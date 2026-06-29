"""Retrieval-augmented generation query engine for discovery insights.

Provides :class:`DiscoveryInsightEngine`, which retrieves relevant reviews from
the vector store and uses an LLM (Claude by default) to synthesize grounded
insights with supporting quotes, source attribution, and an evidence-based
confidence score.

A thin :class:`QueryEngine` alias preserves the simple ``query()`` API used by
the agents and the API layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from .vector_store import VectorStore

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_TOP_K = 20

# Pre-built templates for common discovery questions.
QUERY_TEMPLATES: dict[str, str] = {
    "discovery_struggle": "Why do users struggle to discover new music?",
    "recommendation_frustrations": "What are the most common recommendation frustrations?",
    "desired_behaviors": "What listening behaviors are users trying to achieve?",
    "repetitive_listening": "Why do users repeatedly listen to the same content?",
    "power_vs_casual": "How do power users differ from casual listeners in discovery needs?",
    "unmet_needs": "What unmet needs appear consistently across reviews?",
}


# --------------------------------------------------------------------------- #
# Output dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    quote: str
    source: str
    sentiment: str


@dataclass
class InsightResponse:
    question: str
    insight: str
    confidence: float
    supporting_evidence: list[Evidence] = field(default_factory=list)
    sample_size: int = 0
    themes_identified: list[str] = field(default_factory=list)
    recommended_followup_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "insight": self.insight,
            "confidence": self.confidence,
            "supporting_evidence": [asdict(e) for e in self.supporting_evidence],
            "sample_size": self.sample_size,
            "themes_identified": self.themes_identified,
            "recommended_followup_questions": self.recommended_followup_questions,
        }


@dataclass
class ThemeSummary:
    query: str
    summary: str
    themes: list[dict]  # [{"theme": str, "count": int}]
    sample_size: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonReport:
    segment_a: str
    segment_b: str
    summary: str
    differences: list[str]
    segment_a_traits: list[str]
    segment_b_traits: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureRequest:
    request: str
    source: str
    quote: str

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_json_object(raw: str) -> dict | None:
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class DiscoveryInsightEngine:
    """Synthesize grounded discovery insights from indexed reviews."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.model = model
        self.temperature = temperature
        self._llm = None

    # --- LLM ------------------------------------------------------------- #
    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY must be set.")
            self._llm = ChatAnthropic(
                model=self.model or os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"),
                temperature=self.temperature,
            )
        else:
            from langchain_openai import ChatOpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY must be set.")
            self._llm = ChatOpenAI(
                model=self.model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=self.temperature,
            )
        return self._llm

    def llm_available(self) -> bool:
        """True if an API key is configured for the selected provider."""
        from processors.llm_client import llm_configured

        return llm_configured(self.provider)

    def _invoke(self, prompt: str) -> str:
        from processors.llm_client import chat_complete

        return chat_complete(prompt, temperature=self.temperature, provider=self.provider)

    def _extractive_insight(self, question: str, results: list[dict]) -> InsightResponse:
        """Build a grounded insight WITHOUT an LLM (lexical/statistical only)."""
        sentiments = Counter(
            r.get("metadata", {}).get("sentiment", "unknown") for r in results
        )
        total = len(results)
        dominant, dom_count = (sentiments.most_common(1)[0] if sentiments else ("unknown", 0))
        themes = [t["theme"] for t in self._collect_themes(results)[:5]]
        top_quote = results[0]["content"][:200] if results else ""
        insight = (
            f"Across {total} relevant reviews, sentiment is predominantly "
            f"{dominant} ({(dom_count / total * 100):.0f}%). "
            + (f"Recurring themes: {', '.join(themes)}. " if themes else "")
            + (f"Representative feedback: \u201c{top_quote}\u201d" if top_quote else "")
        )
        return InsightResponse(
            question=question,
            insight=insight,
            confidence=self._confidence(results),
            supporting_evidence=self._build_evidence(results),
            sample_size=total,
            themes_identified=themes,
            recommended_followup_questions=[
                q for q in QUERY_TEMPLATES.values() if q != question
            ][:3],
        )

    # --- retrieval helpers ---------------------------------------------- #
    def _retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K, filters: dict | None = None
    ) -> list[dict]:
        return self.vector_store.similarity_search(query, top_k=top_k, filters=filters)

    @staticmethod
    def _format_context(results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            src = meta.get("source", "unknown")
            sent = meta.get("sentiment", "n/a")
            lines.append(f"[{i}] (source={src}, sentiment={sent}) {r['content']}")
        return "\n".join(lines)

    @staticmethod
    def _build_evidence(results: list[dict], limit: int = 8) -> list[Evidence]:
        evidence: list[Evidence] = []
        for r in results[:limit]:
            meta = r.get("metadata", {})
            quote = r["content"]
            evidence.append(
                Evidence(
                    quote=quote[:300],
                    source=meta.get("source", "unknown"),
                    sentiment=meta.get("sentiment", "n/a"),
                )
            )
        return evidence

    @staticmethod
    def _confidence(results: list[dict]) -> float:
        """Confidence from sample size, sentiment consistency, and similarity."""
        if not results:
            return 0.0
        sample_factor = min(1.0, len(results) / DEFAULT_TOP_K)

        sentiments = [
            r.get("metadata", {}).get("sentiment")
            for r in results
            if r.get("metadata", {}).get("sentiment")
        ]
        if sentiments:
            top = Counter(sentiments).most_common(1)[0][1]
            consistency = top / len(sentiments)
        else:
            consistency = 0.5

        scores = [r.get("score", 0.0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_score = max(0.0, min(1.0, avg_score))

        confidence = 0.5 * consistency + 0.3 * sample_factor + 0.2 * avg_score
        return round(max(0.0, min(1.0, confidence)), 2)

    @staticmethod
    def _collect_themes(results: list[dict]) -> list[dict]:
        counter: Counter[str] = Counter()
        for r in results:
            raw = r.get("metadata", {}).get("themes", "")
            for theme in (t.strip() for t in raw.split(",") if t.strip()):
                counter[theme] += 1
        return [{"theme": t, "count": c} for t, c in counter.most_common()]

    # --- core API -------------------------------------------------------- #
    def answer_question(
        self, question: str, top_k: int = DEFAULT_TOP_K, filters: dict | None = None
    ) -> InsightResponse:
        """Answer a question with a grounded insight + supporting evidence."""
        results = self._retrieve(question, top_k=top_k, filters=filters)
        if not results:
            return InsightResponse(
                question=question,
                insight="No indexed feedback is available to answer this question.",
                confidence=0.0,
                sample_size=0,
            )

        # No API key configured -> grounded extractive answer (no LLM call).
        if not self.llm_available():
            return self._extractive_insight(question, results)

        context = self._format_context(results)
        prompt = (
            "You are a senior product insights analyst for Spotify's music "
            "discovery experience. Using ONLY the user feedback below, answer the "
            "question with a concise, evidence-grounded insight.\n\n"
            f"Question: {question}\n\nFeedback:\n{context}\n\n"
            "Return ONLY valid JSON with keys:\n"
            '  "insight": a 2-4 sentence synthesized insight\n'
            '  "themes_identified": list of short theme strings\n'
            '  "recommended_followup_questions": list of 3 follow-up questions'
        )

        parsed = _parse_json_object(self._invoke(prompt)) or {}
        themes = parsed.get("themes_identified") or [
            t["theme"] for t in self._collect_themes(results)[:5]
        ]
        followups = parsed.get("recommended_followup_questions") or [
            q for q in QUERY_TEMPLATES.values() if q != question
        ][:3]

        return InsightResponse(
            question=question,
            insight=parsed.get("insight", "Unable to synthesize an insight."),
            confidence=self._confidence(results),
            supporting_evidence=self._build_evidence(results),
            sample_size=len(results),
            themes_identified=themes,
            recommended_followup_questions=followups,
        )

    def find_similar_complaints(self, complaint: str, k: int = 10) -> list[dict]:
        """Find complaints most similar to the given one (negative-leaning)."""
        results = self.vector_store.hybrid_search(
            complaint, metadata_filters={"sentiment": "negative"}, top_k=k
        )
        # Fall back to unfiltered search if no negatively-tagged data exists yet.
        if not results:
            results = self._retrieve(complaint, top_k=k)
        return [
            {
                "quote": r["content"][:300],
                "source": r.get("metadata", {}).get("source", "unknown"),
                "sentiment": r.get("metadata", {}).get("sentiment", "n/a"),
                "score": round(r.get("score", 0.0), 3),
            }
            for r in results
        ]

    def aggregate_themes(self, query: str, top_k: int = DEFAULT_TOP_K) -> ThemeSummary:
        """Aggregate themes across the reviews most relevant to a query."""
        results = self._retrieve(query, top_k=top_k)
        themes = self._collect_themes(results)
        if not results:
            return ThemeSummary(query=query, summary="No data available.", themes=[], sample_size=0)

        theme_list = ", ".join(f"{t['theme']} ({t['count']})" for t in themes) or "none"
        if not self.llm_available():
            summary = (
                f"Across {len(results)} reviews, the most common discovery themes are "
                f"{theme_list}."
            )
            return ThemeSummary(query=query, summary=summary, themes=themes, sample_size=len(results))
        prompt = (
            "Summarize the dominant themes in this Spotify discovery feedback in "
            "2-3 sentences. Themes detected: "
            f"{theme_list}.\n\nFeedback:\n{self._format_context(results)}"
        )
        summary = self._invoke(prompt)
        return ThemeSummary(
            query=query, summary=summary, themes=themes, sample_size=len(results)
        )

    def compare_segments(self, segment_a: str, segment_b: str) -> ComparisonReport:
        """Compare discovery needs/pain points between two user segments."""
        results_a = self._retrieve(f"music discovery feedback from {segment_a}", top_k=12)
        results_b = self._retrieve(f"music discovery feedback from {segment_b}", top_k=12)

        prompt = (
            "Compare how two Spotify user segments experience music discovery, "
            "using ONLY the feedback provided.\n\n"
            f"Segment A ({segment_a}):\n{self._format_context(results_a)}\n\n"
            f"Segment B ({segment_b}):\n{self._format_context(results_b)}\n\n"
            "Return ONLY valid JSON with keys: summary (string), "
            "segment_a_traits (list), segment_b_traits (list), "
            "key_differences (list), recommendations (list)."
        )
        parsed = _parse_json_object(self._invoke(prompt)) or {}
        return ComparisonReport(
            segment_a=segment_a,
            segment_b=segment_b,
            summary=parsed.get("summary", ""),
            differences=parsed.get("key_differences", []),
            segment_a_traits=parsed.get("segment_a_traits", []),
            segment_b_traits=parsed.get("segment_b_traits", []),
            recommendations=parsed.get("recommendations", []),
        )

    def extract_feature_requests(self, k: int = DEFAULT_TOP_K) -> list[FeatureRequest]:
        """Extract concrete feature requests from feedback."""
        # Prefer a purpose-built subset index if it has been populated.
        try:
            subset = self.vector_store.get_index("feature_requests")
            results = subset.similarity_search(
                "feature request improvement suggestion add ability", top_k=k
            )
        except Exception:  # noqa: BLE001 - subset may not exist yet
            results = []
        if not results:
            results = self._retrieve(
                "feature request improvement suggestion add ability to", top_k=k
            )
        if not results:
            return []

        prompt = (
            "Extract distinct, concrete feature requests for Spotify music "
            "discovery from the feedback below. Return ONLY valid JSON: a list of "
            '{"request": str, "source": str, "quote": str}. Ignore vague '
            "complaints that aren't actionable requests.\n\n"
            f"Feedback:\n{self._format_context(results)}"
        )
        raw = self._invoke(prompt)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [
            FeatureRequest(
                request=str(item.get("request", "")),
                source=str(item.get("source", "unknown")),
                quote=str(item.get("quote", "")),
            )
            for item in items
            if item.get("request")
        ]

    # --- templates ------------------------------------------------------- #
    def list_templates(self) -> dict[str, str]:
        return dict(QUERY_TEMPLATES)

    def run_template(self, template_key: str) -> InsightResponse:
        if template_key not in QUERY_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_key}'. Options: {list(QUERY_TEMPLATES)}"
            )
        return self.answer_question(QUERY_TEMPLATES[template_key])

    def run_all_templates(self) -> list[dict]:
        return [self.run_template(key).to_dict() for key in QUERY_TEMPLATES]

    # --- backward-compatible simple API --------------------------------- #
    def query(self, question: str, k: int = 6, filter: dict | None = None) -> dict:
        """Legacy helper: returns ``{"answer", "sources"}`` for agents/API."""
        results = self._retrieve(question, top_k=k, filters=filter)
        if not results:
            return {
                "answer": "No indexed feedback is available to answer this question.",
                "sources": [],
            }
        if not self.llm_available():
            bullets = "\n".join(f"- {r['content'][:200]}" for r in results)
            return {
                "answer": f"Most relevant feedback for '{question}':\n{bullets}",
                "sources": [
                    {"content": r["content"], "metadata": r.get("metadata", {})}
                    for r in results
                ],
            }
        prompt = (
            "You are a product insights analyst for Spotify music discovery. "
            "Answer using ONLY the context below; cite concrete pain points.\n\n"
            f"Context:\n{self._format_context(results)}\n\nQuestion: {question}\nAnswer:"
        )
        answer = self._invoke(prompt)
        sources = [
            {"content": r["content"], "metadata": r.get("metadata", {})}
            for r in results
        ]
        return {"answer": answer, "sources": sources}


# Backwards-compatible alias used by agents and the API layer.
QueryEngine = DiscoveryInsightEngine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = DiscoveryInsightEngine()
    print("Templates:")
    for key, q in engine.list_templates().items():
        print(f"  {key}: {q}")
