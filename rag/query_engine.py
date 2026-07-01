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
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
DEFAULT_TOP_K = 20

# Extra retrieval terms so each research question pulls a distinct review slice.
RETRIEVAL_QUERY_BOOST: dict[str, str] = {
    "Why do users struggle to discover new music?": (
        "cannot find new artists struggle discovery algorithm filter bubble onboarding overwhelm"
    ),
    "What are the most common frustrations with recommendations?": (
        "bad recommendations irrelevant annoying repetitive wrong genre stale discover weekly"
    ),
    "What listening behaviors are users trying to achieve?": (
        "mood workout study focus party background vibe context playlist habit goal"
    ),
    "What causes users to repeatedly listen to the same content?": (
        "same songs repeat loop comfort nostalgia stale playlist no new music"
    ),
    "Which user segments experience different discovery challenges?": (
        "casual listener power user new user genre enthusiast different needs segment"
    ),
    "What unmet needs emerge consistently across reviews?": (
        "wish could need want missing lack control explain recommendation feature request"
    ),
}

# Pre-built templates for common discovery questions.
QUERY_TEMPLATES: dict[str, str] = {
    "discovery_struggle": "Why do users struggle to discover new music?",
    "recommendation_frustrations": "What are the most common frustrations with recommendations?",
    "desired_behaviors": "What listening behaviors are users trying to achieve?",
    "repetitive_listening": "What causes users to repeatedly listen to the same content?",
    "segment_challenges": "Which user segments experience different discovery challenges?",
    "unmet_needs": "What unmet needs emerge consistently across reviews?",
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
    pain_points: list[dict] = field(default_factory=list)
    llm_fallback: bool = False
    llm_error: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "insight": self.insight,
            "confidence": self.confidence,
            "supporting_evidence": [asdict(e) for e in self.supporting_evidence],
            "sample_size": self.sample_size,
            "themes_identified": self.themes_identified,
            "recommended_followup_questions": self.recommended_followup_questions,
            "pain_points": self.pain_points,
            "llm_fallback": self.llm_fallback,
            "llm_error": self.llm_error,
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
    """Parse a JSON object from LLM output (handles markdown fences and preamble)."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    match = _JSON_OBJECT_RE.search(text)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_json_array(raw: str) -> list | None:
    """Parse a JSON array from LLM output (handles markdown fences and preamble)."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    match = _JSON_ARRAY_RE.search(text)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize_pain_points(items) -> list[dict]:
    """Validate pain point dicts from LLM or keyword extraction."""
    if not items:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label or label.lower() in seen:
            continue
        seen.add(label.lower())
        try:
            mentions = int(item.get("mentions") or 1)
        except (TypeError, ValueError):
            mentions = 1
        quote = str(item.get("quote") or "").strip()[:240]
        out.append({"label": label, "mentions": max(1, mentions), "quote": quote})
    return out


def _normalize_string_list(items) -> list[str]:
    """Drop empty entries and strip stray quotes from list fields."""
    if not items:
        return []
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        s = str(item).strip().strip('"').strip("'")
        if s and s not in ('""', "''"):
            out.append(s)
    return out


def _strip_json_artifacts(text: str) -> str:
    """Remove trailing JSON keys if the model appended structured fields to prose."""
    cleaned = text.strip()
    for key in ("themes_identified", "recommended_followup_questions"):
        cleaned = re.sub(
            rf'\s*[,|]?\s*"{key}".*$',
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return cleaned.strip().strip('"').strip("'")


def _extract_loose_json_fields(raw: str) -> dict:
    """Best-effort parse when the model returns invalid JSON (e.g. unquoted insight)."""
    parsed = _parse_json_object(raw)
    if parsed:
        return parsed

    out: dict = {}
    insight_match = re.search(
        r'"insight"\s*:\s*("(?:\\.|[^"\\])*"|([^,\n\}\]]+))',
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if insight_match:
        if insight_match.group(1).startswith('"'):
            out["insight"] = json.loads(insight_match.group(1))
        else:
            out["insight"] = insight_match.group(2).strip().strip('"').strip("'")

    themes_match = re.search(
        r'"themes_identified"\s*:\s*\[(.*?)\]',
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if themes_match:
        out["themes_identified"] = _normalize_string_list(
            re.findall(r'"((?:\\.|[^"\\])*)"', themes_match.group(1))
        )

    followups_match = re.search(
        r'"recommended_followup_questions"\s*:\s*\[(.*?)\]',
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if followups_match:
        out["recommended_followup_questions"] = _normalize_string_list(
            re.findall(r'"((?:\\.|[^"\\])*)"', followups_match.group(1))
        )

    return out


def _normalize_insight(raw: str | None) -> str | None:
    """Turn any LLM output into display-ready prose (never raw JSON)."""
    if not raw or not str(raw).strip():
        return None

    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    loose = _extract_loose_json_fields(text)
    if loose.get("insight"):
        return _strip_json_artifacts(str(loose["insight"]))

    extracted = _insight_from_llm_raw(text)
    if extracted:
        return _strip_json_artifacts(extracted)

    if text.startswith("{") and '"insight"' in text[:200]:
        return None

    cleaned = _strip_json_artifacts(text)
    return cleaned if len(cleaned) >= 40 else None


def _insight_from_llm_raw(raw: str) -> str | None:
    """Extract an insight string from LLM output (JSON or plain prose)."""
    parsed = _parse_json_object(raw)
    if parsed:
        insight = parsed.get("insight") or parsed.get("summary") or parsed.get("answer")
        if insight and str(insight).strip():
            return str(insight).strip()
    stripped = raw.strip()
    if len(stripped) < 40:
        return None
    if stripped.startswith("{") and '"insight"' in stripped[:120]:
        return None
    for prefix in ("Answer:", "Insight:", "Summary:", "Response:"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
    return stripped if len(stripped) >= 40 else None


def _is_auth_error_from_msg(msg: str) -> bool:
    lowered = msg.lower()
    return any(
        token in lowered
        for token in ("401", "invalid api key", "invalid_api_key", "authentication", "unauthorized")
    )


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
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        self.model = model
        self.temperature = temperature
        self._llm = None
        self._llm_disabled = False
        self._last_llm_error = ""

    # --- LLM ------------------------------------------------------------- #
    def _effective_provider(self) -> str | None:
        from processors.llm_client import auto_llm_provider

        if self._llm_disabled:
            return None
        resolved = auto_llm_provider()
        if resolved:
            self.provider = resolved
        return resolved

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
        return self._effective_provider() is not None

    def _invoke(self, prompt: str) -> str | None:
        from processors.llm_client import chat_complete_safe, llm_configured

        if self._llm_disabled:
            return None

        tried: set[str] = set()
        candidates: list[str] = []
        first = self._effective_provider()
        if first:
            candidates.append(first)
        for p in ("groq", "openai", "anthropic"):
            if p not in candidates and llm_configured(p):
                candidates.append(p)

        if not candidates:
            self._last_llm_error = "No LLM API key configured."
            return None

        self._last_llm_error = ""
        last_errors: list[str] = []
        for provider in candidates:
            if provider in tried:
                continue
            tried.add(provider)
            text, err = chat_complete_safe(
                prompt, temperature=self.temperature, provider=provider
            )
            if text:
                self.provider = provider
                return text
            if err:
                last_errors.append(err)
                self._last_llm_error = err

        if last_errors and all(_is_auth_error_from_msg(e) for e in last_errors):
            self._llm_disabled = True
        return None

    def _synthesize_prose_insight(self, question: str, context: str) -> str | None:
        """Second-pass LLM call: plain prose synthesis (no JSON)."""
        prompt = (
            "You are a senior UX researcher analyzing Spotify music discovery feedback.\n\n"
            f"Research question: {question}\n\nUser reviews:\n{context}\n\n"
            "Write a clear, meaningful answer in 4-6 sentences that synthesizes patterns "
            "across the reviews. Explain what users struggle with, why it matters, and what "
            "they want from discovery. Use flowing paragraphs only — do NOT use JSON, bullet "
            "lists, or statistics like 'Across X reviews'. Ground every claim in the reviews."
        )
        return self._invoke(prompt)

    def _extractive_insight(self, question: str, results: list[dict]) -> InsightResponse:
        """Build a narrative insight from review text without an LLM."""
        themes_raw = self._collect_themes(results)[:4]
        themes = [t["theme"].replace("_", " ") for t in themes_raw]
        quotes = [r["content"].strip() for r in results[:5] if r.get("content")]

        paragraphs: list[str] = []
        if quotes:
            paragraphs.append(
                "Reviewers describe how Spotify's discovery experience fits — or fails to fit — "
                "their listening habits, often pointing to gaps between what they expect and "
                "what the product delivers."
            )
        if themes:
            if len(themes) == 1:
                theme_text = themes[0]
            elif len(themes) == 2:
                theme_text = f"{themes[0]} and {themes[1]}"
            else:
                theme_text = ", ".join(themes[:-1]) + f", and {themes[-1]}"
            paragraphs.append(
                f"Recurring concerns in the feedback touch on {theme_text}."
            )
        for q in quotes[:3]:
            snippet = q[:160] + ("…" if len(q) > 160 else "")
            paragraphs.append(f'As one user put it: "{snippet}"')
        if not paragraphs:
            paragraphs.append("Limited review text was available for this question.")
        else:
            paragraphs.append(
                "Taken together, the reviews highlight a need for discovery that feels "
                "personal, varied, and easier to navigate."
            )

        return InsightResponse(
            question=question,
            insight=" ".join(paragraphs),
            confidence=self._confidence(results),
            supporting_evidence=self._build_evidence(results),
            sample_size=len(results),
            themes_identified=[t["theme"] for t in themes_raw],
            pain_points=self._resolve_pain_points(question, "", results),
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
    def _enhanced_retrieval_query(question: str) -> str:
        """Append question-specific terms so retrieval differs per research question."""
        q = question.strip()
        boost = RETRIEVAL_QUERY_BOOST.get(q)
        if not boost:
            for template_q, extra in RETRIEVAL_QUERY_BOOST.items():
                if template_q in q:
                    boost = extra
                    break
        return f"{q} {boost}" if boost else q

    @staticmethod
    def _pain_focus_for_question(question: str) -> set[str]:
        """Problems most relevant to each research question."""
        from agents.segment_analyzer import DISCOVERY_PROBLEMS

        q = question.lower()
        if "struggle" in q and "discover" in q:
            return {
                "poor_personalization",
                "filter_bubble",
                "overwhelming_choice",
                "cold_start_onboarding",
                "stale_discover_weekly",
            }
        if "frustration" in q and "recommend" in q:
            return {
                "repetitive_recommendations",
                "poor_personalization",
                "stale_discover_weekly",
                "lack_of_control",
                "mood_context_mismatch",
            }
        if "listening behavior" in q or "trying to achieve" in q:
            return {
                "mood_context_mismatch",
                "overwhelming_choice",
                "lack_of_control",
                "poor_personalization",
            }
        if "repeatedly listen" in q or "same content" in q:
            return {
                "repetitive_recommendations",
                "filter_bubble",
                "stale_discover_weekly",
                "poor_personalization",
            }
        if "segment" in q:
            return set(DISCOVERY_PROBLEMS.keys())
        if "unmet" in q:
            return {
                "lack_of_control",
                "poor_personalization",
                "stale_discover_weekly",
                "overwhelming_choice",
                "filter_bubble",
                "repetitive_recommendations",
            }
        return set(DISCOVERY_PROBLEMS.keys())

    def _extract_pain_points_llm(self, question: str, context: str) -> list[dict]:
        """LLM: pain points tailored to the active research question."""
        prompt = (
            "You are a UX researcher analyzing Spotify discovery feedback.\n\n"
            f"Research question: {question}\n\n"
            f"Reviews:\n{context}\n\n"
            "Extract 4-6 DISTINCT user pain points that are specifically relevant to "
            "THIS research question — not generic Spotify complaints that apply to every topic.\n"
            "Each item needs a short label (3-8 words) and one verbatim quote from the reviews.\n"
            "Return ONLY a JSON array:\n"
            '[{"label": "...", "mentions": 1, "quote": "..."}]'
        )
        raw = self._invoke(prompt)
        if not raw:
            return []
        parsed = _parse_json_array(raw)
        if not parsed:
            obj = _parse_json_object(raw)
            if obj and isinstance(obj.get("pain_points"), list):
                parsed = obj["pain_points"]
        return _normalize_pain_points(parsed)

    @staticmethod
    def _collect_pain_points(question: str, results: list[dict]) -> list[dict]:
        """Rank pain points in retrieved reviews, weighted by question relevance."""
        from agents.segment_analyzer import DISCOVERY_PROBLEMS

        focus = DiscoveryInsightEngine._pain_focus_for_question(question)
        scored: dict[str, dict] = {}

        for record in results:
            text = (record.get("content") or "").lower()
            quote = (record.get("content") or "")[:220]
            weight = max(0.15, min(1.0, float(record.get("score", 0.5) or 0.5)))

            for problem, keywords in DISCOVERY_PROBLEMS.items():
                hits = sum(1 for k in keywords if k in text)
                if not hits:
                    continue
                boost = 2.5 if problem in focus else 0.35
                add = weight * boost * hits
                entry = scored.setdefault(
                    problem, {"score": 0.0, "mentions": 0, "quote": "", "best_hit": 0.0}
                )
                entry["score"] += add
                entry["mentions"] += 1
                if add > entry["best_hit"]:
                    entry["best_hit"] = add
                    entry["quote"] = quote

            meta = record.get("metadata", {})
            indicators = meta.get("frustration_indicators")
            if isinstance(indicators, str):
                for ind in (i.strip() for i in indicators.split(",") if i.strip()):
                    key = ind if ind in DISCOVERY_PROBLEMS else f"frustration:{ind}"
                    boost = 2.0 if ind in focus else 0.5
                    add = weight * boost
                    entry = scored.setdefault(
                        key, {"score": 0.0, "mentions": 0, "quote": "", "best_hit": 0.0}
                    )
                    entry["score"] += add
                    entry["mentions"] += 1
                    if add > entry["best_hit"]:
                        entry["best_hit"] = add
                        entry["quote"] = quote

            raw_themes = meta.get("themes", "")
            for theme in (t.strip() for t in raw_themes.split(",") if t.strip()):
                boost = 2.2 if theme in focus else 0.55
                add = weight * boost
                entry = scored.setdefault(
                    theme, {"score": 0.0, "mentions": 0, "quote": "", "best_hit": 0.0}
                )
                entry["score"] += add
                entry["mentions"] += 1
                if add > entry["best_hit"]:
                    entry["best_hit"] = add
                    entry["quote"] = quote

        ranked = sorted(scored.items(), key=lambda x: x[1]["score"], reverse=True)
        return _normalize_pain_points(
            [
                {
                    "label": problem.replace("_", " ").title().removeprefix("Frustration:"),
                    "mentions": data["mentions"],
                    "quote": data["quote"],
                }
                for problem, data in ranked[:8]
                if data["score"] > 0
            ]
        )

    def _resolve_pain_points(
        self, question: str, context: str, results: list[dict]
    ) -> list[dict]:
        """Question-specific pain points — LLM when available, else weighted extraction."""
        if self.llm_available() and context:
            llm_pains = self._extract_pain_points_llm(question, context)
            if len(llm_pains) >= 2:
                return llm_pains[:8]
        return self._collect_pain_points(question, results)

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
        query = self._enhanced_retrieval_query(question)
        results = self._retrieve(query, top_k=top_k, filters=filters)
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
            "You are a senior UX researcher analyzing Spotify music discovery feedback.\n\n"
            f"Research question: {question}\n\nUser reviews:\n{context}\n\n"
            "Write a clear, meaningful answer in 4-6 sentences that synthesizes patterns "
            "across the reviews. Explain what users are trying to achieve, what frustrates "
            "them, and what they want from discovery.\n\n"
            "Rules:\n"
            "- Plain English paragraphs only\n"
            "- Do NOT use JSON, bullet lists, or key-value format\n"
            "- Do NOT open with statistics like 'Across X reviews'\n"
            "- Ground every claim in the reviews provided"
        )

        raw = self._invoke(prompt)
        if not raw:
            fallback = self._extractive_insight(question, results)
            fallback.llm_fallback = True
            fallback.llm_error = self._last_llm_error or "LLM unavailable"
            return fallback

        insight_text = _normalize_insight(raw)
        loose = _extract_loose_json_fields(raw)

        if not insight_text:
            insight_text = _normalize_insight(self._synthesize_prose_insight(question, context))

        if not insight_text:
            fallback = self._extractive_insight(question, results)
            fallback.llm_fallback = True
            fallback.llm_error = self._last_llm_error or "Could not parse LLM response"
            return fallback

        themes = _normalize_string_list(loose.get("themes_identified")) or [
            t["theme"].replace("_", " ") for t in self._collect_themes(results)[:5]
        ]
        followups = _normalize_string_list(loose.get("recommended_followup_questions")) or [
            q for q in QUERY_TEMPLATES.values() if q != question
        ][:3]

        return InsightResponse(
            question=question,
            insight=insight_text,
            confidence=self._confidence(results),
            supporting_evidence=self._build_evidence(results),
            sample_size=len(results),
            themes_identified=themes,
            pain_points=self._resolve_pain_points(question, context, results),
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
        if summary:
            summary = _normalize_insight(summary) or summary
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
