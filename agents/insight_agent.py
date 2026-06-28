"""Autonomous discovery-research agent.

Defines :class:`DiscoveryResearchAgent`, a LangChain/LangGraph ReAct agent
(Claude by default) equipped with research tools over the indexed review corpus.
It breaks research questions into sub-questions, gathers evidence, forms and
validates hypotheses, identifies affected segments, and synthesizes findings
with cited quotes into a :class:`ResearchReport`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from rag.query_engine import DiscoveryInsightEngine
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a User Research Analyst specializing in music streaming products.
Your goal is to understand why Spotify users struggle with music discovery
and what would help them discover more new music.

You have access to thousands of user reviews from App Store, Play Store,
and Reddit discussions.

For each research question:
1. Break it into sub-questions
2. Search for relevant evidence
3. Look for patterns and contradictions
4. Synthesize findings with supporting quotes
5. Generate actionable hypotheses
6. Identify which user segments are most affected

Always cite specific reviews and quantify when possible."""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass
class ResearchReport:
    """Output of a full research session."""

    questions: list[str]
    findings: list[dict] = field(default_factory=list)
    segments_affected: str = ""
    followup_questions: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DiscoveryResearchAgent:
    """A tool-using agent that researches Spotify discovery pain points."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        engine: DiscoveryInsightEngine | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.engine = engine or DiscoveryInsightEngine(vector_store=self.vector_store)
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.model = model
        self.temperature = temperature
        self._llm = None
        self._agent = None

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model=self.model or os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"),
                temperature=self.temperature,
            )
        else:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=self.temperature,
            )
        return self._llm

    def _invoke(self, prompt: str) -> str:
        response = self._get_llm().invoke(prompt)
        return getattr(response, "content", str(response))

    # ------------------------------------------------------------------ #
    # Tool implementations (return strings for the agent)
    # ------------------------------------------------------------------ #
    def _impl_search_reviews(self, query: str, source: str = "", sentiment: str = "") -> str:
        filters: dict = {}
        if source:
            filters["source"] = source
        if sentiment:
            filters["sentiment"] = sentiment
        results = self.vector_store.similarity_search(
            query, top_k=8, filters=filters or None
        )
        if not results:
            return "No matching reviews found."
        return "\n".join(
            f"- ({r['metadata'].get('source','?')}/"
            f"{r['metadata'].get('sentiment','?')}) {r['content'][:220]}"
            for r in results
        )

    def _impl_sentiment_distribution(self, source: str = "") -> str:
        records = self.vector_store.get_records(
            filters={"source": source} if source else None, limit=2000
        )
        if not records:
            return "No data available for that filter."
        sentiments = Counter(
            r["metadata"].get("sentiment", "unknown") for r in records
        )
        ratings = [
            r["metadata"]["rating"]
            for r in records
            if isinstance(r["metadata"].get("rating"), (int, float))
        ]
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
        total = len(records)
        dist = ", ".join(
            f"{label}: {count} ({count / total:.0%})"
            for label, count in sentiments.most_common()
        )
        return (
            f"Sample size: {total}. Sentiment distribution -> {dist}. "
            f"Average rating: {avg_rating}."
        )

    def _impl_extract_quotes(self, theme: str, count: int = 5) -> str:
        results = self.vector_store.similarity_search(theme, top_k=int(count))
        if not results:
            return "No quotes found for that theme."
        return "\n".join(
            f'"{r["content"][:240]}" — {r["metadata"].get("source","?")}'
            for r in results
        )

    def _impl_compare_ratings(self, theme: str) -> str:
        matching = self.vector_store.get_records(filters={"theme": theme}, limit=2000)
        if not matching:
            matching = self.vector_store.similarity_search(theme, top_k=50)
        all_records = self.vector_store.get_records(limit=2000)

        def avg_rating(recs: list[dict]) -> float | None:
            vals = [
                r["metadata"]["rating"]
                for r in recs
                if isinstance(r["metadata"].get("rating"), (int, float))
            ]
            return round(sum(vals) / len(vals), 2) if vals else None

        theme_avg = avg_rating(matching)
        overall_avg = avg_rating(all_records)
        return (
            f"Theme '{theme}': {len(matching)} reviews, avg rating {theme_avg}. "
            f"Overall avg rating: {overall_avg} (n={len(all_records)}). "
            f"Delta: {round((theme_avg or 0) - (overall_avg or 0), 2)}."
        )

    def _impl_identify_segments(self, criteria: str = "") -> str:
        records = self.vector_store.get_records(limit=2000)
        if not records:
            return "No data available to segment."
        segments = Counter()
        for r in records:
            meta = r["metadata"]
            content_len = len(r["content"])
            helpful = meta.get("helpful_count") or 0
            if helpful and helpful > 10 or content_len > 400:
                segments["power_users"] += 1
            elif content_len < 120:
                segments["casual_listeners"] += 1
            if meta.get("sentiment") == "negative":
                segments["frustrated_users"] += 1
            elif meta.get("sentiment") == "positive":
                segments["enthusiasts"] += 1
        summary = ", ".join(f"{k}: {v}" for k, v in segments.most_common())
        return f"Segment estimates (criteria: {criteria or 'general'}) -> {summary}."

    def _impl_generate_hypothesis(self, observation: str) -> str:
        prompt = (
            "Given this observation about Spotify music discovery, propose 2-3 "
            "specific, testable hypotheses (each with how it could be validated):\n\n"
            f"Observation: {observation}"
        )
        return self._invoke(prompt)

    def _impl_validate_hypothesis(self, hypothesis: str) -> str:
        evidence = self.vector_store.similarity_search(hypothesis, top_k=12)
        context = "\n".join(f"- {r['content'][:200]}" for r in evidence)
        prompt = (
            "Assess whether the evidence supports, contradicts, or is "
            "inconclusive about this hypothesis. Give a verdict, a confidence "
            "(0-1), and cite specific evidence.\n\n"
            f"Hypothesis: {hypothesis}\n\nEvidence:\n{context}"
        )
        return self._invoke(prompt)

    # ------------------------------------------------------------------ #
    # Agent assembly
    # ------------------------------------------------------------------ #
    def _build_tools(self):
        from langchain_core.tools import tool

        agent = self  # capture for closures

        @tool
        def search_reviews(query: str, source: str = "", sentiment: str = "") -> str:
            """RAG search over user reviews. Optionally filter by source
            (app_store, play_store, reddit) and sentiment (positive/negative/neutral)."""
            return agent._impl_search_reviews(query, source, sentiment)

        @tool
        def analyze_sentiment_distribution(source: str = "") -> str:
            """Return sentiment distribution and average rating, optionally for a
            single source (app_store, play_store, reddit)."""
            return agent._impl_sentiment_distribution(source)

        @tool
        def extract_user_quotes(theme: str, count: int = 5) -> str:
            """Get verbatim user quotes (with source attribution) about a theme."""
            return agent._impl_extract_quotes(theme, count)

        @tool
        def compare_ratings(theme: str) -> str:
            """Correlate a theme with star ratings vs the overall average."""
            return agent._impl_compare_ratings(theme)

        @tool
        def identify_user_segments(criteria: str = "") -> str:
            """Estimate user segments (power users, casual, frustrated, enthusiasts)."""
            return agent._impl_identify_segments(criteria)

        @tool
        def generate_hypothesis(observation: str) -> str:
            """Form testable hypotheses from an observation."""
            return agent._impl_generate_hypothesis(observation)

        @tool
        def validate_hypothesis(hypothesis: str) -> str:
            """Check a hypothesis against retrieved review evidence."""
            return agent._impl_validate_hypothesis(hypothesis)

        return [
            search_reviews,
            analyze_sentiment_distribution,
            extract_user_quotes,
            compare_ratings,
            identify_user_segments,
            generate_hypothesis,
            validate_hypothesis,
        ]

    def _get_agent(self):
        if self._agent is not None:
            return self._agent
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "langgraph is required for the agent. See requirements.txt"
            ) from exc
        self._agent = create_react_agent(
            self._get_llm(), self._build_tools(), prompt=SYSTEM_PROMPT
        )
        return self._agent

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, question: str) -> str:
        """Run the agent on a single research question; return its analysis."""
        agent = self._get_agent()
        result = agent.invoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    def generate_followups(self, findings: list[dict]) -> list[str]:
        """Auto-generate follow-up research questions from findings."""
        digest = "\n".join(
            f"- {f['question']}: {f.get('insight', '')[:200]}" for f in findings
        )
        prompt = (
            "Based on these research findings about Spotify music discovery, "
            "propose 5 specific follow-up research questions worth investigating "
            "next. Return ONLY a JSON array of strings.\n\n"
            f"{digest}"
        )
        if self.engine.llm_available():
            raw = self._invoke(prompt)
            match = _JSON_ARRAY_RE.search(raw)
            if match:
                try:
                    items = json.loads(match.group(0))
                    return [str(q) for q in items][:5]
                except json.JSONDecodeError:
                    pass
        # Fallback: reuse engine templates not already covered.
        return [
            q for q in self.engine.list_templates().values()
            if q not in {f["question"] for f in findings}
        ][:5]

    def run_research_session(self, questions: list[str]) -> ResearchReport:
        """Run a multi-question research session and synthesize a report."""
        llm_on = self.engine.llm_available()
        findings: list[dict] = []
        for question in questions:
            logger.info("Researching: %s", question)
            insight = self.engine.answer_question(question, top_k=20)
            # Use the tool-using agent only when an LLM is configured.
            analysis = self.run(question) if llm_on else insight.insight
            findings.append(
                {
                    "question": question,
                    "analysis": analysis,
                    "insight": insight.insight,
                    "confidence": insight.confidence,
                    "sample_size": insight.sample_size,
                    "themes": insight.themes_identified,
                    "evidence": [asdict(e) for e in insight.supporting_evidence],
                }
            )

        followups = self.generate_followups(findings)
        segments = self._impl_identify_segments("most affected by discovery issues")

        digest = "\n".join(
            f"- {f['question']}: {f.get('insight', '')[:200]}" for f in findings
        )
        if llm_on:
            summary = self._invoke(
                "Write a concise executive summary (4-6 sentences) of these Spotify "
                f"discovery research findings:\n{digest}"
            )
        else:
            summary = "Key findings (extractive, no LLM configured):\n" + digest

        return ResearchReport(
            questions=questions,
            findings=findings,
            segments_affected=segments,
            followup_questions=followups,
            summary=summary,
        )


class InsightAgent:
    """Backward-compatible thin wrapper used by the API layer."""

    def __init__(self, vector_store: VectorStore | None = None, query_engine=None) -> None:
        self.agent = DiscoveryResearchAgent(vector_store=vector_store, engine=query_engine)

    def generate_insights(self, focus: str = "overall music discovery experience") -> str:
        return self.agent.run(
            f"Generate a grounded product insights report focused on: {focus}. "
            "Use your tools to gather and cite evidence."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = DiscoveryResearchAgent()
    report = agent.run_research_session(
        ["Why do users struggle to discover new music?"]
    )
    print(json.dumps(report.to_dict(), indent=2))
