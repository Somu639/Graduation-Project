"""Sentiment analysis for Spotify discovery feedback.

Combines two signals:
  * VADER - fast, rule-based polarity scoring (no API cost).
  * An LLM (Claude or GPT via LangChain) - nuanced, discovery-specific analysis
    that judges satisfaction with recommendations and surfaces frustration /
    enthusiasm phrases.

Results are returned as a :class:`SentimentResult` dataclass.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Overall = Literal["positive", "negative", "neutral", "mixed"]

# Rule-based cue phrases used for the quick (no-LLM) path.
FRUSTRATION_CUES: tuple[str, ...] = (
    "repetitive", "same songs", "same artists", "boring", "stuck",
    "frustrating", "annoying", "hate", "worst", "terrible", "broken",
    "useless", "disappointed", "disappointing", "not working", "doesn't work",
    "over and over", "echo chamber", "bubble", "tired of", "keeps playing",
)
ENTHUSIASM_CUES: tuple[str, ...] = (
    "love", "amazing", "great", "perfect", "best", "awesome", "excellent",
    "favorite", "spot on", "fantastic", "brilliant", "obsessed", "incredible",
    "game changer", "lifesaver", "discovered", "introduced me", "nailed it",
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class SentimentResult:
    """Structured sentiment output for a single piece of feedback."""

    overall_sentiment: Overall
    sentiment_score: float  # -1 to 1
    discovery_satisfaction: int  # 1-5 scale
    frustration_indicators: list[str] = field(default_factory=list)
    enthusiasm_indicators: list[str] = field(default_factory=list)


def _parse_json_object(raw: str) -> dict | None:
    """Best-effort extraction of a JSON object from LLM output."""
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class SentimentAnalyzer:
    """Score sentiment via VADER, optionally enriched by an LLM."""

    def __init__(
        self,
        use_llm: bool = False,
        provider: str | None = None,
        model: str | None = None,
        request_delay: float = 1.0,
    ) -> None:
        self.use_llm = use_llm
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        self.model = model
        self.request_delay = request_delay  # rate limit for LLM calls
        self._vader_engine = None
        self._llm = None

    # --- VADER (quick) --------------------------------------------------- #
    def _vader(self):
        if self._vader_engine is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "vaderSentiment is required. Install it via requirements.txt"
                ) from exc
            self._vader_engine = SentimentIntensityAnalyzer()
        return self._vader_engine

    def analyze_quick(self, text: str) -> dict:
        """Return VADER's compound score (-1..1) and a coarse label."""
        if not text:
            return {"label": "neutral", "score": 0.0}
        compound = self._vader().polarity_scores(text)["compound"]
        label = (
            "positive" if compound >= 0.05
            else "negative" if compound <= -0.05
            else "neutral"
        )
        return {"label": label, "score": compound}

    # --- LLM (nuanced) --------------------------------------------------- #
    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY must be set for LLM analysis.")
            self._llm = ChatAnthropic(
                model=self.model or os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"),
                temperature=0,
            )
        else:
            from langchain_openai import ChatOpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY must be set for LLM analysis.")
            self._llm = ChatOpenAI(
                model=self.model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=0,
            )
        return self._llm

    def analyze_nuanced(self, text: str) -> dict | None:
        """Ask the LLM for a discovery-specific sentiment judgment."""
        prompt = (
            "You are an expert at analyzing Spotify music-discovery feedback. "
            "Judge the user's satisfaction with music discovery and "
            "recommendations specifically.\n\n"
            f"Review: {text}\n\n"
            "Return ONLY valid JSON with these keys:\n"
            '  "overall_sentiment": one of "positive","negative","neutral","mixed"\n'
            '  "sentiment_score": float from -1 to 1\n'
            '  "discovery_satisfaction": integer from 1 to 5\n'
            '  "frustration_indicators": list of short quoted phrases\n'
            '  "enthusiasm_indicators": list of short quoted phrases'
        )
        try:
            from processors.llm_client import chat_complete

            content = chat_complete(prompt, temperature=0)
            return _parse_json_object(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM sentiment analysis failed: %s", exc)
            return None

    # --- helpers --------------------------------------------------------- #
    @staticmethod
    def _match_cues(text: str, cues: tuple[str, ...]) -> list[str]:
        lowered = text.lower()
        return [cue for cue in cues if cue in lowered]

    @staticmethod
    def _score_to_satisfaction(score: float) -> int:
        """Map a -1..1 score onto a 1..5 satisfaction scale."""
        return max(1, min(5, round((score + 1) / 2 * 4) + 1))

    @staticmethod
    def _overall(score: float, frustration: list, enthusiasm: list) -> Overall:
        if frustration and enthusiasm:
            return "mixed"
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    # --- public API ------------------------------------------------------ #
    def analyze(self, text: str, use_llm: bool | None = None) -> SentimentResult:
        """Produce a combined :class:`SentimentResult` for one string."""
        if not text:
            return SentimentResult("neutral", 0.0, 3, [], [])

        quick = self.analyze_quick(text)
        score = quick["score"]
        frustration = self._match_cues(text, FRUSTRATION_CUES)
        enthusiasm = self._match_cues(text, ENTHUSIASM_CUES)
        result = SentimentResult(
            overall_sentiment=self._overall(score, frustration, enthusiasm),
            sentiment_score=score,
            discovery_satisfaction=self._score_to_satisfaction(score),
            frustration_indicators=frustration,
            enthusiasm_indicators=enthusiasm,
        )

        should_use_llm = self.use_llm if use_llm is None else use_llm
        if should_use_llm:
            nuanced = self.analyze_nuanced(text)
            if nuanced:
                result = SentimentResult(
                    overall_sentiment=nuanced.get("overall_sentiment", result.overall_sentiment),
                    sentiment_score=float(nuanced.get("sentiment_score", score)),
                    discovery_satisfaction=int(
                        nuanced.get("discovery_satisfaction", result.discovery_satisfaction)
                    ),
                    frustration_indicators=nuanced.get(
                        "frustration_indicators", frustration
                    ),
                    enthusiasm_indicators=nuanced.get(
                        "enthusiasm_indicators", enthusiasm
                    ),
                )
        return result

    def analyze_batch(
        self,
        records: list[dict],
        text_field: str = "clean_text",
        use_llm: bool | None = None,
    ) -> list[dict]:
        """Attach sentiment fields to each record.

        Adds: ``sentiment`` (overall), ``sentiment_score``,
        ``discovery_satisfaction``, ``frustration_indicators``,
        ``enthusiasm_indicators``.
        """
        should_use_llm = self.use_llm if use_llm is None else use_llm
        enriched: list[dict] = []
        for record in records:
            result = self.analyze(record.get(text_field, ""), use_llm=should_use_llm)
            new_record = dict(record)
            data = asdict(result)
            new_record["sentiment"] = data.pop("overall_sentiment")
            new_record.update(data)
            enriched.append(new_record)
            if should_use_llm and self.request_delay:
                time.sleep(self.request_delay)  # rate limit LLM calls
        return enriched


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = SentimentAnalyzer(use_llm=False)
    print(analyzer.analyze("Discover Weekly is amazing, I find new songs every week!"))
    print(analyzer.analyze("The recommendations are repetitive and boring lately."))
    print(analyzer.analyze("Love the app but discovery keeps giving me the same songs."))
