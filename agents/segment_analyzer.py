"""User segmentation agent for Spotify discovery feedback.

Identifies user segments from review language, builds a rich profile per segment
(behavioral indicators, pain points, workarounds, desired outcomes, satisfaction,
sample quotes, size estimate), and produces a Segment Comparison Matrix mapping
discovery problems to the segments they affect most.
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

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Segment -> description + behavioral indicator keywords found in review text.
SEGMENTS: dict[str, dict] = {
    "casual_listeners": {
        "description": "Casual listeners who use Spotify for background music and ready-made playlists.",
        "indicators": [
            "background", "playlist", "chill", "study", "work", "just play",
            "lean back", "shuffle", "don't care", "easy listening",
        ],
    },
    "active_explorers": {
        "description": "Active explorers who constantly seek out new music and artists.",
        "indicators": [
            "new music", "discover", "explore", "new artists", "find new",
            "fresh", "latest", "underground", "deep cuts", "crate digging",
        ],
    },
    "genre_enthusiasts": {
        "description": "Genre enthusiasts who go deep within one or a few genres.",
        "indicators": [
            "genre", "metal", "jazz", "hip hop", "classical", "techno", "indie",
            "k-pop", "only listen to", "specific genre", "subgenre",
        ],
    },
    "mood_based_listeners": {
        "description": "Mood/context-driven listeners who pick music by activity or feeling.",
        "indicators": [
            "mood", "vibe", "workout", "gym", "sleep", "focus", "party",
            "relax", "energy", "context", "for running",
        ],
    },
    "social_listeners": {
        "description": "Social listeners who share playlists and follow friends' music.",
        "indicators": [
            "friends", "share", "shared playlist", "collaborative", "blend",
            "social", "what my friends", "group", "followers",
        ],
    },
    "nostalgic_listeners": {
        "description": "Nostalgic listeners who prefer older, familiar, comfort music.",
        "indicators": [
            "old songs", "oldies", "throwback", "nostalgia", "classics",
            "comfort", "childhood", "90s", "80s", "familiar",
        ],
    },
    "power_users": {
        "description": "Power users on premium with multiple devices and heavy listening hours.",
        "indicators": [
            "premium", "multiple devices", "all day", "hours", "every day",
            "heavy user", "years", "power user", "desktop and phone", "wrapped",
        ],
    },
    "new_users": {
        "description": "New users / recent adopters who may hit onboarding issues.",
        "indicators": [
            "just started", "new to spotify", "recently", "switched from",
            "first time", "signed up", "onboarding", "new user", "just downloaded",
        ],
    },
}

# Discovery problems tracked in the comparison matrix + their trigger keywords.
DISCOVERY_PROBLEMS: dict[str, list[str]] = {
    "repetitive_recommendations": ["same songs", "repetitive", "over and over", "loop", "same artists"],
    "filter_bubble": ["bubble", "echo chamber", "narrow", "same genre", "stuck", "limited"],
    "poor_personalization": ["doesn't know", "not my taste", "generic", "random", "irrelevant"],
    "stale_discover_weekly": ["discover weekly", "release radar", "stale", "nothing new"],
    "mood_context_mismatch": ["wrong mood", "doesn't match", "vibe", "context", "wrong songs"],
    "cold_start_onboarding": ["new account", "onboarding", "just started", "no recommendations yet"],
    "lack_of_control": ["control", "can't tell", "no way to", "dislike button", "feedback"],
    "overwhelming_choice": ["too much", "overwhelming", "where to start", "too many"],
}


@dataclass
class SegmentProfile:
    segment: str
    description: str
    size_estimate_pct: float
    behavioral_indicators: list[str] = field(default_factory=list)
    primary_frustrations: list[str] = field(default_factory=list)
    desired_outcomes: list[str] = field(default_factory=list)
    discovery_pain_points: list[str] = field(default_factory=list)
    workarounds: list[str] = field(default_factory=list)
    features_mentioned: list[str] = field(default_factory=list)
    recommendation_satisfaction: str = "unknown"
    good_discovery_definition: str = ""
    repetitive_triggers: list[str] = field(default_factory=list)
    sample_quotes: list[dict] = field(default_factory=list)

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


class SegmentAnalyzer:
    """Classify reviews into segments and build profiles + a comparison matrix."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        engine: DiscoveryInsightEngine | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.1,
        segments: dict[str, dict] | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.engine = engine or DiscoveryInsightEngine(vector_store=self.vector_store)
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.model = model
        self.temperature = temperature
        self.segments = segments or SEGMENTS
        self._llm = None

    # --- LLM ------------------------------------------------------------- #
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

    # --- classification & sizing ---------------------------------------- #
    def classify_review(self, text: str) -> list[str]:
        """Return segments whose behavioral indicators appear in the text."""
        lowered = (text or "").lower()
        return [
            name
            for name, cfg in self.segments.items()
            if any(ind in lowered for ind in cfg["indicators"])
        ]

    def estimate_sizes(self, records: list[dict] | None = None) -> dict[str, dict]:
        """Estimate each segment's share of reviews.

        Segments can overlap (a review may match several), so percentages need
        not sum to 100. ``records`` defaults to the full indexed corpus.
        """
        if records is None:
            records = self.vector_store.get_records(limit=2000)
        total = len(records) or 1
        counts: Counter[str] = Counter()
        for record in records:
            text = record.get("content") or record.get("clean_text", "")
            for seg in self.classify_review(text):
                counts[seg] += 1
        return {
            name: {
                "count": counts.get(name, 0),
                "pct": round(counts.get(name, 0) / total * 100, 1),
            }
            for name in self.segments
        }

    # --- per-segment profile -------------------------------------------- #
    def analyze_segment(
        self, segment: str, size_estimate: dict | None = None
    ) -> SegmentProfile:
        """Build a full profile for a single segment, grounded in evidence."""
        cfg = self.segments[segment]
        query = f"{cfg['description']} {' '.join(cfg['indicators'][:5])}"
        evidence = self.vector_store.similarity_search(query, top_k=15)

        sample_quotes = [
            {
                "quote": r["content"][:240],
                "source": r.get("metadata", {}).get("source", "unknown"),
            }
            for r in evidence[:4]
        ]
        context = "\n".join(f"- {r['content'][:220]}" for r in evidence)

        prompt = (
            "You are a user research analyst. Profile the Spotify user segment "
            f"'{segment}' ({cfg['description']}) using ONLY this feedback.\n\n"
            f"Feedback:\n{context}\n\n"
            "Return ONLY valid JSON with keys: discovery_pain_points (list), "
            "workarounds (list), features_mentioned (list), "
            "recommendation_satisfaction (one of very_low/low/medium/high), "
            "good_discovery_definition (string), repetitive_triggers (list), "
            "primary_frustrations (list), desired_outcomes (list)."
        )
        parsed = _parse_json_object(self._invoke(prompt)) if evidence else {}
        parsed = parsed or {}

        pct = (size_estimate or {}).get(segment, {}).get("pct", 0.0)
        return SegmentProfile(
            segment=segment,
            description=cfg["description"],
            size_estimate_pct=pct,
            behavioral_indicators=cfg["indicators"],
            primary_frustrations=parsed.get("primary_frustrations", []),
            desired_outcomes=parsed.get("desired_outcomes", []),
            discovery_pain_points=parsed.get("discovery_pain_points", []),
            workarounds=parsed.get("workarounds", []),
            features_mentioned=parsed.get("features_mentioned", []),
            recommendation_satisfaction=parsed.get("recommendation_satisfaction", "unknown"),
            good_discovery_definition=parsed.get("good_discovery_definition", ""),
            repetitive_triggers=parsed.get("repetitive_triggers", []),
            sample_quotes=sample_quotes,
        )

    def build_profiles(self) -> list[SegmentProfile]:
        """Build profiles for every configured segment."""
        sizes = self.estimate_sizes()
        return [self.analyze_segment(name, sizes) for name in self.segments]

    def analyze_all(self) -> list[dict]:
        """Backward-compatible: return all segment profiles as dicts."""
        return [profile.to_dict() for profile in self.build_profiles()]

    # --- comparison matrix ---------------------------------------------- #
    def build_comparison_matrix(self, records: list[dict] | None = None) -> dict:
        """Map discovery problems to the segments they affect most.

        For each (segment, problem) pair, computes the share of that segment's
        reviews mentioning the problem and labels severity high/medium/low.
        """
        if records is None:
            records = self.vector_store.get_records(limit=2000)

        # Group review texts by segment.
        by_segment: dict[str, list[str]] = {name: [] for name in self.segments}
        for record in records:
            text = (record.get("content") or record.get("clean_text", "") or "").lower()
            for seg in self.classify_review(text):
                by_segment[seg].append(text)

        def severity(pct: float) -> str:
            if pct >= 30:
                return "high"
            if pct >= 10:
                return "medium"
            return "low" if pct > 0 else "none"

        matrix: dict[str, dict] = {}
        for seg, texts in by_segment.items():
            total = len(texts) or 1
            row: dict[str, dict] = {}
            for problem, keywords in DISCOVERY_PROBLEMS.items():
                hits = sum(any(k in t for k in keywords) for t in texts)
                pct = round(hits / total * 100, 1)
                row[problem] = {"pct": pct, "severity": severity(pct), "count": hits}
            matrix[seg] = row

        return {
            "segments": list(self.segments.keys()),
            "problems": list(DISCOVERY_PROBLEMS.keys()),
            "matrix": matrix,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_records = [
        {"content": "I just play background playlists while studying, discover weekly is repetitive"},
        {"content": "Always hunting for new music and new artists, but I feel stuck in a bubble"},
        {"content": "Premium power user here, listen all day across multiple devices, same songs over and over"},
        {"content": "New to spotify, onboarding gave me no recommendations yet"},
        {"content": "I only listen to metal, the genre recommendations are too narrow"},
    ]
    analyzer = SegmentAnalyzer()
    print("Sizes:")
    print(json.dumps(analyzer.estimate_sizes(demo_records), indent=2))
    print("\nComparison matrix:")
    print(json.dumps(analyzer.build_comparison_matrix(demo_records), indent=2))
