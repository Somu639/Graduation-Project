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
from collections import Counter
from dataclasses import asdict, dataclass, field

from rag.query_engine import DiscoveryInsightEngine, _extract_loose_json_fields, _normalize_insight, _parse_json_object
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

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
    narrative_summary: str = ""
    profile_tier: str = "basic"
    listening_habits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _humanize_label(name: str) -> str:
    return name.replace("_", " ").title()


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
        from processors.llm_client import auto_llm_provider

        self.provider = provider or auto_llm_provider() or os.getenv("LLM_PROVIDER", "groq")
        self.model = model
        self.temperature = temperature
        self.segments = segments or SEGMENTS

    # --- LLM ------------------------------------------------------------- #
    def _invoke(self, prompt: str) -> tuple[str | None, str | None]:
        from processors.llm_client import auto_llm_provider, chat_complete_safe, llm_configured

        tried: set[str] = set()
        candidates: list[str] = []
        first = auto_llm_provider()
        if first:
            candidates.append(first)
        for p in ("groq", "openai", "anthropic"):
            if p not in candidates and llm_configured(p):
                candidates.append(p)

        last_err = ""
        for provider in candidates:
            if provider in tried:
                continue
            tried.add(provider)
            text, err = chat_complete_safe(
                prompt, temperature=self.temperature, provider=provider
            )
            if text:
                self.provider = provider
                return text, None
            if err:
                last_err = err
        return None, last_err or "LLM unavailable"

    @staticmethod
    def _extractive_profile_fields(evidence: list[dict]) -> dict:
        """Build profile fields from evidence without an LLM."""
        frustrations: Counter[str] = Counter()
        features: Counter[str] = Counter()
        for record in evidence:
            text = (record.get("content") or "").lower()
            meta = record.get("metadata", {})
            for problem, keywords in DISCOVERY_PROBLEMS.items():
                if any(k in text for k in keywords):
                    frustrations[problem] += 1
            for feature in ("discover weekly", "release radar", "daily mix", "radio", "playlist"):
                if feature in text:
                    features[feature] += 1
            inds = meta.get("frustration_indicators")
            if isinstance(inds, str):
                for ind in (i.strip() for i in inds.split(",") if i.strip()):
                    frustrations[ind] += 1
        top_frustrations = [k for k, _ in frustrations.most_common(5)]
        return {
            "primary_frustrations": top_frustrations,
            "desired_outcomes": ["better music discovery", "more variety"],
            "discovery_pain_points": top_frustrations[:3],
            "workarounds": [],
            "features_mentioned": [k for k, _ in features.most_common(5)],
            "recommendation_satisfaction": "unknown",
            "good_discovery_definition": "",
            "repetitive_triggers": [
                k for k in top_frustrations if "repetit" in k or "same" in k
            ],
        }

    @staticmethod
    def _basic_narrative(description: str, evidence: list[dict], parsed: dict) -> str:
        """Short extractive narrative for quick segment snapshots."""
        frustrations = parsed.get("primary_frustrations") or []
        if frustrations:
            fr_text = ", ".join(_humanize_label(f) for f in frustrations[:3])
            opener = f"Users in this segment often mention {fr_text}."
        else:
            opener = f"{description.rstrip('.')}."
        quote = ""
        if evidence:
            snippet = evidence[0]["content"][:180]
            quote = f' One reviewer writes: "{snippet}{"…" if len(evidence[0]["content"]) > 180 else ""}"'
        return opener + quote

    def _segment_evidence(self, segment: str) -> tuple[dict, list[dict], str]:
        cfg = self.segments[segment]
        query = f"{cfg['description']} {' '.join(cfg['indicators'][:5])}"
        evidence = self.vector_store.similarity_search(query, top_k=15)
        context = "\n".join(f"- {r['content'][:220]}" for r in evidence)
        return cfg, evidence, context

    def build_basic_profile(
        self, segment: str, size_estimate: dict | None = None
    ) -> SegmentProfile:
        """Fast extractive profile — always available without LLM."""
        cfg, evidence, _ = self._segment_evidence(segment)
        sample_quotes = [
            {
                "quote": r["content"][:240],
                "source": r.get("metadata", {}).get("source", "unknown"),
            }
            for r in evidence[:4]
        ]
        parsed = self._extractive_profile_fields(evidence) if evidence else {}
        pains = [
            _humanize_label(p) for p in (parsed.get("discovery_pain_points") or [])[:2]
        ]
        pct = (size_estimate or {}).get(segment, {}).get("pct", 0.0)
        return SegmentProfile(
            segment=segment,
            description=cfg["description"],
            size_estimate_pct=pct,
            behavioral_indicators=cfg["indicators"][:4],
            primary_frustrations=pains,
            desired_outcomes=[],
            discovery_pain_points=pains,
            workarounds=[],
            features_mentioned=[],
            recommendation_satisfaction="unknown",
            good_discovery_definition="",
            repetitive_triggers=[],
            sample_quotes=sample_quotes[:1],
            narrative_summary=self._basic_narrative(cfg["description"], evidence, parsed),
            profile_tier="basic",
            listening_habits=[],
        )

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
        """Build a full LLM-enriched profile for a single segment."""
        cfg, evidence, context = self._segment_evidence(segment)
        basic = self.build_basic_profile(segment, size_estimate)

        if not evidence:
            basic.profile_tier = "full"
            return basic

        from processors.llm_client import llm_configured

        parsed: dict = {}
        narrative = ""
        workarounds: list[str] = []
        listening_habits: list[str] = []
        if llm_configured(self.provider) or llm_configured():
            prose_prompt = (
                "You are a senior user researcher writing a deep segment profile for Spotify.\n\n"
                f"Segment: {segment}\n{cfg['description']}\n\nReviews:\n{context}\n\n"
                "Write a detailed 5-7 sentence profile covering how this segment discovers "
                "music, their deepest frustrations, workarounds they use, and what good "
                "discovery means to them. Plain English paragraphs only — no JSON."
            )
            raw, err = self._invoke(prose_prompt)
            if raw:
                narrative = _normalize_insight(raw) or raw.strip()

            struct_prompt = (
                f"Segment '{segment}' Spotify listeners — feedback:\n{context}\n\n"
                "Return ONLY valid JSON with keys: "
                "workarounds (list of strings), desired_outcomes (list), "
                "discovery_pain_points (list), primary_frustrations (list), "
                "features_mentioned (list), listening_habits (list), "
                "recommendation_satisfaction (very_low|low|medium|high), "
                "good_discovery_definition (string)."
            )
            struct_raw, _ = self._invoke(struct_prompt)
            if struct_raw:
                parsed = _parse_json_object(struct_raw) or _extract_loose_json_fields(struct_raw)

            if not parsed:
                parsed = self._extractive_profile_fields(evidence)
            if not narrative:
                narrative = (
                    f"Deep profile — {cfg['description']} "
                    f"Reviews highlight {_humanize_label(parsed.get('primary_frustrations', ['discovery friction'])[0]) if parsed.get('primary_frustrations') else 'discovery friction'}."
                )
            workarounds = parsed.get("workarounds") or []
            listening_habits = parsed.get("listening_habits") or []
            if err and not narrative:
                logger.warning("LLM segment profile failed for %s: %s", segment, err)
        else:
            parsed = self._extractive_profile_fields(evidence)
            narrative = (
                f"LLM unavailable — showing keyword-based profile. "
                f"{self._basic_narrative(cfg['description'], evidence, parsed)}"
            )

        pct = (size_estimate or {}).get(segment, {}).get("pct", 0.0)
        frustrations = [
            _humanize_label(f) for f in (parsed.get("primary_frustrations") or basic.primary_frustrations)
        ]
        pains = [
            _humanize_label(p) for p in (parsed.get("discovery_pain_points") or parsed.get("primary_frustrations") or [])
        ]
        return SegmentProfile(
            segment=segment,
            description=cfg["description"],
            size_estimate_pct=pct,
            behavioral_indicators=cfg["indicators"],
            primary_frustrations=frustrations[:6],
            desired_outcomes=parsed.get("desired_outcomes") or ["More variety", "Better personalization"],
            discovery_pain_points=pains[:5],
            workarounds=workarounds[:5],
            features_mentioned=[
                _humanize_label(f) for f in (parsed.get("features_mentioned") or basic.features_mentioned)
            ],
            recommendation_satisfaction=parsed.get(
                "recommendation_satisfaction", basic.recommendation_satisfaction
            ),
            good_discovery_definition=parsed.get("good_discovery_definition", ""),
            repetitive_triggers=[
                _humanize_label(k) for k in (parsed.get("repetitive_triggers") or basic.repetitive_triggers)
            ],
            sample_quotes=basic.sample_quotes[:4],
            narrative_summary=narrative,
            profile_tier="full",
            listening_habits=listening_habits[:5],
        )

    def build_basic_profiles(self) -> list[SegmentProfile]:
        sizes = self.estimate_sizes()
        return [self.build_basic_profile(name, sizes) for name in self.segments]

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
