"""Prioritize product features from review evidence using RICE and MoSCoW."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Feature backlog mapped to discovery problems seen in reviews.
FEATURE_CATALOG: list[dict] = [
    {
        "feature": "Refresh Discover Weekly & Release Radar more aggressively",
        "problem": "stale_discover_weekly",
        "description": "Surface genuinely new artists each week instead of repeating the same tracks.",
        "reach": 9,
        "impact": 9,
        "confidence": 0.85,
        "effort": 7,
    },
    {
        "feature": "Stronger dislike / not-interested feedback loop",
        "problem": "lack_of_control",
        "description": "Let users explicitly tune recommendations and explain why a song misses.",
        "reach": 8,
        "impact": 8,
        "confidence": 0.9,
        "effort": 5,
    },
    {
        "feature": "Genre bubble breaker & diversity nudges",
        "problem": "filter_bubble",
        "description": "Intentionally introduce adjacent genres and artists outside the user's comfort zone.",
        "reach": 7,
        "impact": 8,
        "confidence": 0.75,
        "effort": 6,
    },
    {
        "feature": "Cold-start onboarding taste quiz",
        "problem": "cold_start_onboarding",
        "description": "Guide new users through artists, moods, and goals before first recommendations.",
        "reach": 6,
        "impact": 9,
        "confidence": 0.8,
        "effort": 5,
    },
    {
        "feature": "Anti-repetition guardrails in radio & mixes",
        "problem": "repetitive_recommendations",
        "description": "Detect loops and over-played artists; rotate fresher alternatives automatically.",
        "reach": 9,
        "impact": 8,
        "confidence": 0.85,
        "effort": 6,
    },
    {
        "feature": "Context-aware mood & activity playlists",
        "problem": "mood_context_mismatch",
        "description": "Match recommendations to workout, focus, party, or relax contexts more reliably.",
        "reach": 8,
        "impact": 7,
        "confidence": 0.7,
        "effort": 6,
    },
    {
        "feature": "Curated starting paths for overwhelmed listeners",
        "problem": "overwhelming_choice",
        "description": "Offer guided entry points instead of dumping users into an infinite catalog.",
        "reach": 6,
        "impact": 7,
        "confidence": 0.75,
        "effort": 4,
    },
    {
        "feature": "Explain-why on recommendations",
        "problem": "poor_personalization",
        "description": "Show why a song was picked and let users correct the signal.",
        "reach": 7,
        "impact": 7,
        "confidence": 0.65,
        "effort": 7,
    },
    {
        "feature": "Social discovery hub (friends, blends, shared playlists)",
        "problem": "poor_personalization",
        "description": "Make friend activity and collaborative lists a first-class discovery surface.",
        "reach": 5,
        "impact": 6,
        "confidence": 0.6,
        "effort": 8,
    },
    {
        "feature": "Deep genre & subgenre exploration mode",
        "problem": "filter_bubble",
        "description": "Help enthusiasts drill into subgenres without leaving their lane entirely.",
        "reach": 5,
        "impact": 6,
        "confidence": 0.7,
        "effort": 5,
    },
]


@dataclass
class FeatureRecommendation:
    feature: str
    description: str
    problem: str
    reach: int
    impact: int
    confidence: float
    effort: int
    rice_score: float
    moscow: str
    evidence_count: int = 0
    example_quote: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConclusionReport:
    summary: str
    features: list[FeatureRecommendation] = field(default_factory=list)
    moscow_groups: dict[str, list[str]] = field(default_factory=dict)
    top_frustrations: list[dict] = field(default_factory=list)
    sample_size: int = 0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "features": [f.to_dict() for f in self.features],
            "moscow_groups": self.moscow_groups,
            "top_frustrations": self.top_frustrations,
            "sample_size": self.sample_size,
        }


def _rice_score(reach: int, impact: int, confidence: float, effort: int) -> float:
    if effort <= 0:
        return 0.0
    return round((reach * impact * confidence) / effort, 2)


def _assign_moscow(rank: int, total: int, rice: float) -> str:
    if rank <= max(1, total // 4) and rice >= 6.0:
        return "Must have"
    if rank <= total // 2 and rice >= 4.0:
        return "Should have"
    if rice >= 2.5:
        return "Could have"
    return "Won't have (this cycle)"


def build_conclusion(
    frustration_counts: dict[str, int],
    examples: dict[str, str],
    sample_size: int,
    llm_summary: str | None = None,
) -> ConclusionReport:
    """Score features from corpus frustration frequency + RICE defaults."""
    scored: list[FeatureRecommendation] = []
    for item in FEATURE_CATALOG:
        problem = item["problem"]
        evidence = frustration_counts.get(problem, 0)
        confidence = min(0.95, item["confidence"] + (0.05 * min(evidence, 4)))
        rice = _rice_score(item["reach"], item["impact"], confidence, item["effort"])
        scored.append(
            FeatureRecommendation(
                feature=item["feature"],
                description=item["description"],
                problem=problem,
                reach=item["reach"],
                impact=item["impact"],
                confidence=round(confidence, 2),
                effort=item["effort"],
                rice_score=rice,
                moscow="",  # filled after sort
                evidence_count=evidence,
                example_quote=examples.get(problem, "")[:220],
            )
        )

    scored.sort(key=lambda f: (-f.rice_score, -f.evidence_count, -f.impact))
    total = len(scored)
    for i, feat in enumerate(scored):
        feat.moscow = _assign_moscow(i + 1, total, feat.rice_score)

    groups: dict[str, list[str]] = {
        "Must have": [],
        "Should have": [],
        "Could have": [],
        "Won't have (this cycle)": [],
    }
    for feat in scored:
        groups[feat.moscow].append(feat.feature)

    top_frustrations = [
        {"frustration": k, "count": frustration_counts[k], "example": examples.get(k, "")}
        for k in sorted(frustration_counts, key=frustration_counts.get, reverse=True)[:8]
    ]

    if llm_summary:
        summary = llm_summary.strip()
    else:
        top = scored[0] if scored else None
        summary = (
            f"Based on {sample_size} indexed reviews, users most need discovery that feels "
            "personal, varied, and controllable. "
            + (
                f"The highest-priority bet is **{top.feature}** "
                f"(RICE {top.rice_score}, {top.moscow.lower()})."
                if top
                else "Fetch more reviews to sharpen prioritization."
            )
        )

    return ConclusionReport(
        summary=summary,
        features=scored,
        moscow_groups=groups,
        top_frustrations=top_frustrations,
        sample_size=sample_size,
    )
