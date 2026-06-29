"""Review fetch source registry — IDs, labels, and normalization for the UI and pipeline."""

from __future__ import annotations

# Canonical stored source values in ReviewData.metadata["source"]
CANONICAL_SOURCES = ("app_store", "play_store", "reddit", "twitter")

# Options shown in the Live Reviews fetcher (multiselect).
FETCH_SOURCES: dict[str, dict[str, str]] = {
    "play_store": {
        "label": "Google Play Store",
        "category": "App stores",
        "canonical": "play_store",
    },
    "app_store": {
        "label": "Apple App Store",
        "category": "App stores",
        "canonical": "app_store",
    },
    "community_forums": {
        "label": "Community forums (Reddit)",
        "category": "Community & social",
        "canonical": "reddit",
    },
    "social_media": {
        "label": "Social media conversations (X / Twitter)",
        "category": "Community & social",
        "canonical": "twitter",
    },
}

# Legacy IDs still accepted from API / old sessions.
_ALIASES: dict[str, str] = {
    "reddit": "reddit",
    "twitter": "twitter",
    "community_forums": "reddit",
    "social_media": "twitter",
    "play_store": "play_store",
    "app_store": "app_store",
}

FETCH_SOURCE_IDS: tuple[str, ...] = tuple(FETCH_SOURCES.keys())

# Filter dropdown: stored metadata values + friendly grouping.
FILTER_SOURCES: list[tuple[str, str]] = [
    ("all", "All sources"),
    ("app_store", "Apple App Store"),
    ("play_store", "Google Play Store"),
    ("reddit", "Community forums (Reddit)"),
    ("twitter", "Social media (X / Twitter)"),
]


def canonical_source(source_id: str) -> str:
    """Map UI/API source id to the value stored on each review."""
    if source_id in _ALIASES:
        return _ALIASES[source_id]
    meta = FETCH_SOURCES.get(source_id)
    if meta:
        return meta["canonical"]
    return source_id


def normalize_sources(sources: list[str]) -> list[str]:
    """Deduplicate and map fetch ids to canonical scraper ids."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in sources:
        canon = canonical_source(raw)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def source_label(source_id: str) -> str:
    """Human-readable label for a canonical or fetch source id."""
    if source_id in FETCH_SOURCES:
        return FETCH_SOURCES[source_id]["label"]
    canon_labels = {
        "app_store": "Apple App Store",
        "play_store": "Google Play Store",
        "reddit": "Community forums (Reddit)",
        "twitter": "Social media conversations (X / Twitter)",
    }
    return canon_labels.get(source_id, source_id.replace("_", " ").title())


def fetch_source_help() -> str:
    lines = []
    for sid in FETCH_SOURCE_IDS:
        meta = FETCH_SOURCES[sid]
        lines.append(f"• **{meta['label']}**")
    return (
        "App store ratings plus community and social feedback. "
        "Forums and social sources work without extra API keys.\n\n"
        + "\n".join(lines)
    )
