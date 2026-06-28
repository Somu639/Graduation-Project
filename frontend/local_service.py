"""In-process backend for the Streamlit dashboard.

Lets the dashboard run standalone (e.g. on Streamlit Cloud) without a separate
FastAPI server: it uses the dependency-free in-memory vector store and the
LLM-free extractive engine, mirroring the API's response shapes so the UI code
works unchanged.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache


@lru_cache(maxsize=1)
def _components():
    """Build (and seed) the shared in-memory components once per process."""
    os.environ.setdefault("VECTOR_STORE", "memory")
    from rag.vector_store import VectorStore
    from rag.query_engine import DiscoveryInsightEngine
    from agents.segment_analyzer import SegmentAnalyzer
    from api.sample_data import seed_store

    vs = VectorStore(backend="memory")
    if not vs.get_records(limit=1):
        try:
            seed_store(vs)
        except Exception:  # noqa: BLE001 - seeding is best-effort
            pass
    engine = DiscoveryInsightEngine(vector_store=vs)
    seg = SegmentAnalyzer(vector_store=vs, engine=engine)
    return vs, engine, seg


def seed() -> dict:
    from api.sample_data import seed_store

    vs = _components()[0]
    return seed_store(vs)


def stats_overview() -> dict:
    vs = _components()[0]
    records = vs.get_records(limit=10000)
    by_source, by_sentiment, ratings = Counter(), Counter(), Counter()
    rating_vals, dates = [], []
    for r in records:
        m = r.get("metadata", {})
        by_source[m.get("source", "unknown")] += 1
        by_sentiment[m.get("sentiment", "unknown")] += 1
        rt = m.get("rating")
        if isinstance(rt, (int, float)):
            ratings[int(rt)] += 1
            rating_vals.append(int(rt))
        if m.get("date"):
            dates.append(m["date"])
    return {
        "total_reviews": len(records),
        "by_source": dict(by_source),
        "by_sentiment": dict(by_sentiment),
        "ratings_distribution": {str(k): ratings.get(k, 0) for k in range(1, 6)},
        "average_rating": round(sum(rating_vals) / len(rating_vals), 2) if rating_vals else None,
        "date_range": {"earliest": min(dates), "latest": max(dates)} if dates else None,
    }


def stats_timeline() -> dict:
    vs = _components()[0]
    buckets: Counter = Counter()
    for r in vs.get_records(limit=10000):
        d = r.get("metadata", {}).get("date")
        if d and len(d) >= 7:
            buckets[d[:7]] += 1
    return {"granularity": "month", "series": [{"period": k, "count": c} for k, c in sorted(buckets.items())]}


def insights_themes(top_k: int = 20, source: str | None = None, sentiment: str | None = None) -> dict:
    vs = _components()[0]
    filters = {}
    if source:
        filters["source"] = source
    if sentiment:
        filters["sentiment"] = sentiment
    records = vs.get_records(filters=filters or None, limit=5000)
    counts: Counter = Counter()
    theme_sent: dict[str, Counter] = {}
    for r in records:
        m = r.get("metadata", {})
        s = m.get("sentiment", "unknown")
        for t in (x.strip() for x in (m.get("themes", "") or "").split(",") if x.strip()):
            counts[t] += 1
            theme_sent.setdefault(t, Counter())[s] += 1
    themes = [
        {"theme": t, "count": c, "sentiment": dict(theme_sent.get(t, {}))}
        for t, c in counts.most_common(top_k)
    ]
    return {"sample_size": len(records), "themes": themes}


def insights_frustrations(top_k: int = 15) -> dict:
    vs = _components()[0]
    counts: Counter = Counter()
    examples: dict[str, dict] = {}
    for r in vs.get_records(limit=5000):
        m = r.get("metadata", {})
        inds = m.get("frustration_indicators")
        if isinstance(inds, str):
            inds = [i.strip() for i in inds.split(",") if i.strip()]
        for ind in inds or []:
            counts[ind] += 1
            examples.setdefault(ind, {"quote": r["content"][:240], "source": m.get("source", "unknown")})
    ranked = [
        {"frustration": n, "count": c, "example": examples.get(n, {})}
        for n, c in counts.most_common(top_k)
    ]
    return {"sample_size": sum(counts.values()), "frustrations": ranked}


def insights_segments(full: bool = False) -> dict:
    seg = _components()[2]
    out = {"sizes": seg.estimate_sizes(), "comparison_matrix": seg.build_comparison_matrix()}
    if full:
        out["profiles"] = seg.analyze_all()
    return out


def insights_question(question: str, segment: str | None = None, source: str | None = None) -> dict:
    engine = _components()[1]
    filters = {"source": source} if source else None
    phrased = f"For {segment} users: {question}" if segment else question
    return engine.answer_question(phrased, top_k=20, filters=filters).to_dict()


def data_search(q: str | None = None, source: str | None = None, sentiment: str | None = None,
                rating: int | None = None, theme: str | None = None, limit: int = 50) -> dict:
    vs = _components()[0]
    filters = {}
    if source:
        filters["source"] = source
    if sentiment:
        filters["sentiment"] = sentiment
    if rating:
        filters["rating"] = rating
    if theme:
        filters["theme"] = theme
    if q:
        results = vs.similarity_search(q, top_k=limit, filters=filters or None)
    else:
        results = vs.get_records(filters=filters or None, limit=limit)
    return {
        "count": len(results),
        "results": [{"content": r.get("content", ""), "metadata": r.get("metadata", {})} for r in results],
    }


def agent_research(questions: list[str]) -> dict:
    from agents.insight_agent import DiscoveryResearchAgent

    vs, engine, _ = _components()
    return DiscoveryResearchAgent(vector_store=vs, engine=engine).run_research_session(questions).to_dict()


def export_report(fmt: str = "markdown", title: str = "Spotify Discovery Research") -> dict:
    """Build a lightweight Markdown report string from in-process data."""
    stats = stats_overview()
    themes = insights_themes(top_k=15)
    frustrations = insights_frustrations(top_k=10)
    lines = [f"# {title}", "", "## Overview",
             f"- Total reviews: {stats['total_reviews']}",
             f"- Average rating: {stats['average_rating']}",
             f"- By source: {stats['by_source']}",
             f"- By sentiment: {stats['by_sentiment']}", "", "## Top Themes"]
    lines += [f"- {t['theme']} ({t['count']})" for t in themes["themes"]]
    lines += ["", "## Top Frustrations"]
    lines += [f"- {f['frustration']} ({f['count']})" for f in frustrations["frustrations"]]
    return {"format": "markdown", "markdown": "\n".join(lines)}
