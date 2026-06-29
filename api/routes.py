"""API routes for the Spotify Discovery Analyzer.

Endpoints cover data ingestion (async job tracking), RAG-backed insights,
theme/segment/frustration analysis, the autonomous research agent, corpus
statistics, and report export. Includes an in-memory/Redis cache, a simple
per-client rate limiter for LLM-heavy endpoints, and consistent error handling.

Shared resources (vector store, engine, agents) are created once in the app
lifespan handler and accessed via ``request.app.state``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("api.routes")

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = DATA_DIR / "reports"

# In-memory job registry for async review processing.
JOBS: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# Caching (Redis if REDIS_URL set, else in-memory TTL cache)
# --------------------------------------------------------------------------- #
class CacheStore:
    """Tiny cache abstraction over Redis or an in-process dict with TTL."""

    def __init__(self, ttl: int = 300) -> None:
        self.ttl = ttl
        self._mem: dict[str, tuple[float, object]] = {}
        self._redis = None
        url = os.getenv("REDIS_URL")
        if url:
            try:
                import redis  # type: ignore

                self._redis = redis.from_url(url, decode_responses=True)
                self._redis.ping()
                logger.info("Cache: using Redis at %s", url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable (%s); using in-memory cache.", exc)
                self._redis = None

    @staticmethod
    def make_key(*parts) -> str:
        return "sda:" + ":".join(str(p) for p in parts)

    def get(self, key: str):
        if self._redis is not None:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        entry = self._mem.get(key)
        if not entry:
            return None
        expiry, value = entry
        if expiry < time.time():
            self._mem.pop(key, None)
            return None
        return value

    def set(self, key: str, value, ttl: int | None = None) -> None:
        ttl = ttl or self.ttl
        if self._redis is not None:
            self._redis.setex(key, ttl, json.dumps(value, default=str))
        else:
            self._mem[key] = (time.time() + ttl, value)


CACHE = CacheStore(ttl=int(os.getenv("CACHE_TTL", "300")))


def cached(key: str, producer, ttl: int | None = None):
    """Return a cached value or compute, store, and return it."""
    hit = CACHE.get(key)
    if hit is not None:
        logger.info("cache hit: %s", key)
        return hit
    value = producer()
    CACHE.set(key, value, ttl)
    return value


# --------------------------------------------------------------------------- #
# Rate limiting (per-client sliding window) for LLM-heavy endpoints
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Sliding-window rate limiter usable as a FastAPI dependency."""

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self._hits: dict[str, list[float]] = {}

    def __call__(self, request: Request) -> None:
        client = request.client.host if request.client else "anonymous"
        now = time.time()
        window = [t for t in self._hits.get(client, []) if now - t < self.period]
        if len(window) >= self.max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({self.max_calls}/{int(self.period)}s).",
            )
        window.append(now)
        self._hits[client] = window


llm_rate_limiter = RateLimiter(
    max_calls=int(os.getenv("LLM_RATE_LIMIT", "20")),
    period=float(os.getenv("LLM_RATE_PERIOD", "60")),
)


# --------------------------------------------------------------------------- #
# State accessors
# --------------------------------------------------------------------------- #
def get_engine(request: Request):
    return request.app.state.engine


def get_vector_store(request: Request):
    return request.app.state.vector_store


def get_research_agent(request: Request):
    return request.app.state.research_agent


def get_segment_analyzer(request: Request):
    return request.app.state.segment_analyzer


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class AnalyzeReviewsRequest(BaseModel):
    reviews: list[dict] = Field(..., description="Raw or partially-cleaned review records.")
    run_sentiment: bool = True
    run_themes: bool = True
    build_indexes: bool = True


class ResearchRequest(BaseModel):
    research_questions: list[str] = Field(..., min_length=1)


class ExportRequest(BaseModel):
    format: str = Field(default="markdown", description="markdown or pdf")
    title: str = "Spotify Music Discovery — Research Report"


class FetchLiveRequest(BaseModel):
    sources: list[str] = Field(default=["play_store"])
    limit: int = Field(default=50, ge=1, le=200)
    use_llm: bool = True
    discovery_filter: bool = False


class ScrapeRequest(BaseModel):
    sources: list[str] = Field(default=["app_store", "play_store"])
    limit: int = Field(default=100, ge=1, le=1000)


@router.post("/data/fetch-live", tags=["data"])
def fetch_live(req: FetchLiveRequest, request: Request) -> dict:
    """Scrape live reviews, run LLM/VADER analysis, and index into the store."""
    from pipelines.live_reviews import fetch_and_ingest

    try:
        summary = fetch_and_ingest(
            req.sources,
            req.limit,
            get_vector_store(request),
            use_llm=req.use_llm,
            discovery_filter=req.discovery_filter,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch-live failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    CACHE._mem.clear()
    return summary


# --------------------------------------------------------------------------- #
# Data collection + async processing
# --------------------------------------------------------------------------- #
@router.post("/scrape", tags=["data"])
def scrape(req: ScrapeRequest) -> dict:
    """Run the requested scrapers and return collected records."""
    collected: list[dict] = []
    try:
        if "app_store" in req.sources:
            from scrapers.app_store_scraper import AppStoreReviewScraper

            collected += AppStoreReviewScraper().scrape(how_many=req.limit, keyword_filter=False)
        if "play_store" in req.sources:
            from scrapers.play_store_scraper import PlayStoreReviewScraper

            collected += PlayStoreReviewScraper().scrape(how_many=req.limit, sort="newest")
        if "reddit" in req.sources:
            from scrapers.reddit_scraper import RedditScraper

            collected += RedditScraper().scrape(limit_per_query=req.limit)
        if "twitter" in req.sources:
            from scrapers.twitter_scraper import TwitterScraper

            collected += TwitterScraper().scrape(max_results=req.limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scrape failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"count": len(collected), "records": collected}


def _process_reviews_job(job_id: str, payload: dict, state) -> None:
    """Background worker: clean -> sentiment -> themes -> index."""
    JOBS[job_id]["status"] = "running"
    try:
        from processors.text_cleaner import process_records

        records = process_records(payload["reviews"], drop_non_english=True)

        if payload.get("run_sentiment"):
            from processors.sentiment_analyzer import SentimentAnalyzer

            records = SentimentAnalyzer().analyze_batch(records)

        theme_counts: dict = {}
        if payload.get("run_themes"):
            from processors.theme_extractor import ThemeExtractor

            result = ThemeExtractor().extract(records)
            theme_counts = result.get("theme_counts", {})
            records = result.get("records", records)

        indexed: dict | int = 0
        if payload.get("build_indexes"):
            indexed = state.vector_store.build_indexes(records)
        else:
            indexed = state.vector_store.add_documents(records)

        JOBS[job_id].update(
            status="completed",
            finished_at=datetime.utcnow().isoformat(),
            result={
                "processed": len(records),
                "indexed": indexed,
                "theme_counts": theme_counts,
            },
        )
        logger.info("Job %s completed (%d records)", job_id, len(records))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        JOBS[job_id].update(status="failed", error=str(exc))


@router.post("/analyze/reviews", tags=["data"])
def analyze_reviews(
    req: AnalyzeReviewsRequest, request: Request, background: BackgroundTasks
) -> dict:
    """Submit reviews for async processing; returns a job_id to track progress."""
    job_id = uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "submitted_at": datetime.utcnow().isoformat(),
        "count": len(req.reviews),
        "result": None,
        "error": None,
    }
    background.add_task(_process_reviews_job, job_id, req.model_dump(), request.app.state)
    return {"job_id": job_id, "status": "queued", "count": len(req.reviews)}


@router.post("/data/seed", tags=["data"])
def seed_sample_data(request: Request) -> dict:
    """Load a built-in sample dataset through the pipeline (for local demos)."""
    from api.sample_data import seed_store

    try:
        summary = seed_store(get_vector_store(request))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Seeding failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # Invalidate cached aggregates so fresh data shows up.
    CACHE._mem.clear()
    return summary


@router.get("/analyze/status/{job_id}", tags=["data"])
def job_status(job_id: str) -> dict:
    """Check the status/result of an async processing job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, **job}


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
@router.get("/insights/question", tags=["insights"], dependencies=[Depends(llm_rate_limiter)])
def insights_question(
    request: Request,
    question: str = Query(..., min_length=3),
    segment: str | None = None,
    source: str | None = None,
) -> dict:
    """Answer a question with a synthesized, evidence-backed insight."""
    engine = get_engine(request)
    filters = {"source": source} if source else None
    phrased = f"For {segment} users: {question}" if segment else question

    def produce():
        try:
            return engine.answer_question(phrased, top_k=20, filters=filters).to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning("insights_question LLM failed: %s", exc)
            results = engine._retrieve(phrased, top_k=20, filters=filters)  # noqa: SLF001
            if results:
                fb = engine._extractive_insight(phrased, results)  # noqa: SLF001
                fb.llm_fallback = True
                fb.llm_error = str(exc)
                return fb.to_dict()
            return {
                "question": phrased,
                "insight": "No indexed feedback is available to answer this question.",
                "confidence": 0.0,
                "supporting_evidence": [],
                "sample_size": 0,
                "themes_identified": [],
                "recommended_followup_questions": [],
                "llm_fallback": True,
                "llm_error": str(exc),
            }

    key = CacheStore.make_key("insight", phrased, source or "all")
    try:
        return cached(key, produce)
    except Exception as exc:  # noqa: BLE001
        logger.exception("insights_question failed")
        return produce()  # bypass cache on error; produce() handles fallback internally


@router.get("/insights/themes", tags=["insights"])
def insights_themes(
    request: Request,
    top_k: int = 20,
    source: str | None = None,
    sentiment: str | None = None,
) -> dict:
    """Return top themes across the corpus with counts and sentiment breakdown."""
    store = get_vector_store(request)
    filters: dict = {}
    if source:
        filters["source"] = source
    if sentiment:
        filters["sentiment"] = sentiment

    def produce():
        records = store.get_records(filters=filters or None, limit=5000)
        theme_sentiment: dict[str, Counter] = {}
        counts: Counter = Counter()
        for r in records:
            meta = r.get("metadata", {})
            sentiment = meta.get("sentiment", "unknown")
            for theme in (t.strip() for t in (meta.get("themes", "") or "").split(",") if t.strip()):
                counts[theme] += 1
                theme_sentiment.setdefault(theme, Counter())[sentiment] += 1
        themes = [
            {
                "theme": theme,
                "count": count,
                "sentiment": dict(theme_sentiment.get(theme, {})),
            }
            for theme, count in counts.most_common(top_k)
        ]
        return {"sample_size": len(records), "themes": themes}

    return cached(CacheStore.make_key("themes", top_k, source or "all", sentiment or "all"), produce)


@router.get("/insights/segments", tags=["insights"], dependencies=[Depends(llm_rate_limiter)])
def insights_segments(request: Request, full: bool = False) -> dict:
    """Return user-segment sizing, a problem-comparison matrix, and (optionally)
    full LLM-built segment profiles."""
    analyzer = get_segment_analyzer(request)

    def produce():
        out = {
            "sizes": analyzer.estimate_sizes(),
            "comparison_matrix": analyzer.build_comparison_matrix(),
        }
        if full:
            out["profiles"] = analyzer.analyze_all()
        return out

    return cached(CacheStore.make_key("segments", full), produce)


@router.get("/insights/frustrations", tags=["insights"])
def insights_frustrations(request: Request, top_k: int = 15) -> dict:
    """Return ranked discovery frustrations with representative quotes."""
    store = get_vector_store(request)

    def produce():
        records = store.get_records(limit=5000)
        counts: Counter = Counter()
        examples: dict[str, dict] = {}
        for r in records:
            meta = r.get("metadata", {})
            indicators = meta.get("frustration_indicators")
            # Stored as a comma-joined string for cross-backend compatibility.
            if isinstance(indicators, str):
                indicators = [i.strip() for i in indicators.split(",") if i.strip()]
            for ind in indicators or []:
                counts[ind] += 1
                examples.setdefault(
                    ind,
                    {"quote": r["content"][:240], "source": meta.get("source", "unknown")},
                )
        ranked = [
            {"frustration": name, "count": count, "example": examples.get(name, {})}
            for name, count in counts.most_common(top_k)
        ]
        return {"sample_size": len(records), "frustrations": ranked}

    return cached(CacheStore.make_key("frustrations", top_k), produce)


# --------------------------------------------------------------------------- #
# Research agent
# --------------------------------------------------------------------------- #
@router.post("/agent/research", tags=["agent"], dependencies=[Depends(llm_rate_limiter)])
def agent_research(req: ResearchRequest, request: Request) -> dict:
    """Run an autonomous multi-question research session."""
    agent = get_research_agent(request)
    key = CacheStore.make_key("research", "|".join(sorted(req.research_questions)))

    def produce():
        return agent.run_research_session(req.research_questions).to_dict()

    try:
        return cached(key, produce, ttl=1800)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_research failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
@router.get("/stats/overview", tags=["stats"])
def stats_overview(request: Request) -> dict:
    """Return corpus statistics: counts, distributions, ratings, date range."""
    store = get_vector_store(request)

    def produce():
        records = store.get_records(limit=10000)
        by_source: Counter = Counter()
        by_sentiment: Counter = Counter()
        ratings: Counter = Counter()
        rating_vals: list[int] = []
        dates: list[str] = []
        for r in records:
            meta = r.get("metadata", {})
            by_source[meta.get("source", "unknown")] += 1
            by_sentiment[meta.get("sentiment", "unknown")] += 1
            rating = meta.get("rating")
            if isinstance(rating, (int, float)):
                ratings[int(rating)] += 1
                rating_vals.append(int(rating))
            if meta.get("date"):
                dates.append(meta["date"])
        avg_rating = round(sum(rating_vals) / len(rating_vals), 2) if rating_vals else None
        return {
            "total_reviews": len(records),
            "by_source": dict(by_source),
            "by_sentiment": dict(by_sentiment),
            "ratings_distribution": {str(k): ratings.get(k, 0) for k in range(1, 6)},
            "average_rating": avg_rating,
            "date_range": {"earliest": min(dates), "latest": max(dates)} if dates else None,
        }

    return cached(CacheStore.make_key("stats"), produce, ttl=120)


@router.get("/stats/timeline", tags=["stats"])
def stats_timeline(request: Request) -> dict:
    """Return review volume bucketed by month (YYYY-MM)."""
    store = get_vector_store(request)

    def produce():
        records = store.get_records(limit=10000)
        buckets: Counter = Counter()
        for r in records:
            date = r.get("metadata", {}).get("date")
            if date and len(date) >= 7:
                buckets[date[:7]] += 1
        series = [{"period": k, "count": c} for k, c in sorted(buckets.items())]
        return {"granularity": "month", "series": series}

    return cached(CacheStore.make_key("timeline"), produce, ttl=120)


# --------------------------------------------------------------------------- #
# Raw data search / browse
# --------------------------------------------------------------------------- #
@router.get("/data/search", tags=["data"])
def data_search(
    request: Request,
    q: str | None = None,
    source: str | None = None,
    sentiment: str | None = None,
    rating: int | None = Query(default=None, ge=1, le=5),
    theme: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    """Search (semantic if ``q`` given) or browse reviews with metadata filters."""
    store = get_vector_store(request)
    filters: dict = {}
    if source:
        filters["source"] = source
    if sentiment:
        filters["sentiment"] = sentiment
    if rating:
        filters["rating"] = rating
    if theme:
        filters["theme"] = theme

    if q:
        results = store.similarity_search(q, top_k=limit, filters=filters or None)
    else:
        results = store.get_records(filters=filters or None, limit=limit)
    return {
        "count": len(results),
        "results": [
            {"content": r.get("content", ""), "metadata": r.get("metadata", {})}
            for r in results
        ],
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _build_markdown_report(request: Request, title: str) -> str:
    """Assemble a Markdown research report from corpus stats and insights."""
    stats = stats_overview(request)
    themes = insights_themes(request, top_k=15)
    frustrations = insights_frustrations(request, top_k=10)
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [f"# {title}", "", f"_Generated: {generated}_", ""]

    lines += ["## Corpus Overview", ""]
    lines.append(f"- **Total reviews:** {stats['total_reviews']}")
    lines.append(f"- **Average rating:** {stats['average_rating']}")
    lines.append(f"- **By source:** {stats['by_source']}")
    lines.append(f"- **By sentiment:** {stats['by_sentiment']}")
    lines.append(f"- **Ratings distribution:** {stats['ratings_distribution']}")
    if stats.get("date_range"):
        lines.append(f"- **Date range:** {stats['date_range']['earliest']} → {stats['date_range']['latest']}")
    lines.append("")

    lines += ["## Top Themes", ""]
    for t in themes["themes"]:
        lines.append(f"- **{t['theme']}** ({t['count']}) — sentiment {t['sentiment']}")
    lines.append("")

    lines += ["## Top Frustrations", ""]
    for f in frustrations["frustrations"]:
        ex = f.get("example", {})
        lines.append(f"- **{f['frustration']}** ({f['count']})")
        if ex.get("quote"):
            lines.append(f"  > \"{ex['quote']}\" — _{ex.get('source','?')}_")
    lines.append("")

    return "\n".join(lines)


@router.post("/export/report", tags=["export"])
def export_report(req: ExportRequest, request: Request):
    """Generate a Markdown (default) or PDF research report and return the file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        markdown = _build_markdown_report(request, req.title)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if req.format.lower() == "pdf":
        pdf_path = REPORTS_DIR / f"report_{timestamp}.pdf"
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)
            for line in markdown.splitlines():
                # Latin-1 safe text for the core PDF fonts.
                safe = line.encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 6, safe)
            pdf.output(str(pdf_path))
            return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
        except ImportError:
            logger.warning("fpdf2 not installed; falling back to Markdown export.")

    md_path = REPORTS_DIR / f"report_{timestamp}.md"
    md_path.write_text(markdown, encoding="utf-8")
    return FileResponse(md_path, media_type="text/markdown", filename=md_path.name)
