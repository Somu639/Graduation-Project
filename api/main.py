"""FastAPI application entry point for the Spotify Discovery Analyzer.

Initializes shared, expensive resources once via a lifespan handler (vector
store, embedding model, LLM-backed engine, and agent instances), enables CORS,
adds request logging and global error handling, and mounts the API routes.

Run locally with:
    uvicorn api.main:app --reload

Interactive docs are served at /docs.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.routes import router

WEB_DIR = Path(__file__).resolve().parent.parent / "frontend" / "web"

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup; clean up on shutdown."""
    logger.info("Starting up: initializing shared resources...")

    # Imported here so app import stays light and failures are localized.
    from rag.embeddings import ReviewEmbedder
    from rag.vector_store import VectorStore
    from rag.query_engine import DiscoveryInsightEngine
    from agents.insight_agent import DiscoveryResearchAgent
    from agents.segment_analyzer import SegmentAnalyzer

    # Embedding model + vector store connection (clients connect lazily on use).
    embedder = ReviewEmbedder()
    vector_store = VectorStore(embedder=embedder)
    # LLM-backed engine + agents share the single vector store / embedder.
    engine = DiscoveryInsightEngine(vector_store=vector_store)
    research_agent = DiscoveryResearchAgent(vector_store=vector_store, engine=engine)
    segment_analyzer = SegmentAnalyzer(vector_store=vector_store, engine=engine)

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.engine = engine
    app.state.research_agent = research_agent
    app.state.segment_analyzer = segment_analyzer

    # Auto-seed sample data for an instant live demo (in-memory backend only).
    if vector_store.backend == "memory":
        try:
            from api.sample_data import seed_store

            if not vector_store.get_records(limit=1):
                summary = seed_store(vector_store)
                logger.info("Auto-seeded sample data: %s", summary.get("seeded"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-seed skipped: %s", exc)

    logger.info(
        "Startup complete (embeddings=%s, vector_store=%s).",
        embedder.provider,
        vector_store.backend,
    )
    try:
        yield
    finally:
        logger.info("Shutting down.")


app = FastAPI(
    title="Spotify Discovery Analyzer",
    description=(
        "Scrape, process, and analyze user feedback about Spotify's music "
        "discovery experience using a RAG + agent pipeline."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each request's method, path, status, and duration."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - ensure unhandled errors are logged
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception("%s %s failed after %.1fms", request.method, request.url.path, elapsed)
        raise
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a consistent JSON envelope for unhandled errors."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    """Serve the single-page web dashboard."""
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"service": "spotify-discovery-analyzer", "docs": "/docs"})


@app.get("/app", include_in_schema=False)
def web_app():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health", tags=["health"])
def health(request: Request) -> dict:
    """Health check reporting which resources are initialized."""
    state = request.app.state
    return {
        "status": "healthy",
        "vector_store": getattr(getattr(state, "vector_store", None), "backend", None),
        "embeddings": getattr(getattr(state, "embedder", None), "provider", None),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
