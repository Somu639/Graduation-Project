"""Fetch live Spotify reviews and run them through the analysis pipeline."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_SOURCES = ("play_store", "app_store", "reddit", "twitter")


def scrape_live(
    sources: list[str],
    limit: int = 50,
    *,
    discovery_filter: bool = False,
) -> tuple[list[dict], list[str]]:
    """Scrape reviews from the requested sources.

    Returns:
        (records, warnings) — warnings describe skipped sources or missing deps.
    """
    collected: list[dict] = []
    warnings: list[str] = []

    for source in sources:
        try:
            if source == "play_store":
                from scrapers.play_store_scraper import PlayStoreReviewScraper

                collected += PlayStoreReviewScraper().scrape(how_many=limit, sort="newest")
            elif source == "app_store":
                from scrapers.app_store_scraper import AppStoreReviewScraper

                collected += AppStoreReviewScraper().scrape(
                    how_many=limit,
                    keyword_filter=discovery_filter,
                )
            elif source == "reddit":
                if not _reddit_configured():
                    warnings.append("reddit: set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
                    continue
                from scrapers.reddit_scraper import RedditScraper

                collected += RedditScraper().scrape(limit_per_query=min(limit, 100))
            elif source == "twitter":
                if not os.getenv("TWITTER_BEARER_TOKEN"):
                    warnings.append("twitter: set TWITTER_BEARER_TOKEN in .env")
                    continue
                from scrapers.twitter_scraper import TwitterScraper

                collected += TwitterScraper().scrape(max_results=limit)
            else:
                warnings.append(f"unknown source: {source}")
        except ImportError as exc:
            warnings.append(f"{source}: missing dependency ({exc})")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scrape failed for %s", source)
            warnings.append(f"{source}: {exc}")

    return collected, warnings


def _reddit_configured() -> bool:
    return bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))


def ingest_reviews(
    records: list[dict],
    vector_store: VectorStore,
    *,
    use_llm: bool = False,
) -> dict:
    """Clean, analyze, and index review records into the vector store."""
    from processors.llm_client import llm_configured
    from processors.sentiment_analyzer import SentimentAnalyzer
    from processors.text_cleaner import process_records
    from processors.theme_extractor import ThemeExtractor

    if not records:
        return {"scraped": 0, "processed": 0, "indexed": 0, "theme_counts": {}}

    cleaned = process_records(records, drop_non_english=True)
    llm_on = use_llm and llm_configured()
    records_out = SentimentAnalyzer(use_llm=llm_on).analyze_batch(cleaned, use_llm=llm_on)

    theme_strategy = "llm" if llm_on else "keyword"
    theme_result = ThemeExtractor(strategy=theme_strategy).extract(records_out)
    records_out = theme_result.get("records", records_out)

    indexed = vector_store.build_indexes(records_out)
    return {
        "scraped": len(records),
        "processed": len(records_out),
        "indexed": indexed,
        "theme_counts": theme_result.get("theme_counts", {}),
        "llm_used": llm_on,
    }


def fetch_and_ingest(
    sources: list[str],
    limit: int,
    vector_store: VectorStore,
    *,
    use_llm: bool = False,
    discovery_filter: bool = False,
) -> dict:
    """Scrape live reviews, analyze them, and index into the vector store."""
    records, warnings = scrape_live(sources, limit, discovery_filter=discovery_filter)
    summary = ingest_reviews(records, vector_store, use_llm=use_llm)
    summary["warnings"] = warnings
    summary["sources"] = sources
    return summary
