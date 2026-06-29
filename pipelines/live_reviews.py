"""Fetch live Spotify reviews and run them through the analysis pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_SOURCES = ("play_store", "app_store", "reddit", "twitter", "community_forums", "social_media")


def scrape_live(
    sources: list[str],
    limit: int = 50,
    *,
    discovery_filter: bool = False,
) -> tuple[list[dict], list[str], dict[str, int]]:
    """Scrape reviews from the requested sources.

    Returns:
        (records, warnings, source_counts) — per-source fetch counts and any warnings.
    """
    from scrapers.source_registry import normalize_sources, source_label

    sources = normalize_sources(sources)
    collected: list[dict] = []
    warnings: list[str] = []
    source_counts: dict[str, int] = {}

    for source in sources:
        before = len(collected)
        try:
            if source == "play_store":
                from scrapers.play_store_scraper import PlayStoreReviewScraper

                batch = PlayStoreReviewScraper().scrape(how_many=limit, sort="newest")
                collected += batch
            elif source == "app_store":
                from scrapers.app_store_scraper import AppStoreReviewScraper

                batch = AppStoreReviewScraper().scrape(
                    how_many=limit,
                    keyword_filter=discovery_filter,
                )
                collected += batch
            elif source == "reddit":
                from scrapers.reddit_scraper import RedditScraper

                batch = RedditScraper().scrape(limit_per_query=limit)
                collected += batch
            elif source == "twitter":
                from scrapers.twitter_scraper import TwitterScraper

                batch = TwitterScraper().scrape(max_results=limit)
                collected += batch
            else:
                warnings.append(f"unknown source: {source}")
                source_counts[source] = 0
                continue

            count = len(collected) - before
            source_counts[source] = count
            if count == 0:
                warnings.append(
                    f"{source_label(source)}: scraper returned 0 reviews (check network or filters)"
                )
        except ImportError as exc:
            warnings.append(f"{source}: missing dependency — {exc}")
            source_counts[source] = 0
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scrape failed for %s", source)
            warnings.append(f"{source}: {exc}")
            source_counts[source] = 0

    return collected, warnings, source_counts


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
    records, warnings, source_counts = scrape_live(
        sources, limit, discovery_filter=discovery_filter
    )
    summary = ingest_reviews(records, vector_store, use_llm=use_llm)
    summary["warnings"] = warnings
    summary["source_counts"] = source_counts
    summary["sources"] = sources
    if not records and warnings:
        summary["error"] = "No reviews fetched. See warnings for details."
    return summary
