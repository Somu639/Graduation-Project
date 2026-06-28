"""Apple App Store review scraper for Spotify.

Built on the ``app_store_scraper`` library. Collects reviews, anonymizes user
names, filters for music-discovery related feedback, supports rate limiting,
retry logic, and incremental ("only new since last run") scrapes, and writes
timestamped JSON + CSV output via the shared :class:`ReviewData` schema.

Run a full scrape:
    python -m scrapers.app_store_scraper

Notes on fields:
    The ``app_store_scraper`` library exposes username, rating, title, review
    text, date, and edit status. It does NOT expose a stable review id,
    app version, or helpful/vote counts. We therefore synthesize a
    deterministic ``review_id`` and default ``version`` / ``helpful_count`` to
    None, populating them defensively if a future library version returns them.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from .schema import (
    DATA_DIR,
    ReviewData,
    anonymize_username,
    make_review_id,
    save_reviews,
    to_iso,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SPOTIFY_APP_ID = 324684580
SPOTIFY_APP_NAME = "spotify"

# Keywords that signal feedback about music discovery (positive or negative).
DISCOVERY_KEYWORDS: tuple[str, ...] = (
    "discover",
    "recommendation",
    "suggest",
    "new music",
    "playlist",
    "algorithm",
    "same songs",
    "repetitive",
    "boring",
    "stuck",
    "bubble",
    "explore",
    "radio",
    "daily mix",
    "release radar",
    "discover weekly",
)

# Incremental-scrape state lives alongside the output data.
STATE_FILE = DATA_DIR / "app_store_state.json"


def _matches_discovery(title: str, text: str) -> bool:
    """True if the title or body mentions any discovery keyword."""
    haystack = f"{title} {text}".lower()
    return any(keyword in haystack for keyword in DISCOVERY_KEYWORDS)


# --------------------------------------------------------------------------- #
# Scraper
# --------------------------------------------------------------------------- #
class AppStoreReviewScraper:
    """Scrape, filter, and persist Spotify App Store reviews."""

    def __init__(
        self,
        app_id: int = SPOTIFY_APP_ID,
        app_name: str = SPOTIFY_APP_NAME,
        country: str = "us",
        request_delay: float = 2.0,
        max_retries: int = 3,
        retry_backoff: float = 5.0,
    ) -> None:
        self.app_id = app_id
        self.app_name = app_name
        self.country = country
        self.request_delay = request_delay  # rate limit: 1 request / N seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    # --- raw fetching with retries -------------------------------------- #
    def _fetch_raw(
        self, how_many: int, after: datetime | None = None
    ) -> list[dict]:
        """Fetch raw review dicts from the library with retry/backoff.

        Args:
            how_many: Target number of reviews to collect.
            after: If provided, stop once reviews older than this datetime are
                reached (used for incremental scrapes).

        Returns:
            The list of raw review dicts produced by ``app_store_scraper``.
        """
        try:
            from app_store_scraper import AppStore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "app-store-scraper is required. Install it via requirements.txt "
                "(pip install app-store-scraper)."
            ) from exc

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Fresh instance each attempt so retries don't double-append.
                app = AppStore(
                    country=self.country,
                    app_name=self.app_name,
                    app_id=self.app_id,
                )
                # ``sleep`` enforces the per-request rate limit inside the lib.
                app.review(
                    how_many=how_many,
                    after=after,
                    sleep=self.request_delay,
                )
                logger.info(
                    "Fetched %d raw reviews (attempt %d)",
                    len(app.reviews),
                    attempt,
                )
                return app.reviews
            except Exception as exc:  # noqa: BLE001 - library raises broadly
                last_error = exc
                wait = self.retry_backoff * attempt
                logger.warning(
                    "Fetch attempt %d/%d failed: %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("All %d fetch attempts failed.", self.max_retries)
        if last_error:
            raise last_error
        return []

    # --- normalization --------------------------------------------------- #
    @staticmethod
    def _normalize(raw: dict) -> ReviewData:
        """Convert a raw library review dict into :class:`ReviewData`."""
        username = raw.get("userName", "") or ""
        date_str = to_iso(raw.get("date"))
        title = raw.get("title", "") or ""
        review_text = raw.get("review", "") or ""

        return ReviewData(
            review_id=make_review_id(
                "app_store", username, date_str or "", review_text
            ),
            source="app_store",
            username=anonymize_username(username),
            rating=raw.get("rating"),
            title=title,
            review_text=review_text,
            date=date_str,
            # Not exposed by app_store_scraper today; populated if available.
            version=raw.get("version") or raw.get("appVersion"),
            helpful_count=raw.get("helpful_count") or raw.get("voteCount"),
        )

    # --- public scrape API ---------------------------------------------- #
    def scrape(
        self,
        how_many: int = 5000,
        keyword_filter: bool = True,
        after: datetime | None = None,
    ) -> list[dict]:
        """Scrape reviews and optionally filter to discovery-related ones.

        Args:
            how_many: Maximum number of reviews to fetch.
            keyword_filter: If True, keep only reviews mentioning discovery
                keywords.
            after: Only return reviews newer than this datetime.

        Returns:
            A list of review dictionaries in the unified schema.
        """
        raw_reviews = self._fetch_raw(how_many=how_many, after=after)
        normalized = [self._normalize(r) for r in raw_reviews]

        if keyword_filter:
            normalized = [
                r
                for r in normalized
                if _matches_discovery(r.title or "", r.review_text)
            ]
            logger.info(
                "Kept %d discovery-related reviews out of %d total",
                len(normalized),
                len(raw_reviews),
            )

        return [r.model_dump() for r in normalized]

    def scrape_incremental(
        self, how_many: int = 5000, keyword_filter: bool = True
    ) -> list[dict]:
        """Scrape only reviews newer than the last successful run.

        Reads the last-run timestamp from the state file, fetches reviews
        after it, then advances the saved state to the newest review seen.

        Returns:
            A list of newly collected review dictionaries.
        """
        state = self._load_state()
        after = None
        if state.get("last_review_date"):
            try:
                after = datetime.fromisoformat(state["last_review_date"])
                logger.info("Incremental scrape: fetching reviews after %s", after)
            except ValueError:
                logger.warning("Invalid stored timestamp; doing a full scrape.")

        reviews = self.scrape(
            how_many=how_many, keyword_filter=keyword_filter, after=after
        )

        newest = self._newest_date(reviews)
        if newest:
            state["last_review_date"] = newest
            state["last_run_at"] = datetime.utcnow().isoformat()
            state["last_run_count"] = len(reviews)
            self._save_state(state)
            logger.info("State advanced to %s (%d new reviews)", newest, len(reviews))
        else:
            logger.info("No new reviews since last run.")

        return reviews

    # --- persistence ----------------------------------------------------- #
    @staticmethod
    def _newest_date(reviews: list[dict]) -> str | None:
        dates = [r["date"] for r in reviews if r.get("date")]
        return max(dates) if dates else None

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read state file: %s", exc)
        return {}

    def _save_state(self, state: dict) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def save(self, reviews: list[dict], prefix: str = "spotify_app_store_reviews"):
        """Persist reviews to timestamped JSON + CSV. Returns the file paths."""
        return save_reviews(reviews, prefix=prefix)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scraper = AppStoreReviewScraper()
    collected = scraper.scrape(how_many=5000, keyword_filter=True)
    print(f"Collected {len(collected)} discovery-related reviews")
    if collected:
        scraper.save(collected)
