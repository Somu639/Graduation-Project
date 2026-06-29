"""Google Play Store review scraper for Spotify.

Built on the ``google_play_scraper`` library. Fetches large volumes of reviews
via continuation-token pagination, supports multiple sort orders, captures
helpfulness (thumbs-up) metrics, and normalizes everything into the shared
:class:`ReviewData` schema so its output matches the App Store scraper exactly.

Run a full scrape:
    python -m scrapers.play_store_scraper
"""

from __future__ import annotations

import logging
import time

from .schema import (
    ReviewData,
    anonymize_username,
    make_review_id,
    save_reviews,
    to_iso,
)

logger = logging.getLogger(__name__)

# Spotify's Android package name.
SPOTIFY_PACKAGE = "com.spotify.music"

# Supported user-facing sort options.
SortOption = str  # one of: "most_relevant", "newest", "rating"
SORT_OPTIONS: tuple[str, ...] = ("most_relevant", "newest", "rating")


class PlayStoreReviewScraper:
    """Scrape, normalize, and persist Spotify Google Play reviews."""

    def __init__(
        self,
        package_name: str = SPOTIFY_PACKAGE,
        lang: str = "en",
        country: str = "us",
        request_delay: float = 2.0,
        max_retries: int = 3,
        retry_backoff: float = 5.0,
        batch_size: int = 200,
    ) -> None:
        self.package_name = package_name
        self.lang = lang  # English only for v1
        self.country = country
        self.request_delay = request_delay  # rate limit between pages
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.batch_size = batch_size

    # --- sort handling --------------------------------------------------- #
    def _sort_enum(self, sort: str):
        """Map a user-facing sort option to a ``google_play_scraper`` Sort.

        The library only natively supports MOST_RELEVANT and NEWEST. For
        "rating" we fetch by relevance and sort client-side (see :meth:`scrape`).
        """
        from google_play_scraper import Sort

        mapping = {
            "most_relevant": Sort.MOST_RELEVANT,
            "newest": Sort.NEWEST,
            "rating": Sort.MOST_RELEVANT,
        }
        if sort not in mapping:
            raise ValueError(
                f"Invalid sort '{sort}'. Choose from: {', '.join(SORT_OPTIONS)}"
            )
        return mapping[sort]

    # --- paginated fetch with retries ----------------------------------- #
    def _fetch_page(self, sort_enum, continuation_token):
        """Fetch one page of reviews with retry/backoff.

        Returns:
            A ``(results, continuation_token)`` tuple. On total failure returns
            ``([], None)`` so the caller stops cleanly.
        """
        try:
            from google_play_scraper import reviews
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "google-play-scraper is required. Install it via requirements.txt"
            ) from exc

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if continuation_token is not None:
                    # Token carries lang/country/sort/count from the first call.
                    result, token = reviews(
                        self.package_name, continuation_token=continuation_token
                    )
                else:
                    result, token = reviews(
                        self.package_name,
                        lang=self.lang,
                        country=self.country,
                        sort=sort_enum,
                        count=self.batch_size,
                    )
                return result, token
            except Exception as exc:  # noqa: BLE001 - library raises broadly
                last_error = exc
                wait = self.retry_backoff * attempt
                logger.warning(
                    "Play page fetch attempt %d/%d failed: %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("All %d page-fetch attempts failed.", self.max_retries)
        raise RuntimeError(
            f"Play Store fetch failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    # --- normalization --------------------------------------------------- #
    @staticmethod
    def _normalize(raw: dict) -> ReviewData:
        """Convert a raw Play Store review dict into :class:`ReviewData`."""
        username = raw.get("userName", "") or ""
        date_str = to_iso(raw.get("at"))
        content = raw.get("content", "") or ""
        # reviewCreatedVersion is the app version at review time; fall back to appVersion.
        version = raw.get("reviewCreatedVersion") or raw.get("appVersion")
        review_id = raw.get("reviewId") or make_review_id(
            "play_store", username, date_str or "", content
        )

        return ReviewData(
            review_id=review_id,
            source="play_store",
            username=anonymize_username(username),
            rating=raw.get("score"),
            title=None,  # Play Store reviews have no title field
            review_text=content,
            date=date_str,
            version=version,
            helpful_count=raw.get("thumbsUpCount"),
        )

    # --- public scrape API ---------------------------------------------- #
    def scrape(
        self, how_many: int = 5000, sort: str = "newest"
    ) -> list[dict]:
        """Scrape reviews using continuation-token pagination.

        Args:
            how_many: Target number of reviews (5000+ supported).
            sort: One of ``most_relevant``, ``newest``, ``rating``.

        Returns:
            A list of review dictionaries in the unified schema.
        """
        sort_enum = self._sort_enum(sort)
        raw_reviews: list[dict] = []
        token = None

        while len(raw_reviews) < how_many:
            batch, token = self._fetch_page(sort_enum, token)
            if not batch:
                break
            raw_reviews.extend(batch)
            logger.info(
                "Play: collected %d/%d reviews", len(raw_reviews), how_many
            )
            if token is None:  # no more pages available
                break
            time.sleep(self.request_delay)

        raw_reviews = raw_reviews[:how_many]
        normalized = [self._normalize(r) for r in raw_reviews]

        # "rating" isn't a native sort; order client-side (highest first).
        if sort == "rating":
            normalized.sort(key=lambda r: (r.rating or 0), reverse=True)

        logger.info("Play: normalized %d reviews", len(normalized))
        return [r.model_dump() for r in normalized]

    def save(self, reviews: list[dict], prefix: str = "spotify_play_store_reviews"):
        """Persist reviews to timestamped JSON + CSV. Returns the file paths."""
        return save_reviews(reviews, prefix=prefix)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scraper = PlayStoreReviewScraper()
    collected = scraper.scrape(how_many=5000, sort="newest")
    print(f"Collected {len(collected)} Play Store reviews")
    if collected:
        scraper.save(collected)
