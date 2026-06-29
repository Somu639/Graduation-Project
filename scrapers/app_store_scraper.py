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
import re
import time
from datetime import datetime

import requests

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
# Apple RSS customer reviews (legacy — often returns empty since ~May 2026).
ITUNES_RSS_URL = (
    "https://itunes.apple.com/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
)
ITUNES_RSS_MAX_PAGES = 10

# App Store product page SSR JSON (works when RSS is empty).
APP_STORE_PAGE_URL = "https://apps.apple.com/{country}/app/id{app_id}"
SSR_SCRIPT_RE = re.compile(
    r'<script[^>]*id="serialized-server-data"[^>]*>(.*?)</script>',
    re.DOTALL,
)
# Storefronts to aggregate unique reviews (~24 helpful reviews per country page).
DEFAULT_COUNTRIES: tuple[str, ...] = (
    "us", "gb", "ca", "au", "de", "fr", "in", "br", "mx", "jp",
    "es", "it", "nl", "se", "pl", "kr", "sg", "ae", "za", "nz",
)
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_ssr_reviews(html: str) -> list[dict]:
    """Parse Review objects from the App Store page ``serialized-server-data`` blob."""
    match = SSR_SCRIPT_RE.search(html)
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    found: list[dict] = []
    seen_ids: set[str] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            kind = obj.get("$kind") or obj.get("kind")
            if kind == "Review":
                rid = str(obj.get("id") or "")
                if rid and rid not in seen_ids:
                    seen_ids.add(rid)
                    found.append(obj)
                return
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    return found


def _ssr_review_to_raw(review: dict) -> dict:
    """Map an SSR Review object to the raw dict shape used by :meth:`_normalize`."""
    return {
        "review_id": str(review.get("id") or ""),
        "userName": review.get("reviewerName", "") or "",
        "rating": review.get("rating"),
        "title": review.get("title", "") or "",
        "review": review.get("contents") or review.get("content") or "",
        "date": review.get("date", "") or "",
        "version": review.get("version") or review.get("appVersion"),
    }


def _parse_rss_entry(entry: dict) -> dict | None:
    """Convert one iTunes RSS entry to the raw dict shape used by :meth:`_normalize`."""
    if "im:rating" not in entry:
        return None  # first row is app metadata, not a review
    content = entry.get("content", {})
    review_text = content.get("label") if isinstance(content, dict) else ""
    if not review_text:
        summary = entry.get("summary", {})
        review_text = summary.get("label", "") if isinstance(summary, dict) else ""
    author = entry.get("author", {})
    username = author.get("name", {}).get("label", "") if isinstance(author, dict) else ""
    title = entry.get("title", {})
    title_text = title.get("label", "") if isinstance(title, dict) else ""
    updated = entry.get("updated", {})
    date_str = updated.get("label", "") if isinstance(updated, dict) else ""
    version = entry.get("im:version", {})
    version_text = version.get("label") if isinstance(version, dict) else None
    rating_raw = entry.get("im:rating", {})
    rating = int(rating_raw.get("label", 0)) if isinstance(rating_raw, dict) else None
    return {
        "userName": username,
        "rating": rating,
        "title": title_text,
        "review": review_text,
        "date": date_str,
        "version": version_text,
    }


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
    def _fetch_web_ssr(
        self,
        how_many: int,
        after: datetime | None = None,
        countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    ) -> list[dict]:
        """Fetch reviews from App Store product pages (SSR JSON, May 2026+ reliable path)."""
        collected: dict[str, dict] = {}

        for country in countries:
            if len(collected) >= how_many:
                break
            url = APP_STORE_PAGE_URL.format(country=country, app_id=self.app_id)
            try:
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
                resp.raise_for_status()
                for review in _extract_ssr_reviews(resp.text):
                    raw = _ssr_review_to_raw(review)
                    rid = raw.get("review_id") or ""
                    if not rid or not raw.get("review"):
                        continue
                    if after and raw.get("date"):
                        try:
                            review_dt = datetime.fromisoformat(
                                raw["date"].replace("Z", "+00:00")
                            )
                            if review_dt.replace(tzinfo=None) <= after.replace(tzinfo=None):
                                continue
                        except ValueError:
                            pass
                    collected.setdefault(rid, raw)
                    if len(collected) >= how_many:
                        break
                logger.info(
                    "App Store web %s: %d unique reviews so far",
                    country,
                    len(collected),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("App Store web fetch failed for %s: %s", country, exc)
            time.sleep(self.request_delay)

        results = list(collected.values())[:how_many]
        logger.info("App Store web SSR fetched %d reviews", len(results))
        return results

    def _fetch_itunes_rss(
        self, how_many: int, after: datetime | None = None
    ) -> list[dict]:
        """Fetch reviews via Apple's public iTunes RSS JSON feed (requests only)."""
        collected: list[dict] = []
        headers = BROWSER_HEADERS

        for page in range(1, ITUNES_RSS_MAX_PAGES + 1):
            if len(collected) >= how_many:
                break
            url = ITUNES_RSS_URL.format(page=page, app_id=self.app_id)
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("iTunes RSS page %d failed: %s", page, exc)
                break

            entries = payload.get("feed", {}).get("entry", [])
            if not isinstance(entries, list):
                entries = [entries] if entries else []

            for entry in entries:
                raw = _parse_rss_entry(entry)
                if not raw or not raw.get("review"):
                    continue
                if after and raw.get("date"):
                    try:
                        review_dt = datetime.fromisoformat(
                            raw["date"].replace("Z", "+00:00")
                        )
                        if review_dt.replace(tzinfo=None) <= after.replace(tzinfo=None):
                            continue
                    except ValueError:
                        pass
                collected.append(raw)
                if len(collected) >= how_many:
                    break

            if len(entries) <= 1:
                break  # no more review pages
            time.sleep(self.request_delay)

        logger.info("iTunes RSS fetched %d reviews", len(collected))
        return collected[:how_many]

    def _fetch_raw(
        self, how_many: int, after: datetime | None = None
    ) -> list[dict]:
        """Fetch raw review dicts — web SSR first, then RSS / library fallbacks."""
        web = self._fetch_web_ssr(how_many=how_many, after=after)
        if web:
            return web

        logger.info("App Store web SSR returned nothing; trying iTunes RSS.")
        rss = self._fetch_itunes_rss(how_many=how_many, after=after)
        if rss:
            return rss

        try:
            return self._fetch_via_library(how_many=how_many, after=after)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("app-store-scraper library failed: %s", exc)

        raise RuntimeError(
            "Could not fetch App Store reviews (web SSR, RSS, and library all failed)."
        )

    def _fetch_via_library(
        self, how_many: int, after: datetime | None = None
    ) -> list[dict]:
        """Fetch raw review dicts from the ``app_store_scraper`` library with retry/backoff.

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
        review_id = raw.get("review_id") or make_review_id(
            "app_store", username, date_str or "", review_text
        )

        return ReviewData(
            review_id=review_id,
            source="app_store",
            username=anonymize_username(username),
            rating=raw.get("rating"),
            title=title,
            review_text=review_text,
            date=date_str,
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
