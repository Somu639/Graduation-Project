"""Twitter / X scraper for Spotify-related conversations.

Uses the official X API v2 when ``TWITTER_BEARER_TOKEN`` is set, and falls back
to public syndication timelines (no API key) for curated Spotify accounts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

from .schema import ReviewData, anonymize_username

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
SYNDICATION_URL = (
    "https://syndication.twitter.com/srv/timeline-profile/screen-name/{screen_name}"
)
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL,
)
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Public syndication timelines (no auth) — customer support + product news.
SYNDICATION_ACCOUNTS: tuple[str, ...] = (
    "SpotifyCares",
    "SpotifyNews",
    "Spotify",
)

DISCOVERY_HINTS: tuple[str, ...] = (
    "spotify",
    "discover",
    "recommendation",
    "recommend",
    "playlist",
    "algorithm",
    "same songs",
    "repetitive",
    "discover weekly",
    "release radar",
    "daily mix",
    "autoplay",
    "new music",
)


class TwitterScraper:
    """Scrape recent tweets mentioning Spotify discovery topics."""

    def __init__(
        self,
        bearer_token: str | None = None,
        timeout: int = 20,
        request_delay: float = 1.0,
    ) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover
            pass

        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        self.timeout = timeout
        self.request_delay = request_delay

    @property
    def has_api_credentials(self) -> bool:
        return bool(self.bearer_token)

    @staticmethod
    def _is_relevant(text: str) -> bool:
        low = text.lower()
        return any(hint in low for hint in DISCOVERY_HINTS)

    @staticmethod
    def _extract_syndication_tweets(html: str) -> list[dict]:
        match = NEXT_DATA_RE.search(html)
        if not match:
            return []
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

        found: list[dict] = []
        seen: set[str] = set()

        def walk(obj) -> None:
            if isinstance(obj, dict):
                text = obj.get("full_text") or obj.get("text")
                tid = obj.get("id_str") or (
                    str(obj.get("id")) if obj.get("id") is not None else ""
                )
                if text and tid and tid not in seen:
                    seen.add(tid)
                    found.append(obj)
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(payload)
        return found

    def _normalize_api_tweet(self, item: dict) -> ReviewData:
        metrics = item.get("public_metrics", {})
        return ReviewData(
            review_id=str(item.get("id", "")),
            source="twitter",
            username=anonymize_username(str(item.get("author_id", "unknown"))),
            rating=None,
            title=None,
            review_text=item.get("text", "") or "",
            date=item.get("created_at"),
            version=None,
            helpful_count=metrics.get("like_count"),
            metadata={
                "author_id": item.get("author_id"),
                "retweet_count": metrics.get("retweet_count", 0),
                "reply_count": metrics.get("reply_count", 0),
                "lang": item.get("lang"),
                "fetch_method": "api_v2",
            },
        )

    def _normalize_syndication_tweet(
        self, tweet: dict, *, fallback_account: str
    ) -> ReviewData:
        user = tweet.get("user") or {}
        screen_name = user.get("screen_name") or fallback_account
        metrics = tweet.get("favorite_count")
        return ReviewData(
            review_id=str(tweet.get("id_str") or tweet.get("id") or ""),
            source="twitter",
            username=anonymize_username(screen_name),
            rating=None,
            title=None,
            review_text=tweet.get("full_text") or tweet.get("text") or "",
            date=tweet.get("created_at"),
            version=None,
            helpful_count=int(metrics) if metrics is not None else None,
            metadata={
                "screen_name": screen_name,
                "retweet_count": tweet.get("retweet_count"),
                "reply_count": tweet.get("reply_count"),
                "fetch_method": "syndication",
            },
        )

    def _scrape_api(
        self,
        query: str,
        max_results: int,
    ) -> list[ReviewData]:
        if not self.bearer_token:
            return []

        params = {
            "query": query,
            "max_results": min(max(max_results, 10), 100),
            "tweet.fields": "created_at,public_metrics,lang,author_id",
        }
        response = requests.get(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {self.bearer_token}"},
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        tweets = [
            self._normalize_api_tweet(item)
            for item in response.json().get("data", [])
            if item.get("text")
        ]
        logger.info("twitter API: collected %d tweets", len(tweets))
        return tweets

    def _scrape_syndication(self, max_results: int) -> list[ReviewData]:
        collected: dict[str, ReviewData] = {}

        for account in SYNDICATION_ACCOUNTS:
            if len(collected) >= max_results:
                break
            url = SYNDICATION_URL.format(screen_name=account)
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": BROWSER_UA},
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    logger.warning("twitter syndication rate-limited for @%s", account)
                    continue
                resp.raise_for_status()
                raw_tweets = self._extract_syndication_tweets(resp.text)
                for raw in raw_tweets:
                    text = raw.get("full_text") or raw.get("text") or ""
                    if not text or not self._is_relevant(text):
                        continue
                    review = self._normalize_syndication_tweet(
                        raw, fallback_account=account
                    )
                    if review.review_id:
                        collected.setdefault(review.review_id, review)
                    if len(collected) >= max_results:
                        break
                logger.info(
                    "twitter syndication @%s -> %d unique tweets",
                    account,
                    len(collected),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("twitter syndication failed for @%s: %s", account, exc)
            time.sleep(self.request_delay)

        results = list(collected.values())[:max_results]
        logger.info("twitter syndication: collected %d tweets", len(results))
        return results

    def scrape(
        self,
        query: str = "(spotify discover OR spotify recommendation OR spotify playlist) lang:en -is:retweet",
        max_results: int = 100,
    ) -> list[dict]:
        """Fetch tweets via API v2 (if configured) or public syndication timelines."""
        results: list[ReviewData] = []

        if self.has_api_credentials:
            try:
                results = self._scrape_api(query, max_results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("twitter API failed (%s); falling back to syndication.", exc)

        if not results:
            results = self._scrape_syndication(max_results)

        return [r.model_dump() for r in results]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = TwitterScraper()
    data = scraper.scrape(max_results=20)
    print(f"Collected {len(data)} tweets")
