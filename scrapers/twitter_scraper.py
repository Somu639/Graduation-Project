"""Twitter / X scraper for Spotify-related conversations.

Uses the official X API v2 recent-search endpoint via a bearer token. The free
tier is heavily rate limited, so keep queries focused and respect the limits.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

import requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


@dataclass
class Tweet:
    source: str
    tweet_id: str
    content: str
    author_id: str
    created_at: str | None
    like_count: int
    retweet_count: int
    reply_count: int
    lang: str | None


class TwitterScraper:
    """Scrape recent tweets mentioning Spotify discovery topics."""

    def __init__(self, bearer_token: str | None = None, timeout: int = 15) -> None:
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self.bearer_token:
            raise RuntimeError(
                "Twitter credentials missing. Set TWITTER_BEARER_TOKEN in your environment."
            )
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def scrape(
        self,
        query: str = "(spotify discover OR spotify recommendation) lang:en -is:retweet",
        max_results: int = 100,
    ) -> list[dict]:
        """Fetch recent tweets matching a query.

        Args:
            query: X API v2 search query.
            max_results: Number of tweets to fetch (10-100 per request).

        Returns:
            A list of tweet dictionaries.
        """
        params = {
            "query": query,
            "max_results": min(max(max_results, 10), 100),
            "tweet.fields": "created_at,public_metrics,lang,author_id",
        }
        try:
            response = requests.get(
                SEARCH_URL, headers=self._headers(), params=params, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("twitter fetch failed: %s", exc)
            return []

        payload = response.json()
        tweets: list[Tweet] = []
        for item in payload.get("data", []):
            metrics = item.get("public_metrics", {})
            tweets.append(
                Tweet(
                    source="twitter",
                    tweet_id=item.get("id", ""),
                    content=item.get("text", ""),
                    author_id=item.get("author_id", ""),
                    created_at=item.get("created_at"),
                    like_count=metrics.get("like_count", 0),
                    retweet_count=metrics.get("retweet_count", 0),
                    reply_count=metrics.get("reply_count", 0),
                    lang=item.get("lang"),
                )
            )
        logger.info("twitter: collected %d tweets", len(tweets))
        return [asdict(t) for t in tweets]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = TwitterScraper()
    data = scraper.scrape(max_results=10)
    print(f"Collected {len(data)} tweets")
