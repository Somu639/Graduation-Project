"""Reddit discussion scraper for Spotify music-discovery sentiment.

Built on PRAW (the Python Reddit API Wrapper). Searches a set of music-focused
subreddits for discovery-related queries, captures each post plus its top
comments, filters to the last two years, deduplicates by post id, and writes
output in the shared :class:`ReviewData` schema.

Credentials are read from the environment (see .env.example):
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Run a full scrape:
    python -m scrapers.reddit_scraper
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from .schema import ReviewData, anonymize_username, save_reviews

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
DEFAULT_SUBREDDITS: tuple[str, ...] = (
    "spotify",
    "Music",
    "LetsTalkMusic",
    "ifyoulikeblank",
    "musicsuggestions",
)

DEFAULT_QUERIES: tuple[str, ...] = (
    "spotify recommendation algorithm",
    "spotify discover weekly",
    "spotify suggestions bad",
    "spotify same songs",
    "spotify not discovering new music",
    "spotify stuck in bubble",
    "spotify release radar",
)

SECONDS_PER_YEAR = 365.25 * 24 * 3600


class RedditScraper:
    """Scrape, normalize, and persist Spotify-related Reddit discussions."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        max_comments: int = 20,
        request_delay: float = 1.0,
    ) -> None:
        # Load .env so standalone runs pick up credentials automatically.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover - optional convenience
            pass

        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT", "spotify-discovery-analyzer/1.0"
        )
        self.max_comments = max_comments
        self.request_delay = request_delay
        self._reddit = None

    # --- client ---------------------------------------------------------- #
    def _client(self):
        if self._reddit is not None:
            return self._reddit
        try:
            import praw
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("praw is required. Install it via requirements.txt") from exc

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Reddit credentials missing. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET in your .env file."
            )

        self._reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            check_for_async=False,
        )
        self._reddit.read_only = True
        return self._reddit

    # --- comment collection --------------------------------------------- #
    def _collect_comments(self, submission) -> list[dict]:
        """Return the top ``max_comments`` comments (by score) for a post."""
        try:
            submission.comments.replace_more(limit=0)
            all_comments = submission.comments.list()
        except Exception as exc:  # noqa: BLE001 - PRAW can raise broadly
            logger.warning("Failed to load comments for %s: %s", submission.id, exc)
            return []

        ranked = sorted(
            all_comments, key=lambda c: getattr(c, "score", 0), reverse=True
        )
        top = ranked[: self.max_comments]
        return [
            {
                "comment_id": c.id,
                "body": c.body,
                "score": getattr(c, "score", 0),
                "author": anonymize_username(str(c.author) if c.author else ""),
            }
            for c in top
        ]

    # --- normalization --------------------------------------------------- #
    def _to_review(self, submission) -> ReviewData:
        """Convert a PRAW submission into the unified schema."""
        date_str = datetime.fromtimestamp(
            submission.created_utc, tz=timezone.utc
        ).isoformat()
        comments = self._collect_comments(submission)

        return ReviewData(
            review_id=submission.id,
            source="reddit",
            username=anonymize_username(
                str(submission.author) if submission.author else ""
            ),
            rating=None,  # Reddit posts have no star rating
            title=submission.title,
            review_text=submission.selftext or "",
            date=date_str,
            version=None,
            helpful_count=submission.score,  # upvote score
            metadata={
                "subreddit": str(submission.subreddit),
                "num_comments": submission.num_comments,
                "score": submission.score,
                "url": submission.url,
                "permalink": f"https://reddit.com{submission.permalink}",
                "top_comments": comments,
            },
        )

    # --- public scrape API ---------------------------------------------- #
    def scrape(
        self,
        subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
        queries: tuple[str, ...] = DEFAULT_QUERIES,
        limit_per_query: int = 100,
        years: float = 2,
    ) -> list[dict]:
        """Search subreddits for discovery-related discussions.

        Args:
            subreddits: Subreddit names to search.
            queries: Search query strings.
            limit_per_query: Max submissions per (subreddit, query) pair.
            years: Only keep posts created within this many years.

        Returns:
            A list of deduplicated review dictionaries in the unified schema.
        """
        reddit = self._client()
        cutoff = time.time() - years * SECONDS_PER_YEAR
        seen: set[str] = set()
        results: list[dict] = []

        for sub_name in subreddits:
            subreddit = reddit.subreddit(sub_name)
            for query in queries:
                try:
                    for submission in subreddit.search(
                        query, limit=limit_per_query, sort="relevance"
                    ):
                        if submission.id in seen:  # dedupe by post_id
                            continue
                        if submission.created_utc < cutoff:  # last N years only
                            continue
                        seen.add(submission.id)
                        results.append(self._to_review(submission).model_dump())
                    logger.info("reddit: r/%s '%s' -> %d total", sub_name, query, len(results))
                    time.sleep(self.request_delay)
                except Exception as exc:  # noqa: BLE001 - PRAW raises various errors
                    logger.warning(
                        "reddit search failed for r/%s '%s': %s", sub_name, query, exc
                    )

        logger.info("reddit: collected %d unique posts", len(results))
        return results

    def save(self, reviews: list[dict], prefix: str = "spotify_reddit_posts"):
        """Persist posts to timestamped JSON + CSV. Returns the file paths."""
        return save_reviews(reviews, prefix=prefix)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scraper = RedditScraper()
    collected = scraper.scrape()
    print(f"Collected {len(collected)} Reddit posts")
    if collected:
        scraper.save(collected)
