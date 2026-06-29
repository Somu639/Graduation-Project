"""Reddit discussion scraper for Spotify music-discovery sentiment.

Uses the Reddit API (PRAW or OAuth) when credentials are configured, and falls
back to public subreddit RSS feeds when they are not — so Live Reviews works
without API keys on Streamlit Cloud.

Credentials (optional, for deeper search):
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Run a full scrape:
    python -m scrapers.reddit_scraper
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

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
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
RSS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
POST_ID_RE = re.compile(r"/comments/([a-z0-9]+)/", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")

# For non-spotify subreddits, keep posts that mention Spotify or discovery terms.
SPOTIFY_HINTS: tuple[str, ...] = (
    "spotify",
    "discover weekly",
    "release radar",
    "recommendation",
    "algorithm",
    "playlist",
    "new music",
    "same songs",
    "repetitive",
)


def _strip_html(text: str) -> str:
    return HTML_TAG_RE.sub("", text).strip()


def _post_id_from_entry(entry_id: str, link: str) -> str:
    for candidate in (entry_id, link):
        match = POST_ID_RE.search(candidate or "")
        if match:
            return match.group(1)
    return entry_id.rsplit("/", 1)[-1] if entry_id else ""


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
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover
            pass

        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT", "spotify-discovery-analyzer/1.0"
        )
        self.max_comments = max_comments
        self.request_delay = request_delay
        self._reddit = None
        self._oauth_token: str | None = None

    @property
    def has_api_credentials(self) -> bool:
        return bool(self.client_id and self.client_secret)

    # --- OAuth (requests) ------------------------------------------------ #
    def _oauth_headers(self) -> dict[str, str]:
        if self._oauth_token is None:
            resp = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": self.user_agent},
                timeout=15,
            )
            resp.raise_for_status()
            self._oauth_token = resp.json()["access_token"]
        return {
            "Authorization": f"bearer {self._oauth_token}",
            "User-Agent": self.user_agent,
        }

    def _search_oauth(
        self,
        subreddit: str,
        query: str,
        limit: int,
        cutoff: float,
        seen: set[str],
        results: list[dict],
    ) -> None:
        resp = requests.get(
            f"https://oauth.reddit.com/r/{subreddit}/search",
            headers=self._oauth_headers(),
            params={
                "q": query,
                "restrict_sr": "true",
                "sort": "relevance",
                "limit": min(limit, 100),
                "t": "year",
            },
            timeout=20,
        )
        resp.raise_for_status()
        for child in resp.json().get("data", {}).get("children", []):
            data = child.get("data", {})
            post_id = data.get("id")
            if not post_id or post_id in seen:
                continue
            if data.get("created_utc", 0) < cutoff:
                continue
            seen.add(post_id)
            results.append(self._dict_to_review(data, subreddit).model_dump())

    @staticmethod
    def _dict_to_review(data: dict, subreddit: str) -> ReviewData:
        date_str = datetime.fromtimestamp(
            data["created_utc"], tz=timezone.utc
        ).isoformat()
        return ReviewData(
            review_id=data["id"],
            source="reddit",
            username=anonymize_username(data.get("author") or ""),
            rating=None,
            title=data.get("title") or "",
            review_text=data.get("selftext") or "",
            date=date_str,
            version=None,
            helpful_count=data.get("score"),
            metadata={
                "subreddit": subreddit,
                "num_comments": data.get("num_comments"),
                "score": data.get("score"),
                "url": data.get("url"),
                "permalink": f"https://reddit.com{data.get('permalink', '')}",
            },
        )

    # --- PRAW client ----------------------------------------------------- #
    def _client(self):
        if self._reddit is not None:
            return self._reddit
        try:
            import praw
        except ImportError as exc:
            raise ImportError("praw is required for PRAW mode. pip install praw") from exc

        if not self.has_api_credentials:
            raise RuntimeError("Reddit credentials missing.")

        self._reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            check_for_async=False,
        )
        self._reddit.read_only = True
        return self._reddit

    def _collect_comments(self, submission) -> list[dict]:
        try:
            submission.comments.replace_more(limit=0)
            all_comments = submission.comments.list()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load comments for %s: %s", submission.id, exc)
            return []

        ranked = sorted(
            all_comments, key=lambda c: getattr(c, "score", 0), reverse=True
        )
        return [
            {
                "comment_id": c.id,
                "body": c.body,
                "score": getattr(c, "score", 0),
                "author": anonymize_username(str(c.author) if c.author else ""),
            }
            for c in ranked[: self.max_comments]
        ]

    def _to_review(self, submission) -> ReviewData:
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
            rating=None,
            title=submission.title,
            review_text=submission.selftext or "",
            date=date_str,
            version=None,
            helpful_count=submission.score,
            metadata={
                "subreddit": str(submission.subreddit),
                "num_comments": submission.num_comments,
                "score": submission.score,
                "url": submission.url,
                "permalink": f"https://reddit.com{submission.permalink}",
                "top_comments": comments,
            },
        )

    def _scrape_praw(
        self,
        subreddits: tuple[str, ...],
        queries: tuple[str, ...],
        limit_per_query: int,
        cutoff: float,
    ) -> list[dict]:
        reddit = self._client()
        seen: set[str] = set()
        results: list[dict] = []

        for sub_name in subreddits:
            subreddit = reddit.subreddit(sub_name)
            for query in queries:
                try:
                    for submission in subreddit.search(
                        query, limit=limit_per_query, sort="relevance"
                    ):
                        if submission.id in seen:
                            continue
                        if submission.created_utc < cutoff:
                            continue
                        seen.add(submission.id)
                        results.append(self._to_review(submission).model_dump())
                    time.sleep(self.request_delay)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("reddit PRAW search failed r/%s: %s", sub_name, exc)
        return results

    def _scrape_oauth(
        self,
        subreddits: tuple[str, ...],
        queries: tuple[str, ...],
        limit_per_query: int,
        cutoff: float,
    ) -> list[dict]:
        seen: set[str] = set()
        results: list[dict] = []
        for sub_name in subreddits:
            for query in queries:
                try:
                    self._search_oauth(
                        sub_name, query, limit_per_query, cutoff, seen, results
                    )
                    time.sleep(self.request_delay)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("reddit OAuth search failed r/%s: %s", sub_name, exc)
        return results

    # --- RSS fallback (no API keys) -------------------------------------- #
    @staticmethod
    def _rss_relevant(subreddit: str, title: str, body: str) -> bool:
        if subreddit.lower() == "spotify":
            return True
        haystack = f"{title} {body}".lower()
        return any(hint in haystack for hint in SPOTIFY_HINTS)

    def _fetch_subreddit_rss(self, subreddit: str) -> list[dict]:
        url = f"https://www.reddit.com/r/{subreddit}/.rss"
        resp = requests.get(
            url,
            headers={"User-Agent": RSS_USER_AGENT},
            timeout=20,
        )
        if resp.status_code == 429:
            logger.warning("reddit RSS rate-limited for r/%s", subreddit)
            return []
        resp.raise_for_status()
        if not resp.text.strip():
            return []

        root = ET.fromstring(resp.text)
        posts: list[dict] = []
        for entry in root.findall("a:entry", ATOM_NS):
            title_el = entry.find("a:title", ATOM_NS)
            content_el = entry.find("a:content", ATOM_NS)
            author_el = entry.find("a:author/a:name", ATOM_NS)
            updated_el = entry.find("a:updated", ATOM_NS) or entry.find(
                "a:published", ATOM_NS
            )
            link_el = entry.find("a:link", ATOM_NS)
            id_el = entry.find("a:id", ATOM_NS)

            title = (title_el.text or "").strip() if title_el is not None else ""
            body = _strip_html((content_el.text or "") if content_el is not None else "")
            if not self._rss_relevant(subreddit, title, body):
                continue

            link = link_el.get("href", "") if link_el is not None else ""
            entry_id = (id_el.text or "") if id_el is not None else ""
            post_id = _post_id_from_entry(entry_id, link)
            if not post_id:
                continue

            date_str = (updated_el.text or "") if updated_el is not None else ""
            try:
                date_iso = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                ).isoformat()
            except ValueError:
                date_iso = date_str or None

            author = (author_el.text or "") if author_el is not None else ""
            posts.append(
                ReviewData(
                    review_id=post_id,
                    source="reddit",
                    username=anonymize_username(author),
                    rating=None,
                    title=title,
                    review_text=body,
                    date=date_iso,
                    version=None,
                    helpful_count=None,
                    metadata={
                        "subreddit": subreddit,
                        "url": link,
                        "permalink": link,
                        "fetch_method": "rss",
                    },
                ).model_dump()
            )
        return posts

    def _scrape_rss(
        self,
        subreddits: tuple[str, ...],
        limit: int,
        cutoff: float,
    ) -> list[dict]:
        seen: set[str] = set()
        results: list[dict] = []

        for sub_name in subreddits:
            if len(results) >= limit:
                break
            try:
                batch = self._fetch_subreddit_rss(sub_name)
                for post in batch:
                    pid = post.get("review_id")
                    if not pid or pid in seen:
                        continue
                    if post.get("date"):
                        try:
                            ts = datetime.fromisoformat(
                                post["date"].replace("Z", "+00:00")
                            ).timestamp()
                            if ts < cutoff:
                                continue
                        except ValueError:
                            pass
                    seen.add(pid)
                    results.append(post)
                    if len(results) >= limit:
                        break
                logger.info("reddit RSS r/%s -> %d total", sub_name, len(results))
            except Exception as exc:  # noqa: BLE001
                logger.warning("reddit RSS failed for r/%s: %s", sub_name, exc)
            time.sleep(self.request_delay)

        return results[:limit]

    # --- public scrape API ---------------------------------------------- #
    def scrape(
        self,
        subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
        queries: tuple[str, ...] = DEFAULT_QUERIES,
        limit_per_query: int = 100,
        years: float = 2,
    ) -> list[dict]:
        """Search subreddits for discovery-related discussions.

        Tries OAuth/PRAW when credentials exist, otherwise uses public RSS feeds.
        """
        cutoff = time.time() - years * SECONDS_PER_YEAR
        results: list[dict] = []

        if self.has_api_credentials:
            try:
                results = self._scrape_oauth(
                    subreddits, queries, limit_per_query, cutoff
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("reddit OAuth failed (%s); trying PRAW.", exc)
                try:
                    results = self._scrape_praw(
                        subreddits, queries, limit_per_query, cutoff
                    )
                except Exception as exc2:  # noqa: BLE001
                    logger.warning("reddit PRAW failed (%s); falling back to RSS.", exc2)

        if not results:
            results = self._scrape_rss(subreddits, limit_per_query, cutoff)

        logger.info("reddit: collected %d unique posts", len(results))
        return results

    def save(self, reviews: list[dict], prefix: str = "spotify_reddit_posts"):
        return save_reviews(reviews, prefix=prefix)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scraper = RedditScraper()
    collected = scraper.scrape(limit_per_query=30)
    print(f"Collected {len(collected)} Reddit posts")
    if collected:
        scraper.save(collected)
