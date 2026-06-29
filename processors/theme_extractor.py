"""Theme extraction from Spotify discovery feedback.

Two complementary paths:
  * keyword: lightweight TF-IDF + seed-theme bucketing (no API cost). Used as
    the fast default and for quick aggregate counts.
  * llm (Claude/GPT): structured per-review theme extraction using a fixed
    prompt, with disk caching, rate limiting, and cross-review aggregation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Project-level data dir for the cache file.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CACHE_PATH = DATA_DIR / "theme_cache.json"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Discovery-centric seed themes used to bucket feedback heuristically.
SEED_THEMES: dict[str, list[str]] = {
    "discover_weekly": ["discover weekly", "weekly playlist", "new music friday"],
    "recommendations": ["recommendation", "recommend", "suggested", "algorithm"],
    "repetition": ["repetitive", "same songs", "loop", "stale", "boring"],
    "personalization": ["personalized", "my taste", "for me", "tailored"],
    "radio_autoplay": ["radio", "autoplay", "queue", "enhance"],
    "podcasts": ["podcast", "episode", "show"],
    "audio_quality": ["quality", "bitrate", "sound", "audio"],
    "pricing": ["price", "premium", "subscription", "expensive", "cost"],
}

# The exact extraction prompt (Claude-friendly), with explicit JSON keys so the
# output can be aggregated reliably.
THEME_PROMPT = """Analyze this Spotify user review for music discovery themes.

Review: {review_text}

Extract:
1. Primary frustration (if any)
2. Desired behavior/outcome
3. Current listening pattern mentioned
4. Feature mentioned (Discover Weekly, Radio, etc.)
5. User segment indicators (casual/power user, genre preferences)
6. Specific improvement suggestion (if any)

Return ONLY valid JSON with exactly these keys (use null when not present):
{{
  "primary_frustration": string | null,
  "desired_outcome": string | null,
  "listening_pattern": string | null,
  "feature_mentioned": string | null,
  "user_segment": string | null,
  "improvement_suggestion": string | null
}}"""

# Keys we aggregate across reviews to surface patterns.
_AGGREGATE_KEYS = (
    "primary_frustration",
    "desired_outcome",
    "listening_pattern",
    "feature_mentioned",
    "user_segment",
    "improvement_suggestion",
)


def _parse_json_object(raw: str) -> dict | None:
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class ThemeExtractor:
    """Extract and aggregate discovery themes from feedback."""

    def __init__(
        self,
        strategy: Literal["keyword", "llm"] = "keyword",
        provider: str | None = None,
        model: str | None = None,
        request_delay: float = 1.0,
        use_cache: bool = True,
        cache_path: Path | None = None,
    ) -> None:
        self.strategy = strategy
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.model = model
        self.request_delay = request_delay  # rate limit between LLM calls
        self.use_cache = use_cache
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self._llm = None
        self._cache: dict | None = None

    # ------------------------------------------------------------------ #
    # Keyword strategy (fast, default)
    # ------------------------------------------------------------------ #
    def extract_keyword_themes(
        self, records: list[dict], text_field: str = "clean_text", top_k: int = 15
    ) -> dict:
        theme_counts: Counter[str] = Counter()
        tagged_records: list[dict] = []

        for record in records:
            text = (record.get(text_field, "") or "").lower()
            matched = [
                theme
                for theme, keywords in SEED_THEMES.items()
                if any(kw in text for kw in keywords)
            ]
            theme_counts.update(matched)
            new_record = dict(record)
            new_record["themes"] = matched
            tagged_records.append(new_record)

        keywords = self._tfidf_keywords(records, text_field, top_k)
        return {
            "theme_counts": dict(theme_counts.most_common()),
            "top_keywords": keywords,
            "records": tagged_records,
        }

    @staticmethod
    def _tfidf_keywords(records: list[dict], text_field: str, top_k: int) -> list[str]:
        corpus = [r.get(text_field, "") for r in records if r.get(text_field)]
        if not corpus:
            return []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:  # pragma: no cover - dependency guard
            logger.warning("scikit-learn missing; falling back to raw frequency.")
            words = Counter(" ".join(corpus).split())
            return [w for w, _ in words.most_common(top_k)]

        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=1000
        )
        matrix = vectorizer.fit_transform(corpus)
        scores = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [term for term, _ in ranked[:top_k]]

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.use_cache and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read theme cache: %s", exc)
                self._cache = {}
        else:
            self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        if not self.use_cache or self._cache is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    # LLM strategy (Claude/GPT)
    # ------------------------------------------------------------------ #
    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY must be set for LLM extraction.")
            self._llm = ChatAnthropic(
                model=self.model or os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"),
                temperature=0,
            )
        else:
            from langchain_openai import ChatOpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY must be set for LLM extraction.")
            self._llm = ChatOpenAI(
                model=self.model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=0,
            )
        return self._llm

    def extract_review_theme(self, text: str) -> dict:
        """Extract structured themes from a single review (cached)."""
        if not text:
            return {}

        cache = self._load_cache()
        key = self._cache_key(text)
        if key in cache:
            return cache[key]

        prompt = THEME_PROMPT.format(review_text=text)
        try:
            from processors.llm_client import chat_complete

            content = chat_complete(prompt, temperature=0)
            result = _parse_json_object(content) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM theme extraction failed: %s", exc)
            result = {}

        cache[key] = result
        return result

    def extract_batch(
        self, records: list[dict], text_field: str = "clean_text"
    ) -> list[dict]:
        """Extract themes for many reviews with caching + rate limiting."""
        results: list[dict] = []
        cache = self._load_cache()
        for record in records:
            text = record.get(text_field, "") or ""
            if not text:
                results.append({})
                continue
            key = self._cache_key(text)
            cached = key in cache
            theme = self.extract_review_theme(text)
            results.append(theme)
            # Only sleep when we actually hit the API (cache miss).
            if not cached and self.request_delay:
                time.sleep(self.request_delay)
        self._save_cache()
        return results

    @staticmethod
    def aggregate_themes(theme_results: list[dict]) -> dict:
        """Aggregate per-review themes into cross-review pattern counts."""
        aggregates: dict[str, Counter] = {key: Counter() for key in _AGGREGATE_KEYS}
        for theme in theme_results:
            for key in _AGGREGATE_KEYS:
                value = theme.get(key)
                if value and str(value).lower() not in ("null", "none", "n/a"):
                    aggregates[key][str(value).strip().lower()] += 1
        return {key: dict(counter.most_common(20)) for key, counter in aggregates.items()}

    def extract_llm_themes(
        self, records: list[dict], text_field: str = "clean_text"
    ) -> dict:
        """Full LLM path: per-review extraction + aggregated patterns."""
        themes = self.extract_batch(records, text_field)
        tagged = [dict(r, theme=t) for r, t in zip(records, themes)]
        patterns = self.aggregate_themes(themes)
        # Surface frustration patterns as theme_counts for API compatibility.
        return {
            "themes": themes,
            "patterns": patterns,
            "theme_counts": patterns.get("primary_frustration", {}),
            "records": tagged,
        }

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def extract(self, records: list[dict], text_field: str = "clean_text") -> dict:
        if self.strategy == "llm":
            return self.extract_llm_themes(records, text_field)
        return self.extract_keyword_themes(records, text_field)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = [
        {"clean_text": "discover weekly keeps giving me the same songs, so repetitive"},
        {"clean_text": "love how personalized the recommendations feel for my taste"},
    ]
    extractor = ThemeExtractor()  # keyword strategy by default
    out = extractor.extract(demo)
    print(json.dumps(out["theme_counts"], indent=2))
