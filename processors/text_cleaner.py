"""Text preprocessing pipeline for Spotify discovery feedback.

Provides:
  * :class:`TextCleaner` - composable cleaning steps (URLs, emojis, whitespace,
    special characters, contraction expansion, lowercasing, app-specific noise).
  * :class:`DiscoveryKeywordExtractor` - detects Spotify discovery features,
    sentiment-bearing recommendation phrases, and tags feedback into discovery
    categories.
  * Language detection (English filtering) and an efficient batch processor that
    returns cleaned text plus metadata tags.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Shared regexes
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002700-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U00002600-\U000026ff"
    "]+",
    flags=re.UNICODE,
)
WHITESPACE_RE = re.compile(r"\s+")
# Version strings like "v8.7.0", "version 8.7", "8.7.0.1234".
VERSION_RE = re.compile(
    r"\b[vV]?\d+\.\d+(?:\.\d+)*(?:\.\d+)?\b|\bversion\s+\d[\d.]*\b",
    flags=re.IGNORECASE,
)
# Edit/update markers reviewers commonly prepend.
EDIT_MARKER_RE = re.compile(
    r"\b(edit|update|edited|updated|edit\s*\d+)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
# Keep letters, numbers, whitespace and a little sentence punctuation.
SPECIAL_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s.,!?']")

# Common English contractions -> expansions.
CONTRACTIONS: dict[str, str] = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
    "y'all": "you all",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "cant": "cannot",
    "wont": "will not",
    "im": "i am",
}
_CONTRACTION_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS if not k.startswith("'")) + r")\b",
    flags=re.IGNORECASE,
)
_CONTRACTION_SUFFIX_RE = re.compile(
    r"(" + "|".join(re.escape(k) for k in CONTRACTIONS if k.startswith("'") or k.startswith("n")) + r")",
    flags=re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# TextCleaner
# --------------------------------------------------------------------------- #
class TextCleaner:
    """Composable text cleaning pipeline.

    Each step is a small, independently usable method. :meth:`clean` runs the
    full configured pipeline.
    """

    def __init__(
        self,
        do_remove_urls: bool = True,
        do_remove_emojis: bool = True,
        do_expand_contractions: bool = True,
        do_remove_special_characters: bool = False,
        do_lowercase: bool = False,
        do_remove_app_noise: bool = True,
    ) -> None:
        self.do_remove_urls = do_remove_urls
        self.do_remove_emojis = do_remove_emojis
        self.do_expand_contractions = do_expand_contractions
        self.do_remove_special_characters = do_remove_special_characters
        self.do_lowercase = do_lowercase
        self.do_remove_app_noise = do_remove_app_noise

    # --- individual steps ----------------------------------------------- #
    def remove_urls(self, text: str) -> str:
        text = URL_RE.sub(" ", text)
        return MENTION_RE.sub(" ", text)

    def remove_emojis(self, text: str) -> str:
        return EMOJI_RE.sub(" ", text)

    def normalize_whitespace(self, text: str) -> str:
        return WHITESPACE_RE.sub(" ", text).strip()

    def remove_special_characters(self, text: str) -> str:
        text = HASHTAG_RE.sub(r"\1", text)  # keep the hashtag word
        return SPECIAL_CHARS_RE.sub(" ", text)

    def expand_contractions(self, text: str) -> str:
        def _suffix(match: re.Match) -> str:
            return CONTRACTIONS.get(match.group(0).lower(), match.group(0))

        def _word(match: re.Match) -> str:
            return CONTRACTIONS.get(match.group(0).lower(), match.group(0))

        # Word-level forms first (e.g. won't -> will not) so suffix rules don't
        # mangle them (n't -> not would turn won't into "wo not").
        text = _CONTRACTION_WORD_RE.sub(_word, text)
        text = _CONTRACTION_SUFFIX_RE.sub(_suffix, text)
        return text

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_app_specific_noise(self, text: str) -> str:
        """Strip 'Edit:'/'Update:' markers and version numbers."""
        text = EDIT_MARKER_RE.sub(" ", text)
        text = VERSION_RE.sub(" ", text)
        return text

    # --- orchestration --------------------------------------------------- #
    def clean(self, text: str) -> str:
        """Run the full configured cleaning pipeline on one string."""
        if not text:
            return ""

        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)

        if self.do_remove_urls:
            text = self.remove_urls(text)
        if self.do_remove_app_noise:
            text = self.remove_app_specific_noise(text)
        if self.do_expand_contractions:
            text = self.expand_contractions(text)
        if self.do_remove_emojis:
            text = self.remove_emojis(text)
        if self.do_remove_special_characters:
            text = self.remove_special_characters(text)
        if self.do_lowercase:
            text = self.lowercase(text)

        return self.normalize_whitespace(text)


# --------------------------------------------------------------------------- #
# DiscoveryKeywordExtractor
# --------------------------------------------------------------------------- #
# Canonical Spotify feature -> aliases that may appear in feedback.
SPOTIFY_FEATURES: dict[str, list[str]] = {
    "Discover Weekly": ["discover weekly", "discovery weekly", "dw playlist"],
    "Release Radar": ["release radar"],
    "Daily Mix": ["daily mix", "daily mixes"],
    "Made For You": ["made for you", "made for u"],
    "Blend": ["blend"],
    "Radio": ["song radio", "artist radio", "radio station", "radio"],
    "Autoplay": ["autoplay", "auto play", "auto-play"],
    "Enhance": ["enhance"],
    "Smart Shuffle": ["smart shuffle"],
    "Spotify DJ": ["spotify dj", "ai dj"],
    "Recommendations": [
        "recommendation",
        "recommend",
        "suggested",
        "suggestion",
        "for you",
    ],
    "Discover/Explore": ["discover", "discovery", "explore", "browse"],
}

# Discovery category -> trigger keywords/phrases.
DISCOVERY_CATEGORIES: dict[str, list[str]] = {
    "algorithm_complaints": [
        "algorithm",
        "algorithmic",
        "recommendation engine",
        "broken",
        "doesn't work",
        "does not work",
        "terrible recommendations",
        "bad recommendations",
        "poor recommendations",
    ],
    "repetitive_content": [
        "same songs",
        "same artists",
        "repetitive",
        "repeat",
        "over and over",
        "loop",
        "stuck",
        "keeps playing",
        "always plays",
        "same playlist",
    ],
    "feature_requests": [
        "wish",
        "should add",
        "please add",
        "would be nice",
        "feature request",
        "need a way",
        "i want",
        "it would be great",
        "hope they add",
    ],
    "positive_discovery": [
        "love discover",
        "found new",
        "discovered",
        "great recommendations",
        "amazing recommendations",
        "best playlist",
        "new favorite",
        "introduced me",
        "spot on",
        "perfectly",
    ],
    "genre_limitations": [
        "genre",
        "only plays",
        "narrow",
        "limited",
        "bubble",
        "echo chamber",
        "same genre",
        "variety",
        "diverse",
    ],
    "mood_mismatch": [
        "mood",
        "vibe",
        "wrong songs",
        "doesn't match",
        "does not match",
        "context",
        "workout",
        "study",
        "sleep",
        "energy",
    ],
}

POSITIVE_WORDS = {
    "love", "great", "amazing", "perfect", "best", "awesome", "excellent",
    "good", "favorite", "enjoy", "spot on", "fantastic", "brilliant",
}
NEGATIVE_WORDS = {
    "hate", "bad", "terrible", "awful", "worst", "boring", "annoying",
    "broken", "repetitive", "stuck", "poor", "disappointing", "useless",
}
RECOMMENDATION_TERMS = (
    "recommend", "recommendation", "suggest", "discover", "discovery",
    "playlist", "algorithm", "new music", "release radar", "daily mix",
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class DiscoveryKeywordExtractor:
    """Extract discovery features, sentiment phrases, and category tags."""

    def __init__(
        self,
        features: dict[str, list[str]] | None = None,
        categories: dict[str, list[str]] | None = None,
    ) -> None:
        self.features = features or SPOTIFY_FEATURES
        self.categories = categories or DISCOVERY_CATEGORIES

    def extract_features(self, text: str) -> list[str]:
        """Return canonical Spotify features mentioned in the text."""
        lowered = text.lower()
        found: list[str] = []
        for feature, aliases in self.features.items():
            if any(alias in lowered for alias in aliases):
                found.append(feature)
        return found

    def categorize(self, text: str) -> list[str]:
        """Tag the text with matching discovery categories."""
        lowered = text.lower()
        tags: list[str] = []
        for category, keywords in self.categories.items():
            if any(keyword in lowered for keyword in keywords):
                tags.append(category)
        return tags

    def extract_sentiment_phrases(self, text: str) -> list[dict]:
        """Return recommendation-related sentences with a rough polarity.

        Splits the text into sentences and keeps those mentioning a
        recommendation/discovery term, labeling each positive/negative/neutral.
        """
        phrases: list[dict] = []
        for sentence in SENTENCE_SPLIT_RE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            lowered = sentence.lower()
            if not any(term in lowered for term in RECOMMENDATION_TERMS):
                continue
            pos = sum(word in lowered for word in POSITIVE_WORDS)
            neg = sum(word in lowered for word in NEGATIVE_WORDS)
            polarity = (
                "positive" if pos > neg else "negative" if neg > pos else "neutral"
            )
            phrases.append({"phrase": sentence, "polarity": polarity})
        return phrases

    def tag(self, text: str) -> dict:
        """Run all extractors and return a consolidated tag dict."""
        return {
            "discovery_features": self.extract_features(text),
            "discovery_categories": self.categorize(text),
            "sentiment_phrases": self.extract_sentiment_phrases(text),
        }


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
_LANGDETECT_WARNED = False


def detect_language(text: str) -> str:
    """Detect the language of ``text`` (ISO 639-1 code), 'unknown' on failure.

    Uses ``langdetect`` if available. If the library is missing, returns
    'unknown' so callers don't accidentally drop everything.
    """
    global _LANGDETECT_WARNED
    if not text or len(text.strip()) < 3:
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic results
        return detect(text)
    except ImportError:
        if not _LANGDETECT_WARNED:
            logger.warning("langdetect not installed; skipping language filtering.")
            _LANGDETECT_WARNED = True
        return "unknown"
    except Exception:  # noqa: BLE001 - langdetect raises LangDetectException
        return "unknown"


def is_english(text: str) -> bool:
    """True if text is detected as English (or language can't be determined)."""
    lang = detect_language(text)
    return lang in ("en", "unknown")


# --------------------------------------------------------------------------- #
# Batch processing
# --------------------------------------------------------------------------- #
_TEXT_FIELD_FALLBACKS = ("review_text", "content", "body_text", "text")


def _resolve_text(record: dict, text_field: str) -> str:
    """Pull the best available text from a record, prepending a title."""
    body = record.get(text_field)
    if not body:
        for field in _TEXT_FIELD_FALLBACKS:
            if record.get(field):
                body = record[field]
                break
    body = body or ""
    title = record.get("title")
    return f"{title}. {body}" if title else body


def process_records(
    records: list[dict],
    text_field: str = "review_text",
    drop_non_english: bool = True,
    cleaner: TextCleaner | None = None,
    extractor: DiscoveryKeywordExtractor | None = None,
) -> list[dict]:
    """Clean and tag a batch of records efficiently.

    For each record this adds:
        * ``clean_text`` - cleaned text
        * ``language`` - detected language code
        * ``discovery_features`` / ``discovery_categories`` / ``sentiment_phrases``

    Records whose cleaned text is empty (or non-English, if filtering) are dropped.

    Returns:
        The list of enriched record dictionaries.
    """
    cleaner = cleaner or TextCleaner()
    extractor = extractor or DiscoveryKeywordExtractor()

    processed: list[dict] = []
    for record in records:
        raw = _resolve_text(record, text_field)
        cleaned = cleaner.clean(raw)
        if not cleaned:
            continue

        language = detect_language(cleaned)
        if drop_non_english and language not in ("en", "unknown"):
            continue

        tags = extractor.tag(cleaned)
        new_record = dict(record)
        new_record["clean_text"] = cleaned
        new_record["language"] = language
        new_record.update(tags)
        processed.append(new_record)

    logger.info("Processed %d/%d records", len(processed), len(records))
    return processed


def clean_records(records: list[dict], text_field: str = "review_text") -> list[dict]:
    """Backward-compatible wrapper that cleans + tags without language filtering."""
    return process_records(records, text_field=text_field, drop_non_english=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = [
        {
            "title": "Edit: still bad",
            "review_text": "Update: v8.7.0 Discover Weekly keeps giving me the "
            "same songs over and over. I wish they'd fix the algorithm! 😡",
        },
        {
            "review_text": "I love Discover Weekly, it introduced me to my new "
            "favorite band. The recommendations are spot on!",
        },
    ]
    for rec in process_records(demo):
        print(rec["clean_text"])
        print("  features:", rec["discovery_features"])
        print("  categories:", rec["discovery_categories"])
        print("  phrases:", rec["sentiment_phrases"])
