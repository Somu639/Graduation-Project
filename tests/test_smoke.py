"""Lightweight smoke tests that run without API keys or network access."""

from __future__ import annotations


def test_text_cleaner_pipeline():
    from processors.text_cleaner import TextCleaner

    cleaner = TextCleaner()
    out = cleaner.clean("Edit: Update v8.7.0 they'd love it https://x.co 🔥")
    assert "http" not in out
    assert "v8.7.0" not in out
    assert "they would" in out


def test_discovery_keyword_extractor():
    from processors.text_cleaner import DiscoveryKeywordExtractor

    extractor = DiscoveryKeywordExtractor()
    tags = extractor.tag("Discover Weekly keeps giving me the same songs, so repetitive")
    assert "Discover Weekly" in tags["discovery_features"]
    assert "repetitive_content" in tags["discovery_categories"]


def test_review_schema_rating_validation():
    from scrapers.schema import ReviewData

    ok = ReviewData(review_id="1", source="app_store", username="anon", rating=5)
    assert ok.rating == 5
    # Out-of-range ratings are coerced to None rather than raising.
    bad = ReviewData(review_id="2", source="app_store", username="anon", rating=9)
    assert bad.rating is None


def test_segment_classification_and_matrix():
    from agents.segment_analyzer import SegmentAnalyzer

    analyzer = SegmentAnalyzer()
    records = [
        {"content": "I just play background playlists while studying"},
        {"content": "always hunting for new music, stuck in a bubble"},
        {"content": "new to spotify, onboarding gave me no recommendations yet"},
    ]
    sizes = analyzer.estimate_sizes(records)
    assert sizes["casual_listeners"]["count"] >= 1
    matrix = analyzer.build_comparison_matrix(records)
    assert "new_users" in matrix["matrix"]
    assert matrix["matrix"]["new_users"]["cold_start_onboarding"]["count"] >= 1
