"""Sample Spotify discovery reviews + a one-call seeder for local demos.

Lets the app run as a live website with realistic data and no API keys: the
records flow through the real cleaning/sentiment/theme pipeline and into the
(in-memory) vector store.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("api.sample_data")

SAMPLE_REVIEWS: list[dict] = [
    {"review_id": "s1", "source": "app_store", "rating": 2, "date": "2025-01-08T00:00:00",
     "title": "Discover Weekly stuck", "review_text": "Discover Weekly keeps giving me the same songs over and over. It feels so repetitive and I'm stuck in a bubble. The algorithm doesn't surface new music anymore.", "helpful_count": 42},
    {"review_id": "s2", "source": "app_store", "rating": 5, "date": "2025-01-15T00:00:00",
     "title": "Love the recommendations", "review_text": "I love Discover Weekly, it introduced me to my new favorite band. The recommendations are spot on and personalized for my taste.", "helpful_count": 18},
    {"review_id": "s3", "source": "play_store", "rating": 1, "date": "2025-02-02T00:00:00",
     "review_text": "The recommendation algorithm is broken. I only listen to metal and it keeps suggesting pop. The genre recommendations are too narrow and irrelevant.", "helpful_count": 30},
    {"review_id": "s4", "source": "play_store", "rating": 3, "date": "2025-02-11T00:00:00",
     "review_text": "I just play background playlists while studying. It's fine but the daily mix gets stale fast and plays the same artists.", "helpful_count": 7},
    {"review_id": "s5", "source": "reddit", "rating": None, "date": "2025-02-20T00:00:00",
     "review_text": "Always hunting for new music and new artists, but lately Spotify feels like an echo chamber. Wish there was a way to push it out of my comfort zone for discovery.", "helpful_count": 88},
    {"review_id": "s6", "source": "reddit", "rating": None, "date": "2025-03-01T00:00:00",
     "review_text": "New to Spotify, just switched from Apple Music. Onboarding gave me no good recommendations yet and I don't know where to start. Too overwhelming.", "helpful_count": 12},
    {"review_id": "s7", "source": "app_store", "rating": 4, "date": "2025-03-09T00:00:00",
     "title": "Release Radar is great", "review_text": "Release Radar is amazing for keeping up with new releases from artists I follow. Best feature for discovery in my opinion.", "helpful_count": 25},
    {"review_id": "s8", "source": "play_store", "rating": 2, "date": "2025-03-18T00:00:00",
     "review_text": "Autoplay and radio just loop the same popular songs. I want mood based playlists that actually match my workout, not random repetitive tracks.", "helpful_count": 19},
    {"review_id": "s9", "source": "reddit", "rating": None, "date": "2025-03-25T00:00:00",
     "review_text": "Power user here, premium for 6 years across multiple devices. The recommendations got worse, same songs on repeat. I wish there was more control to tell it what I dislike.", "helpful_count": 140},
    {"review_id": "s10", "source": "app_store", "rating": 5, "date": "2025-04-03T00:00:00",
     "title": "Perfect for moods", "review_text": "The mood playlists are perfect. I love how it matches my vibe for focus, sleep, and parties. Great personalized discovery.", "helpful_count": 9},
    {"review_id": "s11", "source": "play_store", "rating": 1, "date": "2025-04-12T00:00:00",
     "review_text": "Boring and repetitive. The discover weekly is stale, nothing new for weeks. The algorithm clearly doesn't know my taste anymore.", "helpful_count": 51},
    {"review_id": "s12", "source": "reddit", "rating": None, "date": "2025-04-22T00:00:00",
     "review_text": "I share playlists with friends and use Blend a lot. Social discovery through friends' music is the best way I find new songs, better than the algorithm.", "helpful_count": 64},
    {"review_id": "s13", "source": "app_store", "rating": 3, "date": "2025-05-05T00:00:00",
     "title": "Nostalgia only", "review_text": "I mostly listen to old songs and classics from the 90s for comfort. Spotify keeps pushing new music I don't want. Let me stay in my throwback zone.", "helpful_count": 14},
    {"review_id": "s14", "source": "play_store", "rating": 4, "date": "2025-05-16T00:00:00",
     "review_text": "Smart Shuffle and the AI DJ actually helped me explore new artists. Finally some real discovery, recommendations feel fresh again.", "helpful_count": 22},
    {"review_id": "s15", "source": "reddit", "rating": None, "date": "2025-05-28T00:00:00",
     "review_text": "The biggest unmet need: I want better control over discovery. A dislike button that works and a way to break out of the same genre bubble.", "helpful_count": 103},
    {"review_id": "s16", "source": "app_store", "rating": 2, "date": "2025-06-04T00:00:00",
     "title": "Same songs", "review_text": "Every playlist has the same songs. Discovery is dead. I keep hearing the same 30 tracks no matter what I do.", "helpful_count": 37},
    {"review_id": "s17", "source": "play_store", "rating": 5, "date": "2025-06-12T00:00:00",
     "review_text": "Daily Mix nailed my taste this month, discovered so many new artists. When the algorithm works it's incredible for finding new music.", "helpful_count": 16},
    {"review_id": "s18", "source": "reddit", "rating": None, "date": "2025-06-20T00:00:00",
     "review_text": "Casual listener, I just want chill background music. The recommendations are okay but repetitive over long sessions. Not a dealbreaker though.", "helpful_count": 8},
]


def seed_store(store) -> dict:
    """Run sample reviews through the pipeline and index them. Returns a summary."""
    from processors.text_cleaner import process_records
    from processors.sentiment_analyzer import SentimentAnalyzer
    from processors.theme_extractor import ThemeExtractor

    records = process_records(SAMPLE_REVIEWS, drop_non_english=True)
    records = SentimentAnalyzer().analyze_batch(records)
    result = ThemeExtractor().extract(records)  # keyword strategy (no API key)
    records = result.get("records", records)

    indexed = store.build_indexes(records)
    logger.info("Seeded %d sample reviews", len(records))
    return {"seeded": len(records), "indexed": indexed, "theme_counts": result.get("theme_counts", {})}
