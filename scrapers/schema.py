"""Unified review schema and shared helpers for all scrapers.

Defines a single Pydantic model, :class:`ReviewData`, that both the App Store
and Play Store scrapers normalize into, plus shared utilities for anonymizing
usernames, generating deterministic review ids, and persisting results to
timestamped JSON + CSV files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Project-level data directory shared by every scraper.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Source = Literal["app_store", "play_store", "reddit", "twitter"]


class ReviewData(BaseModel):
    """Canonical, source-agnostic representation of a single review."""

    review_id: str
    source: Source
    username: str  # anonymized pseudonym, never the raw name
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    title: Optional[str] = None
    review_text: str = ""
    date: Optional[str] = None  # ISO 8601 string
    version: Optional[str] = None
    helpful_count: Optional[int] = None
    # Source-specific extras (e.g. Reddit subreddit, num_comments, comments).
    metadata: Optional[dict[str, Any]] = None

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, value):
        """Coerce to int and drop out-of-range values instead of erroring."""
        if value is None or value == "":
            return None
        try:
            rating = int(value)
        except (TypeError, ValueError):
            return None
        return rating if 1 <= rating <= 5 else None

    @field_validator("title", "review_text", mode="before")
    @classmethod
    def _none_to_empty(cls, value):
        return value if value is not None else ""


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def anonymize_username(username: str) -> str:
    """Return a stable, non-reversible pseudonym for a username."""
    if not username:
        return "anon_unknown"
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()
    return f"anon_{digest[:12]}"


def make_review_id(source: str, username: str, date_str: str, text: str) -> str:
    """Create a deterministic id so identical reviews dedupe across runs."""
    raw = f"{source}|{username}|{date_str}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def to_iso(value) -> Optional[str]:
    """Normalize a datetime (or stringy date) to an ISO 8601 string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def save_reviews(
    reviews: list[ReviewData] | list[dict],
    prefix: str,
    data_dir: Path = DATA_DIR,
) -> tuple[Path, Path]:
    """Write reviews to timestamped JSON and CSV files.

    Args:
        reviews: A list of :class:`ReviewData` instances or plain dicts.
        prefix: Filename prefix (e.g. ``spotify_play_store_reviews``).
        data_dir: Output directory (created if missing).

    Returns:
        A ``(json_path, csv_path)`` tuple.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict] = [
        r.model_dump() if isinstance(r, ReviewData) else r for r in reviews
    ]

    json_path = data_dir / f"{prefix}_{timestamp}.json"
    csv_path = data_dir / f"{prefix}_{timestamp}.csv"

    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    field_names = list(ReviewData.model_fields.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Flatten nested structures (e.g. metadata) to JSON for CSV cells.
            csv_row = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(csv_row)

    logger.info("Saved %d reviews -> %s | %s", len(rows), json_path, csv_path)
    return json_path, csv_path
