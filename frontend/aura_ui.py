"""Aura AI Stitch shell — HTML home + theme helpers for Streamlit."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STITCH_HTML = Path(_ROOT) / "screens" / "stitch_ai_review_discovery_dashboard" / "code.html"

# Aura nav label → Streamlit page name
AURA_NAV: list[tuple[str, str, str]] = [
    ("Home", "home", "Home"),
    ("New Discovery Session", "add", "Live Reviews"),
    ("Discovery Lab", "science", "Discovery Questions"),
    ("Library", "library_music", "Corpus Overview"),
    ("Search", "search", "Evidence Explorer"),
    ("Segments", "groups", "Segment Deep Dive"),
    ("AI Roadmap", "smart_toy", "Conclusion"),
    ("Raw Data", "database", "Raw Data"),
]


def stitch_html_path() -> Path:
    return STITCH_HTML


def _load_stitch_html() -> str:
    if not STITCH_HTML.is_file():
        return "<p>Stitch export not found. Add screens/stitch_ai_review_discovery_dashboard/code.html</p>"
    return STITCH_HTML.read_text(encoding="utf-8")


def _greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _inject_live_data(html: str, stats: dict | None, themes: list[dict]) -> str:
    """Patch static Stitch copy with corpus-backed snippets."""
    total = (stats or {}).get("total_reviews", 0)
    avg = (stats or {}).get("average_rating")
    sentiment = (stats or {}).get("by_sentiment", {})
    pos = sentiment.get("positive", 0)
    neg = sentiment.get("negative", 0)

    if total:
        aura = "Positive-leaning" if pos >= neg else "Mixed signals"
        html = html.replace(
            "Your sonic aura is currently <strong class=\"text-primary-container\">Mellow &amp; Focused</strong>.",
            f"Your review corpus is <strong class=\"text-primary-container\">{total} reviews</strong> "
            f"with {aura.lower()} sentiment.",
        )
    html = html.replace("Good evening, <span class=\"text-primary\">Alex</span>", 
                        f"{_greeting()}, <span class=\"text-primary\">Researcher</span>")

    if themes:
        labels = [t["theme"].replace("_", " ").title() for t in themes[:3]]
        chips = "".join(
            f'<span class="px-3 py-1 bg-surface-container-lowest rounded-full font-label-md '
            f'text-label-md text-on-surface-variant">{lb}</span>'
            for lb in labels
        )
        html = re.sub(
            r'<div class="hidden md:flex gap-2">.*?</div>',
            f'<div class="hidden md:flex gap-2">{chips}</div>',
            html,
            count=1,
            flags=re.DOTALL,
        )
        top = themes[0]
        html = html.replace(
            "Neural-network curated ambient textures for deep, uninterrupted workflow.",
            f"Top theme in your corpus: {top['theme'].replace('_', ' ')} ({top.get('count', 0)} mentions).",
        )
        html = html.replace("Cerebral Synapse", "Discovery Insights")
        html = html.replace(
            "Based on your Tuesday listening habits.",
            f"Average rating {avg or '—'} across indexed reviews.",
        )
        html = html.replace("Morning Coffee Acoustic", top["theme"].replace("_", " ").title())

    return html


def _html_for_embed(raw: str) -> str:
    """Hide Stitch chrome — Streamlit sidebar handles navigation."""
    hide = """
    <style>
      nav, header { display: none !important; }
      main { margin-left: 0 !important; padding-top: 1.5rem !important; }
      body { overflow-x: hidden; }
    </style>
    """
    return raw.replace("</head>", hide + "</head>", 1)


def render_aura_home(stats: dict | None, themes: list[dict]) -> None:
    """Full-width Aura AI home canvas (Stitch HTML + live corpus hints)."""
    html = _inject_live_data(_load_stitch_html(), stats, themes)
    components.html(_html_for_embed(html), height=1050, scrolling=True)


def inject_aura_css() -> None:
    """Global Aura / Stitch styling for Streamlit chrome."""
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@600&family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap" rel="stylesheet"/>
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
        <style>
          .stApp {
            background-color: #f8f9fa;
            font-family: 'Plus Jakarta Sans', sans-serif;
          }
          section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #bccbb9;
          }
          section[data-testid="stSidebar"] .stRadio > label {
            display: none;
          }
          section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-family: 'Plus Jakarta Sans', sans-serif;
          }
          .aura-brand-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.75rem;
            font-weight: 800;
            color: #006e2d;
            margin: 0;
            line-height: 1.1;
          }
          .aura-brand-sub {
            font-family: 'Hanken Grotesk', sans-serif;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #3d4a3d;
            margin-top: 4px;
          }
          header[data-testid="stHeader"] {
            background: rgba(248, 249, 250, 0.85);
            backdrop-filter: blur(8px);
          }
          div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.04);
          }
          .stButton>button {
            border-radius: 999px;
            font-family: 'Hanken Grotesk', sans-serif;
            font-weight: 600;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_aura_sidebar_header() -> None:
    st.markdown(
        """
        <p class="aura-brand-title">Discovery Analyzer</p>
        <p class="aura-brand-sub">Discovery Mode</p>
        """,
        unsafe_allow_html=True,
    )


def aura_nav_labels() -> list[str]:
    return [item[2] for item in AURA_NAV]


def aura_page_icons() -> dict[str, str]:
    return {item[2]: item[1] for item in AURA_NAV}
