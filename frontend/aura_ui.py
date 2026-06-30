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
          .home-hero-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            color: #006e2d;
            margin: 0 0 8px 0;
            line-height: 1.2;
          }
          .home-hero-sub {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.05rem;
            color: #3d4a3d;
            margin: 0;
            max-width: 720px;
            line-height: 1.55;
          }
          .home-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid #bccbb9;
            border-radius: 16px;
            padding: 18px 20px;
            height: 100%;
            box-shadow: 0 4px 24px rgba(0,0,0,0.04);
          }
          .home-card h4 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: #006e2d;
            margin: 0 0 6px 0;
          }
          .home-card p {
            font-size: 0.9rem;
            color: #3d4a3d;
            margin: 0;
            line-height: 1.5;
          }
          .home-pipeline-step {
            background: #ffffff;
            border-left: 4px solid #006e2d;
            border-radius: 0 12px 12px 0;
            padding: 12px 16px;
            margin-bottom: 10px;
          }
          .home-pipeline-step strong {
            color: #006e2d;
            font-family: 'Hanken Grotesk', sans-serif;
          }
          .home-nav-row {
            padding: 10px 0;
            border-bottom: 1px solid #e8ede7;
          }
          .home-nav-row:last-child { border-bottom: none; }
          .home-nav-label {
            font-weight: 700;
            color: #006e2d;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_aura_sidebar_header() -> None:
    st.markdown(
        """
        <p class="aura-brand-title">Spotify Discovery Engine</p>
        <p class="aura-brand-sub">AI-Powered Review Research</p>
        """,
        unsafe_allow_html=True,
    )


def render_product_home(
    fetch_sources: dict[str, dict[str, str]],
    *,
    indexed_count: int = 0,
    llm_ready: bool = False,
) -> None:
    """Home tab — product intro, data sources, pipeline; no charts or tab duplicates."""
    st.markdown(
        """
        <p class="home-hero-title">Spotify AI-Powered Review Discovery Engine</p>
        <p class="home-hero-sub">
          Collects real public feedback about Spotify music discovery, analyzes it with NLP and LLMs,
          and turns it into cited insights for product research and graduation-project evaluation.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("#### What this analyzer is for")
        st.markdown(
            "Users talk about discovery everywhere — app store ratings, Reddit threads, and social posts. "
            "This engine gathers that scattered feedback, normalizes it into one corpus, and helps you "
            "investigate **why discovery breaks down** (repetition, irrelevant recommendations, lack of control) "
            "with answers grounded in actual review text — not generic assumptions."
        )
        st.markdown(
            "It is built as an end-to-end research pipeline: **fetch → clean & score → embed → retrieve → synthesize**. "
            "Use it to support UX and product decisions with evidence, segment-level comparisons, and a prioritized feature roadmap."
        )

        st.markdown("#### How analysis works")
        for title, body in (
            (
                "1 · Collect",
                "Live scrapers pull Spotify-related reviews from the sources below. "
                "Each review is stored in a unified schema regardless of origin.",
            ),
            (
                "2 · Understand",
                "Text is cleaned, language-detected, and scored with VADER sentiment. "
                "Discovery-related phrases and themes are tagged automatically.",
            ),
            (
                "3 · Index",
                "Reviews are embedded and indexed for semantic search so you can find "
                "relevant evidence by meaning, not just keywords.",
            ),
            (
                "4 · Investigate",
                "A RAG layer retrieves matching reviews and synthesizes prose answers "
                "with quotes, confidence, and pain points for structured research questions.",
            ),
        ):
            st.markdown(
                f'<div class="home-pipeline-step"><strong>{title}</strong><br/>{body}</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### Where reviews are fetched from")
        by_category: dict[str, list[tuple[str, str]]] = {}
        source_notes = {
            "play_store": "Public Android app reviews — no API key required.",
            "app_store": "Public iOS app reviews from the Apple App Store.",
            "community_forums": "Discovery-related posts from music subreddits (Reddit API credentials).",
            "social_media": "Recent public posts about Spotify discovery on X / Twitter (bearer token).",
        }
        for sid, meta in fetch_sources.items():
            cat = meta.get("category", "Other")
            by_category.setdefault(cat, []).append((meta["label"], source_notes.get(sid, "")))

        for category, items in by_category.items():
            st.caption(category.upper())
            for label, note in items:
                st.markdown(
                    f'<div class="home-card" style="margin-bottom:10px">'
                    f'<h4>{label}</h4><p>{note}</p></div>',
                    unsafe_allow_html=True,
                )

        st.caption(
            "All sources are **public user-generated content**. "
            "This tool does not access Spotify internal analytics or private listening data."
        )

    st.markdown("---")

    st.markdown("#### Where to go in this dashboard")
    nav_guide = [
        ("New Discovery Session", "Pull fresh reviews and run analysis to build your session corpus."),
        ("Discovery Lab", "Six research questions with AI-synthesized answers, pain points, and quotes."),
        ("Library", "Charts and counts for the indexed corpus — sentiment, sources, themes."),
        ("Search", "Semantic evidence search across all indexed reviews."),
        ("Segments", "Listener-type profiles and how discovery pain differs by segment."),
        ("AI Roadmap", "RICE-scored feature prioritization from corpus patterns."),
        ("Raw Data", "Browse and filter individual review records."),
    ]
    ncols = st.columns(2)
    for i, (label, desc) in enumerate(nav_guide):
        with ncols[i % 2]:
            st.markdown(
                f'<div class="home-nav-row">'
                f'<span class="home-nav-label">{label}</span> — {desc}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews in session", indexed_count if indexed_count else "—")
    c2.metric("LLM synthesis", "Ready" if llm_ready else "Extractive mode")
    c3.metric("Research questions", "6 structured")
    if not indexed_count:
        st.info(
            "No reviews indexed yet. Open **New Discovery Session** in the sidebar to fetch "
            "from app stores or community sources, then explore the other tabs."
        )


def aura_nav_labels() -> list[str]:
    return [item[2] for item in AURA_NAV]


def aura_page_icons() -> dict[str, str]:
    return {item[2]: item[1] for item in AURA_NAV}
