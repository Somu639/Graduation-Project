"""Spotify Discovery Analyzer - interactive Streamlit dashboard.

Research-console UI organized around six discovery questions, plus live review
collection, corpus overview, evidence explorer, segments, and raw data.

Run with the API up:
    uvicorn api.main:app --reload
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

# Load .env before any API-key or scraper configuration is read.
try:
    from dotenv import load_dotenv

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Standalone mode on Streamlit Cloud (no separate FastAPI server).
os.environ.setdefault("LOCAL_MODE", "1")
os.environ.setdefault("VECTOR_STORE", "memory")

import requests
import streamlit as st

# Make the project root importable so the in-process backend can load
# rag/agents/api modules when running standalone (e.g. on Streamlit Cloud).
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Local (in-process) mode: used automatically when no FastAPI server is reachable.
# Force it with LOCAL_MODE=1, or force HTTP with USE_API=1.
import local_service as svc  # noqa: E402  (after sys.path setup)
from processors.llm_client import bootstrap_env, llm_configured  # noqa: E402
from scrapers.source_registry import (  # noqa: E402
    FETCH_SOURCES,
    fetch_source_help,
    normalize_sources,
    source_label,
)

bootstrap_env()

# Light theme palette (Spotify green accent).
ACCENT = "#1DB954"
ACCENT_HOVER = "#1ED760"
ACCENT_SOFT = "#E8F8EE"
BG = "#F4F6F9"
SURFACE = "#FFFFFF"
SIDEBAR_BG = "#FFFFFF"
BORDER = "#E2E8F0"
TEXT = "#111827"
TEXT_MUTED = "#6B7280"
TEXT_SOFT = "#9CA3AF"
SHADOW = "0 2px 8px rgba(15, 23, 42, 0.06)"
SHADOW_HOVER = "0 8px 24px rgba(29, 185, 84, 0.15)"
GREEN_SCALE = ["#1DB954", "#34D399", "#059669", "#10B981", "#6EE7B7", "#047857"]
SENTIMENT_COLORS = {
    "positive": "#059669",
    "negative": "#DC2626",
    "neutral": "#6B7280",
    "mixed": "#D97706",
}

# Core research questions the dashboard is designed to answer.
DISCOVERY_QUESTIONS: list[dict] = [
    {
        "id": "discovery_struggle",
        "icon": "🔍",
        "short_title": "Discovery struggle",
        "question": "Why do users struggle to discover new music?",
        "lens": "Root causes — algorithm limits, repetition, onboarding, filter bubbles",
        "summary": (
            "Users often report that recommendations feel repetitive, that Discover Weekly "
            "and Release Radar surface the same artists, and that the app pushes familiar "
            "content instead of genuinely new music. Onboarding rarely teaches exploration habits."
        ),
        "evidence": ["insight", "themes"],
    },
    {
        "id": "recommendation_frustrations",
        "icon": "😤",
        "short_title": "Frustrations",
        "question": "What are the most common frustrations with recommendations?",
        "lens": "Ranked frustrations and representative quotes from the corpus",
        "summary": (
            "Top complaints include irrelevant suggestions, genre mismatch, stale playlists, "
            "over-reliance on past listening, and lack of control over why something was recommended."
        ),
        "evidence": ["insight", "frustrations"],
    },
    {
        "id": "desired_behaviors",
        "icon": "🎧",
        "short_title": "Behaviors",
        "question": "What listening behaviors are users trying to achieve?",
        "lens": "Moods, contexts, exploration goals, and playlist habits",
        "summary": (
            "Reviews describe mood-based listening, workout/study/background contexts, "
            "social sharing, deep genre exploration, and lean-back vs active discovery modes."
        ),
        "evidence": ["insight", "themes", "behaviors"],
    },
    {
        "id": "repetitive_listening",
        "icon": "🔁",
        "short_title": "Repetition",
        "question": "What causes users to repeatedly listen to the same content?",
        "lens": "Repetition triggers, stale playlists, and comfort listening",
        "summary": (
            "Comfort listening, nostalgia, algorithm loops, and limited fresh suggestions "
            "lead users to replay the same songs, artists, and playlists rather than explore."
        ),
        "evidence": ["insight", "quotes_repetitive"],
    },
    {
        "id": "segment_challenges",
        "icon": "👥",
        "short_title": "Segments",
        "question": "Which user segments experience different discovery challenges?",
        "lens": "Segment sizes, problem heatmap, and per-segment pain points",
        "summary": (
            "Casual listeners, power users, genre enthusiasts, and new users describe "
            "different discovery pain points — from overwhelm to filter bubbles to cold-start gaps."
        ),
        "evidence": ["segments"],
    },
    {
        "id": "unmet_needs",
        "icon": "💡",
        "short_title": "Unmet needs",
        "question": "What unmet needs emerge consistently across reviews?",
        "lens": "Cross-cutting gaps users ask for but do not get today",
        "summary": (
            "Users repeatedly ask for better genre depth, more control, fresher discovery feeds, "
            "clearer explanations of recommendations, and tools to break out of repetition."
        ),
        "evidence": ["insight", "themes", "frustrations"],
    },
]

st.set_page_config(page_title="Spotify Discovery Analyzer", page_icon="🎧", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {BG} 0%, #EEF2F7 100%);
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {SIDEBAR_BG};
            border-right: 1px solid {BORDER};
            box-shadow: 2px 0 12px rgba(15, 23, 42, 0.04);
        }}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {TEXT};
        }}
        h1, h2, h3, h4 {{ color: {TEXT}; font-weight: 700; }}
        .stMarkdown, .stText, label, p {{ color: {TEXT}; }}
        div[data-testid="stMetric"] {{
            background-color: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: {SHADOW};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: {SHADOW_HOVER};
        }}
        div[data-testid="stMetricValue"] {{ color: {ACCENT}; font-weight: 800; }}
        div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
        .stButton>button {{
            background-color: {ACCENT};
            color: #FFFFFF;
            border: none;
            border-radius: 999px;
            font-weight: 700;
            padding: 0.5rem 1.4rem;
            transition: background-color 0.2s ease, transform 0.15s ease;
        }}
        .stButton>button:hover {{
            background-color: {ACCENT_HOVER};
            color: #FFFFFF;
            transform: translateY(-1px);
        }}
        .stButton>button[kind="secondary"] {{
            background-color: {SURFACE};
            color: {TEXT};
            border: 1px solid {BORDER};
        }}
        .hero-banner {{
            background: linear-gradient(135deg, {ACCENT_SOFT} 0%, {SURFACE} 55%, #F0FDF4 100%);
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 20px;
            box-shadow: {SHADOW};
        }}
        .hero-title {{ font-size: 1.55rem; font-weight: 800; color: {TEXT}; margin: 0 0 6px 0; }}
        .hero-sub {{ color: {TEXT_MUTED}; font-size: 0.95rem; margin: 0; line-height: 1.5; }}
        .quote-card {{
            background-color: {SURFACE};
            border: 1px solid {BORDER};
            border-left: 4px solid {ACCENT};
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 12px;
            box-shadow: {SHADOW};
            transition: box-shadow 0.2s ease;
        }}
        .quote-card:hover {{ box-shadow: {SHADOW_HOVER}; }}
        .quote-meta {{ color: {TEXT_MUTED}; font-size: 0.82rem; margin-top: 8px; }}
        .research-card {{
            background-color: {SURFACE};
            border: 1px solid {BORDER};
            border-left: 4px solid {ACCENT};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: {SHADOW};
        }}
        .research-lens {{ color: {TEXT_MUTED}; font-size: 0.92rem; line-height: 1.5; }}
        .step-pill {{
            display: inline-block;
            background: {ACCENT_SOFT};
            color: #047857;
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 8px;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; background: transparent; }}
        .stTabs [data-baseweb="tab"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px 10px 0 0;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ACCENT};
            border-bottom: 2px solid {ACCENT};
            font-weight: 700;
        }}
        div[data-testid="stExpander"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}
        .stRadio > label {{ font-weight: 600; color: {TEXT_MUTED}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_banner(title: str, subtitle: str, steps: list[str] | None = None) -> None:
    steps_html = ""
    if steps:
        pills = "".join(f'<span class="step-pill">{s}</span>' for s in steps)
        steps_html = f'<div style="margin-top:12px">{pills}</div>'
    st.markdown(
        f'<div class="hero-banner">'
        f'<p class="hero-title">{title}</p>'
        f'<p class="hero-sub">{subtitle}</p>{steps_html}</div>',
        unsafe_allow_html=True,
    )


def sentiment_badge(sentiment: str) -> str:
    key = (sentiment or "neutral").lower()
    color = SENTIMENT_COLORS.get(key, TEXT_MUTED)
    label = key.capitalize() if key else "Unknown"
    return (
        f'<span style="background:{color}22;color:{color};padding:2px 10px;'
        f'border-radius:999px;font-size:0.75rem;font-weight:700;">{label}</span>'
    )


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def _use_local() -> bool:
    """Decide whether to use the in-process backend (cached per session)."""
    if os.getenv("LOCAL_MODE", "").lower() in ("1", "true", "yes"):
        return True
    if os.getenv("USE_API", "").lower() in ("1", "true", "yes"):
        return False
    if "use_local" not in st.session_state:
        st.session_state["use_local"] = not _check_health()
    return st.session_state["use_local"]


def _local_get(path: str, params: dict):
    p = params or {}
    if path == "/stats/overview":
        return svc.stats_overview()
    if path == "/stats/timeline":
        return svc.stats_timeline()
    if path == "/insights/themes":
        return svc.insights_themes(int(p.get("top_k", 20)), p.get("source"), p.get("sentiment"))
    if path == "/insights/frustrations":
        return svc.insights_frustrations(int(p.get("top_k", 15)))
    if path == "/insights/segments":
        return svc.insights_segments(str(p.get("full", "false")).lower() in ("true", "1", "yes"))
    if path == "/insights/question":
        try:
            return svc.insights_question(p.get("question", ""), p.get("segment"), p.get("source"))
        except Exception as exc:  # noqa: BLE001
            return {
                "question": p.get("question", ""),
                "insight": "Insight generation failed — showing cached or extractive data only.",
                "confidence": 0.0,
                "supporting_evidence": [],
                "sample_size": 0,
                "themes_identified": [],
                "recommended_followup_questions": [],
                "llm_fallback": True,
                "llm_error": str(exc),
            }
    if path == "/data/search":
        return svc.data_search(p.get("q"), p.get("source"), p.get("sentiment"),
                               p.get("rating"), p.get("theme"), int(p.get("limit", 50)))
    raise ValueError(f"No local route for GET {path}")


def _local_post(path: str, body: dict):
    if path == "/data/seed":
        return svc.seed()
    if path == "/data/fetch-live":
        return svc.fetch_live(
            body.get("sources", ["play_store"]),
            int(body.get("limit", 50)),
            use_llm=bool(body.get("use_llm", True)),
            discovery_filter=bool(body.get("discovery_filter", False)),
        )
    if path == "/agent/research":
        try:
            return svc.agent_research(body.get("research_questions", []))
        except Exception as exc:  # noqa: BLE001
            return {
                "questions": body.get("research_questions", []),
                "findings": [],
                "segments_affected": "",
                "followup_questions": [],
                "summary": f"Research session failed: {exc}",
            }
    if path == "/export/report":
        return svc.export_report(body.get("format", "markdown"), body.get("title", "Report"))
    raise ValueError(f"No local route for POST {path}")


def api_get(path: str, params: dict | None = None, timeout: int = 120):
    if _use_local():
        try:
            return _local_get(path, params or {})
        except Exception as exc:  # noqa: BLE001
            if path == "/insights/question":
                return {
                    "question": (params or {}).get("question", ""),
                    "insight": "Could not reach the insight engine.",
                    "confidence": 0.0,
                    "supporting_evidence": [],
                    "sample_size": 0,
                    "themes_identified": [],
                    "recommended_followup_questions": [],
                    "llm_fallback": True,
                    "llm_error": str(exc),
                }
            st.error(f"{path} failed: {exc}")
            return None
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        if path == "/insights/question":
            return {
                "question": (params or {}).get("question", ""),
                "insight": "Could not reach insight API — using offline mode.",
                "confidence": 0.0,
                "supporting_evidence": [],
                "sample_size": 0,
                "themes_identified": [],
                "recommended_followup_questions": [],
                "llm_fallback": True,
                "llm_error": str(exc),
            }
        st.error(f"GET {path} failed: {exc}")
        return None


def api_post(path: str, payload: dict, timeout: int = 300):
    if _use_local():
        try:
            return _local_post(path, payload or {})
        except Exception as exc:  # noqa: BLE001
            st.error(f"{path} failed: {exc}")
            return None
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"POST {path} failed: {exc}")
        return None


def _check_health() -> bool:
    """Ping the root /health endpoint (lives outside the /api/v1 prefix)."""
    root = API_BASE.split("/api/")[0]
    try:
        resp = requests.get(f"{root}/health", timeout=5)
        return resp.ok
    except requests.RequestException:
        return False


def plotly_layout(fig):
    """Apply the light theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=BG,
        font_color=TEXT,
        margin=dict(l=10, r=10, t=40, b=10),
        colorway=GREEN_SCALE,
    )
    return fig


def quote_card(quote: str, source: str = "", sentiment: str = "") -> None:
    border = SENTIMENT_COLORS.get((sentiment or "").lower(), ACCENT)
    badge = sentiment_badge(sentiment) if sentiment and sentiment != "n/a" else ""
    meta_parts = [source_label(source) if source else ""]
    meta = " · ".join(meta_parts)
    st.markdown(
        f'<div class="quote-card" style="border-left-color:{border}">'
        f'"{quote}"<br>'
        f'<span class="quote-meta">{meta} {badge}</span></div>',
        unsafe_allow_html=True,
    )


def _question_by_id(qid: str) -> dict:
    return next(q for q in DISCOVERY_QUESTIONS if q["id"] == qid)


def _fetch_insight(question: str, source: str | None = None, refresh: bool = False) -> dict:
    """Always returns a dict — never None, never raises."""
    cache_key = f"insight::{question}::{source or 'all'}"
    if refresh and cache_key in st.session_state:
        del st.session_state[cache_key]
    if cache_key not in st.session_state:
        src = source if source and source != "all" else None
        with st.spinner("Synthesizing evidence-backed insight…"):
            if _use_local():
                try:
                    result = svc.insights_question(question, source=src)
                except Exception as exc:  # noqa: BLE001
                    result = _empty_insight(question, str(exc))
            else:
                result = api_get("/insights/question", {"question": question, **({"source": src} if src else {})})
                if not result:
                    result = _empty_insight(question, "API request failed")
        st.session_state[cache_key] = result or _empty_insight(question, "No result")
    return st.session_state[cache_key]


def _empty_insight(question: str, error: str = "") -> dict:
    return {
        "question": question,
        "insight": "Fetch live reviews first, then analyze this question.",
        "confidence": 0.0,
        "supporting_evidence": [],
        "sample_size": 0,
        "themes_identified": [],
        "recommended_followup_questions": [],
        "llm_fallback": bool(error),
        "llm_error": error,
    }


def _render_insight_block(result: dict, question: str = "", source: str = "all") -> None:
    if result.get("llm_fallback") and result.get("llm_error"):
        err = result.get("llm_error", "")
        if "401" in err or "invalid api key" in err.lower():
            st.warning(
                "LLM API key rejected — showing review-based summary instead. "
                "Update **GROQ_API_KEY** in Streamlit Secrets (Settings → Secrets), "
                "set `LLM_PROVIDER=groq`, then reboot the app."
            )
        else:
            st.warning(
                "LLM synthesis unavailable — showing extractive summary from reviews. "
                f"({err[:120]})"
            )

    tab_answer, tab_quotes, tab_next = st.tabs(["Answer", "Quotes", "Follow-ups"])
    with tab_answer:
        st.markdown(result.get("insight", ""))
        m1, m2, m3 = st.columns(3)
        conf = result.get("confidence", 0)
        m1.metric("Confidence", f"{conf:.0%}")
        m2.metric("Reviews analyzed", result.get("sample_size", 0))
        m3.metric("Themes found", len(result.get("themes_identified") or []))
        if result.get("themes_identified"):
            st.markdown("**Key themes**")
            theme_cols = st.columns(min(4, len(result["themes_identified"])))
            for i, theme in enumerate(result["themes_identified"][:8]):
                theme_cols[i % len(theme_cols)].markdown(
                    f'<span style="background:{ACCENT_SOFT};color:#047857;padding:6px 12px;'
                    f'border-radius:999px;font-size:0.85rem;font-weight:600;">{theme}</span>',
                    unsafe_allow_html=True,
                )

    evidence = result.get("supporting_evidence", [])
    with tab_quotes:
        if evidence:
            for ev in evidence[:8]:
                quote_card(ev.get("quote", ""), ev.get("source", ""), ev.get("sentiment", ""))
        else:
            st.caption("No supporting quotes returned for this question.")

    with tab_next:
        followups = result.get("recommended_followup_questions") or []
        if followups:
            st.caption("Suggested follow-ups — use the **chat box** below to explore these:")
            for fq in followups:
                st.write(f"- {fq}")
        else:
            st.caption("Use the chat box below to ask follow-up questions.")

    if question and st.button(
        "Refresh insight",
        key=f"refresh_{question[:24]}",
        type="secondary",
    ):
        _fetch_insight(question, source if source != "all" else None, refresh=True)
        st.rerun()


def _render_question_evidence(spec: dict) -> None:
    """Supporting charts and quotes tailored to each research question."""
    for kind in spec["evidence"]:
        if kind == "insight":
            continue
        if kind == "frustrations":
            st.markdown("#### Frustration ranking")
            _render_frustrations_block()
        elif kind == "themes":
            st.markdown("#### Related themes")
            _render_themes_block()
        elif kind == "behaviors":
            _render_behavior_themes()
        elif kind == "quotes_repetitive":
            st.markdown("#### Repetition-related reviews")
            _render_repetitive_quotes()
        elif kind == "segments":
            full = st.toggle(
                "Include full LLM segment profiles (slower)",
                value=False,
                key=f"seg_full_{spec['id']}",
            )
            _render_segments_block(full_profiles=full)


def _chat_history_key(question_id: str) -> str:
    return f"dq_chat_{question_id}"


def _ask_in_chat(spec: dict, user_prompt: str, source: str | None) -> dict:
    """Answer a follow-up in the context of a research question."""
    src = source if source and source != "all" else None
    phrased = (
        f'Research question: "{spec["question"]}"\n'
        f"User follow-up: {user_prompt}\n"
        "Answer using only evidence from indexed Spotify discovery reviews."
    )
    cache_key = f"chat::{spec['id']}::{user_prompt}::{src or 'all'}"
    if cache_key not in st.session_state:
        if _use_local():
            try:
                st.session_state[cache_key] = svc.insights_question(phrased, source=src)
            except Exception as exc:  # noqa: BLE001
                st.session_state[cache_key] = _empty_insight(phrased, str(exc))
        else:
            result = api_get(
                "/insights/question",
                {"question": phrased, **({"source": src} if src else {})},
            )
            st.session_state[cache_key] = result or _empty_insight(phrased, "API failed")
    return st.session_state[cache_key]


def _render_discovery_chat(spec: dict, source: str, insight: dict) -> None:
    """Grounded chat box for follow-ups on the active research question."""
    hist_key = _chat_history_key(spec["id"])
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    st.markdown("#### Ask about this question")
    st.caption(
        "Follow-up questions are answered from your indexed reviews, "
        "in the context of the research question above."
    )

    for msg in st.session_state[hist_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for ev in msg.get("evidence", [])[:3]:
                quote_card(ev.get("quote", ""), ev.get("source", ""), ev.get("sentiment", ""))

    suggested = insight.get("recommended_followup_questions") or []
    if suggested and not st.session_state[hist_key]:
        st.markdown("**Try asking:**")
        sug_cols = st.columns(min(2, len(suggested)))
        for i, sq in enumerate(suggested[:4]):
            if sug_cols[i % len(sug_cols)].button(sq, key=f"suggest_{spec['id']}_{i}"):
                _append_chat_turn(spec, source, sq, hist_key)
                st.rerun()

    prompt = st.chat_input(
        f"Ask about {spec['short_title'].lower()}…",
        key=f"chat_input_{spec['id']}",
    )
    if prompt:
        _append_chat_turn(spec, source, prompt, hist_key)
        st.rerun()


def _append_chat_turn(spec: dict, source: str, prompt: str, hist_key: str) -> None:
    st.session_state[hist_key].append({"role": "user", "content": prompt})
    with st.spinner("Searching reviews for an answer…"):
        answer = _ask_in_chat(spec, prompt, source if source != "all" else None)
    st.session_state[hist_key].append(
        {
            "role": "assistant",
            "content": answer.get("insight", "No answer available."),
            "evidence": answer.get("supporting_evidence", []),
        }
    )


def _render_question_panel(spec: dict, source: str) -> None:
    """Full insight + evidence + chat for one research question."""
    st.markdown(f"### {spec['icon']} {spec['question']}")
    st.markdown(
        f'<div class="research-card">'
        f'<strong>Research lens</strong><br>'
        f'<span class="research-lens">{spec["lens"]}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**What we look for:** {spec['summary']}")

    insight = _fetch_insight(spec["question"], source if source != "all" else None)
    st.markdown("#### Evidence-backed insight")
    _render_insight_block(insight, spec["question"], source)

    with st.expander("Supporting evidence", expanded=False):
        _render_question_evidence(spec)

    _render_discovery_chat(spec, source, insight)
    data = api_get("/insights/frustrations", {"top_k": top_k})
    items = (data or {}).get("frustrations", [])
    if not items:
        st.caption("No frustration patterns indexed yet.")
        return
    try:
        import plotly.express as px

        fig = px.bar(
            x=[f["count"] for f in items],
            y=[f["frustration"] for f in items],
            orientation="h",
            labels={"x": "Mentions", "y": "Frustration"},
            color=[f["count"] for f in items],
            color_continuous_scale="Greens",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(plotly_layout(fig), use_container_width=True)
    except ImportError:
        pass

    labels = [f["frustration"] for f in items]
    chosen = st.selectbox(
        "Drill into a frustration",
        labels,
        index=0,
        key=f"frust_pick_{top_k}",
    )
    st.markdown("#### Representative quotes")
    for item in items:
        if item["frustration"] != chosen:
            continue
        ex = item.get("example", {})
        quote_card(ex.get("quote", item["frustration"]), ex.get("source", ""))
        break


def _render_themes_block(top_k: int = 15) -> None:
    data = api_get("/insights/themes", {"top_k": top_k})
    themes = (data or {}).get("themes", [])
    if not themes:
        st.caption("No themes indexed yet.")
        return
    try:
        import plotly.express as px

        fig = px.bar(
            x=[t["count"] for t in themes],
            y=[t["theme"] for t in themes],
            orientation="h",
            labels={"x": "Mentions", "y": "Theme"},
            color=[t["count"] for t in themes],
            color_continuous_scale="Greens",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(plotly_layout(fig), use_container_width=True)
    except ImportError:
        for t in themes[:10]:
            st.write(f"- **{t['theme']}** ({t['count']})")

    if themes:
        chip_cols = st.columns(min(5, len(themes)))
        for i, t in enumerate(themes[:10]):
            if chip_cols[i % len(chip_cols)].button(
                f"{t['theme']} ({t['count']})",
                key=f"theme_chip_{top_k}_{i}",
                use_container_width=True,
            ):
                st.session_state["theme_drill"] = t["theme"]
        if st.session_state.get("theme_drill"):
            st.markdown(f"**Reviews mentioning:** `{st.session_state['theme_drill']}`")
            data = api_get(
                "/data/search",
                {"q": st.session_state["theme_drill"].replace("_", " "), "limit": 6},
            )
            for r in (data or {}).get("results", []):
                meta = r.get("metadata", {})
                quote_card(r["content"][:280], meta.get("source", ""), meta.get("sentiment", ""))


def _render_behavior_themes() -> None:
    behavior_keys = (
        "mood", "playlist", "background", "workout", "study", "explore",
        "discover", "genre", "social", "nostalg", "casual", "power",
    )
    data = api_get("/insights/themes", {"top_k": 40})
    themes = (data or {}).get("themes", [])
    matched = [
        t for t in themes
        if any(k in t["theme"].lower() for k in behavior_keys)
    ]
    if matched:
        st.markdown("#### Behavior-related themes — click to explore")
        bcols = st.columns(min(4, len(matched)))
        for i, t in enumerate(matched[:12]):
            if bcols[i % len(bcols)].button(
                f"{t['theme']} ({t['count']})",
                key=f"beh_{t['theme']}_{i}",
                use_container_width=True,
            ):
                st.session_state["theme_drill"] = t["theme"]
        if st.session_state.get("theme_drill"):
            data = api_get(
                "/data/search",
                {"q": st.session_state["theme_drill"].replace("_", " "), "limit": 6},
            )
            for r in (data or {}).get("results", []):
                meta = r.get("metadata", {})
                quote_card(r["content"][:280], meta.get("source", ""), meta.get("sentiment", ""))
    else:
        _render_themes_block(12)


def _render_repetitive_quotes() -> None:
    data = api_get(
        "/data/search",
        {"q": "repetitive same songs discover weekly stale loop", "limit": 12},
    )
    results = (data or {}).get("results", [])
    if not results:
        st.caption("No repetition-related reviews found.")
        return
    for r in results:
        meta = r.get("metadata", {})
        quote_card(r["content"][:300], meta.get("source", ""), meta.get("sentiment", ""))


def _render_segments_block(full_profiles: bool = False) -> None:
    data = api_get("/insights/segments", {"full": str(full_profiles).lower()})
    if not data:
        return
    sizes = data.get("sizes", {})
    matrix = data.get("comparison_matrix", {})
    profiles = {p["segment"]: p for p in data.get("profiles", [])}

    st.markdown("#### Segment sizes")
    cols = st.columns(4)
    for i, (seg, info) in enumerate(sizes.items()):
        cols[i % 4].metric(
            seg.replace("_", " ").title(),
            f"{info.get('pct', 0)}%",
            f"{info.get('count', 0)} reviews",
        )

    problems = matrix.get("problems", [])
    seg_matrix = matrix.get("matrix", {})
    try:
        import plotly.express as px

        if problems and seg_matrix:
            st.markdown("#### Discovery challenges by segment")
            segs = list(seg_matrix.keys())
            z = [[seg_matrix[s][p]["pct"] for p in problems] for s in segs]
            fig = px.imshow(
                z, x=problems, y=segs, color_continuous_scale="Greens",
                labels=dict(color="% of segment mentioning problem"),
            )
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
    except ImportError:
        pass

    if full_profiles and profiles:
        st.markdown("#### Segment profiles")
        for seg in sizes:
            prof = profiles.get(seg)
            if not prof:
                continue
            with st.expander(seg.replace("_", " ").title()):
                _bullets("Pain points", prof.get("discovery_pain_points"))
                _bullets("Primary frustrations", prof.get("primary_frustrations"))
                for q in prof.get("sample_quotes", [])[:2]:
                    quote_card(q.get("quote", ""), q.get("source", ""))


def page_discovery_questions() -> None:
    hero_banner(
        "Discovery Research Questions",
        "Six research questions — each with its own evidence-backed insight and grounded chat.",
        ["Choose a question", "Read the insight", "Ask follow-ups in chat"],
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        source = st.selectbox(
            "Filter evidence by source",
            ["all", "app_store", "play_store", "reddit", "twitter"],
            format_func=lambda x: "All sources" if x == "all" else source_label(x),
            key="dq_source",
        )
    with c2:
        if st.button("Refresh insights", use_container_width=True):
            for k in list(st.session_state):
                if k.startswith("insight::") or k.startswith("chat::"):
                    del st.session_state[k]
                if k.startswith("dq_chat_"):
                    del st.session_state[k]
            st.toast("Insights and chat history cleared")
            st.rerun()

    ids = [q["id"] for q in DISCOVERY_QUESTIONS]
    labels = [f"{q['icon']} {q['short_title']}" for q in DISCOVERY_QUESTIONS]
    if "dq_radio" not in st.session_state:
        default_id = st.session_state.get("dq_selected", ids[0])
        default_idx = ids.index(default_id) if default_id in ids else 0
        st.session_state["dq_radio"] = labels[default_idx]

    picked_label = st.radio(
        "Research question",
        labels,
        key="dq_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    spec = DISCOVERY_QUESTIONS[labels.index(picked_label)]
    st.session_state["dq_selected"] = spec["id"]

    st.markdown("---")
    _render_question_panel(spec, source)

    with st.expander("Advanced: multi-question research agent", expanded=False):
        page_research_agent()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    hero_banner(
        "Corpus Overview",
        "See how healthy your review dataset is, then jump into guided research questions.",
    )

    st.markdown("##### Quick start — pick a research question")
    qcols = st.columns(3)
    for i, q in enumerate(DISCOVERY_QUESTIONS):
        if qcols[i % 3].button(
            f"{q['icon']} {q['question'][:40]}…" if len(q["question"]) > 40 else f"{q['icon']} {q['question']}",
            key=f"ov_{q['id']}",
            use_container_width=True,
        ):
            _goto_page("Discovery Questions", question_id=q["id"])

    stats = api_get("/stats/overview")
    if not stats:
        st.info("No data yet. Ingest reviews via the API or the Raw Data page.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reviews", stats.get("total_reviews", 0))
    c2.metric("Average rating", stats.get("average_rating") or "—")
    sources = stats.get("by_source", {})
    c3.metric("Sources", len(sources))
    dr = stats.get("date_range")
    c4.metric("Latest", (dr or {}).get("latest", "—")[:10] if dr else "—")

    try:
        import plotly.express as px
    except ImportError:
        st.warning("Install plotly for charts: pip install plotly")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Rating distribution")
        ratings = stats.get("ratings_distribution", {})
        if ratings:
            fig = px.bar(
                x=list(ratings.keys()), y=list(ratings.values()),
                labels={"x": "Rating", "y": "Reviews"},
                color_discrete_sequence=[ACCENT],
            )
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
    with right:
        st.subheader("Sentiment distribution")
        sentiment = stats.get("by_sentiment", {})
        if sentiment:
            fig = px.pie(
                names=list(sentiment.keys()), values=list(sentiment.values()),
                hole=0.45, color_discrete_sequence=GREEN_SCALE,
            )
            st.plotly_chart(plotly_layout(fig), use_container_width=True)

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["By source", "Over time", "Term cloud"])
    with chart_tab1:
        st.subheader("Reviews by source")
        if sources:
            fig = px.bar(
                x=list(sources.values()), y=list(sources.keys()), orientation="h",
                labels={"x": "Reviews", "y": "Source"},
                color=list(sources.values()),
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
    with chart_tab2:
        st.subheader("Review volume over time")
        timeline = api_get("/stats/timeline")
        series = (timeline or {}).get("series", [])
        if series:
            fig = px.area(
                x=[s["period"] for s in series], y=[s["count"] for s in series],
                labels={"x": "Month", "y": "Reviews"},
                color_discrete_sequence=[ACCENT],
            )
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
        else:
            st.caption("No dated reviews available for a timeline yet.")
    with chart_tab3:
        st.subheader("Discovery term cloud")
        themes = api_get("/insights/themes", {"top_k": 50})
        freqs = {t["theme"]: t["count"] for t in (themes or {}).get("themes", [])}
        if freqs:
            render_wordcloud(freqs)
        else:
            st.caption("No themes available yet.")


def render_wordcloud(freqs: dict) -> None:
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        st.warning("Install wordcloud + matplotlib for the term cloud.")
        st.json(freqs)
        return
    wc = WordCloud(
        width=1000, height=320, background_color=SURFACE, colormap="Greens",
    ).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def page_theme_explorer() -> None:
    """Themes tab content (used by Evidence Explorer)."""
    f1, f2, f3 = st.columns(3)
    source = f1.selectbox(
        "Source",
        ["all", "app_store", "play_store", "reddit", "twitter"],
        format_func=lambda x: "All sources" if x == "all" else source_label(x),
    )
    sentiment = f2.selectbox("Sentiment", ["all", "positive", "negative", "neutral", "mixed"])
    top_k = f3.slider("Themes to show", 5, 50, 20)

    params = {"top_k": top_k}
    if source != "all":
        params["source"] = source
    if sentiment != "all":
        params["sentiment"] = sentiment

    themes = api_get("/insights/themes", params)
    theme_list = (themes or {}).get("themes", [])
    if not theme_list:
        st.info("No themes available for these filters.")
        return

    try:
        import plotly.express as px
    except ImportError:
        st.warning("Install plotly for the treemap.")
        return

    # Build sunburst path: theme -> sentiment.
    rows = []
    for t in theme_list:
        sent_breakdown = t.get("sentiment") or {t.get("theme"): t["count"]}
        for sent, count in (sent_breakdown.items() if sent_breakdown else []):
            rows.append({"theme": t["theme"], "sentiment": sent or "unknown", "count": count})
    if rows:
        st.subheader("Theme sunburst (click to drill down)")
        fig = px.sunburst(
            rows, path=["theme", "sentiment"], values="count",
            color="count", color_continuous_scale="Greens",
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    st.subheader("Representative quotes")
    chosen = st.selectbox("Pick a theme", [t["theme"] for t in theme_list])
    if chosen:
        search_params = {"q": chosen.replace("_", " "), "theme": chosen, "limit": 6}
        if source != "all":
            search_params["source"] = source
        data = api_get("/data/search", search_params)
        for r in (data or {}).get("results", []):
            meta = r.get("metadata", {})
            quote_card(r["content"][:280], meta.get("source", ""), meta.get("sentiment", ""))


def page_evidence_explorer() -> None:
    hero_banner(
        "Evidence Explorer",
        "Filter themes and frustrations interactively — click chips and drill into quotes.",
    )
    tab_themes, tab_frustrations = st.tabs(["Themes", "Frustrations"])

    with tab_themes:
        page_theme_explorer()
    with tab_frustrations:
        st.subheader("Top frustrations")
        _render_frustrations_block(top_k=15)


def page_insight_qa() -> None:
    """Legacy — kept for imports; use Discovery Questions in the nav."""
    page_discovery_questions()


def page_segments() -> None:
    hero_banner(
        "Segment Deep Dive",
        "Compare how casual listeners, power users, and other segments experience discovery differently.",
    )
    full = st.toggle("Include full LLM profiles (slower)", value=False)
    with st.spinner("Loading segments..."):
        data = api_get("/insights/segments", {"full": str(full).lower()})
    if not data:
        return

    sizes = data.get("sizes", {})
    matrix = data.get("comparison_matrix", {})
    profiles = {p["segment"]: p for p in data.get("profiles", [])}

    st.subheader("Segment sizes (% of reviews)")
    cols = st.columns(4)
    for i, (seg, info) in enumerate(sizes.items()):
        cols[i % 4].metric(seg.replace("_", " ").title(), f"{info.get('pct', 0)}%", f"{info.get('count',0)} reviews")

    # Radar comparing segments across discovery problems.
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None
    problems = matrix.get("problems", [])
    seg_matrix = matrix.get("matrix", {})
    if go and problems and seg_matrix:
        st.subheader("Segment comparison radar")
        chosen = st.multiselect(
            "Segments to compare", list(seg_matrix.keys()),
            default=list(seg_matrix.keys())[:3],
        )
        fig = go.Figure()
        for seg in chosen:
            vals = [seg_matrix[seg][p]["pct"] for p in problems]
            fig.add_trace(go.Scatterpolar(r=vals, theta=problems, fill="toself", name=seg))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    # Heatmap of the comparison matrix.
    try:
        import plotly.express as px
    except ImportError:
        px = None
    if px and problems and seg_matrix:
        st.subheader("Discovery problems × segments")
        segs = list(seg_matrix.keys())
        z = [[seg_matrix[s][p]["pct"] for p in problems] for s in segs]
        fig = px.imshow(
            z, x=problems, y=segs, color_continuous_scale="Greens",
            labels=dict(color="% of segment"),
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    st.subheader("Segment profiles")
    for seg, info in sizes.items():
        with st.expander(f"{seg.replace('_', ' ').title()} — {info.get('pct',0)}% of reviews"):
            prof = profiles.get(seg)
            if prof:
                st.write(f"**Satisfaction:** {prof.get('recommendation_satisfaction','—')}")
                if prof.get("good_discovery_definition"):
                    st.write(f"**'Good discovery' means:** {prof['good_discovery_definition']}")
                _bullets("Pain points", prof.get("discovery_pain_points"))
                _bullets("Primary frustrations", prof.get("primary_frustrations"))
                _bullets("Desired outcomes", prof.get("desired_outcomes"))
                _bullets("Workarounds", prof.get("workarounds"))
                _bullets("Features mentioned", prof.get("features_mentioned"))
                _journey(prof)
                for q in prof.get("sample_quotes", []):
                    quote_card(q.get("quote", ""), q.get("source", ""))
            else:
                st.caption("Enable full profiles above to load LLM analysis.")


def _bullets(label: str, items) -> None:
    if items:
        st.write(f"**{label}:**")
        for it in items:
            st.write(f"- {it}")


def _journey(profile: dict) -> None:
    """Render a simple discovery journey across stages for the segment."""
    st.write("**Discovery journey**")
    stages = ["Onboarding", "Exploration", "Recommendation", "Retention"]
    pains = profile.get("discovery_pain_points", []) or ["—"]
    outcomes = profile.get("desired_outcomes", []) or ["—"]
    cols = st.columns(len(stages))
    mapping = [
        ("Onboarding", outcomes[0] if outcomes else "—"),
        ("Exploration", pains[0] if pains else "—"),
        ("Recommendation", pains[min(1, len(pains) - 1)] if pains else "—"),
        ("Retention", outcomes[min(1, len(outcomes) - 1)] if outcomes else "—"),
    ]
    for col, (stage, note) in zip(cols, mapping):
        col.markdown(f"**{stage}**")
        col.caption(str(note)[:120])


def page_research_agent() -> None:
    st.caption(
        "Breaks each question into sub-questions, searches evidence, "
        "forms hypotheses, and synthesizes a report."
    )
    default_qs = "Why do users struggle to discover new music?\nWhat unmet needs appear consistently across reviews?"
    raw = st.text_area("Research questions (one per line)", value=default_qs, height=120)
    questions = [q.strip() for q in raw.splitlines() if q.strip()]

    if st.button("Run research session", type="primary") and questions:
        steps = st.empty()
        steps.info("Agent working: breaking down questions → searching → hypothesizing → synthesizing...")
        with st.spinner("Running autonomous research (this can take a while)..."):
            report = api_post("/agent/research", {"research_questions": questions})
        steps.empty()
        if report:
            st.session_state["research_report"] = report

    report = st.session_state.get("research_report")
    if report:
        st.markdown("### Executive summary")
        st.write(report.get("summary", ""))

        st.markdown("### Findings")
        for f in report.get("findings", []):
            with st.expander(f.get("question", "")):
                st.write(f.get("insight", ""))
                cols = st.columns(2)
                cols[0].metric("Confidence", f"{f.get('confidence', 0):.0%}")
                cols[1].metric("Sample size", f.get("sample_size", 0))
                if f.get("analysis"):
                    st.markdown("**Agent analysis**")
                    st.write(f["analysis"])
                for ev in f.get("evidence", []):
                    quote_card(ev.get("quote", ""), ev.get("source", ""), ev.get("sentiment", ""))

        if report.get("segments_affected"):
            st.markdown("### Segments affected")
            st.write(report["segments_affected"])

        if report.get("followup_questions"):
            st.markdown("### Suggested follow-ups")
            for q in report["followup_questions"]:
                st.write(f"- {q}")

        st.download_button(
            "Export findings (JSON)",
            data=_json_dumps(report),
            file_name="research_report.json",
            mime="application/json",
        )
        if st.button("Export Markdown report"):
            res = api_post("/export/report", {"format": "markdown", "title": "Spotify Discovery Research"})
            if res is not None:
                st.success("Report generated on the server (see data/reports/).")


def _json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


def page_raw_data() -> None:
    hero_banner("Raw Data", "Search and filter individual reviews from the indexed corpus.")
    q = st.text_input("Search query (leave blank to browse)", "", placeholder="e.g. discover weekly repetitive")
    c1, c2, c3, c4 = st.columns(4)
    source = c1.selectbox(
        "Source",
        ["all", "app_store", "play_store", "reddit", "twitter"],
        format_func=lambda x: "All sources" if x == "all" else source_label(x),
    )
    sentiment = c2.selectbox("Sentiment", ["all", "positive", "negative", "neutral", "mixed"])
    rating = c3.selectbox("Rating", ["all", 1, 2, 3, 4, 5])
    limit = c4.slider("Max results", 10, 200, 50)

    if st.button("Search", type="primary"):
        params = {"limit": limit}
        if q:
            params["q"] = q
        if source != "all":
            params["source"] = source
        if sentiment != "all":
            params["sentiment"] = sentiment
        if rating != "all":
            params["rating"] = rating
        data = api_get("/data/search", params)
        results = (data or {}).get("results", [])
        st.write(f"**{len(results)} results**")

        rows = []
        for r in results:
            meta = r.get("metadata", {})
            rows.append(
                {
                    "source": meta.get("source", ""),
                    "rating": meta.get("rating", ""),
                    "sentiment": meta.get("sentiment", ""),
                    "date": meta.get("date", ""),
                    "review": r.get("content", "")[:160],
                }
            )
        if rows:
            try:
                import pandas as pd

                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)
            except ImportError:
                st.table(rows)
            st.markdown("#### Detail")
            for r in results[:20]:
                meta = r.get("metadata", {})
                quote_card(r["content"][:300], meta.get("source", ""), meta.get("sentiment", ""))


def page_live_reviews() -> None:
    hero_banner(
        "Live Reviews",
        "Fetch Spotify feedback from app stores, community forums, and social media — "
        "then analyze with VADER + your LLM.",
        ["Choose sources", "Fetch & analyze", "Explore insights"],
    )

    try:
        from processors.llm_client import llm_configured
        import os

        bootstrap_env()
        provider = os.getenv("LLM_PROVIDER", "openai")
        if llm_configured():
            st.success(f"LLM ready ({provider}) — deep theme + sentiment analysis enabled.")
        else:
            st.warning(
                "No LLM API key found. Reviews will use VADER sentiment + keyword themes only. "
                "Set GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in `.env` or Streamlit Secrets."
            )
    except ImportError:
        st.warning("LLM client unavailable — keyword analysis only.")

    st.markdown("##### App stores")
    store_cols = st.columns(2)
    store_selected: list[str] = []
    with store_cols[0]:
        if st.checkbox(
            FETCH_SOURCES["play_store"]["label"],
            value=True,
            key="src_play_store",
        ):
            store_selected.append("play_store")
    with store_cols[1]:
        if st.checkbox(
            FETCH_SOURCES["app_store"]["label"],
            value=False,
            key="src_app_store",
        ):
            store_selected.append("app_store")

    st.markdown("##### Community & social")
    social_cols = st.columns(2)
    with social_cols[0]:
        if st.checkbox(
            FETCH_SOURCES["community_forums"]["label"],
            value=False,
            key="src_community_forums",
        ):
            store_selected.append("community_forums")
    with social_cols[1]:
        if st.checkbox(
            FETCH_SOURCES["social_media"]["label"],
            value=False,
            key="src_social_media",
        ):
            store_selected.append("social_media")

    sources = normalize_sources(store_selected)
    with st.expander("About review sources", expanded=False):
        st.markdown(fetch_source_help())
    limit = st.slider("Reviews per source", min_value=10, max_value=100, value=30, step=10)
    use_llm = st.checkbox("Use LLM for themes & sentiment", value=True)
    discovery_filter = st.checkbox(
        "App Store: discovery-keyword filter only",
        value=False,
        help="When enabled, App Store reviews must mention discovery-related terms.",
    )

    if st.button("Fetch & analyze live reviews", type="primary", disabled=not sources):
        with st.spinner("Scraping and analyzing reviews — this may take a minute…"):
            result = api_post(
                "/data/fetch-live",
                {
                    "sources": sources,
                    "limit": limit,
                    "use_llm": use_llm,
                    "discovery_filter": discovery_filter,
                },
                timeout=600,
            )
        if not result:
            return

        if result.get("error"):
            st.error(result["error"])

        counts = result.get("source_counts") or {}
        if counts:
            st.markdown("#### Fetched per source")
            for src, n in counts.items():
                st.write(f"- **{source_label(src)}**: {n} reviews")

        st.success(
            f"Done — scraped {result.get('scraped', 0)}, "
            f"processed {result.get('processed', 0)}, "
            f"indexed into store."
        )
        if result.get("llm_used"):
            st.info("LLM analysis was applied to this batch.")
        for warning in result.get("warnings") or []:
            st.warning(warning)

        themes = result.get("theme_counts") or {}
        if themes:
            st.markdown("#### Top themes in this batch")
            tcols = st.columns(min(5, len(themes)))
            for i, (theme, count) in enumerate(list(themes.items())[:10]):
                tcols[i % len(tcols)].metric(theme.replace("_", " ").title(), count)

        st.balloons()
        st.markdown("**Next steps**")
        n1, n2 = st.columns(2)
        if n1.button("Go to Discovery Questions", type="primary", use_container_width=True):
            _goto_page("Discovery Questions")
        if n2.button("View Corpus Overview", use_container_width=True):
            _goto_page("Corpus Overview")


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
_DEFAULT_PAGE = "Live Reviews"

PAGES = {
    "Live Reviews": page_live_reviews,
    "Discovery Questions": page_discovery_questions,
    "Corpus Overview": page_overview,
    "Evidence Explorer": page_evidence_explorer,
    "Segment Deep Dive": page_segments,
    "Raw Data": page_raw_data,
}


def _goto_page(page: str, *, question_id: str | None = None) -> None:
    """Queue a page change — safe to call after widgets (applied on next rerun)."""
    st.session_state["_pending_nav"] = page
    if question_id:
        st.session_state["_pending_dq"] = question_id
    st.rerun()


def _apply_pending_navigation() -> None:
    """Apply queued navigation before rendering the sidebar radio."""
    pending = st.session_state.pop("_pending_nav", None)
    if pending in PAGES:
        st.session_state["nav_choice"] = pending
    dq = st.session_state.pop("_pending_dq", None)
    if dq:
        st.session_state["dq_selected"] = dq
        ids = [q["id"] for q in DISCOVERY_QUESTIONS]
        if dq in ids:
            labels = [f"{q['icon']} {q['short_title']}" for q in DISCOVERY_QUESTIONS]
            st.session_state["dq_radio"] = labels[ids.index(dq)]


def main() -> None:
    inject_css()
    st.sidebar.markdown(
        f"<div style='padding:4px 0 12px 0'>"
        f"<span style='font-size:1.6rem'>🎧</span> "
        f"<span style='color:{ACCENT};font-weight:800;font-size:1.15rem'>Discovery Analyzer</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    page_names = list(PAGES.keys())
    if "nav_choice" not in st.session_state:
        st.session_state["nav_choice"] = _DEFAULT_PAGE

    _apply_pending_navigation()

    choice = st.sidebar.radio("Navigate", page_names, key="nav_choice")

    st.sidebar.markdown("---")

    try:
        from processors.llm_client import auto_llm_provider, llm_configured
        import os

        bootstrap_env()
        provider = auto_llm_provider() or os.getenv("LLM_PROVIDER", "—")
        if llm_configured(provider if provider != "—" else None):
            st.sidebar.success(f"LLM: {provider}")
        else:
            st.sidebar.warning(
                "LLM key missing — insights use extractive summaries only. "
                "Set GROQ_API_KEY + LLM_PROVIDER=groq in Secrets."
            )
    except Exception:
        pass

    if _use_local():
        st.sidebar.info("Running in standalone mode (in-process engine)")
    elif _check_health():
        st.sidebar.success("API connected")
    else:
        st.sidebar.error("API unreachable — start: uvicorn api.main:app --reload")
    st.sidebar.caption(f"API: {API_BASE}" if not _use_local() else "Backend: in-process")

    PAGES[choice]()


if __name__ == "__main__":
    main()
