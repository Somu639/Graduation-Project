"""Spotify Discovery Analyzer - interactive Streamlit dashboard.

Six pages (Overview, Theme Explorer, Insight Q&A, Segment Analysis, Research
Agent, Raw Data) backed by the FastAPI service, with Spotify-inspired dark
styling.

Run with the API up:
    uvicorn api.main:app --reload
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

import requests
import streamlit as st

# Make the project root importable so the in-process backend can load
# rag/agents/api modules when running standalone (e.g. on Streamlit Cloud).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Local (in-process) mode: used automatically when no FastAPI server is reachable.
# Force it with LOCAL_MODE=1, or force HTTP with USE_API=1.
import local_service as svc  # noqa: E402  (after sys.path setup)

# Spotify palette.
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#121212"
SPOTIFY_GREY = "#181818"
SPOTIFY_LIGHT = "#B3B3B3"
GREEN_SCALE = ["#1DB954", "#1ED760", "#168d40", "#0e6b30", "#53e07f", "#0a4f24"]

st.set_page_config(page_title="Spotify Discovery Analyzer", page_icon="🎧", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {SPOTIFY_BLACK}; color: #FFFFFF; }}
        section[data-testid="stSidebar"] {{ background-color: #000000; }}
        h1, h2, h3, h4 {{ color: #FFFFFF; font-weight: 700; }}
        .stMarkdown, .stText, label, p {{ color: #E8E8E8; }}
        div[data-testid="stMetric"] {{
            background-color: {SPOTIFY_GREY};
            border: 1px solid #282828;
            border-radius: 12px;
            padding: 16px;
        }}
        div[data-testid="stMetricValue"] {{ color: {SPOTIFY_GREEN}; }}
        .stButton>button {{
            background-color: {SPOTIFY_GREEN};
            color: #000000;
            border: none;
            border-radius: 500px;
            font-weight: 700;
            padding: 0.5rem 1.4rem;
        }}
        .stButton>button:hover {{ background-color: #1ED760; color: #000000; }}
        .quote-card {{
            background-color: {SPOTIFY_GREY};
            border-left: 4px solid {SPOTIFY_GREEN};
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }}
        .quote-meta {{ color: {SPOTIFY_LIGHT}; font-size: 0.8rem; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [aria-selected="true"] {{ color: {SPOTIFY_GREEN}; }}
        </style>
        """,
        unsafe_allow_html=True,
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
        return svc.insights_question(p.get("question", ""), p.get("segment"), p.get("source"))
    if path == "/data/search":
        return svc.data_search(p.get("q"), p.get("source"), p.get("sentiment"),
                               p.get("rating"), p.get("theme"), int(p.get("limit", 50)))
    raise ValueError(f"No local route for GET {path}")


def _local_post(path: str, body: dict):
    if path == "/data/seed":
        return svc.seed()
    if path == "/agent/research":
        return svc.agent_research(body.get("research_questions", []))
    if path == "/export/report":
        return svc.export_report(body.get("format", "markdown"), body.get("title", "Report"))
    raise ValueError(f"No local route for POST {path}")


def api_get(path: str, params: dict | None = None, timeout: int = 120):
    if _use_local():
        try:
            return _local_get(path, params or {})
        except Exception as exc:  # noqa: BLE001
            st.error(f"{path} failed: {exc}")
            return None
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
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
    """Apply the dark Spotify theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SPOTIFY_BLACK,
        plot_bgcolor=SPOTIFY_BLACK,
        font_color="#FFFFFF",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def quote_card(quote: str, source: str = "", sentiment: str = "") -> None:
    meta = " · ".join(p for p in [source, sentiment] if p and p != "n/a")
    st.markdown(
        f'<div class="quote-card">“{quote}”<br>'
        f'<span class="quote-meta">{meta}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.header("Overview")
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
                color_discrete_sequence=[SPOTIFY_GREEN],
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

    st.subheader("Reviews by source")
    if sources:
        fig = px.bar(
            x=list(sources.values()), y=list(sources.keys()), orientation="h",
            labels={"x": "Reviews", "y": "Source"},
            color_discrete_sequence=[SPOTIFY_GREEN],
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    st.subheader("Review volume over time")
    timeline = api_get("/stats/timeline")
    series = (timeline or {}).get("series", [])
    if series:
        fig = px.area(
            x=[s["period"] for s in series], y=[s["count"] for s in series],
            labels={"x": "Month", "y": "Reviews"},
            color_discrete_sequence=[SPOTIFY_GREEN],
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)
    else:
        st.caption("No dated reviews available for a timeline yet.")

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
        width=1000, height=320, background_color=SPOTIFY_BLACK, colormap="Greens",
    ).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(SPOTIFY_BLACK)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def page_theme_explorer() -> None:
    st.header("Theme Explorer")
    f1, f2, f3 = st.columns(3)
    source = f1.selectbox("Source", ["all", "app_store", "play_store", "reddit", "twitter"])
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


def page_insight_qa() -> None:
    st.header("Insight Q&A")
    st.caption("Ask anything about Spotify music discovery; answers are grounded in user reviews.")

    prebuilt = [
        "Why do users struggle to discover new music?",
        "What are the most common recommendation frustrations?",
        "Why do users repeatedly listen to the same content?",
        "What unmet needs appear consistently across reviews?",
    ]
    st.write("**Quick questions**")
    cols = st.columns(2)
    for i, q in enumerate(prebuilt):
        if cols[i % 2].button(q, key=f"pb_{i}"):
            st.session_state["qa_question"] = q

    question = st.text_input(
        "Your question",
        value=st.session_state.get("qa_question", ""),
        placeholder="e.g. What do power users dislike about Discover Weekly?",
    )
    c1, c2 = st.columns(2)
    source = c1.selectbox("Source filter", ["all", "app_store", "play_store", "reddit"])
    segment = c2.text_input("Segment hint (optional)", "")

    if st.button("Get insight", type="primary") and question:
        params = {"question": question}
        if source != "all":
            params["source"] = source
        if segment:
            params["segment"] = segment
        with st.spinner("Synthesizing insight..."):
            result = api_get("/insights/question", params)
        if result:
            st.markdown("### Insight")
            st.write(result.get("insight", ""))
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{result.get('confidence', 0):.0%}")
            m2.metric("Sample size", result.get("sample_size", 0))
            if result.get("themes_identified"):
                st.write("**Themes:** " + ", ".join(result["themes_identified"]))
            st.markdown("#### Supporting evidence")
            for ev in result.get("supporting_evidence", []):
                quote_card(ev.get("quote", ""), ev.get("source", ""), ev.get("sentiment", ""))
            if result.get("recommended_followup_questions"):
                st.markdown("#### Follow-up questions")
                for fq in result["recommended_followup_questions"]:
                    st.write(f"- {fq}")


def page_segments() -> None:
    st.header("Segment Analysis")
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
    st.header("Research Agent")
    st.caption(
        "The agent breaks each question into sub-questions, searches evidence, "
        "forms hypotheses, and synthesizes a report. (Results stream when complete; "
        "the API returns the final report.)"
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
    st.header("Raw Data")
    st.caption("Search semantically or browse with filters.")
    q = st.text_input("Search query (leave blank to browse)", "")
    c1, c2, c3, c4 = st.columns(4)
    source = c1.selectbox("Source", ["all", "app_store", "play_store", "reddit", "twitter"])
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


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
PAGES = {
    "Overview": page_overview,
    "Theme Explorer": page_theme_explorer,
    "Insight Q&A": page_insight_qa,
    "Segment Analysis": page_segments,
    "Research Agent": page_research_agent,
    "Raw Data": page_raw_data,
}


def main() -> None:
    inject_css()
    st.sidebar.markdown(f"<h1 style='color:{SPOTIFY_GREEN}'>🎧 Discovery Analyzer</h1>", unsafe_allow_html=True)
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.markdown("---")

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
