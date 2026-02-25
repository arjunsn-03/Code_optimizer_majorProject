"""
app.py — Streamlit UI for the Carbon-Aware Code Optimizer.

Tabs
----
  🏠 Dashboard   — system status, Ollama health, last-run summary
  ⚡ Optimize    — full pipeline with live step-by-step logs
  🔍 Analyze     — static analysis only, hotspot table + chart
  📊 Results     — browse saved results, patches, energy savings
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── make sure the project root is importable ──────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    GRID_INTENSITY_KG_PER_KWH,
    OLLAMA_HOST,
    RESULTS_PATH,
)
from estimator.energy import estimate_annual_savings, kwh_to_co2

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon-Aware Code Optimizer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2rem; font-weight:700; color:#2ecc71; }
    .sub-title    { font-size:1rem; color:#7f8c8d; margin-top:-12px; }
    .log-info     { color:#bdc3c7; font-family:monospace; font-size:.82rem; }
    .log-stage    { color:#3498db; font-weight:700; font-family:monospace; font-size:.85rem; }
    .log-success  { color:#2ecc71; font-family:monospace; font-size:.82rem; }
    .log-warning  { color:#f39c12; font-family:monospace; font-size:.82rem; }
    .log-error    { color:#e74c3c; font-family:monospace; font-size:.82rem; }
    .metric-card  { background:#1a1a2e; border-radius:10px; padding:14px 18px; }
    div[data-testid="stMetricValue"] { font-size:1.6rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ollama_status() -> Tuple[bool, List[str]]:
    """Return (is_reachable, model_names)."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=4)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except Exception:
        return False, []


def load_results() -> List[Dict[str, Any]]:
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            return []
    return []


def level_icon(level: str) -> str:
    return {
        "stage":   "🔵",
        "success": "✅",
        "warning": "⚠️",
        "error":   "❌",
        "info":    "📋",
    }.get(level, "📋")


def render_log_line(level: str, msg: str) -> str:
    cls = f"log-{level}"
    icon = level_icon(level)
    return f'<span class="{cls}">{icon} {msg}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="main-title">🌿 CodeCarbon</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">LLM Code Optimizer</div>', unsafe_allow_html=True)
    st.divider()

    repo_path = st.text_input(
        "📁 Project folder path",
        value=str(ROOT),
        help="Absolute path to the project you want to optimise.",
    )

    model = st.selectbox(
        "🤖 Model",
        options=["qwen3:8b", "llama3", "mistral"],
        index=0,
        help="Local Ollama model to use for optimisation.",
    )

    top_n = st.slider(
        "🎯 Top-N hotspots",
        min_value=1, max_value=20, value=5,
        help="Number of hotspot functions to analyse.",
    )

    gpu_mode = st.toggle(
        "⚡ GPU / ML mode",
        value=False,
        help="Enable if the project uses PyTorch/TF — activates GPU-aware prompts.",
    )

    st.divider()
    st.caption("**Grid intensity**")
    grid_intensity = st.number_input(
        "kg CO₂e / kWh",
        value=GRID_INTENSITY_KG_PER_KWH,
        min_value=0.01, max_value=2.0, step=0.01,
        help="Override for your local grid. Default = world average.",
    )

    runs_per_day = st.number_input(
        "Estimated runs / day",
        value=1000, min_value=1,
        help="Used to project annual CO₂ savings.",
    )

    st.divider()
    st.caption("v1.0 · B.Tech Final Year Project")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_dash, tab_opt, tab_analyze, tab_results = st.tabs([
    "🏠 Dashboard",
    "⚡ Optimize",
    "🔍 Analyze",
    "📊 Results",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    st.subheader("System Status")

    reachable, avail_models = ollama_status()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Ollama",
        "🟢 Online" if reachable else "🔴 Offline",
        help="Ollama must be running: `ollama serve`",
    )
    col2.metric("Models available", len(avail_models))
    col3.metric("Selected model", model)
    results_cached = load_results()
    col4.metric("Saved results", len(results_cached))

    if not reachable:
        st.error(
            "**Ollama is not reachable.** "
            "Open a terminal and run:  `ollama serve`"
        )

    st.divider()

    # Available models table
    if avail_models:
        st.markdown("#### 🤖 Available local models")
        model_df = pd.DataFrame(avail_models, columns=["Model name"])
        st.dataframe(model_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No models found. Pull one with `ollama pull qwen3:8b`")

    st.divider()

    # Project folder stats
    st.markdown("#### 📂 Target project overview")
    p = Path(repo_path)
    if p.exists():
        from ingestor.clone import walk_repo
        try:
            with st.spinner("Scanning folder …"):
                files = walk_repo(repo_path)
            lang_count: Dict[str, int] = {}
            total_lines = 0
            for f in files:
                lang_count[f["language"]] = lang_count.get(f["language"], 0) + 1
                total_lines += f["lines"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Source files", len(files))
            c2.metric("Total lines", f"{total_lines:,}")
            c3.metric("Languages", len(lang_count))

            if lang_count:
                fig = px.pie(
                    names=list(lang_count.keys()),
                    values=list(lang_count.values()),
                    title="Language distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=280)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not scan folder: {exc}")
    else:
        st.warning(f"Path does not exist: `{repo_path}`")

    # Last run summary
    if results_cached:
        st.divider()
        st.markdown("#### 📈 Last run quick summary")
        accepted = sum(
            1 for r in results_cached
            if r.get("verification", {}).get("verdict") == "accepted"
        )
        total_kwh = sum(
            abs(min(r.get("estimated_kwh_delta", 0.0) or 0.0, 0.0))
            for r in results_cached
            if r.get("verification", {}).get("verdict") == "accepted"
        )
        co2_g = kwh_to_co2(total_kwh, grid_intensity) * 1000

        ca, cb, cc = st.columns(3)
        ca.metric("Patches accepted", accepted)
        cb.metric("kWh saved / run", f"{total_kwh:.8f}")
        cc.metric("CO₂ saved / run", f"{co2_g:.4f} g")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Optimize (live pipeline)
# ══════════════════════════════════════════════════════════════════════════════

with tab_opt:
    st.subheader("⚡ Full Optimization Pipeline")
    st.caption(
        "Runs all 5 stages: Ingest → Analyse → LLM Optimize → Verify → Report. "
        "Logs stream in real-time below."
    )

    if not Path(repo_path).exists():
        st.error(f"Folder not found: `{repo_path}` — update it in the sidebar.")
        st.stop()

    run_btn = st.button(
        "🚀  Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not reachable,
    )

    if not reachable:
        st.warning("Start Ollama first (`ollama serve`) then refresh.")

    # ── Progress + stage indicators ───────────────────────────────────────────

    STAGES = [
        "Stage 1 — Ingest & Index",
        "Stage 2 — Static Analysis",
        "Stage 3 — LLM Optimise",
        "Stage 4 — Verify",
        "Stage 5 — Report",
    ]

    stage_cols = st.columns(len(STAGES))
    stage_placeholders = [c.empty() for c in stage_cols]

    def render_stages(active: int = -1, done_up_to: int = -1) -> None:
        for i, (col, ph) in enumerate(zip(stage_cols, stage_placeholders)):
            if i < done_up_to:
                ph.success(f"✅ {STAGES[i].split('—')[0].strip()}")
            elif i == active:
                ph.info(f"⏳ {STAGES[i].split('—')[0].strip()}")
            else:
                ph.empty()
                col.caption(f"◦ {STAGES[i].split('—')[0].strip()}")

    render_stages()

    progress_bar   = st.progress(0, text="Waiting …")
    log_area       = st.empty()      # live log container
    results_region = st.container()  # filled after pipeline completes

    # ── Pipeline runner ───────────────────────────────────────────────────────

    if run_btn:
        LOG_Q: queue.Queue = queue.Queue()
        log_lines: List[Tuple[str, str]] = []
        pipeline_error: List[str] = []

        def _run() -> None:
            try:
                from pipeline.orchestrator import Orchestrator
                orch = Orchestrator(
                    repo_path=repo_path,
                    model=model,
                    top_n=top_n,
                    gpu=gpu_mode,
                    log_callback=lambda level, msg: LOG_Q.put((level, msg)),
                )
                orch.run()
                LOG_Q.put(("__done__", json.dumps(orch.results, default=str)))
            except Exception as exc:
                LOG_Q.put(("error", f"Pipeline crashed: {exc}"))
                LOG_Q.put(("__done__", "[]"))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # Stage tracking heuristic: advance stage when keyword appears in msg
        stage_keywords = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
        current_stage  = -1
        done_stages    = 0
        final_results  = []

        while thread.is_alive() or not LOG_Q.empty():
            try:
                level, msg = LOG_Q.get(timeout=0.15)
            except queue.Empty:
                continue

            if level == "__done__":
                try:
                    final_results = json.loads(msg)
                except Exception:
                    final_results = []
                break

            log_lines.append((level, msg))

            # Advance stage indicator
            for si, kw in enumerate(stage_keywords):
                if kw.lower() in msg.lower() and si > current_stage:
                    if current_stage >= 0:
                        done_stages = current_stage + 1
                    current_stage = si
                    break

            render_stages(active=current_stage, done_up_to=done_stages)
            pct = int(min((done_stages / len(STAGES)) * 100, 95))
            progress_bar.progress(pct, text=f"Running {STAGES[current_stage] if current_stage >= 0 else '…'}")

            # Render last 60 log lines
            html_lines = "<br>".join(
                render_log_line(l, m) for l, m in log_lines[-60:]
            )
            log_area.markdown(
                f'<div style="background:#0e1117;padding:12px;border-radius:8px;'
                f'max-height:420px;overflow-y:auto;font-size:.82rem;line-height:1.6;">'
                f'{html_lines}</div>',
                unsafe_allow_html=True,
            )

        thread.join(timeout=5)
        render_stages(active=-1, done_up_to=len(STAGES))
        progress_bar.progress(100, text="✅ Pipeline complete!")

        # ── Results table ─────────────────────────────────────────────────────
        with results_region:
            st.divider()
            st.markdown("### 📋 Optimisation Results")

            if not final_results:
                all_res = load_results()
            else:
                all_res = final_results

            if not all_res:
                st.info("No results yet — pipeline may not have generated any patches.")
            else:
                rows = []
                total_kwh_saved = 0.0
                for r in all_res:
                    verdict = r.get("verification", {}).get("verdict", "unverified")
                    delta   = r.get("estimated_kwh_delta", 0.0) or 0.0
                    if verdict == "accepted":
                        total_kwh_saved += abs(min(delta, 0.0))
                    rows.append({
                        "Function":    r.get("function_name", "?"),
                        "File":        r.get("relative_path", "?"),
                        "Opt. Type":   r.get("optimization_type", "?"),
                        "Δ kWh":       delta,
                        "Confidence":  r.get("confidence", 0.0),
                        "Risk":        r.get("risk_score", 0.0),
                        "Verdict":     verdict,
                    })

                df = pd.DataFrame(rows)

                def _colour_verdict(val: str) -> str:
                    if val == "accepted":  return "background-color:#1e4d2b; color:#2ecc71"
                    if val == "skipped":   return "background-color:#4d3c00; color:#f39c12"
                    return "background-color:#4d1010; color:#e74c3c"

                st.dataframe(
                    df.style.applymap(_colour_verdict, subset=["Verdict"]),
                    use_container_width=True, hide_index=True,
                )

                # ── Energy metrics ────────────────────────────────────────────
                st.divider()
                em1, em2, em3, em4 = st.columns(4)
                co2_per_run = kwh_to_co2(total_kwh_saved, grid_intensity) * 1000
                savings = estimate_annual_savings(total_kwh_saved, int(runs_per_day))
                em1.metric("kWh saved / run",   f"{total_kwh_saved:.8f}")
                em2.metric("g CO₂e saved / run", f"{co2_per_run:.4f}")
                em3.metric("kWh saved / year",  f"{savings['annual_kwh_saved']:.4f}")
                em4.metric("CO₂ saved / year",  f"{savings['annual_co2_g_saved']:.2f} g")

                # ── Bar chart ─────────────────────────────────────────────────
                if rows:
                    fig = px.bar(
                        df, x="Function", y="Δ kWh", color="Verdict",
                        color_discrete_map={
                            "accepted": "#2ecc71",
                            "skipped":  "#f39c12",
                            "rejected": "#e74c3c",
                        },
                        title="Estimated kWh delta per function",
                    )
                    fig.update_layout(
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font_color="#ecf0f1", height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analyze (static only)
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyze:
    st.subheader("🔍 Static Analysis — Hotspot Scanner")
    st.caption("Runs instantly with no LLM or internet required.")

    analyze_btn = st.button(
        "🔎  Scan Project",
        type="secondary",
        use_container_width=True,
    )

    if analyze_btn:
        if not Path(repo_path).exists():
            st.error(f"Folder not found: `{repo_path}`")
        else:
            from ingestor.clone import walk_repo
            from static_analyzer.analyzer import analyze_file

            with st.spinner("Scanning files and building hotspot scores …"):
                files   = walk_repo(repo_path)
                all_funcs: List[Dict] = []

                for f in files:
                    result = analyze_file(f)
                    for func in result.get("functions", []):
                        all_funcs.append({
                            "Function":      func["name"],
                            "File":          result["relative_path"],
                            "LOC":           func.get("loc", 0),
                            "Hotspot Score": func.get("hotspot_score", 0.0),
                            "Line":          func.get("lineno", 0),
                            "Is Async":      func.get("is_async", False),
                        })

            if not all_funcs:
                st.warning("No Python functions found in the target project.")
            else:
                df = pd.DataFrame(all_funcs).sort_values(
                    "Hotspot Score", ascending=False
                ).reset_index(drop=True)

                st.success(f"Found **{len(df)}** functions across **{len(files)}** files.")

                a1, a2, a3 = st.columns(3)
                a1.metric("Total functions", len(df))
                a2.metric("High-risk (score > 0.5)", int((df["Hotspot Score"] > 0.5).sum()))
                a3.metric("Avg hotspot score", f"{df['Hotspot Score'].mean():.3f}")

                # Colour-coded table
                def _heat(val: float) -> str:
                    if val >= 0.6: return "background-color:#4d1010; color:#e74c3c"
                    if val >= 0.35: return "background-color:#4d3c00; color:#f39c12"
                    return ""

                st.dataframe(
                    df.style.applymap(_heat, subset=["Hotspot Score"]),
                    use_container_width=True, hide_index=True,
                )

                # Top-15 bar chart
                top = df.head(15)
                fig = px.bar(
                    top, x="Hotspot Score", y="Function",
                    orientation="h",
                    color="Hotspot Score",
                    color_continuous_scale="RdYlGn_r",
                    title="Top 15 Hotspot Functions",
                    text="File",
                )
                fig.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font_color="#ecf0f1", height=440,
                    yaxis=dict(autorange="reversed"),
                )
                fig.update_traces(textposition="inside", textfont_size=10)
                st.plotly_chart(fig, use_container_width=True)

                # Anti-pattern breakdown
                st.divider()
                st.markdown("#### Anti-pattern summary")
                all_issues: List[Dict] = []
                for f in files:
                    from static_analyzer.analyzer import detect_antipatterns
                    for issue in detect_antipatterns(f["content"], f["path"]):
                        all_issues.append({
                            "Type":     issue["type"],
                            "Severity": issue["severity"],
                            "File":     f["relative_path"],
                            "Line":     issue.get("lineno", 0),
                            "Message":  issue.get("message", ""),
                        })

                if all_issues:
                    idf = pd.DataFrame(all_issues)
                    severity_counts = idf["Type"].value_counts().reset_index()
                    severity_counts.columns = ["Type", "Count"]

                    fig2 = px.pie(
                        severity_counts, names="Type", values="Count",
                        title="Anti-pattern types",
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig2.update_layout(
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font_color="#ecf0f1", height=300,
                    )
                    c1, c2 = st.columns([1, 2])
                    c1.plotly_chart(fig2, use_container_width=True)
                    c2.dataframe(idf, use_container_width=True, hide_index=True)
                else:
                    st.success("No anti-patterns detected 🎉")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Results browser
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.subheader("📊 Results Browser")
    st.caption(f"Loaded from `{RESULTS_PATH}`")

    if st.button("🔄  Reload", key="reload_results"):
        st.rerun()

    results = load_results()
    if not results:
        st.info("No results saved yet — run the **Optimize** pipeline first.")
    else:
        # Summary metrics
        accepted_res = [
            r for r in results
            if r.get("verification", {}).get("verdict") == "accepted"
        ]
        total_kwh = sum(
            abs(min(r.get("estimated_kwh_delta", 0.0) or 0.0, 0.0))
            for r in accepted_res
        )
        savings = estimate_annual_savings(total_kwh, int(runs_per_day))
        co2_run = kwh_to_co2(total_kwh, grid_intensity) * 1000

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total suggestions", len(results))
        m2.metric("Accepted",  len(accepted_res))
        m3.metric("kWh / run", f"{total_kwh:.8f}")
        m4.metric("g CO₂ / run", f"{co2_run:.4f}")
        m5.metric("Equiv. km / yr", f"{savings['equivalent_km_driven']:.1f} km")

        # Annual savings gauge
        st.divider()
        if total_kwh > 0:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=savings["annual_co2_g_saved"],
                delta={"reference": 0},
                title={"text": "g CO₂e saved / year"},
                gauge={
                    "axis": {"range": [0, max(savings["annual_co2_g_saved"] * 2, 1)]},
                    "bar":  {"color": "#2ecc71"},
                    "steps": [
                        {"range": [0, savings["annual_co2_g_saved"] * 0.5], "color": "#1a2e1a"},
                        {"range": [savings["annual_co2_g_saved"] * 0.5,
                                   savings["annual_co2_g_saved"]],      "color": "#1e4d2b"},
                    ],
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0e1117", font_color="#ecf0f1", height=260,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # Per-suggestion expanders
        st.markdown("#### Suggestion details")
        for i, r in enumerate(results, 1):
            verdict   = r.get("verification", {}).get("verdict", "unverified")
            icon      = "✅" if verdict == "accepted" else ("⚠️" if verdict == "skipped" else "❌")
            fname     = r.get("function_name", "unknown")
            ffile     = r.get("relative_path", "")
            opt_type  = r.get("optimization_type", "none")
            delta_kwh = r.get("estimated_kwh_delta", 0.0) or 0.0

            with st.expander(
                f"{icon}  [{i}]  `{fname}`  —  {ffile}  "
                f"({opt_type})  Δ{delta_kwh:+.7f} kWh"
            ):
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Verdict",    verdict)
                tc2.metric("Confidence", f"{r.get('confidence', 0.0):.0%}")
                tc3.metric("Risk score", f"{r.get('risk_score', 0.0):.2f}")
                tc4.metric("Δ kWh",      f"{delta_kwh:+.7f}")

                explanation = r.get("explanation") or r.get("rationale", "")
                if explanation:
                    st.markdown(f"**Explanation:** {explanation}")

                patch = r.get("patch", "")
                if patch and patch.strip():
                    st.markdown("**Patch (unified diff):**")
                    st.code(patch, language="diff")
                else:
                    st.caption("No patch generated for this suggestion.")

                ver = r.get("verification", {})
                if ver.get("stdout"):
                    with st.expander("Test output"):
                        st.code(ver["stdout"], language="text")
