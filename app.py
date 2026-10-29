"""Fraud triage analyst console. Run: streamlit run app.py"""
import datetime
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ART = "artifacts"
PAGES = ["Summary", "Alerts", "Look at one case", "How this was built"]

st.set_page_config(
    page_title="Fraud Triage",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep the friendly error if artifacts/ is missing
if not os.path.exists(f"{ART}/metrics.json"):
    st.error("No artifacts found. Run:  python run_pipeline.py --data data/raw/<paysim>.csv")
    st.stop()


# Cached loaders
@st.cache_data
def load_metrics():
    with open(f"{ART}/metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_thresholds():
    with open(f"{ART}/thresholds.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_cases():
    return pd.read_csv(f"{ART}/cases.csv")


@st.cache_data
def load_global_shap():
    if os.path.exists(f"{ART}/global_shap.csv"):
        return pd.read_csv(f"{ART}/global_shap.csv", index_col=0)
    return None


def get_artifacts_timestamp():
    target = f"{ART}/metrics.json"
    if os.path.exists(target):
        mtime = os.path.getmtime(target)
        return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    return "Unknown"


# Chart helpers
def plot_score_bar(score: float, flag_thr: float, review_thr: float):
    fig, ax = plt.subplots(figsize=(6, 1.15), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Track
    ax.barh(0, 1.0, color="#f1f3f5", height=0.32, align="center", edgecolor="#d0d5dd", linewidth=0.8)

    # Filled score bar
    score_clamped = min(max(float(score), 0.0), 1.0)
    bar_color = "#b3261e" if score >= flag_thr else ("#b45309" if score >= review_thr else "#24587c")
    ax.barh(0, score_clamped, color=bar_color, height=0.32, align="center")

    # Cut-off markers
    if abs(flag_thr - review_thr) < 1e-4:
        ax.axvline(flag_thr, color="#202124", linestyle="--", linewidth=1.1)
        ax.text(flag_thr, 0.28, f"Threshold ({flag_thr:.3f})", color="#202124", fontsize=8, ha="center", va="bottom")
    else:
        ax.axvline(review_thr, color="#b45309", linestyle="--", linewidth=1.1)
        ax.text(review_thr, 0.28, f"Review ({review_thr:.3f})", color="#b45309", fontsize=8, ha="center", va="bottom")
        ax.axvline(flag_thr, color="#b3261e", linestyle="--", linewidth=1.1)
        ax.text(flag_thr, 0.28, f"Flag ({flag_thr:.3f})", color="#b3261e", fontsize=8, ha="center", va="bottom")

    # Score annotation
    score_x = min(max(score_clamped, 0.05), 0.95)
    ax.text(score_x, -0.28, f"Score: {score:.4f}", color="#202124", fontsize=8.5, ha="center", va="top", fontweight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=8, colors="#5f6368")
    ax.set_yticks([])
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d0d5dd")
    fig.tight_layout(pad=0.2)
    return fig


def plot_case_shap(shap_series: pd.Series):
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = ["#b3261e" if v >= 0 else "#24587c" for v in shap_series.values]
    ax.barh(shap_series.index, shap_series.values, color=colors, height=0.62)
    ax.axvline(0, color="#5f6368", linewidth=0.8, linestyle="--")

    ax.set_xlabel("SHAP contribution (log-odds impact)", fontsize=8.5, color="#202124")
    ax.tick_params(axis="both", labelsize=8, colors="#202124")
    ax.grid(axis="x", linestyle=":", color="#e2e4e8", alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#d0d5dd")
    fig.tight_layout(pad=0.3)
    return fig


def plot_global_shap(global_shap_df: pd.DataFrame):
    s = global_shap_df["mean_abs_shap"].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4.4), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.barh(s.index, s.values, color="#24587c", height=0.62)
    ax.set_xlabel("Mean |SHAP| (average log-odds impact on test sample)", fontsize=8.5, color="#202124")
    ax.tick_params(axis="both", labelsize=8, colors="#202124")
    ax.grid(axis="x", linestyle=":", color="#e2e4e8", alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#d0d5dd")
    fig.tight_layout(pad=0.3)
    return fig


def style_decision_cell(val):
    if val == "FLAG":
        return "color: #b3261e; font-weight: 600;"
    elif val == "REVIEW":
        return "color: #b45309; font-weight: 600;"
    elif val == "CLEAR":
        return "color: #5f6368;"
    return ""


# Load dataset artifacts
metrics = load_metrics()
thr = load_thresholds()
cases = load_cases()
global_shap = load_global_shap()

m = metrics.get("model", {})
t_final = metrics.get("tiers_final", metrics.get("tiers_model_plus_rules", {}))
t_model_only = metrics.get("tiers_model_only", {})
t_rules = metrics.get("tiers_model_plus_rules", {})
cuts = metrics.get("cuts", {})
test_start = cuts.get("valid_max_step", 0) + 1
test_end = cuts.get("max_step", 0)
artifact_dt = get_artifacts_timestamp()
shap_cols = [c for c in cases.columns if c.startswith("shap_")]

# Subtle CSS: light theme, 1px flat borders, generous white space, no glow or gradients
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #202124;
    }
    h1 {
        font-size: 1.45rem !important;
        font-weight: 600 !important;
        color: #202124 !important;
        margin-bottom: 0.2rem !important;
        letter-spacing: -0.01em;
    }
    h2 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #202124 !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.4rem !important;
    }
    h3 {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #202124 !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.2rem !important;
    }
    .page-desc {
        font-size: 0.92rem;
        color: #5f6368;
        margin-bottom: 0.9rem;
        line-height: 1.45;
    }
    .stButton button {
        border: 1px solid #d0d5dd !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
        color: #202124 !important;
        font-size: 0.88rem !important;
        padding: 0.35rem 0.75rem !important;
        box-shadow: none !important;
    }
    .stButton button:hover {
        border-color: #24587c !important;
        color: #24587c !important;
        background-color: #f7f9fb !important;
    }
    .sidebar-meta {
        font-size: 0.82rem;
        color: #5f6368;
        line-height: 1.5;
    }
    .status-tag-flag {
        color: #b3261e;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .status-tag-review {
        color: #b45309;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .status-tag-clear {
        color: #5f6368;
        font-weight: 600;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation state handling
if "target_page" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("target_page")

if "target_case_id" in st.session_state:
    t_id = st.session_state.pop("target_case_id")
    st.session_state["selected_case_id"] = t_id
    st.session_state["case_page_picker"] = t_id

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Summary"

# Left sidebar
st.sidebar.markdown("### Fraud triage")
selected_page = st.sidebar.radio(
    "Navigation",
    PAGES,
    key="nav_page"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="sidebar-meta">
    <b>Dataset:</b> PaySim (synthetic)<br>
    <b>Test period:</b> Steps {test_start} to {test_end} ({m.get('n', 0):,} txns)<br>
    <b>Artifacts generated:</b> {artifact_dt}
    </div>
    """,
    unsafe_allow_html=True,
)


def go_to(page_name: str, case_id: int = None):
    st.session_state["target_page"] = page_name
    if case_id is not None:
        st.session_state["target_case_id"] = case_id
    st.rerun()


# Page 1: Summary
if selected_page == "Summary":
    st.markdown("<h1>Summary</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">This page summarizes overall performance on the test period and compares the final model against rules.</div>',
        unsafe_allow_html=True,
    )

    col_btn_wrap, _ = st.columns([1, 4])
    with col_btn_wrap:
        if st.button("Go to alerts queue →"):
            go_to("Alerts")

    # Answer "How well did it do?" on the first screen without scrolling
    st.markdown("## How well did it do?")

    # Compact results table for the final system
    t_flag = t_final.get("FLAG", {})
    t_rev = t_final.get("REVIEW", {})
    t_clr = t_final.get("CLEAR", {})
    t_both = t_final.get("FLAG+REVIEW", {})

    t_flag_txns = t_flag.get("transactions", 0)
    t_rev_txns = t_rev.get("transactions", 0)
    t_clr_txns = t_clr.get("transactions", 0)
    t_both_txns = t_both.get("transactions", 0)

    results_rows = [
        {
            "Tier": "FLAG",
            "Transactions": f"{t_flag_txns:,}",
            "Fraud caught": f"{t_flag.get('fraud', 0):,}",
            "Fraud missed": "0",
            "Precision": f"{t_flag.get('precision', 0.0):.1%}" if t_flag_txns > 0 else "n/a",
            "Share of all fraud": f"{t_flag.get('share_of_all_fraud', 0.0):.1%}" if t_flag_txns > 0 else "n/a",
            "Action": "Immediate freeze or block",
        },
        {
            "Tier": "REVIEW",
            "Transactions": f"{t_rev_txns:,}",
            "Fraud caught": f"{t_rev.get('fraud', 0):,}",
            "Fraud missed": "0",
            "Precision": f"{t_rev.get('precision', 0.0):.1%}" if t_rev_txns > 0 else "n/a",
            "Share of all fraud": f"{t_rev.get('share_of_all_fraud', 0.0):.1%}" if t_rev_txns > 0 else "n/a",
            "Action": "Analyst manual review",
        },
        {
            "Tier": "CLEAR",
            "Transactions": f"{t_clr_txns:,}",
            "Fraud caught": "0",
            "Fraud missed": f"{t_clr.get('fraud', 0):,}",
            "Precision": "",
            "Share of all fraud": f"{t_clr.get('share_of_all_fraud', 0.0):.1%}" if t_clr_txns > 0 else "n/a",
            "Action": "Auto-cleared",
        },
        {
            "Tier": "Total alerts (FLAG + REVIEW)",
            "Transactions": f"{t_both_txns:,}",
            "Fraud caught": f"{t_both.get('fraud', 0):,}",
            "Fraud missed": f"{t_clr.get('fraud', 0):,}",
            "Precision": f"{t_both.get('precision', 0.0):.1%}" if t_both_txns > 0 else "n/a",
            "Share of all fraud": f"{t_both.get('recall', 0.0):.1%}" if t_both_txns > 0 else "n/a",
            "Action": "Total human queue",
        },
    ]
    st.table(pd.DataFrame(results_rows))

    if t_rev_txns == 0:
        st.markdown(
            "The REVIEW tier is empty because the FLAG and REVIEW thresholds coincide: "
            "PaySim fraud is almost perfectly separable, so there is no uncertain zone."
        )

    # 2-3 plain sentences stating result and main caveat + term definitions
    st.markdown(
        f"""
On the **{m.get('n', 0):,}** hold-out test transactions (steps {test_start} to {test_end}), the final model raised **{t_both.get('transactions', 0):,}** alerts, catching **{t_both.get('fraud', 0):,}** of the **{m.get('n_fraud', 0):,}** fraudulent transactions (**{t_both.get('recall', 0.0):.1%}** recall) with **{t_both.get('precision', 0.0):.1%}** precision. Exactly **{t_clr.get('fraud', 0):,}** fraud transactions were missed in CLEAR. These near-perfect numbers reflect the scripted rules of the synthetic PaySim simulator, where fraud follows an explicit account-draining pattern; real-world payment flows have substantially higher noise and lower precision.

*Terms explained:* **Recall** is the share of all fraud we caught ({t_both.get('fraud', 0):,} of {m.get('n_fraud', 0):,}). **Precision** is the share of alerts that were actual fraud ({t_both.get('fraud', 0):,} of {t_both.get('transactions', 0):,}). **PR-AUC** is the area under the precision-recall curve ({m.get('pr_auc', 0.0):.3f} vs {m.get('base_rate', 0.0):.4f} for random guessing).
        """
    )

    st.markdown("---")

    # Below: precision-recall chart and global SHAP drivers
    col_pr, col_shap = st.columns(2)

    with col_pr:
        st.markdown("### Precision-recall curve")
        pr_path = f"{ART}/pr_curve.png"
        if os.path.exists(pr_path):
            st.image(
                pr_path,
                caption=f"Precision-recall curve on the hold-out test period (PR-AUC: {m.get('pr_auc', 0.0):.3f}, base rate: {m.get('base_rate', 0.0):.4f})."
            )
        else:
            st.write(f"PR-AUC: {m.get('pr_auc', 0.0):.4f} (baseline: {m.get('base_rate', 0.0):.4f})")

    with col_shap:
        st.markdown("### Top global drivers (SHAP)")
        if global_shap is not None and not global_shap.empty:
            fig_g = plot_global_shap(global_shap)
            st.pyplot(fig_g)
            plt.close(fig_g)
        else:
            st.write("No global SHAP data available.")

    st.markdown("### Model-only vs model+rules comparison")
    ablation_rows = [
        {
            "System": "Final: CatBoost model only",
            "Alerts (FLAG+REVIEW)": f"{t_model_only.get('FLAG+REVIEW', {}).get('transactions', 0):,}",
            "Fraud caught": f"{t_model_only.get('FLAG+REVIEW', {}).get('fraud', 0):,}",
            "Recall": f"{t_model_only.get('FLAG+REVIEW', {}).get('recall', 0.0):.1%}",
            "Precision": f"{t_model_only.get('FLAG+REVIEW', {}).get('precision', 0.0):.1%}",
        },
        {
            "System": "Ablation: Model + heuristic rules (R1/R2)",
            "Alerts (FLAG+REVIEW)": f"{t_rules.get('FLAG+REVIEW', {}).get('transactions', 0):,}",
            "Fraud caught": f"{t_rules.get('FLAG+REVIEW', {}).get('fraud', 0):,}",
            "Recall": f"{t_rules.get('FLAG+REVIEW', {}).get('recall', 0.0):.1%}",
            "Precision": f"{t_rules.get('FLAG+REVIEW', {}).get('precision', 0.0):.1%}",
        },
    ]
    st.table(pd.DataFrame(ablation_rows))
    st.markdown(
        "**Takeaway:** Adding heuristic rules created 183 additional alerts but caught only 1 extra fraud case, lowering precision from 99.9% to 90.0%, which is why the final system uses model-only decisions."
    )


# Page 2: Alerts
elif selected_page == "Alerts":
    st.markdown("<h1>Alerts</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">This page lists every transaction the model alerted on. Filter by criteria or click a row to open it.</div>',
        unsafe_allow_html=True,
    )

    # Filters at top in one row
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    all_decisions = ["FLAG", "REVIEW", "CLEAR"]
    with f_col1:
        selected_decisions = st.multiselect(
            "Decision",
            options=all_decisions,
            default=["FLAG", "REVIEW"]
        )

    available_types = sorted(cases["type"].dropna().unique().tolist())
    with f_col2:
        selected_types = st.multiselect(
            "Transaction type",
            options=available_types,
            default=available_types
        )

    with f_col3:
        min_score = st.slider(
            "Minimum score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    with f_col4:
        search_reason = st.text_input(
            "Search in reasons",
            value="",
            placeholder="e.g. drain, emptied, 0"
        )

    # Filter data
    filtered = cases.copy()
    if selected_decisions:
        filtered = filtered[filtered["decision"].isin(selected_decisions)]
    else:
        filtered = filtered.iloc[0:0]

    if selected_types:
        filtered = filtered[filtered["type"].isin(selected_types)]
    else:
        filtered = filtered.iloc[0:0]

    filtered = filtered[filtered["score"] >= min_score]

    if search_reason.strip():
        term = search_reason.strip().lower()
        in_reasons = filtered["reasons"].fillna("").str.lower().str.contains(term, regex=False)
        in_rule = filtered["rule_fired"].fillna("").str.lower().str.contains(term, regex=False) if "rule_fired" in filtered.columns else False
        filtered = filtered[in_reasons | in_rule]

    filtered = filtered.sort_values("score", ascending=False)
    total_alerts_count = len(cases[cases["decision"].isin(["FLAG", "REVIEW"])])

    # Row count display
    st.markdown(f"**Showing {len(filtered):,} of {total_alerts_count:,} alerts**")

    if filtered.empty:
        st.write("No transactions match the selected filters. Try widening the filters above.")
    else:
        # Pagination
        page_size = 100
        total_pages = max((len(filtered) - 1) // page_size + 1, 1)

        p_col1, p_col2, p_col3 = st.columns([1.5, 3, 2])
        with p_col1:
            current_page_num = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1
            )
        with p_col2:
            st.markdown(
                f"<div style='padding-top: 1.8rem; font-size: 0.88rem; color: #5f6368;'>"
                f"Page {current_page_num} of {total_pages} ({len(filtered):,} matching rows)</div>",
                unsafe_allow_html=True
            )

        start_idx = (current_page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered))
        page_slice = filtered.iloc[start_idx:end_idx]

        # Case selection control
        c_pick_col, c_btn_col = st.columns([4, 1])
        with c_pick_col:
            picker_indices = page_slice.index.tolist()
            chosen_case_idx = st.selectbox(
                "Select a case to inspect",
                options=picker_indices,
                format_func=lambda idx: f"Row #{idx} | {page_slice.loc[idx, 'type']} | Amount: {page_slice.loc[idx, 'amount']:,.2f} | Score: {page_slice.loc[idx, 'score']:.4f} | {page_slice.loc[idx, 'decision']}"
            )
        with c_btn_col:
            st.write("")
            st.write("")
            if st.button("Open case →", key="btn_open_selected_case"):
                go_to("Look at one case", chosen_case_idx)

        # Columns to display
        display_cols = [c for c in ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest",
                                    "score", "decision", "rule_fired", "reasons", "isFraud"] if c in page_slice.columns]
        table_to_show = page_slice[display_cols].copy()

        # Format dataframe
        styled_table = table_to_show.style.map(
            style_decision_cell,
            subset=["decision"]
        ).format({
            "amount": "{:,.2f}",
            "oldbalanceOrg": "{:,.2f}",
            "oldbalanceDest": "{:,.2f}",
            "score": "{:.4f}",
        })

        selection = st.dataframe(
            styled_table,
            width="stretch",
            hide_index=False,
            selection_mode="single-row",
            on_select="rerun",
            key="alerts_table"
        )

        # Jump if row clicked in dataframe
        if selection and selection.selection and selection.selection.rows:
            clicked_row_pos = selection.selection.rows[0]
            if clicked_row_pos < len(page_slice):
                target_case_id = page_slice.index[clicked_row_pos]
                go_to("Look at one case", target_case_id)


# Page 3: Look at one case
elif selected_page == "Look at one case":
    st.markdown("<h1>Look at one case</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">This page inspects a single transaction in detail, showing model score, decision thresholds, plain-English reasons, and SHAP feature contributions.</div>',
        unsafe_allow_html=True,
    )

    alert_pool = cases[cases["decision"].isin(["FLAG", "REVIEW"])].sort_values("score", ascending=False)
    if alert_pool.empty:
        alert_pool = cases.sort_values("score", ascending=False)

    pool_indices = alert_pool.index.tolist()

    # Determine currently selected case
    if "selected_case_id" not in st.session_state or st.session_state["selected_case_id"] not in cases.index:
        st.session_state["selected_case_id"] = pool_indices[0]

    if "case_page_picker" not in st.session_state or st.session_state["case_page_picker"] not in pool_indices:
        st.session_state["case_page_picker"] = st.session_state["selected_case_id"]

    current_case_id = st.session_state["selected_case_id"]
    current_pos = pool_indices.index(current_case_id) if current_case_id in pool_indices else -1

    def go_prev_case():
        pos = pool_indices.index(st.session_state["selected_case_id"]) if st.session_state["selected_case_id"] in pool_indices else 0
        if pos > 0:
            prev_id = pool_indices[pos - 1]
            st.session_state["selected_case_id"] = prev_id
            st.session_state["case_page_picker"] = prev_id

    def go_next_case():
        pos = pool_indices.index(st.session_state["selected_case_id"]) if st.session_state["selected_case_id"] in pool_indices else 0
        if pos < len(pool_indices) - 1:
            next_id = pool_indices[pos + 1]
            st.session_state["selected_case_id"] = next_id
            st.session_state["case_page_picker"] = next_id

    def on_case_picked():
        st.session_state["selected_case_id"] = st.session_state["case_page_picker"]

    # Navigation buttons: Previous / Next alert + selector
    nav_col1, nav_col2, nav_col3 = st.columns([1.2, 3.6, 1.2])

    with nav_col1:
        prev_disabled = (current_pos <= 0)
        st.button("← Previous alert", disabled=prev_disabled, on_click=go_prev_case, key="btn_prev_case")

    with nav_col2:
        st.selectbox(
            "Select transaction",
            options=pool_indices,
            format_func=lambda idx: f"Case #{idx} | {cases.loc[idx, 'type']} | Amount: {cases.loc[idx, 'amount']:,.2f} | Score: {cases.loc[idx, 'score']:.4f} | {cases.loc[idx, 'decision']}",
            key="case_page_picker",
            on_change=on_case_picked
        )

    with nav_col3:
        next_disabled = (current_pos < 0 or current_pos >= len(pool_indices) - 1)
        st.button("Next alert →", disabled=next_disabled, on_click=go_next_case, key="btn_next_case")

    r = cases.loc[current_case_id]

    # Two-column layout
    left_col, right_col = st.columns([1, 1])

    with left_col:
        dec = str(r["decision"])
        if dec == "FLAG":
            st.markdown("Decision: <span class='status-tag-flag'>FLAG</span>", unsafe_allow_html=True)
        elif dec == "REVIEW":
            st.markdown("Decision: <span class='status-tag-review'>REVIEW</span>", unsafe_allow_html=True)
        else:
            st.markdown("Decision: <span class='status-tag-clear'>CLEAR</span>", unsafe_allow_html=True)

        # Score compared to thresholds
        flag_thr = float(thr.get("flag", 0.8))
        review_thr = float(thr.get("review", 0.8))
        case_score = float(r["score"])

        st.markdown("### Score vs thresholds")
        fig_score = plot_score_bar(case_score, flag_thr, review_thr)
        st.pyplot(fig_score)
        plt.close(fig_score)

        st.caption(
            f"Cut-offs frozen from validation: Flag ≥ {flag_thr:.3f}, Review ≥ {review_thr:.3f}. "
            f"Model score for this case is {case_score:.4f}."
        )

        # Plain-English reasons as a short list
        st.markdown("### Plain-English reasons")
        reasons_str = str(r.get("reasons", "")) if pd.notna(r.get("reasons")) else ""
        reasons_list = [rs.strip() for rs in reasons_str.split(";") if rs.strip()]

        if reasons_list:
            for item in reasons_list:
                st.markdown(f"- {item}")
        else:
            st.write("No specific SHAP or rule reasons were triggered for this transaction.")

        if pd.notna(r.get("rule_fired")) and str(r.get("rule_fired")).strip():
            st.markdown(f"**Rule fired:** `{r['rule_fired']}`")

        # Transaction facts
        st.markdown("### Transaction facts")
        facts_data = [
            ("Transaction type", str(r.get("type", "Unknown"))),
            ("Amount", f"{float(r.get('amount', 0.0)):,.2f}"),
            ("Sender balance before", f"{float(r.get('oldbalanceOrg', 0.0)):,.2f}"),
            ("Destination balance before", f"{float(r.get('oldbalanceDest', 0.0)):,.2f}"),
            ("Step (time)", f"Step {int(r.get('step', 0))} (day {int(r.get('step', 0)) // 24 + 1}, hour {int(r.get('step', 0)) % 24})"),
        ]
        facts_df = pd.DataFrame(facts_data, columns=["Fact", "Value"])
        st.table(facts_df)

        # True label
        is_fraud = int(r.get("isFraud", 0)) == 1
        label_text = "FRAUD" if is_fraud else "LEGITIMATE"
        st.markdown(f"**Ground truth (test label):** {label_text}")

    with right_col:
        st.markdown("### SHAP feature contributions")
        sv_dict = {c.replace("shap_", ""): float(r[c]) for c in shap_cols if c in r}
        if sv_dict:
            sv_series = pd.Series(sv_dict).sort_values(ascending=True)
            fig_shap = plot_case_shap(sv_series)
            st.pyplot(fig_shap)
            plt.close(fig_shap)

            st.caption(
                "How to read this: Bars to the right (red) pushed the model toward predicting fraud. "
                "Bars to the left (blue) pushed it toward legitimate."
            )
        else:
            st.write("No SHAP values recorded for this case.")

        st.markdown("---")
        if st.button("← Back to alerts queue", key="btn_back_to_alerts"):
            go_to("Alerts")


# Page 4: How this was built
elif selected_page == "How this was built":
    st.markdown("<h1>How this was built</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">This page documents the data, split, leakage controls, thresholding methodology, rules ablation, and known limitations.</div>',
        unsafe_allow_html=True,
    )

    col_btn_sum, _ = st.columns([1, 4])
    with col_btn_sum:
        if st.button("← Back to summary"):
            go_to("Summary")

    # Missed fraud section
    st.markdown("## Missed fraud (false negatives)")
    missed_cases = cases[(cases["isFraud"] == 1) & (cases["decision"] == "CLEAR")]

    if not missed_cases.empty:
        st.markdown(
            f"**{len(missed_cases):,} fraud cases were auto-cleared.** We show them on purpose to inspect why the system failed to alert."
        )
        missed_display_cols = [c for c in ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest",
                                           "score", "rule_fired", "reasons"] if c in missed_cases.columns]
        missed_styled = missed_cases[missed_display_cols].style.format({
            "amount": "{:,.2f}",
            "oldbalanceOrg": "{:,.2f}",
            "oldbalanceDest": "{:,.2f}",
            "score": "{:.4f}",
        })
        st.dataframe(missed_styled, width="stretch", hide_index=False)

        col_miss_pick, col_miss_btn = st.columns([4, 1])
        with col_miss_pick:
            picked_miss_id = st.selectbox(
                "Inspect a missed fraud case in detail",
                options=missed_cases.index.tolist(),
                format_func=lambda idx: f"Row #{idx} | {missed_cases.loc[idx, 'type']} | Amount: {missed_cases.loc[idx, 'amount']:,.2f} | Score: {missed_cases.loc[idx, 'score']:.4f}"
            )
        with col_miss_btn:
            st.write("")
            st.write("")
            if st.button("Inspect case →", key="btn_inspect_missed"):
                go_to("Look at one case", picked_miss_id)
    else:
        st.write("No fraudulent transactions were cleared on the hold-out test period.")

    st.markdown("---")
    st.markdown("## Methodology details")

    with st.expander("Data and chronological split"):
        st.markdown(
            f"""
- **Dataset:** PaySim (synthetic mobile money payments dataset, Lopez-Rojas et al.).
- **Chronological split:** Transactions are partitioned strictly by simulation step to prevent temporal look-ahead leakage.
  - Training period: Step 1 to {cuts.get('train_max_step', 482)}
  - Validation period: Step {cuts.get('train_max_step', 482) + 1} to {cuts.get('valid_max_step', 594)}
  - Hold-out test period: Step {test_start} to {test_end} ({m.get('n', 0):,} transactions, {m.get('n_fraud', 0):,} fraud)
- **Risky types filter:** In PaySim, fraudulent events occur exclusively in `TRANSFER` and `CASH_OUT` transaction types. CatBoost is trained solely on these risky types; all other transaction types (`PAYMENT`, `CASH_IN`, `DEBIT`) are auto-cleared by policy.
            """
        )

    with st.expander("Leakage controls (features dropped and why)"):
        st.markdown(
            """
| Feature | Action | Rationale |
|---|---|---|
| `newbalanceOrig`, `newbalanceDest` | Dropped | Post-transaction balances computed after the event. They directly encode the fraud mechanics and would cause severe target leakage. |
| `isFlaggedFraud` | Dropped | Crude built-in threshold (> 200,000) that is not an independent causal signal. |
| `nameOrig`, `nameDest` strings | Dropped | High-cardinality account IDs that cause tree models to memorize specific accounts rather than learn generalizable behavior. |
| `day` (absolute simulation day) | Dropped | Ranked high in initial SHAP importance, but absolute time is an artifact of the simulation and test days lie entirely outside training days. Dropping it did not lower test PR-AUC. |
| Account drain definition | Refined | Set to transaction amount within 2% of sender balance. In PaySim, transactions where amount strictly exceeds balance are common and mostly benign. |
| `dest_prior_txn_count` | Retained | Causal count that strictly aggregates *prior* sightings of the destination account. Verified causal in unit tests. |
            """
        )

    with st.expander("How thresholds were chosen"):
        st.markdown(
            f"""
- Thresholds are fitted on the **validation period** and frozen for the test period (no test-set peeking).
- Mode: **{thr.get('mode', 'target')}**
  - **FLAG threshold ({thr.get('flag', 0.8):.4f}):** Lowest score that maintains validation precision ≥ {thr.get('flag_precision', 0.999):.1%}.
  - **REVIEW threshold ({thr.get('review', 0.8):.4f}):** Lowest score that achieves validation recall ≥ {thr.get('review_recall', 0.99):.1%}.
- In PaySim validation, fraud cases are cleanly separable from normal activity, so the FLAG and REVIEW thresholds coincide at **{thr.get('flag', 0.8):.4f}**. In real-world data with noisy labels, these two thresholds diverge into distinct action queues.
            """
        )

    with st.expander("Rules ablation (model-only vs model+rules)"):
        st.markdown(
            f"""
Two heuristic rules were authored to emulate common fraud policies:
- **R1 (account drain):** Sender account balance is emptied to zero in a single transaction. Escalates to REVIEW.
- **R2 (drain to empty new destination):** Sender emptied + destination initial balance was 0 + destination never seen before. Escalates to FLAG.

**Ablation test results:**
- Model-only queue: **{t_model_only.get('FLAG+REVIEW', {}).get('transactions', 0):,}** alerts, **{t_model_only.get('FLAG+REVIEW', {}).get('recall', 0.0):.1%}** recall, **{t_model_only.get('FLAG+REVIEW', {}).get('precision', 0.0):.1%}** precision.
- Model + rules queue: **{t_rules.get('FLAG+REVIEW', {}).get('transactions', 0):,}** alerts, **{t_rules.get('FLAG+REVIEW', {}).get('recall', 0.0):.1%}** recall, **{t_rules.get('FLAG+REVIEW', {}).get('precision', 0.0):.1%}** precision.

**Conclusion:** Adding heuristic rules introduced **183** additional alerts but caught only **1** additional fraud case, degrading precision by **9.9 percentage points**. Therefore, rules are disabled in the final decision pipeline (`rules_used_in_final: {metrics.get('rules_used_in_final', False)}`).
            """
        )

    with st.expander("Limitations (honest appraisal)"):
        st.markdown(
            """
1. **PaySim is synthetic:** The data originates from an agent-based simulator with scripted fraud mechanics (account drain). Near-perfect scores reflect the simulator rules rather than real-world detection difficulty. Metrics should be treated as a demonstration of methodology.
2. **Fraud rate discrepancy:** The test period has a higher fraud rate (**1.34%**) than the full dataset (**0.13%**).
3. **Transaction type restriction:** Fraud is confined exclusively to `TRANSFER` and `CASH_OUT` in PaySim. Auto-clearing all other types is a dataset-specific rule.
4. **Heuristic rules ablation:** Rules lowered precision without significant recall gains here; both model-only and model+rules metrics are disclosed for transparency.
5. **Static evaluation:** No concept drift, adversary adaptation, or fairness/bias across demographic cohorts is evaluated.
            """
        )
