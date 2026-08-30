"""
views/view_table_one_matching.py - StatioMed AI Table 1 & PSM View (Gradio Native)
===================================================================================
Baseline Characteristics Table (Table 1) generation with Standardized Mean
Differences (SMD), and Propensity Score Matching (PSM) with Love Plot balance checks.
===================================================================================
"""

from __future__ import annotations

import html

import gradio as gr
import plotly.graph_objects as go

from core.common import select_variable_by_keyword
from core.state import AppState
from logger import get_logger
from utils import psm_lib
from utils.table_one_advanced import TableOneGenerator

logger = get_logger(__name__)


def refresh_table_one_columns(
    state: AppState,
) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    """Updates dropdown choices based on active dataset."""
    df = state.get_active_dataframe()
    if df is None or df.empty:
        return (
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
        )

    cols = df.columns.tolist()
    default_grp = select_variable_by_keyword(
        cols, ["treatment", "group", "arm", "therapy", "intervention"]
    )
    var_candidates = [c for c in cols if c != default_grp]

    return (
        gr.Dropdown(
            choices=["Overall (No Group)"] + cols,
            value=default_grp or "Overall (No Group)",
        ),
        gr.Dropdown(choices=cols, value=var_candidates[:8]),
        gr.Dropdown(choices=cols, value=default_grp or cols[0]),
        gr.Dropdown(choices=cols, value=var_candidates[:4]),
    )


def generate_table_one_action(
    group_col: str, selected_vars: list[str], show_smd: bool, state: AppState
) -> str:
    """Action callback: Builds publication-ready Baseline Table 1."""
    df = state.get_active_dataframe()
    if df is None or df.empty:
        return "<div style='color:#b91c1c;'>No active dataset.</div>"
    if not selected_vars:
        return "<div style='color:#b91c1c;'>Please select at least one variable to analyze.</div>"

    try:
        grp = None if group_col in ["Overall (No Group)", "", None] else group_col
        generator = TableOneGenerator(df, state.var_meta)

        html_table = generator.generate(
            selected_vars=selected_vars,
            stratify_by=grp,
        )
        return f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;overflow-x:auto;'>{html_table}</div>"

    except Exception as e:
        logger.error(f"Table 1 Error: {e}")
        return f"<div style='color:#b91c1c;'>Error generating Table 1: {html.escape(str(e))}</div>"


def run_psm_action(
    treatment_col: str,
    covariates: list[str],
    caliper: float,
    ratio: int,
    state: AppState,
) -> tuple[AppState, str, go.Figure]:
    """Action callback: Executes Propensity Score Matching and assesses covariate balance."""
    df = state.df
    if df is None or df.empty:
        return (
            state,
            "<div style='color:#b91c1c;'>No active dataset.</div>",
            go.Figure(),
        )
    if not treatment_col or not covariates:
        return (
            state,
            "<div style='color:#b91c1c;'>Please select Treatment variable and at least one Confounder covariate.</div>",
            go.Figure(),
        )

    try:
        ps_series, _ = psm_lib.calculate_ps(
            df, treatment=treatment_col, covariates=covariates
        )
        df_with_ps = df.copy()
        df_with_ps["_ps"] = ps_series
        df_matched = psm_lib.perform_matching(
            data=df_with_ps,
            treatment_col=treatment_col,
            ps_col="_ps",
            caliper=float(caliper),
            ratio=int(ratio),
        )
        if df_matched.empty:
            return (
                state,
                "<div style='color:#b91c1c;'>No matched pairs found within specified caliper.</div>",
                go.Figure(),
            )

        smd_pre = psm_lib.check_balance(
            df, treatment=treatment_col, covariates=covariates
        )
        smd_post = psm_lib.check_balance(
            df_matched, treatment=treatment_col, covariates=covariates
        )
        fig_love = psm_lib.plot_love_plot(smd_pre, smd_post)

        state.df_matched = df_matched
        state.is_matched = True
        state.matched_treatment_col = treatment_col
        state.matched_covariates = covariates

        n_treated_orig = int(df[treatment_col].sum())
        n_control_orig = len(df) - n_treated_orig
        n_matched = len(df_matched)

        summary_html = f"""
        <div style='background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:16px;'>
            <h4 style='color:#166534;margin-top:0;'>🎯 Propensity Score Matching Complete</h4>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0;'>
                <div style='background:#ffffff;padding:10px;border-radius:6px;border:1px solid #dcfce7;'>
                    <strong>Original Cohort:</strong> {len(df):,} patients ({n_treated_orig} Treated, {n_control_orig} Control)
                </div>
                <div style='background:#ffffff;padding:10px;border-radius:6px;border:1px solid #dcfce7;'>
                    <strong>Matched Cohort:</strong> <span style='color:#059669;font-weight:700;'>{n_matched:,} patients</span> (1:{ratio} Nearest Neighbor, Caliper {caliper} SD)
                </div>
            </div>
            <div style='color:#15803d;font-size:0.88rem;'>
                ✅ Matched dataset is now active in session. Downstream tabs (Survival, Regression) can now analyze this balanced cohort.
            </div>
        </div>
        """
        return state, summary_html, fig_love

    except Exception as e:
        logger.error(f"PSM Error: {e}")
        return (
            state,
            f"<div style='color:#b91c1c;'>Error in PSM matching: {html.escape(str(e))}</div>",
            go.Figure(),
        )


def create_table_one_matching_view(
    app_state: gr.State,
) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for Table 1 and Propensity Score Matching.
    """
    with gr.Tab("👥 Table 1 & Matching", id="tab_table_one") as tab:
        gr.Markdown(
            """
            ### 👥 Baseline Characteristics (Table 1) & Propensity Score Matching (PSM)
            *Generate publication-standard Table 1 with Standardized Mean Differences (SMD), and balance observational cohorts with PSM.*
            """
        )

        with gr.Tabs():
            with gr.Tab("📊 Baseline Table 1"):
                with gr.Row():
                    with gr.Column(scale=4):
                        btn_refresh_t1 = gr.Button(
                            "🔄 Refresh Column List", variant="secondary", size="sm"
                        )
                        t1_group = gr.Dropdown(
                            label="👥 Grouping Variable (e.g. Treatment vs Control):",
                            choices=["Overall (No Group)"],
                        )
                        t1_vars = gr.Dropdown(
                            label="📋 Variables to Include in Table 1:",
                            choices=[],
                            multiselect=True,
                        )
                        t1_smd = gr.Checkbox(
                            label="Calculate Standardized Mean Difference (SMD)",
                            value=True,
                        )
                        btn_gen_t1 = gr.Button("🚀 Generate Table 1", variant="primary")

                    with gr.Column(scale=8):
                        t1_display_html = gr.HTML(
                            """
                            <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; padding: 20px; text-align: center; color: #64748b;">
                                Select variables and click <strong>Generate Table 1</strong>.
                            </div>
                            """
                        )

            with gr.Tab("🎯 Propensity Score Matching (PSM)"):
                with gr.Row():
                    with gr.Column(scale=4):
                        psm_treat = gr.Dropdown(
                            label="💊 Binary Treatment Indicator (1=Treated, 0=Control):",
                            choices=[],
                        )
                        psm_covars = gr.Dropdown(
                            label="⚖️ Confounder Covariates for Matching:",
                            choices=[],
                            multiselect=True,
                        )
                        psm_caliper = gr.Slider(
                            label="Matching Caliper (Standard Deviations):",
                            minimum=0.05,
                            maximum=0.50,
                            value=0.20,
                            step=0.05,
                        )
                        psm_ratio = gr.Radio(
                            label="Matching Ratio (Treated:Control):",
                            choices=[("1:1 Matching", 1), ("1:2 Matching", 2)],
                            value=1,
                        )
                        btn_run_psm = gr.Button(
                            "🚀 Run Propensity Score Matching", variant="primary"
                        )

                    with gr.Column(scale=8):
                        psm_summary = gr.HTML("")
                        psm_love_plot = gr.Plot(
                            label="Covariate Balance: Love Plot (SMD Pre vs Post Matching)"
                        )

        # Events
        btn_refresh_t1.click(
            fn=refresh_table_one_columns,
            inputs=[app_state],
            outputs=[t1_group, t1_vars, psm_treat, psm_covars],
        )

        btn_gen_t1.click(
            fn=generate_table_one_action,
            inputs=[t1_group, t1_vars, t1_smd, app_state],
            outputs=[t1_display_html],
        )

        btn_run_psm.click(
            fn=run_psm_action,
            inputs=[psm_treat, psm_covars, psm_caliper, psm_ratio, app_state],
            outputs=[app_state, psm_summary, psm_love_plot],
        )

    return tab, {
        "btn_refresh_t1": btn_refresh_t1,
        "t1_group": t1_group,
        "t1_vars": t1_vars,
        "btn_gen_t1": btn_gen_t1,
        "t1_display_html": t1_display_html,
        "btn_run_psm": btn_run_psm,
    }
