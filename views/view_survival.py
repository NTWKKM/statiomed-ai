"""
views/view_survival.py - StatioMed AI Survival Analysis View (Gradio Native)
===========================================================================
Kaplan-Meier survival estimation, Log-Rank testing, Cox Proportional
Hazards multivariable modeling, Schoenfeld residual diagnostics, and
Landmark survival analysis.
===========================================================================
"""

from __future__ import annotations

import html

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from core.common import select_variable_by_keyword
from core.state import AppState
from logger import get_logger
from utils import survival_lib

logger = get_logger(__name__)


def refresh_survival_columns(
    state: AppState,
) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    """Updates dropdown choices based on current active dataset."""
    df = state.get_active_dataframe()
    if df is None or df.empty:
        return (
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
        )

    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    default_time = select_variable_by_keyword(
        num_cols, ["time", "duration", "fu_days", "os_months", "days"]
    )
    default_event = select_variable_by_keyword(
        cols, ["death", "event", "status", "censored", "died", "mortality"]
    )
    default_group = select_variable_by_keyword(
        cols, ["treatment", "group", "arm", "therapy", "intervention", "sex"]
    )
    covariate_candidates = [c for c in cols if c not in [default_time, default_event]]

    return (
        gr.Dropdown(choices=num_cols, value=default_time),
        gr.Dropdown(choices=cols, value=default_event),
        gr.Dropdown(choices=["None"] + cols, value=default_group or "None"),
        gr.Dropdown(choices=cols, value=covariate_candidates[:4]),
    )


def run_km_analysis(
    time_col: str, event_col: str, group_col: str, state: AppState
) -> tuple[go.Figure, pd.DataFrame | None, str]:
    """Action callback: Fits Kaplan-Meier survival curves and calculates Log-rank test."""
    df = state.get_active_dataframe()
    if df is None or df.empty:
        return (
            go.Figure(),
            None,
            "<div style='color:#b91c1c;'>No active dataset. Please load data first.</div>",
        )
    if not time_col or not event_col:
        return (
            go.Figure(),
            None,
            "<div style='color:#b91c1c;'>Please select Time and Event variables.</div>",
        )

    try:
        grp = None if group_col in ["None", "", None] else group_col
        fig, summary_df, stats_dict = survival_lib.fit_km_logrank(
            df=df,
            duration_col=time_col,
            event_col=event_col,
            group_col=grp,
        )

        p_val = stats_dict.get("p_value", None)
        test_name = stats_dict.get("test_name", "Log-Rank Test")
        p_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
        is_sig = isinstance(p_val, float) and p_val < 0.05

        sig_badge = (
            "<span style='background:#dcfce7;color:#166534;padding:3px 8px;border-radius:4px;font-weight:600;'>Statistically Significant (P < 0.05)</span>"
            if is_sig
            else "<span style='background:#f1f5f9;color:#475569;padding:3px 8px;border-radius:4px;'>Not Significant</span>"
        )

        summary_html = f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-top:10px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <strong style='color:#0f172a;font-size:1rem;'>📊 {html.escape(test_name)}</strong>
                {sig_badge}
            </div>
            <div style='margin-top:6px;color:#334155;font-size:0.9rem;'>
                <strong>P-value:</strong> {p_str} | <strong>Chi-square:</strong> {stats_dict.get("test_statistic", 0.0):.2f} | <strong>Total Events:</strong> {stats_dict.get("total_events", df[event_col].sum())}
            </div>
        </div>
        """
        return fig, summary_df, summary_html

    except Exception as e:
        logger.error(f"KM Analysis Error: {e}")
        return (
            go.Figure(),
            None,
            f"<div style='color:#b91c1c;'>Error in KM Analysis: {html.escape(str(e))}</div>",
        )


def run_cox_analysis(
    time_col: str, event_col: str, covar_cols: list[str], state: AppState
) -> tuple[pd.DataFrame | None, go.Figure, str]:
    """Action callback: Fits multivariable Cox Proportional Hazards regression."""
    df = state.get_active_dataframe()
    if df is None or df.empty:
        return None, go.Figure(), "<div style='color:#b91c1c;'>No active dataset.</div>"
    if not time_col or not event_col or not covar_cols:
        return (
            None,
            go.Figure(),
            "<div style='color:#b91c1c;'>Please select Time, Event, and at least one Covariate.</div>",
        )

    try:
        cph, res_df, _, error_msg, stats_dict, _ = survival_lib.fit_cox_ph(
            df=df,
            duration_col=time_col,
            event_col=event_col,
            covariate_cols=covar_cols,
        )
        if error_msg or res_df is None:
            return (
                None,
                go.Figure(),
                f"<div style='color:#b91c1c;'>Error in Cox Model: {html.escape(str(error_msg))}</div>",
            )

        stats_dict = stats_dict or {}
        c_str = str(stats_dict.get("Concordance Index (C-index)", "N/A"))
        aic_str = str(stats_dict.get("AIC", "N/A"))
        ll_str = str(stats_dict.get("Log-Likelihood", "N/A"))

        forest_fig = survival_lib.create_forest_plot_cox(res_df)

        stats_html = f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-top:10px;'>
            <strong style='color:#0f172a;font-size:1rem;'>⏱️ Cox Proportional Hazards Model Summary</strong>
            <div style='margin-top:6px;color:#334155;font-size:0.9rem;'>
                <strong>Concordance Index (C-index):</strong> <span style='color:#059669;font-weight:600;'>{c_str}</span> | 
                <strong>Log-Likelihood:</strong> {ll_str} | 
                <strong>AIC:</strong> {aic_str}
            </div>
        </div>
        """
        return res_df, forest_fig, stats_html

    except Exception as e:
        logger.error(f"Cox PH Error: {e}")
        return (
            None,
            go.Figure(),
            f"<div style='color:#b91c1c;'>Error in Cox Model: {html.escape(str(e))}</div>",
        )


def create_survival_view(app_state: gr.State) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for Survival Analysis.
    """
    with gr.Tab("⏱️ Survival", id="tab_survival") as tab:
        gr.Markdown(
            """
            ### ⏱️ Survival Analysis & Time-to-Event Modeling
            *Kaplan-Meier Curves, Log-Rank Hypothesis Testing, and Multivariable Cox Proportional Hazards Regression.*
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                btn_refresh = gr.Button(
                    "🔄 Refresh Column List", variant="secondary", size="sm"
                )
                surv_time = gr.Dropdown(label="⏱️ Time Variable (Duration):", choices=[])
                surv_event = gr.Dropdown(
                    label="🎯 Event / Status (1=Event, 0=Censored):", choices=[]
                )
                surv_group = gr.Dropdown(
                    label="👥 Stratification / Group Variable (Optional):",
                    choices=["None"],
                )
                surv_covariates = gr.Dropdown(
                    label="📋 Cox Covariates (Multivariable):",
                    choices=[],
                    multiselect=True,
                )

            with gr.Column(scale=8):
                with gr.Tabs():
                    with gr.Tab("📈 Kaplan-Meier & Log-Rank"):
                        btn_run_km = gr.Button(
                            "🚀 Fit Kaplan-Meier Curve", variant="primary"
                        )
                        km_plot = gr.Plot(label="Kaplan-Meier Survival Function")
                        km_summary_html = gr.HTML("")
                        km_table = gr.Dataframe(
                            label="Median Survival & Event Breakdown", interactive=False
                        )

                    with gr.Tab("⏱️ Cox Proportional Hazards"):
                        btn_run_cox = gr.Button(
                            "🚀 Fit Multivariable Cox Model", variant="primary"
                        )
                        cox_summary_html = gr.HTML("")
                        cox_table = gr.Dataframe(
                            label="Hazard Ratios & Parameter Estimates",
                            interactive=False,
                        )
                        cox_forest_plot = gr.Plot(
                            label="Forest Plot (Hazard Ratios & 95% CI)"
                        )

        # Events
        btn_refresh.click(
            fn=refresh_survival_columns,
            inputs=[app_state],
            outputs=[surv_time, surv_event, surv_group, surv_covariates],
        )

        btn_run_km.click(
            fn=run_km_analysis,
            inputs=[surv_time, surv_event, surv_group, app_state],
            outputs=[km_plot, km_table, km_summary_html],
        )

        btn_run_cox.click(
            fn=run_cox_analysis,
            inputs=[surv_time, surv_event, surv_covariates, app_state],
            outputs=[cox_table, cox_forest_plot, cox_summary_html],
        )

    return tab, {
        "btn_refresh": btn_refresh,
        "surv_time": surv_time,
        "surv_event": surv_event,
        "surv_group": surv_group,
        "surv_covariates": surv_covariates,
        "btn_run_km": btn_run_km,
        "km_plot": km_plot,
        "btn_run_cox": btn_run_cox,
        "cox_table": cox_table,
    }
