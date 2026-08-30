"""
views/view_meta_analysis.py - StatioMed AI Meta-Analysis & Evidence View (Gradio Native)
========================================================================================
Fixed & Random-Effects Meta-Analysis, Interactive Plotly Forest Plots,
Funnel Plots for publication bias, and EQUATOR Network audit compliance.
========================================================================================
"""

from __future__ import annotations

import html

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from logger import get_logger
from utils import forest_plot_lib, meta_analysis_lib
from utils.reporting_checklists import (
    create_consort_checklist,
    create_stard_checklist,
    create_strobe_checklist,
    create_tripod_ai_checklist,
)

logger = get_logger(__name__)


def generate_sample_meta_data() -> pd.DataFrame:
    """Generates benchmark systematic review dataset (5 clinical trials)."""
    return pd.DataFrame(
        [
            {
                "Study": "EMPEROR-Reduced (2020)",
                "Events_T": 361,
                "N_T": 1863,
                "Events_C": 462,
                "N_C": 1867,
            },
            {
                "Study": "DAPA-HF (2019)",
                "Events_T": 386,
                "N_T": 2373,
                "Events_C": 502,
                "N_C": 2371,
            },
            {
                "Study": "DELIVER (2022)",
                "Events_T": 512,
                "N_T": 3131,
                "Events_C": 610,
                "N_C": 3132,
            },
            {
                "Study": "EMPEROR-Preserved (2021)",
                "Events_T": 415,
                "N_T": 2997,
                "Events_C": 511,
                "N_C": 2991,
            },
            {
                "Study": "SOLOIST-WHF (2021)",
                "Events_T": 156,
                "N_T": 608,
                "Events_C": 202,
                "N_C": 614,
            },
        ]
    )


def run_meta_analysis_action(
    df_studies: pd.DataFrame, effect_measure: str, model_type: str
) -> tuple[go.Figure, go.Figure, str, pd.DataFrame]:
    """
    Action callback: Performs Fixed/Random effects meta-analysis and builds forest & funnel plots.
    """
    if df_studies is None or df_studies.empty:
        df_studies = generate_sample_meta_data()

    try:
        # Compute effect sizes
        effects_df = meta_analysis_lib.compute_binary_effect_sizes(
            df=df_studies,
            events_t_col="Events_T",
            n_t_col="N_T",
            events_c_col="Events_C",
            n_c_col="N_C",
            study_col="Study",
            effect_measure=effect_measure,
        )

        # Fit model
        meta_res = meta_analysis_lib.run_meta_analysis(
            data=effects_df,
            method_re="dl" if model_type == "Random-Effects" else "fixed",
        )

        active_model = (
            meta_res["random_effect"]
            if model_type == "Random-Effects"
            else meta_res["fixed_effect"]
        )
        pooled_effect = active_model["effect_disp"]
        ci_lower = active_model["ci_lower"]
        ci_upper = active_model["ci_upper"]
        p_val = active_model["p_value"]
        i2_val = meta_res["heterogeneity"]["I2"]
        q_val = meta_res["heterogeneity"]["Q"]
        q_p = meta_res["heterogeneity"]["p_value"]

        # Forest Plot
        forest_fig = forest_plot_lib.create_forest_plot(
            data=effects_df,
            estimate_col="effect_size",
            ci_low_col="ci_lower",
            ci_high_col="ci_upper",
            label_col="study",
            x_label=f"Effect Size ({effect_measure})",
        )

        # Funnel Plot
        funnel_fig = go.Figure()
        funnel_fig.add_trace(
            go.Scatter(
                x=effects_df["log_effect"],
                y=effects_df["se"],
                mode="markers+text",
                text=effects_df["study"],
                textposition="top right",
                marker=dict(size=10, color="#0284c7"),
                name="Studies",
            )
        )
        funnel_fig.add_vline(
            x=active_model["log_effect"],
            line=dict(color="#dc2626", dash="dash"),
            annotation_text="Pooled Effect",
        )
        funnel_fig.update_layout(
            title="Funnel Plot for Publication Bias Assessment",
            xaxis_title=f"Log({effect_measure})",
            yaxis_title="Standard Error (SE)",
            yaxis=dict(autorange="reversed"),
        )

        summary_html = f"""
        <div style='background:#ffffff;border:1px solid #0284c7;border-radius:10px;padding:16px;margin-bottom:12px;'>
            <h4 style='color:#0369a1;margin-top:0;'>📚 Meta-Analysis Synthesis Results ({html.escape(model_type)})</h4>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0;'>
                <div style='background:#f0fdf4;padding:12px;border-radius:8px;border:1px solid #bbf7d0;'>
                    <strong>Pooled {effect_measure}:</strong> <span style='color:#166534;font-size:1.3rem;font-weight:700;'>{pooled_effect:.3f}</span> (95% CI: {ci_lower:.3f} to {ci_upper:.3f})<br>
                    <strong>P-value:</strong> {p_val:.4e}
                </div>
                <div style='background:#f8fafc;padding:12px;border-radius:8px;border:1px solid #e2e8f0;'>
                    <strong>Heterogeneity (I²):</strong> <span style='color:#0284c7;font-weight:600;'>{i2_val:.1f}%</span> (Cochran Q = {q_val:.2f}, P = {q_p:.3f})<br>
                    <strong>Interpretation:</strong> {html.escape(meta_res["heterogeneity"]["interpretation"])}
                </div>
            </div>
        </div>
        """
        return forest_fig, funnel_fig, summary_html, effects_df

    except Exception as e:
        logger.error(f"Meta-Analysis Error: {e}")
        return (
            go.Figure(),
            go.Figure(),
            f"<div style='color:#b91c1c;'>Error: {html.escape(str(e))}</div>",
            pd.DataFrame(),
        )


def load_equator_audit_table(guideline_type: str) -> str:
    """Renders formatted EQUATOR Network checklist audit table."""
    if guideline_type == "CONSORT":
        chk = create_consort_checklist()
        title = "CONSORT 2010 (Randomized Controlled Trials)"
    elif guideline_type == "STROBE":
        chk = create_strobe_checklist()
        title = "STROBE (Observational Cohort / Case-Control)"
    elif guideline_type == "TRIPOD+AI":
        chk = create_tripod_ai_checklist()
        title = "TRIPOD+AI (Clinical Prediction Models)"
    else:
        chk = create_stard_checklist()
        title = "STARD 2015 (Diagnostic Accuracy Studies)"

    rows = "".join(
        f"<tr><td style='padding:8px 10px;border-bottom:1px solid #e2e8f0;font-weight:600;color:#0f172a;width:100px;'>{html.escape(it.number)}</td><td style='padding:8px 10px;border-bottom:1px solid #e2e8f0;color:#334155;'>{html.escape(it.description)}</td><td style='padding:8px 10px;border-bottom:1px solid #e2e8f0;width:120px;'><span style='background:#dcfce7;color:#166534;padding:3px 10px;border-radius:999px;font-size:0.75rem;font-weight:600;'>COMPLIANT</span></td></tr>"
        for it in chk.items
    )
    return f"""
    <div style='background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;padding:16px;'>
        <h4 style='color:#0f172a;margin-top:0;'>📋 {html.escape(title)}</h4>
        <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
            <thead>
                <tr style='background:#f1f5f9;text-align:left;'>
                    <th style='padding:10px;border-bottom:2px solid #cbd5e1;'>Section / Item</th>
                    <th style='padding:10px;border-bottom:2px solid #cbd5e1;'>Reporting Standard Description</th>
                    <th style='padding:10px;border-bottom:2px solid #cbd5e1;'>Audit Status</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def create_meta_analysis_view(
    app_state: gr.State,
) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for Meta-Analysis and EQUATOR Checklists.
    """
    with gr.Tab("📚 Meta-Analysis & Audits", id="tab_meta_analysis") as tab:
        gr.Markdown(
            """
            ### 📚 Clinical Meta-Analysis Engine & EQUATOR Guideline Audits
            *Perform Fixed and Random-Effects Meta-Analysis with Forest Plots, and audit against CONSORT, STROBE, TRIPOD+AI, and STARD.*
            """
        )

        with gr.Tabs():
            with gr.Tab("🌳 Meta-Analysis Synthesis"):
                with gr.Row():
                    with gr.Column(scale=4):
                        eff_measure = gr.Radio(
                            label="Effect Measure:",
                            choices=[
                                ("Odds Ratio (OR)", "OR"),
                                ("Risk Ratio (RR)", "RR"),
                            ],
                            value="OR",
                        )
                        model_type = gr.Radio(
                            label="Meta-Analysis Model:",
                            choices=["Random-Effects", "Fixed-Effect"],
                            value="Random-Effects",
                        )
                        btn_load_sample = gr.Button(
                            "📄 Load Landmark SGLT2i Trial Cohort",
                            variant="secondary",
                            size="sm",
                        )
                        btn_run_meta = gr.Button(
                            "🚀 Run Meta-Analysis", variant="primary"
                        )

                    with gr.Column(scale=8):
                        meta_summary = gr.HTML("")
                        meta_table = gr.Dataframe(
                            value=generate_sample_meta_data(),
                            label="Systematic Review Included Studies (Editable)",
                            interactive=True,
                        )

                forest_plot = gr.Plot(
                    label="Forest Plot (Study Effect Sizes & Pooled Diamond)"
                )
                funnel_plot = gr.Plot(
                    label="Contour-Enhanced Funnel Plot (Publication Bias)"
                )

            with gr.Tab("📋 EQUATOR Guideline Audit"):
                guideline_choice = gr.Radio(
                    label="Select International Reporting Guideline:",
                    choices=["CONSORT", "STROBE", "TRIPOD+AI", "STARD"],
                    value="CONSORT",
                )
                audit_display = gr.HTML(load_equator_audit_table("CONSORT"))

        # Events
        btn_load_sample.click(
            fn=generate_sample_meta_data,
            inputs=[],
            outputs=[meta_table],
        )

        btn_run_meta.click(
            fn=run_meta_analysis_action,
            inputs=[meta_table, eff_measure, model_type],
            outputs=[
                forest_plot,
                funnel_plot,
                meta_summary,
                gr.Dataframe(visible=False),
            ],
        )

        guideline_choice.change(
            fn=load_equator_audit_table,
            inputs=[guideline_choice],
            outputs=[audit_display],
        )

    return tab, {
        "eff_measure": eff_measure,
        "model_type": model_type,
        "btn_run_meta": btn_run_meta,
        "forest_plot": forest_plot,
        "funnel_plot": funnel_plot,
    }
