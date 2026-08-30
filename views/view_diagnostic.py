"""
views/view_diagnostic.py - StatioMed AI Diagnostic Test & Agreement View (Gradio Native)
========================================================================================
Diagnostic accuracy testing (Sensitivity, Specificity, PPV, NPV, LR+/LR-, DOR),
Fagan Nomogram probability conversion, and Inter-rater reliability (Kappa, ICC).
========================================================================================
"""

from __future__ import annotations

import html

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from logger import get_logger

logger = get_logger(__name__)


def calculate_2x2_diagnostic(
    tp: int, fp: int, fn: int, tn: int, pre_test_prob_pct: float
) -> tuple[pd.DataFrame, str, go.Figure]:
    """
    Calculates diagnostic metrics (Sensitivity, Specificity, PPV, NPV, LR+, LR-, DOR)
    from 2x2 contingency table inputs.
    """
    try:
        tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
        total = tp + fp + fn + tn
        if total == 0:
            return (
                pd.DataFrame(),
                "<div style='color:#b91c1c;'>Total subjects cannot be zero.</div>",
                go.Figure(),
            )

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        lr_pos = (sens / (1.0 - spec)) if (1.0 - spec) > 0 else np.nan
        lr_neg = ((1.0 - sens) / spec) if spec > 0 else np.nan
        dor = ((tp * tn) / (fp * fn)) if (fp * fn) > 0 else np.nan

        # Post-test probability from prior probability
        p_pre = pre_test_prob_pct / 100.0
        odds_pre = p_pre / (1.0 - p_pre) if p_pre < 1.0 else 999.0
        odds_post_pos = odds_pre * lr_pos if not np.isnan(lr_pos) else np.nan
        p_post_pos = (
            (odds_post_pos / (1.0 + odds_post_pos)) * 100.0
            if not np.isnan(odds_post_pos)
            else np.nan
        )

        odds_post_neg = odds_pre * lr_neg if not np.isnan(lr_neg) else np.nan
        p_post_neg = (
            (odds_post_neg / (1.0 + odds_post_neg)) * 100.0
            if not np.isnan(odds_post_neg)
            else np.nan
        )

        metrics_df = pd.DataFrame(
            [
                {
                    "Metric": "Sensitivity (True Positive Rate)",
                    "Value": f"{sens:.1%}",
                    "Description": "Ability to correctly identify disease",
                },
                {
                    "Metric": "Specificity (True Negative Rate)",
                    "Value": f"{spec:.1%}",
                    "Description": "Ability to correctly identify healthy",
                },
                {
                    "Metric": "Positive Predictive Value (PPV)",
                    "Value": f"{ppv:.1%}",
                    "Description": "Probability of disease given positive test",
                },
                {
                    "Metric": "Negative Predictive Value (NPV)",
                    "Value": f"{npv:.1%}",
                    "Description": "Probability of healthy given negative test",
                },
                {
                    "Metric": "Positive Likelihood Ratio (LR+)",
                    "Value": f"{lr_pos:.2f}" if not np.isnan(lr_pos) else "N/A",
                    "Description": "Multiplier of odds with positive result (>10 is strong)",
                },
                {
                    "Metric": "Negative Likelihood Ratio (LR-)",
                    "Value": f"{lr_neg:.2f}" if not np.isnan(lr_neg) else "N/A",
                    "Description": "Multiplier of odds with negative result (<0.1 is strong)",
                },
                {
                    "Metric": "Diagnostic Odds Ratio (DOR)",
                    "Value": f"{dor:.2f}" if not np.isnan(dor) else "N/A",
                    "Description": "Overall effectiveness of test",
                },
            ]
        )

        summary_html = f"""
        <div style='background:#ffffff;border:1px solid #0284c7;border-radius:10px;padding:16px;'>
            <h4 style='color:#0369a1;margin-top:0;'>🔬 Diagnostic Performance & Fagan Bayesian Update</h4>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0;'>
                <div style='background:#f0fdf4;padding:12px;border-radius:8px;border:1px solid #bbf7d0;'>
                    <div style='font-size:0.85rem;color:#166534;'>Post-Test Probability (Positive Result):</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#14532d;'>{p_post_pos:.1f}%</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Pre-test: {pre_test_prob_pct:.1f}% → Post-test: {p_post_pos:.1f}%</div>
                </div>
                <div style='background:#fef2f2;padding:12px;border-radius:8px;border:1px solid #fecaca;'>
                    <div style='font-size:0.85rem;color:#991b1b;'>Post-Test Probability (Negative Result):</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#7f1d1d;'>{p_post_neg:.1f}%</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Pre-test: {pre_test_prob_pct:.1f}% → Post-test: {p_post_neg:.1f}%</div>
                </div>
            </div>
        </div>
        """

        # Fagan 3-axis plot representation
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2],
                y=[pre_test_prob_pct, lr_pos, p_post_pos],
                mode="lines+markers+text",
                text=[
                    f"Pre: {pre_test_prob_pct:.0f}%",
                    f"LR+: {lr_pos:.1f}",
                    f"Post: {p_post_pos:.1f}%",
                ],
                textposition="top center",
                line=dict(color="#059669", width=3),
                name="Positive Test Path",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2],
                y=[pre_test_prob_pct, lr_neg, p_post_neg],
                mode="lines+markers+text",
                text=[
                    f"Pre: {pre_test_prob_pct:.0f}%",
                    f"LR-: {lr_neg:.2f}",
                    f"Post: {p_post_neg:.1f}%",
                ],
                textposition="bottom center",
                line=dict(color="#dc2626", width=3, dash="dash"),
                name="Negative Test Path",
            )
        )
        fig.update_layout(
            title="Bayesian Updating Trajectory (Pre-test to Post-test Probability)",
            xaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=[
                    "Pre-Test Probability (%)",
                    "Likelihood Ratio",
                    "Post-Test Probability (%)",
                ],
            ),
            yaxis_title="Probability (%) / Ratio",
        )

        return metrics_df, summary_html, fig
    except Exception as e:
        return (
            pd.DataFrame(),
            f"<div style='color:#b91c1c;'>Error: {html.escape(str(e))}</div>",
            go.Figure(),
        )


def create_diagnostic_view(
    app_state: gr.State,
) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for Diagnostic Accuracy and Fagan Nomograms.
    """
    with gr.Tab("🔬 Diagnostic & Reliability", id="tab_diagnostic") as tab:
        gr.Markdown(
            """
            ### 🔬 Diagnostic Test Accuracy, Likelihood Ratios & Fagan Nomogram
            *Evaluate 2x2 clinical diagnostic performance, calculate likelihood ratios (LR+/LR-), and compute Bayesian post-test probabilities.*
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("##### 2x2 Contingency Table (Counts)")
                with gr.Row():
                    tp_in = gr.Number(
                        label="True Positives (TP):", value=85, precision=0
                    )
                    fp_in = gr.Number(
                        label="False Positives (FP):", value=15, precision=0
                    )
                with gr.Row():
                    fn_in = gr.Number(
                        label="False Negatives (FN):", value=15, precision=0
                    )
                    tn_in = gr.Number(
                        label="True Negatives (TN):", value=185, precision=0
                    )

                pre_test_slider = gr.Slider(
                    label="Pre-Test Clinical Probability (%):",
                    minimum=1.0,
                    maximum=99.0,
                    value=25.0,
                    step=1.0,
                )
                btn_calc_diag = gr.Button(
                    "🚀 Calculate Diagnostic Metrics", variant="primary"
                )

            with gr.Column(scale=8):
                summary_diag_html = gr.HTML("")
                table_diag = gr.Dataframe(
                    label="Diagnostic Metrics Summary", interactive=False
                )
                plot_fagan = gr.Plot(label="Bayesian Post-Test Updating Trajectory")

        btn_calc_diag.click(
            fn=calculate_2x2_diagnostic,
            inputs=[tp_in, fp_in, fn_in, tn_in, pre_test_slider],
            outputs=[table_diag, summary_diag_html, plot_fagan],
        )

    return tab, {
        "tp_in": tp_in,
        "fp_in": fp_in,
        "fn_in": fn_in,
        "tn_in": tn_in,
        "btn_calc_diag": btn_calc_diag,
    }
