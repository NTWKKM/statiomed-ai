"""
views/view_sample_size.py - StatioMed AI Sample Size & Power View (Gradio Native)
=================================================================================
Exact closed-form sample size calculation for Two Means (T-test), Two Proportions
(Fleiss formula), and Survival (Schoenfeld formula) with drop-out buffers and
SAMPL-compliant justification generators.
=================================================================================
"""

from __future__ import annotations

import html

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from utils import sample_size_lib


def compute_means_sample_size(
    m1: float,
    m2: float,
    sd1: float,
    sd2: float,
    power: float,
    alpha: float,
    ratio: float,
    dropout: float,
) -> tuple[str, go.Figure]:
    """Calculates sample size for two independent means."""
    try:
        res = sample_size_lib.calculate_sample_size_means(
            power=power, ratio=ratio, mean1=m1, mean2=m2, sd1=sd1, sd2=sd2, alpha=alpha
        )
        if "error" in res:
            return (
                f"<div style='color:#b91c1c;'>Error: {res['error']}</div>",
                go.Figure(),
            )

        n1 = int(np.ceil(res["n1"]))
        n2 = int(np.ceil(res["n2"]))
        total_raw = n1 + n2
        total_adj = int(np.ceil(total_raw / (1.0 - (dropout / 100.0))))

        diff = abs(m1 - m2)
        sd_pool = np.sqrt((sd1**2 + sd2**2) / 2)
        cohen_d = diff / sd_pool if sd_pool > 0 else 0

        # Power curve
        n_range = np.linspace(max(10, total_raw // 3), total_raw * 2, 30)
        power_vals = [
            sample_size_lib.calculate_power_means(
                n1=n, n2=n * ratio, mean1=m1, mean2=m2, sd1=sd1, sd2=sd2, alpha=alpha
            )
            for n in n_range
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=n_range * (1 + ratio),
                y=power_vals,
                mode="lines+markers",
                line=dict(color="#0284c7", width=3),
                name="Power Curve",
            )
        )
        fig.add_hline(
            y=power,
            line=dict(dash="dash", color="#dc2626"),
            annotation_text=f"Target Power ({power:.0%})",
        )
        fig.add_vline(
            x=total_raw,
            line=dict(dash="dot", color="#059669"),
            annotation_text=f"N = {total_raw}",
        )
        fig.update_layout(
            title="Power Curve (Two Means)",
            xaxis_title="Total Sample Size (N)",
            yaxis_title="Statistical Power (1 - β)",
            yaxis=dict(range=[0, 1.05]),
        )

        result_html = f"""
        <div style='background:#ffffff;border:1px solid #0284c7;border-radius:12px;padding:18px;'>
            <h4 style='color:#0369a1;margin-top:0;'>📊 Two Independent Means Sample Size</h4>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0;'>
                <div style='background:#f0f9ff;padding:12px;border-radius:8px;border:1px solid #bae6fd;'>
                    <div style='font-size:0.85rem;color:#0369a1;'>Group 1 (Control):</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#0c4a6e;'>{n1} subjects</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Mean: {m1:.2f} (SD: {sd1:.2f})</div>
                </div>
                <div style='background:#f0f9ff;padding:12px;border-radius:8px;border:1px solid #bae6fd;'>
                    <div style='font-size:0.85rem;color:#0369a1;'>Group 2 (Intervention):</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#0c4a6e;'>{n2} subjects</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Mean: {m2:.2f} (SD: {sd2:.2f})</div>
                </div>
            </div>
            <div style='background:#ecfdf5;border:1px solid #a7f3d0;padding:14px;border-radius:8px;margin-bottom:12px;'>
                <strong style='color:#065f46;font-size:1.05rem;'>Total Target Sample Size: {total_adj} patients</strong>
                <div style='color:#047857;font-size:0.85rem;margin-top:4px;'>Includes +{dropout:.0f}% drop-out buffer (Raw N: {total_raw}, Effect Size Cohen's d: {cohen_d:.3f}, Alpha: {alpha:.3f}, Power: {power:.0%}).</div>
            </div>
            <div style='background:#f8fafc;padding:12px;border-radius:8px;border-left:4px solid #0284c7;font-size:0.88rem;color:#334155;'>
                <strong>SAMPL Justification:</strong> "A total sample size of {total_adj} participants ({n1} in Group 1 and {n2} in Group 2, adjusted for a {dropout:.0f}% loss to follow-up) achieves {power:.0%} power to detect a true difference of {diff:.2f} units between group means with a significance level (alpha) of {alpha:.3f} using a two-sided two-sample t-test."
            </div>
        </div>
        """
        return result_html, fig
    except Exception as e:
        return (
            f"<div style='color:#b91c1c;'>Error: {html.escape(str(e))}</div>",
            go.Figure(),
        )


def compute_proportions_sample_size(
    p1: float, p2: float, power: float, alpha: float, ratio: float, dropout: float
) -> tuple[str, go.Figure]:
    """Calculates sample size for two independent proportions."""
    try:
        res = sample_size_lib.calculate_sample_size_proportions(
            power=power, ratio=ratio, p1=p1, p2=p2, alpha=alpha
        )
        if "error" in res:
            return (
                f"<div style='color:#b91c1c;'>Error: {res['error']}</div>",
                go.Figure(),
            )

        n1 = int(np.ceil(res["n1"]))
        n2 = int(np.ceil(res["n2"]))
        total_raw = n1 + n2
        total_adj = int(np.ceil(total_raw / (1.0 - (dropout / 100.0))))

        n_range = np.linspace(max(10, total_raw // 3), total_raw * 2, 30)
        power_vals = [
            sample_size_lib.calculate_power_proportions(
                n1=n, n2=n * ratio, p1=p1, p2=p2, alpha=alpha
            )
            for n in n_range
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=n_range * (1 + ratio),
                y=power_vals,
                mode="lines+markers",
                line=dict(color="#059669", width=3),
                name="Power Curve",
            )
        )
        fig.add_hline(
            y=power,
            line=dict(dash="dash", color="#dc2626"),
            annotation_text=f"Target Power ({power:.0%})",
        )
        fig.add_vline(
            x=total_raw,
            line=dict(dash="dot", color="#0284c7"),
            annotation_text=f"N = {total_raw}",
        )
        fig.update_layout(
            title="Power Curve (Two Proportions)",
            xaxis_title="Total Sample Size (N)",
            yaxis_title="Statistical Power (1 - β)",
            yaxis=dict(range=[0, 1.05]),
        )

        result_html = f"""
        <div style='background:#ffffff;border:1px solid #059669;border-radius:12px;padding:18px;'>
            <h4 style='color:#065f46;margin-top:0;'>🥧 Two Independent Proportions Sample Size</h4>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0;'>
                <div style='background:#f0fdf4;padding:12px;border-radius:8px;border:1px solid #bbf7d0;'>
                    <div style='font-size:0.85rem;color:#065f46;'>Group 1 Event Rate ($p_1$):</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#14532d;'>{n1} subjects</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Rate: {p1:.1%}</div>
                </div>
                <div style='background:#f0fdf4;padding:12px;border-radius:8px;border:1px solid #bbf7d0;'>
                    <div style='font-size:0.85rem;color:#065f46;'>Group 2 Event Rate ($p_2$):</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#14532d;'>{n2} subjects</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Rate: {p2:.1%}</div>
                </div>
            </div>
            <div style='background:#ecfdf5;border:1px solid #a7f3d0;padding:14px;border-radius:8px;margin-bottom:12px;'>
                <strong style='color:#065f46;font-size:1.05rem;'>Total Target Sample Size: {total_adj} patients</strong>
                <div style='color:#047857;font-size:0.85rem;margin-top:4px;'>Includes +{dropout:.0f}% drop-out adjustment (Raw N: {total_raw}, Absolute Risk Difference: {abs(p1 - p2):.1%}).</div>
            </div>
            <div style='background:#f8fafc;padding:12px;border-radius:8px;border-left:4px solid #059669;font-size:0.88rem;color:#334155;'>
                <strong>SAMPL Justification:</strong> "A target enrollment of {total_adj} patients ({n1} in Group 1 and {n2} in Group 2, assuming a {dropout:.0f}% loss to follow-up) will provide {power:.0%} power to detect a reduction from {p1:.1%} to {p2:.1%} (absolute difference {abs(p1 - p2):.1%}) with a two-sided alpha of {alpha:.3f} using Chi-Square testing."
            </div>
        </div>
        """
        return result_html, fig
    except Exception as e:
        return (
            f"<div style='color:#b91c1c;'>Error: {html.escape(str(e))}</div>",
            go.Figure(),
        )


def compute_survival_sample_size(
    hr: float, p_event: float, power: float, alpha: float, ratio: float, dropout: float
) -> tuple[str, go.Figure]:
    """Calculates Schoenfeld survival sample size."""
    try:
        res = sample_size_lib.calculate_sample_size_survival(
            power=power, ratio=ratio, h0=hr, h1=0.0, alpha=alpha, mode="hr"
        )
        if "error" in res:
            return (
                f"<div style='color:#b91c1c;'>Error: {res['error']}</div>",
                go.Figure(),
            )

        events_req = int(
            np.ceil(res.get("events_required", res.get("total_events", 100)))
        )
        total_raw = int(np.ceil(res.get("total_n", events_req / p_event)))
        total_adj = int(np.ceil(total_raw / (1.0 - (dropout / 100.0))))

        fig = go.Figure()
        hr_range = np.linspace(max(0.2, hr - 0.3), min(0.95, hr + 0.3), 30)
        events_curve = [
            sample_size_lib.calculate_sample_size_survival(
                power=power, ratio=ratio, h0=h, h1=0.0, alpha=alpha, mode="hr"
            ).get("total_events", 100)
            for h in hr_range
        ]
        fig.add_trace(
            go.Scatter(
                x=hr_range,
                y=events_curve,
                mode="lines+markers",
                line=dict(color="#7c3aed", width=3),
                name="Required Events",
            )
        )
        fig.add_vline(
            x=hr,
            line=dict(dash="dash", color="#dc2626"),
            annotation_text=f"Target HR = {hr}",
        )
        fig.update_layout(
            title="Schoenfeld Event Count vs. Hazard Ratio",
            xaxis_title="Target Hazard Ratio (HR)",
            yaxis_title="Required Event Count (E)",
        )

        result_html = f"""
        <div style='background:#ffffff;border:1px solid #7c3aed;border-radius:12px;padding:18px;'>
            <h4 style='color:#6d28d9;margin-top:0;'>⏱️ Schoenfeld Survival & Log-Rank Sample Size</h4>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0;'>
                <div style='background:#f5f3ff;padding:12px;border-radius:8px;border:1px solid #ddd6fe;'>
                    <div style='font-size:0.85rem;color:#6d28d9;'>Required Event Count (E):</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#4c1d95;'>{events_req} events</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Cumulative Event Rate: {p_event:.1%}</div>
                </div>
                <div style='background:#f5f3ff;padding:12px;border-radius:8px;border:1px solid #ddd6fe;'>
                    <div style='font-size:0.85rem;color:#6d28d9;'>Target Hazard Ratio:</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#4c1d95;'>HR = {hr:.2f}</div>
                    <div style='font-size:0.8rem;color:#64748b;'>Allocation Ratio: 1:{ratio:.1f}</div>
                </div>
            </div>
            <div style='background:#faf5ff;border:1px solid #e9d5ff;padding:14px;border-radius:8px;margin-bottom:12px;'>
                <strong style='color:#5b21b6;font-size:1.05rem;'>Total Target Enrollment: {total_adj} patients ({total_raw} raw)</strong>
                <div style='color:#6b21a8;font-size:0.85rem;margin-top:4px;'>Calculated using Schoenfeld formula with +{dropout:.0f}% drop-out allowance for Log-rank survival test.</div>
            </div>
            <div style='background:#f8fafc;padding:12px;border-radius:8px;border-left:4px solid #7c3aed;font-size:0.88rem;color:#334155;'>
                <strong>SAMPL Justification:</strong> "Assuming a cumulative event rate of {p_event:.1%} across the trial duration and a true hazard ratio of {hr:.2f}, observing {events_req} events provides {power:.0%} power at a two-sided alpha of {alpha:.3f}. Accounting for a {dropout:.0f}% withdrawal rate, a total of {total_adj} patients must be enrolled."
            </div>
        </div>
        """
        return result_html, fig
    except Exception as e:
        return (
            f"<div style='color:#b91c1c;'>Error: {html.escape(str(e))}</div>",
            go.Figure(),
        )


def create_sample_size_view(
    app_state: gr.State,
) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for Sample Size & Power Calculation.
    """
    with gr.Tab("📐 Sample Size", id="tab_sample_size") as tab:
        gr.Markdown(
            """
            ### 📐 Biostatistical Sample Size & Power Engine
            *Exact closed-form formulas for Two Means, Two Proportions, and Schoenfeld Survival Log-Rank Tests.*
            """
        )

        with gr.Tabs():
            # 1. MEANS
            with gr.Tab("📊 Two Means (T-Test)"):
                with gr.Row():
                    with gr.Column(scale=4):
                        m_m1 = gr.Number(label="Mean Group 1 (Control):", value=120.0)
                        m_sd1 = gr.Number(label="SD Group 1:", value=15.0)
                        m_m2 = gr.Number(
                            label="Mean Group 2 (Intervention):", value=112.0
                        )
                        m_sd2 = gr.Number(label="SD Group 2:", value=15.0)
                        m_power = gr.Slider(
                            label="Power (1 - β):",
                            minimum=0.5,
                            maximum=0.99,
                            value=0.80,
                            step=0.01,
                        )
                        m_alpha = gr.Slider(
                            label="Significance Level (α):",
                            minimum=0.001,
                            maximum=0.20,
                            value=0.05,
                            step=0.005,
                        )
                        m_dropout = gr.Slider(
                            label="Anticipated Loss / Drop-out (%):",
                            minimum=0,
                            maximum=40,
                            value=15,
                            step=5,
                        )
                        btn_calc_means = gr.Button(
                            "🚀 Calculate Means Sample Size", variant="primary"
                        )

                    with gr.Column(scale=8):
                        means_result_html = gr.HTML("")
                        means_plot = gr.Plot(label="Power Curve")

                btn_calc_means.click(
                    fn=compute_means_sample_size,
                    inputs=[
                        m_m1,
                        m_m2,
                        m_sd1,
                        m_sd2,
                        m_power,
                        m_alpha,
                        gr.State(1.0),
                        m_dropout,
                    ],
                    outputs=[means_result_html, means_plot],
                )

            # 2. PROPORTIONS
            with gr.Tab("🥧 Two Proportions (Chi-Square)"):
                with gr.Row():
                    with gr.Column(scale=4):
                        p_p1 = gr.Slider(
                            label="Proportion Group 1 (p1):",
                            minimum=0.01,
                            maximum=0.99,
                            value=0.35,
                            step=0.01,
                        )
                        p_p2 = gr.Slider(
                            label="Proportion Group 2 (p2):",
                            minimum=0.01,
                            maximum=0.99,
                            value=0.20,
                            step=0.01,
                        )
                        p_power = gr.Slider(
                            label="Power (1 - β):",
                            minimum=0.5,
                            maximum=0.99,
                            value=0.80,
                            step=0.01,
                        )
                        p_alpha = gr.Slider(
                            label="Significance Level (α):",
                            minimum=0.001,
                            maximum=0.20,
                            value=0.05,
                            step=0.005,
                        )
                        p_dropout = gr.Slider(
                            label="Anticipated Loss / Drop-out (%):",
                            minimum=0,
                            maximum=40,
                            value=15,
                            step=5,
                        )
                        btn_calc_prop = gr.Button(
                            "🚀 Calculate Proportions Sample Size", variant="primary"
                        )

                    with gr.Column(scale=8):
                        prop_result_html = gr.HTML("")
                        prop_plot = gr.Plot(label="Power Curve")

                btn_calc_prop.click(
                    fn=compute_proportions_sample_size,
                    inputs=[p_p1, p_p2, p_power, p_alpha, gr.State(1.0), p_dropout],
                    outputs=[prop_result_html, prop_plot],
                )

            # 3. SURVIVAL
            with gr.Tab("⏱️ Survival (Schoenfeld)"):
                with gr.Row():
                    with gr.Column(scale=4):
                        s_hr = gr.Slider(
                            label="Target Hazard Ratio (HR):",
                            minimum=0.1,
                            maximum=0.95,
                            value=0.65,
                            step=0.05,
                        )
                        s_pevent = gr.Slider(
                            label="Cumulative Event Rate in Study:",
                            minimum=0.05,
                            maximum=0.90,
                            value=0.30,
                            step=0.05,
                        )
                        s_power = gr.Slider(
                            label="Power (1 - β):",
                            minimum=0.5,
                            maximum=0.99,
                            value=0.80,
                            step=0.01,
                        )
                        s_alpha = gr.Slider(
                            label="Significance Level (α):",
                            minimum=0.001,
                            maximum=0.20,
                            value=0.05,
                            step=0.005,
                        )
                        s_dropout = gr.Slider(
                            label="Anticipated Loss / Drop-out (%):",
                            minimum=0,
                            maximum=40,
                            value=15,
                            step=5,
                        )
                        btn_calc_surv = gr.Button(
                            "🚀 Calculate Survival Sample Size", variant="primary"
                        )

                    with gr.Column(scale=8):
                        surv_result_html = gr.HTML("")
                        surv_plot = gr.Plot(label="Events Curve")

                btn_calc_surv.click(
                    fn=compute_survival_sample_size,
                    inputs=[s_hr, s_pevent, s_power, s_alpha, gr.State(1.0), s_dropout],
                    outputs=[surv_result_html, surv_plot],
                )

    return tab, {
        "btn_calc_means": btn_calc_means,
        "btn_calc_prop": btn_calc_prop,
        "btn_calc_surv": btn_calc_surv,
    }
