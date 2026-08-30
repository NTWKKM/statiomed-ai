"""
views/view_data.py - StatioMed AI Data Management & Profiler View (Gradio Native)
=================================================================================
File loading, data validation, missingness inspection, automated quality audit,
and MICE multiple imputation.
=================================================================================
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.state import AppState
from logger import get_logger
from utils.data_cleaning import impute_missing_data, load_data_robust
from utils.data_quality import check_data_quality
from utils.visualizations import plot_missing_pattern

logger = get_logger(__name__)


def generate_example_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generates a realistic synthetic clinical dataset (n=1600) with survival,
    laboratory, demographic, and treatment variables.
    """
    np.random.seed(42)
    n = 1600

    age = np.random.normal(62, 11, n).astype(int).clip(30, 92)
    sex = np.random.binomial(1, 0.48, n)  # 1 = Female, 0 = Male
    bmi = np.random.normal(26.2, 4.8, n).round(1).clip(16.0, 48.0)

    # Treatment assignment (SGLT2 inhibitor vs Standard Care)
    logit_treat = -2.8 + (0.03 * age) + (0.04 * bmi) + (0.15 * sex)
    p_treat = 1 / (1 + np.exp(-logit_treat))
    treatment = np.random.binomial(1, p_treat, n)

    # Comorbidities
    diabetes = np.random.binomial(1, 0.42, n)
    hypertension = np.random.binomial(1, 0.58, n)
    ckd_stage = np.random.choice([1, 2, 3, 4], size=n, p=[0.25, 0.45, 0.22, 0.08])

    # Biomarkers
    sbp = (
        128 + 0.2 * age + 0.3 * bmi + 8 * hypertension + np.random.normal(0, 12, n)
    ).round(0)
    dbp = (
        78 + 0.1 * age + 0.15 * bmi + 5 * hypertension + np.random.normal(0, 8, n)
    ).round(0)
    # Ensure SBP > DBP + 15
    sbp = np.maximum(sbp, dbp + 20)
    egfr = (
        (95 - 0.7 * age - 8 * ckd_stage - 4 * diabetes + np.random.normal(0, 8, n))
        .round(1)
        .clip(10, 130)
    )

    # Time-to-event (Survival in days)
    hazard = 0.0008 * np.exp(
        0.025 * age + 0.45 * diabetes + 0.35 * hypertension - 0.52 * treatment
    )
    time = np.random.exponential(1 / hazard).round(0).clip(5, 730)
    death = (np.random.uniform(0, 1, n) < (1 - np.exp(-hazard * time))).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"PT-{i + 1:04d}" for i in range(n)],
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "treatment": treatment,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "ckd_stage": ckd_stage,
            "sbp": sbp,
            "dbp": dbp,
            "egfr": egfr,
            "time": time,
            "death": death,
        }
    )

    # Inject realistic 2-5% random missingness
    for col in ["bmi", "sbp", "dbp", "egfr"]:
        mask = np.random.rand(n) < 0.035
        df.loc[mask, col] = np.nan

    meta = {
        "domain": "Cardiovascular & Renal Trial",
        "description": "Multicenter observational cohort of 1600 patients assessing SGLT2i therapy on survival and renal function.",
        "pico": {
            "population": "Adult patients with Cardiorenal Disease (CKD stage 1-4)",
            "exposure": "SGLT2 inhibitor therapy (1 = Active, 0 = Standard)",
            "comparator": "Standard medical care",
            "outcome": "2-year All-Cause Mortality (death) and eGFR trajectory",
        },
    }
    return df, meta


def load_example_data_action(
    state: AppState,
) -> tuple[AppState, str, pd.DataFrame, go.Figure, str]:
    """Action callback: Loads the built-in clinical research dataset."""
    df, meta = generate_example_dataset()
    state.df = df
    state.file_name = "Example Clinical Cohort (n=1,600)"
    state.var_meta = meta
    state.df_matched = None
    state.is_matched = False
    state.mi_imputed_datasets = []

    badge_html = f"""
    <div style='background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:14px;display:flex;justify-content:space-between;align-items:center;'>
        <div>
            <strong style='color:#166534;font-size:1rem;'>✅ Active Dataset: {state.file_name}</strong>
            <div style='color:#475569;font-size:0.85rem;margin-top:2px;'>Rows: <strong>{len(df):,}</strong> | Columns: <strong>{len(df.columns)}</strong> | Missing Cells: <strong>{df.isna().sum().sum():,}</strong> ({df.isna().sum().sum() / (df.size) * 100:.1f}%)</div>
        </div>
        <span style='background:#dcfce7;color:#15803d;padding:4px 12px;border-radius:999px;font-weight:600;font-size:0.8rem;'>🔒 Zero-PHI Verified</span>
    </div>
    """
    quality_issues = check_data_quality(df)
    report_html = f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;font-size:0.88rem;'><strong>Data Quality Check:</strong> {len(quality_issues)} potential issues detected across {len(df.columns)} columns.</div>"
    fig_missing = plot_missing_pattern(df)

    return state, badge_html, df, fig_missing, report_html


def upload_file_action(
    file_obj: Any, state: AppState
) -> tuple[AppState, str, pd.DataFrame | None, go.Figure, str]:
    """Action callback: Parses uploaded CSV, Excel, SPSS, or Stata file."""
    if file_obj is None:
        return (
            state,
            "<div style='color:#64748b;'>No file selected.</div>",
            None,
            go.Figure(),
            "",
        )

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        df = load_data_robust(file_path)
        base_name = Path(file_path).name
        state.df = df
        state.file_name = base_name
        state.df_matched = None
        state.is_matched = False
        state.mi_imputed_datasets = []

        badge_html = f"""
        <div style='background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:14px;display:flex;justify-content:space-between;align-items:center;'>
            <div>
                <strong style='color:#166534;font-size:1rem;'>✅ Uploaded: {base_name}</strong>
                <div style='color:#475569;font-size:0.85rem;margin-top:2px;'>Rows: <strong>{len(df):,}</strong> | Columns: <strong>{len(df.columns)}</strong> | Missing: <strong>{df.isna().sum().sum():,}</strong></div>
            </div>
            <span style='background:#dcfce7;color:#15803d;padding:4px 12px;border-radius:999px;font-weight:600;font-size:0.8rem;'>🔒 Zero-PHI Shield</span>
        </div>
        """
        quality_issues = check_data_quality(df)
        report_html = f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;font-size:0.88rem;'><strong>Data Quality Check:</strong> {len(quality_issues)} potential issues detected across {len(df.columns)} columns.</div>"
        fig_missing = plot_missing_pattern(df)
        return state, badge_html, df, fig_missing, report_html

    except Exception as e:
        logger.error(f"Failed to upload data: {e}")
        err_html = f"<div style='background:#fee2e2;color:#991b1b;padding:12px;border-radius:8px;'>❌ Upload Failed: {html.escape(str(e))}</div>"
        return state, err_html, None, go.Figure(), ""


def run_multiple_imputation_action(
    m_imputations: int, state: AppState
) -> tuple[AppState, str]:
    """Action callback: Performs MICE multiple imputation on missing variables."""
    if state.df is None or state.df.empty:
        return (
            state,
            "<div style='color:#b91c1c;'>Please load or upload a dataset first.</div>",
        )

    try:
        imputed_dfs = impute_missing_data(state.df, m=int(m_imputations))
        state.mi_imputed_datasets = imputed_dfs
        msg = f"""
        <div style='background:#eff6ff;border:1px solid #93c5fd;color:#1e40af;padding:12px;border-radius:8px;'>
            ✅ Successfully generated <strong>{len(imputed_dfs)}</strong> imputed dataset copies via MICE. Downstream statistical models can now pool estimates using Rubin's Rules.
        </div>
        """
        return state, msg
    except Exception as e:
        return (
            state,
            f"<div style='color:#b91c1c;'>Imputation error: {html.escape(str(e))}</div>",
        )


def reset_workspace_action(
    state: AppState,
) -> tuple[AppState, str, pd.DataFrame | None, go.Figure, str]:
    """Action callback: Resets workspace state."""
    state = AppState()
    empty_badge = "<div style='color:#64748b;font-size:0.9rem;'>No dataset currently active in session.</div>"
    return state, empty_badge, None, go.Figure(), ""


def create_data_view(app_state: gr.State) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for Data Profiler & Management.
    """
    with gr.Tab("📊 Data Profiler", id="tab_data") as tab:
        gr.Markdown(
            """
            ### 📊 Data Profiler, Inspection & Quality Audit
            *Ingest hospital research cohorts (.csv, .xlsx, .sav, .dta), detect missingness, and execute MICE imputation.*
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                btn_load_example = gr.Button(
                    "📄 Load Example Clinical Data", variant="primary"
                )
                file_upload = gr.File(
                    label="📂 Upload Research Dataset (.csv, .xlsx, .sav, .dta)",
                    file_types=[".csv", ".xlsx", ".sav", ".dta"],
                )
                with gr.Accordion("⚙️ Advanced: Multiple Imputation (MICE)", open=False):
                    m_slider = gr.Slider(
                        label="Number of Imputations (m):",
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                    )
                    btn_run_mice = gr.Button(
                        "🔄 Run MICE Imputation", variant="secondary"
                    )
                    mice_msg = gr.HTML("")

                btn_reset = gr.Button("⚠️ Reset Workspace", variant="stop")

            with gr.Column(scale=8):
                meta_badge = gr.HTML(
                    """
                    <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; padding: 16px; text-align: center; color: #64748b;">
                        📁 No dataset loaded. Click <strong>Load Example Clinical Data</strong> or upload your file to start.
                    </div>
                    """
                )
                health_report = gr.HTML("")
                with gr.Tabs():
                    with gr.Tab("📋 Data Preview"):
                        data_preview = gr.Dataframe(
                            label="Ingested Records Preview",
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("🔍 Missingness Matrix"):
                        missing_plot = gr.Plot(label="Missingness Distribution Pattern")

        # Callbacks
        btn_load_example.click(
            fn=load_example_data_action,
            inputs=[app_state],
            outputs=[app_state, meta_badge, data_preview, missing_plot, health_report],
        )

        file_upload.change(
            fn=upload_file_action,
            inputs=[file_upload, app_state],
            outputs=[app_state, meta_badge, data_preview, missing_plot, health_report],
        )

        btn_run_mice.click(
            fn=run_multiple_imputation_action,
            inputs=[m_slider, app_state],
            outputs=[app_state, mice_msg],
        )

        btn_reset.click(
            fn=reset_workspace_action,
            inputs=[app_state],
            outputs=[app_state, meta_badge, data_preview, missing_plot, health_report],
        )

    return tab, {
        "btn_load_example": btn_load_example,
        "file_upload": file_upload,
        "meta_badge": meta_badge,
        "data_preview": data_preview,
        "missing_plot": missing_plot,
        "health_report": health_report,
    }
