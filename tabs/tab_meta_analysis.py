"""
📚 Systematic Review & Meta-Analysis Tab (PRISMA 2020 Compliant)

Provides:
- Binary Outcomes (2x2: OR, RR, RD)
- Continuous Outcomes (Mean/SD/N: MD, SMD / Hedges' g)
- Generic Effect Sizes (Log-Effect / SE)
- Fixed & Random-Effects Models (DerSimonian-Laird, REML, Paule-Mandel)
- Hartung-Knapp-Sidik-Jonkman (HKSJ) Adjustment
- 95% Prediction Interval for future trials
- Heterogeneity (Q, I², τ², τ) & Subgroup Difference Test (Q_between)
- Publication Bias: Egger's Test, Begg's Rank Test, Contour-Enhanced Funnel Plot
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, req, ui

from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from tabs._dataset_mixin import register_dataset_selector
from utils import meta_analysis_lib
from utils.download_helpers import safe_download_html
from utils.pdf_helpers import safe_download_pdf
from utils.ui_helpers import (
    create_placeholder_state,
    create_results_container,
)

logger = get_logger(__name__)
COLORS = get_color_palette()


# ==============================================================================
# EXAMPLE DATASETS
# ==============================================================================


def get_bcg_vaccine_data() -> pd.DataFrame:
    """Colditz et al. (1994) BCG Vaccine Tuberculosis Prevention RCTs."""
    return pd.DataFrame(
        {
            "Study": [
                "Aronson 1948",
                "Ferguson 1949",
                "Rosenthal 1945",
                "Hart 1977",
                "Frimodt-Moller 1973",
                "Stein 1953",
                "Vandiviere 1973",
                "TPT Madras 1980",
                "Coetzee 1968",
                "Rosenthal 1961",
                "Comstock 1974",
                "Comstock 1976",
                "BPT 1979",
            ],
            "Latitude_Group": [
                "High (>40°)",
                "High (>40°)",
                "High (>40°)",
                "High (>40°)",
                "Low (≤40°)",
                "High (>40°)",
                "Low (≤40°)",
                "Low (≤40°)",
                "Low (≤40°)",
                "High (>40°)",
                "Low (≤40°)",
                "Low (≤40°)",
                "Low (≤40°)",
            ],
            "TB_Vaccine": [4, 6, 11, 29, 179, 47, 186, 505, 29, 17, 8, 10, 44],
            "Total_Vaccine": [
                123,
                306,
                231,
                247,
                1697,
                230,
                2498,
                88391,
                7499,
                1716,
                2545,
                1665,
                8743,
            ],
            "TB_Control": [11, 29, 29, 45, 141, 34, 141, 499, 45, 65, 10, 11, 65],
            "Total_Control": [
                139,
                303,
                220,
                234,
                1606,
                231,
                2341,
                88391,
                7277,
                1665,
                629,
                2489,
                8788,
            ],
        }
    )


def get_statin_continuous_data() -> pd.DataFrame:
    """Illustrative Continuous Meta-Analysis: Statin vs Control LDL Reduction (mg/dL)."""
    return pd.DataFrame(
        {
            "Trial": [
                "Trial 1 (Low Dose)",
                "Trial 2 (Low Dose)",
                "Trial 3 (Med Dose)",
                "Trial 4 (Med Dose)",
                "Trial 5 (High Dose)",
                "Trial 6 (High Dose)",
            ],
            "Dose_Tier": ["Low", "Low", "Medium", "Medium", "High", "High"],
            "Mean_Statin": [-28.4, -31.2, -42.5, -39.8, -54.1, -58.3],
            "SD_Statin": [8.2, 7.9, 9.4, 8.8, 10.5, 11.2],
            "N_Statin": [120, 150, 200, 180, 250, 310],
            "Mean_Control": [-4.1, -3.8, -5.2, -4.9, -6.1, -5.8],
            "SD_Control": [7.5, 8.1, 8.6, 9.1, 9.8, 10.4],
            "N_Control": [120, 150, 200, 180, 250, 310],
        }
    )


def get_generic_ratio_data() -> pd.DataFrame:
    """Illustrative Generic Meta-Analysis: SGLT2 Inhibitors vs Placebo for All-Cause Mortality in Heart Failure (Hazard Ratios)."""
    return pd.DataFrame(
        {
            "Trial": [
                "DAPA-HF (2019)",
                "EMPEROR-Reduced (2020)",
                "EMPEROR-Preserved (2021)",
                "DELIVER (2022)",
                "SOLOIST-WHF (2020)",
            ],
            "Subgroup": [
                "HFrEF",
                "HFrEF",
                "HFpEF",
                "HFpEF",
                "Worsening HF",
            ],
            "Hazard_Ratio": [0.83, 0.92, 0.90, 0.94, 0.69],
            "Log_HR": [-0.1863, -0.0834, -0.1054, -0.0619, -0.3711],
            "SE_Log_HR": [0.0820, 0.0880, 0.0750, 0.0730, 0.1650],
        }
    )


def render_demo_table_html(
    df: pd.DataFrame, col_descriptions: dict[str, str] | None = None
) -> str:
    """Generate a high-contrast, cleanly styled Bootstrap HTML table for demo dataset preview."""
    header_th = []
    for col in df.columns:
        desc = col_descriptions.get(col, "") if col_descriptions else ""
        desc_badge = (
            f"<span class='badge bg-light text-dark ms-1 border font-monospace fw-normal'>{desc}</span>"
            if desc
            else ""
        )
        header_th.append(f"<th class='text-nowrap'>{col} {desc_badge}</th>")

    rows_html = []
    for _, row in df.iterrows():
        tds = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, (int, np.integer)):
                formatted_val = f"{val:,}"
                tds.append(f"<td class='text-end font-monospace'>{formatted_val}</td>")
            elif isinstance(val, (float, np.floating)):
                formatted_val = (
                    f"{val:.4f}"
                    if abs(val) < 1.0
                    else (f"{val:.2f}" if abs(val) < 100 else f"{val:.1f}")
                )
                tds.append(f"<td class='text-end font-monospace'>{formatted_val}</td>")
            else:
                tds.append(f"<td class='text-nowrap'>{val}</td>")
        rows_html.append(f"<tr>{''.join(tds)}</tr>")

    return f"""
    <div class="table-responsive border rounded shadow-sm bg-white" style="max-height: 380px; overflow-y: auto;">
        <table class="table table-sm table-hover table-striped table-bordered align-middle mb-0">
            <thead class="table-light sticky-top shadow-sm">
                <tr>{"".join(header_th)}</tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
    </div>
    """


def create_demo_data_modal() -> ui.Tag:
    """Constructs the comprehensive modal dialog showcasing raw demo data tables and upload formats."""
    bcg_df = get_bcg_vaccine_data()
    statin_df = get_statin_continuous_data()
    generic_df = get_generic_ratio_data()

    bcg_desc = {
        "Study": "Study ID / Author",
        "Latitude_Group": "Subgroup",
        "TB_Vaccine": "Events (Treatment)",
        "Total_Vaccine": "Total N (Treatment)",
        "TB_Control": "Events (Control)",
        "Total_Control": "Total N (Control)",
    }

    statin_desc = {
        "Trial": "Trial ID",
        "Dose_Tier": "Subgroup",
        "Mean_Statin": "Mean (Treatment)",
        "SD_Statin": "SD (Treatment)",
        "N_Statin": "N (Treatment)",
        "Mean_Control": "Mean (Control)",
        "SD_Control": "SD (Control)",
        "N_Control": "N (Control)",
    }

    generic_desc = {
        "Trial": "Trial ID",
        "Subgroup": "Subgroup",
        "Hazard_Ratio": "Natural Ratio (HR)",
        "Log_HR": "Log-Ratio (θ)",
        "SE_Log_HR": "Standard Error (SE)",
    }

    modal_content = ui.div(
        ui.div(
            ui.markdown("""
            💡 **Data Formatting & Upload Guide**:
            This reference window displays the exact table structure and column types required for uploading your own systematic review data.
            Select a data format below to inspect the raw data values, column descriptions, and download sample CSV templates.
            """),
            class_="alert alert-light border shadow-sm mb-3",
        ),
        ui.navset_tab(
            # TAB 1: Binary (2x2)
            ui.nav_panel(
                "📊 1. Binary Outcome (2x2 Table)",
                ui.div(
                    ui.div(
                        ui.tags.h6(
                            "📋 Required Column Format for Binary Endpoints (OR, RR, RD)",
                            class_="text-primary fw-bold mb-1",
                        ),
                        ui.tags.p(
                            "For RCTs or cohort studies reporting event counts and total participants in experimental vs control groups.",
                            class_="text-muted small mb-2",
                        ),
                        ui.tags.ul(
                            ui.tags.li(
                                ui.tags.strong("Study Identifier: "),
                                "Study name or author and publication year (e.g. Aronson 1948)",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Treatment Arm (2 columns): "),
                                "Events in Treatment (a) + Total Sample Size (n1)",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Control Arm (2 columns): "),
                                "Events in Control (c) + Total Sample Size (n0)",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Subgroup (Optional): "),
                                "Categorical column for subgroup heterogeneity testing (Q_between)",
                            ),
                            class_="small mb-0",
                        ),
                        class_="card p-3 mb-3 bg-light border-0 shadow-sm",
                    ),
                    ui.div(
                        ui.input_action_button(
                            "btn_modal_load_bcg",
                            "📊 Load BCG Vaccine Dataset",
                            class_="btn-sm btn-primary me-2",
                        ),
                        ui.download_button(
                            "btn_dl_demo_bcg_csv",
                            "📥 Download BCG CSV Template",
                            class_="btn-sm btn-outline-secondary",
                        ),
                        class_="d-flex align-items-center mb-2",
                    ),
                    ui.HTML(render_demo_table_html(bcg_df, bcg_desc)),
                    ui.div(
                        ui.span(
                            f"Displaying {len(bcg_df)} studies from Colditz et al. (1994) BCG Tuberculosis Prevention RCTs.",
                            class_="text-muted small",
                        ),
                        class_="mt-2",
                    ),
                    class_="py-2",
                ),
            ),
            # TAB 2: Continuous
            ui.nav_panel(
                "📈 2. Continuous Outcome (Mean / SD / N)",
                ui.div(
                    ui.div(
                        ui.tags.h6(
                            "📋 Required Column Format for Continuous Endpoints (MD, SMD)",
                            class_="text-primary fw-bold mb-1",
                        ),
                        ui.tags.p(
                            "For clinical trials comparing continuous biomarker values, blood pressure, or symptom scores.",
                            class_="text-muted small mb-2",
                        ),
                        ui.tags.ul(
                            ui.tags.li(
                                ui.tags.strong("Trial Identifier: "),
                                "Study name or trial name (e.g. Trial 1 (Low Dose))",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Treatment Arm (3 columns): "),
                                "Mean (x̄1) + Standard Deviation (s1) + Sample Size (n1)",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Control Arm (3 columns): "),
                                "Mean (x̄0) + Standard Deviation (s0) + Sample Size (n0)",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Subgroup (Optional): "),
                                "Categorical dosage tier or patient category",
                            ),
                            class_="small mb-0",
                        ),
                        class_="card p-3 mb-3 bg-light border-0 shadow-sm",
                    ),
                    ui.div(
                        ui.input_action_button(
                            "btn_modal_load_statin",
                            "📈 Load Statin Trials Dataset",
                            class_="btn-sm btn-primary me-2",
                        ),
                        ui.download_button(
                            "btn_dl_demo_statin_csv",
                            "📥 Download Statin CSV Template",
                            class_="btn-sm btn-outline-secondary",
                        ),
                        class_="d-flex align-items-center mb-2",
                    ),
                    ui.HTML(render_demo_table_html(statin_df, statin_desc)),
                    ui.div(
                        ui.span(
                            f"Displaying {len(statin_df)} trials from Statin vs Control LDL Reduction dataset (mg/dL).",
                            class_="text-muted small",
                        ),
                        class_="mt-2",
                    ),
                    class_="py-2",
                ),
            ),
            # TAB 3: Generic (Log-Ratio / Effect & SE)
            ui.nav_panel(
                "📑 3. Generic Effect Size (Log-Ratio / SE)",
                ui.div(
                    ui.div(
                        ui.tags.h6(
                            "📋 Required Column Format for Generic Effect Sizes & Hazard Ratios",
                            class_="text-primary fw-bold mb-1",
                        ),
                        ui.tags.p(
                            "For pre-calculated effect sizes from survival models (Hazard Ratios), pre-calculated Log-Ratios, or multivariable regression coefficients.",
                            class_="text-muted small mb-2",
                        ),
                        ui.tags.ul(
                            ui.tags.li(
                                ui.tags.strong("Study Identifier: "),
                                "Trial or study name (e.g. DAPA-HF (2019))",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Effect Size (θ or ln(HR)): "),
                                "Point estimate (must be on natural log scale if ratio metric)",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Standard Error (SE): "),
                                "Standard error of the estimate (SE = (ln(CI_upper) - ln(CI_lower)) / 3.92)",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Subgroup (Optional): "),
                                "Clinical phenotype stratification (e.g. HFrEF vs HFpEF)",
                            ),
                            class_="small mb-0",
                        ),
                        class_="card p-3 mb-3 bg-light border-0 shadow-sm",
                    ),
                    ui.div(
                        ui.input_action_button(
                            "btn_modal_load_generic",
                            "📑 Load SGLT2i HR Dataset",
                            class_="btn-sm btn-primary me-2",
                        ),
                        ui.download_button(
                            "btn_dl_demo_generic_csv",
                            "📥 Download SGLT2i CSV Template",
                            class_="btn-sm btn-outline-secondary",
                        ),
                        class_="d-flex align-items-center mb-2",
                    ),
                    ui.HTML(render_demo_table_html(generic_df, generic_desc)),
                    ui.div(
                        ui.span(
                            f"Displaying {len(generic_df)} landmark trials in SGLT2i Heart Failure meta-analysis dataset.",
                            class_="text-muted small",
                        ),
                        class_="mt-2",
                    ),
                    class_="py-2",
                ),
            ),
        ),
    )

    return ui.modal(
        modal_content,
        title="📋 Meta-Analysis Data Formats & Demo Raw Data Viewer",
        size="xl",
        easy_close=True,
        footer=ui.div(
            ui.modal_button("Close", class_="btn-secondary"),
            class_="d-flex justify-content-end",
        ),
    )


# ==============================================================================
# UI DEFINITION
# ==============================================================================


@module.ui
def meta_analysis_ui() -> ui.TagChild:
    """Constructs the Clinical Meta-Analysis UI."""
    return ui.div(
        ui.output_ui("ui_title_with_summary"),
        ui.output_ui("ui_matched_info"),
        ui.br(),
        ui.output_ui("ui_dataset_selector"),
        ui.br(),
        # Example Data Quick Loader
        ui.div(
            ui.div(
                ui.span(
                    "💡 Quick Demo Datasets: ", class_="fw-bold text-secondary me-2"
                ),
                ui.input_action_button(
                    "btn_load_bcg",
                    "📊 Load BCG Vaccine (Binary)",
                    class_="btn-sm btn-outline-primary me-2",
                ),
                ui.input_action_button(
                    "btn_load_statin",
                    "📈 Load Statin LDL (Continuous)",
                    class_="btn-sm btn-outline-secondary me-2",
                ),
                ui.input_action_button(
                    "btn_load_generic",
                    "📑 Load SGLT2i HR (Generic)",
                    class_="btn-sm btn-outline-success me-2",
                ),
                ui.input_action_button(
                    "btn_view_demo_data",
                    "👁️ View Demo Raw Data & Format Guide",
                    class_="btn-sm btn-outline-info me-2 fw-semibold",
                ),
                ui.input_action_button(
                    "btn_reset_demo",
                    "🔄 Exit Demo (Use Uploaded Data)",
                    class_="btn-sm btn-outline-danger",
                ),
                class_="d-flex flex-wrap align-items-center gap-1 mb-3",
            ),
        ),
        ui.output_ui("ui_demo_dataset_banner"),
        ui.navset_tab(
            # TAB 1: Forest Plot & Effect Summary
            ui.nav_panel(
                "🌲 Forest Plot & Pooling",
                ui.markdown(
                    "##### Clinical Meta-Analysis: Forest Plot & Summary Effect"
                ),
                ui.row(
                    ui.column(
                        3,
                        ui.input_select(
                            "meta_data_type",
                            "Data Format:",
                            {
                                "binary": "Binary (2x2 Events / Total)",
                                "continuous": "Continuous (Mean, SD, N)",
                                "generic": "Generic (Effect, SE)",
                            },
                            selected="binary",
                        ),
                    ),
                    ui.column(3, ui.output_ui("ui_meta_measure")),
                    ui.column(3, ui.output_ui("ui_meta_study_col")),
                    ui.column(3, ui.output_ui("ui_meta_subgroup_col")),
                ),
                # Dynamic columns for Binary
                ui.panel_conditional(
                    "input.meta_data_type == 'binary'",
                    ui.row(
                        ui.column(3, ui.output_ui("ui_bin_events_t")),
                        ui.column(3, ui.output_ui("ui_bin_n_t")),
                        ui.column(3, ui.output_ui("ui_bin_events_c")),
                        ui.column(3, ui.output_ui("ui_bin_n_c")),
                    ),
                ),
                # Dynamic columns for Continuous
                ui.panel_conditional(
                    "input.meta_data_type == 'continuous'",
                    ui.row(
                        ui.column(2, ui.output_ui("ui_cont_mean_t")),
                        ui.column(2, ui.output_ui("ui_cont_sd_t")),
                        ui.column(2, ui.output_ui("ui_cont_n_t")),
                        ui.column(2, ui.output_ui("ui_cont_mean_c")),
                        ui.column(2, ui.output_ui("ui_cont_sd_c")),
                        ui.column(2, ui.output_ui("ui_cont_n_c")),
                    ),
                ),
                # Dynamic columns for Generic
                ui.panel_conditional(
                    "input.meta_data_type == 'generic'",
                    ui.row(
                        ui.column(4, ui.output_ui("ui_gen_effect_col")),
                        ui.column(4, ui.output_ui("ui_gen_se_col")),
                        ui.column(
                            4,
                            ui.input_checkbox(
                                "meta_gen_is_ratio",
                                "Effect is Log-Ratio (e.g. Log OR/RR/HR)",
                                value=False,
                            ),
                        ),
                    ),
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.input_select(
                            "meta_method_re",
                            "Random-Effects Method:",
                            {
                                "dl": "DerSimonian-Laird (Standard)",
                                "reml": "Restricted Maximum Likelihood (REML)",
                                "pm": "Paule-Mandel",
                            },
                            selected="dl",
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_checkbox(
                            "meta_use_hksj",
                            "Apply Hartung-Knapp-Sidik-Jonkman (HKSJ)",
                            value=True,
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_checkbox(
                            "meta_use_pi", "Display 95% Prediction Interval", value=True
                        ),
                    ),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_run_meta",
                            "🚀 Run Meta-Analysis",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_meta_html",
                            "📥 HTML Report",
                            class_="btn-secondary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_meta_pdf",
                            "📥 PDF Report",
                            class_="btn-outline-danger w-100",
                        ),
                        ui.output_ui("dl_status_meta"),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_meta_status"),
                ui.br(),
                ui.output_ui("out_meta_results"),
            ),
            # TAB 2: Publication Bias & Funnel Plot
            ui.nav_panel(
                "🎯 Publication Bias & Funnel",
                ui.markdown("##### Small-Study Effects & Publication Bias Assessment"),
                ui.output_ui("out_meta_funnel_results"),
            ),
            # TAB 3: PRISMA 2020 Reference & Methods
            ui.nav_panel(
                "ℹ️ PRISMA 2020 & Methodology",
                ui.markdown("""
                    ### 📚 PRISMA 2020 Systematic Review Guidelines

                    This module adheres to international standards for meta-analysis reporting:
                    
                    - **Fixed-Effect Model (Inverse Variance)**: Assumes all studies share one identical true effect. Weights are inversely proportional to within-study variance ($w_i = 1 / \\text{SE}_i^2$).
                    - **Random-Effects Model (DerSimonian-Laird)**: Assumes true effect varies between studies following $\\mathcal{N}(\\mu, \\tau^2)$. Incorporates between-study variance $\\tau^2$.
                    - **Hartung-Knapp-Sidik-Jonkman (HKSJ)**: Highly recommended by the Cochrane Handbook for meta-analyses with small number of trials ($k < 20$), preventing false-positive statistical significance.
                    - **95% Prediction Interval**: Quantifies the range within which the effect of a *future* clinical trial is expected to fall.
                    - **Contour-Enhanced Funnel Plot**: Overlays statistical significance contours ($p < 0.10, p < 0.05, p < 0.01$) to distinguish publication bias (missing negative studies in white zone) from clinical heterogeneity.
                    - **Egger's Linear Regression**: Tests for funnel plot asymmetry; $p < 0.05$ flags potential publication bias.
                """),
            ),
        ),
    )


# ==============================================================================
# SERVER DEFINITION
# ==============================================================================


@module.server
def meta_analysis_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None] | None = None,
    is_matched: reactive.Value[bool] | None = None,
) -> None:
    """Server module for Clinical Meta-Analysis."""

    local_df = reactive.Value(None)
    local_name = reactive.Value("Dataset")

    # Dataset Selector
    current_df = register_dataset_selector(
        input=input,
        output=output,
        df=df,
        df_matched=df_matched if df_matched is not None else reactive.Value(None),
        is_matched=is_matched if is_matched is not None else reactive.Value(False),
        radio_input_id="radio_meta_source",
        title="📚 Clinical Meta-Analysis",
    )

    @reactive.calc
    def active_df():
        if local_df.get() is not None:
            return local_df.get()
        return current_df()

    # Demo Dataset Loader Handlers
    @reactive.effect
    @reactive.event(input.btn_load_bcg)
    def _load_bcg():
        bcg = get_bcg_vaccine_data()
        local_df.set(bcg)
        local_name.set("BCG Vaccine RCTs (Colditz 1994)")
        meta_state.set(None)
        status_msg.set(None)
        ui.update_select("meta_data_type", selected="binary")

    @reactive.effect
    @reactive.event(input.btn_load_statin)
    def _load_statin():
        statin = get_statin_continuous_data()
        local_df.set(statin)
        local_name.set("Statin LDL Reduction Trials")
        meta_state.set(None)
        status_msg.set(None)
        ui.update_select("meta_data_type", selected="continuous")

    @reactive.effect
    @reactive.event(input.btn_load_generic)
    def _load_generic():
        generic_data = get_generic_ratio_data()
        local_df.set(generic_data)
        local_name.set("SGLT2i Heart Failure Trials (Generic HR)")
        meta_state.set(None)
        status_msg.set(None)
        ui.update_select("meta_data_type", selected="generic")
        ui.update_checkbox("meta_gen_is_ratio", value=True)

    # Demo Modal & Raw Data Preview Handlers
    @reactive.effect
    @reactive.event(input.btn_view_demo_data)
    def _show_demo_modal():
        ui.modal_show(create_demo_data_modal())

    @reactive.effect
    @reactive.event(input.btn_view_demo_banner)
    def _show_demo_modal_banner():
        ui.modal_show(create_demo_data_modal())

    # Modal Dataset Quick Loaders
    @reactive.effect
    @reactive.event(input.btn_modal_load_bcg)
    def _modal_load_bcg():
        bcg = get_bcg_vaccine_data()
        local_df.set(bcg)
        local_name.set("BCG Vaccine RCTs (Colditz 1994)")
        meta_state.set(None)
        status_msg.set(None)
        ui.update_select("meta_data_type", selected="binary")
        ui.modal_remove()
        ui.notification_show(
            "Loaded BCG Vaccine RCTs dataset (Binary 2x2).", type="message"
        )

    @reactive.effect
    @reactive.event(input.btn_modal_load_statin)
    def _modal_load_statin():
        statin = get_statin_continuous_data()
        local_df.set(statin)
        local_name.set("Statin LDL Reduction Trials")
        meta_state.set(None)
        status_msg.set(None)
        ui.update_select("meta_data_type", selected="continuous")
        ui.modal_remove()
        ui.notification_show(
            "Loaded Statin LDL Reduction dataset (Continuous).", type="message"
        )

    @reactive.effect
    @reactive.event(input.btn_modal_load_generic)
    def _modal_load_generic():
        generic_data = get_generic_ratio_data()
        local_df.set(generic_data)
        local_name.set("SGLT2i Heart Failure Trials (Generic HR)")
        meta_state.set(None)
        status_msg.set(None)
        ui.update_select("meta_data_type", selected="generic")
        ui.update_checkbox("meta_gen_is_ratio", value=True)
        ui.modal_remove()
        ui.notification_show(
            "Loaded SGLT2i Heart Failure Trials dataset (Generic HR).", type="message"
        )

    # CSV Template Downloads
    @render.download(filename="bcg_vaccine_binary_template.csv")
    def btn_dl_demo_bcg_csv():
        return get_bcg_vaccine_data().to_csv(index=False)

    @render.download(filename="statin_continuous_template.csv")
    def btn_dl_demo_statin_csv():
        return get_statin_continuous_data().to_csv(index=False)

    @render.download(filename="sglt2i_generic_template.csv")
    def btn_dl_demo_generic_csv():
        return get_generic_ratio_data().to_csv(index=False)

    # Demo Reset / Exit Handlers
    @reactive.effect
    @reactive.event(input.btn_reset_demo)
    def _reset_demo():
        if local_df.get() is not None:
            local_df.set(None)
            local_name.set("Dataset")
            meta_state.set(None)
            status_msg.set(None)
            ui.notification_show("Returned to uploaded dataset mode.", type="message")

    @reactive.effect
    @reactive.event(input.btn_exit_demo_banner)
    def _exit_demo_banner():
        if local_df.get() is not None:
            local_df.set(None)
            local_name.set("Dataset")
            meta_state.set(None)
            status_msg.set(None)
            ui.notification_show("Returned to uploaded dataset mode.", type="message")

    @reactive.effect
    @reactive.event(input.radio_meta_source)
    def _on_source_radio_change():
        if local_df.get() is not None:
            local_df.set(None)
            local_name.set("Dataset")
            meta_state.set(None)
            status_msg.set(None)

    @output
    @render.ui
    def ui_demo_dataset_banner():
        if local_df.get() is not None:
            return ui.div(
                ui.div(
                    ui.span(
                        f"💡 Demo Mode Active: {local_name.get()}",
                        class_="fw-bold text-primary",
                    ),
                    ui.div(
                        ui.input_action_button(
                            "btn_view_demo_banner",
                            "👁️ View Raw Data",
                            class_="btn-sm btn-outline-primary me-2 py-0 px-2",
                        ),
                        ui.input_action_button(
                            "btn_exit_demo_banner",
                            "✕ Exit Demo (Use Uploaded Data)",
                            class_="btn-sm btn-outline-danger py-0 px-2",
                        ),
                        class_="d-flex align-items-center",
                    ),
                    class_="d-flex align-items-center justify-content-between",
                ),
                class_="alert alert-info py-2 px-3 mb-3 border-0 shadow-sm",
            )
        return None

    # Dynamic Column Pickers
    @output
    @render.ui
    def ui_meta_measure():
        dtype = input.meta_data_type()
        if dtype == "binary":
            opts = {
                "OR": "Odds Ratio (OR)",
                "RR": "Risk Ratio (RR)",
                "RD": "Risk Difference (RD)",
            }
        elif dtype == "continuous":
            opts = {
                "SMD": "Standardized Mean Diff (Hedges' g)",
                "MD": "Mean Difference (MD)",
            }
        else:
            opts = {"Generic": "Log-Ratio / Difference"}
        return ui.input_select(
            "meta_measure", "Effect Measure:", opts, selected=list(opts.keys())[0]
        )

    @output
    @render.ui
    def ui_meta_study_col():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        default_study = select_variable_by_keyword(
            cols, ["study", "trial", "author", "name", "id"]
        )
        return ui.input_select(
            "meta_study_col",
            "Study Identifier Column:",
            choices=cols,
            selected=default_study,
        )

    @output
    @render.ui
    def ui_meta_subgroup_col():
        df = active_df()
        cols = ["None"] + (list(df.columns) if df is not None else [])
        default_sg = select_variable_by_keyword(
            cols, ["group", "subgroup", "tier", "stratum", "latitude"]
        )
        return ui.input_select(
            "meta_subgroup_col",
            "Subgroup Column (Optional):",
            choices=cols,
            selected=default_sg if default_sg in cols else "None",
        )

    # Binary Column Pickers
    @output
    @render.ui
    def ui_bin_events_t():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols, ["tb_vaccine", "events_t", "event_t", "event1", "e_t", "cases_t"]
        )
        return ui.input_select(
            "bin_events_t", "Events (Treatment):", choices=cols, selected=sel
        )

    @output
    @render.ui
    def ui_bin_n_t():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols, ["total_vaccine", "n_t", "total_t", "n1", "pop_t"]
        )
        return ui.input_select(
            "bin_n_t", "Total N (Treatment):", choices=cols, selected=sel
        )

    @output
    @render.ui
    def ui_bin_events_c():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols, ["tb_control", "events_c", "event_c", "event0", "e_c", "cases_c"]
        )
        return ui.input_select(
            "bin_events_c", "Events (Control):", choices=cols, selected=sel
        )

    @output
    @render.ui
    def ui_bin_n_c():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols, ["total_control", "n_c", "total_c", "n0", "pop_c"]
        )
        return ui.input_select(
            "bin_n_c", "Total N (Control):", choices=cols, selected=sel
        )

    # Continuous Column Pickers
    @output
    @render.ui
    def ui_cont_mean_t():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols,
            [
                "mean_statin",
                "mean_treatment",
                "mean_tx",
                "mean_t",
                "mean1",
                "m_t",
                "mean_exp",
            ],
        )
        return ui.input_select(
            "cont_mean_t", "Mean (Treatment):", choices=cols, selected=sel
        )

    @output
    @render.ui
    def ui_cont_sd_t():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols,
            [
                "sd_statin",
                "sd_treatment",
                "sd_tx",
                "sd_t",
                "sd1",
                "s_t",
                "sd_exp",
            ],
        )
        return ui.input_select(
            "cont_sd_t", "SD (Treatment):", choices=cols, selected=sel
        )

    @output
    @render.ui
    def ui_cont_n_t():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols,
            [
                "n_statin",
                "n_treatment",
                "n_tx",
                "n_t",
                "total_statin",
                "total_treatment",
                "total_t",
                "n1",
                "pop_t",
            ],
        )
        return ui.input_select("cont_n_t", "N (Treatment):", choices=cols, selected=sel)

    @output
    @render.ui
    def ui_cont_mean_c():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols,
            [
                "mean_control",
                "mean_ctrl",
                "mean_c",
                "mean0",
                "m_c",
            ],
        )
        return ui.input_select(
            "cont_mean_c", "Mean (Control):", choices=cols, selected=sel
        )

    @output
    @render.ui
    def ui_cont_sd_c():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols,
            [
                "sd_control",
                "sd_ctrl",
                "sd_c",
                "sd0",
                "s_c",
            ],
        )
        return ui.input_select("cont_sd_c", "SD (Control):", choices=cols, selected=sel)

    @output
    @render.ui
    def ui_cont_n_c():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols,
            [
                "n_control",
                "n_ctrl",
                "n_c",
                "total_control",
                "total_c",
                "n0",
                "pop_c",
            ],
        )
        return ui.input_select("cont_n_c", "N (Control):", choices=cols, selected=sel)

    # Generic Column Pickers
    @output
    @render.ui
    def ui_gen_effect_col():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols,
            [
                "log_hr",
                "log_effect",
                "effect",
                "yi",
                "or",
                "rr",
                "estimate",
            ],
        )
        return ui.input_select(
            "gen_effect_col", "Effect Size (or Log-Ratio):", choices=cols, selected=sel
        )

    @output
    @render.ui
    def ui_gen_se_col():
        df = active_df()
        cols = list(df.columns) if df is not None else []
        sel = select_variable_by_keyword(
            cols, ["se_log_hr", "se", "vi", "stderr", "std_err"]
        )
        return ui.input_select(
            "gen_se_col", "Standard Error (SE):", choices=cols, selected=sel
        )

    # Analysis State
    meta_state = reactive.Value(None)
    status_msg = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.btn_run_meta)
    def _execute_meta():
        df = active_df()
        if df is None or df.empty:
            status_msg.set("⚠️ Please select or load a dataset first.")
            return

        dtype = input.meta_data_type()
        study_col = input.meta_study_col()
        sg_col = input.meta_subgroup_col()
        sg_name = None if sg_col == "None" else sg_col
        if dtype == "binary":
            measure = (
                input.meta_measure()
                if input.meta_measure() in ["OR", "RR", "RD"]
                else "OR"
            )
        elif dtype == "continuous":
            measure = (
                input.meta_measure() if input.meta_measure() in ["SMD", "MD"] else "SMD"
            )
        else:
            measure = "Generic"
        use_hksj = input.meta_use_hksj()
        method_re = input.meta_method_re()

        try:
            if dtype == "binary":
                req(
                    input.bin_events_t(),
                    input.bin_n_t(),
                    input.bin_events_c(),
                    input.bin_n_c(),
                )
                eff_df = meta_analysis_lib.compute_binary_effect_sizes(
                    df,
                    events_t_col=input.bin_events_t(),
                    n_t_col=input.bin_n_t(),
                    events_c_col=input.bin_events_c(),
                    n_c_col=input.bin_n_c(),
                    study_col=study_col,
                    effect_measure=measure,
                    subgroup_col=sg_name,
                )
            elif dtype == "continuous":
                req(
                    input.cont_mean_t(),
                    input.cont_sd_t(),
                    input.cont_n_t(),
                    input.cont_mean_c(),
                    input.cont_sd_c(),
                    input.cont_n_c(),
                )
                eff_df = meta_analysis_lib.compute_continuous_effect_sizes(
                    df,
                    mean_t_col=input.cont_mean_t(),
                    sd_t_col=input.cont_sd_t(),
                    n_t_col=input.cont_n_t(),
                    mean_c_col=input.cont_mean_c(),
                    sd_c_col=input.cont_sd_c(),
                    n_c_col=input.cont_n_c(),
                    study_col=study_col,
                    effect_measure=measure,
                    subgroup_col=sg_name,
                )
            else:  # Generic
                req(input.gen_effect_col(), input.gen_se_col())
                gen_cols = [study_col, input.gen_effect_col(), input.gen_se_col()]
                if sg_name and sg_name in df.columns:
                    gen_cols.append(sg_name)
                gen_cols = list(dict.fromkeys(gen_cols))
                eff_df = df[gen_cols].dropna().copy()
                rename_map = {
                    study_col: "study",
                    input.gen_effect_col(): "effect_size",
                    input.gen_se_col(): "se",
                }
                if sg_name and sg_name in df.columns:
                    rename_map[sg_name] = "subgroup"
                eff_df.rename(columns=rename_map, inplace=True)
                is_ratio = bool(input.meta_gen_is_ratio())
                eff_df["log_effect"] = eff_df["effect_size"]
                eff_df["ci_lower"] = eff_df["effect_size"] - 1.96 * eff_df["se"]
                eff_df["ci_upper"] = eff_df["effect_size"] + 1.96 * eff_df["se"]
                eff_df["is_ratio"] = is_ratio

            meta_res = meta_analysis_lib.run_meta_analysis(
                eff_df, method_re=method_re, use_hksj=use_hksj
            )
            pub_res = meta_analysis_lib.run_publication_bias_tests(eff_df)

            meta_state.set(
                {
                    "meta": meta_res,
                    "publication": pub_res,
                    "measure": measure,
                }
            )
            status_msg.set(
                f"✅ Meta-analysis successfully executed across {meta_res.get('k', 0)} studies."
            )
        except Exception as e:
            logger.exception(f"Meta-analysis execution failed: {e}")
            status_msg.set(f"❌ Analysis failed: {str(e)}")

    @output
    @render.ui
    def ui_meta_status():
        msg = status_msg()
        if not msg:
            return None
        alert_type = "alert-success" if "✅" in msg else "alert-danger"
        return ui.div(msg, class_=f"alert {alert_type} py-2 mb-3")

    @output
    @render.ui
    def out_meta_results():
        state = meta_state()
        if not state or "error" in state["meta"]:
            return create_placeholder_state(
                "No meta-analysis executed yet. Select variables and click Run."
            )

        res = state["meta"]
        fe = res["fixed_effect"]
        re_eff = res["random_effect"]
        het = res["heterogeneity"]
        measure = state["measure"]

        # Build Forest Plot
        fig_forest = meta_analysis_lib.create_meta_forest_plot(
            res,
            title=f"Forest Plot of {measure} (Pooled Effect)",
            effect_label=f"{measure} (95% CI)",
        )
        forest_html = fig_forest.to_html(include_plotlyjs="cdn", full_html=False)

        # Summary Table HTML
        use_pi = bool(input.meta_use_pi())
        pi_str = ""
        pi_dict = re_eff.get("prediction_interval", {})
        if not np.isnan(pi_dict.get("pi_lower", np.nan)):
            pi_str = f" [{pi_dict['pi_lower']:.2f}, {pi_dict['pi_upper']:.2f}]"

        pi_th = "<th>95% Prediction Interval</th>" if use_pi else ""
        pi_td_fe = (
            '<td><span class="text-muted">N/A (Fixed Effect)</span></td>'
            if use_pi
            else ""
        )
        pi_td_re = (
            f"<td><strong>{pi_str if pi_str else 'N/A'}</strong></td>" if use_pi else ""
        )

        summary_table_html = f"""
        <div class="card mb-4 border-0 shadow-sm">
            <div class="card-body">
                <h5 class="card-title text-primary fw-bold">📊 Pooled Summary Effect Estimates</h5>
                <div class="table-responsive">
                    <table class="table table-bordered table-hover align-middle mb-0">
                        <thead class="table-light">
                            <tr>
                                <th>Model Type</th>
                                <th>Estimate ({measure})</th>
                                <th>95% Confidence Interval</th>
                                <th>p-value</th>
                                {pi_th}
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Fixed Effect (Inverse Variance)</strong></td>
                                <td><span class="badge bg-primary text-white fs-6">{fe["effect_disp"]:.3f}</span></td>
                                <td>{fe["ci_lower"]:.3f} – {fe["ci_upper"]:.3f}</td>
                                <td><strong>{fe["p_value"]:.4f}</strong></td>
                                {pi_td_fe}
                            </tr>
                            <tr class="table-warning">
                                <td><strong>Random Effects ({re_eff["method"]})</strong></td>
                                <td><span class="badge bg-danger text-white fs-6">{re_eff["effect_disp"]:.3f}</span></td>
                                <td>{re_eff["ci_lower"]:.3f} – {re_eff["ci_upper"]:.3f}</td>
                                <td><strong>{re_eff["p_value"]:.4f}</strong></td>
                                {pi_td_re}
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """

        # Heterogeneity Card HTML
        het_html = f"""
        <div class="card mb-4 border-0 shadow-sm">
            <div class="card-body">
                <h5 class="card-title text-secondary fw-bold">🔍 Heterogeneity Assessment (PRISMA 2020)</h5>
                <div class="row text-center mt-3">
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded border">
                            <small class="text-muted d-block">I² Statistic</small>
                            <span class="fs-4 fw-bold text-primary">{het["I2"]:.1f}%</span>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded border">
                            <small class="text-muted d-block">Cochran's Q (p-value)</small>
                            <span class="fs-4 fw-bold">{het["Q"]:.2f} <small class="fs-6 text-muted">(p={het["p_value"]:.3f})</small></span>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded border">
                            <small class="text-muted d-block">Between-Study Variance (τ²)</small>
                            <span class="fs-4 fw-bold text-danger">{het["tau2"]:.4f}</span>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded border">
                            <small class="text-muted d-block">Interpretation</small>
                            <span class="fs-6 fw-bold">{het["interpretation"]}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

        # Subgroup Differences if present
        subgroup_html = ""
        if res.get("subgroups") is not None:
            sg = res["subgroups"]
            subgroup_html = f"""
            <div class="card mb-4 border-0 shadow-sm">
                <div class="card-body">
                    <h5 class="card-title text-info fw-bold">🏷️ Subgroup Analysis & Test for Subgroup Differences</h5>
                    <p class="mb-2"><strong>Between-Subgroup Heterogeneity:</strong> Q_between = <strong>{sg["Q_between"]:.2f}</strong> (df = {sg["df_between"]}, p = <strong>{sg["p_between"]:.4f}</strong>)</p>
                    <div class="alert alert-light border">
                        {"⚠️ Significant effect variation across subgroups (p < 0.05)" if sg["p_between"] < 0.05 else "✅ No significant difference detected between subgroups (p ≥ 0.05)"}
                    </div>
                </div>
            </div>
            """

        content_tags = [
            ui.HTML(summary_table_html),
            ui.HTML(het_html),
        ]
        if subgroup_html:
            content_tags.append(ui.HTML(subgroup_html))
        content_tags.append(
            ui.HTML(
                f"<div class='card border-0 shadow-sm p-3 mb-4'>{forest_html}</div>"
            )
        )

        return create_results_container(
            f"📚 Meta-Analysis Report ({measure})",
            *content_tags,
            class_="fade-in-entry",
        )

    @output
    @render.ui
    def out_meta_funnel_results():
        state = meta_state()
        if not state or "error" in state["meta"]:
            return create_placeholder_state(
                "Run meta-analysis first to generate publication bias diagnostics and funnel plots."
            )

        res = state["meta"]
        pub = state["publication"]

        fig_funnel = meta_analysis_lib.create_contour_enhanced_funnel_plot(
            res,
            title="Contour-Enhanced Funnel Plot (Publication Bias Diagnostic)",
        )
        funnel_html = fig_funnel.to_html(include_plotlyjs="cdn", full_html=False)

        # Egger & Begg Table HTML
        if "error" in pub:
            bias_html = f"""
            <div class="card mb-4 border-0 shadow-sm">
                <div class="card-body">
                    <h5 class="card-title text-primary fw-bold">📉 Small-Study Effects & Statistical Bias Tests</h5>
                    <div class="alert alert-warning mb-0">
                        ⚠️ <strong>Publication Bias Tests Unavailable:</strong> {pub["error"]}
                    </div>
                </div>
            </div>
            """
        else:
            egger = pub.get("egger", {})
            begg = pub.get("begg", {})

            bias_html = f"""
            <div class="card mb-4 border-0 shadow-sm">
                <div class="card-body">
                    <h5 class="card-title text-primary fw-bold">📉 Small-Study Effects & Statistical Bias Tests</h5>
                    <div class="table-responsive">
                        <table class="table table-bordered table-sm align-middle mb-0">
                            <thead class="table-light">
                                <tr>
                                    <th>Test</th>
                                    <th>Statistic</th>
                                    <th>p-value</th>
                                    <th>Interpretation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Egger's Linear Regression Test</strong></td>
                                    <td>Intercept = <strong>{egger.get("intercept", 0):.3f}</strong> (t = {egger.get("t_stat", 0):.2f})</td>
                                    <td><strong>{egger.get("p_value", 1.0):.4f}</strong></td>
                                    <td>{pub.get("interpretation", "")}</td>
                                </tr>
                                <tr>
                                    <td><strong>Begg & Mazumdar Rank Correlation</strong></td>
                                    <td>Kendall's τ = <strong>{begg.get("kendall_tau", 0):.3f}</strong></td>
                                    <td><strong>{begg.get("p_value", 1.0):.4f}</strong></td>
                                    <td>{"Significant rank asymmetry" if begg.get("p_value", 1.0) < 0.05 else "No significant rank asymmetry"}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            """

        return create_results_container(
            "🎯 Publication Bias & Contour Funnel Plot",
            ui.HTML(bias_html),
            ui.HTML(
                f"<div class='card border-0 shadow-sm p-3 mb-4'>{funnel_html}</div>"
            ),
            class_="fade-in-entry",
        )

    # HTML Download Handler
    @render.download(filename="meta_analysis_report.html")
    def btn_dl_meta_html():
        state = meta_state()
        if not state:
            return ""
        res = state["meta"]
        fig = meta_analysis_lib.create_meta_forest_plot(res)
        return safe_download_html(fig, "meta_analysis_report")

    # PDF Download Handler
    @render.download(filename="meta_analysis_report.pdf")
    def btn_dl_meta_pdf():
        state = meta_state()
        if not state:
            return b""
        res = state["meta"]
        fig = meta_analysis_lib.create_meta_forest_plot(res)
        return safe_download_pdf(fig, "meta_analysis_report")
