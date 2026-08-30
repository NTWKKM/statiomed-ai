from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from shiny import module, reactive, render, req, ui

from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from tabs._dataset_mixin import register_dataset_selector
from utils import decision_curve_lib, diag_test, fagan_nomogram_lib
from utils.data_cleaning import prepare_data_for_analysis
from utils.diagnostic_advanced_lib import DiagnosticComparison, DiagnosticTest
from utils.download_helpers import safe_download_html
from utils.formatting import create_missing_data_report_html
from utils.pdf_helpers import safe_download_pdf
from utils.ui_helpers import create_download_status_badge, create_results_container

logger = get_logger(__name__)


def _create_status_elements(msg: str) -> list[dict[str, str]]:
    """
    Split status message into components if it contains the HTML block.
    This prevents wrapping block-level elements (div) inside a logical paragraph (p).
    """
    # Marker for the start of the Likelihood Ratio explanation block
    # (Matches utils/diag_test.py)
    html_block_marker = "<div class='alert alert-light border mt-3'>"

    if html_block_marker in msg:
        parts = msg.split(html_block_marker, 1)
        plain_text = parts[0].strip()
        # Reconstruct the HTML block (marker + rest)
        html_content = html_block_marker + parts[1]

        elements = []
        if plain_text:
            elements.append({"type": "text", "data": plain_text})

        elements.append({"type": "html", "data": html_content})
        return elements

    # Default case: return a single text element
    return [{"type": "text", "data": msg}]


COLORS = get_color_palette()


# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def diag_ui() -> ui.TagChild:
    """
    Construct the Diagnostics page UI with controls and result areas for ROC, Chi-Square, Descriptive, Decision Curve Analysis, and a reference/interpretation guide.

    Returns:
        ui.TagChild: A UI container that includes the page title and dataset selector followed by a tabset of five panels (ROC Curve & AUC, Chi-Square & Risk, Descriptive statistics, Decision Curve Analysis, and Reference & Interpretation) with their corresponding input controls, action/download buttons, status displays, and result output regions.
    """
    return ui.div(
        # Title + Data Summary inline
        ui.output_ui("ui_title_with_summary"),
        # Dataset Info Box
        ui.output_ui("ui_matched_info"),
        ui.br(),
        # Dataset Selector
        ui.output_ui("ui_dataset_selector"),
        ui.br(),
        # Tabs for different analyses
        ui.navset_tab(
            # TAB 1: ROC Curve & AUC
            ui.nav_panel(
                "📈 ROC Curve & AUC",
                ui.markdown("##### ROC Curve Analysis"),
                # --- NEW: Analysis Mode Selection ---
                ui.input_radio_buttons(
                    "roc_mode",
                    "Analysis Mode:",
                    {
                        "single": "Single Test Analysis",
                        "compare": "Compare Two Tests (Paired)",
                    },
                    selected="single",
                    inline=True,
                ),
                ui.hr(),
                # --- SINGLE MODE UI ---
                ui.panel_conditional(
                    "input.roc_mode == 'single'",
                    ui.row(
                        ui.column(3, ui.output_ui("ui_roc_truth")),
                        ui.column(3, ui.output_ui("ui_roc_score")),
                        ui.column(3, ui.output_ui("ui_roc_method")),
                        ui.column(3, ui.output_ui("ui_roc_pos_label")),
                    ),
                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "btn_analyze_roc",
                                "🚀 Analyze ROC",
                                class_="btn-primary w-100",
                                width="100%",
                            ),
                        ),
                        ui.column(
                            3,
                            ui.download_button(
                                "btn_dl_roc_report",
                                "📥 HTML",
                                class_="btn-secondary w-100",
                                width="100%",
                            ),
                        ),
                        ui.column(
                            3,
                            ui.download_button(
                                "btn_dl_roc_pdf",
                                "📥 PDF",
                                class_="btn-outline-danger w-100",
                                width="100%",
                            ),
                            ui.output_ui("dl_status_roc"),
                        ),
                    ),
                ),
                # --- COMPARISON MODE UI ---
                ui.panel_conditional(
                    "input.roc_mode == 'compare'",
                    ui.row(
                        ui.column(4, ui.output_ui("ui_roc_truth_comp")),
                        ui.column(4, ui.output_ui("ui_roc_test1")),
                        ui.column(4, ui.output_ui("ui_roc_test2")),
                    ),
                    ui.row(
                        ui.column(6, ui.output_ui("ui_roc_pos_label_comp")),
                        ui.column(
                            6,
                            ui.div(
                                ui.input_action_button(
                                    "btn_compare_roc",
                                    "🚀 Run Comparison (DeLong Test)",
                                    class_="btn-warning w-100",
                                ),
                                style="margin-top: 25px;",
                            ),
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_roc_status"),  # Status message area
                ui.br(),
                ui.output_ui("out_roc_results"),
            ),
            # TAB 2: Chi-Square & Risk Analysis
            ui.nav_panel(
                "🎲 Chi-Square & Risk (2x2)",
                ui.markdown("##### Chi-Square & Risk Analysis (2x2 Contingency Table)"),
                ui.row(
                    ui.column(3, ui.output_ui("ui_chi_v1")),
                    ui.column(3, ui.output_ui("ui_chi_v2")),
                    ui.column(3, ui.output_ui("ui_chi_method")),
                    ui.column(3, ui.output_ui("ui_chi_caption")),
                ),
                ui.row(
                    ui.column(3, ui.output_ui("ui_chi_v1_pos")),
                    ui.column(3, ui.output_ui("ui_chi_v2_pos")),
                    ui.column(3, ui.output_ui("ui_chi_note")),
                    ui.column(3, ui.output_ui("ui_chi_empty")),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_chi",
                            "🚀 Analyze Chi-Square",
                            class_="btn-primary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_chi_report",
                            "📥 HTML",
                            class_="btn-secondary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_chi_pdf",
                            "📥 PDF",
                            class_="btn-outline-danger w-100",
                            width="100%",
                        ),
                        ui.output_ui("dl_status_chi"),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_chi_status"),  # Status message area
                ui.br(),
                ui.output_ui("out_chi_results"),
            ),
            # TAB 3: Descriptive
            ui.nav_panel(
                "📊 Descriptive",
                ui.markdown("##### Descriptive Statistics"),
                ui.row(ui.column(12, ui.output_ui("ui_desc_var"))),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_run_desc",
                            "Show Stats",
                            class_="btn-primary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_desc_report",
                            "📥 HTML",
                            class_="btn-secondary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_desc_pdf",
                            "📥 PDF",
                            class_="btn-outline-danger w-100",
                            width="100%",
                        ),
                        ui.output_ui("dl_status_desc"),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_desc_status"),  # Status message area
                ui.br(),
                ui.output_ui("out_desc_results"),
            ),
            # TAB 4: Decision Curve Analysis (DCA)
            ui.nav_panel(
                "📉 Decision Curve (DCA)",
                ui.markdown("##### Decision Curve Analysis"),
                ui.row(
                    ui.column(6, ui.output_ui("ui_dca_truth")),
                    ui.column(6, ui.output_ui("ui_dca_prob")),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_run_dca",
                            "🚀 Run DCA",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_dca_report",
                            "📥 HTML",
                            class_="btn-outline-primary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_dca_pdf",
                            "📥 PDF",
                            class_="btn-outline-danger w-100",
                            width="100%",
                        ),
                        ui.output_ui("dl_status_dca"),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_dca_status"),
                ui.br(),
                ui.output_ui("out_dca_results"),
            ),
            # TAB 5: Interactive Fagan's Nomogram
            ui.nav_panel(
                "🧭 Fagan's Nomogram",
                ui.markdown("##### 🧭 Interactive Fagan's Nomogram (Bedside Bayes)"),
                ui.input_radio_buttons(
                    "fagan_mode",
                    "Operating Mode:",
                    {
                        "standalone": "Bedside Calculator (Manual Input)",
                        "dataset": "Linked to Dataset & Cutoff",
                        "multilevel": "Multi-Level Biomarker Tiers (Interval LRs)",
                    },
                    selected="standalone",
                    inline=True,
                ),
                ui.hr(),
                # Mode 1: Standalone
                ui.panel_conditional(
                    "input.fagan_mode == 'standalone'",
                    ui.row(
                        ui.column(
                            4,
                            ui.input_slider(
                                "fagan_manual_pre_prob",
                                "Pre-test Probability (%):",
                                min=0.1,
                                max=99.9,
                                value=20.0,
                                step=0.5,
                            ),
                        ),
                        ui.column(
                            4,
                            ui.input_numeric(
                                "fagan_manual_lr_pos",
                                "Positive Likelihood Ratio (LR+):",
                                value=10.0,
                                min=0.01,
                                step=0.5,
                            ),
                        ),
                        ui.column(
                            4,
                            ui.input_numeric(
                                "fagan_manual_lr_neg",
                                "Negative Likelihood Ratio (LR-):",
                                value=0.10,
                                min=0.001,
                                max=5.0,
                                step=0.05,
                            ),
                        ),
                    ),
                    ui.row(
                        ui.column(
                            6,
                            ui.input_text(
                                "fagan_manual_test_name",
                                "Diagnostic Test / Biomarker Name:",
                                value="Bedside Clinical Test",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.div(
                                ui.markdown(
                                    "<small class='text-muted'>💡 Adjust LR+ and LR- directly. Lines update dynamically to show post-test probabilities.</small>"
                                ),
                                style="margin-top: 25px;",
                            ),
                        ),
                    ),
                ),
                # Mode 2: Linked to Dataset
                ui.panel_conditional(
                    "input.fagan_mode == 'dataset'",
                    ui.row(
                        ui.column(3, ui.output_ui("ui_fagan_truth")),
                        ui.column(3, ui.output_ui("ui_fagan_score")),
                        ui.column(3, ui.output_ui("ui_fagan_pos_label")),
                        ui.column(
                            3,
                            ui.input_slider(
                                "fagan_ds_pre_prob",
                                "Pre-test Probability (%):",
                                min=0.1,
                                max=99.9,
                                value=20.0,
                                step=0.5,
                            ),
                        ),
                    ),
                ),
                # Mode 3: Multi-Level Interval LRs
                ui.panel_conditional(
                    "input.fagan_mode == 'multilevel'",
                    ui.row(
                        ui.column(3, ui.output_ui("ui_fagan_multi_truth")),
                        ui.column(2, ui.output_ui("ui_fagan_multi_pos_label")),
                        ui.column(3, ui.output_ui("ui_fagan_multi_score")),
                        ui.column(
                            2,
                            ui.input_text(
                                "fagan_multi_cutoffs",
                                "Interval Cutoffs:",
                                value="14, 50",
                            ),
                        ),
                        ui.column(
                            2,
                            ui.input_slider(
                                "fagan_multi_pre_prob",
                                "Pre-test Prob (%):",
                                min=0.1,
                                max=99.9,
                                value=20.0,
                                step=0.5,
                            ),
                        ),
                    ),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_generate_fagan",
                            "🚀 Generate Nomogram",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_fagan_report",
                            "📥 HTML",
                            class_="btn-secondary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_fagan_pdf",
                            "📥 PDF",
                            class_="btn-outline-danger w-100",
                        ),
                        ui.output_ui("dl_status_fagan"),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_fagan_status"),
                ui.br(),
                ui.output_ui("out_fagan_results"),
            ),
            # TAB 6: Reference & Interpretation
            ui.nav_panel(
                "ℹ️ Reference & Interpretation",
                ui.markdown("""
                    ## 📚 Reference & Interpretation Guide

                    💡 **Tip:** This section provides detailed explanations and interpretation rules for all the diagnostic tests.

                    ### 🚦 Quick Decision Guide

                    | **Question** | **Recommended Test** | **Example** |
                    | :--- | :--- | :--- |
                    | My test is a **score** (e.g., 0-100) and I want to see how well it predicts a **disease** (Yes/No)? | **ROC Curve & AUC** | Risk Score vs Diabetes |
                    | I want to find the **best cut-off** value for my test score? | **ROC Curve (Youden Index)** | Finding optimal BP for Hypertension |
                    | Are these two **groups** (e.g., Treatment vs Control) different in outcome (Cured vs Not Cured)? | **Chi-Square** | Drug A vs Placebo on Recovery |
                    | Is my model **clinically useful** at a specific threshold? | **Decision Curve (DCA)** | Should we biopsy everyone with PSA > 4? |
                    | *(For Agreement/Kappa, see "Agreement" tab)* | | |

                    ### ⚖️ Interpretation Guidelines

                    #### ROC Curve & AUC
                    - **Single Test**: Detailed analysis of one diagnostic test with threshold optimization.
                    - **Compare Tests**: Uses **Paired DeLong's Test** to statistically compare two ROC curves properly.
                    - **AUC > 0.9:** Excellent discrimination
                    - **AUC 0.8-0.9:** Good discrimination
                    - **AUC 0.7-0.8:** Fair discrimination
                    - **AUC 0.5-0.7:** Poor discrimination
                    - **AUC = 0.5:** No discrimination (random chance)
                    - **Youden J Index:** Sensitivity + Specificity - 1 (higher is better, max = 1)
                    
                    #### Comparison Interpretation (DeLong)
                    - **P-value < 0.05**: Significant difference between the two ROC curves.
                    - **Z-score**: Strength of the difference.

                    #### Chi-Square Test
                    - **P < 0.05:** Statistically significant association
                    - **Odds Ratio (OR):** If 95% CI doesn't include 1.0, it's significant
                    - **Risk Ratio (RR):** Similar interpretation as OR
                    - Use **Fisher's Exact Test** when expected counts < 5

                    #### Decision Curve Analysis (DCA)
                    - **Net Benefit:** The benefit of treating true positives minus the harm of treating false positives.
                    - **Interpretation:** The model is useful if the model's curve (Red) is **higher** than both:
                        - **Treat All** (Gray line): Treating everyone assuming they have the disease.
                        - **Treat None** (Horizontal line at 0): Treating no one.
                    - **Threshold Probability:** The patient's/doctor's preference (e.g., how worried are they about missing a case vs unnecessary treatment?).

                    #### Descriptive Statistics
                    - **Mean:** Average value (affected by outliers)
                    - **Median:** Middle value (robust to outliers)
                    - **SD (Standard Deviation):** Spread of data around mean
                    - **Q1/Q3:** 25th and 75th percentiles
                    """),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def diag_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
) -> None:
    # --- Reactive Results Storage ---
    """
    Register server-side UI renderers, reactive handlers, and analysis workflows for the Diagnostics module (ROC, ROC comparison, Chi-Square, Descriptive, and Decision Curve Analysis). Sets up reactive storage for generated HTML reports and processing flags and wires input-driven analysis effects and download handlers.

    Parameters:
        input: Shiny-like input accessor used to read UI control values and events.
        output: Shiny-like output registry used to attach UI render targets.
        session: Shiny-like session object for the current user connection.
        df (reactive.Value[pd.DataFrame | None]): Primary reactive dataset.
        var_meta (reactive.Value[dict[str, Any]]): Reactive variable metadata used for reports and missing-data summaries.
        df_matched (reactive.Value[pd.DataFrame | None]): Optional reactive matched dataset (e.g., from propensity score matching).
        is_matched (reactive.Value[bool]): Reactive flag indicating whether a matched dataset is available/selected.

    Behavior:
        - Exposes UI renderers for inputs, status indicators, and result containers used by the Diagnostics tab.
        - Maintains reactive storage for generated HTML reports (ROC, ROC comparison, Chi-Square, Descriptive, DCA) and processing flags.
        - Implements analysis event handlers that run when corresponding action buttons are triggered:
            * Single-test ROC analysis (ROC plot, statistics, calibration/sens-spec plots, performance table).
            * Paired ROC comparison using a DeLong paired test (comparison plot, DeLong table, optimal-threshold metrics).
            * Chi-Square / 2x2 analysis (contingency table, statistics, risk/effect measures).
            * Descriptive statistics for a selected variable.
            * Decision Curve Analysis (net benefit calculations, DCA plot, selected-threshold net benefits).
        - Each analysis appends missing-data summaries to reports when applicable and exposes download handlers that yield the generated HTML report content.
        - All processing flags are managed to allow UI status spinners while computations run.

    Note:
        This function configures server-side behavior and does not return a value.
    """
    roc_html: reactive.Value[str | None] = reactive.Value(None)
    chi_html: reactive.Value[str | None] = reactive.Value(None)
    desc_html: reactive.Value[str | None] = reactive.Value(None)

    # --- DCA Reactives ---
    dca_html: reactive.Value[str | None] = reactive.Value(None)
    dca_processing: reactive.Value[bool] = reactive.Value(False)

    roc_processing: reactive.Value[bool] = reactive.Value(False)
    chi_processing: reactive.Value[bool] = reactive.Value(False)
    desc_processing: reactive.Value[bool] = reactive.Value(False)

    # --- Dataset Selection Logic ---
    current_df = register_dataset_selector(
        input=input,
        output=output,
        df=df,
        df_matched=df_matched,
        is_matched=is_matched,
        radio_input_id="radio_diag_source",
        title="🧪 Diagnostic Tests (ROC)",
    )

    # Get all columns
    @reactive.Calc
    def all_cols() -> list[str]:
        d = current_df()
        if d is not None:
            return d.columns.tolist()
        return []

    # --- ROC Inputs UI ---
    @render.ui
    def ui_roc_truth():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["gold", "standard", "truth"], default_to_first=True
        )
        return ui.input_select(
            "sel_roc_truth",
            "Gold Standard (Binary):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_roc_score():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["score", "rapid", "test"], default_to_first=True
        )
        return ui.input_select(
            "sel_roc_score",
            "Test Score (Continuous):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_roc_method():
        return ui.input_radio_buttons(
            "radio_roc_method",
            "CI Method:",
            {"delong": "DeLong et al.", "hanley": "Binomial (Hanley)"},
            inline=True,
        )

    @render.ui
    def ui_roc_pos_label():
        """
        Render a dropdown to select the positive label for the currently selected ROC truth column.

        Builds the choice list from the non-missing unique values of the selected truth column in the current data. If the values include "1" or "1.0", that value is selected by default; otherwise the first unique value is selected. If the truth column is missing or no data is available, returns a select input with no choices.

        Returns:
            A Shiny select input component for choosing the positive label.
        """
        truth_col = input.sel_roc_truth()
        d = current_df()
        if d is not None and truth_col and truth_col in d.columns:
            unique_vals = sorted([str(x) for x in d[truth_col].dropna().unique()])
            default_pos_idx = 0
            # Force default to "1" or "1.0" if available
            if "1" in unique_vals:
                default_pos_idx = unique_vals.index("1")
            elif "1.0" in unique_vals:
                default_pos_idx = unique_vals.index("1.0")

            return ui.input_select(
                "sel_roc_pos_label",
                "Positive Label (1):",
                choices=unique_vals,
                selected=unique_vals[default_pos_idx] if unique_vals else None,
            )
        return ui.input_select(
            "sel_roc_pos_label",
            "Positive Label (1):",
            choices=[],
        )

    # --- ROC Comparison Mode UI ---
    @render.ui
    def ui_roc_truth_comp():
        """
        Render a dropdown for selecting the Gold Standard (binary) column from the current dataset.

        The control pre-selects a column whose name matches keywords like "gold" or "truth" when available.

        Returns:
            A UI select input element allowing selection of the gold-standard column (choices populated from available columns).
        """
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["gold", "truth"], default_to_first=True
        )
        return ui.input_select(
            "sel_roc_truth_comp",
            "Gold Standard (Binary):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_roc_test1():
        """
        Render a select input for choosing the reference test (Test 1) from the available dataset columns.

        Returns:
            A UI select input element (id "sel_roc_test1") labeled "Test 1 (Reference):" whose choices are the current dataset columns and whose default selection is the first column matching the keywords ["rapid", "standard", "test1", "score"], falling back to the first column if no keyword match is found.
        """
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["rapid", "standard", "test1", "score"], default_to_first=True
        )
        return ui.input_select(
            "sel_roc_test1", "Test 1 (Reference):", choices=cols, selected=default
        )

    @render.ui
    def ui_roc_test2():
        """
        Render a dropdown for selecting the second test (comparator) used in ROC comparison.

        The control lists all available dataframe columns and chooses a sensible default using prioritized keyword heuristics (preferring names like "expensive", "new", "test2", "score" and then fallback keywords), with a final fallback to the second column when no keyword match is found.

        Returns:
            ui.input_select: A select input element with id "sel_roc_test2", label "Test 2 (Comparator):", choices set to the available columns, and a heuristically chosen selected value.
        """
        cols = all_cols()
        # prioritized keywords for the second test
        default = select_variable_by_keyword(
            cols, ["expensive", "new", "test2", "score"], default_to_first=False
        )
        # If default matches test1 (highly likely if both match "score"),
        # try to find a different one if possible, but simplest is just keyword matching.
        # select_variable_by_keyword returns the first match.

        # Let's be specific for the example data
        if "Test_Score_Expensive" in cols:
            default = "Test_Score_Expensive"
        elif not default:
            # Try smart keyword matching
            default = select_variable_by_keyword(
                cols, ["score", "prob", "test", "expens"], default_to_first=False
            )

        if not default and len(cols) > 1:
            default = cols[1]

        return ui.input_select(
            "sel_roc_test2", "Test 2 (Comparator):", choices=cols, selected=default
        )

    @render.ui
    def ui_roc_pos_label_comp():
        """
        Render the positive-class selector for ROC comparison using the chosen gold-standard column.

        Constructs a UI control populated with candidate positive labels derived from the currently selected truth (gold-standard) column for the paired ROC comparison workflow.

        Returns:
            ui_element: A UI control allowing the user to choose the positive label for the ROC comparison.
        """
        return _render_pos_label_ui(
            input.sel_roc_truth_comp(), "sel_roc_pos_label_comp"
        )

    def _render_pos_label_ui(truth_col_name, input_id):
        """
        Render a dropdown UI for selecting the positive class label based on values in a dataset column.

        Examines the current dataset's column named by `truth_col_name`, collects its non-missing unique values (as strings), and produces a select input labeled "Positive Label:". If one of the common defaults ("1", "1.0", "Yes", "Positive") is present it will be preselected; otherwise the first value is selected. If the column is missing or has no values, an empty-choice select is returned.

        Parameters:
            truth_col_name (str): Name of the column in the current dataset to derive label choices from.
            input_id (str): Input identifier for the generated select widget.

        Returns:
            UI select input populated with the column's unique values and an appropriate default selection.
        """
        d = current_df()
        if d is not None and truth_col_name and truth_col_name in d.columns:
            vals = sorted([str(x) for x in d[truth_col_name].dropna().unique()])
            default = next(
                (x for x in vals if x in ["1", "1.0", "Yes", "Positive"]),
                vals[0] if vals else None,
            )
            return ui.input_select(
                input_id, "Positive Label:", choices=vals, selected=default
            )
        return ui.input_select(input_id, "Positive Label:", choices=[])

    # --- Chi-Square Inputs UI ---
    @render.ui
    def ui_chi_v1():
        """
        Render a dropdown for selecting Variable 1 (the exposure / row) for the chi-square analysis.

        The dropdown lists all available columns from the current dataset and defaults to "Treatment_Group" when that column exists; otherwise it selects the first column.

        Returns:
            The UI input select element for Variable 1 selection.
        """
        cols = all_cols()
        v1_idx = next((i for i, c in enumerate(cols) if c == "Treatment_Group"), 0)
        return ui.input_select(
            "sel_chi_v1",
            "Variable 1 (Exposure/Row):",
            choices=cols,
            selected=cols[v1_idx] if cols else None,
        )

    @render.ui
    def ui_chi_v2():
        cols = all_cols()
        v2_idx = next(
            (i for i, c in enumerate(cols) if c == "Outcome_Cured"),
            min(1, len(cols) - 1),
        )
        return ui.input_select(
            "sel_chi_v2",
            "Variable 2 (Outcome/Col):",
            choices=cols,
            selected=cols[v2_idx] if cols else None,
        )

    @render.ui
    def ui_chi_method():
        return ui.input_radio_buttons(
            "radio_chi_method",
            "Test Method:",
            {
                "Pearson (Standard)": "Pearson",
                "Yates' correction": "Yates",
                "Fisher's Exact Test": "Fisher",
            },
            inline=True,
        )

    @render.ui
    def ui_chi_caption():
        return ui.div(
            ui.markdown("*(Choose method in Tab 5 for guidance)*"),
            class_="text-secondary text-sm",
        )

    def get_pos_label_settings(
        df_input: pd.DataFrame, col_name: str
    ) -> tuple[list[str], int]:
        if df_input is None or col_name not in df_input.columns:
            return [], 0
        unique_vals = [str(x) for x in df_input[col_name].dropna().unique()]
        unique_vals.sort()
        default_idx = 0

        # Prioritize "1" then "1.0", then "0"
        if "1" in unique_vals:
            default_idx = unique_vals.index("1")
        elif "1.0" in unique_vals:
            default_idx = unique_vals.index("1.0")
        elif "0" in unique_vals:
            default_idx = unique_vals.index("0")

        return unique_vals, default_idx

    @render.ui
    def ui_chi_v1_pos():
        v1_col = input.sel_chi_v1()
        d = current_df()
        if d is None:
            return None
        v1_uv, v1_idx = get_pos_label_settings(d, v1_col)
        if not v1_uv:
            return ui.div(
                ui.markdown(f"⚠️ No values in {v1_col}"),
                class_="text-danger text-sm",
            )
        return ui.input_select(
            "sel_chi_v1_pos",
            f"Positive Label (Row: {v1_col}):",
            choices=v1_uv,
            selected=v1_uv[v1_idx] if v1_uv else None,
        )

    @render.ui
    def ui_chi_v2_pos():
        v2_col = input.sel_chi_v2()
        d = current_df()
        if d is None:
            return None
        v2_uv, v2_idx = get_pos_label_settings(d, v2_col)
        if not v2_uv:
            return ui.div(
                ui.markdown(f"⚠️ No values in {v2_col}"),
                class_="text-danger text-sm",
            )
        return ui.input_select(
            "sel_chi_v2_pos",
            f"Positive Label (Col: {v2_col}):",
            choices=v2_uv,
            selected=v2_uv[v2_idx] if v2_uv else None,
        )

    @render.ui
    def ui_chi_note():
        return ui.div(
            ui.markdown("*Select for Risk/OR calculation*"),
            class_="text-secondary text-sm",
        )

    @render.ui
    def ui_chi_empty():
        return ui.div()

    # --- Descriptive Inputs UI ---
    @render.ui
    def ui_desc_var():
        cols = all_cols()
        return ui.input_select("sel_desc_var", "Select Variable:", choices=cols)

    # --- ROC Status & Results ---
    @render.ui
    def ui_roc_status():
        if roc_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Generating ROC curve and statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_chi_status():
        if chi_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Calculating Chi-Square statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_desc_status():
        if desc_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Calculating descriptive statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    # --- ROC Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_roc)
    def _run_roc():
        """
        Run ROC analysis for the currently selected truth and score columns and produce an HTML report.

        Performs the analysis using the configured CI method and positive label, builds a report containing plots, statistics tables, performance-at-thresholds, and missing-data details when available, and stores the generated HTML in the module state (roc_html). While running, the processing flag (roc_processing) is set to True; on error an error alert HTML is stored in roc_html and the processing flag is cleared.
        """
        d = current_df()
        req(d is not None, input.sel_roc_truth(), input.sel_roc_score())

        # Set processing flag
        roc_processing.set(True)

        try:
            truth_col = input.sel_roc_truth()
            score_col = input.sel_roc_score()
            method = input.radio_roc_method()
            pos_label = input.sel_roc_pos_label()

            res, err, fig, coords = diag_test.analyze_roc(
                d,
                truth_col,
                score_col,
                method=method,
                pos_label_user=pos_label,
                var_meta=var_meta.get() or {},
            )

            if err:
                roc_html.set(f"<div class='alert alert-danger'>📄 Error: {err}</div>")
            else:
                rep = [
                    {
                        "type": "text",
                        "data": f"📊 Analysis: {score_col} vs {truth_col}",
                    },
                ]

                # Add plot if available
                if fig is not None:
                    rep.append({"type": "plot", "data": fig})
                else:
                    rep.append(
                        {
                            "type": "text",
                            "data": "⚠️ Warning: ROC plot could not be generated",
                        }
                    )

                # Add statistics table
                if res is not None:
                    # Filter out metadata from the main stats table
                    res_display = {
                        k: v
                        for k, v in res.items()
                        if k
                        not in [
                            "missing_data_info",
                            "calibration_plot",
                            "sens_spec_plot",
                        ]
                    }
                    rep.append(
                        {
                            "type": "table",
                            "header": "ROC Statistics",
                            "data": pd.DataFrame([res_display]).T,
                        }
                    )

                    # Add Calibration Plot if available
                    calib_plot = res.get("calibration_plot")
                    if calib_plot is not None:
                        rep.append(
                            {
                                "type": "plot",
                                "data": calib_plot,
                            }
                        )

                    # Add Sensitivity/Specificity Plot if available
                    ss_plot = res.get("sens_spec_plot")
                    if ss_plot is not None:
                        rep.append(
                            {
                                "type": "plot",
                                "data": ss_plot,
                            }
                        )

                # Add performance coordinates table
                if coords is not None:
                    rep.append(
                        {
                            "type": "table",
                            "header": "Performance at Different Thresholds",
                            "data": coords,
                        }
                    )

                # Missing Data Report
                if res and "missing_data_info" in res:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                res["missing_data_info"], var_meta.get() or {}
                            ),
                        }
                    )

                html_report = diag_test.generate_report(
                    f"ROC Analysis Report ({method})", rep
                )
                roc_html.set(html_report)

        except Exception as e:
            logger.exception("ROC analysis failed")
            roc_html.set(f"<div class='alert alert-danger'>📄 Error: {str(e)}</div>")
        finally:
            # Clear processing flag
            roc_processing.set(False)

    @render.ui
    def out_roc_results():
        if roc_html.get():
            return create_results_container(
                "ROC Results", ui.HTML(roc_html.get()), class_="fade-in-entry"
            )
        return ui.div(
            "Click 'Analyze ROC' to view results.",
            class_="text-secondary p-3",
        )

    @render.download(filename="roc_report.html")
    def btn_dl_roc_report():
        """
        Provide the generated ROC HTML report for download.

        Returns:
            str: The ROC report HTML content.
        """
        yield safe_download_html(roc_html.get(), label="ROC Report")

    # --- Download Status Badges ---
    @render.ui
    def dl_status_roc():
        return create_download_status_badge(bool(roc_html.get()))

    @render.ui
    def dl_status_chi():
        return create_download_status_badge(bool(chi_html.get()))

    @render.ui
    def dl_status_desc():
        return create_download_status_badge(bool(desc_html.get()))

    @render.ui
    def dl_status_dca():
        return create_download_status_badge(bool(dca_html.get()))

    @render.download(filename="chi2_report.html")
    def btn_dl_chi_report():
        """
        Provide the generated Chi-Square report for download.

        Returns:
            str: The HTML content of the Chi-Square analysis report.
        """
        yield safe_download_html(chi_html.get(), label="Chi-Square Report")

    # --- ACTION: Compare ROC Analysis ---
    @reactive.Effect
    @reactive.event(input.btn_compare_roc)
    def _run_roc_compare():
        """
        Run a paired ROC comparison using the DeLong test and store an HTML report.

        Validates selected gold-standard and two test score columns from the current dataset, filters out missing and non-numeric scores, and performs a paired DeLong test via DiagnosticComparison.delong_paired_test. Builds a Plotly ROC comparison figure (two ROC curves, diagonal reference, and optimal-threshold markers), constructs result tables (DeLong statistics and comparative metrics at the Youden optimal threshold), generates a combined HTML report via diag_test.generate_report, and stores the result in the module's `roc_html` reactive. Updates the `roc_processing` flag while work is in progress. If no valid data or insufficient points are found, or if the comparison fails, sets `roc_html` to an appropriate alert message.
        """
        d = current_df()
        req(
            d is not None,
            input.sel_roc_truth_comp(),
            input.sel_roc_test1(),
            input.sel_roc_test2(),
        )
        roc_processing.set(True)

        try:
            truth_col = input.sel_roc_truth_comp()
            test1_col = input.sel_roc_test1()
            test2_col = input.sel_roc_test2()
            pos_label = input.sel_roc_pos_label_comp()

            # --- Centralized Data Cleaning ---
            # We strictly enforce numeric type on test scores, but "truth" is handled inside specific logic if needed.
            # However, for DeLong, we need valid rows for all 3.
            required_cols = [truth_col, test1_col, test2_col]

            # Note: Truth column is binary, but might be string (Yes/No).
            # We only force numeric on the test scores.
            numeric_cols = [test1_col, test2_col]

            data_clean, missing_info = prepare_data_for_analysis(
                d,
                required_cols=required_cols,
                numeric_cols=numeric_cols,
                handle_missing="complete_case",
                var_meta=var_meta.get() or {},
            )

            if data_clean.empty:
                roc_html.set(
                    "<div class='alert alert-info'>No valid data found after cleaning.</div>"
                )
                return

            # Extract Cleaned Data
            y_true = data_clean[truth_col]
            s1 = data_clean[test1_col]
            s2 = data_clean[test2_col]

            if len(y_true) < 2:
                roc_html.set(
                    "<div class='alert alert-warning'>Not enough data points (need at least 2).</div>"
                )
                return

            # Run DeLong Test
            try:
                res = DiagnosticComparison.delong_paired_test(
                    y_true, s1, s2, pos_label=pos_label
                )
            except Exception as e:
                roc_html.set(
                    f"<div class='alert alert-danger'>Comparison failed: {e}</div>"
                )
                return

            # Generate Comparison Plot
            fig = go.Figure()

            # Curve 1
            fpr1, tpr1, _ = diag_test.roc_curve(
                (y_true.astype(str) == str(pos_label)).astype(int), s1
            )
            fig.add_trace(
                go.Scatter(
                    x=fpr1,
                    y=tpr1,
                    mode="lines",
                    name=f"{test1_col} (AUC={res['auc1']:.3f})",
                    line=dict(color=COLORS["primary"]),
                )
            )

            # Curve 2
            fpr2, tpr2, _ = diag_test.roc_curve(
                (y_true.astype(str) == str(pos_label)).astype(int), s2
            )
            fig.add_trace(
                go.Scatter(
                    x=fpr2,
                    y=tpr2,
                    mode="lines",
                    name=f"{test2_col} (AUC={res['auc2']:.3f})",
                    line=dict(color=COLORS["secondary"]),
                )
            )

            # Diagonal
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    showlegend=False,
                )
            )

            fig.update_layout(
                title=f"ROC Comparison: {test1_col} vs {test2_col}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                height=550,
                width=550,
                legend=dict(
                    x=0.6,
                    y=0.1,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="LightGrey",
                    borderwidth=1,
                ),
                xaxis=dict(constrain="domain"),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )

            # Results Table
            delong_table = pd.DataFrame(
                {
                    "Metric": [
                        "AUC (Test 1)",
                        "AUC (Test 2)",
                        "AUC Difference",
                        "Z-Score",
                        "P-value",
                        "95% CI of Diff",
                    ],
                    "Value": [
                        f"{res['auc1']:.4f}",
                        f"{res['auc2']:.4f}",
                        f"{res['diff']:.4f}",
                        f"{res['z_score']:.4f}",
                        diag_test.format_p_value(res["p_value"]),
                        f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]",
                    ],
                }
            )

            # Helper to get metrics row AND coords
            def get_best_metrics(score_data, label, color):
                """
                Constructs a metrics summary and a Plotly marker for the optimal Youden threshold of a diagnostic score.

                Parameters:
                    score_data (Sequence[float]): Numeric prediction scores for one test.
                    label (str): Display name for the test used in table and marker legend.
                    color (str): Color used for the marker on the ROC plot.

                Returns:
                    tuple: A pair (metrics_row, trace) where
                        - metrics_row (dict): A dictionary with keys "Test", "AUC", "Best Threshold", "Sensitivity",
                          "Specificity", "PPV", "NPV", and "Accuracy". Sensitivity and Specificity values include
                          their 95% confidence interval in the format "value [lower-upper]".
                        - trace (plotly.graph_objs._scatter.Scatter): A Plotly Scatter trace marking the optimal
                          point (FPR, TPR) on the ROC curve with hover text showing threshold, TPR, and FPR.
                """
                dt = DiagnosticTest(y_true, score_data, pos_label=pos_label)
                thresh, idx = dt.find_optimal_threshold(method="youden")
                m = dt.get_metrics_at_threshold(thresh)

                # Create marker trace
                fpr_val = dt.fpr[idx]
                tpr_val = dt.tpr[idx]

                trace = go.Scatter(
                    x=[fpr_val],
                    y=[tpr_val],
                    mode="markers",
                    name=f"Optimal {label}",
                    marker=dict(
                        size=12,
                        symbol="circle",
                        color=color,
                        line=dict(width=2, color="white"),
                    ),
                    hovertemplate=f"<b>{label} Optimal</b><br>Threshold: {thresh:.3f}<br>TPR: {tpr_val:.3f}<br>FPR: {fpr_val:.3f}<extra></extra>",
                )

                metrics_row = {
                    "Test": label,
                    "AUC": f"{dt.auc:.3f}",
                    "Best Threshold": f"{thresh:.3f}",
                    "Sensitivity": f"{m['sensitivity']:.3f} [{m['sensitivity_ci_lower']:.3f}-{m['sensitivity_ci_upper']:.3f}]",
                    "Specificity": f"{m['specificity']:.3f} [{m['specificity_ci_lower']:.3f}-{m['specificity_ci_upper']:.3f}]",
                    "PPV": f"{m['ppv']:.3f}",
                    "NPV": f"{m['npv']:.3f}",
                    "Accuracy": f"{m['accuracy']:.3f}",
                }
                return metrics_row, trace

            m1, t1 = get_best_metrics(s1, test1_col, "#d62728")
            m2, t2 = get_best_metrics(s2, test2_col, "#ff7f7f")

            fig.add_trace(t1)
            fig.add_trace(t2)

            metrics_df = pd.DataFrame([m1, m2])

            # Build Report
            rep = [
                {
                    "type": "text",
                    "data": f"📊 Comparison: {test1_col} vs {test2_col}",
                },
                {"type": "plot", "data": fig},
                {
                    "type": "table",
                    "header": "DeLong Correlation Test (Paired)",
                    "data": delong_table,
                },
                {
                    "type": "table",
                    "header": "Comparative Performance at Optimal Threshold (Youden)",
                    "data": metrics_df,
                },
            ]

            # Missing Data Report
            if missing_info:
                rep.append(
                    {
                        "type": "html",
                        "data": create_missing_data_report_html(
                            missing_info, var_meta.get() or {}
                        ),
                    }
                )

            roc_html.set(diag_test.generate_report("ROC Comparison Report", rep))

        except Exception as e:
            logger.exception("ROC Comparison Error")
            roc_html.set(f"<div class='alert alert-danger'>Error: {str(e)}</div>")
        finally:
            roc_processing.set(False)

    # --- Chi-Square Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_chi)
    def _run_chi():
        d = current_df()
        req(d is not None, input.sel_chi_v1(), input.sel_chi_v2())

        # Set processing flag
        chi_processing.set(True)

        try:
            tab, stats, msg, risk, missing_info = diag_test.calculate_chi2(
                d,
                input.sel_chi_v1(),
                input.sel_chi_v2(),
                method=input.radio_chi_method(),
                v1_pos=input.sel_chi_v1_pos(),
                v2_pos=input.sel_chi_v2_pos(),
                var_meta=var_meta.get() or {},
            )

            if tab is not None:
                status_text = (
                    f"Note: {msg.strip()}"
                    if msg.strip()
                    else "Analysis Status: Completed successfully."
                )
                rep = [
                    {
                        "type": "text",
                        "data": "Analysis: Diagnostic Test / Chi-Square",
                    },
                    {
                        "type": "text",
                        "data": f"Variables: {input.sel_chi_v1()} vs {input.sel_chi_v2()}",
                    },
                ]

                # Add status message (split if contains HTML block)
                rep.extend(_create_status_elements(status_text))

                rep.append(
                    {
                        "type": "contingency_table",
                        "header": "Contingency Table",
                        "data": tab,
                    }
                )
                if stats is not None:
                    rep.append(
                        {
                            "type": "table",
                            "header": "Statistics",
                            "data": stats,
                        }
                    )
                if risk is not None:
                    rep.append(
                        {
                            "type": "table",
                            "header": "Risk & Effect Measures",
                            "data": risk,
                        }
                    )

                # Missing Data Report
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )

                chi_html.set(
                    diag_test.generate_report(
                        f"Chi2: {input.sel_chi_v1()} vs {input.sel_chi_v2()}",
                        rep,
                    )
                )
            else:
                chi_html.set(
                    f"<div class='alert alert-danger'>Analysis failed: {msg}</div>"
                )

        except Exception as e:
            logger.exception("Chi-Square analysis failed")
            chi_html.set(f"<div class='alert alert-danger'>📄 Error: {str(e)}</div>")
        finally:
            # Clear processing flag
            chi_processing.set(False)

    @render.ui
    def out_chi_results():
        if chi_html.get():
            return create_results_container(
                "Chi-Square Results", ui.HTML(chi_html.get()), class_="fade-in-entry"
            )
        return ui.div("Results will appear here.", class_="text-secondary p-3")

    # --- Descriptive Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_run_desc)
    def _run_desc():
        d = current_df()
        req(d is not None, input.sel_desc_var())

        # Set processing flag
        desc_processing.set(True)

        try:
            res, missing_info = diag_test.calculate_descriptive(
                d,
                input.sel_desc_var(),
                var_meta=var_meta.get() or {},
            )

            if res is not None:
                rep = [{"type": "table", "data": res}]
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                desc_html.set(
                    diag_test.generate_report(
                        f"Descriptive: {input.sel_desc_var()}",
                        rep,
                    )
                )
            else:
                desc_html.set(
                    "<div class='alert alert-danger'>No data available for {}</div>".format(
                        input.sel_desc_var()
                    )
                )

        except Exception as e:
            logger.exception("Descriptive analysis failed")
            desc_html.set(f"<div class='alert alert-danger'>📄 Error: {str(e)}</div>")
        finally:
            # Clear processing flag
            desc_processing.set(False)

    @render.ui
    def out_desc_results():
        if desc_html.get():
            return create_results_container(
                "Descriptive Statistics",
                ui.HTML(desc_html.get()),
                class_="fade-in-entry",
            )
        return ui.div("Results will appear here.", class_="text-secondary p-3")

    @render.download(filename="descriptive_report.html")
    def btn_dl_desc_report():
        yield safe_download_html(desc_html.get(), label="Descriptive Statistics Report")

    # --- DCA Logic ---
    @render.ui
    def ui_dca_truth():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["gold", "standard", "outcome"], default_to_first=True
        )
        return ui.input_select(
            "sel_dca_truth",
            "Outcome (Truth):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_dca_prob():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["score", "prob", "pred"], default_to_first=True
        )
        return ui.input_select(
            "sel_dca_prob",
            "Prediction/Score (Probability):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_dca_status():
        if dca_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Calculating Decision Curve... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @reactive.Effect
    @reactive.event(input.btn_run_dca)
    def _run_dca():
        d = current_df()
        req(d is not None, input.sel_dca_truth(), input.sel_dca_prob())

        dca_processing.set(True)

        try:
            truth = input.sel_dca_truth()
            prob = input.sel_dca_prob()

            # Calculate Net Benefits
            nb_model, missing_info = decision_curve_lib.calculate_net_benefit(
                d,
                truth,
                prob,
                model_name="Current Model",
                var_meta=var_meta.get() or {},
            )
            if missing_info and "error" in missing_info:
                raise ValueError(missing_info["error"])
            nb_all = decision_curve_lib.calculate_net_benefit_all(d, truth)
            nb_none = decision_curve_lib.calculate_net_benefit_none()

            # Combine
            df_dca = pd.concat([nb_model, nb_all, nb_none])

            # Plot
            fig = decision_curve_lib.create_dca_plot(df_dca)

            # Report Generation
            df_disp = nb_model[
                ["threshold", "net_benefit", "tp_rate", "fp_rate"]
            ].copy()
            df_disp = df_disp[
                df_disp["threshold"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            ]

            rep = [
                {
                    "type": "text",
                    "data": f"Decision Curve Analysis: {prob} (Model) vs {truth} (Outcome)",
                },
                {"type": "plot", "data": fig},
                {
                    "type": "table",
                    "header": "Net Benefit (Model) at selected thresholds",
                    "data": df_disp,
                },
            ]

            # Missing Data Report
            if missing_info:
                rep.append(
                    {
                        "type": "html",
                        "data": create_missing_data_report_html(
                            missing_info, var_meta.get() or {}
                        ),
                    }
                )

            html_content = diag_test.generate_report(f"DCA: {prob} vs {truth}", rep)
            dca_html.set(html_content)

        except Exception as e:
            logger.exception("DCA analysis failed")
            dca_html.set(f"<div class='alert alert-danger'>Error: {str(e)}</div>")
        finally:
            dca_processing.set(False)

    @render.ui
    def out_dca_results():
        if dca_html.get():
            return create_results_container(
                "Decision Curve Analysis",
                ui.HTML(dca_html.get()),
                class_="fade-in-entry",
            )
        return ui.div("Click 'Run DCA' to view results.", class_="text-secondary p-3")

    @render.download(filename="dca_report.html")
    def btn_dl_dca_report():
        yield safe_download_html(dca_html.get(), label="DCA Report")

    # --- PDF Download Handlers ---
    @render.download(filename="roc_report.pdf")
    def btn_dl_roc_pdf():
        yield safe_download_pdf(roc_html.get(), label="ROC Report")

    @render.download(filename="chi2_report.pdf")
    def btn_dl_chi_pdf():
        yield safe_download_pdf(chi_html.get(), label="Chi-Square Report")

    @render.download(filename="descriptive_report.pdf")
    def btn_dl_desc_pdf():
        yield safe_download_pdf(desc_html.get(), label="Descriptive Statistics Report")

    @render.download(filename="dca_report.pdf")
    def btn_dl_dca_pdf():
        yield safe_download_pdf(dca_html.get(), label="DCA Report")

    # =========================================================================
    # 🧭 FAGAN'S NOMOGRAM SERVER LOGIC
    # =========================================================================

    fagan_html = reactive.Value("")
    fagan_status_msg = reactive.Value("")

    @render.ui
    def ui_fagan_truth():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols,
            ["gold", "outcome", "diag", "status", "disease"],
            default_to_first=True,
        )
        return ui.input_select(
            "sel_fagan_truth",
            "Outcome (Gold Standard):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_fagan_score():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["score", "test", "pred", "prob", "biomarker"], default_to_first=True
        )
        return ui.input_select(
            "sel_fagan_score",
            "Diagnostic Test / Score:",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_fagan_pos_label():
        d = current_df()
        truth = input.sel_fagan_truth()
        if d is not None and truth in d.columns:
            uniques = [str(x) for x in d[truth].dropna().unique()]
            default = uniques[0] if uniques else "1"
            return ui.input_select(
                "sel_fagan_pos_label",
                "Positive Class:",
                choices=uniques,
                selected=default,
            )
        return ui.input_text("sel_fagan_pos_label", "Positive Class:", value="1")

    @render.ui
    def ui_fagan_multi_truth():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["gold", "outcome", "diag", "disease"], default_to_first=True
        )
        return ui.input_select(
            "sel_fagan_multi_truth",
            "Outcome (Gold Standard):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_fagan_multi_pos_label():
        d = current_df()
        truth = input.sel_fagan_multi_truth()
        if d is not None and truth in d.columns:
            uniques = [str(x) for x in d[truth].dropna().unique()]
            default = uniques[0] if uniques else "1"
            return ui.input_select(
                "sel_fagan_multi_pos_label",
                "Positive Class:",
                choices=uniques,
                selected=default,
            )
        return ui.input_text("sel_fagan_multi_pos_label", "Positive Class:", value="1")

    @render.ui
    def ui_fagan_multi_score():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["trop", "dimer", "score", "prob", "biomarker"], default_to_first=True
        )
        return ui.input_select(
            "sel_fagan_multi_score",
            "Biomarker / Score Column:",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_fagan_status():
        msg = fagan_status_msg.get()
        if not msg:
            return None
        alert_type = "alert-success" if "✅" in msg else "alert-danger"
        return ui.div(msg, class_=f"alert {alert_type} py-2 mb-3")

    @reactive.Effect
    @reactive.event(input.btn_generate_fagan)
    def _generate_fagan():
        mode = input.fagan_mode()

        try:
            multilevel_list = None
            interval_df = None

            if mode == "standalone":
                pre_prob = float(input.fagan_manual_pre_prob() or 20.0) / 100.0
                lr_pos = float(input.fagan_manual_lr_pos() or 10.0)
                lr_neg = float(input.fagan_manual_lr_neg() or 0.10)
                test_name = str(input.fagan_manual_test_name() or "Bedside Test")
            elif mode == "dataset":
                d = current_df()
                req(d is not None, input.sel_fagan_truth(), input.sel_fagan_score())
                truth_col = input.sel_fagan_truth()
                score_col = input.sel_fagan_score()
                pos_val = input.sel_fagan_pos_label() or "1"
                pre_prob = float(input.fagan_ds_pre_prob() or 20.0) / 100.0

                # Analyze ROC to get optimal cutoff & LR+/LR-
                roc_res, _ = diag_test.analyze_roc(
                    d,
                    truth_col,
                    score_col,
                    pos_label=pos_val,
                    optimize_by="youden",
                    var_meta=var_meta.get() or {},
                )
                if not roc_res:
                    raise ValueError("Failed to compute ROC metrics for dataset.")

                lr_pos = float(roc_res.get("lr_positive", 10.0))
                lr_neg = float(roc_res.get("lr_negative", 0.10))
                sens = roc_res.get("sensitivity", 0.8) * 100.0
                spec = roc_res.get("specificity", 0.8) * 100.0
                cutoff = roc_res.get("optimal_cutoff", 0.0)
                test_name = f"{score_col} (Cutoff: {cutoff:.2f} | Sens: {sens:.1f}%, Spec: {spec:.1f}%)"
            else:  # Multilevel
                d = current_df()
                req(
                    d is not None,
                    input.sel_fagan_multi_truth(),
                    input.sel_fagan_multi_score(),
                    input.fagan_multi_cutoffs(),
                )
                truth_col = input.sel_fagan_multi_truth()
                score_col = input.sel_fagan_multi_score()
                pre_prob = float(input.fagan_multi_pre_prob() or 20.0) / 100.0
                raw_cutoffs = [
                    float(x.strip())
                    for x in input.fagan_multi_cutoffs().split(",")
                    if x.strip()
                ]

                pos_val = input.sel_fagan_multi_pos_label() or "1"
                interval_df = fagan_nomogram_lib.calculate_multilevel_likelihood_ratios(
                    d,
                    outcome_col=truth_col,
                    score_col=score_col,
                    cutoffs=raw_cutoffs,
                    pos_label=pos_val,
                )
                multilevel_list = []
                for _, r in interval_df.iterrows():
                    multilevel_list.append(
                        {
                            "name": str(r["Tier / Interval"]),
                            "lr": float(r["Interval LR"]),
                        }
                    )

                lr_pos = float(interval_df.iloc[-1]["Interval LR"])
                lr_neg = float(interval_df.iloc[0]["Interval LR"])
                test_name = f"{score_col} (Multi-Tier Likelihood Ratios)"

            # Generate interactive Plotly nomogram
            fig_nomo = fagan_nomogram_lib.create_fagan_nomogram_plot(
                pre_test_prob=pre_prob,
                lr_pos=lr_pos,
                lr_neg=lr_neg,
                test_name=test_name,
                multilevel_lrs=multilevel_list,
            )
            nomo_plot_html = fig_nomo.to_html(include_plotlyjs="cdn", full_html=False)

            # Calculate outcome probability cards
            pos_card = fagan_nomogram_lib.calculate_post_test_probability(
                pre_prob, lr_pos
            )
            neg_card = fagan_nomogram_lib.calculate_post_test_probability(
                pre_prob, lr_neg
            )

            # Build HTML Summary
            cards_html = f"""
            <div class="row g-3 mb-4">
                <div class="col-md-4">
                    <div class="p-3 bg-light rounded border text-center h-100">
                        <span class="text-muted d-block small fw-semibold">PRE-TEST PROBABILITY</span>
                        <span class="fs-2 fw-bold text-dark">{pre_prob * 100:.1f}%</span>
                        <div class="small text-muted mt-1">Clinical Pre-Test Suspicion (Prior)</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-3 rounded border text-center h-100" style="background-color: #FEF2F2; border-color: #F87171 !important;">
                        <span class="text-danger d-block small fw-semibold">POST-TEST IF POSITIVE (LR+ = {lr_pos:.2f})</span>
                        <span class="fs-2 fw-bold text-danger">{pos_card["post_test_prob_pct"]:.1f}%</span>
                        <div class="small fw-semibold mt-1" style="color: {pos_card["zone_color"]};">{pos_card["zone"]}</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-3 rounded border text-center h-100" style="background-color: #ECFDF5; border-color: #34D399 !important;">
                        <span class="text-success d-block small fw-semibold">POST-TEST IF NEGATIVE (LR- = {lr_neg:.2f})</span>
                        <span class="fs-2 fw-bold text-success">{neg_card["post_test_prob_pct"]:.1f}%</span>
                        <div class="small fw-semibold mt-1" style="color: {neg_card["zone_color"]};">{neg_card["zone"]}</div>
                    </div>
                </div>
            </div>
            """

            # Multi-level Interval Table HTML if applicable
            interval_table_html = ""
            if interval_df is not None:
                interval_table_html = f"""
                <div class="card mb-4 border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="card-title text-primary fw-bold">🏷️ Multi-Level Interval Likelihood Ratios ({score_col})</h6>
                        <div class="table-responsive">
                            <table class="table table-sm table-bordered align-middle mb-0">
                                <thead class="table-light">
                                    <tr>
                                        <th>Biomarker Tier</th>
                                        <th>Diseased (D+)</th>
                                        <th>Non-Diseased (D-)</th>
                                        <th>Interval LR</th>
                                        <th>95% CI</th>
                                        <th>Post-test Prob @ {pre_prob * 100:.0f}% Prior</th>
                                    </tr>
                                </thead>
                                <tbody>
                """
                for _, r in interval_df.iterrows():
                    tier_post = fagan_nomogram_lib.calculate_post_test_probability(
                        pre_prob, r["Interval LR"]
                    )
                    interval_table_html += f"""
                                    <tr>
                                        <td><strong>{r["Tier / Interval"]}</strong></td>
                                        <td>{r["Diseased (D+)"]}</td>
                                        <td>{r["Non-Diseased (D-)"]}</td>
                                        <td><span class="badge bg-primary text-white fs-6">{r["Interval LR"]:.2f}</span></td>
                                        <td>{r["95% CI Lower"]:.2f} – {r["95% CI Upper"]:.2f}</td>
                                        <td><strong>{tier_post["post_test_prob_pct"]:.1f}%</strong> <small class="text-muted">({tier_post["zone"]})</small></td>
                                    </tr>
                    """
                interval_table_html += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                """

            full_html = f"""
            <div class="fagan-nomogram-container">
                {cards_html}
                {interval_table_html}
                <div class="card border-0 shadow-sm p-3 mb-3">
                    {nomo_plot_html}
                </div>
            </div>
            """
            fagan_html.set(full_html)
            fagan_status_msg.set("✅ Fagan's Nomogram successfully computed.")
        except Exception as e:
            logger.exception("Fagan's nomogram generation failed")
            fagan_status_msg.set(f"❌ Error generating nomogram: {str(e)}")

    @render.ui
    def out_fagan_results():
        content = fagan_html.get()
        if content:
            return create_results_container(
                "🧭 Interactive Fagan's Nomogram Analysis",
                ui.HTML(content),
                class_="fade-in-entry",
            )
        return ui.div(
            "Click 'Generate Nomogram' to view bedside Bayes probability analysis.",
            class_="text-secondary p-3",
        )

    @render.download(filename="fagan_nomogram_report.html")
    def btn_dl_fagan_report():
        yield safe_download_html(fagan_html.get(), label="Fagan Nomogram Report")

    @render.download(filename="fagan_nomogram_report.pdf")
    def btn_dl_fagan_pdf():
        yield safe_download_pdf(fagan_html.get(), label="Fagan Nomogram Report")
