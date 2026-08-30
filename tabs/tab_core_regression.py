from __future__ import annotations

import gc
import html
import json
import numbers
from itertools import combinations, islice

# Use built-in list/dict/tuple for Python 3.9+ and typing for complex types
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from shiny import module, reactive, render, req, ui

from config import CONFIG
from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from tabs._dataset_mixin import register_dataset_selector
from tabs._styling import get_shiny_css
from utils.calibration_lib import (
    create_calibration_plot,
    create_decision_curve,
    format_calibration_html,
    get_calibration_report,
)
from utils.data_cleaning import prepare_data_for_analysis
from utils.download_helpers import (
    safe_data_download,
    safe_download_html,
    safe_report_generation,
)
from utils.forest_plot_lib import create_forest_plot
from utils.formatting import (
    PublicationFormatter,
    create_missing_data_report_html,
    format_p_value,
)
from utils.linear_lib import (
    analyze_linear_outcome,
    bootstrap_ols,
    format_bootstrap_results,
    format_stepwise_history,
    stepwise_selection,
)
from utils.logic import (
    analyze_outcome,
    calculate_absolute_risk,
    calculate_nnt,
    format_absolute_risk_html,
    generate_mi_pooled_report,
    run_glm,
)
from utils.mi_helpers import get_mi_datasets, has_mi_datasets
from utils.multiple_imputation import pool_estimates
from utils.pdf_helpers import safe_download_pdf, safe_pdf_report_generation
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.poisson_lib import analyze_poisson_outcome
from utils.repeated_measures_lib import (
    create_trajectory_plot,
    extract_model_results,
    run_gee,
    run_lmm,
)
from utils.reporting_checklists import (
    auto_populate_strobe,
    format_strobe_html_compact,
    generate_checklist_markdown,
)
from utils.sensitivity_lib import calculate_e_value
from utils.subgroup_analysis_module import SubgroupAnalysisLogit, SubgroupResult
from utils.ui_helpers import (
    create_download_status_badge,
    create_empty_state_ui,
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
    create_skeleton_loader_ui,
    create_tooltip_label,
)

# Import internal modules


logger = get_logger(__name__)
COLORS = get_color_palette()


# ==============================================================================
# Helper Functions (Pure Logic)
# ==============================================================================
def _build_strobe_metadata(
    res: dict[str, Any],
    d: pd.DataFrame | None,
    outcome_name: str | None,
) -> dict[str, Any]:
    """Helper to build STROBE metadata for logistic regression."""
    aor_res = res.get("aor_res", {})
    n_total = len(d) if d is not None else 0
    n_analyzed = (
        len(res.get("y_true", [])) if res.get("y_true") is not None else n_total
    )

    return {
        "n_total": n_total,
        "n_analyzed": n_analyzed,
        "outcome_name": outcome_name or "Unknown",
        "predictors": list(aor_res.keys()) if aor_res else [],
        "has_missing_report": True,  # We always include missing report
        "has_ci": True,
        "method": "logistic",  # Default method
        "has_sensitivity": bool(aor_res),  # E-values are calculated
        "has_subgroup": False,  # Could check if subgroup was run
    }


def check_perfect_separation(df: pd.DataFrame, target_col: str) -> list[str]:
    try:
        from firthmodels import detect_separation
    except ImportError:
        return []

    risky_vars: list[str] = []
    try:
        y_raw = df[target_col].dropna()
        unique_values = sorted(y_raw.unique(), key=str)
        if len(unique_values) != 2:
            return []

        if set(unique_values).issubset({0, 1}):
            y = y_raw.astype(int)
        else:
            y = y_raw.map({unique_values[0]: 0, unique_values[1]: 1}).astype(int)

        X = df.drop(columns=[target_col]).loc[y.index]
        X_num = pd.get_dummies(X, drop_first=True, dtype=float)

        sep_result = detect_separation(X_num.values, y.values)
        is_separated = False
        if hasattr(sep_result, "separation"):
            is_separated = sep_result.separation
        else:
            is_separated = bool(sep_result)

        if is_separated:
            risky_vars.append("Data Separation Detected (Konis LP)")
    except Exception as e:
        logger.debug("detect_separation failed; using fallback heuristic: %s", e)
        if "X" in locals() and "y" in locals():
            for col in X.columns:
                try:
                    ct = pd.crosstab(X[col], y)
                    if (ct == 0).any().any():
                        risky_vars.append(col)
                except (ValueError, TypeError):
                    continue

    return risky_vars


# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def core_regression_ui() -> ui.TagChild:
    """
    Builds the main user interface for the core regression module.

    Provides the dataset selector and info header plus a tabbed interface with controls, actions, and result panels for:
    Binary Outcomes (logistic), Subgroup Analysis (logit), Count & Special (Poisson, Negative Binomial, GLM), Continuous Outcomes (linear), Repeated Measures (GEE/LMM), and a Reference guide.

    Returns:
        ui.TagChild: A UI fragment containing the dataset selector/info and the tabbed analysis panels with inputs, run/download controls, and result containers.
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
        # Main Analysis Tabs
        ui.navset_tab(
            # =====================================================================
            # TAB 1: Binary Outcomes (formerly Binary Logistic)
            # =====================================================================
            ui.nav_panel(
                "📈 Binary Outcomes",
                # Control section (top)
                ui.card(
                    ui.card_header("📈 Analysis Options"),
                    ui.layout_columns(
                        create_input_group(
                            "Variable Selection",
                            ui.input_select(
                                "sel_outcome",
                                create_tooltip_label(
                                    "Select Outcome (Y)",
                                    "Must be binary (0/1 or Yes/No).",
                                ),
                                choices=[],
                            ),
                            ui.output_ui("ui_separation_warning"),
                            type="required",
                        ),
                        create_input_group(
                            "Method & Settings",
                            ui.input_radio_buttons(
                                "radio_method",
                                "Regression Method:",
                                {
                                    "auto": "Auto (Recommended)",
                                    "bfgs": "Standard (MLE)",
                                    "firth": "Firth's (Penalized)",
                                },
                            ),
                            ui.panel_conditional(
                                "input.radio_method === 'firth'",
                                ui.input_slider(
                                    "firth_penalty_weight",
                                    "Firth Penalty Weight:",
                                    min=0.0,
                                    max=2.0,
                                    value=1.0,
                                    step=0.1,
                                ),
                                ui.tags.small(
                                    "1.0 = Standard Firth | 0 = Unpenalized MLE | >1 = Stronger bias-reduction",
                                    class_="text-muted",
                                ),
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_logit_validation"),
                    ui.div(
                        create_input_group(
                            "Advanced Adjustments",
                            create_tooltip_label(
                                "Exclude Variables",
                                "Remove specific variables from the model.",
                            ),
                            ui.input_selectize(
                                "sel_exclude",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            # Interaction Pairs selector
                            ui.h6("🔗 Interaction Pairs:"),
                            ui.input_selectize(
                                "sel_interactions",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={
                                    "placeholder": "Select variable pairs to test interactions...",
                                    "plugins": ["remove_button"],
                                },
                            ),
                            ui.hr(),
                            ui.input_selectize(
                                "sel_exposure",
                                create_tooltip_label(
                                    "Exposure for AR/NNT",
                                    "Select the primary exposure variable for Absolute Risk calculation.",
                                ),
                                choices=[],
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            type="optional",
                        )
                    ),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_logit",
                            "🚀 Run Regression",
                            class_="btn-primary btn-sm w-100",
                        ),
                        ui.div(
                            ui.download_button(
                                "btn_dl_report",
                                "📥 HTML",
                                class_="btn-secondary btn-sm w-100",
                            ),
                            ui.download_button(
                                "btn_dl_report_pdf",
                                "📥 PDF",
                                class_="btn-outline-danger btn-sm w-100 mt-1",
                            ),
                            ui.output_ui("dl_status_logit"),
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                # Content section (bottom)
                ui.output_ui("out_logit_status"),
                create_results_container(
                    "Analysis Results", ui.output_ui("ui_logit_results_area")
                ),
            ),
            # =====================================================================
            # TAB 1.5: Subgroup Analysis (Logit)
            # =====================================================================
            ui.nav_panel(
                "🔛 Subgroup Analysis",
                ui.card(
                    ui.card_header("Binary Logistic Subgroup Analysis - Heterogeneity"),
                    ui.layout_columns(
                        create_input_group(
                            "Variables",
                            ui.input_select(
                                "sg_logit_outcome",
                                create_tooltip_label(
                                    "Outcome (Y)", "Must be binary (0/1 or Yes/No)."
                                ),
                                choices=["Select..."],
                            ),
                            ui.input_select(
                                "sg_logit_treatment",
                                create_tooltip_label(
                                    "Treatment/Exposure",
                                    "Primary variable of interest.",
                                ),
                                choices=["Select..."],
                            ),
                            create_input_group(
                                "Stratification & Adjustment",
                                ui.input_select(
                                    "sg_logit_subgroup",
                                    create_tooltip_label(
                                        "Stratify By",
                                        "Categorical variable defining subgroups.",
                                    ),
                                    choices=["Select..."],
                                ),
                                ui.input_selectize(
                                    "sg_logit_adjust",
                                    create_tooltip_label(
                                        "Adjustment Variables",
                                        "Covariates to adjust for within subgroups.",
                                    ),
                                    choices=[],
                                    multiple=True,
                                    width="100%",
                                    options={"plugins": ["remove_button"]},
                                ),
                                type="required",
                            ),
                            type="required",
                        ),
                        col_widths=[12],
                    ),
                    ui.accordion(
                        ui.accordion_panel(
                            "⚠️ Advanced Settings",
                            create_input_group(
                                "Minimum Counts",
                                ui.input_numeric(
                                    "sg_logit_min_n",
                                    "Min N per subgroup:",
                                    value=10,
                                    min=5,
                                    max=100,
                                ),
                                type="advanced",
                            ),
                        ),
                        open=False,
                    ),
                    ui.output_ui("out_sg_logit_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_sg_logit",
                            "🚀 Run Subgroup Analysis",
                            class_="btn-primary w-100",
                        ),
                        ui.div(
                            ui.download_button(
                                "btn_dl_sg_logit",
                                "📥 HTML",
                                class_="btn-secondary w-100",
                            ),
                            ui.download_button(
                                "btn_dl_sg_logit_pdf",
                                "📥 PDF",
                                class_="btn-outline-danger w-100 mt-1",
                            ),
                            ui.output_ui("dl_status_sg_logit"),
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.output_ui("out_sg_logit_status"),
                create_results_container(
                    "Subgroup Analysis Results", ui.output_ui("out_sg_logit_result")
                ),
            ),
            # =====================================================================
            # TAB 3: Count & Special (formerly Poisson & GLM)
            # =====================================================================
            # =====================================================================
            # TAB 3: Count & Special (formerly Poisson & GLM)
            # =====================================================================
            ui.nav_panel(
                "🔢 Count & Special",
                ui.navset_card_tab(
                    ui.nav_panel(
                        "📊 Poisson Regression",
                        # Control section (top)
                        ui.card(
                            ui.card_header("📊 Poisson Analysis Options"),
                            ui.layout_columns(
                                create_input_group(
                                    "Variable Selection",
                                    ui.input_select(
                                        "poisson_outcome",
                                        create_tooltip_label(
                                            "Select Count Outcome (Y)",
                                            "Outcome must be positive integers.",
                                        ),
                                        choices=[],
                                    ),
                                    ui.input_select(
                                        "poisson_offset",
                                        create_tooltip_label(
                                            "Offset Column",
                                            "Use for rate calculations (e.g., person-years).",
                                        ),
                                        choices=["None"],
                                    ),
                                    type="required",
                                ),
                                create_input_group(
                                    "Advanced Settings",
                                    create_tooltip_label(
                                        "Exclude Variables",
                                        "Remove specific variables from the model.",
                                    ),
                                    ui.input_selectize(
                                        "poisson_exclude",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={"plugins": ["remove_button"]},
                                    ),
                                    type="advanced",
                                ),
                                col_widths=[6, 6],
                            ),
                            # Interaction Pairs selector
                            ui.div(
                                create_input_group(
                                    "Model Refinement",
                                    ui.h6("🔗 Interaction Pairs:"),
                                    ui.input_selectize(
                                        "poisson_interactions",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={
                                            "placeholder": "Select variable pairs to test interactions...",
                                            "plugins": ["remove_button"],
                                        },
                                    ),
                                    type="optional",
                                )
                            ),
                            ui.output_ui("out_poisson_validation"),
                            ui.hr(),
                            ui.layout_columns(
                                ui.input_action_button(
                                    "btn_run_poisson",
                                    "🚀 Run Poisson Regression",
                                    class_="btn-primary btn-sm w-100",
                                ),
                                ui.div(
                                    ui.download_button(
                                        "btn_dl_poisson_report",
                                        "📥 HTML",
                                        class_="btn-secondary btn-sm w-100",
                                    ),
                                    ui.download_button(
                                        "btn_dl_poisson_pdf",
                                        "📥 PDF",
                                        class_="btn-outline-danger btn-sm w-100 mt-1",
                                    ),
                                    ui.output_ui("dl_status_poisson"),
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        # Content section (bottom)
                        ui.output_ui("out_poisson_status"),
                        create_results_container(
                            "Poisson Results", ui.output_ui("ui_poisson_results_area")
                        ),
                    ),
                    ui.nav_panel(
                        "📉 Negative Binomial",
                        ui.card(
                            ui.card_header("📉 Negative Binomial Analysis Options"),
                            ui.layout_columns(
                                create_input_group(
                                    "Variable Selection",
                                    ui.input_select(
                                        "nb_outcome",
                                        create_tooltip_label(
                                            "Select Count Outcome (Y)",
                                            "Use when data is overdispersed (variance > mean).",
                                        ),
                                        choices=[],
                                    ),
                                    ui.input_select(
                                        "nb_offset",
                                        create_tooltip_label(
                                            "Offset Column",
                                            "Use for rate calculations.",
                                        ),
                                        choices=["None"],
                                    ),
                                    type="required",
                                ),
                                create_input_group(
                                    "Advanced Settings",
                                    create_tooltip_label(
                                        "Exclude Variables",
                                        "Remove specific variables.",
                                    ),
                                    ui.input_selectize(
                                        "nb_exclude",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={"plugins": ["remove_button"]},
                                    ),
                                    type="advanced",
                                ),
                                col_widths=[6, 6],
                            ),
                            ui.div(
                                create_input_group(
                                    "Model Refinement",
                                    ui.h6("🔗 Interaction Pairs:"),
                                    ui.input_selectize(
                                        "nb_interactions",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={
                                            "placeholder": "Select variable pairs to test interactions...",
                                            "plugins": ["remove_button"],
                                        },
                                    ),
                                    type="optional",
                                )
                            ),
                            ui.output_ui("out_nb_validation"),
                            ui.hr(),
                            ui.layout_columns(
                                ui.input_action_button(
                                    "btn_run_nb",
                                    "🚀 Run Negative Binomial",
                                    class_="btn-primary btn-sm w-100",
                                ),
                                ui.div(
                                    ui.download_button(
                                        "btn_dl_nb_report",
                                        "📥 HTML",
                                        class_="btn-secondary btn-sm w-100",
                                    ),
                                    ui.download_button(
                                        "btn_dl_nb_pdf",
                                        "📥 PDF",
                                        class_="btn-outline-danger btn-sm w-100 mt-1",
                                    ),
                                    ui.output_ui("dl_status_nb"),
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        ui.output_ui("out_nb_status"),
                        create_results_container(
                            "Negative Binomial Results",
                            ui.output_ui("ui_nb_results_area"),
                        ),
                    ),
                    ui.nav_panel(
                        "📈 Generalized Linear Model",
                        ui.card(
                            ui.card_header("📈 GLM Options"),
                            ui.layout_columns(
                                create_input_group(
                                    "Variable Selection",
                                    ui.input_select(
                                        "glm_outcome",
                                        create_tooltip_label(
                                            "Outcome (Y)",
                                            "Dependent variable for the model.",
                                        ),
                                        choices=[],
                                    ),
                                    ui.h6("Distribution & Link:"),
                                    ui.input_select(
                                        "glm_family",
                                        "Family:",
                                        {
                                            "Gaussian": "Gaussian (Continuous)",
                                            "Binomial": "Binomial (Binary 0/1)",
                                            "Poisson": "Poisson (Count)",
                                            "Gamma": "Gamma (Continuous +)",
                                            "InverseGaussian": "Inverse Gaussian",
                                        },
                                    ),
                                    ui.input_select(
                                        "glm_link",
                                        "Link Function:",
                                        {
                                            "identity": "Identity",
                                            "log": "Log",
                                            "logit": "Logit",
                                            "probit": "Probit",
                                            "cloglog": "Cloglog",
                                            "inverse_power": "Inverse",
                                            "sqrt": "Sqrt",
                                        },
                                    ),
                                    type="required",
                                ),
                                create_input_group(
                                    "Predictors",
                                    ui.input_selectize(
                                        "glm_predictors",
                                        create_tooltip_label(
                                            "Select Predictors (X)",
                                            "Independent variables.",
                                        ),
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={"plugins": ["remove_button"]},
                                    ),
                                    ui.input_selectize(
                                        "glm_interactions",
                                        "Interactions:",
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={
                                            "placeholder": "Select variable pairs...",
                                            "plugins": ["remove_button"],
                                        },
                                    ),
                                    type="required",
                                ),
                                col_widths=[6, 6],
                            ),
                            ui.output_ui("out_glm_validation"),
                            ui.hr(),
                            ui.layout_columns(
                                ui.input_action_button(
                                    "btn_run_glm",
                                    "🚀 Run GLM",
                                    class_="btn-primary btn-sm w-100",
                                ),
                                ui.div(
                                    ui.download_button(
                                        "btn_dl_glm_report",
                                        "📥 HTML",
                                        class_="btn-secondary btn-sm w-100",
                                    ),
                                    ui.download_button(
                                        "btn_dl_glm_pdf",
                                        "📥 PDF",
                                        class_="btn-outline-danger btn-sm w-100 mt-1",
                                    ),
                                    ui.output_ui("dl_status_glm"),
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        create_results_container(
                            "GLM Results", ui.output_ui("ui_glm_results_area")
                        ),
                    ),
                ),
            ),
            # =====================================================================
            # TAB 2: Continuous Outcomes (formerly Linear & Diagnostics)
            # =====================================================================
            ui.nav_panel(
                "📉 Continuous Outcomes",
                # Control section (top)
                ui.card(
                    ui.card_header("📐 Linear Regression Options"),
                    ui.layout_columns(
                        create_input_group(
                            "Variable Selection",
                            ui.input_select(
                                "linear_outcome",
                                create_tooltip_label(
                                    "Continuous Outcome (Y)",
                                    "Must be numeric/continuous.",
                                ),
                                choices=[],
                            ),
                            ui.input_selectize(
                                "linear_predictors",
                                create_tooltip_label(
                                    "Predictors (X)", "Independent variables."
                                ),
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={
                                    "placeholder": "Select predictors or leave empty for auto-selection...",
                                    "plugins": ["remove_button"],
                                },
                            ),
                            ui.p(
                                "💡 Leave predictors empty to auto-include all numeric variables",
                                style="font-size: 0.8em; color: #666; margin-top: 4px;",
                            ),
                            type="required",
                        ),
                        create_input_group(
                            "Method & Settings",
                            ui.input_radio_buttons(
                                "linear_method",
                                "Regression Method:",
                                {"ols": "Standard OLS", "robust": "Robust (Huber)"},
                                selected="ols",
                            ),
                            ui.input_checkbox(
                                "linear_robust_se",
                                "Use Robust Standard Errors (HC3)",
                                value=False,
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    # Advanced Options Accordion
                    ui.accordion(
                        ui.accordion_panel(
                            "🔧 Advanced Options",
                            ui.layout_columns(
                                create_input_group(
                                    "Stepwise Selection",
                                    ui.input_checkbox(
                                        "linear_stepwise_enable",
                                        "Enable Stepwise Selection",
                                        value=False,
                                    ),
                                    ui.input_radio_buttons(
                                        "linear_stepwise_dir",
                                        "Direction:",
                                        {
                                            "both": "Both",
                                            "forward": "Forward",
                                            "backward": "Backward",
                                        },
                                        selected="both",
                                        inline=True,
                                    ),
                                    ui.input_radio_buttons(
                                        "linear_stepwise_crit",
                                        "Criterion:",
                                        {
                                            "aic": "AIC",
                                            "bic": "BIC",
                                            "pvalue": "P-value",
                                        },
                                        selected="aic",
                                        inline=True,
                                    ),
                                    type="advanced",
                                ),
                                create_input_group(
                                    "Bootstrap CI",
                                    ui.input_checkbox(
                                        "linear_bootstrap_enable",
                                        "Enable Bootstrap CIs",
                                        value=False,
                                    ),
                                    ui.input_numeric(
                                        "linear_bootstrap_n",
                                        "Bootstrap Samples:",
                                        value=1000,
                                        min=100,
                                        max=10000,
                                    ),
                                    ui.input_radio_buttons(
                                        "linear_bootstrap_method",
                                        "CI Method:",
                                        {"percentile": "Percentile", "bca": "BCa"},
                                        selected="percentile",
                                        inline=True,
                                    ),
                                    type="advanced",
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        open=False,
                    ),
                    ui.div(
                        create_input_group(
                            "Ad Hoc Exclusions",
                            create_tooltip_label(
                                "Exclude Variables", "Remove specific variables."
                            ),
                            ui.input_selectize(
                                "linear_exclude",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            type="optional",
                        )
                    ),
                    ui.output_ui("out_linear_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_linear",
                            "🚀 Run Linear Regression",
                            class_="btn-primary btn-sm w-100",
                        ),
                        ui.div(
                            ui.download_button(
                                "btn_dl_linear_report",
                                "📥 HTML",
                                class_="btn-secondary btn-sm w-100",
                            ),
                            ui.download_button(
                                "btn_dl_linear_pdf",
                                "📥 PDF",
                                class_="btn-outline-danger btn-sm w-100 mt-1",
                            ),
                            ui.output_ui("dl_status_linear"),
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                # Content section (bottom)
                ui.output_ui("ui_linear_results_area"),
            ),
            # (Subgroup Analysis Removed - Moved to Causal Inference Tab)
            # =====================================================================
            # TAB 5: Repeated Measures
            # =====================================================================
            ui.nav_panel(
                "🔄 Repeated Measures",
                ui.card(
                    ui.card_header("🔄 GEE & LMM Analysis"),
                    ui.layout_columns(
                        create_input_group(
                            "Variable Selection",
                            ui.input_select(
                                "rep_outcome",
                                create_tooltip_label(
                                    "Outcome (Y)", "Response variable."
                                ),
                                choices=[],
                            ),
                            ui.input_select(
                                "rep_treatment",
                                create_tooltip_label(
                                    "Group/Treatment", "Main group of interest."
                                ),
                                choices=[],
                            ),
                            ui.input_select(
                                "rep_time",
                                create_tooltip_label(
                                    "Time Variable", "Timepoint or sequence."
                                ),
                                choices=[],
                            ),
                            ui.input_select(
                                "rep_subject",
                                create_tooltip_label(
                                    "Subject ID", "Unique identifier."
                                ),
                                choices=[],
                            ),
                            type="required",
                        ),
                        create_input_group(
                            "Model Settings",
                            ui.input_radio_buttons(
                                "rep_model_type",
                                "Model Type:",
                                {
                                    "gee": "GEE (Generalized Estimating Equations)",
                                    "lmm": "LMM (Linear Mixed Model)",
                                },
                            ),
                            ui.panel_conditional(
                                "input.rep_model_type === 'gee'",
                                ui.input_select(
                                    "rep_family",
                                    "Family:",
                                    {
                                        "gaussian": "Gaussian (Continuous)",
                                        "binomial": "Binomial (Binary)",
                                        "poisson": "Poisson (Count)",
                                        "gamma": "Gamma",
                                    },
                                ),
                                ui.input_select(
                                    "rep_cov_struct",
                                    "Correlation:",
                                    {
                                        "exchangeable": "Exchangeable",
                                        "independence": "Independence",
                                        "ar1": "AR(1)",
                                    },
                                ),
                            ),
                            ui.panel_conditional(
                                "input.rep_model_type === 'lmm'",
                                ui.input_checkbox(
                                    "rep_random_slope",
                                    "Random Slope for Time",
                                    value=False,
                                ),
                            ),
                            create_tooltip_label(
                                "Adjustments (Covariates)", "Control variables."
                            ),
                            ui.input_selectize(
                                "rep_covariates",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_repeated_validation"),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_repeated",
                        "🚀 Run Analysis",
                        class_="btn-primary btn-sm w-100",
                    ),
                ),
                ui.output_ui("ui_repeated_results_area"),
            ),
            # =====================================================================
            # TAB 6: Reference
            # =====================================================================
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                ## 📚 Core Regression Reference Guide

                ### 1. 📈 Binary Outcomes (Logistic Regression)
                **Use For:** Predicting Yes/No outcomes (e.g., Disease vs Healthy, Died vs Survived).
                
                **Interpretation:**
                *   **Odds Ratio (OR):**
                    *   **OR > 1:** Risk factor (Increases likelihood of event).
                    *   **OR < 1:** Protective factor (Decreases likelihood).
                    *   **OR = 1:** No association.
                
                **Methods:**
                *   **Standard (MLE):** Best for large datasets. Fails with "Perfect Separation".
                *   **Firth's Penalized:** Use for small samples or rare events. Fixes perfect separation.
                *   **Auto:** Automatically switches to Firth if separation is detected.

                ---

                ### 2. 📉 Continuous Outcomes (Linear Regression)
                **Use For:** Predicting numeric values (e.g., Blood Pressure, Length of Stay, Cost).
                
                **Interpretation:**
                *   **Beta Coefficient (β):**
                    *   **β > 0:** Positive relationship (As X increases, Y increases).
                    *   **β < 0:** Negative relationship (As X increases, Y decreases).
                *   **R-squared (R²):** Percentage of variance explained by the model (>0.7 is usually strong).

                **Assumptions Checking:**
                *   **Linearity:** Residuals vs Fitted plot should be flat.
                *   **Normality:** Q-Q plot points should follow the diagonal line.
                *   **Homoscedasticity:** Scale-Location plot should have constant spread.

                ---

                ### 3. 🔢 Count Outcomes (Poisson / Neg. Binomial)
                **Use For:** Count data (e.g., Number of exacerbations, Days in hospital).
                
                **Model Choice:**
                *   **Poisson:** Variance = Mean. Good for simple counts.
                *   **Negative Binomial:** Variance > Mean (Overdispersion). Use if Poisson fails.
                *   **Zero-Inflated:** If there are excess zeros (e.g., many patients with 0 visits).

                **Interpretation:**
                *   **Incidence Rate Ratio (IRR):** Similar to OR. 
                    *   **IRR = 1.5:** Count increases by 50% for every 1-unit increase in X.

                ---

                ### 4. 🔄 Repeated Measures (GEE / LMM)
                **Use For:** Clustered data (e.g., Multiple visits per patient, Eyes per patient).

                **Model Choice:**
                *   **GEE (Generalized Estimating Equations):** Population-averaged effects. Robust to correlation structure errors. Best for binary/count outcomes.
                *   **LMM (Linear Mixed Models):** Subject-specific effects. Handles missing data better. Best for continuous outcomes.

                **Correlation Structures:**
                *   **Exchangeable:** All time points equally correlated.
                *   **AR(1):** Correlation decays over time.
                *   **Unstructured:** No assumption (requires more data).

                ---

                ### 5. 🔛 Subgroup Analysis
                **Use For:** Checking if treatment effect differs across groups (Heterogeneity).
                
                **Interpretation:**
                *   **P-interaction < 0.05:** Significant difference in effect. Report results separately for each group.
                *   **P-interaction ≥ 0.05:** Consistent effect. Report the overall main effect.
                """),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def core_regression_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
    mi_imputed_datasets: reactive.Value[list[pd.DataFrame]] | None = None,  # NEW: MI
) -> None:
    # --- State Management ---
    # Store main logit results: {'html': str, 'fig_adj': FigureWidget, 'fig_crude': FigureWidget}
    """
    Initialize server-side logic and reactive UI handlers for the logistic/poisson/subgroup analysis module.

    Sets up reactive state, input-driven effects, UI renderers, and download endpoints to:
    - manage dataset selection (original vs matched) and dynamic input updates,
    - run binary logistic regression and generate HTML report + forest plots,
    - run Poisson regression and generate HTML report + forest plots,
    - run subgroup analyses and produce forest plot, summary values, and exportable results,
    - surface separation warnings, progress feedback, notifications, and memory cleanup after analyses.

    Parameters:
        input: object providing UI input accessors (e.g., selected values, buttons).
        output: UI output registry (unused directly but provided by framework).
        session: current UI session object.
        df: reactive.Value containing the primary pandas DataFrame or None.
        var_meta: reactive.Value containing variable metadata dictionary.
        df_matched: reactive.Value containing an optional matched pandas DataFrame.
        is_matched: reactive.Value[bool] indicating whether a matched dataset is available/selected.
    """
    logit_res = reactive.Value(None)
    logit_is_running = reactive.Value(False)  # Track running state
    logit_sg_res = reactive.Value(None)
    logit_sg_is_running = reactive.Value(False)  # Track subgroup running state
    # Store Poisson results: {'html': str, 'fig_adj': FigureWidget, 'fig_crude': FigureWidget}
    poisson_res = reactive.Value(None)
    poisson_is_running = reactive.Value(False)
    # Store Negative Binomial results
    nb_res = reactive.Value(None)
    nb_is_running = reactive.Value(False)
    # Store Linear Regression results: {'html_fragment': str, 'html_full': str, 'plots': dict, 'results': dict}
    linear_res = reactive.Value(None)
    linear_is_running = reactive.Value(False)
    # Store subgroup results: SubgroupResult
    subgroup_res: reactive.Value[SubgroupResult | None] = reactive.Value(None)
    # Store analyzer instance: SubgroupAnalysisLogit
    subgroup_analyzer: reactive.Value[SubgroupAnalysisLogit | None] = reactive.Value(
        None
    )
    # Store Repeated Measures results: {'results': DataFrame, 'plot': Figure, 'model_type': str}
    repeated_res = reactive.Value(None)
    repeated_is_running = reactive.Value(False)

    # Store GLM results
    glm_res = reactive.Value(None)
    glm_processing = reactive.Value(False)

    # --- Cache Clearing on Tab Change ---
    @reactive.Effect
    @reactive.event(
        input.btn_run_logit,
        input.btn_run_poisson,
        input.btn_run_nb,
        input.btn_run_subgroup,
        input.btn_run_subgroup,
        input.btn_run_linear,
    )
    def _cleanup_after_analysis():
        """
        OPTIMIZATION: Clear cache after completing analysis.
        This prevents memory buildup from heavy computations.
        """
        try:
            gc.collect()  # Force garbage collection
            logger.debug("Post-analysis cache cleared")
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")

    # --- Dataset Selection Logic ---
    current_df = register_dataset_selector(
        input=input,
        output=output,
        df=df,
        df_matched=df_matched,
        is_matched=is_matched,
        radio_input_id="radio_logit_source",
        title=None,
    )

    @reactive.Calc
    def has_mi() -> bool:
        """Check if MI datasets are available for auto-pooling."""
        return has_mi_datasets(mi_imputed_datasets)

    @reactive.Calc
    def mi_datasets_list() -> list[pd.DataFrame]:
        """Get MI datasets for pooled analysis."""
        return get_mi_datasets(mi_imputed_datasets)

    @render.ui
    def ui_title_with_summary():
        """Display title with dataset summary and MI status."""
        d = current_df()
        mi_active = has_mi()
        mi_count = len(mi_datasets_list()) if mi_active else 0

        if d is not None:
            mi_badge = ""
            if mi_active:
                mi_badge = ui.span(
                    f"🔄 MI Active (m={mi_count})",
                    class_="badge bg-success ms-2",
                    style="font-size: 0.7em; vertical-align: middle;",
                )
            return ui.div(
                ui.h3("📈 Regression Models", mi_badge),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("📈 Regression Models")

    # --- Dynamic Input Updates ---
    @reactive.Effect
    def _update_inputs():
        """
        Update all regression module input widgets to reflect the currently active dataframe.

        Inspects the active dataframe and refreshes choices and sensible defaults across tabs (Binary Logit, Logit Subgroup, Poisson, Negative Binomial, Linear, GLM, and Repeated Measures). If no dataframe is available or it is empty, no updates are performed.

        Detailed behavior:
        - Detects binary, count (non-negative integer), numeric, and candidate subgroup columns to build choice lists.
        - Chooses preferred default variables by keyword heuristics for outcomes, treatments, subgroups, offsets, time/subject identifiers, and predictors.
        - Updates select and selectize widgets for outcomes, predictors, exclusions, interactions, offsets, subgroup adjustment, repeated-measures fields, and GLM inputs.
        """
        d = current_df()
        if d is None or d.empty:
            return

        cols = d.columns.tolist()

        # Identify binary columns for outcomes
        binary_cols = [c for c in cols if d[c].nunique() == 2]

        # Identify potential subgroups (2-10 levels)
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]

        # Update Tab 1 (Binary Logit) Inputs
        # Update Tab 1 (Binary Logit) Inputs
        default_logit_y = select_variable_by_keyword(
            binary_cols, ["outcome", "cured", "death", "status", "event"]
        )

        ui.update_select("sel_outcome", choices=binary_cols, selected=default_logit_y)
        ui.update_selectize("sel_exclude", choices=cols)

        # Generate interaction pair choices for Logit
        interaction_choices = list(
            islice((f"{a} × {b}" for a, b in combinations(cols, 2)), 50)
        )
        ui.update_selectize("sel_interactions", choices=interaction_choices)
        ui.update_selectize("sel_exposure", choices=binary_cols)

        # Update Tab 2 (Poisson) Inputs
        # Identify count columns (non-negative integers)
        count_cols = [
            c
            for c in cols
            if pd.api.types.is_numeric_dtype(d[c])
            and (d[c].dropna() >= 0).all()
            and (d[c].dropna() % 1 == 0).all()
        ]

        # Prefer "Count_" or "Visits" or "Falls"
        default_poisson_y = select_variable_by_keyword(
            count_cols, ["count", "visit", "fall", "event"]
        )
        if not default_poisson_y and count_cols:
            default_poisson_y = count_cols[0]

        ui.update_select(
            "poisson_outcome",
            choices=count_cols if count_cols else cols,
            selected=default_poisson_y,
        )
        ui.update_select("poisson_offset", choices=["None"] + cols)
        ui.update_selectize("poisson_exclude", choices=cols)
        ui.update_selectize("poisson_interactions", choices=interaction_choices[:50])

        # Update Tab 2.5 (Negative Binomial) Inputs
        ui.update_select(
            "nb_outcome",
            choices=count_cols if count_cols else cols,
            selected=default_poisson_y,
        )
        ui.update_select("nb_offset", choices=["None"] + cols)
        ui.update_selectize("nb_exclude", choices=cols)
        ui.update_selectize("nb_interactions", choices=interaction_choices[:50])

        # Update Tab 3 (Linear Regression) Inputs
        # Identify continuous numeric columns for outcome
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]

        # Prefer "Lab_", "Cost", "Score"
        default_linear_y = select_variable_by_keyword(
            numeric_cols, ["lab_", "cost", "score", "chol", "hba1c"]
        )

        linear_outcome_choices = numeric_cols
        ui.update_select(
            "linear_outcome", choices=linear_outcome_choices, selected=default_linear_y
        )

        # Default predictors: exclude ID and outcome, pick numeric/categorical meaningful ones
        default_linear_x = [
            c
            for c in cols
            if c != default_linear_y
            and c not in ["ID", "id_tvc"]
            and not c.startswith("Time_")
        ][
            :5
        ]  # limit to 5
        ui.update_selectize(
            "linear_predictors", choices=numeric_cols, selected=default_linear_x
        )
        ui.update_selectize("linear_exclude", choices=cols)

        # Update Tab 1.5 (Logit Subgroup) Inputs
        ui.update_select(
            "sg_logit_outcome", choices=binary_cols, selected=default_logit_y
        )

        def_sg_treat = select_variable_by_keyword(
            cols, ["treatment", "group"], default_to_first=True
        )
        ui.update_select("sg_logit_treatment", choices=cols, selected=def_sg_treat)

        def_sg_sub = select_variable_by_keyword(
            sg_cols, ["group", "subgroup"], default_to_first=True
        )
        ui.update_select("sg_logit_subgroup", choices=sg_cols, selected=def_sg_sub)

        ui.update_selectize("sg_logit_adjust", choices=cols)

        # Update Tab 5 (Repeated Measures) Inputs
        # Default: Outcome_Cured, Treatment_Group, Time_Months, ID
        default_rep_y = select_variable_by_keyword(
            cols, ["outcome_cured", "outcome", "cured"], default_to_first=True
        )
        default_rep_treat = select_variable_by_keyword(
            cols, ["treatment_group", "treatment", "group"], default_to_first=True
        )
        default_rep_time = select_variable_by_keyword(
            cols, ["time_months", "time", "month"], default_to_first=True
        )
        default_rep_subj = select_variable_by_keyword(
            cols, ["id", "subject", "subjid"], default_to_first=True
        )

        ui.update_select("rep_outcome", choices=cols, selected=default_rep_y)
        ui.update_select("rep_treatment", choices=cols, selected=default_rep_treat)
        ui.update_select("rep_time", choices=cols, selected=default_rep_time)
        ui.update_select("rep_subject", choices=cols, selected=default_rep_subj)
        ui.update_selectize("rep_covariates", choices=cols)

        # Update Tab 2.5 (GLM) Inputs
        # Similar logic to linear but broader
        ui.update_select("glm_outcome", choices=cols, selected=default_linear_y)
        ui.update_selectize("glm_predictors", choices=cols, selected=default_linear_x)
        ui.update_selectize("glm_interactions", choices=interaction_choices[:50])

    # --- Separation Warning Logic ---
    @render.ui
    def ui_separation_warning():
        d = current_df()
        target = input.sel_outcome()
        if d is None or d.empty or not target:
            return None

        risky = check_perfect_separation(d, target)
        if risky:
            return ui.div(
                ui.h6("⚠️ Perfect Separation Risk", class_="text-warning"),
                ui.p(f"Variables: {', '.join(risky)}"),
                ui.p(
                    "Result: Standard logistic regression may fail (infinite coefficients).",
                    style="font-size: 0.9em;",
                ),
                ui.p(
                    "Recommendation: Select 'Firth's (Penalized)' method or use 'Auto'.",
                    style="font-weight: bold; font-size: 0.9em;",
                ),
            )
        return None

    # --- Validation Logic ---
    @render.ui
    def out_logit_validation():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude() if input.sel_exclude() else []

        if d is None or d.empty:
            return None

        alerts = []

        # Check 1: Target selected?
        if not target:
            return (
                None  # Already handled by dropdown placeholder or validation elsewhere
            )

        # Check 2: Target is binary?
        if target and target in d.columns:
            if d[target].nunique() != 2:
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' is not binary (has {d[target].nunique()} unique values). Please select a binary variable.",
                        title="Invalid Outcome",
                    )
                )

        # Check 3: Target in Exclude list? (Redundant but possible)
        if target in exclude:
            alerts.append(
                create_error_alert(
                    f"Outcome '{target}' is currently excluded from the analysis options.",
                    title="Configuration Error",
                )
            )

        # Check 4: Predictors available?
        # Get all potential predictors (columns - target - exclude)
        potential_predictors = [
            c for c in d.columns if c != target and c not in exclude
        ]
        if not potential_predictors:
            alerts.append(
                create_error_alert(
                    "All variables have been excluded. Please allow at least one predictor.",
                    title="No Predictors",
                )
            )

        if alerts:
            return ui.div(*alerts)

        return None

    @render.ui
    def out_poisson_validation():
        d = current_df()
        target = input.poisson_outcome()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        # Check non-negative integers
        if target in d.columns:
            if not pd.api.types.is_numeric_dtype(d[target]):
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' must be numeric.", title="Invalid Outcome"
                    )
                )
            elif (d[target].dropna() < 0).any():
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' contains negative values.",
                        title="Invalid Outcome",
                    )
                )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_nb_validation():
        d = current_df()
        target = input.nb_outcome()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        if target in d.columns:
            if not pd.api.types.is_numeric_dtype(d[target]):
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' must be numeric.", title="Invalid Outcome"
                    )
                )
            elif (d[target].dropna() < 0).any():
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' contains negative values.",
                        title="Invalid Outcome",
                    )
                )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_glm_validation():
        d = current_df()
        target = input.glm_outcome()
        preds = input.glm_predictors()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        if preds and target in preds:
            alerts.append(
                create_error_alert(
                    "Outcome variable cannot be a predictor.",
                    title="Configuration Error",
                )
            )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_linear_validation():
        d = current_df()
        target = input.linear_outcome()
        preds = input.linear_predictors()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        if preds and target in preds:
            alerts.append(
                create_error_alert(
                    "Outcome variable cannot be a predictor.",
                    title="Configuration Error",
                )
            )
        if target in d.columns and not pd.api.types.is_numeric_dtype(d[target]):
            alerts.append(
                create_error_alert(
                    f"Outcome '{target}' must be continuous/numeric.",
                    title="Invalid Outcome",
                )
            )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_repeated_validation():
        d = current_df()
        target = input.rep_outcome()
        subject = input.rep_subject()
        time_var = input.rep_time()
        covs = input.rep_covariates()

        if d is None or d.empty:
            return None
        alerts = []

        if not target or not subject or not time_var:
            return None  # Wait for selection

        if len({target, subject, time_var}) < 3:
            alerts.append(
                create_error_alert(
                    "Outcome, Subject ID, and Time Variable must be different variables.",
                    title="Configuration Error",
                )
            )

        if covs:
            if target in covs or subject in covs or time_var in covs:
                alerts.append(
                    create_error_alert(
                        "Main variables (Outcome, Subject, Time) cannot be used as covariates.",
                        title="Configuration Error",
                    )
                )

        if alerts:
            return ui.div(*alerts)
        return None

    # --- Results Area Logic (Dynamic Loading) ---
    @render.ui
    def ui_logit_results_area():
        # Check if running
        if logit_is_running.get():
            return ui.div(
                create_loading_state("Running Logistic Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        # Check if results exist
        res = logit_res.get()
        if res:
            if "error" in res:
                return ui.div(create_error_alert(res["error"]), class_="fade-in-entry")

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel("🌳 Forest Plots", ui.output_ui("ui_forest_tabs")),
                    ui.nav_panel("📋 Detailed Report", ui.HTML(res["html_fragment"])),
                    ui.nav_panel(
                        "📊 Model Diagnostics",
                        ui.output_ui("out_logit_calibration"),
                    ),
                    ui.nav_panel(
                        "📉 Absolute Measures",
                        ui.output_ui("out_logit_absolute_risk"),
                    ),
                    ui.nav_panel(
                        "🔬 Sensitivity",
                        ui.output_ui("out_logit_sensitivity"),
                    ),
                    ui.nav_panel(
                        "📋 STROBE",
                        ui.output_ui("out_logit_strobe"),
                    ),
                    ui.nav_panel(
                        "✅ Assumptions",
                        ui.markdown("""
                            ### 🧐 Model Assumptions Checklist
                            
                            1.  **Multicollinearity:** Check standard errors in the report. Extremely large SEs often indicate high correlation between predictors (VIF > 5-10).
                            2.  **Linearity:** Log-odds should be linearly related to continuous predictors (Box-Tidwell test).
                            3.  **Independence:** Observations should be independent. If you have repeated measures per patient, use the **Repeated Measures** tab.
                            4.  **Separation:** If standard errors are huge (e.g., > 1000), you may have "Perfect Separation". Consider using **Firth's Method**.
                            """),
                    ),
                ),
                class_="fade-in-entry",
            )

        # Default Placeholder
        return create_empty_state_ui(
            message="No Logistic Regression Results",
            sub_message="Select an outcome and click '🚀 Run Logistic Regression' to start.",
            icon="📈",
        )

    # ==========================================================================
    # LOGIC: Main Logistic Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        """
        Run the logistic regression analysis for the currently selected outcome and publish results for the UI and download.

        Prepares the analysis dataframe (applying exclusions), parses any interaction pairs, and invokes the analysis backend. If available, builds adjusted and crude forest plots and appends them to an HTML fragment used for in-UI display. Also wraps the fragment into a complete HTML document for download. On success, stores the following keys in `logit_res`: `"html_fragment"`, `"html_full"`, `"fig_adj"`, and `"fig_crude"`, and shows a completion notification. On error, logs the exception and shows an error notification.
        """
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        penalty_weight = (
            float(input.firth_penalty_weight()) if method == "firth" else 1.0
        )
        interactions_raw = input.sel_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select an outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors="ignore")

        # Parse interaction pairs from "var1 × var2" format
        interaction_pairs: list[tuple[str, str]] | None = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Logit: Using {len(interaction_pairs)} interaction pairs")
        else:
            # exclude = []  # FIX: Do not wipe user exclusions
            pass

        # Start Loading State
        logit_is_running.set(True)
        logit_res.set(None)  # Clear previous results

        # Use reactive flush to ensure UI updates before heavy computation
        # (Note: In standard Shiny, this might still block if not async, but we set state first)

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Logistic Regression...", detail="Calculating...")

            try:
                # Check if MI datasets are available for pooled analysis
                mi_active = has_mi()
                mi_dfs = mi_datasets_list() if mi_active else []

                if mi_active and len(mi_dfs) > 0:
                    # ====== MI POOLED ANALYSIS ======
                    p.set(
                        message="Running MI Pooled Logistic Regression...",
                        detail=f"Analyzing {len(mi_dfs)} imputed datasets...",
                    )
                    logger.info(
                        f"Logit: Running pooled analysis on {len(mi_dfs)} MI datasets"
                    )

                    # Run analysis on each imputed dataset
                    all_results = []
                    for i, mi_df in enumerate(mi_dfs):
                        p.set(
                            value=(i + 1) / len(mi_dfs),
                            detail=f"Processing imputed dataset {i + 1}/{len(mi_dfs)}...",
                        )
                        # Apply exclusions to MI dataset
                        mi_df_clean = mi_df.drop(columns=exclude, errors="ignore")

                        _, or_res_i, aor_res_i, _ = analyze_outcome(
                            target,
                            mi_df_clean,
                            var_meta=var_meta.get(),
                            method=method,
                            interaction_pairs=interaction_pairs,
                            adv_stats=CONFIG,
                            penalty_weight=penalty_weight,
                        )
                        all_results.append(
                            {
                                "or_res": or_res_i,
                                "aor_res": aor_res_i,
                            }
                        )

                    # Pool results using Rubin's rules
                    pooled_or = {}
                    pooled_aor = {}

                    # Pool crude ORs
                    if all_results[0].get("or_res"):
                        for var_key in all_results[0]["or_res"].keys():
                            estimates = []
                            variances = []
                            for res in all_results:
                                if res.get("or_res") and var_key in res["or_res"]:
                                    v = res["or_res"][var_key]
                                    # Extract estimate on log scale
                                    or_val = v.get("or")
                                    se = v.get("se")
                                    if (
                                        or_val is not None
                                        and se is not None
                                        and or_val > 0
                                        and se > 0
                                    ):
                                        estimates.append(np.log(or_val))
                                        variances.append(se**2)

                            if len(estimates) == len(mi_dfs):
                                # Calculate n_obs from MI datasets
                                n_obs_mi = int(
                                    np.mean([len(mi_df) for mi_df in mi_dfs])
                                )
                                pooled = pool_estimates(
                                    estimates, variances, n_obs=n_obs_mi
                                )
                                pooled_or[var_key] = {
                                    "or": np.exp(pooled.estimate),
                                    "ci_low": np.exp(pooled.ci_lower),
                                    "ci_high": np.exp(pooled.ci_upper),
                                    "p_value": pooled.p_value,
                                    "fmi": pooled.fmi,
                                    "label": all_results[0]["or_res"][var_key].get(
                                        "label", var_key
                                    ),
                                }

                    # Pool adjusted ORs
                    if all_results[0].get("aor_res"):
                        for var_key in all_results[0]["aor_res"].keys():
                            estimates = []
                            variances = []
                            for res in all_results:
                                if res.get("aor_res") and var_key in res["aor_res"]:
                                    v = res["aor_res"][var_key]
                                    aor_val = v.get("aor")
                                    se = v.get("se")
                                    if (
                                        aor_val is not None
                                        and se is not None
                                        and aor_val > 0
                                        and se > 0
                                    ):
                                        estimates.append(np.log(aor_val))
                                        variances.append(se**2)

                            if len(estimates) == len(mi_dfs):
                                # Calculate n_obs from MI datasets
                                n_obs_mi = int(np.mean([len(d) for d in mi_dfs]))
                                pooled = pool_estimates(
                                    estimates, variances, n_obs=n_obs_mi
                                )
                                pooled_aor[var_key] = {
                                    "aor": np.exp(pooled.estimate),
                                    "ci_low": np.exp(pooled.ci_lower),
                                    "ci_high": np.exp(pooled.ci_upper),
                                    "p_value": pooled.p_value,
                                    "fmi": pooled.fmi,
                                    "label": all_results[0]["aor_res"][var_key].get(
                                        "label", var_key
                                    ),
                                }

                    # Build HTML report with pooled results
                    html_rep = generate_mi_pooled_report(
                        len(mi_dfs), pooled_or, pooled_aor
                    )

                    or_res = pooled_or
                    aor_res = pooled_aor
                    interaction_res = None

                else:
                    # ====== STANDARD ANALYSIS ======
                    # Run Logic from logic.py
                    html_rep, or_res, aor_res, interaction_res = analyze_outcome(
                        target,
                        final_df,
                        var_meta=var_meta.get(),
                        method=method,
                        interaction_pairs=interaction_pairs,
                        adv_stats=CONFIG,
                        penalty_weight=penalty_weight,
                    )
            except Exception as e:
                err_msg = f"Error running logistic regression: {e!s}"
                logit_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Logistic regression error")
                logit_is_running.set(False)
                return

            # Generate Forest Plots using library (for interactive widgets)
            fig_adj = None
            fig_crude = None

            if aor_res:
                df_adj = pd.DataFrame(
                    [{"variable": v.get("label", k), **v} for k, v in aor_res.items()]
                )
                if not df_adj.empty:
                    try:
                        fig_adj = create_forest_plot(
                            df_adj,
                            "aor",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Multivariable: Adjusted OR</b>",
                            x_label="Adjusted OR",
                        )
                    except ValueError as e:
                        logger.warning(
                            "Logit Adjusted Forest Plot creation failed: %s", e
                        )

            if or_res:
                df_crude = pd.DataFrame(
                    [{"variable": v.get("label", k), **v} for k, v in or_res.items()]
                )
                if not df_crude.empty:
                    try:
                        fig_crude = create_forest_plot(
                            df_crude,
                            "or",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Univariable: Crude OR</b>",
                            x_label="Crude OR",
                        )
                    except ValueError as e:
                        logger.warning("Logit Crude Forest Plot creation failed: %s", e)

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Table + Plots) ---
            # 1. Create Fragment for UI (Table + Plots)
            logit_fragment_html = html_rep

            # Note: The "Method Used" banner is now handled inside logic.py -> analyze_outcome
            # to correctly reflect "Auto" decisions.

            # Append Adjusted Plot if available
            if fig_adj:
                plot_html = plotly_figure_to_html(fig_adj, include_plotlyjs="cdn")
                logit_fragment_html += f"<div class='forest-plot-section' ><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"

            # Append Crude Plot if available
            if fig_crude:
                plot_html = plotly_figure_to_html(fig_crude, include_plotlyjs="cdn")
                logit_fragment_html += f"<div class='forest-plot-section' ><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # Append Assumptions Checklist
            logit_fragment_html += """
            <div class='assumptions-section' >
                <h3>✅ Model Assumptions Checklist</h3>
                <ol>
                    <li><strong>Multicollinearity:</strong> Check standard errors. Large SEs indicate high correlation (VIF > 5-10).</li>
                    <li><strong>Linearity:</strong> Log-odds should be linear with continuous predictors (Box-Tidwell).</li>
                    <li><strong>Independence:</strong> Observations should be independent.</li>
                    <li><strong>Separation:</strong> Huge SEs (>1000) imply perfect separation. Consider Firth's method.</li>
                </ol>
            </div>
            """

            # 2. Create Full HTML for Download (Wrapped)
            full_logit_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Logistic Regression Report: {html.escape(target)}</title>
                {get_shiny_css()}
            </head>
            <body>
                <div class="report-container">
                    {logit_fragment_html}
                </div>
            </body>
            </html>
            """

            # --- Calculate Predictions for Calibration ---
            # Note: For MI pooled analysis, we skip calibration (no single model)
            y_true_arr = None
            y_pred_arr = None

            if not mi_active and aor_res:
                try:
                    # Prepare data for prediction
                    y_raw = final_df[target].dropna()
                    unique_vals = set(y_raw.unique())

                    if len(unique_vals) == 2:
                        # Map to 0/1 if needed
                        if not unique_vals.issubset({0, 1}):
                            sorted_vals = sorted(unique_vals, key=str)
                            y_binary = y_raw.map({sorted_vals[0]: 0, sorted_vals[1]: 1})
                        else:
                            y_binary = y_raw.astype(int)

                        # Get predictor columns from aor_res keys
                        pred_cols = []
                        for k in aor_res.keys():
                            # Handle categorical dummy columns
                            if "::" in k:
                                base_col = k.split("::")[0].split(": ")[0]
                                if (
                                    base_col in final_df.columns
                                    and base_col not in pred_cols
                                ):
                                    pred_cols.append(base_col)
                            elif k in final_df.columns:
                                pred_cols.append(k)

                        if pred_cols:
                            # Use centralized cleaning
                            req_cols = [target] + pred_cols
                            final_df_clean, _ = prepare_data_for_analysis(
                                final_df,
                                required_cols=req_cols,
                                var_meta=var_meta.get(),
                                return_info=True,
                            )

                            # Re-define y based on clean data
                            y_raw_clean = final_df_clean[target]
                            if not unique_vals.issubset({0, 1}):
                                sorted_vals = sorted(unique_vals, key=str)
                                y_binary = y_raw_clean.map(
                                    {sorted_vals[0]: 0, sorted_vals[1]: 1}
                                )
                            else:
                                y_binary = y_raw_clean.astype(int)

                            # Prepare design matrix
                            X = pd.get_dummies(
                                final_df_clean[pred_cols], drop_first=True
                            )
                            common_idx = y_binary.index.intersection(X.index)
                            X_clean = X.loc[common_idx].dropna()
                            y_clean = y_binary.loc[X_clean.index]

                            if len(y_clean) > 20:
                                X_const = sm.add_constant(X_clean, has_constant="add")
                                model = sm.Logit(y_clean, X_const)
                                result = model.fit(disp=0, maxiter=50)

                                if not result.mle_retvals.get("converged", False):
                                    logger.warning(
                                        "Logistic calibration model failed to converge"
                                    )
                                    # Skip results assignment
                                else:
                                    y_pred_arr = result.predict(X_const).values
                                    y_true_arr = y_clean.values
                                    logger.info(
                                        "Calibration: Generated %d predictions",
                                        len(y_pred_arr),
                                    )
                except Exception as e:
                    logger.warning(
                        "Could not generate predictions for calibration: %s", e
                    )

            # Store Results
            # Calculate missing info for MI pooled
            mi_missing_info = None
            if has_mi_datasets(mi_imputed_datasets):
                # Use mean length of imputed datasets
                mi_dfs = get_mi_datasets(mi_imputed_datasets)
                mi_n_obs = int(np.mean([len(mi_df) for mi_df in mi_dfs]))
                mi_missing_info = {
                    "mi_pooled": True,
                    "imputation_count": len(mi_dfs),
                    "rows_analyzed": mi_n_obs,
                    "total_missing": "N/A (Imputed)",
                }
                logit_fragment_html += create_missing_data_report_html(
                    mi_missing_info, var_meta.get() or {}
                )

                # Rebuild full HTML now that missing-data summary is included
                full_logit_html = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Logistic Regression Report: {html.escape(target)}</title>
                    {get_shiny_css()}
                </head>
                <body>
                    <div class="report-container">
                        {logit_fragment_html}
                    </div>
                </body>
                </html>
                """

            logit_res.set(
                {
                    "html_fragment": logit_fragment_html,  # For UI
                    "html_full": full_logit_html,  # For Download
                    "fig_adj": fig_adj,
                    "fig_crude": fig_crude,
                    "y_true": y_true_arr,  # For calibration
                    "y_pred": y_pred_arr,  # For calibration
                    "aor_res": aor_res,  # For E-value sensitivity
                    "missing_info": mi_missing_info,
                }
            )

            ui.notification_show("✅ Analysis Complete!", type="message")

        # End Loading State
        logit_is_running.set(False)

    # --- Render Main Results ---
    @render.ui
    def out_logit_status():
        res = logit_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Regression Complete"),
                class_="info-callout",
            )
        return None

    @render.ui
    def logit_detailed_report():
        res = logit_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Detailed Report"), ui.HTML(res["html_fragment"])
            )
        return None

    @render.ui
    def ui_forest_tabs():
        """
        Render tabbed forest plot panels for the most recent logistic regression results.

        Returns:
            ui.Component: A UI element containing "Crude OR" and/or "Adjusted OR" tabs when corresponding forest figures are present.
            If analysis exists but no forest figures are available, returns a muted message indicating no plots are available.
        """
        res = logit_res.get()
        if not res:
            return None  # Should be handled by parent container logic

        tabs = []
        if res["fig_crude"]:
            tabs.append(ui.nav_panel("Crude OR", ui.output_ui("out_forest_crude")))
        if res["fig_adj"]:
            tabs.append(ui.nav_panel("Adjusted OR", ui.output_ui("out_forest_adj")))

        if not tabs:
            return ui.div(
                "No forest plots generated from the model.", class_="text-muted p-3"
            )
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_forest_adj():
        """
        Render the adjusted forest plot panel for logistic regression results.

        Returns:
            ui.Component: A UI HTML component containing the adjusted forest plot when available; otherwise a centered placeholder message indicating results are pending.
        """
        res = logit_res.get()
        if res is None or not res.get("fig_adj"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            res["fig_adj"],
            div_id="plot_logit_forest_adj",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_forest_crude():
        """
        Render the crude (unadjusted) forest plot UI for the current logistic regression results.

        Returns:
            ui_component: A UI element containing the plot HTML when a crude figure is available; otherwise a centered placeholder message indicating results are pending.
        """
        res = logit_res.get()
        if res is None or not res.get("fig_crude"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            res["fig_crude"],
            div_id="plot_logit_forest_crude",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_logit_calibration():
        """
        Render calibration plots and metrics for the logistic regression model.

        Returns calibration plot, decision curve, and key metrics table.
        """
        res = logit_res.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Run analysis to see model diagnostics...*"),
                class_="muted-placeholder",
            )

        # Check if we have prediction data
        y_true = res.get("y_true")
        y_pred = res.get("y_pred")

        if y_true is None or y_pred is None:
            # Fall back to metrics-only display if no predictions stored
            return ui.card(
                ui.card_header("📊 Model Diagnostics"),
                ui.div(
                    ui.markdown("""
                        **Note:** Calibration plots require predicted probabilities.
                        
                        The model diagnostics table is included in the **Detailed Report** tab.
                        
                        To enable full calibration analysis, ensure the multivariate model was successfully fitted.
                    """),
                    style="padding: 20px; color: #666;",
                ),
            )

        try:
            # Generate calibration report
            calib_report = get_calibration_report(y_true, y_pred)
            metrics_html = format_calibration_html(calib_report)

            # Create calibration plot
            fig_calib = create_calibration_plot(
                y_true, y_pred, title="Calibration Plot (Observed vs Predicted)"
            )
            calib_plot_html = plotly_figure_to_html(
                fig_calib,
                div_id="plot_logit_calibration",
                include_plotlyjs="cdn",
                responsive=True,
            )

            # Create decision curve
            fig_dca = create_decision_curve(
                y_true, y_pred, title="Decision Curve Analysis"
            )
            dca_plot_html = plotly_figure_to_html(
                fig_dca,
                div_id="plot_logit_dca",
                include_plotlyjs="cdn",
                responsive=True,
            )

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel(
                        "📈 Calibration Metrics",
                        ui.HTML(metrics_html),
                        ui.markdown("""
                            ---
                            **Interpretation Guide:**
                            - **C-statistic (AUC)**: Measures discrimination (ability to distinguish outcomes). >0.8 is excellent.
                            - **Brier Score**: Measures calibration accuracy. <0.25 is acceptable.
                            - **Calibration Slope**: Should be close to 1. <0.8 or >1.2 suggests recalibration needed.
                            - **Hosmer-Lemeshow**: p > 0.05 indicates adequate fit.
                        """),
                    ),
                    ui.nav_panel("📊 Calibration Plot", ui.HTML(calib_plot_html)),
                    ui.nav_panel("🎯 Decision Curve", ui.HTML(dca_plot_html)),
                ),
            )

        except Exception as e:
            logger.exception("Calibration plot generation failed: %s", e)
            return ui.div(
                ui.div(
                    f"⚠️ Could not generate calibration plots: {e}",
                    class_="alert alert-warning",
                ),
                style="padding: 20px;",
            )

    @render.ui
    def out_logit_absolute_risk():
        """
        Render absolute risk measures (ARD/NNT) for logistic regression.

        Requires a binary treatment/exposure variable to be selected.
        """
        res = logit_res.get()
        d = current_df()
        target = input.sel_outcome()

        if res is None or d is None or not target:
            return ui.div(
                ui.markdown("⏳ *Run analysis to see absolute risk measures...*"),
                class_="muted-placeholder",
            )

        # Get binary predictors from the data
        binary_cols = []
        for col in d.columns:
            if col != target:
                unique_vals = d[col].dropna().unique()
                if len(unique_vals) == 2:
                    binary_cols.append(col)

        if not binary_cols:
            return ui.card(
                ui.card_header("📉 Absolute Risk Measures"),
                ui.div(
                    ui.markdown("""
                        **Note:** Absolute risk measures require a binary treatment/exposure variable.
                        
                        No binary predictors were found in the dataset.
                        
                        **What you need:**
                        - A treatment or exposure variable with exactly 2 groups (e.g., Drug vs Placebo, Exposed vs Unexposed)
                        
                        The relative measures (OR) are still available in the Detailed Report.
                    """),
                    style="padding: 20px; color: #666;",
                ),
            )

        try:
            # Calculate absolute risk for the first binary predictor (primary exposure)
            # Use selected exposure or fallback to first binary column
            exposure_col = input.sel_exposure()
            if not exposure_col or exposure_col not in d.columns:
                if binary_cols:
                    exposure_col = binary_cols[0]

            if not exposure_col or exposure_col not in d.columns:
                # If still no valid exposure, skip AR/NNT
                return None

            exposure_vals = sorted(d[exposure_col].dropna().unique())
            treatment_val = exposure_vals[1] if len(exposure_vals) > 1 else 1
            control_val = exposure_vals[0] if len(exposure_vals) > 0 else 0

            # Clean data before absolute risk calculation
            clean_df, _ = prepare_data_for_analysis(
                d,
                required_cols=[target, exposure_col],
                var_meta=var_meta.get(),
                return_info=True,
            )
            risk_data = calculate_absolute_risk(
                clean_df,
                target,
                exposure_col,
                treatment_value=treatment_val,
                control_value=control_val,
            )

            if "error" in risk_data:
                return ui.card(
                    ui.card_header("📉 Absolute Risk Measures"),
                    ui.div(
                        f"Could not calculate: {risk_data['error']}",
                        style="padding: 20px; color: #666;",
                    ),
                )

            nnt_data = calculate_nnt(
                risk_data["ard"],
                risk_data.get("ard_ci_lower"),
                risk_data.get("ard_ci_upper"),
            )

            html_report = format_absolute_risk_html(risk_data, nnt_data)

            # Escape user-provided strings before inserting into HTML
            esc_exposure = html.escape(str(exposure_col))
            esc_treatment = html.escape(str(treatment_val))
            esc_control = html.escape(str(control_val))

            return ui.card(
                ui.card_header(f"📉 Absolute Risk: {esc_exposure}"),
                ui.HTML(html_report),
                ui.hr(),
                ui.div(
                    ui.markdown(f"""
                        **Exposure Variable:** `{esc_exposure}`
                        - Treatment group: `{esc_treatment}` (n={risk_data["n_treatment"]})
                        - Control group: `{esc_control}` (n={risk_data["n_control"]})
                        
                        ---
                        
                        **Why This Matters (NEJM/Lancet Standard):**
                        
                        Relative measures (OR, RR) can overstate clinical significance. Absolute measures show real-world impact:
                        - **ARD**: The actual difference in event rates
                        - **NNT**: How many patients need treatment to affect 1 outcome
                    """),
                    style="font-size: 0.9em; color: #666;",
                ),
            )

        except Exception as e:
            logger.exception("Absolute risk calculation failed: %s", e)
            return ui.div(
                ui.div(
                    f"⚠️ Could not calculate absolute risk: {e}",
                    class_="alert alert-warning",
                ),
                style="padding: 20px;",
            )

    @render.ui
    def out_logit_sensitivity():
        """
        Render E-value sensitivity analysis for unmeasured confounding.

        E-value quantifies the minimum strength of association an unmeasured
        confounder would need with both treatment and outcome to explain away
        the observed effect.
        """
        res = logit_res.get()

        if res is None or "error" in res:
            return ui.div(
                ui.markdown("⏳ *Run analysis to see sensitivity analysis...*"),
                class_="muted-placeholder",
            )

        # Get adjusted OR results
        aor_res = res.get("aor_res", {})

        if not aor_res:
            return ui.card(
                ui.card_header("🔬 E-value Sensitivity Analysis"),
                ui.div(
                    ui.markdown("""
                        **Note:** E-value calculation requires adjusted odds ratios (aOR).
                        
                        No multivariate results available. Run analysis with predictors to calculate E-values.
                    """),
                    style="padding: 20px; color: #666;",
                ),
            )

        try:
            # Calculate E-values for significant effects
            e_value_rows = []
            significant_count = 0

            for var_name, var_data in aor_res.items():
                aor = var_data.get("aor")
                ci_low = var_data.get("ci_low")
                ci_high = var_data.get("ci_high")
                p_val = var_data.get("p_value")

                if aor is None or np.isnan(aor):
                    continue

                # Calculate E-value
                e_result = calculate_e_value(
                    estimate=aor, lower=ci_low, upper=ci_high, estimate_type="OR"
                )

                if "error" in e_result:
                    continue

                e_val = e_result.get("e_value_estimate", np.nan)
                e_ci = e_result.get("e_value_ci_limit", np.nan)

                # Interpret strength
                if e_val >= 3:
                    strength = "🟢 Strong"
                    note = "Robust to moderate confounding"
                elif e_val >= 2:
                    strength = "🟡 Moderate"
                    note = "Sensitive to moderate confounding"
                else:
                    strength = "🔴 Weak"
                    note = "Easily explained by confounding"

                is_sig = p_val is not None and p_val < 0.05
                if is_sig:
                    significant_count += 1

                sig_badge = "✓" if is_sig else ""

                e_ci_str = (
                    f"{e_ci:.2f}" if (e_ci is not None and not np.isnan(e_ci)) else "—"
                )

                e_value_rows.append(f"""
                    <tr>
                        <td>{html.escape(str(var_name))} {sig_badge}</td>
                        <td>{aor:.2f}</td>
                        <td><strong>{e_val:.2f}</strong></td>
                        <td>{e_ci_str}</td>
                        <td>{strength}</td>
                        <td style='font-size: 0.85em; color: #666;'>{note}</td>
                    </tr>
                """)

            if not e_value_rows:
                return ui.card(
                    ui.card_header("🔬 E-value Sensitivity Analysis"),
                    ui.div(
                        "No valid estimates for E-value calculation.",
                        style="padding: 20px;",
                    ),
                )

            e_table = f"""
            <table class="table table-sm table-hover">
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>aOR</th>
                        <th>E-value (Point)</th>
                        <th>E-value (CI)</th>
                        <th>Robustness</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(e_value_rows)}
                </tbody>
            </table>
            """

            return ui.card(
                ui.card_header("🔬 E-value Sensitivity Analysis"),
                ui.HTML(e_table),
                ui.hr(),
                ui.markdown("""
                    **How to Interpret E-values (VanderWeele & Ding, 2017):**
                    
                    The **E-value** is the minimum strength of association (on the risk ratio scale) that an unmeasured confounder would need to have with **both** the treatment and the outcome to fully explain away the observed effect.
                    
                    | E-value | Interpretation |
                    |---------|----------------|
                    | ≥ 3.0 | **Strong robustness** - Would require a very strong confounder |
                    | 2.0 - 3.0 | **Moderate robustness** - Moderately robust to confounding |
                    | 1.5 - 2.0 | **Weak robustness** - Could be explained by a moderately strong confounder |
                    | < 1.5 | **Very weak** - Easily explained by mild confounding |
                    
                    > **Citation:** VanderWeele TJ, Ding P. Sensitivity Analysis in Observational Research: Introducing the E-Value. *Ann Intern Med.* 2017;167(4):268-274.
                """),
            )

        except Exception as e:
            logger.exception("E-value calculation failed: %s", e)
            return ui.div(
                ui.div(
                    f"⚠️ Could not calculate E-values: {e}",
                    class_="alert alert-warning",
                ),
                style="padding: 20px;",
            )

    @render.ui
    def out_logit_strobe():
        """
        Render STROBE checklist with auto-populated items.

        Auto-marks items based on what was performed in the analysis.
        """
        res = logit_res.get()
        d = current_df()
        target = input.sel_outcome()

        if res is None or "error" in res:
            return ui.div(
                ui.markdown("⏳ *Run analysis to see STROBE checklist...*"),
                class_="muted-placeholder",
            )

        try:
            # Build analysis metadata for auto-population
            metadata = _build_strobe_metadata(res, d, target)

            # Create auto-populated checklist
            checklist = auto_populate_strobe(metadata)

            # Generate HTML
            html_content = format_strobe_html_compact(checklist)

            return ui.card(
                ui.card_header("📋 STROBE Checklist (Auto-filled)"),
                ui.HTML(html_content),
                ui.hr(),
                ui.row(
                    ui.column(
                        6,
                        ui.download_button(
                            "dl_strobe_md",
                            "📥 Download STROBE (Markdown)",
                            class_="btn-outline-secondary btn-sm",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.markdown("""
                            **Legend:** ✅ Complete | 🔶 Partial | ❌ Not addressed
                        """),
                    ),
                ),
            )

        except Exception as e:
            logger.exception("STROBE checklist generation failed: %s", e)
            return ui.div(
                ui.div(
                    f"⚠️ Could not generate STROBE checklist: {e}",
                    class_="alert alert-warning",
                ),
                style="padding: 20px;",
            )

    @render.download(filename="strobe_checklist.md")
    def dl_strobe_md():
        """Download STROBE checklist as markdown."""
        res = logit_res.get()
        d = current_df()

        safe_data_download(res, label="STROBE Checklist", type_="checklist")

        try:
            metadata = _build_strobe_metadata(res, d, input.sel_outcome())

            checklist = auto_populate_strobe(metadata)
            yield generate_checklist_markdown(checklist)
        except Exception as e:
            yield f"# Error\n\nCould not generate checklist: {e}"

    @render.download(filename="logit_report.html")
    def btn_dl_report():
        """
        Yield the complete HTML report for download when a logit analysis result is available.

        Yields the standalone HTML document stored in the current logistic regression result under the key "html_full".

        Returns:
            str: An HTML string containing the full, download-ready report, yielded if present.
        """
        res = logit_res.get()
        yield safe_download_html(
            res.get("html_full") if res else None, label="Logistic Regression Report"
        )

    # --- Download Status Badges ---
    @render.ui
    def dl_status_logit():
        res = logit_res.get()
        return create_download_status_badge(
            res is not None and bool(res.get("html_full"))
        )

    @render.ui
    def dl_status_sg_logit():
        res = logit_sg_res.get()
        return create_download_status_badge(
            res is not None and "error" not in res and res.get("results_df") is not None
        )

    @render.ui
    def dl_status_poisson():
        res = poisson_res.get()
        return create_download_status_badge(
            res is not None and bool(res.get("html_full"))
        )

    @render.ui
    def dl_status_nb():
        res = nb_res.get()
        return create_download_status_badge(
            res is not None and bool(res.get("html_full"))
        )

    @render.ui
    def dl_status_glm():
        res = glm_res.get()
        return create_download_status_badge(
            res is not None and bool(res.get("html_report"))
        )

    @render.ui
    def dl_status_linear():
        res = linear_res.get()
        return create_download_status_badge(
            res is not None and bool(res.get("html_full"))
        )

    # ==========================================================================
    # LOGIC: Poisson Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_poisson)
    def _run_poisson():
        d = current_df()
        target = input.poisson_outcome()
        exclude = input.poisson_exclude()
        offset_col = input.poisson_offset()
        interactions_raw = input.poisson_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select a count outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors="ignore")
        offset = offset_col if offset_col != "None" else None

        # Parse interaction pairs
        interaction_pairs: list[tuple[str, str]] | None = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Poisson: Using {len(interaction_pairs)} interaction pairs")

        # Start Loading State
        poisson_is_running.set(True)
        poisson_res.set(None)

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Poisson Regression...", detail="Calculating...")

            try:
                # Run Poisson Logic
                # Expecting 4 values from the updated poisson_lib.py
                html_rep, irr_res, airr_res, interaction_res, _ = (
                    analyze_poisson_outcome(
                        target,
                        final_df,
                        var_meta=var_meta.get(),
                        offset_col=offset,
                        interaction_pairs=interaction_pairs,
                    )
                )
            except Exception as e:
                err_msg = f"Error running Poisson regression: {e!s}"
                poisson_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Poisson regression error")
                poisson_is_running.set(False)
                return

            # Generate Forest Plots for IRR
            fig_adj = None
            fig_crude = None

            if airr_res:
                df_adj = pd.DataFrame(
                    [{"variable": k, **v} for k, v in airr_res.items()]
                )
                if not df_adj.empty:
                    try:
                        fig_adj = create_forest_plot(
                            df_adj,
                            "airr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Multivariable: Adjusted IRR</b>",
                            x_label="Adjusted IRR",
                        )
                    except ValueError as e:
                        logger.warning(
                            "Poisson Adjusted Forest Plot creation failed: %s", e
                        )

            if irr_res:
                df_crude = pd.DataFrame(
                    [{"variable": k, **v} for k, v in irr_res.items()]
                )
                if not df_crude.empty:
                    try:
                        fig_crude = create_forest_plot(
                            df_crude,
                            "irr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Univariable: Crude IRR</b>",
                            x_label="Crude IRR",
                        )
                    except ValueError as e:
                        logger.warning(
                            "Poisson Crude Forest Plot creation failed: %s", e
                        )

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Combined Table + Plot) ---
            # Unlike logic.py, poisson_lib might return just the table HTML.
            # We inject CSS and append the Forest Plot HTML here to match the requested format.

            # Keep a fragment for in-app rendering
            poisson_fragment_html = html_rep

            # Append Adjusted Plot if available, else Crude
            plot_html = ""
            if fig_adj:
                plot_html = fig_adj.to_html(full_html=False, include_plotlyjs="cdn")
                poisson_fragment_html += f"<div class='forest-plot-section' ><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"
            elif fig_crude:
                plot_html = fig_crude.to_html(full_html=False, include_plotlyjs="cdn")
                poisson_fragment_html += f"<div class='forest-plot-section' ><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # Wrap in standard HTML structure for standalone download correctness
            wrapped_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Poisson Regression Report: {html.escape(target)}</title>
                {get_shiny_css()}
            </head>
            <body>
                <div class="report-container">
                    {poisson_fragment_html}
                </div>
            </body>
            </html>
            """
            full_poisson_html = wrapped_html

            # Store Results
            poisson_res.set(
                {
                    "html_fragment": poisson_fragment_html,  # For UI rendering
                    "html_full": full_poisson_html,  # For downloads
                    "fig_adj": fig_adj,
                    "fig_crude": fig_crude,
                }
            )

            ui.notification_show("✅ Poisson Analysis Complete!", type="message")

        # End Loading State
        poisson_is_running.set(False)

    # --- Render Poisson Results ---
    @render.ui
    def out_poisson_status():
        res = poisson_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Poisson Regression Complete"),
                class_="info-callout",
            )
        return None

    @render.ui
    def ui_poisson_results_area():
        if poisson_is_running.get():
            return ui.div(
                create_loading_state("Running Poisson Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = poisson_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plots",
                        ui.output_ui("ui_poisson_forest_tabs"),
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report",
                        ui.HTML(res["html_fragment"]),
                    ),
                    ui.nav_panel(
                        "📚 Reference",
                        ui.markdown("""
                        ### Poisson Regression Reference
                        
                        **When to Use:**
                        * Count outcomes (e.g., number of events, visits, infections)
                        * Rate data with exposure offset (e.g., events per person-year)
                        
                        **Interpretation:**
                        * **IRR > 1**: Higher incidence rate (Risk factor) 🔴
                        * **IRR < 1**: Lower incidence rate (Protective) 🟢
                        * **IRR = 1**: No effect on rate
                        
                        **Overdispersion:**
                        If variance >> mean, consider Negative Binomial regression.
                        """),
                    ),
                ),
                class_="fade-in-entry",
            )

        # Default Placeholder
        return create_empty_state_ui(
            message="No Poisson Regression Results",
            sub_message="Select count outcome and predictors, then click '🚀 Run Random Forest'.. oops wait 'Run Poisson'.",
            icon="🔢",
        )

    @render.ui
    def ui_poisson_forest_tabs():
        """
        Render a tabbed UI containing Poisson regression forest plots when results are available.

        Returns:
            ui.Component: A UI fragment showing one or both tabs ("Crude IRR", "Adjusted IRR") with embedded outputs, or an informational div if results or plots are not available.
        """
        res = poisson_res.get()
        if not res:
            return ui.div(
                "Run analysis to see forest plots.",
                class_="muted-placeholder",
            )

        tabs = []
        if res["fig_crude"]:
            tabs.append(
                ui.nav_panel("Crude IRR", ui.output_ui("out_poisson_forest_crude"))
            )
        if res["fig_adj"]:
            tabs.append(
                ui.nav_panel("Adjusted IRR", ui.output_ui("out_poisson_forest_adj"))
            )

        if not tabs:
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_poisson_forest_adj():
        """
        Render the adjusted Poisson regression forest plot or a waiting placeholder if results are not ready.

        Returns:
            A UI component containing the plot HTML when an adjusted figure is available; otherwise a centered placeholder div with a "Waiting for results..." message.
        """
        res = poisson_res.get()
        if res is None or not res.get("fig_adj"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            res["fig_adj"],
            div_id="plot_poisson_forest_adj",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_poisson_forest_crude():
        """
        Render the crude Poisson forest plot as a UI HTML component, or a waiting placeholder when results are not available.

        Returns:
            ui.Element: An HTML-wrapped Plotly figure for the crude Poisson forest plot if present; otherwise a centered div containing a "Waiting for results..." message.
        """
        res = poisson_res.get()
        if res is None or not res.get("fig_crude"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            res["fig_crude"],
            div_id="plot_poisson_forest_crude",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.download(filename="poisson_report.html")
    def btn_dl_poisson_report():
        """
        Provide the complete Poisson regression report as a standalone HTML document for download.

        Returns:
            str: Full HTML document containing the Poisson analysis report, or nothing if no results are available.
        """
        res = poisson_res.get()
        yield safe_download_html(
            res.get("html_full") if res else None, label="Poisson Regression Report"
        )

    # ==========================================================================
    # LOGIC: Negative Binomial Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_nb)
    def _run_nb():
        d = current_df()
        target = input.nb_outcome()
        exclude = input.nb_exclude()
        offset_col = input.nb_offset()
        interactions_raw = input.nb_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select a count outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors="ignore")
        offset = offset_col if offset_col != "None" else None

        # Parse interaction pairs
        interaction_pairs: list[tuple[str, str]] | None = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"NB: Using {len(interaction_pairs)} interaction pairs")

        # Start Loading State
        nb_is_running.set(True)
        nb_res.set(None)

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message="Running Negative Binomial Regression...",
                detail="Calculating...",
            )

            try:
                # Run NB Logic (via refactored poisson_lib)
                html_rep, irr_res, airr_res, interaction_res, _ = (
                    analyze_poisson_outcome(
                        target,
                        final_df,
                        var_meta=var_meta.get(),
                        offset_col=offset,
                        interaction_pairs=interaction_pairs,
                        model_type="negative_binomial",
                    )
                )
            except Exception as e:
                err_msg = f"Error running Negative Binomial regression: {e!s}"
                nb_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Negative Binomial regression error")
                nb_is_running.set(False)
                return

            # Generate Forest Plots for IRR
            fig_adj = None
            fig_crude = None

            if airr_res:
                df_adj = pd.DataFrame(
                    [{"variable": k, **v} for k, v in airr_res.items()]
                )
                if not df_adj.empty:
                    try:
                        fig_adj = create_forest_plot(
                            df_adj,
                            "airr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Multivariable: Adjusted IRR</b>",
                            x_label="Adjusted IRR",
                        )
                    except ValueError as e:
                        logger.warning("NB Adjusted Forest Plot creation failed: %s", e)

            if irr_res:
                df_crude = pd.DataFrame(
                    [{"variable": k, **v} for k, v in irr_res.items()]
                )
                if not df_crude.empty:
                    try:
                        fig_crude = create_forest_plot(
                            df_crude,
                            "irr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Univariable: Crude IRR</b>",
                            x_label="Crude IRR",
                        )
                    except ValueError as e:
                        logger.warning("NB Crude Forest Plot creation failed: %s", e)

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Combined Table + Plot) ---
            nb_fragment_html = html_rep

            # Append Adjusted Plot if available, else Crude
            plot_html = ""
            if fig_adj:
                plot_html = fig_adj.to_html(full_html=False, include_plotlyjs="cdn")
                nb_fragment_html += f"<div class='forest-plot-section' ><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"
            elif fig_crude:
                plot_html = fig_crude.to_html(full_html=False, include_plotlyjs="cdn")
                nb_fragment_html += f"<div class='forest-plot-section' ><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # Wrap in standard HTML structure for standalone download correctness
            wrapped_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Negative Binomial Regression Report: {html.escape(target)}</title>
                {get_shiny_css()}
            </head>
            <body>
                <div class="report-container">
                    {nb_fragment_html}
                </div>
            </body>
            </html>
            """
            full_nb_html = wrapped_html

            # Store Results
            nb_res.set(
                {
                    "html_fragment": nb_fragment_html,
                    "html_full": full_nb_html,
                    "fig_adj": fig_adj,
                    "fig_crude": fig_crude,
                }
            )

            ui.notification_show("✅ NB Analysis Complete!", type="message")

        # End Loading State
        nb_is_running.set(False)

    # --- Render NB Results ---
    @render.ui
    def out_nb_status():
        res = nb_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Negative Binomial Regression Complete"),
                class_="info-callout",
            )
        return None

    @render.ui
    def ui_nb_results_area():
        if nb_is_running.get():
            return ui.div(
                create_loading_state("Running Negative Binomial Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = nb_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plots",
                        ui.output_ui("ui_nb_forest_tabs"),
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report",
                        ui.HTML(res["html_fragment"]),
                    ),
                    ui.nav_panel(
                        "📚 Reference",
                        ui.markdown("""
                        ### Negative Binomial Regression Reference
                        
                        **When to Use:**
                        * Overdispersed count data (Variance > Mean)
                        * When Poisson model shows lack of fit due to overdispersion
                        
                        **Interpretation:**
                        * Similar to Poisson (IRR)
                        * **Alpha**: Dispersion parameter estimated by the model
                        * **IRR**: Incidence Rate Ratio
                        """),
                    ),
                ),
                class_="fade-in-entry",
            )

        return create_empty_state_ui(
            message="No Negative Binomial Regression Results",
            sub_message="Select count outcome and predictors, then click 'Run Negative Binomial'.",
            icon="📉",
        )

    @render.ui
    def ui_nb_forest_tabs():
        res = nb_res.get()
        if not res:
            return ui.div(
                "Run analysis to see forest plots.",
                class_="muted-placeholder",
            )

        tabs = []
        if res["fig_crude"]:
            tabs.append(ui.nav_panel("Crude IRR", ui.output_ui("out_nb_forest_crude")))
        if res["fig_adj"]:
            tabs.append(ui.nav_panel("Adjusted IRR", ui.output_ui("out_nb_forest_adj")))

        if not tabs:
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_nb_forest_adj():
        res = nb_res.get()
        if res is None or not res.get("fig_adj"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            res["fig_adj"],
            div_id="plot_nb_forest_adj",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_nb_forest_crude():
        res = nb_res.get()
        if res is None or not res.get("fig_crude"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            res["fig_crude"],
            div_id="plot_nb_forest_crude",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.download(filename="nb_report.html")
    def btn_dl_nb_report():
        res = nb_res.get()
        yield safe_download_html(
            res.get("html_full") if res else None, label="Negative Binomial Report"
        )

    # ==========================================================================
    # LOGIC: Linear Regression (OLS)
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_linear)
    def _run_linear():
        """Run linear regression analysis for continuous outcome with optional MI pooling."""
        d = current_df()
        target = input.linear_outcome()
        predictors = (
            list(input.linear_predictors()) if input.linear_predictors() else None
        )
        exclude = list(input.linear_exclude()) if input.linear_exclude() else []
        method = input.linear_method()
        robust_se = input.linear_robust_se()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show(
                "Please select a continuous outcome variable", type="error"
            )
            return

        # Start Loading State
        linear_is_running.set(True)
        linear_res.set(None)

        # Check for MI datasets
        mi_active = has_mi()
        mi_dfs = mi_datasets_list() if mi_active else []

        with ui.Progress(min=0, max=1) as p:
            try:
                if mi_active and len(mi_dfs) > 0:
                    # ====== MI POOLED LINEAR ANALYSIS ======
                    p.set(
                        message="Running MI Pooled Linear Regression...",
                        detail=f"Analyzing {len(mi_dfs)} imputed datasets...",
                    )

                    all_results = []
                    all_plots = None
                    n_obs_list = []
                    for i, mi_df in enumerate(mi_dfs):
                        p.set(
                            value=(i + 1) / len(mi_dfs),
                            detail=f"Processing linear MI dataset {i + 1}/{len(mi_dfs)}...",
                        )

                        # Centralized cleaning for each imputed dataset
                        # This ensures consistent exclusion and logic (e.g. inf checks)
                        # We pass predictors + target (exclude cols are handled by prepare_data via exclude_columns arg if needed?
                        # Actually prepare_data handles required_columns. We need to define them.
                        req_cols = [target]
                        if predictors:
                            req_cols.extend(predictors)

                        # prepare_data_for_analysis takes [df, required_columns, exclude_columns, var_meta]
                        # Updated to use correct signature (return_info=True returns tuple)
                        cleaned_mi_df, _ = prepare_data_for_analysis(
                            mi_df.drop(columns=exclude, errors="ignore"),
                            required_cols=req_cols,
                            var_meta=var_meta.get(),
                            return_info=True,
                        )

                        n_obs_list.append(len(cleaned_mi_df))

                        _, res_i, plots_i, _ = analyze_linear_outcome(
                            outcome_name=target,
                            df=cleaned_mi_df,
                            predictor_cols=predictors,
                            var_meta=var_meta.get(),
                            exclude_cols=[],  # Already excluded by prepare_data_for_analysis
                            regression_type=method,
                            robust_se=robust_se,
                        )
                        all_results.append(res_i)
                        if all_plots is None:
                            all_plots = plots_i

                    if not all_results:
                        linear_res.set({"error": "All MI models failed"})
                        linear_is_running.set(False)
                        return

                    # Pool β coefficients using Rubin's rules
                    pooled_rows = []
                    first_coef = all_results[0]["coef_table"]
                    for _, row in first_coef.iterrows():
                        var_name = row["Variable"]
                        estimates = []
                        variances = []
                        for res in all_results:
                            coef_tbl = res.get("coef_table")
                            if coef_tbl is not None:
                                match = coef_tbl[coef_tbl["Variable"] == var_name]
                                if len(match) > 0:
                                    beta = match.iloc[0]["Coefficient"]
                                    se = match.iloc[0]["Std. Error"]
                                    if np.isfinite(beta) and se > 0:
                                        estimates.append(beta)
                                        variances.append(se**2)

                        if len(estimates) == len(mi_dfs):
                            # Use average n_obs from cleaned datasets for pooling
                            avg_n_obs = (
                                int(np.mean(n_obs_list)) if n_obs_list else len(d)
                            )
                            pooled = pool_estimates(
                                estimates, variances, n_obs=avg_n_obs
                            )
                            pooled_rows.append(
                                {
                                    "Variable": var_name,
                                    "β": pooled.estimate,
                                    "Std. Error": pooled.se,
                                    "95% CI": f"{pooled.ci_lower:.3f}, {pooled.ci_upper:.3f}",
                                    "p-value": format_p_value(pooled.p_value),
                                    "FMI": f"{pooled.fmi * 100:.1f}%",
                                }
                            )

                    pooled_coef_df = pd.DataFrame(pooled_rows)

                    # Build pooled HTML report
                    html_report = f"""
                    <div class="alert alert-info mb-3">
                        <strong>🔄 Multiple Imputation Analysis</strong><br>
                        Results pooled from {len(mi_dfs)} imputed datasets using Rubin's Rules.
                    </div>
                    <h2>Pooled Coefficients</h2>
                    """
                    # Manually escape columns before rendering to HTML
                    pooled_coef_df_safe = pooled_coef_df.copy()

                    # Escape cell values to prevent XSS (since we use escape=False)
                    pooled_coef_df_safe = pooled_coef_df_safe.map(
                        lambda v: html.escape(str(v))
                    )

                    pooled_coef_df_safe.columns = [
                        html.escape(str(c)) for c in pooled_coef_df_safe.columns
                    ]
                    html_report += pooled_coef_df_safe.to_html(
                        index=False, escape=False, classes="table table-striped"
                    )

                    target_escaped = html.escape(target)
                    full_html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="utf-8">
                        <title>MI Pooled Linear Regression: {target_escaped}</title>
                        <style>
                            body {{ font-family: sans-serif; margin: 20px; }}
                            .table {{ width: 100%; border-collapse: collapse; }}
                            .table th, .table td {{ padding: 8px; border: 1px solid #ddd; }}
                        </style>
                    </head>
                    <body>{html_report}</body>
                    </html>
                    """

                    # Calculate n_obs from MI datasets
                    n_obs_mi = int(np.mean([len(df) for df in mi_dfs]))

                    linear_res.set(
                        {
                            "html_fragment": html_report,
                            "html_full": full_html,
                            "plots": all_plots or {},
                            "results": {
                                "mi_pooled": True,
                                "m": len(mi_dfs),
                                "n_obs": n_obs_mi,
                                "r_squared": float("nan"),
                                "coef_table": pooled_coef_df,
                            },
                            "missing_info": {
                                "mi_pooled": True,
                                "imputation_count": len(mi_dfs),
                                "rows_analyzed": n_obs_mi,
                                "total_missing": "N/A (Imputed)",
                            },
                        }
                    )

                    ui.notification_show(
                        "✅ MI Pooled Linear Regression Complete!", type="message"
                    )

                else:
                    # ====== STANDARD LINEAR ANALYSIS ======
                    p.set(
                        message="Running Linear Regression...",
                        detail="Preparing data...",
                    )

                    html_report, results, plots, missing_info = analyze_linear_outcome(
                        outcome_name=target,
                        df=d,
                        predictor_cols=predictors,
                        var_meta=var_meta.get(),
                        exclude_cols=exclude,
                        regression_type=method,
                        robust_se=robust_se,
                    )

                    target_escaped = html.escape(target)
                    full_linear_html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="utf-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Linear Regression Report: {target_escaped}</title>
                        <style>
                            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
                            .table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                            .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
                            .table th {{ background: #f5f5f5; }}
                            .table-striped tbody tr:nth-child(odd) {{ background: #fafafa; }}
                        </style>
                    </head>
                    <body>
                        <div class="report-container">
                            {html_report}
                        </div>
                    </body>
                    </html>
                    """

                    linear_res.set(
                        {
                            "html_fragment": html_report,
                            "html_full": full_linear_html,
                            "plots": plots,
                            "results": results,
                            "missing_info": missing_info,
                        }
                    )

                    ui.notification_show(
                        "✅ Linear Regression Complete!", type="message"
                    )

            except Exception as e:
                err_msg = f"Error running Linear Regression: {e!s}"
                linear_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Linear regression error")

        # End Loading State
        linear_is_running.set(False)

    # --- Render Linear Regression Results ---
    @render.ui
    def ui_linear_results_area():
        if linear_is_running.get():
            return ui.div(
                create_loading_state("Running Linear Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = linear_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            r2 = res["results"].get("r_squared", 0)
            n_obs = res["results"].get("n_obs", 0)
            r2_text = f"R² = {r2:.4f}" if np.isfinite(r2) else "R² = N/A"

            return create_results_container(
                "Regression Results",
                ui.navset_tab(
                    ui.nav_panel(
                        "📋 Regression Results",
                        ui.div(
                            ui.div(
                                ui.h5(
                                    f"✅ Linear Regression Complete ({r2_text}, n = {n_obs:,})"
                                ),
                                class_="info-callout",
                            ),
                            ui.output_ui("out_linear_html_report"),
                        ),
                    ),
                    ui.nav_panel(
                        "📈 Diagnostic Plots",
                        ui.output_ui("out_linear_diagnostic_plots"),
                    ),
                    ui.nav_panel(
                        "🔍 Variable Selection", ui.output_ui("out_linear_stepwise")
                    ),
                    ui.nav_panel(
                        "🎲 Bootstrap CI", ui.output_ui("out_linear_bootstrap")
                    ),
                    ui.nav_panel(
                        "📚 Reference",
                        ui.markdown("""
                            ### Linear Regression Reference
                            
                            **When to Use:**
                            * Continuous outcomes (blood pressure, glucose, length of stay)
                            * Understanding effect size of predictors (β coefficients)
                            * Analyzing relationships between continuous variables
                            
                            **Interpretation:**
                            * **β > 0**: Positive relationship (Y increases with X)
                            * **β < 0**: Negative relationship (Y decreases with X)
                            * **p < 0.05**: Statistically significant effect
                            * **CI not crossing 0**: Significant effect
                            
                            **Model Fit:**
                            * **R² > 0.7**: Strong explanatory power
                            * **R² 0.4-0.7**: Moderate explanatory power
                            * **R² < 0.4**: Weak explanatory power
                            
                            **Assumptions:**
                            1. **Linearity**: Check Residuals vs Fitted plot
                            2. **Normality**: Check Q-Q plot
                            3. **Homoscedasticity**: Check Scale-Location plot
                            4. **Independence**: Check Durbin-Watson statistic
                            5. **No Multicollinearity**: Check VIF values
                            
                            **Stepwise Selection:**
                            Automatically selects the best subset of variables using AIC, BIC, or p-value criteria.
                            - Forward: Start empty, add significant variables
                            - Backward: Start full, remove non-significant variables
                            - Both: Stepwise forward and backward
                            
                            **Bootstrap CI:**
                            Non-parametric confidence intervals via resampling.
                            - Percentile: Simple quantile-based CIs
                            - BCa: Bias-corrected and accelerated (more accurate)
                            """),
                    ),
                ),
                class_="fade-in-entry",
            )

        return create_empty_state_ui(
            message="No Linear Regression Results",
            sub_message="Select an outcome and predictors, then click 'Run Linear Regression'.",
            icon="📉",
        )

    @render.ui
    def out_linear_html_report():
        """Render the Linear Regression detailed report."""
        res = linear_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Linear Regression Report"),
                ui.HTML(res["html_fragment"]),
            )
        return ui.card(
            ui.card_header("📋 Linear Regression Report"),
            ui.div(
                "Run analysis to see detailed report.",
                class_="muted-placeholder",
            ),
        )

    @render.ui
    def out_linear_diagnostic_plots():
        """Render diagnostic plots for linear regression."""
        res = linear_res.get()
        if not res or not res.get("plots"):
            return ui.div(
                "Run analysis to see diagnostic plots.",
                class_="muted-placeholder",
            )

        plots = res["plots"]
        plot_sections = []

        # Plot descriptions
        plot_info = [
            (
                "residuals_vs_fitted",
                "📊 Residuals vs Fitted",
                "✅ Random scatter = good (linearity, homoscedasticity) | ❌ Pattern = potential issues",
            ),
            (
                "qq_plot",
                "📈 Normal Q-Q Plot",
                "✅ Points on line = normal residuals | ❌ Deviation = non-normality",
            ),
            (
                "scale_location",
                "📉 Scale-Location Plot",
                "✅ Horizontal trend = constant variance | ❌ Slope = heteroscedasticity",
            ),
            (
                "residuals_vs_leverage",
                "🔍 Residuals vs Leverage",
                "✅ Blue points = normal | 🔴 Red points = influential observations (high Cook's D)",
            ),
        ]

        for plot_key, plot_title, plot_desc in plot_info:
            if plot_key in plots and plots[plot_key] is not None:
                plot_html = plotly_figure_to_html(
                    plots[plot_key],
                    div_id=f"linear_diag_{plot_key}",
                    include_plotlyjs="cdn",
                    responsive=True,
                )
                plot_sections.append(
                    ui.card(
                        ui.card_header(plot_title),
                        ui.HTML(plot_html),
                        ui.p(
                            plot_desc,
                            style="font-size: 0.85em; color: #666; margin-top: 10px;",
                        ),
                    )
                )

        if not plot_sections:
            return ui.div("No diagnostic plots available.", class_="text-muted")

        return ui.div(*plot_sections)

    @render.download(filename="linear_regression_report.html")
    def btn_dl_linear_report():
        """Download the complete Linear Regression report as HTML."""
        res = linear_res.get()
        yield safe_download_html(
            res.get("html_full") if res else None, label="Linear Regression Report"
        )

    # --- Stepwise Selection Results ---
    @render.ui
    def out_linear_stepwise():
        """Render stepwise variable selection results."""

        # Check if stepwise is enabled and data available
        if not input.linear_stepwise_enable():
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(
                    "Enable stepwise selection in Advanced Options to use this feature.",
                    class_="muted-placeholder",
                ),
            )

        d = current_df()
        target = input.linear_outcome()
        predictors = (
            list(input.linear_predictors()) if input.linear_predictors() else None
        )
        exclude = list(input.linear_exclude()) if input.linear_exclude() else []

        if d is None or d.empty or not target:
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(
                    "Load data and select outcome to run stepwise selection.",
                    class_="muted-placeholder",
                ),
            )

        # Determine candidate columns
        if predictors:
            candidates = [c for c in predictors if c not in exclude and c != target]
        else:
            numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
            candidates = [c for c in numeric_cols if c != target and c not in exclude]

        if len(candidates) < 2:
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(
                    "Need at least 2 candidate variables for stepwise selection.",
                    class_="muted-placeholder",
                ),
            )

        # Run stepwise selection
        try:
            step_result = stepwise_selection(
                df=d.dropna(subset=[target] + candidates),
                outcome_col=target,
                candidate_cols=candidates,
                direction=input.linear_stepwise_dir(),
                criterion=input.linear_stepwise_crit(),
            )
        except Exception as e:
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(f"Error: {e}", style="color: red; padding: 20px;"),
            )

        # Format history
        history_df = format_stepwise_history(step_result.get("history", []))

        # Sanitize all columns (stepwise doesn't have styled p-values in this tab's helper)
        # Wait, format_stepwise_history might have p-values.
        # Let's check headers.
        df_safe = history_df.copy()
        for col in df_safe.columns:
            df_safe[col] = df_safe[col].astype(str).map(html.escape)

        history_html = df_safe.to_html(
            index=False, escape=False, classes="table table-sm", border=0
        )

        selected = step_result.get("selected_vars", [])
        criterion_val = step_result.get("final_criterion", 0)

        return ui.card(
            ui.card_header(
                f"🔍 Stepwise Selection ({input.linear_stepwise_dir().title()}, {input.linear_stepwise_crit().upper()})"
            ),
            ui.div(
                ui.h5(f"✅ Selected Variables ({len(selected)}):"),
                (
                    ui.tags.ul([ui.tags.li(v) for v in selected])
                    if selected
                    else ui.p("No variables selected")
                ),
                ui.p(
                    f"Final {input.linear_stepwise_crit().upper()}: {criterion_val:.2f}"
                ),
                ui.hr(),
                ui.h5("Selection History:"),
                ui.HTML(history_html),
                style="padding: 15px;",
            ),
        )

    # --- Bootstrap CI Results ---
    @render.ui
    def out_linear_bootstrap():
        """Render bootstrap confidence interval results."""
        if not input.linear_bootstrap_enable():
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    "Enable Bootstrap CIs in Advanced Options to use this feature.",
                    class_="muted-placeholder",
                ),
            )

        d = current_df()
        target = input.linear_outcome()
        predictors = (
            list(input.linear_predictors()) if input.linear_predictors() else None
        )
        exclude = list(input.linear_exclude()) if input.linear_exclude() else []

        if d is None or d.empty or not target:
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    "Load data and select outcome to compute bootstrap CIs.",
                    class_="muted-placeholder",
                ),
            )

        # Determine predictor columns
        if predictors:
            use_predictors = [c for c in predictors if c not in exclude and c != target]
        else:
            numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
            use_predictors = [
                c for c in numeric_cols if c != target and c not in exclude
            ]

        if not use_predictors:
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    "No predictor variables available.",
                    style="color: gray; padding: 20px;",
                ),
            )

        # Run bootstrap
        n_boot = input.linear_bootstrap_n()
        ci_method = input.linear_bootstrap_method()

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message=f"Running {n_boot} bootstrap samples...",
                detail="This may take a moment...",
            )

            try:
                boot_result = bootstrap_ols(
                    df=d.dropna(subset=[target] + use_predictors),
                    outcome_col=target,
                    predictor_cols=use_predictors,
                    n_bootstrap=n_boot,
                    random_state=42,
                )
            except Exception as e:
                return ui.card(
                    ui.card_header("🎲 Bootstrap Confidence Intervals"),
                    ui.div(f"Error: {e}", style="color: red; padding: 20px;"),
                )

        if "error" in boot_result:
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    f"Error: {boot_result['error']}", style="color: red; padding: 20px;"
                ),
            )

        # Format results
        formatted = format_bootstrap_results(boot_result, ci_method=ci_method)

        # Sanitize all columns (bootstrap doesn't have styled p-values usually, but let's be safe)
        df_safe = formatted.copy()
        for col in df_safe.columns:
            df_safe[col] = df_safe[col].astype(str).map(html.escape)

        result_html = df_safe.to_html(
            index=False, escape=False, classes="table table-striped", border=0
        )

        return ui.card(
            ui.card_header(
                f"🎲 Bootstrap CIs (n={boot_result['n_bootstrap']}, {ci_method.upper()})"
            ),
            ui.div(
                ui.HTML(result_html),
                ui.p(
                    f"Failed samples: {boot_result['failed_samples']} | "
                    f"CI Level: {int(boot_result['ci_level'] * 100)}%",
                    style="font-size: 0.85em; color: #666; margin-top: 10px;",
                ),
                style="padding: 15px;",
            ),
        )

    # ==========================================================================
    # LOGIC: Subgroup Analysis
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_subgroup)
    def _run_subgroup():
        d = current_df()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if (
            not input.sg_outcome()
            or not input.sg_treatment()
            or not input.sg_subgroup()
        ):
            ui.notification_show("Please fill all required fields", type="error")
            return

        analyzer = SubgroupAnalysisLogit(d)

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message="Running Subgroup Analysis...", detail="Testing interactions..."
            )

            try:
                results = analyzer.analyze(
                    outcome_col=input.sg_outcome(),
                    treatment_col=input.sg_treatment(),
                    subgroup_col=input.sg_subgroup(),
                    adjustment_cols=list(input.sg_adjust()),
                    min_subgroup_n=input.sg_min_n(),
                    var_meta=var_meta.get(),
                )

                subgroup_res.set(results)
                subgroup_analyzer.set(analyzer)
                ui.notification_show("✅ Subgroup Analysis Complete!", type="message")

            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Subgroup analysis error")

    # --- Render Subgroup Results ---
    @render.ui
    def out_subgroup_status():
        """
        Render a completion banner when subgroup analysis results are available.

        Returns:
            ui.div: A success banner UI element indicating subgroup analysis is complete when results exist, `None` otherwise.
        """
        res = subgroup_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Subgroup Analysis Complete"),
                class_="info-callout",
            )
        return None

    @render.ui
    def out_sg_forest_plot():
        """
        Render the subgroup analysis forest plot as an HTML UI component.

        Generates a Plotly-based forest plot using the current SubgroupAnalysisLogit analyzer and returns it wrapped as a ui.HTML component for embedding in the UI. If the analyzer is not yet available, if plot creation raises a ValueError, or if no figure is produced, returns a styled placeholder ui.div with a short status or warning message instead.

        Returns:
            A UI component: `ui.HTML` containing the plotly-generated HTML when a figure is available, or a `ui.div` placeholder message when results are waiting, missing, or plot creation fails.
        """
        analyzer = subgroup_analyzer.get()
        if analyzer is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        # Use txt_edit_forest_title if provided, fallback to sg_title
        title = input.txt_edit_forest_title() or input.sg_title() or None
        try:
            fig = analyzer.create_forest_plot(title=title)
        except ValueError as e:
            logger.warning("Forest plot creation failed: %s", e)
            return ui.div(
                ui.markdown("⚠️ *Run analysis first to generate forest plot.*"),
                class_="muted-placeholder",
            )
        if fig is None:
            return ui.div(
                ui.markdown("⏳ *No forest plot available...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            fig, div_id="plot_logit_subgroup", include_plotlyjs="cdn", responsive=True
        )
        return ui.HTML(html_str)

    @reactive.Effect
    @reactive.event(input.btn_update_plot_title)
    def _update_sg_title():
        # Invalidate to trigger re-render of the forest plot widget
        subgroup_analyzer.set(subgroup_analyzer.get())

    @render.text
    def val_overall_or():
        res = subgroup_res.get()
        if res:
            overall = res.get("overall", {})
            or_val = overall.get("or")
            return f"{or_val:.3f}" if or_val is not None else "N/A"
        return "-"

    @render.text
    def val_overall_p():
        res = subgroup_res.get()
        if res:
            return format_p_value(res["overall"]["p_value"], use_style=False)
        return "-"

    @render.text
    def val_interaction_p():
        res = subgroup_res.get()
        if res:
            p_int = res["interaction"]["p_value"]
            return (
                format_p_value(p_int, use_style=False) if p_int is not None else "N/A"
            )
        return "-"

    @render.ui
    def out_interpretation_box():
        res = subgroup_res.get()
        analyzer = subgroup_analyzer.get()
        if res and analyzer:
            interp = analyzer.get_interpretation()
            is_het = res["interaction"]["significant"]
            color = "alert-warning" if is_het else "alert-success"
            icon = "⚠️" if is_het else "✅"
            return ui.div(f"{icon} {interp}", class_=f"alert {color}")
        return None

    @render.data_frame
    def out_sg_table():
        res = subgroup_res.get()
        if res:
            df_res = res["results_df"].copy()
            # Simple formatting for display
            cols = ["group", "n", "events", "or", "ci_low", "ci_high", "p_value"]
            available_cols = [c for c in cols if c in df_res.columns]
            return render.DataGrid(df_res[available_cols].round(4))
        return None

    @render.ui
    def out_sg_missing_report() -> ui.TagChild | None:
        res = subgroup_res.get()
        if res and "missing_data_info" in res:
            return ui.HTML(
                create_missing_data_report_html(
                    res["missing_data_info"], var_meta.get() or {}
                )
            )
        return None

    # --- Subgroup Downloads ---
    @render.download(filename=lambda: f"subgroup_plot_{input.sg_subgroup()}.html")
    def dl_sg_html():
        analyzer = subgroup_analyzer.get()
        if analyzer and analyzer.figure:
            html_payload = analyzer.figure.to_html(include_plotlyjs="cdn")
            safe_data_download(html_payload, label="Subgroup Plot HTML")
            yield html_payload
        else:
            safe_data_download(None, label="Subgroup Plot HTML")
            yield "No plot available"

    @render.download(filename=lambda: f"subgroup_res_{input.sg_subgroup()}.csv")
    def dl_sg_csv():
        res = subgroup_res.get()
        if res and "results_df" in res:
            csv_payload = res["results_df"].to_csv(index=False)
            safe_data_download(csv_payload, label="Subgroup Results CSV")
            yield csv_payload
        else:
            safe_data_download(None, label="Subgroup Results CSV")
            yield "No results available"

    @render.download(filename=lambda: f"subgroup_data_{input.sg_subgroup()}.json")
    def dl_sg_json():
        """
        Produce a JSON-formatted representation of the latest subgroup analysis results.

        Yields:
            str: A JSON-formatted string of the subgroup results (indent=2). Non-JSON-native types (e.g., NumPy scalars/arrays) are converted to strings to ensure serializability.
        """
        res = subgroup_res.get()
        if res:
            json_payload = json.dumps(res, indent=2, default=str)
            safe_data_download(json_payload, label="Subgroup Results JSON")
            yield json_payload
        else:
            safe_data_download(None, label="Subgroup Results JSON")
            yield "No results available"

    # =========================================================================
    # LOGISTIC SUBGROUP SERVER LOGIC
    # =========================================================================

    @render.ui
    def out_sg_logit_status():
        """
        Render a loading indicator while the logistic subgroup analysis is running.

        Returns:
            ui.TagChild | None: A loading UI element when the subgroup analysis is in progress, otherwise None.
        """
        if logit_sg_is_running.get():
            return create_loading_state("Running Subgroup Analysis...")
        return None

    @reactive.Effect
    @reactive.event(input.btn_run_sg_logit)
    def _run_sg_logit():
        """
        Execute a logistic subgroup analysis using the current dataset and UI selections.

        Validates that outcome, treatment, and subgroup are selected, shows progress notifications, and sets the running state while the analysis executes. On success stores the analysis results (including a generated `forest_plot`) in `logit_sg_res`; on error stores an error in `logit_sg_res` and displays an error notification. Does not return a value.
        """
        d = current_df()
        y = input.sg_logit_outcome()
        treat = input.sg_logit_treatment()
        subgroup = input.sg_logit_subgroup()
        adjust = input.sg_logit_adjust()
        min_n = input.sg_logit_min_n()

        if d is None:
            return

        if not all([y, treat, subgroup]) or any(
            x == "Select..." for x in [y, treat, subgroup]
        ):
            ui.notification_show("Please select all required variables", type="warning")
            return

        logit_sg_is_running.set(True)
        logit_sg_res.set(None)
        ui.notification_show(
            "Running Subgroup Analysis...", duration=None, id="run_sg_logit"
        )

        try:
            analyzer = SubgroupAnalysisLogit(d)
            result = analyzer.analyze(
                outcome_col=y,
                treatment_col=treat,
                subgroup_col=subgroup,
                adjustment_cols=list(adjust) if adjust else None,
                min_subgroup_n=min_n,
                var_meta=var_meta.get(),
            )

            if "error" in result:
                logit_sg_res.set({"error": result["error"]})
                ui.notification_show(result["error"], type="error")
                ui.notification_remove("run_sg_logit")
                return

            # Generate forest plot
            forest_fig = analyzer.create_forest_plot()
            result["forest_plot"] = forest_fig

            logit_sg_res.set(result)
            ui.notification_remove("run_sg_logit")
            ui.notification_show("✅ Analysis Complete", type="message")

        except Exception as e:
            ui.notification_remove("run_sg_logit")
            ui.notification_show(f"Analysis failed: {e}", type="error")
            logger.exception("Logit Subgroup Analysis Error")
        finally:
            logit_sg_is_running.set(False)

    @render.ui
    def out_sg_logit_result():
        """
        Render the logistic subgroup analysis results UI.

        If no results are available, returns a placeholder prompting the user to run the analysis.
        If the results contain an error, returns an error alert. Otherwise, returns a composed UI
        containing a forest plot card, a detailed results table card, and an interaction test card
        displaying the interaction p-value and a heterogeneity message.

        Returns:
            ui.TagChild: A UI element representing the subgroup analysis output (placeholder, error alert,
            or cards with forest plot, results table, and interaction test).
        """
        res = logit_sg_res.get()
        if res is None:
            return create_placeholder_state("Run analysis to see results", "🔛")
        if "error" in res:
            return create_error_alert(res["error"])

        # Create summary table
        summary_df = pd.DataFrame(res["results_df"])
        # Format P-values
        if "p_value" in summary_df.columns:
            summary_df["p_value"] = summary_df["p_value"].apply(
                lambda x: format_p_value(x) if isinstance(x, numbers.Real) else x
            )

        # Sanitize all non-p-value columns to prevent XSS
        df_safe = summary_df.copy()
        for col in df_safe.columns:
            if col != "p_value":
                df_safe[col] = df_safe[col].astype(str).map(html.escape)

        table_html = df_safe.to_html(
            classes="table table-striped table-hover", index=False, escape=False
        )

        # Plot
        fig = res.get("forest_plot")
        plot_html = plotly_figure_to_html(fig) if fig else ""

        return ui.div(
            ui.card(
                ui.card_header("🌳 Forest Plot (Treatment Effect by Subgroup)"),
                ui.HTML(plot_html),
            ),
            ui.br(),
            ui.card(
                ui.card_header("📊 Detailed Results"),
                ui.HTML(table_html),
            ),
            ui.br(),
            ui.card(
                ui.card_header("Interaction Test"),
                ui.div(
                    ui.p(
                        f"P-interaction: {format_p_value(res['interaction']['p_value'])}"
                        if res["interaction"]["p_value"] is not None
                        else "N/A"
                    ),
                    ui.p(
                        "Significant Heterogeneity detected."
                        if res["interaction"]["significant"]
                        else "No significant heterogeneity detected."
                    ),
                    class_=(
                        "alert alert-info"
                        if not res["interaction"]["significant"]
                        else "alert alert-warning"
                    ),
                ),
            ),
        )

    @render.download(filename="logit_subgroup_report.html")
    def btn_dl_sg_logit():
        """
        Produce an HTML report for the completed logistic subgroup analysis.

        Yields a single HTML string containing a forest plot and a detailed results table when analysis results exist; yields the message "No results available." if no results are present.

        Returns:
            generator (str): Yields the HTML report string or an availability message.
        """

        def _build():
            res = logit_sg_res.get()
            if not res:
                return None

            return f"""
            <html>
            <head><title>Subgroup Analysis Report</title></head>
            <body>
                <h1>Subgroup Analysis (Logistic Regression)</h1>
                <h2>Forest Plot</h2>
                {plotly_figure_to_html(res.get("forest_plot"))}
                <h2>Detailed Results</h2>
                {pd.DataFrame(res["results_df"]).to_html()}
            </body>
            </html>
            """

        yield safe_report_generation(_build, label="Logistic Subgroup Report")

    # ==========================================================================
    # LOGIC: Repeated Measures
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_repeated)
    def _run_repeated():
        d = current_df()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return

        outcome = input.rep_outcome()
        treatment = input.rep_treatment()
        time_var = input.rep_time()
        subject = input.rep_subject()

        if not all([outcome, treatment, time_var, subject]):
            ui.notification_show(
                "Please select all required variables (Outcome, Treatment, Time, Subject)",
                type="error",
            )
            return

        model_type = input.rep_model_type()
        covariates = list(input.rep_covariates()) if input.rep_covariates() else []

        # Exclude rows with missing data in selected columns
        # Exclude rows with missing data in selected columns
        cols_needed = [outcome, treatment, time_var, subject] + covariates

        # Use centralized cleaning
        d, missing_info = prepare_data_for_analysis(
            d, required_cols=cols_needed, var_meta=var_meta.get(), return_info=True
        )

        # Start Loading State
        repeated_is_running.set(True)
        repeated_res.set(None)

        with ui.Progress(min=0, max=1) as p:
            p.set(message=f"Running {model_type.upper()}...", detail="Analyzing...")

            try:
                if model_type == "gee":
                    results, missing_info = run_gee(
                        d,  # Cleaned via prepare_data_for_analysis above
                        outcome_col=outcome,
                        treatment_col=treatment,
                        time_col=time_var,
                        subject_col=subject,
                        covariates=covariates,
                        cov_struct=input.rep_cov_struct(),
                        family_str=input.rep_family(),
                        var_meta=var_meta.get() or {},
                    )
                else:  # lmm
                    results, missing_info = run_lmm(
                        d,
                        outcome_col=outcome,
                        treatment_col=treatment,
                        time_col=time_var,
                        subject_col=subject,
                        covariates=covariates,
                        random_slope=input.rep_random_slope(),
                        var_meta=var_meta.get() or {},
                    )

                # Use the indices from missing_info to get the cleaned df for plotting
                # (since the original code used df_clean for create_trajectory_plot)
                df_clean_subset = (
                    d.loc[missing_info.get("analyzed_indices", [])]
                    if "analyzed_indices" in missing_info
                    else d
                )

                # Check for error string
                if isinstance(results, str):
                    repeated_res.set({"error": results})
                    ui.notification_show("Analysis failed", type="error")
                    return

                # Extract Results
                df_res = extract_model_results(results, model_type)

                # Create Plot
                fig = create_trajectory_plot(
                    df_clean_subset,
                    outcome_col=outcome,
                    time_col=time_var,
                    group_col=treatment,
                    subject_col=subject,
                )

                repeated_res.set(
                    {
                        "results": df_res,
                        "plot": fig,
                        "model_type": model_type,
                        "missing_data_info": missing_info,
                    }
                )

                ui.notification_show(
                    f"✅ {model_type.upper()} Analysis Complete!", type="message"
                )

            except Exception as e:
                err_msg = f"Error running Repeated Measures: {e!s}"
                repeated_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Repeated measures error")
            finally:
                repeated_is_running.set(False)

    @render.ui
    def ui_repeated_results_area():
        if repeated_is_running.get():
            return ui.div(
                create_loading_state("Running Repeated Measures Analysis..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = repeated_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            return create_results_container(
                "Analysis Results",
                ui.div(
                    ui.navset_tab(
                        ui.nav_panel(
                            "📋 Model Results",
                            ui.div(
                                ui.div(
                                    ui.h5(
                                        f"✅ {res['model_type'].upper()} Analysis Complete"
                                    ),
                                    class_="info-callout",
                                ),
                                ui.output_data_frame("out_rep_results"),
                            ),
                        ),
                        ui.nav_panel(
                            "📈 Trajectory Plot", ui.output_ui("out_rep_plot")
                        ),
                    ),
                    ui.hr(),
                    ui.HTML(
                        create_missing_data_report_html(
                            res.get("missing_data_info", {}), var_meta.get() or {}
                        )
                    ),
                ),
                class_="fade-in-entry",
            )

        # Default Placeholder
        return create_empty_state_ui(
            message="No Repeated Measures Results",
            sub_message="Configure Outcome, Subject ID, and Time, then click '🚀 Run analysis'.",
            icon="🔄",
        )

    @render.data_frame
    def out_rep_results():
        res = repeated_res.get()
        if res:
            return render.DataGrid(res["results"])
        return None

    @render.ui
    def out_rep_plot():
        res = repeated_res.get()
        if res and res["plot"]:
            return ui.HTML(plotly_figure_to_html(res["plot"], include_plotlyjs="cdn"))
        return ui.div(
            "Run analysis to see trajectory plot.",
            class_="muted-placeholder",
        )

    # --- GLM Logic (Tab 2.5) ---
    @render.ui
    def ui_glm_results_area():
        if glm_processing.get():
            return ui.div(
                create_loading_state("Running Generalized Linear Model..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = glm_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            metrics = res["fit_metrics"]
            # We can put the status banner inside the report or as a separate div if needed.
            # But create_results_container is mostly content.
            # Let's include the status banner inside the report panel or forest plot panel?
            # Or just render it on top of the generic content.
            # But ui_glm_results_area is inside create_results_container.

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel(
                        "📋 Model Results",
                        ui.div(
                            ui.div(
                                ui.h5(
                                    f"✅ Analysis Complete (AIC: {metrics.get('aic', 'N/A'):.2f}, Deviance: {metrics.get('deviance', 'N/A'):.2f})"
                                ),
                                class_="info-callout",
                            ),
                            ui.HTML(res["html_report"]),
                        ),
                    ),
                    ui.nav_panel(
                        "🌳 Forest Plot",
                        (
                            ui.HTML(
                                plotly_figure_to_html(
                                    res["forest_plot"], include_plotlyjs="cdn"
                                )
                            )
                            if res.get("forest_plot")
                            else ui.div(
                                "No forest plot available", class_="text-muted p-3"
                            )
                        ),
                    ),
                ),
                class_="fade-in-entry",
            )

        return create_placeholder_state(
            "Select an outcome and predictors, then click 'Run GLM'.", icon="📈"
        )

    @reactive.Effect
    @reactive.event(input.btn_run_glm)
    def _run_glm():
        d = current_df()
        req(d is not None, input.glm_outcome(), input.glm_predictors())

        glm_processing.set(True)
        glm_res.set(None)

        try:
            # Prepare Data
            outcome = input.glm_outcome()
            predictors = list(input.glm_predictors())
            interactions = list(input.glm_interactions())

            # Simple interaction handling (create dummy cols in a copy)
            X = d[predictors].copy()
            y = pd.to_numeric(d[outcome], errors="coerce")

            # Create interactions if any
            if interactions:
                for pair_str in interactions:
                    if " × " in pair_str:
                        v1, v2 = pair_str.split(" × ")
                        if v1 in d.columns and v2 in d.columns:
                            # Convert to numeric for interaction
                            p1 = pd.to_numeric(d[v1], errors="coerce")
                            p2 = pd.to_numeric(d[v2], errors="coerce")
                            col_name = f"{v1}:{v2}"
                            X[col_name] = p1 * p2

            # Drop Data with NaNs
            valid_idx = y.notna() & X.notna().all(axis=1)
            y = y[valid_idx]
            X = X[valid_idx]

            # Run GLM
            params, conf_int, pvalues, status, metrics = run_glm(
                y, X, family_name=input.glm_family(), link_name=input.glm_link()
            )

            if status != "OK":
                ui.notification_show(f"GLM Failed: {status}", type="error")
                return

            # Format Results for Forest Plot & Table
            res_df = pd.DataFrame(
                {
                    "var": params.index,
                    "coef": params.values,
                    "ci_low": conf_int[0].values,
                    "ci_high": conf_int[1].values,
                    "p_value": pvalues.values,
                }
            )

            # Exclude constant from forest plot usually
            plot_df = res_df[res_df["var"] != "const"]

            # Generate Forest Plot
            forest_data = []
            link = input.glm_link()
            is_ratio = link in ["log", "logit", "cloglog"]

            for _, row in plot_df.iterrows():
                val = np.exp(row["coef"]) if is_ratio else row["coef"]
                low = np.exp(row["ci_low"]) if is_ratio else row["ci_low"]
                high = np.exp(row["ci_high"]) if is_ratio else row["ci_high"]

                forest_data.append(
                    {
                        "label": row["var"],
                        "mean": val,
                        "lower": low,
                        "upper": high,
                        "p_value": row["p_value"],
                        "is_ratio": is_ratio,
                    }
                )

            try:
                forest_df = pd.DataFrame(forest_data)
                fig = create_forest_plot(
                    forest_df,
                    estimate_col="mean",
                    ci_low_col="lower",
                    ci_high_col="upper",
                    label_col="label",
                    pval_col="p_value",
                    title=f"GLM ({input.glm_family()}/{input.glm_link()}) Results",
                    x_label="Exp(Coef) [OR/RR]" if is_ratio else "Coefficient",
                )
            except ValueError as e:
                logger.warning("GLM Forest Plot creation failed: %s", e)
                fig = None

            # Generate HTML Report
            html_parts = [
                f"<h4>GLM Results: {html.escape(outcome)}</h4>",
                f"<p><b>Family:</b> {input.glm_family()} | <b>Link:</b> {input.glm_link()}</p>",
                f"<p><b>AIC:</b> {metrics.get('aic', 'N/A'):.2f} | <b>Deviance:</b> {metrics.get('deviance', 'N/A'):.2f}</p>",
                "<table class='table table-striped table-sm'>",
                "<thead><tr><th>Variable</th><th>Coef</th><th>Exp(Coef)</th><th>95% CI</th><th>P-value</th></tr></thead>",
                "<tbody>",
            ]

            for _, row in res_df.iterrows():
                coef = row["coef"]
                exp_coef = np.exp(coef)  # Always show exp coef for reference
                ci_l = row["ci_low"]
                ci_h = row["ci_high"]
                p = row["p_value"]

                p_fmt = format_p_value(p)
                p_style = "color:red; font-weight:bold;" if p < 0.05 else ""

                # CI Display based on link
                if is_ratio:
                    ci_disp = PublicationFormatter.format_ci(np.exp(ci_l), np.exp(ci_h))
                else:
                    ci_disp = PublicationFormatter.format_ci(ci_l, ci_h)

                html_parts.append(
                    f"<tr>"
                    f"<td>{html.escape(str(row['var']))}</td>"
                    f"<td>{coef:.3f}</td>"
                    f"<td>{exp_coef:.3f}</td>"
                    f"<td>{ci_disp}</td>"
                    f"<td style='{p_style}'>{p_fmt}</td>"
                    f"</tr>"
                )
            html_parts.append("</tbody></table>")

            glm_res.set(
                {
                    "fit_metrics": metrics,
                    "params": params,
                    "forest_plot": fig,
                    "html_report": "".join(html_parts),
                }
            )

        except Exception as e:
            err_msg = f"Error running GLM: {e!s}"
            glm_res.set({"error": err_msg})
            ui.notification_show("GLM Failed", type="error")
            logger.exception("GLM Fatal Error")

        finally:
            glm_processing.set(False)

    @render.download(filename="glm_report.html")
    def btn_dl_glm_report():
        res = glm_res.get()
        yield safe_download_html(
            res.get("html_report") if res else None, label="GLM Report"
        )

    # --- PDF Download Handlers ---
    @render.download(filename="logit_report.pdf")
    def btn_dl_report_pdf():
        res = logit_res.get()
        yield safe_download_pdf(
            res.get("html_full") if res else None, label="Logistic Regression Report"
        )

    @render.download(filename="logit_subgroup_report.pdf")
    def btn_dl_sg_logit_pdf():
        def _build():
            res = logit_sg_res.get()
            if not res:
                return None
            return f"""
            <html>
            <head><title>Subgroup Analysis Report</title></head>
            <body>
            <h1>Logistic Regression Subgroup Analysis</h1>
            {plotly_figure_to_html(res.get("forest_plot"))}
            {res.get("results_df", pd.DataFrame()).to_html(classes="table table-hover", index=False, escape=True)}
            </body></html>
            """

        yield safe_pdf_report_generation(_build, label="Subgroup Analysis Report")

    @render.download(filename="poisson_report.pdf")
    def btn_dl_poisson_pdf():
        res = poisson_res.get()
        yield safe_download_pdf(
            res.get("html_full") if res else None, label="Poisson Regression Report"
        )

    @render.download(filename="nb_report.pdf")
    def btn_dl_nb_pdf():
        res = nb_res.get()
        yield safe_download_pdf(
            res.get("html_full") if res else None, label="Negative Binomial Report"
        )

    @render.download(filename="glm_report.pdf")
    def btn_dl_glm_pdf():
        res = glm_res.get()
        yield safe_download_pdf(
            res.get("html_report") if res else None, label="GLM Report"
        )

    @render.download(filename="linear_regression_report.pdf")
    def btn_dl_linear_pdf():
        res = linear_res.get()
        yield safe_download_pdf(
            res.get("html_full") if res else None, label="Linear Regression Report"
        )
