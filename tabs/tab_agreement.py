from __future__ import annotations

import html as _html
import re
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
from utils import diag_test
from utils.agreement_lib import AgreementAnalysis
from utils.download_helpers import safe_download_html
from utils.formatting import create_missing_data_report_html
from utils.pdf_helpers import safe_download_pdf
from utils.ui_helpers import (
    create_download_status_badge,
    create_error_alert,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
)

logger = get_logger(__name__)
COLORS = get_color_palette()


def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s[:80] or "value"


def _auto_detect_icc_vars(cols: list[str]) -> list[str]:
    """Auto-detect ICC/Rater variables based on column name patterns."""
    icc_patterns = ["diagnosis", "icc", "rater", "method", "observer", "judge"]
    detected = []
    for col in cols:
        col_lower = col.lower()
        for pattern in icc_patterns:
            if pattern in col_lower:
                detected.append(col)
                break
    return detected


# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def agreement_ui() -> ui.TagChild:
    """
    Constructs the Agreement & Reliability analysis UI for selecting data and running Kappa, Bland–Altman, and ICC analyses.

    The returned UI contains dataset selection and a tabbed interface with:
    - Kappa (Cohen's or Fleiss' Kappa) inputs, validation, actions, results, and report download.
    - Bland–Altman inputs (including CI options), validation, actions, results, and report download.
    - ICC inputs, validation, actions, results, and report download.
    - A reference/interpretation panel describing Kappa, Bland–Altman, and ICC.

    Returns:
        ui.TagChild: A Shiny TagChild representing the complete agreement/reliability module UI.
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
            # TAB 1: Kappa (Categorical Agreement)
            ui.nav_panel(
                "🤝 Kappa",
                ui.markdown("##### Categorical Agreement (Kappa)"),
                ui.input_radio_buttons(
                    "kappa_mode",
                    "Method:",
                    {
                        "cohen": "Cohen's Kappa (2 Raters)",
                        "fleiss": "Fleiss' Kappa (>2 Raters)",
                    },
                    inline=True,
                ),
                ui.panel_conditional(
                    "input.kappa_mode === 'cohen'",
                    ui.row(
                        ui.column(4, ui.output_ui("ui_kappa_v1")),
                        ui.column(4, ui.output_ui("ui_kappa_v2")),
                        ui.column(4, ui.output_ui("ui_kappa_weights")),
                    ),
                ),
                ui.panel_conditional(
                    "input.kappa_mode === 'fleiss'", ui.output_ui("ui_fleiss_vars")
                ),
                ui.output_ui("out_kappa_validation"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_kappa",
                            "🚀 Calculate Kappa",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_kappa_report",
                            "📥 HTML",
                            class_="btn-secondary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_kappa_pdf",
                            "📥 PDF",
                            class_="btn-outline-danger w-100",
                        ),
                        ui.output_ui("dl_status_kappa"),
                    ),
                ),
                ui.br(),
                ui.output_ui("out_kappa_results"),
            ),
            # TAB 2: Bland-Altman (Continuous Agreement)
            ui.nav_panel(
                "📉 Bland-Altman",
                ui.markdown("##### Bland-Altman Analysis (Continuous Data Comparison)"),
                ui.row(
                    ui.column(4, ui.output_ui("ui_ba_v1")),
                    ui.column(4, ui.output_ui("ui_ba_v2")),
                    ui.column(
                        4,
                        ui.input_checkbox("ba_show_ci", "Show CI Bands", value=True),
                        ui.input_numeric(
                            "ba_conf_level",
                            "Confidence Level:",
                            0.95,
                            min=0.80,
                            max=0.99,
                            step=0.01,
                        ),
                    ),
                ),
                ui.output_ui("out_ba_validation"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_ba",
                            "🚀 Analyze Agreement",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_ba_report",
                            "📥 HTML",
                            class_="btn-secondary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_ba_pdf",
                            "📥 PDF",
                            class_="btn-outline-danger w-100",
                        ),
                        ui.output_ui("dl_status_ba"),
                    ),
                ),
                ui.br(),
                ui.output_ui("out_ba_results"),
            ),
            # TAB 3: ICC (Reliability)
            ui.nav_panel(
                "🔍 Reliability (ICC)",
                ui.markdown("##### Intraclass Correlation Coefficient (ICC)"),
                ui.row(
                    ui.column(12, ui.output_ui("ui_icc_vars")),
                ),
                # Optional: Add ICC type selector here if needed
                ui.output_ui("out_icc_validation"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_run_icc",
                            "🔍 Calculate ICC",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_icc_report",
                            "📥 HTML",
                            class_="btn-secondary w-100",
                        ),
                    ),
                    ui.column(
                        3,
                        ui.download_button(
                            "btn_dl_icc_pdf",
                            "📥 PDF",
                            class_="btn-outline-danger w-100",
                        ),
                        ui.output_ui("dl_status_icc"),
                    ),
                ),
                ui.br(),
                ui.output_ui("out_icc_results"),
            ),
            # TAB 4: Reference & Interpretation
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                    ## 📚 Agreement & Reliability Reference Guide

                    ### 🤝 Cohen's Kappa & Fleiss' Kappa
                    - **Cohen's Kappa:** For agreement between **two** raters.
                    - **Fleiss' Kappa:** For agreement between **three or more** raters.
                    
                    **Landis–Koch (1977) scale:**
                    - **> 0.81:** Almost perfect agreement ✅
                    - **0.61–0.80:** Substantial agreement
                    - **0.41–0.60:** Moderate agreement
                    - **0.21–0.40:** Fair agreement ⚠️
                    - **< 0.20:** Slight/Poor agreement ❌

                    ### 📉 Bland-Altman Plot
                    Used for **continuous** data to compare two measurement methods.
                    - **Bias (Mean Difference):** Systematic difference.
                    - **Limits of Agreement (LoA):** Interval containing 95% of differences.
                    - **Confidence Intervals (Shaded):** Shows the precision of the Bias and LoA estimates.

                    ### 🔍 Intraclass Correlation (ICC)
                    Measures reliability/consistency.
                    
                    **ICC Forms (Shrout & Fleiss, 1979):**
                    - **ICC1:** One-way random effects (raters selected at random).
                    - **ICC2:** Two-way random effects (raters and subjects random).
                    - **ICC3:** Two-way mixed effects (fixed raters).
                    
                    **Interpretation (Cicchetti, 1994):**
                    - **> 0.75:** Excellent 🌟
                    - **0.60 – 0.75:** Good
                    - **0.40 – 0.60:** Fair ⚠️
                    - **< 0.40:** Poor ❌
                    """),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def agreement_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
) -> None:
    # --- Reactive Results ---
    """
    Initialize server logic for the Agreement & Reliability Shiny module, wiring UI inputs to reactive analyses for Kappa (Cohen/Fleiss), Bland–Altman, and ICC and providing result rendering and download handlers.

    Parameters:
        input (Any): Shiny input bindings (provides UI control values and events).
        output (Any): Shiny output bindings (used to expose UI renderables).
        session (Any): Shiny session object for managing the user's session.
        df (reactive.Value[pd.DataFrame | None]): Primary dataset reactive value.
        var_meta (reactive.Value[dict[str, Any]]): Reactive variable metadata used for display and missing-data reporting.
        df_matched (reactive.Value[pd.DataFrame | None]): Optional matched dataset reactive value (e.g., from propensity score matching).
        is_matched (reactive.Value[bool]): Reactive flag indicating whether a matched dataset is available and selectable.
    """
    kappa_html: reactive.Value[str | None] = reactive.Value(None)
    ba_html: reactive.Value[str | None] = reactive.Value(None)
    icc_html: reactive.Value[str | None] = reactive.Value(None)

    kappa_processing: reactive.Value[bool] = reactive.Value(False)
    ba_processing: reactive.Value[bool] = reactive.Value(False)
    icc_processing: reactive.Value[bool] = reactive.Value(False)

    # --- Dataset Selection Logic ---
    current_df = register_dataset_selector(
        input=input,
        output=output,
        df=df,
        df_matched=df_matched,
        is_matched=is_matched,
        radio_input_id="radio_source",
        title="🤝 Agreement & Reliability",
    )

    # Get all columns
    @reactive.Calc
    def all_cols() -> list[str]:
        d = current_df()
        if d is not None:
            return d.columns.tolist()
        return []

    @reactive.Calc
    def num_cols() -> list[str]:
        d = current_df()
        if d is not None:
            return d.select_dtypes(include=[np.number]).columns.tolist()
        return []

    # --- Kappa Inputs ---
    @render.ui
    def ui_kappa_v1():
        """
        Render a select input for choosing the first rater (row) used in Kappa analysis.

        The input is populated with all columns from the current dataset and defaults to a column likely to represent a rater (searched in order: "diagnosis_dr_a", "rater1", "obs1", "method1"); if none match, the first column is selected.

        Returns:
            The Shiny select input UI element with id "sel_kappa_v1" and label "Rater 1 (Row):".
        """
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["diagnosis_dr_a", "rater1", "obs1", "method1"], default_to_first=True
        )
        return ui.input_select(
            "sel_kappa_v1", "Rater 1 (Row):", choices=cols, selected=default
        )

    @render.ui
    def ui_kappa_v2():
        """
        Create a select input for choosing the second rater/column for Kappa analysis.

        The control lists all dataset columns except the column currently selected for Rater 1.
        The default selection is inferred from common rater/second-column keywords or falls back to the first available choice.

        Returns:
            A UI input select control (id "sel_kappa_v2") labeled "Rater 2 (Col):" populated with the remaining columns.
        """
        cols = all_cols()
        v1 = input.sel_kappa_v1()
        rem_cols = [c for c in cols if c != v1] if v1 else cols
        default = select_variable_by_keyword(
            rem_cols,
            ["diagnosis_dr_b", "rater2", "obs2", "method2"],
            default_to_first=True,
        )
        return ui.input_select(
            "sel_kappa_v2", "Rater 2 (Col):", choices=rem_cols, selected=default
        )

    @render.ui
    def ui_fleiss_vars():
        """
        Create a selectize input for choosing three or more rater columns.

        The input is populated with the current dataset's column names and pre-selects columns that resemble rater/observer/method variables.

        Returns:
            ui.input_selectize: A Shiny selectize input (id "sel_fleiss_vars") configured for multiple selection of rater columns.
        """
        cols = all_cols()
        defaults = _auto_detect_icc_vars(cols)
        return ui.input_selectize(
            "sel_fleiss_vars",
            "Select 3+ Raters:",
            choices=cols,
            multiple=True,
            selected=defaults,
            width="100%",
        )

    # --- Bland-Altman Inputs ---
    @render.ui
    def ui_ba_v1():
        cols = num_cols()
        default = select_variable_by_keyword(
            cols, ["rater1", "method1", "measure1", "bp_r1"], default_to_first=True
        )
        return ui.input_select(
            "sel_ba_v1", "Method A (Reference):", choices=cols, selected=default
        )

    @render.ui
    def ui_ba_v2():
        cols = num_cols()
        v1 = input.sel_ba_v1()
        rem_cols = [c for c in cols if c != v1] if v1 else cols
        default = select_variable_by_keyword(
            rem_cols, ["rater2", "method2", "measure2", "bp_r2"], default_to_first=True
        )
        return ui.input_select(
            "sel_ba_v2", "Method B (New):", choices=cols, selected=default
        )

    # --- ICC Inputs ---
    @render.ui
    def ui_icc_vars():
        """
        Create a selectize input for choosing two or more numeric variables to use as raters/methods in ICC calculations.

        The control is populated with numeric columns from the current dataset and preselects likely ICC candidates detected by `_auto_detect_icc_vars`.

        Returns:
            A Shiny selectize input component (id "icc_vars") configured for multiple selection and initialized with detected defaults.
        """
        cols = num_cols()
        defaults = _auto_detect_icc_vars(cols)
        return ui.input_selectize(
            "icc_vars",
            "Select Variables (2+ Raters/Methods):",
            choices=cols,
            multiple=True,
            selected=defaults,
            width="100%",
        )

    @render.ui
    def ui_kappa_weights():
        """
        Create a select input for choosing Kappa weighting for ordinal categorical agreement.

        The control has id "sel_kappa_weights" and label "Weights (Ordinal):" with three options:
        Unweighted (empty string), Linear ("linear"), and Quadratic ("quadratic").

        Returns:
            The Shiny select input component configured for Kappa weight selection.
        """
        return ui.input_select(
            "sel_kappa_weights",
            "Weights (Ordinal):",
            choices={
                "": "Unweighted",
                "linear": "Linear",
                "quadratic": "Quadratic",
            },
            selected="",
        )

    # ==================== KAPPA LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_analyze_kappa)
    def _run_kappa():
        """
        Run the configured Kappa agreement analysis (Cohen or Fleiss) and update the module's report state.

        Validates required inputs, sets the processing flag while running, executes the selected analysis:
        - For "cohen": requires two selected rater variables and computes Cohen's Kappa (with optional weights).
        - For "fleiss": requires three or more selected rater variables and computes Fleiss' Kappa.

        On success, stores a generated HTML report (statistics, matrices/plots, and an optional missing-data section) in kappa_html; on failure, stores an error alert HTML in kappa_html. Always clears the processing flag when finished.
        """
        d = current_df()
        mode = input.kappa_mode()

        kappa_processing.set(True)

        try:
            if mode == "cohen":
                v1, v2 = input.sel_kappa_v1(), input.sel_kappa_v2()
                weights = input.sel_kappa_weights() or None
                req(d is not None, v1, v2)

                res, err, conf, missing_info = AgreementAnalysis.cohens_kappa(
                    d, v1, v2, weights=weights, ci=0.95
                )

                if err:
                    kappa_html.set(
                        f"<div class='alert alert-danger'>{_html.escape(str(err))}</div>"
                    )
                else:
                    rep = [
                        {
                            "type": "text",
                            "data": f"Cohen's Kappa Analysis: {v1} vs {v2}",
                        },
                        {"type": "table", "header": "Kappa Statistics", "data": res},
                        {
                            "type": "table",
                            "header": "Confusion Matrix",
                            "data": conf,
                            "safe_html": True,
                        },
                    ]
                    if missing_info:
                        rep.append(
                            {
                                "type": "html",
                                "data": create_missing_data_report_html(
                                    missing_info, var_meta.get() or {}
                                ),
                            }
                        )
                    kappa_html.set(
                        diag_test.generate_report(f"Kappa: {v1} vs {v2}", rep)
                    )

            elif mode == "fleiss":
                cols = input.sel_fleiss_vars()
                req(d is not None, cols, len(cols) >= 3)

                res, err, missing_info = AgreementAnalysis.fleiss_kappa(d, list(cols))

                if err:
                    kappa_html.set(
                        f"<div class='alert alert-danger'>{_html.escape(str(err))}</div>"
                    )
                else:
                    rep = [
                        {
                            "type": "text",
                            "data": "Fleiss' Kappa Analysis (Multi-Rater)",
                        },
                        {"type": "table", "header": "Kappa Statistics", "data": res},
                    ]
                    if missing_info:
                        rep.append(
                            {
                                "type": "html",
                                "data": create_missing_data_report_html(
                                    missing_info, var_meta.get() or {}
                                ),
                            }
                        )
                    kappa_html.set(diag_test.generate_report("Fleiss Kappa", rep))

        except Exception as e:
            logger.exception("Kappa failed")
            kappa_html.set(
                f"<div class='alert alert-danger'>Error: {_html.escape(str(e))}</div>"
            )
        finally:
            kappa_processing.set(False)

    @render.ui
    def out_kappa_results():
        """
        Render the UI block for Kappa agreement results based on current processing state.

        Returns:
            A Shiny UI element showing one of:
            - a loading indicator while the Kappa analysis is running,
            - the generated HTML report for Kappa when available,
            - or a placeholder prompting the user to select variables and run the analysis.
        """
        if kappa_processing.get():
            return create_loading_state("Calculating Agreement Statistics...")
        if kappa_html.get():
            return create_results_container(
                "Kappa Analysis Results",
                ui.HTML(kappa_html.get()),
                class_="fade-in-entry",
            )
        return create_placeholder_state(
            "Select variables and click 'Calculate Kappa'.", icon="🤝"
        )

    @render.download(filename=lambda: "kappa_agreement_report.html")
    def btn_dl_kappa_report():
        """
        Provide the generated Kappa agreement HTML report for download.

        Returns:
            str: The HTML content of the latest Kappa agreement report.
        """
        yield safe_download_html(kappa_html.get(), label="Kappa Report")

    # --- Download Status Badges ---
    @render.ui
    def dl_status_kappa():
        return create_download_status_badge(bool(kappa_html.get()))

    @render.ui
    def dl_status_ba():
        return create_download_status_badge(bool(ba_html.get()))

    @render.ui
    def dl_status_icc():
        return create_download_status_badge(bool(icc_html.get()))

    # ==================== BLAND-ALTMAN LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_analyze_ba)
    def _run_ba():
        """
        Run a Bland–Altman analysis for the currently selected pair of variables and store the generated HTML report.

        Retrieves the selected dataset and variables, executes the Bland–Altman analysis with the configured confidence interval and CI-band option, and builds an HTML report containing the plot and agreement statistics (including a missing-data section when applicable). Sets the `ba_processing` flag while running, updates `ba_html` with either the rendered report or an error alert on failure, and clears the processing flag when finished.
        """
        d = current_df()
        v1, v2 = input.sel_ba_v1(), input.sel_ba_v2()
        show_ci = input.ba_show_ci()
        ci_level = input.ba_conf_level()
        req(d is not None, v1, v2)

        ba_processing.set(True)

        try:
            stats_res, fig, missing_info = AgreementAnalysis.bland_altman_advanced(
                d, v1, v2, ci=ci_level, show_ci_bands=show_ci
            )

            if "error" in stats_res:
                ba_html.set(
                    f"<div class='alert alert-danger'>{_html.escape(str(stats_res['error']))}</div>"
                )
            else:
                stats_df = pd.DataFrame(
                    [
                        {
                            "Mean Diff (Bias)": f"{stats_res.get('mean_diff', 0):.4f}",
                            f"{int(ci_level * 100)}% CI Bias": f"{stats_res.get('ci_mean_diff', [0, 0])[0]:.4f} to {stats_res.get('ci_mean_diff', [0, 0])[1]:.4f}",
                            "Upper LoA": f"{stats_res.get('upper_loa', 0):.4f}",
                            "Lower LoA": f"{stats_res.get('lower_loa', 0):.4f}",
                            "N (Valid Pairs)": stats_res.get("n", 0),
                        }
                    ]
                ).T

                rep = [
                    {"type": "text", "data": f"Bland-Altman Comparison: {v1} vs {v2}"},
                    {"type": "plot", "data": fig},
                    {
                        "type": "table",
                        "header": "Agreement Statistics",
                        "data": stats_df,
                    },
                ]
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                ba_html.set(
                    diag_test.generate_report(f"Bland-Altman: {v1} vs {v2}", rep)
                )
        except Exception as e:
            logger.exception("Bland-Altman failed")
            ba_html.set(
                f"<div class='alert alert-danger'>Error: {_html.escape(str(e))}</div>"
            )
        finally:
            ba_processing.set(False)

    @render.ui
    def out_ba_results():
        if ba_processing.get():
            return create_loading_state("Generating Bland-Altman Analysis...")
        if ba_html.get():
            return create_results_container(
                "Bland-Altman Results",
                ui.HTML(ba_html.get()),
                class_="fade-in-entry",
            )
        return create_placeholder_state(
            "Select variables and click 'Analyze Agreement'.", icon="📉"
        )

    @render.download(
        filename=lambda: f"ba_{_safe_filename_part(input.sel_ba_v1())}_{_safe_filename_part(input.sel_ba_v2())}_report.html"
    )
    def btn_dl_ba_report():
        yield safe_download_html(ba_html.get(), label="Bland-Altman Report")

    # ==================== ICC LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_run_icc)
    def _run_icc():
        """
        Run ICC reliability analysis for the currently selected variables and store an HTML report in the module state.

        Reads the current dataset and user-selected ICC variables (requires at least two), sets the processing flag while computation runs, and invokes AgreementAnalysis.icc to compute ICC results. On success builds an interpretation block, assembles a report (including a missing-data section when present), and writes the generated HTML to the module's icc_html state. On failure writes an error alert HTML to icc_html. Does not return a value.
        """
        d = current_df()
        cols = input.icc_vars()
        req(d is not None, cols, len(cols) >= 2)

        icc_processing.set(True)

        try:
            # Using AgreementAnalysis.icc
            res_df, err, _, missing_info = AgreementAnalysis.icc(d, list(cols))

            if err:
                icc_html.set(
                    f"<div class='alert alert-danger'>{_html.escape(str(err))}</div>"
                )
            else:
                # Custom Interpretation Block
                interp_parts = []
                # res_df from pingouin has ICC, F, df1, df2, pval, CI95%
                # And we added 'Strength' column

                for _, row in res_df.iterrows():
                    val = row.get("ICC", 0)
                    if pd.isna(val) or val is None:
                        continue
                    strength = row.get("Strength", "Poor")
                    icon = (
                        "✅"
                        if strength in ["Excellent", "Good"]
                        else "⚠️"
                        if strength == "Fair"
                        else "❌"
                    )
                    interp_parts.append(
                        f"<li>{icon} <strong>{_html.escape(str(row.get('Description', '')))}</strong> ({_html.escape(str(row.get('Type', '')))}): {val:.3f} ({strength})</li>"
                    )

                interp_html = f"""
                <div class='alert alert-info mt-3'>
                    <h5>📊 ICC Interpretation (Cicchetti, 1994)</h5>
                    <ul>{"".join(interp_parts)}</ul>
                </div>
                """

                rep = [
                    {
                        "type": "text",
                        "data": f"ICC Reliability Analysis: {', '.join(cols)}",
                    },
                    {"type": "html", "data": interp_html},
                    {"type": "table", "header": "ICC Results", "data": res_df},
                ]
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                icc_html.set(diag_test.generate_report("ICC Reliability Report", rep))
        except Exception as e:
            logger.exception("ICC failed")
            icc_html.set(
                f"<div class='alert alert-danger'>Error: {_html.escape(str(e))}</div>"
            )
        finally:
            icc_processing.set(False)

    @render.ui
    def out_icc_results():
        if icc_processing.get():
            return create_loading_state("Calculating ICC Reliability...")
        if icc_html.get():
            return create_results_container(
                "ICC Results", ui.HTML(icc_html.get()), class_="fade-in-entry"
            )
        return create_placeholder_state(
            "Select 2+ variables and click 'Calculate ICC'.", icon="🔍"
        )

    @render.download(
        filename=lambda: f"icc_report_{_safe_filename_part('_'.join((input.icc_vars() or [])[:3]) or 'analysis')}.html"
    )
    def btn_dl_icc_report():
        yield safe_download_html(icc_html.get(), label="ICC Report")

    # --- PDF Download Handlers ---
    @render.download(filename=lambda: "kappa_agreement_report.pdf")
    def btn_dl_kappa_pdf():
        yield safe_download_pdf(kappa_html.get(), label="Kappa Report")

    @render.download(
        filename=lambda: f"ba_{_safe_filename_part(input.sel_ba_v1())}_{_safe_filename_part(input.sel_ba_v2())}_report.pdf"
    )
    def btn_dl_ba_pdf():
        yield safe_download_pdf(ba_html.get(), label="Bland-Altman Report")

    @render.download(
        filename=lambda: f"icc_report_{_safe_filename_part('_'.join((input.icc_vars() or [])[:3]) or 'analysis')}.pdf"
    )
    def btn_dl_icc_pdf():
        yield safe_download_pdf(icc_html.get(), label="ICC Report")

    # --- Validation ---
    @render.ui
    def out_kappa_validation():
        """
        Validate Kappa input selections and return a contextual error alert when the current inputs are invalid.

        Checks the selected kappa mode: for Cohen's Kappa ensures two different variables are selected; for Fleiss' Kappa ensures at least three raters are selected. When a validation rule fails, returns an error alert element describing the issue.

        Returns:
            An alert UI element with an explanatory error message if inputs are invalid, or `None` if inputs pass validation.
        """
        mode = input.kappa_mode()
        if mode == "cohen":
            v1, v2 = input.sel_kappa_v1(), input.sel_kappa_v2()
            if v1 and v2 and v1 == v2:
                return create_error_alert(
                    "Please select two different variables for comparison."
                )
        elif mode == "fleiss":
            cols = input.sel_fleiss_vars()
            if cols and len(cols) < 3:
                return create_error_alert(
                    "Please select at least 3 raters for Fleiss' Kappa."
                )
        return None

    @render.ui
    def out_ba_validation():
        v1, v2 = input.sel_ba_v1(), input.sel_ba_v2()
        if v1 and v2 and v1 == v2:
            return create_error_alert(
                "Please select two different variables for comparison."
            )
        return None

    @render.ui
    def out_icc_validation():
        cols = input.icc_vars()
        if cols and len(cols) < 2:
            return create_error_alert("Please select at least 2 rater/method columns.")
        return None
