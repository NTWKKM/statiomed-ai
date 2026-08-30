from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from shiny import module, reactive, render, ui

from logger import get_logger
from tabs import tab_sample_size  # Import Sample Size Tab
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from utils import psm_lib, table_one
from utils.download_helpers import safe_data_download, safe_download_html
from utils.formatting import create_missing_data_report_html
from utils.pdf_helpers import safe_download_pdf, safe_pdf_report_generation
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.ui_helpers import (
    create_empty_state_ui,
    create_input_group,
    create_results_container,
    create_tooltip_label,
)

logger = get_logger(__name__)
COLORS = get_color_palette()


@module.ui
def baseline_matching_ui() -> ui.TagChild:
    """
    Builds the multi-tab Baseline Matching user interface with controls and results panels.

    The returned UI contains five main subtabs: Baseline Characteristics (Table 1) for generating and downloading Table 1 HTML; Propensity Score Matching (PSM) for configuring, running, and inspecting matching results; Matched Data View for previewing, exporting, and analyzing matched data; Sample Size tools (embedded from sample_size module); and Reference & Interpretation guidance. Each subtab groups configuration controls, action buttons, and result/content containers appropriate to its purpose.

    Returns:
        ui.TagChild: The root navset_tab element representing the complete Baseline Matching UI.
    """
    return ui.navset_tab(
        # ===== SUBTAB 1: BASELINE CHARACTERISTICS (TABLE 1) =====
        ui.nav_panel(
            "📊 Baseline Characteristics (Table 1)",
            # Control section (top)
            ui.card(
                ui.card_header("📊 Table 1 Options"),
                ui.output_ui("ui_matched_status_banner_t1"),
                ui.output_ui("ui_dataset_selector_t1"),
                ui.output_ui("ui_data_info_t1"),
                ui.hr(),
                ui.layout_columns(
                    create_input_group(
                        "Configuration",
                        ui.input_select(
                            "sel_group_col",
                            create_tooltip_label(
                                "Group By (Column)",
                                "Select the variable to split the table columns (e.g. Treatment).",
                            ),
                            choices=[],
                        ),
                        ui.input_radio_buttons(
                            "radio_or_style",
                            "OR Style:",
                            {
                                "all_levels": "All Levels (Every Level vs Ref)",
                                "simple": "Simple (Single Line/Risk vs Ref)",
                            },
                            selected="all_levels",
                        ),
                        type="required",
                    ),
                    create_input_group(
                        "Variables",
                        ui.input_selectize(
                            "sel_t1_vars",
                            create_tooltip_label(
                                "Include Variables",
                                "Select variables to include in the table.",
                            ),
                            choices=[],
                            multiple=True,
                            width="100%",
                            options={"plugins": ["remove_button"]},
                        ),
                        type="required",
                    ),
                    col_widths=[6, 6],
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_gen_table1",
                        "📊 Generate Table 1",
                        class_="btn-primary w-100",
                    ),
                    ui.download_button(
                        "btn_dl_table1",
                        "📥 HTML",
                        class_="btn-success w-100",
                    ),
                    ui.download_button(
                        "btn_dl_table1_pdf",
                        "📥 PDF",
                        class_="btn-outline-danger w-100",
                    ),
                    col_widths=[4, 4, 4],
                ),
            ),
            # Content section (bottom)
            create_results_container(
                "Table 1 Results", ui.output_ui("out_table1_html")
            ),
        ),
        # ===== SUBTAB 2: PROPENSITY SCORE MATCHING =====
        ui.nav_panel(
            "⚖️ Propensity Score Matching",
            # Control section (top)
            ui.card(
                ui.card_header("⚖️ PSM Configuration"),
                ui.div(
                    "💡 ",
                    ui.strong("Need ATE directly?"),
                    " Use ",
                    ui.strong("Clinical Tools → Causal Methods"),
                    " for Inverse Probability Weighting (keeps all data, no matching).",
                    class_="info-callout",
                ),
                ui.layout_columns(
                    create_input_group(
                        "1. Select Variables",
                        ui.input_select(
                            "sel_treat_col",
                            create_tooltip_label(
                                "Treatment Variable (Binary)", "Must be 0/1 or Yes/No."
                            ),
                            choices=[],
                        ),
                        ui.input_select(
                            "sel_outcome_col",
                            create_tooltip_label(
                                "Outcome Variable", "Excluded from matching."
                            ),
                            choices=[],
                        ),
                        ui.input_selectize(
                            "sel_covariates",
                            create_tooltip_label(
                                "Confounding Variables", "Variables to match on."
                            ),
                            choices=[],
                            multiple=True,
                            width="100%",
                            options={"plugins": ["remove_button"]},
                        ),
                        type="required",
                    ),
                    ui.div(
                        create_input_group(
                            "2. Quick Presets",
                            ui.input_radio_buttons(
                                "radio_preset",
                                label=None,
                                choices={
                                    "custom": "🔧 Custom (Manual)",
                                    "demographics": "👥 Demographics",
                                    "full_medical": "🏥 Full Medical",
                                },
                                selected="custom",
                            ),
                            type="optional",
                        ),
                        create_input_group(
                            "3. Matching Settings",
                            ui.input_select(
                                "sel_caliper_preset",
                                create_tooltip_label(
                                    "Caliper Width",
                                    "Stricter (0.1) = better balance, fewer matches.",
                                ),
                                choices={
                                    "1.0": "🔓 Very Loose (1.0×SD)",
                                    "0.5": "📊 Loose (0.5×SD)",
                                    "0.25": "⚖️ Standard (0.25×SD) (Rec)",
                                    "0.1": "🔒 Strict (0.1×SD)",
                                },
                                selected="0.25",
                            ),
                            type="advanced",
                        ),
                    ),
                    col_widths=[6, 6],
                ),
                ui.hr(),
                ui.output_ui("ui_psm_config_summary"),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_psm",
                        "🚀 Run Propensity Score Matching",
                        class_="btn-primary w-100",
                    ),
                    ui.output_ui("ui_psm_run_status"),
                    col_widths=[9, 3],
                ),
            ),
            # Content section (bottom) - with nested tabs for results
            create_results_container(
                "Matching Results", ui.output_ui("ui_psm_main_content")
            ),
        ),
        # ===== SUBTAB 3: MATCHED DATA VIEW =====
        ui.nav_panel(
            "✅ Matched Data View",
            # ... (Existing content) ...
            # Control section (top)
            ui.card(
                ui.card_header("✅ Matched Data Actions"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Export Options:"),
                        ui.download_button(
                            "btn_dl_matched_csv_view",
                            "📥 CSV Format",
                            class_="w-100",
                        ),
                        ui.br(),
                        ui.download_button(
                            "btn_dl_matched_xlsx_view",
                            "📥 Excel Format",
                            class_="w-100",
                        ),
                    ),
                    ui.card(
                        ui.card_header("Filter & Display:"),
                        ui.input_slider(
                            "slider_matched_rows",
                            "Rows to display:",
                            min=1,
                            max=100,
                            value=50,
                            step=10,
                        ),
                    ),
                    ui.card(
                        ui.card_header("Compare Variable:"),
                        ui.input_select("sel_stat_var_tab3", label=None, choices=[]),
                    ),
                    ui.card(
                        ui.card_header("Reset:"),
                        ui.input_action_button(
                            "btn_clear_matched_tab3",
                            "🔄 Clear Matched Data",
                            class_="btn-warning w-100",
                        ),
                    ),
                    col_widths=[3, 3, 3, 3],
                ),
            ),
            # Content section (bottom)
            ui.output_ui("ui_matched_status_tab3"),
            ui.card(
                ui.card_header("📊 Summary Statistics"),
                ui.output_ui("ui_matched_summary_stats"),
            ),
            ui.card(
                ui.card_header("🔍 Data Preview"),
                ui.output_data_frame("out_matched_df_preview"),
            ),
            ui.card(
                ui.card_header("📈 Statistics by Group"),
                ui.navset_card_underline(
                    ui.nav_panel(
                        "📊 Descriptive Stats",
                        ui.output_data_frame("out_matched_stats"),
                    ),
                    ui.nav_panel(
                        "📉 Visualization", ui.output_ui("out_matched_boxplot")
                    ),
                ),
            ),
        ),
        # ===== SUBTAB 5: SAMPLE SIZE =====
        ui.nav_panel(
            "🔢 Sample Size",
            tab_sample_size.sample_size_ui("sample_size"),
        ),
        # ===== SUBTAB 4: REFERENCE & INTERPRETATION =====
        ui.nav_panel(
            "ℹ️ Reference & Interpretation",
            ui.markdown("""
## 📚 Reference & Interpretation Guide

💡 **Tip:** This section provides detailed explanations and interpretation rules for Table 1 and Propensity Score Matching.

### 🚦 Quick Decision Guide

| **Question** | **Recommended Action** | **Goal** |
| :--- | :--- | :--- |
| Do my groups differ at baseline? | **Generate Table 1** (Subtab 1) | Check for significant p-values (< 0.05). |
| My groups are imbalanced. Can I fix? | **Run PSM** (Subtab 2) | Create a "synthetic" RCT where groups are balanced. |
| Did the matching work? | **Check SMD** (Subtab 2 - Results) | Look for **SMD < 0.1** in the Love Plot. |
| What do I do with matched data? | **Export / Use Matched Data** | Go to **Subtab 3** to export, or select "✅ Matched Data" in other analysis tabs. |

---
            """),
            ui.layout_columns(
                ui.card(
                    ui.card_header("📊 Baseline Characteristics (Table 1)"),
                    ui.markdown("""
**Concept:** A standard table in medical research that compares the demographic and clinical characteristics of two or more groups (e.g., Treatment vs Placebo).

**Interpretation:**

* **P-value:** Tests if there is a statistically significant difference between groups.
* **p < 0.05:** Significant difference (Imbalance) ⚠️. This suggests confounding may be present.
* **p ≥ 0.05:** No significant difference (Balanced) ✅.

**Reporting Standards:**

* **Numeric Data (Normal):** Report **Mean ± SD**. (e.g., Age: 45.2 ± 10.1)
* **Numeric Data (Skewed):** Report **Median (IQR)**. (e.g., LOS: 5 (3-10))
* **Categorical Data:** Report **Count (%)**. (e.g., Male: 50 (45%))
                    """),
                ),
                ui.card(
                    ui.card_header("⚖️ Propensity Score Matching (PSM)"),
                    ui.markdown("""
**Concept:** A statistical technique used in observational studies to reduce selection bias. It pairs patients in the treated group with patients in the control group who have similar "propensity scores" (probability of receiving treatment).

**Key Metric: Standardized Mean Difference (SMD):**

* The gold standard for checking balance after matching.
* **SMD < 0.1:** Excellent Balance ✅ (Groups are comparable).
* **SMD 0.1 - 0.2:** Acceptable.
* **SMD > 0.2:** Imbalanced ❌.

**Caliper (Tolerance):**

* Determines how "close" a match must be.
* **Stricter (0.1×SD):** Better balance, but you might lose more patients (fewer matches).
* **Looser (0.5×SD):** More matches, but balance might be worse.
                    """),
                ),
                col_widths=[6, 6],
            ),
            ui.hr(),
            ui.markdown("""
### 📝 Common Workflow

1. **Check Original Data:** Run Table 1 on the "Original Data". Note any variables with p < 0.05.
2. **Match:** Go to Subtab 2, select Treatment, Outcome, and **all confounding variables** (especially those with p < 0.05).
3. **Verify:** After matching, check the **Love Plot**. Ensure all dots (Matched) are within the < 0.1 zone.
4. **Re-check Table 1:** Go back to Subtab 1, switch the dataset selector to **"✅ Matched Data"**, and generate Table 1 again. P-values should now be non-significant (or SMDs low).
            """),
        ),
        id="baseline_matching_tabs",
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def baseline_matching_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
    matched_treatment_col: reactive.Value[str | None],
    matched_covariates: reactive.Value[list[str]],
) -> None:
    """
    Set up server-side reactive logic and UI renderers for the Baseline Matching module.

    This function registers reactive computations, effects, UI render callbacks, and download handlers that implement:
    - Table 1 generation and download,
    - Propensity score matching (PSM) configuration, execution, result summaries, plots, and exports,
    - Matched-data viewing, summaries, visualizations, and exports,
    while updating provided reactive values to reflect matching results.

    Parameters:
        input: Shiny input object for reading UI control values (not documented here).
        output: Shiny output object for registering UI/data outputs (not documented here).
        session: Shiny session object (not documented here).
        df (reactive.Value[pd.DataFrame | None]): Reactive source of the original dataset used for Table 1 and PSM.
        var_meta (reactive.Value[dict[str, Any]]): Reactive metadata for variables (labels, types, display hints) used when generating Table 1.
        df_matched (reactive.Value[pd.DataFrame | None]): Reactive holder for the matched dataset; set by the PSM routine and consumed by matched-data views and exports.
        is_matched (reactive.Value[bool]): Reactive flag indicating whether matched data is available; updated once matching completes successfully.
        matched_treatment_col (reactive.Value[str | None]): Reactive storage for the treatment column name used in the matched dataset (may be an encoded column name).
        matched_covariates (reactive.Value[list[str]]): Reactive list of covariate column names used for matching; updated after running PSM.
    """
    # -------------------------------------------------------------------------
    # SHARED REACTIVE VALUES
    # -------------------------------------------------------------------------
    psm_results: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    html_content: reactive.Value[str | None] = reactive.Value(None)

    # Call Sample Size Server
    tab_sample_size.sample_size_server("sample_size")

    # -------------------------------------------------------------------------
    # HELPER: Get Current Data for Table 1
    # -------------------------------------------------------------------------
    @reactive.Calc
    def current_t1_data() -> tuple[pd.DataFrame | None, str]:
        if (
            is_matched.get()
            and input.radio_dataset_source() == "matched"
            and df_matched.get() is not None
        ):
            return df_matched.get(), "✅ Matched Data"
        return df.get(), "📊 Original Data"

    # -------------------------------------------------------------------------
    # UI UPDATERS (Dropdowns, Selectors)
    # -------------------------------------------------------------------------
    @reactive.Effect
    def _update_common_dropdowns():
        d = df.get()
        if d is None:
            return
        cols = d.columns.tolist()

        # Table 1
        def_t1_group = select_variable_by_keyword(
            cols, ["group", "treatment", "exposure"], default_to_first=True
        )
        ui.update_select(
            "sel_group_col", choices=["None"] + cols, selected=def_t1_group
        )
        ui.update_selectize("sel_t1_vars", choices=cols, selected=cols)

        # PSM
        binary_cols = [c for c in cols if d[c].nunique() == 2]
        def_psm_treat = select_variable_by_keyword(
            binary_cols, ["treatment", "group", "exposure"], default_to_first=True
        )
        def_psm_out = select_variable_by_keyword(
            cols, ["outcome", "cured", "death", "event"], default_to_first=False
        )
        # default_to_first=False for outcome because PSM optional selection often defaults to None/Skip

        ui.update_select("sel_treat_col", choices=cols, selected=def_psm_treat)
        ui.update_select(
            "sel_outcome_col",
            choices=["⊘ None / Skip", *cols],
            selected=def_psm_out if def_psm_out else "⊘ None / Skip",
        )

        # FIX: Update covariates dropdown with all columns
        ui.update_selectize("sel_covariates", choices=cols, selected=[])

        # Matched View
        numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        ui.update_select("sel_stat_var_tab3", choices=numeric_cols)

    # =========================================================================
    # TAB 1: TABLE 1 LOGIC
    # =========================================================================

    @render.ui
    def ui_dataset_selector_t1():
        if is_matched.get():
            return ui.input_radio_buttons(
                "radio_dataset_source",
                "📄 Select Dataset:",
                choices={
                    "original": "📊 Original Data",
                    "matched": "✅ Matched Data (from PSM)",
                },
                selected="original",
                inline=True,
            )
        return None

    @render.ui
    def ui_data_info_t1():
        data, label = current_t1_data()
        if data is None:
            return None
        return ui.p(
            f"**Using:** {label}",
            ui.br(),
            f"**Rows:** {len(data)} | **Columns:** {len(data.columns)}",
            class_="text-muted-sm",
        )

    @render.ui
    def ui_matched_status_banner_t1():
        if is_matched.get():
            return ui.div(
                ui.p(
                    ui.strong("✅ Matched Dataset Available"),
                    " - You can select it above for analysis",
                    style=f"color: {COLORS['success']}; margin-bottom: 5px;",
                ),
                style=(
                    "padding: 10px; border-radius: 6px; margin-bottom: 15px; "
                    "background-color: rgba(34,167,101,0.08); "
                    f"border: 1px solid {COLORS['success']};"
                ),
            )
        return None

    @reactive.Effect
    @reactive.event(input.btn_gen_table1)
    def _generate_table1():
        """
        Generate the Table 1 HTML from the currently selected dataset, variables, and options, and store the result in the shared html_content.

        Validates that at least one variable is selected and shows a warning if not. On success, updates html_content with the generated HTML and removes the running notification. On failure, removes the running notification, shows an error notification containing the exception message, and logs the exception.
        """
        data, label = current_t1_data()
        if data is None:
            return

        group_col = input.sel_group_col()
        if group_col == "None":
            group_col = None

        or_style = input.radio_or_style()

        selected_vars = input.sel_t1_vars()
        if not selected_vars:
            ui.notification_show("Please select at least one variable", type="warning")
            return

        ui.notification_show("Generating Table 1...", duration=None, id="gen_t1_notif")
        try:
            html = table_one.generate_table(
                data,
                selected_vars,
                group_col,
                var_meta.get(),
                or_style=or_style,
            )
            html_content.set(html)
            ui.notification_remove("gen_t1_notif")
        except Exception as e:
            ui.notification_remove("gen_t1_notif")
            ui.notification_show(f"Error: {e}", type="error")
            logger.exception("Table 1 Generation Error")

    @render.ui
    def out_table1_html():
        if html_content.get():
            return ui.div(ui.HTML(html_content.get()), class_="fade-in-entry")
        return create_empty_state_ui(
            message="No Table 1 Generated",
            sub_message="Select variables and click '📊 Generate Table 1' to view baseline characteristics.",
            icon="📋",
        )

    @render.download(filename="table1.html")
    def btn_dl_table1():
        yield safe_download_html(html_content.get(), label="Table 1")

    # =========================================================================
    # TAB 2: PSM LOGIC
    # =========================================================================

    @reactive.Effect
    def _apply_psm_presets():
        d = df.get()
        if d is None:
            return

        preset = input.radio_preset()
        treat = input.sel_treat_col()
        outcome = input.sel_outcome_col()

        # Build list of excluded columns
        excluded = []
        if treat:
            excluded.append(treat)
        if outcome and outcome != "⊘ None / Skip":
            excluded.append(outcome)

        # Get candidate columns (exclude treatment, outcome, and ID-like columns)
        candidates = [
            c
            for c in d.columns
            if c not in excluded and c.lower() not in ["id", "index"]
        ]
        selected = []

        if preset == "demographics":
            # Match columns containing age, sex, bmi
            selected = [
                c
                for c in candidates
                if any(x in c.lower() for x in ["age", "sex", "male", "bmi"])
            ]
        elif preset == "full_medical":
            # Match columns for demographics + comorbidities + lab values
            selected = [
                c
                for c in candidates
                if any(
                    x in c.lower()
                    for x in [
                        "age",
                        "sex",
                        "male",
                        "bmi",
                        "comorb",
                        "hyper",
                        "diab",
                        "lab",
                        "glucose",
                        "hba1c",
                    ]
                )
            ]

        # Only update if preset is not custom
        if preset != "custom":
            ui.update_selectize("sel_covariates", selected=selected)
            logger.info(
                f"Preset '{preset}' applied: selected {len(selected)} variables"
            )

    @render.ui
    def ui_psm_config_summary():
        covs = input.sel_covariates() or []
        treat = input.sel_treat_col()
        outcome = input.sel_outcome_col()

        config_valid = len(covs) > 0

        summary_items = [
            f"💊 **Treatment:** `{treat if treat else '(not selected)'}`",
            f"🎯 **Outcome:** `{outcome if outcome != '⊘ None / Skip' else 'Skip'}`",
            f"📊 **Confounders:** {len(covs)} selected",
        ]

        if not config_valid:
            summary_items.append("❌ **Error:** Please select at least one covariate")

        summary_text = "**✅ Configuration Summary:**\n\n" + "\n".join(
            [f"- {item}" for item in summary_items]
        )

        # เปลี่ยนจาก ui.info_message เป็น UI component ที่ถูกต้อง
        return ui.div(
            ui.markdown(summary_text),
            class_="info-callout",
        )

    @render.ui
    def ui_psm_run_status():
        covs = input.sel_covariates() or []
        if not covs:
            return ui.span(
                "⚠️ Select covariates",
                # class_="text-danger fw-bold" # 🟢 Changed to text-danger (Bootstrap standard)
                class_="text-danger fw-bold",
            )
        return ui.span("✅ Ready to run", class_="text-success fw-bold")

    @reactive.Effect
    @reactive.event(input.btn_run_psm)
    def _run_psm():
        d = df.get()
        treat_col = input.sel_treat_col()
        cov_cols = [
            c
            for c in (input.sel_covariates() or [])
            if c not in {treat_col, input.sel_outcome_col()}
        ]
        caliper = float(input.sel_caliper_preset())

        if d is None or not treat_col or not cov_cols:
            ui.notification_show("Please configure all required fields", type="warning")
            return

        ui.notification_show(
            "Running Propensity Score Matching...", duration=None, id="psm_running"
        )

        try:
            df_analysis = d.copy()

            # Pre-processing
            unique_treat = df_analysis[treat_col].dropna().unique()
            if len(unique_treat) != 2:
                raise ValueError(
                    f"Treatment variable must have exactly 2 values. Found {len(unique_treat)}."
                )

            # Encode if categorical
            final_treat_col = treat_col
            if not pd.api.types.is_numeric_dtype(df_analysis[treat_col]):
                minor_val = df_analysis[treat_col].value_counts().idxmin()
                final_treat_col = f"{treat_col}_encoded"
                df_analysis[final_treat_col] = np.where(
                    df_analysis[treat_col] == minor_val, 1, 0
                )

            # Handle categorical covariates
            cat_covs = [
                c for c in cov_cols if not pd.api.types.is_numeric_dtype(df_analysis[c])
            ]
            if cat_covs:
                df_analysis = pd.get_dummies(
                    df_analysis, columns=cat_covs, drop_first=True
                )
                new_cols = [
                    c
                    for c in df_analysis.columns
                    if c not in d.columns and c != final_treat_col
                ]
                final_cov_cols = [c for c in cov_cols if c not in cat_covs] + new_cols
            else:
                final_cov_cols = cov_cols

            # Calculation
            ps_scores, missing_info = psm_lib.calculate_propensity_score(
                df_analysis, final_treat_col, final_cov_cols, var_meta=var_meta.get()
            )
            df_ps = df_analysis.copy()
            df_ps["propensity_score"] = ps_scores

            # Perform Matching
            df_m = psm_lib.perform_matching(
                df_ps, final_treat_col, "propensity_score", caliper=caliper
            )

            if df_m is None or df_m.empty:
                raise ValueError("No matches found within the specified caliper.")

            # SMD Calculations
            smd_pre = psm_lib.calculate_smd(df_ps, final_treat_col, final_cov_cols)
            smd_post = psm_lib.calculate_smd(df_m, final_treat_col, final_cov_cols)

            # --- Atomic State Update ---
            # เตรียมข้อมูลผลลัพธ์ให้พร้อมก่อนทำการ set ค่าให้กับ reactive values
            new_results = {
                "df_matched": df_m,
                "df_pre_match": df_ps,  # NEW: Store pre-match data for distribution plot
                "smd_pre": smd_pre,
                "smd_post": smd_post,
                "final_treat_col": final_treat_col,
                "msg": "Matching successful",
                "df_ps_len": len(df_ps),
                "df_matched_len": len(df_m),
                "treat_pre_sum": df_ps[final_treat_col].sum(),
                "treat_post_sum": df_m[final_treat_col].sum(),
                "missing_data_info": missing_info,
            }

            # อัปเดตพร้อมกันหลังจากผ่านการตรวจสอบและคำนวณทั้งหมดแล้ว
            psm_results.set(new_results)
            df_matched.set(df_m)
            matched_treatment_col.set(final_treat_col)
            matched_covariates.set(cov_cols)
            is_matched.set(True)  # Set True เป็นลำดับสุดท้ายเพื่อยืนยันว่าทุกอย่างพร้อม

            ui.notification_remove("psm_running")
            ui.notification_show("✅ Matching Successful!", type="message")
            logger.info(f"💾 Matched data stored. Rows: {len(df_m)}")

        except Exception as e:
            ui.notification_remove("psm_running")
            ui.notification_show(f"❌ Matching Failed: {e}", type="error")
            logger.error(f"PSM Error: {e}")

    # --- PSM Main Content Output ---

    @render.ui
    def ui_psm_main_content():
        """
        Builds the main Propensity Score Matching (PSM) results UI, showing a prompt when results are absent or a tabbed results interface when available.

        Returns:
            A Shiny UI element: if no PSM results are present, a card prompting the user to run PSM; otherwise a tabbed card with two panels — Match Quality (summary metrics, balance alert, Love Plot, SMD table, and group comparison) and Export & Next Steps (CSV and HTML report download actions).
        """
        res = psm_results.get()

        if res is None:
            return create_empty_state_ui(
                message="No Matching Results",
                sub_message="Configure parameters and click '🚀 Run Propensity Score Matching' to see results.",
                icon="⚖️",
            )

        # Display results with nested tabs
        return ui.div(
            ui.navset_card_underline(
                # Tab 1: Match Quality
                ui.nav_panel(
                    "📊 Match Quality",
                    ui.h5("Step 3️⃣: Match Quality Summary"),
                    ui.layout_columns(
                        ui.value_box(
                            "Pairs Matched", ui.output_ui("val_pairs"), theme="primary"
                        ),
                        ui.value_box(
                            "Sample Retained",
                            ui.output_ui("val_retained"),
                            theme="primary",
                        ),
                        ui.value_box(
                            "Good Balance", ui.output_ui("val_balance"), theme="success"
                        ),
                        ui.value_box(
                            "SMD Improvement",
                            ui.output_ui("val_smd_imp"),
                            theme="success",
                        ),
                        col_widths=[3, 3, 3, 3],
                    ),
                    ui.output_ui("ui_balance_alert"),
                    ui.div(
                        ui.tags.blockquote(
                            "🔍 Interpretability Guide: Standardized mean differences (SMD) < 0.1 indicate good balance. "
                            "Ideally, all variables in the Love Plot should be within the vertical dashed lines.",
                            style="border-left: 4px solid #ccc; padding-left: 10px; margin-top: 10px; color: #555; background: #f9f9f9; padding: 10px;",
                        ),
                    ),
                    # Missing Data Report
                    ui.HTML(
                        create_missing_data_report_html(
                            res.get("missing_data_info", {}), var_meta.get() or {}
                        )
                    ),
                    ui.hr(),
                    ui.h5("Step 4️⃣: Balance Assessment"),
                    ui.navset_card_underline(
                        ui.nav_panel(
                            "📉 Love Plot (Balance)",
                            ui.output_ui("out_love_plot"),
                            ui.div(
                                ui.span(
                                    "■ Green Zone (<0.1): Excellent Balance",
                                    style="color: green; margin-right: 15px;",
                                ),
                                ui.span(
                                    "■ Yellow Zone (0.1-0.2): Acceptable",
                                    style="color: #d4a017; margin-right: 15px;",
                                ),
                                ui.span(
                                    "■ Red Zone (>0.2): Imbalanced", style="color: red;"
                                ),
                                style="font-size: 0.85em; margin-top: 10px; text-align: center;",
                            ),
                        ),
                        ui.nav_panel(
                            "🏔️ Common Support (Distribution)",
                            ui.output_ui("out_ps_distribution"),  # NEW OUTPUT
                            ui.p(
                                "💡 Checks if treated and control groups overlap enough to be comparable.",
                                class_="text-muted-sm",
                            ),
                        ),
                        ui.nav_panel(
                            "📋 SMD Table",
                            ui.output_data_frame("out_smd_table"),
                        ),
                        ui.nav_panel(
                            "📊 Group Comparison",
                            ui.output_data_frame("out_group_comparison_table"),
                        ),
                    ),
                ),
                # Tab 2: Export
                ui.nav_panel(
                    "📥 Export & Next Steps",
                    ui.h5("Step 5️⃣: Export & Next Steps"),
                    ui.layout_columns(
                        ui.download_button(
                            "btn_dl_psm_csv", "📥 CSV", class_="w-100 btn-sm"
                        ),
                        ui.download_button(
                            "btn_dl_psm_report", "📥 HTML", class_="w-100 btn-sm"
                        ),
                        ui.download_button(
                            "btn_dl_psm_pdf",
                            "📥 PDF",
                            class_="w-100 btn-sm btn-outline-danger",
                        ),
                        col_widths=[4, 4, 4],
                    ),
                    ui.p(
                        "✅ Full matched data available in **Subtab 3 (Matched Data View)**",
                        style="background-color: #f0fdf4; padding: 10px; border-radius: 5px; border: 1px solid #bbf7d0; margin-top: 10px;",
                    ),
                ),
                id="psm_results_tabs",
            ),
            class_="fade-in-entry",
        )

    # --- PSM Output Components ---

    @render.ui
    def val_pairs():
        res = psm_results.get()
        if not res:
            return "-"
        return f"{res['treat_post_sum']:.0f}"

    @render.ui
    def val_retained():
        res = psm_results.get()
        if not res:
            return "-"
        pct = res["df_matched_len"] / res["df_ps_len"] * 100
        return f"{pct:.1f}%"

    @render.ui
    def val_balance():
        res = psm_results.get()
        if not res:
            return "-"
        # Defensive checks for missing/empty SMD data
        smd_post = res.get("smd_post")
        if smd_post is None or smd_post.empty:
            return "-"
        if "SMD" not in smd_post.columns or smd_post["SMD"].dropna().empty:
            return "-"
        good = (smd_post["SMD"] < 0.1).sum()
        total = len(smd_post)
        return f"{good}/{total}"

    @render.ui
    def val_smd_imp():
        res = psm_results.get()
        if not res:
            return "-"
        # Defensive checks for missing SMD DataFrames
        smd_pre = res.get("smd_pre")
        smd_post = res.get("smd_post")
        if smd_pre is None or smd_post is None:
            return "-"
        if smd_pre.empty or smd_post.empty:
            return "-"
        # Merge and verify result has valid numeric data
        try:
            merged = smd_pre.merge(smd_post, on="Variable", suffixes=("_pre", "_post"))
        except Exception:
            return "-"
        if merged.empty:
            return "-"
        if "SMD_pre" not in merged.columns or "SMD_post" not in merged.columns:
            return "-"
        # Check for non-empty numeric values
        smd_pre_vals = merged["SMD_pre"].dropna()
        smd_post_vals = merged["SMD_post"].dropna()
        if smd_pre_vals.empty or smd_post_vals.empty:
            return "-"
        avg_pre = smd_pre_vals.mean()
        avg_post = smd_post_vals.mean()
        # Protect against avg_pre == 0 or NaN
        if pd.isna(avg_pre) or pd.isna(avg_post) or avg_pre == 0:
            return "-"
        imp = (avg_pre - avg_post) / avg_pre * 100
        return f"{imp:.1f}%"

    @render.ui
    def ui_balance_alert():
        """
        Render a balance status banner based on post-matching standardized mean differences.

        Displays a green success banner when all post-matching SMDs are below 0.1; otherwise displays a warning banner
        indicating how many variables remain imbalanced. Returns None when no PSM results are available.

        Returns:
            ui.Component or None: A Shiny UI element containing the status banner, or `None` if PSM results are not present.
        """
        res = psm_results.get()
        if not res:
            return None

        smd_post = res.get("smd_post")
        if smd_post is None or smd_post.empty:
            return None
        if "SMD" not in smd_post.columns or smd_post["SMD"].dropna().empty:
            return None

        good = (smd_post["SMD"] < 0.1).sum()
        total = len(smd_post)

        if good == total:
            return ui.div(
                ui.strong("✅ Excellent balance achieved!"),
                " All variables have SMD < 0.1",
                style=(
                    "padding: 10px; border-radius: 6px; "
                    "background-color: rgba(34,167,101,0.08); "
                    f"border: 1px solid {COLORS['success']}; "
                    f"color: {COLORS['success']};"
                ),
            )
        else:
            bad_count = total - good
            return ui.div(
                ui.strong("⚠️ Imbalance remains"),
                f" on {bad_count} variable(s). Try increasing caliper width or checking for outliers.",
                style=(
                    "padding: 10px; border-radius: 6px; "
                    "background-color: rgba(255,185,0,0.08); "
                    f"border: 1px solid {COLORS['warning']}; "
                    "color: #000;"
                ),
            )

    @render.ui
    def out_love_plot():
        """
        Render the Love plot showing standardized mean differences (SMD) before and after matching.

        If PSM results are not yet available or the plot could not be created, returns a centered placeholder message indicating waiting/no plot. Otherwise returns a UI HTML block containing the Plotly-rendered Love plot (responsive, Plotly JS loaded from CDN).

        Returns:
            ui.UI: A Shiny UI element — either ui.HTML with the Love plot HTML or a ui.div placeholder message.
        """
        res = psm_results.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        fig = psm_lib.plot_love_plot(res["smd_pre"], res["smd_post"])
        if fig is None:
            return ui.div(
                ui.markdown("⏳ *No plot available...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            fig, div_id="plot_balance_love", include_plotlyjs="cdn", responsive=True
        )
        return ui.HTML(html_str)

    @render.ui
    def out_ps_distribution():
        res = psm_results.get()
        if res is None:
            return None

        # Extract data from results
        df_pre = res.get("df_pre_match")
        df_post = res.get("df_matched")
        treat_col = res.get("final_treat_col")

        if df_pre is None:
            return None

        fig = psm_lib.plot_ps_distribution(
            df_pre, df_post, treat_col, ps_col="propensity_score"
        )

        html_str = plotly_figure_to_html(
            fig, div_id="plot_ps_dist", include_plotlyjs="cdn", responsive=True
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_smd_table():
        """
        Render a table comparing standardized mean differences (SMD) before and after matching.

        The table merges pre- and post-matching SMDs by variable, computes percent improvement as
        ((SMD_before - SMD_after) / SMD_before * 100) with divisions-by-zero treated as 0, and rounds
        SMD values to 4 decimal places and improvement to 1 decimal place.

        Returns:
            render.DataGrid: A DataGrid wrapping a DataFrame with columns:
                - 'Variable': variable name
                - 'SMD_before': SMD before matching (rounded)
                - 'SMD_after': SMD after matching (rounded)
                - 'Improvement %': percent improvement (rounded)
        """
        res = psm_results.get()
        if not res:
            return None

        # 1. รวมข้อมูล
        merged = res["smd_pre"].merge(
            res["smd_post"], on="Variable", suffixes=("_before", "_after")
        )

        # 2. คำนวณส่วนต่าง
        merged["Improvement %"] = (
            (merged["SMD_before"] - merged["SMD_after"])
            / merged["SMD_before"].replace(0, np.nan)
            * 100
        ).fillna(0)

        # 3. ปัดเศษตัวเลขใน DataFrame แทนการใช้ .style
        merged["SMD_before"] = merged["SMD_before"].round(4)
        merged["SMD_after"] = merged["SMD_after"].round(4)
        merged["Improvement %"] = merged["Improvement %"].round(1)

        # ✅ คืนค่าเป็น DataFrame เปล่าๆ เข้า DataGrid
        return render.DataGrid(merged)

    @render.data_frame
    def out_group_comparison_table():
        res = psm_results.get()
        if not res:
            return None

        comp_data = pd.DataFrame(
            {
                "Stage": ["Before", "After"],
                "Treated (1)": [
                    res["treat_pre_sum"],
                    res["treat_post_sum"],
                ],
                "Control (0)": [
                    res["df_ps_len"] - res["treat_pre_sum"],
                    (res["df_matched_len"] - res["treat_post_sum"]),
                ],
            }
        )
        return render.DataGrid(comp_data)

    @render.download(filename="matched_data.csv")
    def btn_dl_psm_csv():
        res = psm_results.get()
        safe_data_download(res, label="PSM Matched Data")
        yield res["df_matched"].to_csv(index=False)

    @render.download(filename="psm_report.html")
    def btn_dl_psm_report():
        res = psm_results.get()
        if res:
            fig = psm_lib.plot_love_plot(res["smd_pre"], res["smd_post"])
            merged = res["smd_pre"].merge(
                res["smd_post"], on="Variable", suffixes=("_before", "_after")
            )
            elements = [
                {"type": "text", "data": "PSM Report"},
                {"type": "table", "data": merged},
                {"type": "plot", "data": fig},
            ]
            html = psm_lib.generate_psm_report(
                "Propensity Score Matching Report", elements
            )
            yield safe_download_html(html, label="PSM Report")

    @render.download(filename="table1.pdf")
    def btn_dl_table1_pdf():
        yield safe_download_pdf(html_content.get(), label="Table 1")

    @render.download(filename="psm_report.pdf")
    def btn_dl_psm_pdf():
        def _build():
            res = psm_results.get()
            if not res:
                return None
            fig = psm_lib.plot_love_plot(res["smd_pre"], res["smd_post"])
            merged = res["smd_pre"].merge(
                res["smd_post"], on="Variable", suffixes=("_before", "_after")
            )
            elements = [
                {"type": "text", "data": "PSM Report"},
                {"type": "table", "data": merged},
                {"type": "plot", "data": fig},
            ]
            return psm_lib.generate_psm_report(
                "Propensity Score Matching Report", elements
            )

        yield safe_pdf_report_generation(_build, label="PSM Report")

    # =========================================================================
    # TAB 3: MATCHED DATA VIEW
    # =========================================================================

    @render.ui
    def ui_matched_status_tab3():
        if df_matched.get() is not None:
            df_m = df_matched.get()
            treat_col = matched_treatment_col.get()
            return ui.div(
                ui.h5(
                    ui.span(
                        "✅ Matched Dataset Ready", style=f"color: {COLORS['success']};"
                    ),
                    style="margin-bottom: 10px;",
                ),
                ui.p(
                    f"• Total rows: **{len(df_m):,}**",
                    ui.br(),
                    f"• Treatment variable: **{treat_col}**",
                    style="font-size: 0.95em;",
                ),
                style=(
                    f"background-color: rgba(34,167,101,0.08); padding: 15px; border-radius: 5px; "
                    f"border: 1px solid {COLORS['success']}; margin-bottom: 20px;"
                ),
            )
        else:
            return ui.div(
                ui.markdown(
                    "### ℹ️ No matched data available yet.\n\n"
                    "1. Go to **Subtab 2 (Propensity Score Matching)**\n\n"
                    "2. Configure variables and run PSM matching\n\n"
                    "3. Return here to view and export matched data"
                ),
                class_="info-callout",
            )

    @render.ui
    def ui_matched_summary_stats():
        if df_matched.get() is None:
            return None

        df_m = df_matched.get()
        treat_col = matched_treatment_col.get()

        # Show group sizes
        if treat_col and treat_col in df_m.columns:
            grp_counts = df_m[treat_col].value_counts().sort_index()
            return ui.p(
                ui.strong(f"Group Sizes ({treat_col}):"),
                ui.br(),
                ", ".join([f"{idx}: {count}" for idx, count in grp_counts.items()]),
                class_="text-muted-sm",
            )
        return None

    @render.data_frame
    def out_matched_df_preview():
        if df_matched.get() is not None:
            n_rows = input.slider_matched_rows() or 50
            return render.DataGrid(df_matched.get().head(n_rows), filters=True)
        return None

    @render.data_frame
    def out_matched_stats():
        """
        Render a DataGrid of descriptive statistics for the selected variable grouped by the matched treatment column.

        If the matched dataset, selected statistic variable, or treatment column is missing or not present in the dataset, returns None.

        Returns:
            render.DataGrid or None: A DataGrid containing group-wise describe() results (treatment column as the first column) when data and columns are available; otherwise None.
        """
        d = df_matched.get()
        var = input.sel_stat_var_tab3()
        treat = matched_treatment_col.get()

        if d is not None and var and treat and var in d.columns and treat in d.columns:
            return render.DataGrid(d.groupby(treat)[var].describe().reset_index())
        return None

    @render.ui
    def out_matched_boxplot():
        """
        Render a box plot of the selected variable by treatment for the matched dataset, or a waiting placeholder if data is unavailable.

        Returns:
            ui element: An HTML UI element containing a Plotly box plot of the selected variable grouped by the treatment column, or a centered "Waiting for data..." placeholder `div` when matched data, the variable, or the treatment column is missing or invalid.
        """
        d = df_matched.get()
        var = input.sel_stat_var_tab3()
        treat = matched_treatment_col.get()

        if (
            d is None
            or not var
            or not treat
            or var not in d.columns
            or treat not in d.columns
        ):
            return ui.div(
                ui.markdown("⏳ *Waiting for data...*"),
                class_="muted-placeholder",
            )
        fig = px.box(d, x=treat, y=var, title=f"{var} by {treat}")
        html_str = plotly_figure_to_html(
            fig, div_id="plot_balance_boxplot", include_plotlyjs="cdn", responsive=True
        )
        return ui.HTML(html_str)

    @reactive.Effect
    @reactive.event(input.btn_clear_matched_tab3)
    def _clear_matched():
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        psm_results.set(None)
        html_content.set(None)
        ui.notification_show("Matched data cleared", type="warning")
        logger.info("🔄 Matched data cleared")

    # Exports for Tab 3
    @render.download(filename="matched_data.csv")
    def btn_dl_matched_csv_view():
        data = df_matched.get()
        safe_data_download(data, label="Matched Data CSV")
        yield data.to_csv(index=False)

    @render.download(filename="matched_data.xlsx")
    def btn_dl_matched_xlsx_view():
        data = df_matched.get()
        safe_data_download(data, label="Matched Data Excel", type_="dataset")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            data.to_excel(writer, index=False)
        yield buffer.getvalue()
