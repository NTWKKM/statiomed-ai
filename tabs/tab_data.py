from __future__ import annotations

import html as _html
from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, ui
from shiny.types import FileInfo

from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import (
    check_assumptions,
    handle_outliers,
    impute_missing_data,
    load_data_robust,
    transform_variable,
)
from utils.data_quality import DataQualityReport, check_data_quality
from utils.ui_helpers import (
    create_empty_state_ui,
)
from utils.visualizations import plot_missing_pattern

logger = get_logger(__name__)

COLORS = get_color_palette()  # เรียกใช้ Palette กลาง


# --- 1. UI Definition ---
@module.ui
def data_ui() -> ui.TagChild:
    """
    Constructs the Data Management tab UI with controls and panels for loading, inspecting, configuring, cleaning, imputing, transforming, and previewing a dataset.

    The layout includes a left sidebar for data actions and metadata, a main area with data quality warnings and a data health report, an accordion of tools (Variable Config, Cleaning & Imputation, Transformation, Reference), and a data preview card.

    Returns:
        ui.TagChild: A UI tag tree representing the complete Data Management tab layout.
    """
    return ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.h4("⚙️ Data Controls", class_="mb-3 text-primary"),
                ui.input_action_button(
                    "btn_load_example",
                    "📄 Load Example Data",
                    class_="btn-outline-primary w-100 mb-2 shadow-sm",
                ),
                ui.input_file(
                    "file_upload",
                    "📂 Upload CSV/Excel",
                    accept=[".csv", ".xlsx"],
                    multiple=False,
                    width="100%",
                ),
                ui.output_ui("ui_file_metadata"),
                ui.hr(),
                ui.div(
                    ui.output_ui("ui_btn_clear_match"),
                    ui.input_action_button(
                        "btn_reset_all",
                        "⚠️ Reset Workspace",
                        class_="btn-outline-danger w-100 shadow-sm",
                    ),
                    class_="d-grid gap-2",
                ),
                class_="p-2",
            ),
            width=320,
            bg=COLORS["smoke_white"],
            title="Data Management",
        ),
        ui.div(
            # New: Data Quality Warnings
            ui.output_ui("ui_data_quality_warnings"),
            # New: Data Health Report Section (Visible only when issues exist)
            ui.output_ui("ui_data_report_card"),
            # 1. Variable Settings & Advanced Tools
            ui.accordion(
                ui.accordion_panel(
                    "🛠️ Data Management Tools",
                    ui.navset_card_tab(
                        # Tab 1: Configuration (Existing)
                        ui.nav_panel(
                            "🛠️ Variable Config",
                            ui.accordion(
                                ui.accordion_panel(
                                    ui.tags.span(
                                        "📝 Metadata & Type", class_="fw-bold"
                                    ),
                                    ui.layout_columns(
                                        # LEFT COLUMN: Variable Selection
                                        ui.div(
                                            ui.input_select(
                                                "sel_var_edit",
                                                "Select Variable:",
                                                choices=["Select..."],
                                                width="100%",
                                            ),
                                            ui.div(
                                                ui.tags.strong("Categorical Mapping:"),
                                                " Format as `0=Control`.",
                                                class_="alert alert-info p-2 mb-0",
                                            ),
                                            class_="p-2",
                                        ),
                                        # MIDDLE COLUMN: Variable Settings
                                        ui.div(
                                            ui.output_ui("ui_var_settings"),
                                            class_="p-2",
                                        ),
                                        # RIGHT COLUMN: Missing Data Configuration
                                        ui.div(
                                            ui.h6(
                                                "🔍 Missing Data",
                                                style=f"color: {COLORS['primary']};",
                                            ),
                                            ui.input_text(
                                                "txt_missing_codes",
                                                "Missing Values:",
                                                placeholder="-99, 999",
                                                value="",
                                            ),
                                            ui.output_ui("ui_missing_preview"),
                                            ui.input_action_button(
                                                "btn_save_missing",
                                                "💾 Save Config",
                                                class_="btn-secondary w-100 mt-2",
                                            ),
                                            class_="p-2 bg-light rounded",
                                        ),
                                        col_widths=(3, 6, 3),
                                    ),
                                    value="var_config",
                                ),
                                open=True,
                                id="acc_var_config",
                                class_="border-0",
                            ),
                        ),
                        # Tab 2: Cleaning & Imputation
                        ui.nav_panel(
                            "🧹 Cleaning & Imputation",
                            ui.layout_columns(
                                # Card 1: Missing Data Imputation
                                ui.card(
                                    ui.card_header("🧩 Impute Missing Data"),
                                    ui.input_select(
                                        "sel_impute_method",
                                        "Method:",
                                        choices=["mean", "median", "knn", "mice"],
                                    ),
                                    ui.input_select(
                                        "sel_impute_cols",
                                        "Columns:",
                                        choices=[],
                                        multiple=True,
                                    ),
                                    ui.input_action_button(
                                        "btn_run_impute",
                                        "Run Imputation",
                                        class_="btn-warning",
                                    ),
                                ),
                                # Card 2: Outlier Handling
                                ui.card(
                                    ui.card_header("📈 Outlier Handling"),
                                    ui.input_select(
                                        "sel_outlier_cols",
                                        "Columns:",
                                        choices=[],
                                        multiple=True,
                                    ),
                                    ui.layout_columns(
                                        ui.input_select(
                                            "sel_outlier_method",
                                            "Method:",
                                            choices=["iqr", "zscore"],
                                        ),
                                        ui.input_numeric(
                                            "num_outlier_thresh",
                                            "Threshold:",
                                            value=1.5,
                                            step=0.1,
                                        ),
                                    ),
                                    ui.input_select(
                                        "sel_outlier_action",
                                        "Action:",
                                        choices={
                                            "flag": "Flag (set to NaN)",
                                            "remove": "Set to NaN (same as Flag)",
                                            "winsorize": "Winsorize",
                                            "cap": "Cap",
                                        },
                                    ),
                                    ui.input_action_button(
                                        "btn_run_outlier",
                                        "Handle Outliers",
                                        class_="btn-danger",
                                    ),
                                ),
                                col_widths=(6, 6),
                            ),
                            ui.br(),
                            ui.card(
                                ui.card_header("🗺️ Missing Data Pattern"),
                                ui.output_ui("ui_missing_plot"),
                                class_="mt-3 shadow-sm",
                            ),
                        ),
                        # Tab 2.5: Multiple Imputation (NEW)
                        ui.nav_panel(
                            "🔄 Multiple Imputation",
                            ui.div(
                                # Header with info
                                ui.div(
                                    ui.tags.span(
                                        "📊 Multiple Imputation with Rubin's Rules",
                                        class_="fw-bold fs-5",
                                    ),
                                    ui.tags.span(
                                        " — Generate m imputed datasets and pool estimates",
                                        class_="text-muted",
                                    ),
                                    class_="mb-3",
                                ),
                                # Main controls
                                ui.layout_columns(
                                    # Left: Settings
                                    ui.card(
                                        ui.card_header(
                                            ui.tags.span(
                                                "⚙️ MI Settings", class_="fw-semibold"
                                            )
                                        ),
                                        ui.input_select(
                                            "sel_mi_cols",
                                            "Columns to Impute:",
                                            choices=[],
                                            multiple=True,
                                        ),
                                        ui.layout_columns(
                                            ui.input_numeric(
                                                "num_mi_m",
                                                "Imputations (m):",
                                                value=5,
                                                min=2,
                                                max=50,
                                            ),
                                            ui.input_numeric(
                                                "num_mi_iter",
                                                "Iterations:",
                                                value=10,
                                                min=5,
                                                max=100,
                                            ),
                                            col_widths=(6, 6),
                                        ),
                                        ui.hr(),
                                        ui.input_action_button(
                                            "btn_run_mi",
                                            "🚀 Run Multiple Imputation",
                                            class_="btn-primary w-100",
                                        ),
                                        ui.br(),
                                        ui.div(
                                            ui.tags.small(
                                                "💡 Tip: Use m ≥ 5 imputations. "
                                                "More is better for high % missing data.",
                                                class_="text-muted",
                                            ),
                                            class_="mt-2",
                                        ),
                                        class_="shadow-sm",
                                    ),
                                    # Right: Status & Results
                                    ui.card(
                                        ui.card_header(
                                            ui.tags.span(
                                                "📈 MI Status", class_="fw-semibold"
                                            )
                                        ),
                                        ui.output_ui("ui_mi_status"),
                                        class_="shadow-sm",
                                    ),
                                    col_widths=(5, 7),
                                ),
                                # Diagnostics Section
                                ui.br(),
                                ui.card(
                                    ui.card_header("🔍 Imputation Diagnostics"),
                                    ui.output_ui("ui_mi_diagnostics"),
                                    class_="shadow-sm",
                                ),
                                class_="p-2",
                            ),
                        ),
                        # Tab 3: Transformation
                        ui.nav_panel(
                            "⚡ Transformation",
                            ui.layout_columns(
                                ui.div(
                                    ui.input_select(
                                        "sel_trans_var",
                                        "Variable:",
                                        choices=["Select..."],
                                    ),
                                    ui.input_select(
                                        "sel_trans_method",
                                        "Transformation:",
                                        choices=["log", "sqrt", "zscore"],
                                    ),
                                    ui.input_action_button(
                                        "btn_run_trans",
                                        "Apply Transform",
                                        class_="btn-primary w-100 mb-3",
                                    ),
                                    ui.h6("📊 Assumption Check"),
                                    ui.output_ui("ui_assumption_result"),
                                ),
                                ui.div(
                                    ui.h6("Transformation Preview"),
                                    ui.output_text_verbatim(
                                        "txt_trans_preview"
                                    ),  # Basic placeholder
                                    class_="p-3 border rounded bg-light",
                                ),
                                col_widths=(4, 8),
                            ),
                        ),
                        ui.nav_panel(
                            "ℹ️ Reference & Interpretation",
                            ui.div(
                                ui.markdown(
                                    """
                                ### 🛠️ Variable Config
                                - **Metadata**: Define variable types (Categorical vs Continuous).
                                - **Missing Data**: standardized coding (e.g., `-99`, `NaN`) ensures accurate analysis.

                                ### 🧹 Cleaning & Imputation
                                - **Mean/Median**: Simple, fast, but reduces variance. Use for low missingness (<5%).
                                - **KNN (K-Nearest Neighbors)**: Imputes based on similar rows. Preserves local structure better.
                                - **MICE (Multivariate Imputation)**: Models each variable using others. Best for complex datasets with random missingness (MAR).

                                ### 📈 Outlier Handling
                                - **IQR (Interquartile Range)**: Robust method. Flags points < Q1-1.5*IQR or > Q3+1.5*IQR.
                                - **Z-Score**: Parametric. Flags points > 3 SD from mean. Assumes normality.
                                - **Actions**:
                                    - **Winsorize**: Cap values at the thresholds (preserves sample size).
                                    - **Remove**: Delete values (creates missingness).

                                ### ⚡ Transformation
                                - **Log**: Reduces right-skewness (e.g., income, CRP levels). Handles `x > 0`.
                                - **Sqrt**: Moderate skew reduction. Handles `x >= 0`.
                                - **Z-Score**: Standardizes to Mean=0, SD=1. Essential for algorithms sensitive to scale (e.g., KNN, Clustering).
                                """
                                ),
                                class_="p-3 bg-light border rounded",
                                style="max-height: 500px; overflow-y: auto;",
                            ),
                        ),
                        id="tabs_data_tools",
                    ),
                    value="acc_data_tools",
                ),
                open="acc_data_tools",
                id="acc_data_tools_wrapper",
                class_="mb-3 shadow-sm",
            ),
            # 2. Data Preview Card
            ui.card(
                ui.card_header(ui.tags.span("📄 Data Preview", class_="fw-bold")),
                ui.output_ui("ui_preview_area"),
                height="600px",
                full_screen=True,
                class_="shadow-sm border-0",
            ),
            class_="p-3",
        ),
    )


# --- 2. Server Logic ---
@module.server
def data_server(  # noqa: C901, PLR0915, PLR0913
    input: Any,  # noqa: A002
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    uploaded_file_info: reactive.Value[dict[str, Any] | None],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
    matched_treatment_col: reactive.Value[str | None],
    matched_covariates: reactive.Value[list[str]],
    mi_imputed_datasets: reactive.Value[list[pd.DataFrame]],  # NEW: Shared MI state
) -> None:
    """
    Initialize the server-side logic for the Data Management Shiny module, wiring data loading, metadata handling, cleaning/imputation/outlier/transform workflows, data-quality detection, and UI renderers.

    This function sets up reactive state and event/effect handlers that:
    - generate example clinical data and simulate time-varying covariates,
    - load and infer column types from uploaded CSV/Excel files (with basic quality issue detection),
    - configure per-variable metadata and missing-value codes,
    - run imputation, outlier handling, and variable transformations (creating new variables and updating metadata),
    - provide UI outputs for data preview, data quality warnings, a data quality report card, file metadata, missing-data pattern plots, and transformation assumption checks,
    - manage matched-data state and reset/confirmation flows.

    Parameters:
        df (reactive.Value[pd.DataFrame | None]): Reactive reference that holds the current dataset.
        var_meta (reactive.Value[dict[str, Any]]): Reactive reference for per-variable metadata (type, label, mappings, missing codes).
        uploaded_file_info (reactive.Value[dict[str, Any] | None]): Reactive reference storing uploaded file information (e.g., name).
        df_matched (reactive.Value[pd.DataFrame | None]): Reactive reference holding a matched/processed dataset variant when applicable.
        is_matched (reactive.Value[bool]): Reactive flag indicating whether matching has been performed.
        matched_treatment_col (reactive.Value[str | None]): Reactive reference storing the treatment column name used for matching.
        matched_covariates (reactive.Value[list[str]]): Reactive reference listing covariates used for matching.

    Note:
        This function mutates the provided reactive.Value objects to drive the module's UI and behavior and does not return a value.
    """
    is_loading_data: reactive.Value[bool] = reactive.Value(value=False)
    # เก็บข้อมูลปัญหาของข้อมูล (Row, Col, Value) เพื่อรายงาน
    data_issues: reactive.Value[list[dict[str, Any]]] = reactive.Value([])

    @reactive.Calc
    def quality_warnings() -> list[str]:
        data = df.get()
        if data is None:
            return []
        return check_data_quality(data)

    # --- 1. Data Loading Logic & Immediate Quality Check ---
    def _infer_column_type(df_in: pd.DataFrame, col: str) -> tuple[str, list[dict]]:
        series = df_in[col]
        unique_vals = series.dropna().unique()
        n_unique = len(unique_vals)
        inferred_type = "Categorical"
        issues = []

        # Constants for detection logic
        max_categorical_unique = 12
        numeric_threshold = 0.70
        max_issue_per_col = 10

        # Check 1: Is it already numeric?
        if pd.api.types.is_numeric_dtype(series):
            if n_unique > max_categorical_unique:
                inferred_type = "Continuous"
        # Check 2: Is it Object/String but looks like numbers?
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
            series
        ):
            numeric_conversion = pd.to_numeric(series, errors="coerce")
            valid_count = numeric_conversion.notna().sum()
            total_count = series.notna().sum()

            if total_count > 0 and (valid_count / total_count) > numeric_threshold:
                inferred_type = "Continuous"

                # Identify Bad Rows
                bad_mask = numeric_conversion.isna() & series.notna()
                bad_rows = series[bad_mask]

                for idx, val in bad_rows.items():
                    if len(issues) < max_issue_per_col:
                        issues.append(
                            {
                                "col": col,
                                "row": idx + 2,
                                "value": str(val),
                                "issue": "Non-numeric value in continuous column",
                            }
                        )
                    elif len(issues) == max_issue_per_col:
                        issues.append(
                            {
                                "col": col,
                                "row": "...",
                                "value": "...",
                                "issue": "More issues suppressed...",
                            }
                        )
        return inferred_type, issues

    def _process_ingested_dataset(
        new_df: pd.DataFrame,
        file_name: str,
        preset_meta: dict[str, Any] | None = None,
    ) -> None:
        """Shared dataset ingestion, type inference, and immediate quality-check pipeline."""
        processed_df = new_df.copy()
        uploaded_file_info.set({"name": file_name})

        current_meta = preset_meta.copy() if preset_meta is not None else {}
        current_issues = []

        # --- Infer Types and Detect Quality Issues ---
        for col in processed_df.columns:
            if col in current_meta:
                if current_meta[col].get(
                    "type"
                ) == "Continuous" and not pd.api.types.is_numeric_dtype(
                    processed_df[col]
                ):
                    processed_df[col] = pd.to_numeric(
                        processed_df[col], errors="coerce"
                    )
                continue
            inferred_type, issues = _infer_column_type(processed_df, col)
            if inferred_type == "Continuous" and not pd.api.types.is_numeric_dtype(
                processed_df[col]
            ):
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
            current_meta[col] = {"type": inferred_type, "map": {}, "label": col}
            current_issues.extend(issues)

        df.set(processed_df)
        var_meta.set(current_meta)
        data_issues.set(current_issues)

        msg = (
            f"✅ Loaded {len(new_df)} rows."
            if file_name != "Example Clinical Data"
            else f"✅ Loaded {len(new_df)} Clinical Records (Simulated)"
        )
        if current_issues:
            msg += " ⚠️ Found data quality issues (see report)."
            ui.notification_show(msg, type="warning")
        else:
            ui.notification_show(msg, type="message")

    def generate_example_data_logic():
        logger.info("Generating example data...")
        is_loading_data.set(True)
        data_issues.set([])  # Reset issues
        id_notify = ui.notification_show("🔄 Generating simulation...", duration=None)

        try:
            new_df, meta = _simulate_clinical_data()
            ui.notification_remove(id_notify)
            _process_ingested_dataset(
                new_df, file_name="Example Clinical Data", preset_meta=meta
            )
            logger.info("✅ Successfully generated %d records", len(new_df))

        except Exception as e:
            logger.error("Error generating example data: %s", e)
            ui.notification_remove(id_notify)
            ui.notification_show(f"❌ Error: {e!s}", type="error")

        finally:
            is_loading_data.set(False)

    def _simulate_clinical_data() -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Generate a synthetic clinical dataset and accompanying variable metadata.

        Creates a simulated cohort (n=1600) containing patient identifiers, demographics, treatment assignment,
        comorbidities, survival follow-up (time and death status), clinical outcomes, laboratory results,
        multiple diagnostic rater assessments, an expensive comparator test, repeated-measures/time-varying
        covariate (TVC) fields, resource use and costs, and other derived measurements. Random missing values
        are injected into the produced DataFrame.

        Returns:
            tuple[pd.DataFrame, dict[str, Any]]: A tuple where the first element is the simulated dataset
            (one row per patient, with TVC fields aligned by patient) and the second element is a metadata
            mapping describing example variables (types, labels, and value mappings).
        """
        np.random.seed(42)
        n = 1600

        # --- Base Variables ---
        age = np.random.normal(60, 12, n).astype(int).clip(30, 95)
        sex = np.random.binomial(1, 0.5, n)
        bmi = np.random.normal(25, 5, n).round(1).clip(15, 50)

        # --- Treatment & Comorbidities ---
        logit_treat = -4.5 + (0.05 * age) + (0.08 * bmi) + (0.2 * sex)
        p_treat = 1 / (1 + np.exp(-logit_treat))
        group = np.random.binomial(1, p_treat, n)

        logit_dm = -5 + (0.04 * age) + (0.1 * bmi)
        p_dm = 1 / (1 + np.exp(-logit_dm))
        diabetes = np.random.binomial(1, p_dm, n)

        logit_ht = -4 + (0.06 * age) + (0.05 * bmi)
        p_ht = 1 / (1 + np.exp(-logit_ht))
        hypertension = np.random.binomial(1, p_ht, n)

        # --- Survival Data ---
        lambda_base = 0.002
        linear_predictor = (
            0.03 * age
            + 0.4 * diabetes
            + 0.3 * hypertension
            - 0.6 * group
            + 0.005 * (bmi - 25) ** 2  # Non-linear BMI effect (U-shape)
        )
        hazard = lambda_base * np.exp(linear_predictor)
        surv_time = np.random.exponential(1 / hazard, n)
        censor_time = np.random.uniform(0, 100, n)
        time_obs = np.minimum(surv_time, censor_time).round(1)
        time_obs = np.maximum(time_obs, 0.5)
        status_death = (surv_time <= censor_time).astype(int)

        # --- Outcomes & Labs ---
        logit_cure = 0.5 + 1.2 * group - 0.04 * age - 0.5 * diabetes
        p_cure = 1 / (1 + np.exp(-logit_cure))
        outcome_cured = np.random.binomial(1, p_cure, n)

        gold_std = np.random.binomial(1, 0.3, n)
        rapid_score = np.where(
            gold_std == 0, np.random.normal(20, 10, n), np.random.normal(50, 15, n)
        )
        rapid_score = np.clip(rapid_score, 0, 100).round(1)

        # Second continuous test for comparison (e.g., more expensive but slightly better)
        expensive_score = np.where(
            gold_std == 0,
            np.random.normal(15, 8, n),
            np.random.normal(55, 12, n),
        )
        expensive_score = np.clip(expensive_score, 0, 100).round(1)

        rater_a = np.where(
            gold_std == 1,
            np.random.binomial(1, 0.85, n),
            np.random.binomial(1, 0.10, n),
        )

        agree_prob = 0.85
        rater_b = np.where(
            np.random.binomial(1, agree_prob, n) == 1, rater_a, 1 - rater_a
        )
        rater_c = np.where(
            np.random.binomial(1, agree_prob - 0.05, n) == 1, rater_a, 1 - rater_a
        )
        rater_d = np.where(
            np.random.binomial(1, agree_prob - 0.10, n) == 1, rater_a, 1 - rater_a
        )
        rater_e = np.random.choice([0, 1], n)  # Random rater for contrast

        hba1c = np.random.normal(6.5, 1.5, n).clip(4, 14).round(1)
        glucose = ((hba1c * 15) + np.random.normal(0, 15, n)).round(0)

        icc_rater1 = np.random.normal(120, 15, n).round(1)
        icc_rater2 = (icc_rater1 + 5 + np.random.normal(0, 4, n)).round(1)

        # --- TVC Logic ---
        tvc_df = _generate_tvc_data(n)

        # --- Additional Variables ---
        visits = np.random.poisson(2, n)
        visits = np.where(group == 1, visits + np.random.poisson(1, n), visits)

        cost_base = np.random.gamma(2, 2000, n)
        cost = cost_base + (diabetes * 1000) + (hypertension * 500) + (group * 2000)
        cost = cost.round(2)

        chol = np.random.normal(200, 40, n).astype(int).clip(100, 400)
        statin = np.random.binomial(1, 0.4, n)
        kidney_dz = np.random.binomial(1, 0.15, n)
        falls = np.random.poisson(0.5, n)

        data = {
            "ID": range(1, n + 1),
            "Treatment_Group": group,
            "Age_Years": age,
            "Sex_Male": sex,
            "BMI_kgm2": bmi,
            "Comorb_Diabetes": diabetes,
            "Comorb_Hypertension": hypertension,
            "Comorb_Kidney_Disease": kidney_dz,
            "Medication_Statin": statin,
            "Outcome_Cured": outcome_cured,
            "Time_Months": time_obs,
            "Status_Death": status_death,
            "Count_Hospital_Visits": visits,
            "History_Falls": falls,
            "Cost_Treatment_USD": cost,
            "Lab_Cholesterol_mgdL": chol,
            "Gold_Standard_Disease": gold_std,
            "Test_Score_Rapid": rapid_score,
            "Test_Score_Expensive": expensive_score,
            "Diagnosis_Dr_A": rater_a,
            "Diagnosis_Dr_B": rater_b,
            "Diagnosis_Dr_C": rater_c,
            "Diagnosis_Dr_D": rater_d,
            "Diagnosis_Dr_E": rater_e,
            "Lab_HbA1c": hba1c,
            "Lab_Glucose": glucose,
            "ICC_SysBP_Rater1": icc_rater1,
            "ICC_SysBP_Rater2": icc_rater2,
            "id_tvc": tvc_df["id_tvc"].values,
            "time_start": tvc_df["time_start"].values,
            "time_stop": tvc_df["time_stop"].values,
            "status_event": tvc_df["status_event"].values,
            "TVC_Value": tvc_df["TVC_Value"].values,
            "Static_Age": tvc_df["Static_Age"].values,
            "Static_Sex": tvc_df["Static_Sex"].values,
        }

        new_df = pd.DataFrame(data)
        _apply_random_missingness(new_df)

        meta = _get_example_metadata()
        return new_df, meta

    def _generate_tvc_data(n: int) -> pd.DataFrame:
        tvc_intervals_template = [0, 3, 6, 12, 24]
        tvc_data_rows = []
        current_row_count = 0
        tvc_patient_id = 1

        while current_row_count < n:
            p_age = np.random.randint(30, 80)
            p_sex = np.random.choice([0, 1])
            p_max_followup = np.random.uniform(3, 30)
            p_has_event = np.random.choice([0, 1], p=[0.4, 0.6])

            baseline_val = np.random.normal(50, 10)
            current_val = baseline_val

            for i in range(len(tvc_intervals_template) - 1):
                start_t = tvc_intervals_template[i]
                stop_t = tvc_intervals_template[i + 1]

                if start_t >= p_max_followup:
                    break

                actual_stop = min(stop_t, p_max_followup)
                r_start = round(start_t, 1)
                r_stop = round(actual_stop, 1)
                if r_stop <= r_start:
                    r_stop = r_start + 0.1

                is_final = (actual_stop >= p_max_followup) or (stop_t >= p_max_followup)
                event = 1 if (p_has_event and is_final) else 0

                drift = np.random.normal(2, 3)
                current_val = np.clip(current_val + drift, 10, 120)

                if current_row_count < n:
                    tvc_data_rows.append(
                        {
                            "id_tvc": tvc_patient_id,
                            "time_start": r_start,
                            "time_stop": r_stop,
                            "status_event": event,
                            "TVC_Value": round(current_val, 1),
                            "Static_Age": p_age,
                            "Static_Sex": p_sex,
                        }
                    )
                    current_row_count += 1
                else:
                    break

                if actual_stop >= p_max_followup:
                    break
            tvc_patient_id += 1

        tvc_df = pd.DataFrame(tvc_data_rows)
        # Pad if somehow short
        if len(tvc_df) < n:
            padding = pd.DataFrame(
                np.nan, index=range(n - len(tvc_df)), columns=tvc_df.columns
            )
            tvc_df = pd.concat([tvc_df, padding], ignore_index=True)
        return tvc_df.iloc[:n]

    def _apply_random_missingness(df_in: pd.DataFrame):
        protected_cols = ["ID", "id_tvc", "time_start", "time_stop", "status_event"]
        for col in df_in.columns:
            if col not in protected_cols:
                valid_mask = df_in[col].notna()
                if valid_mask.sum() > 0:
                    random_mask = np.random.choice(
                        [True, False], size=len(df_in), p=[0.00618, 1 - 0.00618]
                    )
                    df_in.loc[valid_mask & random_mask, col] = np.nan

    def _get_example_metadata() -> dict[str, Any]:
        """
        Provide example metadata for columns in the simulated clinical dataset.

        Returns:
            metadata (dict[str, Any]): Mapping from column name to a metadata dictionary containing:
                - "type": either "Categorical" or "Continuous".
                - "label": human-readable display label for the variable.
                - "map": a mapping of coded values to labels for categorical variables (empty dict for continuous variables).
        """
        return {
            "Treatment_Group": {
                "type": "Categorical",
                "map": {0: "Standard Care", 1: "New Drug"},
                "label": "Treatment Group",
            },
            "Sex_Male": {
                "type": "Categorical",
                "map": {0: "Female", 1: "Male"},
                "label": "Sex",
            },
            "Comorb_Diabetes": {
                "type": "Categorical",
                "map": {0: "No", 1: "Yes"},
                "label": "Diabetes",
            },
            "Comorb_Hypertension": {
                "type": "Categorical",
                "map": {0: "No", 1: "Yes"},
                "label": "Hypertension",
            },
            "Comorb_Kidney_Disease": {
                "type": "Categorical",
                "map": {0: "No", 1: "Yes"},
                "label": "Kidney Disease",
            },
            "Medication_Statin": {
                "type": "Categorical",
                "map": {0: "No", 1: "Yes"},
                "label": "Statin Use",
            },
            "Outcome_Cured": {
                "type": "Categorical",
                "map": {0: "Not Cured", 1: "Cured"},
                "label": "Outcome (Cured)",
            },
            "Status_Death": {
                "type": "Categorical",
                "map": {0: "Censored/Alive", 1: "Dead"},
                "label": "Status (Death)",
            },
            "Gold_Standard_Disease": {
                "type": "Categorical",
                "map": {0: "Healthy", 1: "Disease"},
                "label": "Gold Standard",
            },
            "Diagnosis_Dr_A": {
                "type": "Categorical",
                "map": {0: "Normal", 1: "Abnormal"},
                "label": "Diagnosis (Dr. A)",
            },
            "Diagnosis_Dr_B": {
                "type": "Categorical",
                "map": {0: "Normal", 1: "Abnormal"},
                "label": "Diagnosis (Dr. B)",
            },
            "Diagnosis_Dr_C": {
                "type": "Categorical",
                "map": {0: "Normal", 1: "Abnormal"},
                "label": "Diagnosis (Dr. C)",
            },
            "Diagnosis_Dr_D": {
                "type": "Categorical",
                "map": {0: "Normal", 1: "Abnormal"},
                "label": "Diagnosis (Dr. D)",
            },
            "Diagnosis_Dr_E": {
                "type": "Categorical",
                "map": {0: "Normal", 1: "Abnormal"},
                "label": "Diagnosis (Dr. E)",
            },
            "Age_Years": {"type": "Continuous", "label": "Age (Years)", "map": {}},
            "BMI_kgm2": {"type": "Continuous", "label": "BMI (kg/m²)", "map": {}},
            "Time_Months": {"type": "Continuous", "label": "Time (Months)", "map": {}},
            "Count_Hospital_Visits": {
                "type": "Continuous",
                "label": "Hospital Visits (Count)",
                "map": {},
            },
            "History_Falls": {
                "type": "Continuous",
                "label": "History of Falls (Count)",
                "map": {},
            },
            "Cost_Treatment_USD": {
                "type": "Continuous",
                "label": "Treatment Cost ($)",
                "map": {},
            },
            "Lab_Cholesterol_mgdL": {
                "type": "Continuous",
                "label": "Cholesterol (mg/dL)",
                "map": {},
            },
            "Test_Score_Rapid": {
                "type": "Continuous",
                "label": "Rapid Test Score (0-100)",
                "map": {},
            },
            "Test_Score_Expensive": {
                "type": "Continuous",
                "label": "Expensive Test Score (0-100)",
                "map": {},
            },
            "Lab_HbA1c": {"type": "Continuous", "label": "HbA1c (%)", "map": {}},
            "Lab_Glucose": {
                "type": "Continuous",
                "label": "Fasting Glucose (mg/dL)",
                "map": {},
            },
            "ICC_SysBP_Rater1": {
                "type": "Continuous",
                "label": "Sys BP (Rater 1)",
                "map": {},
            },
            "ICC_SysBP_Rater2": {
                "type": "Continuous",
                "label": "Sys BP (Rater 2)",
                "map": {},
            },
            "id_tvc": {"type": "Continuous", "label": "TVC Patient ID", "map": {}},
            "time_start": {
                "type": "Continuous",
                "label": "TVC Interval Start",
                "map": {},
            },
            "time_stop": {
                "type": "Continuous",
                "label": "TVC Interval Stop",
                "map": {},
            },
            "status_event": {
                "type": "Categorical",
                "map": {0: "Censored", 1: "Event"},
                "label": "TVC Event Status",
            },
            "TVC_Value": {
                "type": "Continuous",
                "label": "Time-Varying Covariate",
                "map": {},
            },
            "Static_Age": {
                "type": "Continuous",
                "label": "TVC Static Age",
                "map": {},
            },
            "Static_Sex": {
                "type": "Categorical",
                "map": {0: "Female", 1: "Male"},
                "label": "TVC Static Sex",
            },
        }

    @reactive.effect
    @reactive.event(input.btn_load_example)
    def _handle_btn_load_example():
        generate_example_data_logic()

    @reactive.effect
    @reactive.event(input.btn_load_example_trigger)
    def _handle_btn_load_example_trigger():
        generate_example_data_logic()

    @reactive.Effect
    @reactive.event(input.btn_jump_upload_trigger)
    def _jump_to_upload():
        """Smooth scroll to upload area when clicking the jump button in empty state"""
        file_upload_id = session.ns("file_upload")
        ui.insert_ui(
            ui.tags.script(
                f"document.getElementById('{file_upload_id}')"
                ".scrollIntoView({behavior: 'smooth'});"
            ),
            selector="body",
            where="beforeEnd",
        )

    # --- Data Upload Confirmation Modal ---
    @render.ui
    def modal_confirm_upload():
        return None  # Placeholder, modal is triggered dynamically

    pending_file = reactive.Value(None)

    @reactive.Effect
    @reactive.event(lambda: input.file_upload())
    def _handle_file_upload():
        file_infos: list[FileInfo] = input.file_upload()
        if not file_infos:
            return

        f = file_infos[0]

        # If data already exists, ask for confirmation
        if df.get() is not None:
            pending_file.set(f)
            m = ui.modal(
                "Loading a new file will replace the current dataset. "
                "All unsaved changes and analysis results will be lost. "
                "Are you sure?",
                title="⚠️ Confirm Replace Data",
                footer=ui.div(
                    ui.input_action_button(
                        "btn_confirm_upload", "Yes, Replace", class_="btn-danger"
                    ),
                    ui.modal_button("Cancel", class_="btn-secondary"),
                ),
                easy_close=True,
            )
            ui.modal_show(m)
        else:
            # No existing data, load immediately
            _load_data_file(f)

    @reactive.Effect
    @reactive.event(lambda: input.btn_confirm_upload())
    def _confirm_upload_action():
        f = pending_file.get()
        if f:
            ui.modal_remove()
            _load_data_file(f)
            pending_file.set(None)

    def _load_data_file(f: FileInfo):
        is_loading_data.set(True)
        data_issues.set([])  # Reset report

        try:
            # --- 1. Load File ---
            # Use robust loader to handle encodings, separators, and formats automatically
            new_df = load_data_robust(f["datapath"])

            # --- 2. Limit Data Size ---
            max_rows = 100000
            if len(new_df) > max_rows:
                new_df = new_df.head(max_rows)
                ui.notification_show(
                    f"⚠️ Large file: showing first {max_rows:,} rows", type="warning"
                )

            # --- 3. Unified Ingestion & Quality Check ---
            _process_ingested_dataset(new_df, file_name=f["name"])

        except Exception as e:
            logger.error("Error: %s", e)
            ui.notification_show(f"❌ Error: {e!s}", type="error")
        finally:
            is_loading_data.set(False)

    @reactive.Effect
    @reactive.event(lambda: input.btn_reset_all())
    def _():
        df.set(None)
        var_meta.set({})
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        data_issues.set([])
        is_loading_data.set(False)
        ui.notification_show("All data reset", type="warning")

    # --- 2. Metadata Logic (Simplified with Dynamic UI) ---

    # Update Dropdown list
    @reactive.Effect
    def _update_var_select():
        data = df.get()
        if data is not None:
            cols = ["Select...", *data.columns.tolist()]
            ui.update_select("sel_var_edit", choices=cols)

    # Render Settings UI dynamically when a variable is selected
    @render.ui
    def ui_var_settings():
        var_name = input.sel_var_edit()

        if not var_name or var_name == "Select...":
            return None

        # Retrieve current meta
        meta = var_meta.get()
        current_type = "Continuous"
        map_str = ""

        if meta and var_name in meta:
            m = meta[var_name]
            current_type = m.get("type", "Continuous")
            map_str = "\n".join([f"{k}={v}" for k, v in m.get("map", {}).items()])

        return ui.TagList(
            ui.input_radio_buttons(
                "radio_var_type",
                "Variable Type:",
                choices={"Continuous": "Continuous", "Categorical": "Categorical"},
                selected=current_type,
                inline=True,
            ),
            ui.input_text_area(
                "txt_var_map",
                "Value Labels (Format: 0=No, 1=Yes)",
                value=map_str,
                height="100px",
            ),
            ui.input_action_button(
                "btn_save_meta", "💾 Save Settings", class_="btn-primary"
            ),
        )

    @reactive.Effect
    @reactive.event(lambda: input.btn_save_meta())
    def _save_metadata():
        """
        Save the currently selected variable's metadata (type, label, and value mapping) into the shared var_meta store and notify the user.

        Parses the mapping text from input.txt_var_map() where each non-empty line with an '=' defines a source=>label pair. Keys that parse as numbers are converted to int when integer-valued or float otherwise; all other keys are kept as stripped strings. Malformed mapping lines are ignored and logged at debug level. If no variable is selected ("Select..."), the function exits without changing metadata. After updating, the variable's metadata entry is set with keys "type", "map", and "label", and a success notification is shown.
        """
        var_name = input.sel_var_edit()
        if var_name == "Select...":
            return

        new_map = {}
        map_input = input.txt_var_map()
        if map_input:
            for line in map_input.split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    try:
                        k_clean = k.strip()
                        try:
                            k_num = float(k_clean)
                            k_val = int(k_num) if k_num.is_integer() else k_num
                        except (ValueError, TypeError):
                            k_val = k_clean
                        new_map[k_val] = v.strip()
                    except (ValueError, AttributeError) as e:
                        logger.debug(
                            "Skipping malformed mapping line: %s - %s", line, e
                        )

        current_meta = var_meta.get() or {}
        current_meta[var_name] = {
            "type": input.radio_var_type(),
            "map": new_map,
            "label": var_name,
        }
        var_meta.set(current_meta)
        ui.notification_show(f"✅ Saved settings for {var_name}", type="message")

    @reactive.Effect
    def _update_cleaning_choices():
        """Update choices for imputation, outliers, and transformation when data changes"""
        data = df.get()
        if data is not None:
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            ui.update_select("sel_impute_cols", choices=numeric_cols)
            ui.update_select("sel_outlier_cols", choices=numeric_cols)
            ui.update_select("sel_trans_var", choices=["Select...", *numeric_cols])
        else:
            ui.update_select("sel_impute_cols", choices=[])
            ui.update_select("sel_outlier_cols", choices=[])
            ui.update_select("sel_trans_var", choices=["Select..."])

    @reactive.Effect
    @reactive.event(input.btn_run_impute)
    def _handle_imputation():
        """
        Imputes missing values for the currently loaded dataframe on the user-selected columns using the chosen method.

        If a dataframe and one or more target columns are selected, updates the module's dataframe in-place with the imputed result and shows a success notification; on error the dataframe is left unchanged and a failure notification is shown (exceptions are caught and logged).
        """
        d = df.get()
        cols = input.sel_impute_cols()
        method = input.sel_impute_method()

        if d is not None and cols:
            try:
                with ui.Progress(min=0, max=1) as p:
                    p.set(
                        message=f"Running {method} imputation...", detail="Please wait"
                    )
                    numeric_cols = (
                        d[list(cols)].select_dtypes(include=np.number).columns.tolist()
                    )
                    if not numeric_cols:
                        ui.notification_show(
                            "⚠️ No numeric columns selected for imputation",
                            type="warning",
                        )
                        return
                    new_df = impute_missing_data(d, numeric_cols, method=method)
                    df.set(new_df)
                ui.notification_show(
                    f"✅ Imputed {len(numeric_cols)} columns using {method}",
                    type="message",
                )
            except Exception as e:
                logger.error("Imputation error: %s", e)
                ui.notification_show(f"❌ Imputation failed: {e}", type="error")

    @reactive.Effect
    @reactive.event(input.btn_run_outlier)
    def _handle_outliers():
        """
        Apply the configured outlier handling to the selected columns and update the active dataframe.

        Reads the target columns, method, action, and threshold from the module inputs, applies outlier handling to each selected column, replaces those columns in the active dataframe, and shows a success notification with the number of processed columns. If no dataframe is loaded or no columns are selected, no changes are made. On error, logs the exception and shows an error notification.
        """
        d = df.get()
        cols = input.sel_outlier_cols()
        method = input.sel_outlier_method()
        action = input.sel_outlier_action()
        thresh = input.num_outlier_thresh()

        if d is not None and cols:
            try:
                new_df = d.copy()
                count = 0
                for col in cols:
                    # handle_outliers returns a Series
                    new_df[col] = handle_outliers(
                        new_df[col], method=method, action=action, threshold=thresh
                    )
                    count += 1

                df.set(new_df)
                ui.notification_show(
                    f"✅ Handled outliers in {count} columns ({action})", type="message"
                )
            except Exception as e:
                logger.error("Outlier error: %s", e)
                ui.notification_show(f"❌ Outlier handling failed: {e}", type="error")

    # ==========================================
    # Multiple Imputation (MI) Server Logic
    # ==========================================

    # Reactive state for MI results (local - stores full MICEResult)
    # Note: mi_imputed_datasets is now shared via parameter from app.py
    mi_result: reactive.Value[Any] = reactive.Value(None)

    @reactive.Effect
    def _update_mi_choices():
        """Update MI column choices when data changes"""
        data = df.get()
        if data is not None:
            # Get columns with missing data (numeric only)
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            cols_with_missing = [c for c in numeric_cols if data[c].isna().any()]
            ui.update_select("sel_mi_cols", choices=cols_with_missing)
        else:
            ui.update_select("sel_mi_cols", choices=[])

    @reactive.Effect
    @reactive.event(input.btn_run_mi)
    def _handle_mi():
        """Run Multiple Imputation and store results"""
        from utils.multiple_imputation import MICEImputer

        d = df.get()
        cols = input.sel_mi_cols()
        m = input.num_mi_m() or 5
        max_iter = input.num_mi_iter() or 10

        if d is None or not cols:
            ui.notification_show(
                "⚠️ Please load data and select columns to impute",
                type="warning",
            )
            return

        try:
            with ui.Progress(min=0, max=1) as p:
                p.set(
                    message=f"Running MICE with m={m} imputations...",
                    detail="This may take a moment",
                )

                imputer = MICEImputer(
                    n_imputations=m,
                    max_iter=max_iter,
                    random_state=42,
                )
                result = imputer.fit_transform(d, columns=list(cols))

                # Store results
                mi_result.set(result)
                mi_imputed_datasets.set(result.imputed_datasets)

            ui.notification_show(
                f"✅ Generated {len(result.imputed_datasets)} imputed datasets "
                f"({len(result.columns_imputed)} columns)",
                type="message",
            )
            logger.info("MI complete: %d datasets", len(result.imputed_datasets))

        except Exception as e:
            logger.exception("MI error: %s", e)
            ui.notification_show(f"❌ Multiple Imputation failed: {e}", type="error")

    @render.ui
    def ui_mi_status():
        """Render MI status and summary"""
        result = mi_result.get()
        d = df.get()

        if result is None:
            if d is None:
                return ui.div(
                    ui.tags.i(class_="bi bi-info-circle me-2"),
                    "Load data to begin",
                    class_="text-muted text-center p-4",
                )

            # Show missing data summary
            numeric_cols = d.select_dtypes(include=np.number).columns.tolist()
            cols_with_missing = [c for c in numeric_cols if d[c].isna().any()]

            if not cols_with_missing:
                return ui.div(
                    ui.tags.i(class_="bi bi-check-circle text-success me-2"),
                    "No missing data in numeric columns!",
                    class_="text-center p-4",
                )

            missing_counts = [d[c].isna().sum() for c in cols_with_missing]
            total_missing = sum(missing_counts)

            return ui.div(
                ui.tags.div(
                    ui.tags.span(
                        f"{len(cols_with_missing)}", class_="fs-3 fw-bold text-primary"
                    ),
                    ui.tags.span(" columns with missing data", class_="text-muted"),
                    class_="mb-2",
                ),
                ui.tags.div(
                    ui.tags.span(f"{total_missing:,}", class_="fs-4 fw-semibold"),
                    ui.tags.span(" total missing values", class_="text-muted"),
                    class_="mb-3",
                ),
                ui.tags.small(
                    "Select columns above and click 'Run Multiple Imputation' to generate imputed datasets.",
                    class_="text-muted",
                ),
                class_="p-3",
            )

        # Show MI results summary
        from utils.multiple_imputation import get_imputation_summary

        summary_df = get_imputation_summary(result)

        return ui.div(
            ui.div(
                ui.tags.span("✅ ", class_="text-success"),
                ui.tags.span(
                    f"{result.n_imputations} Imputed Datasets Generated",
                    class_="fw-bold",
                ),
                class_="alert alert-success mb-3",
            ),
            ui.tags.h6("📊 Imputation Summary", class_="mb-2"),
            ui.HTML(
                summary_df.to_html(
                    index=False,
                    classes="table table-sm table-striped",
                    float_format=lambda x: f"{x:.2f}",
                )
            )
            if not summary_df.empty
            else ui.div("No summary available"),
            ui.hr(),
            ui.div(
                ui.input_action_button(
                    "btn_apply_mi",
                    "📥 Apply First Imputation to Data",
                    class_="btn-outline-success btn-sm me-2",
                ),
                ui.input_action_button(
                    "btn_clear_mi",
                    "🗑️ Clear MI Results",
                    class_="btn-outline-secondary btn-sm",
                ),
                class_="d-flex gap-2 mt-3",
            ),
            class_="p-2",
        )

    @reactive.Effect
    @reactive.event(input.btn_apply_mi)
    def _apply_first_imputation():
        """Apply first imputed dataset to main dataframe"""
        datasets = mi_imputed_datasets.get()
        if datasets:
            df.set(datasets[0])
            ui.notification_show(
                "✅ Applied first imputed dataset to data",
                type="message",
            )

    @reactive.Effect
    @reactive.event(input.btn_clear_mi)
    def _clear_mi_results():
        """Clear MI results"""
        mi_result.set(None)
        mi_imputed_datasets.set([])
        ui.notification_show("🗑️ MI results cleared", type="message")

    @render.ui
    def ui_mi_diagnostics():
        """Render MI diagnostic plots"""
        result = mi_result.get()
        d = df.get()

        if result is None:
            return ui.div(
                "Run Multiple Imputation to view diagnostics",
                class_="text-muted text-center p-4",
            )

        try:
            from utils.multiple_imputation import create_imputation_diagnostics
            from utils.plotly_html_renderer import plotly_figure_to_html

            figures = create_imputation_diagnostics(result, d)

            if not figures:
                return ui.div(
                    "No diagnostic plots available",
                    class_="text-muted text-center p-3",
                )

            plots_html = []
            for name, fig in figures.items():
                plots_html.append(
                    ui.div(
                        ui.HTML(plotly_figure_to_html(fig)),
                        class_="mb-3",
                    )
                )

            return ui.div(*plots_html)

        except Exception as e:
            logger.error("MI diagnostics error: %s", e)
            return ui.div(
                f"Error generating diagnostics: {e}",
                class_="alert alert-warning",
            )

    @render.ui
    def ui_missing_plot():
        """
        Render the dataset's missing-value pattern as an embeddable Plotly HTML fragment.

        Returns:
            html_fragment (str): HTML string containing a Plotly figure showing missing-data patterns for the current dataframe, or `None` if no dataframe is loaded.
        """
        d = df.get()
        if d is None:
            return None

        fig = plot_missing_pattern(d)
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    @render.ui
    def ui_assumption_result():
        """
        Render a UI summary of distributional assumption checks for the currently selected transformation variable.

        When a variable is selected and data is available, returns a UI fragment that displays the normality test name, p-value, pass/fail status (with color coding), and optional skewness and kurtosis values. If no data or variable is selected, returns None. If an error occurs while checking assumptions, returns an alert UI element describing the error.

        Returns:
            ui_element: A UI component summarizing normality/assumption results for the selected variable, or `None` when no data/variable is available. If an error occurs during the check, an alert UI element is returned containing the error message.
        """
        var_name = input.sel_trans_var()
        d = df.get()

        if d is None or not var_name or var_name == "Select...":
            return None

        try:
            res = check_assumptions(d[var_name])

            if "error" in res:
                return ui.div(f"Error: {res['error']}", class_="alert alert-danger")
            if res.get("normality_test") == "Insufficient Data":
                return ui.div(
                    "Insufficient data for normality test (minimum 3 observations required).",
                    class_="alert alert-warning",
                )
            if res.get("normality_test") == "Insufficient Variance":
                return ui.div(
                    "Insufficient variance for normality test (data are constant).",
                    class_="alert alert-warning",
                )

            color = COLORS["success"] if res["is_normal"] else COLORS["danger"]
            status_text = (
                "Normal Distribution" if res["is_normal"] else "NOT Normal Distribution"
            )

            return ui.div(
                ui.h6(f"Test: {res['normality_test']}"),
                ui.div(
                    ui.span("P-Value: ", class_="text-muted"),
                    ui.span(
                        f"{res['p_value']}", style=f"color: {color}; font-weight: bold;"
                    ),
                ),
                ui.div(
                    ui.span("Result: ", class_="text-muted"),
                    ui.span(status_text, style=f"color: {color}; font-weight: bold;"),
                ),
                ui.div(
                    ui.tags.small(
                        f"Skewness: {res.get('skewness', 'N/A')} | Kurtosis: {res.get('kurtosis', 'N/A')}"
                    ),
                    class_="mt-1 text-muted",
                ),
                class_="alert alert-light border shadow-sm mt-2",
            )
        except Exception as e:
            return ui.div(
                f"Error checking assumptions: {e}", class_="alert alert-danger"
            )

    @reactive.Effect
    @reactive.event(input.btn_run_trans)
    def _handle_transform():
        """
        Create a new transformed variable from a selected column and add it to the dataset and metadata.

        Applies the chosen transformation method to the currently selected variable, appends the resulting series to the reactive DataFrame under a generated name "<variable>_<method>" (adding a numeric suffix if that name already exists), and records metadata for the new variable with type "Continuous" and label "<original> (<method>)". Shows a success notification on completion or an error notification if the transformation fails.
        """
        d = df.get()
        var_name = input.sel_trans_var()
        method = input.sel_trans_method()

        if d is not None and var_name and var_name != "Select...":
            try:
                col_data = d[var_name]
                new_col = transform_variable(col_data, method=method)

                # Create description for the new variable
                new_var_name = f"{var_name}_{method}"
                if new_var_name in d.columns:
                    suffix = 1
                    while f"{new_var_name}_{suffix}" in d.columns:
                        suffix += 1
                    new_var_name = f"{new_var_name}_{suffix}"

                # Update DataFrame
                new_df = d.copy()
                new_df[new_var_name] = new_col
                df.set(new_df)

                # Update metadata
                current_meta = var_meta.get() or {}
                current_meta[new_var_name] = {
                    "type": "Continuous",
                    "map": {},
                    "label": f"{var_name} ({method})",
                }
                var_meta.set(current_meta)

                ui.notification_show(
                    f"✅ Created new variable: {new_var_name}", type="message"
                )

            except Exception as e:
                logger.error("Transformation error: %s", e)
                ui.notification_show(f"❌ Transformation failed: {e}", type="error")

    @render.text
    def txt_trans_preview():
        """
        Return a short status message describing the selected variable transformation.

        Returns:
            str: `"Please select a variable and transformation method."` when no variable or method is selected, otherwise
            `"Ready to apply '<method>' transformation to '<variable>'."`
        """
        var_name = input.sel_trans_var()
        method = input.sel_trans_method()

        if not var_name or var_name == "Select...":
            return "Please select a variable and transformation method."

        return f"Ready to apply '{method}' transformation to '{var_name}'."

    # --- Missing Data Configuration Handlers ---
    @render.ui
    def ui_missing_preview():
        """
        Render a preview of the configured missing-value codes for the currently selected variable.

        If no variable is selected or the variable has no metadata or missing-value configuration, the returned UI element displays an explanatory message; otherwise it displays the configured codes.

        Returns:
            ui_element: A Shiny UI element that displays either an explanatory message or the list of missing-value codes for the selected variable.
        """
        var_name = input.sel_var_edit()
        if not var_name or var_name == "Select...":
            return ui.div(
                ui.p(
                    "Select a variable to view missing data configuration.",
                    class_="text-muted-sm",
                ),
            )

        meta = var_meta.get()
        if not meta or var_name not in meta:
            return ui.p(
                "No config yet",
                class_="text-muted-sm",
            )

        missing_vals = meta[var_name].get("missing_values", [])
        if not missing_vals:
            return ui.p(
                "No missing codes configured",
                class_="text-muted-sm",
            )

        codes_str = ", ".join(str(v) for v in missing_vals)
        return ui.p(
            f"✓ Codes: {codes_str}",
            style=f"color: {COLORS['success']}; font-weight: 500; font-size: 0.9em;",
        )

    @reactive.Effect
    @reactive.event(lambda: input.btn_save_missing())
    def _save_missing_config():
        """Save missing data configuration for selected variable"""
        var_name = input.sel_var_edit()
        if not var_name or var_name == "Select...":
            ui.notification_show("⚠️ Select a variable first", type="warning")
            return

        # Parse comma-separated missing codes
        missing_input = input.txt_missing_codes()
        missing_codes = []

        if missing_input.strip():
            for raw_item in missing_input.split(","):
                item = raw_item.strip()
                if not item:
                    continue
                # Try to parse as number
                try:
                    num = float(item)
                    num = int(num) if num.is_integer() else num
                    missing_codes.append(num)
                except ValueError:
                    # If not a number, treat as string
                    missing_codes.append(item)

        # Update metadata
        current_meta = var_meta.get() or {}
        if var_name not in current_meta:
            current_meta[var_name] = {
                "type": "Continuous",
                "map": {},
                "label": var_name,
            }

        current_meta[var_name]["missing_values"] = missing_codes
        var_meta.set(current_meta)

        codes_display = (
            ", ".join(str(c) for c in missing_codes) if missing_codes else "None"
        )
        ui.notification_show(
            f"✅ Missing codes for '{var_name}' set to: {codes_display}", type="message"
        )

    # --- 3. Render Outputs ---
    @render.ui
    def ui_preview_area():
        d = df.get()
        if d is None:
            return create_empty_state_ui(
                message="No Data Uploaded",
                sub_message=(
                    "Upload a CSV or Excel (.xlsx) file to start your analysis. "
                    "You can also load example data to explore features."
                ),
                icon="📂",
                action_button=ui.div(
                    ui.input_action_button(
                        "btn_jump_upload_trigger",
                        "⬆️ Upload File",
                        class_="btn-primary me-2 shadow-sm",
                    ),
                    ui.input_action_button(
                        "btn_load_example_trigger",
                        "📄 Example Data",
                        class_="btn-outline-primary shadow-sm",
                    ),
                    class_="d-flex justify-content-center gap-2",
                ),
            )
        return ui.output_data_frame(session.ns("out_df_preview"))

    @render.data_frame
    def out_df_preview():
        """
        Render the current dataframe as a preview table.

        Note: Cell highlighting for data quality issues is disabled to avoid Styler compatibility issues
        with Shiny's DataTable renderer. Detailed issues are available in the Data Quality Scorecard.

        Returns:
            render.DataTable or None: A DataTable rendering of the current dataframe, or `None` when no data is available.
        """
        d = df.get()
        if d is None:
            return None

        return render.DataTable(d, width="100%", filters=False)

    @render.ui
    def ui_btn_clear_match():
        if is_matched.get():
            return ui.input_action_button("btn_clear_match", "🔄 Clear Matched Data")
        return None

    @reactive.Effect
    @reactive.event(lambda: input.btn_clear_match())
    def _():
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])

    # --- New: Data Health Report Renderer ---
    @render.ui
    def ui_file_metadata():
        """
        Render a compact metadata card showing dataset dimensions, approximate memory usage, and the uploaded file name.

        Returns:
            ui_element (Optional[ui.div]): A UI div containing the dataset "rows x cols" summary, approximate memory usage in MB, and the uploaded file name. Returns `None` if no dataset is loaded.
        """
        d = df.get()
        info = uploaded_file_info.get()
        if d is None:
            return None

        # Calculate approximate memory usage or file size if available
        # Simple row/col display as per audit
        mem_usage = d.memory_usage(deep=True).sum() / (1024 * 1024)

        return ui.div(
            ui.div(
                f"📊 {len(d):,} rows x {len(d.columns)} cols", style="font-weight: 600;"
            ),
            ui.div(
                f"📦 Memory: {mem_usage:.2f} MB",
                class_="text-muted-sm",
            ),
            ui.div(
                f"📄 {info.get('name', 'Unknown')}" if info else "",
                style="font-size: 0.85em; font-style: italic; margin-top: 4px;",
            ),
            style=(
                f"background: {COLORS['smoke_white']}; padding: 10px; border-radius: 6px; "
                f"margin-top: 10px; border: 1px solid {COLORS['border']};"
            ),
        )

    @render.ui
    def ui_data_report_card():
        """
        Renders the enhanced Data Quality Report with Value Boxes and Issue List.
        """
        data = df.get()
        if data is None:
            return None

        # Generate report
        report = DataQualityReport(data).generate_report()
        scores = report["dimension_scores"]
        issues = report["issues"]
        recommendations = report["recommendations"]

        # 1. UI: Score Cards
        score_cards = ui.layout_columns(
            ui.value_box(
                "Completeness",
                f"{scores['completeness']:.1f}%",
                "Non-missing values",
                theme="teal" if scores["completeness"] > 90 else "danger",
            ),
            ui.value_box(
                "Consistency",
                f"{scores['consistency']:.1f}%",
                "Type consistency",
                theme="blue" if scores["consistency"] > 90 else "warning",
            ),
            ui.value_box(
                "Uniqueness",
                f"{scores['uniqueness']:.1f}%",
                "Unique rows",
                theme="indigo" if scores["uniqueness"] > 90 else "warning",
            ),
            ui.value_box(
                "Validity",
                f"{scores['validity']:.1f}%",
                "Schema validation",
                theme="green",  # Currently hardcoded to 100
            ),
            col_widths=(3, 3, 3, 3),
        )

        # 2. UI: Details
        details_ui = None
        if issues or recommendations:
            recs_html = (
                "<ul>" + "".join([f"<li>{r}</li>" for r in recommendations]) + "</ul>"
            )

            # Previous issues from "data_issues" (legacy check)
            legacy_issues = data_issues.get()
            legacy_table = ""
            if legacy_issues:
                rows = ""
                for item in legacy_issues:
                    col = _html.escape(str(item["col"]))
                    row = _html.escape(str(item["row"]))
                    value = _html.escape(str(item["value"]))
                    issue = _html.escape(str(item["issue"]))
                    rows += (
                        f"<tr><td>{col}</td><td>{row}</td><td>{value}</td>"
                        f"<td class='text-danger'>{issue}</td></tr>"
                    )
                legacy_table = f"""
                 <h6 class="mt-3">Detailed Anomalies</h6>
                 <div class="table-responsive" style="max-height: 200px; overflow-y: auto;">
                     <table class="table table-sm table-striped table-bordered">
                         <thead class="table-danger">
                             <tr><th>Column</th><th>Row</th><th>Value</th><th>Issue</th></tr>
                         </thead>
                         <tbody>{rows}</tbody>
                     </table>
                 </div>
                 """

            details_ui = ui.accordion(
                ui.accordion_panel(
                    "📋 Detailed Quality Report",
                    ui.div(
                        ui.h6("Recommendations"),
                        ui.HTML(recs_html),
                        ui.HTML(legacy_table) if legacy_table else None,
                    ),
                    value="details",
                ),
                id="acc_dq_details",
                open=False,  # Collapsed by default
            )

        return ui.div(
            ui.h5("🛡️ Data Quality Scorecard", class_="mb-3"),
            score_cards,
            details_ui,
            class_="p-3 border rounded shadow-sm bg-white mb-3",
        )

    @render.ui
    def ui_data_quality_warnings():
        warnings = quality_warnings()
        if not warnings:
            return None

        # Format as list of items
        list_items = [
            ui.tags.li(ui.markdown(w), style="margin-bottom: 8px;") for w in warnings
        ]

        return ui.accordion(
            ui.accordion_panel(
                ui.div(
                    ui.tags.span(
                        "🧐 Data Quality Alerts", class_="fw-bold text-warning"
                    ),
                    ui.tags.span(
                        f"{len(warnings)} affected columns",
                        class_="badge bg-warning text-dark ms-2",
                    ),
                    class_="d-flex justify-content-between align-items-center",
                ),
                ui.div(
                    ui.tags.ul(
                        *list_items,
                        style=(
                            "padding-left: 20px; padding-top: 10px; "
                            "padding-bottom: 10px;"
                        ),
                    ),
                    ui.div(
                        ui.tags.strong("Tip: "),
                        "These issues might impact statistical analysis results. "
                        "Consider cleaning these values in the original file.",
                        class_="alert alert-light border-warning text-dark p-2 mt-2",
                        style="font-size: 0.9em;",
                    ),
                ),
                value="quality_alerts",
            ),
            open=False,
            id="acc_quality_warnings",
            class_="mb-3 border-warning shadow-sm",
        )
