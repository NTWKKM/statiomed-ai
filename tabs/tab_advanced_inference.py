from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from shiny import module, reactive, render, ui

from logger import get_logger
from tabs._common import get_color_palette, select_variable_by_keyword
from utils.collinearity_lib import calculate_vif
from utils.data_cleaning import prepare_data_for_analysis
from utils.formatting import (
    create_missing_data_report_html,
    format_p_value,
)
from utils.heterogeneity_lib import calculate_heterogeneity
from utils.mediation_lib import analyze_mediation
from utils.mi_helpers import get_mi_datasets, has_mi_datasets
from utils.model_diagnostics_lib import (
    calculate_cooks_distance,
    get_diagnostic_plot_data,
    run_heteroscedasticity_test,
    run_reset_test,
)
from utils.multiple_imputation import pool_estimates
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.ui_helpers import (
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
)

COLORS = get_color_palette()
logger = get_logger(__name__)


@module.ui
def advanced_inference_ui():
    """
    Constructs the Advanced Inference user interface with tabs for mediation analysis, collinearity diagnostics, OLS model diagnostics, heterogeneity testing, and an informational reference.

    Builds and returns a composite UI containing:
    - A header with dataset/matching status controls.
    - Tabbed panels for Mediation Analysis, Collinearity, Model Diagnostics (OLS), Heterogeneity Testing, and a Reference card describing key metrics (ACME/ADE/Total Effect, VIF/Tolerance guidance, residual diagnostics, Cook's Distance thresholds, and heterogeneity interpretation).
    """
    return ui.div(
        # Title + Data Summary inline
        ui.div(
            ui.h3("🔍 Advanced Inference"),
            ui.output_ui("ui_matched_info_ai"),
            ui.output_ui("ui_dataset_selector_ai"),
            class_="d-flex justify-content-between align-items-center mb-3",
        ),
        ui.navset_tab(
            ui.nav_panel(
                "🎯 Mediation Analysis",
                ui.card(
                    ui.card_header("Mediation Analysis"),
                    create_input_group(
                        "Variables",
                        ui.input_select("med_outcome", "Outcome (Y):", choices=[]),
                        ui.input_select("med_treatment", "Treatment (X):", choices=[]),
                        ui.input_select("med_mediator", "Mediator (M):", choices=[]),
                        ui.input_selectize(
                            "med_confounders",
                            "Confounders (C):",
                            choices=[],
                            multiple=True,
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_mediation_validation"),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_mediation",
                        "🚀 Run Mediation",
                        class_="btn-primary w-100",
                    ),
                ),
                ui.output_ui("out_mediation_status"),
                create_results_container(
                    "Results", ui.output_ui("out_mediation_container")
                ),
            ),
            ui.nav_panel(
                "🔬 Collinearity",
                ui.card(
                    ui.card_header("Collinearity Diagnostic"),
                    create_input_group(
                        "Variables",
                        ui.input_selectize(
                            "coll_vars", "Predictors:", choices=[], multiple=True
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_coll_validation"),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_collinearity", "Analyze", class_="btn-primary w-100"
                    ),
                ),
                create_results_container(
                    "VIF Results", ui.output_ui("out_vif_container")
                ),
            ),
            ui.nav_panel(
                "📊 Model Diagnostics",
                ui.card(
                    ui.card_header("Model Diagnostics (OLS)"),
                    create_input_group(
                        "Model Specification",
                        ui.input_select("diag_outcome", "Outcome (Y):", choices=[]),
                        ui.input_select(
                            "diag_predictor", "Main Predictor (X):", choices=[]
                        ),
                        ui.input_selectize(
                            "diag_covariates",
                            "Covariates:",
                            choices=[],
                            multiple=True,
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_diag_validation"),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_diag",
                        "🚀 Run Diagnostics",
                        class_="btn-primary w-100",
                    ),
                ),
                ui.output_ui("out_diag_status"),
                create_results_container(
                    "Diagnostic Results", ui.output_ui("out_diag_container")
                ),
            ),
            ui.nav_panel(
                "🏥 Heterogeneity Testing",
                ui.card(
                    ui.card_header("Meta-Analysis Data Entry"),
                    ui.p(
                        "Enter comma-separated values for effect sizes and variances."
                    ),
                    create_input_group(
                        "Data Inputs",
                        ui.input_text(
                            "het_effects", "Effect Sizes (e.g., 0.5, 0.4, 0.6):"
                        ),
                        ui.input_text(
                            "het_variances", "Variances (e.g., 0.01, 0.02, 0.015):"
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_het_validation"),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_het",
                        "🚀 Calculate Heterogeneity",
                        class_="btn-primary w-100",
                    ),
                ),
                create_results_container("Results", ui.output_ui("out_het_container")),
            ),
            ui.nav_panel(
                "ℹ️ Reference",
                ui.card(
                    ui.card_header("Advanced Inference Reference"),
                    ui.markdown("""
                        **Mediation Analysis:**
                        * **ACME (Indirect Effect):** The portion of the effect mediated by M. (Effect of X on Y via M).
                        * **ADE (Direct Effect):** The effect of X on Y, keeping M constant.
                        * **Total Effect:** ACME + ADE.

                        **Collinearity Diagnostics:**
                        * **VIF (Variance Inflation Factor):**
                            * **VIF > 5:** Moderate multicollinearity (Caution).
                            * **VIF > 10:** Severe multicollinearity (Consider removing variable).
                        * **Tolerance:** 1/VIF. Values < 0.1 indicate problems.

                        **Model Diagnostics (OLS):**
                        * **Residuals vs Fitted:** Checks linearity. Ideally, points fluctuate randomly around 0 (horizontal line).
                        * **Q-Q Plot:** Checks normality of residuals. Points should fall along the diagonal line.
                        * **Cook's Distance:** Measures influence. Points with Cook's D > 4/n (or > 1) are highly influential and may skew results.

                        **Heterogeneity (Meta-Analysis):**
                        * **I-squared (I²):**
                            * **< 25%:** Low heterogeneity.
                            * **25-75%:** Moderate heterogeneity.
                            * **> 75%:** High heterogeneity.
                        * **Q-Statistic P-value:**
                            * **P < 0.05:** Significant heterogeneity exists.
                        """),
                ),
            ),
        ),
    )


@module.server
def advanced_inference_server(
    input,
    output,
    session,
    df,
    var_meta,
    df_matched,
    is_matched,
    mi_imputed_datasets=None,  # NEW: MI state (future support)
):
    """Server logic for Advanced Inference."""

    mediation_results = reactive.Value(None)
    vif_results = reactive.Value(None)
    diag_results = reactive.Value(None)
    diag_plot_data = reactive.Value(None)
    cooks_results = reactive.Value(None)
    diag_clean_data = reactive.Value(None)
    het_results = reactive.Value(None)

    # Running States
    med_is_running = reactive.Value(False)
    coll_is_running = reactive.Value(False)
    diag_is_running = reactive.Value(False)
    het_is_running = reactive.Value(False)

    # ==================== MI HELPER FUNCTIONS ====================
    # (Shared utilities — see utils/mi_helpers.py)
    _has_mi_datasets = lambda: has_mi_datasets(mi_imputed_datasets)  # noqa: E731
    _get_mi_datasets = lambda: get_mi_datasets(mi_imputed_datasets)  # noqa: E731

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df():
        try:
            source = input.radio_ai_source()
        except Exception:
            source = "original"
        if is_matched.get() and source == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_matched_info_ai():
        """Display matched dataset availability info."""
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** - You can select it below for analysis",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_dataset_selector_ai():
        """Render dataset selector radio buttons."""
        if is_matched.get():
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_ai_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({original_len:,} rows)",
                    "matched": f"✅ Matched ({matched_len:,} rows)",
                },
                selected="original",
                inline=True,
            )
        return None

    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is not None:
            cols = d.columns.tolist()
            # -- Mediation Defaults --
            # Default: Outcome_Cured, Treatment_Group, Lab_Cholesterol_mgdL
            def_med_out = select_variable_by_keyword(
                cols, ["outcome_cured", "outcome", "cured"], default_to_first=True
            )
            def_med_treat = select_variable_by_keyword(
                cols, ["treatment_group", "treatment", "group"], default_to_first=True
            )
            def_med_mediator = select_variable_by_keyword(
                cols,
                ["lab_cholesterol", "cholesterol", "chol", "lab", "mediator"],
                default_to_first=True,
            )

            # Default Confounders: Age_Years, Sex_Male
            def_med_conf = []
            for c in cols:
                if "age" in c.lower() or "sex" in c.lower():
                    def_med_conf.append(c)

            ui.update_select("med_outcome", choices=cols, selected=def_med_out)
            ui.update_select("med_treatment", choices=cols, selected=def_med_treat)
            ui.update_select("med_mediator", choices=cols, selected=def_med_mediator)
            ui.update_selectize("med_confounders", choices=cols, selected=def_med_conf)

            # -- Collinearity Defaults --
            # Default: Age, BMI, Cholesterol, Cost
            def_coll_vars = []
            desired_coll = [
                "Age_Years",
                "BMI_kgm2",
                "Lab_Cholesterol_mgdL",
                "Cost_Treatment_USD",
            ]
            for d_col in desired_coll:
                matches = [c for c in cols if d_col in c]  # Loose match or exact
                # Prefer exact match from desired list if in cols
                if d_col in cols:
                    if d_col not in def_coll_vars:
                        def_coll_vars.append(d_col)
                elif matches:
                    if matches[0] not in def_coll_vars:
                        def_coll_vars.append(matches[0])

            # If none found from specific list, fallback to broader keywords
            if not def_coll_vars:
                def_coll_vars = [
                    c
                    for c in cols
                    if any(
                        k in c.lower() for k in ["age", "bmi", "lab", "score", "value"]
                    )
                ]
                # If still none, first 3
                if not def_coll_vars:
                    def_coll_vars = cols[:3] if len(cols) >= 3 else cols

            ui.update_selectize("coll_vars", choices=cols, selected=def_coll_vars)

            # -- Diagnostics Defaults --
            # Default: Lab_Glucose (Y), BMI_kgm2 (X), Age_Years, Treatment_Group (Covar)
            def_diag_out = select_variable_by_keyword(
                cols, ["lab_glucose", "glucose", "lab", "y"], default_to_first=True
            )
            def_diag_pred = select_variable_by_keyword(
                cols, ["bmi_kgm2", "bmi", "x"], default_to_first=True
            )

            def_diag_covar = []
            desired_diag_cov = ["Age_Years", "Treatment_Group"]
            for d_c in desired_diag_cov:
                if d_c in cols:
                    def_diag_covar.append(d_c)

            ui.update_select("diag_outcome", choices=cols, selected=def_diag_out)
            ui.update_select("diag_predictor", choices=cols, selected=def_diag_pred)
            ui.update_selectize(
                "diag_covariates", choices=cols, selected=def_diag_covar
            )

    @reactive.Effect
    @reactive.event(input.btn_run_mediation)
    def _run_mediation():
        if current_df() is None:
            ui.notification_show("Load data first", type="error")
            return

        try:
            med_is_running.set(True)
            mediation_results.set(None)

            # Check for MI datasets
            mi_active = _has_mi_datasets()
            mi_dfs = _get_mi_datasets() if mi_active else []

            if mi_active and len(mi_dfs) > 0:
                # ====== MI POOLED MEDIATION ANALYSIS ======
                ui.notification_show(
                    f"Running MI Pooled Mediation ({len(mi_dfs)} datasets)...",
                    duration=None,
                    id="run_med",
                )

                all_results = []
                for _mi_idx, mi_df in enumerate(mi_dfs):
                    try:
                        res_i = analyze_mediation(
                            data=mi_df,
                            outcome=input.med_outcome(),
                            treatment=input.med_treatment(),
                            mediator=input.med_mediator(),
                            confounders=(
                                list(input.med_confounders())
                                if input.med_confounders()
                                else None
                            ),
                            var_meta=var_meta.get() or {},
                        )
                        if "error" not in res_i:
                            all_results.append(res_i)
                    except Exception as exc:
                        logger.warning(
                            "MI mediation failed for imputed dataset %d/%d: %s",
                            _mi_idx + 1,
                            len(mi_dfs),
                            exc,
                            exc_info=True,
                        )

                if not all_results:
                    mediation_results.set({"error": "All MI mediation analyses failed"})
                    ui.notification_remove("run_med")
                    med_is_running.set(False)
                    return

                if len(all_results) != len(mi_dfs):
                    mediation_results.set(
                        {
                            "error": (
                                f"MI pooling requires all imputations to succeed "
                                f"({len(all_results)}/{len(mi_dfs)} succeeded)."
                            )
                        }
                    )
                    ui.notification_remove("run_med")
                    med_is_running.set(False)
                    return

                # Pool ACME and ADE using Rubin's rules
                pooled = {"mi_pooled": True, "m": len(mi_dfs)}
                failed_pool_keys = []

                for key in ["acme", "ade", "total_effect", "proportion_mediated"]:
                    estimates = []
                    variances = []
                    for res in all_results:
                        if key in res and res[key] is not None:
                            est = (
                                res[key].get("estimate", res[key])
                                if isinstance(res[key], dict)
                                else res[key]
                            )
                            se = (
                                res[key].get("se", 0.1)
                                if isinstance(res[key], dict)
                                else 0.1
                            )
                            if np.isfinite(est) and se > 0:
                                estimates.append(est)
                                variances.append(se**2)

                    if len(estimates) == len(mi_dfs):
                        try:
                            p = pool_estimates(
                                estimates, variances, n_obs=len(current_df())
                            )
                            # Map to keys expected by table
                            if key == "acme":
                                pooled["indirect_effect"] = p.estimate
                            elif key == "ade":
                                pooled["direct_effect"] = p.estimate
                            elif key == "total_effect":
                                pooled["total_effect"] = p.estimate
                            elif key == "proportion_mediated":
                                pooled["proportion_mediated"] = p.estimate

                            # Store full stats
                            pooled[f"{key}_stats"] = {
                                "estimate": p.estimate,
                                "se": p.se,
                                "ci_lower": p.ci_lower,
                                "ci_upper": p.ci_upper,
                                "p_value": p.p_value,
                                "fmi": p.fmi,
                            }
                        except Exception:
                            failed_pool_keys.append(key)
                            logger.exception("MI pooling failed for key '%s'", key)

                if failed_pool_keys:
                    mediation_results.set(
                        {
                            "error": (
                                "MI pooling failed for: "
                                + ", ".join(failed_pool_keys)
                                + ". Please review imputation/model stability."
                            )
                        }
                    )
                    ui.notification_remove("run_med")
                    med_is_running.set(False)
                    return

                mediation_results.set(pooled)
                ui.notification_remove("run_med")
                ui.notification_show("MI Pooled Mediation Complete", type="message")

            else:
                # ====== STANDARD MEDIATION ANALYSIS ======
                ui.notification_show(
                    "Running Mediation Analysis...", duration=None, id="run_med"
                )

                results = analyze_mediation(
                    data=current_df(),
                    outcome=input.med_outcome(),
                    treatment=input.med_treatment(),
                    mediator=input.med_mediator(),
                    confounders=(
                        list(input.med_confounders())
                        if input.med_confounders()
                        else None
                    ),
                    var_meta=var_meta.get() or {},
                )
                mediation_results.set(results)
                ui.notification_remove("run_med")
                ui.notification_show("Mediation Analysis Complete", type="message")

        except Exception as e:
            ui.notification_remove("run_med")
            ui.notification_show("Analysis failed", type="error")
            mediation_results.set({"error": f"Error: {str(e)}"})
        finally:
            med_is_running.set(False)

    @render.ui
    def out_mediation_container():
        if med_is_running.get():
            return create_loading_state("Running Mediation Analysis...")

        res = mediation_results.get()
        if res is None:
            return create_placeholder_state(
                "Select variables and run mediation analysis.", icon="🎯"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.div(
                ui.output_data_frame("tbl_mediation"),
                # Missing Data Report
                ui.HTML(
                    create_missing_data_report_html(
                        res.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
            ),
            class_="fade-in-entry",
        )

    @render.ui
    def out_mediation_status():
        res = mediation_results.get()
        if res and "error" not in res:
            return ui.div("✅ Analysis Complete", class_="alert alert-success mt-2")
        return None

    @render.data_frame
    def tbl_mediation():
        res = mediation_results.get()
        if res and "error" not in res:
            return pd.DataFrame(
                {
                    "Effect": [
                        "Total Effect",
                        "Direct Effect (ADE)",
                        "Indirect Effect (ACME)",
                        "Proportion Mediated",
                    ],
                    "Estimate": [
                        f"{res['total_effect']:.4f}",
                        f"{res['direct_effect']:.4f}",
                        f"{res['indirect_effect']:.4f}",
                        f"{res['proportion_mediated']:.2%}",
                    ],
                }
            )
        return None

    @reactive.Effect
    @reactive.event(input.btn_run_collinearity)
    def _run_collinearity():
        if current_df() is None:
            ui.notification_show("Load data first", type="error")
            return

        predictors = list(input.coll_vars())
        if not predictors:
            ui.notification_show("Select predictors first", type="error")
            return

        try:
            coll_is_running.set(True)
            vif_results.set(None)
            ui.notification_show("Calculating VIF...", duration=None, id="run_vif")

            # Pass var_meta to calculate_vif
            vif_df, missing_info = calculate_vif(
                current_df(), predictors, var_meta=var_meta.get()
            )
            vif_results.set({"vif_df": vif_df, "missing_info": missing_info})

            ui.notification_remove("run_vif")
        except Exception as e:
            ui.notification_remove("run_vif")
            ui.notification_show("Analysis failed", type="error")
            vif_results.set({"error": f"Error: {str(e)}"})
        finally:
            coll_is_running.set(False)

    @render.ui
    def out_vif_container():
        if coll_is_running.get():
            return create_loading_state("Calculating Collinearity Diagnostics...")

        res = vif_results.get()
        if res is None:
            return create_placeholder_state(
                "Select predictors to check for multicollinearity.", icon="🔬"
            )

        if isinstance(res, dict) and "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.div(
                ui.output_data_frame("tbl_vif"),
                # Missing Data Report
                ui.HTML(
                    create_missing_data_report_html(
                        res.get("missing_info", {}), var_meta.get() or {}
                    )
                ),
                ui.h5("Correlation Heatmap", class_="mt-4"),
                ui.output_ui("plot_corr_heatmap"),
            ),
            class_="fade-in-entry",
        )

    @render.data_frame
    def tbl_vif():
        res = vif_results.get()
        if res and "error" not in res:
            return res.get("vif_df")
        return None

    @render.ui
    def plot_corr_heatmap():
        """Render correlation heatmap for selected predictors using Plotly."""
        if current_df() is None:
            return None
        predictors = list(input.coll_vars())
        if not predictors or len(predictors) < 2:
            return None

        # Calculate correlation matrix
        d = current_df()[predictors].select_dtypes(include=[np.number]).dropna()
        if d.empty:
            return None

        corr = d.corr()

        # Create interactive heatmap with Plotly
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(title="Correlation Heatmap")

        html_str = plotly_figure_to_html(fig, include_plotlyjs="cdn", responsive=True)
        return ui.HTML(html_str)

    # --- Model Diagnostics Logic ---
    # (Reactive values moved to top init block)

    @reactive.Effect
    @reactive.event(input.btn_run_diag)
    def _run_diagnostics():
        if current_df() is None:
            ui.notification_show("Load data first", type="error")
            return

        try:
            diag_is_running.set(True)
            diag_results.set(None)
            ui.notification_show("Running Diagnostics...", duration=None, id="run_diag")

            y_col = input.diag_outcome()
            x_col = input.diag_predictor()
            covars = list(input.diag_covariates()) if input.diag_covariates() else []

            # Use centralized data preparation
            required_cols = [y_col, x_col] + covars

            # 1. Clean Data (Handle numeric conversion & missing values)
            df_clean, missing_info = prepare_data_for_analysis(
                current_df(),
                required_cols=required_cols,
                numeric_cols=required_cols,  # All variables in specific OLS diagnostics should be numeric
                handle_missing="complete_case",
                var_meta=var_meta.get() or {},
            )

            # Fit OLS for diagnostics
            X = df_clean[[x_col] + covars]
            X = sm.add_constant(X)
            Y = df_clean[y_col]

            model = sm.OLS(Y, X).fit()

            # Run tests
            res_reset = run_reset_test(model)
            res_bp = run_heteroscedasticity_test(model)
            res_cooks = calculate_cooks_distance(model)
            res_plots = get_diagnostic_plot_data(model)

            diag_results.set([res_reset, res_bp])
            diag_plot_data.set(res_plots)
            cooks_results.set(res_cooks)
            diag_clean_data.set(df_clean)

            ui.notification_remove("run_diag")
            ui.notification_show("Diagnostics Complete", type="message")

        except Exception as e:
            ui.notification_remove("run_diag")
            ui.notification_show("Analysis failed", type="error")
            diag_results.set({"error": f"Error: {str(e)}"})
        finally:
            diag_is_running.set(False)

    @render.ui
    def out_diag_container():
        if diag_is_running.get():
            return create_loading_state("Running Model Diagnostics...")

        res = diag_results.get()
        if res is None:
            return create_placeholder_state(
                "Define model parameters and run diagnostics.", icon="📊"
            )

        if isinstance(res, dict) and "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.navset_tab(
                ui.nav_panel("📋 Tests", ui.output_data_frame("tbl_diagnostics")),
                ui.nav_panel(
                    "📉 Plots",
                    ui.layout_columns(
                        ui.output_plot("plot_residuals"),
                        ui.output_plot("plot_qq"),
                    ),
                ),
                ui.nav_panel(
                    "⚠️ Influence",
                    ui.output_data_frame("tbl_cooks"),
                    ui.output_text_verbatim("txt_cooks_summary"),
                ),
            ),
            class_="fade-in-entry",
        )

    @render.data_frame
    def tbl_diagnostics():
        res = diag_results.get()
        if res and not (isinstance(res, dict) and "error" in res):
            df = pd.DataFrame(res)
            # Format P-Values if present
            for col in df.columns:
                if "p-value" in col.lower():
                    df[col] = df[col].apply(format_p_value)
            return df
        return None

    @render.plot
    def plot_residuals():
        data = diag_plot_data.get()
        if not data:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=data["fitted_values"], y=data["residuals"], ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        return fig

    @render.plot
    def plot_qq():
        data = diag_plot_data.get()
        if not data:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        # Use standardized residuals for Q-Q plot
        sm.qqplot(np.array(data["std_residuals"]), line="45", fit=True, ax=ax)
        ax.set_title("Q-Q Plot")
        return fig

    @render.data_frame
    def tbl_cooks():
        res = cooks_results.get()
        if not res or not res["influential_points"]:
            return None

        # Show top 10 influential points by Cook's D
        influential_indices = res["influential_points"]
        cooks_d = res["cooks_d"]
        cooks_values = [cooks_d[i] for i in influential_indices]
        # Sort by Cook's D descending and take top 10
        sorted_order = np.argsort(cooks_values)[::-1][:10].tolist()
        top_indices = [influential_indices[i] for i in sorted_order]
        # Use the exactly aligned cleaned data frame from the diagnostics run
        d_clean = diag_clean_data.get()
        if d_clean is None:
            return None
        return d_clean.iloc[top_indices]

    @render.text
    def txt_cooks_summary():
        res = cooks_results.get()
        if not res:
            return ""
        if res["n_influential"] == 0:
            return "No influential points detected (Cook's D < 4/n)"
        return f"Detected {res['n_influential']} influential points (Cook's D > {res['threshold']:.4f})"

    # --- Heterogeneity Logic --- (het_results in init)

    @reactive.Effect
    @reactive.event(input.btn_run_het)
    def _run_heterogeneity():
        try:
            het_is_running.set(True)
            het_results.set(None)
            ui.notification_show(
                "Calculating Heterogeneity...", duration=None, id="run_het"
            )

            eff_str = input.het_effects()
            var_str = input.het_variances()

            if not eff_str or not var_str:
                ui.notification_show("Enter effects and variances", type="error")
                ui.notification_remove("run_het")
                return

            effects = [float(x.strip()) for x in eff_str.split(",") if x.strip()]
            variances = [float(x.strip()) for x in var_str.split(",") if x.strip()]

            if len(effects) != len(variances):
                ui.notification_show("Mismatched lengths", type="error")
                ui.notification_remove("run_het")
                return

            if any(v <= 0 for v in variances):
                ui.notification_show("Variances must be positive", type="error")
                ui.notification_remove("run_het")
                return

            res = calculate_heterogeneity(effects, variances)
            het_results.set(res)
            ui.notification_remove("run_het")

        except ValueError:
            ui.notification_remove("run_het")
            ui.notification_show("Invalid numeric input", type="error")
            het_results.set({"error": "Invalid numeric input"})
        except Exception as e:
            ui.notification_remove("run_het")
            ui.notification_show("Analysis failed", type="error")
            het_results.set({"error": f"Error: {str(e)}"})
        finally:
            het_is_running.set(False)

    @render.ui
    def out_het_container():
        if het_is_running.get():
            return create_loading_state("Calculating Heterogeneity Statistics...")

        res = het_results.get()
        if res is None:
            return create_placeholder_state(
                "Enter effect sizes and variances to test for heterogeneity.", icon="🏥"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        return ui.div(ui.output_data_frame("tbl_heterogeneity"), class_="fade-in-entry")

    @render.data_frame
    def tbl_heterogeneity():
        res = het_results.get()
        if res and "error" not in res:
            # Format single row DF
            df = pd.DataFrame([res])
            for col in df.columns:
                if "p-value" in col.lower():
                    df[col] = df[col].apply(format_p_value)
            return df
        return None

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_mediation_validation():
        outcome = input.med_outcome()
        treatment = input.med_treatment()
        mediator = input.med_mediator()

        alerts = []
        if not outcome or not treatment or not mediator:
            return None

        if len({outcome, treatment, mediator}) < 3:
            alerts.append(
                create_error_alert(
                    "Outcome, Treatment, and Mediator must be different variables.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_coll_validation():
        preds = input.coll_vars()
        alerts = []

        if not preds:
            return None

        if len(preds) < 2:
            alerts.append(
                create_error_alert(
                    "Please select at least 2 predictors for collinearity analysis.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_diag_validation():
        outcome = input.diag_outcome()
        pred = input.diag_predictor()

        alerts = []
        if not outcome or not pred:
            return None

        if outcome == pred:
            alerts.append(
                create_error_alert(
                    "Outcome and Predictor must be different variables.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_het_validation():
        eff_str = input.het_effects()
        var_str = input.het_variances()

        alerts = []
        if not eff_str or not var_str:
            return None

        try:
            effects = [float(x.strip()) for x in eff_str.split(",") if x.strip()]
            variances = [float(x.strip()) for x in var_str.split(",") if x.strip()]

            if len(effects) != len(variances):
                alerts.append(
                    create_error_alert(
                        f"Mismatch: Found {len(effects)} effects and {len(variances)} variances.",
                        title="Data Error",
                    )
                )
        except ValueError:
            alerts.append(
                create_error_alert(
                    "Please enter valid comma-separated numbers.", title="Format Error"
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None
