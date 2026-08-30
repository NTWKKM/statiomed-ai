import numpy as np
import pandas as pd
from shiny import module, reactive, render, ui

from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.psm_lib import (
    PropensityScoreDiagnostics,
    calculate_ps,
)
from utils.sensitivity_lib import calculate_e_value
from utils.stratified_lib import breslow_day, mantel_haenszel
from utils.ui_helpers import (
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
)

logger = get_logger(__name__)
COLORS = get_color_palette()


@module.ui
def causal_inference_ui():
    """
    Constructs the Causal Inference tab UI with panels for propensity score methods, stratified analysis, sensitivity analysis, balance diagnostics, and reference material.

    The returned container includes dataset info/selector, inputs and run controls for:
    - PSM & IPW: treatment/outcome/covariate selectors, run button, and IPW & balance results.
    - Stratified Analysis: treatment/outcome/stratum selectors, run button, and stratified results.
    - Sensitivity Analysis: observed estimate and CI inputs, E-value calculator, and results.
    - Balance Diagnostics: Love plot output.
    - Reference: explanatory markdown about methods.

    Returns:
        ui_div: A Shiny UI div assembling the causal inference interface and its panels.
    """
    return ui.div(
        ui.h3("🎯 Causal Inference"),
        # Dataset info and selector
        ui.output_ui("ui_matched_info_ci"),
        ui.output_ui("ui_dataset_selector_ci"),
        ui.br(),
        ui.navset_tab(
            ui.nav_panel(
                "⚖️ PSM & IPW",
                ui.layout_sidebar(
                    ui.sidebar(
                        create_input_group(
                            "Model Specification",
                            ui.input_select(
                                "psm_treatment",
                                "Treatment Variable (Binary)",
                                choices=[],
                            ),
                            ui.input_select(
                                "psm_outcome", "Outcome Variable", choices=[]
                            ),
                            ui.input_selectize(
                                "psm_covariates",
                                "Covariates for Matching",
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            type="required",
                        ),
                        ui.input_checkbox(
                            "psm_trim",
                            "Truncate Extreme Weights (1%/99%)",
                            value=False,
                        ),
                        ui.output_ui("out_psm_validation"),
                        ui.br(),
                        ui.input_action_button(
                            "btn_run_psm", "Run Analysis", class_="btn-primary w-100"
                        ),
                    ),
                    ui.div(
                        ui.div(
                            "💡 ",
                            ui.strong("Need matched dataset?"),
                            " Use ",
                            ui.strong("Clinical → Table 1 & Matching"),
                            " to create balanced paired data first.",
                            style="padding: 10px; margin-bottom: 20px; background-color: var(--bs-info-bg-subtle, #e0faff); border-left: 4px solid var(--bs-info); border-radius: 4px;",
                        ),
                        create_results_container(
                            "IPW & Balance Results", ui.output_ui("out_psm_container")
                        ),
                    ),
                ),
            ),
            ui.nav_panel(
                "📊 Stratified Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        create_input_group(
                            "Variables",
                            ui.input_select(
                                "strat_treatment",
                                "Treatment/Exposure (Binary)",
                                choices=[],
                            ),
                            ui.input_select(
                                "strat_outcome", "Outcome (Binary)", choices=[]
                            ),
                            ui.input_select(
                                "strat_stratum", "Stratification Variable", choices=[]
                            ),
                            type="required",
                        ),
                        ui.output_ui("out_strat_validation"),
                        ui.br(),
                        ui.input_action_button(
                            "btn_run_strat",
                            "Run Stratified Analysis",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "Stratified Results", ui.output_ui("out_strat_container")
                    ),
                ),
            ),
            ui.nav_panel(
                "🔍 Sensitivity Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        create_input_group(
                            "Parameters",
                            ui.input_numeric(
                                "sens_est",
                                "Observed Estimate (RR or OR)",
                                2.0,
                                step=0.1,
                            ),
                            ui.input_numeric(
                                "sens_lower", "Lower CI Limit", 1.5, step=0.1
                            ),
                            ui.input_numeric(
                                "sens_upper", "Upper CI Limit", 3.0, step=0.1
                            ),
                            type="required",
                        ),
                        ui.output_ui("out_sens_validation"),
                        ui.br(),
                        ui.input_action_button(
                            "btn_calc_evalue",
                            "Calculate E-Value",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "E-Value Results", ui.output_ui("out_eval_container")
                    ),
                ),
            ),
            ui.nav_panel(
                "⚖️ Balance Diagnostics",
                create_results_container(
                    "Love Plot (Covariate Balance)",
                    ui.output_ui("plot_love"),
                ),
                ui.br(),
                create_results_container(
                    "Common Support (Overlap)",
                    ui.layout_columns(
                        ui.output_ui("plot_overlap"),
                        ui.output_ui("out_overlap_info"),
                        col_widths=(8, 4),
                    ),
                ),
            ),
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                    ### Causal Inference Methods
                    
                    #### 1. Propensity Score Methods (IPW/IPTW)
                    *   **Goal**: Estimate the Average Treatment Effect (ATE) by adjusting for confounding.
                    *   **Mechanism**: Observations are weighted by the Inverse Probability of Treatment Weighting (IPTW).
                    *   **Diagnostics**: Check Standardized Mean Differences (SMD). Ideally, all SMD < 0.1 after weighting.
                    
                    #### 2. Stratified Analysis (Mantel-Haenszel)
                    *   **Goal**: adjust for confounding by a categorical variable (stratum).
                    *   **Mantel-Haenszel OR**: A weighted average of stratum-specific odds ratios.
                    *   **Homogeneity Test**: Checks if the effect of treatment is consistent across all strata.
                    
                    #### 3. Sensitivity Analysis (E-Value)
                    *   **Goal**: Assess robustness to unmeasured confounding.
                    *   **E-Value**: The strength of association an unmeasured confounder must have to explain away the observed effect. Larger E-values imply more robust results.
                 """),
            ),
        ),
    )


@module.server
def causal_inference_server(
    input, output, session, df, var_meta, df_matched, is_matched
):
    """Server logic for Causal Inference."""

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df():
        if is_matched.get() and input.radio_ci_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_matched_info_ci():
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
    def ui_dataset_selector_ci():
        """Render dataset selector radio buttons."""
        if is_matched.get():
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_ci_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({original_len:,} rows)",
                    "matched": f"✅ Matched ({matched_len:,} rows)",
                },
                selected="original",
                inline=True,
            )
        return None

    # --- Update Choices ---
    @reactive.Effect
    def update_inputs():
        d = current_df()
        if d is None:
            return
        cols = list(d.columns)
        num_cols = list(d.select_dtypes(include=np.number).columns)
        cat_cols = list(d.select_dtypes(include=["object", "category"]).columns)

        # Heuristic for binary cols (0/1 or similar small unique count)
        binary_cols = [c for c in cols if d[c].nunique() == 2]

        # Defaults for PSM
        def_psm_treat = select_variable_by_keyword(
            binary_cols,
            ["treatment_group", "treatment", "group", "exposure"],
            default_to_first=True,
        )
        def_psm_out = select_variable_by_keyword(
            cols,
            ["outcome_cured", "outcome", "cured", "death", "event"],
            default_to_first=True,
        )

        # Default Covariates for PSM
        def_psm_covs = []
        desired_covs = [
            "Age_Years",
            "Sex_Male",
            "BMI_kgm2",
            "Comorb_Diabetes",
            "Comorb_Hypertension",
        ]
        for dc in desired_covs:
            if dc in cols:
                def_psm_covs.append(dc)

        # Fallback if specific covariates not found
        if not def_psm_covs:
            def_psm_covs = [
                c for c in num_cols if c not in [def_psm_treat, def_psm_out, "ID"]
            ][:3]

        ui.update_select("psm_treatment", choices=binary_cols, selected=def_psm_treat)
        ui.update_select(
            "psm_outcome", choices=num_cols + binary_cols, selected=def_psm_out
        )
        ui.update_selectize(
            "psm_covariates", choices=num_cols + binary_cols, selected=def_psm_covs
        )

        # Defaults for Stratified
        def_strat_treat = select_variable_by_keyword(
            binary_cols,
            ["treatment_group", "treatment", "group", "exposure"],
            default_to_first=True,
        )
        # Try to find different binary col for outcome
        rem_binary = [c for c in binary_cols if c != def_strat_treat]
        def_strat_out = select_variable_by_keyword(
            rem_binary,
            ["outcome_cured", "outcome", "cured", "event", "response"],
            default_to_first=True,
        )
        # Find categorical stratum
        def_strat_stratum = select_variable_by_keyword(
            cat_cols + binary_cols,  # Allow binary as stratum
            [
                "sex_male",
                "sex",
                "gender",
                "comorb_diabetes",
                "diabetes",
                "strata",
                "category",
            ],
            default_to_first=True,
        )

        ui.update_select(
            "strat_treatment", choices=binary_cols, selected=def_strat_treat
        )
        ui.update_select("strat_outcome", choices=binary_cols, selected=def_strat_out)
        ui.update_select(
            "strat_stratum", choices=cat_cols + binary_cols, selected=def_strat_stratum
        )

    # --- PSM & IPW Logic ---
    psm_res = reactive.Value(None)
    balance_res = reactive.Value(None)
    balance_pre_res = reactive.Value(None)
    common_support_res = reactive.Value(None)
    ps_cache = reactive.Value(None)

    # Running States
    psm_is_running = reactive.Value(False)
    strat_is_running = reactive.Value(False)
    eval_is_running = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.btn_run_psm)
    def run_psm_analysis():
        d = current_df()
        if d is None:
            return

        try:
            psm_is_running.set(True)
            psm_res.set(None)
            balance_res.set(None)
            balance_pre_res.set(None)
            common_support_res.set(None)
            ps_cache.set(None)
            ui.notification_show("Running PSM/IPW...", duration=None, id="run_psm")

            treatment = input.psm_treatment()
            outcome = input.psm_outcome()
            covs = list(input.psm_covariates())

            if not treatment or not outcome or not covs:
                ui.notification_show("Please select all PSM variables.", type="warning")
                ui.notification_remove("run_psm")
                return

            # Calculate PS (Cleaning is now handled inside calculate_ps via prepare_data_for_analysis)
            ps, missing_info = calculate_ps(
                d, treatment, covs, var_meta=var_meta.get() or {}
            )

            if "error" in missing_info:
                raise ValueError(missing_info["error"])

            # Attach PS for further steps (we need d_clean for balance check)
            # Match PS to original data for stats
            d_clean = d.copy()
            d_clean["ps"] = ps
            d_clean = d_clean.dropna(subset=["ps"])  # Keep only those with scores

            # Cache the cleaned dataframe with PS for plotting
            ps_cache.set(d_clean)

            # 1. Assess Common Support
            cs_support = PropensityScoreDiagnostics.assess_common_support(
                d_clean["ps"], d_clean[treatment]
            )
            common_support_res.set(cs_support)

            # 2. Check Unadjusted Balance (Pre-weighting)
            bal_pre = PropensityScoreDiagnostics.calculate_smd(
                d_clean, treatment, covs, weights=None
            )
            balance_pre_res.set(bal_pre)

            # 3. Calculate Weights
            # FIX: Ensure Treatment is numeric (0/1) for weights and regression
            d_clean[treatment] = pd.to_numeric(d_clean[treatment], errors="coerce")
            d_clean[outcome] = pd.to_numeric(d_clean[outcome], errors="coerce")

            # Drop any rows that became NaN during conversion
            d_clean = d_clean.dropna(subset=[treatment, outcome, "ps"])

            # Ensure treatment is coded as 0/1 before weighting
            unique_vals = set(d_clean[treatment].dropna().unique())
            if unique_vals != {0, 1}:
                raise ValueError(
                    f"Treatment '{treatment}' must be coded as 0/1 before weighting (found: {sorted(unique_vals)})"
                )

            if d_clean.empty:
                raise ValueError(
                    "No valid numeric data for Treatment/Outcome after cleaning."
                )

            T = d_clean[treatment]
            ps_vals = d_clean["ps"]

            # FIX: Clip PS to prevent infinite weights
            # Clip PS
            ps_vals = np.clip(ps_vals, 1e-6, 1 - 1e-6)

            # Calculate raw weights
            weights = np.where(T == 1, 1 / ps_vals, 1 / (1 - ps_vals))

            # Optional: Truncate Weights
            if input.psm_trim():
                weights_series = PropensityScoreDiagnostics.truncate_weights(weights)
                weights_array = (
                    weights_series.values
                )  # Convert back to array for regression safety
            else:
                weights_array = weights

            # 4. IPW Analysis (re-run logic with truncated weights if needed)
            # Existing specific function calculate_ipw does internal calculation.
            # To respect truncation, we manually run weighted regression here or just use the weights?
            # calculate_ipw internally calculates weights. To support truncation we'd need to modify it or do it manually.
            # For now, let's recalculate IPW if weights are modified, or trust the weights we just built.

            # Since calculate_ipw creates its own weights, we'll use a manual WLS here if trimmed,
            # OR better: update calculate_ipw to accept weights? No, it takes data/cols.
            # For simplicity/robustness, we perform the WLS here if we want to meaningful output from trimmed weights.
            # BUT: calculate_ipw returns neat dict.
            # Let's rely on our manual WLS for the ATE if trimmed, otherwise standard.

            import statsmodels.api as sm

            # Recalculate ATE with weights
            try:
                # Use T as simple numeric array to avoid index alignment issues in add_constant
                X_vals = T.values
                X_wls = sm.add_constant(X_vals)

                # Use weights as array
                model_wls = sm.WLS(
                    d_clean[outcome].values, X_wls, weights=weights_array
                ).fit()

                # Robust parameter access
                params = model_wls.params

                # Check if we have at least 2 params (const, treatment)
                if len(params) < 2:
                    raise ValueError(
                        "Model failed to estimate treatment effect (treatment variable may be constant)."
                    )

                # Access by position (numpy array or series safe)
                if isinstance(params, (pd.Series, pd.DataFrame)):
                    ate = params.iloc[1]
                    se = model_wls.bse.iloc[1]
                    p_val = model_wls.pvalues.iloc[1]
                else:
                    # Numpy array
                    ate = params[1]
                    se = model_wls.bse[1]
                    p_val = model_wls.pvalues[1]

                ci = model_wls.conf_int()

                # Handle CI structure
                if isinstance(ci, pd.DataFrame):
                    ci_l = ci.iloc[1, 0]
                    ci_u = ci.iloc[1, 1]
                elif isinstance(ci, np.ndarray):
                    ci_l = ci[1, 0]
                    ci_u = ci[1, 1]
                else:
                    # Fallback - indicate CI extraction failed
                    ci_l, ci_u = np.nan, np.nan

                ipw_results = {
                    "ATE": float(ate),
                    "SE": float(se),
                    "p_value": float(p_val),
                    "CI_Lower": float(ci_l),
                    "CI_Upper": float(ci_u),
                    "missing_data_info": missing_info,
                }
                psm_res.set(ipw_results)
            except Exception as e_ipw:
                logger.error(f"Weighted regression failed: {repr(e_ipw)}")
                psm_res.set({"error": f"Regression failed: {str(e_ipw)}"})

            # 5. Check Adjusted Balance (Post-weighting)
            bal_post = PropensityScoreDiagnostics.calculate_smd(
                d_clean,
                treatment,
                covs,
                weights=pd.Series(weights_array, index=d_clean.index),
            )
            balance_res.set(bal_post)

            ui.notification_remove("run_psm")
            ui.notification_show("PSM/IPW Analysis Complete", type="message")

        except Exception as e:
            ui.notification_remove("run_psm")
            ui.notification_show("Analysis failed", type="error")
            psm_res.set({"error": f"Analysis Error: {str(e)}"})
        finally:
            psm_is_running.set(False)

    @render.ui
    def out_psm_container():
        if psm_is_running.get():
            return create_loading_state("Running PSM & IPW Analysis...")

        res = psm_res.get()
        if res is None:
            return create_placeholder_state(
                "Select variables and run analysis.", icon="⚖️"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        ipw_div = ui.div(
            ui.p(f"ATE Estimate: {res['ATE']:.4f}"),
            ui.p(f"95% CI: [{res['CI_Lower']:.4f}, {res['CI_Upper']:.4f}]"),
            ui.p(f"P-value: {res['p_value']:.4f}"),
        )

        from utils.formatting import create_missing_data_report_html

        return ui.div(
            ui.div(
                ui.h5("Inverse Probability of Treatment Weighting (IPTW)"),
                ipw_div,
                ui.hr(),
                ui.h5("Standardized Mean Differences (Balance Check)"),
                ui.output_data_frame("out_balance_table"),
                ui.hr(),
                # Missing Data Report
                ui.HTML(
                    create_missing_data_report_html(
                        res.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
            ),
            class_="fade-in-entry",
        )

    @render.data_frame
    def out_balance_table():
        return balance_res.get()

    @render.ui
    def plot_love():
        bal_post = balance_res.get()
        bal_pre = balance_pre_res.get()

        if bal_post is None or bal_pre is None:
            return ui.div("Run PSM analysis to see balance diagnostics.")

        fig = PropensityScoreDiagnostics.create_love_plot(bal_pre, bal_post)
        return ui.HTML(plotly_figure_to_html(fig))

    @render.ui
    def plot_overlap():
        d = current_df()
        if d is None or common_support_res.get() is None:
            return None

        # We need the dataframe with PS. We re-calculate or store it?
        # Ideally we'd store the processed DF. Re-calculating for plot is safer than large reactive state.
        # But we don't want to re-run logit.
        # Let's assume re-running pure visualization is fast enough,
        # OR better: The "Common Support" is about PS distribution.
        # We can implement a simple reactive that holds the d_clean + ps if needed.
        # For now, let's just use the fact if analysis is run, we can grab it?
        # No, simpler: check common_support_res for data?
        # No, assess_common_support returns dict.

        # To avoid complexity, let's just make the plot use the reactive result IF we stored the PS there.
        # Alternative: We re-calculate PS quickly.

        treatment = input.psm_treatment()
        covs = list(input.psm_covariates())
        if not treatment or not covs:
            return None

        # Use cached PS if available
        d_cached = ps_cache.get()
        if d_cached is not None:
            d_plot = d_cached.copy()
        else:
            # Fallback re-calc
            ps, missing = calculate_ps(
                d, treatment, covs, var_meta=var_meta.get() or {}
            )
            if "error" in missing or ps.empty:
                return ui.div("Unable to compute propensity scores for overlap plot.")
            d_plot = d.copy()
            d_plot["ps"] = ps
            # remove missing
            d_plot = d_plot.dropna(subset=["ps", treatment])

        fig = PropensityScoreDiagnostics.plot_ps_overlap(d_plot, treatment, "ps")
        return ui.HTML(plotly_figure_to_html(fig))

    @render.ui
    def out_overlap_info():
        res = common_support_res.get()
        if not res or "error" in res:
            return None

        rec_color = (
            "text-success" if res["recommendation"] == "Adequate" else "text-warning"
        )
        if "exclusion" in res["recommendation"].lower():
            rec_color = "text-danger"

        return ui.div(
            ui.h5("Common Support Assessment"),
            ui.p(
                ui.strong("Overlap Area: "), f"{res['overlap_percent']:.1f}% of units"
            ),
            ui.p(
                ui.strong("Recommendation: "),
                ui.span(res["recommendation"], class_=rec_color),
            ),
            ui.p(f"Units outside support: {res['excluded_count']}"),
            ui.hr(),
            ui.h6("Propensity Score Ranges:"),
            ui.tags.ul(
                ui.tags.li(
                    f"Treated: [{res['treated_range'][0]:.3f}, {res['treated_range'][1]:.3f}]"
                ),
                ui.tags.li(
                    f"Control: [{res['control_range'][0]:.3f}, {res['control_range'][1]:.3f}]"
                ),
                ui.tags.li(
                    f"Overlap: [{res['overlap_range'][0]:.3f}, {res['overlap_range'][1]:.3f}]"
                ),
            ),
            style="background-color: var(--bs-light); padding: 15px; border-radius: 8px;",
        )

    # --- Stratified Logic ---
    strat_res = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_run_strat)
    def run_stratified():
        d = current_df()
        if d is None:
            return
        try:
            strat_is_running.set(True)
            strat_res.set(None)
            ui.notification_show(
                "Running Stratified Analysis...", duration=None, id="run_strat"
            )

            mh = mantel_haenszel(
                d,
                input.strat_outcome(),
                input.strat_treatment(),
                input.strat_stratum(),
                var_meta=var_meta.get() or {},
            )
            bd = breslow_day(
                d,
                input.strat_outcome(),
                input.strat_treatment(),
                input.strat_stratum(),
                var_meta=var_meta.get() or {},
            )

            strat_res.set({"mh": mh, "bd": bd})
            ui.notification_remove("run_strat")
        except Exception as e:
            ui.notification_remove("run_strat")
            ui.notification_show("Analysis failed", type="error")
            strat_res.set({"error": f"Stratified Error: {str(e)}"})
        finally:
            strat_is_running.set(False)

    @render.ui
    def out_strat_container():
        if strat_is_running.get():
            return create_loading_state("Running Stratified Analysis...")

        res = strat_res.get()
        if not res:
            return create_placeholder_state(
                "Select variables and run stratified analysis.", icon="📊"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        # Prepare helper text
        mh = res["mh"]
        bd = res["bd"]

        mh_ui = (
            create_error_alert(mh["error"])
            if "error" in mh
            else ui.div(ui.h5(f"Mantel-Haenszel OR: {mh['MH_OR']:.4f}"))
        )

        bd_ui = None
        if "error" in bd:
            bd_ui = create_error_alert(bd["error"])
        else:
            bd_ui = ui.div(
                ui.p(f"Test: {bd.get('test')}"),
                ui.p(f"P-value: {bd.get('p_value'):.4f}"),
                ui.p(f"Conclusion: {bd.get('conclusion')}"),
            )

        from utils.formatting import create_missing_data_report_html

        return ui.div(
            ui.div(
                ui.h5("Mantel-Haenszel Odds Ratio"),
                mh_ui,
                ui.br(),
                ui.h5("Test for Homogeneity (Breslow-Day / Interaction)"),
                bd_ui,
                ui.br(),
                ui.h5("Stratum-Specific Estimates"),
                ui.output_data_frame("out_stratum_table"),
                ui.hr(),
                # Missing Data Report (using MH info as representative since both clean same way)
                ui.HTML(
                    create_missing_data_report_html(
                        mh.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
            ),
            class_="fade-in-entry",
        )

    @render.data_frame
    def out_stratum_table():
        res = strat_res.get()
        if not res or "error" in res or "error" in res["mh"]:
            return None
        return res["mh"]["Strata_Results"]

    # --- Sensitivity Logic ---
    eval_res = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_calc_evalue)
    def run_evalue():
        try:
            eval_is_running.set(True)
            eval_res.set(None)

            res = calculate_e_value(
                input.sens_est(), input.sens_lower(), input.sens_upper()
            )
            eval_res.set(res)
        except Exception as e:
            ui.notification_show("Analysis failed", type="error")
            eval_res.set({"error": f"Error: {str(e)}"})
        finally:
            eval_is_running.set(False)

    @render.ui
    def out_eval_container():
        if eval_is_running.get():
            return create_loading_state("Calculating E-Value...")

        res = eval_res.get()
        if not res:
            return create_placeholder_state(
                "Enter parameters and calculate E-Value.", icon="🔍"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.div(
                ui.h5("E-Value for Unmeasured Confounding"),
                ui.div(
                    ui.h5(f"E-Value for Estimate: {res['e_value_estimate']}"),
                    ui.p(f"E-Value for CI Limit: {res['e_value_ci_limit']}"),
                ),
                ui.br(),
                ui.markdown("""
                **Interpretation:** The E-value is the minimum strength of association that an unmeasured confounder would need to have with both the treatment and the outcome to fully explain away a specific treatment-outcome association, conditional on the measured covariates.
                """),
            ),
            class_="fade-in-entry",
        )

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_psm_validation():
        outcome = input.psm_outcome()
        treatment = input.psm_treatment()
        covs = input.psm_covariates()

        alerts = []
        if not outcome or not treatment:
            return None

        if outcome == treatment:
            alerts.append(
                create_error_alert(
                    "Outcome and Treatment variables must be different.",
                    title="Configuration Error",
                )
            )

        if covs and (outcome in covs or treatment in covs):
            alerts.append(
                create_error_alert(
                    "Covariates cannot include Outcome or Treatment variables.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_strat_validation():
        outcome = input.strat_outcome()
        treatment = input.strat_treatment()
        stratum = input.strat_stratum()

        alerts = []
        if not outcome or not treatment or not stratum:
            return None

        if len({outcome, treatment, stratum}) < 3:
            alerts.append(
                create_error_alert(
                    "Outcome, Treatment, and Stratum variables must be different.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_sens_validation():
        est = input.sens_est()
        lower = input.sens_lower()
        upper = input.sens_upper()

        alerts = []
        if est <= 0 or lower <= 0 or upper <= 0:
            alerts.append(
                create_error_alert(
                    "E-Value requires positive estimates/CI (RR or OR).",
                    title="Invalid Input",
                )
            )

        if not (lower <= est <= upper) and not (upper <= est <= lower):
            alerts.append(
                create_error_alert(
                    "Estimate usually falls between Lower and Upper CI limits.",
                    title="Check Values",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None
