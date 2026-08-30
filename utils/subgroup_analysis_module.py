"""
🧪 Subgroup Analysis Module (Shiny Compatible)

Professional subgroup analysis without Streamlit dependencies.
OPTIMIZED for Python 3.12 with strict type hints and TypedDict.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import prepare_data_for_analysis
from utils.forest_plot_lib import create_forest_plot
from utils.formatting import format_p_value

logger = get_logger(__name__)
COLORS = get_color_palette()


class SubgroupResult(TypedDict):
    group: str
    n: int
    events: int
    or_val: float
    ci_low: float
    ci_high: float
    p_value: float
    type: str
    subgroup: str | None


class SubgroupStats(TypedDict):
    n_overall: int
    events_overall: int
    n_subgroups: int
    or_overall: float
    ci_overall: tuple[float, float]
    p_overall: float
    p_interaction: float
    heterogeneous: bool
    or_range: tuple[float, float]


class SubgroupAnalysisLogit:
    """
    Subgroup analysis for logistic regression.
    Shiny-compatible, no Streamlit dependencies.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with dataset copy."""
        self.df = df.copy()
        self.results: pd.DataFrame | None = None
        self.stats: dict[str, Any] | None = None
        self.interaction_result: dict[str, Any] | None = None
        self.figure: go.Figure | None = None
        logger.info(f"SubgroupAnalysisLogit initialized with {len(df)} observations")

    def validate_inputs(
        self,
        outcome_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list[str] | None = None,
    ) -> bool:
        """Validate input columns and data types."""
        required_cols = {outcome_col, treatment_col, subgroup_col}
        if adjustment_cols:
            required_cols.update(adjustment_cols)

        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if self.df[outcome_col].nunique() != 2:
            raise ValueError(f"Outcome '{outcome_col}' must be binary")

        if self.df[subgroup_col].nunique() < 2:
            raise ValueError(f"Subgroup '{subgroup_col}' must have 2+ categories")

        if len(self.df) < 10:
            raise ValueError("Minimum 10 observations required")

        logger.info("Input validation passed")
        return True

    def analyze(
        self,
        outcome_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list[str] | None = None,
        min_subgroup_n: int = 5,
        var_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform logistic regression subgroup analysis."""
        try:
            from scipy import stats
            from statsmodels.formula.api import logit

            self.validate_inputs(
                outcome_col, treatment_col, subgroup_col, adjustment_cols
            )

            if adjustment_cols is None:
                adjustment_cols = []

            cols_to_use = [outcome_col, treatment_col, subgroup_col] + adjustment_cols
            # Deduplicate to prevent dimensionality errors
            cols_to_use = list(dict.fromkeys(cols_to_use))

            # --- MISSING DATA HANDLING ---
            missing_cfg = CONFIG.get("analysis.missing", {}) or {}
            strategy = missing_cfg.get("strategy", "complete-case")
            missing_codes = missing_cfg.get("user_defined_values", [])

            try:
                # Identify numeric columns for cleaning
                numeric_cols = [
                    c for c in cols_to_use if pd.api.types.is_numeric_dtype(self.df[c])
                ]

                df_clean, missing_data_info = prepare_data_for_analysis(
                    self.df,
                    required_cols=cols_to_use,
                    numeric_cols=numeric_cols,
                    var_meta=var_meta,
                    missing_codes=missing_codes,
                    handle_missing=strategy,
                )
                missing_data_info["strategy"] = strategy
            except Exception as e:
                logger.error(f"Subgroup Data preparation failed: {e}")
                return {"error": f"Data preparation failed: {e}"}

            if df_clean.empty:
                logger.warning("No data after cleaning")
                return {"error": "No valid data after cleaning."}

            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data: {len(df_clean)} rows")

            df_clean = df_clean.dropna(subset=[outcome_col])

            formula_base = f"{outcome_col} ~ {treatment_col}"
            if adjustment_cols:
                formula_base += " + " + " + ".join(adjustment_cols)

            results_list: list[dict[str, Any]] = []

            # Overall model
            logger.info("Computing overall model...")
            model_overall = logit(formula_base, data=df_clean).fit(disp=0)

            # Robust parameter lookup for treatment
            # Statsmodels might rename treatment to treatment[T.1] etc.
            param_key = None
            if treatment_col in model_overall.params:
                param_key = treatment_col
            else:
                for k in model_overall.params.index:
                    if k.startswith(f"{treatment_col}["):
                        param_key = k
                        break

            if not param_key:
                logger.warning(
                    f"Treatment variable '{treatment_col}' dropped from model."
                )
                return {}

            or_overall = float(np.exp(model_overall.params[param_key]))
            ci_data = np.exp(model_overall.conf_int().loc[param_key])
            # Robust access to CI values
            ci_low = float(ci_data.iloc[0])
            ci_high = float(ci_data.iloc[1])
            p_overall = float(model_overall.pvalues[param_key])

            results_list.append(
                {
                    "group": f"Overall (N={len(df_clean)})",
                    "n": len(df_clean),
                    "events": int(df_clean[outcome_col].sum()),
                    "or": or_overall,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_overall,
                    "type": "overall",
                    "subgroup": None,
                }
            )

            logger.info(f"Overall: OR={or_overall:.3f}, P={p_overall:.4f}")

            # Subgroup models
            subgroups = sorted(df_clean[subgroup_col].dropna().unique())
            logger.info(f"Computing {len(subgroups)} subgroup models...")

            for subgroup_val in subgroups:
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]

                if len(df_sub) < min_subgroup_n:
                    logger.warning(
                        f"Subgroup {subgroup_val}: N={len(df_sub)} < {min_subgroup_n}, skipping"
                    )
                    continue

                if (
                    df_sub[treatment_col].nunique() < 2
                    or df_sub[outcome_col].nunique() < 2
                ):
                    logger.warning(f"Subgroup {subgroup_val}: No variation, skipping")
                    continue

                try:
                    model_sub = logit(formula_base, data=df_sub).fit(disp=0)

                    # Robust parameter lookup for treatment in subgroup
                    sub_param_key = None
                    if treatment_col in model_sub.params:
                        sub_param_key = treatment_col
                    else:
                        for k in model_sub.params.index:
                            if k.startswith(f"{treatment_col}["):
                                sub_param_key = k
                                break

                    if not sub_param_key:
                        continue

                    or_sub = float(np.exp(model_sub.params[sub_param_key]))
                    ci_sub_data = np.exp(model_sub.conf_int().loc[sub_param_key])
                    # Robust access
                    ci_low = (
                        float(ci_sub_data.iloc[0])
                        if hasattr(ci_sub_data, "iloc")
                        else float(ci_sub_data[0])
                    )
                    ci_high = (
                        float(ci_sub_data.iloc[1])
                        if hasattr(ci_sub_data, "iloc")
                        else float(ci_sub_data[1])
                    )
                    p_sub = float(model_sub.pvalues[sub_param_key])

                    results_list.append(
                        {
                            "group": f"{subgroup_col}={subgroup_val} (N={len(df_sub)})",
                            "subgroup": str(subgroup_val),
                            "n": len(df_sub),
                            "events": int(df_sub[outcome_col].sum()),
                            "or": or_sub,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": p_sub,
                            "type": "subgroup",
                        }
                    )

                    logger.info(
                        f"Subgroup {subgroup_val}: OR={or_sub:.3f}, P={p_sub:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Model failed for {subgroup_val}: {e}")
                    continue

            # Interaction test
            logger.info("Computing interaction test...")
            try:
                formula_reduced = f"{outcome_col} ~ {treatment_col} + C({subgroup_col})"
                formula_full = f"{outcome_col} ~ {treatment_col} * C({subgroup_col})"
                if adjustment_cols:
                    formula_reduced += " + " + " + ".join(adjustment_cols)
                    formula_full += " + " + " + ".join(adjustment_cols)

                model_reduced = logit(formula_reduced, data=df_clean).fit(disp=0)
                model_full = logit(formula_full, data=df_clean).fit(disp=0)

                lr_stat = -2 * (model_reduced.llf - model_full.llf)
                df_diff = model_full.df_model - model_reduced.df_model

                if df_diff > 0:
                    p_interaction = float(stats.chi2.sf(lr_stat, df_diff))
                else:
                    p_interaction = np.nan

                alpha = CONFIG.get("analysis.alpha", 0.05)
                is_sig = pd.notna(p_interaction) and p_interaction < alpha

                self.interaction_result = {
                    "p_value": p_interaction,
                    "significant": is_sig,
                }

                logger.info(f"Interaction P={p_interaction:.4f}")
            except Exception as e:
                logger.warning(f"Interaction test failed: {e}")
                self.interaction_result = {"p_value": np.nan, "significant": False}

            self.results = pd.DataFrame(results_list)
            self.stats = self._compute_summary_statistics()
            logger.info("Analysis complete")
            result = self._format_output()
            if missing_data_info:
                result["missing_data_info"] = missing_data_info
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            if "Missing columns" in str(e) or "not found" in str(e):
                raise
            return {"error": str(e)}

    def _compute_summary_statistics(self) -> dict[str, Any]:
        """Compute summary statistics."""
        if self.results is None or self.results.empty:
            return {}

        overall = self.results[self.results["type"] == "overall"].iloc[0]
        subgroups = self.results[self.results["type"] == "subgroup"]

        p_int = (
            self.interaction_result.get("p_value", np.nan)
            if self.interaction_result
            else np.nan
        )
        is_het = (
            self.interaction_result.get("significant", False)
            if self.interaction_result
            else False
        )

        return {
            "n_overall": int(overall["n"]),
            "events_overall": int(overall["events"]),
            "n_subgroups": len(subgroups),
            "or_overall": float(overall["or"]),
            "ci_overall": (float(overall["ci_low"]), float(overall["ci_high"])),
            "p_overall": float(overall["p_value"]),
            "p_interaction": p_int,
            "heterogeneous": is_het,
            "or_range": (
                (float(subgroups["or"].min()), float(subgroups["or"].max()))
                if not subgroups.empty
                else (0.0, 0.0)
            ),
        }

    def _format_output(self) -> dict[str, Any]:
        """Format results for output."""
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results["type"] == "overall"]
        if overall_rows.empty:
            return {}

        overall = overall_rows.iloc[0]
        p_int = (
            self.interaction_result["p_value"] if self.interaction_result else np.nan
        )
        p_int_val = float(p_int) if pd.notna(p_int) else None

        return {
            "overall": {
                "or": float(overall["or"]),
                "ci": (float(overall["ci_low"]), float(overall["ci_high"])),
                "p_value": float(overall["p_value"]),
                "n": int(overall["n"]),
                "events": int(overall["events"]),
            },
            "subgroups": self.results[self.results["type"] == "subgroup"].to_dict(
                "records"
            ),
            "interaction": {
                "p_value": p_int_val,
                "significant": (
                    bool(self.interaction_result["significant"])
                    if self.interaction_result
                    else False
                ),
            },
            "summary": self.stats,
            "results_df": self.results,
        }

    def create_forest_plot(
        self,
        title: str = "Subgroup Analysis: Logistic Regression",
        color: str | None = None,
    ) -> go.Figure:
        """Create forest plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")

        if color is None:
            color = COLORS["primary"]

        plot_data = self.results[["group", "or", "ci_low", "ci_high", "p_value"]].copy()
        plot_data.columns = ["variable", "or", "ci_low", "ci_high", "p_value"]

        p_int = (
            self.interaction_result.get("p_value", np.nan)
            if self.interaction_result
            else np.nan
        )
        is_het = (
            self.interaction_result.get("significant", False)
            if self.interaction_result
            else False
        )
        het_text = "Heterogeneous" if is_het else "Homogeneous"

        if pd.notna(p_int):
            # Fix: Plotly titles strip nested HTML styles, so we use plain text format
            p_text = format_p_value(p_int, use_style=False)
            title_final = f"{title}<br>P = {p_text} ({het_text})"
        else:
            title_final = title

        self.figure = create_forest_plot(
            data=plot_data,
            estimate_col="or",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            label_col="variable",
            pval_col="p_value",
            title=title_final,
            x_label="Odds Ratio (95% CI)",
            ref_line=1.0,
            color=color,
        )
        # Add annotation for P-interaction
        if pd.notna(p_int):
            self.figure.add_annotation(
                text=f"<b>P<sub>interaction</sub> = {format_p_value(p_int, use_style=False)}</b>",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.05,  # Top right, slightly above plot
                showarrow=False,
                font=dict(size=12, color="black"),
                align="right",
            )

        return self.figure


class SubgroupAnalysisCox:
    """
    Subgroup analysis for Cox proportional hazards regression.
    Shiny-compatible, no Streamlit dependencies.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with dataset copy."""
        self.df = df.copy()
        self.results: pd.DataFrame | None = None
        self.stats: dict[str, Any] | None = None
        self.interaction_result: dict[str, Any] | None = None
        self.figure: go.Figure | None = None
        logger.info(f"SubgroupAnalysisCox initialized with {len(df)} observations")

    def validate_inputs(
        self,
        duration_col: str,
        event_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list[str] | None = None,
    ) -> bool:
        """Validate input columns and data types."""
        required_cols = {duration_col, event_col, treatment_col, subgroup_col}
        if adjustment_cols:
            required_cols.update(adjustment_cols)

        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if self.df[event_col].nunique() != 2:
            raise ValueError(f"Event '{event_col}' must be binary (0/1)")

        if self.df[subgroup_col].nunique() < 2:
            raise ValueError(f"Subgroup '{subgroup_col}' must have 2+ categories")

        if len(self.df) < 10:
            raise ValueError("Minimum 10 observations required")

        logger.info("Input validation passed")
        return True

    def analyze(
        self,
        duration_col: str,
        event_col: str,
        treatment_col: str,
        subgroup_col: str,
        adjustment_cols: list[str] | None = None,
        min_subgroup_n: int = 5,
        min_events: int = 2,
        var_meta: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        """Perform Cox subgroup analysis."""
        try:
            from lifelines import CoxPHFitter

            self.validate_inputs(
                duration_col, event_col, treatment_col, subgroup_col, adjustment_cols
            )

            if adjustment_cols is None:
                adjustment_cols = []

            cols_to_use = [
                duration_col,
                event_col,
                treatment_col,
                subgroup_col,
            ] + adjustment_cols
            # Deduplicate to prevent dimensionality errors
            cols_to_use = list(dict.fromkeys(cols_to_use))

            # --- MISSING DATA HANDLING (Centralized) ---
            missing_cfg = CONFIG.get("analysis.missing", {}) or {}
            strategy = missing_cfg.get("strategy", "complete-case")
            missing_codes = missing_cfg.get("user_defined_values", [])

            # Identify numeric columns for cleaning
            # For Cox models, duration and event MUST be numeric.
            # Adjustment cols might be numeric or categorical.
            numeric_cols = [duration_col, event_col]
            if adjustment_cols:
                for c in adjustment_cols:
                    if pd.api.types.is_numeric_dtype(self.df[c]):
                        numeric_cols.append(c)

            try:
                # Use centralized preparation
                df_clean, missing_data_info = prepare_data_for_analysis(
                    self.df,
                    required_cols=cols_to_use,
                    numeric_cols=numeric_cols,
                    handle_missing=strategy,
                    var_meta=var_meta,
                    missing_codes=missing_codes,
                    return_info=True,
                )
            except Exception as e:
                logger.error(f"Data preparation failed: {e}")
                return None, None, f"Data preparation failed: {e}"

            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data: {len(df_clean)} rows")

            results_list: list[dict[str, Any]] = []

            # Overall Cox model
            logger.info("Computing overall Cox model...")
            try:
                cph_overall = CoxPHFitter()

                # Build formula
                adj_str = ""
                if adjustment_cols:
                    adj_str = " + " + " + ".join(adjustment_cols)
                formula_base = f"{treatment_col}{adj_str}"

                cph_overall.fit(
                    df_clean,
                    duration_col=duration_col,
                    event_col=event_col,
                    formula=formula_base,
                    show_progress=False,
                )

                hr_overall = float(np.exp(cph_overall.params_[treatment_col]))
                # conf_int_ is a DataFrame with cols like 'lower 0.95', 'upper 0.95'
                ci_row = np.exp(cph_overall.confidence_intervals_.loc[treatment_col])
                # FIX: Use positional indexing (.iloc) to avoid KeyError 0
                ci_low = float(ci_row.iloc[0])
                ci_high = float(ci_row.iloc[1])

                p_overall = float(cph_overall.summary.loc[treatment_col, "p"])

                results_list.append(
                    {
                        "group": f"Overall (N={len(df_clean)})",
                        "n": len(df_clean),
                        "events": int(df_clean[event_col].sum()),
                        "hr": hr_overall,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_overall,
                        "type": "overall",
                        "subgroup": None,
                    }
                )
                logger.info(f"Overall: HR={hr_overall:.3f}, P={p_overall:.4f}")
            except Exception as e:
                logger.error(f"Overall Cox model failed: {e}")
                return None, None, f"Overall Cox model failed: {e}"

            # Subgroup Cox models
            subgroups = sorted(df_clean[subgroup_col].dropna().unique())
            logger.info(f"Computing {len(subgroups)} subgroup Cox models...")

            for subgroup_val in subgroups:
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]

                if len(df_sub) < min_subgroup_n:
                    logger.warning(
                        f"Subgroup {subgroup_val}: N={len(df_sub)} < {min_subgroup_n}, skipping"
                    )
                    continue

                if df_sub[event_col].sum() < min_events:
                    logger.warning(
                        f"Subgroup {subgroup_val}: Events={df_sub[event_col].sum()} < {min_events}, skipping"
                    )
                    continue

                try:
                    cph_sub = CoxPHFitter()
                    cph_sub.fit(
                        df_sub,
                        duration_col=duration_col,
                        event_col=event_col,
                        formula=formula_base,
                        show_progress=False,
                    )

                    hr_sub = float(np.exp(cph_sub.params_[treatment_col]))
                    ci_sub_row = np.exp(
                        cph_sub.confidence_intervals_.loc[treatment_col]
                    )
                    # FIX: Use positional indexing (.iloc) to avoid KeyError 0
                    ci_low = float(ci_sub_row.iloc[0])
                    ci_high = float(ci_sub_row.iloc[1])

                    p_sub = float(cph_sub.summary.loc[treatment_col, "p"])

                    results_list.append(
                        {
                            "group": f"{subgroup_col}={subgroup_val} (N={len(df_sub)})",
                            "subgroup": str(subgroup_val),
                            "n": len(df_sub),
                            "events": int(df_sub[event_col].sum()),
                            "hr": hr_sub,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": p_sub,
                            "type": "subgroup",
                        }
                    )
                    logger.info(
                        f"Subgroup {subgroup_val}: HR={hr_sub:.3f}, P={p_sub:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Cox model failed for {subgroup_val}: {e}")
                    continue

            # Interaction test (using treatment*subgroup term)
            logger.info("Computing interaction test...")
            try:
                # Use formula if possible, or manually construct interaction
                # To be robust across lifelines versions, we might need manual dummy creation if formula isn't supported.
                # However, assuming standard environment with formula support.

                # Check for formula support (try-except)
                try:
                    cph_reduced = CoxPHFitter()
                    cph_full = CoxPHFitter()

                    # Construct formulas
                    # We need to handle adjustments
                    adj_str = (
                        " + " + " + ".join(adjustment_cols) if adjustment_cols else ""
                    )

                    f_reduced = f"{treatment_col} + C({subgroup_col}){adj_str}"
                    f_full = f"{treatment_col} * C({subgroup_col}){adj_str}"

                    # Run models
                    cph_reduced.fit(
                        df_clean,
                        duration_col=duration_col,
                        event_col=event_col,
                        formula=f_reduced,
                    )
                    cph_full.fit(
                        df_clean,
                        duration_col=duration_col,
                        event_col=event_col,
                        formula=f_full,
                    )

                    # LRT
                    ll_reduced = cph_reduced.log_likelihood_
                    ll_full = cph_full.log_likelihood_

                    lr_stat = 2 * (ll_full - ll_reduced)
                    # Degrees of freedom difference
                    # full params count - reduced params count
                    df_diff = len(cph_full.params_) - len(cph_reduced.params_)

                    from scipy import stats

                    if df_diff > 0:
                        p_interaction = float(stats.chi2.sf(lr_stat, df_diff))
                        alpha = CONFIG.get("analysis.alpha", 0.05)
                        is_sig = p_interaction < alpha
                    else:
                        p_interaction = np.nan
                        is_sig = False

                    self.interaction_result = {
                        "p_value": p_interaction,
                        "significant": is_sig,
                        "method": "Likelihood Ratio Test (Cox)",
                    }
                    logger.info(f"Cox Interaction P={p_interaction:.4f}")

                except TypeError:
                    # Fallback if formula not supported (older lifelines)
                    logger.warning(
                        "lifelines formula not supported, skipping interaction test"
                    )
                    self.interaction_result = {"p_value": np.nan, "significant": False}

            except Exception as e:
                logger.warning(f"Interaction test failed: {e}")
                self.interaction_result = {"p_value": np.nan, "significant": False}

            self.results = pd.DataFrame(results_list)
            self.stats = self._compute_summary_statistics()
            logger.info("Analysis complete")
            result = self._format_output()
            if missing_data_info:
                result["missing_data_info"] = missing_data_info
            return result, None, None

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None, None, str(e)

    def _compute_summary_statistics(self) -> dict[str, Any]:
        """Compute summary statistics."""
        if self.results is None or self.results.empty:
            return {}

        overall = self.results[self.results["type"] == "overall"].iloc[0]
        subgroups = self.results[self.results["type"] == "subgroup"]

        return {
            "n_overall": int(overall["n"]),
            "events_overall": int(overall["events"]),
            "n_subgroups": len(subgroups),
            "hr_overall": float(overall["hr"]),
            "ci_overall": (float(overall["ci_low"]), float(overall["ci_high"])),
            "p_overall": float(overall["p_value"]),
            "hr_range": (
                (float(subgroups["hr"].min()), float(subgroups["hr"].max()))
                if not subgroups.empty
                else (0.0, 0.0)
            ),
        }

    def _format_output(self) -> dict[str, Any]:
        """Format results for output."""
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results["type"] == "overall"]
        if overall_rows.empty:
            return {}

        overall = overall_rows.iloc[0]

        return {
            "overall": {
                "hr": float(overall["hr"]),
                "ci": (float(overall["ci_low"]), float(overall["ci_high"])),
                "p_value": float(overall["p_value"]),
                "n": int(overall["n"]),
                "events": int(overall["events"]),
            },
            "subgroups": self.results[self.results["type"] == "subgroup"].to_dict(
                "records"
            ),
            "interaction": self.interaction_result,
            "interaction_table": self.results.to_dict("records"),
            "summary": self.stats,
            "results_df": self.results,
        }

    def create_forest_plot(
        self,
        title: str = "Subgroup Analysis: Cox Regression",
        color: str | None = None,
    ) -> go.Figure:
        """Create forest plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")

        if color is None:
            color = COLORS["primary"]

        plot_data = self.results[["group", "hr", "ci_low", "ci_high", "p_value"]].copy()
        plot_data.columns = ["variable", "hr", "ci_low", "ci_high", "p_value"]

        self.figure = create_forest_plot(
            data=plot_data,
            estimate_col="hr",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            label_col="variable",
            pval_col="p_value",
            title=title,
            x_label="Hazard Ratio (95% CI)",
            ref_line=1.0,
            color=color,
        )
        # Add annotation for P-interaction
        p_int = (
            self.interaction_result.get("p_value", np.nan)
            if self.interaction_result
            else np.nan
        )

        if pd.notna(p_int):
            self.figure.add_annotation(
                text=f"<b>P<sub>interaction</sub> = {format_p_value(p_int, use_style=False)}</b>",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.05,
                showarrow=False,
                font=dict(size=12, color="black"),
                align="right",
            )

        return self.figure


# =============================================================================
# ICEMAN CREDIBILITY ASSESSMENT
# =============================================================================


def assess_iceman_credibility(
    subgroup_results: dict[str, Any],
    is_prespecified: bool = False,
    n_subgroups_tested: int = 1,
) -> dict[str, Any]:
    """
    Assess credibility of subgroup heterogeneity using ICEMAN criteria.

    ICEMAN (Instrument to assess the Credibility of Effect Modification Analyses)
    evaluates whether observed subgroup differences are likely to be real.

    Args:
        subgroup_results: Output from SubgroupAnalysisLogit.analyze() or SubgroupAnalysisCox.analyze()
        is_prespecified: Whether the subgroup analysis was pre-specified
        n_subgroups_tested: Total number of subgroup analyses performed

    Returns:
        Dictionary with credibility score and interpretation
    """
    criteria = []

    # Extract key values
    interaction = subgroup_results.get("interaction", {})
    p_interaction = interaction.get("p_value") if interaction else None
    subgroups = subgroup_results.get("subgroups", [])

    # 1. Was the subgroup analysis pre-specified?
    if is_prespecified:
        criteria.append(
            {
                "criterion": "Pre-specified",
                "score": 2,
                "status": "✅",
                "note": "Subgroup was pre-specified before data collection",
            }
        )
    else:
        criteria.append(
            {
                "criterion": "Pre-specified",
                "score": 0,
                "status": "❌",
                "note": "Exploratory (post-hoc) analysis",
            }
        )

    # 2. Is the interaction statistically significant?
    if p_interaction is not None and not np.isnan(p_interaction):
        if p_interaction < 0.05:
            criteria.append(
                {
                    "criterion": "Significant interaction",
                    "score": 2,
                    "status": "✅",
                    "note": f"P-interaction = {p_interaction:.4f}",
                }
            )
        elif p_interaction < 0.10:
            criteria.append(
                {
                    "criterion": "Significant interaction",
                    "score": 1,
                    "status": "🔶",
                    "note": f"P-interaction = {p_interaction:.4f} (trend)",
                }
            )
        else:
            criteria.append(
                {
                    "criterion": "Significant interaction",
                    "score": 0,
                    "status": "❌",
                    "note": f"P-interaction = {p_interaction:.4f} (not significant)",
                }
            )
    else:
        criteria.append(
            {
                "criterion": "Significant interaction",
                "score": 0,
                "status": "❓",
                "note": "Interaction test not performed",
            }
        )

    # 3. Direction consistency with a priori hypothesis?
    # This is typically a qualitative assessment, we give partial credit
    criteria.append(
        {
            "criterion": "A priori direction",
            "score": 1,
            "status": "🔶",
            "note": "Requires clinical review of effect direction",
        }
    )

    # 4. Effect within subgroups clinically meaningful?
    # Check if any subgroup shows strong effects
    has_strong_effect = False
    for sub in subgroups:
        # Fix: handle 0.0 correctly (don't use 'or' operator which treats 0.0 as False)
        val_or = sub.get("or")
        val_hr = sub.get("hr", 1.0)
        or_hr = val_or if val_or is not None else val_hr
        if or_hr and (or_hr < 0.5 or or_hr > 2.0):
            has_strong_effect = True
            break

    if has_strong_effect:
        criteria.append(
            {
                "criterion": "Clinically meaningful effect",
                "score": 2,
                "status": "✅",
                "note": "At least one subgroup shows large effect size",
            }
        )
    else:
        criteria.append(
            {
                "criterion": "Clinically meaningful effect",
                "score": 1,
                "status": "🔶",
                "note": "Effects are modest in magnitude",
            }
        )

    # 5. Limited number of subgroup analyses?
    if n_subgroups_tested <= 3:
        criteria.append(
            {
                "criterion": "Limited testing",
                "score": 2,
                "status": "✅",
                "note": f"Only {n_subgroups_tested} subgroup(s) tested",
            }
        )
    elif n_subgroups_tested <= 6:
        criteria.append(
            {
                "criterion": "Limited testing",
                "score": 1,
                "status": "🔶",
                "note": f"{n_subgroups_tested} subgroups tested - consider multiple testing",
            }
        )
    else:
        criteria.append(
            {
                "criterion": "Limited testing",
                "score": 0,
                "status": "❌",
                "note": f"{n_subgroups_tested} subgroups tested - high risk of false positive",
            }
        )

    # Calculate total score and credibility level
    total_score = sum(c["score"] for c in criteria)
    max_score = 10  # 5 criteria × 2 max points each

    if total_score >= 8:
        credibility = "High"
        interpretation = "Strong evidence for true subgroup effect"
        badge_class = "text-success"
    elif total_score >= 5:
        credibility = "Moderate"
        interpretation = "Some evidence, interpret with caution"
        badge_class = "text-warning"
    else:
        credibility = "Low"
        interpretation = "Likely spurious finding, not reliable"
        badge_class = "text-danger"

    # Bonferroni-adjusted alpha for multiple testing
    adjusted_alpha = 0.05 / max(n_subgroups_tested, 1)

    return {
        "criteria": criteria,
        "total_score": total_score,
        "max_score": max_score,
        "credibility": credibility,
        "interpretation": interpretation,
        "badge_class": badge_class,
        "adjusted_alpha": adjusted_alpha,
        "note": "Based on ICEMAN framework (Schandelmaier et al., 2020)",
    }


def format_iceman_html(iceman_result: dict[str, Any]) -> str:
    """
    Generate HTML display for ICEMAN credibility assessment.

    Args:
        iceman_result: Output from assess_iceman_credibility()

    Returns:
        HTML string for display
    """
    criteria_rows = []
    for c in iceman_result["criteria"]:
        criteria_rows.append(f"""
            <tr>
                <td>{c["status"]}</td>
                <td>{c["criterion"]}</td>
                <td style="text-align: center;">{c["score"]}/2</td>
                <td style="font-size: 0.85em; color: #666;">{c["note"]}</td>
            </tr>
        """)

    html = f"""
    <div class="iceman-assessment">
        <div class="alert alert-{"success" if iceman_result["credibility"] == "High" else "warning" if iceman_result["credibility"] == "Moderate" else "danger"} mb-3">
            <strong>ICEMAN Credibility: {iceman_result["credibility"]}</strong>
            ({iceman_result["total_score"]}/{iceman_result["max_score"]} points)<br>
            <em>{iceman_result["interpretation"]}</em>
        </div>

        <table class="table table-sm">
            <thead>
                <tr>
                    <th style="width: 40px;"></th>
                    <th>Criterion</th>
                    <th style="width: 60px;">Score</th>
                    <th>Note</th>
                </tr>
            </thead>
            <tbody>
                {"".join(criteria_rows)}
            </tbody>
        </table>

        <div class="alert alert-info" style="font-size: 0.85em;">
            <strong>Multiple Testing:</strong>
            Bonferroni-adjusted α = {iceman_result["adjusted_alpha"]:.4f}<br>
            <em>Use this threshold for significance when multiple subgroups are tested.</em>
        </div>

        <div class="text-muted" style="font-size: 0.8em;">
            <strong>Reference:</strong> Schandelmaier S, et al. (2020).
            Development of the ICEMAN tool. <em>BMJ</em> 370:m2508.
        </div>
    </div>
    """
    return html
