"""
🧮 Poisson Regression Library for Count Data Analysis

Handles count outcomes (e.g., number of events, hospital visits)
Returns Incidence Rate Ratios (IRR) instead of Odds Ratios (OR)

✅ Now supports Interaction Terms Analysis
OPTIMIZED for Python 3.12 with strict type hints.
"""

from __future__ import annotations

import html
import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import (
    prepare_data_for_analysis,
)
from utils.formatting import (
    PublicationFormatter,
    create_missing_data_report_html,
    format_p_value,
)

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

# Initialize color palette for consistent styling
COLORS = get_color_palette()


def _build_irr_interpretation_section(
    irr_results: dict[str, Any],
    airr_results: dict[str, Any],
    colors: dict[str, str],
) -> str:
    """Build HTML section for IRR interpretations, avoiding nested f-strings."""
    # Check if we should display
    has_sig_irr = any(v.get("p_value", 1) < 0.05 for v in irr_results.values())
    has_sig_airr = any(v.get("p_value", 1) < 0.05 for v in airr_results.values())

    if not (irr_results or airr_results) or not (has_sig_irr or has_sig_airr):
        return ""

    # Build crude analysis list
    crude_items = []
    for k, v in irr_results.items():
        if v.get("p_value", 1) < 0.05 and v.get("interpretation"):
            crude_items.append(f"<li><b>{k}:</b> {v.get('interpretation', 'N/A')}</li>")
    crude_html = "".join(crude_items)

    # Build adjusted analysis section
    adjusted_html = ""
    if airr_results and has_sig_airr:
        adjusted_items = []
        for k, v in airr_results.items():
            if v.get("p_value", 1) < 0.05 and v.get("interpretation"):
                adjusted_items.append(
                    f"<li><b>{k}:</b> {v.get('interpretation', 'N/A')}</li>"
                )
        adjusted_list = "".join(adjusted_items)
        adjusted_html = f"""
                <div style='margin-top: 16px; padding-top: 12px; border-top: 1px solid {colors["border"]};'>
                    <b style='color: {colors["primary_dark"]}; font-size: 1.1em;'>Adjusted (Multivariate) Analysis:</b><br>
                    <ul style='margin-top: 8px; line-height: 1.8;'>
                        {adjusted_list}
                    </ul>
                </div>
                """

    return f"""
        <div class='table-container' style='margin-top: 20px;'>
            <div class='outcome-title'>💬 Rate Ratio Interpretations</div>
            <div class='summary-box' style='border-radius: 0 0 8px 8px;'>
                <b style='color: {colors["primary_dark"]}; font-size: 1.1em;'>Crude (Univariate) Analysis:</b><br>
                <ul style='margin-top: 8px; line-height: 1.8;'>
                    {crude_html}
                </ul>
                {adjusted_html}
            </div>
        </div>
        """


def run_poisson_regression(
    y: pd.Series,
    X: pd.DataFrame,
    offset: pd.Series | None = None,
    alpha: float = 0.05,
) -> tuple[
    pd.Series | None,
    pd.DataFrame | None,
    pd.Series | None,
    str,
    dict[str, float],
]:
    """
    Fit a Poisson generalized linear model for count outcomes.

    Parameters:
        y (pd.Series): Observed count outcome aligned with rows of X.
        X (pd.DataFrame): Predictor matrix; a constant column will be added if absent.
        offset (pd.Series | None): Additive offset on the linear predictor (e.g., log(exposure)); pass None to omit.

    Returns:
        tuple[pd.Series | None, pd.DataFrame | None, pd.Series | None, str, dict[str, float]]:
            params: Estimated model coefficients indexed by term, or None on failure.
            conf_int: Coefficient confidence intervals as a DataFrame, or None on failure.
            pvalues: Coefficient p-values indexed by term, or None on failure.
            status_msg: "OK" on success or an error message on failure.
            stats_dict: Fit statistics with keys including "deviance", "pearson_chi2", "aic", and "bic" (values may be NaN if unavailable).
    """
    stats_metrics = {
        "deviance": np.nan,
        "pearson_chi2": np.nan,
        "aic": np.nan,
        "bic": np.nan,
    }

    try:
        # Validate inputs are not empty
        if y is None or y.empty:
            return (None, None, None, "Error: Outcome data is empty", stats_metrics)
        if X is None or X.empty:
            # Poisson usually needs at least an intercept
            # But checking empty is good
            return (None, None, None, "Error: Predictor data is empty", stats_metrics)

        # Ensure numeric types
        try:
            y = pd.to_numeric(y, errors="raise")
        except ValueError:
            return (
                None,
                None,
                None,
                "Error: Outcome has non-numeric data",
                stats_metrics,
            )

        # Check if outcome counts are non-negative
        if (y < 0).any():
            return (
                None,
                None,
                None,
                "Error: Count outcome must be non-negative",
                stats_metrics,
            )

        # Add constant if not present
        X_with_const = sm.add_constant(X, has_constant="add")

        # Fit Poisson GLM
        if offset is not None:
            model = sm.GLM(y, X_with_const, family=sm.families.Poisson(), offset=offset)
        else:
            model = sm.GLM(y, X_with_const, family=sm.families.Poisson())

        result = model.fit(disp=0)  # Suppress convergence output

        # Calculate fit statistics
        try:
            stats_metrics = {
                "deviance": result.deviance,
                "pearson_chi2": result.pearson_chi2,
                "aic": result.aic,
                "bic": result.bic,
            }
        except (AttributeError, ZeroDivisionError) as e:
            logger.debug(f"Failed to calculate Poisson fit stats: {e}")

        return (
            result.params,
            result.conf_int(alpha=alpha),
            result.pvalues,
            "OK",
            stats_metrics,
        )

    except Exception as e:
        err_msg = str(e)
        if "Perfect separation" in err_msg or "separation" in err_msg.lower():
            logger.warning("Perfect separation detected in Poisson regression")
            return (
                None,
                None,
                None,
                "Perfect separation detected. Model cannot be reliably fitted.",
                stats_metrics,
            )

        logger.exception("Poisson regression failed")
        return None, None, None, f"Error: {err_msg}", stats_metrics


def run_negative_binomial_regression(
    y: pd.Series,
    X: pd.DataFrame,
    offset: pd.Series | None = None,
    alpha: float = 0.05,
) -> tuple[
    pd.Series | None,
    pd.DataFrame | None,
    pd.Series | None,
    str,
    dict[str, float],
]:
    """
    Fit Negative Binomial regression (alternative to Poisson for overdispersed data).

    Uses sm.NegativeBinomial (discrete model) to estimate the dispersion parameter (alpha) via MLE,
    instead of sm.GLM with fixed alpha.
    """
    # Initialize metrics with alpha
    stats_metrics = {
        "deviance": np.nan,
        "pearson_chi2": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "alpha": np.nan,
    }

    try:
        if y is None or y.empty:
            return (None, None, None, "Error: Outcome data is empty", stats_metrics)
        if X is None or X.empty:
            return (None, None, None, "Error: Predictor data is empty", stats_metrics)

        try:
            y = pd.to_numeric(y, errors="raise")
        except ValueError:
            return (
                None,
                None,
                None,
                "Error: Outcome has non-numeric data",
                stats_metrics,
            )

        X_const = sm.add_constant(X, has_constant="add")

        # Use discrete NegativeBinomial model to estimate alpha
        # Note: For discrete models, offset is passed to the constructor, not fit()
        if offset is not None:
            # sm.NegativeBinomial corresponds to sm.discrete.discrete_model.NegativeBinomial
            model = sm.NegativeBinomial(y, X_const, offset=offset)
        else:
            model = sm.NegativeBinomial(y, X_const)

        # Fit model (disp=0 suppresses convergence output)
        result = model.fit(disp=0)

        # Extract alpha (dispersion parameter)
        # In discrete NegativeBinomial (NB2), alpha is estimated and included in params.
        # It is typically named 'alpha' in the results or is the last parameter.
        alpha_val = np.nan
        if hasattr(result, "params"):
            if isinstance(result.params, pd.Series) and "alpha" in result.params.index:
                alpha_val = result.params["alpha"]
            else:
                # Fallback: alpha is typically the last parameter for NB2
                logger.debug(
                    "Alpha not found by name, using last parameter (index %d) as fallback",
                    len(result.params) - 1,
                )
                alpha_val = (
                    result.params.iloc[-1]
                    if isinstance(result.params, pd.Series)
                    else result.params[-1]
                )

        try:
            # Note: DiscreteResult (MLE) does not have 'deviance' or 'pearson_chi2' attributes
            # like GLMResults. We provide AIC, BIC, and Alpha.
            stats_metrics = {
                "deviance": np.nan,  # Not directly available in discrete MLE results
                "pearson_chi2": np.nan,
                "aic": result.aic,
                "bic": result.bic,
                "alpha": alpha_val,
            }
        except (AttributeError, ZeroDivisionError) as e:
            logger.debug(f"Failed to calculate NB fit stats: {e}")

        params = result.params
        conf = result.conf_int(alpha=alpha)
        pvals = result.pvalues

        # Remove alpha from coefficient outputs (keep only in stats_metrics)
        if isinstance(params, pd.Series) and "alpha" in params.index:
            params = params.drop(index=["alpha"])
            conf = conf.drop(index=["alpha"], errors="ignore")
            pvals = pvals.drop(index=["alpha"], errors="ignore")

        return params, conf, pvals, "OK", stats_metrics

    except Exception as e:
        logger.exception("Negative Binomial regression failed")
        return None, None, None, str(e), stats_metrics


def check_count_outcome(series: pd.Series) -> tuple[bool, str]:
    """
    Validate if series is suitable for Poisson regression.

    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Check if all values are non-negative integers
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()

        if len(numeric_series) == 0:
            return False, "No valid numeric values"

        # Check for negative values
        if (numeric_series < 0).any():
            return False, "Count data cannot contain negative values"

        # Check if mostly integers (allow some floating point for exposure-adjusted rates)
        non_integer_pct = (numeric_series != numeric_series.astype(int)).sum() / len(
            numeric_series
        )
        if non_integer_pct > 0.1:
            logger.warning("More than 10% non-integer values in count outcome")

        # Check for overdispersion indicators
        mean_val = numeric_series.mean()
        var_val = numeric_series.var()
        if var_val > mean_val * 2:
            logger.warning(
                f"Possible overdispersion detected (Var={var_val:.2f}, Mean={mean_val:.2f}). Consider Negative Binomial model."
            )

        return True, "Valid count data"

    except Exception as e:
        logger.exception("Count outcome validation error")
        return False, f"Validation error: {e!s}"


def interpret_irr(
    irr: float,
    ci_low: float,
    ci_high: float,
    var_name: str = "",
    mode: str = "linear",
) -> str:
    """
    Convert IRR (Incidence Rate Ratio) to human-readable interpretation.

    Args:
        irr: Incidence Rate Ratio value
        ci_low: Lower bound of 95% CI
        ci_high: Upper bound of 95% CI
        var_name: Variable name for context (optional)
        mode: "linear" for continuous or "categorical" for categories

    Returns:
        str: Human-readable interpretation in English

    Examples:
        >>> interpret_irr(1.25, 1.10, 1.42, "Age", "linear")
        'Increase 25% per unit (95% CI: 10%-42%)'

        >>> interpret_irr(0.80, 0.65, 0.95, "Treatment", "categorical")
        'Decrease 20% (95% CI: 35% decrease to 5% decrease)'
    """
    try:
        # Calculate percentage change from IRR
        pct_change = (irr - 1) * 100
        pct_ci_low = (ci_low - 1) * 100
        pct_ci_high = (ci_high - 1) * 100

        # Determine effect direction
        if abs(pct_change) < 0.5:  # Essentially no change
            direction_en = "No significant change"
            detail_en = ""
        elif pct_change > 0:  # Increase
            direction_en = "Increase"
            detail_en = f"{abs(pct_change):.1f}%"
        else:  # Decrease
            direction_en = "Decrease"
            detail_en = f"{abs(pct_change):.1f}%"

        # Build interpretation based on mode
        if mode == "linear":
            unit_en = " per unit"
        else:
            unit_en = ""

        # Format confidence interval
        if abs(pct_change) < 0.5:
            ci_en = f"(95% CI: IRR {ci_low:.2f}-{ci_high:.2f})"
        else:
            ci_en = f"(95% CI: {pct_ci_low:+.1f}% to {pct_ci_high:+.1f}%)"

        # Combine into final interpretation
        if abs(pct_change) < 0.5:
            interp_en = f"{direction_en} {ci_en}"
        else:
            interp_en = f"{direction_en} {detail_en}{unit_en} {ci_en}"

        return interp_en

    except Exception as e:
        logger.debug(f"Failed to interpret IRR: {e}")
        ci_str = PublicationFormatter.format_ci(ci_low, ci_high)
        return f"IRR = {irr:.2f} {ci_str}"


def analyze_poisson_outcome(
    outcome_name: str,
    df: pd.DataFrame,
    var_meta: dict[str, Any] | None = None,
    offset_col: str | None = None,
    interaction_pairs: list[tuple[str, str | None]] = None,
    model_type: str = "poisson",
) -> tuple[
    str,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """
    Perform Poisson regression analysis for count outcome.

    Args:
        outcome_name: Name of count outcome column
        df: Input DataFrame
        var_meta: Variable metadata dictionary
        offset_col: Optional column name for exposure offset (e.g., person-years)
        interaction_pairs: List of tuples for interactions [(var1, var2), ...]
        model_type: "poisson" or "negative_binomial"

    Returns:
        tuple: (html_table, irr_results, airr_results, interaction_results, model_metadata)
    """
    # ✅ FIX: Wrap entire function in try-except to prevent None return crashes
    try:
        # ✅ Consolidated fmt_p import: Use logic.py's centralized formatting
        from utils.logic import (
            _robust_sort_key,
            clean_numeric_value,
            get_label,
        )

        model_label = (
            "Negative Binomial" if model_type == "negative_binomial" else "Poisson"
        )
        logger.info("Starting %s analysis for outcome: %s", model_label, outcome_name)

        model_metadata = {
            "model_type": model_type,
            "model_label": model_label,
            "outcome": outcome_name,
            "offset": offset_col,
        }

        # --- MISSING DATA HANDLING ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        # We need to preserve all columns for the univariate analysis loop,
        # so we pass df.columns as required.
        # Outcome and offset must be numeric.
        numeric_cols = [outcome_name]
        if offset_col:
            numeric_cols.append(offset_col)

        # Use unified pipeline
        try:
            df_clean, missing_data_info = prepare_data_for_analysis(
                df,
                required_cols=df.columns.tolist(),
                numeric_cols=numeric_cols,
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return (
                f"<div class='alert'>⚠️ Data preparation failed: {e}</div>",
                {},
                {},
                {},
                model_metadata,
            )

        missing_data_info["strategy"] = strategy  # Ensure strategy name matches
        df = df_clean
        model_metadata["missing_info"] = missing_data_info

        logger.info(
            "Missing data: %s rows excluded", missing_data_info["rows_excluded"]
        )

        if outcome_name not in df.columns:
            msg = f"Outcome '{outcome_name}' not found"
            logger.error(msg)
            return f"<div class='alert'>{msg}</div>", {}, {}, {}, model_metadata

        # Validate count data
        is_valid, msg = check_count_outcome(df[outcome_name])
        if not is_valid:
            logger.error(f"Invalid count data: {msg}")
            return f"<div class='alert'>⚠️ {msg}</div>", {}, {}, {}, model_metadata

        y = pd.to_numeric(df[outcome_name], errors="coerce").dropna()
        if len(y) == 0:
            return (
                "<div class='alert'>⚠️ No valid data rows available.</div>",
                {},
                {},
                {},
                model_metadata,
            )

        df_aligned = df.loc[y.index]
        total_n = len(y)
        model_metadata["n_total"] = total_n

        # Handle offset if provided
        offset = None
        if offset_col and offset_col in df_aligned.columns:
            raw_offset = pd.to_numeric(df_aligned[offset_col], errors="coerce")
            valid_offset = raw_offset.dropna()
            if len(valid_offset) == 0 or (valid_offset <= 0).any():
                return (
                    f"<div class='alert'>⚠️ Offset '{html.escape(str(offset_col))}' must be > 0 for log-offset.</div>",
                    {},
                    {},
                    {},
                )
            offset = np.log(raw_offset)
            logger.info("Using offset column: %s", offset_col)

        candidates = []
        results_db = {}
        sorted_cols = sorted(df.columns.astype(str))
        mode_map = {}
        cat_levels_map = {}

        irr_results = {}

        # Univariate analysis
        logger.info(
            f"Starting univariate {model_label} analysis for {len(sorted_cols) - 1} variables"
        )

        for col in sorted_cols:
            if (
                col == outcome_name
                or col not in df_aligned.columns
                or df_aligned[col].isnull().all()
            ):
                continue
            if col == offset_col:  # Skip offset column
                continue

            res = {"var": col}
            X_raw = df_aligned[col]
            X_num = X_raw.apply(clean_numeric_value)

            unique_vals = X_num.dropna().unique()
            mode = "linear"

            # Auto-detect mode
            if set(unique_vals).issubset({0, 1}):
                mode = "categorical"
            elif len(unique_vals) < 10:
                decimals_pct = (
                    sum(1 for v in unique_vals if not float(v).is_integer())
                    / len(unique_vals)
                    if len(unique_vals) > 0
                    else 0
                )
                if decimals_pct < 0.3:
                    mode = "categorical"

            # Check var_meta
            if var_meta:
                orig_name = col.split("_", 1)[1] if "_" in col else col
                key = col if col in var_meta else orig_name
                if key in var_meta:
                    user_mode = var_meta[key].get("type", "").lower()
                    if "cat" in user_mode or "simp" in user_mode:
                        mode = "categorical"
                    elif "lin" in user_mode or "cont" in user_mode:
                        mode = "linear"

            mode_map[col] = mode

            if mode == "categorical":
                try:
                    levels = sorted(X_raw.dropna().unique(), key=_robust_sort_key)
                except (TypeError, ValueError):
                    levels = sorted(X_raw.astype(str).unique())
                cat_levels_map[col] = levels

                n_used = len(X_raw.dropna())
                mean_y = y.mean()
                desc_tot = [f"n={n_used}", f"Mean count: {mean_y:.2f}"]

                for lvl in levels:
                    try:
                        float_val = float(lvl)
                        lbl_txt = (
                            str(int(float_val)) if float_val.is_integer() else str(lvl)
                        )
                    except (ValueError, TypeError):
                        lbl_txt = str(lvl)
                    mask = X_raw.astype(str) == str(lvl)
                    count = mask.sum()
                    mean_count = y[mask].mean() if count > 0 else 0
                    desc_tot.append(f"{lbl_txt}: n={count}, mean={mean_count:.2f}")

                res["desc_total"] = "<br>".join(desc_tot)

                # Categorical test (Chi-square or Kruskal-Wallis)
                try:
                    groups = [
                        y[X_raw.astype(str) == str(lvl)].dropna() for lvl in levels
                    ]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) > 1:
                        _, p = stats.kruskal(*groups)
                        res["p_comp"] = p
                        res["test_name"] = "Kruskal-Wallis"
                    else:
                        res["p_comp"] = np.nan
                        res["test_name"] = "-"
                except (ValueError, TypeError):
                    res["p_comp"] = np.nan
                    res["test_name"] = "-"

                # Univariate Poisson with dummy variables
                if len(levels) > 1:
                    temp_df = pd.DataFrame({"y": y, "raw": X_raw}).dropna()
                    dummy_cols = []
                    for lvl in levels[1:]:
                        d_name = f"{col}::{lvl}"
                        temp_df[d_name] = (
                            temp_df["raw"].astype(str) == str(lvl)
                        ).astype(int)
                        dummy_cols.append(d_name)

                    if dummy_cols and temp_df[dummy_cols].std().sum() > 0:
                        offset_uni = (
                            offset.loc[temp_df.index] if offset is not None else None
                        )
                        offset_uni = (
                            offset.loc[temp_df.index] if offset is not None else None
                        )

                        if model_type == "negative_binomial":
                            params, conf, pvals, status, _ = (
                                run_negative_binomial_regression(
                                    temp_df["y"], temp_df[dummy_cols], offset=offset_uni
                                )
                            )
                        else:
                            params, conf, pvals, status, _ = run_poisson_regression(
                                temp_df["y"], temp_df[dummy_cols], offset=offset_uni
                            )

                        if status == "OK":
                            irr_lines, coef_lines, p_lines = ["Ref."], ["-"], ["-"]
                            for lvl in levels[1:]:
                                d_name = f"{col}::{lvl}"
                                if d_name in params:
                                    coef = params[d_name]
                                    irr = np.exp(coef)
                                    ci_l, ci_h = (
                                        np.exp(conf.loc[d_name][0]),
                                        np.exp(conf.loc[d_name][1]),
                                    )
                                    pv = pvals[d_name]

                                    coef_lines.append(f"{coef:.3f}")

                                    ci_str = PublicationFormatter.format_ci(ci_l, ci_h)
                                    irr_lines.append(f"{irr:.2f} {ci_str}")
                                    p_lines.append(format_p_value(pv))
                                    interpretation = interpret_irr(
                                        irr,
                                        ci_l,
                                        ci_h,
                                        f"{col}: {lvl}",
                                        mode="categorical",
                                    )
                                    irr_results[f"{col}: {lvl}"] = {
                                        "irr": irr,
                                        "ci_low": ci_l,
                                        "ci_high": ci_h,
                                        "p_value": pv,
                                        "interpretation": interpretation,
                                    }

                            res["irr"] = "<br>".join(irr_lines)
                            res["coef"] = "<br>".join(coef_lines)
                            res["p_irr"] = "<br>".join(p_lines)
                        else:
                            res["irr"] = "-"
                            res["coef"] = "-"
                    else:
                        res["irr"] = "-"
                        res["coef"] = "-"
                else:
                    res["irr"] = "-"
                    res["coef"] = "-"

            else:  # Linear mode
                n_used = len(X_num.dropna())
                m_t, s_t = X_num.mean(), X_num.std()
                res["desc_total"] = f"n={n_used}<br>Mean: {m_t:.2f} (SD {s_t:.2f})"

                # Correlation test
                try:
                    _, p = stats.spearmanr(X_num.dropna(), y.loc[X_num.dropna().index])
                    res["p_comp"] = p
                    res["test_name"] = "Spearman"
                except (ValueError, TypeError):
                    res["p_comp"] = np.nan
                    res["test_name"] = "-"

                # Univariate Poisson
                data_uni = pd.DataFrame({"y": y, "x": X_num}).dropna()
                if not data_uni.empty and data_uni["x"].nunique() > 1:
                    offset_uni = (
                        offset.loc[data_uni.index] if offset is not None else None
                    )
                    offset_uni = (
                        offset.loc[data_uni.index] if offset is not None else None
                    )

                    if model_type == "negative_binomial":
                        params, conf, pvals, status, _ = (
                            run_negative_binomial_regression(
                                data_uni["y"], data_uni[["x"]], offset=offset_uni
                            )
                        )
                    else:
                        params, conf, pvals, status, _ = run_poisson_regression(
                            data_uni["y"], data_uni[["x"]], offset=offset_uni
                        )

                    if status == "OK" and "x" in params:
                        coef = params["x"]
                        irr = np.exp(coef)
                        ci_l, ci_h = np.exp(conf.loc["x"][0]), np.exp(conf.loc["x"][1])
                        pv = pvals["x"]

                        res["coef"] = f"{coef:.3f}"
                        ci_str = PublicationFormatter.format_ci(ci_l, ci_h)
                        res["irr"] = f"{irr:.2f} {ci_str}"
                        res["p_irr"] = pv
                        interpretation = interpret_irr(
                            irr, ci_l, ci_h, col, mode="linear"
                        )
                        irr_results[col] = {
                            "irr": irr,
                            "ci_low": ci_l,
                            "ci_high": ci_h,
                            "p_value": pv,
                            "interpretation": interpretation,
                        }
                    else:
                        res["irr"] = "-"
                        res["coef"] = "-"
                else:
                    res["irr"] = "-"
                    res["coef"] = "-"

            results_db[col] = res

            # Screen for multivariate
            p_screen = res.get("p_comp", np.nan)
            if (
                isinstance(p_screen, (int, float))
                and pd.notna(p_screen)
                and p_screen < 0.20
            ):
                candidates.append(col)

        # ===================================================================
        # 🔗 MULTIVARIATE ANALYSIS WITH INTERACTIONS
        # ===================================================================
        airr_results = {}
        interaction_results = {}
        int_meta = {}
        final_n_multi = 0
        mv_metrics_text = ""

        def _is_candidate_valid(col: str) -> bool:
            mode = mode_map.get(col, "linear")
            series = df_aligned[col]
            if mode == "categorical":
                return series.notna().sum() > 5
            return series.apply(clean_numeric_value).notna().sum() > 5

        cand_valid = [c for c in candidates if _is_candidate_valid(c)]

        # ✅ FIX: Run multivariate analysis if there are candidates OR interaction pairs
        if len(cand_valid) > 0 or interaction_pairs:
            multi_df = pd.DataFrame({"y": y})

            # Track raw categorical columns to drop later
            raw_cat_cols_to_drop = []

            # Add main effects (Include Raw for interactions AND Dummies for regression)
            for c in cand_valid:
                mode = mode_map.get(c, "linear")
                if mode == "categorical":
                    # 1. Add RAW column so interaction_lib can find it (CodeRabbit Fix)
                    multi_df[c] = df_aligned[c]
                    raw_cat_cols_to_drop.append(c)

                    # 2. Add Dummy columns for the regression model
                    levels = cat_levels_map.get(c, [])
                    raw_vals = df_aligned[c]
                    if len(levels) > 1:
                        for lvl in levels[1:]:
                            d_name = f"{c}::{lvl}"
                            multi_df[d_name] = (
                                raw_vals.astype(str) == str(lvl)
                            ).astype(int)
                else:
                    multi_df[c] = df_aligned[c].apply(clean_numeric_value)

            # ✅ Add interaction terms if specified
            if interaction_pairs:
                try:
                    from utils.interaction_lib import (
                        create_interaction_terms,
                        format_interaction_results,
                    )

                    # Now multi_df has the raw columns, so create_interaction_terms will work correctly
                    multi_df, int_meta = create_interaction_terms(
                        multi_df, interaction_pairs, mode_map
                    )
                    logger.info(
                        f"✅ Added {len(int_meta)} interaction terms to {model_label} multivariate model"
                    )
                except (ImportError, ValueError, TypeError, KeyError):
                    logger.exception("Failed to create interaction terms")

            # 🧹 CLEANUP: Remove raw categorical columns before passing to statsmodels
            # Statsmodels GLM with formula expects numeric dummies in design matrix (if not using formula API)
            if raw_cat_cols_to_drop:
                multi_df = multi_df.drop(columns=raw_cat_cols_to_drop, errors="ignore")

            multi_data = multi_df.dropna()
            final_n_multi = len(multi_data)
            predictors = [col for col in multi_data.columns if col != "y"]

            if not multi_data.empty and final_n_multi > 10 and len(predictors) > 0:
                offset_multi = (
                    offset.loc[multi_data.index] if offset is not None else None
                )

                if model_type == "negative_binomial":
                    params, conf, pvals, status, mv_stats = (
                        run_negative_binomial_regression(
                            multi_data["y"], multi_data[predictors], offset=offset_multi
                        )
                    )
                else:
                    params, conf, pvals, status, mv_stats = run_poisson_regression(
                        multi_data["y"], multi_data[predictors], offset=offset_multi
                    )

                if status == "OK":
                    # Format fit statistics
                    r2_parts = []
                    if pd.notna(mv_stats.get("deviance")):
                        r2_parts.append(f"Deviance = {mv_stats['deviance']:.2f}")
                    r2_parts = []
                    if model_type == "negative_binomial" and pd.notna(
                        mv_stats.get("alpha")
                    ):
                        r2_parts.append(f"Alpha (Dispersion) = {mv_stats['alpha']:.3f}")

                    if pd.notna(mv_stats.get("deviance")):
                        r2_parts.append(f"Deviance = {mv_stats['deviance']:.2f}")
                    if pd.notna(mv_stats.get("aic")):
                        r2_parts.append(f"AIC = {mv_stats['aic']:.2f}")
                    if r2_parts:
                        mv_metrics_text = " | ".join(r2_parts)

                    # Process main effects
                    for var in cand_valid:
                        mode = mode_map.get(var, "linear")
                        if mode == "categorical":
                            levels = cat_levels_map.get(var, [])
                            airr_entries = []
                            for lvl in levels[1:]:
                                d_name = f"{var}::{lvl}"
                                if d_name in params:
                                    coef = params[d_name]
                                    airr = np.exp(coef)
                                    ci_low, ci_high = (
                                        np.exp(conf.loc[d_name][0]),
                                        np.exp(conf.loc[d_name][1]),
                                    )
                                    pv = pvals[d_name]
                                    airr_entries.append(
                                        {
                                            "lvl": lvl,
                                            "coef": coef,
                                            "airr": airr,
                                            "l": ci_low,
                                            "h": ci_high,
                                            "p": pv,
                                        }
                                    )
                                    interpretation = interpret_irr(
                                        airr,
                                        ci_low,
                                        ci_high,
                                        f"{var}: {lvl}",
                                        mode="categorical",
                                    )
                                    airr_results[f"{var}: {lvl}"] = {
                                        "airr": airr,
                                        "ci_low": ci_low,
                                        "ci_high": ci_high,
                                        "p_value": pv,
                                        "interpretation": interpretation,
                                    }
                            results_db[var]["multi_res"] = airr_entries
                        else:
                            if var in params:
                                coef = params[var]
                                airr = np.exp(coef)
                                ci_low, ci_high = (
                                    np.exp(conf.loc[var][0]),
                                    np.exp(conf.loc[var][1]),
                                )
                                pv = pvals[var]
                                results_db[var]["multi_res"] = {
                                    "coef": coef,
                                    "airr": airr,
                                    "l": ci_low,
                                    "h": ci_high,
                                    "p": pv,
                                }
                                interpretation = interpret_irr(
                                    airr, ci_low, ci_high, var, mode="linear"
                                )
                                airr_results[var] = {
                                    "airr": airr,
                                    "ci_low": ci_low,
                                    "ci_high": ci_high,
                                    "p_value": pv,
                                    "interpretation": interpretation,
                                }

                    # ✅ Process interaction effects
                    if int_meta:
                        try:
                            interaction_results = format_interaction_results(
                                params, conf, pvals, int_meta, "poisson"
                            )
                            logger.info(
                                f"✅ Formatted {len(interaction_results)} Poisson interaction results"
                            )

                            # ✅ FIX: Merge interaction results into airr_results for forest plot inclusion
                            for int_name, int_res in interaction_results.items():
                                label = f"🔗 {int_res.get('label', int_name)}"
                                airr_results[label] = {
                                    "airr": int_res.get("irr"),
                                    "ci_low": int_res.get("ci_low"),
                                    "ci_high": int_res.get("ci_high"),
                                    "p_value": int_res.get("p_value"),
                                }
                        except Exception:
                            logger.exception("Failed to format interaction results")

        # ===================================================================
        # 🎨 STYLED HTML GENERATION (WITH INTERACTIONS)
        # ===================================================================

        # Professional CSS styling matching the Navy Blue theme
        css_styles = f"""<style>
            body {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                padding: 20px;
                background-color: {COLORS["background"]};
                color: {COLORS["text"]};
                line-height: 1.6;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }}
            
            .table-container {{
                background: {COLORS["surface"]};
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04);
                overflow-x: auto;
                margin-bottom: 16px;
                border: 1px solid {COLORS["border"]};
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            .table-container:hover {{
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12), 0 8px 16px rgba(0, 0, 0, 0.06);
                border-color: {COLORS["primary_light"]};
            }}
            
            table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                font-size: 13px;
            }}
            
            th {{
                background: linear-gradient(135deg, {COLORS["primary_dark"]} 0%, {COLORS["primary"]} 100%);
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                letter-spacing: 0.5px;
                border: none;
                font-size: 13px;
            }}
            
            td {{
                padding: 12px;
                border-bottom: 1px solid {COLORS["border"]};
                vertical-align: middle;
            }}
            
            tbody tr {{
                transition: background-color 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            tbody tr:nth-child(even) {{
                background-color: {COLORS["background"]};
            }}
            
            tbody tr:hover {{
                background-color: {COLORS["primary_light"]};
            }}
            
            .outcome-title {{
                background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
                color: white;
                padding: 15px 16px;
                font-weight: 600;
                font-size: 15px;
                border-radius: 8px 8px 0 0;
                border-bottom: 2px solid {COLORS["primary_dark"]};
            }}
            
            .summary-box {{
                background-color: {COLORS["primary_light"]};
                border-top: 2px solid {COLORS["border"]};
                padding: 14px 16px;
                font-size: 0.9em;
                color: {COLORS["text"]};
                border-radius: 0 0 8px 8px;
                line-height: 1.6;
            }}
            
            .summary-box b {{
                color: {COLORS["primary_dark"]};
            }}
            
            .sig-p {{
                color: #fff;
                background-color: {COLORS["danger"]};
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 3px;
                display: inline-block;
            }}
            
            .sheet-header td {{
                background-color: {COLORS["primary_light"]};
                color: {COLORS["primary_dark"]};
                font-weight: bold;
                padding: 10px 12px;
                border-top: 2px solid {COLORS["primary"]};
                border-bottom: 1px solid {COLORS["primary"]};
                font-size: 13px;
            }}
            
            .alert {{
                background-color: rgba(231, 72, 86, 0.08);
                border: 1px solid {COLORS["danger"]};
                border-radius: 6px;
                padding: 12px 16px;
                color: {COLORS["danger"]};
                margin-bottom: 16px;
                display: flex;
                align-items: flex-start;
                gap: 12px;
            }}
            
            h1, h2, h3 {{
                color: {COLORS["primary_dark"]};
                margin-top: 24px;
                margin-bottom: 16px;
            }}
            
            /* Responsive design */
            @media (max-width: 768px) {{
                body {{ padding: 12px; }}
                table {{ font-size: 12px; }}
                th, td {{ padding: 8px; }}
                .outcome-title {{ padding: 12px 14px; font-size: 14px; }}
                .summary-box {{ padding: 12px 14px; }}
            }}
        </style>"""

        # Build HTML table rows
        html_rows = []
        valid_cols_for_html = [c for c in sorted_cols if c in results_db]
        # Keep sorting by group to ensure variables of same sheet stay together
        grouped_cols = sorted(
            valid_cols_for_html,
            key=lambda x: (x.split("_")[0] if "_" in x else "Variables", x),
        )

        for col in grouped_cols:
            if col == outcome_name or col == offset_col:
                continue

            res = results_db[col]
            mode = mode_map.get(col, "linear")

            lbl = get_label(col, var_meta)
            mode_badge = {"categorical": "📊 (All Levels)", "linear": "📉 (Trend)"}
            if mode in mode_badge:
                lbl += f"<br><span style='font-size:0.8em; color:{COLORS['text_secondary']}'>{mode_badge[mode]}</span>"

            irr_s = res.get("irr", "-")
            coef_s = res.get("coef", "-")

            # Use styled p-value formatting
            if mode == "categorical":
                p_col_display = res.get("p_irr", "-")
            else:
                p_col_display = format_p_value(res.get("p_irr", np.nan))

            # Multivariate results
            airr_s, acoef_s, ap_s = "-", "-", "-"
            multi_res = res.get("multi_res")

            if multi_res:
                if isinstance(multi_res, list):
                    airr_lines, acoef_lines, ap_lines = ["Ref."], ["-"], ["-"]
                    for item in multi_res:
                        p_txt = format_p_value(item["p"])
                        acoef_lines.append(f"{item['coef']:.3f}")
                        ci_str = PublicationFormatter.format_ci(item["l"], item["h"])
                        airr_lines.append(f"{item['airr']:.2f} {ci_str}")
                        ap_lines.append(p_txt)
                    airr_s, acoef_s, ap_s = (
                        "<br>".join(airr_lines),
                        "<br>".join(acoef_lines),
                        "<br>".join(ap_lines),
                    )
                else:
                    if "coef" in multi_res and pd.notna(multi_res["coef"]):
                        acoef_s = f"{multi_res['coef']:.3f}"
                    else:
                        acoef_s = "-"
                    ci_str = PublicationFormatter.format_ci(
                        multi_res["l"], multi_res["h"]
                    )
                    airr_s = f"{multi_res['airr']:.2f} {ci_str}"
                    ap_s = format_p_value(multi_res["p"])

            html_rows.append(f"""<tr>
                <td>{lbl}</td>
                <td>{res.get("desc_total", "")}</td>
                <td>{coef_s}</td>
                <td>{irr_s}</td>
                <td>{res.get("test_name", "-")}</td>
                <td>{p_col_display}</td>
                <td>{acoef_s}</td>
                <td>{airr_s}</td>
                <td>{ap_s}</td>
            </tr>""")

        # ✅ Add interaction terms to HTML table (No header row)
        if interaction_results:
            for int_name, res in interaction_results.items():
                int_label = res.get("label", int_name)
                int_coef = (
                    f"{res.get('coef', 0):.3f}" if pd.notna(res.get("coef")) else "-"
                )
                irr_val = res.get("irr")
                if irr_val is not None and pd.notna(irr_val):
                    ci_str = PublicationFormatter.format_ci(
                        res.get("ci_low", 0), res.get("ci_high", 0)
                    )
                    int_irr = f"{irr_val:.2f} {ci_str}"
                else:
                    int_irr = "-"
                int_p = format_p_value(res.get("p_value", 1))

                html_rows.append(f"""<tr style='background-color: #fff9f0;'>
                    <td><b>🔗 {int_label}</b><br><small style='color: {COLORS["text_secondary"]};'>(Interaction)</small></td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>Interaction</td>
                    <td>-</td>
                    <td>{int_coef}</td>
                    <td><b>{int_irr}</b></td>
                    <td>{int_p}</td>
                </tr>""")

        logger.info(
            f"{model_label} analysis complete. Multivariate n={final_n_multi}, Interactions={len(interaction_results)}"
        )

        # Model fit and interaction info
        model_fit_html = ""
        if mv_metrics_text:
            model_fit_html = f"""<div style='margin-top: 8px; padding-top: 8px; 
                                            border-top: 1px dashed {COLORS["border"]}; 
                                            color: {COLORS["primary_dark"]};'>
                <b>Model Fit:</b> {mv_metrics_text}
            </div>"""

        interaction_info = ""
        if interaction_pairs:
            interaction_info = (
                f"<br><b>Interactions Tested:</b> {len(interaction_pairs)} pairs"
            )
            if interaction_results:
                try:
                    from utils.interaction_lib import interpret_interaction

                    interaction_info += interpret_interaction(
                        interaction_results,
                        "poisson",  # Interaction check uses generalized logic close enough for NB
                    )
                except ImportError:
                    logger.warning("interaction_lib not available for interpretation")

        # Offset information
        offset_info = f" | Offset: {offset_col}" if offset_col else ""

        # Complete HTML report with professional styling
        html_table = f"""{css_styles}
        <div id='{outcome_name}' class='table-container'>
            <div class='outcome-title'>📊 {model_label} Regression: {outcome_name} (n={
            total_n
        }){offset_info}</div>
            <table>
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Description</th>
                        <th>Crude Coef.</th>
                        <th>Crude IRR (95% CI)</th>
                        <th>Test</th>
                        <th>P-value</th>
                        <th>Adj. Coef.</th>
                        <th>aIRR (95% CI)</th>
                        <th>aP-value</th>
                    </tr>
                </thead>
                <tbody>{chr(10).join(html_rows)}</tbody>
            </table>
            <div class='summary-box'>
                <b>Method:</b> {model_label} GLM / MLE<br>
                <div style='margin-top: 8px; padding-top: 8px; 
                            border-top: 1px solid {COLORS["border"]}; 
                            font-size: 0.9em; color: {COLORS["text_secondary"]};'>
                    <b>Selection Criteria:</b> Variables with Crude P &lt; 0.20 included in multivariate model (n={
            final_n_multi
        })<br>
                    <b>Interpretation:</b> IRR = Incidence Rate Ratio (Rate in exposed / Rate in unexposed)<br>
                    <b>Visual Indicators:</b> 📊 Categorical (vs Reference) | 📉 Linear (Per-unit increase)
                    {model_fit_html}
                    {interaction_info}
                </div>
            </div>
        </div>
        <!-- IRR Interpretations Section -->
        {_build_irr_interpretation_section(irr_results, airr_results, COLORS)}
        <!-- Missing Data Section -->
        {
            create_missing_data_report_html(missing_data_info, var_meta or {})
            if CONFIG.get("analysis.missing.report_missing", True)
            else ""
        }
        <br>"""

        return (
            html_table,
            irr_results,
            airr_results,
            interaction_results,
            model_metadata,
        )

    except Exception as e:
        logger.exception(
            "Unexpected error in %s analysis for outcome: %s", model_type, outcome_name
        )
        # ✅ FIX: ALWAYS return 5 elements to prevent "cannot unpack" error
        return (
            f"<div class='alert alert-danger'>Analysis failed: {html.escape(str(e))}</div>",
            {},
            {},
            {},
            {"error": str(e), "model_type": model_type},
        )


# Alias for backward compatibility or different naming conventions
run_negative_binomial = run_negative_binomial_regression
