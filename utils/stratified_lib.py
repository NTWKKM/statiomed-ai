from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import prepare_data_for_analysis

logger = get_logger(__name__)


def mantel_haenszel(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    stratum: str,
    var_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate Mantel-Haenszel Odds Ratio for stratified data.
    """
    try:
        # Validate inputs
        if data is None or data.empty:
            return {"error": "Input data is empty."}

        required_cols = [outcome, treatment, stratum]

        # --- DATA PREPARATION ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        try:
            df_clean, missing_info = prepare_data_for_analysis(
                data,
                required_cols=required_cols,
                numeric_cols=[outcome, treatment],  # stratum might be categorical
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy
        except Exception as e:
            return {"error": f"Data preparation failed: {e}"}

        if df_clean.empty:
            return {"error": "No valid data after cleaning."}

        # Create list of 2x2 tables
        strata_levels = df_clean[stratum].unique()

        num_sum = 0
        den_sum = 0

        results_by_stratum = []

        if len(strata_levels) == 0:
            return {
                "error": "No strata found in data.",
                "missing_data_info": missing_info,
            }

        for s in strata_levels:
            subset = df_clean[df_clean[stratum] == s]

            # Ensure binary outcome/treatment (0/1) for calculation
            # We assume users pass binary columns as requested in UI

            a = ((subset[treatment] == 1) & (subset[outcome] == 1)).sum()
            b = ((subset[treatment] == 1) & (subset[outcome] == 0)).sum()
            c = ((subset[treatment] == 0) & (subset[outcome] == 1)).sum()
            d = ((subset[treatment] == 0) & (subset[outcome] == 0)).sum()

            n_k = a + b + c + d
            if n_k == 0:
                continue

            # MH components
            num_sum += (a * d) / n_k
            den_sum += (b * c) / n_k

            # Stratum OR (with small constant for stability if needed, but standard MH doesn't)
            or_k = (a * d) / (b * c) if (b * c) > 0 else np.nan

            results_by_stratum.append(
                {"Stratum": s, "OR": or_k, "a": a, "b": b, "c": c, "d": d}
            )

        mh_or = num_sum / den_sum if den_sum > 0 else np.nan

        return {
            "MH_OR": float(mh_or),
            "Strata_Results": pd.DataFrame(results_by_stratum),
            "missing_data_info": missing_info,
        }
    except Exception as e:
        logger.exception("Mantel-Haenszel calculation failed")
        return {"error": f"Mantel-Haenszel failed: {str(e)}"}


def breslow_day(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    stratum: str,
    var_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform Breslow-Day test replacement using Likelihood Ratio Test for Homogeneity.
    """
    try:
        if data is None or data.empty:
            return {"error": "Input data is empty."}

        required_cols = [outcome, treatment, stratum]

        # --- DATA PREPARATION ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        try:
            df_clean, missing_info = prepare_data_for_analysis(
                data,
                required_cols=required_cols,
                numeric_cols=[outcome, treatment],
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy
        except Exception as e:
            return {"error": f"Data preparation failed: {e}"}

        if df_clean.empty:
            return {"error": "No valid data after cleaning."}

        # Check for single stratum or no variation
        if df_clean[stratum].nunique() < 2:
            return {
                "error": "Need at least 2 strata for homogeneity test.",
                "missing_data_info": missing_info,
            }

        # Likelihood Ratio Test approach
        formula_h0 = f"{outcome} ~ {treatment} + C({stratum})"
        formula_h1 = f"{outcome} ~ {treatment} * C({stratum})"

        mod0 = smf.logit(formula_h0, data=df_clean).fit(disp=0)
        mod1 = smf.logit(formula_h1, data=df_clean).fit(disp=0)

        lr_stat = 2 * (mod1.llf - mod0.llf)
        df_diff = mod1.df_model - mod0.df_model

        if df_diff <= 0:
            p_val = 1.0
        else:
            p_val = 1 - chi2.cdf(lr_stat, df_diff)

        return {
            "test": "Likelihood Ratio Test for Homogeneity (Interaction)",
            "statistic": float(lr_stat),
            "p_value": float(p_val),
            "conclusion": (
                "Heterogeneity detected (p<0.05)"
                if p_val < 0.05
                else "Homogeneity holds"
            ),
            "missing_data_info": missing_info,
        }

    except Exception as e:
        logger.exception("Homogeneity test failed")
        return {"error": f"Error calculating homogeneity: {str(e)}"}
