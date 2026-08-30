from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import prepare_data_for_analysis

logger = get_logger(__name__)


def calculate_vif(
    data: pd.DataFrame, predictors: list[str], var_meta: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Calculate Variance Inflation Factor (VIF) for a list of predictors.

    Args:
        data: DataFrame containing the data
        predictors: List of column names to check for collinearity
        var_meta: Variable metadata for missing data handling

    Returns:
        Tuple of (VIF DataFrame, missing-data info dict)
    """
    try:
        if not predictors or data is None or data.empty:
            return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"]), {
                "error": "Insufficient data or missing predictors"
            }

        # --- DATA PREPARATION ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        X_clean, missing_info = prepare_data_for_analysis(
            data,
            required_cols=predictors,
            numeric_cols=predictors,
            var_meta=var_meta,
            missing_codes=missing_codes,
            handle_missing=strategy,
        )

        if X_clean.empty or len(X_clean.columns) == 0:
            return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"]), missing_info

        valid_predictors = X_clean.columns.tolist()

        # VIF needs at least 2 variables to assess multicollinearity effectively
        if len(valid_predictors) == 1:
            return (
                pd.DataFrame(
                    [{"Variable": valid_predictors[0], "VIF": 1.0, "Tolerance": 1.0}]
                ),
                missing_info,
            )

        # Add constant for VIF calculation correctness
        X_with_const = sm.add_constant(X_clean)

        # Check for constant predictors in the original data (can cause infinite VIF)
        variances = X_clean.var()
        const_cols = variances[variances < 1e-10].index.tolist()
        if const_cols:
            logger.warning(
                "Constant predictors detected: %s. VIF might be undefined.", const_cols
            )

        vif_data = []
        for col in valid_predictors:
            try:
                # Find index in X_with_const
                idx = list(X_with_const.columns).index(col)
                val = variance_inflation_factor(X_with_const.values, idx)
                if not np.isfinite(val):
                    val = float("inf")
            except Exception as e:
                logger.error("VIF calculation failed for %s: %s", col, e)
                val = np.nan

            # Interpretation
            if not np.isfinite(val) or val >= 10:
                interpretation = "🔴 High"
            elif val >= 5:
                interpretation = "⚠️ Moderate"
            else:
                interpretation = "✅ OK"

            vif_data.append(
                {
                    "Variable": col,
                    "VIF": val,
                    "Tolerance": (
                        1.0 / val if val and val != 0 and np.isfinite(val) else np.nan
                    ),
                    "Interpretation": interpretation,
                }
            )

        df_vif = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
        return df_vif, missing_info

    except Exception as e:
        logger.exception("VIF calculation failed completely")
        return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"]), {"error": str(e)}


def condition_index(
    data: pd.DataFrame, predictors: list[str], var_meta: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Calculate Condition Index (CI) for assessing multicollinearity.

    Returns:
        tuple: (DataFrame with results, dict with missing data info)
    """
    try:
        if not predictors or data is None or data.empty:
            return pd.DataFrame(), {"error": "Insufficient data"}

        # --- DATA PREPARATION ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        X_clean, missing_info = prepare_data_for_analysis(
            data,
            required_cols=predictors,
            numeric_cols=predictors,
            var_meta=var_meta,
            missing_codes=missing_codes,
            handle_missing=strategy,
        )

        if X_clean.empty:
            return pd.DataFrame(), missing_info

        # Scale the data for CI
        # Guard against columns with all zeros to prevent division by zero
        sums_of_squares = (X_clean**2).sum(axis=0)
        sums_of_squares = sums_of_squares.replace(0, 1.0)
        X_norm = X_clean / np.sqrt(sums_of_squares)

        # Use SVD for numerical stability
        U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)

        max_sv = S.max()
        if max_sv == 0:
            return pd.DataFrame(), missing_info

        condition_indices = max_sv / S

        results = []
        for i, ci in enumerate(condition_indices):
            results.append(
                {"Dimension": i + 1, "Condition Index": ci, "Eigenvalue": S[i] ** 2}
            )

        return pd.DataFrame(results), missing_info

    except Exception as e:
        logger.exception("Condition Index calculation failed")
        return pd.DataFrame(), {"error": str(e)}
