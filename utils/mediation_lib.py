from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.api import OLS, add_constant

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import prepare_data_for_analysis

logger = get_logger(__name__)


def analyze_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    confounders: list[str] | None = None,
    var_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform mediation analysis using the product of coefficients method.
    """
    try:
        if data is None or data.empty:
            return {"error": "Input data is empty or None."}

        required_cols = [outcome, treatment, mediator]
        if confounders:
            required_cols.extend(confounders)

        # Clean col list (remove duplicates)
        required_cols = list(dict.fromkeys(required_cols))

        # --- DATA PREPARATION ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        try:
            df_clean, missing_info = prepare_data_for_analysis(
                data,
                required_cols=required_cols,
                numeric_cols=required_cols,
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy
        except Exception as e:
            return {"error": f"Data preparation failed: {e}"}

        if df_clean.empty:
            return {"error": "No valid data after cleaning."}

        # Check for constant variables (Treatment and Mediator MUST vary)
        if df_clean[treatment].nunique() <= 1:
            return {"error": f"Treatment variable '{treatment}' is constant."}
        if df_clean[mediator].nunique() <= 1:
            return {"error": f"Mediator variable '{mediator}' is constant."}

        # Check for perfect collinearity between Treatment and Mediator
        # Using correlation for a quick check
        if len(df_clean) > 1:
            corr = df_clean[[treatment, mediator]].corr().iloc[0, 1]
            if abs(corr) > 1.0 - 1e-12:
                return {"error": "Perfect collinearity between Treatment and Mediator."}

        # 1. Total Effect (c): Y ~ X + C
        X_total = df_clean[[treatment]]
        if confounders:
            X_total = pd.concat([X_total, df_clean[confounders]], axis=1)
        X_total = add_constant(X_total)
        model_total = OLS(df_clean[outcome], X_total).fit()
        c_total = model_total.params[treatment]

        # 2. Mediator Model (a): M ~ X + C
        X_med = df_clean[[treatment]]
        if confounders:
            X_med = pd.concat([X_med, df_clean[confounders]], axis=1)
        X_med = add_constant(X_med)
        model_med = OLS(df_clean[mediator], X_med).fit()
        a_path = model_med.params[treatment]

        # 3. Outcome Model (b, c'): Y ~ X + M + C
        X_out = df_clean[[treatment, mediator]]
        if confounders:
            X_out = pd.concat([X_out, df_clean[confounders]], axis=1)
        X_out = add_constant(X_out)
        model_out = OLS(df_clean[outcome], X_out).fit()
        b_path = model_out.params[mediator]
        c_prime = model_out.params[treatment]  # Direct effect

        # Indirect Effect (a * b)
        indirect_effect = a_path * b_path

        # Proportion Mediated
        if abs(c_total) < 1e-9:
            prop_mediated = 0.0
        else:
            prop_mediated = indirect_effect / c_total

        return {
            "total_effect": float(c_total),
            "direct_effect": float(c_prime),
            "indirect_effect": float(indirect_effect),
            "proportion_mediated": float(prop_mediated),
            "a_path": float(a_path),
            "b_path": float(b_path),
            "n_obs": len(df_clean),
            "missing_data_info": missing_info,
        }
    except Exception as e:
        logger.exception("Mediation analysis failed")
        return {"error": f"Mediation analysis failed: {str(e)}"}
