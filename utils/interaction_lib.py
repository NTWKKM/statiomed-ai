"""
🔗 Interaction Terms Library for Statistical Modeling

Handles creation and analysis of interaction effects between variables
Compatible with both Logistic and Poisson regression
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from logger import get_logger
from utils.formatting import PublicationFormatter, format_p_value
from utils.logic import clean_numeric_value

logger = get_logger(__name__)


def create_interaction_terms(
    df: pd.DataFrame,
    interaction_pairs: list[tuple[str, str]],
    mode_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Create interaction variables from pairs of predictors.

    Args:
        df: Input DataFrame
        interaction_pairs: List of tuples [(var1, var2), ...]
        mode_map: Dictionary mapping variables to 'categorical' or 'linear'

    Returns:
        tuple: (df_with_interactions, interaction_metadata)
    """
    if not interaction_pairs:
        return df, {}

    df_result = df.copy()
    interaction_meta = {}

    for var1, var2 in interaction_pairs:
        if var1 not in df.columns or var2 not in df.columns:
            logger.warning(f"Skipping interaction {var1}×{var2}: variables not found")
            continue

        mode1 = mode_map.get(var1, "linear") if mode_map else "linear"
        mode2 = mode_map.get(var2, "linear") if mode_map else "linear"

        # Get clean numeric values
        x1 = df[var1].apply(clean_numeric_value)
        x2 = df[var2].apply(clean_numeric_value)

        # Create interaction based on modes
        if mode1 == "categorical" and mode2 == "categorical":
            # Both categorical: Create dummy interactions
            levels1 = sorted(x1.dropna().unique())
            levels2 = sorted(x2.dropna().unique())

            for lvl1 in levels1[1:]:  # Skip reference
                for lvl2 in levels2[1:]:
                    int_name = f"{var1}::{lvl1}×{var2}::{lvl2}"
                    df_result[int_name] = ((x1 == lvl1) & (x2 == lvl2)).astype(int)
                    interaction_meta[int_name] = {
                        "type": "cat×cat",
                        "var1": var1,
                        "var2": var2,
                        "level1": lvl1,
                        "level2": lvl2,
                    }

        elif mode1 == "categorical" or mode2 == "categorical":
            # One categorical, one continuous
            if mode1 == "categorical":
                cat_var, cat_vals = var1, x1
                cont_var, cont_vals = var2, x2
            else:
                cat_var, cat_vals = var2, x2
                cont_var, cont_vals = var1, x1

            levels = sorted(cat_vals.dropna().unique())

            for lvl in levels[1:]:  # Skip reference
                int_name = f"{cat_var}::{lvl}×{cont_var}"
                df_result[int_name] = (cat_vals == lvl).astype(int) * cont_vals
                interaction_meta[int_name] = {
                    "type": "cat×cont",
                    "cat_var": cat_var,
                    "cont_var": cont_var,
                    "level": lvl,
                }

        else:
            # Both continuous: Simple product
            int_name = f"{var1}×{var2}"
            df_result[int_name] = x1 * x2
            interaction_meta[int_name] = {
                "type": "cont×cont",
                "var1": var1,
                "var2": var2,
            }

    logger.info(f"Created {len(interaction_meta)} interaction terms")
    return df_result, interaction_meta


def test_interaction_significance(
    y: pd.Series,
    X_base: pd.DataFrame,
    X_with_int: pd.DataFrame,
    model_type: str = "logit",
    offset: pd.Series | None = None,
) -> dict[str, float | int | bool]:
    """
    Test if adding interaction terms significantly improves model fit.

    Args:
        y: Outcome variable
        X_base: Base model predictors (no interactions)
        X_with_int: Full model with interactions
        model_type: 'logit' or 'poisson'
        offset: Exposure offset for Poisson

    Returns:
        dict: {'lr_stat', 'df', 'p_value', 'improvement'}
    """
    try:
        X_base_const = sm.add_constant(X_base, has_constant="add")
        X_full_const = sm.add_constant(X_with_int, has_constant="add")

        # Fit base model
        if model_type == "logit":
            model_base = sm.Logit(y, X_base_const)
            model_full = sm.Logit(y, X_full_const)
        elif model_type == "poisson":
            if offset is not None:
                model_base = sm.GLM(
                    y, X_base_const, family=sm.families.Poisson(), offset=offset
                )
                model_full = sm.GLM(
                    y, X_full_const, family=sm.families.Poisson(), offset=offset
                )
            else:
                model_base = sm.GLM(y, X_base_const, family=sm.families.Poisson())
                model_full = sm.GLM(y, X_full_const, family=sm.families.Poisson())
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        result_base = model_base.fit(disp=0)
        result_full = model_full.fit(disp=0)

        # Likelihood ratio test
        lr_stat = -2 * (result_base.llf - result_full.llf)
        df = len(X_full_const.columns) - len(X_base_const.columns)

        p_value = 1 - chi2.cdf(lr_stat, df)

    except Exception:
        logger.exception("Interaction test failed")
        return {"lr_stat": np.nan, "df": 0, "p_value": np.nan, "improvement": False}
    else:
        return {
            "lr_stat": lr_stat,
            "df": df,
            "p_value": p_value,
            "improvement": p_value < 0.05,
            "aic_base": result_base.aic,
            "aic_full": result_full.aic,
            "bic_base": result_base.bic,
            "bic_full": result_full.bic,
        }


def format_interaction_results(
    params: pd.Series,
    conf_int: pd.DataFrame,
    pvalues: pd.Series,
    interaction_meta: dict[str, Any],
    model_type: str = "logit",
) -> dict[str, Any]:
    """
    Format interaction results for display.
    """
    try:
        results = {}
        effect_name = "OR" if model_type == "logit" else "IRR"

        for int_name, meta in interaction_meta.items():
            if int_name not in params:
                continue

            coef = params[int_name]
            effect = np.exp(coef)
            ci_low, ci_high = (
                np.exp(conf_int.loc[int_name][0]),
                np.exp(conf_int.loc[int_name][1]),
            )
            pval = pvalues[int_name]

            # Create descriptive label
            if meta["type"] == "cat×cat":
                label = (
                    f"{meta['var1']}={meta['level1']} × {meta['var2']}={meta['level2']}"
                )
            elif meta["type"] == "cat×cont":
                label = f"{meta['cat_var']}={meta['level']} × {meta['cont_var']}"
            else:  # cont×cont
                label = f"{meta['var1']} × {meta['var2']}"

            results[int_name] = {
                "label": label,
                "coef": coef,
                effect_name.lower(): effect,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": pval,
                "significant": pval < 0.05,
                "type": meta["type"],
            }

        return results
    except Exception:
        logger.exception("Error formatting interaction results")
        return {}


def interpret_interaction(results: dict[str, Any], model_type: str = "logit") -> str:
    """
    Generate interpretation text for interaction results.
    """
    if not results:
        return "<p>No interaction terms to interpret.</p>"

    effect_name = "Odds Ratio" if model_type == "logit" else "Incidence Rate Ratio"

    sig_interactions = [k for k, v in results.items() if v["significant"]]

    if not sig_interactions:
        return """<div class='alert alert-info'>
            <b>No Significant Interactions Found</b><br>
            The effect of predictors does not significantly vary across levels of other variables.
            Main effects can be interpreted independently.
        </div>"""

    interp_lines = [
        f"<div class='alert alert-warning'><b>⚠️ Significant Interactions Detected ({len(sig_interactions)})</b><br>"
    ]

    for int_name in sig_interactions:
        res = results[int_name]
        effect_val = res.get("or" if model_type == "logit" else "irr")

        if effect_val > 1:
            direction = "synergistic (multiplicative increase)"
        elif effect_val < 1:
            direction = "antagonistic (multiplicative decrease)"
        else:
            direction = "neutral"

        ci_str = PublicationFormatter.format_ci(res["ci_low"], res["ci_high"])
        p_str = format_p_value(res["p_value"])

        interp_lines.append(
            f"<li><b>{res['label']}</b>: {effect_name} = {effect_val:.2f} "
            f"{ci_str}, {p_str} "
            f"→ {direction}</li>"
        )

    interp_lines.append(
        "<br><b>Interpretation:</b> The combined effect of these variables differs from "
        "the sum of their individual effects. Main effects should be interpreted cautiously.</div>"
    )

    return "".join(interp_lines)


def generate_interaction_html_table(
    results: dict[str, Any], model_type: str = "logit"
) -> str:
    """
    Generate HTML table for interaction results.
    """

    if not results:
        return "<p class='text-muted'>No interaction results available.</p>"

    effect_col = "OR" if model_type == "logit" else "IRR"

    rows = []
    for _int_name, res in results.items():
        effect_val = res.get("or" if model_type == "logit" else "irr")
        sig_marker = "✓" if res["significant"] else ""

        ci_str = PublicationFormatter.format_ci(res["ci_low"], res["ci_high"])
        p_str = format_p_value(res["p_value"])

        rows.append(f"""<tr>
            <td>{res["label"]}</td>
            <td>{res["type"]}</td>
            <td>{res["coef"]:.4f}</td>
            <td>{effect_val:.3f} {ci_str}</td>
            <td>{p_str}</td>
            <td>{sig_marker}</td>
        </tr>""")

    html = f"""<div class='table-container'>
    <table>
        <thead>
            <tr>
                <th>Interaction Term</th>
                <th>Type</th>
                <th>Coefficient</th>
                <th>{effect_col} (95% CI)</th>
                <th>P-value</th>
                <th>Sig.</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    </div>"""

    return html
