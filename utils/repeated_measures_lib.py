from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable, Independence
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import prepare_data_for_analysis

logger = get_logger(__name__)


def run_gee(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    subject_col: str,
    covariates: list[str] = [],
    cov_struct: str = "exchangeable",
    family_str: str = "gaussian",
    var_meta: dict[str, Any] | None = None,
) -> Any:
    """Run Generalized Estimating Equations (GEE) model with unified data cleaning.

    Returns the fitted GEE results object, or an error dict/string on failure.
    """
    try:
        # Validate critical columns
        required_cols = [outcome_col, treatment_col, time_col, subject_col] + covariates

        # --- DATA PREPARATION ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        try:
            df_clean, _missing_info = prepare_data_for_analysis(
                df,
                required_cols=required_cols,
                numeric_cols=[outcome_col, time_col] + [c for c in covariates],
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
        except Exception as e:
            logger.error("Data preparation for GEE failed: %s", e)
            return {"error": f"Data preparation failed: {e}"}

        if df_clean.empty:
            return {"error": "No valid data after cleaning."}

        # Construct formula
        formula = (
            f"{outcome_col} ~ {treatment_col} + {time_col} + {treatment_col}:{time_col}"
        )
        if covariates:
            formula += " + " + " + ".join(covariates)

        # Set family
        family_map = {
            "gaussian": sm.families.Gaussian(),
            "binomial": sm.families.Binomial(),
            "poisson": sm.families.Poisson(),
            "gamma": sm.families.Gamma(),
        }
        family = family_map.get(family_str.lower(), sm.families.Gaussian())

        # Set correlation structure
        cov_struct_map = {
            "exchangeable": Exchangeable(),
            "independence": Independence(),
            "ar1": Autoregressive(),
        }
        covariance = cov_struct_map.get(cov_struct.lower(), Exchangeable())

        model = GEE.from_formula(
            formula,
            groups=subject_col,
            data=df_clean,
            family=family,
            cov_struct=covariance,
        )
        results = model.fit()
        return results, _missing_info
    except Exception as e:
        logger.exception("GEE failed")
        return {"error": f"Error running GEE: {str(e)}"}, {}


def run_lmm(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    subject_col: str,
    covariates: list[str] = [],
    random_slope: bool = False,
    var_meta: dict[str, Any] | None = None,
) -> Any:
    """Run Linear Mixed Model (LMM) with unified data cleaning.

    Returns the fitted MixedLM results object, or an error dict/string on failure.
    """
    try:
        # Validate critical columns
        required_cols = [outcome_col, treatment_col, time_col, subject_col] + covariates

        # --- DATA PREPARATION ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        try:
            df_clean, _missing_info = prepare_data_for_analysis(
                df,
                required_cols=required_cols,
                numeric_cols=[outcome_col, time_col] + [c for c in covariates],
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
        except Exception as e:
            logger.error("Data preparation for LMM failed: %s", e)
            return {"error": f"Data preparation failed: {e}"}

        if df_clean.empty:
            return {"error": "No valid data after cleaning."}

        # Construct formula
        formula = (
            f"{outcome_col} ~ {treatment_col} + {time_col} + {treatment_col}:{time_col}"
        )
        if covariates:
            formula += " + " + " + ".join(covariates)

        if random_slope:
            re_formula = f"~{time_col}"
            model = MixedLM.from_formula(
                formula,
                groups=subject_col,
                re_formula=re_formula,
                data=df_clean,
            )
        else:
            # Random intercept only
            model = MixedLM.from_formula(formula, groups=subject_col, data=df_clean)

        results = model.fit()
        return results, _missing_info
    except Exception as e:
        logger.exception("LMM failed")
        return {"error": f"Error running LMM: {str(e)}"}, {}


def create_trajectory_plot(
    df: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    group_col: str,
    subject_col: str | None = None,
) -> go.Figure:
    """Create a trajectory plot showing mean trends with confidence interval bands."""
    # Calculate means and SE per group at each time point
    summary = (
        df.groupby([group_col, time_col])[outcome_col]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary["ci_upper"] = summary["mean"] + 1.96 * summary["sem"]
    summary["ci_lower"] = summary["mean"] - 1.96 * summary["sem"]

    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip("#")
        lv = len(hex_color)
        rgb = tuple(int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

    fig = go.Figure()

    # Get unique groups
    groups = df[group_col].unique()
    colors = px.colors.qualitative.Plotly

    for i, group in enumerate(groups):
        group_data = summary[summary[group_col] == group]
        # Ensure we have enough colors, cycle if needed
        color_hex = colors[i % len(colors)]

        # Add Mean Line
        fig.add_trace(
            go.Scatter(
                x=group_data[time_col],
                y=group_data["mean"],
                mode="lines+markers",
                name=f"{group} (Mean)",
                line=dict(color=color_hex, width=3),
                legendgroup=str(group),
            )
        )

        # Add CI Band
        fill_color = hex_to_rgba(color_hex, 0.2)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([group_data[time_col], group_data[time_col][::-1]]),
                y=pd.concat([group_data["ci_upper"], group_data["ci_lower"][::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=str(group),
            )
        )

    fig.update_layout(
        title=f"Mean Trajectory of {outcome_col} by {group_col}",
        xaxis_title=time_col,
        yaxis_title=outcome_col,
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def extract_model_results(results: Any, model_type: str) -> pd.DataFrame:
    """Extract coeffs, CI, p-values from GEE/LMM results into a DataFrame."""
    if isinstance(results, str) or isinstance(results, dict):  # Error message
        return pd.DataFrame()

    try:
        # Both GEE and MixedLM results have .params, .bse, .pvalues, .conf_int()
        df_res = pd.DataFrame(
            {
                "Beta": results.params,
                "SE": results.bse,
                "P-value": results.pvalues,
                "CI Lower": results.conf_int()[0],
                "CI Upper": results.conf_int()[1],
            }
        )

        # Rounding
        df_res = df_res.round(4)
        df_res["Variable"] = df_res.index
        df_res = df_res[["Variable", "Beta", "SE", "CI Lower", "CI Upper", "P-value"]]
        df_res.reset_index(drop=True, inplace=True)

        return df_res
    except Exception:
        return pd.DataFrame()
