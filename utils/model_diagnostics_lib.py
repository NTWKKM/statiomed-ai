"""
📊 Model Diagnostics Library

Comprehensive regression diagnostics for publication-ready analysis.

Functions:
    - run_reset_test: Ramsey RESET test for misspecification
    - run_heteroscedasticity_test: Breusch-Pagan test
    - calculate_cooks_distance: Cook's Distance for influential points
    - calculate_dfbetas: DFBETAS for coefficient influence
    - calculate_dffits: DFFITS for fitted value influence
    - calculate_leverage: Hat values and leverage diagnostics
    - get_diagnostic_plot_data: Data for diagnostic plots
    - create_diagnostic_plots: Publication-ready Plotly figures

References:
    Belsley, Kuh & Welsch (1980). Regression Diagnostics.
    Cook, R.D. (1977). Detection of Influential Observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import reset_ramsey

from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


def run_reset_test(model_results) -> dict:
    """
    Run Ramsey's RESET test for model specification.

    Args:
        model_results: A fitted statsmodels result object (e.g. from OLS)

    Returns:
        dict with 'statistic', 'p_value', 'conclusion'
    """
    try:
        # Check if model has a method/attribute required (e.g., fitted OLS)
        if model_results is None or not hasattr(model_results, "model"):
            return {"error": "Invalid model object provided."}

        # Run RESET test (power=2,3 by default generally)
        reset = reset_ramsey(model_results, degree=3)

        p_val = reset.pvalue

        return {
            "test": "Ramsey RESET",
            "statistic": reset.statistic,
            "p_value": p_val,
            "conclusion": (
                "Possible misspecification (p < 0.05)"
                if p_val < 0.05
                else "No strong evidence of misspecification"
            ),
        }
    except Exception as e:
        logger.exception("RESET Test failed")
        return {"error": f"RESET Test failed: {e!s}"}


def run_heteroscedasticity_test(model_results) -> dict:
    """
    Run Breusch-Pagan test for heteroscedasticity.

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        dict with 'statistic', 'p_value', 'conclusion'
    """
    try:
        if model_results is None or not hasattr(model_results, "model"):
            return {"error": "Invalid model object provided."}

        # Breusch-Pagan requires residuals and exog
        resid = model_results.resid
        exog = model_results.model.exog

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, exog)

        return {
            "test": "Breusch-Pagan",
            "statistic": lm_stat,
            "p_value": lm_pvalue,
            "conclusion": (
                "Heteroscedasticity present (p < 0.05)"
                if lm_pvalue < 0.05
                else "Homoscedasticity assumption holds"
            ),
        }
    except Exception as e:
        logger.exception("Heteroscedasticity test failed")
        return {"error": f"Heteroscedasticity test failed: {e!s}"}


def calculate_cooks_distance(model_results) -> dict:
    """
    Calculate Cook's Distance for identifying influential points.

    Cook's D measures the effect of deleting a single observation.
    Values > 4/n are typically considered influential.

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        dict with 'cooks_d' (list), 'p_values' (list), 'influential_points' (indices)
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        influence = model_results.get_influence()
        # cooks_d is a tuple (distance, p-value)
        c_d, p_val = influence.cooks_distance

        if c_d is None:
            return {"error": "Could not calculate Cook's distance."}

        # Threshold: 4/n rule of thumb
        n = model_results.nobs
        threshold = 4 / n if n > 0 else 1.0

        influential_indices = np.where(c_d > threshold)[0].tolist()

        return {
            "cooks_d": c_d.tolist(),
            "p_values": p_val.tolist(),
            "threshold": threshold,
            "influential_points": influential_indices,
            "n_influential": len(influential_indices),
            "interpretation": (
                f"{len(influential_indices)} observations exceed 4/n threshold"
                if influential_indices
                else "No highly influential observations detected"
            ),
        }
    except Exception as e:
        logger.exception("Cooks distance calculation failed")
        return {"error": f"Cooks distance calculation failed: {e!s}"}


def calculate_dfbetas(model_results) -> dict[str, Any]:
    """
    Calculate DFBETAS for each coefficient.

    DFBETAS measures how much each coefficient changes when observation i
    is deleted. Observations with |DFBETAS| > 2/√n are influential.

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        Dictionary with:
            - dfbetas: 2D array (n_obs x n_params)
            - var_names: Parameter names
            - threshold: Cutoff for influential observations
            - influential_per_var: Dict of influential indices per variable
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        influence = model_results.get_influence()
        dfbetas = influence.dfbetas

        if dfbetas is None:
            return {"error": "Could not calculate DFBETAS."}

        n = model_results.nobs
        threshold = 2 / np.sqrt(n)

        # Get parameter names
        var_names = model_results.model.exog_names

        # Find influential observations for each variable
        influential_per_var = {}
        for i, var in enumerate(var_names):
            influential_idx = np.where(np.abs(dfbetas[:, i]) > threshold)[0].tolist()
            influential_per_var[var] = {
                "indices": influential_idx,
                "count": len(influential_idx),
                "max_dfbetas": float(np.max(np.abs(dfbetas[:, i]))),
            }

        return {
            "dfbetas": dfbetas.tolist(),
            "var_names": var_names,
            "threshold": round(threshold, 4),
            "influential_per_var": influential_per_var,
            "total_influential": sum(
                1 for row in dfbetas if np.any(np.abs(row) > threshold)
            ),
            "interpretation": (
                f"Threshold for influence: |DFBETAS| > {threshold:.3f} (2/√n)"
            ),
        }

    except Exception as e:
        logger.exception("DFBETAS calculation failed")
        return {"error": f"DFBETAS calculation failed: {e!s}"}


def calculate_dffits(model_results) -> dict[str, Any]:
    """
    Calculate DFFITS for each observation.

    DFFITS measures the change in the predicted value for observation i
    when it is deleted from the model. Observations with
    |DFFITS| > 2√(p/n) are considered influential.

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        Dictionary with dffits values and influential observations
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        influence = model_results.get_influence()
        dffits_vals = influence.dffits[0]  # dffits returns (dffits, dffits_internal)

        if dffits_vals is None:
            return {"error": "Could not calculate DFFITS."}

        n = model_results.nobs
        p = len(model_results.params)  # Number of parameters
        threshold = 2 * np.sqrt(p / n)

        influential_indices = np.where(np.abs(dffits_vals) > threshold)[0].tolist()

        return {
            "dffits": dffits_vals.tolist(),
            "threshold": round(threshold, 4),
            "influential_points": influential_indices,
            "n_influential": len(influential_indices),
            "max_dffits": float(np.max(np.abs(dffits_vals))),
            "interpretation": (
                f"Threshold: |DFFITS| > {threshold:.3f} (2√(p/n)). "
                f"{len(influential_indices)} influential observations detected."
            ),
        }

    except Exception as e:
        logger.exception("DFFITS calculation failed")
        return {"error": f"DFFITS calculation failed: {e!s}"}


def calculate_leverage(model_results) -> dict[str, Any]:
    """
    Calculate leverage (hat values) for each observation.

    Leverage measures how far an observation's predictor values are
    from the mean of all predictors. High leverage points have
    more influence on the regression line.

    Threshold: leverage > 2p/n (or 3p/n for stricter)

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        Dictionary with leverage values and high-leverage observations
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        influence = model_results.get_influence()
        hat_diag = influence.hat_matrix_diag

        if hat_diag is None:
            return {"error": "Could not calculate leverage."}

        n = model_results.nobs
        p = len(model_results.params)

        # Thresholds
        threshold_moderate = 2 * p / n
        threshold_high = 3 * p / n

        moderate_leverage = np.where(
            (hat_diag > threshold_moderate) & (hat_diag <= threshold_high)
        )[0].tolist()
        high_leverage = np.where(hat_diag > threshold_high)[0].tolist()

        return {
            "leverage": hat_diag.tolist(),
            "mean_leverage": round(float(np.mean(hat_diag)), 4),
            "threshold_moderate": round(threshold_moderate, 4),
            "threshold_high": round(threshold_high, 4),
            "moderate_leverage_points": moderate_leverage,
            "high_leverage_points": high_leverage,
            "interpretation": (
                f"Average leverage: {np.mean(hat_diag):.4f}. "
                f"{len(high_leverage)} high-leverage observations (>3p/n)."
            ),
        }

    except Exception as e:
        logger.exception("Leverage calculation failed")
        return {"error": f"Leverage calculation failed: {e!s}"}


def get_diagnostic_plot_data(model_results) -> dict:
    """
    Extract data needed for diagnostic plots (Residuals vs Fitted, Q-Q).

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        dict with 'fitted_values', 'residuals', 'std_residuals'
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        return {
            "fitted_values": model_results.fittedvalues.tolist(),
            "residuals": model_results.resid.tolist(),
            "std_residuals": model_results.get_influence().resid_studentized_internal.tolist(),
        }
    except Exception as e:
        logger.exception("Diagnostic plot data extraction failed")
        return {"error": f"Diagnostic plot data extraction failed: {e!s}"}


def create_diagnostic_plots(
    model_results,
    title_prefix: str = "",
) -> dict[str, Any]:
    """
    Create publication-ready diagnostic plots using Plotly.

    Generates four standard diagnostic plots:
    1. Residuals vs Fitted
    2. Q-Q Plot
    3. Scale-Location
    4. Residuals vs Leverage

    Args:
        model_results: A fitted statsmodels result object
        title_prefix: Optional prefix for plot titles

    Returns:
        Dictionary of Plotly figures:
            - combined: 2x2 subplot of all diagnostics
            - residuals_fitted: Residuals vs Fitted
            - qq_plot: Normal Q-Q plot
            - scale_location: Scale-Location plot
            - residuals_leverage: Residuals vs Leverage
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        influence = model_results.get_influence()
        fitted = model_results.fittedvalues
        residuals = model_results.resid
        std_resid = influence.resid_studentized_internal
        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]

        # Create individual figures

        # 1. Residuals vs Fitted
        fig_resid = go.Figure()
        fig_resid.add_trace(
            go.Scatter(
                x=fitted,
                y=residuals,
                mode="markers",
                marker=dict(color=COLORS.get("info", "#3B82F6"), size=6, opacity=0.7),
                name="Residuals",
                hovertemplate="Fitted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
            )
        )
        fig_resid.add_hline(
            y=0, line_dash="dash", line_color=COLORS.get("danger", "red")
        )
        fig_resid.update_layout(
            title=f"{title_prefix}Residuals vs Fitted",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            template="plotly_white",
            font=dict(family="Inter, Arial, sans-serif", size=11),
        )

        # 2. Q-Q Plot
        (theoretical_q, sorted_std_resid), (slope, intercept, _r) = stats.probplot(
            std_resid, dist="norm", plot=None
        )

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=sorted_std_resid,
                mode="markers",
                marker=dict(
                    color=COLORS.get("success", "#10B981"), size=6, opacity=0.7
                ),
                name="Q-Q",
            )
        )
        # Add diagonal reference line
        qq_x = np.array([theoretical_q.min(), theoretical_q.max()])
        qq_y = slope * qq_x + intercept

        fig_qq.add_trace(
            go.Scatter(
                x=qq_x,
                y=qq_y,
                mode="lines",
                line=dict(color=COLORS.get("danger", "red"), dash="dash"),
                showlegend=False,
            )
        )
        fig_qq.update_layout(
            title=f"{title_prefix}Normal Q-Q",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Standardized Residuals",
            template="plotly_white",
            font=dict(family="Inter, Arial, sans-serif", size=11),
        )

        # 3. Scale-Location
        sqrt_abs_resid = np.sqrt(np.abs(std_resid))
        fig_scale = go.Figure()
        fig_scale.add_trace(
            go.Scatter(
                x=fitted,
                y=sqrt_abs_resid,
                mode="markers",
                marker=dict(
                    color=COLORS.get("warning", "#F59E0B"), size=6, opacity=0.7
                ),
                name="√|Std Resid|",
            )
        )
        fig_scale.update_layout(
            title=f"{title_prefix}Scale-Location",
            xaxis_title="Fitted Values",
            yaxis_title="√|Std Residuals|",
            template="plotly_white",
            font=dict(family="Inter, Arial, sans-serif", size=11),
        )

        # 4. Residuals vs Leverage
        # Size points by Cook's D
        cooks_sizes = (
            5 + 50 * (cooks_d / cooks_d.max())
            if cooks_d.max() > 0
            else np.full_like(cooks_d, 6)
        )

        fig_lev = go.Figure()
        fig_lev.add_trace(
            go.Scatter(
                x=leverage,
                y=std_resid,
                mode="markers",
                marker=dict(
                    color=cooks_d,
                    colorscale="Reds",
                    size=cooks_sizes,
                    opacity=0.7,
                    colorbar=dict(title="Cook's D"),
                ),
                name="Leverage",
                hovertemplate=(
                    "Leverage: %{x:.4f}<br>"
                    "Std Resid: %{y:.3f}<br>"
                    "Cook's D: %{marker.color:.4f}<extra></extra>"
                ),
            )
        )
        fig_lev.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_lev.update_layout(
            title=f"{title_prefix}Residuals vs Leverage",
            xaxis_title="Leverage",
            yaxis_title="Standardized Residuals",
            template="plotly_white",
            font=dict(family="Inter, Arial, sans-serif", size=11),
        )

        # Create combined subplot by adding traces from individual plots
        # (Plotly copies traces internally so this is safe)
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Residuals vs Fitted",
                "Normal Q-Q",
                "Scale-Location",
                "Residuals vs Leverage",
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.15,
        )

        # trace 1
        fig.add_trace(fig_resid.data[0], row=1, col=1)
        fig.add_hline(
            y=0, line_dash="dash", line_color=COLORS.get("danger", "red"), row=1, col=1
        )

        # trace 2
        fig.add_trace(fig_qq.data[0], row=1, col=2)
        fig.add_trace(fig_qq.data[1], row=1, col=2)  # diagonal line

        # trace 3
        fig.add_trace(fig_scale.data[0], row=2, col=1)

        # trace 4
        fig.add_trace(fig_lev.data[0], row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

        # Update layout for combined
        fig.update_layout(
            title=f"{title_prefix}Regression Diagnostics"
            if title_prefix
            else "Regression Diagnostics",
            height=700,
            width=900,
            showlegend=False,
            font=dict(family="Inter, Arial, sans-serif", size=11),
            template="plotly_white",
        )

        # Update axes labels for combined
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Standardized Residuals", row=1, col=2)
        fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
        fig.update_yaxes(title_text="√|Std Residuals|", row=2, col=1)
        fig.update_xaxes(title_text="Leverage", row=2, col=2)
        fig.update_yaxes(title_text="Standardized Residuals", row=2, col=2)

        return {
            "combined": fig,
            "residuals_fitted": fig_resid,
            "qq_plot": fig_qq,
            "scale_location": fig_scale,
            "residuals_leverage": fig_lev,
        }

    except Exception as e:
        logger.exception("Diagnostic plot creation failed")
        return {"error": f"Plot creation failed: {e!s}"}


def get_comprehensive_diagnostics(model_results) -> dict[str, Any]:
    """
    Run all diagnostic tests and return comprehensive results.

    Combines Cook's D, DFBETAS, DFFITS, leverage, and specification tests.

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        Comprehensive diagnostics dictionary
    """
    if model_results is None:
        return {"error": "Invalid model object provided."}

    return {
        "cooks_distance": calculate_cooks_distance(model_results),
        "dfbetas": calculate_dfbetas(model_results),
        "dffits": calculate_dffits(model_results),
        "leverage": calculate_leverage(model_results),
        "reset_test": run_reset_test(model_results),
        "heteroscedasticity": run_heteroscedasticity_test(model_results),
        "plot_data": get_diagnostic_plot_data(model_results),
    }
