from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.spatial.distance import cdist

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import prepare_data_for_analysis

logger = get_logger(__name__)


def calculate_ps(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    var_meta: dict[str, Any] | None = None,
) -> pd.Series:
    """
    Calculate propensity scores using logistic regression with unified data cleaning.
    Returns propensity_scores as pd.Series (index matches original data).
    """
    try:
        # --- MISSING DATA HANDLING ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        # Deduplicate required columns
        req_cols = list(dict.fromkeys([treatment] + covariates))

        df_subset, missing_info = prepare_data_for_analysis(
            data,
            required_cols=req_cols,
            var_meta=var_meta,
            missing_codes=missing_codes,
            handle_missing=strategy,
        )

        if df_subset.empty:
            return pd.Series(dtype=float, index=data.index)

        X = df_subset[covariates]
        X = sm.add_constant(X)
        y = df_subset[treatment]

        # Simple logistic regression
        model = sm.Logit(y, X).fit(disp=0)
        ps = model.predict(X)

        # Reindex to original data (fill missing with NaN)
        ps_full = pd.Series(index=data.index, dtype=float)
        ps_full.loc[df_subset.index] = ps.values

        return ps_full, missing_info

    except Exception as e:
        logger.error(f"Propensity score calculation failed: {str(e)}")
        return pd.Series(dtype=float, index=data.index), {"error": str(e)}


# Wrapper for backward compatibility
def calculate_propensity_score(*args, **kwargs):
    return calculate_ps(*args, **kwargs)


def perform_matching(
    data: pd.DataFrame,
    treatment_col: str,
    ps_col: str,
    caliper: float = 0.2,
    ratio: int = 1,
) -> pd.DataFrame:
    """
    Perform nearest-neighbor propensity score matching with caliper.
    """
    try:
        df = data.copy().dropna(subset=[ps_col, treatment_col])

        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        if treated.empty or control.empty:
            return pd.DataFrame()

        # Get propensity scores
        treated_ps = treated[[ps_col]].values
        control_ps = control[[ps_col]].values

        # Calculate distance matrix
        distances = cdist(treated_ps, control_ps, metric="euclidean")

        matched_treated_idx = []
        matched_control_idx = []
        used_controls = set()

        # Greedy matching
        for i, treat_idx in enumerate(treated.index):
            best_dist = float("inf")
            best_control = None

            for j, ctrl_idx in enumerate(control.index):
                if ctrl_idx in used_controls:
                    continue
                dist = distances[i, j]
                if dist < best_dist and dist <= caliper:
                    best_dist = dist
                    best_control = ctrl_idx

            if best_control is not None:
                matched_treated_idx.append(treat_idx)
                matched_control_idx.append(best_control)
                used_controls.add(best_control)

        if not matched_treated_idx:
            return pd.DataFrame()

        # Combine matched pairs with explicit copy and index preservation
        matched_df = pd.concat(
            [
                df.loc[matched_treated_idx].copy(),
                df.loc[matched_control_idx].copy(),
            ]
        )

        # Add a weight column for convenience (1.0 for matched pairs)
        matched_df["matching_weight"] = 1.0

        return matched_df
    except Exception:
        return pd.DataFrame()


def calculate_ipw(
    data: pd.DataFrame, treatment: str, outcome: str, ps_col: str
) -> dict:
    """
    Calculate Average Treatment Effect (ATE) using Inverse Probability Weighting.
    """
    try:
        df = data.copy().dropna(subset=[treatment, outcome, ps_col])
        if df.empty:
            return {"error": "No valid data after removing NaNs"}

        T = df[treatment].values
        Y = df[outcome].values
        ps = df[ps_col].values

        # Avoid division by zero
        ps = np.clip(ps, 0.01, 0.99)

        # Calculate weights
        weights = np.where(T == 1, 1 / ps, 1 / (1 - ps))

        # ATE is coefficient of T in weighted regression
        X = sm.add_constant(T)
        model = sm.WLS(Y, X, weights=weights).fit()

        # --- FIX: Use iloc/positional access safely ---
        # Check if model has enough parameters (Constant + Treatment)
        if len(model.params) < 2:
            return {
                "error": "Model could not estimate treatment effect (treatment may be constant or collinear)."
            }

        # Use positional indexing based on return type
        if isinstance(model.params, pd.Series):
            ate = model.params.iloc[1]
            se = model.bse.iloc[1]
            p_val = model.pvalues.iloc[1]
        else:
            # Numpy array fallback
            ate = model.params[1]
            se = model.bse[1]
            p_val = model.pvalues[1]

        # Handle conf_int return type (could be DataFrame or array depending on statsmodels version/context)
        conf_int = model.conf_int()
        if isinstance(conf_int, pd.DataFrame):
            ci_lower = conf_int.iloc[1, 0]
            ci_upper = conf_int.iloc[1, 1]
        else:
            # Assume numpy array
            ci_lower = conf_int[1, 0]
            ci_upper = conf_int[1, 1]

        return {
            "ATE": float(ate),
            "SE": float(se),
            "p_value": float(p_val),
            "CI_Lower": float(ci_lower),
            "CI_Upper": float(ci_upper),
        }
    except Exception as e:
        logger.error(f"IPW calculation failed: {str(e)}")
        return {"error": str(e)}


def check_balance(
    data: pd.DataFrame, treatment: str, covariates: list[str], weights: pd.Series = None
) -> pd.DataFrame:
    """
    Calculate Standardized Mean Differences (SMD) to check balance.
    """
    results = []

    # Ensure all columns exist
    all_needed = [treatment] + covariates
    missing_cols = [c for c in all_needed if c not in data.columns]
    if missing_cols:
        logger.warning(f"Missing columns in check_balance: {missing_cols}")
        return pd.DataFrame(columns=["Covariate", "SMD", "Status"])

    # Ensure clean data for balance check
    df = data.dropna(subset=all_needed)
    if df.empty:
        return pd.DataFrame(columns=["Covariate", "SMD", "Status"])

    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    if treated.empty or control.empty:
        return pd.DataFrame(columns=["Covariate", "SMD", "Status"])

    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(df[cov]):
            continue

        mean_t = np.average(
            treated[cov],
            weights=weights[treated.index].values if weights is not None else None,
        )
        mean_c = np.average(
            control[cov],
            weights=weights[control.index].values if weights is not None else None,
        )

        var_t = np.var(treated[cov])
        var_c = np.var(control[cov])
        pooled_sd = np.sqrt((var_t + var_c) / 2)

        smd = (mean_t - mean_c) / pooled_sd if pooled_sd != 0 else 0

        results.append(
            {
                "Covariate": cov,
                "Variable": cov,  # For test compatibility
                "SMD": abs(smd),
                "Status": "Balanced (<0.1)" if abs(smd) < 0.1 else "Unbalanced",
            }
        )

    return pd.DataFrame(results)


def calculate_smd(data, treatment, covariates, weights=None):
    """Test-compatible alias for check_balance."""
    return check_balance(data, treatment, covariates, weights)


def plot_love_plot(smd_pre: pd.DataFrame, smd_post: pd.DataFrame) -> go.Figure:
    """
    Create a Love Plot (SMD Comparison) using Plotly with enhanced diagnostics zones.
    """
    fig = go.Figure()

    # --- ENHANCEMENT: Add colored background zones for interpretation ---
    # Green Zone (Excellent Balance: < 0.1)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=0,
        y0=0,
        x1=0.1,
        y1=1,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    # Yellow Zone (Acceptable: 0.1 - 0.2)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=0.1,
        y0=0,
        x1=0.2,
        y1=1,
        fillcolor="yellow",
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    # Red Zone (Imbalanced: > 0.2) - Optional visual indicator, usually implied by outside zones

    # Pre-matching
    fig.add_trace(
        go.Scatter(
            x=smd_pre["SMD"],
            y=smd_pre.get("Variable", smd_pre.get("Covariate")),
            mode="markers",
            name="Unadjusted",
            marker=dict(color="#d62728", size=10, symbol="circle-open"),  # Standard Red
            hovertemplate="Unadjusted SMD: %{x:.3f}<extra></extra>",
        )
    )

    # Post-matching
    fig.add_trace(
        go.Scatter(
            x=smd_post["SMD"],
            y=smd_post.get("Variable", smd_post.get("Covariate")),
            mode="markers",
            name="Adjusted (Matched)",
            marker=dict(color="#2ca02c", size=10, symbol="circle"),  # Standard Green
            hovertemplate="Adjusted SMD: %{x:.3f}<extra></extra>",
        )
    )

    # Reference lines
    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color="black",
        opacity=0.5,
        annotation_text="0.1 (Excellent)",
    )
    fig.add_vline(
        x=0.2,
        line_dash="dot",
        line_color="black",
        opacity=0.5,
        annotation_text="0.2 (Limit)",
    )

    fig.update_layout(
        title="<b>Covariate Balance (Love Plot)</b><br><sup>Target: All dots within the green zone (<0.1)</sup>",
        xaxis_title="Absolute Standardized Mean Difference (SMD)",
        yaxis_title="",
        template="plotly_white",
        height=max(500, 300 + (len(smd_pre) * 35)),  # Dynamic height
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=150),  # Ensure long variable names are visible
    )

    return fig


def plot_ps_distribution(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    treatment_col: str,
    ps_col: str = "propensity_score",
) -> go.Figure:
    """
    NEW: Plot Propensity Score Distribution (Common Support).
    Compares Treated vs Control distributions before and after matching.
    """
    fig = go.Figure()

    # Check if data exists
    if df_pre is None or df_pre.empty:
        return fig

    # Validate required columns exist
    if treatment_col not in df_pre.columns or ps_col not in df_pre.columns:
        return fig

    # colors
    c_treat = "rgba(255, 100, 100, 0.6)"
    c_control = "rgba(100, 100, 255, 0.6)"

    # 1. Before Matching - Control
    fig.add_trace(
        go.Histogram(
            x=df_pre[df_pre[treatment_col] == 0][ps_col],
            name="Control (Raw)",
            marker_color=c_control,
            opacity=0.5,
            bingroup=1,
            legendgroup="Raw",
            legendgrouptitle_text="Before Matching",
        )
    )

    # 2. Before Matching - Treated
    fig.add_trace(
        go.Histogram(
            x=df_pre[df_pre[treatment_col] == 1][ps_col],
            name="Treated (Raw)",
            marker_color=c_treat,
            opacity=0.5,
            bingroup=1,
            legendgroup="Raw",
        )
    )

    # 3. After Matching - Control (Overlay as line or darker bar)
    if df_post is not None and not df_post.empty:
        fig.add_trace(
            go.Histogram(
                x=df_post[df_post[treatment_col] == 0][ps_col],
                name="Control (Matched)",
                marker=dict(color="blue", pattern=dict(shape="/")),
                opacity=0.6,
                bingroup=1,
                visible="legendonly",  # Hidden by default to avoid clutter
                legendgroup="Matched",
                legendgrouptitle_text="After Matching",
            )
        )

        fig.add_trace(
            go.Histogram(
                x=df_post[df_post[treatment_col] == 1][ps_col],
                name="Treated (Matched)",
                marker=dict(color="red", pattern=dict(shape="/")),
                opacity=0.6,
                bingroup=1,
                visible="legendonly",
                legendgroup="Matched",
            )
        )

    fig.update_layout(
        title="<b>Propensity Score Distribution (Common Support)</b>",
        xaxis_title="Propensity Score",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


class PropensityScoreDiagnostics:
    """
    Diagnostic tools for Propensity Score methods.
    Includes Common Support assessment, Weight Truncation, and Balance Checks.
    """

    @staticmethod
    def assess_common_support(
        ps: pd.Series | np.ndarray,
        treatment: pd.Series | np.ndarray,
        min_support: float | None = None,
    ) -> dict[str, Any]:
        """
        Assess whether treated and control have overlapping PS distributions.

        Returns:
            Dictionary containing overlap ranges, statistics, and recommendations.
        """
        try:
            ps = np.array(ps)
            treat = np.array(treatment)

            treated_ps = ps[treat == 1]
            control_ps = ps[treat == 0]

            if len(treated_ps) == 0 or len(control_ps) == 0:
                return {"error": "Insufficient data"}

            min_treated, max_treated = np.min(treated_ps), np.max(treated_ps)
            min_control, max_control = np.min(control_ps), np.max(control_ps)

            # Common support region
            overlap_min = max(min_treated, min_control)
            overlap_max = min(max_treated, max_control)

            in_support = (ps >= overlap_min) & (ps <= overlap_max)
            overlap_percent = np.mean(in_support) * 100

            units_to_exclude = np.where(~in_support)[0]

            recommendation = "Adequate"
            if overlap_percent < 80:
                recommendation = (
                    "Recommend exclusion or alternative method (limited overlap)"
                )
            elif overlap_percent < 95:
                recommendation = "Consider trimming (some non-overlap)"

            return {
                "overlap_range": (float(overlap_min), float(overlap_max)),
                "treated_range": (float(min_treated), float(max_treated)),
                "control_range": (float(min_control), float(max_control)),
                "overlap_percent": float(overlap_percent),
                "units_to_exclude": units_to_exclude.tolist(),
                "excluded_count": len(units_to_exclude),
                "recommendation": recommendation,
            }
        except Exception as e:
            logger.error(f"Common support assessment failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def truncate_weights(
        weights: pd.Series | np.ndarray,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> pd.Series:
        """
        Truncate (trim) extreme weights to reduce variance.
        Values below lower_percentile are set to lower_percentile value.
        Values above upper_percentile are set to upper_percentile value.
        """
        try:
            w = pd.Series(weights).copy()
            lower = w.quantile(lower_percentile)
            upper = w.quantile(upper_percentile)
            return w.clip(lower=lower, upper=upper)
        except Exception as e:
            logger.error(f"Weight truncation failed: {e}")
            return pd.Series(weights)

    @staticmethod
    def calculate_smd(
        df: pd.DataFrame,
        treatment_col: str,
        covariates: list[str],
        weights: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Standardized Mean Difference (SMD).
        Wrapper designed to be class-method compatible with user request.
        """
        return check_balance(df, treatment_col, covariates, weights)

    @staticmethod
    def create_love_plot(smd_pre: pd.DataFrame, smd_post: pd.DataFrame) -> go.Figure:
        """
        Generate a 'Love Plot' showing SMD comparison.
        Wrapper for plot_love_plot.
        """
        return plot_love_plot(smd_pre, smd_post)

    @staticmethod
    def plot_ps_overlap(
        df_pre: pd.DataFrame,
        treatment_col: str,
        ps_col: str,
        df_post: pd.DataFrame | None = None,
    ) -> go.Figure:
        """
        Visualize PS Overlap.
        Wrapper for plot_ps_distribution.
        """
        return plot_ps_distribution(
            df_pre=df_pre,
            df_post=df_post,
            treatment_col=treatment_col,
            ps_col=ps_col,
        )
