"""
📚 Clinical Meta-Analysis Library for Systematic Reviews (PRISMA 2020 Compliant)

Provides:
- Fixed-Effect & Random-Effects Models (Inverse-Variance, DerSimonian-Laird, REML, Paule-Mandel)
- Hartung-Knapp-Sidik-Jonkman (HKSJ) Adjustment for small study counts
- 95% Prediction Interval for individual future studies
- Heterogeneity assessment: Cochran's Q, I², τ², τ
- Subgroup Meta-Analysis with test for subgroup differences (Q_between)
- Publication bias suite: Egger's linear regression test, Begg's rank test
- Interactive Plotly Forest Plot & Contour-Enhanced Funnel Plot

References:
    DerSimonian R, Laird N. Control Clin Trials. 1986;7(3):177-188.
    IntHout J, Ioannidis JP, Borm GF. BMJ. 2014;348:g170.
    Higgins JPT, Thompson SG. Stat Med. 2002;21(11):1539-1558.
    Peters JL, Sutton AJ, et al. J Clin Epidemiol. 2008;61(10):991-996.
    Page MJ, et al. PRISMA 2020 explanation and elaboration. BMJ. 2021;372:n160.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import optimize, stats

from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


# =============================================================================
# DATA PREPARATION & EFFECT SIZE CALCULATORS
# =============================================================================


def compute_binary_effect_sizes(
    df: pd.DataFrame,
    events_t_col: str,
    n_t_col: str,
    events_c_col: str,
    n_c_col: str,
    study_col: str,
    effect_measure: str = "OR",
    subgroup_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute effect sizes (OR, RR, RD) and standard errors from 2x2 contingency table columns.

    Args:
        df: Input DataFrame
        events_t_col: Number of events in treatment arm
        n_t_col: Total sample size in treatment arm
        events_c_col: Number of events in control arm
        n_c_col: Total sample size in control arm
        study_col: Study identification column
        effect_measure: "OR" (Odds Ratio), "RR" (Risk Ratio), or "RD" (Risk Difference)
        subgroup_col: Optional column for subgroup stratification

    Returns:
        DataFrame with study, effect_size, log_effect, se, ci_lower, ci_upper, weight data
    """
    for c in [study_col, events_t_col, n_t_col, events_c_col, n_c_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in dataset.")

    if len({events_t_col, n_t_col, events_c_col, n_c_col}) < 4:
        raise ValueError(
            "Events and Total N columns for treatment and control must be distinct. "
            "Please check column mappings."
        )

    cols = [study_col, events_t_col, n_t_col, events_c_col, n_c_col]
    if subgroup_col and subgroup_col in df.columns:
        cols.append(subgroup_col)

    unique_cols = list(dict.fromkeys(cols))
    clean_df = df[unique_cols].dropna().copy()

    records = []
    for _, row in clean_df.iterrows():
        study = str(row[study_col])
        subgroup = (
            str(row[subgroup_col])
            if subgroup_col and subgroup_col in df.columns
            else "Overall"
        )

        try:
            a = float(row[events_t_col])
            n1 = float(row[n_t_col])
            c = float(row[events_c_col])
            n0 = float(row[n_c_col])
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Non-numeric value encountered in 2x2 data for study '{study}': {err}"
            ) from err

        b = n1 - a
        d = n0 - c

        # Continuity correction of 0.5 if any cell is 0
        cc = 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else 0.0

        a_c = a + cc
        b_c = b + cc
        c_c = c + cc
        d_c = d + cc
        n1_c = n1 + 2 * cc
        n0_c = n0 + 2 * cc

        if effect_measure == "OR":
            or_val = (a_c * d_c) / (b_c * c_c)
            log_eff = math.log(or_val)
            se = math.sqrt(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
            eff_disp = or_val
            ci_low = math.exp(log_eff - 1.96 * se)
            ci_high = math.exp(log_eff + 1.96 * se)
        elif effect_measure == "RR":
            p1 = a_c / n1_c
            p0 = c_c / n0_c
            rr_val = p1 / p0
            log_eff = math.log(rr_val)
            se = math.sqrt((1.0 / a_c - 1.0 / n1_c) + (1.0 / c_c - 1.0 / n0_c))
            eff_disp = rr_val
            ci_low = math.exp(log_eff - 1.96 * se)
            ci_high = math.exp(log_eff + 1.96 * se)
        else:  # "RD"
            p1 = a / n1 if n1 > 0 else 0
            p0 = c / n0 if n0 > 0 else 0
            rd_val = p1 - p0
            log_eff = rd_val  # linear scale
            se = (
                math.sqrt((p1 * (1 - p1)) / n1 + (p0 * (1 - p0)) / n0)
                if (n1 > 0 and n0 > 0)
                else 0.1
            )
            eff_disp = rd_val
            ci_low = rd_val - 1.96 * se
            ci_high = rd_val + 1.96 * se

        records.append(
            {
                "study": study,
                "subgroup": subgroup,
                "events_t": int(a),
                "n_t": int(n1),
                "events_c": int(c),
                "n_c": int(n0),
                "effect_size": eff_disp,
                "log_effect": log_eff,
                "se": se,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "is_ratio": effect_measure in ["OR", "RR"],
            }
        )

    return pd.DataFrame(records)


def compute_continuous_effect_sizes(
    df: pd.DataFrame,
    mean_t_col: str,
    sd_t_col: str,
    n_t_col: str,
    mean_c_col: str,
    sd_c_col: str,
    n_c_col: str,
    study_col: str,
    effect_measure: str = "SMD",
    subgroup_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute Mean Difference (MD) or Standardized Mean Difference (SMD / Hedges' g) for continuous endpoints.
    """
    for c in [study_col, mean_t_col, sd_t_col, n_t_col, mean_c_col, sd_c_col, n_c_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in dataset.")

    if len({mean_t_col, sd_t_col, n_t_col, mean_c_col, sd_c_col, n_c_col}) < 6:
        raise ValueError(
            "Mean, SD, and N columns for treatment and control must be distinct. "
            "Please check column mappings."
        )

    cols = [study_col, mean_t_col, sd_t_col, n_t_col, mean_c_col, sd_c_col, n_c_col]
    if subgroup_col and subgroup_col in df.columns:
        cols.append(subgroup_col)

    unique_cols = list(dict.fromkeys(cols))
    clean_df = df[unique_cols].dropna().copy()

    records = []
    for _, row in clean_df.iterrows():
        study = str(row[study_col])
        subgroup = (
            str(row[subgroup_col])
            if subgroup_col and subgroup_col in df.columns
            else "Overall"
        )

        try:
            m1 = float(row[mean_t_col])
            s1 = float(row[sd_t_col])
            n1 = float(row[n_t_col])
            m0 = float(row[mean_c_col])
            s0 = float(row[sd_c_col])
            n0 = float(row[n_c_col])
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Non-numeric value encountered in continuous data for study '{study}': {err}"
            ) from err

        if n1 < 2 or n0 < 2 or s1 <= 0 or s0 <= 0:
            continue

        if effect_measure == "MD":
            diff = m1 - m0
            se = math.sqrt((s1**2 / n1) + (s0**2 / n0))
            eff_disp = diff
            log_eff = diff
            ci_low = diff - 1.96 * se
            ci_high = diff + 1.96 * se
        else:  # "SMD" (Hedges' g)
            df_deg = n1 + n0 - 2
            s_pooled = (
                math.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / df_deg)
                if df_deg > 0
                else 1.0
            )
            d = (m1 - m0) / s_pooled if s_pooled > 0 else 0.0
            # Hedges' small-sample correction factor J
            j = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0)) if (n1 + n0) > 3 else 1.0
            g = d * j
            se = math.sqrt((n1 + n0) / (n1 * n0) + (g**2) / (2.0 * (n1 + n0)))
            eff_disp = g
            log_eff = g
            ci_low = g - 1.96 * se
            ci_high = g + 1.96 * se

        records.append(
            {
                "study": study,
                "subgroup": subgroup,
                "mean_t": m1,
                "sd_t": s1,
                "n_t": int(n1),
                "mean_c": m0,
                "sd_c": s0,
                "n_c": int(n0),
                "effect_size": eff_disp,
                "log_effect": log_eff,
                "se": se,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "is_ratio": False,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# STATISTICAL POOLING & HETEROGENEITY ENGINE
# =============================================================================


def _estimate_tau2_pm(
    theta: np.ndarray, se: np.ndarray, q_stat: float, k: int
) -> float:
    """Estimate tau^2 between-study variance using Paule-Mandel method."""
    if q_stat <= (k - 1) or k <= 1:
        return 0.0

    def f_pm(t2: float) -> float:
        w = 1.0 / (se**2 + t2)
        mu = float(np.sum(w * theta) / np.sum(w))
        return float(np.sum(w * ((theta - mu) ** 2)) - (k - 1))

    b = 10.0
    while f_pm(b) > 0 and b < 1e7:
        b *= 4.0
    if f_pm(b) > 0:
        return float(b)

    try:
        root = optimize.brentq(f_pm, 0.0, b, xtol=1e-6, maxiter=100)
        return max(0.0, float(root))
    except Exception:
        return 0.0


def _estimate_tau2_reml(
    theta: np.ndarray, se: np.ndarray, q_stat: float, k: int
) -> float:
    """Estimate tau^2 between-study variance using Restricted Maximum Likelihood (REML)."""
    if k <= 1:
        return 0.0

    def f_reml(t2: float) -> float:
        w = 1.0 / (se**2 + t2)
        mu = float(np.sum(w * theta) / np.sum(w))
        return float(
            np.sum((w**2) * ((theta - mu) ** 2)) - np.sum(w) + np.sum(w**2) / np.sum(w)
        )

    if f_reml(0.0) <= 0:
        return 0.0

    b = 10.0
    while f_reml(b) > 0 and b < 1e7:
        b *= 4.0
    if f_reml(b) > 0:
        return float(b)

    try:
        root = optimize.brentq(f_reml, 0.0, b, xtol=1e-6, maxiter=100)
        return max(0.0, float(root))
    except Exception:
        return 0.0


def run_meta_analysis(
    data: pd.DataFrame,
    method_re: str = "dl",  # "dl" (DerSimonian-Laird), "reml", "pm"
    use_hksj: bool = True,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Fit Fixed-Effect and Random-Effects Meta-Analysis models.

    Args:
        data: DataFrame containing 'log_effect' (or effect size) and 'se' columns
        method_re: Random-effects variance estimator ("dl", "reml", "pm")
        use_hksj: Whether to apply Hartung-Knapp-Sidik-Jonkman adjustment for random effects
        alpha: Significance level (default 0.05 for 95% CIs)

    Returns:
        Dictionary with pooled effects, weights, heterogeneity stats, and prediction intervals
    """
    df = data.dropna(subset=["log_effect", "se"]).copy()
    df = df[df["se"] > 0].reset_index(drop=True)

    k = len(df)
    if k < 2:
        return {"error": "Meta-analysis requires at least 2 valid studies.", "k": k}

    theta = df["log_effect"].to_numpy(dtype=float)
    se = df["se"].to_numpy(dtype=float)
    is_ratio = bool(df["is_ratio"].iloc[0]) if "is_ratio" in df.columns else False

    # 1. Fixed Effect Model (Inverse Variance)
    w_fe = 1.0 / (se**2)
    w_fe_pct = (w_fe / np.sum(w_fe)) * 100.0
    sum_w_fe = np.sum(w_fe)

    theta_fe = np.sum(w_fe * theta) / sum_w_fe
    se_fe = math.sqrt(1.0 / sum_w_fe)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_fe_low = theta_fe - z_crit * se_fe
    ci_fe_high = theta_fe + z_crit * se_fe
    z_val_fe = theta_fe / se_fe if se_fe > 0 else 0
    p_val_fe = 2.0 * (1.0 - stats.norm.cdf(abs(z_val_fe)))

    # 2. Heterogeneity Assessment (Cochran's Q, I², τ²)
    q_stat = float(np.sum(w_fe * ((theta - theta_fe) ** 2)))
    df_q = k - 1
    p_q = float(1.0 - stats.chi2.cdf(q_stat, df_q)) if df_q > 0 else 1.0

    # Between-study variance tau^2 estimation
    method_re_clean = str(method_re).lower().strip()
    if method_re_clean == "reml":
        tau2 = _estimate_tau2_reml(theta, se, q_stat, k)
        method_label = "REML"
    elif method_re_clean == "pm":
        tau2 = _estimate_tau2_pm(theta, se, q_stat, k)
        method_label = "Paule-Mandel"
    else:
        # Default DerSimonian-Laird
        c_const = sum_w_fe - (np.sum(w_fe**2) / sum_w_fe)
        tau2 = max(0.0, (q_stat - df_q) / c_const) if c_const > 0 else 0.0
        method_label = "DerSimonian-Laird"

    tau = math.sqrt(tau2)

    # I^2 statistic and Higgins 95% CI
    i2 = max(0.0, ((q_stat - df_q) / q_stat) * 100.0) if q_stat > 0 else 0.0

    # 3. Random Effects Model
    w_re = 1.0 / (se**2 + tau2)
    w_re_pct = (w_re / np.sum(w_re)) * 100.0
    sum_w_re = np.sum(w_re)

    theta_re = float(np.sum(w_re * theta) / sum_w_re)
    se_re_standard = math.sqrt(1.0 / sum_w_re)

    # Hartung-Knapp-Sidik-Jonkman (Modified HKSJ) adjustment
    if use_hksj and k >= 3:
        q_hksj = float((1.0 / (k - 1)) * np.sum(w_re * ((theta - theta_re) ** 2)))
        hksj_scale = max(1.0, q_hksj)
        se_re = math.sqrt(hksj_scale / sum_w_re)
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=k - 1)
        ci_re_low = theta_re - t_crit * se_re
        ci_re_high = theta_re + t_crit * se_re
        t_val = theta_re / se_re if se_re > 0 else 0
        p_val_re = float(2.0 * (1.0 - stats.t.cdf(abs(t_val), df=k - 1)))
        method_name = f"Random Effects ({method_label} + Modified Hartung-Knapp)"
    else:
        se_re = se_re_standard
        ci_re_low = theta_re - z_crit * se_re
        ci_re_high = theta_re + z_crit * se_re
        z_val = theta_re / se_re if se_re > 0 else 0
        p_val_re = float(2.0 * (1.0 - stats.norm.cdf(abs(z_val))))
        method_name = f"Random Effects ({method_label})"

    # 4. 95% Prediction Interval (for future trial setting)
    pi_low, pi_high = np.nan, np.nan
    if k >= 3:
        t_crit_pi = stats.t.ppf(1.0 - alpha / 2.0, df=max(1, k - 2))
        se_pi = math.sqrt(tau2 + se_re**2)
        pi_low = theta_re - t_crit_pi * se_pi
        pi_high = theta_re + t_crit_pi * se_pi

    # Transform back to natural scale if ratio (OR/RR)
    def disp(x: float) -> float:
        return math.exp(x) if is_ratio else x

    df["weight_fe_pct"] = w_fe_pct
    df["weight_re_pct"] = w_re_pct

    # Subgroup Analysis if available
    subgroups_res = {}
    if "subgroup" in df.columns and df["subgroup"].nunique() > 1:
        q_within = 0.0
        df_within = 0
        for sg_name, sg_data in df.groupby("subgroup"):
            if len(sg_data) >= 1:
                sg_meta = run_meta_analysis(
                    sg_data, method_re=method_re, use_hksj=False
                )
                subgroups_res[sg_name] = sg_meta
                if "heterogeneity" in sg_meta and not np.isnan(
                    sg_meta["heterogeneity"]["Q"]
                ):
                    q_within += sg_meta["heterogeneity"]["Q"]
                    df_within += max(0, len(sg_data) - 1)

        q_between = max(0.0, q_stat - q_within)
        df_between = df["subgroup"].nunique() - 1
        p_between = (
            float(1.0 - stats.chi2.cdf(q_between, df_between))
            if df_between > 0
            else 1.0
        )
        subgroup_diff = {
            "Q_between": q_between,
            "df_between": df_between,
            "p_between": p_between,
            "subgroups": subgroups_res,
        }
    else:
        subgroup_diff = None

    return {
        "k": k,
        "is_ratio": is_ratio,
        "studies": df,
        "fixed_effect": {
            "log_effect": theta_fe,
            "effect_disp": disp(theta_fe),
            "se": se_fe,
            "ci_lower": disp(ci_fe_low),
            "ci_upper": disp(ci_fe_high),
            "z_value": z_val_fe,
            "p_value": p_val_fe,
        },
        "random_effect": {
            "method": method_name,
            "log_effect": theta_re,
            "effect_disp": disp(theta_re),
            "se": se_re,
            "ci_lower": disp(ci_re_low),
            "ci_upper": disp(ci_re_high),
            "p_value": p_val_re,
            "prediction_interval": {
                "pi_lower": disp(pi_low) if not np.isnan(pi_low) else np.nan,
                "pi_upper": disp(pi_high) if not np.isnan(pi_high) else np.nan,
            },
        },
        "heterogeneity": {
            "Q": q_stat,
            "df": df_q,
            "p_value": p_q,
            "I2": i2,
            "tau2": tau2,
            "tau": tau,
            "interpretation": (
                "Low Heterogeneity (I² < 25%)"
                if i2 < 25
                else "Moderate Heterogeneity (25-50%)"
                if i2 < 50
                else "Substantial Heterogeneity (50-75%)"
                if i2 < 75
                else "Considerable Heterogeneity (I² ≥ 75%)"
            ),
        },
        "subgroups": subgroup_diff,
    }


# =============================================================================
# PUBLICATION BIAS & SMALL STUDY EFFECTS
# =============================================================================


def run_publication_bias_tests(data: pd.DataFrame) -> dict[str, Any]:
    """
    Run Egger's linear regression test and Begg's rank test for publication bias.

    Args:
        data: DataFrame containing 'log_effect' and 'se'

    Returns:
        Dictionary with Egger's intercept, p-value, Begg's tau, and interpretation
    """
    try:
        df = data.dropna(subset=["log_effect", "se"]).copy()
        df = df[df["se"] > 0]
        k = len(df)
        if k < 3:
            return {
                "error": "Publication bias tests require at least 3 studies.",
                "k": k,
            }

        theta = df["log_effect"].to_numpy()
        se = df["se"].to_numpy()

        # Egger's Linear Regression: (theta / se) ~ (1 / se)
        std_effect = theta / se
        precision = 1.0 / se

        import statsmodels.api as sm

        X = sm.add_constant(precision)
        egger_model = sm.OLS(std_effect, X).fit()

        intercept = float(egger_model.params[0])
        intercept_p = float(egger_model.pvalues[0])
        intercept_ci = egger_model.conf_int()[0].tolist()

        # Begg's Rank Correlation Test (Kendall's tau between standardized effect and variance)
        tau_val, tau_p = stats.kendalltau(std_effect, se**2)

        interpretation = (
            "⚠️ Potential Publication Bias detected (Egger p < 0.05)"
            if intercept_p < 0.05
            else "✅ No significant small-study asymmetry detected (Egger p ≥ 0.05)"
        )

        return {
            "k": k,
            "egger": {
                "intercept": intercept,
                "ci_lower": float(intercept_ci[0]),
                "ci_upper": float(intercept_ci[1]),
                "t_stat": float(egger_model.tvalues[0]),
                "p_value": intercept_p,
            },
            "begg": {
                "kendall_tau": float(tau_val) if not np.isnan(tau_val) else 0.0,
                "p_value": float(tau_p) if not np.isnan(tau_p) else 1.0,
            },
            "interpretation": interpretation,
        }
    except Exception as e:
        logger.error(f"Publication bias test failed: {e}")
        return {"error": str(e)}


# =============================================================================
# INTERACTIVE PLOTLY VISUALIZATIONS
# =============================================================================


def create_meta_forest_plot(
    meta_results: dict[str, Any],
    title: str = "Meta-Analysis Forest Plot",
    effect_label: str = "Odds Ratio (95% CI)",
) -> go.Figure:
    """
    Generate an interactive Plotly Forest Plot with study weights, summary diamond, and prediction interval.
    """
    if "error" in meta_results:
        fig = go.Figure()
        fig.add_annotation(text=meta_results["error"], x=0.5, y=0.5, showarrow=False)
        return fig

    df = meta_results["studies"].copy()
    is_ratio = meta_results.get("is_ratio", False)
    null_val = 1.0 if is_ratio else 0.0

    k = len(df)
    # Sort studies from top to bottom
    df = df.iloc[::-1].reset_index(drop=True)

    fig = go.Figure()

    # Vertical Reference Line (No effect)
    fig.add_vline(
        x=null_val,
        line=dict(color="#9CA3AF", width=1.5, dash="dash"),
    )

    y_positions = list(range(len(df)))
    study_names = df["study"].tolist()

    # 1. Individual Study Markers & Error Bars
    for idx, row in df.iterrows():
        est = row["effect_size"]
        ci_low = row["ci_lower"]
        ci_high = row["ci_upper"]
        wt_re = row.get("weight_re_pct", 100.0 / k)

        # Marker size scaled by Random Effects weight
        marker_size = max(6, min(18, 6 + (wt_re / (100.0 / k)) * 4))

        fig.add_trace(
            go.Scatter(
                x=[est],
                y=[idx],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="#1E3A5F",
                    symbol="square",
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[ci_high - est],
                    arrayminus=[est - ci_low],
                    color="#1E3A5F",
                    thickness=2,
                    width=4,
                ),
                name=str(row["study"]),
                hovertemplate=f"<b>{row['study']}</b><br>Estimate: {est:.2f} [{ci_low:.2f}, {ci_high:.2f}]<br>Weight: {wt_re:.1f}%<extra></extra>",
                showlegend=False,
            )
        )

    # 2. Add Pooled Diamonds
    # Fixed Effect Diamond
    fe = meta_results["fixed_effect"]
    fe_y = -1.2
    fe_est = fe["effect_disp"]
    fe_low = fe["ci_lower"]
    fe_high = fe["ci_upper"]

    fig.add_trace(
        go.Scatter(
            x=[fe_low, fe_est, fe_high, fe_est, fe_low],
            y=[fe_y, fe_y + 0.35, fe_y, fe_y - 0.35, fe_y],
            fill="toself",
            fillcolor="#60A5FA",
            line=dict(color="#2563EB", width=1.5),
            name=f"Fixed Effect: {fe_est:.2f} [{fe_low:.2f}, {fe_high:.2f}]",
            hovertemplate=f"<b>Fixed Effect Model</b><br>Estimate: {fe_est:.2f} [{fe_low:.2f}, {fe_high:.2f}]<br>p = {fe['p_value']:.4f}<extra></extra>",
        )
    )

    # Random Effect Diamond
    re = meta_results["random_effect"]
    re_y = -2.4
    re_est = re["effect_disp"]
    re_low = re["ci_lower"]
    re_high = re["ci_upper"]

    fig.add_trace(
        go.Scatter(
            x=[re_low, re_est, re_high, re_est, re_low],
            y=[re_y, re_y + 0.4, re_y, re_y - 0.4, re_y],
            fill="toself",
            fillcolor="#EF4444",
            line=dict(color="#B91C1C", width=2),
            name=f"Random Effects: {re_est:.2f} [{re_low:.2f}, {re_high:.2f}]",
            hovertemplate=f"<b>Random Effects Model ({re['method']})</b><br>Estimate: {re_est:.2f} [{re_low:.2f}, {re_high:.2f}]<br>p = {re['p_value']:.4f}<extra></extra>",
        )
    )

    # 3. 95% Prediction Interval Line
    pi_dict = re.get("prediction_interval", {})
    pi_low = pi_dict.get("pi_lower", np.nan)
    pi_high = pi_dict.get("pi_upper", np.nan)

    if not np.isnan(pi_low) and not np.isnan(pi_high):
        fig.add_trace(
            go.Scatter(
                x=[pi_low, pi_high],
                y=[re_y, re_y],
                mode="lines+markers",
                line=dict(color="#B91C1C", width=3, dash="dot"),
                marker=dict(symbol="line-ns", size=10, color="#B91C1C"),
                name=f"95% Prediction Interval: [{pi_low:.2f}, {pi_high:.2f}]",
                hovertemplate=f"<b>95% Prediction Interval</b><br>[{pi_low:.2f}, {pi_high:.2f}]<extra></extra>",
            )
        )

    # Heterogeneity Subtitle
    het = meta_results.get("heterogeneity", {})
    het_text = (
        f"Heterogeneity: I² = {het.get('I2', 0):.1f}%, τ² = {het.get('tau2', 0):.3f}, "
        f"Q = {het.get('Q', 0):.2f} (p = {het.get('p_value', 1.0):.3f})"
    )

    all_y = y_positions + [fe_y, re_y]
    all_labels = study_names + [
        "<b>Fixed Effect (IV)</b>",
        "<b>Random Effects (DL)</b>",
    ]

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: 11px; color: #4B5563;'>{het_text}</span>",
            x=0.5,
        ),
        xaxis=dict(
            title=effect_label,
            type="log" if is_ratio else "linear",
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=all_y,
            ticktext=all_labels,
            autorange="reversed",
            showgrid=True,
            gridcolor="#F3F4F6",
        ),
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E5E7EB",
            borderwidth=1,
        ),
        height=max(480, 200 + k * 35),
        margin=dict(l=160, r=40, t=70, b=80),
    )

    return fig


def create_contour_enhanced_funnel_plot(
    meta_results: dict[str, Any],
    title: str = "Contour-Enhanced Funnel Plot",
) -> go.Figure:
    """
    Generate an interactive Plotly Contour-Enhanced Funnel Plot.
    Shades regions of significance (p < 0.10, p < 0.05, p < 0.01) around null to test publication bias.
    """
    if "error" in meta_results:
        fig = go.Figure()
        fig.add_annotation(text=meta_results["error"], x=0.5, y=0.5, showarrow=False)
        return fig

    df = meta_results["studies"]
    is_ratio = meta_results.get("is_ratio", False)

    theta = df["log_effect"].to_numpy()
    se = df["se"].to_numpy()

    max_se = float(np.max(se)) * 1.25 if len(se) > 0 else 1.0
    se_grid = np.linspace(1e-4, max_se, 100)

    # Significance boundaries centered at 0 (null effect)
    z_90 = 1.645
    z_95 = 1.960
    z_99 = 2.576

    fig = go.Figure()

    def _add_band(z_inner: float, z_outer: float, color: str, label: str) -> None:
        """Shade the mirrored band z_inner*se <= |x| < z_outer*se."""
        if z_inner == 0.0:
            # Central non-significant region: |x| < z_outer * se
            x_left = -z_outer * se_grid
            x_right = z_outer * se_grid
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_left, x_right[::-1]]),
                    y=np.concatenate([se_grid, se_grid[::-1]]),
                    fill="toself",
                    fillcolor=color,
                    line=dict(color="rgba(0,0,0,0)"),
                    name=label,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        else:
            # Mirrored symmetric side bands (both negative and positive effect sides)
            for idx, sign in enumerate((-1.0, 1.0)):
                inner = sign * z_inner * se_grid
                outer = sign * z_outer * se_grid
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([outer, inner[::-1]]),
                        y=np.concatenate([se_grid, se_grid[::-1]]),
                        fill="toself",
                        fillcolor=color,
                        line=dict(color="rgba(0,0,0,0)"),
                        name=label,
                        showlegend=(idx == 0),
                        hoverinfo="skip",
                    )
                )

    # 1. Central non-significant region: |x| < z_90 * se -> p >= 0.10
    _add_band(0.0, z_90, "rgba(243, 244, 246, 0.8)", "p ≥ 0.10")

    # 2. Shaded Contour: 0.05 <= p < 0.10 (between z_90 and z_95)
    _add_band(z_90, z_95, "rgba(219, 234, 254, 0.7)", "0.05 ≤ p < 0.10")

    # 3. Shaded Contour: 0.01 <= p < 0.05 (between z_95 and z_99)
    _add_band(z_95, z_99, "rgba(191, 219, 254, 0.7)", "0.01 ≤ p < 0.05")

    # 4. Shaded Contour: p < 0.01 (beyond z_99 * se)
    z_outer = max(z_99 * 1.6, 4.0)
    _add_band(z_99, z_outer, "rgba(147, 197, 253, 0.5)", "p < 0.01")

    # Pooled summary line
    re_effect = float(meta_results["random_effect"]["log_effect"])
    fig.add_vline(
        x=re_effect,
        line=dict(color="#EF4444", width=2, dash="dash"),
        annotation_text="<b>Pooled Effect</b>",
        annotation_position="top left" if re_effect < 0 else "top right",
        annotation=dict(
            font=dict(color="#EF4444", size=11, family="Inter, sans-serif"),
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#EF4444",
            borderwidth=1,
            borderpad=3,
        ),
    )
    fig.add_vline(
        x=0.0,
        line=dict(color="#4B5563", width=1.5, dash="dot"),
        annotation_text="<b>Null Line (0)</b>",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(color="#4B5563", size=10, family="Inter, sans-serif"),
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#9CA3AF",
            borderwidth=1,
            borderpad=3,
        ),
    )

    # Study Points
    fig.add_trace(
        go.Scatter(
            x=theta,
            y=se,
            mode="markers",
            marker=dict(
                size=9,
                color="#1E3A5F",
                symbol="circle",
                line=dict(width=1.5, color="white"),
            ),
            text=df["study"],
            name="Studies",
            hovertemplate="<b>%{text}</b><br>Effect: %{x:.3f}<br>SE: %{y:.3f}<extra></extra>",
        )
    )

    # Publication bias test caption
    pb = run_publication_bias_tests(df)
    egger_text = ""
    if "egger" in pb:
        egger_text = f"Egger's test intercept: {pb['egger']['intercept']:.2f} (p = {pb['egger']['p_value']:.3f})"

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: 11px; color: #4B5563;'>{egger_text}</span>",
            x=0.5,
        ),
        xaxis=dict(
            title="Log Effect Size" if is_ratio else "Effect Size",
            zeroline=False,
        ),
        yaxis=dict(
            title="Standard Error (SE)",
            autorange="reversed",  # Inverted SE: smaller SE (higher precision) on top
            range=[max_se, 0],
        ),
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(255,255,255,0.85)"),
        height=480,
    )

    return fig
