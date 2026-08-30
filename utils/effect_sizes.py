"""
📊 Effect Size Calculations for Medical Research

Publication-ready effect size calculations following Cohen (1988) conventions
and international medical journal standards (NEJM, Lancet, JAMA, BMJ).

Functions:
    - cohens_d: Standardized mean difference with CI
    - hedges_g: Bias-corrected effect size for small samples (n < 50)
    - glass_delta: Effect size with control group SD only
    - eta_squared: Effect size for ANOVA
    - omega_squared: Less biased ANOVA effect size
    - interpret_effect_size: Cohen's convention interpretation

References:
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    Hedges, L.V. (1981). Distribution theory for Glass's estimator of effect size.
    Lakens, D. (2013). Calculating and reporting effect sizes. Frontiers in Psychology.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats

from config import COHEN_D_THRESHOLDS, ETA_SQUARED_THRESHOLDS
from logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# EFFECT SIZE INTERPRETATION (Cohen's Conventions)
# =============================================================================


def interpret_effect_size(
    value: float,
    metric: Literal["d", "eta_squared", "omega_squared"] = "d",
) -> str:
    """
    Interpret effect size magnitude using Cohen's conventions.

    Args:
        value: The effect size value (absolute value used for d)
        metric: Type of effect size ('d', 'eta_squared', 'omega_squared')

    Returns:
        Interpretation string: 'Negligible', 'Small', 'Medium', or 'Large'

    Examples:
        >>> interpret_effect_size(0.3, "d")
        'Small'
        >>> interpret_effect_size(0.10, "eta_squared")
        'Medium'
    """
    if not np.isfinite(value):
        return "Undefined"

    abs_val = abs(value)

    if metric == "d":
        if abs_val < COHEN_D_THRESHOLDS["negligible"]:
            return "Negligible"
        elif abs_val < COHEN_D_THRESHOLDS["small"]:
            return "Small"
        elif abs_val < COHEN_D_THRESHOLDS["medium"]:
            return "Medium"
        else:
            return "Large"
    elif metric in ("eta_squared", "omega_squared"):
        if abs_val < ETA_SQUARED_THRESHOLDS["small"]:
            return "Negligible"
        elif abs_val < ETA_SQUARED_THRESHOLDS["medium"]:
            return "Small"
        elif abs_val < ETA_SQUARED_THRESHOLDS["large"]:
            return "Medium"
        else:
            return "Large"
    else:
        return "Unknown metric"


# =============================================================================
# STANDARDIZED MEAN DIFFERENCE EFFECT SIZES
# =============================================================================


def cohens_d(
    group1: np.ndarray | list,
    group2: np.ndarray | list,
    pooled: bool = True,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Calculate Cohen's d with confidence intervals.

    Cohen's d is the standardized difference between two means.
    Recommended for comparing two groups of equal or similar size.

    Args:
        group1: Data for group 1 (e.g., treatment group)
        group2: Data for group 2 (e.g., control group)
        pooled: If True, use pooled SD (recommended). If False, use group2 SD only.
        alpha: Significance level for CI (default 0.05 for 95% CI)

    Returns:
        Dictionary containing:
            - d: Cohen's d value
            - ci_lower: Lower bound of CI
            - ci_upper: Upper bound of CI
            - se: Standard error
            - interpretation: Cohen's convention (Small/Medium/Large)
            - variance: Variance of d
            - n1, n2: Sample sizes

    Examples:
        >>> result = cohens_d([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        >>> print(f"d = {result['d']:.3f}")
        d = -0.632

    References:
        Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    try:
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)

        # Remove NaN values
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        n1, n2 = len(g1), len(g2)

        if n1 < 2 or n2 < 2:
            return {"error": "Each group must have at least 2 observations"}

        mean1, mean2 = np.mean(g1), np.mean(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

        # Pooled standard deviation (Hedges' pooled SD formula)
        if pooled:
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            sd_pooled = np.sqrt(pooled_var)
        else:
            # Use group2 (control) SD only (Glass's approach)
            sd_pooled = np.sqrt(var2)

        if np.isclose(sd_pooled, 0):
            logger.warning("Cohen's d: Pooled SD is zero")
            return {"error": "Standard deviation is zero - cannot compute effect size"}

        # Cohen's d
        d = (mean1 - mean2) / sd_pooled

        # Standard error and CI using non-central t approximation
        # Hedges & Olkin (1985) variance formula
        se = np.sqrt((n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2)))
        variance = se**2

        # CI using normal approximation
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = d - z_crit * se
        ci_upper = d + z_crit * se

        interpretation = interpret_effect_size(d, "d")

        return {
            "d": round(d, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "se": round(se, 4),
            "variance": round(variance, 6),
            "interpretation": interpretation,
            "n1": n1,
            "n2": n2,
            "mean_diff": round(mean1 - mean2, 4),
            "pooled_sd": round(sd_pooled, 4),
            # Return raw values for internal use (e.g. hedges_g)
            "_d_raw": d,
            "_se_raw": se,
        }

    except Exception as e:
        logger.exception("Cohen's d calculation failed")
        return {"error": f"Calculation failed: {e!s}"}


def hedges_g(
    group1: np.ndarray | list,
    group2: np.ndarray | list,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Calculate Hedges' g with confidence intervals.

    Hedges' g is a bias-corrected version of Cohen's d, recommended
    when sample sizes are small (n < 50 per group).

    The correction factor J approximately equals: 1 - 3/(4*(n1+n2-2) - 1)

    Args:
        group1: Data for group 1
        group2: Data for group 2
        alpha: Significance level for CI

    Returns:
        Dictionary containing:
            - g: Hedges' g (bias-corrected)
            - ci_lower, ci_upper: Confidence interval
            - d_uncorrected: Original Cohen's d before correction
            - correction_factor: The J correction applied
            - interpretation: Effect size magnitude
            - recommendation: Whether correction was meaningful

    Examples:
        >>> result = hedges_g([1, 2, 3], [4, 5, 6])  # Small sample
        >>> print(result['recommendation'])
        'Hedges' g recommended (small sample)'

    References:
        Hedges, L.V. (1981). Distribution theory for Glass's estimator.
    """
    try:
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)

        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        n1, n2 = len(g1), len(g2)

        if n1 < 2 or n2 < 2:
            return {"error": "Each group must have at least 2 observations"}

        # First calculate Cohen's d
        d_result = cohens_d(g1, g2, pooled=True, alpha=alpha)
        if "error" in d_result:
            return d_result

        # Use raw values if available to avoid compounding rounding errors
        d = d_result.get("_d_raw", d_result["d"])

        # Hedges' correction factor J
        # Exact formula: J = gamma((n1+n2-2)/2) / (sqrt((n1+n2-2)/2) * gamma((n1+n2-3)/2))
        # Approximation (Hedges, 1981):
        df = n1 + n2 - 2
        j = 1 - (3 / (4 * df - 1))

        # Apply correction
        g = d * j

        # Corrected variance
        se_d = d_result.get("_se_raw", d_result["se"])
        se_g = se_d * j
        variance = se_g**2

        # CI
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = g - z_crit * se_g
        ci_upper = g + z_crit * se_g

        interpretation = interpret_effect_size(g, "d")

        # Recommendation based on sample size
        total_n = n1 + n2
        if total_n < 50:
            recommendation = "Hedges' g recommended (small sample)"
        else:
            recommendation = "Cohen's d and Hedges' g similar for large samples"

        return {
            "g": round(g, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "se": round(se_g, 4),
            "variance": round(variance, 6),
            "d_uncorrected": round(d, 4),
            "correction_factor": round(j, 4),
            "interpretation": interpretation,
            "recommendation": recommendation,
            "n1": n1,
            "n2": n2,
        }

    except Exception as e:
        logger.exception("Hedges' g calculation failed")
        return {"error": f"Calculation failed: {e!s}"}


def glass_delta(
    treatment: np.ndarray | list,
    control: np.ndarray | list,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Calculate Glass's Δ (delta) effect size.

    Glass's Δ uses only the control group's standard deviation,
    appropriate when treatment may affect variability.

    Args:
        treatment: Data from the treatment/experimental group
        control: Data from the control group (SD used as denominator)
        alpha: Significance level for CI

    Returns:
        Dictionary with delta, CI, and interpretation

    Examples:
        >>> result = glass_delta([5, 6, 7, 8], [1, 2, 3, 4])
        >>> print(f"Δ = {result['delta']:.2f}")

    References:
        Glass, G.V. (1976). Primary, secondary, and meta-analysis.
    """
    try:
        t = np.asarray(treatment, dtype=float)
        c = np.asarray(control, dtype=float)

        t = t[~np.isnan(t)]
        c = c[~np.isnan(c)]

        n_t, n_c = len(t), len(c)

        if n_t < 2 or n_c < 2:
            return {"error": "Each group must have at least 2 observations"}

        mean_t, mean_c = np.mean(t), np.mean(c)
        sd_control = np.std(c, ddof=1)

        if np.isclose(sd_control, 0):
            return {"error": "Control group SD is zero"}

        delta = (mean_t - mean_c) / sd_control

        # Variance approximation
        # Hedges & Olkin approximation for Glass's delta SE
        se = np.sqrt(1 / n_t + 1 / n_c + (delta**2) / (2 * n_c))

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = delta - z_crit * se
        ci_upper = delta + z_crit * se

        return {
            "delta": round(delta, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "se": round(se, 4),
            "interpretation": interpret_effect_size(delta, "d"),
            "control_sd": round(sd_control, 4),
            "n_treatment": n_t,
            "n_control": n_c,
        }

    except Exception as e:
        logger.exception("Glass's delta calculation failed")
        return {"error": f"Calculation failed: {e!s}"}


# =============================================================================
# ANOVA EFFECT SIZES
# =============================================================================


def eta_squared(
    ss_effect: float,
    ss_total: float,
) -> dict[str, Any]:
    """
    Calculate η² (eta-squared) for ANOVA.

    η² = SS_effect / SS_total

    Note: η² tends to overestimate population effect size.
    Consider using omega_squared for less biased estimate.

    Args:
        ss_effect: Sum of squares for the effect (between-groups SS)
        ss_total: Total sum of squares

    Returns:
        Dictionary with eta_squared, interpretation

    Examples:
        >>> result = eta_squared(120, 500)
        >>> print(f"η² = {result['eta_squared']:.3f}")  # 0.240
    """
    try:
        if ss_total <= 0 or np.isclose(ss_total, 0):
            return {"error": "Total SS must be positive"}

        eta_sq = ss_effect / ss_total

        if not 0 <= eta_sq <= 1:
            logger.warning("η² outside [0, 1] range: %.4f", eta_sq)

        return {
            "eta_squared": round(eta_sq, 4),
            "interpretation": interpret_effect_size(eta_sq, "eta_squared"),
            "percentage_variance": round(eta_sq * 100, 2),
            "note": "η² may overestimate population effect; consider ω²",
        }

    except Exception as e:
        logger.exception("Eta squared calculation failed")
        return {"error": f"Calculation failed: {e!s}"}


def omega_squared(
    ss_effect: float,
    ss_error: float,
    ms_error: float,
    df_effect: int,
    n_total: int,
) -> dict[str, Any]:
    """
    Calculate ω² (omega-squared) for ANOVA.

    ω² is a less biased estimator of population effect size than η².

    Formula: ω² = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)

    Args:
        ss_effect: Sum of squares for the effect
        ss_error: Sum of squares for error (within-groups)
        ms_error: Mean square error
        df_effect: Degrees of freedom for the effect
        n_total: Total sample size

    Returns:
        Dictionary with omega_squared and interpretation

    References:
        Olejnik, S. & Algina, J. (2003). Generalized eta and omega squared statistics.
    """
    try:
        ss_total = ss_effect + ss_error

        if ss_total <= 0 or np.isclose(ss_total, 0) or ms_error < 0:
            return {"error": "Invalid SS or MS values"}

        # ω² formula
        numerator = ss_effect - df_effect * ms_error
        denominator = ss_total + ms_error

        omega_sq = numerator / denominator

        # ω² can be negative for very small effects; floor at 0
        omega_sq = max(0, omega_sq)

        return {
            "omega_squared": round(omega_sq, 4),
            "interpretation": interpret_effect_size(omega_sq, "omega_squared"),
            "percentage_variance": round(omega_sq * 100, 2),
            "note": "ω² is less biased than η² for population inference",
        }

    except Exception as e:
        logger.exception("Omega squared calculation failed")
        return {"error": f"Calculation failed: {e!s}"}


def partial_eta_squared(
    ss_effect: float,
    ss_error: float,
) -> dict[str, Any]:
    """
    Calculate partial η² for factorial ANOVA designs.

    Partial η² = SS_effect / (SS_effect + SS_error)

    Args:
        ss_effect: Sum of squares for the specific effect
        ss_error: Sum of squares for error

    Returns:
        Dictionary with partial_eta_squared and interpretation
    """
    try:
        denominator = ss_effect + ss_error

        if denominator <= 0 or np.isclose(denominator, 0):
            return {"error": "SS values must be positive"}

        partial_eta_sq = ss_effect / denominator

        return {
            "partial_eta_squared": round(partial_eta_sq, 4),
            "interpretation": interpret_effect_size(partial_eta_sq, "eta_squared"),
            "percentage_variance": round(partial_eta_sq * 100, 2),
        }

    except Exception as e:
        logger.exception("Partial eta squared calculation failed")
        return {"error": f"Calculation failed: {e!s}"}


# =============================================================================
# PUBLICATION FORMATTING HELPER
# =============================================================================


def format_effect_size_for_publication(
    result: dict[str, Any],
    metric: str = "d",
    style: Literal["NEJM", "JAMA", "Lancet", "APA"] = "APA",
) -> str:
    """
    Format effect size result for publication.

    Args:
        result: Dictionary from cohens_d, hedges_g, etc.
        metric: Type of effect size ('d', 'g', 'delta', 'eta', 'omega')
        style: Publication style

    Returns:
        Formatted string ready for manuscript

    Examples:
        >>> result = cohens_d([1,2,3], [4,5,6])
        >>> print(format_effect_size_for_publication(result, "d", "NEJM"))
        'd = -1.89 (95% CI, -3.54 to -0.24)'
    """
    if "error" in result:
        return f"Effect size calculation failed: {result['error']}"

    # Get the main value
    if metric == "d":
        value = result.get("d", result.get("g", result.get("delta")))
        symbol = "d"
    elif metric == "g":
        value = result.get("g")
        symbol = "g"
    elif metric == "delta":
        value = result.get("delta")
        symbol = "Δ"
    elif metric in ("eta", "eta_squared"):
        value = result.get("eta_squared", result.get("partial_eta_squared"))
        symbol = "η²"
    elif metric in ("omega", "omega_squared"):
        value = result.get("omega_squared")
        symbol = "ω²"
    else:
        symbol = metric
        value = result.get(metric)

    ci_low = result.get("ci_lower")
    ci_high = result.get("ci_upper")

    if value is None:
        return "Effect size not available"

    # Format based on style
    if style == "NEJM":
        if ci_low is not None and ci_high is not None:
            return f"{symbol} = {value:.2f} (95% CI, {ci_low:.2f} to {ci_high:.2f})"
        return f"{symbol} = {value:.2f}"

    elif style == "JAMA":
        if ci_low is not None and ci_high is not None:
            return f"{symbol} = {value:.2f} (95% CI, {ci_low:.2f}-{ci_high:.2f})"
        return f"{symbol} = {value:.2f}"

    elif style == "Lancet":
        if ci_low is not None and ci_high is not None:
            return f"{symbol} {value:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})"
        return f"{symbol} = {value:.2f}"

    else:  # APA
        if ci_low is not None and ci_high is not None:
            return f"{symbol} = {value:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]"
        return f"{symbol} = {value:.2f}"
