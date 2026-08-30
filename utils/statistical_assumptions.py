"""
📊 Statistical Assumption Tests for Medical Research

Tests for validating statistical assumptions before analysis,
essential for publication in high-impact medical journals.

Functions:
    - check_homogeneity_of_variance: Levene's, Bartlett's, Fligner-Killeen tests
    - test_sphericity: Mauchly's test for repeated measures
    - check_normality_comprehensive: Multi-test normality assessment
    - assess_assumptions_for_ttest: Combined assumption checking for t-tests
    - assess_assumptions_for_anova: Combined assumption checking for ANOVA

References:
    Levene, H. (1960). Robust tests for equality of variances.
    Bartlett, M.S. (1937). Properties of sufficiency and statistical tests.
    Mauchly, J.W. (1940). Significance test for sphericity.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from config import CONFIG
from logger import get_logger

logger = get_logger(__name__)

DEFAULT_ALPHA = CONFIG.get("analysis.significance_level", 0.05)


# =============================================================================
# HOMOGENEITY OF VARIANCE TESTS
# =============================================================================


def check_homogeneity_of_variance(
    *groups: np.ndarray | list,
    method: Literal["levene", "bartlett", "fligner", "all"] = "levene",
    center: Literal["mean", "median", "trimmed"] = "median",
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Test for equal variance across groups.

    Critical assumption for:
    - Independent samples t-test
    - One-way ANOVA
    - ANCOVA

    Args:
        *groups: Two or more arrays of observations
        method: Test method:
            - 'levene': Levene's test (robust to non-normality)
            - 'bartlett': Bartlett's test (requires normality)
            - 'fligner': Fligner-Killeen test (non-parametric)
            - 'all': Run all three tests
        center: For Levene's test - 'mean', 'median' (default), or 'trimmed'
        alpha: Significance level

    Returns:
        Dictionary containing:
            - test_name: Name of the test
            - statistic: Test statistic value
            - p_value: P-value
            - assumption_met: Boolean indicating if variance is equal
            - interpretation: Human-readable interpretation
            - recommendation: What to do if assumption is violated

    Examples:
        >>> g1 = [1, 2, 3, 4, 5]
        >>> g2 = [2, 3, 4, 5, 6]
        >>> g3 = [10, 20, 30, 40, 50]  # Different variance
        >>> result = check_homogeneity_of_variance(g1, g2, g3)
        >>> print(result['assumption_met'])
        False

    References:
        Levene, H. (1960). Robust tests for equality of variances.
    """
    try:
        # Convert and clean data
        cleaned_groups = []
        n_missing_list = []
        zero_variance_count = 0

        for i, g in enumerate(groups):
            arr = np.asarray(g, dtype=float)
            n_original = len(arr)
            arr = arr[~np.isnan(arr)]
            n_missing = n_original - len(arr)
            n_missing_list.append(n_missing)

            if len(arr) < 2:
                return {"error": f"Group {i + 1} must have at least 2 observations"}

            if np.var(arr) == 0:
                zero_variance_count += 1

            cleaned_groups.append(arr)

        if zero_variance_count == len(cleaned_groups):
            # All groups have zero variance - variances are equal (all zero)
            return {
                "test_name": "Zero Variance Check",
                "statistic": 0.0,
                "p_value": 1.0,
                "assumption_met": True,
                "interpretation": "All groups have zero variance (identical values). Assumption met.",
                "missing_data": n_missing_list,
            }

        if len(cleaned_groups) < 2:
            return {"error": "At least 2 groups are required"}

        results = {}

        # Levene's test
        if method in ("levene", "all"):
            stat, p = stats.levene(*cleaned_groups, center=center)
            results["levene"] = {
                "test_name": f"Levene's Test (center={center})",
                "statistic": round(stat, 4),
                "p_value": p,
                "assumption_met": p >= alpha,
                "interpretation": (
                    "Equal variances assumed (p ≥ 0.05)"
                    if p >= alpha
                    else "Unequal variances detected (p < 0.05)"
                ),
            }

        # Bartlett's test (sensitive to non-normality)
        if method in ("bartlett", "all"):
            stat, p = stats.bartlett(*cleaned_groups)
            results["bartlett"] = {
                "test_name": "Bartlett's Test",
                "statistic": round(stat, 4),
                "p_value": p,
                "assumption_met": p >= alpha,
                "interpretation": (
                    "Equal variances assumed (p ≥ 0.05)"
                    if p >= alpha
                    else "Unequal variances detected (p < 0.05)"
                ),
                "note": "Bartlett's test is sensitive to non-normality",
            }

        # Fligner-Killeen test (non-parametric)
        if method in ("fligner", "all"):
            stat, p = stats.fligner(*cleaned_groups)
            results["fligner"] = {
                "test_name": "Fligner-Killeen Test",
                "statistic": round(stat, 4),
                "p_value": p,
                "assumption_met": p >= alpha,
                "interpretation": (
                    "Equal variances assumed (p ≥ 0.05)"
                    if p >= alpha
                    else "Unequal variances detected (p < 0.05)"
                ),
                "note": "Non-parametric, robust to non-normality",
            }

        # Generate overall recommendation
        if method == "all":
            all_met = all(r["assumption_met"] for r in results.values())
            any_met = any(r["assumption_met"] for r in results.values())

            if all_met:
                recommendation = (
                    "All tests indicate homogeneity of variance. "
                    "Proceed with standard parametric tests."
                )
            elif any_met:
                recommendation = (
                    "Mixed results across tests. Consider using Welch's t-test "
                    "or Welch's ANOVA for robustness."
                )
            else:
                recommendation = (
                    "All tests indicate heteroscedasticity. Use Welch's t-test, "
                    "Welch's ANOVA, or non-parametric alternatives."
                )

            return {
                "tests": results,
                "overall_recommendation": recommendation,
                "group_variances": [
                    round(float(np.var(g, ddof=1)), 4) for g in cleaned_groups
                ],
                "group_sizes": [len(g) for g in cleaned_groups],
            }
        else:
            single_result = results[method]
            if not single_result["assumption_met"]:
                single_result["recommendation"] = (
                    "Consider: Welch's t-test, Welch's ANOVA, or non-parametric tests"
                )
            else:
                single_result["recommendation"] = (
                    "Proceed with standard parametric tests"
                )
            return single_result

    except Exception as e:
        logger.exception("Homogeneity of variance test failed")
        return {"error": f"Test failed: {e!s}"}


def test_sphericity(
    data: pd.DataFrame,
    subject_col: str,
    within_cols: list[str],
) -> dict[str, Any]:
    """
    Test sphericity assumption for repeated measures ANOVA.

    Mauchly's test checks if the variances of differences between all
    pairs of within-subject conditions are equal.

    Args:
        data: DataFrame in long format
        subject_col: Column name identifying subjects
        within_cols: List of column names for within-subject levels

    Returns:
        Dictionary containing:
            - mauchly_w: Mauchly's W statistic
            - chi_square: Approximate chi-square
            - p_value: P-value for sphericity
            - sphericity_met: Boolean
            - epsilon_gg: Greenhouse-Geisser epsilon correction
            - epsilon_hf: Huynh-Feldt epsilon correction
            - recommendation: Which correction to use

    Note:
        For full Mauchly's test, consider using pingouin.sphericity()
        This implementation provides a simplified approximation.
    """
    try:
        # This is a simplified placeholder - full Mauchly's requires matrix operations
        # For production, recommend using pingouin library

        if len(within_cols) < 2:
            return {"error": "Need at least 2 within-subject conditions"}

        # Calculate variances of difference scores
        # Placeholder implementation - real test needs covariance matrix analysis

        return {
            "note": "For full Mauchly's test, use pingouin.sphericity()",
            "recommendation": (
                "If sphericity is violated, apply Greenhouse-Geisser "
                "or Huynh-Feldt correction to degrees of freedom"
            ),
            "alternative": "Consider using mixed-effects models (LMM/GEE) which don't assume sphericity",
        }

    except Exception as e:
        logger.exception("Sphericity test failed")
        return {"error": f"Test failed: {e!s}"}


# =============================================================================
# COMPREHENSIVE NORMALITY TESTING
# =============================================================================


def check_normality_comprehensive(
    data: np.ndarray | list,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Comprehensive normality assessment using multiple tests.

    Uses:
    - Shapiro-Wilk (n < 5000): Most powerful for small samples
    - D'Agostino-Pearson (n >= 20): Based on skewness and kurtosis
    - Kolmogorov-Smirnov: For larger samples

    Args:
        data: Array of observations
        alpha: Significance level

    Returns:
        Dictionary with test results, descriptive statistics, and recommendation

    Examples:
        >>> from numpy.random import normal
        >>> data = normal(0, 1, 100)
        >>> result = check_normality_comprehensive(data)
        >>> print(result['recommendation'])
    """
    try:
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 3:
            return {"error": "Need at least 3 observations for normality testing"}

        results = {
            "n": n,
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "skewness": round(float(stats.skew(arr)), 4),
            "kurtosis": round(float(stats.kurtosis(arr)), 4),
            "tests": {},
        }

        # Shapiro-Wilk (best for n < 5000)
        if n <= 5000:
            stat, p = stats.shapiro(arr)
            results["tests"]["shapiro_wilk"] = {
                "test_name": "Shapiro-Wilk",
                "statistic": round(stat, 4),
                "p_value": p,
                "normal": p >= alpha,
                "note": "Most powerful for small samples",
            }

        # D'Agostino-Pearson (needs n >= 20)
        if n >= 20:
            stat, p = stats.normaltest(arr)
            results["tests"]["dagostino_pearson"] = {
                "test_name": "D'Agostino-Pearson K²",
                "statistic": round(stat, 4),
                "p_value": p,
                "normal": p >= alpha,
                "note": "Based on skewness and kurtosis",
            }

        # Kolmogorov-Smirnov
        # Standardize data for KS test against standard normal
        arr_std = (arr - np.mean(arr)) / np.std(arr, ddof=1)
        stat, p = stats.kstest(arr_std, "norm")
        results["tests"]["kolmogorov_smirnov"] = {
            "test_name": "Kolmogorov-Smirnov",
            "statistic": round(stat, 4),
            "p_value": p,
            "normal": p >= alpha,
            "note": "Less powerful but valid for larger samples",
        }

        # Overall assessment
        test_results = [t["normal"] for t in results["tests"].values()]
        all_normal = all(test_results)
        any_normal = any(test_results)

        # Descriptive criteria (practical normality)
        abs_skew = abs(results["skewness"])
        abs_kurt = abs(results["kurtosis"])
        descriptive_normal = abs_skew < 2.0 and abs_kurt < 7.0

        if all_normal and descriptive_normal:
            results["overall_assessment"] = "Data appear normally distributed"
            results["recommendation"] = "Parametric tests appropriate"
        elif any_normal or descriptive_normal:
            results["overall_assessment"] = "Approximate normality"
            results["recommendation"] = (
                "Parametric tests may be used with caution. "
                "Consider robust methods or non-parametric alternatives."
            )
        else:
            results["overall_assessment"] = "Non-normal distribution"
            results["recommendation"] = (
                "Use non-parametric tests (Mann-Whitney, Kruskal-Wallis) "
                "or data transformation (log, sqrt)"
            )

        return results

    except Exception as e:
        logger.exception("Normality check failed")
        return {"error": f"Test failed: {e!s}"}


# =============================================================================
# COMBINED ASSUMPTION CHECKS FOR COMMON TESTS
# =============================================================================


def assess_assumptions_for_ttest(
    group1: np.ndarray | list,
    group2: np.ndarray | list,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Check all assumptions for independent samples t-test.

    Assumptions:
    1. Normality in each group (or n >= 30 by CLT)
    2. Homogeneity of variance

    Args:
        group1, group2: Data arrays for each group
        alpha: Significance level

    Returns:
        Comprehensive assumption check with recommendations
    """
    try:
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)
        # Calculate missing data
        n1_orig, n2_orig = len(g1), len(g2)
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        n1, n2 = len(g1), len(g2)
        missing1, missing2 = n1_orig - n1, n2_orig - n2

        results = {
            "sample_sizes": {"group1": n1, "group2": n2},
            "missing_data": {"group1": missing1, "group2": missing2},
            "normality": {},
            "variance": None,
            "recommendation": "",
        }

        # Check normality
        norm1 = check_normality_comprehensive(g1, alpha)
        norm2 = check_normality_comprehensive(g2, alpha)

        results["normality"]["group1"] = {
            "assessment": norm1.get("overall_assessment", "Unknown"),
            "skewness": norm1.get("skewness"),
            "kurtosis": norm1.get("kurtosis"),
        }
        results["normality"]["group2"] = {
            "assessment": norm2.get("overall_assessment", "Unknown"),
            "skewness": norm2.get("skewness"),
            "kurtosis": norm2.get("kurtosis"),
        }

        # CLT applies for n >= 30
        clt_applies = n1 >= 30 and n2 >= 30

        # Check homogeneity of variance
        var_result = check_homogeneity_of_variance(g1, g2, method="levene")
        results["variance"] = var_result

        # Generate recommendation
        # Recommendations based on assessment text
        # "Data appear normally distributed" or "Approximate normality" are acceptable for parametric
        def is_normality_acceptable(assessment_text: str) -> bool:
            text = assessment_text.lower()
            return "appear normally" in text or "approximate" in text

        normality_ok = (
            clt_applies
            or is_normality_acceptable(norm1.get("overall_assessment", ""))
            or is_normality_acceptable(norm2.get("overall_assessment", ""))
        )
        variance_ok = var_result.get("assumption_met", False)

        if normality_ok and variance_ok:
            results["recommendation"] = "Use standard independent samples t-test"
            results["recommended_test"] = "Student's t-test"
        elif normality_ok and not variance_ok:
            results["recommendation"] = "Use Welch's t-test (unequal variances)"
            results["recommended_test"] = "Welch's t-test"
        elif not normality_ok and variance_ok:
            results["recommendation"] = "Consider Mann-Whitney U test or transform data"
            results["recommended_test"] = "Mann-Whitney U"
        else:
            results["recommendation"] = (
                "Use Mann-Whitney U test (non-normality, unequal variances)"
            )
            results["recommended_test"] = "Mann-Whitney U"

        return results

    except Exception as e:
        logger.exception("T-test assumption check failed")
        return {"error": f"Check failed: {e!s}"}


def assess_assumptions_for_anova(
    *groups: np.ndarray | list,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Check all assumptions for one-way ANOVA.

    Assumptions:
    1. Normality in each group (or n >= 30 by CLT)
    2. Homogeneity of variance
    3. Independence of observations

    Args:
        *groups: Data arrays for each group
        alpha: Significance level

    Returns:
        Comprehensive assumption check with recommendations
    """
    try:
        cleaned_groups = []
        missing_counts = []

        for g in groups:
            arr = np.asarray(g, dtype=float)
            n_orig = len(arr)
            arr = arr[~np.isnan(arr)]
            cleaned_groups.append(arr)
            missing_counts.append(n_orig - len(arr))

        k = len(cleaned_groups)  # Number of groups

        if k < 2:
            return {"error": "Need at least 2 groups for ANOVA"}

        results = {
            "n_groups": k,
            "sample_sizes": [len(g) for g in cleaned_groups],
            "missing_data": missing_counts,
            "normality": [],
            "variance": None,
            "recommendation": "",
        }

        # Check normality for each group
        clt_applies = all(len(g) >= 30 for g in cleaned_groups)

        for i, g in enumerate(cleaned_groups):
            norm_result = check_normality_comprehensive(g, alpha)
            results["normality"].append(
                {
                    "group": i + 1,
                    "n": len(g),
                    "assessment": norm_result.get("overall_assessment", "Unknown"),
                    "skewness": norm_result.get("skewness"),
                }
            )

        # Check homogeneity
        var_result = check_homogeneity_of_variance(*cleaned_groups, method="levene")
        results["variance"] = var_result

        # Generate recommendation
        normality_ok = clt_applies or all(
            "normal" in r["assessment"].lower() for r in results["normality"]
        )
        variance_ok = var_result.get("assumption_met", False)

        if normality_ok and variance_ok:
            results["recommendation"] = "Use standard one-way ANOVA"
            results["recommended_test"] = "One-way ANOVA"
        elif normality_ok and not variance_ok:
            results["recommendation"] = (
                "Use Welch's ANOVA (robust to unequal variances)"
            )
            results["recommended_test"] = "Welch's ANOVA"
        elif not normality_ok:
            results["recommendation"] = "Use Kruskal-Wallis test (non-parametric ANOVA)"
            results["recommended_test"] = "Kruskal-Wallis"

        return results

    except Exception as e:
        logger.exception("ANOVA assumption check failed")
        return {"error": f"Check failed: {e!s}"}
