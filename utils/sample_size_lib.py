"""
Sample Size and Power Calculation Library
Implements computations for Study Design:
- Means (T-test)
- Proportions (Chi-sq/Z-test)
- Survival (Log-rank/Cox)
- Correlation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.stats.power as smp
import statsmodels.stats.proportion as smprop
from scipy import stats


def calculate_power_means(
    n1: float,
    n2: float | None,
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Calculate Power for Two Independent Means (T-test).
    Returns power (0.0 - 1.0).
    """
    try:
        # pooled sd
        # Cohen's d
        if n2 is None:
            n2 = n1

        # Validation
        if n1 <= 1 or n2 <= 1:
            return np.nan
        if sd1 <= 0 or sd2 <= 0:
            return np.nan

        # Weighted pooled SD
        # If n1+n2=2, div by zero in (n1+n2-2). But checked n1>1, n2>1 so denom >= 2.
        sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

        if sd_pooled == 0:
            return np.nan

        effect_size = abs(mean1 - mean2) / sd_pooled
        ratio = n2 / n1

        analysis = smp.TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect_size,
            nobs1=n1,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative,
        )
        return float(power)
    except Exception:
        return np.nan


def calculate_sample_size_means(
    power: float,
    ratio: float,
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, float | str]:
    """
    Calculate Sample Size for Two Independent Means.
    Returns dictionary with n1, n2, and total_n, or error message.
    """
    try:
        # Initial guess for pooled SD (assuming equal N for SD calculation approx, refined by solver)
        # Actually, we need sd_pooled to get effect size.
        # Approx: sd_pooled ~= sqrt((sd1^2 + sd2^2)/2) for planning
        if sd1 <= 0 or sd2 <= 0:
            return {"error": "Standard deviations must be positive."}

        sd_pooled = np.sqrt((sd1**2 + sd2**2) / 2)

        if sd_pooled == 0:
            return {"error": "Pooled standard deviation is zero."}

        effect_size = abs(mean1 - mean2) / sd_pooled

        if effect_size == 0:
            return {"error": "Effect size is zero (means are equal)."}

        analysis = smp.TTestIndPower()
        n1 = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative,
        )
        n2 = n1 * ratio
        return {
            "n1": np.ceil(n1),
            "n2": np.ceil(n2),
            "total": np.ceil(n1) + np.ceil(n2),
        }
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}


def calculate_power_proportions(
    n1: float,
    n2: float | None,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Calculate Power for Two Proportions (Z-test/Chi-sq approx).
    """
    try:
        if n2 is None:
            n2 = n1

        if n1 <= 0 or n2 <= 0:
            return np.nan
        if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
            return np.nan

        ratio = n2 / n1

        # Effect size (h)
        effect_size = smprop.proportion_effectsize(p1, p2)

        # Using GofChisquarePower or NormalIndPower
        # NormalIndPower is standard for 2 proportions z-test
        analysis = smp.NormalIndPower()
        power = analysis.solve_power(
            effect_size=effect_size,
            nobs1=n1,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative,
        )
        return float(power)
    except Exception:
        return np.nan


def calculate_sample_size_proportions(
    power: float,
    ratio: float,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, float | str]:
    """
    Calculate Sample Size for Two Proportions.
    """
    try:
        if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
            return {"error": "Proportions must be between 0 and 1."}

        effect_size = smprop.proportion_effectsize(p1, p2)

        if effect_size == 0:
            return {"error": "Effect size is zero (proportions are equal)."}

        analysis = smp.NormalIndPower()
        n1 = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative,
        )
        n2 = n1 * ratio
        return {
            "n1": np.ceil(n1),
            "n2": np.ceil(n2),
            "total": np.ceil(n1) + np.ceil(n2),
        }
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}


def calculate_sample_size_survival(
    power: float,
    ratio: float,
    h0: float,  # Hazard Ratio or Median Survival 1
    h1: float,  # Median Survival 2 (if using medians) or just HR if param2 is None
    alpha: float = 0.05,
    mode: str = "hr",  # "hr" for Hazard Ratio input, "median" for Medians input
) -> dict[str, float | str]:
    """
    Calculate Sample Size for Log-Rank Test (Survival).
    Freedman's method is commonly used.
    Total Events (E) = ((z_alpha + z_beta)^2 * (1 + ratio*HR)^2) / ( (1-ratio)*HR )?

    Actually simple formula for Total Events (E):
    E = 4 * (z_alpha + z_beta)^2 / ln(HR)^2   (for 1:1 ratio)
    Adjusted for Ratio k=n2/n1:
    E = (z_alpha + z_beta)^2 * ( (1+k)^2 / (k * ln(HR)^2) )
    """
    try:
        if mode == "median":
            # H0 = m1, H1 = m2. HR = m1/m2 (assuming exp dist)
            if h1 == 0:
                return {"error": "Median survival 2 cannot be zero."}
            hr = h0 / h1
        else:
            hr = h0  # h0 is treated as HR

        if hr <= 0:
            return {"error": "Hazard Ratio must be positive."}
        if hr == 1:
            return {"error": "Hazard Ratio cannot be 1 (No difference)."}
        if ratio <= 0:
            return {"error": "Ratio must be positive."}

        z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-sided
        z_beta = stats.norm.ppf(power)

        # Schoenberg/Richter formula for Events
        num = (1 + ratio) ** 2
        den = ratio * (np.log(hr)) ** 2
        total_events = (z_alpha + z_beta) ** 2 * (num / den)

        # We only return required events, N depends on censoring/follow-up which is complex
        # Usually we estimate N assuming probability of event P_event
        # We will return "Total Events Required"

        return {"total_events": np.ceil(total_events), "hr": hr}
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}


def calculate_sample_size_correlation(
    power: float, r: float, alpha: float = 0.05, alternative: str = "two-sided"
) -> float | dict[str, str]:
    """
    Calculate Sample Size for Pearson Correlation.
    Use Fisher's Z transformation.
    """
    # Approx N = 3 + ((z_alpha + z_beta) / (0.5 * ln((1+r)/(1-r))))^2
    try:
        if abs(r) >= 1:
            return {
                "error": "Correlation coefficient must be between -1 and 1 (exclusive)."
            }

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        c = 0.5 * np.log((1 + r) / (1 - r))
        if c == 0:
            return {"error": "Correlation coefficient cannot be zero."}

        n = 3 + ((z_alpha + z_beta) / c) ** 2
        return {
            "total": np.ceil(n)
        }  # Returning dict to be consistent for 'sample size' calculators
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}


def calculate_power_survival(
    total_events: float,
    ratio: float,
    h0: float,
    h1: float = 0,
    alpha: float = 0.05,
    mode: str = "hr",
) -> float:
    """Calculate Power for Log-Rank Test given number of events."""
    try:
        if mode == "median":
            if h1 == 0:
                return np.nan
            hr = h0 / h1
        else:
            hr = h0

        if hr <= 0 or hr == 1:
            return np.nan
        if ratio <= 0:
            return np.nan

        z_alpha = stats.norm.ppf(1 - alpha / 2)

        # E = (z_alpha + z_beta)^2 * ( (1+k)^2 / (k * ln(HR)^2) )
        # z_beta = sqrt( E * k * ln(HR)^2 / (1+k)^2 ) - z_alpha

        num = total_events * ratio * (np.log(hr)) ** 2
        den = (1 + ratio) ** 2

        z_beta = np.sqrt(num / den) - z_alpha
        return float(stats.norm.cdf(z_beta))
    except Exception:
        return np.nan


def calculate_power_correlation(
    n: float, r: float, alpha: float = 0.05, alternative: str = "two-sided"
) -> float:
    """Calculate Power for Pearson Correlation given N."""
    try:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        if abs(r) >= 1:
            return np.nan
        c = 0.5 * np.log((1 + r) / (1 - r))

        # z_beta = C * sqrt(N - 3) - z_alpha
        # Ensure n > 3
        if n <= 3:
            return 0.0

        z_beta = abs(c) * np.sqrt(n - 3) - z_alpha
        return float(stats.norm.cdf(z_beta))
    except Exception:
        return np.nan


def calculate_power_curve(
    target_n: int,
    ratio: float,
    calc_func: callable,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate data for Power Curve (Power vs Sample Size).

    Args:
        target_n: The calculated total sample size (or events)
        ratio: Allocation ratio (n2/n1)
        calc_func: The power calculation function to use
        **kwargs: Additional arguments for calc_func

    Returns:
        pd.DataFrame with columns ['total_n', 'power']
    """
    # Define range: ~50% to ~150% of target_n, or specific range if total_n is small
    n_center = max(20, target_n)

    # Generate ~30 points
    n_min = max(10, int(n_center * 0.5))
    n_max = max(n_min + 20, int(n_center * 1.5))

    # If range is too small, expand it
    if n_max - n_min < 20:
        n_max = n_min + 20

    # Generate evenly spaced sample sizes
    n_values = np.linspace(n_min, n_max, 30).astype(int)
    n_values = np.unique(n_values)  # Remove duplicates

    results = []

    for total_n in n_values:
        try:
            pwr = np.nan
            fname = calc_func.__name__

            if fname == "calculate_power_means":
                n1 = np.ceil(total_n / (1 + ratio))
                n2 = total_n - n1
                pwr = calc_func(n1, n2, **kwargs)
            elif fname == "calculate_power_proportions":
                n1 = np.ceil(total_n / (1 + ratio))
                n2 = total_n - n1
                pwr = calc_func(n1, n2, **kwargs)
            elif fname == "calculate_power_survival":
                # total_n here represents total_events
                pwr = calc_func(total_events=total_n, ratio=ratio, **kwargs)
            elif fname == "calculate_power_correlation":
                pwr = calc_func(n=total_n, **kwargs)

            results.append({"total_n": total_n, "power": pwr})

        except Exception:
            continue

    return pd.DataFrame(results)


def generate_methods_text(study_design: str, params: dict) -> str:
    """
    Generate methods text for publication.
    params dict should contain input parameters and calculated results.
    """
    text = f"Sample size calculation was performed for a {study_design} study. "

    if study_design == "Independent Means (T-test)":
        m1 = params.get("mean1")
        m2 = params.get("mean2")
        sd1 = params.get("sd1")
        sd2 = params.get("sd2")

        text += f"To detect a difference in means of {m1} versus {m2} "
        text += f"(assuming SD1={sd1}, SD2={sd2}), "
        text += f"with a power of {params['power'] * 100:.0f}% and a two-sided significant level (alpha) of {params['alpha']}, "
        text += f"a total sample size of {int(params['total'])} subjects ({int(params['n1'])} in group 1 and {int(params['n2'])} in group 2) is required."

    elif study_design == "Independent Proportions":
        text += f"To detect a difference between proportions of {params['p1']} and {params['p2']}, "
        text += f"with a power of {params['power'] * 100:.0f}% and a two-sided significant level (alpha) of {params['alpha']}, "
        text += f"a total sample size of {int(params['total'])} subjects ({int(params['n1'])} in group 1 and {int(params['n2'])} in group 2) is required."

    elif study_design == "Survival (Log-Rank)":
        text += f"To detect a Hazard Ratio of {params['hr']:.2f}, "
        text += f"with a power of {params['power'] * 100:.0f}% and a two-sided significant level (alpha) of {params['alpha']}, "
        text += f"a total of {int(params['total_events'])} events are required."

    elif study_design == "Pearson Correlation":
        text += f"To detect a correlation coefficient (r) of {params['r']}, "
        text += f"with a power of {params['power'] * 100:.0f}% and a two-sided significant level (alpha) of {params['alpha']}, "
        text += f"a total sample size of {int(params['total'])} subjects is required."

    return text
