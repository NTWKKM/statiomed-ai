"""
📊 Sensitivity Analysis Library

Functions for assessing robustness and uncertainty of statistical estimates.

Includes:
    - E-value calculation for unmeasured confounding
    - Bootstrap confidence intervals (percentile, BCa)
    - Jackknife estimates
    - Leave-one-out cross-validation

References:
    VanderWeele, T.J. & Ding, P. (2017). Sensitivity Analysis in
        Observational Research. Annals of Internal Medicine.
    Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
from scipy import stats

from logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "bootstrap_confidence_interval",
    "jackknife_estimate",
    "leave_one_out_cv",
    "calculate_e_value",
]


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================


def bootstrap_confidence_interval(
    data: np.ndarray | list,
    statistic_func: Callable[[np.ndarray], float],
    n_iterations: int = 10000,
    alpha: float = 0.05,
    method: Literal["percentile", "bca", "basic"] = "percentile",
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for any statistic.

    Supports multiple CI methods:
    - percentile: Simple percentile method
    - bca: Bias-corrected and accelerated (recommended)
    - basic: Basic bootstrap (reverse percentile)

    Args:
        data: Array of observations
        statistic_func: Function that takes an array and returns a scalar statistic
        n_iterations: Number of bootstrap resamples (default 10,000)
        alpha: Significance level (0.05 for 95% CI)
        method: CI method - 'percentile', 'bca', or 'basic'
        random_seed: Optional seed for reproducibility

    Returns:
        Dictionary containing:
            - point_estimate: Original statistic on full data
            - ci_lower: Lower bound of CI
            - ci_upper: Upper bound of CI
            - se_bootstrap: Bootstrap standard error
            - bootstrap_distribution: Array of bootstrap estimates
            - method: CI method used
            - n_iterations: Number of bootstrap samples

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> result = bootstrap_confidence_interval(
        ...     data,
        ...     statistic_func=np.mean,
        ...     n_iterations=5000
        ... )
        >>> print(f"Mean: {result['point_estimate']:.2f} "
        ...       f"({result['ci_lower']:.2f}, {result['ci_upper']:.2f})")

    References:
        Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
    """
    try:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 0:
            arr = np.atleast_1d(arr)

        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 2:
            return {"error": "Need at least 2 observations for bootstrap"}

        if random_seed is not None:
            np.random.seed(random_seed)

        # Original statistic
        point_estimate = statistic_func(arr)

        # Generate bootstrap samples
        bootstrap_estimates = np.empty(n_iterations)
        for i in range(n_iterations):
            resample = np.random.choice(arr, size=n, replace=True)
            bootstrap_estimates[i] = statistic_func(resample)

        # Remove any NaN bootstrap estimates
        bootstrap_estimates = bootstrap_estimates[~np.isnan(bootstrap_estimates)]

        if len(bootstrap_estimates) < 100:
            return {"error": "Too many bootstrap samples failed"}

        # Calculate CI based on method
        if method == "percentile":
            ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

        elif method == "basic":
            # Basic bootstrap: 2*theta - quantiles
            q_lower = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
            q_upper = np.percentile(bootstrap_estimates, 100 * alpha / 2)
            ci_lower = 2 * point_estimate - q_lower
            ci_upper = 2 * point_estimate - q_upper

        elif method == "bca":
            # BCa: Bias-corrected and accelerated
            # Bias correction factor
            p0 = np.mean(bootstrap_estimates < point_estimate)
            z0 = stats.norm.ppf(p0) if 0 < p0 < 1 else 0

            # Acceleration factor using jackknife
            jackknife_estimates = np.empty(n)
            for i in range(n):
                jack_sample = np.delete(arr, i)
                jackknife_estimates[i] = statistic_func(jack_sample)

            jack_mean = np.mean(jackknife_estimates)
            numerator = np.sum((jack_mean - jackknife_estimates) ** 3)
            denominator = 6 * (np.sum((jack_mean - jackknife_estimates) ** 2) ** 1.5)

            a = numerator / denominator if denominator != 0 else 0

            # Adjusted percentiles
            z_alpha_low = stats.norm.ppf(alpha / 2)
            z_alpha_high = stats.norm.ppf(1 - alpha / 2)

            # BCa adjusted percentile positions
            p_low = stats.norm.cdf(
                z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low))
            )
            p_high = stats.norm.cdf(
                z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high))
            )

            # Bound percentiles to valid range
            p_low = max(0.001, min(0.999, p_low))
            p_high = max(0.001, min(0.999, p_high))

            ci_lower = np.percentile(bootstrap_estimates, 100 * p_low)
            ci_upper = np.percentile(bootstrap_estimates, 100 * p_high)

        else:
            return {"error": f"Unknown method: {method}"}

        se_bootstrap = np.std(bootstrap_estimates, ddof=1)

        return {
            "point_estimate": round(float(point_estimate), 4),
            "ci_lower": round(float(ci_lower), 4),
            "ci_upper": round(float(ci_upper), 4),
            "se_bootstrap": round(float(se_bootstrap), 4),
            "bootstrap_distribution": bootstrap_estimates.tolist(),
            "method": method,
            "n_iterations": len(bootstrap_estimates),
            "confidence_level": 1 - alpha,
        }

    except Exception as e:
        logger.exception("Bootstrap CI calculation failed")
        return {"error": f"Bootstrap failed: {e!s}"}


def jackknife_estimate(
    data: np.ndarray | list,
    statistic_func: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    """
    Calculate jackknife estimate and standard error.

    The jackknife is a resampling method that systematically leaves out
    one observation at a time. Useful for:
    - Bias estimation
    - Standard error estimation
    - Detecting influential observations

    Args:
        data: Array of observations
        statistic_func: Function that computes the statistic

    Returns:
        Dictionary with jackknife estimate, SE, and leave-one-out values

    Examples:
        >>> data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        >>> result = jackknife_estimate(data, np.mean)
        >>> print(f"SE: {result['se_jackknife']:.2f}")
    """
    try:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 0:
            arr = np.atleast_1d(arr)

        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 2:
            return {"error": "Need at least 2 observations"}

        # Full sample estimate
        theta_hat = statistic_func(arr)

        # Leave-one-out estimates
        jackknife_estimates = np.empty(n)
        for i in range(n):
            jack_sample = np.delete(arr, i)
            jackknife_estimates[i] = statistic_func(jack_sample)

        # Jackknife estimate of the statistic
        theta_jack = n * theta_hat - (n - 1) * np.mean(jackknife_estimates)

        # Bias
        bias = (n - 1) * (np.mean(jackknife_estimates) - theta_hat)

        # Standard error
        se_jackknife = np.sqrt(
            (n - 1)
            / n
            * np.sum((jackknife_estimates - np.mean(jackknife_estimates)) ** 2)
        )

        # Identify influential observations
        influence = theta_hat - jackknife_estimates
        influence_threshold = 2 * np.std(influence)
        influential_indices = np.where(np.abs(influence) > influence_threshold)[
            0
        ].tolist()

        return {
            "theta_original": round(float(theta_hat), 4),
            "theta_jackknife": round(float(theta_jack), 4),
            "bias": round(float(bias), 4),
            "se_jackknife": round(float(se_jackknife), 4),
            "jackknife_estimates": jackknife_estimates.tolist(),
            "influence_values": influence.tolist(),
            "influential_observations": influential_indices,
            "n_influential": len(influential_indices),
        }

    except Exception as e:
        logger.exception("Jackknife estimation failed")
        return {"error": f"Jackknife failed: {e!s}"}


def leave_one_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fit_func: Callable,
    predict_func: Callable,
) -> dict[str, Any]:
    """
    Perform Leave-One-Out Cross-Validation.

    LOOCV provides an almost unbiased estimate of prediction error,
    but has high variance.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        model_fit_func: Function(X_train, y_train) -> fitted_model
        predict_func: Function(fitted_model, X_test) -> predictions

    Returns:
        Dictionary with MSE, RMSE, R², and predictions
    """
    try:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).flatten()

        # Remove rows with NaN in X or y
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        n = len(y)

        if n < 3:
            return {"error": "Need at least 3 non-NaN observations for LOOCV"}

        predictions = np.empty(n)
        errors = np.empty(n)

        for i in range(n):
            # Create train/test split
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False

            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]

            # Fit and predict
            model = model_fit_func(X_train, y_train)
            pred = predict_func(model, X_test)

            predictions[i] = pred[0] if hasattr(pred, "__len__") else pred
            errors[i] = y_test[0] - predictions[i]

        # Calculate metrics
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "predictions": predictions.tolist(),
            "residuals": errors.tolist(),
            "n_samples": n,
        }

    except Exception as e:
        logger.exception("LOOCV failed")
        return {"error": f"LOOCV failed: {e!s}"}


# =============================================================================
# E-VALUE FOR UNMEASURED CONFOUNDING
# =============================================================================


def calculate_e_value(
    estimate: float,
    lower: float | None = None,
    upper: float | None = None,
    estimate_type: Literal["RR", "OR"] = "RR",
) -> dict:
    """
    Calculate E-value for a risk ratio (RR) or odds ratio (OR).
    If OR provided and outcome is common, E-value is approximate.

    E-value = RR + sqrt(RR * (RR - 1))
    """
    try:
        # Validate inputs
        if estimate is None or not isinstance(estimate, (int, float)):
            return {"error": "Estimate must be a number."}

        if estimate <= 0:
            return {"error": "Estimate must be positive."}

        if estimate_type not in ("RR", "OR"):
            raise ValueError(
                f"Invalid estimate_type: {estimate_type}. Must be 'RR' or 'OR'."
            )

        if estimate == 1.0:
            return {
                "original_estimate": estimate,
                "e_value_estimate": 1.0,
                "e_value_ci_limit": 1.0,
            }

        original_estimate = estimate

        # Convert OR to approximate RR if needed (conservative sqrt heuristic)
        if estimate_type == "OR":
            estimate = np.sqrt(estimate)
            if lower is not None and lower > 0:
                lower = np.sqrt(lower)
            if upper is not None and upper > 0:
                upper = np.sqrt(upper)

            # Bounds sanity check for OR->RR conversion
            if lower is not None and lower > estimate:
                logger.warning(
                    f"OR->RR conversion produced inconsistent lower bound: {lower} > {estimate}. Setting lower to None."
                )
                lower = None
            if upper is not None and upper < estimate:
                logger.warning(
                    f"OR->RR conversion produced inconsistent upper bound: {upper} < {estimate}. Setting upper to None."
                )
                upper = None

            logger.info(
                "Converted OR to approximate RR using sqrt heuristic for E-value calculation"
            )

        # Handle protective effects (RR < 1) by taking inverse
        if estimate < 1:
            est_prime = 1 / estimate
            # CI flips
            # If upper is provided, it becomes the lower bound in the flipped scale
            l_prime = 1 / upper if (upper is not None and upper > 0) else None

            if l_prime is None and lower is not None:
                logger.warning(
                    "Protective effect (RR<1): upper CI bound needed for E-value of CI limit, "
                    "but only lower bound was provided"
                )
            # Note: valid lower bound is ignored here because for RR < 1, the upper bound
            # corresponds to the limit closest to the null (1.0) on the inverted scale.
        else:
            est_prime = estimate
            l_prime = lower

        def compute_e(val):
            if val is None or val <= 1:
                return 1.0
            return val + np.sqrt(val * (val - 1))

        e_est = compute_e(est_prime)

        # For reporting, we usually report the E-value for the estimate
        # and the E-value for the CI limit closest to the null (1).

        limit_e_val = 1.0
        if l_prime is not None and l_prime > 1:
            limit_e_val = compute_e(l_prime)
        # If CI crosses 1, the limit E-value is 1.

        return {
            "original_estimate": original_estimate,
            "e_value_estimate": round(e_est, 3),
            "e_value_ci_limit": round(limit_e_val, 3),
        }
    except Exception as e:
        logger.exception("E-value calculation failed")
        return {"error": f"E-value calculation failed: {e!s}"}
