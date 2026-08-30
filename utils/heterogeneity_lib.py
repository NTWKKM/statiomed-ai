import numpy as np
from scipy import stats


def calculate_heterogeneity(effect_sizes: list[float], variances: list[float]) -> dict:
    """
    Calculate heterogeneity statistics for meta-analysis (Cochran's Q and I^2).

    Args:
        effect_sizes: List of effect estimates (e.g., log OR, mean diff)
        variances: List of variances (SE^2) for each effect size

    Returns:
        dict containing Q, df, p_value, and I_squared
    """
    if not effect_sizes or not variances:
        return {"Q": 0.0, "df": 0, "p_value": 1.0, "I_squared": 0.0, "tau_squared": 0.0}

    k = len(effect_sizes)
    if k < 2:
        return {"Q": 0.0, "df": 0, "p_value": 1.0, "I_squared": 0.0, "tau_squared": 0.0}

    try:
        y = np.array(effect_sizes)
        v = np.array(variances)
        w = 1 / v  # Inverse variance weights

        # Weighted mean effect
        w_sum = np.sum(w)
        weighted_mean = np.sum(w * y) / w_sum

        # Cochran's Q
        Q = np.sum(w * (y - weighted_mean) ** 2)
        df = k - 1

        # P-value for Q
        p_value = 1 - stats.chi2.cdf(Q, df)

        # I-squared
        # I^2 = max(0, (Q - df) / Q) * 100
        if Q <= df:
            I2 = 0.0
        else:
            I2 = 100 * (Q - df) / Q

        # Tau-squared (DerSimonian-Laird estimator)
        # C = sum(w) - sum(w^2)/sum(w)
        c_const = w_sum - (np.sum(w**2) / w_sum)
        if c_const > 0:
            tau2 = max(0, (Q - df) / c_const)
        else:
            tau2 = 0.0

        return {
            "Q": float(Q),
            "df": df,
            "p_value": float(p_value),
            "I_squared": float(I2),
            "tau_squared": float(tau2),
        }
    except Exception:
        return {"Q": 0.0, "df": 0, "p_value": 1.0, "I_squared": 0.0, "tau_squared": 0.0}
