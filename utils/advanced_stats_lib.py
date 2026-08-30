from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from logger import get_logger
from utils.collinearity_lib import calculate_vif as _run_vif

logger = get_logger(__name__)


def apply_mcc(
    p_values: Union[List[float], np.ndarray, pd.Series],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> pd.Series:
    """
    Apply Multiple Comparison Correction to a list of p-values.
    """
    # ✅ FIX 1: เพิ่มการตรวจสอบ None เพื่อป้องกัน AttributeError
    if p_values is None:
        return pd.Series(dtype=float)

    if isinstance(p_values, (list, np.ndarray)):
        p_series = pd.Series(p_values)
    else:
        p_series = p_values.copy()

    if p_series.empty:
        return p_series

    mask = p_series.notna()
    p_valid = p_series[mask]

    if p_valid.empty:
        return p_series

    try:
        reject, pvals_corrected, _, _ = multipletests(
            p_valid.values, alpha=alpha, method=method
        )
        p_series.loc[mask] = pvals_corrected
    except Exception as e:
        logger.error(f"MCC calculation failed: {e}")
        return p_series

    return p_series


def get_ci_configuration(
    method: str, n: int, events: int, n_params: int
) -> Dict[str, str]:
    """
    Configure Confidence Interval method based on sample size and events.
    """
    config = {"method": method, "note": ""}

    if method == "profile":
        if n > 5000:
            config["method"] = "wald"
            config["note"] = "Switched to Wald due to large sample size (N>5000)"

    return config


def calculate_vif(
    data: pd.DataFrame,
    predictor_cols: Optional[List[str]] = None,
    var_meta: Optional[Dict[str, Any]] = None,
) -> Any:  # ✅ Updated return type hint
    """
    Calculate Variance Inflation Factor (VIF) for predictors.
    Wrapper around collinearity_lib.calculate_vif.
    """
    # ✅ FIX 2: Handle missing predictor_cols by using all columns except 'const'
    if predictor_cols is None:
        if data is None or data.empty:
            predictor_cols = []
        else:
            predictor_cols = [c for c in data.columns if c != "const"]

    return _run_vif(data, predictor_cols, var_meta=var_meta)
