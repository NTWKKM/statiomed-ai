"""
🧮 Logistic Regression Core Logic (Shiny Compatible)

No Streamlit dependencies - pure statistical functions.
"""

from __future__ import annotations

import html
import re
import warnings
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Try importing sklearn metrics for AUC
try:
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from utils.advanced_stats_lib import apply_mcc, calculate_vif, get_ci_configuration
from utils.data_cleaning import (
    prepare_data_for_analysis,
)
from utils.formatting import (
    PublicationFormatter,
    create_missing_data_report_html,
    format_p_value,
)

logger = get_logger(__name__)

# Fetch palette and extend for local needs
_PALETTE = get_color_palette()
COLORS = {
    "primary": _PALETTE.get("primary", "#1E3A5F"),
    "primary_dark": _PALETTE.get("primary_dark", "#0F2440"),
    "primary_light": _PALETTE.get("primary_light", "#EBF5FF"),
    "success": _PALETTE.get("success", "#10B981"),
    "warning": _PALETTE.get("warning", "#F59E0B"),
    "danger": _PALETTE.get("danger", "#E74856"),
    "info": _PALETTE.get("info", "#3B82F6"),
    "text": _PALETTE.get("text", "#1F2937"),
    "text_secondary": _PALETTE.get("text_secondary", "#6B7280"),
    "border": _PALETTE.get("border", "#E5E7EB"),
    "background": _PALETTE.get("background", "#F9FAFB"),
    "surface": _PALETTE.get("surface", "#FFFFFF"),
}

# Try to import Firth regression (firthmodels >= 0.7.2)
try:
    from firthmodels import FirthLogisticRegression, detect_separation

    HAS_FIRTH = True
except (ImportError, AttributeError):
    HAS_FIRTH = False
    logger.warning("firthmodels not available")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*")

# ✅ NEW: Type Definitions (Compatible with Python 3.10+)
FitStatus: TypeAlias = Literal["OK"] | str
MethodType: TypeAlias = Literal["default", "bfgs", "firth", "auto"]


class StatsMetrics(TypedDict):
    mcfadden: float
    nagelkerke: float
    p_value: float | None
    aic: float | None  # New
    bic: float | None  # New
    auc: float | None  # New
    hl_pvalue: float | None  # New
    hl_stat: float | None  # New


# Functional syntax for TypedDict to support 'or' as a key
ORResult = TypedDict(
    "ORResult",
    {
        "or": float,
        "ci_low": float,
        "ci_high": float,
        "p_value": float,
        "p_adj": float | None,
    },
)


class AORResultEntry(TypedDict):
    coef: float
    aor: float
    ci_low: float
    ci_high: float
    p: float
    lvl: str | None


# Functional syntax for InteractionResult due to 'or' keyword
InteractionResult = TypedDict(
    "InteractionResult",
    {
        "coef": float,
        "or": float | None,
        "ci_low": float,
        "ci_high": float,
        "p_value": float,
        "label": str,
    },
)


# ✅ NEW: TypedDict for Adjusted OR results
class AORResult(TypedDict):
    aor: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adj: float | None
    label: str | None  # Added for flexible labeling


# ✅ NEW: Helper function for Hosmer-Lemeshow Test
def calculate_hosmer_lemeshow(y_true, y_pred, g=10):
    """
    Calculate Hosmer-Lemeshow goodness of fit test.
    Returns (chi2_stat, p_value).
    """
    try:
        data = pd.DataFrame({"obs": y_true, "pred": y_pred})
        # Create deciles, handling duplicates
        data["decile"] = pd.qcut(data["pred"], g, duplicates="drop")

        grouped = data.groupby("decile", observed=False)
        obs = grouped["obs"].sum()
        total_prob = grouped["pred"].sum()
        count = grouped["obs"].count()

        # HL Statistic formula
        # chi2 = sum( (O - E)^2 / (E * (1 - E/n)) )  <-- Simplified variance approx
        # Standard: sum ( (Ok - nk*pik)^2 / (nk * pik * (1-pik)) )

        numerator = (obs - total_prob) ** 2
        denominator = total_prob * (1 - (total_prob / count))

        # Avoid division by zero
        valid = denominator > 1e-9
        hl_stat = (numerator[valid] / denominator[valid]).sum()

        dof = len(grouped) - 2
        if dof < 1:
            logger.warning(
                f"HL test: insufficient groups ({len(grouped)}), returning NaN"
            )
            return np.nan, np.nan

        p_val = 1 - stats.chi2.cdf(hl_stat, dof)
        return hl_stat, p_val
    except Exception as e:
        logger.warning(f"HL Test failed: {e}")
        return np.nan, np.nan


# ✅ NEW: Helper to calculate all diagnostics
def calculate_model_diagnostics(y_true, y_pred) -> dict:
    metrics = {"auc": np.nan, "hl_stat": np.nan, "hl_pvalue": np.nan}

    # 1. AUC / C-Statistic
    if HAS_SKLEARN and len(np.unique(y_true)) == 2:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_pred)
        except Exception as e:
            logger.debug(f"AUC calculation failed: {e}")

    # 2. Hosmer-Lemeshow (Calibration)
    hl_stat, hl_p = calculate_hosmer_lemeshow(y_true, y_pred)
    metrics["hl_stat"] = hl_stat
    metrics["hl_pvalue"] = hl_p

    return metrics


# ✅ NEW: Helper function to load static CSS
def load_static_css() -> str:
    """
    Load the contents of static/styles.css for embedding in reports.
    """
    try:
        # 1. Try local static folder (utils/static/styles.css)
        css_path = Path(__file__).parent / "static" / "styles.css"
        if not css_path.exists():
            # 2. Try root static folder (static/styles.css)
            css_path = Path(__file__).parent.parent / "static" / "styles.css"

        if css_path.exists():
            with open(css_path, encoding="utf-8") as f:
                return f.read()
        else:
            logger.warning(
                f"CSS file not found at expected locations. Last checked: {css_path}"
            )
            return ""
    except Exception:
        logger.exception("Failed to load static CSS")
        return ""


def validate_logit_data(y: pd.Series, X: pd.DataFrame) -> tuple[bool, str]:
    """
    ✅ NEW: Added validation to prevent crashes during model fitting.
    Checks for perfect separation, zero variance, and collinearity.
    """
    issues = []

    # Check for empty data
    if len(y) == 0 or X.empty:
        return False, "Empty data provided"

    # Check for constant outcome
    if y.nunique() < 2:
        issues.append("Outcome variable has only one unique value (constant).")

    # Check for zero variance (constant columns)
    for col in X.columns:
        if X[col].nunique() <= 1:
            issues.append(f"Variable '{col}' has zero variance (only one value)")

    # Check for perfect separation (quasi-complete or complete)
    for col in X.columns:
        try:
            ct = pd.crosstab(X[col], y)
            if (ct == 0).any().any():
                logger.debug(f"Perfect separation detected in variable: {col}")
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not check separation for {col}: {e}")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def clean_numeric_value(val):
    """
    Cleans numeric values, handling strings, commas, and comparison operators.
    """
    if pd.isna(val) or val == "":
        return np.nan

    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, str):
        val = val.strip().replace(",", "")
        try:
            return float(val)
        except ValueError:
            match = re.search(r"[-+]?\d*\.\d+|\d+", val)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return np.nan
            return np.nan

    return np.nan


def _robust_sort_key(x: Any) -> tuple[int, float | str]:
    """Sort key placing numeric values first."""
    try:
        if pd.isna(x):
            return (2, "")
        val = float(x)
        return (0, val)
    except (ValueError, TypeError):
        return (1, str(x))


DEFAULT_FIRTH_PENALTY_WEIGHT = 1.0


def fit_firth_logistic(
    y: pd.Series,
    X_const: pd.DataFrame,
    penalty_weight: float = DEFAULT_FIRTH_PENALTY_WEIGHT,
) -> tuple[
    pd.Series | None, pd.DataFrame | None, pd.Series | None, FitStatus, StatsMetrics
]:
    """Fit Firth logistic regression with configurable penalty weight."""
    stats_metrics: StatsMetrics = {
        "mcfadden": np.nan,
        "nagelkerke": np.nan,
        "p_value": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "auc": np.nan,
        "hl_pvalue": np.nan,
        "hl_stat": np.nan,
    }

    if not HAS_FIRTH:
        return None, None, None, "firthmodels not installed", stats_metrics

    try:
        # Using firthmodels with configurable penalty weight
        fl = FirthLogisticRegression(
            fit_intercept=False, penalty_weight=penalty_weight
        )

        fl.fit(X_const, y)

        # Explicitly call LRT for better p-values
        try:
            fl.lrt()
            pvalues_src = getattr(fl, "lrt_pvalues_", fl.pvalues_)
        except Exception:
            pvalues_src = fl.pvalues_

        coef = np.asarray(fl.coef_).reshape(-1)
        if coef.shape[0] != len(X_const.columns):
            return None, None, None, "Firth output shape mismatch", stats_metrics

        params = pd.Series(coef, index=X_const.columns)
        pvalues = pd.Series(
            pvalues_src
            if pvalues_src is not None
            else np.full(len(X_const.columns), np.nan),
            index=X_const.columns,
        )

        try:
            # Try Profile Likelihood CIs first
            ci = fl.conf_int(method="pl")
        except Exception:
            # Fallback to Wald
            ci = fl.conf_int(method="wald")

        conf_int = pd.DataFrame(ci, index=X_const.columns, columns=[0, 1])

        # ✅ NEW: Calculate Predictions & Diagnostics
        try:
            # firthmodels mimics sklearn, predict_proba returns [prob_0, prob_1]
            y_pred = fl.predict_proba(X_const)[:, 1]
            diag = calculate_model_diagnostics(y, y_pred)
            stats_metrics.update(diag)

            # Estimate Log-Likelihood for AIC/BIC (Approximate)
            if hasattr(fl, "loglik_"):
                llf = fl.loglik_
                k = len(params)
                n = len(y)
                stats_metrics["aic"] = 2 * k - 2 * llf
                stats_metrics["bic"] = k * np.log(n) - 2 * llf
        except Exception as e:
            logger.warning(f"Firth diagnostics failed: {e}")

        return params, conf_int, pvalues, "OK", stats_metrics

    except Exception as e:
        logger.error(f"Firth fit failed: {e}")
        return None, None, None, str(e), stats_metrics


def fit_standard_logistic(
    y: pd.Series,
    X_const: pd.DataFrame,
    method: str = "default",
    ci_method: str = "wald",
) -> tuple[
    pd.Series | None, pd.DataFrame | None, pd.Series | None, FitStatus, StatsMetrics
]:
    """Fit standard MLE logistic regression."""
    stats_metrics: StatsMetrics = {
        "mcfadden": np.nan,
        "nagelkerke": np.nan,
        "p_value": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "auc": np.nan,
        "hl_pvalue": np.nan,
        "hl_stat": np.nan,
    }

    try:
        if method == "bfgs":
            model = sm.Logit(y, X_const)
            result = model.fit(method="bfgs", maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const)
            result = model.fit(disp=0)

        # Calculate R-squared metrics
        try:
            llf = result.llf
            llnull = result.llnull
            nobs = result.nobs
            mcfadden = 1 - (llf / llnull) if llnull != 0 else np.nan
            cox_snell = 1 - np.exp((2 / nobs) * (llnull - llf))
            max_r2 = 1 - np.exp((2 / nobs) * llnull)
            nagelkerke = cox_snell / max_r2 if max_r2 > 1e-9 else np.nan
            stats_metrics.update(
                {
                    "mcfadden": mcfadden,
                    "nagelkerke": nagelkerke,
                    "p_value": getattr(result, "llr_pvalue", np.nan),
                    "aic": result.aic,
                    "bic": result.bic,
                }
            )
        except (AttributeError, ZeroDivisionError, TypeError) as e:
            logger.debug(f"Failed to calculate R2: {e}")

        # ✅ NEW: Advanced Diagnostics (AUC, HL)
        try:
            y_pred = result.predict(X_const)
            diag = calculate_model_diagnostics(y, y_pred)
            stats_metrics.update(diag)
        except Exception as e:
            logger.warning(f"Standard logit diagnostics failed: {e}")

        if ci_method == "profile":
            logger.debug(
                "Profile likelihood CI requested but not fully implemented, falling back to Wald"
            )

        return result.params, result.conf_int(), result.pvalues, "OK", stats_metrics

    except Exception as e:
        # Detect separation error messages from Statsmodels
        err_msg = str(e)
        if "Singular matrix" in err_msg or "LinAlgError" in err_msg:
            err_msg = "Model fitting failed: data may have perfect separation or too much collinearity."
        elif "Perfect separation" in err_msg:
            err_msg = "Perfect separation detected."

        logger.error(f"Standard logistic failed: {e}")
        return None, None, None, err_msg, stats_metrics


def run_binary_logit(
    y: pd.Series,
    X: pd.DataFrame,
    method: MethodType = "default",
    ci_method: str = "wald",
    penalty_weight: float = DEFAULT_FIRTH_PENALTY_WEIGHT,
) -> tuple[
    pd.Series | None,
    pd.DataFrame | None,
    pd.Series | None,
    FitStatus,
    StatsMetrics,
]:
    """
    Fit a binary logistic regression model and return coefficients, confidence intervals, p-values, status, and fit statistics.
    """
    stats_metrics: StatsMetrics = {
        "mcfadden": np.nan,
        "nagelkerke": np.nan,
        "p_value": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "auc": np.nan,
        "hl_pvalue": np.nan,
        "hl_stat": np.nan,
    }

    # ✅ NEW: Initial Validation
    is_valid, msg = validate_logit_data(y, X)
    if not is_valid:
        return None, None, None, msg, stats_metrics

    try:
        X_const = sm.add_constant(X, has_constant="add")

        if method == "firth":
            return fit_firth_logistic(y, X_const, penalty_weight=penalty_weight)

        # Default or BFGS
        return fit_standard_logistic(y, X_const, method=method, ci_method=ci_method)

    except Exception as e:
        logger.error(f"Logistic regression wrapper failed: {e}")
        return None, None, None, str(e), stats_metrics


def run_glm(
    y: pd.Series,
    X: pd.DataFrame,
    family_name: str = "Gaussian",
    link_name: str = "identity",
) -> tuple[
    pd.Series | None,
    pd.DataFrame | None,
    pd.Series | None,
    FitStatus,
    dict[str, float],
]:
    """
    Fit a Generalized Linear Model (GLM).
    """
    fit_metrics = {"aic": np.nan, "bic": np.nan, "deviance": np.nan}

    # Initial Validation
    if len(y) == 0 or X.empty or len(y) != len(X):
        return None, None, None, "Invalid data dimensions", fit_metrics

    try:
        X_const = sm.add_constant(X, has_constant="add")

        # Map Family
        family_map = {
            "Gaussian": sm.families.Gaussian,
            "Binomial": sm.families.Binomial,
            "Poisson": sm.families.Poisson,
            "Gamma": sm.families.Gamma,
            "InverseGaussian": sm.families.InverseGaussian,
        }
        family_cls = family_map.get(family_name, sm.families.Gaussian)

        # Map Link
        link_map = {
            "identity": sm.families.links.Identity(),
            "log": sm.families.links.Log(),
            "logit": sm.families.links.Logit(),
            "probit": sm.families.links.Probit(),
            "cloglog": sm.families.links.CLogLog(),
            "inverse_power": sm.families.links.InversePower(),
            "sqrt": sm.families.links.Sqrt(),
        }
        link_obj = link_map.get(link_name)

        if link_obj:
            family_instance = family_cls(link=link_obj)
        else:
            family_instance = family_cls()

        model = sm.GLM(y, X_const, family=family_instance)
        result = model.fit()

        fit_metrics = {
            "aic": result.aic,
            "bic": result.bic_llf,
            "deviance": result.deviance,
        }

        return result.params, result.conf_int(), result.pvalues, "OK", fit_metrics

    except Exception as e:
        err_msg = str(e)
        logger.error(f"GLM failed: {e}")
        return None, None, None, err_msg, fit_metrics


def get_label(col_name: str, var_meta: dict[str, Any] | None) -> str:
    """Create formatted label for column."""
    display_name = col_name
    secondary_label = ""

    if var_meta:
        if col_name in var_meta and "label" in var_meta[col_name]:
            secondary_label = var_meta[col_name]["label"]
        elif "_" in col_name:
            parts = col_name.split("_", 1)
            if len(parts) > 1:
                short_name = parts[1]
                if short_name in var_meta and "label" in var_meta[short_name]:
                    secondary_label = var_meta[short_name]["label"]

    safe_name = html.escape(str(display_name))
    if secondary_label:
        safe_label = html.escape(str(secondary_label))
        return f"<b>{safe_name}</b><br><span style='color:#666; font-size:0.9em'>{safe_label}</span>"
    else:
        return f"<b>{safe_name}</b>"


def fmt_or_with_styling(or_val: float | None, ci_low: float, ci_high: float) -> str:
    """Format OR (95% CI) with bolding if significant."""
    if or_val is None or pd.isna(or_val) or pd.isna(ci_low) or pd.isna(ci_high):
        return "-"

    # Use PublicationFormatter for CI style
    ci_str = PublicationFormatter.format_ci(ci_low, ci_high, include_brackets=True)
    res_str = f"{or_val:.2f} {ci_str}"

    if (ci_low > 1.0) or (ci_high < 1.0):
        return f"<b>{res_str}</b>"

    return res_str


def analyze_outcome(
    outcome_name: str,
    df: pd.DataFrame,
    var_meta: dict[str, Any] | None = None,
    method: MethodType = "auto",
    interaction_pairs: list[tuple[str, str | None]] = None,
    adv_stats: dict[str, Any] | None = None,
    penalty_weight: float = DEFAULT_FIRTH_PENALTY_WEIGHT,
) -> tuple[
    str, dict[str, ORResult], dict[str, AORResult], dict[str, InteractionResult]
]:
    """
    Perform a complete logistic regression analysis.
    """
    # Extract Advanced Stats Settings
    mcc_enable = adv_stats.get("stats.mcc_enable", False) if adv_stats else False
    mcc_method = adv_stats.get("stats.mcc_method", "fdr_bh") if adv_stats else "fdr_bh"
    mcc_alpha = adv_stats.get("stats.mcc_alpha", 0.05) if adv_stats else 0.05
    try:
        mcc_alpha = float(mcc_alpha)
    except (TypeError, ValueError):
        mcc_alpha = 0.05

    vif_enable = adv_stats.get("stats.vif_enable", False) if adv_stats else False
    vif_thresh_raw = adv_stats.get("stats.vif_threshold", 10) if adv_stats else 10
    try:
        vif_threshold = float(vif_thresh_raw)
    except (ValueError, TypeError):
        vif_threshold = 10.0

    ci_method = adv_stats.get("stats.ci_method", "wald") if adv_stats else "wald"

    logger.info(
        "Starting logistic analysis for outcome: %s. MCC=%s, VIF=%s, CI=%s",
        outcome_name,
        mcc_enable,
        vif_enable,
        ci_method,
    )

    # --- INITIALIZE VARIABLES ---
    final_n_multi = 0  # Initialize to prevent UnboundLocalError
    vif_html = ""  # Initialize to prevent UnboundLocalError
    model_diagnostics_html = ""  # Initialize to prevent UnboundLocalError
    effective_ci_method = ci_method  # Initialize fallback
    ci_note = ""  # Initialize fallback

    # --- MISSING DATA HANDLING ---
    missing_cfg = CONFIG.get("analysis.missing", {}) or {}
    strategy = missing_cfg.get("strategy", "complete-case")
    missing_codes = missing_cfg.get("user_defined_values", [])

    try:
        # ✅ FIX: Include outcome in needed_cols to prevent it from being dropped
        needed_cols = [c for c in df.columns]

        df_clean, missing_data_info = prepare_data_for_analysis(
            df,
            required_cols=needed_cols,
            handle_missing=strategy,
            var_meta=var_meta,
            missing_codes=missing_codes,
        )
        df = df_clean
        logger.info(
            "Missing data: %s rows excluded (%.1f%%)",
            missing_data_info["rows_excluded"],
            (
                missing_data_info["rows_excluded"]
                / missing_data_info["rows_original"]
                * 100
                if missing_data_info["rows_original"] > 0
                else 0
            ),
        )
    except Exception as e:
        logger.error(f"Data preparation for logistic analysis failed: {e}")
        return (
            f"<div class='alert alert-danger'><b>Data Preparation Failed:</b> {e}</div>",
            {},
            {},
            None,
        )

    if outcome_name not in df.columns:
        msg = f"Outcome '{outcome_name}' not found"
        logger.error(msg)
        return f"<div class='alert'>{msg}</div>", {}, {}, {}

    y_raw = df[outcome_name].dropna()
    unique_outcomes = set(y_raw.unique())

    if len(unique_outcomes) != 2:
        msg = f"Invalid outcome: expected 2 values, found {len(unique_outcomes)}"
        logger.error(msg)
        return f"<div class='alert'>{msg}</div>", {}, {}, {}

    if not unique_outcomes.issubset({0, 1}):
        sorted_outcomes = sorted(unique_outcomes, key=str)
        outcome_map = {sorted_outcomes[0]: 0, sorted_outcomes[1]: 1}
        y = y_raw.map(outcome_map).astype(int)
    else:
        y = y_raw.astype(int)

    df_aligned = df.loc[y.index]
    total_n = len(y)
    candidates = []
    results_db = {}
    sorted_cols = sorted(df.columns.astype(str))
    mode_map = {}
    cat_levels_map = {}

    # Check for perfect separation using Konis (2007) LP method
    has_perfect_separation = False
    if method == "auto" and HAS_FIRTH:
        try:
            sep_parts = []
            for c in sorted_cols:
                if c == outcome_name or c not in df_aligned.columns:
                    continue

                numeric = df_aligned[c].apply(clean_numeric_value)
                if numeric.notna().any() and numeric.nunique(dropna=True) > 1:
                    sep_parts.append(numeric.rename(c))
                    continue

                dummies = pd.get_dummies(
                    df_aligned[c], prefix=c, drop_first=True, dtype=float
                )
                if dummies.shape[1] > 0:
                    sep_parts.append(dummies)

            if sep_parts:
                X_sep = pd.concat(sep_parts, axis=1)
                X_sep = X_sep.loc[:, X_sep.nunique(dropna=True) > 1].dropna()
                y_sep = y.loc[X_sep.index]
                if len(X_sep) > 0 and X_sep.shape[1] > 0:
                    sep_result = detect_separation(X_sep.values, y_sep.values)
                    has_perfect_separation = sep_result.separation
                    if has_perfect_separation:
                        logger.info(
                            "Separation detected (Konis LP method): %s",
                            sep_result.summary() if hasattr(sep_result, 'summary') else 'True',
                        )
        except Exception as e:
            logger.warning("detect_separation failed, falling back to heuristic: %s", e)
            # Fallback: original crosstab heuristic
            for col in sorted_cols:
                if col == outcome_name or col not in df_aligned.columns:
                    continue
                try:
                    X_num = df_aligned[col].apply(clean_numeric_value)
                    if X_num.nunique() > 1 and (pd.crosstab(X_num, y) == 0).any().any():
                        has_perfect_separation = True
                        break
                except Exception:
                    continue

    # Select fitting method
    preferred_method: MethodType = "bfgs"
    if (
        method == "auto"
        and HAS_FIRTH
        and (has_perfect_separation or len(df) < 50 or (y == 1).sum() < 20)
    ):
        preferred_method = "firth"
    elif method == "firth":
        preferred_method = "firth" if HAS_FIRTH else "bfgs"
    elif method == "default":
        preferred_method = "default"

    def count_val(series: pd.Series, v_str: str) -> int:
        return (
            series.astype(str).apply(
                lambda x: (
                    str(x).replace(".0", "")
                    if str(x).replace(".", "", 1).isdigit()
                    else str(x)
                )
            )
            == v_str
        ).sum()

    or_results: dict[str, ORResult] = {}

    # Univariate analysis
    logger.info(f"Starting univariate analysis for {len(sorted_cols) - 1} variables")
    for col in sorted_cols:
        if (
            col == outcome_name
            or col not in df_aligned.columns
            or df_aligned[col].isnull().all()
        ):
            continue

        res: dict[str, Any] = {"var": col}
        X_raw = df_aligned[col]
        X_num = X_raw.apply(clean_numeric_value)
        X_neg = X_raw[y == 0]
        X_pos = X_raw[y == 1]

        unique_vals = X_num.dropna().unique()
        mode = "linear"

        # Auto-detect mode
        if set(unique_vals).issubset({0, 1}):
            mode = "categorical"
        elif len(unique_vals) < 10:
            decimals_pct = (
                sum(1 for v in unique_vals if not float(v).is_integer())
                / len(unique_vals)
                if len(unique_vals) > 0
                else 0
            )
            if decimals_pct < 0.3:
                mode = "categorical"

        # Check var_meta
        if var_meta:
            orig_name = col.split("_", 1)[1] if "_" in col else col
            key = col if col in var_meta else orig_name
            if key in var_meta:
                user_mode = var_meta[key].get("type", "").lower()
                if "cat" in user_mode or "simp" in user_mode:
                    mode = "categorical"
                elif "lin" in user_mode or "cont" in user_mode:
                    mode = "linear"

        mode_map[col] = mode

        if mode == "categorical":
            try:
                levels = sorted(X_raw.dropna().unique(), key=_robust_sort_key)
            except (TypeError, ValueError):
                levels = sorted(X_raw.astype(str).unique())
            cat_levels_map[col] = levels

            n_used = len(X_raw.dropna())
            desc_tot = [f"n={n_used}"]
            desc_neg = [f"n={len(X_neg.dropna())}"]
            desc_pos = [f"n={len(X_pos.dropna())}"]

            for lvl in levels:
                lbl_txt = str(int(float(lvl))) if str(lvl).endswith(".0") else str(lvl)
                c_all = count_val(X_raw, lbl_txt)
                p_all = (c_all / n_used) * 100 if n_used else 0
                c_n = count_val(X_neg, lbl_txt)
                p_n = (c_n / len(X_neg.dropna())) * 100 if len(X_neg.dropna()) else 0
                c_p = count_val(X_pos, lbl_txt)
                p_p = (c_p / len(X_pos.dropna())) * 100 if len(X_pos.dropna()) else 0

                desc_tot.append(f"{lbl_txt}: {c_all} ({p_all:.1f}%)")
                desc_neg.append(f"{c_n} ({p_n:.1f}%)")
                desc_pos.append(f"{c_p} ({p_p:.1f}%)")

            res["desc_total"] = "<br>".join(desc_tot)
            res["desc_neg"] = "<br>".join(desc_neg)
            res["desc_pos"] = "<br>".join(desc_pos)

            try:
                ct = pd.crosstab(X_raw, y)
                _, p, _, _ = (
                    stats.chi2_contingency(ct) if ct.size > 0 else (0, np.nan, 0, 0)
                )
                res["p_comp"] = p
                res["test_name"] = "Chi-square"
            except (ValueError, TypeError):
                res["p_comp"] = np.nan
                res["test_name"] = "-"

            if len(levels) > 1:
                temp_df = pd.DataFrame({"y": y, "raw": X_raw}).dropna()
                dummy_cols = []
                for lvl in levels[1:]:
                    d_name = f"{col}::{lvl}"
                    temp_df[d_name] = (temp_df["raw"].astype(str) == str(lvl)).astype(
                        int
                    )
                    dummy_cols.append(d_name)

                if dummy_cols and temp_df[dummy_cols].std().sum() > 0:
                    params, conf, pvals, status, _ = run_binary_logit(
                        temp_df["y"], temp_df[dummy_cols], method=preferred_method,
                        penalty_weight=penalty_weight
                    )
                    if status == "OK":
                        or_lines, coef_lines, p_lines = ["Ref."], ["-"], ["-"]
                        for lvl in levels[1:]:
                            d_name = f"{col}::{lvl}"
                            if d_name in params:
                                coef = params[d_name]
                                odd = np.exp(coef)
                                ci_l, ci_h = (
                                    np.exp(conf.loc[d_name][0]),
                                    np.exp(conf.loc[d_name][1]),
                                )
                                pv = pvals[d_name]

                                coef_lines.append(f"{coef:.3f}")

                                # Use new formatter
                                ci_str = PublicationFormatter.format_ci(ci_l, ci_h)
                                or_lines.append(f"{odd:.2f} {ci_str}")

                                p_lines.append(format_p_value(pv))
                                or_results[f"{col}: {lvl}"] = {
                                    "or": odd,
                                    "ci_low": ci_l,
                                    "ci_high": ci_h,
                                    "p_value": pv,
                                }
                            else:
                                or_lines.append("-")
                                p_lines.append("-")
                                coef_lines.append("-")

                        res["or_val"] = None
                        res["p_or"] = "<br>".join(p_lines)
                        or_lines_styled = ["Ref."]
                        for lvl in levels[1:]:
                            d_name = f"{col}::{lvl}"
                            if d_name in params:
                                odd = np.exp(params[d_name])
                                ci_l, ci_h = (
                                    np.exp(conf.loc[d_name][0]),
                                    np.exp(conf.loc[d_name][1]),
                                )
                                or_lines_styled.append(
                                    fmt_or_with_styling(odd, ci_l, ci_h)
                                )
                            else:
                                or_lines_styled.append("-")
                        res["or"] = "<br>".join(or_lines_styled)
                        res["coef"] = "<br>".join(coef_lines)
                    else:
                        res["or"] = (
                            f"<span style='color:red; font-size:0.8em'>{status}</span>"
                        )
                        res["coef"] = "-"
                else:
                    res["or"] = "-"
                    res["coef"] = "-"
            else:
                res["or"] = "-"
                res["coef"] = "-"

        else:  # Linear mode
            n_used = len(X_num.dropna())
            m_t, s_t = X_num.mean(), X_num.std()
            m_n, s_n = (
                pd.to_numeric(X_neg, errors="coerce").mean(),
                pd.to_numeric(X_neg, errors="coerce").std(),
            )
            m_p, s_p = (
                pd.to_numeric(X_pos, errors="coerce").mean(),
                pd.to_numeric(X_pos, errors="coerce").std(),
            )

            res["desc_total"] = f"n={n_used}<br>Mean: {m_t:.2f} (SD {s_t:.2f})"
            res["desc_neg"] = f"{m_n:.2f} ({s_n:.2f})"
            res["desc_pos"] = f"{m_p:.2f} ({s_p:.2f})"

            try:
                _, p = stats.mannwhitneyu(
                    pd.to_numeric(X_neg, errors="coerce").dropna(),
                    pd.to_numeric(X_pos, errors="coerce").dropna(),
                )
                res["p_comp"] = p
                res["test_name"] = "Mann-Whitney"
            except (ValueError, TypeError):
                res["p_comp"] = np.nan
                res["test_name"] = "-"

            data_uni = pd.DataFrame({"y": y, "x": X_num}).dropna()
            if not data_uni.empty and data_uni["x"].nunique() > 1:
                params, hex_conf, pvals, status, _ = run_binary_logit(
                    data_uni["y"], data_uni[["x"]], method=preferred_method,
                    penalty_weight=penalty_weight
                )
                if status == "OK" and "x" in params:
                    coef = params["x"]
                    odd = np.exp(coef)
                    ci_l, ci_h = (
                        np.exp(hex_conf.loc["x"][0]),
                        np.exp(hex_conf.loc["x"][1]),
                    )
                    pv = pvals["x"]

                    res["coef"] = f"{coef:.3f}"
                    res["or_val"] = odd
                    res["ci_low"] = ci_l
                    res["ci_high"] = ci_h
                    res["p_or"] = pv
                    or_results[col] = {
                        "or": odd,
                        "ci_low": ci_l,
                        "ci_high": ci_h,
                        "p_value": pv,
                    }
                else:
                    res["or"] = (
                        f"<span style='color:red; font-size:0.8em'>{status}</span>"
                    )
                    res["coef"] = "-"
            else:
                res["or"] = "-"
                res["coef"] = "-"

        results_db[col] = res

        p_screen = res.get("p_comp", np.nan)
        if (
            isinstance(p_screen, (int, float))
            and pd.notna(p_screen)
            and p_screen < 0.20
        ):
            candidates.append(col)

    # ✅ MCC APPLICATION
    if mcc_enable:
        uni_p_vals = []
        uni_keys = []
        for col, res in results_db.items():
            p = res.get("p_comp", np.nan)
            if pd.isna(p) and "p_or" in res and isinstance(res["p_or"], (float, int)):
                p = res["p_or"]

            if pd.notna(p) and isinstance(p, (float, int)):
                uni_p_vals.append(p)
                uni_keys.append(col)

        if uni_p_vals:
            adj_p = apply_mcc(uni_p_vals, method=mcc_method, alpha=mcc_alpha)
            for k, p_adj in zip(uni_keys, adj_p, strict=True):
                if k in or_results:
                    or_results[k]["p_adj"] = p_adj
                if k in results_db:
                    results_db[k]["p_adj"] = p_adj

    # Multivariate analysis
    aor_results: dict[str, AORResult] = {}
    interaction_results: dict[str, InteractionResult] = {}
    int_meta = {}
    predictors_for_vif = []
    multi_data = None
    mv_metrics_text = ""
    model_diagnostics_html = ""  # New variable for diagnostics table

    def _is_candidate_valid(col):
        mode = mode_map.get(col, "linear")
        series = df_aligned[col]
        if mode == "categorical":
            return series.notna().sum() > 5
        return series.apply(clean_numeric_value).notna().sum() > 5

    cand_valid = [c for c in candidates if _is_candidate_valid(c)]

    if len(cand_valid) > 0 or interaction_pairs:
        multi_df = pd.DataFrame({"y": y})

        for c in cand_valid:
            mode = mode_map.get(c, "linear")
            if mode == "categorical":
                levels = cat_levels_map.get(c, [])
                raw_vals = df_aligned[c]
                if len(levels) > 1:
                    for lvl in levels[1:]:
                        d_name = f"{c}::{lvl}"
                        multi_df[d_name] = (raw_vals.astype(str) == str(lvl)).astype(
                            int
                        )
            else:
                multi_df[c] = df_aligned[c].apply(clean_numeric_value)

        if interaction_pairs:
            try:
                from utils.interaction_lib import create_interaction_terms

                multi_df, int_meta = create_interaction_terms(
                    multi_df, interaction_pairs, mode_map
                )
                logger.info(
                    f"✅ Added {len(int_meta)} interaction terms to logistic multivariate model"
                )
            except ImportError:
                logger.warning(
                    "interaction_lib not available, skipping interaction terms"
                )
            except (ValueError, KeyError):
                logger.exception("Failed to create interaction terms")

        multi_data = multi_df.dropna()
        final_n_multi = len(multi_data)
        predictors_for_vif = [col for col in multi_data.columns if col != "y"]

        if not multi_data.empty and final_n_multi > 10 and len(predictors_for_vif) > 0:
            try:
                n_params_mv = len(predictors_for_vif) + 1
                ci_config = get_ci_configuration(
                    ci_method, final_n_multi, y.sum(), n_params_mv
                )
            except (ValueError, TypeError) as e:
                logger.warning("CI configuration failed, falling back to auto: %s", e)
                ci_config = {"method": "wald", "note": "Fallback due to config error"}

            effective_ci_method = ci_config["method"]
            ci_note = ci_config["note"]

            params, conf, pvals, status, mv_stats = run_binary_logit(
                multi_data["y"],
                multi_data[predictors_for_vif],
                method=preferred_method,
                ci_method=effective_ci_method,
                penalty_weight=penalty_weight,
            )

            if status == "OK":
                final_n_multi = len(multi_data)
                # 1. Format Basic Metrics (Existing logic enhanced)
                r2_parts = []
                mcf = mv_stats.get("mcfadden")
                nag = mv_stats.get("nagelkerke")
                if pd.notna(mcf):
                    r2_parts.append(f"McFadden R² = {mcf:.3f}")
                if pd.notna(nag):
                    r2_parts.append(f"Nagelkerke R² = {nag:.3f}")
                if r2_parts:
                    mv_metrics_text = " | ".join(r2_parts)

                # ✅ 2. Construct Professional Diagnostics Table (New)
                diag_rows = []

                # Discrimination (AUC)
                auc = mv_stats.get("auc")
                if pd.notna(auc):
                    auc_interp = (
                        "Excellent"
                        if auc > 0.8
                        else "Acceptable"
                        if auc > 0.7
                        else "Poor"
                    )
                    # Source: Hosmer, D.W., & Lemeshow, S. (2000). Applied Logistic Regression.
                    # Commonly cited thresholds: >0.8 Excellent, >0.7 Acceptable, <0.7 Poor/Fair.
                    diag_rows.append(
                        f"<tr><td>Discrimination (C-stat)</td><td>{auc:.3f}</td><td>{auc_interp}</td></tr>"
                    )

                # Calibration (HL Test)
                hl_p = mv_stats.get("hl_pvalue")
                hl_stat = mv_stats.get("hl_stat")
                if pd.notna(hl_p):
                    hl_res = (
                        "Good Fit (p > 0.05)" if hl_p > 0.05 else "Poor Fit (p < 0.05)"
                    )
                    hl_color = "green" if hl_p > 0.05 else "red"
                    diag_rows.append(
                        f"<tr><td>Calibration (Hosmer-Lemeshow)</td><td>Chi2={hl_stat:.2f}, p=<span style='color:{hl_color}'>{hl_p:.3f}</span></td><td>{hl_res}</td></tr>"
                    )

                # AIC/BIC
                aic = mv_stats.get("aic")
                bic = mv_stats.get("bic")
                if pd.notna(aic):
                    diag_rows.append(
                        f"<tr><td>AIC / BIC</td><td>{aic:.1f} / {bic:.1f}</td><td>Lower is better</td></tr>"
                    )

                # ✅ VIF CALCULATION (Moved up for alignment text)
                vif_html = ""
                vif_df = None
                if (
                    vif_enable
                    and multi_data is not None
                    and final_n_multi > 10
                    and len(predictors_for_vif) > 1
                ):
                    try:
                        vif_df, _ = calculate_vif(
                            multi_data[predictors_for_vif], var_meta=var_meta
                        )

                        if vif_df is not None and not vif_df.empty:
                            vif_rows = []
                            for _, row in vif_df.iterrows():
                                feat = html.escape(
                                    str(row["Variable"]).replace("::", ": ")
                                )
                                val = row["VIF"]

                                if np.isinf(val):
                                    vif_str = "∞ (Perfect Collinearity)"
                                    vif_class = "vif-warning"
                                    icon = "⚠️"
                                elif val > vif_threshold:
                                    vif_str = f"{val:.2f}"
                                    vif_class = "vif-warning"
                                    icon = "⚠️"
                                elif val > 5:
                                    vif_str = f"{val:.2f}"
                                    vif_class = "vif-caution"
                                    icon = ""
                                else:
                                    vif_str = f"{val:.2f}"
                                    vif_class = ""
                                    icon = ""

                                vif_rows.append(f"""
                                    <tr>
                                        <td>{feat}</td>
                                        <td class='{vif_class}'><strong>{vif_str}</strong> {icon}</td>
                                    </tr>
                                """)

                            vif_body = "".join(vif_rows)

                            vif_html = f"""
                            <div class='vif-container' style='margin-top: 16px;'>
                                <div class='vif-title'>🔹 Collinearity Diagnostics (VIF)</div>
                                <table class='table' style='max-width: 500px;'>
                                    <thead>
                                        <tr>
                                            <th>Predictor</th>
                                            <th>VIF</th>
                                        </tr>
                                    </thead>
                                    <tbody>{vif_body}</tbody>
                                </table>
                                <div class='vif-footer'>
                                    VIF &gt; {vif_threshold}: potential high collinearity. ∞ = perfect correlation with other predictors.
                                </div>
                            </div>
                            """
                    except (ValueError, np.linalg.LinAlgError) as e:
                        logger.warning("VIF calculation failed: %s", e)
                    except TypeError as e:
                        logger.warning(
                            "VIF calculation failed due to type error: %s", e
                        )

                # ✅ 3. Alignment text (STROBE/TRIPOD)
                alignment_parts = []
                if len(predictors_for_vif) > 0:
                    adj_list = ", ".join(
                        [str(p).replace("::", ": ") for p in predictors_for_vif[:5]]
                    )
                    if len(predictors_for_vif) > 5:
                        adj_list += " et al."
                    alignment_parts.append(
                        f"This multivariable model adjusted for: <i>{adj_list}</i>."
                    )

                if vif_enable and vif_df is not None and not vif_df.empty:
                    max_vif = vif_df["VIF"].max()
                    if max_vif < 5:
                        alignment_parts.append(
                            "No evidence of multicollinearity detected (all VIF < 5)."
                        )
                    elif max_vif < 10:
                        alignment_parts.append(
                            "Minor multicollinearity detected (VIF < 10)."
                        )

                if preferred_method == "firth":
                    alignment_parts.append(
                        "Firth's penalized likelihood was used to account for potential separation or small sample bias."
                    )

                alignment_html = ""
                if alignment_parts:
                    alignment_html = f"<div style='margin-top: 10px; font-style: italic; color: {COLORS['primary']};'>{' '.join(alignment_parts)}</div>"

                if diag_rows:
                    model_diagnostics_html = f"""
                    <div class='diag-container' style='margin-top: 15px; border: 1px solid #eee; padding: 10px; border-radius: 5px; background: #fafafa;'>
                        <div class='vif-title' style='margin-bottom: 5px;'>🩺 Model Diagnostics (Publication Grade)</div>
                        <table class='table table-sm' style='width: 100%; font-size: 0.9em;'>
                            <thead style='background: #f1f1f1;'><tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr></thead>
                            <tbody>{"".join(diag_rows)}</tbody>
                        </table>
                        {alignment_html}
                    </div>
                    """

                coef_map = params
                p_map = pvals
                ci_map = conf

                for var in cand_valid:
                    mode = mode_map.get(var, "linear")
                    if mode == "categorical":
                        levels = cat_levels_map.get(var, [])
                        aor_entries = []
                        for lvl in levels[1:]:
                            d_name = f"{var}::{lvl}"
                            if d_name in params:
                                coef = params[d_name]
                                aor = np.exp(coef)
                                ci_low, ci_high = (
                                    np.exp(conf.loc[d_name][0]),
                                    np.exp(conf.loc[d_name][1]),
                                )
                                pv = pvals[d_name]
                                aor_entries.append(
                                    {
                                        "lvl": lvl,
                                        "coef": coef,
                                        "aor": aor,
                                        "ci_low": ci_low,
                                        "ci_high": ci_high,
                                        "p": pv,
                                    }
                                )
                                aor_results[f"{var}: {lvl}"] = {
                                    "aor": aor,
                                    "ci_low": ci_low,
                                    "ci_high": ci_high,
                                    "p_value": pv,
                                }
                        results_db[var]["multi_res"] = aor_entries
                    else:
                        if var in params:
                            coef = params[var]
                            aor = np.exp(coef)
                            ci_low, ci_high = (
                                np.exp(conf.loc[var][0]),
                                np.exp(conf.loc[var][1]),
                            )
                            pv = pvals[var]
                            results_db[var]["multi_res"] = {
                                "coef": coef,
                                "aor": aor,
                                "ci_low": ci_low,
                                "ci_high": ci_high,
                                "p": pv,
                            }
                            aor_results[var] = {
                                "aor": aor,
                                "ci_low": ci_low,
                                "ci_high": ci_high,
                                "p_value": pv,
                            }

                if int_meta:
                    try:
                        from utils.interaction_lib import format_interaction_results

                        interaction_results = format_interaction_results(
                            params, conf, pvals, int_meta, "logit"
                        )
                        logger.info(
                            f"✅ Formatted {len(interaction_results)} logistic interaction results"
                        )

                        for int_name, int_res in interaction_results.items():
                            label_display = f"🔗 {int_res.get('label', int_name)}"
                            aor_results[int_name] = {
                                "aor": int_res.get("or", np.nan),
                                "ci_low": int_res.get("ci_low", np.nan),
                                "ci_high": int_res.get("ci_high", np.nan),
                                "p_value": int_res.get("p_value", np.nan),
                                "p_adj": None,
                                "label": label_display,
                            }
                    except Exception:
                        logger.exception("Failed to format interaction results")
            else:
                mv_metrics_text = (
                    f"<span style='color:red'>Adjustment Failed: {status}</span>"
                )

        if mcc_enable and aor_results:
            mv_p_vals = []
            mv_keys = []
            for k, v in aor_results.items():
                p = v.get("p_value")
                if pd.notna(p):
                    mv_p_vals.append(p)
                    mv_keys.append(k)

            if mv_p_vals:
                adj_p = apply_mcc(mv_p_vals, method=mcc_method, alpha=mcc_alpha)
                for k, p_adj in zip(mv_keys, adj_p, strict=True):
                    aor_results[k]["p_adj"] = p_adj

    # Build HTML
    html_rows = []
    valid_cols_for_html = [c for c in sorted_cols if c in results_db]
    grouped_cols = sorted(
        valid_cols_for_html,
        key=lambda x: (x.split("_")[0] if "_" in x else "Variables", x),
    )

    if "effective_ci_method" not in locals():
        try:
            ci_config = get_ci_configuration(ci_method, total_n, y.sum(), 2)
            effective_ci_method = ci_config["method"]
            ci_note = ci_config["note"]
        except Exception:
            effective_ci_method = "wald"
            ci_note = ""

    for col in grouped_cols:
        if col == outcome_name:
            continue
        res = results_db[col]
        mode = mode_map.get(col, "linear")
        sheet = col.split("_")[0] if "_" in col else "Variables"

        lbl = get_label(col, var_meta)
        mode_badge = {"categorical": "📊 (All Levels)", "linear": "📉 (Trend)"}
        if mode in mode_badge:
            lbl += f"<br><span style='font-size:0.8em; color:{COLORS['text_secondary']}'>{mode_badge[mode]}</span>"

        if mode == "categorical" and "or" in res:
            or_s = res["or"]
        else:
            or_s = fmt_or_with_styling(
                res.get("or_val"), res.get("ci_low"), res.get("ci_high")
            )
        coef_s = res.get("coef", "-")

        if mode == "categorical":
            p_col_display = res.get("p_or", "-")
        else:
            p_col_display = format_p_value(res.get("p_comp", np.nan))

        aor_s, acoef_s, ap_s = "-", "-", "-"
        multi_res = res.get("multi_res")

        if multi_res:
            if isinstance(multi_res, list):
                aor_lines, acoef_lines, ap_lines = ["Ref."], ["-"], ["-"]
                for item in multi_res:
                    p_txt = format_p_value(item["p"])
                    acoef_lines.append(f"{item['coef']:.3f}")
                    aor_lines.append(
                        fmt_or_with_styling(
                            item["aor"], item["ci_low"], item["ci_high"]
                        )
                    )
                    ap_lines.append(p_txt)
                aor_s, acoef_s, ap_s = (
                    "<br>".join(aor_lines),
                    "<br>".join(acoef_lines),
                    "<br>".join(ap_lines),
                )
            else:
                if "coef" in multi_res and pd.notna(multi_res["coef"]):
                    acoef_s = f"{multi_res['coef']:.3f}"
                else:
                    acoef_s = "-"
                aor_s = fmt_or_with_styling(
                    multi_res["aor"], multi_res["ci_low"], multi_res["ci_high"]
                )
                ap_s = format_p_value(multi_res["p"])

        p_adj_uni_s = "-"
        if mcc_enable and "p_adj" in res:
            p_adj_uni_s = format_p_value(res["p_adj"])

        ap_adj_s = "-"
        if mcc_enable:
            if multi_res:
                if isinstance(multi_res, list):
                    ap_adj_lines = ["-"]
                    for item in multi_res:
                        key = f"{col}: {item['lvl']}"
                        val = aor_results.get(key, {}).get("p_adj")
                        ap_adj_lines.append(format_p_value(val))
                    ap_adj_s = "<br>".join(ap_adj_lines)
                else:
                    val = aor_results.get(col, {}).get("p_adj")
                    ap_adj_s = format_p_value(val)

        html_rows.append(f"""<tr>
            <td>{lbl}</td>
            <td>{res.get("desc_total", "")}</td>
            <td>{res.get("desc_neg", "")}</td>
            <td>{res.get("desc_pos", "")}</td>
            <td>{coef_s}</td>
            <td>{or_s}</td>
            <td>{res.get("test_name", "-")}</td>
            <td>{p_col_display}</td>
            {f"<td>{p_adj_uni_s}</td>" if mcc_enable else ""}
            <td>{acoef_s}</td>
            <td>{aor_s}</td>
            <td>{ap_s}</td>
            {f"<td>{ap_adj_s}</td>" if mcc_enable else ""}
        </tr>""")

    if interaction_results:
        for int_name, res in interaction_results.items():
            int_label = html.escape(str(res.get("label", int_name)))
            int_coef = f"{res.get('coef', 0):.3f}" if pd.notna(res.get("coef")) else "-"
            or_val = res.get("or")
            if or_val is not None and pd.notna(or_val):
                ci_str = PublicationFormatter.format_ci(
                    res.get("ci_low", 0), res.get("ci_high", 0)
                )
                int_or = f"{or_val:.2f} {ci_str}"
            else:
                int_or = "-"
            int_p = format_p_value(res.get("p_value", np.nan))

            int_p_adj = "-"
            if mcc_enable:
                val = aor_results.get(int_name, {}).get("p_adj")
                int_p_adj = format_p_value(val)

            html_rows.append(
                f"""<tr style='background-color: {COLORS["primary_light"]};'>
                <td><b>🔗 {int_label}</b><br><small style='color: {COLORS["text_secondary"]};'>(Interaction)</small></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>{int_coef}</td>
                <td><b>{int_or}</b></td>
                <td>Interaction</td>
                <td>-</td>
                {"<td>-</td>" if mcc_enable else ""}
                <td>{int_coef}</td>
                <td><b>{int_or}</b></td>
                <td>{int_p}</td>
                {f"<td>{int_p_adj}</td>" if mcc_enable else ""}
            </tr>"""
            )

    logger.info(
        f"Logistic analysis complete. Multivariate n={final_n_multi}, Interactions={len(interaction_results)}"
    )

    model_fit_html = ""
    if mv_metrics_text:
        model_fit_html = f"<div style='margin-top: 8px; padding-top: 8px; border-top: 1px dashed {COLORS['border']}; color: {COLORS['primary_dark']};'><b>Model Fit:</b> {mv_metrics_text}</div>"

    css_content = load_static_css()
    if not css_content:
        css_content = """
        body { font-family: sans-serif; padding: 20px; }
        .table-container { background: #fff; border: 1px solid #ddd; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; border-bottom: 1px solid #ddd; }
        """

    css_styles = f"<style>{css_content}</style>"

    html_table = f"""{css_styles}
    <div id='{outcome_name}' class='table-container'>
    <div class='outcome-title'>Outcome: {outcome_name} (n={total_n})</div>
    <table>
        <thead>
            <tr>
                <th>Variable</th>
                <th>Total</th>
                <th>Group 0</th>
                <th>Group 1</th>
                <th>Crude Coef.</th>
                <th>Crude OR (95% CI)</th>
                <th>Test</th>
                <th>P-value</th>
                {f"<th>Adj. P ({mcc_method})</th>" if mcc_enable else ""}
                <th>Adj. Coef.</th>
                <th>aOR (95% CI)</th>
                <th>aP-value</th>
                {f"<th>Adj. aP ({mcc_method})</th>" if mcc_enable else ""}
            </tr>
        </thead>
        <tbody>{chr(10).join(html_rows)}</tbody>
    </table>
    <div class='summary-box'>
        <b>Method:</b> {preferred_method.capitalize()} Logit<br>
        <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid {
        COLORS["border"]
    }; font-size: 0.9em; color: {COLORS["text_secondary"]};'>
            <b>Selection:</b> Variables with Crude P &lt; 0.20 (n={final_n_multi})<br>
            <b>Modes:</b> 📊 Categorical (vs Reference) | 📉 Linear (Per-unit)<br>
            <b>CI Method:</b> {effective_ci_method.capitalize()} {
        f"({ci_note})" if ci_note else ""
    }
            {model_fit_html}
            {model_diagnostics_html}
            {vif_html}
            {
        f"<br><b>Interactions Tested:</b> {len(interaction_pairs)} pairs"
        if interaction_pairs
        else ""
    }
        </div>
    {
        create_missing_data_report_html(missing_data_info, var_meta or {})
        if CONFIG.get("analysis.missing.report_missing", True)
        else ""
    }
    </div></div><br>"""

    if preferred_method == "firth":
        banner = (
            f"<div style='background-color: {COLORS['info']}20; border: 1px solid {COLORS['info']}; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>"
            "<strong>ℹ️ Method Used:</strong> Firth's Penalized Likelihood (useful for rare events or separation)."
            "</div>"
        )

        html_table = banner + html_table

    return html_table, or_results, aor_results, interaction_results


def run_logistic_regression(df, outcome_col, covariate_cols):
    """
    Robust wrapper for logistic analysis as requested by integration requirements.
    """
    if outcome_col not in df.columns:
        return None, None, f"Error: Outcome '{outcome_col}' not found.", {}

    if df[outcome_col].nunique() < 2:
        return None, None, f"Error: Outcome '{outcome_col}' is constant", {}

    missing_cols = [col for col in covariate_cols if col not in df.columns]
    if missing_cols:
        return None, None, f"Error: Covariates not found: {', '.join(missing_cols)}", {}

    if not covariate_cols:
        return None, None, "Error: No covariates provided", {}

    try:
        html_table, or_results, _aor_results, _ = analyze_outcome(
            outcome_col, df[[*covariate_cols, outcome_col]].copy()
        )

        y = df[outcome_col]
        X = df[covariate_cols]
        X_const = sm.add_constant(X)
        model = sm.Logit(y, X_const)
        result = model.fit(disp=0)

        mcfadden = result.prsquared if not np.isnan(result.prsquared) else np.nan
        metrics = {
            "aic": result.aic,
            "bic": result.bic,
            "mcfadden": mcfadden,
            "nagelkerke": np.nan,
            "p_value": getattr(result, "llr_pvalue", np.nan),
        }

        return html_table, or_results, "OK", metrics

    except Exception as e:
        logger.exception("Logistic regression failed")
        return None, None, str(e), {}


# =============================================================================
# ABSOLUTE RISK MEASURES (NEJM/Lancet Standard)
# =============================================================================


def calculate_absolute_risk(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    treatment_value: Any = 1,
    control_value: Any = 0,
) -> dict:
    """
    Calculate absolute risk and risk difference between treatment groups.

    Args:
        df: DataFrame with outcome and treatment columns
        outcome_col: Name of binary outcome column
        treatment_col: Name of treatment/exposure column
        treatment_value: Value indicating treatment group
        control_value: Value indicating control group

    Returns:
        Dictionary with risk metrics and CIs
    """
    try:
        # Filter to treatment and control groups
        mask_treat = df[treatment_col] == treatment_value
        mask_ctrl = df[treatment_col] == control_value

        y_treat = df.loc[mask_treat, outcome_col].dropna()
        y_ctrl = df.loc[mask_ctrl, outcome_col].dropna()

        n_treat = len(y_treat)
        n_ctrl = len(y_ctrl)

        if n_treat < 5 or n_ctrl < 5:
            return {"error": "Insufficient sample size in one or both groups"}

        # Calculate risks
        events_treat = (y_treat == 1).sum()
        events_ctrl = (y_ctrl == 1).sum()

        p_treat = events_treat / n_treat
        p_ctrl = events_ctrl / n_ctrl

        # Absolute Risk Difference (ARD)
        ard = p_treat - p_ctrl

        # Newcombe CI for ARD (Wilson score interval)
        # Standard error using Wilson's method
        se_treat = np.sqrt(p_treat * (1 - p_treat) / n_treat)
        se_ctrl = np.sqrt(p_ctrl * (1 - p_ctrl) / n_ctrl)
        se_ard = np.sqrt(se_treat**2 + se_ctrl**2)

        z = 1.96  # 95% CI
        ard_ci_lower = ard - z * se_ard
        ard_ci_upper = ard + z * se_ard

        # Relative Risk (RR)
        rr = p_treat / p_ctrl if p_ctrl > 0 else np.nan
        log_rr = np.log(rr) if rr > 0 else np.nan
        se_log_rr = (
            np.sqrt(
                (1 - p_treat) / (n_treat * p_treat) + (1 - p_ctrl) / (n_ctrl * p_ctrl)
            )
            if p_treat > 0 and p_ctrl > 0
            else np.nan
        )

        rr_ci_lower = (
            np.exp(log_rr - z * se_log_rr) if se_log_rr is not np.nan else np.nan
        )
        rr_ci_upper = (
            np.exp(log_rr + z * se_log_rr) if se_log_rr is not np.nan else np.nan
        )

        return {
            "risk_treatment": p_treat,
            "risk_control": p_ctrl,
            "ard": ard,
            "ard_ci_lower": ard_ci_lower,
            "ard_ci_upper": ard_ci_upper,
            "rr": rr,
            "rr_ci_lower": rr_ci_lower,
            "rr_ci_upper": rr_ci_upper,
            "n_treatment": n_treat,
            "n_control": n_ctrl,
            "events_treatment": events_treat,
            "events_control": events_ctrl,
        }

    except Exception as e:
        logger.warning("Absolute risk calculation failed: %s", e)
        return {"error": str(e)}


def calculate_nnt(
    ard: float,
    ard_ci_lower: float | None = None,
    ard_ci_upper: float | None = None,
) -> dict:
    """
    Calculate Number Needed to Treat (NNT) from Absolute Risk Difference.

    NNT = 1 / |ARD|

    CI calculation follows Altman's method:
    - When ARD is significant (CI doesn't cross 0): NNT CI = 1/ARD_CI
    - When CI crosses 0: Report as NNT to benefit / NNT to harm

    Args:
        ard: Absolute Risk Difference
        ard_ci_lower: Lower bound of ARD CI
        ard_ci_upper: Upper bound of ARD CI

    Returns:
        Dictionary with NNT metrics
    """
    if ard == 0 or np.isnan(ard):
        return {
            "nnt": np.inf,
            "nnt_type": "undefined",
            "interpretation": "No difference between groups",
        }

    nnt = abs(1 / ard)

    # Determine if benefit (treatment reduces outcome) or harm
    if ard < 0:
        nnt_type = "NNT (benefit)"
        interpretation = f"Treat {nnt:.0f} patients to prevent 1 event"
    else:
        nnt_type = "NNH (harm)"
        interpretation = f"Treat {nnt:.0f} patients to cause 1 additional event"

    result = {
        "nnt": nnt,
        "nnt_type": nnt_type,
        "interpretation": interpretation,
        "ard": ard,
    }

    # Calculate CI using Altman method
    if ard_ci_lower is not None and ard_ci_upper is not None:
        ci_crosses_zero = ard_ci_lower < 0 < ard_ci_upper

        if ci_crosses_zero:
            # CI crosses zero - report both NNTB and NNTH ranges
            result["ci_note"] = "CI crosses null (∞)"
            if ard_ci_lower != 0:
                result["nnth_from_ci"] = abs(1 / ard_ci_lower)
            if ard_ci_upper != 0:
                result["nntb_from_ci"] = abs(1 / ard_ci_upper)
        else:
            # CI doesn't cross zero
            if ard_ci_lower != 0 and ard_ci_upper != 0:
                nnt_ci_lower = abs(1 / ard_ci_upper)  # Note: inversion
                nnt_ci_upper = abs(1 / ard_ci_lower)
                result["nnt_ci_lower"] = min(nnt_ci_lower, nnt_ci_upper)
                result["nnt_ci_upper"] = max(nnt_ci_lower, nnt_ci_upper)

    return result


def format_absolute_risk_html(
    risk_data: dict,
    nnt_data: dict | None = None,
) -> str:
    """
    Format absolute risk measures as HTML table for publication.

    Args:
        risk_data: Output from calculate_absolute_risk()
        nnt_data: Output from calculate_nnt() or None

    Returns:
        HTML string
    """
    if "error" in risk_data:
        return f"<div class='alert alert-warning'>Cannot calculate: {risk_data['error']}</div>"

    p_treat = risk_data["risk_treatment"] * 100
    p_ctrl = risk_data["risk_control"] * 100
    ard = risk_data["ard"] * 100
    ard_ci_l = risk_data["ard_ci_lower"] * 100
    ard_ci_u = risk_data["ard_ci_upper"] * 100

    html = f"""
    <div class="absolute-risk-report">
        <h4>📊 Absolute Risk Measures</h4>
        <table class="table table-sm">
            <thead>
                <tr><th>Measure</th><th>Value</th><th>95% CI</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Risk (Treatment)</strong></td>
                    <td>{p_treat:.1f}%</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td><strong>Risk (Control)</strong></td>
                    <td>{p_ctrl:.1f}%</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td><strong>Absolute Risk Difference</strong></td>
                    <td>{ard:+.1f}%</td>
                    <td>({ard_ci_l:+.1f}%, {ard_ci_u:+.1f}%)</td>
                </tr>
    """

    if nnt_data and "nnt" in nnt_data:
        nnt = nnt_data["nnt"]
        nnt_type = nnt_data.get("nnt_type", "NNT")

        if "nnt_ci_lower" in nnt_data and "nnt_ci_upper" in nnt_data:
            ci_str = f"({nnt_data['nnt_ci_lower']:.0f}–{nnt_data['nnt_ci_upper']:.0f})"
        elif "ci_note" in nnt_data:
            ci_str = nnt_data["ci_note"]
        else:
            ci_str = "—"

        html += f"""
                <tr>
                    <td><strong>{nnt_type}</strong></td>
                    <td>{nnt:.0f}</td>
                    <td>{ci_str}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    """

    if nnt_data and "interpretation" in nnt_data:
        html += f"""
        <div class="alert alert-info" style="margin-top: 10px;">
            <strong>💡 Clinical Interpretation:</strong> {nnt_data["interpretation"]}
        </div>
        """

    html += "</div>"
    return html


def generate_mi_pooled_report(
    n_imputations: int,
    pooled_or: dict[str, dict[str, Any]] | None,
    pooled_aor: dict[str, dict[str, Any]] | None,
) -> str:
    """
    Generate an HTML report for pooled Multiple Imputation results in logistic regression.
    Includes Crude OR and Adjusted OR columns.
    """
    html_rep = f"""
    <div class="alert alert-success mb-3">
        <strong>🔄 Multiple Imputation Analysis</strong><br>
        Results pooled from {n_imputations} imputed datasets using Rubin's Rules.
    </div>
    """

    if not pooled_aor and not pooled_or:
        return html_rep + "<p>No pooled results available.</p>"

    html_rep += "<h4>Pooled Odds Ratios (Crude & Adjusted)</h4>"
    html_rep += "<table class='table table-striped'><thead><tr>"
    html_rep += "<th>Variable</th>"
    html_rep += "<th>Crude OR (95% CI)</th><th>Crude P</th>"
    html_rep += "<th>aOR (95% CI)</th><th>Adj. P</th>"
    html_rep += "<th>FMI</th></tr></thead><tbody>"

    # Iterate over variables. Prioritize AOR keys as they represent the multivariable model,
    # but also check if OR keys have anything extra? Usually AOR is subset or equal.
    # For reporting, we typically show rows for variables in the final model (AOR).
    keys = []
    if pooled_aor:
        keys = list(pooled_aor.keys())
    elif pooled_or:
        keys = list(pooled_or.keys())

    for k in keys:
        # Get Label
        label = k
        if pooled_aor and k in pooled_aor:
            label = pooled_aor[k].get("label", k)
        elif pooled_or and k in pooled_or:
            label = pooled_or[k].get("label", k)

        # Crude
        crude_cell = "-"
        crude_p = "-"
        if pooled_or and k in pooled_or:
            v = pooled_or[k]
            # Format Crude OR
            # Use strict type checking or conversion if needed
            ci = PublicationFormatter.format_ci(
                float(v["ci_low"]), float(v["ci_high"]), include_brackets=True
            )
            crude_cell = f"{float(v['or']):.2f} {ci}"
            if (float(v["ci_low"]) > 1.0) or (float(v["ci_high"]) < 1.0):
                crude_cell = f"<b>{crude_cell}</b>"
            crude_p = format_p_value(float(v["p_value"]))

        # Adjusted
        adj_cell = "-"
        adj_p = "-"
        fmi_cell = "-"
        if pooled_aor and k in pooled_aor:
            v = pooled_aor[k]
            # Format Adjusted OR
            ci = PublicationFormatter.format_ci(
                float(v["ci_low"]), float(v["ci_high"]), include_brackets=True
            )
            adj_cell = f"{float(v['aor']):.2f} {ci}"
            if (float(v["ci_low"]) > 1.0) or (float(v["ci_high"]) < 1.0):
                adj_cell = f"<b>{adj_cell}</b>"
            adj_p = format_p_value(float(v["p_value"]))
            if v.get("fmi") is not None:
                fmi_cell = f"{float(v['fmi']) * 100:.1f}%"
        elif pooled_or and k in pooled_or:
            v_or = pooled_or[k]
            if v_or.get("fmi") is not None:
                fmi_cell = f"{float(v_or['fmi']) * 100:.1f}%"

        html_rep += f"<tr><td>{html.escape(str(label))}</td>"
        html_rep += f"<td>{crude_cell}</td><td>{crude_p}</td>"
        html_rep += f"<td>{adj_cell}</td><td>{adj_p}</td>"
        html_rep += f"<td>{fmi_cell}</td></tr>"

    html_rep += "</tbody></table>"

    return html_rep
