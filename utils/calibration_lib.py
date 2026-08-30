"""
📊 Calibration Library for Publication-Quality Model Validation

Provides calibration metrics and plots essential for NEJM/Lancet standards:
- Calibration plots (observed vs predicted)
- Brier score
- Calibration slope/intercept
- Net Benefit / Decision Curve Analysis

References:
    Steyerberg EW. Clinical Prediction Models. 2nd ed. Springer; 2019.
    Vickers AJ, Elkin EB. Med Decis Making. 2006;26(6):565-574.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


# =============================================================================
# CALIBRATION METRICS
# =============================================================================


def calculate_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate Brier Score and its decomposition.

    Brier Score ranges from 0 (perfect) to 1 (worst).
    Brier < 0.25 is generally acceptable for clinical prediction.

    Args:
        y_true: Binary outcome (0/1)
        y_pred: Predicted probabilities

    Returns:
        Dictionary with Brier score and interpretation
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Remove NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        brier = brier_score_loss(y_true, y_pred)

        # Reference Brier (using prevalence as prediction)
        prevalence = np.mean(y_true)
        brier_ref = prevalence * (1 - prevalence)

        # Scaled Brier (0 = perfect, 1 = no better than reference)
        brier_scaled = 1 - (brier / brier_ref) if brier_ref > 0 else np.nan

        interpretation = (
            "Excellent"
            if brier < 0.1
            else "Good"
            if brier < 0.2
            else "Acceptable"
            if brier < 0.25
            else "Poor"
        )

        return {
            "brier_score": brier,
            "brier_scaled": brier_scaled,
            "brier_reference": brier_ref,
            "interpretation": interpretation,
            "n": len(y_true),
        }
    except Exception as e:
        logger.warning("Brier score calculation failed: %s", e)
        return {"brier_score": np.nan, "error": str(e)}


def calculate_ici(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    span: float = 0.75,
    n_bootstrap: int = 100,
) -> dict:
    """
    Calculate Integrated Calibration Index (ICI), E50, E90, and Emax.

    According to Austin & Steyerberg (2019) and TRIPOD+AI standards:
    - ICI: Weighted absolute difference between predicted and LOESS-smoothed observed probabilities.
    - E50: Median absolute calibration error.
    - E90: 90th percentile of absolute calibration error.
    - Emax: Maximum absolute calibration error.

    References:
        Austin PC, Steyerberg EW. Stat Med. 2019;38(21):4051-4065.
        Collins GS, et al. BMJ. 2024;385:e078378 (TRIPOD+AI Statement).
    """
    try:
        y_true = np.asarray(y_true, dtype=float).flatten()
        y_pred = np.asarray(y_pred, dtype=float).flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = np.clip(y_pred[mask], 1e-6, 1.0 - 1e-6)

        n = len(y_true)
        if n < 10 or len(np.unique(y_true)) < 2:
            return {
                "ici": np.nan,
                "e50": np.nan,
                "e90": np.nan,
                "emax": np.nan,
                "error": "Insufficient sample size or classes for continuous calibration",
            }

        from statsmodels.nonparametric.smoothers_lowess import lowess

        # Fit LOESS: y_true ~ y_pred
        smooth_res = lowess(
            y_true, y_pred, frac=span, it=0, is_sorted=False, return_sorted=True
        )
        loess_x = smooth_res[:, 0]
        loess_y = np.clip(smooth_res[:, 1], 0.0, 1.0)

        # Interpolate smoothed observed probability for each predicted probability
        y_obs_smooth = np.interp(y_pred, loess_x, loess_y)
        abs_errors = np.abs(y_pred - y_obs_smooth)

        ici = float(np.mean(abs_errors))
        e50 = float(np.median(abs_errors))
        e90 = float(np.percentile(abs_errors, 90))
        emax = float(np.max(abs_errors))

        # Bootstrap 95% Confidence Intervals if sample size allows
        ici_boots = []
        if n_bootstrap > 0 and n >= 30:
            rng = np.random.default_rng(42)
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                if len(np.unique(y_true[idx])) < 2:
                    continue
                try:
                    b_smooth = lowess(
                        y_true[idx],
                        y_pred[idx],
                        frac=span,
                        it=0,
                        is_sorted=False,
                        return_sorted=True,
                    )
                    b_obs = np.clip(
                        np.interp(y_pred[idx], b_smooth[:, 0], b_smooth[:, 1]), 0.0, 1.0
                    )
                    ici_boots.append(np.mean(np.abs(y_pred[idx] - b_obs)))
                except Exception:
                    continue

        ici_ci_lower = (
            float(np.percentile(ici_boots, 2.5)) if len(ici_boots) >= 20 else np.nan
        )
        ici_ci_upper = (
            float(np.percentile(ici_boots, 97.5)) if len(ici_boots) >= 20 else np.nan
        )

        interpretation = (
            "Excellent (ICI < 0.02)"
            if ici < 0.02
            else "Good (ICI < 0.05)"
            if ici < 0.05
            else "Moderate (ICI < 0.10)"
            if ici < 0.10
            else "Poor (Recalibration Recommended)"
        )

        return {
            "ici": ici,
            "ici_ci_lower": ici_ci_lower,
            "ici_ci_upper": ici_ci_upper,
            "e50": e50,
            "e90": e90,
            "emax": emax,
            "interpretation": interpretation,
            "loess_x": loess_x.tolist(),
            "loess_y": loess_y.tolist(),
            "n": n,
        }
    except Exception as e:
        logger.warning("ICI calculation failed: %s", e)
        return {
            "ici": np.nan,
            "e50": np.nan,
            "e90": np.nan,
            "emax": np.nan,
            "error": str(e),
        }


def calculate_calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate calibration slope and intercept via logistic regression.

    Perfect calibration: slope = 1, intercept = 0

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities

    Returns:
        Dictionary with slope, intercept, and CIs
    """
    try:
        import statsmodels.api as sm

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Remove NaN and clip predictions
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = np.clip(y_pred[mask], 1e-6, 1 - 1e-6)

        # Logit transform of predictions
        logit_pred = np.log(y_pred / (1 - y_pred))

        # Fit logistic regression: logit(y) ~ intercept + slope * logit(pred)
        X = sm.add_constant(logit_pred)
        model = sm.Logit(y_true, X)
        result = model.fit(disp=0)

        intercept = result.params[0]
        slope = result.params[1]

        # Handle both DataFrame and numpy array from conf_int()
        ci = result.conf_int()
        if hasattr(ci, "iloc"):
            ci_slope = ci.iloc[1].tolist()
            ci_intercept = ci.iloc[0].tolist()
        else:
            # numpy array case
            ci_slope = [float(ci[1, 0]), float(ci[1, 1])]
            ci_intercept = [float(ci[0, 0]), float(ci[0, 1])]

        # Interpretation
        slope_status = (
            "✅ Well calibrated" if 0.8 <= slope <= 1.2 else "⚠️ Needs recalibration"
        )
        intercept_status = (
            "✅ Good calibration-in-the-large"
            if abs(intercept) < 0.2
            else "⚠️ Systematic over/underestimation"
        )

        return {
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "slope_ci_lower": ci_slope[0],
            "slope_ci_upper": ci_slope[1],
            "intercept_ci_lower": ci_intercept[0],
            "intercept_ci_upper": ci_intercept[1],
            "slope_interpretation": slope_status,
            "intercept_interpretation": intercept_status,
        }
    except Exception as e:
        logger.warning("Calibration slope calculation failed: %s", e)
        return {"calibration_slope": np.nan, "error": str(e)}


def calculate_c_statistic_with_ci(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05
) -> dict:
    """
    Calculate C-statistic (AUC) with confidence interval.

    Uses DeLong method for variance estimation.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        alpha: Significance level for CI

    Returns:
        Dictionary with C-statistic and 95% CI
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(np.unique(y_true)) < 2:
            return {"c_statistic": np.nan, "error": "Only one class present in outcome"}

        auc = roc_auc_score(y_true, y_pred)
        n = len(y_true)

        # Variance using Hanley-McNeil approximation
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)

        q1 = auc / (2 - auc)
        q2 = (2 * auc * auc) / (1 + auc)

        var_auc = (
            auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)
        ) / (n1 * n0)

        se_auc = np.sqrt(var_auc)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = max(0, auc - z * se_auc)
        ci_upper = min(1, auc + z * se_auc)

        interpretation = (
            "Excellent"
            if auc >= 0.9
            else "Good"
            if auc >= 0.8
            else "Acceptable"
            if auc >= 0.7
            else "Poor"
        )

        return {
            "c_statistic": auc,
            "se": se_auc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": n,
            "interpretation": interpretation,
        }
    except Exception as e:
        logger.warning("C-statistic calculation failed: %s", e)
        return {"c_statistic": np.nan, "error": str(e)}


# =============================================================================
# CALIBRATION PLOTS
# =============================================================================


def create_calibration_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Plot (TRIPOD+AI)",
    strategy: str = "quantile",
    show_loess: bool = True,
) -> go.Figure:
    """
    Create a publication-grade calibration plot with LOESS smooth curve, binned points, and histogram.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        n_bins: Number of bins for grouping
        title: Plot title
        strategy: 'uniform' or 'quantile' binning
        show_loess: Whether to display continuous LOESS smoothed calibration curve

    Returns:
        Plotly Figure object
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Compute continuous ICI & LOESS
        ici_res = calculate_ici(y_true, y_pred)

        # Use sklearn's calibration_curve for binned points
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy=strategy
        )

        # Create figure
        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1.5),
                name="Ideal (Slope = 1, Intercept = 0)",
                showlegend=True,
            )
        )

        # Continuous LOESS smoothed curve if available
        if show_loess and "loess_x" in ici_res and len(ici_res["loess_x"]) > 1:
            fig.add_trace(
                go.Scatter(
                    x=ici_res["loess_x"],
                    y=ici_res["loess_y"],
                    mode="lines",
                    line=dict(color=COLORS.get("primary", "#1E3A5F"), width=3),
                    name=f"LOESS Smooth (ICI: {ici_res.get('ici', 0):.3f})",
                    showlegend=True,
                )
            )

        # Observed vs predicted binned points
        fig.add_trace(
            go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode="markers",
                marker=dict(
                    size=9,
                    color=COLORS.get("accent", "#3B82F6"),
                    symbol="circle",
                    line=dict(width=1.5, color="white"),
                ),
                name=f"Binned Deciles ({strategy.title()})",
                showlegend=True,
            )
        )

        # Add histogram/distribution of predictions at bottom
        fig.add_trace(
            go.Histogram(
                x=y_pred,
                nbinsx=30,
                marker=dict(
                    color=COLORS.get("secondary", "#64748B"),
                    opacity=0.35,
                ),
                yaxis="y2",
                name="Risk Distribution",
                showlegend=False,
            )
        )

        # Subtitle metrics annotation
        metrics_subtitle = ""
        if not np.isnan(ici_res.get("ici", np.nan)):
            metrics_subtitle = (
                f"ICI: {ici_res['ici']:.3f} | E50: {ici_res.get('e50', 0):.3f} | "
                f"E90: {ici_res.get('e90', 0):.3f} | Emax: {ici_res.get('emax', 0):.3f}"
            )

        # Layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>"
                + (
                    f"<br><span style='font-size: 11px; color: #4B5563;'>{metrics_subtitle}</span>"
                    if metrics_subtitle
                    else ""
                ),
                x=0.5,
            ),
            xaxis=dict(
                title="Predicted Probability",
                range=[0, 1],
                tickformat=".1f",
                domain=[0, 1],
            ),
            yaxis=dict(
                title="Observed Proportion",
                range=[0, 1],
                tickformat=".1f",
                domain=[0.22, 1],  # Top 78% for calibration plot
            ),
            yaxis2=dict(
                domain=[0, 0.16],  # Bottom 16% for histogram
                showticklabels=False,
                showgrid=False,
                title="Density",
            ),
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
            height=460,
        )

        return fig
    except Exception as e:
        logger.exception("Calibration plot creation failed: %s", e)
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {e}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig
    except Exception as e:
        logger.exception("Calibration plot creation failed: %s", e)
        # Return empty figure with error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {e}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig


# =============================================================================
# DECISION CURVE ANALYSIS
# =============================================================================


def calculate_net_benefit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate Net Benefit for Decision Curve Analysis.

    Net Benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        thresholds: Probability thresholds to evaluate

    Returns:
        DataFrame with threshold, net_benefit_model, net_benefit_all
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)

    results = []
    prevalence = np.mean(y_true)

    for thresh in thresholds:
        # Model predictions
        pred_positive = y_pred >= thresh
        tp = np.sum((pred_positive) & (y_true == 1))
        fp = np.sum((pred_positive) & (y_true == 0))

        # Net benefit for model
        if thresh < 1:
            nb_model = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        else:
            nb_model = 0

        # Net benefit for "treat all"
        if thresh < 1:
            nb_all = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
        else:
            nb_all = 0

        results.append(
            {
                "threshold": thresh,
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": 0,
            }
        )

    return pd.DataFrame(results)


def create_decision_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Decision Curve Analysis",
) -> go.Figure:
    """
    Create Decision Curve Analysis plot.

    Shows net benefit across risk thresholds compared to
    'treat all' and 'treat none' strategies.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        title: Plot title

    Returns:
        Plotly Figure object
    """
    nb_df = calculate_net_benefit(y_true, y_pred)

    fig = go.Figure()

    # Model line
    fig.add_trace(
        go.Scatter(
            x=nb_df["threshold"],
            y=nb_df["net_benefit_model"],
            mode="lines",
            line=dict(color=COLORS.get("primary", "#1E3A5F"), width=2),
            name="Model",
        )
    )

    # Treat All line
    fig.add_trace(
        go.Scatter(
            x=nb_df["threshold"],
            y=nb_df["net_benefit_all"],
            mode="lines",
            line=dict(color=COLORS.get("warning", "#F59E0B"), width=1.5, dash="dash"),
            name="Treat All",
        )
    )

    # Treat None line
    fig.add_trace(
        go.Scatter(
            x=nb_df["threshold"],
            y=nb_df["net_benefit_none"],
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            name="Treat None",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis=dict(
            title="Threshold Probability",
            range=[0, 1],
            tickformat=".1f",
        ),
        yaxis=dict(
            title="Net Benefit",
        ),
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(x=0.7, y=0.98),
        height=400,
    )

    return fig


# =============================================================================
# HOSMER-LEMESHOW TEST (Surfacing from logic.py)
# =============================================================================


def hosmer_lemeshow_test(y_true: np.ndarray, y_pred: np.ndarray, g: int = 10) -> dict:
    """
    Perform Hosmer-Lemeshow goodness-of-fit test.

    H0: Model fits well (predicted = observed)
    p > 0.05 suggests adequate fit

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        g: Number of groups (typically 10)

    Returns:
        Dictionary with chi2, p-value, and interpretation
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Sort by predicted probability and create groups
        order = np.argsort(y_pred)
        y_true_sorted = y_true[order]
        y_pred_sorted = y_pred[order]

        # Split into g groups
        groups = np.array_split(np.arange(len(y_true)), g)

        chi2 = 0.0
        df = g - 2  # Degrees of freedom

        for group_idx in groups:
            n_group = len(group_idx)
            if n_group == 0:
                continue

            obs_events = np.sum(y_true_sorted[group_idx])
            exp_events = np.sum(y_pred_sorted[group_idx])

            # Avoid division by zero
            if exp_events > 0 and exp_events < n_group:
                chi2 += ((obs_events - exp_events) ** 2) / (
                    exp_events * (1 - exp_events / n_group)
                )

        p_value = 1 - stats.chi2.cdf(chi2, df) if df > 0 else np.nan

        interpretation = (
            "✅ Good fit (p ≥ 0.05)" if p_value >= 0.05 else "⚠️ Poor fit (p < 0.05)"
        )

        return {
            "chi2": chi2,
            "df": df,
            "p_value": p_value,
            "interpretation": interpretation,
            "g": g,
        }
    except Exception as e:
        logger.warning("Hosmer-Lemeshow test failed: %s", e)
        return {"chi2": np.nan, "p_value": np.nan, "error": str(e)}


# =============================================================================
# COMPREHENSIVE CALIBRATION REPORT
# =============================================================================


def get_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Generate comprehensive calibration report for publication (TRIPOD+AI Compliant).

    Combines discrimination (C-statistic/AUC), continuous calibration (ICI, E50, E90, Emax),
    calibration-in-the-large, calibration slope, Brier score, and goodness-of-fit.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities

    Returns:
        Dictionary with all calibration and discrimination metrics
    """
    return {
        "c_statistic": calculate_c_statistic_with_ci(y_true, y_pred),
        "ici": calculate_ici(y_true, y_pred),
        "brier": calculate_brier_score(y_true, y_pred),
        "calibration": calculate_calibration_slope(y_true, y_pred),
        "hosmer_lemeshow": hosmer_lemeshow_test(y_true, y_pred),
    }


def format_calibration_html(report: dict) -> str:
    """
    Format calibration report as HTML table adhering to TRIPOD+AI standards.

    Args:
        report: Output from get_calibration_report()

    Returns:
        HTML string
    """
    c_stat = report.get("c_statistic", {})
    ici_dict = report.get("ici", {})
    brier = report.get("brier", {})
    calib = report.get("calibration", {})
    hl = report.get("hosmer_lemeshow", {})

    html = """
    <div class="calibration-report">
        <h5 class="mb-2">📊 Model Discrimination & Calibration Assessment (TRIPOD+AI)</h5>
        <div class="table-responsive">
            <table class="table table-sm table-bordered align-middle">
                <thead class="table-light">
                    <tr>
                        <th style="width: 30%;">Metric</th>
                        <th style="width: 35%;">Value (95% CI)</th>
                        <th style="width: 35%;">Interpretation</th>
                    </tr>
                </thead>
                <tbody>
    """

    # 1. Discrimination (C-statistic)
    if "c_statistic" in c_stat and not np.isnan(c_stat.get("c_statistic", np.nan)):
        c_val = c_stat["c_statistic"]
        c_ci = f"({c_stat.get('ci_lower', 0):.3f}–{c_stat.get('ci_upper', 0):.3f})"
        html += f"""
                    <tr>
                        <td><strong>C-statistic (AUC)</strong><br><small class="text-muted">Discrimination</small></td>
                        <td><span class="badge bg-primary text-white fs-6">{c_val:.3f}</span> <small class="text-muted">{c_ci}</small></td>
                        <td>{c_stat.get("interpretation", "")}</td>
                    </tr>
        """

    # 2. Integrated Calibration Index (ICI)
    if "ici" in ici_dict and not np.isnan(ici_dict.get("ici", np.nan)):
        ici_val = ici_dict["ici"]
        ici_ci_str = ""
        if not np.isnan(ici_dict.get("ici_ci_lower", np.nan)):
            ici_ci_str = (
                f" ({ici_dict['ici_ci_lower']:.3f}–{ici_dict['ici_ci_upper']:.3f})"
            )
        html += f"""
                    <tr>
                        <td><strong>Integrated Calibration Index (ICI)</strong><br><small class="text-muted">Mean absolute error (TRIPOD standard)</small></td>
                        <td><strong>{ici_val:.4f}</strong><small class="text-muted">{ici_ci_str}</small></td>
                        <td>{ici_dict.get("interpretation", "")}</td>
                    </tr>
                    <tr>
                        <td><strong>Calibration Error Percentiles</strong><br><small class="text-muted">Median (E50) & 90th percentile (E90)</small></td>
                        <td>E50: <strong>{ici_dict.get("e50", 0):.4f}</strong> | E90: <strong>{ici_dict.get("e90", 0):.4f}</strong> | Emax: <strong>{ici_dict.get("emax", 0):.4f}</strong></td>
                        <td><small class="text-muted">E90 indicates error for 90% of patient predictions</small></td>
                    </tr>
        """

    # 3. Calibration Slope & Intercept
    if "calibration_slope" in calib and not np.isnan(
        calib.get("calibration_slope", np.nan)
    ):
        slope = calib["calibration_slope"]
        slope_ci = f"({calib.get('slope_ci_lower', 0):.2f}–{calib.get('slope_ci_upper', 0):.2f})"
        intercept = calib.get("calibration_intercept", 0.0)
        intercept_ci = f"({calib.get('intercept_ci_lower', 0):.2f}–{calib.get('intercept_ci_upper', 0):.2f})"
        html += f"""
                    <tr>
                        <td><strong>Calibration Slope</strong><br><small class="text-muted">Ideal = 1.0 (over/under-fitting check)</small></td>
                        <td><strong>{slope:.3f}</strong> <small class="text-muted">{slope_ci}</small></td>
                        <td>{calib.get("slope_interpretation", "")}</td>
                    </tr>
                    <tr>
                        <td><strong>Calibration Intercept</strong><br><small class="text-muted">Calibration-in-the-large (Ideal = 0.0)</small></td>
                        <td><strong>{intercept:.3f}</strong> <small class="text-muted">{intercept_ci}</small></td>
                        <td>{calib.get("intercept_interpretation", "")}</td>
                    </tr>
        """

    # 4. Brier Score
    if "brier_score" in brier and not np.isnan(brier.get("brier_score", np.nan)):
        brier_val = brier["brier_score"]
        scaled_str = (
            f" (Scaled: {brier.get('brier_scaled', 0):.3f})"
            if not np.isnan(brier.get("brier_scaled", np.nan))
            else ""
        )
        html += f"""
                    <tr>
                        <td><strong>Brier Score</strong><br><small class="text-muted">Overall accuracy (&lt;0.25 is acceptable)</small></td>
                        <td><strong>{brier_val:.4f}</strong><small class="text-muted">{scaled_str}</small></td>
                        <td>{brier.get("interpretation", "")}</td>
                    </tr>
        """

    # 5. Hosmer-Lemeshow (Legacy / Supplementary)
    if "p_value" in hl and not np.isnan(hl.get("p_value", np.nan)):
        html += f"""
                    <tr>
                        <td><strong>Hosmer-Lemeshow Test</strong><br><small class="text-muted">Binned goodness-of-fit (Legacy metric)</small></td>
                        <td>χ² = {hl["chi2"]:.2f}, df = {hl["df"]}, p = {hl["p_value"]:.3f}</td>
                        <td>{hl.get("interpretation", "")}</td>
                    </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return html
