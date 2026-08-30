from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import prepare_data_for_analysis

logger = get_logger(__name__)


def calculate_net_benefit(
    df: pd.DataFrame,
    truth_col: str,
    prob_col: str,
    thresholds: list[float] | np.ndarray | None = None,
    model_name: str = "Model",
    var_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Calculate Net Benefit for a prediction model across a range of thresholds.
    """
    try:
        # Validate inputs
        if df is None or df.empty:
            return pd.DataFrame(), {}

        # --- DATA PREPARATION ---
        required_cols = [truth_col, prob_col]
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        try:
            df_clean, missing_info = prepare_data_for_analysis(
                df,
                required_cols=required_cols,
                numeric_cols=required_cols,
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy
        except Exception as e:
            return pd.DataFrame(), {"error": f"Data preparation failed: {e}"}

        if df_clean.empty:
            return pd.DataFrame(), missing_info

        if thresholds is None:
            thresholds = np.arange(0.01, 1.00, 0.01)

        y_true = df_clean[truth_col].values
        y_prob = df_clean[prob_col].values
        n = len(y_true)

        results = []

        for pt in thresholds:
            y_pred = (y_prob >= pt).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))

            weight = pt / (1 - pt) if pt < 1.0 else 0
            net_benefit = (tp / n) - (fp / n) * weight

            results.append(
                {
                    "threshold": pt,
                    "net_benefit": net_benefit,
                    "model": model_name,
                    "tp_rate": tp / n,
                    "fp_rate": fp / n,
                }
            )

        return pd.DataFrame(results), missing_info
    except Exception as e:
        logger.error("DCA calculation failed: %s", e)
        return pd.DataFrame(), {"error": str(e)}


def calculate_net_benefit_all(
    df: pd.DataFrame,
    truth_col: str,
    thresholds: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate Net Benefit assuming ALL patients are treated (Prevalence strategy).
    Note: df here should ALREADY be cleaned by calculate_net_benefit or caller.
    """
    try:
        if df is None or df.empty or truth_col not in df.columns:
            return pd.DataFrame()

        if thresholds is None:
            thresholds = np.arange(0.01, 1.00, 0.01)

        y_true = df[truth_col].values
        n = len(y_true)

        tp = np.sum(y_true == 1)
        fp = np.sum(y_true == 0)

        results = []
        for pt in thresholds:
            weight = pt / (1 - pt) if pt < 1.0 else 0
            net_benefit = (tp / n) - (fp / n) * weight

            results.append(
                {
                    "threshold": pt,
                    "net_benefit": net_benefit,
                    "model": "Treat All",
                }
            )

        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()


def calculate_net_benefit_none(
    thresholds: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate Net Benefit assuming NO patients are treated.
    Always zero.
    """
    try:
        if thresholds is None:
            thresholds = np.arange(0.01, 1.00, 0.01)

        results = [
            {"threshold": pt, "net_benefit": 0.0, "model": "Treat None"}
            for pt in thresholds
        ]
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()


def create_dca_plot(dca_df: pd.DataFrame) -> go.Figure:
    """
    Create a Decision Curve Analysis plot using Plotly.
    Expects DataFrame with columns: 'threshold', 'net_benefit', 'model'.
    """
    fig = go.Figure()

    models = dca_df["model"].unique()

    for model in models:
        subset = dca_df[dca_df["model"] == model]

        # Determine style
        line_dict = dict(width=2)
        if model == "Treat All":
            line_dict = dict(color="gray", width=2, dash="dash")
        elif model == "Treat None":
            line_dict = dict(color="black", width=2)
        else:
            # Random or specific color for actual models
            line_dict = dict(width=3)

        fig.add_trace(
            go.Scatter(
                x=subset["threshold"],
                y=subset["net_benefit"],
                mode="lines",
                name=model,
                line=line_dict,
            )
        )

    # Layout improvements
    fig.update_layout(
        title="Decision Curve Analysis",
        xaxis_title="Threshold Probability",
        yaxis_title="Net Benefit",
        yaxis=dict(
            range=[-0.05, dca_df["net_benefit"].max() * 1.1]
        ),  # Zoom in on relevant area, avoid huge negative
        xaxis=dict(range=[0, 1]),
        template="plotly_white",
        legend=dict(x=0.8, y=0.95),
        hovermode="x unified",
    )

    return fig
