"""
🧭 Fagan's Nomogram & Likelihood Ratio Module for Bedside Diagnostic Translation

Provides:
- Bayes' Theorem translation from Pre-test to Post-test probability on the log-odds scale
- Interactive Plotly 3-Axis Nomogram (Pre-test % -> Likelihood Ratio -> Post-test %)
- Multi-level / Interval Likelihood Ratios calculation for continuous biomarker strata

References:
    Fagan TJ. Nomogram for Bayes's theorem. N Engl J Med. 1975;293(5):257.
    Simel DL, Rennie D. The Rational Clinical Examination. JAMA Evidence, 2008.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


def calculate_post_test_probability(pre_test_prob: float, lr: float) -> dict[str, Any]:
    """
    Calculate post-test odds and probability given pre-test probability and likelihood ratio.

    Args:
        pre_test_prob: Probability between 0.0001 and 0.9999 (e.g. 0.25 for 25%)
        lr: Likelihood Ratio (LR+ or LR- or Interval LR)

    Returns:
        Dictionary with pre_test_prob, lr, post_test_prob, post_test_odds, interpretation
    """
    try:
        p = float(np.clip(pre_test_prob, 1e-4, 1.0 - 1e-4))
        likelihood_ratio = float(max(1e-4, lr))

        pre_odds = p / (1.0 - p)
        post_odds = pre_odds * likelihood_ratio
        post_prob = post_odds / (1.0 + post_odds)

        # Risk zone interpretation
        if post_prob < 0.02:
            zone = "🟢 Rule-Out (<2%)"
            zone_color = "#10B981"
        elif post_prob < 0.15:
            zone = "🔵 Low Risk (2-15%)"
            zone_color = "#3B82F6"
        elif post_prob < 0.50:
            zone = "🟡 Intermediate Risk (15-50%)"
            zone_color = "#F59E0B"
        elif post_prob < 0.85:
            zone = "🟠 High Risk (50-85%)"
            zone_color = "#F97316"
        else:
            zone = "🔴 Rule-In / Diagnostic Confirmed (>85%)"
            zone_color = "#EF4444"

        # Diagnostic impact rating
        if likelihood_ratio >= 10.0:
            impact = "Strong evidence to rule in disease"
        elif likelihood_ratio >= 5.0:
            impact = "Moderate evidence to rule in disease"
        elif likelihood_ratio >= 2.0:
            impact = "Weak/minimal shift toward disease"
        elif likelihood_ratio > 0.5:
            impact = "Inconclusive / No significant shift"
        elif likelihood_ratio > 0.2:
            impact = "Weak shift toward ruling out disease"
        elif likelihood_ratio > 0.1:
            impact = "Moderate evidence to rule out disease"
        else:
            impact = "Strong evidence to rule out disease"

        return {
            "pre_test_prob": p,
            "pre_test_prob_pct": p * 100.0,
            "lr": likelihood_ratio,
            "post_test_prob": post_prob,
            "post_test_prob_pct": post_prob * 100.0,
            "pre_test_odds": pre_odds,
            "post_test_odds": post_odds,
            "zone": zone,
            "zone_color": zone_color,
            "impact": impact,
        }
    except Exception as e:
        logger.error(f"Error calculating post-test probability: {e}")
        return {
            "pre_test_prob": pre_test_prob,
            "lr": lr,
            "post_test_prob": np.nan,
            "error": str(e),
        }


def calculate_multilevel_likelihood_ratios(
    df: pd.DataFrame,
    outcome_col: str,
    score_col: str,
    cutoffs: list[float],
    pos_label: Any = 1,
) -> pd.DataFrame:
    """
    Calculate Interval / Multi-level Likelihood Ratios for continuous biomarker strata.

    Args:
        df: Input DataFrame
        outcome_col: Disease outcome column (0/1 or True/False)
        score_col: Biomarker score / continuous variable
        cutoffs: List of cutoffs (e.g. [14, 50] creates <14, 14-50, >=50)
        pos_label: Value representing disease positive

    Returns:
        DataFrame with Tiers, Diseased counts, Non-diseased counts, Stratum sensitivity/specificity, Interval LR, 95% CI
    """
    clean_df = df[[outcome_col, score_col]].dropna().copy()
    clean_df["_disease"] = (
        (clean_df[outcome_col] == pos_label)
        | (clean_df[outcome_col].astype(str) == str(pos_label))
    ).astype(int)
    scores = clean_df[score_col].astype(float)

    total_d_pos = int(clean_df["_disease"].sum())
    total_d_neg = int(len(clean_df) - total_d_pos)

    if total_d_pos == 0 or total_d_neg == 0:
        raise ValueError(
            "Both diseased and non-diseased cases are required for interval LR calculation."
        )

    sorted_cutoffs = sorted(list(set(cutoffs)))
    bins = [-np.inf] + sorted_cutoffs + [np.inf]

    tier_labels = []
    for i in range(len(bins) - 1):
        if i == 0:
            tier_labels.append(f"< {sorted_cutoffs[0]}")
        elif i == len(bins) - 2:
            tier_labels.append(f"≥ {sorted_cutoffs[-1]}")
        else:
            tier_labels.append(f"[{sorted_cutoffs[i - 1]}, {sorted_cutoffs[i]})")

    clean_df["_tier"] = pd.cut(scores, bins=bins, labels=tier_labels, right=False)

    records = []
    for tier in tier_labels:
        sub = clean_df[clean_df["_tier"] == tier]
        d_pos_count = int(sub["_disease"].sum())
        d_neg_count = int(len(sub) - d_pos_count)

        prob_d_pos = (d_pos_count + 0.5) / (total_d_pos + 1.0)
        prob_d_neg = (d_neg_count + 0.5) / (total_d_neg + 1.0)

        lr_interval = prob_d_pos / prob_d_neg
        se_log_lr = math.sqrt(
            (1.0 / (d_pos_count + 0.5))
            - (1.0 / (total_d_pos + 1.0))
            + (1.0 / (d_neg_count + 0.5))
            - (1.0 / (total_d_neg + 1.0))
        )
        ci_lower = math.exp(math.log(lr_interval) - 1.96 * se_log_lr)
        ci_upper = math.exp(math.log(lr_interval) + 1.96 * se_log_lr)

        records.append(
            {
                "Tier / Interval": tier,
                "Diseased (D+)": d_pos_count,
                "Non-Diseased (D-)": d_neg_count,
                "Total": len(sub),
                "P(Score | D+) (%)": round((d_pos_count / total_d_pos) * 100, 2),
                "P(Score | D-) (%)": round((d_neg_count / total_d_neg) * 100, 2),
                "Interval LR": round(lr_interval, 3),
                "95% CI Lower": round(ci_lower, 3),
                "95% CI Upper": round(ci_upper, 3),
                "Post-Test Prob @ 20% Pre-test (%)": round(
                    calculate_post_test_probability(0.20, lr_interval)[
                        "post_test_prob_pct"
                    ],
                    1,
                ),
            }
        )

    return pd.DataFrame(records)


def create_fagan_nomogram_plot(
    pre_test_prob: float,
    lr_pos: float = 10.0,
    lr_neg: float = 0.1,
    test_name: str = "Diagnostic Test",
    multilevel_lrs: list[dict] | None = None,
) -> go.Figure:
    """
    Generate an interactive 3-axis Plotly Fagan's Nomogram.

    Axes:
    - Left (x=0): Pre-test probability (0.1% to 99.9% on logit scale)
    - Middle (x=0.5): Likelihood Ratio (0.001 to 1000 on 0.5*ln(LR) scale)
    - Right (x=1.0): Post-test probability (0.1% to 99.9% on logit scale)

    Args:
        pre_test_prob: Pre-test probability (0.001 to 0.999)
        lr_pos: Positive Likelihood Ratio (LR+)
        lr_neg: Negative Likelihood Ratio (LR-)
        test_name: Name of diagnostic test or biomarker
        multilevel_lrs: Optional list of dicts for multi-tier interval LRs

    Returns:
        Plotly Figure object
    """

    # Helper to convert probability to y-coordinate (logit scale)
    def prob_to_y_pre(p: float) -> float:
        """Pre-test probability axis is inverted (0.1% at top, 99.9% at bottom)."""
        p_clamped = float(np.clip(p, 0.001, 0.999))
        return -math.log(p_clamped / (1.0 - p_clamped))

    def prob_to_y_post(p: float) -> float:
        """Post-test probability axis is standard (0.1% at bottom, 99.9% at top)."""
        p_clamped = float(np.clip(p, 0.001, 0.999))
        return math.log(p_clamped / (1.0 - p_clamped))

    # Center axis y-coordinate for a given LR (geometric midpoint)
    def lr_to_y(lr_val: float) -> float:
        """Likelihood Ratio axis (1000 at top, 1 at center, 0.001 at bottom)."""
        lr_clamped = float(max(1e-4, lr_val))
        return 0.5 * math.log(lr_clamped)

    fig = go.Figure()

    # Predefined tick marks and values for Probabilities (%)
    prob_ticks_pct = [
        0.1,
        0.2,
        0.5,
        1,
        2,
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        95,
        98,
        99,
        99.9,
    ]
    prob_ticks_val = [p / 100.0 for p in prob_ticks_pct]
    prob_ticks_y_pre = [prob_to_y_pre(p) for p in prob_ticks_val]
    prob_ticks_y_post = [prob_to_y_post(p) for p in prob_ticks_val]

    # Predefined tick marks and values for Likelihood Ratios
    lr_ticks_val = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
    ]
    lr_ticks_labels = [
        "0.001",
        "0.002",
        "0.005",
        "0.01",
        "0.02",
        "0.05",
        "0.1",
        "0.2",
        "0.5",
        "1",
        "2",
        "5",
        "10",
        "20",
        "50",
        "100",
        "200",
        "500",
        "1000",
    ]
    lr_ticks_y = [lr_to_y(val) for val in lr_ticks_val]

    y_max = prob_to_y_post(0.999)
    y_min = -y_max

    # 1. Draw Axis Lines
    # Left Axis (Pre-test)
    fig.add_shape(
        type="line",
        x0=0,
        y0=y_min,
        x1=0,
        y1=y_max,
        line=dict(color="#1F2937", width=2.5),
    )
    # Middle Axis (Likelihood Ratio)
    fig.add_shape(
        type="line",
        x0=0.5,
        y0=y_min,
        x1=0.5,
        y1=y_max,
        line=dict(color="#1F2937", width=2.5),
    )
    # Right Axis (Post-test)
    fig.add_shape(
        type="line",
        x0=1.0,
        y0=y_min,
        x1=1.0,
        y1=y_max,
        line=dict(color="#1F2937", width=2.5),
    )

    # 2. Add Tick Marks & Labels
    # Left Axis Ticks (Pre-test)
    for p_label, y_pos in zip(prob_ticks_pct, prob_ticks_y_pre):
        fig.add_shape(
            type="line",
            x0=-0.015,
            y0=y_pos,
            x1=0,
            y1=y_pos,
            line=dict(color="#4B5563", width=1),
        )
        fig.add_annotation(
            x=-0.02,
            y=y_pos,
            text=f"<b>{p_label}%</b>",
            showarrow=False,
            xanchor="right",
            font=dict(size=9, color="#1F2937"),
        )

    # Middle Axis Ticks (Likelihood Ratio)
    for lr_lbl, y_pos in zip(lr_ticks_labels, lr_ticks_y):
        fig.add_shape(
            type="line",
            x0=0.485,
            y0=y_pos,
            x1=0.515,
            y1=y_pos,
            line=dict(color="#4B5563", width=1),
        )
        fig.add_annotation(
            x=0.525,
            y=y_pos,
            text=f"<b>{lr_lbl}</b>",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color="#1F2937"),
        )

    # Right Axis Ticks (Post-test)
    for p_label, y_pos in zip(prob_ticks_pct, prob_ticks_y_post):
        fig.add_shape(
            type="line",
            x0=1.0,
            y0=y_pos,
            x1=1.015,
            y1=y_pos,
            line=dict(color="#4B5563", width=1),
        )
        fig.add_annotation(
            x=1.02,
            y=y_pos,
            text=f"<b>{p_label}%</b>",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color="#1F2937"),
        )

    # 3. Axis Headers
    fig.add_annotation(
        x=0,
        y=y_max + 0.6,
        text="<b>Pre-test Probability</b><br>(Prior)",
        showarrow=False,
        font=dict(size=12, color="#1E3A5F"),
        xanchor="center",
    )
    fig.add_annotation(
        x=0.5,
        y=y_max + 0.6,
        text="<b>Likelihood Ratio</b><br>(LR)",
        showarrow=False,
        font=dict(size=12, color="#1E3A5F"),
        xanchor="center",
    )
    fig.add_annotation(
        x=1.0,
        y=y_max + 0.6,
        text="<b>Post-test Probability</b><br>(Posterior)",
        showarrow=False,
        font=dict(size=12, color="#1E3A5F"),
        xanchor="center",
    )

    # 4. Draw Diagnostic Trajectories
    y_pre = prob_to_y_pre(pre_test_prob)

    # Standard Binary Test (LR+ and LR-)
    res_pos = calculate_post_test_probability(pre_test_prob, lr_pos)
    res_neg = calculate_post_test_probability(pre_test_prob, lr_neg)

    y_post_pos = prob_to_y_post(res_pos["post_test_prob"])
    y_post_neg = prob_to_y_post(res_neg["post_test_prob"])

    # Line for LR+ (Positive Result - Red/Orange)
    fig.add_trace(
        go.Scatter(
            x=[0, 0.5, 1.0],
            y=[y_pre, (y_pre + y_post_pos) / 2.0, y_post_pos],
            mode="lines+markers",
            line=dict(color="#DC2626", width=3.5),
            marker=dict(
                size=10, color="#DC2626", symbol=["circle", "diamond", "square"]
            ),
            name=f"Test Positive (LR+: {lr_pos:.2f} → {res_pos['post_test_prob_pct']:.1f}%)",
            hovertemplate="<b>Positive Test</b><br>Pre-test: "
            + f"{pre_test_prob * 100:.1f}%"
            + "<br>LR+: "
            + f"{lr_pos:.2f}"
            + "<br>Post-test: "
            + f"{res_pos['post_test_prob_pct']:.1f}%<extra></extra>",
        )
    )

    # Line for LR- (Negative Result - Green/Blue)
    fig.add_trace(
        go.Scatter(
            x=[0, 0.5, 1.0],
            y=[y_pre, (y_pre + y_post_neg) / 2.0, y_post_neg],
            mode="lines+markers",
            line=dict(color="#059669", width=3.5, dash="dash"),
            marker=dict(
                size=10, color="#059669", symbol=["circle", "diamond", "square"]
            ),
            name=f"Test Negative (LR-: {lr_neg:.2f} → {res_neg['post_test_prob_pct']:.1f}%)",
            hovertemplate="<b>Negative Test</b><br>Pre-test: "
            + f"{pre_test_prob * 100:.1f}%"
            + "<br>LR-: "
            + f"{lr_neg:.2f}"
            + "<br>Post-test: "
            + f"{res_neg['post_test_prob_pct']:.1f}%<extra></extra>",
        )
    )

    # If Multi-level / Tiered LRs are provided, plot additional trajectories
    if multilevel_lrs:
        tier_palette = ["#8B5CF6", "#EC4899", "#3B82F6", "#F59E0B", "#10B981"]
        for idx, tier in enumerate(multilevel_lrs):
            tier_name = tier.get("name", f"Tier {idx + 1}")
            tier_lr = tier.get("lr", 1.0)
            tier_res = calculate_post_test_probability(pre_test_prob, tier_lr)
            y_post_tier = prob_to_y_post(tier_res["post_test_prob"])
            t_color = tier_palette[idx % len(tier_palette)]

            fig.add_trace(
                go.Scatter(
                    x=[0, 0.5, 1.0],
                    y=[y_pre, (y_pre + y_post_tier) / 2.0, y_post_tier],
                    mode="lines+markers",
                    line=dict(color=t_color, width=2, dash="dot"),
                    marker=dict(size=8, color=t_color),
                    name=f"{tier_name} (LR: {tier_lr:.2f} → {tier_res['post_test_prob_pct']:.1f}%)",
                    hovertemplate=f"<b>{tier_name}</b><br>LR: {tier_lr:.2f}<br>Post-test: {tier_res['post_test_prob_pct']:.1f}%<extra></extra>",
                )
            )

    # Layout styling
    fig.update_layout(
        title=dict(
            text=f"<b>🧭 Interactive Fagan's Nomogram: {test_name}</b><br><span style='font-size: 11px; color: #6B7280;'>Pre-test: {pre_test_prob * 100:.1f}% | Post-test (+): {res_pos['post_test_prob_pct']:.1f}% | Post-test (-): {res_neg['post_test_prob_pct']:.1f}%</span>",
            x=0.5,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.12, 1.12],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[y_min - 0.5, y_max + 0.9],
        ),
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E5E7EB",
            borderwidth=1,
        ),
        height=560,
        margin=dict(l=60, r=60, t=70, b=80),
    )

    return fig
