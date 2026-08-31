"""
agent/clinical_analyst.py - Biostatistical Reasoning & Execution Harness
=============================================================================
Translates clinical objectives and research proposals into deterministic
statistical workflows. Uses utils/ as an execution harness to eliminate
LLM mathematical hallucination and guarantee SAMPL/EQUATOR compliance.
Includes PubMed Evidence Exploration & 5-Direction Research Synthesis.
=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

from agent.agent_runner import ClinicalAgentRunner
from agent.tools.tool_pubmed import PubMedEvidenceTool
from agent.tools.tool_sample_size import SampleSizeTool
from agent.tools.tool_synthetic_data import SyntheticDataTool
from agent.topic_ideator import ClinicalTopicIdeator
from core.common import select_variable_by_keyword
from core.state import AppState
from logger import get_logger
from utils import linear_lib, logic, psm_lib, survival_lib
from utils.data_cleaning import load_data_robust
from utils.data_quality import check_data_quality
from utils.proposal_parser import ProposalMetadata, ProposalParser
from utils.table_one_advanced import TableOneGenerator
from utils.visualizations import plot_missing_pattern

logger = get_logger(__name__)


def _coerce_to_binary_series(series: pd.Series) -> pd.Series:
    """Coerces boolean, numeric, or string clinical labels to binary 0/1 integers."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1}):
            return series.astype(int)
        return (series > 0).astype(int)
    # String / categorical mapping
    s_str = series.astype(str).str.strip().str.lower()
    pos_tokens = {
        "1",
        "true",
        "yes",
        "positive",
        "pos",
        "case",
        "present",
        "abnormal",
        "intervention",
        "diseased",
        "reactive",
    }
    return s_str.apply(
        lambda v: 1
        if (
            v in pos_tokens
            or any(tok in v for tok in ["pos", "true", "abnormal", "yes"])
        )
        else 0
    )


@dataclass
class AnalystResult:
    """Structured response from the Clinical Analyst Engine."""

    message_text: str
    figure: go.Figure | None = None
    preview_df: pd.DataFrame | None = None
    action_type: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


class StatHarness:
    """
    Deterministic statistical calculation harness.
    Wraps tested biostatistical libraries (lifelines, statsmodels, scipy, pingouin).
    """

    @staticmethod
    def run_table_one(
        df: pd.DataFrame,
        group_col: str | None = None,
        selected_vars: list[str] | None = None,
        var_meta: dict[str, Any] | None = None,
    ) -> tuple[str, pd.DataFrame]:
        """Generates Baseline Table 1 with Standardized Mean Differences (SMD)."""
        cols = df.columns.tolist()
        if not selected_vars:
            selected_vars = [c for c in cols if c != group_col][:10]

        generator = TableOneGenerator(df, var_meta or {})
        html_table = generator.generate(
            selected_vars=selected_vars,
            stratify_by=group_col,
        )
        return html_table, df[selected_vars]

    @staticmethod
    def run_survival(
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str | None = None,
        covar_cols: list[str] | None = None,
    ) -> tuple[go.Figure, pd.DataFrame, dict[str, Any]]:
        """Fits Kaplan-Meier Log-rank & Multivariable Cox Proportional Hazards."""
        km_fig, km_summary, missing_info = survival_lib.fit_km_logrank(
            df=df,
            duration_col=time_col,
            event_col=event_col,
            group_col=group_col,
        )

        p_val = "N/A"
        if (
            isinstance(km_summary, pd.DataFrame)
            and not km_summary.empty
            and "P-value" in km_summary.columns
        ):
            p_val = km_summary.iloc[0]["P-value"]

        cox_df = None
        cox_stats = {}
        if covar_cols and len(covar_cols) > 0:
            cph, res_df, _, err, c_stats, _ = survival_lib.fit_cox_ph(
                df=df,
                duration_col=time_col,
                event_col=event_col,
                covariate_cols=covar_cols,
            )
            if not err and res_df is not None:
                cox_df = res_df
                cox_stats = c_stats or {}

        return (
            km_fig,
            km_summary,
            {
                "km_stats": {"p_value": p_val},
                "cox_df": cox_df,
                "cox_stats": cox_stats,
                "missing_info": missing_info,
            },
        )

    @staticmethod
    def run_logistic(
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure]:
        """Fits binary logistic regression with Odds Ratios and 95% CIs."""
        clean_cols = [outcome_col] + [c for c in predictor_cols if c in df.columns]
        df_clean = df[clean_cols].dropna()

        html_table, or_results, status, metrics = logic.run_logistic_regression(
            df=df_clean, outcome_col=outcome_col, covariate_cols=predictor_cols
        )
        rows = []
        if or_results:
            for var_name, r in or_results.items():
                or_val = (
                    r.get("or", 1.0)
                    if isinstance(r, dict)
                    else getattr(r, "or_val", 1.0)
                )
                ci_l = (
                    r.get("ci_low", r.get("ci_lower", 1.0))
                    if isinstance(r, dict)
                    else getattr(r, "ci_low", getattr(r, "ci_lower", 1.0))
                )
                ci_u = (
                    r.get("ci_high", r.get("ci_upper", 1.0))
                    if isinstance(r, dict)
                    else getattr(r, "ci_high", getattr(r, "ci_upper", 1.0))
                )
                p_v = (
                    r.get("p_value", 1.0)
                    if isinstance(r, dict)
                    else getattr(r, "p_value", 1.0)
                )
                rows.append(
                    {
                        "Variable": var_name,
                        "Odds Ratio (OR)": f"{float(or_val):.3f}"
                        if isinstance(or_val, (int, float))
                        else str(or_val),
                        "95% CI Lower": f"{float(ci_l):.3f}"
                        if isinstance(ci_l, (int, float))
                        else str(ci_l),
                        "95% CI Upper": f"{float(ci_u):.3f}"
                        if isinstance(ci_u, (int, float))
                        else str(ci_u),
                        "P-value": f"{float(p_v):.4f}"
                        if isinstance(p_v, (int, float))
                        else str(p_v),
                    }
                )
        coef_df = pd.DataFrame(rows)

        fig = go.Figure()
        if not coef_df.empty and "Odds Ratio (OR)" in coef_df.columns:
            vars_list = coef_df["Variable"].tolist()
            or_vals = [float(x) for x in coef_df["Odds Ratio (OR)"]]
            ci_lows = [float(x) for x in coef_df["95% CI Lower"]]
            ci_highs = [float(x) for x in coef_df["95% CI Upper"]]

            fig.add_trace(
                go.Scatter(
                    x=or_vals,
                    y=vars_list,
                    mode="markers",
                    marker=dict(color="#0284c7", size=10),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[h - o for h, o in zip(ci_highs, or_vals)],
                        arrayminus=[
                            o - low_val for o, low_val in zip(or_vals, ci_lows)
                        ],
                        color="#0284c7",
                    ),
                    name="Odds Ratio (95% CI)",
                )
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444")
            fig.update_layout(
                title="Multivariable Logistic Regression (Forest Plot of Odds Ratios)",
                xaxis_title="Odds Ratio (Log Scale)",
                yaxis_title="Covariates",
                xaxis_type="log",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )

        return coef_df, metrics or {}, fig

    @staticmethod
    def run_diagnostic(
        df: pd.DataFrame | None = None,
        index_test_col: str | None = None,
        ref_standard_col: str | None = None,
        pre_test_prob: float | None = None,
        tp: int | None = None,
        fp: int | None = None,
        fn: int | None = None,
        tn: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure]:
        """Calculates Diagnostic Test Accuracy (STARD 2015) & Bayesian Fagan Nomogram."""
        calc_tp, calc_fp, calc_fn, calc_tn = None, None, None, None

        if df is not None and not df.empty:
            cols = df.columns.tolist()
            idx_col = (
                index_test_col
                or select_variable_by_keyword(
                    cols,
                    [
                        "pocus",
                        "index",
                        "test",
                        "biomarker",
                        "screen",
                        "predict",
                        "treatment",
                        "arm",
                    ],
                )
                or (cols[0] if len(cols) > 0 else None)
            )

            ref_col = (
                ref_standard_col
                or select_variable_by_keyword(
                    [c for c in cols if c != idx_col],
                    [
                        "gold",
                        "reference",
                        "ref",
                        "disease",
                        "diagnosis",
                        "status",
                        "event",
                        "death",
                        "outcome",
                        "target",
                    ],
                )
                or (cols[1] if len(cols) > 1 else None)
            )

            if idx_col in df.columns and ref_col in df.columns:
                sub = df[[idx_col, ref_col]].dropna()
                if not sub.empty:
                    y_test = _coerce_to_binary_series(sub[idx_col])
                    y_ref = _coerce_to_binary_series(sub[ref_col])
                    calc_tp = int(((y_test == 1) & (y_ref == 1)).sum())
                    calc_fp = int(((y_test == 1) & (y_ref == 0)).sum())
                    calc_fn = int(((y_test == 0) & (y_ref == 1)).sum())
                    calc_tn = int(((y_test == 0) & (y_ref == 0)).sum())
                    index_test_col = idx_col
                    ref_standard_col = ref_col

        # Strict validation: require complete explicit 2x2 matrix or valid DataFrame inputs
        has_explicit = any(c is not None for c in (tp, fp, fn, tn))
        if has_explicit:
            if not all(c is not None for c in (tp, fp, fn, tn)):
                raise ValueError(
                    "Incomplete 2x2 contingency matrix: tp, fp, fn, and tn must all be specified when providing explicit cell counts."
                )
            for name, val in [("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)]:
                if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                    raise ValueError(
                        f"2x2 contingency matrix cell count '{name}' must be an integer, got {val!r}."
                    )
                if val < 0:
                    raise ValueError(
                        f"2x2 contingency matrix cell count '{name}' cannot be negative, got {val}."
                    )
            final_tp, final_fp, final_fn, final_tn = int(tp), int(fp), int(fn), int(tn)
            used_example_counts = False
        elif calc_tp is not None:
            final_tp, final_fp, final_fn, final_tn = calc_tp, calc_fp, calc_fn, calc_tn
            used_example_counts = False
        else:
            raise ValueError(
                "No valid data provided for diagnostic accuracy calculation. "
                "Please provide a DataFrame with valid index test and reference standard columns, "
                "or specify all 4 contingency matrix cell counts (tp, fp, fn, tn)."
            )

        total = final_tp + final_fp + final_fn + final_tn
        if total == 0:
            raise ValueError(
                "Total sample size across 2x2 contingency matrix cannot be zero."
            )
        if pre_test_prob is None:
            if (final_tp + final_fn) > 0 and total > 0:
                pre_test_prob = ((final_tp + final_fn) / total) * 100.0
            else:
                pre_test_prob = 25.0

        sens = final_tp / (final_tp + final_fn) if (final_tp + final_fn) > 0 else 0.0
        spec = final_tn / (final_tn + final_fp) if (final_tn + final_fp) > 0 else 0.0
        ppv = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0.0
        npv = final_tn / (final_tn + final_fn) if (final_tn + final_fn) > 0 else 0.0

        # Haldane-Anscombe continuity correction avoids reporting an uninformative LR of 1.0
        # when a cell count is zero or when sensitivity/specificity reach boundary values (1.0 or 0.0).
        if (
            (1.0 - spec) > 0
            and spec > 0
            and (final_tp + final_fn) > 0
            and (final_tn + final_fp) > 0
        ):
            lr_pos = sens / (1.0 - spec)
            lr_neg = (1.0 - sens) / spec
        else:
            c_tp, c_fp = final_tp + 0.5, final_fp + 0.5
            c_fn, c_tn = final_fn + 0.5, final_tn + 0.5
            c_sens = c_tp / (c_tp + c_fn)
            c_spec = c_tn / (c_tn + c_fp)
            lr_pos = c_sens / (1.0 - c_spec)
            lr_neg = (1.0 - c_sens) / c_spec

        dor = (
            ((final_tp * final_tn) / (final_fp * final_fn))
            if (final_fp * final_fn) > 0
            else (
                ((final_tp + 0.5) * (final_tn + 0.5))
                / ((final_fp + 0.5) * (final_fn + 0.5))
            )
        )

        p_pre = pre_test_prob / 100.0
        odds_pre = p_pre / (1.0 - p_pre) if p_pre < 1.0 else 999.0
        odds_post_pos = odds_pre * lr_pos
        p_post_pos = (odds_post_pos / (1.0 + odds_post_pos)) * 100.0

        odds_post_neg = odds_pre * lr_neg
        p_post_neg = (odds_post_neg / (1.0 + odds_post_neg)) * 100.0

        metrics_df = pd.DataFrame(
            [
                {"Metric": "Sensitivity (True Positive Rate)", "Value": f"{sens:.1%}"},
                {"Metric": "Specificity (True Negative Rate)", "Value": f"{spec:.1%}"},
                {"Metric": "Positive Predictive Value (PPV)", "Value": f"{ppv:.1%}"},
                {"Metric": "Negative Predictive Value (NPV)", "Value": f"{npv:.1%}"},
                {"Metric": "Positive Likelihood Ratio (LR+)", "Value": f"{lr_pos:.2f}"},
                {"Metric": "Negative Likelihood Ratio (LR-)", "Value": f"{lr_neg:.2f}"},
                {"Metric": "Diagnostic Odds Ratio (DOR)", "Value": f"{dor:.2f}"},
            ]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2],
                y=[pre_test_prob, lr_pos, p_post_pos],
                mode="lines+markers+text",
                text=[
                    f"Pre: {pre_test_prob:.0f}%",
                    f"LR+: {lr_pos:.1f}",
                    f"Post: {p_post_pos:.1f}%",
                ],
                textposition="top center",
                line=dict(color="#059669", width=3),
                name="Positive Result (+LR)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2],
                y=[pre_test_prob, lr_neg, p_post_neg],
                mode="lines+markers+text",
                text=[
                    f"Pre: {pre_test_prob:.0f}%",
                    f"LR-: {lr_neg:.2f}",
                    f"Post: {p_post_neg:.1f}%",
                ],
                textposition="bottom center",
                line=dict(color="#dc2626", width=3, dash="dash"),
                name="Negative Result (-LR)",
            )
        )
        fig.update_layout(
            title="Bayesian Updating Trajectory (Pre-test to Post-test Probability)",
            xaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=[
                    "Pre-Test Prob (%)",
                    "Likelihood Ratio",
                    "Post-Test Prob (%)",
                ],
            ),
            yaxis_title="Probability (%) / Ratio",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        metrics = {
            "tp": final_tp,
            "fp": final_fp,
            "fn": final_fn,
            "tn": final_tn,
            "used_example_counts": used_example_counts,
            "pre_test_prob": pre_test_prob,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "lr_pos": lr_pos,
            "lr_neg": lr_neg,
            "dor": dor,
            "post_prob_pos": p_post_pos,
            "post_prob_neg": p_post_neg,
            "index_test_col": index_test_col,
            "ref_standard_col": ref_standard_col,
        }
        return metrics_df, metrics, fig

    @staticmethod
    def run_binary_rct(
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure]:
        """
        Evaluates a 2-arm Randomized Controlled Trial (CONSORT compliant) for binary primary outcomes.
        Calculates Chi-Square / Fisher's exact tests, Relative Risk (RR), Risk Difference (RD),
        Relative Risk Reduction (RRR), and Number Needed to Treat (NNT) with 95% Confidence Intervals.
        """
        clean_df = df[[treatment_col, outcome_col]].dropna()
        t_bin = _coerce_to_binary_series(clean_df[treatment_col])
        y_bin = _coerce_to_binary_series(clean_df[outcome_col])

        # Control (0) vs Intervention (1)
        n_ctrl = int((t_bin == 0).sum())
        events_ctrl = int(((t_bin == 0) & (y_bin == 1)).sum())
        p_ctrl = events_ctrl / n_ctrl if n_ctrl > 0 else 0.0

        n_treat = int((t_bin == 1).sum())
        events_treat = int(((t_bin == 1) & (y_bin == 1)).sum())
        p_treat = events_treat / n_treat if n_treat > 0 else 0.0

        # Risk Difference (RD = p_treat - p_ctrl)
        rd = p_treat - p_ctrl
        se_rd = (
            np.sqrt(
                (p_treat * (1 - p_treat) / n_treat) + (p_ctrl * (1 - p_ctrl) / n_ctrl)
            )
            if (n_treat > 0 and n_ctrl > 0)
            else 0.0
        )
        rd_ci_low = rd - 1.96 * se_rd
        rd_ci_high = rd + 1.96 * se_rd

        # Relative Risk (RR = p_treat / p_ctrl)
        if p_ctrl > 0 and p_treat > 0 and events_treat > 0 and events_ctrl > 0:
            rr = p_treat / p_ctrl
            se_ln_rr = np.sqrt(
                ((n_treat - events_treat) / (n_treat * events_treat))
                + ((n_ctrl - events_ctrl) / (n_ctrl * events_ctrl))
            )
            rr_ci_low = float(np.exp(np.log(rr) - 1.96 * se_ln_rr))
            rr_ci_high = float(np.exp(np.log(rr) + 1.96 * se_ln_rr))
        else:
            rr = (p_treat / p_ctrl) if p_ctrl > 0 else 1.0
            rr_ci_low, rr_ci_high = rr, rr

        # Relative Risk Reduction (RRR)
        rrr = (1.0 - rr) * 100.0 if rr > 0 else 0.0

        # Number Needed to Treat (NNT = 1 / |RD|)
        nnt = (1.0 / abs(rd)) if abs(rd) > 1e-6 else float("inf")

        # 2x2 Contingency Table for Chi-square and Fisher's exact
        table_2x2 = np.array(
            [
                [events_treat, max(0, n_treat - events_treat)],
                [events_ctrl, max(0, n_ctrl - events_ctrl)],
            ]
        )
        try:
            chi2_res = stats.chi2_contingency(table_2x2, correction=True)
            chi2_stat = float(chi2_res.statistic)
            chi2_p = float(chi2_res.pvalue)
        except Exception:
            chi2_stat, chi2_p = 0.0, 1.0

        try:
            fisher_res = stats.fisher_exact(table_2x2)
            fisher_or = float(fisher_res.statistic)
            fisher_p = float(fisher_res.pvalue)
        except Exception:
            fisher_or, fisher_p = 1.0, 1.0

        summary_df = pd.DataFrame(
            [
                {
                    "Metric": "Control Event Rate",
                    "Value": f"{events_ctrl}/{n_ctrl} ({p_ctrl:.1%})",
                },
                {
                    "Metric": "Intervention Event Rate",
                    "Value": f"{events_treat}/{n_treat} ({p_treat:.1%})",
                },
                {
                    "Metric": "Absolute Risk Difference (RD)",
                    "Value": f"{rd:+.1%} (95% CI {rd_ci_low:+.1%} to {rd_ci_high:+.1%})",
                },
                {
                    "Metric": "Relative Risk (RR)",
                    "Value": f"{rr:.3f} (95% CI {rr_ci_low:.3f} to {rr_ci_high:.3f})",
                },
                {
                    "Metric": "Relative Risk Reduction (RRR)",
                    "Value": f"{rrr:.1f}%",
                },
                {
                    "Metric": "Number Needed to Treat (NNT)",
                    "Value": f"{nnt:.1f}" if nnt < 1000 else ">1000",
                },
                {
                    "Metric": "Chi-Square Test (with Yates)",
                    "Value": f"χ² = {chi2_stat:.3f}, P = {chi2_p:.4f}",
                },
                {
                    "Metric": "Fisher's Exact Test",
                    "Value": f"Odds Ratio = {fisher_or:.3f}, P = {fisher_p:.4f}",
                },
            ]
        )

        metrics = {
            "n_control": n_ctrl,
            "events_control": events_ctrl,
            "p_control": p_ctrl,
            "n_intervention": n_treat,
            "events_intervention": events_treat,
            "p_intervention": p_treat,
            "risk_diff": rd,
            "risk_diff_ci": (rd_ci_low, rd_ci_high),
            "relative_risk": rr,
            "relative_risk_ci": (rr_ci_low, rr_ci_high),
            "relative_risk_reduction": rrr,
            "nnt": nnt,
            "chi2_stat": chi2_stat,
            "chi2_p": chi2_p,
            "fisher_or": fisher_or,
            "fisher_p": fisher_p,
        }

        fig = go.Figure()
        groups = ["Control Arm", "Intervention Arm"]
        rates = [p_ctrl * 100.0, p_treat * 100.0]
        ci_lower = [
            max(
                0.0,
                (p_ctrl - 1.96 * np.sqrt(p_ctrl * (1 - p_ctrl) / max(1, n_ctrl)))
                * 100.0,
            ),
            max(
                0.0,
                (p_treat - 1.96 * np.sqrt(p_treat * (1 - p_treat) / max(1, n_treat)))
                * 100.0,
            ),
        ]
        ci_upper = [
            min(
                100.0,
                (p_ctrl + 1.96 * np.sqrt(p_ctrl * (1 - p_ctrl) / max(1, n_ctrl)))
                * 100.0,
            ),
            min(
                100.0,
                (p_treat + 1.96 * np.sqrt(p_treat * (1 - p_treat) / max(1, n_treat)))
                * 100.0,
            ),
        ]
        fig.add_trace(
            go.Bar(
                x=groups,
                y=rates,
                text=[
                    f"{r:.1f}%<br>({e}/{n})"
                    for r, e, n in zip(
                        rates, [events_ctrl, events_treat], [n_ctrl, n_treat]
                    )
                ],
                textposition="auto",
                marker=dict(color=["#94a3b8", "#0284c7"]),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[u - r for u, r in zip(ci_upper, rates)],
                    arrayminus=[r - low_val for r, low_val in zip(rates, ci_lower)],
                ),
            )
        )
        fig.update_layout(
            title=f"RCT Primary Outcome Comparison ({outcome_col})",
            yaxis_title="Event Rate (%)",
            yaxis=dict(range=[0, min(100, max(ci_upper) * 1.25 + 5)]),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return summary_df, metrics, fig

    @staticmethod
    def run_psm(
        df: pd.DataFrame,
        treatment_col: str,
        covariate_cols: list[str],
        outcome_col: str | None = None,
        caliper: float = 0.20,
        ratio: int = 1,
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure, pd.DataFrame]:
        """
        Executes 1:1 Nearest-Neighbor Propensity Score Matching (PSM),
        evaluates Standardized Mean Differences (SMD) before and after matching (Love plot),
        and analyzes the matched cohort.
        """
        ps_series, _missing_info = psm_lib.calculate_ps(
            df, treatment=treatment_col, covariates=covariate_cols
        )
        df_with_ps = df.copy()
        df_with_ps["_ps"] = ps_series

        df_matched = psm_lib.perform_matching(
            data=df_with_ps,
            treatment_col=treatment_col,
            ps_col="_ps",
            caliper=float(caliper),
            ratio=int(ratio),
        )

        smd_pre = psm_lib.check_balance(
            df, treatment=treatment_col, covariates=covariate_cols
        )
        smd_post = psm_lib.check_balance(
            df_matched if not df_matched.empty else df,
            treatment=treatment_col,
            covariates=covariate_cols,
        )
        fig_love = psm_lib.plot_love_plot(smd_pre, smd_post)

        balance_rows = []
        if (
            isinstance(smd_pre, pd.DataFrame)
            and not smd_pre.empty
            and isinstance(smd_post, pd.DataFrame)
            and not smd_post.empty
        ):
            merged = pd.merge(
                smd_pre[["Covariate", "SMD"]].rename(columns={"SMD": "SMD_pre"}),
                smd_post[["Covariate", "SMD"]].rename(columns={"SMD": "SMD_post"}),
                on="Covariate",
                how="outer",
            )
            for _, r in merged.iterrows():
                cov = str(r["Covariate"])
                pre_val = float(r["SMD_pre"]) if pd.notnull(r["SMD_pre"]) else 0.0
                post_val = float(r["SMD_post"]) if pd.notnull(r["SMD_post"]) else 0.0
                balanced = (
                    "✅ Balanced (<0.10)"
                    if abs(post_val) < 0.10
                    else "⚠️ Imbalanced (≥0.10)"
                )
                balance_rows.append(
                    {
                        "Covariate": cov,
                        "SMD Before": f"{pre_val:.3f}",
                        "SMD After": f"{post_val:.3f}",
                        "Status": balanced,
                    }
                )
        balance_df = pd.DataFrame(balance_rows)

        matched_outcome_stats: dict[str, Any] = {}
        matched_coef_df = None
        if not df_matched.empty and outcome_col and outcome_col in df_matched.columns:
            clean_matched = df_matched[
                [outcome_col, treatment_col] + covariate_cols
            ].dropna()
            if not clean_matched.empty:
                try:
                    matched_coef_df, matched_metrics, _ = StatHarness.run_logistic(
                        clean_matched,
                        outcome_col=outcome_col,
                        predictor_cols=[treatment_col] + covariate_cols,
                    )
                    matched_outcome_stats = matched_metrics
                except Exception as e:
                    logger.warning(f"Matched cohort regression error: {e}")

        stats_dict = {
            "n_original": len(df),
            "n_matched": len(df_matched),
            "n_treated_matched": int(df_matched[treatment_col].sum())
            if not df_matched.empty and treatment_col in df_matched.columns
            else 0,
            "caliper": caliper,
            "ratio": ratio,
            "balance_df": balance_df,
            "matched_outcome_stats": matched_outcome_stats,
            "matched_coef_df": matched_coef_df,
        }

        return balance_df, stats_dict, fig_love, df_matched

    @staticmethod
    def run_linear(
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure]:
        """Fits OLS Linear Regression."""
        clean_cols = [outcome_col] + [c for c in predictor_cols if c in df.columns]
        df_clean = df[clean_cols].dropna()

        res = linear_lib.run_ols_regression(
            df=df_clean, outcome_col=outcome_col, predictor_cols=predictor_cols
        )
        coef_df = res["coef_table"]
        diag_plots = linear_lib.create_diagnostic_plots(res)
        fig = diag_plots.get("residuals_vs_fitted", go.Figure())
        return coef_df, res, fig

    @staticmethod
    def run_sample_size(
        p1: float = 0.30,
        p2: float = 0.15,
        power: float = 0.80,
        alpha: float = 0.05,
        dropout_rate: float = 0.15,
    ) -> dict[str, Any]:
        """Calculates Sample Size and Power with Drop-out adjustment."""
        return SampleSizeTool.calculate_two_proportions(
            p1=p1, p2=p2, power=power, alpha=alpha, dropout_rate=dropout_rate
        )


class ClinicalAnalystEngine:
    """
    Main reasoning engine that inspects proposals & datasets,
    selects appropriate biostatistical tools, and immediately executes them.
    """

    @classmethod
    def process_turn(
        cls,
        user_message: str,
        file_paths: list[str] | None,
        state: AppState,
    ) -> tuple[str, AppState, go.Figure | None, pd.DataFrame | None]:
        """
        Processes a multi-turn user message + attached files, executes statistical
        methods deterministically, and updates global AppState.
        """
        user_msg = (user_message or "").strip()
        lower_msg = user_msg.lower()
        file_paths = file_paths or []

        proposal_meta: ProposalMetadata | None = None
        loaded_new_data = False

        # 1. Handle File Ingestion
        for f in file_paths:
            p = Path(f)
            suffix = p.suffix.lower()
            if suffix in [".docx", ".doc", ".pdf", ".txt", ".md"]:
                logger.info(f"Parsing Proposal document: {p.name}")
                proposal_meta = ProposalParser.parse_proposal(p)
            elif suffix in [".csv", ".xlsx", ".xlsm", ".xls", ".sav", ".dta"]:
                logger.info(f"Loading Research dataset: {p.name}")
                try:
                    df = load_data_robust(p)
                    state.df = df
                    state.file_name = p.name
                    state.df_matched = None
                    state.is_matched = False
                    loaded_new_data = True
                except Exception as e:
                    logger.error(f"Error loading dataset {p.name}: {e}")

        # If user pasted raw proposal text in the prompt
        if not proposal_meta and any(
            k in lower_msg
            for k in [
                "pico",
                "population",
                "intervention",
                "primary outcome",
                "proposal",
                "ระเบียบวิธีวิจัย",
                "โครงร่างงานวิจัย",
            ]
        ):
            proposal_meta = ProposalParser.parse_proposal(user_msg)

        # 2. Determine Action Mode & Statistical Path

        # Case A: User selected a specific option from proposed directions (e.g. "เลือกข้อ 2", "option 2", "ขอข้อ 1", "เอาข้อ 3")
        opt_match = re.search(r"(?:เลือก|option|ข้อ|แนวทางที่)\s*([1-5])", lower_msg)
        if opt_match and not state.has_data():
            opt_id = int(opt_match.group(1))
            topic = re.sub(r"(?:เลือก|option|ข้อ|แนวทางที่|\d)", "", user_msg).strip()
            topic = topic or "Acute Dyspnea Clinical Investigation"

            df_gen, meta = SyntheticDataTool.generate_topic_aware_cohort(
                topic, n=250, seed=42
            )
            state.df = df_gen
            state.file_name = (
                f"Option {opt_id} Cohort: {meta.get('domain', 'Clinical Research')}"
            )
            state.var_meta = meta

            cols = df_gen.columns.tolist()
            time_col = select_variable_by_keyword(cols, ["time", "duration", "days"])
            event_col = select_variable_by_keyword(cols, ["death", "event", "status"])
            treat_col = select_variable_by_keyword(
                cols, ["treatment", "group", "therapy"]
            )
            covars = [
                c
                for c in cols
                if c not in [time_col, event_col, treat_col]
                and not any(
                    id_k in c.lower() for id_k in ["id", "patient", "subject", "hn"]
                )
            ][:4]

            fig = go.Figure()
            if opt_id == 1:
                t_col = treat_col or cols[0]
                o_col = event_col or (cols[1] if len(cols) > 1 else cols[0])
                summary_df, metrics, fig = StatHarness.run_binary_rct(
                    df_gen,
                    treatment_col=t_col,
                    outcome_col=o_col,
                )
                response_md = f"""### 🚀 Executing Option {opt_id} (Randomized Controlled Trial - CONSORT 2010)

**Generated Synthetic Cohort:** `{state.file_name}` (n = {len(df_gen):,} patients)
- **Treatment Arm:** `{t_col}` (Control: `{metrics["n_control"]}` vs Intervention: `{metrics["n_intervention"]}`)
- **Primary Endpoint:** `{o_col}` (Binary Event)

#### 1. Primary Endpoint Comparison & Effect Sizes:
- **Control Event Rate:** `{metrics["events_control"]}/{metrics["n_control"]}` (**`{metrics["p_control"]:.1%}`**)
- **Intervention Event Rate:** `{metrics["events_intervention"]}/{metrics["n_intervention"]}` (**`{metrics["p_intervention"]:.1%}`**)
- **Relative Risk (RR):** **`{metrics["relative_risk"]:.3f}`** (95% CI `{metrics["relative_risk_ci"][0]:.3f}` to `{metrics["relative_risk_ci"][1]:.3f}`)
- **Absolute Risk Difference (RD):** **`{metrics["risk_diff"]:+.1%}`** (95% CI `{metrics["risk_diff_ci"][0]:+.1%}` to `{metrics["risk_diff_ci"][1]:+.1%}`)
- **Number Needed to Treat (NNT):** **`{metrics["nnt"]:.1f}`** patients | **Relative Risk Reduction (RRR):** `{metrics["relative_risk_reduction"]:.1f}%`

#### 2. Hypothesis Testing & Significance:
- **Chi-Square Test (with Yates correction):** $\\chi^2$ = `{metrics["chi2_stat"]:.3f}` (P-value = **`{metrics["chi2_p"]:.4f}`**)
- **Fisher's Exact Test:** P-value = **`{metrics["fisher_p"]:.4f}`** (Odds Ratio = `{metrics["fisher_or"]:.3f}`)

{summary_df.to_markdown(index=False)}
"""
                return response_md, state, fig, df_gen
            elif opt_id == 2 and time_col and event_col:
                fig, km_df, stats_dict = StatHarness.run_survival(
                    df_gen,
                    time_col=time_col,
                    event_col=event_col,
                    group_col=treat_col,
                    covar_cols=covars,
                )
                km_p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
                km_p_val_str = (
                    f"{km_p_val:.4f}"
                    if isinstance(km_p_val, (int, float))
                    else str(km_p_val)
                )
                response_md = f"""### 🚀 Executing Option {opt_id} (Kaplan-Meier & Cox Proportional Hazards - STROBE)

**Generated Synthetic Cohort:** `{state.file_name}` (n = {len(df_gen):,} patients)

#### 1. Kaplan-Meier Survival Analysis & Log-Rank Test:
- **Duration / Time:** `{time_col}` | **Event / Status:** `{event_col}`
- **Log-Rank P-value:** `{km_p_val_str}`
- **Total Observed Events:** `{df_gen[event_col].sum() if event_col in df_gen.columns else "N/A"}` out of `{len(df_gen)}` subjects

#### 2. Multivariable Cox Proportional Hazards Model:
- **Confounders Adjusted:** {", ".join([f"`{c}`" for c in covars]) if covars else "None"}
- **Status:** Cohort saved to session state and Kaplan-Meier plot rendered in the Visual Output panel.
"""
                return response_md, state, fig, df_gen
            elif opt_id == 3:
                idx_col = (
                    select_variable_by_keyword(
                        cols,
                        [
                            "pocus",
                            "index_test",
                            "test",
                            "biomarker",
                            "screening",
                            "treatment",
                            "arm",
                            "group",
                        ],
                    )
                    or cols[0]
                )
                ref_col = select_variable_by_keyword(
                    [c for c in cols if c != idx_col],
                    [
                        "gold_standard",
                        "reference",
                        "ref",
                        "disease",
                        "diagnosis",
                        "status",
                        "mortality",
                        "death",
                        "event",
                        "outcome",
                    ],
                ) or (cols[1] if len(cols) > 1 else cols[0])

                metrics_df, metrics, fig = StatHarness.run_diagnostic(
                    df_gen,
                    index_test_col=idx_col,
                    ref_standard_col=ref_col,
                )
                matrix_label = (
                    "2x2 Matrix Counts (Demonstration Example Data)"
                    if metrics.get("used_example_counts")
                    else "2x2 Matrix Counts (Derived from Cohort)"
                )
                response_md = f"""### 🚀 Executing Option {opt_id} (Diagnostic Accuracy & Fagan Nomogram - STARD 2015)

**Generated Synthetic Cohort:** `{state.file_name}` (n = {len(df_gen):,} subjects)
- **Index Diagnostic Test:** `{idx_col}` | **Reference Standard:** `{ref_col}`
- **{matrix_label}:** TP = `{metrics["tp"]}`, FP = `{metrics["fp"]}`, FN = `{metrics["fn"]}`, TN = `{metrics["tn"]}`

#### 1. Diagnostic Performance Metrics:
- **Sensitivity:** `{metrics["sensitivity"]:.1%}` | **Specificity:** `{metrics["specificity"]:.1%}`
- **Positive Likelihood Ratio (LR+):** `{metrics["lr_pos"]:.2f}` | **Negative Likelihood Ratio (LR-):** `{metrics["lr_neg"]:.2f}`
- **Diagnostic Odds Ratio (DOR):** `{metrics["dor"]:.2f}`

#### 2. Bayesian Pre-test to Post-test Updating (Fagan Nomogram):
- **Pre-Test Probability:** `{metrics["pre_test_prob"]:.1f}%`
- **Post-Test Probability (Positive Test):** `{metrics["post_prob_pos"]:.1f}%`
- **Post-Test Probability (Negative Test):** `{metrics["post_prob_neg"]:.1f}%`

{metrics_df.to_markdown(index=False)}
"""
                return response_md, state, fig, df_gen
            elif opt_id == 4:
                o_col = event_col or (cols[1] if len(cols) > 1 else cols[0])
                p_cols = covars or [c for c in cols if c != o_col][:4]
                coef_df, metrics, fig = StatHarness.run_logistic(
                    df_gen,
                    outcome_col=o_col,
                    predictor_cols=p_cols,
                )
                response_md = f"""### 🚀 Executing Option {opt_id} (Clinical Prediction Model - TRIPOD+AI)

**Dataset:** `{state.file_name}` (n = {len(df_gen):,} records)  
**Dependent Outcome:** `{o_col}` (Binary Event)  
**McFadden Pseudo-$R^2$:** `{metrics.get("mcfadden", 0.0):.4f}` | **AIC:** `{metrics.get("aic", 0.0):.1f}`

#### Odds Ratios & Multivariable Model Summary:
{coef_df.to_markdown(index=False)}
"""
                return response_md, state, fig, df_gen
            elif opt_id == 5:
                t_col = treat_col or cols[0]
                o_col = event_col or (cols[1] if len(cols) > 1 else cols[0])
                psm_covars = covars or [c for c in cols if c not in [t_col, o_col]][:4]

                balance_df, stats_dict, fig, df_matched = StatHarness.run_psm(
                    df_gen,
                    treatment_col=t_col,
                    covariate_cols=psm_covars,
                    outcome_col=o_col,
                    caliper=0.20,
                    ratio=1,
                )

                state.df_matched = df_matched
                state.is_matched = not df_matched.empty
                state.matched_treatment_col = t_col
                state.matched_covariates = psm_covars

                matched_reg_md = ""
                if (
                    stats_dict.get("matched_coef_df") is not None
                    and not stats_dict["matched_coef_df"].empty
                ):
                    matched_reg_md = f"""
#### 3. Matched Cohort Analysis:
{stats_dict["matched_coef_df"].to_markdown(index=False)}
"""

                response_md = f"""### 🚀 Executing Option {opt_id} (Propensity Score Matching & Causal Inference)

**Generated Synthetic Cohort:** `{state.file_name}` (n = {len(df_gen):,} records)
- **Treatment Exposure:** `{t_col}` | **Outcome:** `{o_col}`
- **Covariates Adjusted in PS Model:** {", ".join([f"`{c}`" for c in psm_covars])}

#### 1. Matching Results (1:1 Nearest-Neighbor, Caliper 0.20 SD):
- **Original Cohort:** `{stats_dict["n_original"]}` subjects
- **Matched Balanced Cohort:** **`{stats_dict["n_matched"]}` subjects** ({stats_dict["n_treated_matched"]} pairs)
- **Status:** Balanced cohort saved to session (`AppState.df_matched`).

#### 2. Covariate Balance Assessment (Love Plot & SMD):
{balance_df.to_markdown(index=False) if not balance_df.empty else "No covariate balance table available."}
{matched_reg_md}
"""
                return response_md, state, fig, df_gen
            else:
                html_table, df_sub = StatHarness.run_table_one(
                    df_gen, group_col=treat_col
                )
                response_md = f"""### 🚀 Executing Option {opt_id} (Baseline Characteristics & Table 1)

**Generated Synthetic Cohort:** `{state.file_name}` (n = {len(df_gen):,} subjects)

Dataset saved to session and ready for downstream analysis.
"""
                return response_md, state, fig, df_gen

        # Case B: Broad Topic Ideation & PubMed Evidence Search (e.g. "dyspnea", "sepsis", "triage", "คัดกรอง", "เสนอแนวทางวิจัย", "หัวข้อวิจัย")
        is_topic_query = (
            not state.has_data()
            and not file_paths
            and not proposal_meta
            and (
                len(user_msg.split()) <= 12
                or any(
                    k in lower_msg
                    for k in [
                        "dyspnea",
                        "sepsis",
                        "cardiac arrest",
                        "aki",
                        "stroke",
                        "heart failure",
                        "pneumonia",
                        "asthma",
                        "copd",
                        "triage",
                        "screening",
                        "emergency",
                        "เหนื่อย",
                        "หอบ",
                        "หัวข้อวิจัย",
                        "เสนอแนวทาง",
                        "ไอเดียวิจัย",
                        "หัวข้อ",
                        "directions",
                        "research ideas",
                        "topic",
                        "เจ็บหน้าอก",
                        "ติดเชื้อ",
                        "คัดกรอง",
                        "ฉุกเฉิน",
                        "ห้องฉุกเฉิน",
                        "er",
                    ]
                )
            )
            and not any(
                k in lower_msg
                for k in [
                    "sample size",
                    "table 1",
                    "synthetic",
                    "cohort",
                    "จำลอง",
                    "สร้าง",
                    "คำนวณ",
                    "สร้างข้อมูล",
                ]
            )
        )

        if is_topic_query:
            options, articles, norm_q = (
                ClinicalTopicIdeator.generate_research_directions(user_msg)
            )

            # If LLM is available, synthesize tailored clinical proposals
            if ClinicalAgentRunner.is_llm_available():
                llm_synth = ClinicalAgentRunner.synthesize_proposals_with_llm(
                    user_msg, articles
                )
                if llm_synth:
                    return llm_synth, state, None, None

            response_md = ClinicalTopicIdeator.format_proposals_markdown(
                user_msg, options, articles
            )
            return response_md, state, None, None

        # Case C: Synthetic Cohort Generation requested
        if any(
            k in lower_msg
            for k in [
                "synthetic",
                "สร้างข้อมูล",
                "จำลองข้อมูล",
                "mock cohort",
                "ตัวอย่างข้อมูล",
            ]
        ):
            topic = user_msg if len(user_msg) > 5 else "SGLT2 inhibitor in HFrEF Trial"
            df_gen, meta = SyntheticDataTool.generate_topic_aware_cohort(
                topic, n=200, seed=42
            )
            state.df = df_gen
            state.file_name = (
                f"Synthetic Cohort: {meta.get('domain', 'Clinical Research')}"
            )
            state.var_meta = meta

            time_col = select_variable_by_keyword(
                df_gen.columns.tolist(), ["time", "duration", "days"]
            )
            event_col = select_variable_by_keyword(
                df_gen.columns.tolist(), ["death", "event", "status"]
            )
            treat_col = select_variable_by_keyword(
                df_gen.columns.tolist(), ["treatment", "group", "therapy"]
            )

            fig = go.Figure()
            stats_dict = {}
            if time_col and event_col:
                fig, km_df, stats_dict = StatHarness.run_survival(
                    df_gen, time_col=time_col, event_col=event_col, group_col=treat_col
                )

            km_p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
            km_p_val_str = (
                f"{km_p_val:.4f}" if isinstance(km_p_val, float) else str(km_p_val)
            )

            pico = meta.get("pico", {})
            response_md = f"""### 🧬 Synthetic Clinical Cohort Generated Successfully

**Dataset:** `{state.file_name}` (n = {len(df_gen):,} records, {len(df_gen.columns)} variables)

#### 📋 PICO Framework:
- **👥 Population (P):** {pico.get("population", "Adult clinical cohort")}
- **💊 Intervention/Exposure (I):** {pico.get("exposure", "Active treatment")}
- **⚖️ Comparator (C):** {pico.get("comparator", "Standard of care")}
- **🎯 Primary Outcome (O):** {pico.get("outcome", "All-cause mortality / Event")}

---

#### 🚀 Immediate Statistical Execution:
1. **Kaplan-Meier Survival Analysis:** Fit survival curves stratified by treatment arm (`{treat_col}`)
   - **Log-Rank Test P-value:** `{km_p_val_str}`
   - **Total Events:** `{df_gen[event_col].sum() if event_col else "N/A"}` events
2. **Session State Synced:** You can now switch to **📊 Data Profiler**, **📈 Regression**, or **👥 Table 1 & Matching** tabs for further in-depth analysis.
"""
            return response_md, state, fig, df_gen

        # Case D: Sample Size & Power Calculation requested
        if any(
            k in lower_msg
            for k in [
                "sample size",
                "power",
                "คำนวณกลุ่มตัวอย่าง",
                "คำนวณตัวอย่าง",
                "ขนาดตัวอย่าง",
            ]
        ):
            p1 = 0.30
            p2 = 0.15
            power = 0.80
            alpha = 0.05
            p_nums = re.findall(r"(\d+(?:\.\d+)?)\s*%", user_msg)
            if len(p_nums) >= 2:
                p1 = float(p_nums[0]) / 100.0
                p2 = float(p_nums[1]) / 100.0

            res = StatHarness.run_sample_size(
                p1=p1, p2=p2, power=power, alpha=alpha, dropout_rate=0.15
            )
            response_md = f"""### 📐 Sample Size & Statistical Power Calculation

**Standard Formula:** Fleiss formula with Continuity Correction (SAMPL Compliant)

| Parameter | Assigned Value |
| :--- | :--- |
| **Control Group Event Rate ($p_1$)** | `{res["p1_control"]:.1%}` |
| **Intervention Group Event Rate ($p_2$)** | `{res["p2_intervention"]:.1%}` |
| **Type I Error ($\\alpha$, 2-sided)** | `{alpha}` (95% Confidence Level) |
| **Statistical Power ($1 - \\beta$)** | `{power:.0%}` |
| **Anticipated Drop-out Rate** | `15.0%` |

#### 🎯 Target Sample Size:
- **Control Group:** `{res["n_control_adjusted"]}` subjects
- **Intervention Group:** `{res["n_intervention_adjusted"]}` subjects
- **Total Required Enrollment:** **`{res["n_total_adjusted"]}` patients**

> 💡 **Methodology Justification Text:**  
> *"{res["justification_text"]}"*
"""
            return response_md, state, None, state.df

        # Case E: Proposal Uploaded or Parsed
        if proposal_meta:
            recs_list = "\n".join(
                [f"- ✔️ **{m}**" for m in proposal_meta.recommended_methods]
            )
            var_list = ", ".join([f"`{v}`" for v in proposal_meta.variables_identified])

            dataset_exec_section = ""
            fig = None
            preview_df = state.df

            if state.has_data() and state.df is not None:
                df = state.df
                cols = df.columns.tolist()
                time_col = select_variable_by_keyword(
                    cols, ["time", "duration", "days", "fu_time"]
                )
                event_col = select_variable_by_keyword(
                    cols, ["death", "event", "status", "mortality"]
                )
                treat_col = select_variable_by_keyword(
                    cols, ["treatment", "group", "arm", "therapy"]
                )
                covar_candidates = [
                    c
                    for c in cols
                    if c not in [time_col, event_col, treat_col]
                    and not any(
                        id_k in c.lower() for id_k in ["id", "patient", "subject", "hn"]
                    )
                ][:4]
                covar_str = ", ".join(covar_candidates)

                if time_col and event_col:
                    fig, km_df, stats_dict = StatHarness.run_survival(
                        df,
                        time_col=time_col,
                        event_col=event_col,
                        group_col=treat_col,
                        covar_cols=covar_candidates,
                    )
                    p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
                    p_val_str = (
                        f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
                    )

                    dataset_exec_section = f"""
---
### 🚀 Execution on Active Session Dataset:
- **Dataset:** `{state.file_name}` (n = {len(df):,} records)
- **Detected Variables:** Time = `{time_col}`, Event = `{event_col}`, Group = `{treat_col or "Overall"}`
- **Kaplan-Meier & Log-Rank Test:** P-value = **`{p_val_str}`**
- **Multivariable Cox Model:** Adjusted for confounders ({covar_str})

*(Kaplan-Meier Survival Function is displayed in the Visual Output panel on the right)*
"""
                elif event_col and treat_col:
                    coef_df, metrics, fig = StatHarness.run_logistic(
                        df,
                        outcome_col=event_col,
                        predictor_cols=covar_candidates or [treat_col],
                    )
                    dataset_exec_section = f"""
---
### 🚀 Multivariable Logistic Regression Execution:
- **Dataset:** `{state.file_name}` (n = {len(df):,} records)
- **Primary Binary Outcome:** `{event_col}`
- **Pseudo $R^2$ (McFadden):** `{metrics.get("mcfadden", 0.0):.4f}` | **AIC:** `{metrics.get("aic", 0.0):.1f}`
"""
            else:
                sample_calc = StatHarness.run_sample_size(
                    p1=0.30, p2=0.18, power=0.80, alpha=0.05
                )
                dataset_exec_section = f"""
---
### 📐 Initial Sample Size Planning:
- **Target Effect Size:** $p_1 = 30.0\\%$ vs $p_2 = 18.0\\%$ ($\\Delta = 12.0\\%$)
- **Recommended Sample Size (with 15% Drop-out):** **`{sample_calc["n_total_adjusted"]}` subjects** (`{sample_calc["n_control_adjusted"]}` per arm)

💡 *Would you like to generate a Synthetic Clinical Cohort matching this proposal to test the statistical pipeline? Type "generate synthetic data".*
"""

            response_md = f"""### 📄 Research Proposal & Protocol Analysis

**Study Title:** `{proposal_meta.title}`  
**Study Design:** **{proposal_meta.study_design}**

#### 📋 PICO Framework:
- **👥 Population (P):** {proposal_meta.population}
- **💊 Intervention / Exposure (I):** {proposal_meta.intervention_exposure}
- **⚖️ Comparator (C):** {proposal_meta.comparator}
- **🎯 Primary Outcome (O):** {proposal_meta.primary_outcome}
- **📊 Detected Variables:** {var_list}

#### 📐 Recommended Statistical Pipeline (SAMPL & EQUATOR Compliant):
{recs_list}
{dataset_exec_section}
"""
            return response_md, state, fig, preview_df

        # Case F: Dataset Uploaded without Proposal or Statistical Command on Active Data
        if state.has_data() and state.df is not None:
            df = state.df
            cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            time_col = select_variable_by_keyword(
                num_cols, ["time", "duration", "days", "fu_days"]
            )
            event_col = select_variable_by_keyword(
                cols, ["death", "event", "status", "mortality", "died"]
            )
            treat_col = select_variable_by_keyword(
                cols, ["treatment", "group", "arm", "therapy", "intervention"]
            )
            covariates = [
                c
                for c in cols
                if c not in [time_col, event_col, treat_col]
                and not any(
                    id_k in c.lower() for id_k in ["id", "patient", "subject", "hn"]
                )
            ][:4]
            covar_str = ", ".join([f"`{c}`" for c in covariates])

            # F1. Survival command
            if any(
                k in lower_msg
                for k in ["survival", "kaplan", "cox", "log-rank", "การรอดชีพ"]
            ):
                if not time_col or not event_col:
                    return (
                        "⚠️ Time or Event variable not detected in dataset. Please specify column names.",
                        state,
                        None,
                        df,
                    )

                fig, km_summary, stats_dict = StatHarness.run_survival(
                    df,
                    time_col=time_col,
                    event_col=event_col,
                    group_col=treat_col,
                    covar_cols=covariates,
                )
                p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
                p_val_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
                c_idx = stats_dict.get("cox_stats", {}).get(
                    "Concordance Index (C-index)", "N/A"
                )

                response_md = f"""### ⏱️ Survival Analysis Execution

**Dataset:** `{state.file_name}` (n = {len(df):,})  
**Variables:** Duration = `{time_col}`, Event = `{event_col}`, Group = `{treat_col or "None"}`

#### 1. Kaplan-Meier & Log-Rank Test:
- **Log-Rank P-value:** **`{p_val_str}`**
- **Total Events:** `{df[event_col].sum()}` out of `{len(df)}` subjects

#### 2. Multivariable Cox Proportional Hazards Model:
- **Concordance Index (C-index):** `{c_idx}`
- **Covariates Adjusted:** {covar_str}

*(Kaplan-Meier Survival Function has been rendered in the Visual Output window)*
"""
                return response_md, state, fig, df

            # F2. Table 1 Baseline command
            if any(
                k in lower_msg
                for k in ["table 1", "baseline", "ตารางที่ 1", "ลักษณะพื้นฐาน"]
            ):
                html_t1, df_t1 = StatHarness.run_table_one(
                    df, group_col=treat_col, selected_vars=cols[:8]
                )
                response_md = f"""### 👥 Baseline Characteristics (Table 1)

**Stratified by:** `{treat_col or "Overall"}`  
**Difference Metric:** Standardized Mean Differences (SMD < 0.10 indicates balance)

{html_t1}
"""
                return response_md, state, None, df

            # F3. Regression command
            if any(
                k in lower_msg
                for k in ["regression", "logistic", "linear", "ถดถอย", "odds ratio"]
            ):
                target_outcome = event_col or (num_cols[0] if num_cols else cols[0])
                if df[target_outcome].nunique() <= 3:
                    coef_df, metrics, fig = StatHarness.run_logistic(
                        df,
                        outcome_col=target_outcome,
                        predictor_cols=covariates or cols[:4],
                    )
                    table_md = coef_df.to_markdown(index=False)
                    response_md = f"""### 🎯 Multivariable Logistic Regression

**Dependent Outcome (Y):** `{target_outcome}` (Binary)  
**McFadden Pseudo-$R^2$:** `{metrics.get("mcfadden", 0.0):.4f}` | **AIC:** `{metrics.get("aic", 0.0):.1f}`

{table_md}
"""
                    return response_md, state, fig, df
                else:
                    coef_df, res, fig = StatHarness.run_linear(
                        df,
                        outcome_col=target_outcome,
                        predictor_cols=covariates or num_cols[1:4],
                    )
                    table_md = coef_df.to_markdown(index=False)
                    response_md = f"""### 📈 Multivariable Linear Regression (OLS)

**Dependent Outcome (Y):** `{target_outcome}` (Continuous)  
**$R^2$:** `{res.get("r_squared", 0.0):.4f}` | **Adjusted $R^2$:** `{res.get("adj_r_squared", 0.0):.4f}` | **F-statistic P-value:** `{res.get("f_pvalue", 0.0):.4e}`

{table_md}
"""
                    return response_md, state, fig, df

            # F4. General Data Profile on new upload
            if loaded_new_data or not user_msg:
                quality_issues = check_data_quality(df)
                fig = plot_missing_pattern(df)
                response_md = f"""### 📊 Ingested Dataset: `{state.file_name}`

**Cohort Size:** `{len(df):,}` rows | `{len(df.columns)}` columns | Missing Cells: `{df.isna().sum().sum():,}` ({df.isna().sum().sum() / df.size * 100:.1f}%)

#### 🔍 Detected Clinical Schema:
- **⏱️ Time Variable:** `{time_col or "None"}`
- **🎯 Event / Outcome:** `{event_col or "None"}`
- **👥 Group / Exposure:** `{treat_col or "None"}`
- **📋 Covariates:** {covar_str}
- **⚠️ Data Quality Check:** Detected {len(quality_issues)} potential issues (Zero-PHI Verified)

💡 *Would you like to run a specific analysis? For example: "run survival analysis", "generate Table 1", "run logistic regression", or "calculate sample size".*
"""
                return response_md, state, fig, df

        # Default conversational / General biostatistical query
        pubmed_query = user_msg if len(user_msg) > 5 else "clinical trial evidence"
        if ClinicalAgentRunner.is_llm_available():
            extracted_q = ClinicalAgentRunner.extract_biomedical_search_terms(user_msg)
            if extracted_q:
                pubmed_query = extracted_q

        articles: list[dict[str, Any]] = []
        try:
            articles = ClinicalTopicIdeator._pubmed_tool.search_and_extract(
                pubmed_query,
                max_results=3,
            )
        except Exception as e:
            logger.warning(f"PubMed search error in clinical consultation: {e}")

        # Live LLM Agent Consultation if token is present
        if ClinicalAgentRunner.is_llm_available():
            session_ctx = (
                f"Active Dataset: {state.file_name} (n={len(state.df)} records, variables={state.df.columns.tolist()[:8]})"
                if state.has_data() and state.df is not None
                else "No active dataset loaded in session."
            )
            llm_consult = ClinicalAgentRunner.consult_llm(
                user_query=user_msg,
                articles=articles,
                session_context=session_ctx,
            )
            if llm_consult:
                return llm_consult, state, None, state.df

        vancouver_list = ""
        if articles:
            vancouver_list = "\n".join(
                [f"- **{a['title']}**\n  *{a['vancouver_citation']}*" for a in articles]
            )

        pubmed_section = (
            f"#### 📚 Benchmark Evidence from PubMed:\n{vancouver_list}"
            if vancouver_list
            else ""
        )

        response_md = f"""### 🤖 Clinical Biostatistical & Study Design Consultation

**Topic / Query:** {user_msg}

#### 💡 Methodological Guidance (SAMPL & EQUATOR Guidelines):
1. **Endpoint & Study Seam:** Clearly define your primary endpoint as Time-to-Event (Kaplan-Meier / Cox PH), Binary Proportion (Logistic Regression / Chi-square), or Continuous Metric (Linear Regression / ANCOVA).
2. **Confounder Control:** For observational cohorts, propensity score matching (PSM) or multivariable regression is recommended to reduce treatment selection bias.
3. **Statistical Reporting:** Always report Effect Sizes (HR, OR, RR) with 95% Confidence Intervals and exact P-values.

{pubmed_section}

📁 *You can specify a clinical topic (e.g., "dyspnea", "sepsis") or upload a Research Proposal (`.docx`) or Dataset (`.csv`, `.xlsx`, `.sav`) to execute statistical workflows immediately.*
"""
        return response_md, state, None, state.df
