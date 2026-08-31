"""
agent/tools/tool_stat_harness.py - smolagents Tool Implementations for Biostatistical Methods
=============================================================================
Wraps deterministic StatHarness methods into formal smolagents Tool classes:
  - SurvivalAnalysisTool (Kaplan-Meier, Log-Rank, Cox PH)
  - BaselineTableOneTool (Table 1 with SMD, Continuous/Categorical auto-stratification)
  - LogisticRegressionTool (Multivariable Logistic, Odds Ratios, 95% CIs)
  - DiagnosticAccuracyTool (2x2 Sensitivity, Specificity, Likelihood Ratios, Fagan Nomogram)
  - BinaryRCTTool (CONSORT 2-Arm Trial, RR, RD, NNT, Chi-Square, Fisher's Exact)
  - PropensityScoreMatchingTool (1:1 Nearest-Neighbor PSM, Love Plot, SMD Balance)
  - LinearRegressionTool (Multivariable OLS, Beta coefficients, R²)
=============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go

try:
    from smolagents import Tool
except ImportError:

    class Tool:
        name: str = ""
        description: str = ""
        inputs: Dict[str, Any] = {}
        output_type: str = "string"

        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            raise NotImplementedError


class SurvivalAnalysisTool(Tool):
    name = "survival_analysis"
    description = (
        "Executes time-to-event survival analysis adhering to STROBE guidelines. "
        "Fits Kaplan-Meier survival curves, Log-Rank test between groups, and multivariable "
        "Cox Proportional Hazards model with Hazard Ratios (HR) and 95% Confidence Intervals. "
        "Call this tool when analyzing patient follow-up time, mortality, time to recurrence, or hazard rates."
    )
    inputs = {
        "time_col": {
            "type": "string",
            "description": "Column name representing time-to-event or follow-up duration (e.g. 'time', 'os_months', 'days').",
        },
        "event_col": {
            "type": "string",
            "description": "Column name representing binary status/event (1 = event/death, 0 = censored).",
        },
        "group_col": {
            "type": "string",
            "description": "Optional column name for stratification/comparison groups (e.g. 'treatment', 'arm', 'stage').",
            "nullable": True,
        },
        "covar_cols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of covariate column names for multivariable Cox PH adjustment.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, state_df_provider: Optional[Any] = None):
        super().__init__()
        self.state_df_provider = state_df_provider

    def run_with_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: Optional[str] = None,
        covar_cols: Optional[List[str]] = None,
    ) -> tuple[str, go.Figure, pd.DataFrame, dict[str, Any]]:
        from agent.clinical_analyst import StatHarness

        fig, km_summary, stats_dict = StatHarness.run_survival(
            df=df,
            time_col=time_col,
            event_col=event_col,
            group_col=group_col,
            covar_cols=covar_cols,
        )
        p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
        p_val_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)

        events_count = int(df[event_col].sum()) if event_col in df.columns else 0
        total_n = len(df)

        report = [
            "### ⏱️ Survival Analysis & Cox Proportional Hazards (STROBE Compliant)",
            f"- **Time / Follow-up Variable:** `{time_col}` | **Event Indicator:** `{event_col}`",
            f"- **Observed Events:** {events_count:,} / {total_n:,} ({events_count / total_n:.1%})"
            if total_n > 0
            else "",
            f"- **Stratification Group:** `{group_col}` (Log-Rank P-value: **{p_val_str}**)"
            if group_col
            else "- **Single Cohort KM Curve Generated**",
        ]

        cox_df = stats_dict.get("cox_df")
        if cox_df is not None and isinstance(cox_df, pd.DataFrame) and not cox_df.empty:
            report.append("\n#### Multivariable Cox Proportional Hazards Model:")
            report.append(cox_df.to_markdown())

        text_out = "\n".join([line for line in report if line])
        return text_out, fig, km_summary, stats_dict

    def forward(
        self,
        time_col: str,
        event_col: str,
        group_col: Optional[str] = None,
        covar_cols: Optional[List[str]] = None,
    ) -> str:
        if self.state_df_provider is None or self.state_df_provider() is None:
            return "Error: No active dataset loaded in session to perform survival analysis."
        df = self.state_df_provider()
        text_out, _, _, _ = self.run_with_dataframe(
            df=df,
            time_col=time_col,
            event_col=event_col,
            group_col=group_col,
            covar_cols=covar_cols,
        )
        return text_out


class BaselineTableOneTool(Tool):
    name = "table_one_baseline"
    description = (
        "Generates a publication-grade baseline characteristics Table 1 with Standardized Mean "
        "Differences (SMD), means ± SD, medians [IQR], and counts (%). "
        "Call this tool to describe participant demographics and check baseline balance across study arms."
    )
    inputs = {
        "group_col": {
            "type": "string",
            "description": "Optional column name to stratify comparison (e.g. 'treatment', 'intervention').",
            "nullable": True,
        },
        "selected_vars": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of variable columns to include in Table 1.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, state_df_provider: Optional[Any] = None):
        super().__init__()
        self.state_df_provider = state_df_provider

    def run_with_dataframe(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        selected_vars: Optional[List[str]] = None,
        var_meta: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str, pd.DataFrame]:
        from agent.clinical_analyst import StatHarness

        html_table, preview_df = StatHarness.run_table_one(
            df=df,
            group_col=group_col,
            selected_vars=selected_vars,
            var_meta=var_meta,
        )
        cols_count = len(selected_vars) if selected_vars else len(df.columns)
        strat_desc = (
            f"stratified by `{group_col}`" if group_col else "for overall cohort"
        )
        text_summary = (
            f"### 👥 Baseline Characteristics (Table 1 - SAMPL Compliant)\n"
            f"Generated baseline table {strat_desc} across {cols_count} variables with Standardized Mean Differences (SMD).\n\n"
            f"{html_table}"
        )
        return text_summary, html_table, preview_df

    def forward(
        self,
        group_col: Optional[str] = None,
        selected_vars: Optional[List[str]] = None,
    ) -> str:
        if self.state_df_provider is None or self.state_df_provider() is None:
            return "Error: No active dataset loaded in session to generate Table 1."
        df = self.state_df_provider()
        text_summary, _, _ = self.run_with_dataframe(
            df=df, group_col=group_col, selected_vars=selected_vars
        )
        return text_summary


class LogisticRegressionTool(Tool):
    name = "logistic_regression"
    description = (
        "Fits multivariable binary logistic regression adhering to TRIPOD/STROBE guidelines. "
        "Calculates adjusted Odds Ratios (OR), 95% Confidence Intervals, Wald P-values, "
        "and produces a Forest Plot. Call this tool when analyzing binary endpoints (death, readmission, complication)."
    )
    inputs = {
        "outcome_col": {
            "type": "string",
            "description": "Binary dependent outcome variable (e.g. 'mortality_30d', 'aki_stage3').",
        },
        "predictor_cols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of independent predictors and confounder variables.",
        },
    }
    output_type = "string"

    def __init__(self, state_df_provider: Optional[Any] = None):
        super().__init__()
        self.state_df_provider = state_df_provider

    def run_with_dataframe(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: List[str],
    ) -> tuple[str, go.Figure, pd.DataFrame, dict[str, Any]]:
        from agent.clinical_analyst import StatHarness

        coef_df, metrics, fig = StatHarness.run_logistic(
            df=df, outcome_col=outcome_col, predictor_cols=predictor_cols
        )
        table_md = (
            coef_df.to_markdown(index=False)
            if isinstance(coef_df, pd.DataFrame)
            else ""
        )
        text_out = (
            f"### 📊 Multivariable Logistic Regression Analysis\n"
            f"- **Primary Binary Outcome:** `{outcome_col}`\n"
            f"- **Adjusted Covariates:** {', '.join([f'`{c}`' for c in predictor_cols])}\n"
            f"- **Sample Size Analyzed:** {len(df.dropna(subset=[outcome_col] + [c for c in predictor_cols if c in df.columns])):,} records\n\n"
            f"#### Odds Ratios & 95% Confidence Intervals:\n"
            f"{table_md}\n"
        )
        return text_out, fig, coef_df, metrics

    def forward(
        self,
        outcome_col: str,
        predictor_cols: List[str],
    ) -> str:
        if self.state_df_provider is None or self.state_df_provider() is None:
            return (
                "Error: No active dataset loaded in session to run logistic regression."
            )
        df = self.state_df_provider()
        text_out, _, _, _ = self.run_with_dataframe(
            df=df, outcome_col=outcome_col, predictor_cols=predictor_cols
        )
        return text_out


class DiagnosticAccuracyTool(Tool):
    name = "diagnostic_accuracy"
    description = (
        "Calculates clinical diagnostic test accuracy metrics (STARD 2015 compliant). "
        "Computes Sensitivity, Specificity, Positive/Negative Predictive Values (PPV/NPV), "
        "Likelihood Ratios (LR+, LR-), Diagnostic Odds Ratio (DOR), and Bayesian Fagan updating. "
        "Call this tool when validating diagnostic biomarkers, screening tests, or POCUS imaging."
    )
    inputs = {
        "index_test_col": {
            "type": "string",
            "description": "Column name for index diagnostic test / screening tool (binary 0/1).",
            "nullable": True,
        },
        "ref_standard_col": {
            "type": "string",
            "description": "Column name for gold standard / reference diagnosis (binary 0/1).",
            "nullable": True,
        },
        "tp": {
            "type": "integer",
            "description": "True Positive count (if computing directly from 2x2 matrix).",
            "nullable": True,
        },
        "fp": {
            "type": "integer",
            "description": "False Positive count.",
            "nullable": True,
        },
        "fn": {
            "type": "integer",
            "description": "False Negative count.",
            "nullable": True,
        },
        "tn": {
            "type": "integer",
            "description": "True Negative count.",
            "nullable": True,
        },
        "pre_test_prob": {
            "type": "number",
            "description": "Pre-test clinical probability percentage (0.0 to 100.0, default 25.0%).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, state_df_provider: Optional[Any] = None):
        super().__init__()
        self.state_df_provider = state_df_provider

    def run_with_dataframe(
        self,
        df: Optional[pd.DataFrame] = None,
        index_test_col: Optional[str] = None,
        ref_standard_col: Optional[str] = None,
        pre_test_prob: Optional[float] = None,
        tp: Optional[int] = None,
        fp: Optional[int] = None,
        fn: Optional[int] = None,
        tn: Optional[int] = None,
    ) -> tuple[str, go.Figure, pd.DataFrame, dict[str, Any]]:
        from agent.clinical_analyst import StatHarness

        metrics_df, metrics, fig = StatHarness.run_diagnostic(
            df=df,
            index_test_col=index_test_col,
            ref_standard_col=ref_standard_col,
            pre_test_prob=pre_test_prob,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        )
        text_out = (
            f"### 🎯 Diagnostic Accuracy & Bayesian Nomogram (STARD 2015)\n"
            f"- **Index Test:** `{metrics.get('index_test_col', 'Specified Test')}` | **Reference Standard:** `{metrics.get('ref_standard_col', 'Gold Standard')}`\n"
            f"- **2x2 Matrix Counts:** TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']} (Total N={metrics['tp'] + metrics['fp'] + metrics['fn'] + metrics['tn']:,})\n\n"
            f"#### Performance Parameters:\n"
            f"{metrics_df.to_markdown(index=False)}\n\n"
            f"#### Bayesian Clinical Updating:\n"
            f"- **Pre-Test Probability:** `{metrics['pre_test_prob']:.1f}%`\n"
            f"- **Post-Test Prob with Positive Test (+LR):** **`{metrics['post_prob_pos']:.1f}%`**\n"
            f"- **Post-Test Prob with Negative Test (-LR):** **`{metrics['post_prob_neg']:.1f}%`**\n"
        )
        return text_out, fig, metrics_df, metrics

    def forward(
        self,
        index_test_col: Optional[str] = None,
        ref_standard_col: Optional[str] = None,
        tp: Optional[int] = None,
        fp: Optional[int] = None,
        fn: Optional[int] = None,
        tn: Optional[int] = None,
        pre_test_prob: Optional[float] = None,
    ) -> str:
        df = self.state_df_provider() if self.state_df_provider else None
        text_out, _, _, _ = self.run_with_dataframe(
            df=df,
            index_test_col=index_test_col,
            ref_standard_col=ref_standard_col,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            pre_test_prob=pre_test_prob,
        )
        return text_out


class BinaryRCTTool(Tool):
    name = "binary_rct_analysis"
    description = (
        "Evaluates 2-arm Randomized Controlled Trials (CONSORT 2010 compliant) with binary endpoints. "
        "Calculates Relative Risk (RR), Absolute Risk Difference (RD), Relative Risk Reduction (RRR), "
        "Number Needed to Treat (NNT), Chi-Square with Yates continuity correction, and Fisher's exact test."
    )
    inputs = {
        "treatment_col": {
            "type": "string",
            "description": "Column name for randomized treatment arm (1 = intervention, 0 = control/placebo).",
        },
        "outcome_col": {
            "type": "string",
            "description": "Column name for primary binary clinical endpoint (1 = event, 0 = no event).",
        },
    }
    output_type = "string"

    def __init__(self, state_df_provider: Optional[Any] = None):
        super().__init__()
        self.state_df_provider = state_df_provider

    def run_with_dataframe(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
    ) -> tuple[str, go.Figure, pd.DataFrame, dict[str, Any]]:
        from agent.clinical_analyst import StatHarness

        summary_df, metrics, fig = StatHarness.run_binary_rct(
            df=df, treatment_col=treatment_col, outcome_col=outcome_col
        )
        text_out = (
            f"### 💊 Randomized Controlled Trial Evaluation (CONSORT 2010)\n"
            f"- **Treatment Arm:** `{treatment_col}` (Control n={metrics['n_control']:,}, Intervention n={metrics['n_intervention']:,})\n"
            f"- **Primary Endpoint:** `{outcome_col}`\n\n"
            f"#### Effect Sizes & Event Rates:\n"
            f"- **Control Event Rate:** `{metrics['events_control']}/{metrics['n_control']}` (**`{metrics['p_control']:.1%}`**)\n"
            f"- **Intervention Event Rate:** `{metrics['events_intervention']}/{metrics['n_intervention']}` (**`{metrics['p_intervention']:.1%}`**)\n"
            f"- **Relative Risk (RR):** **`{metrics['relative_risk']:.3f}`** (95% CI `{metrics['relative_risk_ci'][0]:.3f}` to `{metrics['relative_risk_ci'][1]:.3f}`)\n"
            f"- **Absolute Risk Difference (RD):** **`{metrics['risk_diff']:+.1%}`** (95% CI `{metrics['risk_diff_ci'][0]:+.1%}` to `{metrics['risk_diff_ci'][1]:+.1%}`)\n"
            f"- **Number Needed to Treat (NNT):** **`{metrics['nnt']:.1f}`** participants | **RRR:** `{metrics['relative_risk_reduction']:.1f}%`\n\n"
            f"#### Hypothesis Tests:\n"
            f"- **Chi-Square (Yates):** $\\chi^2 = {metrics['chi2_stat']:.3f}$ (P = **{metrics['chi2_p']:.4f}**)\n"
            f"- **Fisher's Exact Test:** P = **{metrics['fisher_p']:.4f}** (Odds Ratio = {metrics['fisher_or']:.3f})\n"
        )
        return text_out, fig, summary_df, metrics

    def forward(self, treatment_col: str, outcome_col: str) -> str:
        if self.state_df_provider is None or self.state_df_provider() is None:
            return "Error: No active dataset loaded in session to evaluate RCT."
        df = self.state_df_provider()
        text_out, _, _, _ = self.run_with_dataframe(
            df=df, treatment_col=treatment_col, outcome_col=outcome_col
        )
        return text_out


class PropensityScoreMatchingTool(Tool):
    name = "propensity_score_matching"
    description = (
        "Executes 1:1 Nearest-Neighbor Propensity Score Matching (PSM) with caliper check "
        "to reduce confounding in observational comparative effectiveness studies. "
        "Calculates Standardized Mean Differences (SMD) before/after matching and produces Love plots."
    )
    inputs = {
        "treatment_col": {
            "type": "string",
            "description": "Binary treatment indicator column (1 = exposed/treated, 0 = control/unexposed).",
        },
        "covariate_cols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of confounding covariate column names for propensity score estimation.",
        },
        "outcome_col": {
            "type": "string",
            "description": "Optional outcome variable to analyze in the matched cohort.",
            "nullable": True,
        },
        "caliper": {
            "type": "number",
            "description": "Caliper width in SD of logit PS (default: 0.20).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, state_df_provider: Optional[Any] = None):
        super().__init__()
        self.state_df_provider = state_df_provider

    def run_with_dataframe(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        covariate_cols: List[str],
        outcome_col: Optional[str] = None,
        caliper: float = 0.20,
    ) -> tuple[str, go.Figure, pd.DataFrame, dict[str, Any], pd.DataFrame]:
        from agent.clinical_analyst import StatHarness

        balance_df, stats_dict, fig_love, df_matched = StatHarness.run_psm(
            df=df,
            treatment_col=treatment_col,
            covariate_cols=covariate_cols,
            outcome_col=outcome_col,
            caliper=caliper,
        )
        text_out = (
            f"### ⚖️ Propensity Score Matching (1:1 Nearest Neighbor, Caliper={caliper:.2f})\n"
            f"- **Cohort Size:** Original n={stats_dict['n_original']:,} ➔ Matched n={stats_dict['n_matched']:,} ({stats_dict['n_treated_matched']:,} pairs)\n"
            f"- **Confounders Adjusted:** {', '.join([f'`{c}`' for c in covariate_cols])}\n\n"
            f"#### Covariate Balance (Standardized Mean Difference < 0.10):\n"
            f"{balance_df.to_markdown(index=False)}\n"
        )
        return text_out, fig_love, balance_df, stats_dict, df_matched

    def forward(
        self,
        treatment_col: str,
        covariate_cols: List[str],
        outcome_col: Optional[str] = None,
        caliper: Optional[float] = 0.20,
    ) -> str:
        if self.state_df_provider is None or self.state_df_provider() is None:
            return "Error: No active dataset loaded in session to perform PSM."
        df = self.state_df_provider()
        text_out, _, _, _, _ = self.run_with_dataframe(
            df=df,
            treatment_col=treatment_col,
            covariate_cols=covariate_cols,
            outcome_col=outcome_col,
            caliper=caliper or 0.20,
        )
        return text_out


class LinearRegressionTool(Tool):
    name = "linear_regression"
    description = (
        "Fits multivariable Ordinary Least Squares (OLS) Linear Regression for continuous outcomes. "
        "Calculates Beta coefficients, standard errors, 95% CIs, R-squared, adjusted R-squared, and F-statistic."
    )
    inputs = {
        "outcome_col": {
            "type": "string",
            "description": "Continuous dependent outcome variable (e.g. 'systolic_bp', 'creatinine', 'los_days').",
        },
        "predictor_cols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of continuous and categorical predictor variables.",
        },
    }
    output_type = "string"

    def __init__(self, state_df_provider: Optional[Any] = None):
        super().__init__()
        self.state_df_provider = state_df_provider

    def run_with_dataframe(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: List[str],
    ) -> tuple[str, go.Figure, pd.DataFrame, dict[str, Any]]:
        from agent.clinical_analyst import StatHarness

        coef_df, res, fig = StatHarness.run_linear(
            df=df, outcome_col=outcome_col, predictor_cols=predictor_cols
        )
        r2 = res.get("r_squared", 0.0)
        adj_r2 = res.get("adj_r_squared", 0.0)
        table_md = coef_df.to_markdown() if isinstance(coef_df, pd.DataFrame) else ""
        text_out = (
            f"### 📈 Multivariable Linear Regression (OLS)\n"
            f"- **Outcome Variable:** `{outcome_col}` (Continuous)\n"
            f"- **Model Fit:** $R^2 = {r2:.3f}$, Adjusted $R^2 = {adj_r2:.3f}$\n"
            f"- **Predictors:** {', '.join([f'`{c}`' for c in predictor_cols])}\n\n"
            f"#### Parameter Estimates & Significance:\n"
            f"{table_md}\n"
        )
        return text_out, fig, coef_df, res

    def forward(
        self,
        outcome_col: str,
        predictor_cols: List[str],
    ) -> str:
        if self.state_df_provider is None or self.state_df_provider() is None:
            return (
                "Error: No active dataset loaded in session to run linear regression."
            )
        df = self.state_df_provider()
        text_out, _, _, _ = self.run_with_dataframe(
            df=df, outcome_col=outcome_col, predictor_cols=predictor_cols
        )
        return text_out
