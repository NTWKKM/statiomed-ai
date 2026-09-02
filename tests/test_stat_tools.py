"""
tests/test_stat_tools.py - Unit Tests for smolagents Biostatistical Tool Classes
=============================================================================
Verifies forward execution, dataframe integration, and schema validation for:
  - SurvivalAnalysisTool
  - BaselineTableOneTool
  - LogisticRegressionTool
  - DiagnosticAccuracyTool
  - BinaryRCTTool
  - PropensityScoreMatchingTool
  - LinearRegressionTool
=============================================================================
"""

import pandas as pd
import pytest

from agent.agent_runner import create_clinical_agent
from agent.tools.tool_stat_harness import (
    BaselineTableOneTool,
    BinaryRCTTool,
    DiagnosticAccuracyTool,
    LinearRegressionTool,
    LogisticRegressionTool,
    PropensityScoreMatchingTool,
    SurvivalAnalysisTool,
)
from views.view_data import generate_example_dataset


@pytest.fixture
def sample_clinical_df() -> pd.DataFrame:
    df, _ = generate_example_dataset()
    return df


def test_survival_analysis_tool(sample_clinical_df):
    tool = SurvivalAnalysisTool()
    text_out, fig, summary_df, stats_dict = tool.run_with_dataframe(
        df=sample_clinical_df,
        time_col="time",
        event_col="death",
        group_col="treatment",
        covar_cols=["age", "diabetes"],
    )
    assert "Survival Analysis" in text_out
    assert fig is not None
    assert isinstance(summary_df, pd.DataFrame)
    assert "km_stats" in stats_dict
    assert "Observed Events:" in text_out


def test_survival_analysis_tool_with_string_events():
    tool = SurvivalAnalysisTool()
    n = 50
    df = pd.DataFrame(
        {
            "time": [10.0, 20.0] * (n // 2),
            "status_str": ["Alive", "Dead"] * (n // 2),
            "treatment": [0, 1] * (n // 2),
        }
    )
    text_out, fig, summary_df, stats_dict = tool.run_with_dataframe(
        df=df,
        time_col="time",
        event_col="status_str",
        group_col="treatment",
    )
    assert "**Observed Events:** 25 / 50 (50.0%)" in text_out

    # With non-standard numeric encoding and positive_val
    df_num = pd.DataFrame(
        {
            "time": [10.0, 20.0] * (n // 2),
            "status_code": [1, 2] * (n // 2),
            "treatment": [0, 1] * (n // 2),
        }
    )
    text_out2, _, _, _ = tool.run_with_dataframe(
        df=df_num,
        time_col="time",
        event_col="status_code",
        group_col="treatment",
        positive_val=2,
    )
    assert "**Observed Events:** 25 / 50 (50.0%)" in text_out2


def test_baseline_table_one_tool(sample_clinical_df):
    tool = BaselineTableOneTool()
    text_out, html_table, preview_df = tool.run_with_dataframe(
        df=sample_clinical_df,
        group_col="treatment",
        selected_vars=["age", "bmi", "diabetes", "hypertension"],
    )
    assert "Baseline Characteristics" in text_out
    assert "<table" in html_table
    assert len(preview_df) == len(sample_clinical_df)


def test_logistic_regression_tool(sample_clinical_df):
    tool = LogisticRegressionTool()
    text_out, fig, coef_df, metrics = tool.run_with_dataframe(
        df=sample_clinical_df,
        outcome_col="death",
        predictor_cols=["treatment", "age", "diabetes"],
    )
    assert "Logistic Regression" in text_out
    assert fig is not None
    assert isinstance(coef_df, pd.DataFrame)
    assert not coef_df.empty


def test_diagnostic_accuracy_tool(sample_clinical_df):
    tool = DiagnosticAccuracyTool()
    text_out, fig, metrics_df, metrics = tool.run_with_dataframe(
        tp=85, fp=15, fn=15, tn=185, pre_test_prob=25.0
    )
    assert "Diagnostic Accuracy" in text_out
    assert fig is not None
    assert metrics["sensitivity"] == 0.85
    assert metrics["specificity"] == 0.925


def test_binary_rct_tool(sample_clinical_df):
    tool = BinaryRCTTool()
    text_out, fig, summary_df, metrics = tool.run_with_dataframe(
        df=sample_clinical_df,
        treatment_col="treatment",
        outcome_col="death",
    )
    assert "Randomized Controlled Trial" in text_out
    assert fig is not None
    assert "relative_risk" in metrics
    assert "risk_diff" in metrics


def test_psm_tool(sample_clinical_df):
    tool = PropensityScoreMatchingTool()
    text_out, fig_love, balance_df, stats_dict, df_matched = tool.run_with_dataframe(
        df=sample_clinical_df,
        treatment_col="treatment",
        covariate_cols=["age", "bmi", "diabetes"],
        caliper=0.20,
    )
    assert "Propensity Score Matching" in text_out
    assert fig_love is not None
    assert not df_matched.empty
    assert stats_dict["n_matched"] > 0


def test_linear_regression_tool(sample_clinical_df):
    tool = LinearRegressionTool()
    text_out, fig, coef_df, res = tool.run_with_dataframe(
        df=sample_clinical_df,
        outcome_col="sbp",
        predictor_cols=["age", "bmi", "hypertension"],
    )
    assert "Linear Regression" in text_out
    assert fig is not None
    assert "r_squared" in res


def test_create_clinical_agent_registration(sample_clinical_df):
    agent = create_clinical_agent(state_df_provider=lambda: sample_clinical_df)
    assert agent is not None
    assert "survival_analysis" in agent.tools
    assert "table_one_baseline" in agent.tools
    assert "logistic_regression" in agent.tools
    assert "diagnostic_accuracy" in agent.tools
    assert "binary_rct_analysis" in agent.tools
    assert "propensity_score_matching" in agent.tools
    assert "linear_regression" in agent.tools
    assert "pubmed_evidence_search" in agent.tools
    assert "sample_size_calculator" in agent.tools
    assert "synthetic_cohort_generator" in agent.tools
