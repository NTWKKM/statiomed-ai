"""
tests/test_critique_engine.py - Unit Tests for Automated Clinical Appraisal Engine
=============================================================================
Verifies EPV, quasi-separation, missingness, small-cell counts, and appraisal verdicts.
=============================================================================
"""

import numpy as np
import pandas as pd

from agent.critique_engine import CritiqueEngine, CritiqueVerdict


def test_epv_evaluation():
    # Severe deficit: 8 events, 5 covariates -> EPV = 1.6
    finding_severe = CritiqueEngine.evaluate_epv(
        n_events=8, n_non_events=100, n_covariates=5
    )
    assert finding_severe is not None
    assert finding_severe.severity == "HIGH"
    assert "Severe EPV Deficit" in finding_severe.title

    # Suboptimal: 24 events, 3 covariates -> EPV = 8.0
    finding_mod = CritiqueEngine.evaluate_epv(
        n_events=24, n_non_events=100, n_covariates=3
    )
    assert finding_mod is not None
    assert finding_mod.severity == "MODERATE"
    assert "Suboptimal EPV" in finding_mod.title

    # Adequate: 50 events, 3 covariates -> EPV = 16.6
    finding_ok = CritiqueEngine.evaluate_epv(
        n_events=50, n_non_events=100, n_covariates=3
    )
    assert finding_ok is None


def test_separation_evaluation():
    # Extreme OR indicating quasi-complete separation
    df_sep = pd.DataFrame(
        [
            {
                "Variable": "biomarker_x",
                "Odds Ratio (OR)": "250.50",
                "P-value": "0.001",
            },
            {"Variable": "age", "Odds Ratio (OR)": "1.04", "P-value": "0.02"},
        ]
    )
    finding = CritiqueEngine.evaluate_separation(df_sep)
    assert finding is not None
    assert finding.severity == "HIGH"
    assert "Quasi-Complete Separation" in finding.title

    # Normal ORs
    df_norm = pd.DataFrame(
        [
            {"Variable": "treatment", "Odds Ratio (OR)": "1.85", "P-value": "0.01"},
            {"Variable": "age", "Odds Ratio (OR)": "1.02", "P-value": "0.05"},
        ]
    )
    assert CritiqueEngine.evaluate_separation(df_norm) is None


def test_missingness_evaluation():
    # 30% missing data
    df = pd.DataFrame(
        {
            "y": [1] * 70 + [None] * 30,
            "x1": [1.0] * 100,
        }
    )
    finding = CritiqueEngine.evaluate_missingness(df, ["y", "x1"])
    assert finding is not None
    assert finding.severity == "HIGH"
    assert "High Missing Data Rate" in finding.title


def test_rct_sparse_cell_counts():
    # Table with expected cell count < 5:
    # Row 1: 1 event / 50 total (49 non-events)
    # Row 2: 3 events / 50 total (47 non-events)
    # Grand total = 100, Total events = 4 -> Expected E11 = 50*4/100 = 2.0 < 5
    finding_sparse = CritiqueEngine.evaluate_rct_cell_counts(
        events_ctrl=1, n_ctrl=50, events_treat=3, n_treat=50
    )
    assert finding_sparse is not None
    assert finding_sparse.severity == "MODERATE"
    assert "Sparse Contingency Cell" in finding_sparse.title

    # Table with expected cell counts >= 5 (even though observed events_ctrl = 2):
    # Grand total = 100, Total events = 17 -> Expected E11 = 50*17/100 = 8.5 >= 5
    finding_adequate = CritiqueEngine.evaluate_rct_cell_counts(
        events_ctrl=2, n_ctrl=50, events_treat=15, n_treat=50
    )
    assert finding_adequate is None


def test_separation_evaluation_or_zero():
    # Odds Ratio of 0 should trigger quasi-complete / complete separation
    df_zero = pd.DataFrame(
        [
            {"Variable": "treatment_arm", "Odds Ratio (OR)": "0", "P-value": "0.001"},
            {"Variable": "age", "Odds Ratio (OR)": "1.05", "P-value": "0.02"},
        ]
    )
    finding = CritiqueEngine.evaluate_separation(df_zero)
    assert finding is not None
    assert finding.severity == "HIGH"
    assert "Quasi-Complete Separation" in finding.title

    # Appraise analysis with OR=0 should produce HIGH_RISK_BIAS verdict
    df_data = pd.DataFrame(
        {"outcome": [1, 0, 1, 0] * 20, "treatment": [1, 0, 1, 0] * 20}
    )
    verdict = CritiqueEngine.appraise_analysis(
        "logistic",
        df=df_data,
        results_meta={
            "outcome_col": "outcome",
            "predictor_cols": ["treatment_arm", "age"],
            "coef_df": df_zero,
        },
    )
    assert verdict.overall_status == "HIGH_RISK_BIAS"


def test_survival_ph_diagnostic_states():
    df = pd.DataFrame(
        {
            "time": [10, 20, 30, 40, 50] * 20,
            "event": [1, 0, 1, 0, 1] * 20,
            "treatment": [1, 1, 0, 0, 1] * 20,
            "age": np.random.randint(40, 80, size=100),
        }
    )

    # 1. PH Diagnostic Passed
    verdict_passed = CritiqueEngine.appraise_analysis(
        "survival",
        df=df,
        results_meta={
            "time_col": "time",
            "event_col": "event",
            "covar_cols": ["treatment", "age"],
            "ph_diagnostic": {"passed": True, "p_value": 0.35},
        },
    )
    assert any(
        "Proportional hazards (PH) assumption verified" in s
        for s in verdict_passed.strengths
    )
    assert not any(f.category == "PH_Assumption" for f in verdict_passed.findings)

    # 2. PH Diagnostic Failed
    verdict_failed = CritiqueEngine.appraise_analysis(
        "survival",
        df=df,
        results_meta={
            "time_col": "time",
            "event_col": "event",
            "covar_cols": ["treatment", "age"],
            "ph_diagnostic": {"passed": False, "p_value": 0.012},
        },
    )
    assert not any(
        "Proportional hazards (PH) assumption verified" in s
        for s in verdict_failed.strengths
    )
    ph_finding = next(
        (f for f in verdict_failed.findings if f.category == "PH_Assumption"), None
    )
    assert ph_finding is not None
    assert ph_finding.severity == "HIGH"
    assert "Violation of Proportional Hazards" in ph_finding.title

    # 3. PH Diagnostic Absent / Unassessed
    verdict_absent = CritiqueEngine.appraise_analysis(
        "survival",
        df=df,
        results_meta={
            "time_col": "time",
            "event_col": "event",
            "covar_cols": ["treatment", "age"],
        },
    )
    assert not any("Proportional hazards" in s for s in verdict_absent.strengths)
    assert not any(f.category == "PH_Assumption" for f in verdict_absent.findings)


def test_appraise_analysis_verdict():
    df = pd.DataFrame(
        {
            "time": [10, 20, 30, 40, 50] * 20,
            "event": [1, 0, 1, 0, 1] * 20,
            "treatment": [1, 1, 0, 0, 1] * 20,
            "age": np.random.randint(40, 80, size=100),
        }
    )
    verdict = CritiqueEngine.appraise_analysis(
        analysis_type="survival",
        df=df,
        results_meta={
            "time_col": "time",
            "event_col": "event",
            "covar_cols": ["treatment", "age"],
        },
    )
    assert isinstance(verdict, CritiqueVerdict)
    assert verdict.overall_status in [
        "ROBUST",
        "VALID_WITH_LIMITATIONS",
        "HIGH_RISK_BIAS",
    ]
    md = verdict.to_markdown()
    assert "Automated Clinical Critique & Appraisal" in md


def test_psm_balance_critique_finding():
    df = pd.DataFrame(
        {
            "treatment": [1, 0] * 50,
            "age": np.random.normal(60, 10, 100),
            "bmi": np.random.normal(25, 4, 100),
        }
    )
    # 1. PSM with imbalanced covariate
    balance_df_imbalanced = pd.DataFrame(
        [
            {
                "Covariate": "age",
                "SMD Before": "0.350",
                "SMD After": "0.150",
                "Status": "⚠️ Imbalanced (≥0.10)",
            },
            {
                "Covariate": "bmi",
                "SMD Before": "0.200",
                "SMD After": "0.040",
                "Status": "✅ Balanced (<0.10)",
            },
        ]
    )
    verdict_imbalanced = CritiqueEngine.appraise_analysis(
        "psm",
        df=df,
        results_meta={
            "n_original": 100,
            "n_matched": 80,
            "balance_df": balance_df_imbalanced,
        },
    )
    assert verdict_imbalanced.overall_status == "VALID_WITH_LIMITATIONS"
    imb_finding = next(
        (f for f in verdict_imbalanced.findings if f.category == "Confounding_Bias"),
        None,
    )
    assert imb_finding is not None
    assert imb_finding.severity == "MODERATE"
    assert "Residual Post-Match Covariate Imbalance" in imb_finding.title
    assert "`age`" in imb_finding.description

    # 2. PSM with perfectly balanced covariates (< 0.10)
    balance_df_balanced = pd.DataFrame(
        [
            {
                "Covariate": "age",
                "SMD Before": "0.350",
                "SMD After": "0.050",
                "Status": "✅ Balanced (<0.10)",
            },
            {
                "Covariate": "bmi",
                "SMD Before": "0.200",
                "SMD After": "0.040",
                "Status": "✅ Balanced (<0.10)",
            },
        ]
    )
    verdict_balanced = CritiqueEngine.appraise_analysis(
        "psm",
        df=df,
        results_meta={
            "n_original": 100,
            "n_matched": 80,
            "balance_df": balance_df_balanced,
        },
    )
    assert verdict_balanced.overall_status == "ROBUST"
    assert not any(f.category == "Confounding_Bias" for f in verdict_balanced.findings)


def test_survival_epv_from_fitted_cohort():
    # 100 rows, 30 events in full df
    # Missing covariates reduce fitted complete-case events to 4
    df = pd.DataFrame(
        {
            "time": [10, 20, 30, 40, 50] * 20,
            "event": [1, 0, 1, 0, 1] * 20,
            "covar1": [1.0] * 100,
            "covar2": [2.0] * 100,
        }
    )
    # When fitted_events=4 is provided for 2 covariates -> EPV = 4/2 = 2.0 (HIGH risk)
    verdict = CritiqueEngine.appraise_analysis(
        "survival",
        df=df,
        results_meta={
            "time_col": "time",
            "event_col": "event",
            "covar_cols": ["covar1", "covar2"],
            "fitted_events": 4,
            "fitted_non_events": 50,
        },
    )
    epv_finding = next((f for f in verdict.findings if f.category == "EPV"), None)
    assert epv_finding is not None
    assert epv_finding.severity == "HIGH"
    assert "Severe EPV Deficit" in epv_finding.title
    assert "2.0" in epv_finding.title
