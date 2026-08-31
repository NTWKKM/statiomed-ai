"""
tests/test_critique_engine.py - Unit Tests for Automated Clinical Appraisal Engine
=============================================================================
Verifies EPV, quasi-separation, missingness, small-cell counts, and appraisal verdicts.
=============================================================================
"""

import numpy as np
import pandas as pd
import pytest

from agent.critique_engine import CritiqueEngine, CritiqueFinding, CritiqueVerdict


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
    # Sparse cell: 2 events in control
    finding = CritiqueEngine.evaluate_rct_cell_counts(
        events_ctrl=2, n_ctrl=50, events_treat=15, n_treat=50
    )
    assert finding is not None
    assert finding.severity == "MODERATE"
    assert "Sparse Contingency Cell" in finding.title


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
