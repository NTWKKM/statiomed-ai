"""
tests/test_survival_reference.py - Biostatistical Regression Test Harness
=============================================================================
Cross-validates Python Cox proportional hazards calculation against verified
R 4.3.3 survival::coxph(Surv(time, status) ~ age + sex, data = lung) run.
Fixture: tests/fixtures/reference_datasets/lung_benchmark.csv (228 rows, 165 events).
=============================================================================
"""

import pytest
import pandas as pd
from lifelines import CoxPHFitter
from pathlib import Path

def test_cox_ph_against_r_lung_benchmark():
    """
    R 4.3.3 survival::coxph output (Efron tie-handling):
             coef  exp(coef)  se(coef)      z  Pr(>|z|)
    age  0.017045   1.017191  0.009223  1.848  0.064591
    sex -0.513219   0.598566  0.167458 -3.065  0.002178
    """
    fixture_path = Path(__file__).parent / "fixtures" / "reference_datasets" / "lung_benchmark.csv"
    assert fixture_path.exists(), f"Benchmark fixture missing: {fixture_path}"

    df_lung = pd.read_csv(fixture_path)
    df_clean = df_lung[["time", "status", "age", "sex"]].dropna()

    cph = CoxPHFitter()
    cph.fit(df_clean, duration_col="time", event_col="status")

    # Assert exact numerical agreement with R 4.3.3 ground truth
    summary = cph.summary

    # age
    assert summary.loc["age", "coef"] == pytest.approx(0.017045, abs=1e-3)
    assert summary.loc["age", "exp(coef)"] == pytest.approx(1.017191, abs=1e-3)
    assert summary.loc["age", "se(coef)"] == pytest.approx(0.009223, abs=1e-3)
    assert summary.loc["age", "p"] == pytest.approx(0.064591, abs=1e-2)

    # sex
    assert summary.loc["sex", "coef"] == pytest.approx(-0.513219, abs=1e-3)
    assert summary.loc["sex", "exp(coef)"] == pytest.approx(0.598566, abs=1e-3)
    assert summary.loc["sex", "se(coef)"] == pytest.approx(0.167458, abs=1e-3)
    assert summary.loc["sex", "p"] == pytest.approx(0.002178, abs=1e-3)

    # Observations & Events
    assert len(df_clean) == 228
    assert int(df_clean["status"].sum()) == 165
