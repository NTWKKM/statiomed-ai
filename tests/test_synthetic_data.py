"""
tests/test_synthetic_data.py - Unit Tests for Synthetic Cohort Generator
"""

from agent.tools.tool_synthetic_data import SyntheticDataTool

def test_synthetic_cohort_physiological_bounds():
    df = SyntheticDataTool.generate_rct_cohort(n=200, seed=42)

    assert len(df) == 200
    assert "sbp_mmhg" in df.columns
    assert "dbp_mmhg" in df.columns

    # Physiological constraint: SBP must exceed DBP by at least 20 mmHg
    pulse_pressure = df["sbp_mmhg"] - df["dbp_mmhg"]
    assert (pulse_pressure >= 20.0).all(), "Pulse pressure violation detected!"

    # Positive values for creatinine and glucose
    assert (df["serum_creatinine_mg_dl"] > 0).all()
    assert (df["fasting_glucose_mg_dl"] > 0).all()
    assert set(df["treatment_arm"].unique()) == {"Intervention", "Control"}
