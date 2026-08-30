"""
tests/test_synthetic_data.py - Unit Tests for Synthetic Cohort Generator
"""

from agent.tools.tool_synthetic_data import SyntheticDataTool, calculate_ckd_epi_2021


def test_synthetic_cohort_physiological_bounds():
    df = SyntheticDataTool.generate_rct_cohort(n=200, seed=42)

    assert len(df) == 200
    assert "sbp_mmhg" in df.columns
    assert "dbp_mmhg" in df.columns
    assert "map_mmhg" in df.columns
    assert "egfr_ckd_epi_ml_min" in df.columns

    # Physiological constraint: SBP must exceed DBP by at least 20 mmHg
    pulse_pressure = df["sbp_mmhg"] - df["dbp_mmhg"]
    assert (pulse_pressure >= 20.0).all(), "Pulse pressure violation detected!"

    # MAP formula validation: MAP = DBP + 1/3(SBP - DBP)
    expected_map = (df["dbp_mmhg"] + (pulse_pressure / 3.0)).round(1)
    assert (df["map_mmhg"] == expected_map).all()

    # Positive values for creatinine, glucose, and eGFR
    assert (df["serum_creatinine_mg_dl"] > 0).all()
    assert (df["fasting_glucose_mg_dl"] > 0).all()
    assert (df["egfr_ckd_epi_ml_min"] > 0).all()
    assert set(df["treatment_arm"].unique()) == {"Intervention", "Control"}

    # Age capping at 90+
    assert (df["age_years"] <= 90.0).all()


def test_ckd_epi_2021_formula():
    # 60yo Male, Cr 1.0 -> eGFR should be ~86-88
    egfr_m = calculate_ckd_epi_2021(creatinine_mg_dl=1.0, age_years=60.0, sex="Male")
    assert 75.0 <= egfr_m <= 95.0

    # 60yo Female, Cr 0.8 -> eGFR should be ~88-92
    egfr_f = calculate_ckd_epi_2021(creatinine_mg_dl=0.8, age_years=60.0, sex="Female")
    assert 75.0 <= egfr_f <= 95.0


def test_redcap_data_dictionary():
    dict_df = SyntheticDataTool.generate_redcap_data_dictionary()
    assert len(dict_df) >= 10
    assert "Variable / Field Name" in dict_df.columns
    assert "Field Type" in dict_df.columns
    assert "Field Label" in dict_df.columns
    assert "subject_id" in dict_df["Variable / Field Name"].values
    assert "map_mmhg" in dict_df["Variable / Field Name"].values


def test_synthetic_data_tool_forward_smolagents():
    tool = SyntheticDataTool()
    assert tool.name == "synthetic_cohort_generator"

    out = tool.forward(n=100, study_design="rct", intervention_ratio=0.5)
    assert "Synthetic Patient Cohort Generated (100 records)" in out
    assert "Physiological Invariant Compliance" in out
