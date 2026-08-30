from __future__ import annotations

from tabs._common import select_variable_by_keyword


def test_select_variable_by_keyword_statin_dataset_anti_collision() -> None:
    """
    Verify that in continuous datasets like Statin LDL, N_Statin and N_Control
    are correctly matched and do not collide with Mean_Statin or Mean_Control.
    """
    cols = [
        "Study",
        "Year",
        "Mean_Statin",
        "SD_Statin",
        "N_Statin",
        "Mean_Control",
        "SD_Control",
        "N_Control",
        "Country",
        "Design",
    ]

    mean_t = select_variable_by_keyword(
        cols, ["mean_statin", "mean_treatment", "mean_tx", "mean_t", "mean1"]
    )
    sd_t = select_variable_by_keyword(
        cols, ["sd_statin", "sd_treatment", "sd_tx", "sd_t", "sd1"]
    )
    n_t = select_variable_by_keyword(
        cols, ["n_statin", "n_treatment", "n_tx", "n_t", "n1"]
    )

    mean_c = select_variable_by_keyword(
        cols, ["mean_control", "mean_ctrl", "mean_c", "mean0"]
    )
    sd_c = select_variable_by_keyword(cols, ["sd_control", "sd_ctrl", "sd_c", "sd0"])
    n_c = select_variable_by_keyword(cols, ["n_control", "n_ctrl", "n_c", "n0"])

    assert mean_t == "Mean_Statin"
    assert sd_t == "SD_Statin"
    assert n_t == "N_Statin"
    assert mean_c == "Mean_Control"
    assert sd_c == "SD_Control"
    assert n_c == "N_Control"

    # Verify all 6 continuous variables are completely distinct
    selected_vars = [mean_t, sd_t, n_t, mean_c, sd_c, n_c]
    assert len(set(selected_vars)) == 6


def test_select_variable_by_keyword_exact_and_boundary_rules() -> None:
    """
    Test individual matching tiers: exact match, word boundary, and delimited prefix/suffix.
    """
    # 1. Exact match tier (case-insensitive)
    cols1 = ["PATIENT_ID", "Age", "GENDER", "status_event"]
    assert select_variable_by_keyword(cols1, ["gender"]) == "GENDER"
    assert select_variable_by_keyword(cols1, ["patient_id"]) == "PATIENT_ID"

    # 2. Token boundary match (regex)
    cols2 = ["baseline_sbp_mmhg", "followup_sbp_mmhg", "patient_age_years"]
    assert select_variable_by_keyword(cols2, ["age"]) == "patient_age_years"

    # 3. Delimited prefix/suffix match
    cols3 = ["time_months", "status_30d", "treatment_group"]
    assert select_variable_by_keyword(cols3, ["time"]) == "time_months"
    assert select_variable_by_keyword(cols3, ["status"]) == "status_30d"

    # 4. Anti-collision: Substrings that form a part of an alphanumeric word must NOT match
    cols4 = ["mean_treatment", "sd_treatment", "n_treatment"]
    # Keyword 'n_treatment' must match 'n_treatment', not 'mean_treatment'
    assert select_variable_by_keyword(cols4, ["n_treatment"]) == "n_treatment"
    # Keyword 'sd_treatment' must match 'sd_treatment', not 'mean_treatment'
    assert select_variable_by_keyword(cols4, ["sd_treatment"]) == "sd_treatment"


def test_select_variable_by_keyword_empty_and_fallback() -> None:
    """
    Test empty column list and fallback behavior.
    """
    assert select_variable_by_keyword([], ["age"]) is None
    assert (
        select_variable_by_keyword(
            ["colA", "colB"], ["nonexistent"], default_to_first=True
        )
        == "colA"
    )
    assert (
        select_variable_by_keyword(
            ["colA", "colB"], ["nonexistent"], default_to_first=False
        )
        is None
    )
