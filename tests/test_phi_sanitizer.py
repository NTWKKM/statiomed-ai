"""
tests/test_phi_sanitizer.py - Unit & Security Tests for On-Prem De-identifier
"""

import pandas as pd
from tools_local.phi_sanitizer_cli import sanitize_dataframe

def test_sanitize_dataframe_drops_direct_identifiers():
    df_raw = pd.DataFrame({
        "HN": ["MOCK_HN_01", "MOCK_HN_02"],
        "Patient_Name": ["Patient Alpha", "Patient Beta"],
        "Citizen_ID": ["MOCK_CID_101", "MOCK_CID_102"],
        "Age": [45, 62],
        "SBP": [130, 145],
        "Outcome": [0, 1]
    })

    df_clean = sanitize_dataframe(df_raw, keep_uuid=True)

    # Assert direct identifiers are completely dropped
    assert "HN" not in df_clean.columns
    assert "Patient_Name" not in df_clean.columns
    assert "Citizen_ID" not in df_clean.columns

    # Assert non-PHI clinical variables are preserved
    assert "Age" in df_clean.columns
    assert "SBP" in df_clean.columns
    assert "Outcome" in df_clean.columns

    # Assert UUID4 surrogate key is inserted
    assert "Deidentified_Patient_ID" in df_clean.columns
    assert len(df_clean["Deidentified_Patient_ID"].unique()) == 2

def test_sanitize_dataframe_temporal_shift():
    df_raw = pd.DataFrame({
        "HN": ["MOCK_HN_01", "MOCK_HN_02"],
        "admission_date": ["2024-01-01", "2024-02-01"],
        "discharge_date": ["2024-01-08", "2024-02-15"],
        "icu_event_date": ["2024-01-03", "2024-02-05"]
    })

    df_clean = sanitize_dataframe(df_raw, time_zero_col="admission_date")

    # Calendar dates should be gone
    assert "admission_date" not in df_clean.columns
    assert "discharge_date" not in df_clean.columns

    # Elapsed days should be calculated exactly
    assert "discharge_date_elapsed_days" in df_clean.columns
    assert "icu_event_date_elapsed_days" in df_clean.columns

    assert df_clean.loc[0, "discharge_date_elapsed_days"] == 7.0
    assert df_clean.loc[0, "icu_event_date_elapsed_days"] == 2.0
    assert df_clean.loc[1, "discharge_date_elapsed_days"] == 14.0
    assert df_clean.loc[1, "icu_event_date_elapsed_days"] == 4.0
