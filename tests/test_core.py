"""
tests/test_core.py - Unit Tests for StatioMed AI Core Layer
===========================================================
Verifies AppState lifecycle, deep methods, clinical formatting helpers,
constants validation, and SAMPL P-value & CI renderers.
===========================================================
"""

import pandas as pd

from core import (
    APP_NAME,
    CLINICAL_UNITS,
    EQUATOR_GUIDELINES,
    PHI_RESTRICTED_COLUMNS,
    AppState,
    format_sampl_ci,
    format_sampl_p_value,
    get_color_palette,
    render_card_html,
    select_variable_by_keyword,
)


def test_app_state_lifecycle():
    """Verify AppState dataset management, column inference, and reset."""
    state = AppState()
    assert state.has_data() is False
    assert state.has_matched_data() is False
    assert state.get_columns() == []

    # Load mock dataframe
    df = pd.DataFrame(
        {
            "age": [55, 62, 70],
            "sex": ["M", "F", "M"],
            "treatment": [1, 0, 1],
            "sbp": [130.0, 145.0, 120.0],
            "status": ["active", "control", "active"],
        }
    )
    state.df = df
    state.file_name = "test_cohort.csv"

    assert state.has_data() is True
    assert len(state.get_columns()) == 5
    assert set(state.get_numeric_columns()) == {"age", "treatment", "sbp"}
    assert "sex" in state.get_categorical_columns()

    # Summary dict
    summary = state.get_summary_dict()
    assert summary["loaded"] is True
    assert summary["rows"] == 3
    assert summary["cols"] == 5

    # Test matched data
    df_matched = df.iloc[:2].copy()
    state.df_matched = df_matched
    state.is_matched = True

    assert state.has_matched_data() is True
    assert len(state.get_active_dataframe(use_matched=True)) == 2
    assert len(state.get_active_dataframe(use_matched=False)) == 3

    # Reset
    state.reset_dataset()
    assert state.has_data() is False
    assert state.has_matched_data() is False


def test_core_constants():
    """Verify standard units, equator guidelines, and PHI definitions."""
    assert APP_NAME == "StatioMed AI"
    assert "egfr" in CLINICAL_UNITS
    assert CLINICAL_UNITS["egfr"] == "mL/min/1.73m²"
    assert "CONSORT" in EQUATOR_GUIDELINES
    assert "hn" in PHI_RESTRICTED_COLUMNS


def test_sampl_formatting_helpers():
    """Verify P-value and CI formatting per SAMPL standards."""
    assert format_sampl_p_value(0.0004) == "P < 0.001"
    assert format_sampl_p_value(0.0423) == "P = 0.042"
    assert format_sampl_p_value(0.24) == "P = 0.24"
    assert format_sampl_p_value(0.999) == "P > 0.99"
    assert format_sampl_p_value(None) == "N/A"

    ci_str = format_sampl_ci(1.45, 1.12, 1.89)
    assert "1.45 (95% CI: 1.12 to 1.89)" == ci_str


def test_render_card_html():
    """Verify card HTML template generator."""
    card = render_card_html(
        title="Test Card",
        content_html="<p>Analysis Results</p>",
        badge_text="COMPLIANT",
        badge_color="green",
    )
    assert "Test Card" in card
    assert "COMPLIANT" in card
    assert "<p>Analysis Results</p>" in card


def test_color_palette_and_var_selector():
    """Verify color tokens and keyword matching."""
    palette = get_color_palette()
    assert "primary" in palette
    assert "success" in palette

    cols = ["patient_id", "treatment_arm", "overall_survival_months"]
    matched = select_variable_by_keyword(cols, ["treatment", "arm", "group"])
    assert matched == "treatment_arm"
