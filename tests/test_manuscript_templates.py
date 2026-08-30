"""
tests/test_manuscript_templates.py - Unit Tests for Deterministic Jinja2 Manuscript Engine
"""

from agent.manuscript_engine import (
    ManuscriptEngine,
    format_sampl_p_value,
    format_sampl_ci,
    format_sampl_pct,
)


def test_sampl_formatters():
    # P-value formatting
    assert format_sampl_p_value(0.0002) == "< 0.001"
    assert format_sampl_p_value(0.0423) == "= 0.042"
    assert format_sampl_p_value(0.354) == "= 0.35"
    assert format_sampl_p_value(None) == "—"

    # 95% CI formatting
    assert format_sampl_ci(0.552, 0.891, decimals=2) == "95% CI: 0.55 to 0.89"

    # Percentage formatting
    assert format_sampl_pct(45, 100) == "45.0%"
    assert format_sampl_pct(12, 50) == "24%"


def test_render_sap():
    context = {
        "study_title": "EMPEROR-Preserved SGLT2 Trial",
        "primary_objective": "Determine if Empagliflozin reduces CV death or HF hospitalization",
        "n_total": 5988,
        "n_control": 2991,
        "n_intervention": 2997,
        "power_pct": "90",
        "alpha": "0.05",
    }
    sap_md = ManuscriptEngine.render_sap(context)
    assert "Statistical Analysis Plan (SAP)" in sap_md
    assert "EMPEROR-Preserved" in sap_md
    assert "Austin (2009)" in sap_md
    assert "Efron's tie-handling" in sap_md


def test_render_methods_rct():
    context = {
        "n_intervention": 250,
        "n_control": 250,
        "m_imputations": 20,
    }
    methods = ManuscriptEngine.render_methods("rct", context)
    assert "CONSORT Compliant" in methods
    assert "Austin (2009)" in methods
    assert "Barnard-Rubin (1999)" in methods


def test_render_results_survival():
    context = {
        "median_followup": "730",
        "n_intervention": 250,
        "events_intervention": 35,
        "pct_events_intervention": "14.0%",
        "n_control": 250,
        "events_control": 60,
        "pct_events_control": "24.0%",
        "hr": "0.65",
        "hr_ci_lower": "0.48",
        "hr_ci_upper": "0.88",
        "hr_p_str": "= 0.005",
        "c_index": "0.79",
    }
    results = ManuscriptEngine.render_results("survival", context)
    assert "Results: Time-to-Event Survival Analysis" in results
    assert "Hazard Ratio [HR] = 0.65" in results
    assert "Harrell's C-index" in results
