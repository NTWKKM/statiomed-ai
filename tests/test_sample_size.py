"""
tests/test_sample_size.py - Unit Tests for Sample Size & Power Calculations
"""

from agent.tools.tool_sample_size import SampleSizeTool

def test_sample_size_two_proportions():
    res = SampleSizeTool.calculate_two_proportions(
        p1=0.20,
        p2=0.10,
        power=0.80,
        alpha=0.05,
        dropout_rate=0.15
    )

    assert res["n_total_raw"] > 0
    assert res["n_total_adjusted"] > res["n_total_raw"]
    assert "A total sample size of" in res["justification_text"]

def test_sample_size_survival_logrank():
    res = SampleSizeTool.calculate_survival_logrank(
        hr=0.70,
        p_event=0.30,
        power=0.80,
        alpha=0.05,
        dropout_rate=0.15
    )

    assert res["required_events"] > 0
    assert res["n_total_adjusted"] > res["required_events"]
    assert "To detect a hazard ratio of 0.70" in res["justification_text"]
