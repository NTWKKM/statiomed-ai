"""
tests/test_sample_size.py - Unit Tests for Sample Size & Power Calculations
"""

from agent.tools.tool_sample_size import SampleSizeTool


def test_sample_size_two_proportions():
    res = SampleSizeTool.calculate_two_proportions(
        p1=0.20, p2=0.10, power=0.80, alpha=0.05, dropout_rate=0.15
    )

    assert res["n_total_raw"] > 0
    assert res["n_total_adjusted"] > res["n_total_raw"]
    assert "A total sample size of" in res["justification_text"]


def test_sample_size_survival_logrank():
    res = SampleSizeTool.calculate_survival_logrank(
        hr=0.70, p_event=0.30, power=0.80, alpha=0.05, dropout_rate=0.15
    )

    assert res["required_events"] > 0
    assert res["n_total_adjusted"] > res["required_events"]
    assert "To detect a hazard ratio of 0.70" in res["justification_text"]


def test_sample_size_two_means():
    res = SampleSizeTool.calculate_two_means(
        mean1=120.0,
        mean2=110.0,
        sd1=15.0,
        sd2=15.0,
        power=0.80,
        alpha=0.05,
        dropout_rate=0.15,
    )

    assert res["n_total_raw"] > 0
    assert res["n_total_adjusted"] > res["n_total_raw"]
    assert "Two Independent Means" in res["test_type"]


def test_sample_size_tool_forward_smolagents():
    tool = SampleSizeTool()
    assert tool.name == "sample_size_calculator"

    # Test two_proportions forward
    out_prop = tool.forward(test_type="two_proportions", p1=0.25, p2=0.15)
    assert "Sample Size Result: Two Independent Proportions" in out_prop

    # Test survival_logrank forward
    out_surv = tool.forward(test_type="survival_logrank", hr=0.75, p_event=0.25)
    assert "Sample Size Result: Time-to-Event" in out_surv

    # Test two_means forward
    out_means = tool.forward(
        test_type="two_means", mean1=130.0, mean2=120.0, sd1=15.0, sd2=15.0
    )
    assert "Sample Size Result: Two Independent Means" in out_means
