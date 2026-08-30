"""
agent/tools/tool_sample_size.py - Sample Size & Power Calculation Tool (smolagents.Tool Compliant)
=============================================================================
Calculates required sample size for clinical trials and observational studies
with explicit power, alpha, allocation ratio, and drop-out compensation.
Adheres to SAMPL guidelines and smolagents v1.x Tool interface.
=============================================================================
"""

import math
from typing import Any, Dict, Optional

try:
    from smolagents import Tool
except ImportError:

    class Tool:
        name: str = ""
        description: str = ""
        inputs: Dict[str, Any] = {}
        output_type: str = "string"

        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            raise NotImplementedError


from utils.sample_size_lib import (
    calculate_sample_size_means,
    calculate_sample_size_proportions,
    calculate_sample_size_survival,
)


class SampleSizeTool(Tool):
    name = "sample_size_calculator"
    description = (
        "Calculates required sample sizes and statistical power for clinical studies "
        "including two independent proportions (Z-test/Chi-square), continuous means (t-test), "
        "and time-to-event log-rank tests (Schoenfeld formula) with dropout compensation."
    )
    inputs = {
        "test_type": {
            "type": "string",
            "description": "Type of statistical test: 'two_proportions', 'survival_logrank', or 'two_means'.",
            "nullable": True,
        },
        "p1": {
            "type": "number",
            "description": "Anticipated event rate in control group (for two_proportions, e.g., 0.25).",
            "nullable": True,
        },
        "p2": {
            "type": "number",
            "description": "Anticipated event rate in intervention group (for two_proportions, e.g., 0.15).",
            "nullable": True,
        },
        "hr": {
            "type": "number",
            "description": "Expected Hazard Ratio (for survival_logrank, e.g., 0.70).",
            "nullable": True,
        },
        "p_event": {
            "type": "number",
            "description": "Expected overall event rate during study period (for survival_logrank, e.g., 0.30).",
            "nullable": True,
        },
        "mean1": {
            "type": "number",
            "description": "Expected mean in group 1 (for two_means, e.g., 120.0).",
            "nullable": True,
        },
        "mean2": {
            "type": "number",
            "description": "Expected mean in group 2 (for two_means, e.g., 110.0).",
            "nullable": True,
        },
        "sd1": {
            "type": "number",
            "description": "Standard deviation in group 1 (for two_means, e.g., 15.0).",
            "nullable": True,
        },
        "sd2": {
            "type": "number",
            "description": "Standard deviation in group 2 (for two_means, e.g., 15.0).",
            "nullable": True,
        },
        "power": {
            "type": "number",
            "description": "Target statistical power (default 0.80 = 80%).",
            "nullable": True,
        },
        "alpha": {
            "type": "number",
            "description": "Two-sided type I error rate (default 0.05).",
            "nullable": True,
        },
        "ratio": {
            "type": "number",
            "description": "Allocation ratio n2/n1 (default 1.0 = 1:1 equal allocation).",
            "nullable": True,
        },
        "dropout_rate": {
            "type": "number",
            "description": "Anticipated loss to follow-up / drop-out rate (default 0.15 = 15%).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_two_proportions(
        p1: float,
        p2: float,
        power: float = 0.80,
        alpha: float = 0.05,
        ratio: float = 1.0,
        dropout_rate: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Calculates sample size for comparing two independent proportions (Fleiss formula).
        """
        res = calculate_sample_size_proportions(
            power=power, ratio=ratio, p1=p1, p2=p2, alpha=alpha
        )
        n1 = int(res.get("n1", 0))
        n2 = int(res.get("n2", 0))
        n_total = int(res.get("total", n1 + n2))

        # Adjust for drop-out
        n_total_adj = math.ceil(n_total / (1.0 - dropout_rate))
        n1_adj = math.ceil(n_total_adj / (1.0 + ratio))
        n2_adj = n_total_adj - n1_adj

        return {
            "test_type": "Two Independent Proportions (Z-test / Chi-square)",
            "p1_control": p1,
            "p2_intervention": p2,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": ratio,
            "n_control_raw": n1,
            "n_intervention_raw": n2,
            "n_total_raw": n_total,
            "dropout_rate_assumed": dropout_rate,
            "n_control_adjusted": n1_adj,
            "n_intervention_adjusted": n2_adj,
            "n_total_adjusted": n_total_adj,
            "justification_text": (
                f"A total sample size of {n_total_adj} participants ({n1_adj} in control, {n2_adj} in intervention) "
                f"provides {int(power * 100)}% power to detect an absolute difference between {p1:.1%} and {p2:.1%} "
                f"at a two-sided significance level of {alpha} (alpha), assuming a {int(dropout_rate * 100)}% loss to follow-up."
            ),
        }

    @staticmethod
    def calculate_survival_logrank(
        hr: float,
        p_event: float,
        power: float = 0.80,
        alpha: float = 0.05,
        ratio: float = 1.0,
        dropout_rate: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Calculates required events and total sample size for time-to-event log-rank test (Schoenfeld formula).
        """
        res = calculate_sample_size_survival(
            power=power, ratio=ratio, h0=hr, h1=1.0, alpha=alpha, mode="hr"
        )
        events_needed = int(math.ceil(float(res.get("total_events", 0))))
        if p_event > 0:
            n_total = int(math.ceil(events_needed / p_event))
        else:
            n_total = events_needed
        n_total_adj = math.ceil(n_total / (1.0 - dropout_rate))

        return {
            "test_type": "Time-to-Event (Log-Rank / Cox PH)",
            "hazard_ratio": hr,
            "event_rate": p_event,
            "power": power,
            "alpha": alpha,
            "required_events": events_needed,
            "n_total_raw": n_total,
            "n_total_adjusted": n_total_adj,
            "justification_text": (
                f"To detect a hazard ratio of {hr:.2f} with {int(power * 100)}% power and two-sided alpha of {alpha}, "
                f"a total of {events_needed} endpoint events are required. Based on an anticipated event rate of {p_event:.1%} "
                f"and {int(dropout_rate * 100)}% drop-out, a total of {n_total_adj} patients must be enrolled."
            ),
        }

    @staticmethod
    def calculate_two_means(
        mean1: float,
        mean2: float,
        sd1: float,
        sd2: float,
        power: float = 0.80,
        alpha: float = 0.05,
        ratio: float = 1.0,
        dropout_rate: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Calculates sample size for comparing two independent continuous means (two-sample t-test).
        """
        res = calculate_sample_size_means(
            power=power,
            ratio=ratio,
            mean1=mean1,
            mean2=mean2,
            sd1=sd1,
            sd2=sd2,
            alpha=alpha,
        )
        n1 = int(res.get("n1", 0))
        n2 = int(res.get("n2", 0))
        n_total = int(res.get("total", n1 + n2))

        n_total_adj = math.ceil(n_total / (1.0 - dropout_rate))
        n1_adj = math.ceil(n_total_adj / (1.0 + ratio))
        n2_adj = n_total_adj - n1_adj

        return {
            "test_type": "Two Independent Means (Two-Sample t-test)",
            "mean1": mean1,
            "mean2": mean2,
            "sd1": sd1,
            "sd2": sd2,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": ratio,
            "n_group1_raw": n1,
            "n_group2_raw": n2,
            "n_total_raw": n_total,
            "dropout_rate_assumed": dropout_rate,
            "n_group1_adjusted": n1_adj,
            "n_group2_adjusted": n2_adj,
            "n_total_adjusted": n_total_adj,
            "justification_text": (
                f"A sample size of {n_total_adj} ({n1_adj} in group 1, {n2_adj} in group 2) "
                f"achieves {int(power * 100)}% power to detect a difference of {abs(mean1 - mean2):.2f} "
                f"(Group 1 mean={mean1:.2f} vs Group 2 mean={mean2:.2f}) at a two-sided alpha={alpha}, "
                f"accounting for {int(dropout_rate * 100)}% drop-out."
            ),
        }

    def forward(
        self,
        test_type: str = "two_proportions",
        p1: Optional[float] = 0.25,
        p2: Optional[float] = 0.15,
        hr: Optional[float] = 0.70,
        p_event: Optional[float] = 0.30,
        mean1: Optional[float] = 120.0,
        mean2: Optional[float] = 110.0,
        sd1: Optional[float] = 15.0,
        sd2: Optional[float] = 15.0,
        power: Optional[float] = 0.80,
        alpha: Optional[float] = 0.05,
        ratio: Optional[float] = 1.0,
        dropout_rate: Optional[float] = 0.15,
    ) -> str:
        """
        smolagents Tool forward execution method.
        """
        power = power or 0.80
        alpha = alpha or 0.05
        ratio = ratio or 1.0
        dropout_rate = dropout_rate or 0.15

        if test_type in ["survival_logrank", "survival", "logrank"]:
            hr = hr or 0.70
            p_event = p_event or 0.30
            res = self.calculate_survival_logrank(
                hr=hr,
                p_event=p_event,
                power=power,
                alpha=alpha,
                ratio=ratio,
                dropout_rate=dropout_rate,
            )
            return (
                f"**Sample Size Result: {res['test_type']}**\n"
                f"- Hazard Ratio: {res['hazard_ratio']:.2f}\n"
                f"- Event Rate: {res['event_rate']:.1%}\n"
                f"- Required Endpoint Events: **{res['required_events']}**\n"
                f"- Total Sample Size (with {int(dropout_rate * 100)}% drop-out): **{res['n_total_adjusted']}** patients\n\n"
                f"*Justification (SAMPL):* {res['justification_text']}"
            )
        elif test_type in ["two_means", "means", "t_test"]:
            mean1 = mean1 or 120.0
            mean2 = mean2 or 110.0
            sd1 = sd1 or 15.0
            sd2 = sd2 or 15.0
            res = self.calculate_two_means(
                mean1=mean1,
                mean2=mean2,
                sd1=sd1,
                sd2=sd2,
                power=power,
                alpha=alpha,
                ratio=ratio,
                dropout_rate=dropout_rate,
            )
            return (
                f"**Sample Size Result: {res['test_type']}**\n"
                f"- Mean Difference: {abs(mean1 - mean2):.2f}\n"
                f"- Group 1 (N): **{res['n_group1_adjusted']}**, Group 2 (N): **{res['n_group2_adjusted']}**\n"
                f"- Total Sample Size (with {int(dropout_rate * 100)}% drop-out): **{res['n_total_adjusted']}** participants\n\n"
                f"*Justification (SAMPL):* {res['justification_text']}"
            )
        else:
            p1 = p1 or 0.25
            p2 = p2 or 0.15
            res = self.calculate_two_proportions(
                p1=p1,
                p2=p2,
                power=power,
                alpha=alpha,
                ratio=ratio,
                dropout_rate=dropout_rate,
            )
            return (
                f"**Sample Size Result: {res['test_type']}**\n"
                f"- Control Rate ($p_1$): {res['p1_control']:.1%}, Intervention Rate ($p_2$): {res['p2_intervention']:.1%}\n"
                f"- Control (N): **{res['n_control_adjusted']}**, Intervention (N): **{res['n_intervention_adjusted']}**\n"
                f"- Total Target Enrollment: **{res['n_total_adjusted']}** participants\n\n"
                f"*Justification (SAMPL):* {res['justification_text']}"
            )
