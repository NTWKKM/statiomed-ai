"""
agent/tools/tool_sample_size.py - Sample Size & Power Calculation Tool
=============================================================================
Calculates required sample size for clinical trials and observational studies
with explicit power, alpha, allocation ratio, and drop-out compensation.
=============================================================================
"""

import math
from typing import Dict, Any
from utils.sample_size_lib import (
    calculate_sample_size_proportions,
    calculate_sample_size_means,
    calculate_sample_size_survival,
)

class SampleSizeTool:
    @staticmethod
    def calculate_two_proportions(
        p1: float,
        p2: float,
        power: float = 0.80,
        alpha: float = 0.05,
        ratio: float = 1.0,
        dropout_rate: float = 0.15
    ) -> Dict[str, Any]:
        """
        Calculates sample size for comparing two independent proportions (e.g. mortality, response rate).
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
                f"provides {int(power*100)}% power to detect an absolute difference between {p1:.1%} and {p2:.1%} "
                f"at a two-sided significance level of {alpha} (alpha), assuming a {int(dropout_rate*100)}% loss to follow-up."
            )
        }

    @staticmethod
    def calculate_survival_logrank(
        hr: float,
        p_event: float,
        power: float = 0.80,
        alpha: float = 0.05,
        ratio: float = 1.0,
        dropout_rate: float = 0.15
    ) -> Dict[str, Any]:
        """
        Calculates required events and total sample size for time-to-event log-rank test.
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
                f"To detect a hazard ratio of {hr:.2f} with {int(power*100)}% power and two-sided alpha of {alpha}, "
                f"a total of {events_needed} endpoint events are required. Based on an anticipated event rate of {p_event:.1%} "
                f"and {int(dropout_rate*100)}% drop-out, a total of {n_total_adj} patients must be enrolled."
            )
        }
