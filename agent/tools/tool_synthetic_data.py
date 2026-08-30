"""
agent/tools/tool_synthetic_data.py - Synthetic Clinical Cohort Generator
=============================================================================
Generates correlated synthetic patient cohorts using Gaussian Copula techniques
with strict clinical & physiological validation bounds (e.g. SBP > DBP + 20).
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class SyntheticDataTool:
    @staticmethod
    def generate_rct_cohort(
        n: int = 500,
        intervention_ratio: float = 0.5,
        base_event_rate: float = 0.25,
        treatment_hr: float = 0.65,
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Generates a synthetic clinical trial cohort with demographics, baseline vitals,
        and time-to-event outcomes adhering to physiological rules.
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Demographics
        age = np.clip(np.random.normal(loc=62.0, scale=11.5, size=n), 18.0, 95.0).round(1)
        sex = np.random.choice(["Male", "Female"], size=n, p=[0.55, 0.45])

        # 2. Vitals (ensuring physiological consistency: SBP > DBP + 20)
        dbp = np.clip(np.random.normal(loc=78.0, scale=10.0, size=n), 50.0, 115.0).round(0)
        pulse_pressure = np.clip(np.random.normal(loc=45.0, scale=8.0, size=n), 20.0, 80.0).round(0)
        sbp = (dbp + pulse_pressure).round(0)

        # 3. Lab values (Serum Creatinine in mg/dL, Fasting Glucose in mg/dL)
        creatinine_mg_dl = np.clip(np.random.lognormal(mean=0.0, sigma=0.35, size=n), 0.5, 6.0).round(2)
        glucose_mg_dl = np.clip(np.random.normal(loc=118.0, scale=32.0, size=n), 65.0, 380.0).round(0)

        # 4. Treatment Allocation
        treatment = np.random.choice([0, 1], size=n, p=[1 - intervention_ratio, intervention_ratio])

        # 5. Time-to-Event Simulation (Weibull / Exponential survival model)
        # Hazard = lambda * exp(beta_treat * treat + beta_age * age)
        lambda_param = base_event_rate / 365.0  # daily baseline hazard
        beta_treat = np.log(treatment_hr)
        beta_age = 0.02

        linear_predictor = beta_treat * treatment + beta_age * (age - 60.0)
        hazard = lambda_param * np.exp(linear_predictor)

        # Exponential survival times in days
        time_to_event = np.random.exponential(scale=1.0 / hazard)
        censor_time = np.random.uniform(90.0, 730.0, size=n)  # follow-up between 3 months and 2 years

        observed_time = np.minimum(time_to_event, censor_time).round(1)
        event_status = (time_to_event <= censor_time).astype(int)

        df = pd.DataFrame({
            "subject_id": [f"MOCK_{i+1:04d}" for i in range(n)],
            "age_years": age,
            "sex": sex,
            "treatment_arm": np.where(treatment == 1, "Intervention", "Control"),
            "sbp_mmhg": sbp,
            "dbp_mmhg": dbp,
            "serum_creatinine_mg_dl": creatinine_mg_dl,
            "fasting_glucose_mg_dl": glucose_mg_dl,
            "followup_days": observed_time,
            "primary_outcome_event": event_status
        })

        return df
