"""
tests/test_agent_runner.py - Unit Tests for Two-Tier Model Router & Clinical Agent
"""

import os

from agent.agent_runner import (
    CLINICAL_TECH_LEAD_SYSTEM_PROMPT,
    create_clinical_agent,
    execute_agent_turn,
    get_model,
)


def test_get_model_tiers():
    # Test Tier A (ZeroGPU Local)
    model_zerogpu = get_model("zerogpu-local")
    assert model_zerogpu.model_id == os.getenv(
        "LOCAL_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct-AWQ"
    )

    # Test Tier B (Inference Providers)
    model_providers = get_model("inference-providers")
    assert model_providers.model_id == os.getenv(
        "PROVIDER_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"
    )


def test_create_clinical_agent():
    agent = create_clinical_agent(backend="zerogpu-local")
    assert agent is not None
    assert hasattr(agent, "tools")
    assert "pubmed_evidence_search" in agent.tools
    assert "sample_size_calculator" in agent.tools
    assert "synthetic_cohort_generator" in agent.tools
    assert "Zero-PHI" in CLINICAL_TECH_LEAD_SYSTEM_PROMPT


def test_execute_agent_turn_deterministic_fallback():
    agent = create_clinical_agent()
    response = execute_agent_turn(
        agent, "Calculate sample size for mortality trial with 25% vs 15%"
    )
    assert "Sample Size Result" in response or "Clinical AI Co-Pilot" in response


def test_diagnostic_accuracy_derives_from_active_dataset():
    import pandas as pd

    # Dataset 1: 80 TP, 20 FP, 10 FN, 90 TN
    df1 = pd.DataFrame(
        {
            "pocus_scan": [1] * 100 + [0] * 100,
            "gold_standard": [1] * 80 + [0] * 20 + [1] * 10 + [0] * 90,
        }
    )
    # Dataset 2: 50 TP, 50 FP, 40 FN, 60 TN
    df2 = pd.DataFrame(
        {
            "pocus_scan": [1] * 100 + [0] * 100,
            "gold_standard": [1] * 50 + [0] * 50 + [1] * 40 + [0] * 60,
        }
    )

    current_df = df1
    agent = create_clinical_agent(state_df_provider=lambda: current_df)

    res1 = execute_agent_turn(agent, "Run diagnostic accuracy evaluation for pocus")
    assert "TP=80" in res1 or "80.0%" in res1

    # Switch active dataset to df2
    current_df = df2
    res2 = execute_agent_turn(agent, "Run diagnostic accuracy evaluation for pocus")
    assert "TP=50" in res2 or "50.0%" in res2
    assert res1 != res2

    # Without active dataset, it requests dataset or 4 counts
    agent_no_df = create_clinical_agent(state_df_provider=lambda: None)
    res_no_df = execute_agent_turn(agent_no_df, "Run diagnostic accuracy evaluation")
    assert "No active dataset loaded" in res_no_df or "tp, fp, fn, tn" in res_no_df


def test_tool_routing_non_canonical_columns():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df_non_canonical = pd.DataFrame(
        {
            "follow_up_days": [10, 20, 30, 40, 50] * 20,
            "death_status": [1, 0, 1, 0, 1] * 20,
            "study_arm": [1, 1, 0, 0, 1] * 20,
            "aki_endpoint": [0, 1, 0, 1, 0] * 20,
            "sbp_mmhg": rng.normal(130, 15, 100),
            "age_years": rng.normal(60, 10, 100),
            "bmi_value": rng.normal(25, 4, 100),
        }
    )

    agent = create_clinical_agent(state_df_provider=lambda: df_non_canonical)

    # 1. Survival routing with non-canonical columns: assert exact columns & appraisal
    res_surv = execute_agent_turn(agent, "Run survival analysis kaplan meier")
    assert "Survival Analysis" in res_surv
    assert "`follow_up_days`" in res_surv
    assert "`death_status`" in res_surv
    assert "Automated Clinical Critique & Appraisal" in res_surv

    # 2. Logistic routing with non-canonical columns: assert exact outcome column & appraisal
    res_logit = execute_agent_turn(agent, "Run multivariable logistic regression")
    assert "Logistic Regression" in res_logit
    assert "`aki_endpoint`" in res_logit
    assert "Automated Clinical Critique & Appraisal" in res_logit

    # 3. RCT routing with non-canonical columns: assert exact treatment & outcome & appraisal
    res_rct = execute_agent_turn(agent, "Analyze randomized trial consort")
    assert "Randomized Controlled Trial" in res_rct
    assert "`study_arm`" in res_rct
    assert "`aki_endpoint`" in res_rct
    assert "Automated Clinical Critique & Appraisal" in res_rct

    # 4. PSM routing with non-canonical columns: assert exact treatment & appraisal
    res_psm = execute_agent_turn(agent, "Run propensity score matching psm")
    assert "Propensity Score Matching" in res_psm
    assert "`study_arm`" in res_psm
    assert "`aki_endpoint`" not in res_psm
    assert "`death_status`" not in res_psm
    assert "`follow_up_days`" not in res_psm
    assert "Automated Clinical Critique & Appraisal" in res_psm

    # 5. Linear regression (OLS) with non-canonical columns: assert exact continuous outcome & appraisal
    res_linear = execute_agent_turn(agent, "Run linear regression ols")
    assert "Linear Regression" in res_linear
    assert "`sbp_mmhg`" in res_linear
    assert "Automated Clinical Critique & Appraisal" in res_linear


def test_cox_routing_passes_non_empty_covariates():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df_cox = pd.DataFrame(
        {
            "follow_up_days": [10, 20, 30, 40, 50] * 20,
            "death_status": [1, 0, 1, 0, 1] * 20,
            "treatment": [1, 1, 0, 0, 1] * 20,
            "age": rng.normal(60, 10, 100),
            "bmi": rng.normal(25, 4, 100),
        }
    )

    agent = create_clinical_agent(state_df_provider=lambda: df_cox)
    res_cox = execute_agent_turn(agent, "Fit Cox proportional hazards regression model")

    assert "Cox Proportional Hazards" in res_cox
    assert "Multivariable Cox Proportional Hazards Model" in res_cox
    # Check that covariates (e.g. age, bmi) were passed and included in model
    assert "`age`" in res_cox or "age" in res_cox
    assert "Automated Clinical Critique & Appraisal" in res_cox


def test_tool_routing_rejects_unresolved_columns_without_positional_fallbacks():
    import pandas as pd

    # Dataset with ambiguous column names that do not match semantic keywords
    df_ambiguous = pd.DataFrame(
        {
            "col_alpha": [1, 2, 3, 4, 5] * 20,
            "col_beta": ["a", "b", "c", "d", "e"] * 20,
            "col_gamma": [10.5, 20.1, 30.2, 40.8, 50.9] * 20,
        }
    )

    agent = create_clinical_agent(state_df_provider=lambda: df_ambiguous)

    # 1. Survival unresolved error
    res_surv = execute_agent_turn(agent, "Run survival analysis")
    assert "Error: Could not resolve valid time duration" in res_surv

    # 2. Logistic unresolved error
    res_logit = execute_agent_turn(agent, "Run logistic regression")
    assert "Error: Could not resolve a valid binary outcome column" in res_logit

    # 3. RCT unresolved error
    res_rct = execute_agent_turn(agent, "Run RCT trial analysis")
    assert "Error: Could not resolve valid treatment and outcome columns" in res_rct

    # 4. Diagnostic unresolved error
    res_diag = execute_agent_turn(agent, "Run diagnostic accuracy evaluation")
    assert (
        "Error: Could not resolve valid index test and reference standard columns"
        in res_diag
    )

    # 5. PSM unresolved error
    res_psm = execute_agent_turn(agent, "Run propensity score matching psm")
    assert (
        "Error: Could not resolve a valid binary treatment indicator column" in res_psm
    )

    # 6. Linear unresolved error
    res_linear = execute_agent_turn(agent, "Run linear regression ols")
    assert (
        "Error: Could not resolve a valid continuous/numeric outcome column"
        in res_linear
    )

    # --- Validation Gate Failures ---

    # Case A: Constant treatment (nunique=1) -> rejected by RCT and PSM
    df_const_treat = pd.DataFrame(
        {
            "study_arm": [1] * 20,
            "outcome": [0, 1] * 10,
            "age": [50, 60] * 10,
        }
    )
    agent_const = create_clinical_agent(state_df_provider=lambda: df_const_treat)
    res_rct_const = execute_agent_turn(agent_const, "Analyze randomized trial consort")
    assert (
        "Error: Treatment and outcome columns in RCT analysis must be binary."
        in res_rct_const
    )

    res_psm_const = execute_agent_turn(agent_const, "Run propensity score matching psm")
    assert (
        "Error: Could not resolve a valid binary treatment indicator column in active dataset for PSM."
        in res_psm_const
    )

    # Case B: Non-numeric / string event labels ("Alive"/"Dead") -> rejected by Survival
    df_str_event = pd.DataFrame(
        {
            "follow_up_days": [10, 20] * 10,
            "death_status": ["Alive", "Dead"] * 10,
        }
    )
    agent_str_event = create_clinical_agent(state_df_provider=lambda: df_str_event)
    res_surv_str = execute_agent_turn(
        agent_str_event, "Run survival analysis kaplan meier"
    )
    assert (
        "Error: Could not resolve valid time duration and binary event indicator columns"
        in res_surv_str
    )

    # Case C: Multiclass outcome (>2 distinct values) -> rejected by Logistic
    df_multi_outcome = pd.DataFrame(
        {
            "outcome": [0, 1, 2] * 10,
            "treatment": [0, 1] * 15,
            "age": [50, 60, 70] * 10,
        }
    )
    agent_multi = create_clinical_agent(state_df_provider=lambda: df_multi_outcome)
    res_logit_multi = execute_agent_turn(
        agent_multi, "Run multivariable logistic regression"
    )
    assert "Error: Could not resolve a valid binary outcome column" in res_logit_multi

    # Case D: Non-numeric outcome -> rejected by Linear regression
    df_str_outcome = pd.DataFrame(
        {
            "outcome": ["high", "low"] * 10,
            "age": [50, 60] * 10,
        }
    )
    agent_str_outcome = create_clinical_agent(state_df_provider=lambda: df_str_outcome)
    res_linear_str = execute_agent_turn(agent_str_outcome, "Run linear regression ols")
    assert (
        "Error: Could not resolve a valid continuous/numeric outcome column"
        in res_linear_str
    )


def test_psm_covariates_exclude_endpoints_and_outcomes():
    import pandas as pd

    from agent.agent_runner import _resolve_psm_columns

    df = pd.DataFrame(
        {
            "treatment": [1, 0, 1, 0] * 10,
            "patient_id": [f"P_{i}" for i in range(40)],
            "age_years": [50, 60, 55, 65] * 10,
            "baseline_status": [1, 0, 1, 0] * 10,
            "smoking_status": [0, 1, 0, 1] * 10,
            "creatinine": [1.1, 0.9, 1.4, 0.8] * 10,
            "aki_endpoint": [1, 0, 0, 1] * 10,
            "death_status": [0, 1, 0, 0] * 10,
            "follow_up_days": [100, 200, 150, 300] * 10,
        }
    )

    resolved, err = _resolve_psm_columns(df, "Run propensity score matching")
    assert err is None
    assert resolved is not None
    assert resolved["treatment_col"] == "treatment"
    assert "aki_endpoint" not in resolved["covariate_cols"]
    assert "death_status" not in resolved["covariate_cols"]
    assert "follow_up_days" not in resolved["covariate_cols"]
    assert "patient_id" not in resolved["covariate_cols"]
    assert "age_years" in resolved["covariate_cols"]
    assert "baseline_status" in resolved["covariate_cols"]
    assert "smoking_status" in resolved["covariate_cols"]
    assert set(resolved["covariate_cols"]).issubset(
        {"age_years", "baseline_status", "smoking_status", "creatinine"}
    )
