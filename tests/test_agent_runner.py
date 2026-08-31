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

    df_non_canonical = pd.DataFrame(
        {
            "follow_up_days": [10, 20, 30, 40, 50] * 20,
            "death_status": [1, 0, 1, 0, 1] * 20,
            "study_arm": [1, 1, 0, 0, 1] * 20,
            "aki_endpoint": [0, 1, 0, 1, 0] * 20,
            "sbp_mmhg": np.random.normal(130, 15, 100),
            "age_years": np.random.normal(60, 10, 100),
            "bmi_value": np.random.normal(25, 4, 100),
        }
    )

    agent = create_clinical_agent(state_df_provider=lambda: df_non_canonical)

    # 1. Survival routing with non-canonical columns
    res_surv = execute_agent_turn(agent, "Run survival analysis kaplan meier")
    assert "Survival Analysis" in res_surv
    assert "follow_up_days" in res_surv or "death_status" in res_surv

    # 2. Logistic routing with non-canonical columns
    res_logit = execute_agent_turn(agent, "Run multivariable logistic regression")
    assert "Logistic Regression" in res_logit
    assert "aki_endpoint" in res_logit or "Odds Ratio" in res_logit

    # 3. RCT routing with non-canonical columns
    res_rct = execute_agent_turn(agent, "Analyze randomized trial consort")
    assert "Randomized Controlled Trial" in res_rct
    assert "study_arm" in res_rct or "Relative Risk" in res_rct

    # 4. PSM routing with non-canonical columns
    res_psm = execute_agent_turn(agent, "Run propensity score matching psm")
    assert "Propensity Score Matching" in res_psm

    # 5. Linear regression (OLS) with non-canonical columns
    res_linear = execute_agent_turn(agent, "Run linear regression ols")
    assert "Linear Regression" in res_linear
