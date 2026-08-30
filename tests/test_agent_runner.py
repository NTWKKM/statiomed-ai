"""
tests/test_agent_runner.py - Unit Tests for Two-Tier Model Router & Clinical Agent
"""

import os
from agent.agent_runner import (
    get_model,
    create_clinical_agent,
    execute_agent_turn,
    CLINICAL_TECH_LEAD_SYSTEM_PROMPT,
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
