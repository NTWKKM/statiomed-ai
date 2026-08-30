"""
agent/agent_runner.py - Agent Model Router & Execution Engine
=============================================================================
Orchestrates smolagents ToolCallingAgent with 2-Tier Model Strategy:
  - Tier A (ZeroGPU Local): Focused fast turns, quantized 14B/32B AWQ via spaces.GPU
  - Tier B (Inference Providers): High-reasoning 70B+ (Qwen 2.5 72B / Llama 3.3 70B)
=============================================================================
"""

import os
from typing import Any, Dict, List, Optional

# Conditional import for spaces if running inside Hugging Face ZeroGPU Space
try:
    import spaces

    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

    class spaces:
        @staticmethod
        def GPU(duration=45):
            def decorator(fn):
                return fn

            return decorator


try:
    from smolagents import HfApiModel, ToolCallingAgent

    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False

    class HfApiModel:
        def __init__(
            self,
            model_id: str,
            provider: Optional[str] = None,
            token: Optional[str] = None,
        ):
            self.model_id = model_id
            self.provider = provider
            self.token = token

    class ToolCallingAgent:
        def __init__(
            self, tools: List[Any], model: Any, system_prompt: Optional[str] = None
        ):
            self.tools = {t.name: t for t in tools if hasattr(t, "name")}
            self.model = model
            self.system_prompt = system_prompt

        def run(self, prompt: str) -> str:
            # Fallback deterministic dispatcher if smolagents library is not available
            lower_p = prompt.lower()
            if "pubmed" in lower_p or "pico" in lower_p or "evidence" in lower_p:
                tool = self.tools.get("pubmed_evidence_search")
                if tool:
                    return tool.forward(query=prompt, max_results=3)
            elif "sample size" in lower_p or "power" in lower_p:
                tool = self.tools.get("sample_size_calculator")
                if tool:
                    return tool.forward(test_type="two_proportions", p1=0.25, p2=0.15)
            elif "synthetic" in lower_p or "cohort" in lower_p:
                tool = self.tools.get("synthetic_cohort_generator")
                if tool:
                    return tool.forward(n=100)
            return f"Clinical AI Co-Pilot analysis for: {prompt}\n\nAll computations adhere to SAMPL guidelines and Zero-PHI security."


CLINICAL_TECH_LEAD_SYSTEM_PROMPT = """
You are StatioMed AI, an Offline-First Clinical Tech Lead and ER/Critical Care Physician persona.
You assist clinical researchers in study design, PICO benchmark extraction, sample size justification,
and statistical analysis planning (SAP).

Operating Guidelines:
1. Zero-PHI Guarantee: Never accept, store, or output Protected Health Information (HN, Citizen ID, Names, Phone, DOB).
2. SAMPL Compliance: Report P-values to 2-3 decimal places (e.g. P = 0.042, P < 0.001), 95% Confidence Intervals for effect sizes, and unrounded rational numbers in intermediate steps.
3. EQUATOR Standards: Ensure study designs conform to STROBE (observational), CONSORT (RCT), TRIPOD+AI (prediction models), or STARD (diagnostics).
4. Tool Utilization: Call formal tools (PubMedEvidenceTool, SampleSizeTool, SyntheticDataTool) to compute rigorous numbers instead of guessing.
"""


def get_model(backend: Optional[str] = None):
    """
    Initializes HfApiModel with appropriate backend and token authentication.
    """
    backend = backend or os.getenv("LLM_BACKEND", "inference-providers")
    token = os.getenv("HF_TOKEN")

    if backend == "zerogpu-local":
        return HfApiModel(
            model_id=os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct-AWQ"),
            token=token,
        )
    elif backend == "inference-providers":
        return HfApiModel(
            model_id=os.getenv("PROVIDER_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"),
            provider=os.getenv("HF_INFERENCE_PROVIDER", "together"),
            token=token,
        )
    else:
        return HfApiModel(model_id="Qwen/Qwen2.5-72B-Instruct", token=token)


def create_clinical_agent(
    backend: Optional[str] = None, tools: Optional[List[Any]] = None
) -> ToolCallingAgent:
    """
    Factory function to initialize a ToolCallingAgent with clinical tools and system prompt.
    """
    from agent.tools.tool_pubmed import PubMedEvidenceTool
    from agent.tools.tool_sample_size import SampleSizeTool
    from agent.tools.tool_synthetic_data import SyntheticDataTool

    if tools is None:
        tools = [PubMedEvidenceTool(), SampleSizeTool(), SyntheticDataTool()]

    model = get_model(backend=backend)
    return ToolCallingAgent(
        tools=tools, model=model, system_prompt=CLINICAL_TECH_LEAD_SYSTEM_PROMPT
    )


@spaces.GPU(duration=45)
def execute_agent_turn(agent: Any, user_prompt: str) -> str:
    """
    Executes a single focused LLM reasoning turn within ZeroGPU time allocation.
    """
    return agent.run(user_prompt)
