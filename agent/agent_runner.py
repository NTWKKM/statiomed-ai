"""
agent/agent_runner.py - Agent Model Router & Execution Engine
=============================================================================
Orchestrates smolagents ToolCallingAgent with 2-Tier Model Strategy:
  - Tier A: ZeroGPU local inference (quantized 14-32B AWQ via spaces.GPU)
  - Tier B: Cloud Inference Providers (70B+ models, opt-in)
=============================================================================
"""

import os
from typing import Any, Dict, Optional

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

from smolagents import HfApiModel, ToolCallingAgent
from agent.tools.tool_pubmed import PubMedEvidenceTool
from agent.tools.tool_sample_size import SampleSizeTool
from agent.tools.tool_synthetic_data import SyntheticDataTool

def get_model(backend: Optional[str] = None):
    backend = backend or os.getenv("LLM_BACKEND", "inference-providers")
    token = os.getenv("HF_TOKEN")

    if backend == "zerogpu-local":
        return HfApiModel(
            model_id=os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
            token=token
        )
    elif backend == "inference-providers":
        return HfApiModel(
            model_id=os.getenv("PROVIDER_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"),
            provider=os.getenv("HF_INFERENCE_PROVIDER", "together"),
            token=token
        )
    else:
        # Fallback to standard HF model
        return HfApiModel(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            token=token
        )

@spaces.GPU(duration=45)
def execute_agent_turn(agent: Any, user_prompt: str) -> str:
    """
    Executes a single focused LLM reasoning turn within ZeroGPU time allocation.
    """
    return agent.run(user_prompt)
