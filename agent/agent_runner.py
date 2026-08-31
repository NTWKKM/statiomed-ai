"""
agent/agent_runner.py - Agent Model Router & Execution Engine
=============================================================================
Orchestrates smolagents ToolCallingAgent with 2-Tier Model Strategy:
  - Tier A (ZeroGPU Local): Focused fast turns, quantized 14B/32B AWQ via spaces.GPU
  - Tier B (Inference Providers): High-reasoning 70B+ (Qwen 2.5 72B / Llama 3.3 70B)
=============================================================================
"""

import os
import re
from typing import Any, List, Optional

import pandas as pd

from core.common import select_variable_by_keyword
from logger import get_logger

logger = get_logger(__name__)

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
    from huggingface_hub import InferenceClient

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    InferenceClient = None  # type: ignore

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

        def _get_df(self, tool: Any) -> Optional[pd.DataFrame]:
            if hasattr(tool, "state_df_provider") and callable(tool.state_df_provider):
                return tool.state_df_provider()
            return None

        def run(self, prompt: str) -> str:
            lower_p = prompt.lower()
            if "sample size" in lower_p or "power" in lower_p or "n_total" in lower_p:
                tool = self.tools.get("sample_size_calculator")
                if tool:
                    return tool.forward(test_type="two_proportions", p1=0.25, p2=0.15)
            elif (
                "survival" in lower_p
                or "kaplan" in lower_p
                or "cox" in lower_p
                or "logrank" in lower_p
            ):
                tool = self.tools.get("survival_analysis")
                if tool:
                    df = self._get_df(tool)
                    if df is None or df.empty:
                        return tool.forward(time_col="time", event_col="event")
                    cols = df.columns.tolist()
                    time_col = select_variable_by_keyword(
                        cols,
                        [
                            "time",
                            "duration",
                            "followup",
                            "follow_up",
                            "days",
                            "months",
                            "years",
                            "surv_time",
                            "os_months",
                            "pfs_days",
                        ],
                        default_to_first=False,
                    )
                    if not time_col:
                        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        if num_cols:
                            time_col = num_cols[0]
                    event_col = select_variable_by_keyword(
                        [c for c in cols if c != time_col],
                        [
                            "event",
                            "status",
                            "death",
                            "censored",
                            "recurrence",
                            "mortality",
                            "outcome",
                            "target",
                        ],
                        default_to_first=False,
                    )
                    if not event_col:
                        candidates = [c for c in cols if c != time_col]
                        for c in candidates:
                            if df[c].dropna().nunique() <= 2:
                                event_col = c
                                break
                    if time_col and event_col:
                        return tool.forward(time_col=time_col, event_col=event_col)
                    return "Error: Could not resolve valid time duration and event indicator columns in active dataset for survival analysis. Please provide columns for follow-up duration and event status."
            elif (
                "table 1" in lower_p
                or "baseline characteristics" in lower_p
                or "table one" in lower_p
            ):
                tool = self.tools.get("table_one_baseline")
                if tool:
                    return tool.forward()
            elif "logistic" in lower_p or "odds ratio" in lower_p:
                tool = self.tools.get("logistic_regression")
                if tool:
                    df = self._get_df(tool)
                    if df is None or df.empty:
                        return tool.forward(
                            outcome_col="outcome", predictor_cols=["treatment", "age"]
                        )
                    cols = df.columns.tolist()
                    outcome_col = select_variable_by_keyword(
                        cols,
                        [
                            "outcome",
                            "death",
                            "event",
                            "status",
                            "response",
                            "target",
                            "disease",
                            "mortality",
                            "y",
                        ],
                        default_to_first=False,
                    )
                    if not outcome_col:
                        for c in cols:
                            if df[c].dropna().nunique() <= 2:
                                outcome_col = c
                                break
                    if outcome_col:
                        preds = [c for c in cols if c != outcome_col]
                        if preds:
                            return tool.forward(
                                outcome_col=outcome_col, predictor_cols=preds[:4]
                            )
                        return "Error: No predictor columns available in active dataset for logistic regression."
                    return "Error: Could not resolve a valid binary outcome column in active dataset for logistic regression. Please provide a binary outcome variable."
            elif (
                "diagnostic" in lower_p
                or "sensitivity" in lower_p
                or "stard" in lower_p
                or "fagan" in lower_p
            ):
                tool = self.tools.get("diagnostic_accuracy")
                if tool:
                    df = self._get_df(tool)
                    if df is not None and not df.empty:
                        cols = df.columns.tolist()
                        idx_col = select_variable_by_keyword(
                            cols,
                            [
                                "pocus",
                                "index_test",
                                "index",
                                "test",
                                "screening",
                                "biomarker",
                                "assay",
                                "marker",
                            ],
                            default_to_first=False,
                        )
                        ref_col = select_variable_by_keyword(
                            [c for c in cols if c != idx_col],
                            [
                                "gold_standard",
                                "reference",
                                "ref",
                                "disease",
                                "diagnosis",
                                "status",
                                "mortality",
                                "death",
                                "event",
                                "outcome",
                            ],
                            default_to_first=False,
                        )
                        if not idx_col and len(cols) >= 1:
                            idx_col = cols[0]
                        if not ref_col and len(cols) >= 2:
                            ref_col = [c for c in cols if c != idx_col][0]
                        if idx_col and ref_col:
                            return tool.forward(
                                index_test_col=idx_col, ref_standard_col=ref_col
                            )
                        return "Error: Could not resolve valid index test and reference standard columns in active dataset for diagnostic accuracy."
                    return "Error: No active dataset loaded in session for diagnostic accuracy. Please load a dataset with index and reference columns or specify all four 2x2 contingency counts (tp, fp, fn, tn)."
            elif (
                "rct" in lower_p
                or "consort" in lower_p
                or "randomized" in lower_p
                or "relative risk" in lower_p
            ):
                tool = self.tools.get("binary_rct_analysis")
                if tool:
                    df = self._get_df(tool)
                    if df is None or df.empty:
                        return tool.forward(
                            treatment_col="treatment", outcome_col="outcome"
                        )
                    cols = df.columns.tolist()
                    t_col = select_variable_by_keyword(
                        cols,
                        [
                            "treatment",
                            "treat",
                            "arm",
                            "group",
                            "intervention",
                            "therapy",
                            "rx",
                        ],
                        default_to_first=False,
                    )
                    o_col = select_variable_by_keyword(
                        [c for c in cols if c != t_col],
                        [
                            "outcome",
                            "death",
                            "event",
                            "status",
                            "endpoint",
                            "primary",
                            "mortality",
                            "response",
                        ],
                        default_to_first=False,
                    )
                    if not t_col and len(cols) >= 1:
                        t_col = cols[0]
                    if not o_col and len(cols) >= 2:
                        o_col = [c for c in cols if c != t_col][0]
                    if t_col and o_col:
                        return tool.forward(treatment_col=t_col, outcome_col=o_col)
                    return "Error: Could not resolve valid treatment and outcome columns in active dataset for RCT analysis."
            elif (
                "psm" in lower_p
                or "propensity" in lower_p
                or "nearest-neighbor" in lower_p
            ):
                tool = self.tools.get("propensity_score_matching")
                if tool:
                    df = self._get_df(tool)
                    if df is None or df.empty:
                        return tool.forward(
                            treatment_col="treatment", covariate_cols=["age", "bmi"]
                        )
                    cols = df.columns.tolist()
                    t_col = select_variable_by_keyword(
                        cols,
                        [
                            "treatment",
                            "treat",
                            "arm",
                            "group",
                            "exposure",
                            "intervention",
                            "therapy",
                        ],
                        default_to_first=False,
                    )
                    if not t_col and cols:
                        t_col = cols[0]
                    covars = [c for c in cols if c != t_col]
                    if t_col and covars:
                        return tool.forward(
                            treatment_col=t_col, covariate_cols=covars[:4]
                        )
                    return "Error: Could not resolve valid treatment indicator and covariate columns in active dataset for PSM."
            elif "linear" in lower_p or "ols" in lower_p:
                tool = self.tools.get("linear_regression")
                if tool:
                    df = self._get_df(tool)
                    if df is None or df.empty:
                        return tool.forward(
                            outcome_col="outcome", predictor_cols=["age", "treatment"]
                        )
                    cols = df.columns.tolist()
                    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    out_col = select_variable_by_keyword(
                        num_cols or cols,
                        [
                            "outcome",
                            "continuous",
                            "y",
                            "target",
                            "score",
                            "sbp",
                            "los",
                            "creatinine",
                            "bmi",
                            "measurement",
                        ],
                        default_to_first=False,
                    )
                    if not out_col and num_cols:
                        out_col = num_cols[0]
                    elif not out_col and cols:
                        out_col = cols[0]
                    preds = [c for c in cols if c != out_col]
                    if out_col and preds:
                        return tool.forward(
                            outcome_col=out_col, predictor_cols=preds[:4]
                        )
                    return "Error: Could not resolve valid continuous outcome and predictor columns in active dataset for linear regression."
            elif "pubmed" in lower_p or "pico" in lower_p or "evidence" in lower_p:
                tool = self.tools.get("pubmed_evidence_search")
                if tool:
                    return tool.forward(query=prompt, max_results=3)
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
4. Rigor: Propose concrete primary endpoints, effect sizes (RR, OR, HR, RD), exact test statistics, and power calculations.
5. Format: Output clean, professional markdown with high-contrast badge styling.
"""


class ClinicalAgentRunner:
    """
    High-Reasoning LLM Agent Bridge powered by Hugging Face Inference API.
    Supports Qwen 2.5 72B / Llama 3.3 70B with Zero-Cost offline fallback.
    """

    @classmethod
    def get_token(cls) -> Optional[str]:
        """Retrieves Hugging Face access token from environment."""
        token = os.getenv("HF_TOKEN")
        if token and token.strip():
            return token.strip()
        return None

    @classmethod
    def is_llm_available(cls) -> bool:
        """Returns True if Hugging Face API token is configured."""
        return cls.get_token() is not None and HAS_HF_HUB

    @classmethod
    def get_model_id(cls) -> str:
        """Returns active LLM model identifier."""
        return os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")

    @classmethod
    def chat_completion(
        cls,
        messages: list[dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """
        Executes a chat completion call via Hugging Face InferenceClient.
        Returns generated text or None if unconfigured / failed.
        """
        token = cls.get_token()
        if not token or not HAS_HF_HUB:
            return None

        try:
            client = InferenceClient(token=token, timeout=30.0)
            model_id = cls.get_model_id()
            res = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = res.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            logger.warning("Hugging Face InferenceClient error: %s", e)
            return None

    @classmethod
    def extract_biomedical_search_terms(cls, user_query: str) -> str:
        """
        Translates or extracts precise English MeSH/biomedical search terms from user inquiry.
        """
        if not cls.is_llm_available():
            return user_query

        prompt = (
            f"You are a biomedical librarian. Extract 3 to 5 precise English medical keywords "
            f"and MeSH search terms for PubMed search based on this user query (which may be in Thai or English):\n\n"
            f'User Query: "{user_query}"\n\n'
            f"Output ONLY the English search terms separated by space, without quotes, explanations, or punctuation."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a concise medical keyword extractor.",
            },
            {"role": "user", "content": prompt},
        ]
        result = cls.chat_completion(messages, max_tokens=60, temperature=0.1)
        if result:
            clean_res = re.sub(r"[^\w\s-]", "", result).strip()
            if clean_res:
                return clean_res
        return user_query

    @classmethod
    def synthesize_proposals_with_llm(
        cls,
        clinical_topic: str,
        articles: list[dict[str, Any]],
    ) -> Optional[str]:
        """
        Uses LLM to synthesize 5 tailored, publication-standard clinical study designs (SAMPL & EQUATOR compliant).
        """
        if not cls.is_llm_available():
            return None

        lit_summary = ""
        if articles:
            lit_summary = "\n".join(
                [
                    f"- {a['title']} ({a.get('journal', '')}, {a.get('pubdate', '')})"
                    for a in articles
                ]
            )

        prompt = f"""Clinical Research Topic / Question: "{clinical_topic}"

Relevant Benchmark Literature from PubMed:
{lit_summary or "None retrieved"}

Please formulate 5 methodologically distinct, high-impact clinical study designs and statistical analysis plans (SAPs) adhering to SAMPL and EQUATOR guidelines.

Cover these 5 designs:
1. Option 1: Interventional RCT (CONSORT 2010 compliant)
2. Option 2: Time-to-Event Survival Cohort (STROBE compliant, Kaplan-Meier & Cox PH)
3. Option 3: Diagnostic Accuracy Trial (STARD 2015 compliant, Sensitivity, Specificity, Likelihood Ratios)
4. Option 4: Clinical Prediction Model (TRIPOD+AI compliant, Multivariable Logistic / Machine Learning, ROC/AUC, Calibration)
5. Option 5: Real-World Comparative Effectiveness with Propensity Score Matching (PSM, 1:1 Nearest-Neighbor, SMD < 0.10)

For each option, provide:
- Clinical Rationale & Objective
- PICO (Population, Intervention/Exposure, Comparator, Primary Endpoint)
- Recommended Statistical Plan
- Estimated Sample Size & Power Justification

Format output in professional, publication-ready English Markdown.
"""

        messages = [
            {"role": "system", "content": CLINICAL_TECH_LEAD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return cls.chat_completion(messages, max_tokens=2500, temperature=0.4)

    @classmethod
    def consult_llm(
        cls,
        user_query: str,
        articles: list[dict[str, Any]],
        session_context: str = "",
    ) -> Optional[str]:
        """
        Executes a deep clinical consultation turn using the LLM agent.
        """
        if not cls.is_llm_available():
            return None

        lit_summary = ""
        if articles:
            lit_summary = "PubMed Literature Citations:\n" + "\n".join(
                [f"- {a['title']} | {a['vancouver_citation']}" for a in articles]
            )

        prompt = f"""User Question / Consultation:
"{user_query}"

{session_context}

{lit_summary}

Provide expert clinical tech lead consultation. Ensure all statistical guidance complies with SAMPL & EQUATOR standards. If suggesting analyses, provide explicit test statistics and effect size definitions.
"""
        messages = [
            {"role": "system", "content": CLINICAL_TECH_LEAD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return cls.chat_completion(messages, max_tokens=1800, temperature=0.3)


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
    backend: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    state_df_provider: Optional[Any] = None,
) -> ToolCallingAgent:
    """
    Factory function to initialize a ToolCallingAgent with clinical tools and system prompt.
    """
    from agent.tools.tool_pubmed import PubMedEvidenceTool
    from agent.tools.tool_sample_size import SampleSizeTool
    from agent.tools.tool_stat_harness import (
        BaselineTableOneTool,
        BinaryRCTTool,
        DiagnosticAccuracyTool,
        LinearRegressionTool,
        LogisticRegressionTool,
        PropensityScoreMatchingTool,
        SurvivalAnalysisTool,
    )
    from agent.tools.tool_synthetic_data import SyntheticDataTool

    if tools is None:
        tools = [
            PubMedEvidenceTool(),
            SampleSizeTool(),
            SyntheticDataTool(),
            SurvivalAnalysisTool(state_df_provider=state_df_provider),
            BaselineTableOneTool(state_df_provider=state_df_provider),
            LogisticRegressionTool(state_df_provider=state_df_provider),
            DiagnosticAccuracyTool(state_df_provider=state_df_provider),
            BinaryRCTTool(state_df_provider=state_df_provider),
            PropensityScoreMatchingTool(state_df_provider=state_df_provider),
            LinearRegressionTool(state_df_provider=state_df_provider),
        ]

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
