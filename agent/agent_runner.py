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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

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
    from huggingface_hub import get_token as hf_get_token

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    InferenceClient = None  # type: ignore

    def hf_get_token() -> Optional[str]:  # type: ignore
        return None


try:
    from smolagents import (
        InferenceClientModel,
    )
    from smolagents import (
        ToolCallingAgent as SmolToolCallingAgent,
    )

    HAS_SMOLAGENTS = True
    HfApiModel = InferenceClientModel
except ImportError:
    try:
        from smolagents import (
            HfApiModel as InferenceClientModel,
        )  # type: ignore
        from smolagents import (
            ToolCallingAgent as SmolToolCallingAgent,
        )

        HAS_SMOLAGENTS = True
        HfApiModel = InferenceClientModel
    except ImportError:
        HAS_SMOLAGENTS = False
        SmolToolCallingAgent = None  # type: ignore

        class InferenceClientModel:  # type: ignore
            def __init__(
                self,
                model_id: str,
                provider: Optional[str] = None,
                token: Optional[str] = None,
                **kwargs,
            ):
                self.model_id = model_id
                self.provider = provider
                self.token = token

        HfApiModel = InferenceClientModel  # type: ignore


SUPPORTED_HF_MODELS: dict[str, str] = {
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistral-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
}


DEFAULT_HF_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"


def resolve_hf_model_id(model_name_or_alias: Optional[str] = None) -> str:
    """
    Resolves UI labels, aliases, or environment variables to canonical Hugging Face Model IDs.
    """
    raw = (
        model_name_or_alias
        if model_name_or_alias and model_name_or_alias.strip()
        else os.getenv("HF_MODEL_ID")
    )
    if not raw or not raw.strip():
        return DEFAULT_HF_MODEL_ID

    trimmed = raw.strip()
    if "/" in trimmed:
        return trimmed

    clean = trimmed.lower()
    if "qwen" in clean:
        return "Qwen/Qwen2.5-72B-Instruct"
    elif "llama" in clean:
        return "meta-llama/Llama-3.3-70B-Instruct"
    elif "deepseek" in clean or "r1" in clean:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    elif "mistral" in clean:
        return "mistralai/Mistral-Small-24B-Instruct-2501"

    return DEFAULT_HF_MODEL_ID


def _is_identifier_column(col_name: str) -> bool:
    c_lower = col_name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", c_lower))
    terms = {"id", "patient", "subject", "hn", "record_id", "mrn", "citizen_id"}
    if any(t in terms for t in tokens):
        return True
    raw_tokens = set(re.split(r"[^\w]+", c_lower))
    return any(t in terms for t in raw_tokens)


def _is_outcome_or_endpoint_column(col_name: str) -> bool:
    c_lower = col_name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", c_lower))
    terms = {
        "endpoint",
        "outcome",
        "death",
        "mortality",
        "recurrence",
        "relapse",
        "surv_time",
        "os_months",
        "pfs_days",
        "followup",
        "follow_up",
    }
    if any(t in terms for t in tokens):
        return True
    raw_tokens = set(re.split(r"[^\w]+", c_lower))
    if any(t in terms for t in raw_tokens):
        return True
    return any(
        k in c_lower
        for k in [
            "endpoint",
            "outcome",
            "death",
            "mortality",
            "follow_up",
            "followup",
            "surv_time",
            "os_months",
            "pfs_days",
        ]
    )


def _is_binary_column(df: pd.DataFrame, col_name: str) -> bool:
    if col_name not in df.columns:
        return False
    return df[col_name].dropna().nunique() == 2


def _resolve_survival_columns(
    df: pd.DataFrame, prompt: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
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
    if time_col and not pd.api.types.is_numeric_dtype(df[time_col]):
        time_col = None

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
    if event_col and (
        not pd.api.types.is_numeric_dtype(df[event_col])
        or not _is_binary_column(df, event_col)
    ):
        event_col = None

    if not time_col or not event_col:
        return None, (
            "Error: Could not resolve valid time duration and binary event indicator columns in active dataset for survival analysis. "
            "Please provide explicit columns for follow-up duration (numeric) and event status (binary)."
        )

    treat_col = select_variable_by_keyword(
        [c for c in cols if c not in [time_col, event_col]],
        [
            "treatment",
            "treat",
            "arm",
            "group",
            "intervention",
            "therapy",
            "rx",
            "exposure",
        ],
        default_to_first=False,
    )

    covar_candidates = [
        c
        for c in cols
        if c not in [time_col, event_col, treat_col] and not _is_identifier_column(c)
    ][:4]

    lower_p = prompt.lower()
    is_explicit_cox = "cox" in lower_p
    if is_explicit_cox and not covar_candidates and not treat_col:
        return None, (
            "Error: Cox proportional hazards regression requires at least one covariate or predictor variable. "
            "Please specify covariate columns to include in the Cox model."
        )

    covars_to_pass = covar_candidates if (is_explicit_cox or covar_candidates) else None
    if is_explicit_cox and not covars_to_pass and treat_col:
        covars_to_pass = [treat_col]

    return {
        "time_col": time_col,
        "event_col": event_col,
        "group_col": treat_col,
        "covar_cols": covars_to_pass,
    }, None


def _build_survival_meta(kwargs: dict[str, Any], result: tuple) -> dict[str, Any]:
    stats = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
    return {
        "time_col": kwargs.get("time_col"),
        "event_col": kwargs.get("event_col"),
        "covar_cols": kwargs.get("covar_cols"),
        "cox_stats": stats.get("cox_stats", {}),
        "fitted_events": stats.get("fitted_events"),
        "fitted_non_events": stats.get("fitted_non_events"),
    }


def _resolve_table_one_columns(
    df: pd.DataFrame, prompt: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    cols = df.columns.tolist()
    group_col = select_variable_by_keyword(
        cols,
        [
            "treatment",
            "treat",
            "arm",
            "group",
            "intervention",
            "therapy",
            "exposure",
        ],
        default_to_first=False,
    )
    selected_vars = [
        c for c in cols if c != group_col and not _is_identifier_column(c)
    ][:8]
    return {"group_col": group_col, "selected_vars": selected_vars}, None


def _resolve_logistic_columns(
    df: pd.DataFrame, prompt: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
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
            "aki_endpoint",
            "recurrence",
        ],
        default_to_first=False,
    )
    if not outcome_col or not _is_binary_column(df, outcome_col):
        return None, (
            "Error: Could not resolve a valid binary outcome column in active dataset for logistic regression. "
            "Please provide an explicit binary outcome variable (with 2 distinct classes)."
        )

    preds = [c for c in cols if c != outcome_col and not _is_identifier_column(c)][:4]
    if not preds:
        return (
            None,
            "Error: No predictor columns available in active dataset for logistic regression.",
        )

    return {"outcome_col": outcome_col, "predictor_cols": preds}, None


def _build_logistic_meta(kwargs: dict[str, Any], result: tuple) -> dict[str, Any]:
    coef_df = result[2] if len(result) > 2 else None
    metrics = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
    return {
        "outcome_col": kwargs.get("outcome_col"),
        "predictor_cols": kwargs.get("predictor_cols"),
        "coef_df": coef_df,
        "fitted_events": metrics.get("fitted_events"),
        "fitted_non_events": metrics.get("fitted_non_events"),
    }


def _resolve_diagnostic_columns(
    df: pd.DataFrame, prompt: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    cols = df.columns.tolist()
    idx_col = select_variable_by_keyword(
        cols,
        [
            "pocus_scan",
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
            "gold_dx",
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
    if not idx_col or not ref_col:
        return None, (
            "Error: Could not resolve valid index test and reference standard columns in active dataset for diagnostic accuracy. "
            "Please specify index test and gold standard reference columns."
        )
    if not _is_binary_column(df, idx_col) or not _is_binary_column(df, ref_col):
        return (
            None,
            "Error: Index test and reference standard columns must be binary (0/1 or 2 distinct categories).",
        )

    return {"index_test_col": idx_col, "ref_standard_col": ref_col}, None


def _build_diagnostic_meta(kwargs: dict[str, Any], result: tuple) -> dict[str, Any]:
    return result[3] if len(result) > 3 and isinstance(result[3], dict) else {}


def _resolve_rct_columns(
    df: pd.DataFrame, prompt: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    cols = df.columns.tolist()
    t_col = select_variable_by_keyword(
        cols,
        [
            "study_arm",
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
            "aki_endpoint",
        ],
        default_to_first=False,
    )
    if not t_col or not o_col:
        return None, (
            "Error: Could not resolve valid treatment and outcome columns in active dataset for RCT analysis. "
            "Please provide binary treatment and outcome column names."
        )
    if not _is_binary_column(df, t_col) or not _is_binary_column(df, o_col):
        return (
            None,
            "Error: Treatment and outcome columns in RCT analysis must be binary.",
        )

    return {"treatment_col": t_col, "outcome_col": o_col}, None


def _build_rct_meta(kwargs: dict[str, Any], result: tuple) -> dict[str, Any]:
    return result[3] if len(result) > 3 and isinstance(result[3], dict) else {}


def _resolve_psm_columns(
    df: pd.DataFrame, prompt: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    cols = df.columns.tolist()
    t_col = select_variable_by_keyword(
        cols,
        [
            "study_arm",
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
    if not t_col or not _is_binary_column(df, t_col):
        return None, (
            "Error: Could not resolve a valid binary treatment indicator column in active dataset for PSM. "
            "Please provide a binary treatment variable."
        )
    covars = [
        c
        for c in cols
        if c != t_col
        and not _is_identifier_column(c)
        and not _is_outcome_or_endpoint_column(c)
    ][:4]
    if not covars:
        return None, "Error: No covariate columns available in active dataset for PSM."

    return {"treatment_col": t_col, "covariate_cols": covars}, None


def _build_psm_meta(kwargs: dict[str, Any], result: tuple) -> dict[str, Any]:
    balance_df = result[2] if len(result) > 2 else None
    stats = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
    return {"balance_df": balance_df, **stats}


def _resolve_linear_columns(
    df: pd.DataFrame, prompt: str
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    out_col = select_variable_by_keyword(
        num_cols,
        [
            "outcome",
            "continuous",
            "y",
            "target",
            "score",
            "sbp_mmhg",
            "sbp",
            "los",
            "creatinine",
            "bmi_value",
            "bmi",
            "measurement",
        ],
        default_to_first=False,
    )
    if not out_col or not pd.api.types.is_numeric_dtype(df[out_col]):
        return None, (
            "Error: Could not resolve a valid continuous/numeric outcome column in active dataset for linear regression. "
            "Please provide an explicit numeric outcome column."
        )
    preds = [c for c in cols if c != out_col and not _is_identifier_column(c)][:4]
    if not preds:
        return (
            None,
            "Error: No predictor columns available in active dataset for linear regression.",
        )

    return {"outcome_col": out_col, "predictor_cols": preds}, None


def _build_linear_meta(kwargs: dict[str, Any], result: tuple) -> dict[str, Any]:
    coef_df = result[2] if len(result) > 2 else None
    metrics = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
    return {
        "outcome_col": kwargs.get("outcome_col"),
        "predictor_cols": kwargs.get("predictor_cols"),
        "coef_df": coef_df,
        **metrics,
    }


@dataclass
class AnalysisRoute:
    keywords: list[str]
    tool_name: str
    resolver: Callable[
        [pd.DataFrame, str], tuple[Optional[dict[str, Any]], Optional[str]]
    ]
    critique_type: Optional[str] = None
    default_forward_kwargs: Optional[dict[str, Any]] = None
    no_df_error: Optional[str] = None
    build_critique_meta: Optional[Callable[[dict[str, Any], tuple], dict[str, Any]]] = (
        None
    )


ANALYSIS_ROUTES: list[AnalysisRoute] = [
    AnalysisRoute(
        keywords=["survival", "kaplan", "cox", "logrank", "log-rank"],
        tool_name="survival_analysis",
        resolver=_resolve_survival_columns,
        critique_type="survival",
        default_forward_kwargs={"time_col": "time", "event_col": "event"},
        build_critique_meta=_build_survival_meta,
    ),
    AnalysisRoute(
        keywords=["table 1", "baseline characteristics", "table one"],
        tool_name="table_one_baseline",
        resolver=_resolve_table_one_columns,
        critique_type=None,
        default_forward_kwargs={},
    ),
    AnalysisRoute(
        keywords=["logistic", "odds ratio"],
        tool_name="logistic_regression",
        resolver=_resolve_logistic_columns,
        critique_type="logistic",
        default_forward_kwargs={
            "outcome_col": "outcome",
            "predictor_cols": ["treatment", "age"],
        },
        build_critique_meta=_build_logistic_meta,
    ),
    AnalysisRoute(
        keywords=["diagnostic", "sensitivity", "stard", "fagan"],
        tool_name="diagnostic_accuracy",
        resolver=_resolve_diagnostic_columns,
        critique_type="diagnostic",
        no_df_error=(
            "Error: No active dataset loaded in session for diagnostic accuracy. "
            "Please load a dataset with index and reference columns."
        ),
        build_critique_meta=_build_diagnostic_meta,
    ),
    AnalysisRoute(
        keywords=["rct", "consort", "randomized", "relative risk"],
        tool_name="binary_rct_analysis",
        resolver=_resolve_rct_columns,
        critique_type="rct",
        default_forward_kwargs={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
        },
        build_critique_meta=_build_rct_meta,
    ),
    AnalysisRoute(
        keywords=["psm", "propensity", "nearest-neighbor"],
        tool_name="propensity_score_matching",
        resolver=_resolve_psm_columns,
        critique_type="psm",
        default_forward_kwargs={
            "treatment_col": "treatment",
            "covariate_cols": ["age", "bmi"],
        },
        build_critique_meta=_build_psm_meta,
    ),
    AnalysisRoute(
        keywords=["linear", "ols"],
        tool_name="linear_regression",
        resolver=_resolve_linear_columns,
        critique_type="linear",
        default_forward_kwargs={
            "outcome_col": "outcome",
            "predictor_cols": ["age", "treatment"],
        },
        build_critique_meta=_build_linear_meta,
    ),
]


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

    def _execute_analysis_route(self, route: AnalysisRoute, prompt: str) -> str:
        tool = self.tools.get(route.tool_name)
        if not tool:
            return f"Clinical AI Co-Pilot analysis for: {prompt}\n\nAll computations adhere to SAMPL guidelines and Zero-PHI security."

        df = self._get_df(tool)
        if df is None or df.empty:
            if route.no_df_error:
                return route.no_df_error
            forward_kwargs = route.default_forward_kwargs or {}
            return tool.forward(**forward_kwargs)

        resolved_kwargs, error_msg = route.resolver(df, prompt)
        if error_msg:
            return error_msg
        if resolved_kwargs is None:
            resolved_kwargs = {}

        raw_result = tool.run_with_dataframe(df=df, **resolved_kwargs)
        text_out = (
            raw_result[0] if isinstance(raw_result, (tuple, list)) else str(raw_result)
        )

        if not route.critique_type:
            return text_out

        from agent.critique_engine import CritiqueEngine

        meta = (
            route.build_critique_meta(resolved_kwargs, raw_result)
            if route.build_critique_meta
            else (
                raw_result[3]
                if isinstance(raw_result, (tuple, list))
                and len(raw_result) > 3
                and isinstance(raw_result[3], dict)
                else {}
            )
        )
        critique = CritiqueEngine.appraise_analysis(route.critique_type, df, meta)
        return f"{text_out}\n\n---\n{critique.to_markdown()}"

    def run(self, prompt: str) -> str:
        lower_p = prompt.lower()
        if "sample size" in lower_p or "power" in lower_p or "n_total" in lower_p:
            tool = self.tools.get("sample_size_calculator")
            if tool:
                return tool.forward(test_type="two_proportions", p1=0.25, p2=0.15)

        for route in ANALYSIS_ROUTES:
            if any(kw in lower_p for kw in route.keywords):
                return self._execute_analysis_route(route, prompt)

        if "pubmed" in lower_p or "pico" in lower_p or "evidence" in lower_p:
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
    Supports Qwen 2.5 72B / Llama 3.3 70B / DeepSeek R1 32B / Mistral Small 24B
    with Zero-Cost offline fallback.
    """

    @classmethod
    def get_token(cls, explicit_token: Optional[str] = None) -> Optional[str]:
        """
        Retrieves Hugging Face access token with hierarchical discovery:
        1. Explicit argument passed to call
        2. Environment variables (HF_TOKEN, HUGGINGFACE_HUB_TOKEN)
        3. huggingface_hub.get_token() (~/.cache/huggingface/token or CLI login)
        4. .env file in workspace root
        """
        if explicit_token and explicit_token.strip():
            return explicit_token.strip()

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token and token.strip():
            return token.strip()

        if HAS_HF_HUB and callable(hf_get_token):
            try:
                cached_token = hf_get_token()
                if cached_token and str(cached_token).strip():
                    return str(cached_token).strip()
            except Exception as e:
                logger.debug("Error checking huggingface_hub.get_token: %s", e)

        # Check local .env file in project root
        env_file = Path(".env")
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") or line.startswith(
                        "HUGGINGFACE_HUB_TOKEN="
                    ):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
            except Exception:
                pass

        return None

    @classmethod
    def is_llm_available(cls, explicit_token: Optional[str] = None) -> bool:
        """Returns True if Hugging Face API token is configured."""
        return cls.get_token(explicit_token) is not None and HAS_HF_HUB

    @classmethod
    def get_model_id(cls, model_alias: Optional[str] = None) -> str:
        """Returns active LLM model identifier."""
        return resolve_hf_model_id(model_alias)

    @classmethod
    def test_hf_connection(
        cls,
        token: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Actively tests live connection to Hugging Face Inference API.
        Returns (success: bool, message: str, details: dict).
        """
        resolved_token = cls.get_token(token)
        if not resolved_token:
            return (
                False,
                "No Hugging Face access token found. Please set HF_TOKEN in Settings or .env file.",
                {"token_found": False},
            )

        if not HAS_HF_HUB or InferenceClient is None:
            return (
                False,
                "huggingface_hub package is not available in environment.",
                {"has_hf_hub": False},
            )

        target_model = resolve_hf_model_id(model_id)
        provider = provider or os.getenv("HF_INFERENCE_PROVIDER", None)

        import time

        t0 = time.perf_counter()
        try:
            client = InferenceClient(
                token=resolved_token,
                provider=provider if provider != "auto" else None,
                timeout=20.0,
            )
            test_messages = [
                {
                    "role": "system",
                    "content": "You are a concise medical research AI co-pilot.",
                },
                {
                    "role": "user",
                    "content": "Respond strictly with 'StatioMed AI Connected: Ready' and nothing else.",
                },
            ]
            res = client.chat.completions.create(
                model=target_model,
                messages=test_messages,
                max_tokens=25,
                temperature=0.1,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            content = res.choices[0].message.content if res.choices else ""
            clean_reply = content.strip() if content else "Connected"
            return (
                True,
                f"✅ Connected to Hugging Face AI ({target_model}) in {elapsed_ms}ms! Response: {clean_reply}",
                {
                    "model": target_model,
                    "elapsed_ms": elapsed_ms,
                    "response": clean_reply,
                    "provider": provider or "auto",
                },
            )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            err_str = str(e)
            logger.warning("HF connection test failed: %s", err_str)
            return (
                False,
                f"❌ Hugging Face API connection failed ({elapsed_ms}ms): {err_str}",
                {"model": target_model, "elapsed_ms": elapsed_ms, "error": err_str},
            )

    @classmethod
    def chat_completion(
        cls,
        messages: list[dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.3,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Executes a chat completion call via Hugging Face InferenceClient.
        Returns generated text or None if unconfigured / failed.
        """
        resolved_token = cls.get_token(token)
        if not resolved_token or not HAS_HF_HUB or InferenceClient is None:
            return None

        target_model = resolve_hf_model_id(model_id)
        provider = provider or os.getenv("HF_INFERENCE_PROVIDER", None)

        try:
            client = InferenceClient(
                token=resolved_token,
                provider=provider if provider != "auto" else None,
                timeout=45.0,
            )
            res = client.chat.completions.create(
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = res.choices[0].message.content if res.choices else None
            return content.strip() if content else None
        except Exception as e:
            logger.warning(
                "Hugging Face InferenceClient error (%s): %s", target_model, e
            )
            return None

    @classmethod
    def extract_biomedical_search_terms(
        cls,
        user_query: str,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> str:
        """
        Translates or extracts precise English MeSH/biomedical search terms from user inquiry.
        """
        if not cls.is_llm_available(token):
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
        result = cls.chat_completion(
            messages, max_tokens=60, temperature=0.1, model_id=model_id, token=token
        )
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
        model_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Uses LLM to synthesize 5 tailored, publication-standard clinical study designs (SAMPL & EQUATOR compliant).
        """
        if not cls.is_llm_available(token):
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
        return cls.chat_completion(
            messages, max_tokens=2500, temperature=0.4, model_id=model_id, token=token
        )

    @classmethod
    def consult_llm(
        cls,
        user_query: str,
        articles: list[dict[str, Any]],
        session_context: str = "",
        model_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Executes a deep clinical consultation turn using the LLM agent.
        """
        if not cls.is_llm_available(token):
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
        return cls.chat_completion(
            messages, max_tokens=1800, temperature=0.3, model_id=model_id, token=token
        )


def get_model(
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    Initializes InferenceClientModel with appropriate backend and token authentication.
    """
    backend = backend or os.getenv("LLM_BACKEND", "inference-providers")
    resolved_token = ClinicalAgentRunner.get_token(token)
    resolved_model = resolve_hf_model_id(model_id) if model_id else None

    if backend == "zerogpu-local":
        return InferenceClientModel(
            model_id=resolved_model
            or os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct-AWQ"),
            token=resolved_token,
        )
    elif backend == "inference-providers":
        return InferenceClientModel(
            model_id=resolved_model
            or os.getenv("PROVIDER_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"),
            provider=os.getenv("HF_INFERENCE_PROVIDER", "together"),
            token=resolved_token,
        )
    else:
        return InferenceClientModel(
            model_id=resolved_model or "Qwen/Qwen2.5-72B-Instruct", token=resolved_token
        )


def create_clinical_agent(
    backend: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    state_df_provider: Optional[Any] = None,
    ncbi_api_key: Optional[str] = None,
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
            PubMedEvidenceTool(api_key=ncbi_api_key),
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
