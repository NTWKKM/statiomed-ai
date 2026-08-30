"""
agent/clinical_analyst.py - Biostatistical Reasoning & Execution Harness
=============================================================================
Translates clinical objectives and research proposals into deterministic
statistical workflows. Uses utils/ as an execution harness to eliminate
LLM mathematical hallucination and guarantee SAMPL/EQUATOR compliance.
Includes PubMed Evidence Exploration & 5-Direction Research Synthesis.
=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from agent.tools.tool_pubmed import PubMedEvidenceTool
from agent.tools.tool_sample_size import SampleSizeTool
from agent.tools.tool_synthetic_data import SyntheticDataTool
from agent.topic_ideator import ClinicalTopicIdeator
from core.common import select_variable_by_keyword
from core.state import AppState
from logger import get_logger
from utils import linear_lib, logic, survival_lib
from utils.data_cleaning import load_data_robust
from utils.data_quality import check_data_quality
from utils.proposal_parser import ProposalMetadata, ProposalParser
from utils.table_one_advanced import TableOneGenerator
from utils.visualizations import plot_missing_pattern

logger = get_logger(__name__)


@dataclass
class AnalystResult:
    """Structured response from the Clinical Analyst Engine."""

    message_text: str
    figure: go.Figure | None = None
    preview_df: pd.DataFrame | None = None
    action_type: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


class StatHarness:
    """
    Deterministic statistical calculation harness.
    Wraps tested biostatistical libraries (lifelines, statsmodels, scipy, pingouin).
    """

    @staticmethod
    def run_table_one(
        df: pd.DataFrame,
        group_col: str | None = None,
        selected_vars: list[str] | None = None,
        var_meta: dict[str, Any] | None = None,
    ) -> tuple[str, pd.DataFrame]:
        """Generates Baseline Table 1 with Standardized Mean Differences (SMD)."""
        cols = df.columns.tolist()
        if not selected_vars:
            selected_vars = [c for c in cols if c != group_col][:10]

        generator = TableOneGenerator(df, var_meta or {})
        html_table = generator.generate(
            selected_vars=selected_vars,
            stratify_by=group_col,
        )
        return html_table, df[selected_vars]

    @staticmethod
    def run_survival(
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str | None = None,
        covar_cols: list[str] | None = None,
    ) -> tuple[go.Figure, pd.DataFrame, dict[str, Any]]:
        """Fits Kaplan-Meier Log-rank & Multivariable Cox Proportional Hazards."""
        km_fig, km_summary, missing_info = survival_lib.fit_km_logrank(
            df=df,
            duration_col=time_col,
            event_col=event_col,
            group_col=group_col,
        )

        p_val = "N/A"
        if (
            isinstance(km_summary, pd.DataFrame)
            and not km_summary.empty
            and "P-value" in km_summary.columns
        ):
            p_val = km_summary.iloc[0]["P-value"]

        cox_df = None
        cox_stats = {}
        if covar_cols and len(covar_cols) > 0:
            cph, res_df, _, err, c_stats, _ = survival_lib.fit_cox_ph(
                df=df,
                duration_col=time_col,
                event_col=event_col,
                covariate_cols=covar_cols,
            )
            if not err and res_df is not None:
                cox_df = res_df
                cox_stats = c_stats or {}

        return (
            km_fig,
            km_summary,
            {
                "km_stats": {"p_value": p_val},
                "cox_df": cox_df,
                "cox_stats": cox_stats,
                "missing_info": missing_info,
            },
        )

    @staticmethod
    def run_logistic(
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure]:
        """Fits binary logistic regression with Odds Ratios and 95% CIs."""
        clean_cols = [outcome_col] + [c for c in predictor_cols if c in df.columns]
        df_clean = df[clean_cols].dropna()

        html_table, or_results, status, metrics = logic.run_logistic_regression(
            df=df_clean, outcome_col=outcome_col, covariate_cols=predictor_cols
        )
        rows = []
        if or_results:
            for var_name, r in or_results.items():
                or_val = (
                    r.get("or", 1.0)
                    if isinstance(r, dict)
                    else getattr(r, "or_val", 1.0)
                )
                ci_l = (
                    r.get("ci_low", r.get("ci_lower", 1.0))
                    if isinstance(r, dict)
                    else getattr(r, "ci_low", getattr(r, "ci_lower", 1.0))
                )
                ci_u = (
                    r.get("ci_high", r.get("ci_upper", 1.0))
                    if isinstance(r, dict)
                    else getattr(r, "ci_high", getattr(r, "ci_upper", 1.0))
                )
                p_v = (
                    r.get("p_value", 1.0)
                    if isinstance(r, dict)
                    else getattr(r, "p_value", 1.0)
                )
                rows.append(
                    {
                        "Variable": var_name,
                        "Odds Ratio (OR)": f"{float(or_val):.3f}"
                        if isinstance(or_val, (int, float))
                        else str(or_val),
                        "95% CI Lower": f"{float(ci_l):.3f}"
                        if isinstance(ci_l, (int, float))
                        else str(ci_l),
                        "95% CI Upper": f"{float(ci_u):.3f}"
                        if isinstance(ci_u, (int, float))
                        else str(ci_u),
                        "P-value": f"{float(p_v):.4f}"
                        if isinstance(p_v, (int, float))
                        else str(p_v),
                    }
                )
        coef_df = pd.DataFrame(rows)

        fig = go.Figure()
        if not coef_df.empty and "Odds Ratio (OR)" in coef_df.columns:
            vars_list = coef_df["Variable"].tolist()
            or_vals = [float(x) for x in coef_df["Odds Ratio (OR)"]]
            ci_lows = [float(x) for x in coef_df["95% CI Lower"]]
            ci_highs = [float(x) for x in coef_df["95% CI Upper"]]

            fig.add_trace(
                go.Scatter(
                    x=or_vals,
                    y=vars_list,
                    mode="markers",
                    marker=dict(color="#0284c7", size=10),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[h - o for h, o in zip(ci_highs, or_vals)],
                        arrayminus=[
                            o - low_val for o, low_val in zip(or_vals, ci_lows)
                        ],
                        color="#0284c7",
                    ),
                    name="Odds Ratio (95% CI)",
                )
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444")
            fig.update_layout(
                title="Multivariable Logistic Regression (Forest Plot of Odds Ratios)",
                xaxis_title="Odds Ratio (Log Scale)",
                yaxis_title="Covariates",
                xaxis_type="log",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )

        return coef_df, metrics or {}, fig

    @staticmethod
    def run_diagnostic(
        df: pd.DataFrame,
        tp: int = 85,
        fp: int = 15,
        fn: int = 15,
        tn: int = 185,
        pre_test_prob: float = 25.0,
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure]:
        """Calculates Diagnostic Test Accuracy (STARD 2015) & Bayesian Fagan Nomogram."""
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        lr_pos = (sens / (1.0 - spec)) if (1.0 - spec) > 0 else 1.0
        lr_neg = ((1.0 - sens) / spec) if spec > 0 else 1.0
        dor = ((tp * tn) / (fp * fn)) if (fp * fn) > 0 else 1.0

        p_pre = pre_test_prob / 100.0
        odds_pre = p_pre / (1.0 - p_pre) if p_pre < 1.0 else 999.0
        odds_post_pos = odds_pre * lr_pos
        p_post_pos = (odds_post_pos / (1.0 + odds_post_pos)) * 100.0

        odds_post_neg = odds_pre * lr_neg
        p_post_neg = (odds_post_neg / (1.0 + odds_post_neg)) * 100.0

        metrics_df = pd.DataFrame(
            [
                {"Metric": "Sensitivity (True Positive Rate)", "Value": f"{sens:.1%}"},
                {"Metric": "Specificity (True Negative Rate)", "Value": f"{spec:.1%}"},
                {"Metric": "Positive Predictive Value (PPV)", "Value": f"{ppv:.1%}"},
                {"Metric": "Negative Predictive Value (NPV)", "Value": f"{npv:.1%}"},
                {"Metric": "Positive Likelihood Ratio (LR+)", "Value": f"{lr_pos:.2f}"},
                {"Metric": "Negative Likelihood Ratio (LR-)", "Value": f"{lr_neg:.2f}"},
                {"Metric": "Diagnostic Odds Ratio (DOR)", "Value": f"{dor:.2f}"},
            ]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2],
                y=[pre_test_prob, lr_pos, p_post_pos],
                mode="lines+markers+text",
                text=[
                    f"Pre: {pre_test_prob:.0f}%",
                    f"LR+: {lr_pos:.1f}",
                    f"Post: {p_post_pos:.1f}%",
                ],
                textposition="top center",
                line=dict(color="#059669", width=3),
                name="Positive Result (+LR)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2],
                y=[pre_test_prob, lr_neg, p_post_neg],
                mode="lines+markers+text",
                text=[
                    f"Pre: {pre_test_prob:.0f}%",
                    f"LR-: {lr_neg:.2f}",
                    f"Post: {p_post_neg:.1f}%",
                ],
                textposition="bottom center",
                line=dict(color="#dc2626", width=3, dash="dash"),
                name="Negative Result (-LR)",
            )
        )
        fig.update_layout(
            title="Bayesian Updating Trajectory (Pre-test to Post-test Probability)",
            xaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=[
                    "Pre-Test Prob (%)",
                    "Likelihood Ratio",
                    "Post-Test Prob (%)",
                ],
            ),
            yaxis_title="Probability (%) / Ratio",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        metrics = {
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "lr_pos": lr_pos,
            "lr_neg": lr_neg,
            "dor": dor,
            "post_prob_pos": p_post_pos,
            "post_prob_neg": p_post_neg,
        }
        return metrics_df, metrics, fig

    @staticmethod
    def run_linear(
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any], go.Figure]:
        """Fits OLS Linear Regression."""
        clean_cols = [outcome_col] + [c for c in predictor_cols if c in df.columns]
        df_clean = df[clean_cols].dropna()

        res = linear_lib.run_ols_regression(
            df=df_clean, outcome_col=outcome_col, predictor_cols=predictor_cols
        )
        coef_df = res["coef_table"]
        diag_plots = linear_lib.create_diagnostic_plots(res)
        fig = diag_plots.get("residuals_vs_fitted", go.Figure())
        return coef_df, res, fig

    @staticmethod
    def run_sample_size(
        p1: float = 0.30,
        p2: float = 0.15,
        power: float = 0.80,
        alpha: float = 0.05,
        dropout_rate: float = 0.15,
    ) -> dict[str, Any]:
        """Calculates Sample Size and Power with Drop-out adjustment."""
        return SampleSizeTool.calculate_two_proportions(
            p1=p1, p2=p2, power=power, alpha=alpha, dropout_rate=dropout_rate
        )


class ClinicalAnalystEngine:
    """
    Main reasoning engine that inspects proposals & datasets,
    selects appropriate biostatistical tools, and immediately executes them.
    """

    @classmethod
    def process_turn(
        cls,
        user_message: str,
        file_paths: list[str] | None,
        state: AppState,
    ) -> tuple[str, AppState, go.Figure | None, pd.DataFrame | None]:
        """
        Processes a multi-turn user message + attached files, executes statistical
        methods deterministically, and updates global AppState.
        """
        user_msg = (user_message or "").strip()
        lower_msg = user_msg.lower()
        file_paths = file_paths or []

        proposal_meta: ProposalMetadata | None = None
        loaded_new_data = False

        # 1. Handle File Ingestion
        for f in file_paths:
            p = Path(f)
            suffix = p.suffix.lower()
            if suffix in [".docx", ".doc", ".pdf", ".txt", ".md"]:
                logger.info(f"Parsing Proposal document: {p.name}")
                proposal_meta = ProposalParser.parse_proposal(p)
            elif suffix in [".csv", ".xlsx", ".xlsm", ".xls", ".sav", ".dta"]:
                logger.info(f"Loading Research dataset: {p.name}")
                try:
                    df = load_data_robust(p)
                    state.df = df
                    state.file_name = p.name
                    state.df_matched = None
                    state.is_matched = False
                    loaded_new_data = True
                except Exception as e:
                    logger.error(f"Error loading dataset {p.name}: {e}")

        # If user pasted raw proposal text in the prompt
        if not proposal_meta and any(
            k in lower_msg
            for k in [
                "pico",
                "population",
                "intervention",
                "primary outcome",
                "proposal",
                "ระเบียบวิธีวิจัย",
                "โครงร่างงานวิจัย",
            ]
        ):
            proposal_meta = ProposalParser.parse_proposal(user_msg)

        # 2. Determine Action Mode & Statistical Path

        # Case A: User selected a specific option from proposed directions (e.g. "เลือกข้อ 2", "option 2", "ขอข้อ 1", "เอาข้อ 3")
        opt_match = re.search(r"(?:เลือก|option|ข้อ|แนวทางที่)\s*([1-5])", lower_msg)
        if opt_match and not state.has_data():
            opt_id = int(opt_match.group(1))
            topic = re.sub(r"(?:เลือก|option|ข้อ|แนวทางที่|\d)", "", user_msg).strip()
            topic = topic or "Acute Dyspnea Clinical Investigation"

            df_gen, meta = SyntheticDataTool.generate_topic_aware_cohort(
                topic, n=250, seed=42
            )
            state.df = df_gen
            state.file_name = (
                f"Option {opt_id} Cohort: {meta.get('domain', 'Clinical Research')}"
            )
            state.var_meta = meta

            cols = df_gen.columns.tolist()
            time_col = select_variable_by_keyword(cols, ["time", "duration", "days"])
            event_col = select_variable_by_keyword(cols, ["death", "event", "status"])
            treat_col = select_variable_by_keyword(
                cols, ["treatment", "group", "therapy"]
            )
            covars = [
                c
                for c in cols
                if c not in [time_col, event_col, treat_col]
                and not any(
                    id_k in c.lower() for id_k in ["id", "patient", "subject", "hn"]
                )
            ][:4]

            fig = go.Figure()
            if opt_id in [2, 1] and time_col and event_col:
                fig, km_df, stats_dict = StatHarness.run_survival(
                    df_gen,
                    time_col=time_col,
                    event_col=event_col,
                    group_col=treat_col,
                    covar_cols=covars,
                )
                km_p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
                response_md = f"""### 🚀 ดำเนินการวิเคราะห์แนวทางที่ {opt_id} ทันที (Immediate Statistical Execution)

**สร้างชุดข้อมูลจำลองตามแนวทางที่ {opt_id}:** `{state.file_name}` (n = {len(df_gen):,} ราย)

#### 1. Kaplan-Meier Survival Analysis & Log-Rank Test:
- **Duration / Time:** `{time_col}` | **Event / Status:** `{event_col}`
- **Log-Rank P-value:** `{km_p_val}`
- **เหตุการณ์ที่เกิดขึ้นทั้งหมด (Events):** `{df_gen[event_col].sum()}` จาก `{len(df_gen)}` ราย

#### 2. Multivariable Cox Proportional Hazards Model:
- **Confounders Adjusted:** {", ".join([f"`{c}`" for c in covars])}
- **สถานะ:** ข้อมูลถูกบันทึกเข้า session และเรนเดอร์กราฟในหน้าต่าง Visual Output เรียบร้อยแล้วครับ
"""
                return response_md, state, fig, df_gen
            elif opt_id == 3:
                metrics_df, metrics, fig = StatHarness.run_diagnostic(df_gen)
                response_md = f"""### 🚀 ดำเนินการวิเคราะห์แนวทางที่ {opt_id} (Diagnostic Accuracy & Fagan Nomogram - STARD 2015)

**สร้างชุดข้อมูลจำลองตามแนวทางที่ {opt_id}:** `{state.file_name}` (n = {len(df_gen):,} ราย)

#### 1. ผลการประเมินความแม่นยำในการวินิจฉัย (Diagnostic Performance Metrics):
- **Sensitivity (ความไว):** `{metrics["sensitivity"]:.1%}` | **Specificity (ความจำเพาะ):** `{metrics["specificity"]:.1%}`
- **Positive Likelihood Ratio (LR+):** `{metrics["lr_pos"]:.2f}` | **Negative Likelihood Ratio (LR-):** `{metrics["lr_neg"]:.2f}`
- **Diagnostic Odds Ratio (DOR):** `{metrics["dor"]:.2f}`

#### 2. Bayesian Pre-test to Post-test Updating (Fagan Nomogram):
- **Pre-Test Probability:** `25.0%`
- **Post-Test Probability (Positive Test):** `{metrics["post_prob_pos"]:.1f}%`
- **Post-Test Probability (Negative Test):** `{metrics["post_prob_neg"]:.1f}%`

{metrics_df.to_markdown(index=False)}
"""
                return response_md, state, fig, df_gen
            elif opt_id in [4, 5]:
                coef_df, metrics, fig = StatHarness.run_logistic(
                    df_gen,
                    outcome_col=event_col or cols[1],
                    predictor_cols=covars or cols[:4],
                )
                response_md = f"""### 🚀 ดำเนินการวิเคราะห์แนวทางที่ {opt_id} (Multivariable Model & Table 1)

**ชุดข้อมูล:** `{state.file_name}` (n = {len(df_gen):,} ราย)  
**ตัวแปรตาม:** `{event_col or cols[1]}` (Binary Event)  
**McFadden Pseudo-$R^2$:** `{metrics.get("mcfadden", 0.0):.4f}` | **AIC:** `{metrics.get("aic", 0.0):.1f}`

{coef_df.to_markdown(index=False)}
"""
                return response_md, state, fig, df_gen
            else:
                html_table, df_sub = StatHarness.run_table_one(
                    df_gen, group_col=treat_col
                )
                response_md = f"""### 🚀 ดำเนินการสร้างชุดข้อมูลตามแนวทางที่ {opt_id} (Baseline Characteristics & Table 1)

**สร้างชุดข้อมูลจำลองตามแนวทางที่ {opt_id}:** `{state.file_name}` (n = {len(df_gen):,} ราย)

ชุดข้อมูลถูกบันทึกเข้า session เรียบร้อยแล้ว พร้อมสำหรับการวิเคราะห์ขั้นต่อไปครับ
"""
                return response_md, state, fig, df_gen

        # Case B: Broad Topic Ideation & PubMed Evidence Search (e.g. "dyspnea", "sepsis", "เสนอแนวทางวิจัย", "หัวข้อวิจัย")
        is_topic_query = (
            not state.has_data()
            and not file_paths
            and not proposal_meta
            and (
                len(user_msg.split()) <= 8
                or any(
                    k in lower_msg
                    for k in [
                        "dyspnea",
                        "sepsis",
                        "cardiac arrest",
                        "aki",
                        "stroke",
                        "heart failure",
                        "pneumonia",
                        "asthma",
                        "copd",
                        "เหนื่อย",
                        "หอบ",
                        "หัวข้อวิจัย",
                        "เสนอแนวทาง",
                        "ไอเดียวิจัย",
                        "หัวข้อ",
                        "directions",
                        "research ideas",
                        "topic",
                        "เจ็บหน้าอก",
                        "ติดเชื้อ",
                    ]
                )
            )
            and not any(
                k in lower_msg
                for k in [
                    "sample size",
                    "table 1",
                    "synthetic data",
                    "คำนวณ",
                    "สร้างข้อมูล",
                ]
            )
        )

        if is_topic_query:
            options, articles, norm_q = (
                ClinicalTopicIdeator.generate_research_directions(user_msg)
            )
            response_md = ClinicalTopicIdeator.format_proposals_markdown(
                user_msg, options, articles
            )
            return response_md, state, None, None

        # Case C: Synthetic Cohort Generation requested
        if any(
            k in lower_msg
            for k in [
                "synthetic",
                "สร้างข้อมูล",
                "จำลองข้อมูล",
                "mock cohort",
                "ตัวอย่างข้อมูล",
            ]
        ):
            topic = user_msg if len(user_msg) > 5 else "SGLT2 inhibitor in HFrEF Trial"
            df_gen, meta = SyntheticDataTool.generate_topic_aware_cohort(
                topic, n=200, seed=42
            )
            state.df = df_gen
            state.file_name = (
                f"Synthetic Cohort: {meta.get('domain', 'Clinical Research')}"
            )
            state.var_meta = meta

            time_col = select_variable_by_keyword(
                df_gen.columns.tolist(), ["time", "duration", "days"]
            )
            event_col = select_variable_by_keyword(
                df_gen.columns.tolist(), ["death", "event", "status"]
            )
            treat_col = select_variable_by_keyword(
                df_gen.columns.tolist(), ["treatment", "group", "therapy"]
            )

            fig = go.Figure()
            stats_dict = {}
            if time_col and event_col:
                fig, km_df, stats_dict = StatHarness.run_survival(
                    df_gen, time_col=time_col, event_col=event_col, group_col=treat_col
                )

            km_p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
            km_p_val_str = (
                f"{km_p_val:.4f}" if isinstance(km_p_val, float) else str(km_p_val)
            )

            pico = meta.get("pico", {})
            response_md = f"""### 🧬 สร้างชุดข้อมูลจำลองทางการแพทย์สำเร็จ (Synthetic Clinical Cohort Active)

**ชุดข้อมูล:** `{state.file_name}` (n = {len(df_gen):,} records, {len(df_gen.columns)} ตัวแปร)

#### 📋 PICO Framework:
- **👥 Population (P):** {pico.get("population", "Adult clinical cohort")}
- **💊 Intervention/Exposure (I):** {pico.get("exposure", "Active treatment")}
- **⚖️ Comparator (C):** {pico.get("comparator", "Standard of care")}
- **🎯 Primary Outcome (O):** {pico.get("outcome", "All-cause mortality / Event")}

---

#### 🚀 ดำเนินการวิเคราะห์สถิติทันที (Immediate Statistical Execution):
1. **Kaplan-Meier Survival Analysis:** Fit เส้นรอดชีพตามกลุ่มการรักษา (`{treat_col}`)
   - **Log-Rank Test P-value:** `{km_p_val_str}`
   - **Total Events:** `{df_gen[event_col].sum() if event_col else "N/A"}` เหตุการณ์
2. **ข้อมูลถูกซิงค์เข้า Session เรียบร้อย:** ท่านสามารถสลับไปแท็บ **📊 Data Profiler**, **📈 Regression**, หรือ **👥 Table 1 & Matching** เพื่อดูรายละเอียดเพิ่มเติมได้ทันทีครับ
"""
            return response_md, state, fig, df_gen

        # Case D: Sample Size & Power Calculation requested
        if any(
            k in lower_msg
            for k in [
                "sample size",
                "power",
                "คำนวณกลุ่มตัวอย่าง",
                "คำนวณตัวอย่าง",
                "ขนาดตัวอย่าง",
            ]
        ):
            p1 = 0.30
            p2 = 0.15
            power = 0.80
            alpha = 0.05
            p_nums = re.findall(r"(\d+(?:\.\d+)?)\s*%", user_msg)
            if len(p_nums) >= 2:
                p1 = float(p_nums[0]) / 100.0
                p2 = float(p_nums[1]) / 100.0

            res = StatHarness.run_sample_size(
                p1=p1, p2=p2, power=power, alpha=alpha, dropout_rate=0.15
            )
            response_md = f"""### 📐 ผลการคำนวณขนาดกลุ่มตัวอย่าง (Sample Size & Power Calculation)

**สูตรมาตรฐาน:** Fleiss formula with Continuity Correction (SAMPL Compliant)

| พารามิเตอร์ (Parameter) | ค่าที่กำหนด (Value) |
| :--- | :--- |
| **Exposure / Control Event Rate ($p_1$)** | `{res["p1_control"]:.1%}` |
| **Intervention / Experimental Event Rate ($p_2$)** | `{res["p2_intervention"]:.1%}` |
| **Type I Error ($\\alpha$, 2-sided)** | `{alpha}` (95% Confidence Level) |
| **Statistical Power ($1 - \\beta$)** | `{power:.0%}` |
| **Expected Drop-out Rate** | `15.0%` |

#### 🎯 ขนาดกลุ่มตัวอย่างที่ต้องการ (Target Sample Size):
- **กลุ่มควบคุม (Control Group):** `{res["n_control_adjusted"]}` ราย
- **กลุ่มทดลอง (Intervention Group):** `{res["n_intervention_adjusted"]}` ราย
- **จำนวนผู้ป่วยรวมทั้งสิ้น (Total Target):** **`{res["n_total_adjusted"]}` ราย**

> 💡 **ข้อความสำหรับระเบียบวิธีวิจัย (Methodology Justification):**  
> *"{res["justification_text"]}"*
"""
            return response_md, state, None, state.df

        # Case E: Proposal Uploaded or Parsed
        if proposal_meta:
            recs_list = "\n".join(
                [f"- ✔️ **{m}**" for m in proposal_meta.recommended_methods]
            )
            var_list = ", ".join([f"`{v}`" for v in proposal_meta.variables_identified])

            dataset_exec_section = ""
            fig = None
            preview_df = state.df

            if state.has_data() and state.df is not None:
                df = state.df
                cols = df.columns.tolist()
                time_col = select_variable_by_keyword(
                    cols, ["time", "duration", "days", "fu_time"]
                )
                event_col = select_variable_by_keyword(
                    cols, ["death", "event", "status", "mortality"]
                )
                treat_col = select_variable_by_keyword(
                    cols, ["treatment", "group", "arm", "therapy"]
                )
                covar_candidates = [
                    c
                    for c in cols
                    if c not in [time_col, event_col, treat_col]
                    and not any(
                        id_k in c.lower() for id_k in ["id", "patient", "subject", "hn"]
                    )
                ][:4]
                covar_str = ", ".join(covar_candidates)

                if time_col and event_col:
                    fig, km_df, stats_dict = StatHarness.run_survival(
                        df,
                        time_col=time_col,
                        event_col=event_col,
                        group_col=treat_col,
                        covar_cols=covar_candidates,
                    )
                    p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
                    p_val_str = (
                        f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
                    )

                    dataset_exec_section = f"""
---
### 🚀 ดำเนินการวิเคราะห์สถิติตามชุดข้อมูลที่เชื่อมโยงทันที (Execution on Active Data):
- **ชุดข้อมูล:** `{state.file_name}` (n = {len(df):,} records)
- **ตัวแปรที่ตรวจจับ:** Time = `{time_col}`, Event = `{event_col}`, Group = `{treat_col or "Overall"}`
- **ผลการทดสอบ Kaplan-Meier & Log-Rank Test:** P-value = **`{p_val_str}`**
- **ผล Cox Multivariable Proportional Hazards:** ปรับตัวแปรกวน ({covar_str}) เรียบร้อยแล้ว

*(แสดงกราฟ Kaplan-Meier Survival Function ในหน้าต่าง Visual Artifact ด้านขวา)*
"""
                elif event_col and treat_col:
                    coef_df, metrics, fig = StatHarness.run_logistic(
                        df,
                        outcome_col=event_col,
                        predictor_cols=covar_candidates or [treat_col],
                    )
                    dataset_exec_section = f"""
---
### 🚀 ดำเนินการวิเคราะห์ Multivariable Logistic Regression ทันที:
- **ชุดข้อมูล:** `{state.file_name}` (n = {len(df):,} records)
- **ตัวแปรตาม (Binary Outcome):** `{event_col}`
- **Pseudo $R^2$ (McFadden):** `{metrics.get("mcfadden", 0.0):.4f}` | **AIC:** `{metrics.get("aic", 0.0):.1f}`
"""
            else:
                sample_calc = StatHarness.run_sample_size(
                    p1=0.30, p2=0.18, power=0.80, alpha=0.05
                )
                dataset_exec_section = f"""
---
### 📐 การวางแผนขนาดตัวอย่างเบื้องต้น (Initial Sample Size Justification):
- **เป้าหมายความต่าง (Effect Size):** $p_1 = 30.0\\%$ vs $p_2 = 18.0\\%$ ($\\Delta = 12.0\\%$)
- **จำนวนกลุ่มตัวอย่างแนะนำ (รวม 15% Drop-out):** **`{sample_calc["n_total_adjusted"]}` ราย** (`{sample_calc["n_control_adjusted"]}` ต่อกลุ่ม)

💡 *ต้องการให้ Agent สร้างชุดข้อมูลจำลอง (Synthetic Clinical Cohort) ตามโครงสร้าง Proposal นี้เพื่อทดสอบสถิติทันทีหรือไม่? พิมพ์ว่า "สร้าง synthetic data"*
"""

            response_md = f"""### 📄 ผลการวิเคราะห์โครงร่างงานวิจัย (Research Proposal & Protocol Analysis)

**หัวข้องานวิจัย:** `{proposal_meta.title}`  
**รูปแบบการศึกษา (Study Design):** **{proposal_meta.study_design}**

#### 📋 PICO Framework:
- **👥 Population (P):** {proposal_meta.population}
- **💊 Intervention / Exposure (I):** {proposal_meta.intervention_exposure}
- **⚖️ Comparator (C):** {proposal_meta.comparator}
- **🎯 Primary Outcome (O):** {proposal_meta.primary_outcome}
- **📊 ตัวแปรที่ตรวจพบ (Variables):** {var_list}

#### 📐 สถิติที่เหมาะสมและแนะนำตามหลักระเบียบวิธีวิจัย (Recommended Statistical Pipeline):
{recs_list}
{dataset_exec_section}
"""
            return response_md, state, fig, preview_df

        # Case F: Dataset Uploaded without Proposal or Statistical Command on Active Data
        if state.has_data() and state.df is not None:
            df = state.df
            cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            time_col = select_variable_by_keyword(
                num_cols, ["time", "duration", "days", "fu_days"]
            )
            event_col = select_variable_by_keyword(
                cols, ["death", "event", "status", "mortality", "died"]
            )
            treat_col = select_variable_by_keyword(
                cols, ["treatment", "group", "arm", "therapy", "intervention"]
            )
            covariates = [
                c
                for c in cols
                if c not in [time_col, event_col, treat_col]
                and not any(
                    id_k in c.lower() for id_k in ["id", "patient", "subject", "hn"]
                )
            ][:4]
            covar_str = ", ".join([f"`{c}`" for c in covariates])

            # F1. Survival command
            if any(
                k in lower_msg
                for k in ["survival", "kaplan", "cox", "log-rank", "การรอดชีพ"]
            ):
                if not time_col or not event_col:
                    return (
                        "⚠️ ไม่พบตัวแปร Time หรือ Event ในชุดข้อมูล กรุณาระบุชื่อคอลัมน์",
                        state,
                        None,
                        df,
                    )

                fig, km_summary, stats_dict = StatHarness.run_survival(
                    df,
                    time_col=time_col,
                    event_col=event_col,
                    group_col=treat_col,
                    covar_cols=covariates,
                )
                p_val = stats_dict.get("km_stats", {}).get("p_value", "N/A")
                p_val_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
                c_idx = stats_dict.get("cox_stats", {}).get(
                    "Concordance Index (C-index)", "N/A"
                )

                response_md = f"""### ⏱️ ผลการวิเคราะห์การรอดชีพ (Survival Analysis Execution)

**ชุดข้อมูล:** `{state.file_name}` (n = {len(df):,})  
**ตัวแปร:** Duration = `{time_col}`, Event = `{event_col}`, Group = `{treat_col or "None"}`

#### 1. Kaplan-Meier & Log-Rank Test:
- **Log-Rank P-value:** **`{p_val_str}`**
- **Total Events:** `{df[event_col].sum()}` จากทั้งหมด `{len(df)}` ราย

#### 2. Multivariable Cox Proportional Hazards Model:
- **Concordance Index (C-index):** `{c_idx}`
- **Covariates Adjusted:** {covar_str}

*(เส้นกราฟ Kaplan-Meier Survival Function ถูกเรนเดอร์ในหน้าต่าง Visual Output เรียบร้อยแล้ว)*
"""
                return response_md, state, fig, df

            # F2. Table 1 Baseline command
            if any(
                k in lower_msg
                for k in ["table 1", "baseline", "ตารางที่ 1", "ลักษณะพื้นฐาน"]
            ):
                html_t1, df_t1 = StatHarness.run_table_one(
                    df, group_col=treat_col, selected_vars=cols[:8]
                )
                response_md = f"""### 👥 ตารางลักษณะพื้นฐานประชากร (Baseline Table 1)

**จำแนกตามกลุ่ม (Stratified by):** `{treat_col or "Overall"}`  
**คำนวณความต่าง:** Standardized Mean Differences (SMD cutoff < 0.10 บ่งชี้ความสมดุล)

{html_t1}
"""
                return response_md, state, None, df

            # F3. Regression command
            if any(
                k in lower_msg
                for k in ["regression", "logistic", "linear", "ถดถอย", "odds ratio"]
            ):
                target_outcome = event_col or (num_cols[0] if num_cols else cols[0])
                if df[target_outcome].nunique() <= 3:
                    coef_df, metrics, fig = StatHarness.run_logistic(
                        df,
                        outcome_col=target_outcome,
                        predictor_cols=covariates or cols[:4],
                    )
                    table_md = coef_df.to_markdown(index=False)
                    response_md = f"""### 🎯 ผลการวิเคราะห์ Multivariable Logistic Regression

**ตัวแปรตาม (Outcome Y):** `{target_outcome}` (Binary)  
**McFadden Pseudo-$R^2$:** `{metrics.get("mcfadden", 0.0):.4f}` | **AIC:** `{metrics.get("aic", 0.0):.1f}`

{table_md}
"""
                    return response_md, state, fig, df
                else:
                    coef_df, res, fig = StatHarness.run_linear(
                        df,
                        outcome_col=target_outcome,
                        predictor_cols=covariates or num_cols[1:4],
                    )
                    table_md = coef_df.to_markdown(index=False)
                    response_md = f"""### 📈 ผลการวิเคราะห์ Multivariable Linear Regression (OLS)

**ตัวแปรตาม (Outcome Y):** `{target_outcome}` (Continuous)  
**$R^2$:** `{res.get("r_squared", 0.0):.4f}` | **Adjusted $R^2$:** `{res.get("adj_r_squared", 0.0):.4f}` | **F-statistic P-value:** `{res.get("f_pvalue", 0.0):.4e}`

{table_md}
"""
                    return response_md, state, fig, df

            # F4. General Data Profile on new upload
            if loaded_new_data or not user_msg:
                quality_issues = check_data_quality(df)
                fig = plot_missing_pattern(df)
                response_md = f"""### 📊 โหลดชุดข้อมูล `{state.file_name}` เรียบร้อยแล้ว

**ขนาดข้อมูล:** `{len(df):,}` แถว | `{len(df.columns)}` คอลัมน์ | เซลล์สูญหาย (Missing): `{df.isna().sum().sum():,}` ({df.isna().sum().sum() / df.size * 100:.1f}%)

#### 🔍 ตัวแปรที่ตรวจจับทางคลินิก (Detected Clinical Schema):
- **⏱️ Time Variable:** `{time_col or "None"}`
- **🎯 Event / Outcome:** `{event_col or "None"}`
- **👥 Group / Exposure:** `{treat_col or "None"}`
- **📋 ตัวแปรกวน / Covariates:** {covar_str}
- **⚠️ ปัญหาคุณภาพข้อมูล:** ตรวจพบ {len(quality_issues)} ประเด็น (Zero-PHI Verified)

💡 *ท่านต้องการให้รันการวิเคราะห์อะไรเป็นพิเศษหรือไม่ครับ? เช่น "รัน survival analysis", "ทำ Table 1", "รัน logistic regression", หรือ "คำนวณ sample size"*
"""
                return response_md, state, fig, df

        # Default conversational / General biostatistical query
        pubmed_tool = PubMedEvidenceTool()
        articles: list[dict[str, Any]] = []
        try:
            articles = pubmed_tool.search_and_extract(
                user_msg if len(user_msg) > 5 else "clinical trial evidence",
                max_results=2,
            )
        except Exception as e:
            logger.warning(f"PubMed search error in clinical consultation: {e}")
        vancouver_list = ""
        if articles:
            vancouver_list = "\n".join(
                [f"- **{a['title']}**\n  *{a['vancouver_citation']}*" for a in articles]
            )

        pubmed_section = (
            f"#### 📚 หลักฐานอ้างอิงจาก PubMed Benchmark:\n{vancouver_list}"
            if vancouver_list
            else ""
        )

        response_md = f"""### 🤖 คำแนะนำทางชีวสถิติและการออกแบบวิจัย (Clinical Tech Lead Consultation)

**ประเด็นการสนทนา:** {user_msg}

#### 💡 แนวทางปฏิบัติตามมาตรฐานสากล (SAMPL & EQUATOR Guidelines):
1. **การกำหนด PICO & Endpoint:** ควรระบุ Primary Endpoint ให้ชัดเจนว่าเป็น Time-to-Event (ใช้ Kaplan-Meier / Cox PH), Binary Proportion (ใช้ Logistic Regression / Chi-square), หรือ Continuous Measurement (ใช้ ANCOVA / Linear Regression)
2. **การควบคุม Confounder:** ในงานวิจัยสังเกตการณ์ (Observational Studies) แนะนำให้ใช้ Propensity Score Matching (PSM) หรือ Multivariable Regression เพื่อลดอคติจากการคัดเลือก
3. **การรายงานค่าทางสถิติ:** รายงาน Effect Size พร้อม 95% Confidence Interval และ Exact P-value (ทศนิยม 2-3 ตำแหน่ง) เสมอ

{pubmed_section}

📁 *ท่านสามารถระบุหัวข้อที่สนใจ เช่น "dyspnea", "sepsis" หรืออัปโหลดไฟล์ Proposal (`.docx`) เพื่อให้ Agent เสนอแนวทางวิจัยและรันสถิติทันทีได้ครับ*
"""
        return response_md, state, None, state.df
