"""
agent/topic_ideator.py - Clinical Topic Exploration & Research Direction Engine
================================================================================
Takes broad clinical topics (e.g., dyspnea, sepsis, cardiac arrest, AKI),
retrieves benchmark evidence from PubMed, and synthesizes 5 methodologically
diverse, publication-standard clinical research proposals with PICO & statistical plans.
================================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

from agent.tools.tool_pubmed import PubMedEvidenceTool
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResearchProposalOption:
    """A structured clinical research proposal direction."""

    option_id: int
    title: str
    design_type: str
    design_badge: str
    clinical_rationale: str
    population: str
    intervention_exposure: str
    comparator: str
    primary_outcome: str
    recommended_stats: list[str]
    sample_size_estimate: str
    synthetic_topic_prompt: str


class ClinicalTopicIdeator:
    """
    Ideates and structures 5 publication-standard clinical research directions
    from any broad clinical keyword or medical condition.
    """

    # Keyword normalization dictionary (Thai -> English medical terms)
    TOPIC_SYNONYMS: ClassVar[dict[str, str]] = {
        "dyspnea": "acute dyspnea heart failure COPD",
        "เหนื่อย": "acute dyspnea heart failure COPD",
        "หอบ": "acute dyspnea asthma COPD exacerbation",
        "sepsis": "sepsis septic shock ICU resuscitation mortality",
        "ติดเชื้อ": "sepsis bacteremia septic shock infection",
        "cardiac arrest": "out of hospital cardiac arrest CPR post resuscitation",
        "หัวใจหยุดเต้น": "cardiac arrest CPR resuscitation ROSC",
        "chest pain": "acute coronary syndrome non-ST elevation MI troponin",
        "เจ็บหน้าอก": "acute chest pain emergency department troponin ACS",
        "stroke": "acute ischemic stroke thrombolysis thrombectomy",
        "หลอดเลือดสมอง": "acute stroke ischemic hemorrhagic NIHSS",
        "aki": "acute kidney injury renal replacement therapy biomarker",
        "ไตวาย": "acute kidney injury CKD progression cardiorenal",
        "pneumonia": "community acquired pneumonia severity score CURB-65",
        "ปอดอักเสบ": "severe pneumonia respiratory failure ICU",
        "heart failure": "heart failure with reduced ejection fraction SGLT2i",
        "หัวใจวาย": "acute decompensated heart failure diuretic resistance",
        "trauma": "major trauma massive transfusion protocol damage control",
        "อุบัติเหตุ": "trauma hemorrhagic shock resuscitation",
        "diabetic ketoacidosis": "diabetic ketoacidosis fluid protocol insulin",
        "dka": "diabetic ketoacidosis fluid protocol insulin",
    }

    _pubmed_tool: ClassVar[PubMedEvidenceTool] = PubMedEvidenceTool()
    _pubmed_cache: ClassVar[dict[str, list[dict[str, Any]]]] = {}

    @classmethod
    def normalize_query(cls, query: str) -> str:
        """Normalizes user query to optimal PubMed search string."""
        lower_q = query.lower().strip()
        for k, v in cls.TOPIC_SYNONYMS.items():
            if k in lower_q:
                return v
        # Default cleaning
        cleaned = re.sub(r"[^\w\s]", "", query)
        return cleaned or "emergency medicine clinical trial"

    @classmethod
    def generate_research_directions(
        cls, clinical_topic: str
    ) -> tuple[list[ResearchProposalOption], list[dict[str, Any]], str]:
        """
        Retrieves PubMed evidence and formulates 5 structured research directions.
        """
        norm_query = cls.normalize_query(clinical_topic)
        if norm_query in cls._pubmed_cache:
            articles = cls._pubmed_cache[norm_query]
        else:
            articles = []
            try:
                articles = cls._pubmed_tool.search_and_extract(
                    norm_query, max_results=3
                )
                cls._pubmed_cache[norm_query] = articles
            except Exception as e:
                logger.warning(f"PubMed search error: {e}")

        topic_clean = clinical_topic.strip().title()

        # Build 5 Methodologically Diverse Clinical Proposals
        options = [
            # Option 1: Interventional RCT
            ResearchProposalOption(
                option_id=1,
                title=f"Efficacy of Early Targeted Protocol vs Standard Care in Acute {topic_clean}",
                design_type="Randomized Controlled Trial (RCT)",
                design_badge="💊 Interventional (CONSORT)",
                clinical_rationale=f"Evaluates whether an intensified, protocolized early intervention improves short-term clinical resolution in patients presenting with acute {topic_clean}.",
                population=f"Adult patients presenting to the Emergency Department / Acute Care with acute {topic_clean}",
                intervention_exposure="Early Protocolized Bundle (High-Intensity Active Arm)",
                comparator="Standard Guideline-Directed Care",
                primary_outcome="72-Hour Clinical Deterioration / Treatment Failure Rate (Binary Proportion)",
                recommended_stats=[
                    "Chi-Square Test & Fisher's Exact Test",
                    "Relative Risk (RR) and Risk Difference with 95% CI",
                    "Intention-to-Treat (ITT) and Per-Protocol Sensitivity Analysis",
                    "Fleiss Sample Size with 15% Drop-Out Allowance",
                ],
                sample_size_estimate="n = 280 (140 per group, Power 80%, Alpha 0.05, Delta = 15%)",
                synthetic_topic_prompt=f"RCT of Early Protocol vs Standard Care in Acute {topic_clean}",
            ),
            # Option 2: Prognostic Survival Cohort (Time-to-Event)
            ResearchProposalOption(
                option_id=2,
                title=f"30-Day and 1-Year Survival Predictors in Hospitalized Patients with {topic_clean}",
                design_type="Prospective Observational Cohort Study",
                design_badge="⏱️ Survival (STROBE)",
                clinical_rationale=f"Investigates time-to-event outcomes, physiological trajectories, and independent risk factors of mortality in patients admitted with {topic_clean}.",
                population=f"Inpatients admitted with primary diagnosis of acute/chronic {topic_clean}",
                intervention_exposure="High-Risk Clinical Biomarker / Severe Physiological Subtype",
                comparator="Low-Risk / Standard Physiological Subtype",
                primary_outcome="All-Cause 30-Day and 1-Year Mortality (Time-to-Event Days)",
                recommended_stats=[
                    "Kaplan-Meier Survival Estimation with Log-Rank Test",
                    "Multivariable Cox Proportional Hazards Model (Efron Tie Handling)",
                    "Schoenfeld Residual Proportional Hazards Assumption Check",
                    "Harrell's Concordance Index (C-index) for Model Discrimination",
                ],
                sample_size_estimate="n = 350 (Target >= 85 Survival Events via Schoenfeld Rule)",
                synthetic_topic_prompt=f"Observational Survival Cohort in Hospitalized {topic_clean} Patients",
            ),
            # Option 3: Diagnostic Accuracy Study (POCUS / Novel Biomarkers)
            ResearchProposalOption(
                option_id=3,
                title=f"Diagnostic Accuracy of Point-of-Care Ultrasound (POCUS) vs Conventional Workup in {topic_clean}",
                design_type="Cross-Sectional Diagnostic Accuracy Trial",
                design_badge="🔍 Diagnostic (STARD)",
                clinical_rationale=f"Quantifies the sensitivity, specificity, and Bayesian likelihood ratios of rapid bedside ultrasound/biomarkers in etiology differentiation for {topic_clean}.",
                population=f"Consecutive emergency patients presenting with undifferentiated {topic_clean}",
                intervention_exposure="Rapid Bedside POCUS Protocol (Index Diagnostic Test)",
                comparator="Comprehensive Multi-Specialist Consensus / CT / Expert Adjudication (Reference Standard)",
                primary_outcome="Accurate Etiology Identification within 1 Hour (Binary Diagnostic Matrix)",
                recommended_stats=[
                    "2x2 Diagnostic Matrix (Sensitivity, Specificity, PPV, NPV)",
                    "Positive & Negative Likelihood Ratios (LR+, LR-)",
                    "Receiver Operating Characteristic (ROC) & Area Under Curve (AUC)",
                    "Fagan Bayesian Nomogram for Post-Test Probability",
                ],
                sample_size_estimate="n = 220 (Pre-test probability 35%, Target Sensitivity > 90%)",
                synthetic_topic_prompt=f"Diagnostic POCUS Accuracy Trial in Undifferentiated {topic_clean}",
            ),
            # Option 4: Clinical Prediction Model (Machine Learning & Multivariable Score)
            ResearchProposalOption(
                option_id=4,
                title=f"Development and Validation of a Bedside Risk Score for ICU Transfer in {topic_clean}",
                design_type="Clinical Prediction Model (Derivation & Validation)",
                design_badge="🧠 Prediction (TRIPOD+AI)",
                clinical_rationale=f"Derives a parsimonious multivariable prediction model to stratify risk of early clinical decompensation requiring critical care in {topic_clean}.",
                population=f"Adult patients presenting with acute presentation of {topic_clean}",
                intervention_exposure="Multivariable Clinical Candidate Predictors (Age, Vitals, Labs)",
                comparator="Univariable Risk Assessment / Physician Gestalt",
                primary_outcome="Unplanned ICU Admission or Mechanical Ventilation within 24 Hours",
                recommended_stats=[
                    "Multivariable Binary Logistic Regression (Odds Ratios & 95% CI)",
                    "Calibration Slope and Hosmer-Lemeshow Goodness-of-Fit",
                    "Discrimination via ROC/AUC Analysis (DeLong Test)",
                    "Decision Curve Analysis (Net Clinical Benefit)",
                ],
                sample_size_estimate="n = 500 (Minimum 15-20 Events Per Variable - EPV Rule)",
                synthetic_topic_prompt=f"Prediction Model for ICU Transfer in Emergency {topic_clean}",
            ),
            # Option 5: Comparative Effectiveness with Propensity Score Matching (PSM)
            ResearchProposalOption(
                option_id=5,
                title=f"Real-World Comparative Effectiveness of Therapy A vs Therapy B in {topic_clean}",
                design_type="Retrospective Comparative Cohort Study (PSM)",
                design_badge="👥 Balance (PSM & Table 1)",
                clinical_rationale=f"Employs causal inference and propensity score matching to balance baseline confounding when comparing real-world therapies in {topic_clean}.",
                population=f"Real-world observational cohort receiving treatment for {topic_clean}",
                intervention_exposure="Novel Therapeutic Agent / Early Invasive Management",
                comparator="Conventional Standard Medical Therapy",
                primary_outcome="In-Hospital Length of Stay & 30-Day Readmission",
                recommended_stats=[
                    "Propensity Score 1:1 Nearest Neighbor Matching (Caliper 0.20 SD)",
                    "Love Plot & Standardized Mean Differences (SMD < 0.10 Balance)",
                    "Baseline Characteristics Table 1 Generation",
                    "Matched-Cohort Multivariable Generalized Linear Models (GLM)",
                ],
                sample_size_estimate="n = 600 (300 Treated vs 300 Control before 1:1 Caliper Match)",
                synthetic_topic_prompt=f"Comparative Effectiveness PSM Study in {topic_clean}",
            ),
        ]

        return options, articles, norm_query

    @classmethod
    def format_proposals_markdown(
        cls,
        clinical_topic: str,
        options: list[ResearchProposalOption],
        articles: list[dict[str, Any]],
    ) -> str:
        """Formats the 5 proposals and PubMed evidence into an interactive Markdown response."""
        topic_title = clinical_topic.strip().title()

        pubmed_section = ""
        if articles:
            lines = []
            for a in articles:
                lines.append(f"- **{a['title']}**  \n  *{a['vancouver_citation']}*")
            pubmed_section = f"""#### 📚 หลักฐานเชิงประจักษ์ล่าสุดจาก PubMed (Recent Benchmark Evidence):
{chr(10).join(lines)}

---
"""

        proposals_cards = []
        for opt in options:
            stats_bullets = "\n".join([f"    - ✔️ {s}" for s in opt.recommended_stats])
            card = f"""### 📌 แนวทางที่ {opt.option_id}: {opt.title}
<span style='background:#e0f2fe;color:#0369a1;padding:2px 8px;border-radius:6px;font-size:0.8rem;font-weight:600;'>{opt.design_badge}</span>

- **🎯 วัตถุประสงค์และเหตุผลทางคลินิก:** {opt.clinical_rationale}
- **👥 ประชากรศึกษา (Population):** {opt.population}
- **💊 สิ่งแทรกแซง/ปัจจัยเสี่ยง (Intervention/Exposure):** {opt.intervention_exposure}
- **⚖️ กลุ่มเปรียบเทียบ (Comparator):** {opt.comparator}
- **🎯 ผลลัพธ์หลัก (Primary Endpoint):** `{opt.primary_outcome}`
- **📐 ระเบียบวิธีวิจัยและสถิติที่แนะนำ (Statistical Plan):**
{stats_bullets}
- **📊 ประมาณการขนาดกลุ่มตัวอย่าง (Sample Size Target):** `{opt.sample_size_estimate}`
"""
            proposals_cards.append(card)

        all_cards_md = "\n\n".join(proposals_cards)

        full_md = f"""### 💡 ข้อเสนอแนวทางการทำวิจัยและสถิติ 5 รูปแบบสำหรับ: **"{topic_title}"**

{pubmed_section}
{all_cards_md}

---

### 🚀 เลือกดำเนินการวิเคราะห์สถิติทันที (Immediate Interactive Action):
ท่านสามารถพิมพ์เลือกแนวทางที่ต้องการได้เลยครับ เช่น:
- พิมพ์ **`"เลือกข้อ 2 สร้าง synthetic data แล้วรัน survival ให้ดู"`** ➔ *Agent จะสร้างข้อมูลจำลองและฟิตกราฟ Kaplan-Meier & Cox PH ให้ทันที*
- พิมพ์ **`"คำนวณ sample size ของข้อ 1"`** ➔ *Agent จะคำนวณขนาดตัวอย่างพร้อมสูตร SAMPL*
- พิมพ์ **`"ทำ Table 1 ของข้อ 5"`** ➔ *Agent จะสร้าง Baseline Table 1 พร้อมคำนวณ SMD*
"""
        return full_md
