"""
utils/proposal_parser.py - Clinical Research Proposal & Protocol Parser
=============================================================================
Zero-external-dependency parser for Word (.docx), text, and markdown files.
Extracts PICO components, study design, hypothesis, variables, and endpoints
to enable automated biostatistical methodology selection.
=============================================================================
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProposalMetadata:
    """Structured clinical proposal / protocol metadata."""

    title: str = "Clinical Research Proposal"
    raw_text: str = ""
    study_design: str = "Observational Cohort"
    population: str = ""
    intervention_exposure: str = ""
    comparator: str = ""
    primary_outcome: str = ""
    secondary_outcomes: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    variables_identified: list[str] = field(default_factory=list)
    recommended_methods: list[str] = field(default_factory=list)
    sample_size_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "study_design": self.study_design,
            "population": self.population,
            "intervention_exposure": self.intervention_exposure,
            "comparator": self.comparator,
            "primary_outcome": self.primary_outcome,
            "secondary_outcomes": self.secondary_outcomes,
            "hypotheses": self.hypotheses,
            "variables_identified": self.variables_identified,
            "recommended_methods": self.recommended_methods,
            "sample_size_info": self.sample_size_info,
        }


class ProposalParser:
    """
    Parser for research proposals, study protocols, and synopses.
    Supports .docx (via standard zipfile + XML), .txt, .md, and raw text.
    """

    @staticmethod
    def extract_text_from_docx(file_path: str | Path) -> str:
        """
        Extracts plain text from a Microsoft Word .docx file using standard library zipfile + XML.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with zipfile.ZipFile(path) as z:
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)

                # XML namespaces in Word documents
                ns = {
                    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                }
                paragraphs = []
                for p in tree.iterfind(".//w:p", ns):
                    texts = [
                        node.text for node in p.iterfind(".//w:t", ns) if node.text
                    ]
                    if texts:
                        paragraphs.append("".join(texts))

                return "\n\n".join(paragraphs)
        except Exception as e:
            logger.warning(f"Zipfile/XML docx parsing failed: {e}. Trying fallback.")
            try:
                import docx  # type: ignore

                doc = docx.Document(path)
                return "\n\n".join([p.text for p in doc.paragraphs if p.text])
            except Exception as e2:
                logger.error(f"Fallback docx parser also failed: {e2}")
                raise ValueError(f"Could not read .docx file: {e}") from e

    @classmethod
    def extract_text(cls, file_path: str | Path) -> str:
        """
        Extracts raw text based on file suffix (.docx, .txt, .md, etc.).
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in [".docx", ".doc"]:
            try:
                return cls.extract_text_from_docx(path)
            except Exception:
                # If binary .doc or corrupted, attempt text read
                return path.read_text(errors="ignore")
        elif suffix in [".txt", ".md", ".json", ".log"]:
            return path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            try:
                import pypdf  # type: ignore

                reader = pypdf.PdfReader(path)
                pages = [
                    page.extract_text() for page in reader.pages if page.extract_text()
                ]
                return "\n\n".join(pages)
            except Exception as e:
                logger.warning(f"PDF extraction failed: {e}")
                return path.read_text(errors="ignore")
        else:
            return path.read_text(encoding="utf-8", errors="ignore")

    @classmethod
    def parse_proposal(cls, text_or_path: str | Path) -> ProposalMetadata:
        """
        Parses proposal text or file and extracts structured clinical PICO,
        study design, and statistical recommendations.
        """
        raw_text = ""
        if isinstance(text_or_path, (str, Path)) and Path(str(text_or_path)).is_file():
            raw_text = cls.extract_text(text_or_path)
        else:
            raw_text = str(text_or_path)

        meta = ProposalMetadata(raw_text=raw_text)
        if not raw_text.strip():
            return meta

        lower_text = raw_text.lower()

        # 1. Extract Title
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if lines:
            meta.title = lines[0][:150]

        # 2. Identify Study Design
        if any(
            k in lower_text
            for k in [
                "randomized controlled",
                "rct",
                "randomised",
                "clinical trial",
                "ทดลองทางคลินิก",
            ]
        ):
            meta.study_design = "Randomized Controlled Trial (RCT)"
        elif any(
            k in lower_text
            for k in [
                "survival",
                "time to event",
                "mortality",
                "follow-up",
                "cohort",
                "prospective",
                "retrospective",
                "การรอดชีพ",
                "ติดตามผล",
            ]
        ):
            meta.study_design = "Observational Cohort Study (Time-to-Event / Survival)"
        elif any(
            k in lower_text
            for k in [
                "case-control",
                "case control",
                "odds ratio",
                "กลุ่มควบคุม",
            ]
        ):
            meta.study_design = "Case-Control Study"
        elif any(
            k in lower_text
            for k in [
                "cross-sectional",
                "prevalence",
                "survey",
                "ภาคตัดขวาง",
            ]
        ):
            meta.study_design = "Cross-Sectional Study"
        elif any(
            k in lower_text
            for k in [
                "diagnostic",
                "sensitivity",
                "specificity",
                "roc",
                "fagan",
                "ความไว",
                "ความจำเพาะ",
            ]
        ):
            meta.study_design = "Diagnostic Accuracy Study"
        else:
            meta.study_design = "Clinical Cohort Investigation"

        # 3. PICO Extraction via Regex & Patterns
        def extract_field(keywords: list[str], default_val: str) -> str:
            kw_pattern = "|".join(keywords)
            # Try anchored line match first (e.g. "Population: ...")
            m = re.search(
                rf"(?im)^\s*(?:{kw_pattern})\s*[:\-\=]\s*([^\n\r]+)",
                raw_text,
            )
            if m:
                return m.group(1).strip()
            # Try inline match
            m2 = re.search(
                rf"(?:{kw_pattern})\s*[:\-\=]\s*([^\n\.\;]{{5,200}})",
                raw_text,
                re.IGNORECASE,
            )
            if m2:
                return m2.group(1).strip()
            return default_val

        meta.population = extract_field(
            [
                "population",
                "participants",
                "subjects",
                "patients",
                "กลุ่มประชากร",
                "กลุ่มตัวอย่าง",
                "ผู้ป่วย",
            ],
            "Target adult clinical population meeting inclusion criteria",
        )

        meta.intervention_exposure = extract_field(
            [
                "intervention",
                "exposure",
                "treatment",
                "investigational drug",
                "therapy",
                "กลุ่มศึกษา",
                "การรักษา",
                "ปัจจัยเสี่ยง",
            ],
            "Investigational intervention / Active exposure group",
        )

        meta.comparator = extract_field(
            [
                "comparator",
                "control",
                "standard of care",
                "placebo",
                "กลุ่มควบคุม",
                "การรักษามาตรฐาน",
            ],
            "Standard of care / Placebo control",
        )

        meta.primary_outcome = extract_field(
            [
                "primary outcome",
                "primary endpoint",
                "outcome",
                "endpoint",
                "ผลลัพธ์หลัก",
                "ผลลัพธ์",
            ],
            "All-cause mortality / Time-to-event endpoint"
            if "survival" in lower_text or "mortality" in lower_text
            else "Primary clinical effectiveness endpoint",
        )

        # 4. Extract Variables
        var_candidates = [
            "age",
            "sex",
            "gender",
            "bmi",
            "treatment",
            "diabetes",
            "hypertension",
            "ckd",
            "sbp",
            "dbp",
            "egfr",
            "creatinine",
            "crp",
            "cholesterol",
            "time",
            "death",
            "status",
            "event",
            "icu_stay",
            "readmission",
            "mortality",
        ]
        found_vars = [v for v in var_candidates if v in lower_text]
        meta.variables_identified = found_vars or [
            "age",
            "sex",
            "treatment",
            "outcome",
            "time",
            "death",
        ]

        # 5. Determine Recommended Biostatistical Methods
        recommended = []
        recommended.append(
            "Baseline Characteristics Table (Table 1) with Standardized Mean Differences (SMD)"
        )

        if "survival" in meta.study_design.lower() or any(
            k in lower_text
            for k in [
                "survival",
                "time to event",
                "mortality",
                "cox",
                "kaplan-meier",
                "km",
            ]
        ):
            recommended.append("Kaplan-Meier Survival Curves with Log-Rank Test")
            recommended.append(
                "Multivariable Cox Proportional Hazards Model (Efron tie handling & Schoenfeld residual test)"
            )

        if (
            "logistic" in lower_text
            or "binary" in lower_text
            or "case-control" in meta.study_design.lower()
            or "odds ratio" in lower_text
        ):
            recommended.append(
                "Multivariable Logistic Regression (Odds Ratios, 95% CI & McFadden Pseudo-R²)"
            )

        if "continuous" in lower_text or "linear" in lower_text or "mean" in lower_text:
            recommended.append(
                "Multivariable Ordinary Least Squares (OLS) Linear Regression with Homoscedasticity & Normality Diagnostics"
            )

        if (
            "observational" in meta.study_design.lower()
            or "cohort" in meta.study_design.lower()
        ):
            recommended.append(
                "Propensity Score Matching (PSM) with Caliper Balance Assessment (Love Plot)"
            )

        if "diagnostic" in meta.study_design.lower() or "accuracy" in lower_text:
            recommended.append(
                "Diagnostic Accuracy Matrix (Sensitivity, Specificity, PPV, NPV, ROC/AUC, Fagan Nomogram)"
            )

        recommended.append(
            "SAMPL-Compliant Sample Size & Statistical Power Calculation with Drop-Out Adjustment"
        )

        meta.recommended_methods = recommended

        # 6. Sample size defaults
        meta.sample_size_info = {
            "alpha": 0.05,
            "power": 0.80,
            "dropout_rate": 0.15,
            "test_type": "two_proportions"
            if "proportion" in lower_text or "rate" in lower_text
            else "survival_events",
        }

        return meta
