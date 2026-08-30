"""
tests/test_proposal_parser.py - Unit tests for clinical proposal parser
"""

import zipfile
from pathlib import Path

from utils.proposal_parser import ProposalMetadata, ProposalParser


def create_mock_docx(file_path: Path, paragraphs: list[str]) -> Path:
    """Helper to generate a minimal valid .docx file in memory/disk."""
    w_p_elements = "".join(
        [f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs]
    )
    xml_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body>
            {w_p_elements}
        </w:body>
    </w:document>
    """
    with zipfile.ZipFile(file_path, "w") as z:
        z.writestr("word/document.xml", xml_content)
    return file_path


def test_proposal_parser_from_text():
    raw_text = """
    Clinical Study Protocol: Evaluating SGLT2 Inhibitors in Diabetic Kidney Disease
    Study Design: Prospective Observational Cohort Study
    Population: Adult patients with Type 2 Diabetes and CKD Stage 3-4
    Intervention: Dapagliflozin 10mg daily
    Comparator: Standard glycemic control without SGLT2i
    Primary Outcome: All-cause mortality and 2-year survival (time-to-event)
    Secondary Outcome: Rate of eGFR decline and SBP reduction
    Variables: age, sex, bmi, diabetes, hypertension, ckd, sbp, dbp, egfr, time, death
    """
    meta = ProposalParser.parse_proposal(raw_text)
    assert isinstance(meta, ProposalMetadata)
    assert "SGLT2" in meta.title or "Clinical Study" in meta.title
    assert "Survival" in meta.study_design or "Cohort" in meta.study_design
    assert "Diabetes" in meta.population
    assert "Dapagliflozin" in meta.intervention_exposure
    assert (
        "mortality" in meta.primary_outcome.lower()
        or "survival" in meta.primary_outcome.lower()
    )
    assert any("Kaplan-Meier" in m for m in meta.recommended_methods)
    assert any("Cox" in m for m in meta.recommended_methods)
    assert any("Table 1" in m for m in meta.recommended_methods)


def test_proposal_parser_from_mock_docx(tmp_path):
    docx_file = tmp_path / "sample_proposal.docx"
    paragraphs = [
        "A Randomized Controlled Trial of Novel Antiarrhythmic Drug",
        "Population: Patients with symptomatic paroxysmal atrial fibrillation",
        "Intervention: Drug X 50mg twice daily",
        "Comparator: Placebo standard therapy",
        "Primary Endpoint: Recurrence of AF at 6 months (binary outcome)",
        "Variables: age, sex, bmi, hypertension, recurrence",
    ]
    create_mock_docx(docx_file, paragraphs)

    meta = ProposalParser.parse_proposal(docx_file)
    assert "Randomized Controlled" in meta.study_design or "RCT" in meta.study_design
    assert "atrial fibrillation" in meta.population.lower()
    assert "Drug X" in meta.intervention_exposure
    assert any("Logistic" in m or "Table 1" in m for m in meta.recommended_methods)
