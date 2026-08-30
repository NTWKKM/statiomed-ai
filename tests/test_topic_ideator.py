"""
tests/test_topic_ideator.py - Unit tests for Clinical Topic Ideator & Proposal Synthesis
"""

import pytest

from agent.clinical_analyst import ClinicalAnalystEngine
from agent.tools.tool_pubmed import PubMedEvidenceTool
from agent.topic_ideator import ClinicalTopicIdeator, ResearchProposalOption
from core.state import AppState


@pytest.fixture(autouse=True)
def stub_pubmed_search(monkeypatch):
    """Avoid live NCBI HTTP requests in topic ideator unit tests."""
    monkeypatch.setattr(
        PubMedEvidenceTool, "search_and_extract", lambda self, q, max_results=3: []
    )


def test_normalize_query():
    assert "dyspnea" in ClinicalTopicIdeator.normalize_query("dyspnea").lower()
    assert "dyspnea" in ClinicalTopicIdeator.normalize_query("เหนื่อย").lower()
    assert "sepsis" in ClinicalTopicIdeator.normalize_query("sepsis").lower()


def test_generate_research_directions_dyspnea():
    options, _articles, _norm_q = ClinicalTopicIdeator.generate_research_directions(
        "dyspnea"
    )
    assert len(options) == 5
    assert all(isinstance(opt, ResearchProposalOption) for opt in options)

    # Check design diversity
    designs = [opt.design_type for opt in options]
    assert any("Randomized Controlled" in d or "RCT" in d for d in designs)
    assert any("Survival" in d or "Cohort" in d for d in designs)
    assert any("Diagnostic" in d for d in designs)
    assert any("Prediction" in d for d in designs)
    assert any("Comparative" in d or "PSM" in d for d in designs)


def test_format_proposals_markdown():
    options, articles, _norm_q = ClinicalTopicIdeator.generate_research_directions(
        "dyspnea"
    )
    md = ClinicalTopicIdeator.format_proposals_markdown("dyspnea", options, articles)
    assert "ข้อเสนอแนวทางการทำวิจัย" in md
    assert "แนวทางที่ 1" in md
    assert "แนวทางที่ 5" in md
    assert "PICO" in md or "ประชากรศึกษา" in md


def test_clinical_analyst_turn_dyspnea_ideation():
    state = AppState()
    msg = "dyspnea"
    resp_md, _new_state, _fig, _preview_df = ClinicalAnalystEngine.process_turn(
        user_message=msg,
        file_paths=None,
        state=state,
    )
    assert "ข้อเสนอแนวทางการทำวิจัย" in resp_md
    assert "แนวทางที่ 1" in resp_md
    assert "แนวทางที่ 2" in resp_md


def test_clinical_analyst_turn_select_option_1():
    state = AppState()
    msg = "เลือกข้อ 1 รัน RCT primary outcome analysis"
    resp_md, new_state, fig, _preview_df = ClinicalAnalystEngine.process_turn(
        user_message=msg,
        file_paths=None,
        state=state,
    )
    assert new_state.has_data()
    assert "Randomized Controlled Trial" in resp_md or "CONSORT" in resp_md
    assert "Relative Risk" in resp_md or "Risk Difference" in resp_md
    assert fig is not None


def test_clinical_analyst_turn_select_option_2():
    state = AppState()
    msg = "เลือกข้อ 2 สร้าง synthetic data แล้วรัน survival ให้ดู"
    resp_md, new_state, fig, _preview_df = ClinicalAnalystEngine.process_turn(
        user_message=msg,
        file_paths=None,
        state=state,
    )
    assert new_state.has_data()
    assert "Kaplan-Meier" in resp_md
    assert "Option 2" in new_state.file_name or "Cohort" in new_state.file_name
    assert fig is not None


def test_clinical_analyst_turn_select_option_3():
    state = AppState()
    msg = "เลือกข้อ 3 วิเคราะห์ diagnostic accuracy"
    resp_md, new_state, fig, _preview_df = ClinicalAnalystEngine.process_turn(
        user_message=msg,
        file_paths=None,
        state=state,
    )
    assert new_state.has_data()
    assert "Option 3" in new_state.file_name or "Diagnostic" in resp_md
    assert "Sensitivity" in resp_md
    assert "2x2 Matrix Counts" in resp_md
    assert fig is not None


def test_clinical_analyst_turn_select_option_5():
    state = AppState()
    msg = "เลือกข้อ 5 ทำ propensity score matching"
    resp_md, new_state, fig, _preview_df = ClinicalAnalystEngine.process_turn(
        user_message=msg,
        file_paths=None,
        state=state,
    )
    assert new_state.has_data()
    assert "Propensity Score Matching" in resp_md
    assert "Nearest-Neighbor" in resp_md
    assert fig is not None
