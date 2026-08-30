"""
tests/test_reporting_checklists.py - Unit Tests for EQUATOR Network Reporting Checklists
"""

from utils.reporting_checklists import (
    auto_populate_consort,
    auto_populate_stard,
    auto_populate_tripod_ai,
    create_consort_checklist,
    create_prisma_checklist,
    create_stard_checklist,
    create_strobe_checklist,
    create_tripod_ai_checklist,
    generate_checklist_markdown,
)


def test_create_equator_checklists():
    consort = create_consort_checklist()
    assert len(consort.items) >= 25
    assert consort.name == "CONSORT 2010"

    strobe = create_strobe_checklist("cohort")
    assert len(strobe.items) >= 22
    assert "STROBE" in strobe.name

    tripod_ai = create_tripod_ai_checklist()
    assert len(tripod_ai.items) >= 20
    assert tripod_ai.name == "TRIPOD+AI 2024"

    stard = create_stard_checklist()
    assert len(stard.items) >= 20
    assert stard.name == "STARD 2015"

    prisma = create_prisma_checklist()
    assert len(prisma.items) >= 20
    assert prisma.name == "PRISMA 2020"


def test_auto_populate_tripod_ai():
    meta = {
        "n_total": 1250,
        "model_name": "Gradient Boosted Survival Ensemble",
        "c_index": 0.842,
        "calibration_slope": 0.98,
        "calibration_intercept": 0.02,
        "brier_score": 0.112,
        "has_dca": True,
    }
    checklist = auto_populate_tripod_ai(meta)
    summary = checklist.get_completion_summary()
    assert summary["complete"] >= 4
    md = generate_checklist_markdown(checklist)
    assert "TRIPOD+AI 2024 Checklist" in md
    assert "C-index: 0.842" in md


def test_auto_populate_stard():
    meta = {
        "tp": 85,
        "fp": 15,
        "fn": 10,
        "tn": 190,
        "sensitivity": 0.895,
        "specificity": 0.927,
        "positive_lr": 12.26,
        "negative_lr": 0.11,
        "auc": 0.952,
        "has_fagan": True,
    }
    checklist = auto_populate_stard(meta)
    summary = checklist.get_completion_summary()
    assert summary["complete"] >= 3
    md = generate_checklist_markdown(checklist)
    assert "STARD 2015 Checklist" in md
    assert "Sensitivity: 89.5%" in md


def test_auto_populate_consort():
    meta = {
        "n_assigned": 500,
        "n_control": 250,
        "n_intervention": 250,
        "has_sample_size": True,
        "primary_effect": "HR 0.68 (95% CI 0.52-0.89), P = 0.005",
    }
    checklist = auto_populate_consort(meta)
    summary = checklist.get_completion_summary()
    assert summary["complete"] >= 3
    assert summary["completion_rate"] > 0
