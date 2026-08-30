"""
tests/test_clinical_analyst.py - Unit tests for Clinical Analyst Engine & Statistical Harness
"""

from agent.clinical_analyst import ClinicalAnalystEngine, StatHarness
from core.state import AppState
from views.view_data import generate_example_dataset


def test_stat_harness_sample_size():
    res = StatHarness.run_sample_size(p1=0.30, p2=0.15, power=0.80, alpha=0.05)
    assert res["n_total_adjusted"] > 0
    assert res["n_control_adjusted"] > 0
    assert "justification_text" in res


def test_stat_harness_survival():
    df, _ = generate_example_dataset()
    fig, km_df, stats_dict = StatHarness.run_survival(
        df=df,
        time_col="time",
        event_col="death",
        group_col="treatment",
        covar_cols=["age", "bmi", "diabetes"],
    )
    assert fig is not None
    assert km_df is not None
    assert "km_stats" in stats_dict
    assert stats_dict["km_stats"]["p_value"] is not None


def test_stat_harness_logistic():
    df, _ = generate_example_dataset()
    coef_df, metrics, _fig = StatHarness.run_logistic(
        df=df,
        outcome_col="death",
        predictor_cols=["age", "bmi", "treatment"],
    )
    assert not coef_df.empty
    assert "Odds Ratio (OR)" in coef_df.columns
    assert "mcfadden" in metrics


def test_stat_harness_table_one():
    df, _ = generate_example_dataset()
    html_table, _df_sub = StatHarness.run_table_one(
        df=df,
        group_col="treatment",
        selected_vars=["age", "bmi", "sbp", "diabetes"],
    )
    assert "<table" in html_table.lower()


def test_stat_harness_diagnostic():
    df, _ = generate_example_dataset()
    metrics_df, metrics, fig = StatHarness.run_diagnostic(
        df, index_test_col="treatment", ref_standard_col="death"
    )
    assert not metrics_df.empty
    assert "sensitivity" in metrics
    assert "tp" in metrics and "tn" in metrics
    assert metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == len(
        df.dropna(subset=["treatment", "death"])
    )
    assert fig is not None


def test_stat_harness_diagnostic_explicit_counts():
    import pandas as pd

    # Construct DataFrame with exactly TP=80, FP=20, FN=10, TN=90
    data = (
        [{"pocus": 1, "gold_dx": 1}] * 80
        + [{"pocus": 1, "gold_dx": 0}] * 20
        + [{"pocus": 0, "gold_dx": 1}] * 10
        + [{"pocus": 0, "gold_dx": 0}] * 90
    )
    df_test = pd.DataFrame(data)

    metrics_df, metrics, fig = StatHarness.run_diagnostic(
        df_test, index_test_col="pocus", ref_standard_col="gold_dx"
    )
    assert metrics["tp"] == 80
    assert metrics["fp"] == 20
    assert metrics["fn"] == 10
    assert metrics["tn"] == 90
    assert round(metrics["sensitivity"], 4) == round(80 / 90, 4)
    assert round(metrics["specificity"], 4) == round(90 / 110, 4)
    assert round(metrics["ppv"], 4) == round(80 / 100, 4)
    assert round(metrics["npv"], 4) == round(90 / 100, 4)
    assert round(metrics["pre_test_prob"], 2) == 45.0
    assert fig is not None


def test_stat_harness_binary_rct():
    df, _ = generate_example_dataset()
    summary_df, metrics, fig = StatHarness.run_binary_rct(
        df=df,
        treatment_col="treatment",
        outcome_col="death",
    )
    assert not summary_df.empty
    assert "relative_risk" in metrics
    assert "risk_diff" in metrics
    assert "chi2_p" in metrics
    assert "n_control" in metrics and "n_intervention" in metrics
    assert metrics["n_control"] + metrics["n_intervention"] == len(
        df.dropna(subset=["treatment", "death"])
    )
    assert fig is not None


def test_stat_harness_psm():
    df, _ = generate_example_dataset()
    balance_df, stats_dict, fig_love, df_matched = StatHarness.run_psm(
        df=df,
        treatment_col="treatment",
        covariate_cols=["age", "bmi", "diabetes"],
        outcome_col="death",
        caliper=0.20,
        ratio=1,
    )
    assert not df_matched.empty
    assert stats_dict["n_matched"] > 0
    assert stats_dict["n_original"] == len(df)
    assert not balance_df.empty
    assert fig_love is not None


def test_clinical_analyst_turn_synthetic_generation():
    state = AppState()
    msg = "สร้าง synthetic cohort สำหรับการทดลองรักษา SGLT2 inhibitor vs Standard care"
    response_md, new_state, _fig, _preview_df = ClinicalAnalystEngine.process_turn(
        user_message=msg,
        file_paths=None,
        state=state,
    )
    assert new_state.has_data()
    assert len(new_state.df) > 0
    assert "PICO" in response_md
    assert "Kaplan-Meier" in response_md


def test_clinical_analyst_turn_sample_size():
    state = AppState()
    msg = "คำนวณ sample size สำหรับ clinical trial 2 กลุ่ม event rate 35% vs 18% power 80%"
    response_md, _new_state, _fig, _preview_df = ClinicalAnalystEngine.process_turn(
        user_message=msg,
        file_paths=None,
        state=state,
    )
    assert "ผลการคำนวณขนาดกลุ่มตัวอย่าง" in response_md
    assert "Fleiss" in response_md
    assert "35" in response_md and "%" in response_md
    assert "18" in response_md and "%" in response_md
    assert (
        "\\alpha" in response_md or "alpha" in response_md.lower() or "α" in response_md
    )
