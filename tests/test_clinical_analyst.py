import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter

from agent.clinical_analyst import ClinicalAnalystEngine, StatHarness
from agent.critique_engine import CritiqueEngine
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
    assert metrics["used_example_counts"] is False
    assert round(metrics["sensitivity"], 4) == round(80 / 90, 4)
    assert round(metrics["specificity"], 4) == round(90 / 110, 4)
    assert round(metrics["ppv"], 4) == round(80 / 100, 4)
    assert round(metrics["npv"], 4) == round(90 / 100, 4)
    assert round(metrics["pre_test_prob"], 2) == 45.0
    assert fig is not None


def test_stat_harness_diagnostic_strict_validation():
    import pytest

    # When no df or explicit counts are provided, ValueError must be raised
    with pytest.raises(ValueError, match="No valid data provided"):
        StatHarness.run_diagnostic()

    # When counts are partially specified, ValueError must be raised
    with pytest.raises(ValueError, match="Incomplete 2x2 contingency matrix"):
        StatHarness.run_diagnostic(tp=50)

    with pytest.raises(ValueError, match="Incomplete 2x2 contingency matrix"):
        StatHarness.run_diagnostic(tp=50, fp=10, fn=5)

    # When negative counts are specified, ValueError must be raised
    with pytest.raises(ValueError, match="cannot be negative"):
        StatHarness.run_diagnostic(tp=-5, fp=10, fn=5, tn=20)

    # When fractional or non-integer counts are specified, ValueError must be raised without silent truncation
    with pytest.raises(ValueError, match="must be an integer"):
        StatHarness.run_diagnostic(tp=-0.5, fp=10, fn=5, tn=20)

    with pytest.raises(ValueError, match="must be an integer"):
        StatHarness.run_diagnostic(tp=1.9, fp=10, fn=5, tn=20)

    with pytest.raises(ValueError, match="must be an integer"):
        StatHarness.run_diagnostic(tp=True, fp=10, fn=5, tn=20)

    # When all 4 counts sum to zero, ValueError must be raised
    with pytest.raises(ValueError, match="cannot be zero"):
        StatHarness.run_diagnostic(tp=0, fp=0, fn=0, tn=0)

    # When all 4 explicit counts are provided, it must succeed with used_example_counts=False
    _metrics_df2, metrics2, _fig2 = StatHarness.run_diagnostic(tp=50, fp=5, fn=5, tn=50)
    assert metrics2["used_example_counts"] is False
    assert metrics2["tp"] == 50
    assert metrics2["fp"] == 5
    assert metrics2["fn"] == 5
    assert metrics2["tn"] == 50


def test_stat_harness_diagnostic_haldane_anscombe_continuity():
    # Test degenerate case: Perfect specificity (spec = 1.0, fp = 0)
    # Haldane-Anscombe correction should prevent LR+ from collapsing to 1.0 (uninformative)
    _metrics_df, metrics, _fig = StatHarness.run_diagnostic(
        tp=100, fp=0, fn=10, tn=100, pre_test_prob=50.0
    )
    assert metrics["specificity"] == 1.0
    assert metrics["lr_pos"] > 10.0  # Highly informative positive test
    assert metrics["post_prob_pos"] > 50.0  # Post-test probability increases

    # Test degenerate case: Perfect sensitivity (sens = 1.0, fn = 0)
    _metrics_df2, metrics2, _fig2 = StatHarness.run_diagnostic(
        tp=100, fp=10, fn=0, tn=100, pre_test_prob=50.0
    )
    assert metrics2["sensitivity"] == 1.0
    assert metrics2["lr_neg"] < 0.1  # Highly informative negative test
    assert metrics2["post_prob_neg"] < 50.0  # Post-test probability decreases


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
    assert "Sample Size" in response_md
    assert "Fleiss" in response_md
    assert "35" in response_md and "%" in response_md
    assert "18" in response_md and "%" in response_md
    assert (
        "\\alpha" in response_md or "alpha" in response_md.lower() or "α" in response_md
    )


def test_logistic_epv_with_predictor_missingness():
    # 100 subjects: 25 events, 75 non-events in full df
    # 20 out of 25 event cases have missing biomarker -> listwise deletion drops them
    # Fitted cohort has only 5 events -> for 2 predictors, fitted EPV = 5/2 = 2.5 (Severe EPV Deficit < 5)
    np.random.seed(42)
    n = 100
    y = np.array([1] * 25 + [0] * 75)
    age = np.random.normal(55, 10, n)
    biomarker = np.random.normal(10, 2, n)
    # Introduce NaN in biomarker for 20 event cases
    biomarker[:20] = np.nan

    df = pd.DataFrame({"death": y, "age": age, "biomarker": biomarker})

    coef_df, metrics, fig = StatHarness.run_logistic(
        df=df, outcome_col="death", predictor_cols=["age", "biomarker"]
    )
    assert metrics["fitted_events"] == 5
    assert metrics["fitted_non_events"] == 75
    assert metrics["n_clean"] == 80

    verdict = CritiqueEngine.appraise_analysis(
        "logistic",
        df=df,
        results_meta={
            "outcome_col": "death",
            "predictor_cols": ["age", "biomarker"],
            "coef_df": coef_df,
            "fitted_events": metrics["fitted_events"],
            "fitted_non_events": metrics["fitted_non_events"],
        },
    )

    # EPV should be assessed against fitted cohort (5 events / 2 predictors = 2.5 -> HIGH risk)
    epv_finding = next((f for f in verdict.findings if f.category == "EPV"), None)
    assert epv_finding is not None
    assert epv_finding.severity == "HIGH"
    assert "Severe EPV Deficit" in epv_finding.title
    assert "2.5" in epv_finding.title or "5 effective events" in epv_finding.description
    assert verdict.overall_status == "HIGH_RISK_BIAS"

    # Turn processing preserves critique in state
    state = AppState(df=df, file_name="epv_test.csv")
    resp_md, new_state, _, _ = ClinicalAnalystEngine.process_turn(
        user_message="logistic regression death on age and biomarker",
        file_paths=None,
        state=state,
    )
    assert new_state.last_analysis_type == "logistic"
    assert new_state.last_critique_md is not None
    assert "Severe EPV Deficit" in new_state.last_critique_md


def test_dataset_reload_clears_cached_analysis_state(tmp_path):
    # Setup state with previous analysis results and matched cohort
    df_old = pd.DataFrame({"old_x": [1, 2, 3], "old_y": [0, 1, 0]})
    df_matched_old = pd.DataFrame({"old_x": [1, 2], "old_y": [0, 1]})
    state = AppState(
        df=df_old,
        file_name="old_cohort.csv",
        df_matched=df_matched_old,
        is_matched=True,
        last_analysis_type="survival",
        last_analysis_results={"km_stats": {"p_value": 0.04}},
        last_critique_md="### Previous Appraisal",
    )

    # Ingest a new CSV dataset
    new_csv_path = tmp_path / "new_dataset.csv"
    pd.DataFrame(
        {"time": [10, 20, 30], "death": [1, 0, 1], "treatment": [1, 0, 1]}
    ).to_csv(new_csv_path, index=False)

    resp_md, new_state, _fig, _df = ClinicalAnalystEngine.process_turn(
        user_message="",
        file_paths=[str(new_csv_path)],
        state=state,
    )

    assert new_state.file_name == "new_dataset.csv"
    assert new_state.is_matched is False
    assert new_state.df_matched is None
    assert new_state.last_analysis_type is None
    assert new_state.last_analysis_results == {}
    assert new_state.last_critique_md is None


def test_synthetic_cohort_and_proposal_inspector_sync():
    # 1. Synthetic survival (Case C)
    state1 = AppState()
    msg_synth = (
        "สร้าง synthetic cohort สำหรับการทดลองรักษา SGLT2 inhibitor vs Standard care"
    )
    resp1, new_state1, _fig1, _ = ClinicalAnalystEngine.process_turn(
        user_message=msg_synth,
        file_paths=None,
        state=state1,
    )
    assert new_state1.last_analysis_type == "survival"
    assert new_state1.last_analysis_results is not None
    assert "km_stats" in new_state1.last_analysis_results
    assert new_state1.last_critique_md is not None
    assert "Automated Clinical Critique & Appraisal" in new_state1.last_critique_md
    assert "Automated Clinical Critique & Appraisal" in resp1

    # 2. Proposal-triggered survival on active dataset (Case E)
    df_surv = pd.DataFrame(
        {
            "time": [10, 20, 30, 40, 50] * 20,
            "death": [1, 0, 1, 0, 1] * 20,
            "treatment": [1, 1, 0, 0, 1] * 20,
            "age": np.random.normal(60, 10, 100),
        }
    )
    state2 = AppState(df=df_surv, file_name="active_trial.csv")
    proposal_text = "PICO Framework\nPopulation: Heart failure patients\nIntervention: SGLT2i\nComparator: Placebo\nPrimary Outcome: All-cause mortality time to event"
    resp2, new_state2, _fig2, _ = ClinicalAnalystEngine.process_turn(
        user_message=proposal_text,
        file_paths=None,
        state=state2,
    )
    assert new_state2.last_analysis_type == "survival"
    assert new_state2.last_critique_md is not None
    assert "Automated Clinical Critique & Appraisal" in resp2

    # 3. Proposal-triggered logistic regression on active dataset (Case E)
    df_logit = pd.DataFrame(
        {
            "event": [1, 0, 1, 0, 1] * 20,
            "treatment": [1, 1, 0, 0, 1] * 20,
            "age": np.random.normal(60, 10, 100),
        }
    )
    state3 = AppState(df=df_logit, file_name="binary_trial.csv")
    resp3, new_state3, _fig3, _ = ClinicalAnalystEngine.process_turn(
        user_message=proposal_text,
        file_paths=None,
        state=state3,
    )
    assert new_state3.last_analysis_type == "logistic"
    assert new_state3.last_critique_md is not None
    assert "Automated Clinical Critique & Appraisal" in resp3


def test_coerce_to_binary_series_extended():
    from agent.clinical_analyst import _coerce_to_binary_series

    # Numeric 1/2 coding without explicit positive_val must raise ValueError (prohibits inference from numeric ordering)
    s1 = pd.Series([1, 2, 1, 2, 2])
    with pytest.raises(
        ValueError, match="Binary outcome column contains non-standard numeric values"
    ):
        _coerce_to_binary_series(s1)

    # With explicit positive_val=2 (e.g. 2=Event, 1=Censored)
    res1_p2 = _coerce_to_binary_series(s1, positive_val=2)
    assert list(res1_p2) == [0, 1, 0, 1, 1]

    # With explicit positive_val=1 (e.g. 1=Event, 2=Censored)
    res1_p1 = _coerce_to_binary_series(s1, positive_val=1)
    assert list(res1_p1) == [1, 0, 1, 0, 0]

    # Negative/Positive numeric without positive_val must raise ValueError
    s2 = pd.Series([-1, 1, -1, 1])
    with pytest.raises(
        ValueError, match="Binary outcome column contains non-standard numeric values"
    ):
        _coerce_to_binary_series(s2)

    # Negative/Positive numeric with explicit positive_val=1
    res2 = _coerce_to_binary_series(s2, positive_val=1)
    assert list(res2) == [0, 1, 0, 1]

    # Standard 0/1 (explicit 0/1 outcomes preserved automatically)
    s3 = pd.Series([0, 1, 0, 1])
    res3 = _coerce_to_binary_series(s3)
    assert list(res3) == [0, 1, 0, 1]

    # Clinical string tokens: Dead / Alive
    s_dead = pd.Series(["Dead", "Alive", "Dead", "Alive"])
    res_dead = _coerce_to_binary_series(s_dead)
    assert list(res_dead) == [1, 0, 1, 0]

    # Clinical string tokens: Death / Survived
    s_death = pd.Series(["Death", "Survived", "Death"])
    res_death = _coerce_to_binary_series(s_death)
    assert list(res_death) == [1, 0, 1]

    # Clinical string tokens: Case / Control
    s_case = pd.Series(["Control", "Case", "Control"])
    res_case = _coerce_to_binary_series(s_case)
    assert list(res_case) == [0, 1, 0]

    # Ambiguous strings without positive_val must raise ValueError
    s_ambig = pd.Series(["Cohort_A", "Cohort_B", "Cohort_A"])
    with pytest.raises(
        ValueError,
        match="Binary outcome column contains unrecognized category values",
    ):
        _coerce_to_binary_series(s_ambig)

    # Ambiguous strings with explicit positive_val
    res_ambig = _coerce_to_binary_series(s_ambig, positive_val="Cohort_B")
    assert list(res_ambig) == [0, 1, 0]


def test_run_survival_with_non_standard_binary_event():
    np.random.seed(42)
    n = 60
    # event_col encoded as 1 (censored/alive) and 2 (dead)
    df = pd.DataFrame(
        {
            "time": np.random.uniform(10, 100, n),
            "status": [1, 2] * (n // 2),
            "treatment": [0, 1] * (n // 2),
            "age": np.random.normal(55, 8, n),
        }
    )

    # Without positive_val, non-0/1 numeric raises ValueError for clinical safety
    with pytest.raises(
        ValueError, match="Binary outcome column contains non-standard numeric values"
    ):
        StatHarness.run_survival(
            df=df,
            time_col="time",
            event_col="status",
            group_col="treatment",
            covar_cols=["age"],
        )

    # With explicit positive_val=2 (dead=2 is event, alive=1 is censored)
    fig, summary, meta = StatHarness.run_survival(
        df=df,
        time_col="time",
        event_col="status",
        group_col="treatment",
        covar_cols=["age"],
        positive_val=2,
    )

    assert meta["fitted_events"] == 30
    assert meta["fitted_non_events"] == 30
    assert meta["fitted_non_events"] >= 0
    assert meta["km_stats"]["km_events"] == 30
    assert meta["km_stats"]["km_censored"] == 30

    # Regression assertion: lifelines Kaplan-Meier event count matches fitted_events
    # (verifying that lifelines received normalized 0/1 binary event series, avoiding treating 1 and 2 as all observed events)
    kmf = KaplanMeierFitter()
    norm_status = (df["status"] == 2).astype(int)
    kmf.fit(df["time"], norm_status)
    assert int(kmf.event_observed.sum()) == 30
    assert int(kmf.event_observed.sum()) == meta["fitted_events"]
    assert int((kmf.event_observed == 0).sum()) == meta["fitted_non_events"]


def test_run_logistic_with_string_outcome():
    np.random.seed(42)
    n = 80
    df = pd.DataFrame(
        {
            "mortality": ["Alive", "Dead"] * (n // 2),
            "age": np.random.normal(60, 10, n),
            "biomarker": np.random.normal(5, 2, n),
        }
    )

    coef_df, metrics, fig = StatHarness.run_logistic(
        df=df,
        outcome_col="mortality",
        predictor_cols=["age", "biomarker"],
    )

    assert metrics["fitted_events"] == 40
    assert metrics["fitted_non_events"] == 40
    assert metrics["n_clean"] == 80


def test_app_state_file_size_updated_on_ingest(tmp_path):
    # Create temp CSV file
    csv_file = tmp_path / "test_upload.csv"
    csv_file.write_text("id,age,death\n1,50,0\n2,60,1\n3,70,0\n")
    expected_size = csv_file.stat().st_size

    state = AppState()
    _resp, new_state, _fig, _df = ClinicalAnalystEngine.process_turn(
        user_message="Analyze uploaded dataset",
        file_paths=[str(csv_file)],
        state=state,
    )

    assert new_state.file_name == "test_upload.csv"
    assert new_state.file_size_bytes == expected_size
    assert new_state.file_size_bytes > 0
