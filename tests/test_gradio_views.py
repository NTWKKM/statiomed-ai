"""
tests/test_gradio_views.py - Unit & Integration Tests for Native Gradio Views
=============================================================================
Verifies that all Gradio view modules, callbacks, and top-level Blocks demo
construct properly with complete component graphs and valid state contracts.
=============================================================================
"""

import pytest

pytest.importorskip("gradio")

from app import build_app
from core.state import AppState
from views.view_ai_copilot import run_ai_copilot_action
from views.view_data import generate_example_dataset, load_example_data_action
from views.view_diagnostic import calculate_2x2_diagnostic
from views.view_meta_analysis import generate_sample_meta_data, run_meta_analysis_action
from views.view_regression import run_regression_analysis
from views.view_sample_size import (
    compute_means_sample_size,
    compute_proportions_sample_size,
    compute_survival_sample_size,
)
from views.view_survival import run_cox_analysis, run_km_analysis
from views.view_table_one_matching import generate_table_one_action, run_psm_action


def test_gradio_app_structure():
    """Verify that top-level Gradio Blocks app compiles and contains blocks."""
    app_demo = build_app()
    assert app_demo is not None
    assert len(app_demo.blocks) > 0


def test_data_view_load_example():
    """Verify example dataset generation and data profiler action."""
    state = AppState()
    state, badge_html, df, fig, report = load_example_data_action(state)

    assert state.df is not None
    assert len(state.df) == 1600
    assert "Active Dataset" in badge_html
    assert df is not None
    assert fig is not None


def test_ai_copilot_synthetic_cohort():
    """Verify AI Co-Pilot synthetic cohort generation action."""
    state = AppState()
    html_out, state, df_gen = run_ai_copilot_action(
        mode="synthetic_cohort",
        prompt="SGLT2 inhibitor vs Placebo in Heart Failure",
        state=state,
    )
    assert state.df is not None
    assert "Synthetic Clinical Cohort" in html_out
    assert df_gen is not None
    assert len(df_gen) > 0


def test_ai_copilot_manuscript_draft():
    """Verify deterministic manuscript drafting."""
    state = AppState()
    html_out, state, _ = run_ai_copilot_action(
        mode="manuscript_draft",
        prompt="SGLT2 inhibitors in Heart Failure",
        state=state,
    )
    assert "Methods" in html_out
    assert "Results" in html_out


def test_survival_view_km_and_cox():
    """Verify Kaplan-Meier and Cox regression in survival view."""
    df, _ = generate_example_dataset()
    state = AppState(df=df)

    # KM
    fig, summary_df, summary_html = run_km_analysis(
        time_col="time", event_col="death", group_col="treatment", state=state
    )
    assert fig is not None
    assert "Log-Rank" in summary_html

    # Cox
    cox_df, forest_fig, stats_html = run_cox_analysis(
        time_col="time",
        event_col="death",
        covar_cols=["treatment", "age", "diabetes"],
        state=state,
    )
    assert cox_df is not None
    assert (
        "treatment" in cox_df.index.tolist() or "treatment" in cox_df.columns.tolist()
    )
    assert forest_fig is not None
    assert "Concordance Index" in stats_html


def test_regression_view():
    """Verify linear and logistic regression actions."""
    df, _ = generate_example_dataset()
    state = AppState(df=df)

    # Linear (OLS)
    coef_df, summary_html, fig = run_regression_analysis(
        model_family="linear",
        outcome_col="sbp",
        predictor_cols=["age", "bmi", "hypertension"],
        state=state,
    )
    assert coef_df is not None
    assert "R-squared" in summary_html
    assert fig is not None

    # Logistic
    coef_logit, summary_logit, _ = run_regression_analysis(
        model_family="logistic",
        outcome_col="death",
        predictor_cols=["treatment", "age", "diabetes"],
        state=state,
    )
    assert coef_logit is not None
    assert "Logistic Regression" in summary_logit


def test_sample_size_view():
    """Verify sample size calculation callbacks for means, proportions, and survival."""
    # Means
    res_m, fig_m = compute_means_sample_size(
        m1=120, m2=110, sd1=15, sd2=15, power=0.80, alpha=0.05, ratio=1.0, dropout=15.0
    )
    assert "Total Target Sample Size" in res_m
    assert fig_m is not None

    # Proportions
    res_p, fig_p = compute_proportions_sample_size(
        p1=0.35, p2=0.20, power=0.80, alpha=0.05, ratio=1.0, dropout=15.0
    )
    assert "Total Target Sample Size" in res_p
    assert fig_p is not None

    # Survival
    res_s, fig_s = compute_survival_sample_size(
        hr=0.65, p_event=0.30, power=0.80, alpha=0.05, ratio=1.0, dropout=15.0
    )
    assert "Total Target Enrollment" in res_s
    assert fig_s is not None


def test_table_one_and_psm():
    """Verify Table 1 and PSM actions."""
    df, _ = generate_example_dataset()
    state = AppState(df=df)

    # Table 1
    t1_html = generate_table_one_action(
        group_col="treatment",
        selected_vars=["age", "sex", "bmi", "diabetes"],
        show_smd=True,
        state=state,
    )
    assert "<table" in t1_html

    # PSM
    state, psm_summary, love_plot = run_psm_action(
        treatment_col="treatment",
        covariates=["age", "bmi", "diabetes", "hypertension"],
        caliper=0.20,
        ratio=1,
        state=state,
    )
    assert state.is_matched is True
    assert state.df_matched is not None
    assert len(state.df_matched) > 0
    assert "Propensity Score Matching Complete" in psm_summary
    assert love_plot is not None


def test_diagnostic_view():
    """Verify 2x2 diagnostic testing and Fagan updating."""
    df_metrics, summary_html, fig = calculate_2x2_diagnostic(
        tp=85, fp=15, fn=15, tn=185, pre_test_prob_pct=25.0
    )
    assert not df_metrics.empty
    assert any("Sensitivity" in str(m) for m in df_metrics["Metric"].values)
    assert "Fagan Bayesian Update" in summary_html
    assert fig is not None


def test_meta_analysis_view():
    """Verify systematic review meta-analysis synthesis and forest plot."""
    df_studies = generate_sample_meta_data()
    forest_fig, funnel_fig, summary_html, effects_df = run_meta_analysis_action(
        df_studies=df_studies, effect_measure="OR", model_type="Random-Effects"
    )
    assert forest_fig is not None
    assert funnel_fig is not None
    assert "Pooled" in summary_html
    assert not effects_df.empty


def test_app_theme_compatibility():
    """Verify that app theme compares cleanly against all Gradio built-in themes."""
    from gradio import utils

    from app import theme

    theme_dict = theme.to_dict()
    assert theme_dict is not None
    for name, built_in_theme in utils.BUILT_IN_THEMES.items():
        # Ensure comparison does not raise AttributeError on font string comparisons
        match = theme_dict == built_in_theme.to_dict()
        assert isinstance(match, bool)


def test_ai_copilot_chat_submit_action_dynamic_critique():
    """Verify dynamic critique rendering in chat submit action."""
    from views.view_ai_copilot import chat_submit_action

    state = AppState()

    # 1. Non-analysis prompt
    (
        history,
        _,
        _,
        new_state,
        _,
        _,
        _,
        critique_summary_non_analysis,
    ) = chat_submit_action(
        user_message="Hello, what can you do?",
        uploaded_files=None,
        chat_history=[],
        state=state,
    )
    assert (
        "No statistical analysis executed in this turn" in critique_summary_non_analysis
    )

    # 2. Analysis prompt (e.g. synthetic survival)
    (
        history2,
        _,
        _,
        new_state2,
        _,
        _,
        _,
        critique_summary_analysis,
    ) = chat_submit_action(
        user_message="สร้าง synthetic cohort สำหรับการทดลองรักษา SGLT2 inhibitor vs Standard care",
        uploaded_files=None,
        chat_history=[],
        state=state,
    )
    # When option 2 / analysis is selected in next step or survival executed:
    assert new_state2.has_data()


def test_ai_copilot_controls_interactivity():
    """Verify that model_dropdown is interactive (wired to HF AI) while unwired workspace, storage, and microphone controls remain non-interactive."""
    import gradio as gr

    from views.view_ai_copilot import create_ai_copilot_view

    with gr.Blocks():
        tab, components = create_ai_copilot_view(app_state=gr.State())

    assert components["workspace_selector"].interactive is False
    assert components["model_dropdown"].interactive is True
    assert components["storage_dropdown"].interactive is False
    assert components["btn_mic"].interactive is False


def test_settings_view_actions(monkeypatch, tmp_path):
    """Verify settings save and HF connection test actions in Gradio Settings tab."""
    import gradio as gr

    from core.state import AppState
    from views.view_settings import (
        create_settings_view,
        test_hf_connection_action,
        update_settings_action,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    state = AppState()
    with gr.Blocks():
        tab, components = create_settings_view(app_state=gr.State())

    assert "btn_test_hf" in components
    assert "btn_save" in components

    # Test update settings action
    save_html = update_settings_action(
        ncbi_key="test_ncbi", hf_token="hf_test_123", state=state
    )
    assert "Settings updated successfully" in save_html
    assert state.hf_token == "hf_test_123"

    # Test individual credential update preserving existing .env keys
    save_ncbi_only = update_settings_action(
        ncbi_key="new_ncbi_key", hf_token="", state=state
    )
    assert "Settings updated successfully" in save_ncbi_only
    env_content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "NCBI_API_KEY=new_ncbi_key" in env_content
    assert "HF_TOKEN=hf_test_123" in env_content

    # Test error handling when writing to .env fails
    from unittest.mock import patch

    with patch(
        "views.view_settings.Path.write_text",
        side_effect=OSError("Read-only filesystem"),
    ):
        fail_html = update_settings_action(
            ncbi_key="fail_key", hf_token="fail_token", state=state
        )
        assert "Failed to persist settings" in fail_html

    # Test HF connection test action (mocked)
    from unittest.mock import MagicMock

    mock_choice = MagicMock()
    mock_choice.message.content = "StatioMed AI Connected: Ready"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    monkeypatch.setattr(
        "agent.agent_runner.InferenceClient", lambda **kwargs: mock_client
    )

    test_html = test_hf_connection_action(hf_token="hf_test_123", state=state)
    assert "Connected to Hugging Face AI" in test_html
