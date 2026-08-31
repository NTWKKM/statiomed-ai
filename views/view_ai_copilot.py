"""
views/view_ai_copilot.py - StatioMed AI Conversational Chatbot View (Gradio Native)
=============================================================================
Conversational AI Chatbot UI (ChatGPT / Claude / Gemini style) with integrated
proposal (.docx) and dataset (.csv, .xlsx, .sav) upload, automatic biostatistical
methodology determination, and immediate deterministic execution harness.
=============================================================================
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from agent.clinical_analyst import ClinicalAnalystEngine
from agent.manuscript_engine import ManuscriptEngine
from agent.tools.tool_synthetic_data import SyntheticDataTool
from core.state import AppState
from logger import get_logger

logger = get_logger(__name__)

INITIAL_BOT_MESSAGE = """### 🏥 Hello! I am StatioMed AI — Clinical Biostatistical Co-Pilot
I am an AI-driven clinical biostatistics and research methodology engine (Zero-PHI Compliant).

**How I can assist your clinical investigation:**
1. **💡 5-Direction Research Ideation:** Specify broad topics like *'dyspnea'*, *'sepsis'*, or *'acute kidney injury'*. I will synthesize recent evidence from **PubMed** and formulate 5 publication-ready study designs (RCT, Survival Cohort, Diagnostic Accuracy, Prediction Model, PSM) with statistical analysis plans.
2. **📄 Analyze Research Proposal / Protocol (`.docx`, `.pdf`, `.txt`):** Extract PICO, study variables, and construct a SAMPL & EQUATOR compliant statistical pipeline.
3. **📊 Ingest Clinical Research Datasets (`.csv`, `.xlsx`, `.sav`, `.dta`):** Automatically detect schemas and execute appropriate biostatistical workflows (Baseline Table 1, Kaplan-Meier, Cox PH, Logistic Regression).
4. **🧬 Generate Synthetic Clinical Cohorts:** Instant mock datasets for hypothesis testing and model validation.
5. **📐 Sample Size & Power Calculations:** Closed-form formulas with manuscript-ready methodology text.

*Type any clinical topic of interest (e.g., 'dyspnea') or click the quick action chips below!*
"""


def run_ai_copilot_action(
    mode: str, prompt: str, state: AppState
) -> tuple[str, AppState, pd.DataFrame | None]:
    """
    Backward-compatible action executor for programmatic testing and legacy workflows.
    """
    if mode == "synthetic_cohort":
        df_gen, meta = SyntheticDataTool.generate_topic_aware_cohort(
            prompt or "Clinical Cohort", n=200, seed=42
        )
        state.df = df_gen
        state.file_name = f"Synthetic Cohort: {meta.get('domain', 'Clinical Research')}"
        state.var_meta = meta
        return (
            "<div style='color:#059669;'>Synthetic Clinical Cohort generated successfully.</div>",
            state,
            df_gen,
        )
    elif mode == "manuscript_draft":
        ctx = {
            "study_title": prompt or "Clinical Investigation",
            "n_total": 500,
            "n_intervention": 250,
            "n_control": 250,
            "m_imputations": 20,
            "median_followup": "365",
            "primary_hazard_ratio": "0.68",
            "hr_ci_lower": "0.52",
            "hr_ci_upper": "0.89",
            "logrank_p_val": "0.004",
            "population_desc": "adult clinical cohort",
            "intervention_name": "target therapy",
            "comparator_name": "standard of care",
            "primary_endpoint_desc": "all-cause mortality",
        }
        methods = ManuscriptEngine.render_methods("cohort", ctx)
        results = ManuscriptEngine.render_results("survival", ctx)
        html_out = f"<div><h4>Methods</h4><pre>{methods}</pre><h4>Results</h4><pre>{results}</pre></div>"
        return html_out, state, state.df
    else:
        resp_md, new_state, _, preview_df = ClinicalAnalystEngine.process_turn(
            user_message=prompt, file_paths=None, state=state
        )
        return resp_md, new_state, preview_df


def chat_submit_action(
    user_message: str,
    uploaded_files: list[Any] | None,
    chat_history: list[dict[str, str]],
    state: AppState,
) -> tuple[
    list[dict[str, str]],
    str,
    list[Any] | None,
    AppState,
    go.Figure,
    pd.DataFrame | None,
    str,
]:
    """
    Handles user chat submission with optional file attachments, executes statistical harness,
    and returns updated chat stream and visual artifacts.
    """
    chat_history = chat_history or []
    files_list = []
    file_names = []

    if uploaded_files:
        for f in uploaded_files:
            p = f.name if hasattr(f, "name") else str(f)
            files_list.append(p)
            file_names.append(Path(p).name)

    msg_to_send = (user_message or "").strip()
    if not msg_to_send and not files_list:
        return chat_history, "", None, state, go.Figure(), state.df, ""

    display_user_msg = msg_to_send
    if file_names:
        files_badge = " ".join([f"`📎 {fn}`" for fn in file_names])
        display_user_msg = (
            f"{display_user_msg}\n\n{files_badge}" if display_user_msg else files_badge
        )

    chat_history.append({"role": "user", "content": display_user_msg})

    try:
        response_md, new_state, fig, preview_df = ClinicalAnalystEngine.process_turn(
            user_message=msg_to_send,
            file_paths=files_list,
            state=state,
        )
        chat_history.append({"role": "assistant", "content": response_md})
        fig_out = fig if fig is not None else go.Figure()

        active_status_html = (
            f"""
        <div style='background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:8px 12px;font-size:0.85rem;color:#166534;'>
            ✅ <strong>Active Session:</strong> {new_state.file_name} ({len(new_state.df):,} rows)
        </div>
        """
            if new_state.has_data()
            else "<div style='background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;padding:8px 12px;color:#475569;font-size:0.85rem;'>📁 No dataset loaded. Upload files or ask AI to generate synthetic data.</div>"
        )

        return (
            chat_history,
            "",
            None,
            new_state,
            fig_out,
            preview_df,
            active_status_html,
        )

    except Exception as e:
        logger.exception("Chat Action Error: %s", e)
        err_msg = f"❌ Error during processing: {html.escape(str(e))}"
        chat_history.append({"role": "assistant", "content": err_msg})
        return chat_history, "", None, state, go.Figure(), state.df, ""


def handle_prompt_chip(
    prompt_text: str,
    chat_history: list[dict[str, str]],
    state: AppState,
) -> tuple[
    list[dict[str, str]],
    str,
    list[Any] | None,
    AppState,
    go.Figure,
    pd.DataFrame | None,
    str,
]:
    """Helper to execute predefined prompt chip directly."""
    return chat_submit_action(
        user_message=prompt_text,
        uploaded_files=None,
        chat_history=chat_history,
        state=state,
    )


def clear_chat_action(
    state: AppState,
) -> tuple[list[dict[str, str]], str, list[Any] | None, go.Figure, pd.DataFrame | None]:
    """Resets chatbot conversation."""
    initial_history = [{"role": "assistant", "content": INITIAL_BOT_MESSAGE}]
    return initial_history, "", None, go.Figure(), state.df


def create_ai_copilot_view(
    app_state: gr.State,
) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for the Conversational AI Biostatistical Co-Pilot.
    """
    with gr.Tab("🤖 AI Co-Pilot", id="tab_ai_copilot") as tab:
        with gr.Row():
            # Left Column: Conversational Chat Interface (Anthropic/ChatGPT/Gemini style)
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": INITIAL_BOT_MESSAGE}],
                    height=560,
                    render_markdown=True,
                    elem_classes=["ai-chat-window"],
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )

                # Quick Action Chips
                with gr.Row(elem_classes=["prompt-chips-row"]):
                    btn_chip_dyspnea = gr.Button(
                        "💡 Research Ideation: Dyspnea", size="sm", variant="secondary"
                    )
                    btn_chip_sepsis = gr.Button(
                        "💡 Research Ideation: Sepsis", size="sm", variant="secondary"
                    )
                    btn_chip_proposal = gr.Button(
                        "📄 Analyze Proposal & Stats", size="sm", variant="secondary"
                    )
                    btn_chip_synth = gr.Button(
                        "🧬 Synthetic Data & Kaplan-Meier",
                        size="sm",
                        variant="secondary",
                    )
                    btn_chip_sample = gr.Button(
                        "📐 Sample Size (80% Power)",
                        size="sm",
                        variant="secondary",
                    )
                    btn_chip_t1 = gr.Button(
                        "👥 Generate Table 1 Baseline", size="sm", variant="secondary"
                    )
                    btn_chip_surv = gr.Button(
                        "⏱️ Run Survival & Cox PH", size="sm", variant="secondary"
                    )

                # Chat Input Box + File Upload
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="💬 Type your research question, objective, or attach proposal/dataset below...",
                        lines=2,
                        max_lines=6,
                        scale=9,
                        show_label=False,
                        container=False,
                    )
                    btn_send = gr.Button("🚀 Send Message", variant="primary", scale=2)

                with gr.Row():
                    file_uploader = gr.File(
                        label="📎 Attach Proposal (.docx, .pdf, .txt) or Dataset (.csv, .xlsx, .sav, .dta)",
                        file_types=[
                            ".docx",
                            ".doc",
                            ".pdf",
                            ".txt",
                            ".md",
                            ".csv",
                            ".xlsx",
                            ".xlsm",
                            ".xls",
                            ".sav",
                            ".dta",
                        ],
                        file_count="multiple",
                        scale=9,
                    )
                    btn_clear = gr.Button(
                        "🗑️ Clear Chat", variant="secondary", size="sm", scale=2
                    )

            # Right Column: Live Visual Artifacts & Dataset Inspection
            with gr.Column(scale=5):
                active_status_badge = gr.HTML(
                    """
                    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; color: #475569; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);">
                        📁 No dataset loaded. Upload files or ask AI to generate synthetic data.
                    </div>
                    """
                )

                with gr.Tabs():
                    with gr.Tab("📈 Visual Output (Charts & Plots)"):
                        plot_output = gr.Plot(
                            label="Interactive Statistical Visualizations"
                        )

                    with gr.Tab("📋 Active Dataset Preview"):
                        dataset_preview = gr.Dataframe(
                            label="Session Dataframe Records",
                            interactive=False,
                            wrap=True,
                        )

                    with gr.Tab("ℹ️ System Principles"):
                        gr.Markdown(
                            """
                            #### 🔒 Zero-PHI & SAMPL Certified Engine
                            - **Zero Hallucination:** LLM selects calibrated deterministic statistical functions from `utils/` (benchmarked against R 4.3.3 & statsmodels).
                            - **Dual Ingestion:** Supports research proposals (Word `.docx`, PDF, text) and clinical datasets (Excel, CSV, SPSS, Stata).
                            - **Immediate Execution:** Automatically executes statistical tests and renders interactive plots with full reactive session state.
                            """
                        )

        # Callbacks & Event Handlers
        btn_send.click(
            fn=chat_submit_action,
            inputs=[chat_input, file_uploader, chatbot, app_state],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        chat_input.submit(
            fn=chat_submit_action,
            inputs=[chat_input, file_uploader, chatbot, app_state],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        # Prompt chip handlers
        btn_chip_dyspnea.click(
            fn=handle_prompt_chip,
            inputs=[
                gr.State(
                    "Propose 5 clinical research study designs and statistical analysis plans for acute dyspnea based on PubMed evidence"
                ),
                chatbot,
                app_state,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        btn_chip_sepsis.click(
            fn=handle_prompt_chip,
            inputs=[
                gr.State(
                    "Propose 5 clinical research study designs and statistical analysis plans for sepsis in the emergency department/ICU based on PubMed evidence"
                ),
                chatbot,
                app_state,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        btn_chip_proposal.click(
            fn=handle_prompt_chip,
            inputs=[
                gr.State(
                    "Analyze this clinical research proposal and recommend appropriate statistical methodology for the primary endpoint"
                ),
                chatbot,
                app_state,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        btn_chip_synth.click(
            fn=handle_prompt_chip,
            inputs=[
                gr.State(
                    "Generate a synthetic clinical trial dataset comparing SGLT2 inhibitor vs Placebo and execute Kaplan-Meier survival analysis"
                ),
                chatbot,
                app_state,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        btn_chip_sample.click(
            fn=handle_prompt_chip,
            inputs=[
                gr.State(
                    "Calculate sample size for an RCT comparing two groups with 30% vs 15% event rates, 80% power, alpha 0.05"
                ),
                chatbot,
                app_state,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        btn_chip_t1.click(
            fn=handle_prompt_chip,
            inputs=[
                gr.State(
                    "Generate a baseline characteristics Table 1 with Standardized Mean Differences (SMD)"
                ),
                chatbot,
                app_state,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        btn_chip_surv.click(
            fn=handle_prompt_chip,
            inputs=[
                gr.State(
                    "Execute Kaplan-Meier survival analysis and multivariable Cox Proportional Hazards model adjusting for confounders"
                ),
                chatbot,
                app_state,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
            ],
        )

        btn_clear.click(
            fn=clear_chat_action,
            inputs=[app_state],
            outputs=[chatbot, chat_input, file_uploader, plot_output, dataset_preview],
        )

    return tab, {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "file_uploader": file_uploader,
        "btn_send": btn_send,
        "btn_clear": btn_clear,
        "plot_output": plot_output,
        "dataset_preview": dataset_preview,
    }
