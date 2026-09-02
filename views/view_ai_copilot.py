"""
views/view_ai_copilot.py - StatioMed AI Clean Conversational Chatbot View
=============================================================================
Conversational AI Chatbot UI (Anthropic / Gemini / Antigravity Style) with integrated
proposal (.docx, .pdf) and dataset (.csv, .xlsx, .sav, .dta) upload, automatic biostatistical
methodology determination, smolagents tool calling, and automated critique appraisal.
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

INITIAL_BOT_MESSAGE = """### 🏥 StatioMed AI — Clinical Biostatistical Co-Pilot
I am an AI-driven biostatistics and research methodology engine (Zero-PHI Compliant).

**How I can assist your clinical investigation:**
1. **💡 5-Direction Research Ideation:** Specify clinical topics (e.g. *'dyspnea'*, *'sepsis'*, *'acute kidney injury'*). I synthesize PubMed evidence and formulate 5 publication-ready study designs with SAPs.
2. **📄 Analyze Research Protocols (`.docx`, `.pdf`):** Extract PICO, study variables, and construct a SAMPL & EQUATOR compliant statistical pipeline.
3. **📊 Analyze Research Datasets (`.csv`, `.xlsx`, `.sav`, `.dta`):** Automatically detect schemas, execute biostatistical workflows (Baseline Table 1, Kaplan-Meier, Cox PH, Logistic Regression, PSM), and appraise bias risks.
4. **🧬 Generate Synthetic Cohorts:** Instant calibrated mock datasets for hypothesis testing and protocol validation.
5. **📐 Sample Size & Power Calculations:** Closed-form formulas with manuscript-ready methodology justification.

*Type any research question below, click the prompt chips, or attach your protocol/dataset to begin.*
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
    model_name: str = "Qwen 2.5 72B (Hugging Face)",
) -> tuple[
    list[dict[str, str]],
    str,
    list[Any] | None,
    AppState,
    go.Figure,
    pd.DataFrame | None,
    str,
    str,
]:
    """
    Handles user chat submission with optional file attachments, executes statistical harness,
    and returns updated chat stream, visual artifacts, and critique appraisal.
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
        return (
            chat_history,
            "",
            None,
            state,
            go.Figure(),
            state.df,
            "",
            "ℹ️ No active analysis executed yet.",
        )

    display_user_msg = msg_to_send
    if file_names:
        files_badge = " ".join([f"`📎 {fn}`" for fn in file_names])
        display_user_msg = (
            f"{display_user_msg}\n\n{files_badge}" if display_user_msg else files_badge
        )

    chat_history.append({"role": "user", "content": display_user_msg})

    try:
        state.active_model = model_name
        response_md, new_state, fig, preview_df = ClinicalAnalystEngine.process_turn(
            user_message=msg_to_send,
            file_paths=files_list,
            state=state,
            model_name=model_name,
        )
        chat_history.append({"role": "assistant", "content": response_md})
        fig_out = fig if fig is not None else go.Figure()

        active_status_html = (
            f"""
        <div class="active-dataset-card active">
            <span class="status-indicator-dot"></span>
            <strong>Active Session:</strong> {new_state.file_name} <span class="badge-count">({len(new_state.df):,} rows)</span>
        </div>
        """
            if new_state.has_data()
            else """
        <div class="active-dataset-card empty">
            <span class="status-indicator-dot empty"></span>
            <span>📁 No dataset loaded. Attach files with <code>+</code> or ask AI to generate data.</span>
        </div>
        """
        )

        critique_summary = (
            getattr(new_state, "last_critique_md", None)
            or (
                new_state.last_analysis_results.get("critique_md")
                if isinstance(new_state.last_analysis_results, dict)
                else None
            )
            or """### 🛡️ Automated Clinical Appraisal
*No statistical analysis executed in this turn. Run a statistical analysis or select an option to view automated bias appraisal, EPV checks, and assumption tests.*"""
        )

        return (
            chat_history,
            "",
            None,
            new_state,
            fig_out,
            preview_df,
            active_status_html,
            critique_summary,
        )

    except Exception as e:
        logger.exception("Chat Action Error: %s", e)
        err_msg = f"❌ Error during processing: {html.escape(str(e))}"
        chat_history.append({"role": "assistant", "content": err_msg})
        return (
            chat_history,
            "",
            None,
            state,
            go.Figure(),
            state.df,
            "",
            "❌ Error encountered in analysis.",
        )


def handle_prompt_chip(
    prompt_text: str,
    chat_history: list[dict[str, str]],
    state: AppState,
    model_name: str = "Qwen 2.5 72B (Hugging Face)",
) -> tuple[
    list[dict[str, str]],
    str,
    list[Any] | None,
    AppState,
    go.Figure,
    pd.DataFrame | None,
    str,
    str,
]:
    """Helper to execute predefined prompt chip directly."""
    return chat_submit_action(
        user_message=prompt_text,
        uploaded_files=None,
        chat_history=chat_history,
        state=state,
        model_name=model_name,
    )


def clear_chat_action(
    state: AppState,
) -> tuple[
    list[dict[str, str]], str, list[Any] | None, go.Figure, pd.DataFrame | None, str
]:
    """Resets chatbot conversation."""
    initial_history = [{"role": "assistant", "content": INITIAL_BOT_MESSAGE}]
    return (
        initial_history,
        "",
        None,
        go.Figure(),
        state.df,
        "ℹ️ Conversation reset to initial state.",
    )


def toggle_file_upload_visibility(is_visible: bool) -> tuple[bool, str]:
    """Toggles the visibility of the file upload container."""
    new_vis = not is_visible
    btn_text = "−" if new_vis else "+"
    return new_vis, btn_text


def create_ai_copilot_view(
    app_state: gr.State,
) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native View for the Conversational AI Biostatistical Co-Pilot
    with warm minimalist aesthetic (Antigravity/Gemini style).
    """
    with gr.Tab("💬 AI Co-Pilot", id="tab_ai_copilot") as tab:
        with gr.Row():
            # =========================================================================
            # LEFT COLUMN: Clean Conversational Chat & Unified Prompt Card (scale=7)
            # =========================================================================
            with gr.Column(scale=7, elem_classes=["chat-main-column"]):
                # Top Workspace Context Header: 📁 statiomed-ai ⌄
                with gr.Row(elem_classes=["workspace-header-row"]):
                    workspace_selector = gr.Dropdown(
                        choices=[
                            "📁 statiomed-ai",
                            "📁 clinical-icu-cohort",
                            "📁 rct-sglt2-hf-trial",
                        ],
                        value="📁 statiomed-ai",
                        show_label=False,
                        container=False,
                        elem_classes=["workspace-pill-select"],
                        interactive=False,
                    )
                    btn_clear = gr.Button(
                        "🗑️ Reset",
                        variant="secondary",
                        size="sm",
                        elem_classes=["btn-reset-chat"],
                    )

                # Conversational Chat Window
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": INITIAL_BOT_MESSAGE}],
                    height=520,
                    render_markdown=True,
                    elem_classes=["clean-chat-window"],
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )

                # Quick Action Suggestion Chips (Minimalist pill styling)
                with gr.Row(elem_classes=["clean-prompt-chips"]):
                    btn_chip_dyspnea = gr.Button(
                        "💡 Ideation: Dyspnea", size="sm", variant="secondary"
                    )
                    btn_chip_sepsis = gr.Button(
                        "💡 Ideation: Sepsis", size="sm", variant="secondary"
                    )
                    btn_chip_proposal = gr.Button(
                        "📄 Analyze Proposal", size="sm", variant="secondary"
                    )
                    btn_chip_synth = gr.Button(
                        "🧬 Synthetic KM", size="sm", variant="secondary"
                    )
                    btn_chip_sample = gr.Button(
                        "📐 Sample Size (80%)", size="sm", variant="secondary"
                    )
                    btn_chip_t1 = gr.Button(
                        "👥 Table 1 SMD", size="sm", variant="secondary"
                    )
                    btn_chip_surv = gr.Button(
                        "⏱️ Cox Survival", size="sm", variant="secondary"
                    )

                # =====================================================================
                # UNIFIED PROMPT CARD (Antigravity / Minimalist Style)
                # =====================================================================
                with gr.Group(elem_classes=["clean-input-card"]):
                    # Main Textarea
                    chat_input = gr.Textbox(
                        placeholder="Ask anything, @ to mention, / for actions",
                        lines=2,
                        max_lines=6,
                        show_label=False,
                        container=False,
                        elem_classes=["clean-chat-textarea"],
                    )

                    # Collapsible File Attachment Dropzone
                    with gr.Row(
                        visible=False, elem_classes=["clean-file-row"]
                    ) as file_upload_row:
                        file_uploader = gr.File(
                            label="Attach Proposal (.docx, .pdf) or Dataset (.csv, .xlsx, .sav, .dta)",
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
                            elem_classes=["clean-file-box"],
                        )

                    # Inner Action Toolbar
                    with gr.Row(elem_classes=["clean-action-toolbar"]):
                        # Left Toolbar Group: + (attachment toggle) & Model Selector Pill
                        with gr.Row(elem_classes=["toolbar-left-group"]):
                            btn_attach_toggle = gr.Button(
                                "+",
                                variant="secondary",
                                elem_classes=["btn-icon-attach"],
                            )
                            model_dropdown = gr.Dropdown(
                                choices=[
                                    "Qwen 2.5 72B (Hugging Face)",
                                    "Llama 3.3 70B (Hugging Face)",
                                    "DeepSeek R1 32B (Hugging Face)",
                                    "Mistral Small 24B (Hugging Face)",
                                    "Local Deterministic Engine (Offline)",
                                ],
                                value="Qwen 2.5 72B (Hugging Face)",
                                show_label=False,
                                container=False,
                                elem_classes=["clean-model-pill"],
                                interactive=True,
                            )

                        # Right Toolbar Group: Mic icon & Circular Send button
                        with gr.Row(elem_classes=["toolbar-right-group"]):
                            btn_mic = gr.Button(
                                "🎙️",
                                variant="secondary",
                                elem_classes=["btn-icon-mic"],
                                interactive=False,
                            )
                            btn_send = gr.Button(
                                "➔",
                                variant="primary",
                                elem_classes=["btn-icon-send"],
                            )

                # Sub-bar Context & Storage Pill
                with gr.Row(elem_classes=["sub-context-bar"]):
                    storage_dropdown = gr.Dropdown(
                        choices=[
                            "🗄️ Local (IndexedDB/OPFS)",
                            "☁️ Encrypted Cloud Sync",
                            "🧬 Synthetic Sandbox",
                        ],
                        value="🗄️ Local (IndexedDB/OPFS)",
                        show_label=False,
                        container=False,
                        elem_classes=["storage-pill-select"],
                        interactive=False,
                    )

            # =========================================================================
            # RIGHT COLUMN: Inspector & Artifacts Panel (scale=5)
            # =========================================================================
            with gr.Column(scale=5, elem_classes=["inspector-panel-column"]):
                active_status_badge = gr.HTML(
                    """
                    <div class="active-dataset-card empty">
                        <span class="status-indicator-dot empty"></span>
                        <span>📁 No dataset loaded. Attach files with <code>+</code> or ask AI to generate data.</span>
                    </div>
                    """
                )

                with gr.Tabs(elem_classes=["inspector-tabs"]):
                    with gr.Tab("📈 Visual Output (Charts)"):
                        plot_output = gr.Plot(
                            label="Interactive Visualizations",
                            elem_classes=["inspector-plot"],
                        )

                    with gr.Tab("📋 Active Dataset Preview"):
                        dataset_preview = gr.Dataframe(
                            label="Session Dataframe Records",
                            interactive=False,
                            wrap=True,
                            elem_classes=["inspector-df"],
                        )

                    with gr.Tab("🛡️ Critical Appraisal & Diagnostics"):
                        critique_inspector = gr.Markdown(
                            """
                            ### 🛡️ Automated Clinical Appraisal
                            *No statistical analysis executed yet. Run an analysis from chat to view automated bias appraisal, EPV checks, and assumption tests.*
                            """,
                            elem_classes=["inspector-critique"],
                        )

                    with gr.Tab("ℹ️ Zero-PHI Principles"):
                        gr.Markdown(
                            """
                            #### 🔒 Zero-PHI & SAMPL Certified Engine
                            - **Zero Hallucination:** LLM routes deterministically to calibrated statistical tools (`lifelines`, `statsmodels`, `pingouin`).
                            - **Dual Ingestion:** Ingests research protocols (`.docx`, `.pdf`) and clinical datasets (`.csv`, `.xlsx`, `.sav`, `.dta`).
                            - **Deterministic Verification:** Automatic EPV, quasi-separation, and proportional hazards checks on all models.
                            """,
                            elem_classes=["inspector-principles"],
                        )

        # =========================================================================
        # EVENT HANDLERS & CALLBACKS
        # =========================================================================
        file_upload_visible_state = gr.State(False)

        btn_attach_toggle.click(
            fn=toggle_file_upload_visibility,
            inputs=[file_upload_visible_state],
            outputs=[file_upload_visible_state, btn_attach_toggle],
        ).then(
            fn=lambda vis: gr.update(visible=vis),
            inputs=[file_upload_visible_state],
            outputs=[file_upload_row],
        )

        btn_send.click(
            fn=chat_submit_action,
            inputs=[chat_input, file_uploader, chatbot, app_state, model_dropdown],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
            ],
        )

        chat_input.submit(
            fn=chat_submit_action,
            inputs=[chat_input, file_uploader, chatbot, app_state, model_dropdown],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
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
                model_dropdown,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
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
                model_dropdown,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
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
                model_dropdown,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
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
                model_dropdown,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
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
                model_dropdown,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
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
                model_dropdown,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
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
                model_dropdown,
            ],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                app_state,
                plot_output,
                dataset_preview,
                active_status_badge,
                critique_inspector,
            ],
        )

        btn_clear.click(
            fn=clear_chat_action,
            inputs=[app_state],
            outputs=[
                chatbot,
                chat_input,
                file_uploader,
                plot_output,
                dataset_preview,
                critique_inspector,
            ],
        )

    return tab, {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "file_uploader": file_uploader,
        "btn_send": btn_send,
        "btn_mic": btn_mic,
        "btn_clear": btn_clear,
        "plot_output": plot_output,
        "dataset_preview": dataset_preview,
        "model_dropdown": model_dropdown,
        "workspace_selector": workspace_selector,
        "storage_dropdown": storage_dropdown,
    }
