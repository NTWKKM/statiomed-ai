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

INITIAL_BOT_MESSAGE = """### 🏥 สวัสดีครับ! ผมคือ StatioMed AI — Clinical Biostatistical Co-Pilot
ผมเป็นระบบผู้ช่วยปัญญาประดิษฐ์และเครื่องมือวิเคราะห์ชีวสถิติทางการแพทย์ (Zero-PHI Compliant)

**สิ่งที่ผมสามารถช่วยเหลือท่านได้ทันที:**
1. **💡 เสนอแนวทางการทำวิจัย 4-5 รูปแบบจากหัวข้อกว้างๆ:** ระบุหัวข้อ เช่น *'dyspnea'*, *'sepsis'*, *'acute kidney injury'* ระบบจะดึงหลักฐานจาก **PubMed** และสังเคราะห์โจทย์วิจัย (RCT, Survival, Diagnostic, Prediction, PSM) พร้อมแผนสถิติให้เลือก
2. **📄 อัปโหลด Research Proposal / Protocol (`.docx`, `.pdf`, `.txt`):** เพื่อให้ระบบวิเคราะห์ PICO, ตัวแปร, และเลือกสถิติที่เหมาะสมตามมาตรฐาน SAMPL & EQUATOR
3. **📊 อัปโหลดชุดข้อมูลวิจัย (`.csv`, `.xlsx`, `.sav`, `.dta`):** เพื่อให้ระบบรันสถิติที่เหมาะสม (เช่น Table 1, Kaplan-Meier, Cox PH, Logistic Regression) ให้ทันที
4. **🧬 สร้างข้อมูลจำลอง (Synthetic Clinical Cohort):** เพื่อทดสอบโมเดลสถิติตามโจทย์ทางคลินิก
5. **📐 คำนวณขนาดกลุ่มตัวอย่าง (Sample Size & Statistical Power):** พร้อมข้อความสำหรับเขียนในระเบียบวิธีวิจัย

*พิมพ์ชื่อหัวข้อที่สนใจ (เช่น 'dyspnea') หรือคลิกปุ่มด้านล่างได้เลยครับ!*
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
            file_names.append(pd.io.common.os.path.basename(p))

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
            else "<div style='color:#64748b;font-size:0.85rem;'>No dataset currently active in session.</div>"
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
        logger.error(f"Chat Action Error: {e}")
        err_msg = f"❌ เกิดข้อผิดพลาดในการประมวลผล: {html.escape(str(e))}"
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
                        "💡 เสนอแนวทางวิจัย: Dyspnea", size="sm", variant="secondary"
                    )
                    btn_chip_sepsis = gr.Button(
                        "💡 เสนอแนวทางวิจัย: Sepsis", size="sm", variant="secondary"
                    )
                    btn_chip_proposal = gr.Button(
                        "📄 วิเคราะห์ Proposal & สถิติ", size="sm", variant="secondary"
                    )
                    btn_chip_synth = gr.Button(
                        "🧬 สร้าง Synthetic Data & รัน KM",
                        size="sm",
                        variant="secondary",
                    )
                    btn_chip_sample = gr.Button(
                        "📐 คำนวณ Sample Size (80% Power)",
                        size="sm",
                        variant="secondary",
                    )
                    btn_chip_t1 = gr.Button(
                        "👥 สร้าง Table 1 Baseline", size="sm", variant="secondary"
                    )
                    btn_chip_surv = gr.Button(
                        "⏱️ รัน Survival & Cox PH", size="sm", variant="secondary"
                    )

                # Chat Input Box + File Upload
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="💬 พิมพ์คำถาม, ระบุวัตถุประสงค์วิจัย, หรือแนบไฟล์ Proposal/Data ด้านล่าง...",
                        lines=2,
                        max_lines=6,
                        scale=9,
                        show_label=False,
                        container=False,
                    )
                    btn_send = gr.Button("🚀 ส่งข้อความ", variant="primary", scale=2)

                with gr.Row():
                    file_uploader = gr.File(
                        label="📎 แนบไฟล์ Proposal (.docx, .pdf, .txt) หรือ Dataset (.csv, .xlsx, .sav)",
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
                        "🗑️ ล้างแชท", variant="secondary", size="sm", scale=2
                    )

            # Right Column: Live Visual Artifacts & Dataset Inspection
            with gr.Column(scale=5):
                active_status_badge = gr.HTML(
                    """
                    <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 8px 12px; font-size: 0.85rem; color: #64748b;">
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
                            - **Zero Hallucination:** LLM ไม่คิดเลขเอง แต่เลือกใช้ฟังก์ชันสถิติที่ผ่านการสอบเทียบจาก `utils/` (R 4.3.3 & statsmodels benchmarked)
                            - **Dual Ingestion:** รองรับทั้งโครงร่างงานวิจัย (Word `.docx`) และชุดข้อมูลจริง (Excel/CSV/SPSS)
                            - **Immediate Execution:** สั่งการสถิติและพล็อตกราฟให้ทันที พร้อมส่งต่อข้อมูลไปยังแท็บอื่นในระบบแบบ Reactive State
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
                    "เสนอแนวทางการทำวิจัยทางคลินิก 5 รูปแบบสำหรับหัวข้อ Dyspnea (ภาวะหายใจลำบาก) พร้อมหลักฐานจาก PubMed"
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
                    "เสนอแนวทางการทำวิจัยทางคลินิก 5 รูปแบบสำหรับหัวข้อ Sepsis ในแผนกฉุกเฉิน/ICU พร้อมหลักฐานจาก PubMed"
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
                    "ช่วยวิเคราะห์โครงร่างงานวิจัย (Proposal) และแนะนำสถิติที่เหมาะสมสำหรับ Primary Endpoint"
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
                    "สร้าง Synthetic Data การทดลองทางคลินิก SGLT2 inhibitor vs Placebo แล้วรัน Kaplan-Meier survival analysis ให้ดูทันที"
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
                    "คำนวณ sample size สำหรับ RCT เปรียบเทียบ 2 กลุ่ม Event rate 30% vs 15% Power 80% Alpha 0.05"
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
                    "สร้าง Table 1 Baseline characteristics พร้อมคำนวณ Standardized Mean Differences (SMD)"
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
                    "รัน Kaplan-Meier survival curves และ Multivariable Cox Proportional Hazards model ปรับตัวแปรกวน"
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
