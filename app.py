"""
app.py - StatioMed AI (Pure Native Gradio 6.x Architecture)
=============================================================================
Clinical Research, Biostatistical Analysis & Manuscript Co-Pilot Engine
Built natively for Hugging Face Spaces with ZeroGPU & smolagents support.
Warm Minimalist Theming (Anthropic / Gemini / Antigravity Aesthetic).
=============================================================================
"""

from __future__ import annotations

import os

# Hugging Face ZeroGPU Bootstrap Hook
try:
    import spaces

    @spaces.GPU(duration=45)
    def _zerogpu_startup_probe(text: str = "") -> str:
        """Hugging Face ZeroGPU supervisor startup probe."""
        return f"StatioMed AI ZeroGPU Ready: {text}"

except ImportError:

    def _zerogpu_startup_probe(text: str = "") -> str:
        return "StatioMed AI CPU Mode"


import gradio as gr

from core.state import AppState
from logger import LoggerFactory, get_logger
from views import (
    create_ai_copilot_view,
    create_data_view,
    create_diagnostic_view,
    create_meta_analysis_view,
    create_regression_view,
    create_sample_size_view,
    create_settings_view,
    create_survival_view,
    create_table_one_matching_view,
)

# Initialize Logger
LoggerFactory.configure()
logger = get_logger(__name__)

# =============================================================================
# WARM MINIMALIST THEME & STYLING TOKENS (Anthropic / Antigravity Aesthetic)
# =============================================================================
theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="stone",
    font=[
        "Inter",
        "ui-sans-serif",
        "-apple-system",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        "JetBrains Mono",
        "ui-monospace",
        "Consolas",
        "monospace",
    ],
).set(
    body_background_fill="#FAF8F5",
    body_text_color="#1C1917",
    body_text_color_subdued="#57534E",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F5F2EB",
    block_background_fill="#FFFFFF",
    block_border_color="#E7E5E0",
    block_label_text_color="#292524",
    block_title_text_color="#1C1917",
    input_background_fill="#FFFFFF",
    input_border_color="#E7E5E0",
    input_border_color_focus="#C2410C",
    button_primary_background_fill="#1C1917",
    button_primary_background_fill_hover="#292524",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#F5F2EB",
    button_secondary_background_fill_hover="#EAE6DF",
    button_secondary_border_color="#E7E5E0",
    button_secondary_text_color="#1C1917",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root, body, html {
    color-scheme: light !important;
    background-color: #FAF8F5 !important;
    color: #1C1917 !important;
}

/* Force warm light theme tokens across dark mode overrides */
.dark, body.dark, html.dark {
    background-color: #FAF8F5 !important;
    color: #1C1917 !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F5F2EB !important;
    --body-background-fill: #FAF8F5 !important;
    --body-text-color: #1C1917 !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: #E7E5E0 !important;
    --input-background-fill: #FFFFFF !important;
    --input-text-color: #1C1917 !important;
}

.gradio-container {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    max-width: 1440px !important;
    margin: auto !important;
    background-color: #FAF8F5 !important;
    padding: 12px 24px !important;
}

/* Header Branding (Warm Minimalist) */
.header-container {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 10px 24px !important;
    border: 1px solid #E7E5E0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 12px !important;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F0FDF4 !important;
    color: #166534 !important;
    border: 1px solid #BBF7D0 !important;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.80rem;
    font-weight: 600;
}

/* Navigation Tabs */
.tab-nav {
    border-bottom: 1px solid #E7E5E0 !important;
    margin-bottom: 12px !important;
}

.tab-nav button {
    font-weight: 500 !important;
    color: #78716C !important;
    font-size: 0.90rem !important;
    transition: all 0.15s ease-in-out !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
}

.tab-nav button:hover {
    color: #1C1917 !important;
    background: #F5F2EB !important;
}

.tab-nav button.selected {
    color: #C2410C !important;
    border-bottom: 2px solid #C2410C !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* Workspace Header Row */
.workspace-header-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 8px !important;
    padding: 0 4px !important;
}

.workspace-pill-select {
    max-width: 220px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.workspace-pill-select input {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #1C1917 !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer !important;
}

.btn-reset-chat {
    background: #F5F2EB !important;
    border: 1px solid #E7E5E0 !important;
    border-radius: 8px !important;
    font-size: 0.80rem !important;
    color: #57534E !important;
    padding: 4px 10px !important;
}

/* Clean Chat Window & Bubbles */
.clean-chat-window {
    background: #FAF8F5 !important;
    border: 1px solid #E7E5E0 !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
    overflow-y: auto !important;
}

.clean-chat-window .message-wrap .user,
.clean-chat-window [data-testid="user"] {
    background-color: #F5F2EB !important;
    border: 1px solid #E7E5E0 !important;
    color: #1C1917 !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
    font-size: 0.93rem !important;
    line-height: 1.55 !important;
}

.clean-chat-window .message-wrap .bot,
.clean-chat-window [data-testid="bot"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E7E5E0 !important;
    color: #1C1917 !important;
    border-radius: 14px !important;
    padding: 14px 20px !important;
    font-size: 0.93rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
}

/* Prompt Suggestion Chips */
.clean-prompt-chips {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    margin: 8px 0 10px 0 !important;
}

.clean-prompt-chips button {
    background: #FFFFFF !important;
    color: #44403C !important;
    border: 1px solid #E7E5E0 !important;
    border-radius: 9999px !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    padding: 5px 12px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.15s ease-in-out !important;
}

.clean-prompt-chips button:hover {
    background: #F5F2EB !important;
    color: #1C1917 !important;
    border-color: #D6D3CD !important;
    transform: translateY(-1px) !important;
}

/* =============================================================================
   UNIFIED PROMPT CARD (Antigravity Style)
   ============================================================================= */
.clean-input-card {
    background: #FFFFFF !important;
    border: 1px solid #E7E5E0 !important;
    border-radius: 18px !important;
    padding: 12px 16px 10px 16px !important;
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02) !important;
    margin-top: 6px !important;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
}

.clean-input-card:focus-within {
    border-color: #D6D3CD !important;
    box-shadow: 0 6px 20px -2px rgba(0, 0, 0, 0.06), 0 2px 6px -1px rgba(0, 0, 0, 0.03) !important;
}

.clean-chat-textarea textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #1C1917 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    padding: 4px 0 !important;
    resize: none !important;
}

.clean-chat-textarea textarea::placeholder {
    color: #A8A29E !important;
}

.clean-action-toolbar {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-top: 8px !important;
    padding-top: 6px !important;
}

.toolbar-left-group {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.toolbar-right-group {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.btn-icon-attach {
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    border-radius: 50% !important;
    background: #F5F2EB !important;
    color: #57534E !important;
    border: 1px solid #E7E5E0 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    padding: 0 !important;
    line-height: 1 !important;
}

.btn-icon-attach:hover {
    background: #EAE6DF !important;
    color: #1C1917 !important;
}

.clean-model-pill {
    background: #F5F2EB !important;
    border: 1px solid #E7E5E0 !important;
    border-radius: 9999px !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: #44403C !important;
    padding: 0 6px !important;
    max-width: 260px !important;
}

.clean-model-pill input {
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: #44403C !important;
    background: transparent !important;
    border: none !important;
}

.btn-icon-mic {
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    border-radius: 50% !important;
    background: #F5F2EB !important;
    border: 1px solid #E7E5E0 !important;
    font-size: 0.90rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
}

.btn-icon-send {
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    border-radius: 50% !important;
    background: #1C1917 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-size: 0.90rem !important;
    font-weight: 700 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    padding: 0 !important;
    transition: transform 0.1s ease-in-out !important;
}

.btn-icon-send:hover {
    background: #292524 !important;
    transform: scale(1.05) !important;
}

/* Sub-context Bar */
.sub-context-bar {
    display: flex !important;
    align-items: center !important;
    margin-top: 6px !important;
    padding: 0 4px !important;
}

.storage-pill-select {
    max-width: 220px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.storage-pill-select input {
    font-size: 0.80rem !important;
    color: #78716C !important;
    background: transparent !important;
    border: none !important;
}

/* Right-side Inspector Panel */
.active-dataset-card {
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.active-dataset-card.active {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534;
}

.active-dataset-card.empty {
    background: #FFFFFF;
    border: 1px solid #E7E5E0;
    color: #78716C;
}

.status-indicator-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22C55E;
    display: inline-block;
}

.status-indicator-dot.empty {
    background: #CBD5E1;
}

.badge-count {
    color: #15803D;
    font-weight: 600;
}

.inspector-tabs {
    background: #FFFFFF !important;
    border: 1px solid #E7E5E0 !important;
    border-radius: 14px !important;
    padding: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
}

/* Code Blocks & Tables in Clean Style */
code {
    background-color: #F5F2EB !important;
    color: #1C1917 !important;
    border: 1px solid #E7E5E0 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.88em !important;
}

pre {
    background-color: #F8FAF8 !important;
    color: #1C1917 !important;
    border: 1px solid #E7E5E0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.86rem !important;
    overflow-x: auto !important;
}
"""

startup_js = """
() => {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    if (window.localStorage) {
        localStorage.setItem('theme', 'light');
    }
}
"""


def build_app() -> gr.Blocks:
    """
    Constructs the top-level Gradio application Blocks structure.
    """
    with gr.Blocks(
        title="🏥 StatioMed AI — Clinical Research & Biostatistical Co-Pilot",
    ) as demo:
        # Global Session State
        app_state = gr.State(AppState())

        # Header branding
        with gr.Row(elem_classes=["header-container"]):
            gr.HTML(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; width: 100%;">
                    <div>
                        <h2 style="margin: 0; color: #1C1917; font-size: 1.5rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                            <span>🏥</span> StatioMed AI
                        </h2>
                        <p style="margin: 2px 0 0 0; color: #78716C; font-size: 0.86rem;">
                            Agentic Biostatistical Analysis, Study Design & Manuscript Generation Engine
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span class="header-badge">
                            <span>🔒</span> Zero-PHI Boundary Verified
                        </span>
                        <div style="color: #78716C; font-size: 0.76rem; margin-top: 3px;">
                            SAMPL & R 4.3.3 Benchmark Compliance
                        </div>
                    </div>
                </div>
                """
            )

        # Tab Navigation (Chat-First as Default)
        with gr.Tabs(selected="tab_ai_copilot"):
            create_ai_copilot_view(app_state)
            create_data_view(app_state)
            create_survival_view(app_state)
            create_regression_view(app_state)
            create_sample_size_view(app_state)
            create_table_one_matching_view(app_state)
            create_diagnostic_view(app_state)
            create_meta_analysis_view(app_state)
            create_settings_view(app_state)

        # Footer
        gr.HTML(
            """
            <div style="text-align: center; color: #A8A29E; font-size: 0.80rem; padding: 18px 0 10px 0; border-top: 1px solid #E7E5E0; margin-top: 20px;">
                StatioMed AI © 2026 | Built for Clinical Researchers & Biostatisticians | Powered by Gradio 6.x & smolagents
            </div>
            """
        )

    return demo


demo = build_app()
app = demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    logger.info(
        f"🚀 Launching StatioMed AI Native Gradio on port {port} (Warm Minimalist Theme)..."
    )
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css,
        js=startup_js,
        ssr_mode=False,
    )
