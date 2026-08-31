"""
app.py - StatioMed AI (Pure Native Gradio 6.x Architecture)
=============================================================================
Clinical Research, Biostatistical Analysis & Manuscript Co-Pilot Engine
Built natively for Hugging Face Spaces with ZeroGPU & smolagents support.
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
# CLINICAL LIGHT THEME & STYLING TOKENS
# =============================================================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
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
    body_background_fill="#f8fafc",
    body_text_color="#0f172a",
    body_text_color_subdued="#475569",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f1f5f9",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_label_text_color="#1e293b",
    block_title_text_color="#0f172a",
    input_background_fill="#ffffff",
    input_border_color="#cbd5e1",
    input_border_color_focus="#0284c7",
    button_primary_background_fill="#0284c7",
    button_primary_background_fill_hover="#0369a1",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f8fafc",
    button_secondary_border_color="#cbd5e1",
    button_secondary_text_color="#0f172a",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root, body, html {
    color-scheme: light !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Force light theme tokens across dark mode overrides */
.dark, body.dark, html.dark {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f1f5f9 !important;
    --body-background-fill: #f8fafc !important;
    --body-text-color: #0f172a !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e2e8f0 !important;
    --input-background-fill: #ffffff !important;
    --input-text-color: #0f172a !important;
}

.gradio-container {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
    background-color: #f8fafc !important;
}

.header-container {
    background: #ffffff;
    border-radius: 12px;
    padding: 8px 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    margin-bottom: 16px;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}

.tab-nav button {
    font-weight: 500 !important;
    color: #475569 !important;
    font-size: 0.92rem !important;
}

.tab-nav button.selected {
    color: #0284c7 !important;
    border-bottom: 2px solid #0284c7 !important;
    font-weight: 600 !important;
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
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; width: 100%;">
                    <div>
                        <h2 style="margin: 0; color: #0f172a; font-size: 1.6rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                            <span>🏥</span> StatioMed AI
                        </h2>
                        <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.9rem;">
                            Agentic Biostatistical Analysis, Study Design & Manuscript Generation Engine
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span class="header-badge">
                            <span>🔒</span> Zero-PHI Boundary Verified
                        </span>
                        <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px;">
                            SAMPL & R 4.3.3 Benchmark Compliance
                        </div>
                    </div>
                </div>
                """
            )

        # Tab Navigation
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
            <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 24px 0 12px 0; border-top: 1px solid #e2e8f0; margin-top: 24px;">
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
        f"🚀 Launching StatioMed AI Native Gradio on port {port} (Clinical Light Theme)..."
    )
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css,
        js=startup_js,
        ssr_mode=False,
    )
