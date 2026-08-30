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

# Custom Theme & Styling for Clinical Grade UI
custom_css = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#0284c7",
    button_primary_background_fill_hover="#0369a1",
    button_primary_text_color="#ffffff",
)


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
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0 16px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 12px;">
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
                        <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
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
            <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 24px 0 12px 0; border-top: 1px solid #f1f5f9; margin-top: 24px;">
                StatioMed AI © 2026 | Built for Clinical Researchers & Biostatisticians | Powered by Gradio & smolagents
            </div>
            """
        )

    return demo


demo = build_app()
app = demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    logger.info(f"🚀 Launching StatioMed AI Native Gradio on port {port}...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css,
        ssr_mode=False,
    )
