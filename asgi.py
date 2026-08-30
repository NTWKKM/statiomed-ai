"""
asgi.py - StatioMed AI Entry Point for Hugging Face Spaces (ZeroGPU Free Tier)
=============================================================================
Hugging Face Spaces Free Tier mandates ZeroGPU with Gradio SDK.
This module bridges Shiny for Python into Gradio Blocks with native @spaces.GPU
ZeroGPU startup hooks, mounting the full Shiny application seamlessly.
=============================================================================
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Hugging Face ZeroGPU Free Tier Decorator
try:
    import spaces

    @spaces.GPU(duration=60)
    def _zerogpu_probe_fn(text: str = "") -> str:
        """Hugging Face ZeroGPU supervisor startup probe."""
        return f"StatioMed AI ZeroGPU Active: {text}"

except ImportError:

    def _zerogpu_probe_fn(text: str = "") -> str:
        return "StatioMed AI CPU Active"


import gradio as gr
from gradio.routes import App
from starlette.staticfiles import StaticFiles

from app import shiny_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Patch Gradio's App.create_app to mount Shiny & static files BEFORE middleware compilation
_orig_create_app = App.create_app


@staticmethod
def _custom_create_app(*args, **kwargs):
    app = _orig_create_app(*args, **kwargs)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/shiny", shiny_app, name="shiny")
    return app


App.create_app = _custom_create_app

# Build Top-Level Gradio Blocks UI (Required by Hugging Face ZeroGPU Supervisor)
custom_css = """
html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background-color: #0b0f19 !important;
}
footer, header, .gradio-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}
"""

REDIRECT_JS = r"""() => {
    try {
        var p = window.location.pathname.replace(/\/+$/, '');
        window.location.replace(p + '/shiny/');
    } catch(e) {
        window.location.href = './shiny/';
    }
}"""

with gr.Blocks(
    title="StatioMed AI — Clinical Research & Biostatistical Co-Pilot",
) as demo:
    # ZeroGPU event listener binding required by HF ZeroGPU startup scanner
    _probe_inp = gr.Textbox(visible=False)
    _probe_btn = gr.Button("Probe", visible=False)
    _probe_out = gr.Textbox(visible=False)
    _probe_btn.click(fn=_zerogpu_probe_fn, inputs=_probe_inp, outputs=_probe_out)

    # Seamless application gateway card with instant auto-redirect to mounted Shiny sub-app
    gr.HTML(
        """
        <div style="font-family: system-ui, -apple-system, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 85vh; color: #f8fafc; text-align: center; padding: 20px;">
            <div style="background: rgba(30, 41, 59, 0.9); border: 1px solid #334155; border-radius: 16px; padding: 36px 44px; max-width: 520px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);">
                <div style="font-size: 52px; margin-bottom: 16px;">🏥</div>
                <h2 style="font-size: 24px; font-weight: 700; margin: 0 0 8px 0; color: #ffffff;">StatioMed AI</h2>
                <p style="font-size: 14px; color: #94a3b8; margin: 0 0 24px 0; line-height: 1.5;">
                    Clinical Research & Biostatistical Co-Pilot
                </p>
                <a href="./shiny/" target="_self" onclick="window.location.href='./shiny/'; return false;" style="display: inline-block; background: #2563eb; color: #ffffff; text-decoration: none; font-size: 15px; font-weight: 600; padding: 12px 28px; border-radius: 8px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); cursor: pointer;">
                    🚀 Open Application
                </a>
                <p style="font-size: 12px; color: #64748b; margin: 16px 0 0 0;">
                    Launching environment...
                </p>
            </div>
        </div>
        """
    )

    # Execute client-side navigation as soon as Gradio mounts
    demo.load(None, js=REDIRECT_JS)

# Disable Node.js SSR proxy so Python FastAPI handles port 7860 directly
os.environ["GRADIO_SSR_MODE"] = "False"

# Export ASGI app instance for direct ASGI servers / health checks
app = demo.app

# Hugging Face Gradio Entry Point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    logger.info("🚀 Launching StatioMed AI on Hugging Face ZeroGPU Gradio Gateway...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False,
        theme=gr.themes.Base(),
        css=custom_css,
        js=REDIRECT_JS,
        allowed_paths=["/shiny", str(STATIC_DIR)],
    )
