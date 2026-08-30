"""
asgi.py - StatioMed AI Entry Point for Hugging Face Spaces (ZeroGPU Free Tier)
=============================================================================
Hugging Face Spaces Free Tier (Mid-2026+) mandates ZeroGPU with Gradio SDK.
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

# Build Top-Level Gradio Blocks UI (Required by Hugging Face ZeroGPU Supervisor)
custom_css = """
body, html {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background-color: #ffffff !important;
}
footer {
    display: none !important;
}
.gradio-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    overflow: hidden !important;
    background-color: #ffffff !important;
}
iframe {
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    background-color: #ffffff !important;
}
"""

with gr.Blocks(
    title="StatioMed AI — Clinical Research & Biostatistical Co-Pilot",
) as demo:
    # ZeroGPU event listener binding required by HF ZeroGPU startup scanner
    _probe_inp = gr.Textbox(visible=False)
    _probe_btn = gr.Button("Probe", visible=False)
    _probe_out = gr.Textbox(visible=False)
    _probe_btn.click(fn=_zerogpu_probe_fn, inputs=_probe_inp, outputs=_probe_out)

    # Full-screen responsive iframe hosting the mounted Shiny for Python application
    gr.HTML(
        '<iframe src="/shiny/" '
        'style="width: 100vw; height: 100vh; border: none; overflow: hidden; display: block; background: #ffffff;" '
        'allow="camera; microphone; clipboard-read; clipboard-write;"></iframe>'
    )

# Wrap demo.launch to guarantee Shiny is mounted on the active FastAPI server
_orig_launch = demo.launch


def custom_launch(*args, **kwargs):
    kwargs.setdefault("ssr_mode", False)
    app, local_url, share_url = _orig_launch(*args, **kwargs)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/shiny", shiny_app, name="shiny")
    return app, local_url, share_url


demo.launch = custom_launch

# Mount on demo.app initial reference as fallback
if STATIC_DIR.exists():
    demo.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

demo.app.mount("/shiny", shiny_app, name="shiny")

# Export ASGI app instance for direct ASGI servers / health checks
app = demo.app

# Hugging Face Gradio Entry Point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    logger.info("🚀 Launching StatioMed AI on Hugging Face ZeroGPU Gradio Gateway...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Base(),
        css=custom_css,
        allowed_paths=["/shiny", str(STATIC_DIR)],
    )
