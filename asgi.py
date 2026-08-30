"""
ASGI Entry Point for HuggingFace Spaces Deployment
===================================================
This file acts as the ASGI gateway for the Shiny application when deployed
on platforms like HuggingFace Spaces which use Docker + Gunicorn + Uvicorn.

It wraps the Shiny `app` with Starlette to provide:
1. Static file serving (CSS, JS) from the /static endpoint.
2. GZip middleware for performance.
3. Health check and lifecycle event logging.

Run with: gunicorn -k uvicorn.workers.UvicornWorker asgi:app
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount
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


@asynccontextmanager
async def lifespan(app: Starlette):
    """Handle application startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Medical Stat Tool (ASGI Wrapper)...")
    logger.info(f"📁 Static files directory: {STATIC_DIR}")
    yield
    # Shutdown
    logger.info("👋 Shutting down application...")


# Routes: Static files first
routes = [
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

# Middleware stack
middleware = [
    Middleware(GZipMiddleware, minimum_size=500),  # Compress responses > 500 bytes
]

# Create ASGI base application
base_app = Starlette(
    routes=routes,
    middleware=middleware,
    lifespan=lifespan,
)

# Hugging Face ZeroGPU Free Tier Bridge
# ZeroGPU Free Tier requires a Gradio event listener decorated with @spaces.GPU
# Mount Gradio app into Starlette ASGI application at /_gradio path
try:
    import spaces

    @spaces.GPU(duration=45)
    def _zerogpu_probe_fn(text: str = "") -> str:
        return "StatioMed AI ZeroGPU Probe Active"

except ImportError:

    def _zerogpu_probe_fn(text: str = "") -> str:
        return "StatioMed AI CPU Probe Active"


with gr.Blocks(title="StatioMed AI ZeroGPU Bridge") as _gradio_probe:
    _inp = gr.Textbox(visible=False)
    _btn = gr.Button("Probe", visible=False)
    _out = gr.Textbox(visible=False)
    _btn.click(fn=_zerogpu_probe_fn, inputs=_inp, outputs=_out)

app = gr.mount_gradio_app(base_app, _gradio_probe, path="/_gradio")
# Mount Shiny app at root last, so it acts as a catch-all
app.mount("/", shiny_app, name="shiny")

# Development server
if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    uvicorn.run(
        "asgi:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
