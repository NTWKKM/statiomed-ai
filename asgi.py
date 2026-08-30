"""
asgi.py - Production ASGI Entry Point for Docker & Hugging Face Spaces
=============================================================================
Hosts Shiny for Python natively on Starlette ASGI with:
1. Direct root routing (/) for Shiny with full reactivity & WebSocket support.
2. Static file serving (/static) with GZip compression.
3. Zero-overhead Gunicorn + UvicornWorker execution in Docker containers.
4. Top-level Gradio Blocks bridge & @spaces.GPU probe for ZeroGPU compatibility.
=============================================================================
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# ZeroGPU Compatibility Hook
try:
    import spaces

    @spaces.GPU(duration=60)
    def _zerogpu_probe_fn(text: str = "") -> str:
        return f"StatioMed AI ZeroGPU Active: {text}"

except ImportError:

    def _zerogpu_probe_fn(text: str = "") -> str:
        return "StatioMed AI CPU Active"


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
    logger.info("🚀 Starting StatioMed AI (Starlette ASGI Gateway)...")
    logger.info(f"📁 Static files directory: {STATIC_DIR}")
    yield
    logger.info("👋 Shutting down application...")


# Routes: Static files first, Shiny app at root (catch-all)
routes = []
if STATIC_DIR.exists():
    routes.append(
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static")
    )
routes.append(Mount("/", app=shiny_app, name="shiny"))

# Middleware stack
middleware = [
    Middleware(GZipMiddleware, minimum_size=500),
]

# Canonical ASGI application for Gunicorn / Uvicorn (Docker SDK)
app = Starlette(
    routes=routes,
    middleware=middleware,
    lifespan=lifespan,
)

# Optional Gradio Blocks bridge for ZeroGPU supervisor discovery
try:
    import gradio as gr

    with gr.Blocks(title="StatioMed AI") as demo:
        _inp = gr.Textbox(visible=False)
        _btn = gr.Button("Probe", visible=False)
        _out = gr.Textbox(visible=False)
        _btn.click(fn=_zerogpu_probe_fn, inputs=_inp, outputs=_out)

    # Pre-generate config so Gradio never crashes on missing config
    demo.config = demo.get_config_file()
except Exception:
    demo = None

# Development / Direct Execution
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    logger.info(f"🚀 Starting Uvicorn on 0.0.0.0:{port}...")
    uvicorn.run(
        "asgi:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
