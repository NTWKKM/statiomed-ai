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
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from app import app as shiny_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Medical Stat Tool (ASGI Wrapper)...")
    logger.info(f"📁 Static files directory: {STATIC_DIR}")
    yield
    # Shutdown
    logger.info("👋 Shutting down application...")


# Routes: Static files first, then Shiny app at root
routes = [
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    Mount("/", app=shiny_app, name="shiny"),
]

# Middleware stack
middleware = [
    Middleware(GZipMiddleware, minimum_size=500),  # Compress responses > 500 bytes
]

# Create ASGI application
app = Starlette(
    routes=routes,
    middleware=middleware,
    lifespan=lifespan,
)

# Development server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "asgi:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
        log_level="info",
    )
