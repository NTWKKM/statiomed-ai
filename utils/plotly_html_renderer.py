"""
📊 Plotly HTML Renderer Utility

Converts Plotly figures to HTML strings using CDN for better compatibility
in restricted environments (corporate firewalls, Private Spaces, etc.)

Usage:
    from utils.plotly_html_renderer import plotly_figure_to_html

    html_str = plotly_figure_to_html(
        fig=my_plotly_figure,
        div_id="my_unique_id",
        include_plotlyjs='cdn',
        responsive=True
    )
    return ui.HTML(html_str)
"""

from __future__ import annotations

import logging
import re
import uuid

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore

logger = logging.getLogger(__name__)


def plotly_figure_to_html(
    fig: "go.Figure" | None = None,
    div_id: str | None = None,
    include_plotlyjs: str | bool = "cdn",
    height: int | None = None,
    width: int | None = None,
    responsive: bool = True,
) -> str:
    """
    Render a Plotly Figure to an embeddable HTML string.

    Returns a sanitized HTML fragment suitable for embedding (e.g., in a UI).
    If rendering fails, returns a styled pulse-animated placeholder.

    Performance: Avoids heavy Figure duplication unless layout overrides are required.
    """
    if fig is None:
        return _create_placeholder_html("⏳ Waiting for data...")

    if go is None:
        return _create_placeholder_html("⚠️ Plotly is not installed")

    if not isinstance(fig, go.Figure):
        return _create_placeholder_html(f"⚠️ Invalid figure type: {type(fig).__name__}")

    # Generate or sanitize div_id
    div_id = _sanitize_div_id(div_id) if div_id else f"plotly-{uuid.uuid4().hex[:12]}"

    try:
        # PERFORMANCE OPTIMIZATION:
        # Only copy/mutate the figure if we actually need to change properties.
        # This saves memory for large datasets.
        needs_update = (height is not None) or (width is not None) or responsive

        target_fig = fig
        if needs_update:
            target_fig = go.Figure(fig)
            if responsive:
                target_fig.update_layout(autosize=True)
            if height is not None:
                target_fig.update_layout(height=height)
            if width is not None:
                target_fig.update_layout(width=width)

        # Generate HTML
        html_str = target_fig.to_html(
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            div_id=div_id,
            config={
                "responsive": responsive,
                "displayModeBar": True,
                "displaylogo": False,
            },
        )

        return html_str

    except Exception:
        logger.exception("plotly_figure_to_html: Error generating HTML")
        return _create_placeholder_html("Error rendering plot")


def _sanitize_div_id(div_id: str) -> str:
    """Produce a safe HTML id by removing invalid characters."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", str(div_id))
    if sanitized and not sanitized[0].isalpha():
        sanitized = "plot-" + sanitized
    return sanitized if sanitized else f"plot-{uuid.uuid4().hex[:8]}"


def _create_placeholder_html(message: str) -> str:
    """Create a styled pulse-animated placeholder (Skeleton UI)."""
    import html

    from tabs._common import get_color_palette

    COLORS = get_color_palette()
    escaped_message = html.escape(str(message))

    # Using a unique ID for the style ensures no collisions if multiple placeholders exist
    style_id = f"pulse-{uuid.uuid4().hex[:8]}"

    return f"""
    <style>
        .shiny-stat-placeholder {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 200px;
            color: {COLORS["secondary"]};
            text-align: center;
            padding: 40px 20px;
            background: {COLORS["background"]}; /* #FAFAFA */
            border-radius: 12px;
            border: 1px solid {COLORS["border"]}; /* #E2E8F0 */
            font-family: 'Inter', -apple-system, system-ui, sans-serif;
            font-size: 14px;
            position: relative;
            overflow: hidden;
        }}
        .shiny-stat-placeholder::after {{
            content: "";
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: {style_id}-pulse 1.5s infinite;
        }}
        @keyframes {style_id}-pulse {{
            0%   {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
    </style>
    <div class="shiny-stat-placeholder">
        <div style="position: relative; z-index: 1;">
            {escaped_message}
        </div>
    </div>
    """
