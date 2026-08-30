"""
tabs/_common.py - Backward Compatibility Adapter for Core Shared Services
==========================================================================
Re-exports from `core.common` to preserve backward compatibility with legacy
modules and tests while centralizing the true implementation in `core/`.
==========================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.common import (
    get_color_palette,
    select_variable_by_keyword,
)

if TYPE_CHECKING:
    from shiny.ui import TagChild


def wrap_with_container(content: "TagChild") -> "TagChild":
    """Wraps UI content with the .app-container CSS class (legacy helper)."""
    try:
        from shiny import ui

        return ui.div(content, class_="app-container")
    except ImportError:
        return content


__all__ = ["get_color_palette", "select_variable_by_keyword", "wrap_with_container"]
