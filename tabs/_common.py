from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shiny.ui import TagChild


def wrap_with_container(content: "TagChild") -> "TagChild":
    """
    Wraps UI content with the .app-container CSS class.

    Args:
        content: The UI content (TagChild) to wrap.

    Returns:
        A shiny.ui.div element with class='app-container'.
    """
    from shiny import ui  # Import inside function to avoid circular imports

    return ui.div(content, class_="app-container")


def get_color_palette() -> dict[str, str]:
    """
    Returns a unified color palette dictionary for all modules.
    Ensures consistency across the application.

    Returns:
        dict[str, str]: A dictionary mapping color names to hex codes.
    """
    slate_50 = "#F8FAFC"  # Slate 50
    return {
        # Primary colors - Slate theme
        "primary": "#0F172A",  # Slate 900
        "primary_dark": "#020617",  # Slate 950
        "primary_light": slate_50,
        "secondary": "#64748B",  # Slate 500
        # Neutral colors - Light theme
        "smoke_white": slate_50,
        "text": "#0F172A",  # Slate 900
        "text_secondary": "#64748B",  # Slate 500
        "border": "#E2E8F0",  # Slate 200
        "background": "#FAFAFA",  # Very clean off-white
        "surface": "#FFFFFF",
        # Status/Semantic colors - Soft/desaturated pastels
        "success": "#059669",  # Emerald 600
        "danger": "#DC2626",  # Red 600
        "warning": "#D97706",  # Amber 600
        "info": "#475569",  # Slate 600
        "neutral": "#CBD5E1",  # Slate 300
    }


def select_variable_by_keyword(
    columns: list[str], keywords: list[str], default_to_first: bool = True
) -> str | None:
    """
    Intelligently attempts to select a default variable from a list of columns
    based on a prioritized list of keywords with token-boundary matching to prevent
    false-positive substring collisions (e.g., 'n_statin' matching 'mean_statin').

    Matching Tiers:
    1. Exact Match (case-insensitive)
    2. Token & Word-boundary Match (case-insensitive regex)
    3. Prefix/Suffix Delimited Match (with '_' or '.')
    """
    if not columns:
        return None

    # Tier 1: Exact match (case-insensitive)
    for k in keywords:
        k_lower = k.strip().lower()
        for col in columns:
            if k_lower == col.strip().lower():
                return col

    # Tier 2: Token / Word-boundary Match (case-insensitive regex)
    for k in keywords:
        k_lower = k.strip().lower()
        pattern = re.compile(
            rf"(^|[^a-zA-Z0-9]){re.escape(k_lower)}([^a-zA-Z0-9]|$)", re.IGNORECASE
        )
        for col in columns:
            if pattern.search(col):
                return col

    # Tier 3: Delimited prefix / suffix match
    for k in keywords:
        k_lower = k.strip().lower()
        for col in columns:
            col_lower = col.strip().lower()
            if (
                col_lower.startswith(f"{k_lower}_")
                or col_lower.endswith(f"_{k_lower}")
                or col_lower.startswith(f"{k_lower}.")
                or col_lower.endswith(f".{k_lower}")
            ):
                return col

    # Default fallback
    if default_to_first:
        return columns[0]

    return None
