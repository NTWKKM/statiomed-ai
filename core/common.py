"""
core/common.py - StatioMed AI Shared Utilities & Formatting Helpers
===================================================================
Color palette tokens, variable matching heuristics, SAMPL-compliant
formatting helpers (P-values, effect sizes, percentages), and UI cards.
===================================================================
"""

from __future__ import annotations

import html
import re


def get_color_palette() -> dict[str, str]:
    """
    Returns a unified color palette dictionary for all modules.
    Ensures consistency across the application.
    """
    slate_50 = "#F8FAFC"
    return {
        # Primary colors - Slate theme
        "primary": "#0F172A",
        "primary_dark": "#020617",
        "primary_light": slate_50,
        "secondary": "#64748B",
        # Neutral colors - Light theme
        "smoke_white": slate_50,
        "text": "#0F172A",
        "text_secondary": "#64748B",
        "border": "#E2E8F0",
        "background": "#FAFAFA",
        "surface": "#FFFFFF",
        # Status/Semantic colors
        "success": "#059669",
        "danger": "#DC2626",
        "warning": "#D97706",
        "info": "#475569",
        "neutral": "#CBD5E1",
    }


def select_variable_by_keyword(
    columns: list[str], keywords: list[str], default_to_first: bool = True
) -> str | None:
    """
    Intelligently attempts to select a default variable from a list of columns
    based on a prioritized list of keywords with token-boundary matching.
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


def format_sampl_p_value(p: float | None) -> str:
    """
    Formats P-values according to SAMPL guidelines:
    - P < 0.001 for values < 0.001
    - P = 0.042 (3 decimals if < 0.10)
    - P = 0.24 (2 decimals if >= 0.10)
    - P > 0.99 for values > 0.99
    """
    if p is None or not isinstance(p, (int, float)):
        return "N/A"
    if p < 0.001:
        return "P < 0.001"
    elif p > 0.99:
        return "P > 0.99"
    elif p < 0.10:
        return f"P = {p:.3f}"
    else:
        return f"P = {p:.2f}"


def format_sampl_ci(point: float, lower: float, upper: float, decimals: int = 2) -> str:
    """
    Formats point estimate with 95% Confidence Interval according to SAMPL.
    e.g., '1.45 (95% CI: 1.12 to 1.89)'
    """
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(point)} (95% CI: {fmt.format(lower)} to {fmt.format(upper)})"


def render_card_html(
    title: str,
    content_html: str,
    badge_text: str | None = None,
    badge_color: str = "green",
) -> str:
    """
    Renders a unified styled card container for Gradio HTML outputs.
    """
    badge_html = ""
    if badge_text:
        bg = "#dcfce7" if badge_color == "green" else "#e0f2fe"
        fg = "#166534" if badge_color == "green" else "#0369a1"
        badge_html = f"<span style='background:{bg};color:{fg};padding:3px 10px;border-radius:999px;font-size:0.75rem;font-weight:600;'>{html.escape(badge_text)}</span>"

    return f"""
    <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.02);'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
            <h4 style='color:#0f172a;margin:0;font-size:1.05rem;font-weight:600;'>{html.escape(title)}</h4>
            {badge_html}
        </div>
        <div>
            {content_html}
        </div>
    </div>
    """
