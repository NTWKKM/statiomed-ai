"""
core - StatioMed AI Core State & Shared Services
================================================
Central exports for application state, constants, color tokens,
SAMPL formatters, and column matching heuristics.
================================================
"""

from core.common import (
    format_sampl_ci,
    format_sampl_p_value,
    get_color_palette,
    render_card_html,
    select_variable_by_keyword,
)
from core.constants import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    CLINICAL_UNITS,
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    EQUATOR_GUIDELINES,
    PHI_RESTRICTED_COLUMNS,
)
from core.state import AppState

__all__ = [
    "APP_DESCRIPTION",
    "APP_NAME",
    "APP_VERSION",
    "AppState",
    "CLINICAL_UNITS",
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "EQUATOR_GUIDELINES",
    "PHI_RESTRICTED_COLUMNS",
    "format_sampl_ci",
    "format_sampl_p_value",
    "get_color_palette",
    "render_card_html",
    "select_variable_by_keyword",
]
