"""
⏱️ Time-Varying Covariate (TVC) UI Components Module

Provides reusable Shiny UI components for time-varying covariate Cox analysis.

Components:
- Data format selector and validator
- Column configuration cards (ID, time columns, covariates)
- Risk interval definition picker
- Long-format data preview table
- Example dataset loader

Usage:
    >>> from tabs._tvc_components import (
    ...     tvc_data_format_selector_ui,
    ...     tvc_column_config_ui,
    ...     tvc_risk_interval_picker_ui
    ... )
    >>>
    >>> # In your Shiny UI
    >>> ui.div(
    ...     tvc_data_format_selector_ui(),
    ...     tvc_column_config_ui(),
    ...     tvc_risk_interval_picker_ui()
    ... )
"""

from __future__ import annotations

import pandas as pd
from shiny import ui

from logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# DATA FORMAT COMPONENTS
# ==============================================================================


def tvc_data_format_selector_ui() -> ui.TagChild:
    """
    Create UI for selecting input data format (Wide vs Long).
    Minimal version: Moved detailed explanation to Reference tab.
    """
    return ui.card(
        ui.card_header("📄 Data Format Selection"),
        ui.div(
            ui.span("Select your data structure below. "),
            ui.span(
                "(See 'Reference' tab for detailed format specifications)",
                class_="text-muted",
                style="font-size: 0.9em;",
            ),
            class_="mb-2",
        ),
        ui.input_radio_buttons(
            "tvc_data_format",
            None,  # Label removed for cleaner look
            {
                "long": "📊 Long Format (Multiple rows per patient - Recommended)",
                "wide": "📈 Wide Format (One row per patient)",
            },
            selected="long",
            inline=False,
        ),
        class_="mb-3",
    )


# ==============================================================================
# COLUMN CONFIGURATION COMPONENTS
# ==============================================================================


def tvc_column_config_ui() -> ui.TagChild:
    """
    Create UI for configuring required columns for TVC analysis.

    Returns:
        ui.TagChild: Card with column selectors
    """
    return ui.card(
        ui.card_header("⚙️ Column Configuration"),
        ui.markdown("**Columns for Time-Varying Cox analysis:**"),
        # --- Identification & Time Columns ---
        ui.div(ui.strong("Identification & Time"), class_="text-muted mt-3 mb-2"),
        ui.layout_columns(
            ui.input_select(
                "tvc_id_col",
                "🆔 Patient ID Column:",
                choices={"Select...": "Select..."},
            ),
            ui.div(
                ui.input_select(
                    "tvc_start_col",
                    "⏱️ Interval Start Time:",
                    choices={"Select...": "Select..."},
                ),
                id="div_tvc_start_col",
            ),
            ui.input_select(
                "tvc_stop_col",
                "⏱️ Interval Stop Time:",
                choices={"Select...": "Select..."},
            ),
            col_widths=[4, 4, 4],
        ),
        ui.input_select(
            "tvc_event_col",
            "🚫 Event Indicator (0=censored, 1=event):",
            choices={"Select...": "Select..."},
        ),
        # --- Covariate Selection ---
        ui.div(ui.strong("Covariates"), class_="text-muted mt-4 mb-2"),
        ui.markdown("""
            **Time-Varying Covariates:** Columns that can change over time within intervals
            (e.g., treatment status, lab values, symptoms)
            """),
        ui.input_selectize(
            "tvc_tvc_cols",
            "Select Time-Varying Covariates:",
            choices={},
            selected=[],
            multiple=True,
            options={"placeholder": "Select covariates..."},
        ),
        ui.markdown("""
            **Static Covariates:** Columns constant within each patient
            (e.g., age at baseline, sex, initial diagnosis)
            """),
        ui.input_selectize(
            "tvc_static_cols",
            "Select Static Covariates:",
            choices={},
            selected=[],
            multiple=True,
            options={"placeholder": "Select covariates..."},
        ),
    )


# ==============================================================================
# RISK INTERVAL COMPONENTS
# ==============================================================================


def tvc_risk_interval_picker_ui() -> ui.TagChild:
    """
    Create UI for defining risk intervals (Wide format only).

    Returns:
        ui.TagChild: Card with interval picker and options
    """
    return ui.card(
        ui.card_header("📊 Risk Intervals Definition (Wide Format Only)"),
        ui.markdown("""
            **Risk intervals** divide the follow-up period into time windows.
            For example: [0, 1m, 3m, 6m, 12m] creates 4 intervals.
            
            Last-observed values of time-varying covariates are "carried forward" within intervals.
            """),
        ui.input_radio_buttons(
            "tvc_interval_method",
            "How to define intervals:",
            {
                "auto": "🤖 Auto-detect (from TVC column names like 'tvc_3m', 'tvc_6m')",
                "quantile": "📈 Quantile-based (equal number of events in each interval)",
                "manual": "✍️ Manual (specify time points below)",
            },
            selected="manual",  # Default to Manual as per clinical preference
            inline=False,
        ),
        # Manual interval input (conditionally shown)
        ui.div(
            ui.markdown("**Specify time points (comma-separated):**"),
            # Preset Buttons
            ui.div(
                ui.span("Presets: ", class_="tvc-preset-label"),
                ui.input_action_button(
                    "btn_tvc_preset_quarterly",
                    "Quarterly (3m)",
                    class_="btn-sm btn-outline-primary tvc-preset-btn",
                ),
                ui.input_action_button(
                    "btn_tvc_preset_biannual",
                    "Biannual (6m)",
                    class_="btn-sm btn-outline-primary tvc-preset-btn",
                ),
                ui.input_action_button(
                    "btn_tvc_preset_yearly",
                    "Yearly (12m)",
                    class_="btn-sm btn-outline-primary",
                ),
                class_="tvc-preset-container",
            ),
            ui.input_text(
                "tvc_manual_intervals",
                "Enter custom intervals:",
                placeholder="0, 1, 3, 6, 12, 24",
                value="",
            ),
            id="tvc_manual_interval_div",
            style="display:none;",  # Controlled by JS/Reactive
        ),
        ui.markdown("""
            **Preview:**
            <span id='tvc_interval_preview' style='font-family: monospace; color: #555;'>
            Intervals will appear here
            </span>
            """),
    )


# ==============================================================================
# DATA PREVIEW COMPONENTS
# ==============================================================================


def tvc_data_preview_card_ui() -> ui.TagChild:
    """
    Create UI card for previewing long-format data structure.

    Returns:
        ui.TagChild: Card with data preview table and row count
    """
    return ui.card(
        ui.card_header("🛥️ Data Preview (First 50 Rows)"),
        ui.div(
            ui.strong("Data Structure Summary:"),
            ui.output_ui("tvc_preview_summary"),
            style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 15px;",
        ),
        ui.output_data_frame("tvc_preview_table"),
        ui.div(
            ui.markdown(
                "**Note:** Only first 50 rows displayed. Full dataset is used for analysis."
            ),
            style="font-size: 0.85em; color: #666; margin-top: 10px;",
        ),
    )


# ==============================================================================
# MODEL CONFIGURATION COMPONENTS
# ==============================================================================


def tvc_model_config_ui() -> ui.TagChild:
    """
    Create UI for TVC Cox model configuration options.

    Returns:
        ui.TagChild: Card with model options
    """
    return ui.card(
        ui.card_header("🔧 Model Configuration"),
        ui.accordion(
            ui.accordion_panel(
                "🤖 Fitting Method",
                ui.input_radio_buttons(
                    "tvc_fitting_method",
                    "Select algorithm:",
                    {
                        "auto": "Auto (CoxTimeVaryingFitter)",
                        "ctvf": "CoxTimeVaryingFitter (recommended)",
                    },
                    selected="auto",
                    inline=True,
                ),
            ),
            ui.accordion_panel(
                "⚠️ Advanced Options",
                ui.input_numeric(
                    "tvc_penalizer",
                    "Ridge Penalizer (L2 regularization, 0=none):",
                    value=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                ),
                ui.markdown("""
                    - **0.0:** Standard Cox (no regularization)
                    - **0.1-1.0:** Small penalty (useful if unstable)
                    - **>1.0:** Strong regularization (shrinks coefficients)
                    """),
            ),
            open=False,
        ),
    )


# ==============================================================================
# INFORMATION & HELP COMPONENTS
# ==============================================================================


def tvc_info_panel_ui() -> ui.TagChild:
    """
    Create info panel with quick reference and common questions.

    Returns:
        ui.TagChild: Card with info and FAQ
    """
    return ui.card(
        ui.card_header("ℹ️ Information & FAQ"),
        ui.accordion(
            ui.accordion_panel(
                "🆔 What are time-varying covariates?",
                ui.markdown("""
                    **Time-varying covariates** are variables that can change their values during follow-up.
                    
                    *Examples:*
                    - Drug dosage (adjusted over time)
                    - Lab values (measured periodically)
                    - Treatment status (switched during study)
                    
                    Unlike standard Cox regression (assumes constant covariates), TVC Cox allows modeling
                    of dynamic effects.
                    """),
            ),
            ui.accordion_panel(
                "👀 When to use Long vs Wide format?",
                ui.markdown("""
                    | Aspect | Long Format | Wide Format |
                    |--------|-------------|-------------|
                    | **Rows/Patient** | Multiple | One |
                    | **Structure** | Interval-based | Measurement-based |
                    | **Best for** | Complex follow-up schedules | Regular measurements |
                    | **Preparation** | Ready to use | Requires transformation |
                    | **Example** | Medical visit notes | Quarterly labs |
                    """),
            ),
            ui.accordion_panel(
                "📊 How are intervals created?",
                ui.markdown("""
                    Three methods to define intervals:
                    
                    1. **Auto-detect (Recommended):**
                       - Extracts time points from column names
                       - E.g., 'tvc_3m' → extracts time=3
                       - Creates intervals: [0-3m], [3m-6m], [6m-12m], etc.
                    
                    2. **Quantile-based:**
                       - Divides data into roughly equal event frequencies
                       - Ensures sufficient data per interval
                    
                    3. **Manual:**
                       - User specifies exact time points
                       - E.g., "0, 1, 3, 6, 12, 24"
                       - Maximum control but requires planning
                    """),
            ),
            ui.accordion_panel(
                "🛠️ Data transformation example",
                ui.markdown("""
                    **Input (Wide Format):**
                    ```
                    ID | time | event | tvc_0m | tvc_3m | tvc_6m
                    1  | 12   | 1     | 100    | 110    | 120
                    2  | 6    | 0     | 95     | 98     | NA
                    ```
                    
                    **Output (Long Format):**
                    ```
                    ID | start | stop | event | tvc
                    1  | 0     | 3    | 0     | 100
                    1  | 3     | 6    | 0     | 110
                    1  | 6     | 12   | 1     | 120
                    2  | 0     | 3    | 0     | 95
                    2  | 3     | 6    | 0     | 98
                    ```
                    
                    **Key Points:**
                    - Event=1 only in final interval
                    - Covariate values carried forward (last observed)
                    - Each patient split into multiple rows
                    """),
            ),
            open=False,
        ),
    )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of numeric column names from DataFrame.

    Parameters:
        df: Input DataFrame

    Returns:
        list[str]: Column names with numeric dtype
    """
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of categorical column names from DataFrame.

    Parameters:
        df: Input DataFrame

    Returns:
        list[str]: Column names with object/categorical dtype
    """
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def detect_tvc_columns(df: pd.DataFrame, prefix: str = "tvc_") -> list[str]:
    """
    Auto-detect time-varying covariate columns by name pattern.

    Looks for columns matching pattern like 'tvc_0m', 'tvc_3m', 'tvc_baseline', etc.

    Parameters:
        df: Input DataFrame
        prefix: Column name prefix to search for (default: 'tvc_')

    Returns:
        list[str]: Detected TVC column names, sorted by implied time
    """
    import re

    tvc_cols = []
    for col in df.columns:
        if col.lower().startswith(prefix):
            tvc_cols.append(col)

    # Sort by extracted time value
    def extract_time(col_name: str) -> float:
        match = re.search(r"(\d+)", col_name)
        return float(match.group(1)) if match else 0

    return sorted(tvc_cols, key=extract_time)


def detect_static_columns(df: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
    """
    Auto-detect static covariate columns by exclusion.

    Assumes static columns are not identified as ID, time, event, or TVC columns.
    Typically: age, sex, baseline diagnosis, etc.

    Parameters:
        df: Input DataFrame
        exclude_cols: Columns to exclude (ID, time, event, TVC)

    Returns:
        list[str]: Detected static column names
    """
    all_cols = set(df.columns)
    excluded = set(exclude_cols)
    static_cols = list(all_cols - excluded)

    return sorted(static_cols)


def format_interval_preview(intervals: list[float]) -> str:
    """
    Format risk intervals for display.

    Parameters:
        intervals: List of time points [0, 1, 3, 6, 12]

    Returns:
        str: Formatted string like "[0-1], [1-3], [3-6], [6-12]"
    """
    if not intervals or len(intervals) < 2:
        return "Invalid intervals"

    intervals = sorted(set(intervals))
    interval_strings = []

    for i in range(len(intervals) - 1):
        interval_strings.append(f"[{intervals[i]}-{intervals[i + 1]}]")

    return ", ".join(interval_strings)
