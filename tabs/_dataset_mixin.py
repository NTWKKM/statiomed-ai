"""
📦 Dataset Selector Mixin — Shared factory for multi-dataset tab modules.

All analysis tabs that accept both an original and a matched/imputed dataset
share identical boilerplate for:
  - current_df()     : reactive to select between original / matched
  - render_matched_info()   : banner when matched data is available
  - render_dataset_selector(): radio buttons to switch datasets

Usage
-----
Call ``register_dataset_selector(...)`` inside your ``@module.server`` function.
It registers the necessary render/reactive functions into the module's session
and returns a ``current_df`` ``reactive.Calc`` that other reactives can depend on.

Example
-------
::

    @module.server
    def my_server(input, output, session, df, var_meta, df_matched, is_matched):
        from tabs._dataset_mixin import register_dataset_selector

        current_df = register_dataset_selector(
            input=input,
            output=output,
            df=df,
            df_matched=df_matched,
            is_matched=is_matched,
            radio_input_id="radio_my_module_source",  # Must be unique per module
            title="📊 My Analysis Module",
        )

        @reactive.Effect
        def _update():
            d = current_df()  # Use like a normal reactive.Calc
            ...

Notes
-----
- The ``radio_input_id`` MUST match the ``ui.output_ui`` IDs declared in the
  corresponding ``@module.ui`` function:
    * ``ui.output_ui("ui_matched_info")``
    * ``ui.output_ui("ui_dataset_selector")``
    * ``ui.output_ui("ui_title_with_summary")``  (optional)
- Each module MUST use a globally unique ``radio_input_id`` to prevent Shiny
  input namespace collisions between module instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shiny import reactive, render, ui

if TYPE_CHECKING:
    import pandas as pd


def register_dataset_selector(
    input: Any,
    output: Any,
    df: "reactive.Value[pd.DataFrame | None]",
    df_matched: "reactive.Value[pd.DataFrame | None]",
    is_matched: "reactive.Value[bool]",
    radio_input_id: str,
    title: str = "",
) -> "reactive.Calc":
    """
    Register dataset selection UI and reactive logic inside a Shiny module.

    Attaches three output renderers to the module session (``ui_matched_info``,
    ``ui_dataset_selector``, and ``ui_title_with_summary``) and returns a
    ``reactive.Calc`` that resolves to the currently selected DataFrame.

    Parameters
    ----------
    input : module input object
        The ``input`` argument provided to the calling ``@module.server`` function.
    output : module output object
        The ``output`` argument provided to the calling ``@module.server`` function.
    df : reactive.Value[pd.DataFrame | None]
        Reactive holding the original (full) dataset.
    df_matched : reactive.Value[pd.DataFrame | None]
        Reactive holding the matched/imputed dataset (may be None).
    is_matched : reactive.Value[bool]
        Reactive flag — True when a matched dataset has been generated.
    radio_input_id : str
        The Shiny input ID for the dataset-selector radio buttons.
        Must be unique per module and match the ID declared in the UI.
    title : str
        Display title for the module header (e.g. "📊 Correlation Analysis").
        Used in the ``ui_title_with_summary`` renderer. Pass ``""`` to skip.

    Returns
    -------
    reactive.Calc
        A reactive calc resolving to ``pd.DataFrame | None`` — the currently
        active dataset (original or matched) as selected by the user.
    """

    # ------------------------------------------------------------------
    # 1. Core reactive: resolve active dataset
    # ------------------------------------------------------------------
    @reactive.Calc
    def current_df() -> "pd.DataFrame | None":
        """Return the user-selected dataset (original or matched)."""
        try:
            source = getattr(input, radio_input_id)()
        except Exception:
            source = "original"
        if is_matched.get() and source == "matched":
            return df_matched.get()
        return df.get()

    # ------------------------------------------------------------------
    # 2. Optional title with row/column summary
    # ------------------------------------------------------------------
    if title is not None:
        @output
        @render.ui
        def ui_title_with_summary():
            """Render title with dataset dimensions."""
            if not title:
                return None
            d = current_df()
            if d is not None:
                return ui.div(
                    ui.h3(title),
                    ui.p(
                        f"{len(d):,} rows | {len(d.columns)} columns",
                        class_="text-secondary mb-3",
                    ),
                )
            return ui.h3(title)

    # ------------------------------------------------------------------
    # 3. Matched dataset availability banner
    # ------------------------------------------------------------------
    @output
    @render.ui
    def ui_matched_info():
        """Display a banner when a matched dataset is available."""
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** — select it below to analyse matched data",
                    class_="alert alert-info",
                )
            )
        return None

    # ------------------------------------------------------------------
    # 4. Dataset selector radio buttons
    # ------------------------------------------------------------------
    @output
    @render.ui
    def ui_dataset_selector():
        """Render radio buttons to switch between original and matched datasets."""
        if not is_matched.get():
            return None

        original = df.get()
        matched = df_matched.get()
        original_len = len(original) if original is not None else 0
        matched_len = len(matched) if matched is not None else 0

        return ui.input_radio_buttons(
            radio_input_id,
            "📊 Select Dataset:",
            {
                "original": f"📊 Original ({original_len:,} rows)",
                "matched": f"✅ Matched ({matched_len:,} rows)",
            },
            selected="original",
            inline=True,
        )

    return current_df
