"""
📈 Correlation & ICC Analysis Module (Enhanced) - FIXED INTERPRETATION

Enhanced Features:
- Comprehensive statistics (CI, R², effect size)
- Matrix summary statistics
- HTML report download for all analyses
- Detailed interpretations (Fixed ICC display issue)

Updated: Uses dataset selector pattern like tab_diag.py
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, ui

from logger import get_logger
from tabs._common import (
    select_variable_by_keyword,
)
from tabs._dataset_mixin import register_dataset_selector
from utils import (
    correlation,  # Import from utils
)
from utils.download_helpers import safe_report_generation
from utils.formatting import create_missing_data_report_html, format_p_value
from utils.pdf_helpers import safe_pdf_report_generation
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.ui_helpers import create_download_status_badge


def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s[:80] or "value"


logger = get_logger(__name__)


# ✅ Use @module.ui decorator
@module.ui
def corr_ui() -> ui.TagChild:
    """
    Create the UI for correlation analysis tab.
    NO manual namespace needed - Shiny handles it automatically.
    """
    return ui.div(
        # Title + Data Summary inline
        ui.output_ui("ui_title_with_summary"),
        # Dataset Info Box
        ui.output_ui("ui_matched_info"),
        ui.br(),
        # Dataset Selector
        ui.output_ui("ui_dataset_selector"),
        ui.br(),
        # Main Analysis Tabs
        ui.navset_tab(
            # TAB 1: Pearson/Spearman Correlation (Pairwise)
            ui.nav_panel(
                "📈 Pairwise Correlation",
                ui.card(
                    ui.card_header("📈 Continuous Correlation Analysis"),
                    ui.layout_columns(
                        ui.input_select(
                            "coeff_type",
                            "Correlation Coefficient:",
                            choices={
                                "pearson": "Pearson",
                                "spearman": "Spearman",
                                "kendall": "Kendall",
                            },
                            selected="pearson",
                        ),
                        ui.input_select(
                            "cv1", "Variable 1 (X-axis):", choices=["Select..."]
                        ),
                        ui.input_select(
                            "cv2", "Variable 2 (Y-axis):", choices=["Select..."]
                        ),
                        col_widths=[3, 4, 4],
                    ),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_corr",
                            "📈 Analyze Correlation",
                            class_="btn-primary",
                            width="100%",
                        ),
                        # ✅ CHANGED: Use download_button
                        ui.div(
                            ui.download_button(
                                "btn_dl_corr",
                                "📥 HTML",
                                class_="btn-secondary",
                                width="100%",
                            ),
                            ui.download_button(
                                "btn_dl_corr_pdf",
                                "📥 PDF",
                                class_="btn-outline-danger mt-1",
                                width="100%",
                            ),
                            ui.output_ui("dl_status_corr"),
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_corr_result"),
                    full_screen=True,
                ),
            ),
            # TAB 2: Matrix/Heatmap (New!)
            ui.nav_panel(
                "📊 Matrix/Heatmap",
                ui.card(
                    ui.card_header("📊 Correlation Matrix & Heatmap"),
                    ui.input_selectize(
                        "matrix_vars",
                        "Select Variables (Multi-select):",
                        choices=["Select..."],
                        multiple=True,
                        selected=[],
                    ),
                    ui.input_select(
                        "matrix_method",
                        "Correlation Method:",
                        choices={
                            "pearson": "Pearson",
                            "spearman": "Spearman",
                            "kendall": "Kendall",
                        },
                        selected="pearson",
                    ),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_matrix",
                            "🎨 Generate Heatmap",
                            class_="btn-primary",
                            width="100%",
                        ),
                        # ✅ CHANGED: Use download_button
                        ui.div(
                            ui.download_button(
                                "btn_dl_matrix",
                                "📥 HTML",
                                class_="btn-secondary",
                                width="100%",
                            ),
                            ui.download_button(
                                "btn_dl_matrix_pdf",
                                "📥 PDF",
                                class_="btn-outline-danger mt-1",
                                width="100%",
                            ),
                            ui.output_ui("dl_status_matrix"),
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_matrix_result"),
                    full_screen=True,
                ),
            ),
            # TAB 3: Reference & Interpretation
            ui.nav_panel(
                "📖 Reference",
                ui.card(
                    ui.card_header("📚 Reference & Interpretation Guide"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("📈 Correlation (Relationship)"),
                            ui.markdown("""
                            **Concept:** Measures the strength and direction of the relationship between 
                            **two continuous variables**.

                            <div class="alert alert-warning" role="alert">
                            <strong>Warning:</strong> Correlation does not imply causation. A strong relationship between two variables does not mean one causes the other.
                            </div>

                            **1. Pearson (r):**
                            * **Best for:** Linear relationships (straight line), normally distributed data.
                            * **Sensitive to:** Outliers.
                            * **Returns:** R-squared (R²) = proportion of variance explained

                            **2. Spearman (rho) & Kendall (tau):**
                            * **Best for:** Monotonic relationships, non-normal data, or ranks.
                            * **Robust to:** Outliers.
                            * **Kendall's Tau** is often preferred for small datasets with many tied ranks.

                            **Interpretation of Coefficient (r, rho, or tau):**
                            * **+1.0:** Perfect Positive (As X goes up, Y goes up).
                            * **-1.0:** Perfect Negative (As X goes up, Y goes down).
                            * **0.0:** No relationship.

                            **Strength Guidelines:**
                            * **0.9 - 1.0:** Very Strong 🔥
                            * **0.7 - 0.9:** Strong 📈
                            * **0.5 - 0.7:** Moderate 📊
                            * **0.3 - 0.5:** Weak 📉
                            * **< 0.3:** Very Weak/Negligible
                            
                            **Confidence Intervals (95% CI):**
                            * Shows the range where the true correlation likely falls
                            * Wider CI = less precise estimate (usually with small samples)
                            """),
                        ),
                        ui.card(
                            ui.card_header("💡 Common Questions"),
                            ui.markdown("""
                            **Q: What is R-squared (R²)?**
                            * **A:** R² tells you the proportion of variance in Y that is explained by X. 
                            For example, R² = 0.64 means 64% of the variation in Y is explained by X.

                            **Q: Why use ICC instead of Pearson for reliability?**
                            * **A:** Pearson only measures linearity. If Rater A always gives exactly 10 points 
                            higher than Rater B, Pearson = 1.0 but they don't agree! ICC accounts for this.

                            **Q: What if p-value is significant but r is low (0.1)?**
                            * **A:** P-value means it's likely not zero. With large samples, tiny correlations 
                            can be "significant". **Focus on r-value magnitude** for clinical relevance.

                            **Q: How to interpret confidence intervals?**
                            * **A:** If 95% CI includes 0, the correlation is not statistically significant. 
                            Narrow CI = more precise estimate, Wide CI = less precise (need more data).
                            
                            **Q: How many variables do I need for ICC?**
                            * **A:** At least 2 (to compare two raters/methods). More raters = more reliable ICC.
                            """),
                        ),
                        col_widths=[6, 6],
                    ),
                    full_screen=True,
                ),
            ),
        ),
    )


# ✅ Use @module.server decorator properly
@module.server
def corr_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
) -> None:
    """
    Register server-side reactives, event handlers, and UI outputs for the Correlation & ICC Analysis tab.
    """

    # ==================== REACTIVE STATES ====================

    corr_result: reactive.Value[dict[str, Any] | None] = reactive.Value(
        None
    )  # Pairwise result
    matrix_result: reactive.Value[dict[str, Any] | None] = reactive.Value(
        None
    )  # Matrix result
    numeric_cols_list: reactive.Value[list[str]] = reactive.Value(
        []
    )  # List of numeric columns

    # ==================== DATASET SELECTION LOGIC ====================
    current_df = register_dataset_selector(
        input=input,
        output=output,
        df=df,
        df_matched=df_matched,
        is_matched=is_matched,
        radio_input_id="radio_corr_source",
        title="📈 Correlation Analysis",
    )

    # ==================== UPDATE NUMERIC COLUMNS ====================

    @reactive.Effect
    def _update_numeric_cols():
        """Update list of numeric columns when data changes."""
        data = current_df()
        if data is not None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols_list.set(cols)

            if cols:
                # ✅ FILTER: Filter columns starting with 'lab', 'value', 'values'
                filtered_cols = [
                    c for c in cols if c.lower().startswith(("lab", "value", "values"))
                ]

                # If no columns match, fallback to all numeric columns
                final_cols = filtered_cols if filtered_cols else cols

                # Pairwise selectors
                selected_v1 = select_variable_by_keyword(
                    final_cols, ["glucose", "lab_glucose"], default_to_first=True
                )
                ui.update_select("cv1", choices=final_cols, selected=selected_v1)

                remaining_cols = [c for c in final_cols if c != selected_v1]
                selected_v2 = select_variable_by_keyword(
                    remaining_cols, ["hba1c", "lab_hba1c"], default_to_first=True
                )
                ui.update_select("cv2", choices=final_cols, selected=selected_v2)

                # Matrix selector
                ui.update_selectize("matrix_vars", choices=cols, selected=cols[:5])

    # ==================== PAIRWISE CORRELATION ====================

    @reactive.Effect
    @reactive.event(input.btn_run_corr)
    def _run_correlation() -> None:
        """Run pairwise correlation analysis."""
        data = current_df()

        if data is None:
            ui.notification_show("No data available", type="error")
            return

        col1 = input.cv1()
        col2 = input.cv2()
        method = input.coeff_type()

        if not col1 or not col2:
            ui.notification_show("Please select two variables", type="warning")
            return

        if col1 == col2:
            ui.notification_show("Please select different variables", type="warning")
            return

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Calculating correlation...", detail="This may take a moment")
            res_stats, err, fig = correlation.calculate_correlation(
                data, col1, col2, method=method, var_meta=var_meta.get() or {}
            )

        if err:
            ui.notification_show(f"Error: {err}", type="error")
            corr_result.set(None)
        else:
            # Determine data label
            if is_matched.get() and input.radio_corr_source() == "matched":
                data_label = f"✅ Matched Data ({len(data)} rows)"
            else:
                data_label = f"📊 Original Data ({len(data)} rows)"

            corr_result.set(
                {
                    "stats": res_stats,
                    "figure": fig,
                    "method": method,
                    "var1": col1,
                    "var2": col2,
                    "data_label": data_label,
                }
            )
            ui.notification_show("✅ Correlation analysis complete", type="default")

    @render.ui
    def out_corr_result():
        """Display pairwise correlation results."""
        result = corr_result.get()
        if result is None:
            return ui.markdown(
                "*Results will appear here after clicking '📈 Analyze Correlation'*"
            )

        stats = result["stats"]

        # Format interpretation
        var1 = _html.escape(str(result["var1"]))
        var2 = _html.escape(str(result["var2"]))
        interpretation = _html.escape(str(stats.get("Interpretation", "")))
        sample_note = _html.escape(str(stats.get("Sample Note", "")))
        r2_raw = stats.get("R-squared (R²)", None)
        r2 = (
            float(r2_raw)
            if isinstance(r2_raw, (int, float)) and not pd.isna(r2_raw)
            else None
        )
        interp_html = f"""
        <div class='info-callout'>
            <strong>Interpretation:</strong> {interpretation}<br>
            <strong>R² = {f"{r2:.3f}" if r2 is not None else "N/A"}</strong> →
            {f"{r2 * 100:.1f}" if r2 is not None else "N/A"}% of variance in {var2} is explained by {var1}<br>
            <strong>Sample:</strong> {sample_note}
        </div>
        """

        return ui.div(
            ui.card(
                ui.card_header("Results"),
                ui.markdown(f"**Data Source:** {result['data_label']}"),
                ui.markdown(f"**Method:** {result['method'].title()}"),
                ui.output_data_frame("out_corr_table"),
                ui.HTML(interp_html),
                # Missing Data Report
                ui.HTML(
                    create_missing_data_report_html(
                        stats.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
                ui.card_header("Scatter Plot"),
                ui.output_ui("out_corr_plot_widget"),
            ),
            class_="fade-in-entry",
        )

    @render.data_frame
    def out_corr_table():
        """
        Create a formatted table of the most relevant pairwise correlation statistics for the current result.
        """
        result = corr_result.get()
        if result is None:
            return None

        # Create formatted table
        stats = result["stats"]
        # Helper to get coefficient safely
        coef_key = (
            "Coefficient (r/rho/tau)"
            if "Coefficient (r/rho/tau)" in stats
            else "Coefficient (r)"
        )
        coef_val = stats.get(coef_key)
        coef_display = (
            f"{coef_val:.4f}"
            if isinstance(coef_val, (int, float)) and not pd.isna(coef_val)
            else "N/A"
        )

        display_data = {
            "Metric": [
                "Method",
                "Correlation Coefficient",
                "95% CI Lower",
                "95% CI Upper",
                "R-squared (R²)",
                "P-value",
                "Sample Size (N)",
                "Interpretation",
            ],
            "Value": [
                stats["Method"],
                coef_display,
                f"{stats.get('95% CI Lower', float('nan')):.4f}",
                f"{stats.get('95% CI Upper', float('nan')):.4f}",
                f"{stats.get('R-squared (R²)', float('nan')):.4f}",
                format_p_value(stats.get("P-value", float("nan")), use_style=False),
                str(stats.get("N", "N/A")),
                stats.get("Interpretation", "N/A"),
            ],
        }

        df_display = pd.DataFrame(display_data)
        return render.DataGrid(df_display, width="100%")

    @render.ui
    def out_corr_plot_widget():
        """Render the correlation scatter plot as an HTML UI element."""
        result = corr_result.get()
        if result is None or result["figure"] is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            result["figure"],
            div_id="plot_corr_scatter",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.download(
        filename=lambda: (
            (
                lambda r: f"correlation_{_safe_filename_part(r['var1'])}_{_safe_filename_part(r['var2'])}.html"
            )(corr_result.get())
            if corr_result.get() is not None
            else "correlation_report.html"
        ),
    )
    def btn_dl_corr():
        """Generate and download correlation report."""

        def _build():
            result = corr_result.get()
            if not result or "error" in result:
                return None

            stats = result["stats"]

            # Build report elements
            elements = [
                {"type": "text", "data": f"Data Source: {result['data_label']}"},
                {"type": "text", "data": f"Method: {result['method'].title()}"},
                {
                    "type": "text",
                    "data": f"Variables: {result['var1']} vs {result['var2']}",
                },
                {"type": "text", "header": "Statistical Results", "data": ""},
            ]

            # Add statistics
            coef_key = (
                "Coefficient (r/rho/tau)"
                if "Coefficient (r/rho/tau)" in stats
                else "Coefficient (r)"
            )

            for key in [
                "Method",
                coef_key,
                "95% CI Lower",
                "95% CI Upper",
                "R-squared (R\u00b2)",
                "P-value",
                "N",
            ]:
                val = stats.get(key, "N/A")
                if key == "P-value" and isinstance(val, (int, float, np.number)):
                    elements.append(
                        {
                            "type": "html",
                            "data": f"<strong>P-value:</strong> {format_p_value(val, use_style=True)}",
                            "safe_html": True,
                        }
                    )
                elif isinstance(val, (int, float, np.number)):
                    elements.append(
                        {
                            "type": "text",
                            "data": (
                                f"{key if key != coef_key else 'Correlation Coefficient'}: {val:.4f}"
                                if isinstance(val, float)
                                else f"{key}: {val}"
                            ),
                        }
                    )
                else:
                    elements.append({"type": "text", "data": f"{key}: {val}"})

            # Add interpretation
            interp = stats.get("Interpretation", "N/A")
            r2 = stats.get("R-squared (R\u00b2)", float("nan"))
            if isinstance(r2, (int, float)) and not pd.isna(r2):
                r2_text = f"R\u00b2 = {r2:.3f} means {r2 * 100:.1f}% of variance is explained."
            else:
                r2_text = "R\u00b2 is not available."
            elements.append(
                {
                    "type": "interpretation",
                    "data": f"{interp}. {r2_text}",
                }
            )

            elements.append({"type": "text", "data": stats.get("Sample Note", "")})

            # Add plot
            elements.append(
                {"type": "plot", "header": "Scatter Plot", "data": result["figure"]}
            )

            # Missing Data Report
            if "missing_data_info" in stats:
                elements.append(
                    {
                        "type": "html",
                        "data": create_missing_data_report_html(
                            stats["missing_data_info"], var_meta.get() or {}
                        ),
                        "safe_html": True,
                    }
                )

            return correlation.generate_report(
                title=f"Correlation Analysis: {result['var1']} vs {result['var2']}",
                elements=elements,
            )

        yield safe_report_generation(_build, label="Correlation Report")

    # --- Download Status Badges ---
    def _is_download_ready(result: dict[str, Any] | None) -> bool:
        """Return True only if result is non-None and not an error."""
        return bool(result) and "error" not in result

    @render.ui
    def dl_status_corr():
        return create_download_status_badge(_is_download_ready(corr_result.get()))

    @render.ui
    def dl_status_matrix():
        return create_download_status_badge(_is_download_ready(matrix_result.get()))

    # ==================== CORRELATION MATRIX / HEATMAP ====================

    @reactive.Effect
    @reactive.event(input.btn_run_matrix)
    def _run_matrix() -> None:
        """Run correlation matrix and heatmap generation."""
        data = current_df()

        if data is None:
            ui.notification_show("No data available", type="error")
            return

        cols = input.matrix_vars()
        method = input.matrix_method()

        if not cols or len(cols) < 2:
            ui.notification_show("Please select at least 2 variables", type="warning")
            return

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message="Generating Heatmap...",
                detail=f"Processing {len(cols)} variables",
            )
            corr_matrix, fig, summary = correlation.compute_correlation_matrix(
                data, list(cols), method=method, var_meta=var_meta.get() or {}
            )

        if corr_matrix is not None:
            # Determine data label
            if is_matched.get() and input.radio_corr_source() == "matched":
                data_label = f"✅ Matched Data ({len(data)} rows)"
            else:
                data_label = f"📊 Original Data ({len(data)} rows)"

            matrix_result.set(
                {
                    "matrix": corr_matrix,
                    "figure": fig,
                    "method": method,
                    "summary": summary,
                    "data_label": data_label,
                    "strategy": summary.get("missing_data_info", {}).get(
                        "strategy", "pairwise-complete"
                    ),
                }
            )
            ui.notification_show("✅ Heatmap generated!", type="default")
        else:
            matrix_result.set(None)
            ui.notification_show("Failed to generate matrix", type="error")

    @render.ui
    def out_matrix_result():
        """Render the matrix/heatmap results card for the current analysis."""
        result = matrix_result.get()
        if result is None:
            return ui.markdown(
                "*Results will appear here after clicking '🎨 Generate Heatmap'*"
            )

        summary = result["summary"]

        # Format summary statistics — defensive .get() access
        strongest_pos = _html.escape(str(summary.get("strongest_positive", "N/A")))
        strongest_neg = _html.escape(str(summary.get("strongest_negative", "N/A")))
        n_vars = summary.get("n_variables", "N/A")
        n_corrs = summary.get("n_correlations", "N/A")
        mean_corr = summary.get("mean_correlation", "N/A")
        n_sig = summary.get("n_significant", "N/A")
        pct_sig = summary.get("pct_significant", "N/A")
        mean_corr_str = (
            f"{mean_corr:.3f}"
            if isinstance(mean_corr, (int, float))
            else str(mean_corr)
        )
        pct_sig_str = (
            f"{pct_sig:.1f}" if isinstance(pct_sig, (int, float)) else str(pct_sig)
        )
        summary_html = f"""
        <div class='info-callout'>
            <h4 style='margin-top: 0;'>📊 Matrix Summary</h4>
            <p><strong>Variables:</strong> {_html.escape(str(n_vars))}</p>
            <p><strong>Correlations Computed:</strong> {_html.escape(str(n_corrs))} (unique pairs)</p>
            <p><strong>Mean |Correlation|:</strong> {_html.escape(str(mean_corr_str))}</p>
            <p><strong>Strongest Positive:</strong> {strongest_pos}</p>
            <p><strong>Strongest Negative:</strong> {strongest_neg}</p>
            <p><strong>Significant Correlations (p<0.05):</strong> {_html.escape(str(n_sig))} ({_html.escape(str(pct_sig_str))}%)</p>
        </div>
        """

        return ui.div(
            ui.card(
                ui.card_header("Matrix Results"),
                ui.markdown(f"**Data Source:** {result['data_label']}"),
                ui.markdown(f"**Method:** {result['method'].title()}"),
                ui.markdown(f"**Missing Data Strategy:** {result['strategy'].title()}"),
                ui.HTML(summary_html),
                # Missing Data Report
                ui.HTML(
                    create_missing_data_report_html(
                        summary.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
                ui.card_header("Heatmap"),
                ui.output_ui("out_heatmap_widget"),
                ui.card_header("Correlation Table"),
                ui.markdown(
                    "*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*"
                ),
                ui.output_data_frame("out_matrix_table"),
            ),
            class_="fade-in-entry",
        )

    @render.ui
    def out_heatmap_widget():
        """Render the correlation heatmap plot or a waiting placeholder as a Shiny UI element."""
        result = matrix_result.get()
        if result is None or result["figure"] is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                class_="muted-placeholder",
            )
        html_str = plotly_figure_to_html(
            result["figure"],
            div_id="plot_corr_heatmap",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_matrix_table():
        """Render the correlation matrix as a DataGrid suitable for display."""
        result = matrix_result.get()
        if result is None:
            return None
        # Add index as a column for better display in DataGrid
        df_display = (
            result["matrix"].reset_index().rename(columns={"index": "Variable"})
        )
        return render.DataGrid(df_display, width="100%")

    @render.download(
        filename=lambda: (
            (lambda r: f"correlation_matrix_{_safe_filename_part(r['method'])}.html")(
                matrix_result.get()
            )
            if matrix_result.get() is not None
            else "correlation_matrix.html"
        ),
    )
    def btn_dl_matrix():
        """Generate and download matrix report."""

        def _build():
            result = matrix_result.get()
            if not result or "error" in result:
                return None

            summary = result["summary"]

            # Build report elements — defensive .get() access
            n_vars = summary.get("n_variables", "N/A")
            elements = [
                {"type": "text", "data": f"Data Source: {result['data_label']}"},
                {"type": "text", "data": f"Method: {result['method'].title()}"},
                {
                    "type": "text",
                    "data": f"Number of Variables: {n_vars}",
                },
            ]

            # Add summary statistics — safe access with .get()
            n_corrs = summary.get("n_correlations", "N/A")
            mean_corr = summary.get("mean_correlation", "N/A")
            max_corr = summary.get("max_correlation", "N/A")
            min_corr = summary.get("min_correlation", "N/A")
            n_sig = summary.get("n_significant", "N/A")
            pct_sig = summary.get("pct_significant", "N/A")
            strongest_pos = summary.get("strongest_positive", "N/A")
            strongest_neg = summary.get("strongest_negative", "N/A")

            max_corr_str = (
                f"{max_corr:.3f}"
                if isinstance(max_corr, (int, float))
                else _html.escape(str(max_corr))
            )
            min_corr_str = (
                f"{min_corr:.3f}"
                if isinstance(min_corr, (int, float))
                else _html.escape(str(min_corr))
            )
            pct_sig_str = (
                f"{pct_sig:.1f}"
                if isinstance(pct_sig, (int, float))
                else _html.escape(str(pct_sig))
            )
            mean_corr_str = (
                f"{mean_corr:.3f}"
                if isinstance(mean_corr, (int, float))
                else _html.escape(str(mean_corr))
            )

            summary_text = f"""
            <h3>Matrix Summary Statistics</h3>
            <p><strong>Correlations Computed:</strong> {_html.escape(str(n_corrs))} unique pairs</p>
            <p><strong>Mean |Correlation|:</strong> {mean_corr_str}</p>
            <p><strong>Maximum |Correlation|:</strong> {max_corr_str}</p>
            <p><strong>Minimum |Correlation|:</strong> {min_corr_str}</p>
            <p><strong>Significant Correlations (p<0.05):</strong> {_html.escape(str(n_sig))} out of {_html.escape(str(n_corrs))} ({pct_sig_str}%)</p>
            <p><strong>Strongest Positive:</strong> {_html.escape(str(strongest_pos))}</p>
            <p><strong>Strongest Negative:</strong> {_html.escape(str(strongest_neg))}</p>
            """

            elements.append(
                {"type": "summary", "data": summary_text, "safe_html": True}
            )

            # Add heatmap
            elements.append(
                {
                    "type": "plot",
                    "header": "Correlation Heatmap",
                    "data": result["figure"],
                }
            )

            # Add matrix table
            elements.append(
                {
                    "type": "table",
                    "header": "Correlation Matrix",
                    "data": result["matrix"],
                }
            )

            elements.append(
                {
                    "type": "text",
                    "data": "Significance levels: * p<0.05, ** p<0.01, *** p<0.001",
                }
            )

            # Missing Data Report
            if "missing_data_info" in summary:
                elements.append(
                    {
                        "type": "html",
                        "data": create_missing_data_report_html(
                            summary["missing_data_info"], var_meta.get() or {}
                        ),
                        "safe_html": True,
                    }
                )

            return correlation.generate_report(
                title=f"Correlation Matrix Analysis ({result['method'].title()})",
                elements=elements,
            )

        yield safe_report_generation(_build, label="Correlation Matrix Report")

    # --- PDF Download Handlers ---
    @render.download(
        filename=lambda: (
            (
                lambda r: f"correlation_{_safe_filename_part(r['var1'])}_{_safe_filename_part(r['var2'])}.pdf"
            )(corr_result.get())
            if corr_result.get() is not None
            else "correlation_report.pdf"
        ),
    )
    def btn_dl_corr_pdf():
        """Generate and download correlation report as PDF."""

        def _build():
            result = corr_result.get()
            if not result or "error" in result:
                return None

            stats = result["stats"]
            elements = [
                {"type": "text", "data": f"Data Source: {result['data_label']}"},
                {"type": "text", "data": f"Method: {result['method'].title()}"},
                {
                    "type": "text",
                    "data": f"Variables: {result['var1']} vs {result['var2']}",
                },
                {"type": "text", "header": "Statistical Results", "data": ""},
            ]

            coef_key = (
                "Coefficient (r/rho/tau)"
                if "Coefficient (r/rho/tau)" in stats
                else "Coefficient (r)"
            )

            for key in [
                "Method",
                coef_key,
                "95% CI Lower",
                "95% CI Upper",
                "R-squared (R\u00b2)",
                "P-value",
                "N",
            ]:
                val = stats.get(key, "N/A")
                if key == "P-value" and isinstance(val, (int, float, np.number)):
                    elements.append(
                        {
                            "type": "html",
                            "data": f"<strong>P-value:</strong> {format_p_value(val, use_style=True)}",
                            "safe_html": True,
                        }
                    )
                elif isinstance(val, (int, float, np.number)):
                    elements.append(
                        {
                            "type": "text",
                            "data": (
                                f"{key if key != coef_key else 'Correlation Coefficient'}: {val:.4f}"
                                if isinstance(val, float)
                                else f"{key}: {val}"
                            ),
                        }
                    )
                else:
                    elements.append({"type": "text", "data": f"{key}: {val}"})

            interp = stats.get("Interpretation", "N/A")
            r2 = stats.get("R-squared (R\u00b2)", float("nan"))
            if isinstance(r2, (int, float)) and not pd.isna(r2):
                r2_text = f"R\u00b2 = {r2:.3f} means {r2 * 100:.1f}% of variance is explained."
            else:
                r2_text = "R\u00b2 is not available."
            elements.append(
                {
                    "type": "interpretation",
                    "data": f"{interp}. {r2_text}",
                }
            )

            elements.append({"type": "text", "data": stats.get("Sample Note", "")})
            elements.append(
                {"type": "plot", "header": "Scatter Plot", "data": result["figure"]}
            )

            if "missing_data_info" in stats:
                elements.append(
                    {
                        "type": "html",
                        "data": create_missing_data_report_html(
                            stats["missing_data_info"], var_meta.get() or {}
                        ),
                        "safe_html": True,
                    }
                )

            return correlation.generate_report(
                title=f"Correlation Analysis: {result['var1']} vs {result['var2']}",
                elements=elements,
            )

        yield safe_pdf_report_generation(_build, label="Correlation Report")

    @render.download(
        filename=lambda: (
            (lambda r: f"correlation_matrix_{_safe_filename_part(r['method'])}.pdf")(
                matrix_result.get()
            )
            if matrix_result.get() is not None
            else "correlation_matrix.pdf"
        ),
    )
    def btn_dl_matrix_pdf():
        """Generate and download matrix report as PDF."""

        def _build():
            result = matrix_result.get()
            if not result or "error" in result:
                return None

            summary = result["summary"]
            n_vars = summary.get("n_variables", "N/A")
            elements = [
                {"type": "text", "data": f"Data Source: {result['data_label']}"},
                {"type": "text", "data": f"Method: {result['method'].title()}"},
                {"type": "text", "data": f"Number of Variables: {n_vars}"},
            ]

            n_corrs = summary.get("n_correlations", "N/A")
            mean_corr = summary.get("mean_correlation", "N/A")
            max_corr = summary.get("max_correlation", "N/A")
            min_corr = summary.get("min_correlation", "N/A")
            n_sig = summary.get("n_significant", "N/A")
            pct_sig = summary.get("pct_significant", "N/A")
            strongest_pos = summary.get("strongest_positive", "N/A")
            strongest_neg = summary.get("strongest_negative", "N/A")

            max_corr_str = (
                f"{max_corr:.3f}"
                if isinstance(max_corr, (int, float))
                else _html.escape(str(max_corr))
            )
            min_corr_str = (
                f"{min_corr:.3f}"
                if isinstance(min_corr, (int, float))
                else _html.escape(str(min_corr))
            )
            pct_sig_str = (
                f"{pct_sig:.1f}"
                if isinstance(pct_sig, (int, float))
                else _html.escape(str(pct_sig))
            )
            mean_corr_str = (
                f"{mean_corr:.3f}"
                if isinstance(mean_corr, (int, float))
                else _html.escape(str(mean_corr))
            )

            summary_text = f"""
            <h3>Matrix Summary Statistics</h3>
            <p><strong>Correlations Computed:</strong> {_html.escape(str(n_corrs))} unique pairs</p>
            <p><strong>Mean |Correlation|:</strong> {mean_corr_str}</p>
            <p><strong>Maximum |Correlation|:</strong> {max_corr_str}</p>
            <p><strong>Minimum |Correlation|:</strong> {min_corr_str}</p>
            <p><strong>Significant Correlations (p<0.05):</strong> {_html.escape(str(n_sig))} out of {_html.escape(str(n_corrs))} ({pct_sig_str}%)</p>
            <p><strong>Strongest Positive:</strong> {_html.escape(str(strongest_pos))}</p>
            <p><strong>Strongest Negative:</strong> {_html.escape(str(strongest_neg))}</p>
            """

            elements.append(
                {"type": "summary", "data": summary_text, "safe_html": True}
            )
            elements.append(
                {
                    "type": "plot",
                    "header": "Correlation Heatmap",
                    "data": result["figure"],
                }
            )
            elements.append(
                {
                    "type": "table",
                    "header": "Correlation Matrix",
                    "data": result["matrix"],
                }
            )
            elements.append(
                {
                    "type": "text",
                    "data": "Significance levels: * p<0.05, ** p<0.01, *** p<0.001",
                }
            )

            if "missing_data_info" in summary:
                elements.append(
                    {
                        "type": "html",
                        "data": create_missing_data_report_html(
                            summary["missing_data_info"], var_meta.get() or {}
                        ),
                        "safe_html": True,
                    }
                )

            return correlation.generate_report(
                title=f"Correlation Matrix Analysis ({result['method'].title()})",
                elements=elements,
            )

        yield safe_pdf_report_generation(_build, label="Correlation Matrix Report")
