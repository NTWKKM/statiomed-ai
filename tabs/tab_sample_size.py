from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
from shiny import module, reactive, render, ui

from tabs._common import get_color_palette
from utils import sample_size_lib
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.ui_helpers import (
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
)

COLORS = get_color_palette()


@module.ui
def sample_size_ui() -> ui.TagChild:
    return ui.div(
        ui.h4("🔢 Sample Size & Power Calculator"),
        ui.br(),
        ui.navset_card_pill(
            # --- 1. MEANS (T-Test) ---
            ui.nav_panel(
                "📊 Means (T-test)",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Means (T-test) Setup"),
                        ui.layout_columns(
                            create_input_group(
                                "Group 1",
                                ui.input_numeric("ss_mean1", "Mean Group 1:", value=0),
                                ui.input_numeric("ss_sd1", "SD Group 1:", value=10),
                                type="required",
                            ),
                            create_input_group(
                                "Group 2",
                                ui.input_numeric("ss_mean2", "Mean Group 2:", value=5),
                                ui.input_numeric("ss_sd2", "SD Group 2:", value=10),
                                type="required",
                            ),
                            col_widths=[6, 6],
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                "⚙️ Statistical Parameters (Power, Alpha)",
                                ui.layout_columns(
                                    ui.input_slider(
                                        "ss_means_power",
                                        "Power (1-β):",
                                        min=0.5,
                                        max=0.99,
                                        value=0.8,
                                        step=0.01,
                                    ),
                                    ui.input_slider(
                                        "ss_means_alpha",
                                        "Alpha (Sig. Level):",
                                        min=0.001,
                                        max=0.2,
                                        value=0.05,
                                        step=0.001,
                                    ),
                                    col_widths=[6, 6],
                                ),
                                ui.input_numeric(
                                    "ss_means_ratio",
                                    "Allocation Ratio (N2/N1):",
                                    value=1,
                                    min=0.1,
                                    step=0.1,
                                ),
                            ),
                            open=False,
                        ),
                        ui.output_ui("out_means_validation"),
                        ui.hr(),
                        ui.input_action_button(
                            "btn_calc_means",
                            "🚀 Calculate N",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "Results",
                        ui.output_ui("out_means_result"),
                        ui.output_ui("out_means_plot"),
                        ui.output_ui("out_means_methods"),
                    ),
                    col_widths=[5, 7],
                ),
            ),
            # --- 2. PROPORTIONS (Chi-Sq) ---
            ui.nav_panel(
                "🎲 Proportions",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Proportions Setup"),
                        ui.layout_columns(
                            create_input_group(
                                "Expected Proportions",
                                ui.input_numeric(
                                    "ss_p1",
                                    "Proportion 1 (0-1):",
                                    value=0.1,
                                    min=0,
                                    max=1,
                                    step=0.01,
                                ),
                                ui.input_numeric(
                                    "ss_p2",
                                    "Proportion 2 (0-1):",
                                    value=0.2,
                                    min=0,
                                    max=1,
                                    step=0.01,
                                ),
                                type="required",
                            ),
                            col_widths=[12],
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                "⚙️ Statistical Parameters",
                                ui.layout_columns(
                                    ui.input_slider(
                                        "ss_props_power",
                                        "Power (1-β):",
                                        min=0.5,
                                        max=0.99,
                                        value=0.8,
                                        step=0.01,
                                    ),
                                    ui.input_slider(
                                        "ss_props_alpha",
                                        "Alpha (Sig. Level):",
                                        min=0.001,
                                        max=0.2,
                                        value=0.05,
                                        step=0.001,
                                    ),
                                    col_widths=[6, 6],
                                ),
                                ui.input_numeric(
                                    "ss_props_ratio",
                                    "Allocation Ratio (N2/N1):",
                                    value=1,
                                    min=0.1,
                                    step=0.1,
                                ),
                            ),
                            open=False,
                        ),
                        ui.output_ui("out_props_validation"),
                        ui.hr(),
                        ui.input_action_button(
                            "btn_calc_props",
                            "🚀 Calculate N",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "Results",
                        ui.output_ui("out_props_result"),
                        ui.output_ui("out_props_plot"),
                        ui.output_ui("out_props_methods"),
                    ),
                    col_widths=[5, 7],
                ),
            ),
            # --- 3. SURVIVAL (Log-Rank) ---
            ui.nav_panel(
                "⏳ Survival (Log-Rank)",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Survival Analysis Setup"),
                        create_input_group(
                            "Input Mode",
                            ui.input_radio_buttons(
                                "ss_surv_mode",
                                "Input Mode:",
                                {
                                    "hr": "Hazard Ratio",
                                    "median": "Median Survival Time",
                                },
                            ),
                            type="optional",
                        ),
                        create_input_group(
                            "Parameters",
                            ui.panel_conditional(
                                "input.ss_surv_mode == 'hr'",
                                ui.input_numeric(
                                    "ss_surv_hr",
                                    "Hazard Ratio (HR):",
                                    value=2.0,
                                    min=0.001,
                                ),
                            ),
                            ui.panel_conditional(
                                "input.ss_surv_mode == 'median'",
                                ui.input_numeric(
                                    "ss_surv_m1", "Median Survival 1:", value=12
                                ),
                                ui.input_numeric(
                                    "ss_surv_m2", "Median Survival 2:", value=24
                                ),
                            ),
                            ui.input_slider(
                                "ss_surv_power",
                                "Power (1-β):",
                                min=0.5,
                                max=0.99,
                                value=0.8,
                                step=0.01,
                            ),
                            ui.input_slider(
                                "ss_surv_alpha",
                                "Alpha (Sig. Level):",
                                min=0.001,
                                max=0.2,
                                value=0.05,
                                step=0.001,
                            ),
                            ui.input_numeric(
                                "ss_surv_ratio",
                                "Ratio (N2/N1):",
                                value=1,
                                min=0.1,
                                step=0.1,
                            ),
                            type="required",
                        ),
                        ui.output_ui("out_surv_validation"),
                        ui.hr(),
                        ui.input_action_button(
                            "btn_calc_surv",
                            "🚀 Calculate Events",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "Results",
                        ui.output_ui("out_surv_result"),
                        ui.output_ui("out_surv_plot"),
                        ui.output_ui("out_surv_methods"),
                        ui.markdown(
                            "*Note: This calculates required number of EVENTS, not total subjects.*"
                        ),
                    ),
                    col_widths=[5, 7],
                ),
            ),
            # --- 4. CORRELATION ---
            ui.nav_panel(
                "📈 Correlation",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Correlation Setup"),
                        create_input_group(
                            "Parameters",
                            ui.input_numeric(
                                "ss_corr_r",
                                "Expected Correlation (r):",
                                value=0.3,
                                min=-1,
                                max=1,
                                step=0.05,
                            ),
                            ui.input_slider(
                                "ss_corr_power",
                                "Power (1-β):",
                                min=0.5,
                                max=0.99,
                                value=0.8,
                                step=0.01,
                            ),
                            ui.input_slider(
                                "ss_corr_alpha",
                                "Alpha (Sig. Level):",
                                min=0.001,
                                max=0.2,
                                value=0.05,
                                step=0.001,
                            ),
                            type="required",
                        ),
                        ui.output_ui("out_corr_validation"),
                        ui.hr(),
                        ui.input_action_button(
                            "btn_calc_corr",
                            "🚀 Calculate N",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "Results",
                        ui.output_ui("out_corr_result"),
                        ui.output_ui("out_corr_plot"),
                        ui.output_ui("out_corr_methods"),
                    ),
                    col_widths=[5, 7],
                ),
            ),
        ),
    )


@module.server
def sample_size_server(input: Any, output: Any, session: Any) -> None:
    # --- RESULT HOLDERS ---
    res_means = reactive.Value(None)
    plot_means = reactive.Value(None)
    res_props = reactive.Value(None)
    plot_props = reactive.Value(None)
    res_surv = reactive.Value(None)
    plot_surv = reactive.Value(None)
    res_corr = reactive.Value(None)
    plot_corr = reactive.Value(None)

    # Methods Text Holders
    methods_means = reactive.Value(None)
    methods_props = reactive.Value(None)
    methods_surv = reactive.Value(None)
    methods_corr = reactive.Value(None)

    # Running States
    means_is_running = reactive.Value(False)
    props_is_running = reactive.Value(False)
    surv_is_running = reactive.Value(False)
    corr_is_running = reactive.Value(False)

    # --- CALCULATORS ---

    @reactive.Effect
    @reactive.event(input.btn_calc_means)
    def _calc_means():
        means_is_running.set(True)
        res_means.set(None)
        plot_means.set(None)
        methods_means.set(None)

        try:
            res = sample_size_lib.calculate_sample_size_means(
                power=input.ss_means_power(),
                ratio=input.ss_means_ratio(),
                mean1=input.ss_mean1(),
                mean2=input.ss_mean2(),
                sd1=input.ss_sd1(),
                sd2=input.ss_sd2(),
                alpha=input.ss_means_alpha(),
            )
            res_means.set(res)

            # Power Curve
            df_plot = sample_size_lib.calculate_power_curve(
                target_n=int(res["total"]),
                ratio=input.ss_means_ratio(),
                calc_func=sample_size_lib.calculate_power_means,
                mean1=input.ss_mean1(),
                mean2=input.ss_mean2(),
                sd1=input.ss_sd1(),
                sd2=input.ss_sd2(),
                alpha=input.ss_means_alpha(),
            )
            plot_means.set(df_plot)

            # Methods Text
            txt = sample_size_lib.generate_methods_text(
                "Independent Means (T-test)",
                {
                    "mean1": input.ss_mean1(),
                    "mean2": input.ss_mean2(),
                    "sd1": input.ss_sd1(),
                    "sd2": input.ss_sd2(),
                    "power": input.ss_means_power(),
                    "alpha": input.ss_means_alpha(),
                    "total": res["total"],
                    "n1": res["n1"],
                    "n2": res["n2"],
                },
            )
            methods_means.set(txt)

        except Exception as e:
            res_means.set({"error": f"Error: {e!s}"})
            plot_means.set(None)
            methods_means.set(None)
        finally:
            means_is_running.set(False)

    @reactive.Effect
    @reactive.event(input.btn_calc_props)
    def _calc_props():
        props_is_running.set(True)
        res_props.set(None)
        plot_props.set(None)
        methods_props.set(None)

        try:
            res = sample_size_lib.calculate_sample_size_proportions(
                power=input.ss_props_power(),
                ratio=input.ss_props_ratio(),
                p1=input.ss_p1(),
                p2=input.ss_p2(),
                alpha=input.ss_props_alpha(),
            )
            res_props.set(res)

            # Power Curve
            df_plot = sample_size_lib.calculate_power_curve(
                target_n=int(res["total"]),
                ratio=input.ss_props_ratio(),
                calc_func=sample_size_lib.calculate_power_proportions,
                p1=input.ss_p1(),
                p2=input.ss_p2(),
                alpha=input.ss_props_alpha(),
            )
            plot_props.set(df_plot)

            # Methods Text
            txt = sample_size_lib.generate_methods_text(
                "Independent Proportions",
                {
                    "p1": input.ss_p1(),
                    "p2": input.ss_p2(),
                    "power": input.ss_props_power(),
                    "alpha": input.ss_props_alpha(),
                    "total": res["total"],
                    "n1": res["n1"],
                    "n2": res["n2"],
                },
            )
            methods_props.set(txt)
        except Exception as e:
            res_props.set({"error": f"Error: {e!s}"})
            plot_props.set(None)
            methods_props.set(None)
        finally:
            props_is_running.set(False)

    @reactive.Effect
    @reactive.event(input.btn_calc_surv)
    def _calc_surv():
        surv_is_running.set(True)
        res_surv.set(None)
        plot_surv.set(None)
        methods_surv.set(None)

        try:
            h0 = (
                input.ss_surv_hr()
                if input.ss_surv_mode() == "hr"
                else input.ss_surv_m1()
            )
            h1 = 0 if input.ss_surv_mode() == "hr" else input.ss_surv_m2()

            res = sample_size_lib.calculate_sample_size_survival(
                power=input.ss_surv_power(),
                ratio=input.ss_surv_ratio(),
                h0=h0,
                h1=h1,
                alpha=input.ss_surv_alpha(),
                mode=input.ss_surv_mode(),
            )
            res_surv.set(res)

            # Power Curve (Events)
            df_plot = sample_size_lib.calculate_power_curve(
                target_n=int(res["total_events"]),
                ratio=input.ss_surv_ratio(),
                calc_func=sample_size_lib.calculate_power_survival,
                h0=h0,
                h1=h1,
                alpha=input.ss_surv_alpha(),
                mode=input.ss_surv_mode(),
            )
            plot_surv.set(df_plot)

            # Methods Text
            txt = sample_size_lib.generate_methods_text(
                "Survival (Log-Rank)",
                {
                    "hr": res["hr"],
                    "power": input.ss_surv_power(),
                    "alpha": input.ss_surv_alpha(),
                    "total_events": res["total_events"],
                },
            )
            methods_surv.set(txt)
        except Exception as e:
            res_surv.set({"error": f"Error: {e!s}"})
            plot_surv.set(None)
            methods_surv.set(None)
        finally:
            surv_is_running.set(False)

    @reactive.Effect
    @reactive.event(input.btn_calc_corr)
    def _calc_corr():
        corr_is_running.set(True)
        res_corr.set(None)
        plot_corr.set(None)
        methods_corr.set(None)

        try:
            res = sample_size_lib.calculate_sample_size_correlation(
                power=input.ss_corr_power(),
                r=input.ss_corr_r(),
                alpha=input.ss_corr_alpha(),
            )
            res_corr.set(res)

            # Power Curve
            df_plot = sample_size_lib.calculate_power_curve(
                target_n=int(res),
                ratio=1.0,  # Not used for corr, but required arg
                calc_func=sample_size_lib.calculate_power_correlation,
                r=input.ss_corr_r(),
                alpha=input.ss_corr_alpha(),
            )
            plot_corr.set(df_plot)

            # Methods Text
            txt = sample_size_lib.generate_methods_text(
                "Pearson Correlation",
                {
                    "r": input.ss_corr_r(),
                    "power": input.ss_corr_power(),
                    "alpha": input.ss_corr_alpha(),
                    "total": int(res),
                },
            )
            methods_corr.set(txt)
        except Exception as e:
            res_corr.set({"error": f"Error: {e!s}"})
            plot_corr.set(None)
            methods_corr.set(None)
        finally:
            corr_is_running.set(False)

    # --- RENDERERS ---

    def _render_n_result(res):
        if res is None:
            return None

        if "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.div(
                ui.h2(
                    f"Total N = {int(res['total'])}", class_="text-primary text-center"
                ),
                ui.hr(),
                ui.p(f"Group 1 (n1): {int(res['n1'])}"),
                ui.p(f"Group 2 (n2): {int(res['n2'])}"),
                style="background: #f8f9fa; padding: 20px; border-radius: 10px;",
            ),
            class_="fade-in-entry",
        )

    def _render_power_curve_plot(
        df: pd.DataFrame | None, x_label: str = "Sample Size (Total N)"
    ):
        if df is None or df.empty:
            return None

        fig = px.line(
            df,
            x="total_n",
            y="power",
            title="Power Curve",
            markers=True,
            labels={"total_n": x_label, "power": "Power (1-β)"},
            template="plotly_white",
        )
        # Add threshold line
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="red",
            annotation_text="80% Power",
            annotation_position="bottom right",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
        return ui.HTML(plotly_figure_to_html(fig))

    @render.ui
    def out_means_result():
        if means_is_running.get():
            return create_loading_state("Calculating sample size for Means...")

        res = res_means.get()
        if res is None:
            return create_placeholder_state(
                "Configure parameters and click 'Calculate N'.", icon="📊"
            )

        return _render_n_result(res)

    @render.ui
    def out_means_plot():
        return _render_power_curve_plot(plot_means.get())

    @render.ui
    def out_props_result():
        if props_is_running.get():
            return create_loading_state("Calculating sample size for Proportions...")

        res = res_props.get()
        if res is None:
            return create_placeholder_state(
                "Configure parameters and click 'Calculate N'.", icon="🎲"
            )

        return _render_n_result(res)

    @render.ui
    def out_props_plot():
        return _render_power_curve_plot(plot_props.get())

    @render.ui
    def out_surv_result():
        if surv_is_running.get():
            return create_loading_state("Calculating sample size for Survival...")

        res = res_surv.get()
        if res is None:
            return create_placeholder_state(
                "Configure parameters and click 'Calculate Events'.", icon="⏳"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.div(
                ui.h2(
                    f"Total Events = {int(res['total_events'])}",
                    class_="text-primary text-center",
                ),
                ui.p(
                    f"Hazard Ratio detected: {res['hr']:.2f}",
                    class_="text-center text-muted",
                ),
                style="background: #f8f9fa; padding: 20px; border-radius: 10px;",
            ),
            class_="fade-in-entry",
        )

    @render.ui
    def out_surv_plot():
        return _render_power_curve_plot(plot_surv.get(), x_label="Number of Events")

    @render.ui
    def out_corr_result():
        if corr_is_running.get():
            return create_loading_state("Calculating sample size for Correlation...")

        res = res_corr.get()
        if res is None:
            return create_placeholder_state(
                "Configure parameters and click 'Calculate N'.", icon="📈"
            )

        if isinstance(res, dict) and "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.div(
                ui.h2(f"Total N = {int(res)}", class_="text-primary text-center"),
                style="background: #f8f9fa; padding: 20px; border-radius: 10px;",
            ),
            class_="fade-in-entry",
        )

    @render.ui
    def out_corr_plot():
        return _render_power_curve_plot(plot_corr.get())

    def _render_methods_text(txt):
        if not txt:
            return None
        return ui.div(
            ui.div(
                ui.h5("📝 Methods Text (Copy-Ready)"),
                ui.pre(
                    txt,
                    style="background: #f8f9fa; padding: 10px; border-radius: 5px; white-space: pre-wrap;",
                ),
                ui.p(
                    "You can copy/paste this into your protocol or manuscript.",
                    style="font-size: 0.8em; color: gray;",
                ),
            ),
            class_="fade-in-entry",
        )

    @render.ui
    def out_means_methods():
        return _render_methods_text(methods_means.get())

    @render.ui
    def out_props_methods():
        return _render_methods_text(methods_props.get())

    @render.ui
    def out_surv_methods():
        return _render_methods_text(methods_surv.get())

    @render.ui
    def out_corr_methods():
        return _render_methods_text(methods_corr.get())

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_means_validation():
        sd1 = input.ss_sd1()
        sd2 = input.ss_sd2()
        ratio = input.ss_means_ratio()

        alerts = []
        if sd1 <= 0 or sd2 <= 0:
            alerts.append(
                create_error_alert(
                    "Standard Deviations must be positive (> 0).",
                    title="Invalid Parameter",
                )
            )

        if ratio <= 0:
            alerts.append(
                create_error_alert(
                    "Ratio (N2/N1) must be positive (> 0).", title="Invalid Parameter"
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_props_validation():
        p1 = input.ss_p1()
        p2 = input.ss_p2()
        alerts = []

        if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
            alerts.append(
                create_error_alert(
                    "Proportions must be between 0 and 1.", title="Invalid Parameter"
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_surv_validation():
        mode = input.ss_surv_mode()
        alerts = []

        if mode == "hr":
            hr = input.ss_surv_hr()
            if hr <= 0:
                alerts.append(
                    create_error_alert(
                        "Hazard Ratio must be positive (> 0).",
                        title="Invalid Parameter",
                    )
                )
        elif mode == "median":
            m1 = input.ss_surv_m1()
            m2 = input.ss_surv_m2()
            if m1 <= 0 or m2 <= 0:
                alerts.append(
                    create_error_alert(
                        "Median Survival Times must be positive (> 0).",
                        title="Invalid Parameter",
                    )
                )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_corr_validation():
        r = input.ss_corr_r()
        alerts = []

        if not (-1 <= r <= 1):
            alerts.append(
                create_error_alert(
                    "Correlation (r) must be between -1 and 1.",
                    title="Invalid Parameter",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None
