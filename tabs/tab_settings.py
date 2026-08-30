"""Settings Tab for Shiny Application

Provides interactive configuration management for statistical analysis parameters,
UI settings, logging configuration, and performance tuning.

Usage:
    from tabs import tab_settings

    # In app.py
    ui.nav_panel("⚡ Settings", tab_settings.settings_ui("settings"))

    # In server function
    tab_settings.settings_server("settings", CONFIG)
"""

from __future__ import annotations

from typing import Any

from shiny import reactive, render, ui
from shiny.session import get_current_session

from config import CONFIG
from logger import get_logger
from tabs import tab_advanced_stats
from tabs._common import (
    get_color_palette,
)
from utils.ui_helpers import (
    create_input_group,
    create_tooltip_label,
)

logger = get_logger(__name__)
COLORS = get_color_palette()


def settings_ui(id: str) -> ui.TagChild:
    """
    Construct the Settings tab UI containing six themed panels.
    """
    return ui.navset_tab(
        # ==========================================
        # 1. ANALYSIS SETTINGS TAB
        # ==========================================
        ui.nav_panel(
            "📊 Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Analysis Parameters"),
                    # Logistic Regression Section
                    create_input_group(
                        "Logistic Regression",
                        ui.input_select(
                            f"{id}_logit_method",
                            "Method",
                            choices={
                                "auto": "Auto",
                                "firth": "Firth",
                                "bfgs": "BFGS",
                                "default": "Default",
                            },
                            selected=CONFIG.get("analysis.logit_method"),
                            width="100%",
                        ),
                        ui.input_slider(
                            f"{id}_logit_screening_p",
                            create_tooltip_label(
                                "Screening P-value",
                                "P-value threshold for univariate variable selection (default: 0.20)",
                            ),
                            min=0.0,
                            max=1.0,
                            value=CONFIG.get("analysis.logit_screening_p"),
                            step=0.01,
                            width="100%",
                        ),
                        ui.input_numeric(
                            f"{id}_logit_max_iter",
                            create_tooltip_label(
                                "Max Iterations",
                                "Maximum number of iterations for the solver to converge (default: 1000)",
                            ),
                            value=CONFIG.get("analysis.logit_max_iter"),
                            min=10,
                            max=5000,
                            width="100%",
                        ),
                        ui.input_numeric(
                            f"{id}_logit_min_cases",
                            create_tooltip_label(
                                "Min Cases for Multivariate",
                                "Minimum number of outcome events required to attempt multivariate modeling",
                            ),
                            value=CONFIG.get("analysis.logit_min_cases"),
                            min=1,
                            max=100,
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # Survival Analysis Section
                    create_input_group(
                        "Survival Analysis",
                        ui.input_select(
                            f"{id}_survival_method",
                            create_tooltip_label(
                                "Survival Method",
                                "Statistical method for survival analysis (Kaplan-Meier is standard for visualization)",
                            ),
                            choices={
                                "kaplan-meier": "Kaplan-Meier",
                                "weibull": "Weibull",
                            },
                            selected=CONFIG.get("analysis.survival_method"),
                            width="100%",
                        ),
                        ui.input_select(
                            f"{id}_cox_method",
                            create_tooltip_label(
                                "Cox Method (Tie Handling)",
                                "Method for handling tied event times. Efron is generally more accurate than Breslow.",
                            ),
                            choices={"efron": "Efron", "breslow": "Breslow"},
                            selected=CONFIG.get("analysis.cox_method"),
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # Variable Detection Section
                    create_input_group(
                        "Variable Detection",
                        ui.input_numeric(
                            f"{id}_var_detect_threshold",
                            "Unique Value Threshold",
                            value=CONFIG.get("analysis.var_detect_threshold"),
                            min=1,
                            max=50,
                            width="100%",
                        ),
                        ui.input_slider(
                            f"{id}_var_detect_decimal_pct",
                            "Decimal % Threshold",
                            min=0.0,
                            max=1.0,
                            value=CONFIG.get("analysis.var_detect_decimal_pct"),
                            step=0.05,
                            width="100%",
                        ),
                        type="advanced",
                    ),
                    # Reporting Style Section
                    create_input_group(
                        "Reporting Style",
                        ui.input_select(
                            f"{id}_publication_style",
                            "Publication Style",
                            choices={
                                "nejm": "NEJM",
                                "jama": "JAMA",
                                "lancet": "The Lancet",
                                "bmj": "BMJ",
                            },
                            selected=CONFIG.get("analysis.publication_style"),
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # P-value Formatting Section
                    create_input_group(
                        "P-value Bounds (NEJM)",
                        ui.input_slider(
                            f"{id}_pvalue_bounds_lower",
                            "Lower Bound",
                            min=0.0,
                            max=0.1,
                            value=CONFIG.get("analysis.pvalue_bounds_lower"),
                            step=0.001,
                            width="100%",
                        ),
                        ui.input_slider(
                            f"{id}_pvalue_bounds_upper",
                            "Upper Bound",
                            min=0.9,
                            max=1.0,
                            value=CONFIG.get("analysis.pvalue_bounds_upper"),
                            step=0.001,
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # Missing Data Section
                    create_input_group(
                        "Missing Data",
                        ui.input_select(
                            f"{id}_missing_strategy",
                            "Missing Data Strategy",
                            choices={"complete-case": "Complete-case", "drop": "Drop"},
                            selected=CONFIG.get("analysis.missing_strategy"),
                            width="100%",
                        ),
                        ui.input_numeric(
                            f"{id}_missing_threshold_pct",
                            "Missing Flag Threshold (%)",
                            value=CONFIG.get("analysis.missing_threshold_pct"),
                            min=0,
                            max=100,
                            width="100%",
                        ),
                        type="optional",
                    ),
                    ui.br(),
                    ui.input_action_button(
                        f"{id}_btn_save_analysis",
                        "💾 Save Analysis Settings",
                        class_="btn-primary",
                        width="100%",
                    ),
                    width=350,
                    bg=COLORS["smoke_white"],
                ),
                ui.card(
                    ui.card_header("📋 Analysis Settings Guide"),
                    ui.markdown("""
                    ### Logistic Regression
                    - **Auto**: Best for beginners
                    - **Firth**: Stable for small samples or rare events
                    - **Screening P**: 0.05-0.20 typical range
                    
                    ### Survival Analysis
                    - **Kaplan-Meier**: Non-parametric (recommended)
                    - **Efron**: Better for tied event times
                    
                    ### P-value Format (NEJM Standard)
                    - Lower: 0.001 (display as "<0.001")
                    - Upper: 0.999 (display as ">0.999")
                    """),
                    full_screen=True,
                ),
            ),
        ),
        # ==========================================
        # 2. UI & DISPLAY TAB
        # ==========================================
        ui.nav_panel(
            "🎎 UI & Display",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Display Settings"),
                    # Page Setup
                    create_input_group(
                        "Page Setup",
                        ui.input_text(
                            f"{id}_page_title",
                            "Page Title",
                            value=CONFIG.get("ui.page_title"),
                            width="100%",
                        ),
                        ui.input_select(
                            f"{id}_theme",
                            "Theme",
                            choices={"light": "Light", "dark": "Dark", "auto": "Auto"},
                            selected=CONFIG.get("ui.theme"),
                            width="100%",
                        ),
                        ui.input_select(
                            f"{id}_layout",
                            "Layout",
                            choices={"wide": "Wide", "centered": "Centered"},
                            selected=CONFIG.get("ui.layout"),
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # Tables
                    create_input_group(
                        "Tables",
                        ui.input_numeric(
                            f"{id}_table_max_rows",
                            "Max Table Rows",
                            value=CONFIG.get("ui.table_max_rows"),
                            min=10,
                            max=10000,
                            width="100%",
                        ),
                        ui.input_checkbox(
                            f"{id}_table_pagination",
                            "Enable Pagination",
                            value=CONFIG.get("ui.table_pagination"),
                        ),
                        ui.input_numeric(
                            f"{id}_table_decimal_places",
                            "Decimal Places",
                            value=CONFIG.get("ui.table_decimal_places"),
                            min=0,
                            max=10,
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # Plots
                    create_input_group(
                        "Plots",
                        ui.input_numeric(
                            f"{id}_plot_width",
                            "Plot Width (inches)",
                            value=CONFIG.get("ui.plot_width"),
                            min=5,
                            max=50,
                            width="100%",
                        ),
                        ui.input_numeric(
                            f"{id}_plot_height",
                            "Plot Height (inches)",
                            value=CONFIG.get("ui.plot_height"),
                            min=3,
                            max=30,
                            width="100%",
                        ),
                        ui.input_numeric(
                            f"{id}_plot_dpi",
                            "Plot DPI",
                            value=CONFIG.get("ui.plot_dpi"),
                            min=50,
                            max=600,
                            width="100%",
                        ),
                        ui.input_text(
                            f"{id}_plot_style",
                            "Plot Style",
                            value=CONFIG.get("ui.plot_style"),
                            width="100%",
                        ),
                        type="optional",
                    ),
                    ui.br(),
                    ui.input_action_button(
                        f"{id}_btn_save_ui",
                        "💾 Save UI Settings",
                        class_="btn-primary",
                        width="100%",
                    ),
                    width=350,
                    bg=COLORS["smoke_white"],
                ),
                ui.card(
                    ui.card_header("🎏 UI Settings Guide"),
                    ui.markdown("""
                    ### Recommended Settings
                    - **Plot Width**: 10-14 inches
                    - **Plot Height**: 5-8 inches
                    - **Plot DPI**: 100-300 (higher = sharper)
                    - **Theme**: Auto follows system preference
                    - **Decimal Places**: 3 for most stats
                    """),
                    full_screen=True,
                ),
            ),
        ),
        # ==========================================
        # 3. LOGGING TAB
        # ==========================================
        ui.nav_panel(
            "📝 Logging",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Logging Configuration"),
                    # Global Settings
                    create_input_group(
                        "Global",
                        ui.input_checkbox(
                            f"{id}_logging_enabled",
                            "Enable Logging",
                            value=CONFIG.get("logging.enabled"),
                        ),
                        ui.input_select(
                            f"{id}_logging_level",
                            "Log Level",
                            choices={
                                "DEBUG": "DEBUG",
                                "INFO": "INFO",
                                "WARNING": "WARNING",
                                "ERROR": "ERROR",
                                "CRITICAL": "CRITICAL",
                            },
                            selected=CONFIG.get("logging.level"),
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # File Logging
                    create_input_group(
                        "File Logging",
                        ui.input_checkbox(
                            f"{id}_file_enabled",
                            "Enable File Logging",
                            value=CONFIG.get("logging.file_enabled"),
                        ),
                        ui.input_text(
                            f"{id}_log_dir",
                            "Log Directory",
                            value=CONFIG.get("logging.log_dir"),
                            width="100%",
                        ),
                        ui.input_text(
                            f"{id}_log_file",
                            "Log Filename",
                            value=CONFIG.get("logging.log_file"),
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # Console Logging
                    create_input_group(
                        "Console",
                        ui.input_checkbox(
                            f"{id}_console_enabled",
                            "Enable Console Logging",
                            value=CONFIG.get("logging.console_enabled"),
                        ),
                        ui.input_select(
                            f"{id}_console_level",
                            "Console Level",
                            choices={
                                "DEBUG": "DEBUG",
                                "INFO": "INFO",
                                "WARNING": "WARNING",
                                "ERROR": "ERROR",
                                "CRITICAL": "CRITICAL",
                            },
                            selected=CONFIG.get("logging.console_level"),
                            width="100%",
                        ),
                        type="optional",
                    ),
                    # Event Logging
                    create_input_group(
                        "Log Events",
                        ui.input_checkbox(
                            f"{id}_log_file_ops",
                            "File Operations",
                            value=CONFIG.get("logging.log_file_operations"),
                        ),
                        ui.input_checkbox(
                            f"{id}_log_data_ops",
                            "Data Operations",
                            value=CONFIG.get("logging.log_data_operations"),
                        ),
                        ui.input_checkbox(
                            f"{id}_log_analysis_ops",
                            "Analysis Operations",
                            value=CONFIG.get("logging.log_analysis_operations"),
                        ),
                        ui.input_checkbox(
                            f"{id}_log_performance",
                            "Performance Timing",
                            value=CONFIG.get("logging.log_performance"),
                        ),
                        type="advanced",
                    ),
                    ui.br(),
                    ui.input_action_button(
                        f"{id}_btn_save_logging",
                        "💾 Save Logging Settings",
                        class_="btn-primary",
                        width="100%",
                    ),
                    width=350,
                    bg=COLORS["smoke_white"],
                ),
                ui.card(
                    ui.card_header("📊 Logging Status"),
                    ui.output_text(f"{id}_txt_logging_status"),
                    full_screen=True,
                ),
            ),
        ),
        # ==========================================
        # 4. PERFORMANCE TAB
        # ==========================================
        ui.nav_panel(
            "⚡ Performance",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Performance Tuning"),
                    create_input_group(
                        "Caching & Threads",
                        ui.input_checkbox(
                            f"{id}_caching_enabled",
                            "Enable Caching",
                            value=CONFIG.get("performance.enable_caching"),
                        ),
                        ui.input_numeric(
                            f"{id}_cache_ttl",
                            "Cache TTL (seconds)",
                            value=CONFIG.get("performance.cache_ttl"),
                            min=60,
                            max=86400,
                            width="100%",
                        ),
                        ui.input_checkbox(
                            f"{id}_compression_enabled",
                            "Enable Compression",
                            value=CONFIG.get("performance.enable_compression"),
                        ),
                        ui.input_numeric(
                            f"{id}_num_threads",
                            "Number of Threads",
                            value=CONFIG.get("performance.num_threads"),
                            min=1,
                            max=32,
                            width="100%",
                        ),
                        type="advanced",
                    ),
                    ui.br(),
                    ui.input_action_button(
                        f"{id}_btn_save_perf",
                        "💾 Save Performance Settings",
                        class_="btn-primary",
                        width="100%",
                    ),
                    width=350,
                    bg=COLORS["smoke_white"],
                ),
                ui.card(
                    ui.card_header("Info: Performance Guide"),
                    ui.markdown("""
                    ### Caching
                    - **TTL**: How long to keep cached results (seconds)
                    - **Typical**: 3600 (1 hour)
                    
                    ### Threading
                    - **Threads**: CPU cores available for parallel processing
                    - **Typical**: Set to # of CPU cores
                    
                    ### Compression
                    - Reduces memory usage for large datasets
                    - May slightly increase CPU usage
                    """),
                    full_screen=True,
                ),
            ),
        ),
        # ==========================================
        # 5. ADVANCED STATISTICS TAB
        # ==========================================
        ui.nav_panel(
            "📈 Advanced Stats", tab_advanced_stats.advanced_stats_ui(f"{id}_adv_stats")
        ),
        # ==========================================
        # 6. APP ADVANCED SETTINGS TAB
        # ==========================================
        ui.nav_panel(
            "🛠️ Advanced",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Advanced Settings"),
                    # Validation
                    create_input_group(
                        "Validation",
                        ui.input_checkbox(
                            f"{id}_strict_mode",
                            "Strict Mode",
                            value=CONFIG.get("validation.strict_mode"),
                        ),
                        ui.input_checkbox(
                            f"{id}_validate_inputs",
                            "Validate Inputs",
                            value=CONFIG.get("validation.validate_inputs"),
                        ),
                        ui.input_checkbox(
                            f"{id}_validate_outputs",
                            "Validate Outputs",
                            value=CONFIG.get("validation.validate_outputs"),
                        ),
                        ui.input_checkbox(
                            f"{id}_auto_fix_errors",
                            "Auto-fix Errors",
                            value=CONFIG.get("validation.auto_fix_errors"),
                        ),
                        type="advanced",
                    ),
                    # Debug
                    create_input_group(
                        "Debug",
                        ui.input_checkbox(
                            f"{id}_debug_enabled",
                            "Enable Debug Mode",
                            value=CONFIG.get("debug.enabled"),
                        ),
                        ui.input_checkbox(
                            f"{id}_debug_verbose",
                            "Verbose Output",
                            value=CONFIG.get("debug.verbose"),
                        ),
                        ui.input_checkbox(
                            f"{id}_profile_performance",
                            "Profile Performance",
                            value=CONFIG.get("debug.profile_performance"),
                        ),
                        ui.input_checkbox(
                            f"{id}_show_timings",
                            "Show Timings",
                            value=CONFIG.get("debug.show_timings"),
                        ),
                        type="advanced",
                    ),
                    ui.br(),
                    ui.input_action_button(
                        f"{id}_btn_save_advanced",
                        "💾 Save Advanced Settings",
                        class_="btn-primary",
                        width="100%",
                    ),
                    width=350,
                    bg=COLORS["smoke_white"],
                ),
                ui.card(
                    ui.card_header("⚠️ Advanced Options"),
                    ui.markdown("""
                    ### Validation
                    - **Strict**: Stop on validation errors
                    - **Validate**: Check data types and formats
                    
                    ### Debug
                    - **Debug Mode**: Show detailed debugging info
                    - **Verbose**: Print intermediate steps
                    - **Profile**: Measure CPU/memory usage
                    """),
                    full_screen=True,
                ),
            ),
        ),
        id=f"{id}_tabs",
    )


def settings_server(id: str, config: Any) -> None:
    """
    Wire server-side reactive handlers for the Settings UI and persist changes to the provided config.

    Registers a dynamic logging-status output, attaches save handlers for Analysis, UI, Logging, Performance, and Advanced settings that update corresponding config keys and emit user notifications on success or error, and initializes the Advanced Statistics submodule.

    Parameters:
        id (str): Module namespace used to build dynamic input/output IDs for the settings UI.
        config: Mutable config-like object (expects `.get(key)` and `.update(key, value)`) used to read and persist settings.
    """
    session = get_current_session()
    if session is None:
        logger.warning("No active session found for settings_server")
        return

    input = session.input
    output = session.output

    # ---------------------------------------------------------
    # Manual Output Registration for Dynamic IDs
    # ---------------------------------------------------------
    @render.text
    def _txt_logging_status() -> str:
        """Display current logging configuration status."""
        enabled = config.get("logging.enabled")
        level = config.get("logging.level")
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        return f"Logging Status: {status}\nLevel: {level}"

    setattr(output, f"{id}_txt_logging_status", _txt_logging_status)

    # ==========================================
    # ANALYSIS SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(input[f"{id}_btn_save_analysis"])
    def _save_analysis_settings() -> None:
        """Save analysis settings when button clicked."""
        try:
            config.update("analysis.logit_method", input[f"{id}_logit_method"]())
            config.update(
                "analysis.logit_screening_p", float(input[f"{id}_logit_screening_p"]())
            )
            config.update(
                "analysis.logit_max_iter", int(input[f"{id}_logit_max_iter"]())
            )
            config.update(
                "analysis.logit_min_cases", int(input[f"{id}_logit_min_cases"]())
            )
            config.update("analysis.survival_method", input[f"{id}_survival_method"]())
            config.update("analysis.cox_method", input[f"{id}_cox_method"]())
            config.update(
                "analysis.var_detect_threshold",
                int(input[f"{id}_var_detect_threshold"]()),
            )
            config.update(
                "analysis.var_detect_decimal_pct",
                float(input[f"{id}_var_detect_decimal_pct"]()),
            )
            config.update(
                "analysis.publication_style", input[f"{id}_publication_style"]()
            )
            config.update(
                "analysis.pvalue_bounds_lower",
                float(input[f"{id}_pvalue_bounds_lower"]()),
            )
            config.update(
                "analysis.pvalue_bounds_upper",
                float(input[f"{id}_pvalue_bounds_upper"]()),
            )
            config.update(
                "analysis.missing_strategy", input[f"{id}_missing_strategy"]()
            )
            config.update(
                "analysis.missing_threshold_pct",
                int(input[f"{id}_missing_threshold_pct"]()),
            )

            logger.info("✅ Analysis settings saved")
            ui.notification_show("✅ Analysis settings saved", type="message")
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error saving analysis settings")
            ui.notification_show(f"❌ Error: {e}", type="error")

    # ==========================================
    # UI SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(input[f"{id}_btn_save_ui"])
    def _save_ui_settings() -> None:
        """Save UI settings when button clicked."""
        try:
            config.update("ui.page_title", input[f"{id}_page_title"]())
            config.update("ui.theme", input[f"{id}_theme"]())
            config.update("ui.layout", input[f"{id}_layout"]())
            config.update("ui.table_max_rows", int(input[f"{id}_table_max_rows"]()))
            config.update(
                "ui.table_pagination", bool(input[f"{id}_table_pagination"]())
            )
            config.update(
                "ui.table_decimal_places", int(input[f"{id}_table_decimal_places"]())
            )
            config.update("ui.plot_width", int(input[f"{id}_plot_width"]()))
            config.update("ui.plot_height", int(input[f"{id}_plot_height"]()))
            config.update("ui.plot_dpi", int(input[f"{id}_plot_dpi"]()))
            config.update("ui.plot_style", input[f"{id}_plot_style"]())

            logger.info("✅ UI settings saved")
            ui.notification_show("✅ UI settings saved", type="message")
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error saving UI settings")
            ui.notification_show(f"❌ Error: {e}", type="error")

    # ==========================================
    # LOGGING SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(input[f"{id}_btn_save_logging"])
    def _save_logging_settings() -> None:
        """Save logging settings when button clicked."""
        try:
            config.update("logging.enabled", bool(input[f"{id}_logging_enabled"]()))
            config.update("logging.level", input[f"{id}_logging_level"]())
            config.update("logging.file_enabled", bool(input[f"{id}_file_enabled"]()))
            config.update("logging.log_dir", input[f"{id}_log_dir"]())
            config.update("logging.log_file", input[f"{id}_log_file"]())
            config.update(
                "logging.console_enabled", bool(input[f"{id}_console_enabled"]())
            )
            config.update("logging.console_level", input[f"{id}_console_level"]())
            config.update(
                "logging.log_file_operations", bool(input[f"{id}_log_file_ops"]())
            )
            config.update(
                "logging.log_data_operations", bool(input[f"{id}_log_data_ops"]())
            )
            config.update(
                "logging.log_analysis_operations",
                bool(input[f"{id}_log_analysis_ops"]()),
            )
            config.update(
                "logging.log_performance", bool(input[f"{id}_log_performance"]())
            )

            logger.info("✅ Logging settings saved")
            ui.notification_show("✅ Logging settings saved", type="message")
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error saving logging settings")
            ui.notification_show(f"❌ Error: {e}", type="error")

    # ==========================================
    # PERFORMANCE SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(input[f"{id}_btn_save_perf"])
    def _save_perf_settings() -> None:
        """
        Persist performance-related settings into the application's configuration.

        Updates the following config keys from input values: 'performance.enable_caching', 'performance.cache_ttl',
        'performance.enable_compression', and 'performance.num_threads'. On success logs an informational message
        and displays a success notification; on failure logs the exception and displays an error notification.
        """
        try:
            config.update(
                "performance.enable_caching", bool(input[f"{id}_caching_enabled"]())
            )
            config.update("performance.cache_ttl", int(input[f"{id}_cache_ttl"]()))
            config.update(
                "performance.enable_compression",
                bool(input[f"{id}_compression_enabled"]()),
            )
            config.update("performance.num_threads", int(input[f"{id}_num_threads"]()))

            logger.info("✅ Performance settings saved")
            ui.notification_show("✅ Performance settings saved", type="message")
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error saving performance settings")
            ui.notification_show(f"❌ Error: {e}", type="error")

    # ==========================================
    # ADVANCED STATS MODULE
    # ==========================================
    tab_advanced_stats.advanced_stats_server(f"{id}_adv_stats", config)

    # ==========================================
    # ADVANCED SETTINGS SAVE
    # ==========================================
    @reactive.Effect
    @reactive.event(input[f"{id}_btn_save_advanced"])
    def _save_advanced_settings() -> None:
        """Save advanced settings when button clicked."""
        try:
            config.update("validation.strict_mode", bool(input[f"{id}_strict_mode"]()))
            config.update(
                "validation.validate_inputs", bool(input[f"{id}_validate_inputs"]())
            )
            config.update(
                "validation.validate_outputs", bool(input[f"{id}_validate_outputs"]())
            )
            config.update(
                "validation.auto_fix_errors", bool(input[f"{id}_auto_fix_errors"]())
            )
            config.update("debug.enabled", bool(input[f"{id}_debug_enabled"]()))
            config.update("debug.verbose", bool(input[f"{id}_debug_verbose"]()))
            config.update(
                "debug.profile_performance", bool(input[f"{id}_profile_performance"]())
            )
            config.update("debug.show_timings", bool(input[f"{id}_show_timings"]()))

            logger.info("✅ Advanced settings saved")
            ui.notification_show("✅ Advanced settings saved", type="message")
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error saving advanced settings")
            ui.notification_show(f"❌ Error: {e}", type="error")
