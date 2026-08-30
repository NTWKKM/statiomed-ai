from pathlib import Path
from typing import Any

# Hugging Face ZeroGPU Free Tier Bootstrap Hook
# Hugging Face Free Tier runs Gradio Spaces on ZeroGPU and requires @spaces.GPU at module top-level.
try:
    import spaces

    @spaces.GPU(duration=45)
    def _zerogpu_startup_probe():
        """Hugging Face ZeroGPU supervisor startup probe."""
        pass
except ImportError:
    pass

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, ui

from config import CONFIG
from logger import LoggerFactory, get_logger

# Import Tabs
from tabs import (
    tab_ai_copilot,
    tab_baseline_matching,
    tab_core_regression,
    tab_data,
    tab_sample_size,
    tab_settings,
    tab_survival,
)

# Initialize Logger
LoggerFactory.configure()
logger = get_logger(__name__)

# UI Definition
app_ui = ui.page_navbar(
    ui.nav_panel(
        "🤖 AI Co-Pilot", tab_ai_copilot.ai_copilot_ui("ai_copilot"), value="ai_copilot"
    ),
    ui.nav_panel("📊 Data Profiler", tab_data.data_ui("data"), value="data"),
    ui.nav_panel("⏱️ Survival", tab_survival.survival_ui("survival"), value="survival"),
    ui.nav_panel(
        "📈 Regression",
        tab_core_regression.core_regression_ui("regression"),
        value="regression",
    ),
    ui.nav_panel(
        "📐 Sample Size",
        tab_sample_size.sample_size_ui("sample_size"),
        value="sample_size",
    ),
    ui.nav_panel(
        "👥 Table 1 & Matching",
        tab_baseline_matching.baseline_matching_ui("bm"),
        value="bm",
    ),
    ui.nav_panel("⚙️ Settings", tab_settings.settings_ui("settings"), value="settings"),
    title="🏥 StatioMed AI",
    id="main_navbar",
    selected="ai_copilot",
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    # Global Reactive Dataset and Shared Clinical Analysis State
    shared_dataset: reactive.Value[pd.DataFrame | None] = reactive.Value(None)
    var_meta: reactive.Value[dict[str, Any]] = reactive.Value({})
    uploaded_file_info: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    df_matched: reactive.Value[pd.DataFrame | None] = reactive.Value(None)
    is_matched: reactive.Value[bool] = reactive.Value(False)
    matched_treatment_col: reactive.Value[str | None] = reactive.Value(None)
    matched_covariates: reactive.Value[list[str]] = reactive.Value([])
    mi_imputed_datasets: reactive.Value[list[pd.DataFrame]] = reactive.Value([])

    # Initialize Modules
    tab_ai_copilot.ai_copilot_server("ai_copilot", shared_dataset)
    tab_data.data_server(
        "data",
        shared_dataset,
        var_meta,
        uploaded_file_info,
        df_matched,
        is_matched,
        matched_treatment_col,
        matched_covariates,
        mi_imputed_datasets,
    )
    tab_survival.survival_server(
        "survival",
        shared_dataset,
        var_meta,
        df_matched,
        is_matched,
        mi_imputed_datasets,
    )
    tab_core_regression.core_regression_server(
        "regression",
        shared_dataset,
        var_meta,
        df_matched,
        is_matched,
        mi_imputed_datasets,
    )
    tab_sample_size.sample_size_server("sample_size")
    tab_baseline_matching.baseline_matching_server(
        "bm",
        shared_dataset,
        var_meta,
        df_matched,
        is_matched,
        matched_treatment_col,
        matched_covariates,
    )
    tab_settings.settings_server("settings", CONFIG)


# Create Shiny App
shiny_app = App(app_ui, server, static_assets=Path(__file__).parent / "static")
app = shiny_app

if __name__ == "__main__":
    import os

    import uvicorn

    from asgi import app as asgi_app

    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    uvicorn.run(
        asgi_app,
        host="0.0.0.0",
        port=port,
        loop="asyncio",
    )
