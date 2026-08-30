from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, ui

from config import CONFIG
from logger import LoggerFactory, get_logger

# Import Tabs
from tabs import (
    tab_ai_copilot,
    tab_home,
    tab_data,
    tab_baseline_matching,
    tab_core_regression,
    tab_survival,
    tab_diag,
    tab_corr,
    tab_agreement,
    tab_meta_analysis,
    tab_causal_inference,
    tab_sample_size,
    tab_settings,
)
from tabs._common import wrap_with_container

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
    # Global Reactive Dataset
    shared_dataset: reactive.Value[pd.DataFrame | None] = reactive.Value(None)

    # Initialize Modules
    tab_ai_copilot.ai_copilot_server("ai_copilot", shared_dataset)
    tab_data.data_server("data", shared_dataset)
    tab_survival.survival_server("survival", shared_dataset)
    tab_core_regression.core_regression_server("regression", shared_dataset)
    tab_sample_size.sample_size_server("sample_size", shared_dataset)
    tab_baseline_matching.baseline_matching_server("bm", shared_dataset)
    tab_settings.settings_server("settings")


app = App(app_ui, server, static_assets=Path(__file__).parent / "static")

if __name__ == "__main__":
    import uvicorn
    from asgi import app as asgi_app

    uvicorn.run(asgi_app, host="0.0.0.0", port=7860)
