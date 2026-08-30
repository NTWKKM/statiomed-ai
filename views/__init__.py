"""
views - StatioMed AI Native Gradio Views Package
"""

from views.view_ai_copilot import create_ai_copilot_view
from views.view_data import create_data_view
from views.view_diagnostic import create_diagnostic_view
from views.view_meta_analysis import create_meta_analysis_view
from views.view_regression import create_regression_view
from views.view_sample_size import create_sample_size_view
from views.view_settings import create_settings_view
from views.view_survival import create_survival_view
from views.view_table_one_matching import create_table_one_matching_view

__all__ = [
    "create_ai_copilot_view",
    "create_data_view",
    "create_diagnostic_view",
    "create_meta_analysis_view",
    "create_regression_view",
    "create_sample_size_view",
    "create_settings_view",
    "create_survival_view",
    "create_table_one_matching_view",
]
