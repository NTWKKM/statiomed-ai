"""
core/state.py - StatioMed AI Session State Management
=====================================================
Session-isolated application state dataclass for StatioMed AI.
Enables reactive state sharing across clinical analysis tabs,
propensity score matching, multiple imputation, and LLM co-pilot.
=====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AppState:
    """
    Session-isolated application state for StatioMed AI Gradio application.
    Enables reactive state sharing across clinical analysis tabs.
    """

    # Primary Dataset State
    df: pd.DataFrame | None = None
    file_name: str = "No dataset loaded"
    file_size_bytes: int = 0
    var_meta: dict[str, Any] = field(default_factory=dict)

    # Matched Cohort State (Propensity Score Matching)
    df_matched: pd.DataFrame | None = None
    is_matched: bool = False
    matched_treatment_col: str | None = None
    matched_covariates: list[str] = field(default_factory=list)

    # Multiple Imputation Datasets (MICE)
    mi_imputed_datasets: list[pd.DataFrame] = field(default_factory=list)

    # Cached Model Results / Analysis Outputs
    last_analysis_type: str | None = None
    last_analysis_results: dict[str, Any] = field(default_factory=dict)

    def has_data(self) -> bool:
        """Returns True if a primary dataset is loaded and non-empty."""
        return self.df is not None and not self.df.empty

    def has_matched_data(self) -> bool:
        """Returns True if a balanced propensity-score matched cohort is active."""
        return (
            self.is_matched
            and self.df_matched is not None
            and not self.df_matched.empty
        )

    def get_active_dataframe(self, use_matched: bool = False) -> pd.DataFrame | None:
        """
        Returns the currently active DataFrame.
        If `use_matched` is True and matched data exists, returns the matched cohort.
        """
        if use_matched and self.has_matched_data():
            return self.df_matched
        return self.df

    def get_columns(self, numeric_only: bool = False) -> list[str]:
        """Returns column names from active DataFrame."""
        if not self.has_data() or self.df is None:
            return []
        if numeric_only:
            return self.df.select_dtypes(include=["number"]).columns.tolist()
        return self.df.columns.tolist()

    def get_numeric_columns(self) -> list[str]:
        """Returns strictly numeric columns."""
        return self.get_columns(numeric_only=True)

    def get_categorical_columns(self, max_cardinality: int = 12) -> list[str]:
        """
        Returns binary and low-cardinality categorical candidate columns.
        """
        if not self.has_data() or self.df is None:
            return []
        cat_cols = []
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if (
                dtype == "object"
                or dtype.name == "category"
                or dtype == "bool"
                or self.df[col].nunique(dropna=True) <= max_cardinality
            ):
                cat_cols.append(col)
        return cat_cols

    def get_summary_dict(self) -> dict[str, Any]:
        """Returns high-level summary statistics of the active dataset."""
        if not self.has_data() or self.df is None:
            return {
                "loaded": False,
                "rows": 0,
                "cols": 0,
                "file_name": self.file_name,
                "is_matched": False,
            }
        return {
            "loaded": True,
            "rows": len(self.df),
            "cols": len(self.df.columns),
            "columns": self.df.columns.tolist(),
            "missing_cells": int(self.df.isna().sum().sum()),
            "missing_pct": float((self.df.isna().sum().sum() / self.df.size) * 100.0)
            if self.df.size > 0
            else 0.0,
            "file_name": self.file_name,
            "is_matched": self.is_matched,
            "matched_rows": len(self.df_matched) if self.df_matched is not None else 0,
        }

    def reset_dataset(self) -> None:
        """Resets the state back to default empty state."""
        self.df = None
        self.file_name = "No dataset loaded"
        self.file_size_bytes = 0
        self.var_meta = {}
        self.df_matched = None
        self.is_matched = False
        self.matched_treatment_col = None
        self.matched_covariates = []
        self.mi_imputed_datasets = []
        self.last_analysis_type = None
        self.last_analysis_results = {}
