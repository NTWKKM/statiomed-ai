"""
🔬 Multiple Imputation (MI) Helper Utilities.

Shared helpers for modules that support MI-pooled analyses.
Replaces duplicated ``_has_mi_datasets`` / ``_get_mi_datasets`` closures
previously defined inside individual module server functions.

Usage
-----
::

    from utils.mi_helpers import has_mi_datasets, get_mi_datasets

    mi_active = has_mi_datasets(mi_imputed_datasets)
    mi_dfs = get_mi_datasets(mi_imputed_datasets) if mi_active else []
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from shiny import reactive


def has_mi_datasets(
    mi_imputed_datasets: "reactive.Value[list[pd.DataFrame]] | None",
) -> bool:
    """
    Return True if multiple imputation datasets are available.

    Parameters
    ----------
    mi_imputed_datasets : reactive.Value[list[pd.DataFrame]] | None
        Reactive value holding the list of imputed datasets, or ``None``
        when the module does not expose MI support.

    Returns
    -------
    bool
        ``True`` if the reactive holds a non-empty list of DataFrames.
    """
    if mi_imputed_datasets is None:
        return False
    datasets = mi_imputed_datasets.get()
    return isinstance(datasets, list) and len(datasets) > 0


def get_mi_datasets(
    mi_imputed_datasets: "reactive.Value[list[pd.DataFrame]] | None",
) -> "list[pd.DataFrame]":
    """
    Retrieve the list of MI imputed DataFrames (safe, never raises).

    Parameters
    ----------
    mi_imputed_datasets : reactive.Value[list[pd.DataFrame]] | None
        Reactive value holding the list of imputed datasets, or ``None``.

    Returns
    -------
    list[pd.DataFrame]
        The list of imputed DataFrames, or an empty list if unavailable.
    """
    if mi_imputed_datasets is None:
        return []
    return mi_imputed_datasets.get() or []
