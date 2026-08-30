"""
📈 Table One Generator Module (Wrapper for Advanced Generator)
Backwards compatible wrapper around `utils.table_one_advanced`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger

# New Advanced Implementation
from utils.table_one_advanced import (
    StatisticalEngine,
    TableOneGenerator,
    VariableClassifier,
)

logger = get_logger(__name__)

# --- Legacy Functions (Kept for compatibility if imported directly) ---


def check_normality(series: pd.Series) -> bool:
    """
    Determine whether a pandas Series is classified as continuous normal.

    Parameters:
        series (pd.Series): The data series to classify.

    Returns:
        bool: True if the series is classified as continuous normal, False otherwise.
    """
    return VariableClassifier.classify(series) == "continuous_normal"


def format_p(p: float | str) -> str:
    """
    Format a p-value for display.

    Parameters:
        p (float | str): The p-value to format; may be numeric, NaN, or a string.

    Returns:
        str: "-" if `p` is NaN or a string, "<0.001" if `p` is numeric and less than 0.001, otherwise the p-value formatted to three decimal places.
    """
    if pd.isna(p) or isinstance(p, str):
        return "-"
    if float(p) < 0.001:
        return "<0.001"
    return f"{float(p):.3f}"


# Note: These legacy calculation functions are kept just in case external scripts import them.
# Internally, generate_table will use the new Engine.


def compute_or_ci(a, b, c, d) -> str:
    # See original implementation if needed, but for now we provide a functional stub or copy logic
    # Copying original logic to be safe for direct imports
    """
    Compute an odds ratio and its 95% confidence interval from a 2x2 contingency table.

    Parameters:
        a (float): Count in cell [0,0] of the 2x2 table.
        b (float): Count in cell [0,1] of the 2x2 table.
        c (float): Count in cell [1,0] of the 2x2 table.
        d (float): Count in cell [1,1] of the 2x2 table.

    Notes:
        If any input cell equals zero, a 0.5 continuity correction is added to all cells before calculation.

    Returns:
        str: Odds ratio and 95% confidence interval formatted as "OR (LCL-UCL)" with two decimal places, or "-" if the calculation cannot be performed.
    """
    try:
        if min(a, b, c, d) == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
        or_val = (a * d) / (b * c)
        if or_val == 0 or np.isinf(or_val):
            return "-"
        ln_or = np.log(or_val)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        return f"{or_val:.2f} ({np.exp(ln_or - 1.96 * se):.2f}-{np.exp(ln_or + 1.96 * se):.2f})"
    except Exception:
        return "-"


# --- Legacy Function Wrappers (Delegating to new StatisticalEngine) ---


def calculate_p_continuous(
    groups: list[pd.Series], normal: bool = True
) -> tuple[float | None, str]:
    """
    Compute the p-value and the name of the statistical test for comparing a continuous variable across groups.

    Parameters:
        groups (list[pd.Series]): A list of pandas Series, one per group, containing the continuous observations to compare.
        normal (bool): If True, assume group data are normally distributed (use parametric test); if False, use a nonparametric test.

    Returns:
        tuple[float | None, str]: A tuple (p_value, test_name) where `p_value` is the computed p-value or `None` if it cannot be computed, and `test_name` is the name of the statistical test applied.
    """
    # Note: StatisticalEngine expects list of Series
    return StatisticalEngine.calculate_p_continuous(groups, normal)


def calculate_p_categorical(
    df: pd.DataFrame, col: str, group_col: str
) -> tuple[float | None, str]:
    """
    Compute the p-value and test identifier for association between a categorical column and a grouping column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Name of the categorical column to test.
        group_col (str): Name of the column defining groups/strata.

    Returns:
        tuple:
            p_value (float | None): The p-value from the categorical association test, or `None` if the test could not be computed.
            test_name (str): A short identifier or description of the statistical test performed.
    """
    return StatisticalEngine.calculate_p_categorical(df, col, group_col)


def calculate_smd(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    g1_val: Any,
    g2_val: Any,
    is_cat: bool,
) -> str:
    """
    Compute the standardized mean difference (SMD) for a variable between two groups.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name of the variable to compare.
        group_col (str): Column name used to define groups.
        g1_val (Any): Value in `group_col` identifying the first group.
        g2_val (Any): Value in `group_col` identifying the second group.
        is_cat (bool): Whether `col` is categorical; alters the SMD calculation.

    Returns:
        str: The SMD formatted as a string for display.
    """
    return StatisticalEngine.calculate_smd(df, col, group_col, g1_val, g2_val, is_cat)


# Re-expose other functions if critical...
# For now, we assume most consumers only use generate_table


# --- MAIN ENTRY POINT ---


def generate_table(
    df: pd.DataFrame,
    selected_vars: list[str],
    group_col: str | None,
    var_meta: dict[str, Any] | None,
    or_style: str = "all_levels",
) -> str:
    """
    Generate a Baseline Characteristics HTML table from the given DataFrame using the Advanced TableOneGenerator.

    Parameters:
        df (pd.DataFrame): Input data frame containing variables and grouping column.
        selected_vars (list[str]): Variables to include in the table, in desired order.
        group_col (str | None): Column name to stratify by. If falsy or the string "None", no stratification is applied.
        var_meta (dict[str, Any] | None): Optional metadata for variables (labels, ordering, display settings).
        or_style (str): Odds ratio display style; "all_levels" to show ORs for each level, or "simple" for a compact display.

    Returns:
        str: HTML string containing the generated Baseline Characteristics table. Ensures the output includes a "Baseline Characteristics" header.
    """
    try:
        # Instantiate the new Generator
        generator = TableOneGenerator(df, var_meta=var_meta)

        # Determine stratify_by
        stratify = group_col if (group_col and group_col != "None") else None

        # Generate HTML
        # Note: The new generate method handles classification, stats, and HTML rendering internally.
        html_output = generator.generate(
            selected_vars, stratify_by=stratify, or_style=or_style
        )

        # Helper to match legacy output expectations (Integration Tests)
        if "Baseline Characteristics" not in html_output:
            html_output = f"<h3>Baseline Characteristics</h3>\n{html_output}"

        return html_output

    except Exception as e:
        logger.error(f"Error in Table One generation (Advanced Wrapper): {e}")
        # Fallback or re-raise?
        # Re-raise to match previous behavior so UI handles the error message
        raise
