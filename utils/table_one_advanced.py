"""
📈 Advanced Table One Generator (OOP Architecture)

This module implements an object-oriented approach to Table 1 generation,
separating classification, statistical calculation, and formatting logic.
Refactored from `utils/table_one.py` for better maintainability and testing.
"""

from __future__ import annotations

import html as _html
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import (
    clean_numeric_vector,
    prepare_data_for_analysis,
)
from utils.formatting import create_missing_data_report_html
from utils.logic import run_binary_logit

logger = get_logger(__name__)


def _numeric_sort_key(x: Any) -> tuple[int, float | str]:
    """
    Create a sort key that places numeric-like values before non-numeric values and preserves numeric order.

    Returns:
        tuple: A two-element tuple where the first element is 0 for inputs parseable as a float and 1 otherwise; the second element is the parsed float for numeric inputs or the original string representation for non-numeric inputs.
    """
    s = str(x)
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


# --- 1. Data Structures ---
@dataclass
class VariableAnalysis:
    """
    Holds the full analysis results for a single variable row in Table 1.
    """

    name: str  # Column name
    label: str  # Display label
    var_type: Literal[
        "continuous_normal",
        "continuous_non_normal",
        "categorical",
        "ordinal",
        "unknown",
    ]
    stats_overall: str  # e.g., "50.2 ± 10.1" or "120 (45.3%)"
    stats_groups: dict[str, str] = field(default_factory=dict)  # GroupID -> Stat string
    p_value: float | None = None
    test_name: str | None = None
    or_test_name: str | None = None
    warning: str | None = None
    extra_stats: dict[str, Any] = field(default_factory=dict)  # e.g., OR, SMD
    categorical_counts: pd.Series | None = None  # Store raw counts for OR/SMD calcs


# --- 2. Classifier Logic ---
class VariableClassifier:
    """
    Intelligent variable type inference and distribution checking.
    """

    @staticmethod
    def classify(series: pd.Series, max_cat_unique: int = 10) -> str:
        """
        Infer the reporting type of a variable for Table 1.

        Parameters:
            series (pd.Series): Input column to classify; missing values are ignored.
            max_cat_unique (int): Maximum number of unique values to treat a numeric-like series as categorical.

        Returns:
            var_type (str): One of "categorical", "continuous_normal", "continuous_non_normal", or "unknown".
        """
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return "unknown"

        is_numeric = pd.api.types.is_numeric_dtype(clean_series)
        unique_count = clean_series.nunique()

        # Heuristic for Categorical
        if (
            not is_numeric
            or pd.api.types.is_object_dtype(clean_series)
            or isinstance(clean_series.dtype, pd.CategoricalDtype)
            or unique_count <= max_cat_unique
        ):
            return "categorical"

        # Continuous Distribution Check
        return VariableClassifier._check_normality_advanced(clean_series)

    @staticmethod
    def _check_normality_advanced(series: pd.Series) -> str:
        """
        Determine whether a numeric series should be treated as normally distributed for Table 1 reporting.

        Uses Shapiro-Wilk for samples smaller than 5000 and Jarque-Bera for larger samples; also considers descriptive criteria (absolute skewness < 1.0 and absolute kurtosis < 2.0) when sample size exceeds 50. Series with fewer than 3 observations are classified as non-normal. If an internal error occurs, the function logs a warning and returns non-normal.

        Returns:
            'continuous_normal' if the series is assessed as normal, 'continuous_non_normal' otherwise.
        """
        n = len(series)
        if n < 3:
            return "continuous_non_normal"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if n < 5000:
                    stat, p = stats.shapiro(series)
                else:
                    stat, p = stats.jarque_bera(series)

            # P > 0.05 indicates normality
            is_normal_test = p > 0.05

            # Descriptive Checks
            skew = stats.skew(series)
            kurt = stats.kurtosis(series)
            is_normal_desc = abs(skew) < 1.0 and abs(kurt) < 2.0

            if is_normal_test or (n > 50 and is_normal_desc):
                return "continuous_normal"
            else:
                return "continuous_non_normal"

        except Exception as e:
            logger.warning(f"Normality check failed: {e}")
            return "continuous_non_normal"


# --- 3. Statistical Engine ---
class StatisticalEngine:
    """
    Encapsulates all statistical calculations.
    """

    @staticmethod
    def get_stats_continuous(series: pd.Series, normal: bool = True) -> str:
        """
        Return a concise summary string for a continuous variable using mean±SD for normally distributed data or median [Q1, Q3] for non-normal data.

        Parameters:
            series (pd.Series): Numeric values; non-numeric entries will be coerced/cleaned and NA values dropped.
            normal (bool): If True, format as mean ± SD; if False, format as median [Q1, Q3].

        Returns:
            str: Formatted summary string (e.g., "50.2 ± 10.1" or "120.0 [100.0, 140.0]"), or "-" if no non-missing numeric values are present.
        """
        clean = clean_numeric_vector(series).dropna()
        if len(clean) == 0:
            return "-"

        if normal:
            mean = clean.mean()
            sd = clean.std()
            if not np.isfinite(sd):
                return f"{mean:.1f} ± 0.0"
            return f"{mean:.1f} ± {sd:.1f}"
        else:
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            return f"{clean.median():.1f} [{q1:.1f}, {q3:.1f}]"

    @staticmethod
    def get_stats_categorical(
        series: pd.Series, total_n: int | None = None
    ) -> tuple[str, pd.Series]:
        """
        Produce an HTML-ready categorical summary showing counts and percentages per category.

        Parameters:
            series (pd.Series): Values for the categorical variable; missing values are ignored.
            total_n (int | None): Optional denominator used to compute percentages. If None, percentages use the number of non-missing values in `series`.

        Returns:
            summary_html (str): A string with category lines like "Category: count (x.x%)" joined by `<br>`. Returns "-" when there are no non-missing values. Category labels are HTML-escaped.
            counts (pd.Series): Value counts for each category sorted by category index.
        """
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return "-", pd.Series([], dtype=object)

        counts = clean_series.value_counts()
        sorted_cats = sorted(counts.index.tolist(), key=_numeric_sort_key)
        counts = counts.reindex(sorted_cats)
        denominator = total_n if total_n is not None else len(clean_series)

        if denominator == 0:
            return "-", counts

        res_parts = []
        for cat, count in counts.items():
            pct = (count / denominator) * 100
            res_parts.append(f"{_html.escape(str(cat))}: {int(count)} ({pct:.1f}%)")

        return "<br>".join(res_parts), counts

    @staticmethod
    def calculate_p_continuous(
        groups: list[pd.Series], normal: bool = True
    ) -> tuple[float | None, str]:
        """
        Choose and run an appropriate hypothesis test for comparing a numeric variable across groups.

        Parameters:
            groups (list[pd.Series]): List of group vectors to compare; each series may contain non-numeric or missing values which will be dropped.
            normal (bool): If True, use parametric tests (t-test for two groups, ANOVA for >2); if False, use nonparametric alternatives.

        Returns:
            tuple:
                p_value (float | None): The p-value from the selected test, or `None` if the test was not performed.
                test_name (str): The name of the test used ("t-test", "ANOVA", "Mann-Whitney U", "Kruskal-Wallis"), "-" when fewer than two valid groups are provided, or "Error" if the test could not be completed.
        """
        clean_groups = [clean_numeric_vector(g).dropna() for g in groups]
        clean_groups = [g for g in clean_groups if len(g) > 0]

        if len(clean_groups) < 2:
            return None, "-"

        try:
            if normal:
                if len(clean_groups) == 2:
                    _, p = stats.ttest_ind(
                        clean_groups[0], clean_groups[1], nan_policy="omit"
                    )
                    return p, "t-test"
                else:
                    _, p = stats.f_oneway(*clean_groups)
                    return p, "ANOVA"
            else:
                if len(clean_groups) == 2:
                    _, p = stats.mannwhitneyu(
                        clean_groups[0], clean_groups[1], alternative="two-sided"
                    )
                    return p, "Mann-Whitney U"
                else:
                    _, p = stats.kruskal(*clean_groups)
                    return p, "Kruskal-Wallis"
        except Exception:
            return None, "Error"

    @staticmethod
    def calculate_p_categorical(
        df: pd.DataFrame, col: str, group_col: str
    ) -> tuple[float | None, str]:
        """
        Compute a p-value for association between a categorical variable and a grouping variable using chi-square or Fisher's exact test.

        Parameters:
            df (pd.DataFrame): Data frame containing the data.
            col (str): Column name of the categorical variable to test.
            group_col (str): Column name of the grouping variable.

        Returns:
            tuple[float | None, str]: A tuple (p_value, test_name).
                - p_value: The test p-value, or `None` if the test could not be performed (e.g., fewer than two categories or groups, empty contingency table, or an error).
                - test_name: Name of the test used ("Chi-square", "Chi-square (Low N)", "Fisher's Exact"), "-" when there are not enough categories/groups to test, or "Error" if an unexpected exception occurred.
        """
        try:
            if df[group_col].dropna().nunique() < 2 or df[col].dropna().nunique() < 2:
                return None, "-"
            tab = pd.crosstab(df[col], df[group_col])
            if tab.size == 0:
                return None, "-"

            _, p, _, expected = stats.chi2_contingency(tab)
            min_expected = expected.min()

            # Use Fisher if small samples (2x2 only mostly)
            if tab.shape == (2, 2) and min_expected < 5:
                try:
                    _, p_fisher = stats.fisher_exact(tab)
                    return p_fisher, "Fisher's Exact"
                except (ValueError, TypeError):
                    # Fisher's exact not applicable (non-2x2 or invalid table);
                    # fall through to return chi2 result below
                    pass

            test_name = "Chi-square" if min_expected >= 5 else "Chi-square (Low N)"
            return p, test_name
        except Exception:
            return None, "Error"

    @staticmethod
    def calculate_smd(
        df: pd.DataFrame,
        col: str,
        group_col: str,
        g1_val: Any,
        g2_val: Any,
        is_cat: bool,
    ) -> str:
        """
        Compute the standardized mean difference (SMD) between two groups for a specified column.

        For numeric columns this returns a single SMD value formatted to three decimals. For categorical columns this returns per-category SMDs joined with HTML <br> separators in the order of category sorting. Values with magnitude >= 0.1 are wrapped in <b>...</b> to highlight imbalance. The string "<b>—</b>" denotes an undefined SMD due to effectively zero pooled standard deviation for that comparison. A single hyphen "-" indicates insufficient data or an internal error.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            col (str): Column name to compare.
            group_col (str): Column name used to define groups.
            g1_val (Any): Value in group_col identifying the first group.
            g2_val (Any): Value in group_col identifying the second group.
            is_cat (bool): If True, treat `col` as categorical and compute per-category SMDs; otherwise treat as numeric.

        Returns:
            str: Formatted SMD result as described above.
        """
        try:
            # Reusing robust logic from original table_one for reliability
            mask1 = df[group_col].astype(str) == str(g1_val)
            mask2 = df[group_col].astype(str) == str(g2_val)

            if is_cat:
                s1 = df.loc[mask1, col].dropna()
                s2 = df.loc[mask2, col].dropna()
                if len(s1) == 0 or len(s2) == 0:
                    return "-"

                # Get union of categories
                cats = set(s1.unique()) | set(s2.unique())
                if not cats:
                    return "-"

                smd_vals = []
                for cat in sorted(list(cats), key=_numeric_sort_key):
                    p1 = (s1.astype(str) == str(cat)).mean()
                    p2 = (s2.astype(str) == str(cat)).mean()
                    pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)

                    if not np.isfinite(pooled_sd) or pooled_sd <= 1e-8:
                        if abs(p1 - p2) < 1e-8:
                            val_str = "0.000"
                        else:
                            val_str = "<b>—</b>"
                            logger.warning(
                                f"SMD undefined (pooled_sd~0) for {col}={cat} with p1={p1:.3f}, p2={p2:.3f}"
                            )
                    else:
                        smd = abs(p1 - p2) / pooled_sd
                        val_str = f"{smd:.3f}"
                        if smd >= 0.1:
                            val_str = f"<b>{val_str}</b>"
                    smd_vals.append(val_str)
                return "<br>".join(smd_vals)
            else:
                v1 = clean_numeric_vector(df.loc[mask1, col]).dropna()
                v2 = clean_numeric_vector(df.loc[mask2, col]).dropna()
                if len(v1) == 0 or len(v2) == 0:
                    return "-"

                pooled_sd = np.sqrt((v1.std() ** 2 + v2.std() ** 2) / 2)
                if not np.isfinite(pooled_sd) or pooled_sd <= 1e-8:
                    if abs(v1.mean() - v2.mean()) < 1e-8:
                        val_str = "0.000"
                    else:
                        val_str = "<b>—</b>"
                        logger.warning(
                            f"SMD undefined (pooled_sd~0) for {col} with means {v1.mean():.3f}, {v2.mean():.3f}"
                        )
                    return val_str
                smd = abs(v1.mean() - v2.mean()) / pooled_sd

                val_str = f"{smd:.3f}"
                if smd >= 0.1:
                    val_str = f"<b>{val_str}</b>"
                return val_str
        except Exception:
            return "-"

    @staticmethod
    def calculate_or(
        df: pd.DataFrame,
        col: str,
        group_col: str,
        g1_val: Any,
        g2_val: Any,
        is_cat: bool,
        or_style: str = "all_levels",
    ) -> tuple[str | dict[str, str], str]:
        """
        Compute odds ratios (ORs) comparing two groups for a given variable and return the OR(s) with the method name.

        Computes an OR and 95% confidence interval for a numeric variable using univariate logistic regression or for categorical variables using 2x2 tables (each level versus a reference). Group g1_val is treated as the reference and g2_val as the comparison. For categorical variables, `or_style` controls whether to return a single comparison ("simple") or ORs for all levels versus the reference ("all_levels").

        Parameters:
            df (pd.DataFrame): Source table containing the variable and group column.
            col (str): Column name of the predictor variable for which ORs are calculated.
            group_col (str): Column name used to define the two groups to compare.
            g1_val (Any): Value in `group_col` treated as the reference group.
            g2_val (Any): Value in `group_col` treated as the comparison group.
            is_cat (bool): True if `col` is categorical, False if numeric (continuous).
            or_style (str): "all_levels" to return ORs for every category versus reference,
                or "simple" to return only the last category versus the reference for categorical variables.

        Returns:
            tuple[result, method_name]:
                result: For numeric predictors, a formatted string "OR (LB-UB)" or "-" on failure.
                        For categorical predictors with `or_style="all_levels"`, a dict mapping category -> "OR (LB-UB)" or "Ref."/"-".
                        For categorical predictors with `or_style="simple"`, a single string "OR (LB-UB)" or "-".
                method_name: Short descriptor of the method used ("Univar. Logit", "2x2 (vs Ref)", "2x2 (<cat> vs <ref>)", or "-" / "Error").
        """
        try:
            # Filter to only the 2 groups
            mask_g1 = df[group_col].astype(str) == str(g1_val)
            mask_g2 = df[group_col].astype(str) == str(g2_val)

            sub_df = df[mask_g1 | mask_g2].copy()
            # Create binary target: 0 = g1 (Ref), 1 = g2 (Case/Comp)
            # Note: We want OR of being in G2 given the variable.
            # Logit P(Group=G2) ~ Variable.
            sub_df["_target"] = np.where(
                sub_df[group_col].astype(str) == str(g2_val), 1, 0
            )

            clean_df = sub_df[[col, "_target"]].dropna()

            if len(clean_df) < 10:  # Min sample size heuristic
                return "-", "-"

            if not is_cat:
                # Continuous: Univariate Logistic Regression
                # Fix: Ensure numeric
                x = pd.to_numeric(clean_df[col], errors="coerce")
                valid_mask = x.notna()
                x_clean = x[valid_mask]
                y = clean_df.loc[valid_mask, "_target"]

                if len(x_clean) < 10:
                    return "-", "-"

                # Check variance
                if x_clean.std() <= 1e-8:
                    return "-", "-"

                # Logit via shared module - use cleaned numeric data
                x_df = x_clean.to_frame(name=col)
                params, conf, _, status, _ = run_binary_logit(y, x_df)

                if status == "OK" and params is not None and col in params:
                    or_val = np.exp(params[col])
                    ci_lower = np.exp(conf.loc[col, 0])
                    ci_upper = np.exp(conf.loc[col, 1])

                    return (
                        f"{or_val:.2f} ({ci_lower:.2f}-{ci_upper:.2f})",
                        "Univar. Logit",
                    )
                else:
                    return "-", "Univar. Logit"

            else:
                # Categorical: 2x2 Tables for each level vs Reference
                # Logic: Compare specific level (Target) vs Reference level (Ref).
                # Ref = First Category (cats[0]).

                s1 = sub_df.loc[sub_df["_target"] == 1, col]  # G2 (Cases)
                s0 = sub_df.loc[sub_df["_target"] == 0, col]  # G1 (Ref)

                # Get all categories and sort naturally to identify Ref/Target
                unique_cats = set(s1.dropna().unique()) | set(s0.dropna().unique())

                # Sort logic (try numeric if possible)
                sorted_cats = sorted(list(unique_cats), key=_numeric_sort_key)

                if not sorted_cats:
                    return "-", "-"

                # 1. Identify Reference (First Category)
                ref_cat = sorted_cats[0]

                # Pre-calculate Reference Counts (c, d in standard 2x2)
                # c = Cases with Ref
                # d = Controls with Ref
                c_base = (s1.astype(str) == str(ref_cat)).sum()
                d_base = (s0.astype(str) == str(ref_cat)).sum()

                # Helper 2x2 Calc
                def calc_2x2_or(cat_val, c_ref, d_ref):
                    """
                    Compute the odds ratio and 95% confidence interval for a 2x2 comparison of a categorical level versus a reference level.

                    Parameters:
                        cat_val: value of the category level to compare against the reference.
                        c_ref (float | int): count of reference-level events in the case group (cell c).
                        d_ref (float | int): count of reference-level events in the control group (cell d).

                    Returns:
                        str: Formatted result as "OR (lower-upper)" where OR and bounds are shown with two decimals;
                             returns "Ref." if `cat_val` equals the reference category, "-" when the OR is undefined
                             (e.g., division by zero even after Haldane-Anscombe correction) or infinite.
                    """
                    if str(cat_val) == str(ref_cat):
                        return "Ref."

                    # a = Cases with Target
                    # b = Controls with Target
                    a = (s1.astype(str) == str(cat_val)).sum()
                    b = (s0.astype(str) == str(cat_val)).sum()

                    # Cells for 2x2:
                    #        Case   Control
                    # Target   a      b
                    # Ref      c      d
                    c_h = c_ref
                    d_h = d_ref

                    # Haldane-Anscombe correction if any cell is 0
                    if 0 in [a, b, c_h, d_h]:
                        a += 0.5
                        b += 0.5
                        c_h += 0.5
                        d_h += 0.5

                    if (b * c_h) == 0:
                        return "-"

                    or_val = (a * d_h) / (b * c_h)

                    if np.isinf(or_val):
                        return "-"

                    ln_or = np.log(or_val)
                    se = np.sqrt(1 / a + 1 / b + 1 / c_h + 1 / d_h)
                    ci_lower = np.exp(ln_or - 1.96 * se)
                    ci_upper = np.exp(ln_or + 1.96 * se)

                    return f"{or_val:.2f} ({ci_lower:.2f}-{ci_upper:.2f})"

                if or_style == "simple":
                    # Simple: Last vs First (Ref)
                    target_cat = sorted_cats[-1]
                    res_str = calc_2x2_or(target_cat, c_base, d_base)

                    # Method Name
                    method = f"2x2 ({target_cat} vs {ref_cat})"
                    return res_str, method

                else:
                    # All Levels: Each vs Ref
                    results = {}
                    for cat in sorted_cats:
                        results[str(cat)] = calc_2x2_or(cat, c_base, d_base)

                    return results, "2x2 (vs Ref)"

        except Exception as e:
            logger.warning(f"Failed to calc OR for {col}: {e}")
            return "-", "Error"


# --- 4. Formatter ---
class TableOneFormatter:
    """
    Handles HTML rendering matching Shiny style.
    """

    COLORS = get_color_palette()

    @staticmethod
    def format_p(p: float | None) -> str:
        """
        Format a p-value for display in Table 1.

        Parameters:
            p (float | None): The p-value to format; may be None or NaN.

        Returns:
            str: "-" if `p` is None or NaN, "<0.001" if `p` < 0.001, otherwise the p-value rounded to three decimal places (e.g., "0.123").
        """
        if p is None or np.isnan(p):
            return "-"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    @classmethod
    def render_html(
        cls,
        results: list[VariableAnalysis],
        groups: list[dict],
        group_masks: dict,
        total_n: int,
        missing_info: dict,
        var_meta: dict | None = None,
    ) -> str:
        # Helper helpers
        """
        Render an HTML Table 1 from precomputed variable analyses and grouping information.

        Parameters:
            results (list[VariableAnalysis]): Ordered list of per-variable analysis results (labels, stats, p-values, extra stats).
            groups (list[dict]): List of group descriptors, each with keys "val" (group value) and "label" (display label).
            group_masks (dict): Mapping from group value to boolean mask (or boolean Series) selecting rows belonging to that group.
            total_n (int): Total sample size used for the "Total (N=...)" column.
            missing_info (dict): Information about missing data used to generate a supplemental missing-data report appended to the HTML.
            var_meta (dict | None): Optional variable metadata used when rendering the missing-data report (may be None).

        Returns:
            str: HTML string containing the formatted Table 1 (table markup and inline CSS) followed by the missing-data report.
        """

        def _safe_text(val):
            return _html.escape(str(val)) if val is not None else "-"

        def _p_cell(p, test):
            if p is None:
                return "<td>-</td><td>-</td>"
            p_str = cls.format_p(p)
            is_sig = p is not None and p < 0.05
            p_cls = "p-significant" if is_sig else "p-not-significant"
            star = "*" if is_sig else ""

            # Combine P and Test Name, if OR method exists, append it
            test_html = f"{test}"
            return (
                f"<td class='numeric-cell'><span class='{p_cls}'>{p_str}{star}</span></td>"
                f"<td class='numeric-cell'>{test_html}</td>"
            )

        # Basic CSS (same as original to ensure UI consistency)
        css = f"""
        <style>
            .numeric-cell {{ font-family: 'Courier New', monospace; font-size: 12px; }}
            .p-significant {{ color: {cls.COLORS["danger"]}; font-weight: 600; }}
            .p-not-significant {{ color: {cls.COLORS["success"]}; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            th {{ background-color: {cls.COLORS["primary_dark"]}; color: white; padding: 10px; text-align: center; }}
            td {{ padding: 8px; border: 1px solid {cls.COLORS["border"]}; }}
            tr:nth-child(even) {{ background-color: {cls.COLORS["primary_light"]}; }}
        </style>
        """

        # Table Header
        header_cols = (
            f"<th>Characteristic</th><th class='numeric-cell'>Total (N={total_n})</th>"
        )
        for g in groups:
            n_g = group_masks[g["val"]].sum()
            header_cols += f"<th class='numeric-cell'>{_html.escape(str(g['label']))} (n={n_g})</th>"

        if len(groups) == 2:
            header_cols += (
                "<th>SMD</th><th>OR (95% CI)</th><th>P-value</th><th>Test</th>"
            )
        elif len(groups) > 0:
            header_cols += "<th>P-value</th><th>Test</th>"

        rows_html = ""
        for res in results:
            row = f"<tr><td><strong>{_html.escape(res.label)}</strong>"
            if res.var_type == "continuous_non_normal":
                row += "<br><small>(Median [IQR])</small>"
            elif res.var_type == "continuous_normal":
                row += "<br><small>(Mean ± SD)</small>"
            row += "</td>"

            # Total
            row += f"<td class='numeric-cell'>{res.stats_overall}</td>"

            # Groups
            for g in groups:
                val = res.stats_groups.get(g["val"], "-")
                row += f"<td class='numeric-cell'>{val}</td>"

            # Stats (SMD/P)
            if len(groups) == 2:
                smd = res.extra_stats.get("smd", "-")
                or_val = res.extra_stats.get("or", "-")

                # Handle OR if it's a dict (categorical levels)
                row += f"<td class='numeric-cell'>{smd}</td>"
                row += f"<td class='numeric-cell'>{or_val}</td>"

                # Append OR method to test name if available
                final_test_name = _safe_text(res.test_name)
                if res.or_test_name and res.or_test_name != "-":
                    final_test_name += (
                        f"<br><small>({_safe_text(res.or_test_name)})</small>"
                    )

                row += _p_cell(res.p_value, final_test_name)
            elif len(groups) > 0:
                row += _p_cell(res.p_value, _safe_text(res.test_name))

            row += "</tr>"
            rows_html += row

        html = f"""
        {css}
        <div class="table-wrapper">
            <table>
                <thead><tr>{header_cols}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
        # Append missing data info
        html += create_missing_data_report_html(missing_info, var_meta or {})
        return html


# --- 5. Main Generator ---
class TableOneGenerator:
    """
    Orchestrator class.
    """

    def __init__(self, df: pd.DataFrame, var_meta: dict | None = None):
        """
        Initialize the TableOneGenerator with source data and optional variable metadata.

        Parameters:
            df (pd.DataFrame): Source dataframe used for all analyses.
            var_meta (dict | None): Optional mapping of variable names to metadata (labels, display options); defaults to empty dict.

        Attributes:
            raw_df (pd.DataFrame): Reference to the provided source dataframe.
            var_meta (dict): Stored variable metadata.
            classifier (VariableClassifier): Instance used to infer variable types.
            stats_engine (StatisticalEngine): Instance used to compute summary statistics and tests.
        """
        self.raw_df = df
        self.var_meta = var_meta or {}
        self.classifier = VariableClassifier()
        self.stats_engine = StatisticalEngine()

    def _format_categorical_or(
        self, var: str, or_calc: str | dict, df_clean: pd.DataFrame
    ) -> str:
        """
        Format odds-ratio results for a categorical variable into a '<br>'-separated string aligned with the variable's displayed category order.

        Parameters:
            var (str): Column name of the categorical variable.
            or_calc (str | dict): OR results either as a dictionary mapping category -> formatted OR string, or a single formatted OR string (simple/binary style).
            df_clean (pd.DataFrame): DataFrame used to derive the category order for `var`.

        Returns:
            str: A '<br>'-joined string containing one entry per category in display order. When `or_calc` is a dict, each category's mapped value is used (missing entries become '-'). When `or_calc` is a single string and there are two categories, the first category is rendered as the reference ('-') and the second as the provided string; for more than two categories the single string is placed first and remaining entries are empty strings.
        """
        # Reconstruct the string sequence to match the visual order of categories.
        _, counts_series = self.stats_engine.get_stats_categorical(df_clean[var])
        cats_order = counts_series.index.tolist()

        or_strs = []

        if isinstance(or_calc, dict):
            for cat in cats_order:
                or_strs.append(or_calc.get(str(cat), "-"))

        else:
            # Case: "Simple" style for categorical (returns str) - usually Binary
            if len(cats_order) == 2:
                or_strs.append("-")  # Ref (First level)
                or_strs.append(or_calc)  # Event (Second level)
            else:
                # Fallback for >2 levels if simple passed
                or_strs.append(or_calc)
                for _ in range(len(cats_order) - 1):
                    or_strs.append("")

        return "<br>".join(or_strs)

    def generate(
        self,
        selected_vars: list[str],
        stratify_by: str | None = None,
        or_style: str = "all_levels",
    ) -> str:
        """
        Generate an HTML Table 1 for the given variables, optionally stratified by a grouping variable and with configurable odds-ratio formatting.

        This method computes per-variable overall summaries and, when a stratification variable is provided, per-group summaries, hypothesis tests (p-values and test names), standardized mean differences (for two-group comparisons), and odds ratios. Results are rendered into an HTML table along with missing-data reporting.

        Parameters:
            selected_vars (list[str]): List of column names to include as rows in the table.
            stratify_by (str | None): Column name to stratify (create group columns) by; use None for no stratification.
            or_style (str): Style for categorical odds-ratio output when two groups are present. Supported values:
                - "all_levels": produce ORs for each category versus the reference.
                - "simple": produce a single OR comparing the last category to the reference.

        Returns:
            str: Rendered HTML string containing the Table 1 and missing-data report.
        """
        if stratify_by == "None":
            stratify_by = None
        # 1. Clean Data (Reuse robust logic)
        valid_vars = [c for c in selected_vars if c in self.raw_df.columns]
        df_clean, missing_info = prepare_data_for_analysis(
            self.raw_df,
            numeric_cols=[
                c for c in valid_vars if pd.api.types.is_numeric_dtype(self.raw_df[c])
            ],
            required_cols=list(
                set(list(valid_vars) + ([stratify_by] if stratify_by else []))
            ),
            var_meta=self.var_meta,
            handle_missing="complete-case",
        )

        # 2. Setup Groups
        groups = []
        group_masks = {}
        if stratify_by and stratify_by != "None" and stratify_by in df_clean.columns:
            uniques = df_clean[stratify_by].dropna().unique()
            # Sort naturally
            uniques = sorted(
                uniques,
                key=_numeric_sort_key,
            )
            for u in uniques:
                label = str(u)  # Could map here using var_meta
                groups.append({"val": u, "label": label})
                group_masks[u] = df_clean[stratify_by] == u

        # 3. Analyze Variables
        results = []
        for var in valid_vars:
            if var == stratify_by:
                continue

            # Classify
            vtype = self.classifier.classify(df_clean[var])
            res = VariableAnalysis(
                name=var,
                label=self.var_meta.get(var, {}).get("label", var),
                var_type=vtype,
                stats_overall="-",
            )

            # Stats Logic
            is_cat = vtype == "categorical"

            # Overall Stats
            if is_cat:
                res.stats_overall, _ = self.stats_engine.get_stats_categorical(
                    df_clean[var]
                )
            else:
                is_normal = vtype == "continuous_normal"
                res.stats_overall = self.stats_engine.get_stats_continuous(
                    df_clean[var], is_normal
                )

            # Group Stats
            if groups:
                group_series_list = []
                for g in groups:
                    sub_s = df_clean.loc[group_masks[g["val"]], var]
                    group_series_list.append(sub_s)

                    if is_cat:
                        val, _ = self.stats_engine.get_stats_categorical(
                            sub_s, total_n=len(sub_s)
                        )
                        res.stats_groups[g["val"]] = val
                    else:
                        is_normal = vtype == "continuous_normal"
                        res.stats_groups[g["val"]] = (
                            self.stats_engine.get_stats_continuous(sub_s, is_normal)
                        )

                # Hypothesis Testing
                if is_cat:
                    res.p_value, res.test_name = (
                        self.stats_engine.calculate_p_categorical(
                            df_clean, var, stratify_by
                        )
                    )
                else:
                    is_normal = vtype == "continuous_normal"
                    res.p_value, res.test_name = (
                        self.stats_engine.calculate_p_continuous(
                            group_series_list, is_normal
                        )
                    )

                # SMD (if 2 groups)
                if len(groups) == 2:
                    smd = self.stats_engine.calculate_smd(
                        df_clean,
                        var,
                        stratify_by,
                        groups[0]["val"],
                        groups[1]["val"],
                        is_cat,
                    )
                    res.extra_stats["smd"] = smd

                    # Calculate OR
                    or_calc, or_method = self.stats_engine.calculate_or(
                        df_clean,
                        var,
                        stratify_by,
                        groups[0]["val"],
                        groups[1]["val"],
                        is_cat,
                        or_style=or_style,
                    )
                    res.extra_stats["or"] = or_calc
                    res.or_test_name = or_method

                    if is_cat:
                        res.extra_stats["or"] = self._format_categorical_or(
                            var, or_calc, df_clean
                        )

            results.append(res)

        # 4. Render
        return TableOneFormatter.render_html(
            results, groups, group_masks, len(df_clean), missing_info, self.var_meta
        )
