from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from config import CONFIG
from tabs._common import get_color_palette
from utils.data_cleaning import prepare_data_for_analysis

COLORS = get_color_palette()


class AgreementAnalysis:
    """
    Comprehensive inter-rater/inter-method agreement analysis.
    Includes: Cohen's Kappa, Fleiss' Kappa, ICC, and Advanced Bland-Altman.
    """

    @staticmethod
    def cohens_kappa(
        df: pd.DataFrame,
        rater1: str,
        rater2: str,
        weights: str | None = None,
        ci: float = 0.95,
    ) -> tuple[pd.DataFrame, str | None, pd.DataFrame, dict]:
        """
        Compute Cohen's Kappa for two categorical raters, provide an approximate confidence interval, and return a labeled confusion matrix and missing-data info.

        Parameters:
            df (pd.DataFrame): Input dataframe containing rater columns.
            rater1 (str): Column name for the first rater.
            rater2 (str): Column name for the second rater.
            weights (str | None): Weighting scheme for kappa (e.g., "linear", "quadratic") or None for unweighted.
            ci (float): Confidence level for the interval (e.g., 0.95 for a 95% CI).

        Returns:
            tuple:
                stats_df (pd.DataFrame): Summary metrics including Cohen's Kappa, sample size (N), standard error, and CI bounds.
                error (str | None): Error message if computation failed, otherwise None.
                confusion_matrix_df (pd.DataFrame): Confusion matrix with labeled rows/columns for each rater category.
                missing_info (dict): Metadata about missing-data handling applied (includes chosen strategy and any details from data preparation).
        """
        try:
            # Data Cleaning via central pipeline
            missing_cfg = CONFIG.get("analysis.missing", {}) or {}
            strategy = missing_cfg.get("strategy", "complete-case")
            missing_codes = missing_cfg.get("user_defined_values", [])

            clean_df, missing_info = prepare_data_for_analysis(
                df,
                required_cols=[rater1, rater2],
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy

            if clean_df.empty:
                return (
                    pd.DataFrame(),
                    "No valid data pairs found after cleaning.",
                    pd.DataFrame(),
                    missing_info,
                )

            y1, y2 = clean_df[rater1], clean_df[rater2]

            # Kappa Calculation
            kappa = cohen_kappa_score(y1, y2, weights=weights)

            # Standard Error Calculation (Approximate)
            n = len(clean_df)

            # Use analytical variance if possible (not standard in sklearn)
            # We will use a basic CI based on asymptotic standard error for H0=0 for significance,
            # but for the CI around the estimate, we assume normality.
            z_score = stats.norm.ppf(1 - (1 - ci) / 2)

            # Better SE approximation for CI construction
            # Using simple asymptotic standard error for binary/nominal case
            p_o = np.sum(y1 == y2) / n
            pe = 0  # Calculate expected chance agreement manually to get correct SE
            labels = np.unique(np.concatenate([y1, y2]))
            for k in labels:
                p1 = np.sum(y1 == k) / n
                p2 = np.sum(y2 == k) / n
                pe += p1 * p2

            se_kappa = (
                np.sqrt((p_o * (1 - p_o)) / (n * (1 - pe) ** 2)) if (1 - pe) != 0 else 0
            )

            # For weighted kappa, SE is more complex; falling back to simple approximation if needed
            if weights:
                # Placeholder for complex weighted SE - often approximated
                # Using a very simplified heuristic if se_kappa is 0 or invalid
                if se_kappa == 0:
                    se_kappa = np.sqrt((kappa * (1 - kappa)) / n)

            lower = kappa - z_score * se_kappa
            upper = kappa + z_score * se_kappa

            # Results
            stats_data = {
                "Metric": [
                    "Cohen's Kappa",
                    "Sample Size (N)",
                    "Standard Error",
                    f"{int(ci * 100)}% CI Lower",
                    f"{int(ci * 100)}% CI Upper",
                ],
                "Value": [kappa, n, se_kappa, lower, upper],
            }

            # Confusion Matrix
            cm = confusion_matrix(y1, y2, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=[f"{rater1}: {lbl}" for lbl in labels],
                columns=[f"{rater2}: {lbl}" for lbl in labels],
            )

            return pd.DataFrame(stats_data), None, cm_df, missing_info

        except Exception as e:
            return pd.DataFrame(), str(e), pd.DataFrame(), {}

    @staticmethod
    def fleiss_kappa(
        df: pd.DataFrame, raters: list[str]
    ) -> tuple[pd.DataFrame, str | None, dict]:
        """
        Calculate Fleiss' Kappa for multiple raters.
        """
        try:
            # Data Cleaning via central pipeline
            missing_cfg = CONFIG.get("analysis.missing", {}) or {}
            strategy = missing_cfg.get("strategy", "complete-case")
            missing_codes = missing_cfg.get("user_defined_values", [])

            clean_df, missing_info = prepare_data_for_analysis(
                df,
                required_cols=raters,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy

            if clean_df.empty:
                return (
                    pd.DataFrame(),
                    "No valid data found after cleaning.",
                    missing_info,
                )

            # Convert to Subject x Category counts format
            # aggregate_raters expects (Subject, Rater) but we have columns as raters.
            # We need to ensure we are passing the correct format to aggregate_raters.
            # statsmodels aggregate_raters takes a 2d array (n_subjects, n_raters) containing category labels.

            agg_data, categories = aggregate_raters(clean_df.values)

            kappa = fleiss_kappa(agg_data)

            stats_data = {
                "Metric": ["Fleiss' Kappa", "Subjects (N)", "Raters (k)", "Categories"],
                "Value": [kappa, len(clean_df), len(raters), len(categories)],
            }

            return pd.DataFrame(stats_data), None, missing_info

        except Exception as e:
            return pd.DataFrame(), str(e), {}

    @staticmethod
    def bland_altman_advanced(
        df: pd.DataFrame,
        method1: str,
        method2: str,
        ci: float = 0.95,
        show_ci_bands: bool = True,
    ) -> tuple[dict, go.Figure, dict]:
        """
        Generate Bland–Altman summary statistics and a Plotly Bland–Altman figure comparing two measurement methods.

        Parameters:
            df (pd.DataFrame): Input data containing the two measurement columns.
            method1 (str): Column name for the first measurement method (will be subtracted by method2).
            method2 (str): Column name for the second measurement method.
            ci (float): Confidence level for confidence intervals (e.g., 0.95 for 95% CI).
            show_ci_bands (bool): If True, include shaded confidence-interval bands for the mean difference and limits of agreement in the figure.

        Returns:
            tuple:
                stats_res (dict): Summary statistics including:
                    - "n": sample size (int)
                    - "mean_diff": mean of pairwise differences
                    - "ci_mean_diff": [lower, upper] CI for the mean difference
                    - "sd_diff": standard deviation of differences
                    - "upper_loa": upper limit of agreement (mean_diff + 1.96*sd_diff)
                    - "lower_loa": lower limit of agreement (mean_diff - 1.96*sd_diff)
                    - "ci_upper_loa": [lower, upper] CI for the upper LoA
                    - "ci_lower_loa": [lower, upper] CI for the lower LoA
                  If an error occurs, a dict with key "error" and a string message is returned instead of the stats dictionary.
                fig (plotly.graph_objects.Figure): Bland–Altman plot with points, mean line, LoA lines, and optional CI bands.
                missing_info (dict): Information about how missing data were handled and any related metadata from the cleaning pipeline.
        """
        try:
            # Data Cleaning via central pipeline
            missing_cfg = CONFIG.get("analysis.missing", {}) or {}
            strategy = missing_cfg.get("strategy", "complete-case")
            missing_codes = missing_cfg.get("user_defined_values", [])

            clean_df, missing_info = prepare_data_for_analysis(
                df,
                required_cols=[method1, method2],
                numeric_cols=[method1, method2],
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy

            if len(clean_df) < 2:
                return (
                    {"error": "Insufficient data (N < 2) after cleaning"},
                    go.Figure(),
                    missing_info,
                )

            m1 = clean_df[method1].values
            m2 = clean_df[method2].values

            diffs = m1 - m2
            means = (m1 + m2) / 2
            n = len(diffs)

            mean_diff = np.mean(diffs)
            sd_diff = np.std(diffs, ddof=1)
            # SE of the mean difference
            se_mean_diff = sd_diff / np.sqrt(n)

            # Limits of Agreement (1.96 * SD)
            # For 95% LoA, we use +/- 1.96 * SD
            loa_range = 1.96 * sd_diff
            upper_loa = mean_diff + loa_range
            lower_loa = mean_diff - loa_range

            # Confidence Intervals (Carkeet, 2015 / Bland & Altman 1999)
            # t-score for CI construction
            t_crit = stats.t.ppf(1 - (1 - ci) / 2, df=n - 1)

            # CI for Mean Difference (Bias)
            ci_md_low = mean_diff - t_crit * se_mean_diff
            ci_md_high = mean_diff + t_crit * se_mean_diff

            # Standard Error for Limits of Agreement
            # Approx SE for LoA limits = sqrt(3 * SD^2 / n)
            se_loa = np.sqrt(3 * sd_diff**2 / n)

            ci_upper_loa_low = upper_loa - t_crit * se_loa
            ci_upper_loa_high = upper_loa + t_crit * se_loa
            ci_lower_loa_low = lower_loa - t_crit * se_loa
            ci_lower_loa_high = lower_loa + t_crit * se_loa

            # --- Plotting ---
            fig = go.Figure()

            # Scatter points
            fig.add_trace(
                go.Scatter(
                    x=means,
                    y=diffs,
                    mode="markers",
                    marker=dict(color="rgba(0,0,0,0.5)", size=6),
                    name="Difference",
                    hovertemplate="Mean: %{x:.2f}<br>Diff: %{y:.2f}<extra></extra>",
                )
            )

            # Mean Difference Line
            fig.add_hline(
                y=mean_diff,
                line_dash="solid",
                line_color=COLORS["primary"],
                annotation_text=f"Mean Diff: {mean_diff:.2f}",
                annotation_position="top right",
            )

            # LoA Lines
            fig.add_hline(
                y=upper_loa,
                line_dash="dash",
                line_color=COLORS["danger"],
                annotation_text=f"+1.96 SD: {upper_loa:.2f}",
                annotation_position="top right",
            )
            fig.add_hline(
                y=lower_loa,
                line_dash="dash",
                line_color=COLORS["danger"],
                annotation_text=f"-1.96 SD: {lower_loa:.2f}",
                annotation_position="bottom right",
            )

            # CI Bands (Optional)
            if show_ci_bands:
                # CI for Mean Diff
                fig.add_hrect(
                    y0=ci_md_low,
                    y1=ci_md_high,
                    line_width=0,
                    fillcolor=COLORS["primary"],
                    opacity=0.15,
                    annotation_text="CI Mean",
                    annotation_position="left",
                )

                # CI for Upper LoA
                fig.add_hrect(
                    y0=ci_upper_loa_low,
                    y1=ci_upper_loa_high,
                    line_width=0,
                    fillcolor=COLORS["danger"],
                    opacity=0.1,
                    annotation_text="CI Upper LoA",
                    annotation_position="left",
                )

                # CI for Lower LoA
                fig.add_hrect(
                    y0=ci_lower_loa_low,
                    y1=ci_lower_loa_high,
                    line_width=0,
                    fillcolor=COLORS["danger"],
                    opacity=0.1,
                    annotation_text="CI Lower LoA",
                    annotation_position="left",
                )

            fig.update_layout(
                title=f"Bland-Altman Plot: {method1} vs {method2}",
                xaxis_title=f"Mean of {method1} & {method2}",
                yaxis_title=f"Difference ({method1} - {method2})",
                template="plotly_white",
                height=500,
                hovermode="closest",
            )

            stats_res = {
                "n": n,
                "mean_diff": mean_diff,
                "ci_mean_diff": [ci_md_low, ci_md_high],
                "sd_diff": sd_diff,
                "upper_loa": upper_loa,
                "lower_loa": lower_loa,
                "ci_upper_loa": [ci_upper_loa_low, ci_upper_loa_high],
                "ci_lower_loa": [ci_lower_loa_low, ci_lower_loa_high],
            }

            return stats_res, fig, missing_info

        except Exception as e:
            return {"error": str(e)}, go.Figure(), {}

    @staticmethod
    def icc(
        df: pd.DataFrame,
        cols: list[str],
        icc_type: str = "ICC2k",  # Default to Two-way random, average measures
    ) -> tuple[pd.DataFrame, str | None, dict, dict]:
        """
        Compute the Intraclass Correlation Coefficient (ICC) for ratings provided in wide format.

        Parameters:
            df (pd.DataFrame): Input data with subjects as rows and raters as columns.
            cols (list[str]): Column names to use as rater measurements; must contain at least two columns.
            icc_type (str): ICC definition to select from Pingouin's output (e.g., "ICC2k").

        Returns:
            tuple:
                - icc_res (pd.DataFrame): ICC results filtered to the requested `icc_type` and augmented with a `Strength` column categorizing the ICC value as "Poor", "Fair", "Good", or "Excellent".
                - error (str | None): Error message when computation fails or selection is invalid, otherwise `None`.
                - (dict): Placeholder empty dict for backward-compatible return structure.
                - missing_info (dict): Information about how missing data were handled and any related metadata.
        """
        if len(cols) < 2:
            return pd.DataFrame(), "Need at least 2 columns for ICC", {}, {}

        try:
            # Data Cleaning via central pipeline
            missing_cfg = CONFIG.get("analysis.missing", {}) or {}
            strategy = missing_cfg.get("strategy", "complete-case")
            missing_codes = missing_cfg.get("user_defined_values", [])

            clean_df, missing_info = prepare_data_for_analysis(
                df,
                required_cols=cols,
                numeric_cols=cols,
                missing_codes=missing_codes,
                handle_missing=strategy,
            )
            missing_info["strategy"] = strategy

            if clean_df.empty:
                return (
                    pd.DataFrame(),
                    "No valid data found after cleaning.",
                    {},
                    missing_info,
                )

            # Pingouin expects Long format: [Subject, Rater, Rating]
            # Add Subject ID
            clean_df = clean_df.copy()
            clean_df["Subject_ID"] = range(len(clean_df))

            # Melt to Long format
            df_long = clean_df.melt(
                id_vars=["Subject_ID"],
                value_vars=cols,
                var_name="Rater",
                value_name="Rating",
            )

            # Calculate ICC
            icc_res = pg.intraclass_corr(
                data=df_long, targets="Subject_ID", raters="Rater", ratings="Rating"
            )

            if "Type" in icc_res.columns:
                icc_res = icc_res[icc_res["Type"] == icc_type].copy()
                if icc_res.empty:
                    return (
                        pd.DataFrame(),
                        f"ICC type '{icc_type}' not found.",
                        {},
                        missing_info,
                    )
            else:
                return (
                    pd.DataFrame(),
                    "ICC output missing 'Type' column.",
                    {},
                    missing_info,
                )

            # Add interpretation helper
            def interpret_icc(v):
                """
                Map an intraclass correlation coefficient (ICC) value to a qualitative strength category.

                Parameters:
                    v (float): ICC value (can be any real number; typical range is 0 to 1).

                Returns:
                    str: One of "Poor" (v < 0.40), "Fair" (0.40 <= v < 0.60), "Good" (0.60 <= v < 0.75), or "Excellent" (v >= 0.75).
                """
                if v < 0.40:
                    return "Poor"
                if v < 0.60:
                    return "Fair"
                if v < 0.75:
                    return "Good"
                return "Excellent"

            icc_res["Strength"] = icc_res["ICC"].apply(interpret_icc)

            return icc_res, None, {}, missing_info

        except Exception as e:
            return pd.DataFrame(), str(e), {}, {}
