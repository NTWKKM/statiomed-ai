"""
utils/diagnostic_advanced_lib.py
Advanced Diagnostic Test Analysis & Comparison Library (Shiny Compatible)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def calculate_ci_wilson_score(
    k: float, n: float, ci: float = 0.95, alpha: float | None = None
) -> tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    If n <= 0, returns (NaN, NaN). If `alpha` is provided it overrides `ci` (alpha = 1 - ci).

    Parameters:
        k (float): Number of successes.
        n (float): Number of trials.
        ci (float): Desired confidence level between 0 and 1 (default 0.95).
        alpha (float | None): Optional significance level; when provided it overrides `ci`.

    Returns:
        tuple[float, float]: (lower_bound, upper_bound) bounded to [0, 1]; may be (NaN, NaN) when n <= 0.
    """
    if n <= 0:
        return np.nan, np.nan

    if alpha is None:
        alpha = 1 - ci

    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    term1 = p + z**2 / (2 * n)
    term2 = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return max(0, (term1 - term2) / denom), min(1, (term1 + term2) / denom)


def _get_binary_labels(y_true: np.ndarray, pos_label: int | str) -> np.ndarray:
    """Helper to safely convert y_true to 0/1 binary based on pos_label with type alignment."""
    y_true = np.array(y_true)

    # 1. Direct comparison
    binary = (y_true == pos_label).astype(int)
    if np.sum(binary) > 0:
        return binary

    # 2. Type mismatch fallback
    # If y_true is numeric but pos_label is string -> try converting pos_label
    if np.issubdtype(y_true.dtype, np.number) and isinstance(pos_label, str):
        try:
            pos_num = float(pos_label)
            binary = (y_true == pos_num).astype(int)
            if np.sum(binary) > 0:
                return binary
        except ValueError:
            pass

    # If y_true is string/object but pos_label is numeric -> try converting y_true to numeric?
    # Or convert pos_label to string matches?
    # Usually easier to convert y_true to string for comparison if pos_label is string
    if not np.issubdtype(y_true.dtype, np.number):
        y_str = y_true.astype(str)
        binary = (y_str == str(pos_label)).astype(int)
        if np.sum(binary) > 0:
            return binary

    # Warn if no positives found
    if np.sum(binary) == 0:
        warnings.warn(
            f"No samples matched pos_label={pos_label!r}. Returning all-zeros."
        )

    return binary


class DiagnosticTest:
    """
    Class for analyzing a single diagnostic test with advanced metrics and threshold optimization.
    """

    def __init__(
        self,
        y_true: np.ndarray | pd.Series,
        y_score: np.ndarray | pd.Series,
        pos_label: int | str = 1,
    ):
        # Convert inputs to numpy arrays
        """
        Initialize a DiagnosticTest for a single predictive score and its ground truth.

        Parameters:
            y_true (np.ndarray | pd.Series): Ground-truth labels for each sample.
            y_score (np.ndarray | pd.Series): Continuous scores or probabilities produced by the diagnostic test.
            pos_label (int | str): Value in `y_true` treated as the positive class.


        Description:
            Stores inputs as numpy arrays, converts ground-truth labels to binary 0/1 using `pos_label`, and computes ROC curve (FPR, TPR, thresholds) and AUC when both classes are present. If only a single class exists in `y_true`, `fpr`, `tpr`, and `thresholds` are set to empty arrays and `auc` is set to `NaN`.
        """
        self.y_true = np.array(y_true)
        self.y_score = np.array(y_score)
        self.pos_label = pos_label

        # Ensure binary truth (0/1) using robust helper
        self.y_true_binary = _get_binary_labels(self.y_true, self.pos_label)

        # Basic ROC calculation
        # Only compute if we have both classes
        if len(np.unique(self.y_true_binary)) > 1:
            self.fpr, self.tpr, self.thresholds = roc_curve(
                self.y_true_binary, self.y_score
            )
            self.auc = roc_auc_score(self.y_true_binary, self.y_score)
        else:
            self.fpr, self.tpr, self.thresholds = (
                np.array([]),
                np.array([]),
                np.array([]),
            )
            self.auc = np.nan

    def find_optimal_threshold(self, method: str = "youden") -> tuple[float, int]:
        """
        Selects an optimal decision threshold from the instance's ROC thresholds according to a specified criterion.

        Parameters:
            method (str): Selection method — "youden" (maximizes TPR - FPR), "distance" (minimizes Euclidean distance to point (0,1) on ROC), or "f1" (maximizes F1 score computed at each ROC threshold). Defaults to "youden".

        Returns:
            tuple:
                threshold (float): The chosen threshold value (NaN and -1 index if no thresholds are available).
                index (int): Index of the chosen threshold in the stored threshold array (or -1 if no thresholds).
        """
        if len(self.thresholds) == 0:
            return np.nan, -1

        if method == "youden":
            # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
            idx = np.argmax(self.tpr - self.fpr)

        elif method == "distance":
            # Minimize distance to top-left corner (0,1) on ROC plot
            # dist = sqrt((1-TPR)^2 + FPR^2)
            idx = np.argmin(np.sqrt((1 - self.tpr) ** 2 + self.fpr**2))

        elif method == "f1":
            # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
            # Note: roc_curve thresholds might not align perfectly with precision_recall_curve
            # We calculate F1 for each ROC threshold for consistency
            f1_scores = []
            epsilon = 1e-7
            n_pos = np.sum(self.y_true_binary)
            n_neg = len(self.y_true_binary) - n_pos

            for i, thresh in enumerate(self.thresholds):
                # sensitivity = tpr[i], specificity = 1 - fpr[i]
                tp = self.tpr[i] * n_pos
                fp = self.fpr[i] * n_neg

                precision = tp / (tp + fp + epsilon)
                recall = self.tpr[i]

                f1 = 2 * (precision * recall) / (precision + recall + epsilon)
                f1_scores.append(f1)

            idx = np.argmax(f1_scores)

        else:  # Default to Youden
            idx = np.argmax(self.tpr - self.fpr)

        return self.thresholds[idx], idx

    def get_metrics_at_threshold(self, threshold: float) -> dict[str, float]:
        """
        Compute diagnostic performance metrics and Wilson confidence intervals at a given score threshold.

        Parameters:
            threshold (float): Score cutoff used to binarize predictions (samples with score >= threshold are positive).

        Raises:
            ValueError: If the ground truth contains fewer than two classes (both positive and negative samples required).

        Returns:
            dict[str, float]: A dictionary containing:
                - "threshold": the provided threshold.
                - "auc": AUC of the test (NaN if not available).
                - "sensitivity": true positive rate.
                - "sensitivity_ci_lower", "sensitivity_ci_upper": Wilson CI bounds for sensitivity.
                - "specificity": true negative rate.
                - "specificity_ci_lower", "specificity_ci_upper": Wilson CI bounds for specificity.
                - "ppv": positive predictive value (precision).
                - "ppv_ci_lower", "ppv_ci_upper": Wilson CI bounds for PPV.
                - "npv": negative predictive value.
                - "npv_ci_lower", "npv_ci_upper": Wilson CI bounds for NPV.
                - "accuracy": overall accuracy.
                - "accuracy_ci_lower", "accuracy_ci_upper": Wilson CI bounds for accuracy.
                - "f1_score": F1 score at the threshold.
                - "lr_plus": positive likelihood ratio (LR+), NaN if undefined.
                - "lr_minus": negative likelihood ratio (LR-), NaN if undefined.
                - "dor": diagnostic odds ratio, NaN if undefined.
                - "tp", "tn", "fp", "fn": confusion matrix counts.
        """
        if len(np.unique(self.y_true_binary)) < 2:
            raise ValueError(
                "get_metrics_at_threshold requires both positive and negative samples."
            )
        y_pred = (self.y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(self.y_true_binary, y_pred).ravel()

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Likelihood Ratios
        lr_plus = sensitivity / (1 - specificity) if (1 - specificity) > 0 else np.nan
        lr_minus = (1 - sensitivity) / specificity if specificity > 0 else np.nan

        # Diagnostic Odds Ratio
        dor = (
            (lr_plus / lr_minus)
            if (not np.isnan(lr_minus) and lr_minus > 0)
            else np.nan
        )

        # Calculate CIs
        sens_ci = calculate_ci_wilson_score(tp, tp + fn)
        spec_ci = calculate_ci_wilson_score(tn, tn + fp)
        ppv_ci = calculate_ci_wilson_score(tp, tp + fp)
        npv_ci = calculate_ci_wilson_score(tn, tn + fn)
        acc_ci = calculate_ci_wilson_score(tp + tn, tp + tn + fp + fn)

        return {
            "threshold": threshold,
            "auc": self.auc,
            "sensitivity": sensitivity,
            "sensitivity_ci_lower": sens_ci[0],
            "sensitivity_ci_upper": sens_ci[1],
            "specificity": specificity,
            "specificity_ci_lower": spec_ci[0],
            "specificity_ci_upper": spec_ci[1],
            "ppv": ppv,
            "ppv_ci_lower": ppv_ci[0],
            "ppv_ci_upper": ppv_ci[1],
            "npv": npv,
            "npv_ci_lower": npv_ci[0],
            "npv_ci_upper": npv_ci[1],
            "accuracy": accuracy,
            "accuracy_ci_lower": acc_ci[0],
            "accuracy_ci_upper": acc_ci[1],
            "f1_score": f1,
            "lr_plus": lr_plus,
            "lr_minus": lr_minus,
            "dor": dor,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }


class DiagnosticComparison:
    """
    Class for comparing two diagnostic tests using Paired DeLong Test.
    """

    @staticmethod
    def _delong_covariance(
        y_true: np.ndarray, score1: np.ndarray, score2: np.ndarray
    ) -> float:
        """
        Compute the DeLong covariance component between two AUC estimates for paired ROC curves.

        Parameters:
            y_true (np.ndarray): Binary ground-truth labels (0 for negative, 1 for positive).
            score1 (np.ndarray): Continuous scores or prediction probabilities for the first classifier, aligned with y_true.
            score2 (np.ndarray): Continuous scores or prediction probabilities for the second classifier, aligned with y_true.

        Returns:
            float: Covariance between the two AUC estimates as computed by DeLong's method.

        Raises:
            ValueError: If y_true does not contain both positive and negative samples.
        """
        n_pos = np.sum(y_true == 1)
        n_neg = len(y_true) - n_pos

        if n_pos < 2 or n_neg < 2:
            raise ValueError(
                "DeLong covariance requires at least 2 positives and 2 negatives."
            )

        # Indices
        pos_indices = np.where(y_true == 1)[0]
        neg_indices = np.where(y_true == 0)[0]

        # V10 calculation (Placement values for positives)
        def compute_v10(score):
            # Compare every positive score against every negative score
            # Matrix broadcasting: (n_pos, 1) > (1, n_neg) -> (n_pos, n_neg)
            """
            Compute per-positive placement values against all negatives for a score vector.

            Parameters:
                score (array-like): Scores for all samples; comparisons use `pos_indices` and `neg_indices` from the enclosing scope.

            Returns:
                numpy.ndarray: 1-D array of length equal to the number of positive samples where each element is the average over negatives of: `1.0` if positive score > negative score, `0.5` if equal, and `0.0` if less (values in [0.0, 1.0]).
            """
            pos_scores = score[pos_indices]
            neg_scores = score[neg_indices]

            # 1.0 if pos > neg, 0.5 if pos == neg, 0 if pos < neg
            res = (pos_scores[:, np.newaxis] > neg_scores).astype(float)
            res += 0.5 * (pos_scores[:, np.newaxis] == neg_scores).astype(float)

            # Mean over negative samples for each positive sample
            return res.mean(axis=1)

        # V01 calculation (Placement values for negatives)
        def compute_v01(score):
            """
            Compute average placement values for each negative sample by comparing positive scores to that negative.

            Parameters:
                score (array-like): Array of scores for all samples; indices referenced by the outer-scope
                    variables `pos_indices` and `neg_indices`.

            Returns:
                numpy.ndarray: Array of placement values with shape (n_neg,), where each element is the mean
                across positive samples of: 1.0 if positive score > negative score, 0.5 if equal, and 0.0 otherwise.
            """
            pos_scores = score[pos_indices]
            neg_scores = score[neg_indices]

            # 1.0 if pos > neg ... same logic but mean over positive samples
            res = (pos_scores[:, np.newaxis] > neg_scores).astype(float)
            res += 0.5 * (pos_scores[:, np.newaxis] == neg_scores).astype(float)

            # Mean over positive samples for each negative sample
            return res.mean(axis=0)

        v10_1 = compute_v10(score1)
        v10_2 = compute_v10(score2)

        v01_1 = compute_v01(score1)
        v01_2 = compute_v01(score2)

        # Covariance components
        s10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
        s01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]

        covariance = (s10 / n_pos) + (s01 / n_neg)
        return covariance

    @staticmethod
    def delong_paired_test(
        y_true: np.ndarray | pd.Series,
        score1: np.ndarray | pd.Series,
        score2: np.ndarray | pd.Series,
        pos_label: int | str = 1,
        ci: float = 0.95,
    ) -> dict:
        """
        Compare two correlated ROC AUCs using DeLong's paired test.

        Parameters:
            y_true (np.ndarray | pd.Series): True labels.
            score1 (np.ndarray | pd.Series): Scores/probabilities from the first classifier.
            score2 (np.ndarray | pd.Series): Scores/probabilities from the second classifier.
            pos_label (int | str): Value in `y_true` treated as the positive class.
            ci (float): Confidence level for the returned confidence interval (e.g., 0.95).

        Returns:
            result (dict): Dictionary with the following keys:
                auc1 (float): AUC of `score1` computed on `y_true`.
                auc2 (float): AUC of `score2` computed on `y_true`.
                diff (float): Difference `auc1 - auc2`.
                z_score (float): Z statistic for the paired DeLong test.
                p_value (float): Two-tailed p-value for the Z statistic.
                ci_lower (float): Lower bound of the `ci` confidence interval for the AUC difference.
                ci_upper (float): Upper bound of the `ci` confidence interval for the AUC difference.
                se_diff (float): Standard error of the AUC difference.
        """
        # Prepare data
        y_true = np.array(y_true)
        y_true_binary = _get_binary_labels(y_true, pos_label)
        s1 = np.array(score1)
        s2 = np.array(score2)

        # Calculate individual AUCs and Variances (using DeLong)
        # We can reuse the covariance logic: Var(AUC) = Cov(AUC, AUC)

        # 1. Variances
        var_auc1 = DiagnosticComparison._delong_covariance(y_true_binary, s1, s1)
        var_auc2 = DiagnosticComparison._delong_covariance(y_true_binary, s2, s2)

        # 2. Covariance
        cov_auc1_auc2 = DiagnosticComparison._delong_covariance(y_true_binary, s1, s2)

        # 3. AUCs
        auc1 = roc_auc_score(y_true_binary, s1)
        auc2 = roc_auc_score(y_true_binary, s2)

        # 4. Z-test
        auc_diff = auc1 - auc2
        var_diff = var_auc1 + var_auc2 - 2 * cov_auc1_auc2

        # Avoid division by zero
        if var_diff <= 0:
            z_score = 0
            p_value = 1.0
            se_diff = 0
        else:
            se_diff = np.sqrt(var_diff)
            z_score = auc_diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed

        z_crit = stats.norm.ppf(1 - (1 - ci) / 2)
        # 95% CI of the difference
        ci_lower = auc_diff - z_crit * se_diff
        ci_upper = auc_diff + z_crit * se_diff

        return {
            "auc1": auc1,
            "auc2": auc2,
            "diff": auc_diff,
            "z_score": z_score,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "se_diff": se_diff,
        }
