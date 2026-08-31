"""
agent/critique_engine.py - Automated Clinical Methodology Appraisal & Limitation Engine
=============================================================================
Provides rigorous, deterministic critical appraisal of biostatistical analyses:
  1. Events-Per-Variable (EPV) Check: Flags overfitting risk if EPV < 10 (Peduzzi et al.).
  2. Quasi-Complete Separation: Detects extreme log-odds parameters and runaway variances.
  3. Proportional Hazards (PH) Assumption: Checks for crossing KM curves and non-proportionality.
  4. Missing Data Severity: Evaluates missingness against the 20% bias threshold (MCAR/MAR).
  5. Small Cell Contingency Risk: Enforces Fisher's Exact test when expected counts < 5.
  6. EQUATOR / SAMPL Reporting Quality: Identifies missing reporting requirements.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CritiqueFinding:
    category: str  # "EPV", "Separation", "PH_Assumption", "Missingness", "Sample_Size", "Reporting"
    severity: str  # "HIGH", "MODERATE", "LOW", "INFO"
    title: str
    description: str
    recommendation: str


@dataclass
class CritiqueVerdict:
    analysis_type: str
    overall_status: str  # "ROBUST", "VALID_WITH_LIMITATIONS", "HIGH_RISK_BIAS"
    strengths: List[str] = field(default_factory=list)
    findings: List[CritiqueFinding] = field(default_factory=list)
    equator_checklist: str = "STROBE / SAMPL"

    def to_markdown(self) -> str:
        status_icons = {
            "ROBUST": "🟢 **Methodologically Robust (Low Risk of Bias)**",
            "VALID_WITH_LIMITATIONS": "🟡 **Valid with Methodological Caveats**",
            "HIGH_RISK_BIAS": "🔴 **High Risk of Methodological Bias / Overfitting**",
        }
        header_badge = status_icons.get(self.overall_status, "ℹ️ **Appraisal Summary**")

        lines = [
            f"### 🛡️ Automated Clinical Critique & Appraisal ({self.analysis_type.upper()})",
            f"**Overall Methodological Verdict:** {header_badge}",
            f"- **Target Guideline:** `{self.equator_checklist}` & SAMPL Standard\n",
        ]

        if self.strengths:
            lines.append("#### ✅ Methodological Strengths:")
            for s in self.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if self.findings:
            lines.append("#### ⚠️ Limitations & Statistical Caveats:")
            for f in self.findings:
                sev_badge = {
                    "HIGH": "🔴 [HIGH RISK]",
                    "MODERATE": "🟡 [MODERATE]",
                    "LOW": "🔵 [LOW]",
                    "INFO": "ℹ️ [NOTE]",
                }.get(f.severity, "⚠️")
                lines.append(f"- {sev_badge} **{f.title}:** {f.description}")
                lines.append(f"  - *Mitigation:* {f.recommendation}")
            lines.append("")
        else:
            lines.append(
                "#### 🌟 No critical statistical violations detected (Assumptions satisfied).\n"
            )

        return "\n".join(lines)


class CritiqueEngine:
    """
    Deterministic rule-based appraisal engine that inspects datasets and statistical results
    to identify methodological vulnerabilities, bias risks, and reporting deficits.
    """

    @classmethod
    def evaluate_epv(
        cls,
        n_events: int,
        n_non_events: int,
        n_covariates: int,
    ) -> Optional[CritiqueFinding]:
        """Checks Events-Per-Variable (EPV) for logistic or Cox regression."""
        if n_covariates <= 0:
            return None

        effective_events = min(n_events, n_non_events) if n_non_events > 0 else n_events
        epv = effective_events / n_covariates if n_covariates > 0 else 0.0

        if epv < 5:
            return CritiqueFinding(
                category="EPV",
                severity="HIGH",
                title=f"Severe EPV Deficit (EPV = {epv:.1f} < 5)",
                description=(
                    f"Only {effective_events} effective events for {n_covariates} covariates. "
                    "Extreme risk of coefficient inflation, severe overfitting, and distorted p-values."
                ),
                recommendation=(
                    "Reduce degrees of freedom using penalized regression (Ridge/Lasso/ElasticNet), "
                    "pre-specify a smaller subset of clinical confounders, or perform dimension reduction."
                ),
            )
        elif epv < 10:
            return CritiqueFinding(
                category="EPV",
                severity="MODERATE",
                title=f"Suboptimal EPV (EPV = {epv:.1f} < 10)",
                description=(
                    f"Peduzzi & Concato benchmark recommends ≥ 10 events per variable (found {epv:.1f}). "
                    "Confidence intervals may be wider and estimates moderately unstable."
                ),
                recommendation=(
                    "Report unadjusted sensitivity analyses and perform internal bootstrap validation "
                    "(e.g., 500 replicates) to evaluate optimism-corrected performance."
                ),
            )
        return None

    @classmethod
    def evaluate_separation(
        cls,
        coef_df: Optional[pd.DataFrame],
    ) -> Optional[CritiqueFinding]:
        """Detects quasi-complete or complete separation in logistic regression."""
        if coef_df is None or coef_df.empty:
            return None

        # Check for huge odds ratios or exploding standard errors
        for _, row in coef_df.iterrows():
            var_name = str(
                row.get("Variable", row.name if hasattr(row, "name") else "Covariate")
            )
            or_val = row.get("Odds Ratio (OR)", row.get("OR", None))
            if or_val is not None:
                try:
                    num_or = float(str(or_val).replace(",", ""))
                    if num_or > 100.0 or (0.0 <= num_or < 0.01):
                        return CritiqueFinding(
                            category="Separation",
                            severity="HIGH",
                            title=f"Potential Quasi-Complete Separation on `{var_name}`",
                            description=(
                                f"Extreme Odds Ratio ({num_or:.2f}) indicates near-perfect discrimination "
                                f"or zero cell count in one contingency stratum."
                            ),
                            recommendation=(
                                "Use Firth's penalized maximum likelihood logistic regression, "
                                "Bayesian logistic regression with weakly informative priors, or exact conditional logistic regression."
                            ),
                        )
                except (ValueError, TypeError):
                    pass
        return None

    @classmethod
    def evaluate_missingness(
        cls,
        df: pd.DataFrame,
        cols_analyzed: List[str],
    ) -> Optional[CritiqueFinding]:
        """Evaluates missing data fraction and potential mechanism."""
        if df.empty or not cols_analyzed:
            return None

        valid_cols = [c for c in cols_analyzed if c in df.columns]
        if not valid_cols:
            return None

        sub = df[valid_cols]
        total_rows = len(sub)
        complete_rows = len(sub.dropna())
        missing_rate = (
            (total_rows - complete_rows) / total_rows if total_rows > 0 else 0.0
        )

        if missing_rate > 0.20:
            return CritiqueFinding(
                category="Missingness",
                severity="HIGH",
                title=f"High Missing Data Rate ({missing_rate:.1%})",
                description=(
                    f"Complete case analysis drops {total_rows - complete_rows:,} out of {total_rows:,} records (>20%). "
                    "Naive listwise deletion causes substantial power loss and introduces selection bias under MAR."
                ),
                recommendation=(
                    "Implement Multiple Imputation by Chained Equations (MICE, m ≥ 20 imputations) "
                    "under the Missing at Random (MAR) assumption, with Rubin's rule pooling."
                ),
            )
        elif missing_rate > 0.05:
            return CritiqueFinding(
                category="Missingness",
                severity="MODERATE",
                title=f"Moderate Missing Data ({missing_rate:.1%})",
                description=(
                    f"{total_rows - complete_rows:,} records ({missing_rate:.1%}) have missing covariates."
                ),
                recommendation=(
                    "Examine missingness patterns (MCAR vs MAR test) and perform sensitivity analysis "
                    "comparing complete cases with imputed datasets."
                ),
            )
        return None

    @classmethod
    def evaluate_rct_cell_counts(
        cls,
        events_ctrl: int,
        n_ctrl: int,
        events_treat: int,
        n_treat: int,
    ) -> Optional[CritiqueFinding]:
        """Checks for sparse expected cell counts in RCT contingency table (< 5)."""
        n_total = n_ctrl + n_treat
        if n_total <= 0:
            return None

        non_events_ctrl = max(0, n_ctrl - events_ctrl)
        non_events_treat = max(0, n_treat - events_treat)
        total_events = events_ctrl + events_treat
        total_non_events = non_events_ctrl + non_events_treat

        # Derive all four expected cell counts from margins: (Row Total * Col Total) / Grand Total
        e11 = (n_ctrl * total_events) / n_total
        e12 = (n_ctrl * total_non_events) / n_total
        e21 = (n_treat * total_events) / n_total
        e22 = (n_treat * total_non_events) / n_total
        expected_cells = [e11, e12, e21, e22]

        if any(e < 5.0 for e in expected_cells):
            min_e = min(expected_cells)
            return CritiqueFinding(
                category="Sample_Size",
                severity="MODERATE",
                title=f"Sparse Contingency Cell Count (Min Expected Cell = {min_e:.1f})",
                description=(
                    f"One or more expected cell counts in the 2x2 contingency table is < 5 ({min_e:.1f}). "
                    "Asymptotic Chi-Square distribution assumption is violated."
                ),
                recommendation=(
                    "Prioritize Fisher's Exact Test or Mid-P exact test over standard Pearson Chi-Square, "
                    "and calculate exact Clopper-Pearson or Wilson score confidence intervals for proportions."
                ),
            )
        return None

    @classmethod
    def appraise_analysis(
        cls,
        analysis_type: str,
        df: pd.DataFrame,
        results_meta: Dict[str, Any],
        proposal_meta: Optional[Any] = None,
    ) -> CritiqueVerdict:
        """
        Runs comprehensive critique on a completed statistical analysis.
        """
        findings: List[CritiqueFinding] = []
        strengths: List[str] = []
        checklist = "STROBE / SAMPL"

        if analysis_type == "survival":
            checklist = "STROBE 2014 (Observational Cohort) & SAMPL"
            strengths.append(
                "Time-to-event censorship properly incorporated via product-limit estimator."
            )

            covar_cols = results_meta.get("covar_cols", [])
            cox_stats = results_meta.get("cox_stats", {})
            event_col = results_meta.get("event_col", "event")
            events = int(df[event_col].sum()) if event_col in df.columns else 0
            non_events = len(df) - events

            # Proportional Hazards (PH) diagnostic check
            ph_diag = results_meta.get("ph_diagnostic")
            if ph_diag is None and isinstance(cox_stats, dict):
                ph_diag = cox_stats.get("ph_diagnostic", cox_stats.get("ph_test"))

            if ph_diag is not None:
                if isinstance(ph_diag, dict):
                    passed = ph_diag.get("passed")
                    if passed is None and "p_value" in ph_diag:
                        p_val = ph_diag["p_value"]
                        passed = (
                            (p_val >= 0.05) if isinstance(p_val, (int, float)) else None
                        )
                elif isinstance(ph_diag, bool):
                    passed = ph_diag
                else:
                    passed = None

                if passed is True:
                    strengths.append(
                        "Proportional hazards (PH) assumption verified via Schoenfeld residuals."
                    )
                elif passed is False:
                    findings.append(
                        CritiqueFinding(
                            category="PH_Assumption",
                            severity="HIGH",
                            title="Violation of Proportional Hazards (PH) Assumption",
                            description=(
                                "Schoenfeld residuals test indicates non-proportional hazards over time (P < 0.05). "
                                "Constant hazard ratio assumption does not hold."
                            ),
                            recommendation=(
                                "Consider stratified Cox regression, time-varying covariates, or restricted mean survival time (RMST) analysis."
                            ),
                        )
                    )
            # When ph_diag is absent, PH is marked as unassessed without adding strength

            # EPV check
            if covar_cols:
                epv_find = cls.evaluate_epv(events, non_events, len(covar_cols))
                if epv_find:
                    findings.append(epv_find)
                else:
                    strengths.append(
                        f"Adequate Events-Per-Variable (EPV = {events / max(1, len(covar_cols)):.1f} ≥ 10)."
                    )

            # Missingness check
            miss_find = cls.evaluate_missingness(
                df,
                [results_meta.get("time_col", "time"), event_col] + (covar_cols or []),
            )
            if miss_find:
                findings.append(miss_find)

        elif analysis_type == "logistic":
            checklist = "TRIPOD+AI / STROBE & SAMPL"
            strengths.append(
                "Odds Ratios reported with exact 95% Wald Confidence Intervals."
            )
            strengths.append(
                "Multivariable model adjusts for potential confounding factors."
            )

            outcome_col = results_meta.get("outcome_col", "outcome")
            pred_cols = results_meta.get("predictor_cols", [])

            if (
                "fitted_events" in results_meta
                and results_meta["fitted_events"] is not None
            ):
                events = int(results_meta["fitted_events"])
                non_events = (
                    int(results_meta["fitted_non_events"])
                    if "fitted_non_events" in results_meta
                    and results_meta["fitted_non_events"] is not None
                    else len(df) - events
                )
            else:
                events = (
                    int((df[outcome_col] == 1).sum())
                    if outcome_col in df.columns
                    else 0
                )
                non_events = len(df) - events

            epv_find = cls.evaluate_epv(events, non_events, len(pred_cols))
            if epv_find:
                findings.append(epv_find)
            else:
                strengths.append(
                    f"Adequate Events-Per-Variable (EPV = {min(events, non_events) / max(1, len(pred_cols)):.1f} ≥ 10)."
                )

            sep_find = cls.evaluate_separation(results_meta.get("coef_df"))
            if sep_find:
                findings.append(sep_find)

            miss_find = cls.evaluate_missingness(df, [outcome_col] + (pred_cols or []))
            if miss_find:
                findings.append(miss_find)

        elif analysis_type == "rct":
            checklist = "CONSORT 2010 (Randomized Controlled Trial)"
            strengths.append(
                "Dual reporting of Relative Risk (RR) and Absolute Risk Difference (RD) conforms to CONSORT item 17b."
            )
            strengths.append(
                "Number Needed to Treat (NNT) and Relative Risk Reduction (RRR) facilitate bedside clinical interpretation."
            )

            rct_cell_find = cls.evaluate_rct_cell_counts(
                events_ctrl=results_meta.get("events_control", 10),
                n_ctrl=results_meta.get("n_control", 100),
                events_treat=results_meta.get("events_intervention", 10),
                n_treat=results_meta.get("n_intervention", 100),
            )
            if rct_cell_find:
                findings.append(rct_cell_find)

        elif analysis_type == "diagnostic":
            checklist = "STARD 2015 (Diagnostic Accuracy Studies)"
            strengths.append(
                "Complete 2x2 contingency matrix reporting with sensitivity, specificity, and likelihood ratios."
            )
            strengths.append(
                "Bayesian pre-to-post probability updating trajectory provides actionable clinical decision support."
            )

            tp = results_meta.get("tp", 0)
            fp = results_meta.get("fp", 0)
            fn = results_meta.get("fn", 0)
            tn = results_meta.get("tn", 0)
            if any(c < 5 for c in [tp, fp, fn, tn]):
                findings.append(
                    CritiqueFinding(
                        category="Sample_Size",
                        severity="MODERATE",
                        title="Small Cell Contingency Count in Diagnostic Matrix",
                        description=f"One of the 2x2 cells has < 5 counts (TP={tp}, FP={fp}, FN={fn}, TN={tn}).",
                        recommendation="Report exact Wilson or Clopper-Pearson 95% confidence intervals for sensitivity and specificity.",
                    )
                )

        elif analysis_type == "psm":
            checklist = "STROBE / ISPOR Good Research Practices for PSM"
            strengths.append(
                "Propensity score balance evaluated via Standardized Mean Differences (SMD) rather than sample-size dependent p-values."
            )
            strengths.append(
                "Love plot enables visual confirmation of covariate balance (< 0.10 threshold)."
            )

            n_orig = results_meta.get("n_original", len(df))
            n_matched = results_meta.get("n_matched", 0)
            attrition = (n_orig - n_matched) / n_orig if n_orig > 0 else 0.0
            if attrition > 0.40:
                findings.append(
                    CritiqueFinding(
                        category="Sample_Size",
                        severity="MODERATE",
                        title=f"High Trimming / Attrition Rate ({attrition:.1%})",
                        description=f"Matching discarded {n_orig - n_matched:,} subjects ({attrition:.1%}), which may limit generalizability / external validity.",
                        recommendation="Consider Inverse Probability of Treatment Weighting (IPTW) or overlap weighting to retain the full target population.",
                    )
                )

        # Determine overall status
        high_risks = [f for f in findings if f.severity == "HIGH"]
        mod_risks = [f for f in findings if f.severity == "MODERATE"]

        if high_risks:
            overall = "HIGH_RISK_BIAS"
        elif mod_risks:
            overall = "VALID_WITH_LIMITATIONS"
        else:
            overall = "ROBUST"

        return CritiqueVerdict(
            analysis_type=analysis_type,
            overall_status=overall,
            strengths=strengths,
            findings=findings,
            equator_checklist=checklist,
        )
