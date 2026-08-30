"""
📋 Reporting Checklists Module

Publication guideline checklists for medical research reporting.

Includes:
    - CONSORT: RCT reporting (Consolidated Standards of Reporting Trials)
    - STROBE: Observational study reporting (STrengthening the Reporting of OBservational studies in Epidemiology)
    - Checklist tracking and validation utilities

References:
    Schulz KF, et al. (2010). CONSORT 2010 Statement.
    von Elm E, et al. (2007). STROBE Statement.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class ChecklistStatus(Enum):
    """Status of a checklist item."""

    NOT_APPLICABLE = "N/A"
    NOT_DONE = "Not addressed"
    PARTIAL = "Partially addressed"
    COMPLETE = "Complete"


@dataclass
class ChecklistItem:
    """Individual checklist item."""

    number: str
    item: str
    description: str
    section: str
    status: ChecklistStatus = ChecklistStatus.NOT_DONE
    page_number: str = ""
    notes: str = ""


@dataclass
class ReportingChecklist:
    """Base class for reporting checklists."""

    name: str
    items: list[ChecklistItem] = field(default_factory=list)

    def get_completion_summary(self) -> dict[str, Any]:
        """Get summary of checklist completion."""
        total = 0
        complete = 0
        partial = 0
        not_done = 0

        for item in self.items:
            if item.status == ChecklistStatus.NOT_APPLICABLE:
                continue
            total += 1
            if item.status == ChecklistStatus.COMPLETE:
                complete += 1
            elif item.status == ChecklistStatus.PARTIAL:
                partial += 1
            elif item.status == ChecklistStatus.NOT_DONE:
                not_done += 1

        return {
            "total_applicable": total,
            "complete": complete,
            "partial": partial,
            "not_done": not_done,
            "completion_rate": round(complete / total * 100, 1) if total > 0 else 0,
        }

    def update_item(
        self,
        number: str,
        status: ChecklistStatus,
        page: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Update a checklist item status."""
        for item in self.items:
            if item.number == number:
                item.status = status
                if page is not None:
                    item.page_number = page
                if notes is not None:
                    item.notes = notes
                return True
        return False

    def to_html(self) -> str:
        """Generate HTML table of checklist."""
        summary = self.get_completion_summary()

        html_content = f"""
        <div class="checklist-container">
            <h3>{self.name} Checklist</h3>
            <p class="completion-badge">
                Completion: {summary["completion_rate"]}% 
                ({summary["complete"]}/{summary["total_applicable"]} items)
            </p>
            <table class="checklist-table">
                <thead>
                    <tr>
                        <th>Item</th>
                        <th>Description</th>
                        <th>Status</th>
                        <th>Page</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
        """

        current_section = ""
        for item in self.items:
            if item.section != current_section:
                current_section = item.section
                html_content += f'<tr class="section-header"><td colspan="5"><strong>{current_section}</strong></td></tr>'

            status_class = {
                ChecklistStatus.COMPLETE: "status-complete",
                ChecklistStatus.PARTIAL: "status-partial",
                ChecklistStatus.NOT_DONE: "status-not-done",
                ChecklistStatus.NOT_APPLICABLE: "status-na",
            }.get(item.status, "")

            html_content += f"""
                <tr>
                    <td>{item.number}</td>
                    <td>{html.escape(item.item)}</td>
                    <td class="{status_class}">{html.escape(item.status.value)}</td>
                    <td>{html.escape(item.page_number)}</td>
                    <td>{html.escape(item.notes)}</td>
                </tr>
            """

        html_content += """
                </tbody>
            </table>
        </div>
        """
        return html_content


def create_consort_checklist() -> ReportingChecklist:
    """
    Create CONSORT 2010 checklist for RCT reporting.

    CONSORT (Consolidated Standards of Reporting Trials) is the
    standard for reporting randomized controlled trials.

    Returns:
        ReportingChecklist with all CONSORT 2010 items
    """
    items = [
        # Title and Abstract
        ChecklistItem(
            "1a",
            "Title",
            "Identification as a randomised trial in the title",
            "Title and Abstract",
        ),
        ChecklistItem(
            "1b",
            "Abstract",
            "Structured summary of trial design, methods, results, conclusions",
            "Title and Abstract",
        ),
        # Introduction
        ChecklistItem(
            "2a",
            "Background",
            "Scientific background and explanation of rationale",
            "Introduction",
        ),
        ChecklistItem(
            "2b", "Objectives", "Specific objectives or hypotheses", "Introduction"
        ),
        # Methods - Trial Design
        ChecklistItem(
            "3a",
            "Trial design",
            "Description of trial design (parallel, factorial) with allocation ratio",
            "Methods",
        ),
        ChecklistItem(
            "3b",
            "Changes to trial design",
            "Important changes to methods after trial commencement with reasons",
            "Methods",
        ),
        # Participants
        ChecklistItem(
            "4a",
            "Eligibility criteria",
            "Eligibility criteria for participants",
            "Methods",
        ),
        ChecklistItem(
            "4b",
            "Settings and locations",
            "Settings and locations where data collected",
            "Methods",
        ),
        # Interventions
        ChecklistItem(
            "5",
            "Interventions",
            "Interventions for each group with sufficient details for replication",
            "Methods",
        ),
        # Outcomes
        ChecklistItem(
            "6a",
            "Primary outcome",
            "Completely defined primary outcome measures including how and when assessed",
            "Methods",
        ),
        ChecklistItem(
            "6b",
            "Secondary outcomes",
            "Any changes to trial outcomes after trial commenced",
            "Methods",
        ),
        # Sample Size
        ChecklistItem("7a", "Sample size", "How sample size was determined", "Methods"),
        ChecklistItem(
            "7b",
            "Interim analyses",
            "Explanation of any interim analyses and stopping guidelines",
            "Methods",
        ),
        # Randomisation
        ChecklistItem(
            "8a",
            "Sequence generation",
            "Method used to generate random allocation sequence",
            "Methods",
        ),
        ChecklistItem(
            "8b",
            "Type of randomisation",
            "Type of randomisation; details of any restriction",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Allocation concealment",
            "Mechanism used to implement random allocation sequence",
            "Methods",
        ),
        ChecklistItem(
            "10",
            "Implementation",
            "Who generated sequence, enrolled participants, assigned to interventions",
            "Methods",
        ),
        # Blinding
        ChecklistItem(
            "11a",
            "Blinding",
            "Who was blinded after assignment to interventions",
            "Methods",
        ),
        ChecklistItem(
            "11b",
            "Blinding details",
            "Description of similarity of interventions",
            "Methods",
        ),
        # Statistical Methods
        ChecklistItem(
            "12a",
            "Statistical methods",
            "Statistical methods used to compare groups for primary and secondary outcomes",
            "Methods",
        ),
        ChecklistItem(
            "12b",
            "Additional analyses",
            "Methods for additional analyses (subgroup, adjusted)",
            "Methods",
        ),
        # Results - Participant Flow
        ChecklistItem(
            "13a",
            "Flow diagram",
            "Numbers of participants at each stage (flow diagram)",
            "Results",
        ),
        ChecklistItem(
            "13b",
            "Exclusions",
            "Reasons for exclusion or discontinuation at each stage",
            "Results",
        ),
        # Recruitment
        ChecklistItem(
            "14a",
            "Recruitment dates",
            "Dates defining periods of recruitment and follow-up",
            "Results",
        ),
        ChecklistItem(
            "14b", "Trial ended", "Why the trial ended or was stopped", "Results"
        ),
        # Baseline Data
        ChecklistItem(
            "15",
            "Baseline data",
            "Table showing baseline demographic and clinical characteristics",
            "Results",
        ),
        # Numbers Analysed
        ChecklistItem(
            "16",
            "Numbers analysed",
            "Number in each group included in each analysis and whether ITT",
            "Results",
        ),
        # Outcomes and Estimation
        ChecklistItem(
            "17a",
            "Outcomes",
            "For each outcome, results for each group with effect size and precision",
            "Results",
        ),
        ChecklistItem(
            "17b",
            "Binary outcomes",
            "For binary outcomes, presentation of absolute and relative effect sizes",
            "Results",
        ),
        # Ancillary Analyses
        ChecklistItem(
            "18",
            "Ancillary analyses",
            "Results of any other analyses performed",
            "Results",
        ),
        # Harms
        ChecklistItem(
            "19",
            "Harms",
            "All important harms or unintended effects in each group",
            "Results",
        ),
        # Discussion
        ChecklistItem(
            "20",
            "Limitations",
            "Trial limitations, addressing sources of potential bias",
            "Discussion",
        ),
        ChecklistItem(
            "21",
            "Generalisability",
            "Generalisability (external validity) of the trial findings",
            "Discussion",
        ),
        ChecklistItem(
            "22",
            "Interpretation",
            "Interpretation consistent with results, balancing benefits and harms",
            "Discussion",
        ),
        # Other Information
        ChecklistItem(
            "23",
            "Registration",
            "Registration number and name of trial registry",
            "Other Information",
        ),
        ChecklistItem(
            "24",
            "Protocol",
            "Where the full trial protocol can be accessed",
            "Other Information",
        ),
        ChecklistItem(
            "25",
            "Funding",
            "Sources of funding and other support; role of funders",
            "Other Information",
        ),
    ]

    return ReportingChecklist(name="CONSORT 2010", items=items)


def create_strobe_checklist(study_type: str = "cohort") -> ReportingChecklist:
    """
    Create STROBE checklist for observational study reporting.

    STROBE (STrengthening the Reporting of OBservational studies in
    Epidemiology) covers cohort, case-control, and cross-sectional studies.

    Args:
        study_type: "cohort", "case_control", or "cross_sectional"

    Returns:
        ReportingChecklist with appropriate STROBE items
    """
    items = [
        # Title and Abstract
        ChecklistItem(
            "1a",
            "Title",
            "Indicate the study's design with a commonly used term",
            "Title and Abstract",
        ),
        ChecklistItem(
            "1b",
            "Abstract",
            "Provide informative and balanced summary of what was done and found",
            "Title and Abstract",
        ),
        # Introduction
        ChecklistItem(
            "2",
            "Background/rationale",
            "Explain the scientific background and rationale for the investigation",
            "Introduction",
        ),
        ChecklistItem(
            "3",
            "Objectives",
            "State specific objectives, including any prespecified hypotheses",
            "Introduction",
        ),
        # Methods
        ChecklistItem(
            "4",
            "Study design",
            "Present key elements of study design early in the paper",
            "Methods",
        ),
        ChecklistItem(
            "5",
            "Setting",
            "Describe the setting, locations, and relevant dates",
            "Methods",
        ),
        ChecklistItem(
            "6a",
            "Participants",
            "Give eligibility criteria, sources and methods of selection",
            "Methods",
        ),
    ]

    # Study-type specific items
    if study_type == "cohort":
        items.append(
            ChecklistItem("6b", "Cohort", "Give the methods of follow-up", "Methods")
        )
    elif study_type == "case_control":
        items.append(
            ChecklistItem(
                "6b",
                "Cases/Controls",
                "Give rationale for choice of cases and controls, and methods of ascertainment",
                "Methods",
            )
        )
    elif study_type == "cross_sectional":
        items.append(
            ChecklistItem(
                "6b",
                "Cross-sectional",
                "Give the sampling strategy and participant selection",
                "Methods",
            )
        )
    else:
        raise ValueError(
            f"Invalid study_type: '{study_type}'. "
            "Must be 'cohort', 'case_control', or 'cross_sectional'."
        )

    items.extend(
        [
            ChecklistItem(
                "7",
                "Variables",
                "Clearly define all outcomes, exposures, predictors, confounders, effect modifiers",
                "Methods",
            ),
            ChecklistItem(
                "8",
                "Data sources",
                "Give sources of data and details of methods of assessment",
                "Methods",
            ),
            ChecklistItem(
                "9",
                "Bias",
                "Describe any efforts to address potential sources of bias",
                "Methods",
            ),
            ChecklistItem(
                "10",
                "Study size",
                "Explain how the study size was arrived at",
                "Methods",
            ),
            ChecklistItem(
                "11",
                "Quantitative variables",
                "Explain how quantitative variables were handled in analyses",
                "Methods",
            ),
            ChecklistItem(
                "12a",
                "Statistical methods",
                "Describe all statistical methods including those for confounding",
                "Methods",
            ),
            ChecklistItem(
                "12b",
                "Subgroups",
                "Describe any methods used to examine subgroups and interactions",
                "Methods",
            ),
            ChecklistItem(
                "12c",
                "Missing data",
                "Explain how missing data were addressed",
                "Methods",
            ),
            ChecklistItem(
                "12d",
                "Sensitivity analyses",
                "Explain any sensitivity analyses",
                "Methods",
            ),
        ]
    )

    # Results
    items.extend(
        [
            ChecklistItem(
                "13a",
                "Participants",
                "Report numbers at each stage of study",
                "Results",
            ),
            ChecklistItem(
                "13b",
                "Non-participation",
                "Give reasons for non-participation at each stage",
                "Results",
            ),
            ChecklistItem(
                "13c", "Flow diagram", "Consider use of a flow diagram", "Results"
            ),
            ChecklistItem(
                "14a",
                "Descriptive data",
                "Give characteristics of study participants and information on exposures",
                "Results",
            ),
            ChecklistItem(
                "14b",
                "Missing data",
                "Indicate number of participants with missing data for each variable",
                "Results",
            ),
        ]
    )

    if study_type == "cohort":
        items.append(
            ChecklistItem("14c", "Follow-up", "Summarise follow-up time", "Results")
        )

    items.extend(
        [
            ChecklistItem(
                "15",
                "Outcome data",
                "Report numbers of outcome events or summary measures",
                "Results",
            ),
            ChecklistItem(
                "16a",
                "Main results",
                "Give unadjusted estimates and, if applicable, confounder-adjusted estimates",
                "Results",
            ),
            ChecklistItem(
                "16b",
                "Continuous variables",
                "Report category boundaries when continuous variables were categorized",
                "Results",
            ),
            ChecklistItem(
                "16c",
                "Relative risk",
                "If relevant, consider translating relative risk into absolute risk",
                "Results",
            ),
            ChecklistItem(
                "17",
                "Other analyses",
                "Report other analyses done—e.g., sensitivity or subgroup analyses",
                "Results",
            ),
            # Discussion
            ChecklistItem(
                "18",
                "Key results",
                "Summarise key results with reference to study objectives",
                "Discussion",
            ),
            ChecklistItem(
                "19",
                "Limitations",
                "Discuss limitations, including sources of potential bias",
                "Discussion",
            ),
            ChecklistItem(
                "20",
                "Interpretation",
                "Give a cautious overall interpretation of results",
                "Discussion",
            ),
            ChecklistItem(
                "21",
                "Generalisability",
                "Discuss the generalisability (external validity) of the study results",
                "Discussion",
            ),
            # Other
            ChecklistItem(
                "22",
                "Funding",
                "Give the source of funding and the role of the funders",
                "Other Information",
            ),
        ]
    )

    return ReportingChecklist(
        name=f"STROBE ({study_type.replace('_', ' ').title()})", items=items
    )


def generate_checklist_markdown(checklist: ReportingChecklist) -> str:
    """
    Generate a markdown version of the checklist for export.

    Args:
        checklist: The reporting checklist to export

    Returns:
        Markdown formatted string
    """
    summary = checklist.get_completion_summary()

    md = f"# {checklist.name} Checklist\n\n"
    md += f"**Completion Rate:** {summary['completion_rate']}% ({summary['complete']}/{summary['total_applicable']} items)\n\n"

    current_section = ""
    for item in checklist.items:
        if item.section != current_section:
            current_section = item.section
            md += f"\n## {current_section}\n\n"

        status_emoji = {
            ChecklistStatus.COMPLETE: "✅",
            ChecklistStatus.PARTIAL: "🔶",
            ChecklistStatus.NOT_DONE: "❌",
            ChecklistStatus.NOT_APPLICABLE: "➖",
        }.get(item.status, "❓")

        md += f"- {status_emoji} **{item.number}. {item.item}**: {item.description}"
        if item.page_number:
            md += f" (p. {item.page_number})"
        if item.notes:
            md += f"\n  - *Note: {item.notes}*"
        md += "\n"

    return md


def auto_populate_strobe(
    analysis_metadata: dict[str, Any],
    study_type: str = "cohort",
) -> ReportingChecklist:
    """
    Auto-populate STROBE checklist based on analysis metadata.

    Automatically marks items as complete/partial based on what was
    performed in the statistical analysis.

    Args:
        analysis_metadata: Dictionary containing analysis details:
            - n_total: Total sample size
            - n_analyzed: Number in final analysis
            - outcome_name: Name of outcome variable
            - predictors: List of predictor variables
            - has_missing_report: Whether missing data was reported
            - has_ci: Whether confidence intervals were reported
            - method: Statistical method used
            - has_sensitivity: Whether sensitivity analysis was done
            - has_subgroup: Whether subgroup analysis was done
        study_type: "cohort", "case_control", or "cross_sectional"

    Returns:
        Pre-populated ReportingChecklist
    """
    checklist = create_strobe_checklist(study_type)

    # Extract metadata with defaults
    n_total = analysis_metadata.get("n_total", 0)
    n_analyzed = analysis_metadata.get("n_analyzed", 0)
    outcome = analysis_metadata.get("outcome_name", "")
    predictors = analysis_metadata.get("predictors", [])
    has_missing = analysis_metadata.get("has_missing_report", False)
    has_ci = analysis_metadata.get("has_ci", False)
    method = analysis_metadata.get("method", "logistic")
    has_sensitivity = analysis_metadata.get("has_sensitivity", False)
    has_subgroup = analysis_metadata.get("has_subgroup", False)

    # Auto-mark items based on metadata
    auto_marks: dict[str, tuple[ChecklistStatus, str]] = {}

    # 5. Setting: Study setting - partial if we have data
    if n_total > 0:
        auto_marks["5"] = (
            ChecklistStatus.PARTIAL,
            f"Data available: n={n_total}",
        )

    # 7. Variables: Outcomes and exposures
    if outcome and predictors:
        auto_marks["7"] = (
            ChecklistStatus.COMPLETE,
            f"Outcome: {outcome}; Predictors: {len(predictors)} variables",
        )

    # 10. Study size
    if n_analyzed > 0:
        auto_marks["10"] = (
            ChecklistStatus.PARTIAL,
            f"Analyzed n={n_analyzed}. Explain how sample size was determined.",
        )

    # 12a. Statistical methods
    method_desc = {
        "logistic": "Logistic regression",
        "firth": "Firth's penalized logistic regression",
        "auto": "Logistic regression (auto-selected)",
    }.get(method, method)
    ci_suffix = " with 95% CI" if has_ci else ""
    auto_marks["12a"] = (
        ChecklistStatus.COMPLETE,
        f"{method_desc}{ci_suffix}",
    )

    # 12b. Subgroups and interactions
    if has_subgroup:
        auto_marks["12b"] = (
            ChecklistStatus.COMPLETE,
            "Subgroup analysis performed",
        )

    # 12c. Missing data
    if has_missing:
        auto_marks["12c"] = (
            ChecklistStatus.COMPLETE,
            "Missing data handling documented",
        )
    else:
        auto_marks["12c"] = (
            ChecklistStatus.NOT_DONE,
            "Add missing data summary",
        )

    # 12d. Sensitivity analyses
    if has_sensitivity:
        auto_marks["12d"] = (
            ChecklistStatus.COMPLETE,
            "E-value sensitivity analysis included",
        )

    # 13. Participants
    if n_total > 0 and n_analyzed > 0:
        excluded = max(0, n_total - n_analyzed)
        auto_marks["13a"] = (
            ChecklistStatus.PARTIAL,
            f"Total: {n_total}, Analyzed: {n_analyzed}, Excluded: {excluded}",
        )

    # 16. Main results
    if has_ci:
        auto_marks["16a"] = (
            ChecklistStatus.COMPLETE,
            "OR/aOR with 95% CI reported",
        )

    # Apply auto-marks
    for number, (status, notes) in auto_marks.items():
        checklist.update_item(number, status, notes=notes)

    logger.info(
        "Auto-populated STROBE: %d items marked",
        len(auto_marks),
    )

    return checklist


def format_strobe_html_compact(checklist: ReportingChecklist) -> str:
    """
    Generate compact HTML for STROBE checklist display in UI.

    Args:
        checklist: The populated STROBE checklist

    Returns:
        HTML string for display
    """
    summary = checklist.get_completion_summary()

    status_icons = {
        ChecklistStatus.COMPLETE: "✅",
        ChecklistStatus.PARTIAL: "🔶",
        ChecklistStatus.NOT_DONE: "❌",
        ChecklistStatus.NOT_APPLICABLE: "➖",
    }

    rows = []
    for item in checklist.items:
        icon = status_icons.get(item.status, "❓")
        status_class = {
            ChecklistStatus.COMPLETE: "text-success",
            ChecklistStatus.PARTIAL: "text-warning",
            ChecklistStatus.NOT_DONE: "text-danger",
            ChecklistStatus.NOT_APPLICABLE: "text-muted",
        }.get(item.status, "")

        rows.append(f"""
            <tr class="{status_class}">
                <td>{icon}</td>
                <td><strong>{item.number}</strong></td>
                <td>{html.escape(item.item)}</td>
                <td style="font-size: 0.85em;">{html.escape(item.notes) if item.notes else "—"}</td>
            </tr>
        """)

    html_content = f"""
    <div class="strobe-checklist">
        <div class="alert alert-info mb-3">
            <strong>STROBE Completion:</strong>
            {summary["complete"]}/{summary["total_applicable"]} items complete
            ({summary["completion_rate"]}%)
        </div>
        <table class="table table-sm table-hover">
            <thead>
                <tr>
                    <th style="width: 30px;">Status</th>
                    <th style="width: 60px;">#</th>
                    <th>Item</th>
                    <th>Auto-filled Notes</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        <div class="text-muted" style="font-size: 0.85em; margin-top: 10px;">
            <strong>Reference:</strong> von Elm E, et al. (2007). STROBE Statement.
            <em>PLoS Medicine</em> 4(10): e296.
        </div>
    </div>
    """
    return html_content


def create_tripod_ai_checklist() -> ReportingChecklist:
    """
    Create TRIPOD+AI 2024 checklist for clinical AI & machine learning prediction models.

    References:
        Collins GS, et al. (2024). TRIPOD+AI statement: updated guidance for reporting
        clinical prediction models that use machine learning or artificial intelligence.
        BMJ 2024; 385:e078378.
    """
    items = [
        ChecklistItem(
            "1",
            "Title",
            "Identify the study as developing, validating, or updating an AI/ML prediction model",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2",
            "Abstract",
            "Summary of objectives, data sources, ML architecture, validation method, calibration, and discrimination",
            "Title and Abstract",
        ),
        ChecklistItem(
            "3a",
            "Background",
            "Clinical rationale and medical problem being addressed",
            "Introduction",
        ),
        ChecklistItem(
            "3b",
            "Objectives",
            "Specific objectives including intended clinical setting, user, and target population",
            "Introduction",
        ),
        ChecklistItem(
            "4a",
            "Source of data",
            "Study design, data sources (EHR, registries, trials), data collection period, and institutional settings",
            "Methods",
        ),
        ChecklistItem(
            "4b",
            "Data provenance",
            "Data provenance, preparation, de-identification, and data linkage protocols",
            "Methods",
        ),
        ChecklistItem(
            "5a",
            "Eligibility criteria",
            "Inclusion and exclusion criteria for study participants",
            "Methods",
        ),
        ChecklistItem(
            "5b",
            "Participant flow",
            "Details of participant selection and handling of multiple episodes/records",
            "Methods",
        ),
        ChecklistItem(
            "6a",
            "Outcome definition",
            "Definition of the outcome predicted, including how and when it was assessed",
            "Methods",
        ),
        ChecklistItem(
            "6b",
            "Outcome blinding",
            "Whether outcome assessors were blinded to predictor variables",
            "Methods",
        ),
        ChecklistItem(
            "7a",
            "Predictors",
            "Clearly define all candidate predictor variables, measurement timing, and units",
            "Methods",
        ),
        ChecklistItem(
            "7b",
            "Feature engineering",
            "Feature selection, transformation, scaling, encoding, and dimensionality reduction",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Sample size",
            "Rationale for sample size, number of events, and events per candidate parameter (EPV)",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Missing data",
            "Handling of missing data in predictors and outcome (e.g. MICE, complete-case)",
            "Methods",
        ),
        ChecklistItem(
            "10a",
            "AI/ML Model architecture",
            "Algorithm details (Logistic, XGBoost, Neural Nets), hyperparameters, and tuning protocol",
            "Methods",
        ),
        ChecklistItem(
            "10b",
            "Validation strategy",
            "Internal validation (cross-validation, bootstrapping) and external validation cohorts",
            "Methods",
        ),
        ChecklistItem(
            "11a",
            "Discrimination",
            "Concordance index (C-index) or AUC-ROC with 95% confidence intervals",
            "Results",
        ),
        ChecklistItem(
            "11b",
            "Calibration",
            "Calibration intercept and slope, calibration plots, and Brier score",
            "Results",
        ),
        ChecklistItem(
            "12",
            "Decision Curve Analysis (DCA)",
            "Net Benefit curves across relevant clinical decision threshold probabilities",
            "Results",
        ),
        ChecklistItem(
            "13",
            "Subgroup & Fairness",
            "Performance stratified across key clinical and demographic subgroups",
            "Results",
        ),
        ChecklistItem(
            "14",
            "Model presentation",
            "Format of final model (nomogram, score card, API, web calculator)",
            "Results",
        ),
        ChecklistItem(
            "15",
            "Limitations",
            "Study limitations, dataset biases, and generalizability bounds",
            "Discussion",
        ),
        ChecklistItem(
            "16",
            "Interpretation",
            "Overall clinical interpretation, net clinical utility, and potential harms",
            "Discussion",
        ),
        ChecklistItem(
            "17",
            "Availability",
            "Availability of code, model weights, and de-identified benchmark datasets",
            "Other Information",
        ),
        ChecklistItem(
            "18",
            "Funding & Ethics",
            "Funding sources, role of sponsors, and IRB/Ethics approvals",
            "Other Information",
        ),
    ]
    return ReportingChecklist(name="TRIPOD+AI 2024", items=items)


def create_stard_checklist() -> ReportingChecklist:
    """
    Create STARD 2015 checklist for diagnostic test accuracy studies.

    References:
        Bossuyt PM, et al. (2015). STARD 2015: An Updated List of Essential Items
        for Reporting Diagnostic Accuracy Studies. Radiology 277(3): 826-832.
    """
    items = [
        ChecklistItem(
            "1",
            "Title",
            "Identify the study as a diagnostic accuracy study in title",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2",
            "Abstract",
            "Structured summary of background, methods, results, and conclusions",
            "Title and Abstract",
        ),
        ChecklistItem(
            "3",
            "Scientific rationale",
            "Scientific background and diagnostic problem, intended clinical role",
            "Introduction",
        ),
        ChecklistItem(
            "4",
            "Study objectives",
            "Study objectives and diagnostic hypotheses",
            "Introduction",
        ),
        ChecklistItem(
            "5",
            "Study design",
            "Prospective or retrospective diagnostic design",
            "Methods",
        ),
        ChecklistItem(
            "6",
            "Participant eligibility",
            "Inclusion/exclusion criteria, clinical setting, recruitment dates",
            "Methods",
        ),
        ChecklistItem(
            "7",
            "Sampling method",
            "Method of recruitment (consecutive, random, convenience)",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Index test",
            "Description of index test, technology, cut-offs, and measurement procedures",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Reference standard",
            "Reference standard definition, rationale, and diagnostic criteria",
            "Methods",
        ),
        ChecklistItem(
            "10",
            "Blinding",
            "Blinding between index test readers and reference standard assessors",
            "Methods",
        ),
        ChecklistItem(
            "11",
            "Statistical methods",
            "Methods for calculating Sensitivity, Specificity, +LR, -LR, DOR, AUC-ROC with 95% CIs",
            "Methods",
        ),
        ChecklistItem(
            "12",
            "Indeterminate results",
            "Handling of indeterminate, invalid, or missing test results",
            "Methods",
        ),
        ChecklistItem(
            "13",
            "Participant flow",
            "Flow of participants (numbers eligible, tested, verified by reference)",
            "Results",
        ),
        ChecklistItem(
            "14",
            "Baseline demographics",
            "Clinical and demographic characteristics of tested cohort",
            "Results",
        ),
        ChecklistItem(
            "15",
            "Diagnostic 2x2 Table",
            "Cross-tabulation of index test vs reference standard (TP, FP, FN, TN)",
            "Results",
        ),
        ChecklistItem(
            "16",
            "Diagnostic Accuracy Metrics",
            "Estimates of Sensitivity, Specificity, +LR, -LR, DOR with 95% CIs",
            "Results",
        ),
        ChecklistItem(
            "17",
            "Fagan Nomogram / Post-test prob",
            "Post-test probabilities calculated from pre-test probabilities and likelihood ratios",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Adverse events",
            "Adverse events from performing index test or reference standard",
            "Results",
        ),
        ChecklistItem(
            "19",
            "Limitations",
            "Study limitations, verification bias, spectrum bias",
            "Discussion",
        ),
        ChecklistItem(
            "20",
            "Clinical implications",
            "Implications for clinical practice and diagnostic pathways",
            "Discussion",
        ),
        ChecklistItem(
            "21",
            "Funding & Protocol",
            "Registration, protocol location, and funding disclosures",
            "Other Information",
        ),
    ]
    return ReportingChecklist(name="STARD 2015", items=items)


def create_prisma_checklist() -> ReportingChecklist:
    """
    Create PRISMA 2020 checklist for systematic reviews and meta-analyses.

    References:
        Page MJ, et al. (2021). The PRISMA 2020 statement: an updated guideline
        for reporting systematic reviews. BMJ 2021; 372:n71.
    """
    items = [
        ChecklistItem(
            "1",
            "Title",
            "Identify the report as a systematic review and/or meta-analysis",
            "Title",
        ),
        ChecklistItem(
            "2",
            "Abstract",
            "Structured summary of PICO, search, synthesis, results, and conclusions",
            "Abstract",
        ),
        ChecklistItem(
            "3",
            "Rationale",
            "Rationale for review in context of existing knowledge",
            "Introduction",
        ),
        ChecklistItem(
            "4",
            "Objectives",
            "Explicit statement of questions addressed using PICO framework",
            "Introduction",
        ),
        ChecklistItem(
            "5",
            "Eligibility criteria",
            "Inclusion and exclusion criteria and study grouping rationale",
            "Methods",
        ),
        ChecklistItem(
            "6",
            "Information sources",
            "Databases searched, registries, dates searched, contact with authors",
            "Methods",
        ),
        ChecklistItem(
            "7",
            "Search strategy",
            "Full reproducible electronic search strategies with limits/filters",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Selection process",
            "Process of screening and selecting studies, independent reviewers",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Data extraction",
            "Data collection process, variables extracted, consensus mechanisms",
            "Methods",
        ),
        ChecklistItem(
            "10",
            "Risk of bias in studies",
            "Methods used to assess risk of bias (e.g. RoB 2, ROBINS-I)",
            "Methods",
        ),
        ChecklistItem(
            "11",
            "Effect measures",
            "Summary effect measures (OR, RR, HR, MD, SMD)",
            "Methods",
        ),
        ChecklistItem(
            "12",
            "Synthesis methods",
            "Statistical model (fixed vs random effects, REML, DerSimonian-Laird)",
            "Methods",
        ),
        ChecklistItem(
            "13",
            "Heterogeneity assessment",
            "Statistical heterogeneity assessment (Cochran Q, I^2, tau^2)",
            "Methods",
        ),
        ChecklistItem(
            "14",
            "Publication bias",
            "Methods for assessing publication/reporting bias (Funnel plot, Egger test)",
            "Methods",
        ),
        ChecklistItem(
            "15",
            "Study selection results",
            "PRISMA Flow Diagram (records identified, screened, excluded, included)",
            "Results",
        ),
        ChecklistItem(
            "16",
            "Study characteristics",
            "Summary table of included studies (population, interventions, outcomes)",
            "Results",
        ),
        ChecklistItem(
            "17",
            "Meta-analysis & Forest plots",
            "Forest plots displaying individual study effects and pooled estimate with 95% CI",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Risk of bias results",
            "Risk of bias judgments across included studies",
            "Results",
        ),
        ChecklistItem(
            "19",
            "Heterogeneity & Subgroups",
            "Results of subgroup analysis, meta-regression, and sensitivity analysis",
            "Results",
        ),
        ChecklistItem(
            "20",
            "Certainty of evidence",
            "GRADE assessment of certainty of evidence",
            "Discussion",
        ),
        ChecklistItem(
            "21",
            "Limitations",
            "Limitations of included evidence and review processes",
            "Discussion",
        ),
        ChecklistItem(
            "22",
            "Registration & Protocol",
            "PROSPERO registration number and protocol amendments",
            "Other Information",
        ),
        ChecklistItem(
            "23",
            "Funding & Conflicts",
            "Funding sources and conflicts of interest",
            "Other Information",
        ),
    ]
    return ReportingChecklist(name="PRISMA 2020", items=items)


def auto_populate_tripod_ai(analysis_metadata: dict[str, Any]) -> ReportingChecklist:
    """
    Auto-populates TRIPOD+AI (2024) checklist from prediction model metadata.
    """
    checklist = create_tripod_ai_checklist()
    c_index = analysis_metadata.get("c_index")
    brier_score = analysis_metadata.get("brier_score")
    calib_slope = analysis_metadata.get("calibration_slope")
    calib_intercept = analysis_metadata.get("calibration_intercept")
    has_dca = analysis_metadata.get("has_dca", False)
    n_total = analysis_metadata.get("n_total", 0)
    model_name = analysis_metadata.get("model_name", "Multivariable Model")

    if n_total > 0:
        checklist.update_item(
            "5a", ChecklistStatus.COMPLETE, notes=f"Cohort sample size: N={n_total}"
        )
    if model_name:
        checklist.update_item(
            "10a", ChecklistStatus.COMPLETE, notes=f"Architecture: {model_name}"
        )
    if c_index is not None:
        checklist.update_item(
            "11a", ChecklistStatus.COMPLETE, notes=f"C-index: {c_index:.3f} (95% CI)"
        )
    if calib_slope is not None or calib_intercept is not None:
        notes = (
            f"Calibration slope={calib_slope:.2f}, intercept={calib_intercept:.2f}"
            if calib_slope
            else ""
        )
        if brier_score is not None:
            notes += f", Brier score={brier_score:.3f}"
        checklist.update_item("11b", ChecklistStatus.COMPLETE, notes=notes)
    if has_dca:
        checklist.update_item(
            "12",
            ChecklistStatus.COMPLETE,
            notes="Decision Curve Analysis (Net Benefit curve) generated",
        )

    return checklist


def auto_populate_stard(analysis_metadata: dict[str, Any]) -> ReportingChecklist:
    """
    Auto-populates STARD 2015 checklist from diagnostic test accuracy metadata.
    """
    checklist = create_stard_checklist()
    sensitivity = analysis_metadata.get("sensitivity")
    specificity = analysis_metadata.get("specificity")
    auc = analysis_metadata.get("auc")
    plr = analysis_metadata.get("positive_lr")
    nlr = analysis_metadata.get("negative_lr")
    has_fagan = analysis_metadata.get("has_fagan", False)
    tp = analysis_metadata.get("tp")
    fp = analysis_metadata.get("fp")
    fn = analysis_metadata.get("fn")
    tn = analysis_metadata.get("tn")

    if tp is not None and fp is not None and fn is not None and tn is not None:
        checklist.update_item(
            "15",
            ChecklistStatus.COMPLETE,
            notes=f"2x2 Table: TP={tp}, FP={fp}, FN={fn}, TN={tn} (Total N={tp + fp + fn + tn})",
        )

    if sensitivity is not None and specificity is not None:
        notes = f"Sensitivity: {sensitivity:.1%}, Specificity: {specificity:.1%}"
        if plr is not None and nlr is not None:
            notes += f", +LR: {plr:.2f}, -LR: {nlr:.2f}"
        checklist.update_item("16", ChecklistStatus.COMPLETE, notes=notes)

    if auc is not None:
        checklist.update_item(
            "11", ChecklistStatus.COMPLETE, notes=f"AUC-ROC: {auc:.3f} with 95% CI"
        )

    if has_fagan:
        checklist.update_item(
            "17",
            ChecklistStatus.COMPLETE,
            notes="Fagan nomogram and post-test probabilities calculated",
        )

    return checklist


def auto_populate_consort(analysis_metadata: dict[str, Any]) -> ReportingChecklist:
    """
    Auto-populates CONSORT 2010 checklist from RCT metadata.
    """
    checklist = create_consort_checklist()
    n_assigned = analysis_metadata.get("n_assigned", 0)
    n_control = analysis_metadata.get("n_control", 0)
    n_intervention = analysis_metadata.get("n_intervention", 0)
    primary_effect = analysis_metadata.get("primary_effect", "")
    has_sample_size = analysis_metadata.get("has_sample_size", False)

    if n_assigned > 0:
        checklist.update_item(
            "13a",
            ChecklistStatus.COMPLETE,
            notes=f"Assigned N={n_assigned} (Control={n_control}, Intervention={n_intervention})",
        )
        checklist.update_item(
            "16",
            ChecklistStatus.COMPLETE,
            notes=f"Analyzed N={n_assigned} by Intention-to-Treat",
        )

    if has_sample_size:
        checklist.update_item(
            "7a",
            ChecklistStatus.COMPLETE,
            notes="Sample size calculated with power 80% and alpha 0.05",
        )

    if primary_effect:
        checklist.update_item(
            "17a",
            ChecklistStatus.COMPLETE,
            notes=f"Primary outcome estimate: {primary_effect}",
        )

    return checklist
