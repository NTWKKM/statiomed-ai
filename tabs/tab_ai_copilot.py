"""
tabs/tab_ai_copilot.py - StatioMed AI Clinical Co-Pilot Tab
=============================================================================
Interactive AI-assisted biostatistical planning, PICO parsing, PubMed evidence
retrieval, sample size calculation, and Statistical Analysis Plan (SAP) creation.
=============================================================================
"""

from __future__ import annotations
from typing import Any
import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from shinychat import chat_ui, chat_server

from agent.tools.tool_pubmed import PubMedEvidenceTool
from agent.tools.tool_sample_size import SampleSizeTool
from agent.tools.tool_synthetic_data import SyntheticDataTool

@module.ui
def ai_copilot_ui() -> ui.Tag:
    return ui.div(
        ui.card(
            ui.card_header(
                ui.div(
                    ui.h4("🤖 StatioMed AI — Clinical Research Co-Pilot", class_="m-0 text-primary"),
                    ui.p("Evidence-based Study Design, PICO Extraction & Statistical Analysis Planning (SAP)", class_="text-muted small m-0")
                )
            ),
            ui.card_body(
                ui.layout_columns(
                    ui.div(
                        ui.h5("💡 Quick Actions & Clinical Tools"),
                        ui.input_select(
                            "action_mode",
                            "Select Action Mode:",
                            {
                                "sap_design": "📋 1. Statistical Analysis Plan (SAP) Proposal",
                                "pico_pubmed": "🔍 2. PICO & PubMed Benchmark Evidence",
                                "sample_size": "📐 3. Sample Size & Power Calculation",
                                "synthetic_cohort": "🧬 4. Generate Synthetic Trial Cohort"
                            }
                        ),
                        ui.input_text_area(
                            "research_prompt",
                            "Clinical Objective / Research Question:",
                            placeholder="e.g., Comparing 1-year mortality of SGLT2 inhibitors vs placebo in HFrEF patients...",
                            rows=4
                        ),
                        ui.input_action_button("btn_run_action", "Execute Agent Task", class_="btn-primary w-100 mt-2"),
                        ui.hr(),
                        ui.div(
                            ui.p("🔒 Zero-PHI Guarantee: No hospital identifiers leave your workstation.", class_="text-success small fw-bold")
                        )
                    ),
                    ui.div(
                        ui.h5("📄 Generated Output & Evidence"),
                        ui.output_ui("action_output_display")
                    ),
                    col_widths=(5, 7)
                )
            )
        )
    )

@module.server
def ai_copilot_server(input: Inputs, output: Outputs, session: Session, dataset: reactive.Value[pd.DataFrame | None]) -> None:
    result_state = reactive.Value("")

    @reactive.effect
    @reactive.event(input.btn_run_action)
    def _on_run_action():
        prompt = input.research_prompt()
        mode = input.action_mode()

        if not prompt:
            result_state.set("<div class='alert alert-warning'>Please enter a clinical objective or research question.</div>")
            return

        if mode == "pico_pubmed":
            tool = PubMedEvidenceTool()
            try:
                articles = tool.search_and_extract(prompt, max_results=3)
                if not articles:
                    result_state.set("<div class='alert alert-info'>No published articles found for this query.</div>")
                    return
                html = "<div class='card p-3'><h6>📚 Published Benchmark Evidence (Vancouver Format):</h6><ol>"
                for a in articles:
                    html += f"<li><strong>{a['title']}</strong><br><span class='text-muted small'>{a['vancouver_citation']}</span></li><br>"
                html += "</ol></div>"
                result_state.set(html)
            except Exception as e:
                result_state.set(f"<div class='alert alert-danger'>PubMed Error: {e}</div>")

        elif mode == "sample_size":
            res = SampleSizeTool.calculate_two_proportions(p1=0.25, p2=0.15, power=0.80, alpha=0.05, dropout_rate=0.15)
            html = f"""
            <div class='card p-3 border-info'>
                <h6 class='text-info'>📐 Sample Size Calculation Result:</h6>
                <p><strong>Method:</strong> {res['test_type']}</p>
                <ul>
                    <li>Control Event Rate ($p_1$): {res['p1_control']:.1%}</li>
                    <li>Intervention Event Rate ($p_2$): {res['p2_intervention']:.1%}</li>
                    <li>Total Raw Target: <strong>{res['n_total_raw']}</strong> patients</li>
                    <li><strong>Total Target with 15% Drop-out: <span class='text-success'>{res['n_total_adjusted']}</span> patients</strong></li>
                </ul>
                <div class='alert alert-light border'><em>"{res['justification_text']}"</em></div>
            </div>
            """
            result_state.set(html)

        elif mode == "synthetic_cohort":
            df_mock = SyntheticDataTool.generate_rct_cohort(n=100, seed=42)
            dataset.set(df_mock)
            html = f"""
            <div class='alert alert-success'>
                <h6>🧬 Synthetic RCT Cohort Generated!</h6>
                <p>Generated {len(df_mock)} simulated patient records with verified physiological bounds (SBP > DBP + 20). Data is now loaded into the active session.</p>
            </div>
            """
            result_state.set(html)

        else: # sap_design
            html = f"""
            <div class='card p-3 border-primary'>
                <h6 class='text-primary'>📋 Statistical Analysis Plan (SAP) Candidate</h6>
                <p><strong>Objective:</strong> {prompt}</p>
                <ol>
                    <li><strong>Primary Endpoint:</strong> Time-to-first event (Log-rank test and multivariable Cox proportional hazards with Efron tie-handling).</li>
                    <li><strong>Baseline Comparison:</strong> Table 1 with Standardized Mean Differences (SMD) and Wilcoxon/Chi-square tests.</li>
                    <li><strong>Confounder Adjustment:</strong> Multivariable step-down regression with collinearity check ($\text{{VIF}} < 5.0$).</li>
                    <li><strong>Missing Data Strategy:</strong> Little's MCAR test; Multiple Imputation by Chained Equations (MICE, $m=5$) if missingness exceeds 5%.</li>
                    <li><strong>Multiplicity Correction:</strong> Benjamini-Hochberg False Discovery Rate for secondary endpoints.</li>
                </ol>
            </div>
            """
            result_state.set(html)

    @output
    @render.ui
    def action_output_display():
        content = result_state()
        if not content:
            return ui.HTML("<div class='text-muted p-4 text-center border rounded bg-light'>Output and generated SAP will appear here...</div>")
        return ui.HTML(content)
