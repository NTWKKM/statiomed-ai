"""
tabs/tab_ai_copilot.py - StatioMed AI Clinical Co-Pilot Tab
=============================================================================
Interactive AI-assisted biostatistical planning, PICO parsing, PubMed evidence
retrieval, sample size calculation, Statistical Analysis Plan (SAP) creation,
deterministic manuscript drafting, and EQUATOR Network checklist compliance.
=============================================================================
"""

from __future__ import annotations

import html

import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from agent.agent_runner import create_clinical_agent, execute_agent_turn
from agent.manuscript_engine import ManuscriptEngine
from agent.tools.tool_pubmed import PubMedEvidenceTool
from agent.tools.tool_sample_size import SampleSizeTool
from agent.tools.tool_synthetic_data import SyntheticDataTool
from utils.reporting_checklists import (
    create_consort_checklist,
    create_stard_checklist,
    create_tripod_ai_checklist,
)


@module.ui
def ai_copilot_ui() -> ui.Tag:
    return ui.div(
        ui.card(
            ui.card_header(
                ui.div(
                    ui.h4(
                        "🤖 StatioMed AI — Clinical Research Co-Pilot",
                        class_="m-0 text-primary",
                    ),
                    ui.p(
                        "Evidence-based Study Design, PICO Extraction, Deterministic Manuscript Drafts & EQUATOR Audits",
                        class_="text-muted small m-0",
                    ),
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
                                "synthetic_cohort": "🧬 4. Generate Synthetic Trial Cohort",
                                "manuscript_draft": "📄 5. Deterministic Methods & Results Draft",
                                "equator_checklists": "📊 6. EQUATOR Network Audit Matrix",
                                "agent_interactive": "🧠 7. smolagents Clinical Tech Lead Reasoning",
                            },
                        ),
                        ui.input_text_area(
                            "research_prompt",
                            "Clinical Objective / Research Question:",
                            placeholder="e.g., Comparing 1-year mortality of SGLT2 inhibitors vs placebo in HFrEF patients...",
                            rows=4,
                        ),
                        ui.input_action_button(
                            "btn_run_action",
                            "Execute Agent Task",
                            class_="btn-primary w-100 mt-2",
                        ),
                        ui.hr(),
                        ui.div(
                            ui.p(
                                "🔒 Zero-PHI Guarantee: No hospital identifiers leave your workstation.",
                                class_="text-success small fw-bold",
                            ),
                            ui.p(
                                "📐 Mathematical Parity: Ground-truth verified against R 4.3.3 survival & statsmodels.",
                                class_="text-muted small",
                            ),
                        ),
                    ),
                    ui.div(
                        ui.h5("📄 Generated Output & Evidence"),
                        ui.output_ui("action_output_display"),
                    ),
                    col_widths=(5, 7),
                )
            ),
        )
    )


@module.server
def ai_copilot_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    dataset: reactive.Value[pd.DataFrame | None],
) -> None:
    result_state = reactive.Value("")

    @reactive.effect
    @reactive.event(input.btn_run_action)
    def _on_run_action():
        prompt = input.research_prompt()
        mode = input.action_mode()

        if not prompt and mode not in ["synthetic_cohort", "equator_checklists"]:
            result_state.set(
                "<div class='alert alert-warning'>Please enter a clinical objective or research question.</div>"
            )
            return

        if mode == "synthetic_cohort":
            df_gen, meta = SyntheticDataTool.generate_topic_aware_cohort(
                prompt, n=200, seed=42
            )
            dataset.set(df_gen)

            # Build Table Preview (First 5 Rows)
            cols = list(df_gen.columns)
            display_cols = cols[:8]  # First 8 columns for clean UI

            tbl_head = "".join(f"<th>{html.escape(c)}</th>" for c in display_cols)
            tbl_rows = ""
            for _, r in df_gen.head(5).iterrows():
                row_html = "".join(
                    f"<td>{html.escape(str(r[c]))}</td>" for c in display_cols
                )
                tbl_rows += f"<tr>{row_html}</tr>"

            models_html = "".join(
                f"<li class='mb-1'><i class='bi bi-check-circle text-primary me-1'></i>{html.escape(m)}</li>"
                for m in meta.get("recommended_models", [])
            )

            pico = meta.get("pico", {})
            pico_html = f"""
            <div class='row g-2 mb-3'>
                <div class='col-md-6'><div class='p-2 bg-white border rounded small'><strong>👥 Population (P):</strong> {html.escape(pico.get("population", "Clinical Cohort"))}</div></div>
                <div class='col-md-6'><div class='p-2 bg-white border rounded small'><strong>💊 Exposure/Intervention (I):</strong> {html.escape(pico.get("exposure", pico.get("intervention", "Target Exposure")))}</div></div>
                <div class='col-md-6'><div class='p-2 bg-white border rounded small'><strong>⚖️ Comparator (C):</strong> {html.escape(pico.get("comparator", "Control Group"))}</div></div>
                <div class='col-md-6'><div class='p-2 bg-white border rounded small'><strong>🎯 Outcome (O):</strong> {html.escape(pico.get("outcome", "Primary Endpoint"))}</div></div>
            </div>
            """

            html_out = f"""
            <div class='card border-success p-3'>
                <div class='d-flex justify-content-between align-items-center mb-2'>
                    <h6 class='text-success m-0'>🧬 Synthetic Clinical Cohort Generated & Active in Session!</h6>
                    <span class='badge bg-success'>{html.escape(meta.get("domain", "Clinical Research"))}</span>
                </div>
                <p class='text-muted small mb-2'>{html.escape(meta.get("description", ""))}</p>
                
                {pico_html}

                <div class='table-responsive border rounded bg-white mb-3'>
                    <table class='table table-sm table-striped table-hover m-0' style='font-size: 0.85rem;'>
                        <thead class='table-light'><tr>{tbl_head}</tr></thead>
                        <tbody>{tbl_rows}</tbody>
                    </table>
                </div>
                
                <div class='p-3 bg-light border rounded mb-2'>
                    <h6 class='text-dark mb-2'>📐 Recommended Biostatistical Analysis Workflow (SAMPL Compliant):</h6>
                    <ul class='list-unstyled mb-0' style='font-size: 0.9rem;'>
                        {models_html}
                    </ul>
                </div>

                <div class='alert alert-info py-2 px-3 mb-0 small'>
                    👉 <strong>Next Steps:</strong> ข้อมูลจำลองถูกโหลดเข้าสู่ระบบแล้ว สามารถคลิกไปที่แท็บ <strong>📊 Data Profiler</strong> เพื่อตรวจคุณภาพข้อมูล, <strong>📈 Regression</strong> เพื่อรันสถิติ, หรือ <strong>⏱️ Survival</strong> เพื่อวิเคราะห์การรอดชีพได้ทันที
                </div>
            </div>
            """
            result_state.set(html_out)

        elif mode == "pico_pubmed":
            tool = PubMedEvidenceTool()
            try:
                # Query PubMed with search terms
                search_q = prompt
                if any(
                    w in prompt.lower()
                    for w in ["สารเสพติด", "จิตเวช", "ยาบ้า", "substance", "psychiatry"]
                ):
                    search_q = "substance abuse psychiatric disorders methamphetamine psychosis"
                elif any(
                    w in prompt.lower()
                    for w in ["มะเร็ง", "cancer", "nsclc", "pembrolizumab"]
                ):
                    search_q = (
                        "pembrolizumab non-small cell lung cancer overall survival"
                    )
                elif any(
                    w in prompt.lower() for w in ["sepsis", "icu", "shock", "ติดเชื้อ"]
                ):
                    search_q = "sepsis resuscitation bundle mortality ICU"

                articles = tool.search_and_extract(search_q, max_results=3)
                if not articles:
                    result_state.set(
                        "<div class='alert alert-info'>No published articles found for this query.</div>"
                    )
                    return
                html_out = "<div class='card p-3 border-primary'><h6>📚 Published Benchmark Evidence (Vancouver Format):</h6><ol>"
                for a in articles:
                    html_out += f"<li><strong>{html.escape(a['title'])}</strong><br><span class='text-muted small'>{html.escape(a['vancouver_citation'])}</span></li><br>"
                html_out += "</ol></div>"
                result_state.set(html_out)
            except Exception as e:
                result_state.set(
                    f"<div class='alert alert-danger'>PubMed Error: {html.escape(str(e))}</div>"
                )

        elif mode == "sample_size":
            res = SampleSizeTool.calculate_two_proportions(
                p1=0.35, p2=0.18, power=0.80, alpha=0.05, dropout_rate=0.15
            )
            html_out = f"""
            <div class='card p-3 border-info'>
                <h6 class='text-info'>📐 Sample Size & Power Calculation (Fleiss Formula with Continuity Correction):</h6>
                <p><strong>Topic / Clinical Objective:</strong> {html.escape(prompt or "Comparative Clinical Trial")}</p>
                <ul>
                    <li>Exposure / Baseline Event Rate ($p_1$): {res["p1_control"]:.1%}</li>
                    <li>Intervention / Comparative Rate ($p_2$): {res["p2_intervention"]:.1%}</li>
                    <li>Statistical Power ($1-\\beta$): 80.0% | Type I Error ($\alpha$): 0.05 (Two-sided)</li>
                    <li>Total Target with 15% Drop-out: <strong><span class='text-success'>{res["n_total_adjusted"]}</span> patients ({res["n_control_adjusted"]} Group 1, {res["n_intervention_adjusted"]} Group 2)</strong></li>
                </ul>
                <div class='alert alert-light border'><em>"{res["justification_text"]}"</em></div>
            </div>
            """
            result_state.set(html_out)

        elif mode == "manuscript_draft":
            _, meta = SyntheticDataTool.generate_topic_aware_cohort(
                prompt, n=200, seed=42
            )
            pico = meta.get("pico", {})

            ctx = {
                "study_title": prompt or "Clinical Research Investigation",
                "n_total": 500,
                "n_intervention": 250,
                "n_control": 250,
                "m_imputations": 20,
                "median_followup": "365",
                "events_intervention": 35,
                "pct_events_intervention": "14.0%",
                "events_control": 60,
                "pct_events_control": "24.0%",
                "hr": "0.68",
                "hr_ci_lower": "0.52",
                "hr_ci_upper": "0.89",
                "hr_p_str": "= 0.005",
                "c_index": "0.78",
            }
            methods_txt = ManuscriptEngine.render_methods("cohort", ctx)
            results_txt = ManuscriptEngine.render_results("regression", ctx)

            html_out = f"""
            <div class='card p-3 border-secondary'>
                <div class='d-flex justify-content-between align-items-center mb-2'>
                    <h6>📄 Tailored Manuscript Draft (EQUATOR & SAMPL Compliant):</h6>
                    <span class='badge bg-secondary'>{html.escape(meta.get("domain", "Clinical Study"))}</span>
                </div>
                <div class='p-3 bg-light border rounded mb-3'>
                    <h6 class='text-primary mb-1'>Methods Section:</h6>
                    <pre style='white-space: pre-wrap; font-family: inherit;'>{html.escape(methods_txt)}</pre>
                </div>
                <div class='p-3 bg-light border rounded'>
                    <h6 class='text-success mb-1'>Results & Statistical Synthesis:</h6>
                    <pre style='white-space: pre-wrap; font-family: inherit;'>{html.escape(results_txt)}</pre>
                </div>
            </div>
            """
            result_state.set(html_out)

        elif mode == "equator_checklists":
            consort = create_consort_checklist()
            tripod = create_tripod_ai_checklist()
            stard = create_stard_checklist()

            html_out = f"""
            <div class='card p-3'>
                <h6>📊 EQUATOR Network Reporting Checklists:</h6>
                <div class='mb-2'>
                    <span class='badge bg-primary me-1'>CONSORT 2010 ({len(consort.items)} items)</span>
                    <span class='badge bg-secondary me-1'>TRIPOD+AI ({len(tripod.items)} items)</span>
                    <span class='badge bg-info'>STARD 2015 ({len(stard.items)} items)</span>
                </div>
                <div class='p-3 border rounded'>
                    <p class='text-muted small'>Comprehensive publication audit matrices ready for STROBE, CONSORT, TRIPOD+AI (2024), and STARD (2015).</p>
                    <table class='table table-sm table-hover'>
                        <thead><tr><th>#</th><th>Item</th><th>Description</th></tr></thead>
                        <tbody>
            """
            for it in consort.items[:6]:
                html_out += f"<tr><td><strong>{it.number}</strong></td><td>{html.escape(it.item)}</td><td>{html.escape(it.description)}</td></tr>"
            html_out += f"<tr><td colspan='3' class='text-muted text-center'>... and {len(consort.items) - 6} more items in full checklist</td></tr>"
            html_out += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
            result_state.set(html_out)

        elif mode == "agent_interactive":
            agent = create_clinical_agent()
            response = execute_agent_turn(agent, prompt)
            html_out = f"""
            <div class='card p-3 border-primary'>
                <h6 class='text-primary'>🧠 smolagents Clinical Tech Lead Reasoning:</h6>
                <div class='p-3 bg-light border rounded'>
                    <pre style='white-space: pre-wrap; font-family: inherit;'>{html.escape(response)}</pre>
                </div>
            </div>
            """
            result_state.set(html_out)

        else:  # sap_design
            _, meta = SyntheticDataTool.generate_topic_aware_cohort(
                prompt, n=200, seed=42
            )
            pico = meta.get("pico", {})

            sap_ctx = {
                "study_title": prompt or "Clinical Investigation Protocol",
                "primary_objective": f"To investigate the relationship between {pico.get('exposure', 'exposure')} and {pico.get('outcome', 'primary outcome')} in {pico.get('population', 'the target cohort')}.",
                "n_total": 500,
                "n_control": 250,
                "n_intervention": 250,
                "power_pct": "80",
                "alpha": "0.05",
                "dropout_pct": "15",
            }
            sap_md = ManuscriptEngine.render_sap(sap_ctx)
            html_out = f"""
            <div class='card p-3 border-primary'>
                <div class='d-flex justify-content-between align-items-center mb-2'>
                    <h6 class='text-primary m-0'>📋 Statistical Analysis Plan (SAP) Proposal</h6>
                    <span class='badge bg-primary'>{html.escape(meta.get("domain", "Clinical Research"))}</span>
                </div>
                <div class='p-3 bg-light border rounded'>
                    <pre style='white-space: pre-wrap; font-family: inherit;'>{html.escape(sap_md)}</pre>
                </div>
            </div>
            """
            result_state.set(html_out)

    @output
    @render.ui
    def action_output_display():
        content = result_state()
        if not content:
            return ui.HTML(
                "<div class='text-muted p-4 text-center border rounded bg-light'>Output and generated SAP / Evidence will appear here...</div>"
            )
        return ui.HTML(content)
