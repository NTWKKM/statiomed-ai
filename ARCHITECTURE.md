# 🏗️ StatioMed AI — System Architecture & Technical Specifications

> **Role & Domain**: Agentic Medical Statistical Analysis, Study Design & Manuscript Generation Engine  
> **Target Environment**: Local Hospital Workstation (Zero-PHI CLI) + Cloud / On-Prem ASGI Hosting (Hugging Face Spaces / Docker)

---

## 1. System Topology & Zero-PHI Trust Boundary

```
[ Hospital Internal Workstation ]
               │
               │ (Raw EHR / Excel / CSV / SPSS with PHI: HN, Citizen ID, Names, Dates)
               ▼
┌─────────────────────────────────────────────────────────────┐
│  tools_local/phi_sanitizer_cli.py (Zero-PHI Isolation)      │
│  - Strips direct identifiers (HMAC-SHA256 pseudonymization) │
│  - Relative Temporal Shifting (T0 = 0, elapsed days)        │
│  - Physiological Boundary Validation                        │
└─────────────────────────────────────────────────────────────┘
               │
               │ (Sanitized, De-identified Research Dataset)
               ▼
[ Cloud / Production Boundary: Hugging Face Spaces / On-Prem ]
┌─────────────────────────────────────────────────────────────┐
│  ASGI Gateway (asgi.py / Starlette + GZip + Static Files)   │
│  └─► Shiny for Python Reactive Web UI (app.py, tabs/*)      │
│        ├─► Tab 1: AI Co-Pilot (smolagents AgentRunner)      │
│        ├─► Tab 2: Data Profiler & Missingness Inspection    │
│        ├─► Tab 3: Survival Analysis (Kaplan-Meier / Cox PH) │
│        ├─► Tab 4: Core Regression (Linear/Logistic/Poisson) │
│        ├─► Tab 5: Sample Size & Power Calculation           │
│        ├─► Tab 6: Baseline Matching & Table 1 Generation    │
│        └─► Tab 7: Configuration & Model Settings            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Layers

### 2.1 Zero-PHI Preprocessing Boundary (`tools_local/`)
- **HMAC De-identification**: Transforms patient names, citizen IDs, and hospital numbers (`HN`) into irreversible surrogate UUIDs.
- **$T_0$ Temporal Transformation**: Converts calendar dates (admission, surgery, discharge, event) into elapsed days relative to an index date $T_0$, preserving exact time intervals for Cox proportional hazards and Kaplan-Meier estimation without exposing actual hospital stay dates.

### 2.2 Deterministic Computation Engine (`utils/`)
- **Zero Hallucination Principle**: The Large Language Model (LLM) never performs arithmetic, p-value calculations, sample size formulas, or regression fitting. All mathematical computations are delegated to deterministic Python libraries:
  - `statsmodels`: GLM, Ordinary Least Squares, Logistic Regression, Firth penalized regression (`firthmodels`).
  - `lifelines`: Kaplan-Meier, Log-Rank testing, Cox Proportional Hazards (benchmarked against R `survival::coxph` with Efron tie handling).
  - `pingouin` & `scipy`: Parametric / non-parametric hypothesis testing, ICC, Cohen's kappa.
  - `utils/sample_size_lib.py`: Exact closed-form formulas for two proportions, two means, and Schoenfeld survival event counts with drop-out adjustment.

### 2.3 Agentic Co-Pilot & Tool Architecture (`agent/`)
- Built upon `smolagents` (Hugging Face Agent Framework) with strict typed tools:
  - `PubMedEvidenceTool` (`agent/tools/tool_pubmed.py`): Queries NCBI E-Utilities with rate-limiting, extracts control event rates, hazard ratios, and formats Vancouver citations.
  - `SampleSizeTool` (`agent/tools/tool_sample_size.py`): Performs power/sample size calculations and outputs structured SAMPL-compliant justifications.
  - `SyntheticDataTool` (`agent/tools/tool_synthetic_data.py`): Generates Gaussian Copula cohorts with physiological boundary constraints ($\text{SBP} > \text{DBP} + 20$, CKD-EPI 2021 eGFR).
  - `ManuscriptEngine` (`agent/manuscript_engine.py`): Generates publication-ready Methods and Results sections via deterministic Jinja2 templates.

### 2.4 Reporting & Compliance Framework (`utils/reporting_checklists.py`)
- Automated compliance verification against international reporting standards:
  - **STROBE**: Observational cohort, case-control, and cross-sectional studies.
  - **CONSORT**: Parallel group randomized clinical trials.
  - **TRIPOD+AI**: Clinical prediction models and machine learning diagnostics.
  - **STARD**: Diagnostic accuracy studies.
  - **PRISMA**: Systematic reviews and meta-analyses.

---

## 3. Deployment & CI/CD Pipeline

```
Developer Local Environment
       │
       │  git push origin main
       ▼
GitHub Repository (NTWKKM/statiomed-ai)
       │
       ├──► .github/workflows/quality_check.yml (Pytest Suite + Syntax Verification)
       │
       └──► .github/workflows/sync_to_hf.yml (Mirroring to Hugging Face Hub)
                 │
                 │  git push https://hf.co/spaces/ntwkkm/statiomed-ai
                 ▼
Hugging Face Spaces (ntwkkm/statiomed-ai)
       │
       ▼
Container Build & ASGI Execution (Port 7860)
```
