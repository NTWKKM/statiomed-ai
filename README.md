---
title: StatioMed AI
emoji: 🏥
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
app_port: 7860
pinned: false
license: apache-2.0
---

# 🏥 StatioMed AI — Clinical Research & Biostatistical Co-Pilot

[![Quality CI](https://github.com/NTWKKM/statiomed-ai/actions/workflows/quality_check.yml/badge.svg)](https://github.com/NTWKKM/statiomed-ai/actions/workflows/quality_check.yml)
[![Sync to Hugging Face](https://github.com/NTWKKM/statiomed-ai/actions/workflows/sync_to_hf.yml/badge.svg)](https://github.com/NTWKKM/statiomed-ai/actions/workflows/sync_to_hf.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-brightgreen.svg)](https://www.python.org/)

> **StatioMed AI** is an agentic biostatistical co-pilot and clinical study design engine. It bridges on-prem de-identified electronic health record data with deterministic biostatistical execution and AI-assisted Statistical Analysis Plans (SAP).

---

## 🌟 Key Architecture & Capabilities

1. **Zero-PHI Trust Boundary (Thailand PDPA / HIPAA)**:
   - On-prem local de-identification CLI (`tools_local/phi_sanitizer_cli.py`) strips all direct identifiers before cloud transmission.
   - Temporal shifting ($T_0=0$) transforms calendar dates into elapsed days, enabling full time-to-event survival models without exposing hospital stay dates.
   - Cryptographic surrogate IDs (`uuid.uuid4()`).

2. **Deterministic Statistical Execution (Zero Hallucination)**:
   - LLM plans the study and interprets findings; all mathematical computation is executed by verified Python statistical libraries (`statsmodels`, `lifelines`, `scikit-learn`, `pingouin`).
   - Ground-truth regression tests benchmarked against **R 4.3.3 (`survival::coxph`)** with Efron tie handling.
   - Publication results generated via **Jinja2 templates** directly from computation dictionaries.

3. **Clinical Research Suite**:
   - **PICO & PubMed Parameter Extraction**: NCBI E-Utilities integration with rate-limiting and Vancouver citation generation.
   - **Sample Size & Power Engine**: Two-sample proportions, means, log-rank survival tests with $+15\text{--}20\%$ drop-out buffers.
   - **Synthetic Cohort Studio**: Gaussian Copulas with strict physiological boundary checks ($\text{SBP} > \text{DBP} + 20$, CKD-EPI eGFR).
   - **EQUATOR Reporting Checklists**: Built-in verification for STROBE, CONSORT, TRIPOD, and PRISMA.

---

## 🛠️ On-Prem Data Sanitization (Run Before Upload)

Execute locally on your hospital workstation:

```bash
# Sanitize raw clinical records and convert dates to elapsed days
python tools_local/phi_sanitizer_cli.py hospital_cohort.xlsx -o sanitized_cohort.csv --t0 admission_date
```

---

## 🚀 Local Development

```bash
# 1. Clone repository
git clone https://github.com/NTWKKM/statiomed-ai.git
cd statiomed-ai

# 2. Create virtual environment (Python 3.12+)
uv venv .venv --python 3.12
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Run tests
pytest tests/ -v

# 5. Launch web application
shiny run app.py --port 7860 --reload
```

---

## 📜 License & Compliance

Distributed under the **Apache-2.0 License**.  
Designed for compliance with **Thailand PDPA** and **ICMJE / SAMPL / EQUATOR** medical publication guidelines.
>>>>>>> 4fdeca8 (feat: initialize StatioMed AI clinical research suite (Gradio SDK))
