# 📋 StatioMed AI — Architecture Decision Records (ADRs) & Context

> **Project**: StatioMed AI — Clinical Research & Biostatistical Co-Pilot  
> **Repository**: [github.com/NTWKKM/statiomed-ai](https://github.com/NTWKKM/statiomed-ai)  
> **Production Space**: [huggingface.co/spaces/ntwkkm/statiomed-ai](https://huggingface.co/spaces/ntwkkm/statiomed-ai)

---

## ADR-001: Pure Native Gradio 6.x Blocks Architecture on Hugging Face Spaces

- **Status**: Accepted (Supercedes hybrid ASGI bridge)
- **Context**: Hugging Face Spaces provides first-class support, ZeroGPU hardware acceleration (`@spaces.GPU`), iframe embed stability, and native client APIs for Gradio applications. The previous hybrid ASGI bridge (mounting Shiny inside Gradio with JS redirects) suffered from redirect loops and WebSocket session drops on sleep/wake cycles.
- **Decision**: Refactor the frontend presentation layer into modular, deep Gradio 6.x Blocks views (`views/`) with session-isolated `gr.State(AppState())` containers, while preserving 100% of deterministic biostatistical computation in `utils/`.
- **Consequences**: Instant cold boot on Hugging Face Spaces without redirect loops, seamless ZeroGPU integration, robust mobile & iframe responsiveness, and cleaner dependency tree.

---

## ADR-002: Dual-Cloud CI/CD via GitHub Actions Hub Mirroring

- **Status**: Accepted
- **Context**: Code development occurs with Git version control, but continuous integration (Pytest, linting) and open-source collaboration are optimal on GitHub, whereas production hosting is on Hugging Face Spaces.
- **Decision**: Maintain GitHub (`origin`) as the primary repository and configure `.github/workflows/sync_to_hf.yml` with `HF_TOKEN` secret to push a mirror to Hugging Face Spaces automatically on every `main` branch push.
- **Consequences**: Single command `git push origin main` triggers automated test suites on GitHub CI and immediately deploys the verified application to Hugging Face Spaces.

---

## ADR-003: Zero-PHI Trust Boundary & Temporal Shifting ($T_0=0$)

- **Status**: Accepted
- **Context**: Clinical data containing patient identifiers (HN, Citizen ID, Names, Admission Dates) cannot be transmitted to external servers under Thailand PDPA and HIPAA regulations.
- **Decision**: Provide an isolated local CLI utility (`tools_local/phi_sanitizer_cli.py`) executed entirely on hospital workstations. Dates are converted into relative elapsed days from index $T_0$, and direct identifiers are salted and hashed into surrogate UUIDs.
- **Consequences**: Datasets uploaded to the cloud contain strictly anonymized numerical and categorical metrics, preserving survival modeling capacity while ensuring zero PHI exposure.

---

## ADR-004: Deterministic Statistical Execution & SAMPL Manuscript Templates

- **Status**: Accepted
- **Context**: Generative LLMs hallucinate numerical statistics, p-values, and confidence intervals when performing raw calculations.
- **Decision**: LLMs are restricted to study planning and qualitative synthesis. All statistical analyses (Cox PH, Kaplan-Meier, Logistic Regression, Sample Size) are executed deterministically using `statsmodels`, `lifelines`, `scikit-learn`, and `pingouin`. Manuscript Methods and Results sections are rendered using Jinja2 templates directly from computed dictionary results.
- **Consequences**: Zero mathematical hallucination, full reproducibility against benchmark standard (R `survival::coxph`), and conformity with SAMPL / EQUATOR guidelines.
