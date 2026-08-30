# 🏥 StatioMed On-Prem PHI Sanitizer CLI

**Compliance Target**: Thailand PDPA (B.E. 2562) & HIPAA Safe Harbor Standard

This tool **MUST** be executed locally on the hospital workstation or research laboratory computer before uploading any dataset to the StatioMed AI cloud application.

---

## 🔒 What it does

1. **Strips Direct Identifiers**: Automatically detects and drops columns matching patient names, Hospital Numbers (HN), Citizen IDs, Social Security Numbers, Phone Numbers, Addresses, and Emails.
2. **Temporal Duration Shifting ($T_0=0$)**: Transforms sensitive calendar dates (e.g. Admission Date, ICU Transfer Date, Death Date) into relative elapsed days/hours from baseline, preserving exact time-to-event intervals for survival analysis while eliminating actual dates.
3. **Surrogate Key Generation**: Replaces original patient identifiers with cryptographically secure UUID4 values (`Deidentified_Patient_ID`).
4. **Supports Medical Formats**: Reads `.xlsx`, `.csv`, SPSS `.sav`, and Stata `.dta` formats.

---

## 💻 Usage

```bash
# Basic de-identification
python tools_local/phi_sanitizer_cli.py cohort_raw.xlsx -o cohort_sanitized.csv

# Survival Analysis dataset with baseline admission date conversion
python tools_local/phi_sanitizer_cli.py stroke_registry.sav -o stroke_clean.csv --t0 admission_date
```
