"""
core/constants.py - StatioMed AI Clinical & Biostatistical Constants
====================================================================
Canonical clinical constants, standard units, EQUATOR reporting guideline
identifiers, SAMPL formatting thresholds, and Zero-PHI boundary rules.
====================================================================
"""

from __future__ import annotations

from typing import Final

# =====================================================================
# 1. APP & PLATFORM METADATA
# =====================================================================
APP_NAME: Final[str] = "StatioMed AI"
APP_VERSION: Final[str] = "2.0.0"
APP_DESCRIPTION: Final[str] = (
    "Clinical Research, Biostatistical Analysis & Manuscript Generation Engine"
)
DEFAULT_SERVER_PORT: Final[int] = 7860
DEFAULT_SERVER_NAME: Final[str] = "0.0.0.0"

# =====================================================================
# 2. CLINICAL UNITS & METRICS
# =====================================================================
CLINICAL_UNITS: Final[dict[str, str]] = {
    "sbp": "mmHg",
    "dbp": "mmHg",
    "map": "mmHg",
    "heart_rate": "bpm",
    "respiratory_rate": "/min",
    "temperature": "°C",
    "spo2": "%",
    "egfr": "mL/min/1.73m²",
    "creatinine": "mg/dL",
    "bun": "mg/dL",
    "glucose": "mg/dL",
    "hba1c": "%",
    "alt": "U/L",
    "ast": "U/L",
    "bilirubin": "mg/dL",
    "albumin": "g/dL",
    "sodium": "mEq/L",
    "potassium": "mEq/L",
    "chloride": "mEq/L",
    "bicarbonate": "mEq/L",
    "hemoglobin": "g/dL",
    "platelets": "x10³/µL",
    "wbc": "x10³/µL",
    "crp": "mg/L",
    "lactate": "mmol/L",
    "weight": "kg",
    "height": "cm",
    "bmi": "kg/m²",
}

# =====================================================================
# 3. STATISTICAL & REPORTING GUIDELINES (EQUATOR NETWORK)
# =====================================================================
EQUATOR_GUIDELINES: Final[dict[str, dict[str, str]]] = {
    "CONSORT": {
        "name": "CONSORT 2010",
        "description": "Consolidated Standards of Reporting Trials (Randomized Controlled Trials)",
        "url": "https://www.equator-network.org/reporting-guidelines/consort/",
    },
    "STROBE": {
        "name": "STROBE",
        "description": "Strengthening the Reporting of Observational Studies in Epidemiology (Cohort, Case-Control, Cross-Sectional)",
        "url": "https://www.equator-network.org/reporting-guidelines/strobe/",
    },
    "TRIPOD_AI": {
        "name": "TRIPOD+AI 2024",
        "description": "Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis Or Diagnosis - Artificial Intelligence",
        "url": "https://www.equator-network.org/reporting-guidelines/tripod-ai/",
    },
    "STARD": {
        "name": "STARD 2015",
        "description": "Standards for Reporting of Diagnostic Accuracy Studies",
        "url": "https://www.equator-network.org/reporting-guidelines/stard-statement-2015/",
    },
    "PRISMA": {
        "name": "PRISMA 2020",
        "description": "Preferred Reporting Items for Systematic Reviews and Meta-Analyses",
        "url": "https://www.equator-network.org/reporting-guidelines/prisma/",
    },
}

# =====================================================================
# 4. STATISTICAL THRESHOLDS & SAMPL DEFAULTS
# =====================================================================
DEFAULT_ALPHA: Final[float] = 0.05
DEFAULT_POWER: Final[float] = 0.80
DEFAULT_CALIPER_SD: Final[float] = 0.20
DEFAULT_DROPOUT_RATE_PCT: Final[float] = 10.0
DEFAULT_IMPUTATION_MICE_SETS: Final[int] = 5

P_VALUE_FORMAT_THRESHOLDS: Final[dict[str, float]] = {
    "min_exact": 0.001,
    "max_exact": 0.999,
}

# =====================================================================
# 5. ZERO-PHI SECURITY RESTRICTIONS
# =====================================================================
PHI_RESTRICTED_COLUMNS: Final[list[str]] = [
    "hn",
    "hospital_number",
    "mrn",
    "citizen_id",
    "national_id",
    "id_card",
    "first_name",
    "last_name",
    "full_name",
    "name",
    "patient_name",
    "dob",
    "date_of_birth",
    "birth_date",
    "phone",
    "phone_number",
    "telephone",
    "mobile",
    "address",
    "street",
    "zipcode",
    "postal_code",
    "ssn",
]
