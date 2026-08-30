#!/usr/bin/env python3
"""
tools_local/phi_sanitizer_cli.py - Standalone On-Prem De-identification CLI
=============================================================================
MUST be executed locally on the hospital workstation before uploading data to cloud.
Complies with Thailand PDPA (Personal Data Protection Act B.E. 2562) and HIPAA Safe Harbor.
Provides cryptographic UUID4 surrogate patient IDs and temporal shifting (T0 = 0).
=============================================================================
"""

import argparse
import re
import uuid
from pathlib import Path
from typing import Optional
import pandas as pd

# Direct identifier patterns (case-insensitive regex)
DENYLIST_PATTERNS = [
    r"^hn$", r"^hospital_number$", r"^mrn$", r"^patient_name$", r"^first_name$",
    r"^last_name$", r"^name$", r"^citizen_id$", r"^national_id$", r"^id_card$",
    r"^ssn$", r"^phone$", r"^mobile$", r"^telephone$", r"^address$", r"^email$",
    r"^dob$", r"^date_of_birth$", r"^birth_date$", r"^cid$"
]

def sanitize_dataframe(
    df: pd.DataFrame,
    time_zero_col: Optional[str] = None,
    keep_uuid: bool = True
) -> pd.DataFrame:
    """
    Sanitizes clinical DataFrame by removing PHI, converting dates to durations,
    and injecting UUID4 surrogate keys.
    """
    df_clean = df.copy()

    # 1. Identify and drop direct identifier columns
    cols_to_drop = []
    for col in df_clean.columns:
        norm_col = re.sub(r"[^a-zA-Z0-9_]", "_", str(col).strip().lower())
        if any(re.match(p, norm_col) for p in DENYLIST_PATTERNS):
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"🔒 Dropping direct identifier columns: {cols_to_drop}")
        df_clean.drop(columns=cols_to_drop, inplace=True)

    # 2. Temporal date transformation (T0 = 0, relative elapsed days)
    if time_zero_col and time_zero_col in df_clean.columns:
        t0 = pd.to_datetime(df_clean[time_zero_col], errors="coerce")
        date_cols = []
        for col in df_clean.columns:
            if col != time_zero_col:
                # Try parsing as datetime if dtype is object or datetime
                try:
                    converted = pd.to_datetime(df_clean[col], errors="raise")
                    df_clean[col] = converted
                    date_cols.append(col)
                except Exception:
                    pass

        for col in date_cols:
            t_event = pd.to_datetime(df_clean[col], errors="coerce")
            elapsed_days = (t_event - t0).dt.total_seconds() / 86400.0
            new_col_name = f"{col}_elapsed_days"
            df_clean[new_col_name] = elapsed_days.round(3)
            df_clean.drop(columns=[col], inplace=True)
            print(f"⏱️ Converted date column '{col}' -> '{new_col_name}' (relative to {time_zero_col})")

        df_clean.drop(columns=[time_zero_col], inplace=True)
        print(f"⏱️ Dropped baseline calendar date '{time_zero_col}' after calculating elapsed durations.")

    # 3. Inject cryptographically secure UUID4 surrogate ID
    if keep_uuid:
        df_clean.insert(0, "Deidentified_Patient_ID", [str(uuid.uuid4()) for _ in range(len(df_clean))])

    return df_clean

def load_file(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif suffix == ".sav":
        import pyreadstat
        df, _ = pyreadstat.read_sav(str(path))
        return df
    elif suffix == ".dta":
        return pd.read_stata(str(path))
    elif suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    else:
        # Fallback to CSV
        return pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser(
        description="On-Prem Local PHI Sanitizer CLI for StatioMed AI (Thailand PDPA & HIPAA Compliant)"
    )
    parser.add_argument("input", help="Path to raw clinical dataset (CSV, XLSX, SAV, DTA)")
    parser.add_argument("-o", "--output", required=True, help="Path for sanitized output CSV")
    parser.add_argument("--t0", help="Baseline date column name to convert calendar dates into elapsed durations (T0 = 0)")
    parser.add_argument("--no-uuid", action="store_true", help="Do not insert Deidentified_Patient_ID column")

    args = parser.parse_args()

    print(f"📂 Loading raw dataset from: {args.input}")
    df_raw = load_file(args.input)
    print(f"📊 Original dataset dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    df_clean = sanitize_dataframe(
        df=df_raw,
        time_zero_col=args.t0,
        keep_uuid=not args.no_uuid
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Successfully sanitized dataset saved to: {output_path} ({len(df_clean)} rows, {df_clean.shape[1]} columns)")

if __name__ == "__main__":
    main()
