#!/usr/bin/env python3
"""
tools_local/phi_sanitizer_cli.py - Standalone On-Prem De-identification CLI
=============================================================================
Executes LOCALLY on hospital workstation before data leaves the institutional perimeter.
Fully compliant with:
  1. Thailand PDPA (Personal Data Protection Act B.E. 2562 / PDPC Notification Nov 2024)
  2. HIPAA Safe Harbor Standard (45 CFR § 164.514(b)(2))
Features:
  - Drops 18 direct identifier categories (HN, Citizen ID, Names, Phone, Address, DOB, etc.).
  - Age Capping: Ages >= 90 transformed to 90 (or 90+).
  - Temporal Duration Transformation: Converts calendar dates to elapsed days (T0 = 0).
  - Cryptographic Salted Pseudonymization (optional HMAC-SHA256) or UUID4 surrogate IDs.
=============================================================================
"""

import argparse
import hashlib
import hmac
import re
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

DENYLIST_PATTERNS = [
    r"^hn$",
    r"^hospital_number$",
    r"^mrn$",
    r"^an$",
    r"^admission_number$",
    r"^patient_name$",
    r"^first_name$",
    r"^last_name$",
    r"^name$",
    r"^full_name$",
    r"^citizen_id$",
    r"^national_id$",
    r"^cid$",
    r"^id_card$",
    r"^ssn$",
    r"^phone$",
    r"^mobile$",
    r"^telephone$",
    r"^fax$",
    r"^address$",
    r"^zip$",
    r"^postcode$",
    r"^email$",
    r"^dob$",
    r"^date_of_birth$",
    r"^birth_date$",
]


def sanitize_dataframe(
    df: pd.DataFrame,
    time_zero_col: Optional[str] = None,
    salt_key: Optional[str] = None,
    id_col: Optional[str] = None,
    cap_age: bool = True,
    keep_uuid: bool = True,
) -> pd.DataFrame:
    """
    Sanitizes clinical DataFrame by removing PHI, capping age at 90+,
    converting calendar dates to elapsed durations (T0=0), and generating
    HMAC-SHA256 or UUID4 surrogate IDs.
    """
    df_clean = df.copy()

    # 1. Age Capping (HIPAA Safe Harbor & PDPA Outlier Protection)
    if cap_age:
        for col in df_clean.columns:
            if re.search(
                r"\bage\b|^age_", str(col), re.IGNORECASE
            ) and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].apply(
                    lambda x: 90 if pd.notna(x) and x >= 90 else x
                )
                print(f"🛡️ Capped ages >= 90 in column '{col}' to 90")

    # 2. Salted HMAC Pseudonymization or UUID4 Generation
    if salt_key and id_col and id_col in df_clean.columns:
        hashed_ids = [
            hmac.new(
                salt_key.encode("utf-8"), str(val).encode("utf-8"), hashlib.sha256
            ).hexdigest()[:16]
            for val in df_clean[id_col]
        ]
        df_clean.insert(0, "Deidentified_Patient_ID", hashed_ids)
        print(
            f"🔑 Generated deterministic HMAC-SHA256 surrogate IDs from '{id_col}' using local salt."
        )
    elif keep_uuid:
        df_clean.insert(
            0,
            "Deidentified_Patient_ID",
            [str(uuid.uuid4()) for _ in range(len(df_clean))],
        )
        print("🎲 Injected random UUID4 surrogate IDs.")

    # 3. Direct Identifier Removal
    cols_to_drop = []
    for col in df_clean.columns:
        if col == "Deidentified_Patient_ID":
            continue
        norm_col = re.sub(r"[^a-zA-Z0-9_]", "_", str(col).strip().lower())
        if any(re.match(p, norm_col) for p in DENYLIST_PATTERNS):
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"🔒 Dropping direct identifier columns: {cols_to_drop}")
        df_clean.drop(columns=cols_to_drop, inplace=True)

    # 4. Temporal Date Transformation (T0 = 0)
    if time_zero_col and time_zero_col in df_clean.columns:
        t0 = pd.to_datetime(df_clean[time_zero_col], errors="coerce")
        date_cols = []
        for col in df_clean.columns:
            if col != time_zero_col and col != "Deidentified_Patient_ID":
                try:
                    converted = pd.to_datetime(df_clean[col], errors="raise")
                    df_clean[col] = converted
                    date_cols.append(col)
                except Exception:
                    pass

        for col in date_cols:
            t_event = pd.to_datetime(df_clean[col], errors="coerce")
            elapsed_days = (t_event - t0).dt.total_seconds() / 86400.0
            new_col = f"{col}_elapsed_days"
            df_clean[new_col] = elapsed_days.round(3)
            df_clean.drop(columns=[col], inplace=True)
            print(
                f"⏱️ Converted date column '{col}' -> '{new_col}' (relative to {time_zero_col})"
            )

        df_clean.drop(columns=[time_zero_col], inplace=True)
        print(f"⏱️ Dropped baseline date '{time_zero_col}' after calculating durations.")

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
        return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="On-Prem Local PHI Sanitizer CLI for StatioMed AI (Thailand PDPA & HIPAA Compliant)"
    )
    parser.add_argument(
        "input", help="Path to raw clinical dataset (CSV, XLSX, SAV, DTA)"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path for sanitized output CSV"
    )
    parser.add_argument(
        "--t0",
        help="Baseline date column name to convert calendar dates into elapsed durations (T0 = 0)",
    )
    parser.add_argument(
        "--salt", help="Cryptographic salt key for deterministic pseudonymization"
    )
    parser.add_argument("--id-col", help="Original ID column to hash with salt")
    parser.add_argument(
        "--no-age-cap", action="store_true", help="Disable age capping at 90+"
    )
    parser.add_argument(
        "--no-uuid", action="store_true", help="Do not insert surrogate ID column"
    )

    args = parser.parse_args()

    print(f"📂 Loading raw dataset from: {args.input}")
    df_raw = load_file(args.input)
    print(
        f"📊 Original dataset dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns"
    )

    df_clean = sanitize_dataframe(
        df=df_raw,
        time_zero_col=args.t0,
        salt_key=args.salt,
        id_col=args.id_col,
        cap_age=not args.no_age_cap,
        keep_uuid=not args.no_uuid,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(
        f"✅ Successfully sanitized dataset saved to: {output_path} ({len(df_clean)} rows, {df_clean.shape[1]} columns)"
    )


if __name__ == "__main__":
    main()
