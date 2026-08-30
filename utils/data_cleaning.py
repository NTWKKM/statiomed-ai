"""
🧹 Enhanced Data Cleaning Utilities
Comprehensive data cleaning with robust error handling, validation, and logging.

This module provides:
- Robust numeric cleaning with comprehensive error handling
- Outlier detection and handling
- Data type validation and conversion
- Data quality validation
- Input validation and edge case handling
- Informative logging and error messages
- Support for various data formats

Driven by central configuration from config.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config import CONFIG
from logger import get_logger

logger = get_logger(__name__)


class DataCleaningError(Exception):
    """Custom exception for data cleaning errors."""


class DataValidationError(Exception):
    """Custom exception for data validation errors."""


def validate_input_data(data: Any) -> pd.DataFrame:
    """
    Validate input data and convert to DataFrame if needed.

    Parameters:
        data: Input data (DataFrame, dict, list, or similar)

    Returns:
        pd.DataFrame: Validated DataFrame

    Raises:
        DataValidationError: If input cannot be converted to DataFrame
    """
    try:
        match data:
            case pd.DataFrame():
                df = data.copy()
            case dict():
                df = pd.DataFrame(data)
            case list():
                df = pd.DataFrame(data)
            case _:
                raise DataValidationError(f"Unsupported data type: {type(data)}")

        if df.empty:
            logger.warning("Input data is empty")

        logger.debug(
            "Validated input data: %s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )
        return df

    except Exception as e:
        logger.exception("Failed to validate input data")
        raise DataValidationError(f"Input validation failed: {e}") from e


def load_data_robust(file_path: str | Path) -> pd.DataFrame:
    """
    Robustly load data from CSV or Excel files with extensive error handling and format detection.

    Capabilities:
    - CSV:
        - Auto-detects separators (,, ;, tab, |)
        - Handles various encodings (utf-8, utf-8-sig, cp1252, latin1, cp874/tis-620 for Thai)
        - Handles bad lines and mixed types
    - Excel:
        - Support for .xlsx and .xlsm
        - Auto-trims headers
    - General:
        - Sanitizes column names (strip whitespace)
        - Basic empty row/column cleanup

    Parameters:
        file_path: Path to the file (str or Path object)

    Returns:
        pd.DataFrame: Loaded and sanitized DataFrame

    Raises:
        DataValidationError: If file format is invalid or file cannot be read
    """
    path = Path(file_path)
    if not path.exists():
        raise DataValidationError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    df = None
    error_log = []

    # --- CSV Handling ---
    if suffix == ".csv":
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "cp874", "tis-620"]
        separators = [None, ",", ";", "\t", "|"]

        # Try various encoding/separator combinations
        for encoding in encodings:
            for sep in separators:
                try:
                    # Use python engine for more stable separator detection if sep is None
                    engine = "python" if sep is None else "c"

                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        sep=sep,
                        engine=engine,
                        # Removed on_bad_lines="skip" to ensure full data integrity.
                        # We want to catch errors rather than silently dropping rows.
                    )

                    # Basic validation: If it read into 1 column but we expect more, maybe wrong separator
                    if df.shape[1] == 1 and sep is None:
                        # User didn't specify separator, and engine='python' likely failed to sniff it correctly.
                        # Try explicit common separators to see if we get more columns.
                        for trial_sep in [",", ";", "\t", "|"]:
                            try:
                                # Use C engine for speed on re-try, or python if needed
                                df_trial = pd.read_csv(
                                    path, sep=trial_sep, encoding=encoding, nrows=5
                                )
                                if df_trial.shape[1] > 1:
                                    # Found a better separator, re-read full file
                                    logger.info(
                                        "Auto-detected better separator: '%s'",
                                        trial_sep,
                                    )
                                    df = pd.read_csv(
                                        path, sep=trial_sep, encoding=encoding
                                    )
                                    break
                            except Exception:
                                continue

                    logger.info(
                        "Successfully read CSV with encoding=%s, sep=%s", encoding, sep
                    )
                    break
                except Exception as e:
                    error_log.append(f"CSV fail (enc={encoding}, sep={sep}): {e}")
            if df is not None:
                break

        if df is None:
            logger.error("Failed to read CSV with all attempts. Errors: %s", error_log)
            raise DataValidationError(
                f"Could not read CSV file. Tried encodings: {encodings}"
            )

    # --- Excel Handling ---
    elif suffix in [".xlsx", ".xlsm"]:
        try:
            # Default to first sheet
            df = pd.read_excel(path)
            logger.info("Successfully read Excel file")
        except Exception as e:
            logger.exception("Failed to read Excel file")
            raise DataValidationError(f"Failed to read Excel file: {e}")

    else:
        raise DataValidationError(f"Unsupported file format: {suffix}")

    if df is None:
        raise DataValidationError("Unknown error: DataFrame is None after loading")

    # --- Post-Load Sanitization ---

    # 0. Header Heuristic: Check for metadata rows above header
    # If >50% of columns are "Unnamed", scan first 10 rows for a better candidate
    if sum(1 for c in df.columns if "Unnamed:" in str(c)) > len(df.columns) * 0.5:
        logger.info(
            "Potential malformed header detected (many 'Unnamed' columns). Scanning rows..."
        )
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            # Criterion: Row must have >50% non-null values
            if row.count() >= len(df.columns) * 0.5:
                # Promote row to header
                new_cols = row.fillna(f"Col_{i}").astype(str).str.strip()
                # Ensure unique columns to prevent pandas errors
                if new_cols.duplicated().any():
                    # Deduplicate: Col, Col.1, Col.2
                    new_cols = pd.Index(new_cols)
                    counts = new_cols.value_counts()
                    if counts.max() > 1:
                        # Simple deduplication strategy
                        seen = {}
                        deduped = []
                        for c in new_cols:
                            if c not in seen:
                                seen[c] = 0
                                deduped.append(c)
                            else:
                                seen[c] += 1
                                deduped.append(f"{c}_{seen[c]}")
                        new_cols = deduped

                df = df.iloc[i + 1 :].copy()
                df.columns = new_cols
                df = df.reset_index(drop=True)
                logger.info("Promoted row %d to header", i)
                break

    # 1. Clean Column Names
    df.columns = df.columns.astype(str).str.strip()

    # Removed: df.dropna(how="all")
    # reason: User strictly requested NO automatic data deletion/modification.
    # We intentionally keep empty rows/cols so they show up in quality reports if needed,
    # or the user can remove them explicitly.

    # 3. Reset index
    df = df.reset_index(drop=True)

    logger.info("Loaded data shape: %s", df.shape)
    return df


def clean_numeric(
    val: Any, handle_special_chars: bool = True, remove_whitespace: bool = True
) -> float:
    """
    Convert value to float with robust error handling.

    Handles:
    - Missing values (returns np.nan)
    - Strings with comparison operators (>, <)
    - Comma separators in thousands
    - Whitespace
    - Invalid data types
    - Special characters

    Parameters:
        val: Value to convert to float
        handle_special_chars: Whether to handle special characters like >, <
        remove_whitespace: Whether to remove whitespace

    Returns:
        float: Cleaned numeric value or np.nan if conversion fails

    Examples:
        >>> clean_numeric(">100")
        100.0
        >>> clean_numeric("1,234.56")
        1234.56
        >>> clean_numeric(None)
        nan
    """
    if pd.isna(val):
        return np.nan

    try:
        # Convert to string for processing
        s = str(val)

        # Remove whitespace if enabled
        if remove_whitespace:
            s = s.strip()

        is_negative_parentheses = False
        # Handle special characters if enabled
        if handle_special_chars:
            if s.startswith("(") and s.endswith(")"):
                is_negative_parentheses = True
                
            # Remove common comparison operators
            s = s.replace(">", "").replace("<", "")
            # Remove comma separators
            s = s.replace(",", "")
            # Remove other common non-numeric characters
            s = s.replace("$", "").replace("€", "").replace("£", "")
            s = s.replace("%", "").replace("(", "").replace(")", "")

        # Handle empty string after cleaning
        if not s:
            return np.nan

        # Try to convert to float
        result = float(s)
        if is_negative_parentheses:
            result = -abs(result)

        logger.debug("Cleaned numeric: %s -> %s", val, result)
        return result

    except (TypeError, ValueError, AttributeError) as e:
        logger.debug("Failed to clean numeric value '%s': %s", val, e)
        return np.nan


def clean_numeric_vector(series: pd.Series | np.ndarray | list[Any]) -> pd.Series:
    """
    Vectorized numeric cleaning for entire series with comprehensive error handling.

    Uses pandas string operations for optimal performance (10x faster than apply).
    It effectively handles cell-level errors by coercing them to NaN.

    Parameters:
        series: Input data series (pd.Series, np.ndarray, or list)

    Returns:
        pd.Series: Cleaned numeric series

    Examples:
        >>> data = pd.Series([">100", "1,234.56", None, "abc"])
        >>> clean_numeric_vector(data)
        0    100.0
        1    1234.56
        2       NaN
        3       NaN
        dtype: float64
    """
    try:
        # Convert to Series if needed
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        # Check if all values are NA
        if series.isna().all():
            logger.warning("All values in series are NA")
            return pd.Series(dtype=float, index=series.index)

        # Vectorized string operations
        s = series.astype(str)
        s = s.str.strip()
        s = s.str.replace(">", "", regex=False)
        s = s.str.replace("<", "", regex=False)
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace("$", "", regex=False)
        s = s.str.replace("€", "", regex=False)
        s = s.str.replace("£", "", regex=False)
        s = s.str.replace("%", "", regex=False)

        # Convert to numeric with error coercion
        # This is the key step: 'coerce' turns problematic cells into NaN
        result = pd.to_numeric(s, errors="coerce")

        # Log conversion statistics
        na_count = result.isna().sum()
        total_count = len(result)
        if na_count > 0:
            logger.debug(
                "Converted %d/%d values successfully (%d NA)",
                total_count - na_count,
                total_count,
                na_count,
            )
        else:
            logger.debug("Successfully cleaned all %d values", total_count)

        return result

    except Exception as e:
        logger.exception("Failed in clean_numeric_vector")
        raise DataCleaningError(f"Vectorized cleaning failed: {e}") from e


def detect_outliers(
    series: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Detect outliers in numeric series using specified method.

    Parameters:
        series: Input numeric series
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
                    - IQR: multiplier (default 1.5)
                    - Z-score: threshold (default 3.0)

    Returns:
        tuple[pd.Series, Dict]: Boolean mask of outliers and statistics dict
    """
    try:
        # Ensure input is a Series
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        # Clean the series first
        cleaned = clean_numeric_vector(series)
        cleaned = cleaned.dropna()

        if len(cleaned) == 0:
            logger.warning("Cannot detect outliers: series is empty after cleaning")
            return pd.Series(False, index=series.index), {}

        outlier_mask = pd.Series(False, index=series.index)
        stats = {}

        match method:
            case "iqr":
                # IQR method
                Q1 = cleaned.quantile(0.25)
                Q3 = cleaned.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (cleaned < lower_bound) | (cleaned > upper_bound)
                outlier_mask = outlier_mask.reindex(series.index, fill_value=False)

                stats = {
                    "method": "iqr",
                    "Q1": Q1,
                    "Q3": Q3,
                    "IQR": IQR,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_count": outlier_mask.sum(),
                    "outlier_pct": (outlier_mask.sum() / len(series)) * 100,
                }

            case "zscore":
                # Z-score method
                mean = cleaned.mean()
                std = cleaned.std()

                if std == 0:
                    logger.warning(
                        "Standard deviation is zero, cannot use z-score method"
                    )
                    return outlier_mask, stats

                z_scores = np.abs((cleaned - mean) / std)
                outlier_mask = z_scores > threshold
                outlier_mask = outlier_mask.reindex(series.index, fill_value=False)

                stats = {
                    "method": "zscore",
                    "mean": mean,
                    "std": std,
                    "threshold": threshold,
                    "outlier_count": outlier_mask.sum(),
                    "outlier_pct": (outlier_mask.sum() / len(series)) * 100,
                }

            case _:
                raise ValueError(f"Unknown outlier detection method: {method}")

        logger.info(
            "Detected %d outliers (%.2f%%) using %s method",
            stats["outlier_count"],
            stats["outlier_pct"],
            method,
        )
        return outlier_mask, stats

    except Exception as e:
        logger.exception("Failed to detect outliers")
        raise DataCleaningError(f"Outlier detection failed: {e}") from e


def handle_outliers(
    series: pd.Series, method: str = "iqr", action: str = "flag", **kwargs
) -> pd.Series:
    """
    Handle outliers in numeric series.

    Parameters:
        series: Input numeric series
        method: Detection method ('iqr' or 'zscore')
        action: How to handle outliers ('flag', 'remove' (set to NaN), 'winsorize', 'cap')
        **kwargs: Additional arguments for detect_outliers

    Returns:
        pd.Series: Series with outliers handled
    """
    try:
        # Ensure input is a Series
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        # Clean the series first
        cleaned = clean_numeric_vector(series)

        # Detect outliers
        outlier_mask, stats = detect_outliers(series, method=method, **kwargs)
        if not stats:
            logger.info("No outliers detected (empty or non-numeric series)")
            return cleaned

        match action:
            case "flag" | "remove":
                # Flag/Remove outliers (set to NaN)
                result = cleaned.copy()
                result[outlier_mask] = np.nan
                logger.info(
                    "Handled %d outliers via '%s' (set to NaN)",
                    stats["outlier_count"],
                    action,
                )

            case "winsorize":
                # Winsorize outliers to boundary values
                result = cleaned.copy().astype(float)

                if stats["method"] == "iqr":
                    lower_bound = stats["lower_bound"]
                    upper_bound = stats["upper_bound"]
                    result = result.clip(lower=lower_bound, upper=upper_bound)
                    logger.info(
                        "Winsorized %d outliers to [%.2f, %.2f]",
                        stats["outlier_count"],
                        lower_bound,
                        upper_bound,
                    )
                else:
                    raise ValueError(
                        f"Winsorizing requires 'iqr' method, got '{stats['method']}'"
                    )

            case "cap":
                # Cap outliers to nearest non-outlier value
                result = cleaned.copy().astype(float)

                if stats["method"] == "iqr":
                    lower_bound = stats["lower_bound"]
                    upper_bound = stats["upper_bound"]
                    result[result < lower_bound] = lower_bound
                    result[result > upper_bound] = upper_bound
                    logger.info("Capped %d outliers at bounds", stats["outlier_count"])
                else:
                    raise ValueError(
                        f"Capping requires 'iqr' method, got '{stats['method']}'"
                    )

            case _:
                raise ValueError(f"Unknown outlier handling action: {action}")

        return result

    except Exception as e:
        logger.exception("Failed to handle outliers")
        raise DataCleaningError(f"Outlier handling failed: {e}") from e


def robust_sort_key(x: Any) -> tuple:
    """
    Sort key placing numeric values first, then strings, then NA.
    """
    try:
        if pd.isna(x):
            return (2, "")
        val = float(x)
        return (0, val)
    except (ValueError, TypeError):
        return (1, str(x))


def is_continuous_variable(series: pd.Series) -> bool:
    """
    Determine if a variable should be treated as continuous based on CONFIG.
    """
    try:
        # Get threshold from config
        threshold = CONFIG.get("analysis.var_detect_threshold", 10)
        decimal_pct = CONFIG.get("analysis.var_detect_decimal_pct", 0.30)

        unique_count = series.nunique()

        # If unique values exceed threshold, treat as continuous
        if unique_count > threshold:
            return True

        # Check if most values are decimal (float-like)
        if series.dtype in [np.float64, np.float32]:
            return True

        # Check percentage of decimal values
        try:
            cleaned = clean_numeric_vector(series)
            non_na = cleaned.dropna()
            if len(non_na) > 0:
                decimal_count = (non_na % 1 != 0).sum()
                decimal_ratio = decimal_count / len(non_na)
                if decimal_ratio >= decimal_pct:
                    return True
        except (ValueError, TypeError, AttributeError):
            # Numeric conversion failed, treat as non-continuous
            logger.debug("Could not determine decimal ratio for series")

        return False

    except (TypeError, ValueError, AttributeError) as e:
        logger.warning("Failed to determine if variable is continuous: %s", e)
        return False


def validate_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Perform comprehensive data quality validation.
    """
    results = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": {},
        "missing_pct": {},
        "warnings": [],
        "errors": [],
    }

    try:
        # Handle empty DataFrame
        if df.empty:
            results["summary"] = {
                "total_rows": 0,
                "total_columns": 0,
                "total_missing": 0,
                "overall_missing_pct": 0,
                "has_warnings": False,
                "has_errors": False,
            }
            logger.warning("DataFrame is empty")
            return results

        # Check missing data
        missing_threshold = CONFIG.get("analysis.missing_threshold_pct", 50)

        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100

            results["missing_counts"][col] = missing_count
            results["missing_pct"][col] = round(missing_pct, 2)

            if missing_pct > missing_threshold:
                warning = f"Column '{col}' has {missing_pct:.2f}% missing data (threshold: {missing_threshold}%)"
                results["warnings"].append(warning)
                logger.warning(warning)

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warning = f"Found {duplicate_count} duplicate rows ({(duplicate_count / len(df)) * 100:.2f}%)"
            results["warnings"].append(warning)
            logger.warning(warning)

        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            results["warnings"].append(f"Completely empty columns: {empty_cols}")
            logger.warning(f"Completely empty columns: {empty_cols}")

        # Summary
        total_missing = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_missing_pct = (
            (total_missing / total_cells) * 100 if total_cells > 0 else 0
        )

        results["summary"] = {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "total_missing": total_missing,
            "overall_missing_pct": round(overall_missing_pct, 2),
            "has_warnings": len(results["warnings"]) > 0,
            "has_errors": len(results["errors"]) > 0,
        }

        logger.info(
            "Data quality validation complete: %.2f%% missing data", overall_missing_pct
        )
        return results

    except Exception as e:
        logger.exception("Data quality validation failed")
        raise DataValidationError(f"Validation failed: {e}") from e


def clean_dataframe(
    df: pd.DataFrame,
    handle_outliers_flag: bool = False,
    outlier_method: str = "iqr",
    outlier_action: str = "flag",
    validate_quality: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Comprehensive data cleaning for entire DataFrame.

    This function:
    1. Validates input data
    2. Cleans all numeric columns (handling individual cell errors)
    3. Handles outliers if requested
    4. Validates data quality if requested
    5. Returns cleaned DataFrame and cleaning report

    CRITICAL: Original DataFrame is NEVER modified. A copy is returned.
    """
    logger.info("Starting comprehensive data cleaning...")

    cleaning_report = {
        "original_shape": df.shape,
        "cleaning_steps": [],
        "column_reports": {},
        "quality_report": {},
    }

    try:
        # Validate input
        df_validated = validate_input_data(df)
        cleaning_report["cleaning_steps"].append("Input validation")

        # Create a copy for cleaning
        df_cleaned = df_validated.copy()
        logger.info("Created copy for cleaning: %s", df_cleaned.shape)

        # Use lower threshold to be more aggressive in finding numeric columns
        # Default lowered to 30% to support granular cleaning of dirty data
        numeric_threshold = CONFIG.get("analysis.numeric_conversion_threshold", 0.30)

        # Clean each column
        for col in df_cleaned.columns:
            col_report = {"original_dtype": str(df_cleaned[col].dtype), "actions": []}

            try:
                # Try to clean as numeric if it's object/category
                if (
                    pd.api.types.is_object_dtype(df_cleaned[col])
                    or pd.api.types.is_string_dtype(df_cleaned[col])
                    or isinstance(df_cleaned[col].dtype, pd.CategoricalDtype)
                ):
                    original_na = df_cleaned[col].isna().sum()

                    # Clean numeric values (this sets problematic cells to NaN)
                    cleaned_col = clean_numeric_vector(df_cleaned[col])

                    # Check if enough values are numeric
                    non_na_count = cleaned_col.notna().sum()
                    total_count = len(cleaned_col)
                    non_na_ratio = non_na_count / total_count if total_count > 0 else 0

                    # UPDATED LOGIC: Be more permissive (threshold > 0.3)
                    # If the column has at least 30% valid numbers, assume it is numeric
                    # and treat the rest as missing data (NaN) instead of rejecting the whole column.
                    if non_na_ratio > numeric_threshold:
                        new_na = cleaned_col.isna().sum()
                        coerced_na = new_na - original_na

                        df_cleaned[col] = cleaned_col.values

                        action_msg = (
                            f"Converted to numeric ({non_na_ratio * 100:.1f}% valid)"
                        )
                        if coerced_na > 0:
                            action_msg += f", set {coerced_na} problematic cells to NaN"

                        col_report["actions"].append(action_msg)
                        col_report["new_dtype"] = "float64"
                    else:
                        col_report["actions"].append(
                            "Kept as object (insufficient numeric signal)"
                        )

                # Handle outliers for numeric columns
                if handle_outliers_flag and pd.api.types.is_numeric_dtype(
                    df_cleaned[col]
                ):
                    df_cleaned[col] = handle_outliers(
                        df_cleaned[col], method=outlier_method, action=outlier_action
                    )
                    col_report["actions"].append(
                        f"Handled outliers ({outlier_method} method, {outlier_action} action)"
                    )

                cleaning_report["column_reports"][col] = col_report

            except Exception as e:
                warning = f"Failed to clean column '{col}': {e}"
                logger.warning(warning)
                col_report["error"] = str(e)
                cleaning_report["column_reports"][col] = col_report

        cleaning_report["cleaned_shape"] = df_cleaned.shape
        cleaning_report["cleaning_steps"].append("Column-wise cleaning")

        # Validate data quality
        if validate_quality:
            quality_report = validate_data_quality(df_cleaned)
            cleaning_report["quality_report"] = quality_report
            cleaning_report["cleaning_steps"].append("Data quality validation")

        cleaning_report["cleaning_steps"].append("Cleaning complete")

        logger.info("Data cleaning complete: %s -> %s", df.shape, df_cleaned.shape)
        return df_cleaned, cleaning_report

    except Exception as e:
        logger.exception("Data cleaning failed")
        raise DataCleaningError(f"Failed to clean DataFrame: {e}") from e


def get_cleaning_summary(report: dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the cleaning report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DATA CLEANING SUMMARY")
    lines.append("=" * 60)

    lines.append(f"\nOriginal shape: {report['original_shape']}")
    lines.append(f"Cleaned shape: {report['cleaned_shape']}")

    lines.append("\nCleaning steps:")
    for i, step in enumerate(report["cleaning_steps"], 1):
        lines.append(f"  {i}. {step}")

    if report["column_reports"]:
        lines.append("\nColumn reports:")
        for col, col_report in report["column_reports"].items():
            lines.append(f"\n  Column: {col}")
            lines.append(f"    Original dtype: {col_report['original_dtype']}")
            if "actions" in col_report:
                for action in col_report["actions"]:
                    lines.append(f"    - {action}")
            if "error" in col_report:
                lines.append(f"    ERROR: {col_report['error']}")

    if report["quality_report"]:
        qr = report["quality_report"]
        lines.append("\nData quality:")
        lines.append(f"  Overall missing: {qr['summary']['overall_missing_pct']}%")
        if qr["warnings"]:
            lines.append(f"  Warnings: {len(qr['warnings'])}")
            for warning in qr["warnings"]:
                lines.append(f"    - {warning}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


# ============================================================
# Missing Data Management Utilities
# ============================================================


def apply_missing_values_to_df(
    df: pd.DataFrame,
    var_meta: dict[str, Any],
    missing_codes: list[Any] | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Replace user-specified missing value codes with NaN.

    Parameters:
        df: Input DataFrame
        var_meta: Variable metadata with 'missing_values' per variable
        missing_codes: Global missing value codes (list) or per-column (dict)

    Returns:
        DataFrame with missing codes replaced by NaN
    """
    df_copy = df.copy()
    var_meta = var_meta or {}

    for col in df_copy.columns:
        # Get variable-specific missing values from metadata
        if col in var_meta and "missing_values" in var_meta[col]:
            missing_vals = var_meta[col]["missing_values"]
        elif isinstance(missing_codes, dict):
            missing_vals = missing_codes.get(col, [])
        else:
            missing_vals = missing_codes or []

        # Normalize missing_vals to list if scalar or unexpected type
        if isinstance(missing_vals, (str, bytes)) or not isinstance(
            missing_vals, (list, tuple, set, np.ndarray)
        ):
            missing_vals = [missing_vals]

        if len(missing_vals) == 0:
            continue

        # Normalize missing codes across string/numeric representations
        is_numeric = pd.api.types.is_numeric_dtype(df_copy[col])
        normalized_vals = set()
        for val in missing_vals:
            if pd.isna(val):
                continue
            normalized_vals.add(val)
            normalized_vals.add(str(val))
            if is_numeric:
                try:
                    normalized_vals.add(float(val))
                except (ValueError, TypeError):
                    pass

        df_copy[col] = df_copy[col].replace(list(normalized_vals), np.nan)

    logger.debug(f"Applied missing value codes to {len(df_copy.columns)} columns")
    return df_copy


def detect_missing_in_variable(
    series: pd.Series,
    missing_codes: list[Any] | None = None,
    already_normalized: bool = False,
) -> dict[str, Any]:
    """
    Detect and count missing values in a single variable.

    Parameters:
        series: Input pandas Series
        missing_codes: Optional list of values to treat as missing
        already_normalized: If True, assume coded values are already converted to NaN

    Returns:
        Dictionary with missing value statistics
    """
    total_count = len(series)

    # Count entries that are already standard NaN
    missing_nan_count = int(series.isna().sum())

    # Count user-specified missing codes
    missing_coded_count = 0
    if missing_codes and not already_normalized:
        for code in missing_codes:
            # Count occurrences of this code (only if not already NaN)
            count = int((series == code).sum())
            missing_coded_count += count

    # Total missing = NaN + coded missing
    # If already_normalized=True, coded missing will be 0 and NaN will include them
    total_missing = missing_nan_count + missing_coded_count

    missing_pct = (total_missing / total_count * 100) if total_count > 0 else 0.0

    return {
        "total_count": total_count,
        "missing_count": total_missing,
        "missing_pct": round(missing_pct, 2),
        "missing_coded_count": missing_coded_count,
        "missing_nan_count": missing_nan_count,
        "valid_count": total_count - total_missing,
    }


def get_missing_summary_df(
    df: pd.DataFrame,
    var_meta: dict[str, Any],
    missing_codes: list[Any] | None = None,
    already_normalized: bool = False,
) -> pd.DataFrame:
    """
    Builds a per-variable missing-data summary DataFrame.

    Parameters:
        df (pd.DataFrame): Input table to summarize.
        var_meta (dict[str, Any]): Optional variable metadata; if a column name is present, its "type" key will be used to label the variable.
        missing_codes (list[Any] | None): Global list of coded missing values to count as missing when normal NaNs are not present.
        already_normalized (bool): If True, treat coded missing values in `df` as already converted to `NaN` and do not re-check `missing_codes`.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - Variable: column name from `df`.
            - Type: variable type from `var_meta` if available, otherwise "Continuous" for numeric dtypes or "Categorical".
            - N_Total: total number of observations for the variable.
            - N_Valid: number of non-missing observations.
            - N_Missing: number of missing observations (including coded missings when applicable).
            - Pct_Missing: missing percentage formatted as a string with a percent sign (sorted descending by this value).
    """
    # Initialize list to store summary data
    summary_data = []

    # Process each column
    for col in df.columns:
        # Determine variable type from metadata
        var_type = "Unknown"
        if col in var_meta:
            var_type = var_meta[col].get("type", "Unknown")
        elif pd.api.types.is_numeric_dtype(df[col]):
            var_type = "Continuous"
        else:
            var_type = "Categorical"

        # Get missing stats
        stats = detect_missing_in_variable(
            df[col], missing_codes=missing_codes, already_normalized=already_normalized
        )

        summary_data.append(
            {
                "Variable": col,
                "Type": var_type,
                "N_Total": stats["total_count"],
                "N_Valid": stats["valid_count"],
                "N_Missing": stats["missing_count"],
                # FIX: Format percentage as string with % to match test expectations
                "Pct_Missing": stats["missing_pct"],
            }
        )

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Sort by missing percentage (descending) before formatting for display
    summary_df = summary_df.sort_values("Pct_Missing", ascending=False)
    summary_df["Pct_Missing"] = summary_df["Pct_Missing"].map(lambda v: f"{v}%")

    return summary_df


# ============================================================
# Advanced Data Cleaning & Imputation
# ============================================================


def impute_missing_data(
    df: pd.DataFrame, cols: list[str], method: str = "knn", **kwargs
) -> pd.DataFrame:
    """
    Imputes missing values for specified numeric columns using the chosen strategy.

    Performs imputation only on numeric columns among `cols`; non-numeric columns are skipped with a warning.
    Supported methods:
    - "knn": K-Nearest Neighbors imputation (uses `n_neighbors`).
    - "mice": Multiple imputation by chained equations (uses `random_state`, `max_iter`).
    - "mean": Column-wise mean imputation.
    - "median": Column-wise median imputation.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        cols (list[str]): Columns to consider for imputation; must exist in `df`. Non-numeric columns in this list are ignored.
        method (str): Imputation method to apply. One of "knn", "mice", "mean", or "median".
        **kwargs: Additional method-specific options:
            - n_neighbors (int): For "knn" (default 5).
            - random_state (int): For "mice" (default 42).
            - max_iter (int): For "mice" (default 10).

    Returns:
        pd.DataFrame: A copy of `df` with missing values imputed for the processed numeric columns.

    Raises:
        ValueError: If any column in `cols` is not present in `df` or if `method` is unknown.
        DataCleaningError: If the imputation process fails for other reasons.
    """
    df_out = df.copy()

    # Validate columns exist
    missing_cols = [c for c in cols if c not in df_out.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")

    if not cols:
        logger.warning("No columns specified for imputation")
        return df_out

    # Select only numeric columns for advanced imputation
    # Note: Categorical columns should be encoded before using KNN/MICE
    numeric_df = df_out[cols].select_dtypes(include=[np.number])

    if numeric_df.empty:
        logger.warning("No numeric columns found among selected columns for imputation")
        return df_out

    ignored_cols = set(cols) - set(numeric_df.columns)
    if ignored_cols:
        logger.warning(f"Skipping non-numeric columns for imputation: {ignored_cols}")

    try:
        if method == "knn":
            from sklearn.impute import KNNImputer

            n_neighbors = kwargs.get("n_neighbors", 5)
            logger.info(f"Running KNN Imputation (k={n_neighbors})...")
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_out[numeric_df.columns] = imputer.fit_transform(numeric_df)

        elif method == "mice":
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer

            random_state = kwargs.get("random_state", 42)
            max_iter = kwargs.get("max_iter", 10)
            logger.info(f"Running MICE Imputation (iter={max_iter})...")
            imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)
            df_out[numeric_df.columns] = imputer.fit_transform(numeric_df)

        elif method == "mean":
            logger.info("Running Mean Imputation...")
            for col in numeric_df.columns:
                df_out[col] = df_out[col].fillna(df_out[col].mean())

        elif method == "median":
            logger.info("Running Median Imputation...")
            for col in numeric_df.columns:
                df_out[col] = df_out[col].fillna(df_out[col].median())

        else:
            raise ValueError(f"Unknown imputation method: {method}")

        logger.info(
            f"Imputed missing data using {method} on {len(numeric_df.columns)} columns"
        )
        return df_out

    except Exception as e:
        logger.exception("Imputation failed")
        raise DataCleaningError(f"Imputation failed: {e}") from e


def transform_variable(series: pd.Series, method: str = "log") -> pd.Series:
    """
    Apply a statistical transformation to a numeric series.

    This function first cleans the input to numeric values and then applies one of:
    - "log": natural logarithm; if the series contains values <= 0 it is shifted by (abs(min) + 1) before taking the log.
    - "sqrt": square root; negative values are converted to `NaN` before transformation.
    - "zscore": standard score (mean 0, standard deviation 1); if the standard deviation is zero the result is the series centered at zero (all zeros).

    Parameters:
        series (pd.Series): Input values to be cleaned and transformed.
        method (str): Transformation to apply. One of "log", "sqrt", or "zscore".

    Returns:
        pd.Series: Transformed numeric series.

    Raises:
        ValueError: If `method` is not one of the supported transformations.
        DataCleaningError: If an error occurs during cleaning or transformation.
    """
    # Ensure numeric
    clean_s = clean_numeric_vector(series)

    try:
        if method == "log":
            # Handle zeros/negative for log
            min_val = clean_s.min()
            if min_val <= 0:
                # Shift if negative or zero
                shift = abs(min_val) + 1
                logger.info(
                    f"Log transform: shifting data by {shift} to handle non-positive values"
                )
                return np.log(clean_s + shift)
            return np.log(clean_s)

        elif method == "sqrt":
            # Warn if negative
            if (clean_s < 0).any():
                logger.warning(
                    "Sqrt transform: negative values encountered (set to NaN)"
                )
            return np.sqrt(clean_s)

        elif method == "zscore":
            std = clean_s.std()
            if std == 0:
                logger.warning("Z-score transform: standard deviation is zero")
                return clean_s - clean_s.mean()  # Returns zeros
            return (clean_s - clean_s.mean()) / std

        else:
            raise ValueError(f"Unknown transformation method: {method}")

    except Exception as e:
        logger.error(f"Transformation {method} failed: {e}")
        raise DataCleaningError(f"Transformation failed: {e}") from e


def check_assumptions(series: pd.Series) -> dict[str, Any]:
    """
    Evaluate basic distributional assumptions for a numeric series.

    The function cleans the input, drops missing values, computes sample size, skewness, and kurtosis,
    and performs a normality test (Shapiro-Wilk for n < 5000; Kolmogorov-Smirnov against a standard normal for n >= 5000).
    If fewer than 3 non-missing observations are available the function returns an "Insufficient Data" result.
    Any exceptions encountered are captured in the returned dictionary under the `error` key.

    Returns:
        dict: Dictionary with the following keys:
            - n (int): Number of non-missing observations used.
            - normality_test (str): Name of the normality test performed or a short status ("Insufficient Data").
            - statistic (float): Test statistic value (NaN if not available).
            - p_value (float): p-value from the normality test (NaN if not available).
            - is_normal (bool): `true` if p_value > 0.05, `false` otherwise.
            - skewness (float): Sample skewness rounded to 4 decimal places (NaN if not available).
            - kurtosis (float): Sample kurtosis rounded to 4 decimal places (NaN if not available).
            - error (str, optional): Error message if an exception occurred.
    """
    clean_s = clean_numeric_vector(series).dropna()

    result = {
        "n": len(clean_s),
        "normality_test": "None",
        "statistic": np.nan,
        "p_value": np.nan,
        "is_normal": False,
        "skewness": np.nan,
        "kurtosis": np.nan,
    }

    if len(clean_s) < 3:
        result["normality_test"] = "Insufficient Data"
        return result

    try:
        # Calculate moments
        result["skewness"] = float(round(clean_s.skew(), 4))
        result["kurtosis"] = float(round(clean_s.kurt(), 4))

        # Normality Test
        # Shapiro-Wilk (N < 5000) or Kolmogorov-Smirnov / Anderson-Darling
        # For simplicity and robustness, we use Shapiro for N < 5000,
        # and K-S Test comparing to standard normal (after standardization) for N >= 5000

        if len(clean_s) < 5000:
            stat, p_val = stats.shapiro(clean_s)
            test_name = "Shapiro-Wilk"
        else:
            # Standardize for KS test against standard normal
            std = clean_s.std()
            if std == 0 or np.isnan(std):
                result["normality_test"] = "Insufficient Variance"
                return result
            std_s = (clean_s - clean_s.mean()) / std
            stat, p_val = stats.kstest(std_s, "norm")
            test_name = "K-S Test"

        result.update(
            {
                "normality_test": test_name,
                "statistic": float(round(stat, 4)),
                "p_value": float(round(p_val, 4)),
                "is_normal": bool(p_val > 0.05),
            }
        )

        return result

    except Exception as e:
        logger.error(f"Assumption check failed: {e}")
        result["error"] = str(e)
        return result


def handle_missing_for_analysis(
    df: pd.DataFrame,
    var_meta: dict[str, Any],
    missing_codes: list[Any] | None = None,
    strategy: str = "complete-case",
    return_counts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Handle missing values in a DataFrame according to a specified strategy.

    Applies configured or provided missing-value codes to the DataFrame (converting them to NaN) and then performs the chosen missing-data handling strategy.

    Parameters:
        df (pd.DataFrame): Input DataFrame to process.
        var_meta (dict[str, Any]): Variable metadata used to determine per-column missing-value codes (passed to apply_missing_values_to_df).
        missing_codes (list[Any] | None): Global missing-value codes to apply if not specified per-variable in var_meta.
        strategy (str): Strategy to apply after normalizing missing codes. Supported values:
            - "complete-case": drop rows containing any NaN.
            - "drop": drop columns containing any NaN (keep only complete variables).
        return_counts (bool): If True, also return a counts dictionary summarizing rows before/after and percentage removed.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: Cleaned DataFrame. If `return_counts` is True, returns a tuple of (cleaned DataFrame, counts) where `counts` contains:
            - original_rows: number of rows before handling
            - final_rows: number of rows after handling
            - rows_removed: number of rows removed
            - pct_removed: percentage of rows removed (rounded to two decimals)

    Raises:
        ValueError: If `strategy` is not one of the supported values.
    """
    # Step 1: Apply missing codes → NaN
    df_processed = apply_missing_values_to_df(df, var_meta, missing_codes)

    original_rows = len(df_processed)

    # Step 2: Apply strategy
    if strategy == "complete-case":
        # Drop rows with any NaN
        df_clean = df_processed.dropna()
    elif strategy == "drop":
        # Drop columns with any NaN (keep only complete variables)
        df_clean = df_processed.dropna(axis=1)
    else:
        raise ValueError(
            f"Unknown missing data strategy: '{strategy}'. "
            f"Supported strategies: 'complete-case', 'drop'"
        )

    rows_removed = original_rows - len(df_clean)

    logger.info(
        f"Missing data handling: {original_rows} → {len(df_clean)} rows "
        f"({rows_removed} removed, strategy='{strategy}')"
    )

    if return_counts:
        counts = {
            "original_rows": original_rows,
            "final_rows": len(df_clean),
            "rows_removed": rows_removed,
            "pct_removed": round(
                (rows_removed / original_rows * 100) if original_rows > 0 else 0, 2
            ),
        }
        return df_clean, counts

    return df_clean


def check_missing_data_impact(
    df_original: pd.DataFrame,
    df_clean: pd.DataFrame,
    var_meta: dict[str, Any],
    missing_codes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare before/after to report impact of missing data handling.

    Provides accurate counts even if source data contained coded missing values
    (e.g., -99, 999) by normalizing internally if missing_codes is provided.

    Parameters:
        df_original: Original DataFrame (before cleaning)
        df_clean: Cleaned DataFrame (after handling)
        var_meta: Variable metadata
        missing_codes: Optional dictionary mapping variable names to missing values

    Returns:
        Dictionary with impact statistics
    """
    # Normalize if missing codes provided to ensure accurate counting
    df_orig_norm = apply_missing_values_to_df(df_original, var_meta, missing_codes)
    df_clean_norm = apply_missing_values_to_df(df_clean, var_meta, missing_codes)

    rows_removed = len(df_orig_norm) - len(df_clean_norm)
    pct_removed = (
        (rows_removed / len(df_orig_norm) * 100) if len(df_orig_norm) > 0 else 0
    )

    # Find which variables had missing data
    variables_affected = []
    observations_lost = {}

    for col in df_orig_norm.columns:
        if col not in df_clean_norm.columns:
            continue

        original_missing = df_orig_norm[col].isna().sum()

        if original_missing > 0:
            var_label = var_meta.get(col, {}).get("label", col)
            variables_affected.append(col)
            observations_lost[col] = {
                "label": var_label,
                "count": int(original_missing),
                "pct": round((original_missing / len(df_orig_norm) * 100), 1),
            }

    return {
        "rows_removed": rows_removed,
        "pct_removed": round(pct_removed, 2),
        "variables_affected": variables_affected,
        "observations_lost": observations_lost,
    }


# Convenience functions for common operations


def prepare_data_for_analysis(
    df: pd.DataFrame,
    required_cols: list[str],
    numeric_cols: list[str] | None = None,
    handle_missing: str = "complete_case",
    var_meta: dict[str, Any] | None = None,
    missing_codes: list[Any] | dict[str, Any] | None = None,
    return_info: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Standardized data preparation for statistical analysis.

    Performs:
    1. Validation of required columns
    2. Missing value normalization (custom codes -> NaN)
    3. Numeric conversion for specified columns
    4. Missing data handling (default: listwise deletion)
    5. Generation of missing data report

    Parameters:
        df: Input DataFrame
        required_cols: List of column names required for analysis
        numeric_cols: List of columns that must be numeric (subset of required_cols)
        handle_missing: Strategy for missing data
        var_meta: Variable metadata
        missing_codes: Global or per-column missing value codes
        return_info: Whether to return missing data info dict

    Returns:
        tuple: (Cleaned DataFrame, Missing Data Info Dictionary)
    """
    try:
        # 1. Column Validation
        # Deduplicate columns to prevent "Data must be 1-dimensional" errors when a col is repeated
        required_cols = list(dict.fromkeys(required_cols))
        if numeric_cols:
            numeric_cols = list(dict.fromkeys(numeric_cols))

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        df_subset = df[required_cols].copy()

        # 2. Missing Value Normalization (User Codes -> NaN)
        if var_meta or missing_codes:
            df_subset = apply_missing_values_to_df(df_subset, var_meta, missing_codes)

        original_rows = len(df_subset)

        # 3. Numeric Conversion
        if numeric_cols:
            for col in numeric_cols:
                if col in df_subset.columns:
                    df_subset[col] = clean_numeric_vector(df_subset[col])

        # 4. Analyze Missing Data (Before Deletion)
        missing_summary = []
        for col in df_subset.columns:
            n_missing = df_subset[col].isna().sum()
            if n_missing > 0:
                missing_summary.append(
                    {
                        "Variable": col,
                        "Type": str(df_subset[col].dtype),
                        "N_Valid": int(original_rows - n_missing),
                        "N_Missing": int(n_missing),
                        "Pct_Missing": f"{(n_missing / original_rows) * 100:.1f}%",
                    }
                )

        # 5. Handle Missing Data
        strategy = handle_missing.lower().replace("_", "-")
        if strategy == "complete-case":
            df_clean = df_subset.dropna()
            rows_excluded = original_rows - len(df_clean)
        elif strategy in ["none", "pairwise"]:
            df_clean = df_subset
            rows_excluded = 0
        else:
            raise ValueError(
                f"Unknown missing data strategy: '{handle_missing}'. "
                f"Supported: 'complete-case', 'none', 'pairwise'"
            )

        if df_clean.empty:
            raise DataValidationError(
                "No valid data remaining after cleaning (all rows contained missing values)"
            )

        info = {
            "strategy": strategy,
            "rows_original": original_rows,
            "rows_analyzed": len(df_clean),
            "rows_excluded": rows_excluded,
            "analyzed_indices": df_clean.index.tolist(),
            "summary_before": missing_summary,
        }

        logger.info(
            f"Data prepared: {original_rows} -> {len(df_clean)} rows (excluded {rows_excluded})"
        )

        if return_info:
            return df_clean, info
        return df_clean

    except Exception as e:
        logger.exception("Failed to prepare data for analysis")
        raise DataCleaningError(f"Data preparation failed: {e}") from e


def quick_clean_numeric(series: pd.Series | np.ndarray | list[Any]) -> pd.Series:
    """
    Quick numeric cleaning without extensive validation.
    """
    return clean_numeric_vector(series)


def quick_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick DataFrame cleaning without extensive validation or outlier handling.
    """
    cleaned, _ = clean_dataframe(df, handle_outliers_flag=False, validate_quality=False)
    return cleaned
