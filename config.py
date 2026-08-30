"""
Configuration Management System for Medical Statistical Tool

This module provides centralized configuration management for the application,
including analysis parameters, UI settings, logging configuration, and runtime options.

Usage:
    from config import CONFIG

    # Access config
    print(CONFIG.get('analysis.logit_method'))

    # Update config (runtime)
    CONFIG.update('analysis.logit_method', 'firth')

    # Get with default
    value = CONFIG.get('some.nested.key', default='default_value')
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, cast

from logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """
    Centralized configuration management with hierarchical key access.

    Supports:
    - Nested dictionary access with dot notation
    - Default values and fallbacks
    - Environment variable overrides
    - Config validation
    - Runtime updates
    """

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        """
        Create a ConfigManager populated with the given configuration or the module defaults and apply environment variable overrides.

        Parameters:
            config_dict (dict | None): Optional initial configuration dictionary to use instead of the built-in defaults. If None, the manager is initialized from the default configuration.
        """
        self._config: dict[str, Any] = (
            config_dict if config_dict is not None else self._get_default_config()
        )
        self._env_prefix = "MEDSTAT_"
        self._load_env_overrides()
        self._sync_missing_legacy()

    def _sync_missing_legacy(self) -> None:
        """
        Synchronize the nested 'analysis.missing' dictionary with legacy top-level keys
        ('missing_strategy', 'missing_threshold_pct') to maintain backward compatibility
        while preserving explicit overrides.
        """
        analysis = self._config.get("analysis", {})
        missing = analysis.get("missing", {})
        if not isinstance(missing, dict):
            return

        legacy_strategy = analysis.get("missing_strategy")
        legacy_threshold = analysis.get("missing_threshold_pct")

        # 1. Backfill nested from legacy if nested keys are missing
        if legacy_strategy is not None and "strategy" not in missing:
            missing["strategy"] = legacy_strategy
        if legacy_threshold is not None and "report_threshold_pct" not in missing:
            missing["report_threshold_pct"] = legacy_threshold

        # 2. Sync to legacy ONLY if legacy key is absent (preserve overrides)
        if "missing_strategy" not in analysis and "strategy" in missing:
            analysis["missing_strategy"] = missing["strategy"]
        if (
            "missing_threshold_pct" not in analysis
            and "report_threshold_pct" in missing
        ):
            analysis["missing_threshold_pct"] = missing["report_threshold_pct"]

    @staticmethod
    def _get_default_config() -> dict[str, Any]:
        """
        Return the module's default nested configuration for the application.

        Returns:
            dict[str, Any]: Default configuration dictionary with top-level sections:
            'analysis', 'stats', 'ui', 'logging', 'performance', 'validation', and 'debug'.
        """
        return {
            # ========== ANALYSIS SETTINGS ==========
            "analysis": {
                # Logistic Regression
                "logit_method": "auto",  # 'auto', 'firth', 'bfgs', 'default'
                "logit_max_iter": 100,
                "logit_screening_p": 0.20,  # Variables with p < this get into multivariate
                "logit_min_cases": 10,  # Minimum cases for multivariate analysis
                # Variable Detection
                "var_detect_threshold": 10,  # Unique values threshold for categorical/continuous
                "var_detect_decimal_pct": 0.30,  # Decimal % for continuous classification
                # Reporting Style
                "publication_style": "nejm",  # 'nejm', 'jama', 'lancet', 'bmj'
                # P-value Handling (NEJM-oriented defaults)
                "pvalue_bounds_lower": 0.001,  # NEJM: show P<0.001 for smaller values
                "pvalue_bounds_upper": 0.999,  # NEJM: often cap display at >0.99
                "pvalue_clip_tolerance": 0.00001,  # tighter tolerance for extreme p
                "pvalue_format_small": "<0.001",
                "pvalue_format_large": ">0.999",
                "significance_level": 0.05,  # Added for utils/formatting.py
                # Survival Analysis
                "survival_method": "kaplan-meier",  # 'kaplan-meier', 'weibull'
                "cox_method": "efron",  # 'efron', 'breslow'
                # Missing Data
                "missing": {
                    "strategy": "complete-case",  # 'complete-case', 'drop', 'impute' (future)
                    "user_defined_values": [],  # User-specified missing codes: [-99, -999, 99]
                    "treat_empty_as_missing": True,
                    "report_missing": True,
                    "report_threshold_pct": 50,  # Flag if >X% missing
                },
                "missing_strategy": "complete-case",  # Legacy - kept for backward compatibility
                "missing_threshold_pct": 50,  # Legacy - kept for backward compatibility
            },
            # ========== ADVANCED STATS SETTINGS ==========
            "stats": {
                "mcc_enable": True,
                "mcc_method": "fdr_bh",
                "mcc_alpha": 0.05,
                "vif_enable": True,
                "vif_threshold": 10,
                "ci_method": "auto",
            },
            # ========== UI & DISPLAY SETTINGS ==========
            "ui": {
                # Page Setup
                "page_title": "Medical Stat Tool",
                "layout": "wide",
                "theme": "light",  # 'light', 'dark', 'auto'
                # Sidebar
                "sidebar_width": 300,
                "show_sidebar_logo": True,
                # Tables
                "table_max_rows": 1000,  # Max rows to display in data table
                "table_pagination": True,
                "table_decimal_places": 3,
                "styles": {
                    "sig_p_value": "font-weight: bold; color: #DC2626;",
                    "sig_ci": "font-weight: bold; color: #059669;",
                },
                # Plots
                "plot_width": 10,
                "plot_height": 6,
                "plot_dpi": 100,
                "plot_style": "seaborn",
            },
            # ========== LOGGING SETTINGS ==========
            "logging": {
                "enabled": True,
                "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
                "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                # File Logging
                "file_enabled": False,  # Set to True to enable file logging
                "log_dir": "logs",
                "log_file": "app.log",
                "max_log_size": 10485760,  # 10MB in bytes
                "backup_count": 5,
                # Console Logging
                "console_enabled": True,
                "console_level": "INFO",
                # What to Log
                "log_file_operations": True,
                "log_data_operations": True,
                "log_analysis_operations": True,
                "log_ui_events": False,  # Can be verbose
                "log_performance": True,  # Timing information
            },
            # ========== PERFORMANCE SETTINGS ==========
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600,  # seconds
                "enable_compression": False,
                "num_threads": 4,
            },
            # ========== VALIDATION SETTINGS ==========
            "validation": {
                "strict_mode": False,  # Warn vs Error on validation failures
                "validate_inputs": True,
                "validate_outputs": True,
                "auto_fix_errors": True,  # Try to fix issues automatically
            },
            # ========== DEVELOPER SETTINGS ==========
            "debug": {
                "enabled": False,
                "verbose": False,
                "profile_performance": False,
                "show_timings": False,
            },
        }

    def _load_env_overrides(self) -> None:
        """
        Apply configuration overrides from environment variables that start with the MEDSTAT_ prefix.
        """
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                parts = key[len(self._env_prefix) :].lower().split("_")
                if len(parts) < 2:
                    continue
                section = parts[0]
                key_name = "_".join(parts[1:])
                try:
                    self.update(f"{section}.{key_name}", value)
                except (KeyError, ValueError, TypeError) as e:
                    warnings.warn(
                        f"Failed to set env override {key}={value}: {e}", stacklevel=2
                    )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using a dot-separated key path.
        """
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, key: str, value: Any) -> None:
        """
        Set an existing configuration value identified by a dot-separated path.
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                raise KeyError(f"Config path '{'.'.join(keys[:-1])}' does not exist")
            config = config[k]
        final_key = keys[-1]
        if final_key not in config:
            raise KeyError(f"Config key '{key}' does not exist")
        config[final_key] = value
        if key.startswith("analysis.missing_"):
            analysis = self._config.get("analysis", {})
            missing = analysis.get("missing")
            if isinstance(missing, dict):
                if key.endswith("missing_strategy"):
                    missing["strategy"] = value
                elif key.endswith("missing_threshold_pct"):
                    missing["report_threshold_pct"] = value
        if key.startswith("analysis.missing"):
            self._sync_missing_legacy()

    def set_nested(self, key: str, value: Any, create: bool = False) -> None:
        """
        Set a value in the configuration using a dot-separated path, optionally creating missing intermediate dictionaries.
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                if create:
                    config[k] = {}
                else:
                    raise KeyError(f"Config path '{k}' does not exist")
            config = config[k]
        config[keys[-1]] = value

    def get_section(self, section: str) -> dict[str, Any]:
        """
        Return a deep copy of a top-level configuration section.
        """
        import copy

        result = self.get(section, {})
        return copy.deepcopy(result) if isinstance(result, dict) else result

    def to_dict(self) -> dict[str, Any]:
        """
        Get a deep copy of the entire configuration dictionary.
        """
        import copy

        return copy.deepcopy(self._config)

    def to_json(self, filepath: str | None = None, pretty: bool = True) -> str:
        """
        Serialize the current configuration to a JSON string.
        """
        try:
            json_str = json.dumps(self._config, indent=2 if pretty else None)
        except (TypeError, ValueError):
            logger.exception("Failed to serialize config to JSON")
            json_str = "{}"
        if filepath:
            try:
                Path(filepath).write_text(json_str)
            except OSError:
                logger.exception("Failed to write config to %s", filepath)
        return json_str

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate key configuration constraints and collect any violations.
        """
        errors = []
        screening_p = cast(float | None, self.get("analysis.logit_screening_p"))
        if screening_p is None or not (0 < screening_p < 1):
            errors.append("analysis.logit_screening_p must be between 0 and 1")

        lower = cast(float | None, self.get("analysis.pvalue_bounds_lower"))
        upper = cast(float | None, self.get("analysis.pvalue_bounds_upper"))
        if lower is None or upper is None or not (lower < upper):
            errors.append("pvalue_bounds_lower must be < pvalue_bounds_upper")

        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.get("logging.level") not in valid_levels:
            errors.append(f"logging.level must be one of {valid_levels}")

        valid_methods = ["auto", "firth", "bfgs", "default"]
        if self.get("analysis.logit_method") not in valid_methods:
            errors.append(f"analysis.logit_method must be one of {valid_methods}")

        valid_styles = ["nejm", "jama", "lancet", "bmj"]
        if self.get("analysis.publication_style") not in valid_styles:
            errors.append(f"analysis.publication_style must be one of {valid_styles}")

        return len(errors) == 0, errors

    def __repr__(self) -> str:
        return f"ConfigManager({len(self._config)} sections)"


# Global config instance
CONFIG = ConfigManager()


# =============================================================================
# CONSTANTS & THRESHOLDS
# =============================================================================

COHEN_D_THRESHOLDS = {
    "negligible": 0.2,
    "small": 0.5,
    "medium": 0.8,
    "large": float("inf"),
}

ETA_SQUARED_THRESHOLDS = {
    "small": 0.01,
    "medium": 0.06,
    "large": 0.14,
}


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("\n" + "=" * 60)
    print("Configuration Management System - Test")
    print("=" * 60)

    # Test 1: Get values
    print("\n[Test 1] Getting configuration values:")
    print(f"  Logit method: {CONFIG.get('analysis.logit_method')}")
    print(f"  Logging level: {CONFIG.get('logging.level')}")
    print(f"  Log file enabled: {CONFIG.get('logging.file_enabled')}")

    # Test 2: Get with default
    print("\n[Test 2] Getting with defaults:")
    print(f"  Nonexistent key: {CONFIG.get('some.fake.key', 'default_value')}")

    # Test 3: Update config
    print("\n[Test 3] Updating configuration:")
    try:
        CONFIG.update("logging.level", "DEBUG")
        print(f"  ✓ Updated logging.level to: {CONFIG.get('logging.level')}")
    except KeyError as e:
        print(f"  ✗ Error: {e}")

    # Test 4: Get section
    print("\n[Test 4] Getting section:")
    logging_section = CONFIG.get_section("logging")
    print(f"  Logging section keys: {list(logging_section.keys())}")

    # Test 5: Validate
    print("\n[Test 5] Validating configuration:")
    is_valid, validation_errors = CONFIG.validate()
    print(f"  Valid: {is_valid}")
    if validation_errors:
        for err in validation_errors:
            print(f"    ✗ {err}")
    else:
        print("    ✓ No errors found")

    # Test 6: Export to JSON
    print("\n[Test 6] Exporting configuration:")
    json_output = CONFIG.to_json(pretty=False)
    print(f"  JSON length: {len(json_output)} characters")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")
