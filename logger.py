"""
🧮 Logging Framework for Statistical Analysis

Simplified, Streamlit-free logging system.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

# Default configuration (can be overridden)
DEFAULT_CONFIG: dict[str, Any] = {
    "logging.enabled": True,
    "logging.level": "WARNING",
    "logging.format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "logging.date_format": "%Y-%m-%d %H:%M:%S",
    "logging.file_enabled": False,
    "logging.console_enabled": True,
    "logging.console_level": "INFO",
    "logging.log_dir": "logs",
    "logging.log_file": "app.log",
    "logging.max_log_size": 10485760,
    "logging.backup_count": 5,
    "logging.log_performance": False,
    "logging.log_data_operations": True,
    "logging.log_analysis_operations": True,
}


class PerformanceLogger:
    """Track and log performance metrics."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize with logger."""
        self.logger = logger
        self.timings: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @contextmanager
    def track_time(
        self, operation: str, log_level: str = "DEBUG"
    ) -> Generator[None, None, None]:
        """Context manager to track operation timing."""
        if not DEFAULT_CONFIG.get("logging.log_performance"):
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time

            with self._lock:
                if operation not in self.timings:
                    self.timings[operation] = []
                self.timings[operation].append(elapsed)

            log_method = getattr(self.logger, log_level.lower(), self.logger.debug)
            log_method(f"⏱️ {operation} completed in {elapsed:.3f}s")

    def get_timings(self, operation: str | None = None) -> dict[str, list[float]]:
        """Get recorded timings."""
        if operation:
            return {operation: self.timings.get(operation, [])}
        return self.timings

    def print_summary(self) -> None:
        """Print performance summary."""
        if not self.timings:
            return

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Performance Summary")
        self.logger.info("=" * 60)

        for operation, times in self.timings.items():
            if times:
                avg = sum(times) / len(times)
                min_t = min(times)
                max_t = max(times)
                self.logger.info(
                    f"  {operation}: avg={avg:.3f}s, min={min_t:.3f}s, "
                    f"max={max_t:.3f}s (n={len(times)})"
                )

        self.logger.info("=" * 60)


class ContextFilter(logging.Filter):
    """Add context information to log records."""

    def __init__(self) -> None:
        """Initialize context filter."""
        super().__init__()
        self.context: dict[str, Any] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach context to log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True

    def set_context(self, **kwargs: Any) -> None:
        """Set context values."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear context."""
        self.context.clear()


class Logger:
    """Wrapper around standard logger with features."""

    def __init__(
        self,
        standard_logger: logging.Logger,
        context_filter: ContextFilter | None = None,
    ) -> None:
        """Initialize logger."""
        self._logger = standard_logger
        self._context_filter = context_filter
        self._perf_logger = LoggerFactory.get_performance_logger()

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(msg, *args, **kwargs)

    def log_operation(
        self, operation: str, status: str = "started", **details: Any
    ) -> None:
        """Log operation event."""
        msg_parts = [f"[{operation}]"]
        if status:
            msg_parts.append(f"{status.upper()}")
        if details:
            detail_str = " | ".join([f"{k}={v}" for k, v in details.items()])
            msg_parts.append(detail_str)

        msg = " ".join(msg_parts)
        if status.lower() == "failed":
            self.error(msg)
        else:
            self.info(msg)

    def log_data_summary(
        self, df_name: str, shape: tuple[int, ...], dtypes: dict[str, str]
    ) -> None:
        """Log data summary."""
        if DEFAULT_CONFIG.get("logging.log_data_operations"):
            self.info(
                f"📈 {df_name}: shape={shape}, "
                f"numeric={sum(1 for t in dtypes.values() if 'int' in t.lower() or 'float' in t.lower())}, "
                f"object={sum(1 for t in dtypes.values() if 'object' in t)}"
            )

    def log_analysis(
        self, analysis_type: str, outcome: str, n_vars: int, n_samples: int
    ) -> None:
        """Log analysis summary."""
        if DEFAULT_CONFIG.get("logging.log_analysis_operations"):
            self.info(
                f"📈 {analysis_type}: outcome='{outcome}', "
                f"predictors={n_vars}, n={n_samples}"
            )

    @contextmanager
    def track_time(
        self, operation: str, log_level: str = "DEBUG"
    ) -> Generator[None, None, None]:
        """Track operation timing."""
        with self._perf_logger.track_time(operation, log_level):
            yield

    def get_timings(self) -> dict[str, list[float]]:
        """Get performance timings."""
        return self._perf_logger.get_timings()

    def set_context(self, **kwargs: Any) -> None:
        """Set context."""
        if self._context_filter:
            self._context_filter.set_context(**kwargs)

    def clear_context(self) -> None:
        """Clear context."""
        if self._context_filter:
            self._context_filter.clear_context()


class LoggerFactory:
    """Factory for creating and managing loggers."""

    _loggers: ClassVar[dict[str, Logger]] = {}
    _context_filter: ClassVar[ContextFilter | None] = None
    _perf_logger: ClassVar[PerformanceLogger | None] = None
    _configured: ClassVar[bool] = False
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def configure(cls) -> None:
        """Configure logging system."""
        if cls._configured:
            return

        try:
            if not DEFAULT_CONFIG.get("logging.enabled"):
                logging.disable(logging.CRITICAL)
                cls._configured = True
                return

            log_level = cast(str, DEFAULT_CONFIG.get("logging.level", "INFO"))
            log_format = cast(str, DEFAULT_CONFIG.get("logging.format"))
            date_format = cast(str, DEFAULT_CONFIG.get("logging.date_format"))

            formatter = logging.Formatter(log_format, datefmt=date_format)

            root_logger = logging.getLogger()
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            root_logger.setLevel(numeric_level)

            if root_logger.handlers:
                root_logger.handlers.clear()

            cls._context_filter = ContextFilter()

            if DEFAULT_CONFIG.get("logging.console_enabled"):
                cls._setup_console_logging(root_logger, formatter)

            if DEFAULT_CONFIG.get("logging.file_enabled"):
                cls._setup_file_logging(root_logger, formatter)

            cls._configured = True

        except Exception as e:
            print(f"[WARNING] Logging config failed: {e}", file=sys.stderr)
            cls._configured = True

    @classmethod
    def _setup_console_logging(
        cls, root_logger: logging.Logger, formatter: logging.Formatter
    ) -> None:
        """Setup console logging."""
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_level = cast(
                str, DEFAULT_CONFIG.get("logging.console_level", "INFO")
            )
            console_handler.setLevel(getattr(logging, console_level))
            console_handler.setFormatter(formatter)
            if cls._context_filter:
                console_handler.addFilter(cls._context_filter)  # type: ignore
            root_logger.addHandler(console_handler)
        except Exception as e:
            print(f"[WARNING] Console logging setup failed: {e}", file=sys.stderr)

    @classmethod
    def _setup_file_logging(
        cls, root_logger: logging.Logger, formatter: logging.Formatter
    ) -> None:
        """Setup file logging."""
        try:
            log_dir_str = cast(str, DEFAULT_CONFIG.get("logging.log_dir", "logs"))
            log_dir = Path(log_dir_str)
            log_dir.mkdir(exist_ok=True, parents=True)

            log_file_name = cast(str, DEFAULT_CONFIG.get("logging.log_file", "app.log"))
            log_file = log_dir / log_file_name
            max_size = cast(int, DEFAULT_CONFIG.get("logging.max_log_size", 10485760))
            backup_count = cast(int, DEFAULT_CONFIG.get("logging.backup_count", 5))

            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_size, backupCount=backup_count
            )
            handler.setFormatter(formatter)
            if cls._context_filter:
                handler.addFilter(cls._context_filter)  # type: ignore
            root_logger.addHandler(handler)
        except Exception as e:
            print(f"[WARNING] File logging setup failed: {e}", file=sys.stderr)

    @classmethod
    def get_logger(cls, name: str) -> Logger:
        """Get or create logger."""
        if not cls._configured:
            cls.configure()

        with cls._lock:
            if name not in cls._loggers:
                standard_logger = logging.getLogger(name)
                logger = Logger(standard_logger, cls._context_filter)
                cls._loggers[name] = logger
            return cls._loggers[name]

    @classmethod
    def get_performance_logger(cls) -> PerformanceLogger:
        """Get performance logger."""
        if cls._perf_logger is None:
            perf_std_logger = logging.getLogger("performance")
            cls._perf_logger = PerformanceLogger(perf_std_logger)
        return cls._perf_logger


def get_logger(name: str) -> Logger:
    """
    Get a configured logger.

    Parameters:
        name (str): Logger name (typically __name__)

    Returns:
        Logger: Configured logger instance
    """
    return LoggerFactory.get_logger(name)
