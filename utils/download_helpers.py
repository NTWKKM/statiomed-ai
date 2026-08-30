"""
Centralized download helpers for HTML report generation.

Provides utilities to ensure download handlers yield valid HTML,
handle errors gracefully, and show user notifications.
"""

import html as _html
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _notify(message: str, *, type: str = "message", duration: int = 3) -> None:
    """Show a Shiny notification, silently skipping if no session is active."""
    try:
        from shiny import ui

        ui.notification_show(message, type=type, duration=duration)
    except Exception:
        # Outside a Shiny session (e.g. tests, CLI) – just log instead
        logger.info("Notification (%s): %s", type, message)


def _build_error_html(title: str, message: str) -> str:
    """Build a minimal standalone HTML error page."""
    safe_title = _html.escape(str(title))
    safe_message = _html.escape(str(message))
    return (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        "display: flex; justify-content: center; align-items: center; min-height: 80vh; "
        "background: #FAFAFA; color: #0F172A; }"
        ".error-box { text-align: center; max-width: 480px; background: white; "
        "padding: 40px; border-radius: 8px; border: 1px solid #E2E8F0; }"
        ".error-icon { font-size: 3em; margin-bottom: 12px; }"
        "h1 { color: #DC2626; font-size: 1.5em; margin-bottom: 8px; font-weight: 500; }"
        "p { color: #64748B; line-height: 1.6; }"
        "</style>"
        "</head><body>"
        "<div class='error-box'>"
        "<div class='error-icon'>⚠️</div>"
        f"<h1>{safe_title}</h1>"
        f"<p>{safe_message}</p>"
        f"<p style='font-size:0.8em;color:#999;margin-top:20px;'>"
        f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        "</div></body></html>"
    )


def safe_download_html(
    content: str | None,
    *,
    label: str = "Report",
) -> str:
    """
    Validate HTML content for download, returning an error page if empty.

    Use this inside ``@render.download`` handlers to guarantee the user
    always receives a valid HTML file and sees a notification about the
    outcome.

    Args:
        content: The HTML string to download. ``None`` or empty strings
            are treated as "no results available".
        label: Human-readable report name used in notifications and the
            fallback error page (e.g. ``"ROC Report"``).

    Returns:
        A valid HTML string – either the original *content* or a styled
        error page explaining that results are unavailable.

    Example::

        @render.download(filename="roc_report.html")
        def btn_dl_roc_report():
            yield safe_download_html(roc_html.get(), label="ROC Report")
    """
    if not content or not content.strip():
        logger.warning("Download attempted with no content for: %s", label)
        _notify(
            f"⚠️ {label}: No results available. Please run the analysis first.",
            type="warning",
            duration=5,
        )
        return _build_error_html(
            f"{label} – No Results",
            "No analysis results are available for download. "
            "Please run the analysis first, then try downloading again.",
        )

    # Content is valid – notify success
    _notify(
        f"✅ {label} downloaded successfully.",
        type="message",
        duration=3,
    )
    return content


def safe_report_generation(
    generate_fn,
    *,
    label: str = "Report",
) -> str:
    """
    Safely execute a report-generation callable, catching exceptions.

    Wraps *generate_fn* in a try/except so that any error during inline
    report building (e.g. Plotly rendering, DataFrame formatting) is
    caught and converted to a downloadable error page instead of
    silently yielding nothing.

    Args:
        generate_fn: A zero-argument callable that returns an HTML string.
        label: Human-readable name for notifications.

    Returns:
        The generated HTML string, or an error-page HTML on failure.

    Example::

        @render.download(filename="cox_report.html")
        def btn_dl_cox():
            def _build():
                res = cox_result.get()
                if not res:
                    return None
                elements = [...]
                return survival_lib.generate_report_survival(...)

            yield safe_report_generation(_build, label="Cox Report")
    """
    try:
        result = generate_fn()
        return safe_download_html(result, label=label)
    except Exception:
        logger.exception("Report generation failed for: %s", label)
        _notify(
            f"❌ {label}: Report generation failed. See logs for details.",
            type="error",
            duration=8,
        )
        return _build_error_html(
            f"{label} – Generation Failed",
            "An error occurred while generating the report. "
            "Please check the analysis results and try again.",
        )


def safe_data_download(data: Any, *, label: str = "Data", type_: str = "dataset") -> None:
    """
    Validate data exists before allowing a download, aborting gracefully if empty.

    Use this inside ``@render.download`` handlers before yielding raw data
    (like CSVs, Excels, or JSONs) to prevent server errors or empty file downloads.
    If the data is missing or empty (e.g. an empty pandas DataFrame), this will
    show a notification and silently cancel the download.

    Args:
        data: The dataset (e.g. pd.DataFrame, list, dict) or file content to check.
            If None or empty, the download is aborted.
        label: Human-readable name used in the notification.
        type_: Description of what is being downloaded (e.g. "dataset", "checklist").

    Raises:
        shiny.SilentException: If data is missing/empty (via req(False)).
    """
    import pandas as pd
    from shiny import req

    is_valid = False
    if data is not None:
        if isinstance(data, pd.DataFrame):
            is_valid = not data.empty
        elif isinstance(data, (str, list, dict, set, tuple)):
            is_valid = len(data) > 0
        else:
            is_valid = True  # Assuming true for other objects if not None

    if not is_valid:
        logger.warning("Download aborted - empty %s for: %s", type_, label)
        _notify(
            f"⚠️ {label}: No {type_} available. Please check your inputs.",
            type="warning",
            duration=5,
        )
        req(False)

    _notify(f"✅ {label} download initiated.", type="message", duration=3)
