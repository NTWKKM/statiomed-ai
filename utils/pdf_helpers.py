"""
Centralized PDF download helpers for report generation.

Converts HTML reports to PDF using Playwright's headless Chromium.
Mirrors the API of ``download_helpers.py`` for consistency.
"""

import logging
import tempfile
from pathlib import Path

from utils.download_helpers import _build_error_html, _notify

logger = logging.getLogger(__name__)


def html_to_pdf(html_content: str) -> bytes:
    """
    Convert an HTML string to PDF bytes using Playwright (headless Chromium).

    Args:
        html_content: A complete HTML document string.

    Returns:
        PDF file content as bytes.

    Raises:
        RuntimeError: If Playwright or Chromium is not available.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for PDF generation. "
            "Install it with: pip install playwright && playwright install chromium"
        ) from exc

    # Write HTML to a temp file so Chromium can open it via file:// URL
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(html_content)
        tmp_path = Path(tmp.name)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(tmp_path.as_uri(), wait_until="networkidle")

            # Inject Print CSS to optimize PDF appearance
            page.add_style_tag(
                content="""
                @media print {
                    /* Typography and sizing */
                    body { 
                        font-size: 11pt !important; 
                        color: #111 !important; 
                        background: #fff !important;
                    }
                    
                    /* Table formatting: Prevent breaks inside rows, show headers on new pages */
                    table { page-break-inside: auto; border-collapse: collapse !important; width: 100% !important; }
                    tr { page-break-inside: avoid; page-break-after: auto; }
                    thead { display: table-header-group; }
                    tfoot { display: table-footer-group; }
                    th, td { border: 1px solid #ddd !important; padding: 6px !important; }
                    
                    /* Headings: avoid page breaks immediately after */
                    h1, h2, h3, h4, h5, h6 { 
                        page-break-after: avoid; 
                        margin-top: 15px !important;
                    }
                    
                    /* Hide interactive UI elements that don't belong in a static report */
                    .btn, button, input, select, textarea, .download-status-badge { 
                        display: none !important; 
                    }
                    
                    /* Clean up containers (e.g. Bootstrap cards) */
                    .card { 
                        border: 1px solid #ddd !important; 
                        box-shadow: none !important; 
                        page-break-inside: avoid;
                        margin-bottom: 20px !important;
                    }
                    .card-header {
                        background-color: #f8f9fa !important;
                        border-bottom: 1px solid #ddd !important;
                        -webkit-print-color-adjust: exact;
                        color-adjust: exact;
                    }
                }
            """
            )

            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                margin={
                    "top": "15mm",
                    "right": "10mm",
                    "bottom": "15mm",
                    "left": "10mm",
                },
            )
            browser.close()
        return pdf_bytes
    finally:
        tmp_path.unlink(missing_ok=True)


def safe_download_pdf(
    content: str | None,
    *,
    label: str = "Report",
) -> bytes:
    """
    Validate HTML content, convert to PDF, and return PDF bytes.

    Mirrors :func:`download_helpers.safe_download_html` but yields PDF.

    Args:
        content: The HTML string to convert. ``None`` or empty strings
            are treated as "no results available".
        label: Human-readable report name used in notifications.

    Returns:
        PDF bytes — either the converted report or a styled error page.
    """
    if not content or not content.strip():
        logger.warning("PDF download attempted with no content for: %s", label)
        _notify(
            f"⚠️ {label}: No results available. Please run the analysis first.",
            type="warning",
            duration=5,
        )
        error_html = _build_error_html(
            f"{label} – No Results",
            "No analysis results are available for download. "
            "Please run the analysis first, then try downloading again.",
        )
        return html_to_pdf(error_html)

    try:
        pdf_bytes = html_to_pdf(content)
        _notify(
            f"✅ {label} (PDF) downloaded successfully.",
            type="message",
            duration=3,
        )
        return pdf_bytes
    except Exception:
        logger.exception("PDF conversion failed for: %s", label)
        _notify(
            f"❌ {label}: PDF generation failed. See logs for details.",
            type="error",
            duration=8,
        )
        error_html = _build_error_html(
            f"{label} – PDF Generation Failed",
            "An error occurred while generating the PDF. "
            "Please check that Playwright/Chromium is installed.",
        )
        # Return error page as a minimal PDF (plain HTML fallback)
        try:
            return html_to_pdf(error_html)
        except Exception:
            # Absolute fallback: return a minimal text-based PDF-like content
            return error_html.encode("utf-8")


def safe_pdf_report_generation(
    generate_fn,
    *,
    label: str = "Report",
) -> bytes:
    """
    Safely execute a report-generation callable and convert to PDF.

    Mirrors :func:`download_helpers.safe_report_generation` but yields PDF.

    Args:
        generate_fn: A zero-argument callable that returns an HTML string.
        label: Human-readable name for notifications.

    Returns:
        PDF bytes — either the converted report or an error-page PDF.
    """
    try:
        result = generate_fn()
        return safe_download_pdf(result, label=label)
    except Exception:
        logger.exception("Report generation failed for: %s", label)
        _notify(
            f"❌ {label}: Report generation failed. See logs for details.",
            type="error",
            duration=8,
        )
        error_html = _build_error_html(
            f"{label} – Generation Failed",
            "An error occurred while generating the report. "
            "Please check the analysis results and try again.",
        )
        try:
            return html_to_pdf(error_html)
        except Exception:
            return error_html.encode("utf-8")
