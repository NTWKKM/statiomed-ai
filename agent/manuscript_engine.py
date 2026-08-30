"""
agent/manuscript_engine.py - Deterministic SAMPL & EQUATOR Manuscript Engine
=============================================================================
Renders publication-ready Methods, Results, and SAP sections using Jinja2
templates strictly adhering to SAMPL (Statistical Analyses and Methods in the
Published Literature) and EQUATOR Network standards with ZERO numeric hallucinations.
=============================================================================
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import jinja2

    TEMPLATE_DIR = Path(__file__).parent / "templates"
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False
    TEMPLATE_DIR = Path(__file__).parent / "templates"
    jinja_env = None


def _fallback_render(template_text: str, context: Dict[str, Any]) -> str:
    """
    Lightweight fallback template renderer when jinja2 is not available.
    Supports {{ var | default(...) }}, {% if ... %}, and {% for ... %}.
    """
    res = template_text

    # Replace simple variables: {{ var | default(...) }} and {{ var }}
    def _sub_var(m):
        expr = m.group(1).strip()
        if "|" in expr:
            parts = expr.split("|", 1)
            var_name = parts[0].strip()
            def_part = parts[1].strip()
            # extract default value
            def_match = re.search(r'default\((["\']?)(.*?)\1\)', def_part)
            default_val = def_match.group(2) if def_match else ""
            val = context.get(var_name, default_val)
            return str(val if val is not None else default_val)
        else:
            var_name = expr
            val = context.get(var_name, "")
            return str(val if val is not None else "")

    # Handle if blocks
    res = re.sub(
        r"\{%\s*if\s+([a-zA-Z0-9_]+)\s+is\s+defined.*?\s*%\}(.*?)\{%\s*endif\s*%\}",
        lambda m: m.group(2) if m.group(1) in context else "",
        res,
        flags=re.DOTALL,
    )

    # Handle for loops (like table1_rows, regression_terms)
    def _sub_for(m):
        item_name = m.group(1).strip()
        list_expr = m.group(2).strip().split("|")[0].strip()
        body = m.group(3)
        items = context.get(list_expr, [])
        out_items = []
        for it in items:
            it_text = body
            for k, v in it.items():
                it_text = it_text.replace(f"{{{{ {item_name}.{k} }}}}", str(v))
                it_text = re.sub(
                    rf"\{{\{{\s*{item_name}\.{k}\s*\|\s*default\((.*?)\)\s*\}}\}}",
                    str(v),
                    it_text,
                )
            # Remove any un-evaluated inner conditionals if needed
            it_text = re.sub(r"\{%\s*if.*?\s*%\}(.*?)\{%\s*endif\s*%\}", r"\1", it_text)
            out_items.append(it_text)
        return "".join(out_items)

    res = re.sub(
        r"\{%\s*for\s+([a-zA-Z0-9_]+)\s+in\s+(.*?)\s*%\}(.*?)\{%\s*endfor\s*%\}",
        _sub_for,
        res,
        flags=re.DOTALL,
    )

    # Substitute remaining variables
    res = re.sub(r"\{\{\s*(.*?)\s*\}\}", _sub_var, res)
    return res


def format_sampl_p_value(p: Optional[float]) -> str:
    """
    Formats P-values according to SAMPL guidelines:
    - P < 0.001 for values < 0.001
    - P = 0.042 (3 decimals if < 0.10)
    - P = 0.45 (2 decimals if >= 0.10)
    """
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    elif p < 0.10:
        return f"= {p:.3f}"
    else:
        return f"= {p:.2f}"


def format_sampl_ci(lower: float, upper: float, decimals: int = 2) -> str:
    """
    Formats 95% Confidence Intervals: (95% CI: lower to upper)
    """
    return f"95% CI: {lower:.{decimals}f} to {upper:.{decimals}f}"


def format_sampl_pct(numerator: int, denominator: int) -> str:
    """
    Formats percentages per SAMPL:
    - 1 decimal place if N >= 100
    - Whole integer if N < 100
    """
    if denominator <= 0:
        return "0%"
    pct = (numerator / denominator) * 100.0
    if denominator >= 100:
        return f"{pct:.1f}%"
    else:
        return f"{int(round(pct))}%"


class ManuscriptEngine:
    @staticmethod
    def _render_file(template_name: str, context: Dict[str, Any]) -> str:
        template_file = TEMPLATE_DIR / template_name
        if HAS_JINJA and jinja_env:
            try:
                template = jinja_env.get_template(template_name)
                return template.render(context)
            except Exception:
                pass
        # Fallback reading raw text
        if template_file.exists():
            raw_text = template_file.read_text(encoding="utf-8")
            return _fallback_render(raw_text, context)
        return f"Template {template_name} not found."

    @staticmethod
    def render_sap(context: Dict[str, Any]) -> str:
        """
        Renders a candidate Statistical Analysis Plan (SAP).
        """
        return ManuscriptEngine._render_file("sap_candidate.jinja2", context)

    @staticmethod
    def render_methods(study_type: str, context: Dict[str, Any]) -> str:
        """
        Renders EQUATOR-compliant Methods section ('rct', 'cohort', 'diagnostic').
        """
        template_name = f"methods_{study_type.lower()}.jinja2"
        res = ManuscriptEngine._render_file(template_name, context)
        if "not found" in res:
            return ManuscriptEngine._render_file("methods_cohort.jinja2", context)
        return res

    @staticmethod
    def render_results(analysis_type: str, context: Dict[str, Any]) -> str:
        """
        Renders SAMPL-compliant Results section ('table1', 'survival', 'regression', 'diagnostic').
        """
        template_name = f"results_{analysis_type.lower()}.jinja2"
        res = ManuscriptEngine._render_file(template_name, context)
        if "not found" in res:
            return ManuscriptEngine._render_file("results_table1.jinja2", context)
        return res

    @staticmethod
    def render_full_draft(study_type: str, context: Dict[str, Any]) -> str:
        """
        Renders complete Methods & Results draft for manuscript preparation.
        """
        methods = ManuscriptEngine.render_methods(study_type, context)
        results = ManuscriptEngine.render_results(
            context.get("primary_analysis_type", "survival"), context
        )

        draft = f"# Draft Manuscript Section: {study_type.upper()} Study\n\n"
        draft += f"**Reporting Standard:** {context.get('reporting_standard', 'ICMJE / SAMPL Guidelines')}\n\n"
        draft += "---\n\n"
        draft += methods + "\n\n---\n\n"
        draft += results + "\n\n---\n\n"
        draft += "*Generated deterministically via StatioMed AI Manuscript Engine (Zero-LLM Numeric Hallucinations).*"
        return draft
