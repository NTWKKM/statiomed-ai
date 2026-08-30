"""
🚀 CSS Generator Script
-----------------------
This script regenerates 'static/styles.css' by extracting CSS from 'tabs/_styling.py'.
Run this after making changes to the styling module.

Includes a lightweight minification step to reduce file size.
"""

import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from tabs._styling import get_shiny_css
except ImportError as e:
    print(f"❌ Error: Could not import get_shiny_css. {e}")
    sys.exit(1)


def minify_css(css: str) -> str:
    """
    Lightweight CSS minification using regex patterns.

    Removes:
    - Block comments (/* ... */)
    - Leading/trailing whitespace per line
    - Blank lines
    - Whitespace around braces, colons, semicolons
    - Trailing semicolons before closing braces

    Does NOT require any external dependencies.
    """
    # Remove block comments (non-greedy)
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    # Collapse multiple whitespace/newlines into single space
    css = re.sub(r"\s+", " ", css)
    # Remove space around { } ; : ,
    css = re.sub(r"\s*{\s*", "{", css)
    css = re.sub(r"\s*}\s*", "}", css)
    css = re.sub(r"\s*;\s*", ";", css)
    css = re.sub(r"\s*:\s*", ":", css)
    css = re.sub(r"\s*,\s*", ",", css)
    # Remove trailing semicolons before }
    css = css.replace(";}", "}")
    return css.strip()


def generate_static_css():
    print("🎨 Generating static CSS from tabs._styling.py...")

    # Get the Style Tag object
    style_tag = get_shiny_css()

    # Convert Tag to string -> <style>...css...</style>
    raw_html = str(style_tag)

    # Extract content inside <style> tags
    match = re.search(r"<style.*?>(.*?)</style>", raw_html, re.DOTALL)

    if not match:
        print("❌ Error: Could not extract CSS content.")
        print("Debug Raw Output:", raw_html[:100])
        sys.exit(1)

    css_content = match.group(1).strip()

    # Add Header Comment
    header = "/* DO NOT EDIT — auto-generated from tabs/_styling.py */\n"
    final_css = header + css_content + "\n"

    # Output Path
    output_path = project_root / "static" / "styles.css"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_css)

    original_size = len(final_css.encode("utf-8"))

    # Write minified version alongside
    minified = header + minify_css(css_content) + "\n"
    minified_path = project_root / "static" / "styles.min.css"
    with open(minified_path, "w", encoding="utf-8") as f:
        f.write(minified)

    minified_size = len(minified.encode("utf-8"))
    saved_pct = (1 - minified_size / original_size) * 100 if original_size else 0

    print(f"✅ Full CSS saved to: {output_path} ({original_size:,} bytes)")
    print(
        f"✅ Minified CSS saved to: {minified_path} "
        f"({minified_size:,} bytes, {saved_pct:.1f}% smaller)"
    )


if __name__ == "__main__":
    generate_static_css()

