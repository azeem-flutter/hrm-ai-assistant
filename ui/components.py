"""
components.py
--------------
Small render helpers shared between the "live" response and the replayed
chat history in app.py.

IMPORTANT: results are rendered as a real HTML <table>, not st.dataframe.
st.dataframe renders through a canvas-based grid (glide-data-grid) that
pulls its colors from Streamlit's *own* theme engine — our CSS cannot
reach inside it, which is why it was rendering dark/cramped regardless
of styles.py. A plain HTML table is a normal DOM element, so every rule
in styles.py (full width, dark-blue header, alternating rows, hover)
applies to it directly.
"""

import base64
import html
from pathlib import Path

import pandas as pd

_ASSETS_DIR = Path(__file__).parent.parent / "assets"  # ui/ -> project root -> assets/


def get_logo_data_uri(filename: str = "logo.png") -> str:
    """Return the brand logo from /assets as a base64 data URI.

    The logo is embedded directly inside our own custom-positioned
    <div class="app-header"> markup (st.image() would render it as its own
    Streamlit element and we'd lose control over placement). If no logo file
    exists in /assets, a neutral built-in icon is used instead, so the app
    runs out of the box.
    """
    path = _ASSETS_DIR / filename
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="14" fill="#4f46e5"/>'
        '<text x="32" y="42" font-size="26" font-family="Arial" font-weight="700" '
        'text-anchor="middle" fill="#fff">HR</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode("ascii")


def render_result_badge(count: int) -> str:
    return f'<span class="result-badge">✅ {count} result(s) found</span>'


def render_clarify_badge() -> str:
    return '<span class="clarify-badge">🤔 Need a bit more info</span>'


def render_sql_pill(sql: str) -> str:
    return f'<div class="sql-pill">{html.escape(sql)}</div>'


def render_table(df: pd.DataFrame) -> str:
    """Renders a DataFrame as a themed, full-width, scrollable HTML table."""
    table_html = df.to_html(
        index=False,
        border=0,
        classes="hrm-table",
        escape=True,
        na_rep="—",
    )
    return f'<div class="table-scroll">{table_html}</div>'