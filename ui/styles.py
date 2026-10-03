"""
styles.py
---------
The entire visual design system for the chat UI, isolated from app.py so
the layout/markup logic and the styling can be edited independently.
get_css() returns one <style> block that app.py injects once via
st.markdown(..., unsafe_allow_html=True).

Design tokens (see :root):
  - background: ChatGPT-style flat dark charcoal (main #212121 / sidebar #171717)
  - surfaces: dark elevated cards for the header, bubbles, table, sql pill
  - accent: electric blue #3B82F6 (kept from the original palette, tuned
    for contrast on a dark background) — used only for focus rings, the
    send button, and small status accents
  - user bubbles: dark neutral surface, right-aligned, light text
  - AI bubbles: slightly lighter dark surface, left-aligned, light text
  - tables: full width, dark header row, subtly alternating dark rows
  - input: pill-shaped, floating, glowing on focus (ChatGPT-style composer)
  - scrollbars: slim, custom, dark-on-theme
"""

CSS = """
<style>
/* =====================================================================
   DESIGN TOKENS
   ===================================================================== */
:root {
    --bg-1: #FAFAF7;      /* main chat background — Claude light-mode warm white */
    --bg-2: #F4F3EE;      /* sidebar — slightly darker warm off-white */
    --surface: #FFFFFF;   /* header, bubbles, cards */
    --surface-2: #F0EFE9; /* input, hover states, table rows */
    --surface-3: #E8E6DE; /* raised hover / active states */

    --navy: #3D3929;      /* primary text (kept the var name so every
                              existing `color: var(--navy)` below just
                              works on the new light background) */
    --royal: #B45309;     /* accent — headings/links on light bg */
    --blue: #3B82F6;      /* accent — buttons, focus rings */
    --blue-soft: rgba(59, 130, 246, 0.12);
    --muted: #6B6860;     /* secondary text */
    --muted-2: #8F8C82;   /* tertiary/placeholder text */
    --white: #FFFFFF;     /* panel surface */
    --border: rgba(61, 57, 41, 0.10);
    --shadow-sm: 0 4px 18px rgba(0, 0, 0, 0.06);
    --shadow-md: 0 12px 40px rgba(0, 0, 0, 0.10);
    --radius-lg: 24px;
    --radius-md: 18px;

    --header-h: 96px; /* space reserved at the top for the fixed logo */
    --sidebar-w: 340px; /* sidebar width */
}

/* Light-native browser chrome (scrollbars, form controls). */
html { color-scheme: light only; }

html, body, .stApp,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"],
section[data-testid="stMain"] {
    background: var(--bg-1) !important;
    color: var(--navy);
    min-height: 100vh;
}

/* The fixed header bar and the fixed bottom bar (which wraps the chat
   input) are separate elements with their own opaque theme background —
   make both transparent so the flat dark background shows through
   everywhere, no seams. */
div[data-testid="stHeader"],
div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    position: relative;
    z-index: 1;
    max-width: 1320px !important;
    padding-top: calc(var(--header-h) + 12px) !important;
    padding-bottom: 150px !important;
    padding-left: 28px !important;
    padding-right: 28px !important;
}

*, *::before, *::after { box-sizing: border-box; }

* {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
        "Segoe UI", system-ui, sans-serif;
    letter-spacing: -0.006em;
}

html, body, .stApp, p, li, span, div { font-size: 15px; }

code, pre, .sql-pill {
    font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace !important;
    letter-spacing: 0 !important;
}

/* =====================================================================
   HEADER — logo only, fixed to the viewport (not just sticky), centered,
   in a rounded card. Always visible, never scrolls away, never pushed
   around by long answers.
   ===================================================================== */
div[data-testid="element-container"]:has(> .app-header) {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    pointer-events: none;
}

.app-header {
    position: fixed;
    top: 0;
    left: var(--sidebar-w);
    right: 0;
    height: var(--header-h);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    pointer-events: auto;
}

.brand-logo-wrap {
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 9px;
    background: #FFFFFF;
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    pointer-events: auto;
}

.brand-logo {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}

/* =====================================================================
   SIDEBAR — Quick questions + Clear live here now (ChatGPT-style dark
   sidebar, distinct from the main chat pane).
   ===================================================================== */
section[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border);
    width: var(--sidebar-w) !important;
    min-width: var(--sidebar-w) !important;
    max-width: var(--sidebar-w) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 22px;
    width: var(--sidebar-w) !important;
}

.sidebar-title {
    color: var(--navy);
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 4px 0 12px 2px;
}

.sidebar-title.secondary { color: var(--muted); margin-top: 22px; }

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 8px !important; }

section[data-testid="stSidebar"] button {
    min-height: 42px !important;
    width: 100% !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--navy) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    transition: background 150ms ease, border-color 150ms ease, transform 150ms ease !important;
}

section[data-testid="stSidebar"] button:hover:not(:disabled) {
    background: var(--surface-3) !important;
    border-color: rgba(96, 165, 250, 0.35) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}

section[data-testid="stSidebar"] button:disabled {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}

/* Clear button — visually distinct (muted/destructive), still same shape. */
section[data-testid="stSidebar"] .clear-chat-btn button {
    color: #F87171 !important;
    border-color: rgba(248, 113, 113, 0.22) !important;
    justify-content: center !important;
}

section[data-testid="stSidebar"] .clear-chat-btn button:hover:not(:disabled) {
    background: rgba(248, 113, 113, 0.10) !important;
    border-color: rgba(248, 113, 113, 0.40) !important;
    color: #FCA5A5 !important;
}

hr.sidebar-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 18px 0 0;
}

/* =====================================================================
   CHAT WORKSPACE SHELL
   ===================================================================== */
.chat-shell {
    width: 100%;
    min-height: 360px;
    padding: 20px 22px 16px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: 28px;
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.empty-state { min-height: 300px; display: flex; align-items: center; justify-content: center; text-align: center; }
.empty-state-inner { max-width: 470px; padding: 42px 20px; }

.empty-icon {
    width: 58px; height: 58px; margin: 0 auto 17px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 18px;
    background: var(--surface-2);
    color: var(--royal); font-size: 25px;
    box-shadow: inset 0 0 0 1px var(--border);
}

.empty-title { margin: 0; color: var(--navy); font-size: 20px; font-weight: 750; letter-spacing: -0.02em; }
.empty-text { margin: 9px auto 0; color: var(--muted); font-size: 14.5px; line-height: 1.7; }

/* =====================================================================
   CHAT MESSAGES — user right / AI left, no overlap, clean bubble shapes
   ===================================================================== */
div[data-testid="stChatMessage"] {
    width: 100% !important;
    margin: 0 0 16px 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: flex-end !important;
    gap: 10px !important;
}

div[data-testid="stChatMessageAvatarUser"],
div[data-testid="stChatMessageAvatarAssistant"] {
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    border-radius: 11px !important;
    flex-shrink: 0 !important;
}

div[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--surface-2) !important;
    color: var(--royal) !important;
}

/* Neutralize Streamlit's own inner wrapper layers so none of them paint
   their own background over our bubble. */
div[data-testid="stChatMessage"] .stMarkdown,
div[data-testid="stChatMessage"] .stVerticalBlock,
div[data-testid="stChatMessage"] div[data-testid="element-container"] {
    background: transparent !important;
}

/* --- USER turn: ChatGPT-style bubble — dark neutral surface, right
   aligned, fully rounded. Right edge lines up with the AI bubble/table
   (same 8% inset used on the assistant side) instead of floating way
   out to the left, and the message text itself is centered inside the
   bubble. --- */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row !important;
    justify-content: flex-end !important;
    padding-left: 8% !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:last-child {
    order: 1;
    flex: 0 0 auto !important;
    align-self: flex-end !important;
    width: fit-content;
    max-width: 1040px;
    margin: 0;
    padding: 14px 20px;
    background: var(--surface-2) !important;
    color: var(--navy) !important;
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    overflow: hidden;
    text-align: center;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) p {
    color: var(--navy) !important;
    text-align: center;
}

/* User avatar — same blue as the composer's send button, instead of the
   old neutral dark-gray gradient. Ordered last so it sits at the far
   right, right after the bubble. */
div[data-testid="stChatMessageAvatarUser"] {
    order: 2;
    background: linear-gradient(145deg, #3B82F6, #2563EB) !important;
    color: white !important;
}

/* --- AI turn: bubble on the left --- */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    flex-direction: row !important;
    padding-right: 8% !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
    width: 100%;
    max-width: 1040px;
    padding: 19px 22px;
    background: var(--surface) !important;
    color: var(--navy);
    border: 1px solid var(--border);
    border-radius: 22px 22px 22px 4px;
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li {
    font-size: 15px !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}

/* =====================================================================
   RESULT / CLARIFY BADGES
   ===================================================================== */
.result-badge, .clarify-badge {
    display: inline-flex; align-items: center;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px; font-weight: 750; letter-spacing: 0.01em;
    margin-bottom: 12px;
}

.result-badge { background: rgba(5, 150, 105, 0.12); color: #047857; border: 1px solid rgba(5, 150, 105, 0.35); }
.clarify-badge { background: rgba(217, 119, 6, 0.12); color: #B45309; border: 1px solid rgba(217, 119, 6, 0.35); }

/* =====================================================================
   SQL DISPLAY PILL
   ===================================================================== */
.sql-pill {
    width: 100%;
    margin-top: 16px;
    padding: 14px 17px;
    background: #1E1E1E !important;
    border: 1px solid var(--border);
    border-radius: 14px;
    color: #93C5FD;
    font-size: 13px;
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

/* =====================================================================
   FULL-WIDTH RESULT TABLES
   Results render as a real HTML <table> (see components.render_table) —
   not st.dataframe, which pulls its colors from Streamlit's own theme
   engine and ignores page CSS.
   ===================================================================== */
.table-scroll {
    width: 100%;
    max-height: 480px;
    overflow: auto;
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    margin: 14px 0 8px;
    background: var(--surface);
}

.table-scroll table.hrm-table {
    width: 100% !important;
    min-width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    background: var(--surface);
    color: var(--navy);
}

.table-scroll table.hrm-table thead th {
    background: #262626 !important;
    color: #FFFFFF !important;
    font-weight: 700;
    text-align: left;
    padding: 13px 16px;
    white-space: nowrap;
    position: sticky;
    top: 0;
    z-index: 1;
}

.table-scroll table.hrm-table tbody td {
    padding: 11px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--navy);
    white-space: nowrap;
}

.table-scroll table.hrm-table tbody tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.02);
}

.table-scroll table.hrm-table tbody tr:hover {
    background: var(--blue-soft);
}

/* =====================================================================
   STREAMLIT ALERTS
   ===================================================================== */
div[data-testid="stAlert"] { border-radius: 14px !important; border-width: 1px !important; }

/* =====================================================================
   CHAT INPUT — floating pill, glowing focus ring (ChatGPT-style composer)
   ===================================================================== */
div[data-testid="stChatInput"],
div[data-testid="stChatInputContainer"],
div[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stChatInput"] {
    position: fixed !important;
    left: calc(50% + (var(--sidebar-w) / 2)) !important;
    transform: translateX(-50%) !important;
    bottom: 24px !important;
    width: min(calc(100% - var(--sidebar-w) - 56px), 1264px) !important;
    z-index: 1000 !important;
    padding: 6px !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.45), 0 2px 8px rgba(0, 0, 0, 0.30) !important;
    transition: box-shadow 200ms ease, border-color 200ms ease, opacity 200ms ease !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: rgba(59, 130, 246, 0.55) !important;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45), 0 0 0 4px rgba(59, 130, 246, 0.18) !important;
}

div[data-testid="stChatInput"]:has(textarea:disabled) {
    opacity: 0.55 !important;
}

div[data-testid="stChatInput"] textarea {
    min-height: 48px !important;
    max-height: 140px !important;
    padding: 13px 54px 13px 22px !important;
    background: transparent !important;
    color: var(--navy) !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    border-radius: 999px !important;
    -webkit-text-fill-color: var(--navy) !important;
    caret-color: var(--blue) !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted-2) !important;
    opacity: 1 !important;
}

div[data-testid="stChatInput"] button {
    width: 40px !important;
    height: 40px !important;
    margin-right: 6px !important;
    border: 0 !important;
    border-radius: 14px !important;
    background: linear-gradient(145deg, #3B82F6, #2563EB) !important;
    color: white !important;
    box-shadow: 0 5px 13px rgba(37, 99, 235, 0.35) !important;
    transition: transform 150ms ease, box-shadow 150ms ease !important;
}

div[data-testid="stChatInput"] button:hover {
    transform: translateY(-1px) scale(1.05);
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.45) !important;
}

div[data-testid="stChatInput"] button svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* =====================================================================
   SCROLLBARS — slim, on-theme, dark
   ===================================================================== */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.16); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.28); }

* { scrollbar-width: thin; scrollbar-color: rgba(255, 255, 255, 0.16) transparent; }

/* =====================================================================
   RESPONSIVE
   ===================================================================== */
@media (max-width: 900px) {
    :root { --header-h: 84px; }

    .block-container { padding-left: 15px !important; padding-right: 15px !important; }
    .app-header { left: 0 !important; }
    .brand-logo-wrap { width: 60px; height: 60px; border-radius: 16px; }

    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) { padding-left: 5% !important; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:last-child { max-width: 88%; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) { padding-right: 2% !important; }

    div[data-testid="stChatInput"] {
        left: 50% !important;
        width: calc(100% - 28px) !important;
        bottom: 14px !important;
    }
}

@media (max-width: 600px) {
    :root { --header-h: 72px; }

    .brand-logo-wrap { width: 52px; height: 52px; border-radius: 14px; }
    .empty-state { min-height: 250px; }

    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li { font-size: 12.5px !important; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:last-child { max-width: 92%; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


def get_css() -> str:
    """Returns the full CSS block, ready to pass to st.markdown(..., unsafe_allow_html=True)."""
    return CSS