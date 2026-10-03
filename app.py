"""
app.py
------
HRM AI Assistant — Streamlit entry point.

This file only wires the UI together. Every other concern lives in its
own module/package:
    config/config.py              -> environment settings
    integrations/db.py            -> Oracle connection, schema introspection, queries
    integrations/llm.py           -> system prompt + Ollama/Groq call
    integrations/rag.py           -> Vanna / keyword retrieval
    core/sql_safety.py            -> SQL cleaning / validation / read-only guardrails
    core/query_service.py         -> orchestrates question -> SQL (with retries)
    core/agent_graph.py           -> LangGraph generate -> execute -> repair loop
    core/conversation_context.py  -> resolves safe multi-turn follow-up context
    ui/styles.py                  -> the entire visual design system (CSS)
    ui/components.py              -> small render helpers (badges, tables, logo)

PROCESSING LOCK — READ THIS BEFORE TOUCHING THE INPUT/PROCESSING SECTION:
Streamlit reruns the whole script on every widget interaction, and — this
is the part that bit us before — a NEW interaction (another chip click, a
fresh chat_input submit) does not wait politely; it ABORTS whatever script
run is currently in flight and starts a new one immediately. That meant a
quick second click while the first question was still "Thinking locally…"
could cancel the first LLM/DB call mid-request.
The fix is a two-phase lock using `st.session_state.is_processing`:
    Phase 1 (fast): a new question is captured, we set is_processing=True,
        stash the question, and immediately st.rerun() WITHOUT doing any
        of the slow LLM/DB work yet.
    Phase 2: this next run renders the sidebar buttons and chat_input as
        `disabled=True` FIRST (top of script) — Streamlit pushes that to
        the browser as soon as those widgets are drawn — and only THEN
        does the actual slow processing. Because the disabled state is
        already live in the browser before the slow work starts, a stray
        click during that window has nothing to interrupt.
    Phase 3: processing finishes, is_processing resets to False, rerun
        re-enables everything.

SETUP (one-time):
    1. Install Ollama: https://ollama.com/download
    2. Pull the model:  ollama pull qwen2.5-coder:3b
    3. pip install -r requirements.txt
    4. Fill in your .env file (copy .env.example -> .env)

RUN:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from PIL import Image

# Proper package imports -- config/, core/, integrations/, ui/ are real
# Python packages (each has an __init__.py). Streamlit puts this file's
# folder (the project root) on sys.path automatically when you run
# `streamlit run app.py` from the project root, so these just work with no
# extra bootstrap code.
from config import config
from core import conversation_context, query_service, agent_graph
from core.sql_safety import is_safe_select
from integrations import db
from ui import components, styles

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

# =============================================================================
# PAGE CONFIG  (must be the first Streamlit call)
# =============================================================================
try:
    _page_icon = Image.open(ASSETS_DIR / "logo.png")
except Exception:
    _page_icon = config.APP_ICON  # fallback emoji if the asset is ever missing

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(styles.get_css(), unsafe_allow_html=True)

# =============================================================================
# STARTUP SAFETY CHECK — confirm the DB account is actually least-privilege
# =============================================================================
_risky_privs = db.verify_read_only_user()
if _risky_privs:
    # st.error(
    #     "⚠️ Security check failed: the connected database account "
    #     f"(**{config.ORACLE_USER}**) holds privileges beyond read-only "
    #     f"access: {', '.join(_risky_privs)}.\n\n"
    #     "This app relies on the DB account being genuinely read-only as "
    #     "a second line of defense. Please connect with a dedicated "
    #     "least-privilege user (GRANT CREATE SESSION + SELECT only) "
    #     "before using this app against real data."
    # )
    # st.stop()
    pass

# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_history" not in st.session_state:
    st.session_state.model_history = []
if "conversation_context" not in st.session_state:
    # Compact successful-turn state used only for resolving follow-up questions.
    st.session_state.conversation_context = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "pending_clarification" not in st.session_state:
    # Set right after we ask the user a CLARIFY question (either origin --
    # conversation_context's own ambiguity check, or the LLM/preflight
    # layer inside query_service). Consumed (read + reset to None) at the
    # start of the very next turn, so it's used to merge at most once.
    st.session_state.pending_clarification = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "active_question" not in st.session_state:
    st.session_state.active_question = None

LOCKED = st.session_state.is_processing  # read once; used to disable every input below

# =============================================================================
# SIDEBAR — Quick questions + Clear (moved out of the main pane)
# =============================================================================
SUGGESTIONS = [
    "Show all IT employees who joined after 2023",
    "How many employees are in each department?",
    "Top 10 highest paid employees",
    "Departments with no employees",
]

with st.sidebar:
    st.markdown('<div class="sidebar-title">Quick questions</div>', unsafe_allow_html=True)
    for i, example in enumerate(SUGGESTIONS):
        if st.button(example, key=f"chip_{i}", use_container_width=True, disabled=LOCKED):
            st.session_state.pending_question = example
            st.rerun()

    st.markdown('<hr class="sidebar-divider" />', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title secondary">Session</div>', unsafe_allow_html=True)
    st.markdown('<div class="clear-chat-btn">', unsafe_allow_html=True)
    if st.button("Clear chat", key="clear_chat", use_container_width=True, disabled=LOCKED):
        st.session_state.messages = []
        st.session_state.model_history = []
        st.session_state.conversation_context = []
        st.session_state.pending_question = None
        st.session_state.pending_clarification = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# HEADER — logo only, fixed + centered + rounded
# =============================================================================
st.markdown(
    f"""
    <div class="app-header">
        <div class="brand-logo-wrap">
            <img class="brand-logo" src="{components.get_logo_data_uri()}" alt="HRM AI Assistant" />
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CHAT HISTORY
# =============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg.get("clarify"):
            st.markdown(
                components.render_clarify_badge(), unsafe_allow_html=True
            )
            st.markdown(msg["clarify"])
        elif msg.get("error"):
            st.warning(msg["error"])
            if msg.get("sql"):
                st.markdown(
                    f'<div class="sql-pill">{msg["sql"]}</div>',
                    unsafe_allow_html=True,
                )
        elif msg.get("df") is not None and not msg["df"].empty:
            st.markdown(
                components.render_result_badge(len(msg["df"])),
                unsafe_allow_html=True,
            )
            st.markdown(
                components.render_table(msg["df"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                components.render_sql_pill(msg.get("sql", "")),
                unsafe_allow_html=True,
            )
        else:
            st.info("No matching records found.")
            st.markdown(
                components.render_sql_pill(msg.get("sql", "")),
                unsafe_allow_html=True,
            )

# =============================================================================
# INPUT
# =============================================================================
typed_question = st.chat_input(
    "Ask about employees, departments, salaries…",
    disabled=LOCKED,
)
new_question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

# --- Phase 1: capture + lock, but do NOT process yet -----------------------
# See the module docstring: reruning immediately (before any slow work)
# is what actually gets the disabled inputs onto the screen in time.
if new_question and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.active_question = new_question
    st.rerun()

# =============================================================================
# PROCESSING — only runs once the lock is already visible in the browser
# =============================================================================
if st.session_state.is_processing and st.session_state.active_question:
    question = st.session_state.active_question

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally…"):
            try:
                effective_question = question
                context_decision = None
                pending = st.session_state.pending_clarification
                st.session_state.pending_clarification = None

                try:
                    context_decision = conversation_context.resolve_question(
                        question,
                        st.session_state.conversation_context[-config.CONVERSATION_CONTEXT_TURNS:],
                        enabled=config.ENABLE_CONVERSATION_CONTEXT,
                        pending_clarification=pending,
                    )
                    if context_decision.action == "CLARIFY":
                        raw_response = f"CLARIFY: {context_decision.clarification}"
                        graph_result = {"status": "clarify", "sql": raw_response, "attempt": 0}
                    else:
                        effective_question = context_decision.effective_question
                        history = st.session_state.model_history[-(config.MAX_HISTORY_TURNS * 2):]
                        # LangGraph now owns generation -> Oracle execution ->
                        # bounded repair/retry. query_service remains the
                        # generation/validation brain inside the graph.
                        graph_result = agent_graph.run(effective_question, history=history)
                        raw_response = graph_result.get("sql", "")
                except Exception as context_exc:
                    print(f"[CONTEXT/GRAPH] failed — falling back to original pipeline: {context_exc}")
                    history = st.session_state.model_history[-(config.MAX_HISTORY_TURNS * 2):]
                    raw_response = query_service.question_to_sql(question, history=history)
                    graph_result = {"status": "generated", "sql": raw_response, "attempt": 0}

                if raw_response.strip().upper().startswith("CLARIFY:"):
                    clarify_msg = raw_response.split(":", 1)[1].strip()
                    st.markdown(components.render_clarify_badge(), unsafe_allow_html=True)
                    st.markdown(clarify_msg)
                    st.session_state.messages.append({"role": "assistant", "clarify": clarify_msg})
                    st.session_state.model_history.append({"role": "user", "content": question})
                    st.session_state.model_history.append({"role": "assistant", "content": raw_response})
                    st.session_state.pending_clarification = {
                        "original_question": effective_question,
                        "clarify_asked": clarify_msg,
                    }

                elif graph_result.get("status") == "success":
                    sql = graph_result.get("sql", raw_response)
                    df = graph_result.get("result")
                    if df is None:
                        raise RuntimeError("Workflow reported success but returned no result.")
                    if df.empty:
                        st.info("No matching records found.")
                    else:
                        st.markdown(components.render_result_badge(len(df)), unsafe_allow_html=True)
                        st.markdown(components.render_table(df), unsafe_allow_html=True)
                    st.markdown(components.render_sql_pill(sql), unsafe_allow_html=True)
                    retry_count = int(graph_result.get("attempt", 0))
                    if retry_count:
                        st.caption(f"SQL was repaired and validated after {retry_count} retries.")
                    st.session_state.messages.append({"role": "assistant", "df": df, "sql": sql})
                    st.session_state.model_history.append({"role": "user", "content": effective_question})
                    st.session_state.model_history.append({"role": "assistant", "content": sql})
                    st.session_state.conversation_context.append(
                        conversation_context.make_context_record(
                            question=question,
                            effective_question=effective_question,
                            sql=sql,
                        )
                    )
                    st.session_state.conversation_context = st.session_state.conversation_context[-config.CONVERSATION_CONTEXT_TURNS:]

                else:
                    error_msg = graph_result.get("error") or "The SQL workflow could not produce a valid executable query after the allowed retries."
                    st.error(error_msg)
                    if raw_response:
                        st.markdown(components.render_sql_pill(raw_response), unsafe_allow_html=True)
                    retry_count = int(graph_result.get("attempt", 0))
                    if retry_count:
                        st.caption(f"Stopped after {retry_count} SQL repair retries to avoid an endless loop.")
                    st.session_state.messages.append({"role": "assistant", "error": error_msg, "sql": raw_response})
                    st.session_state.model_history.append({"role": "user", "content": question})
                    st.session_state.model_history.append({"role": "assistant", "content": f"(No valid SQL - {error_msg})"})

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "error": error_msg, "sql": ""})
                st.session_state.model_history.append({"role": "user", "content": question})
                st.session_state.model_history.append({"role": "assistant", "content": f"(No valid SQL - error:{e})"})

    # Phase 3: unlock and refresh so the next question can be picked.
    st.session_state.is_processing = False
    st.session_state.active_question = None
    st.rerun()