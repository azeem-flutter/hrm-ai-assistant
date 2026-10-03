"""Agentic orchestration for the HRM SQL assistant.

The existing query_service remains the SQL-generation brain. This module adds
an explicit LangGraph workflow around it:

question -> generate -> execute -> repair -> execute -> finish

LangGraph is optional at runtime. If it is not installed, the same workflow is
executed by a small Python fallback so the existing app does not break merely
because the optional dependency is unavailable.
"""
from typing import Any, Dict, Optional, TypedDict

from config import config
from integrations import db, llm, rag
from core import query_service, sql_safety


class GraphState(TypedDict, total=False):
    question: str
    history: list
    sql: str
    result: Any
    error: str
    attempt: int
    status: str


def _repair_sql(question: str, sql: str, error: str) -> str:
    """Repair a real Oracle execution error with small verified context."""
    schema = db.build_table_chunks()
    try:
        retrieved = rag.retrieve_for_prompt(question, schema)
        schema_text = "\n\n".join(retrieved.values()) if retrieved else db.build_schema_text()
    except Exception:
        schema_text = db.build_schema_text()

    relationships = db.build_relationships_text(db.get_connection())
    template = """You are repairing Oracle 11g SQL after a real database execution error.
Return ONLY one safe SELECT statement. Do not explain anything.

USER QUESTION:
{question}

PREVIOUS SQL:
{sql}

ORACLE ERROR:
{error}

RELEVANT VERIFIED SCHEMA:
{schema_text}

VERIFIED RELATIONSHIPS:
{relationships}

Rules:
- Oracle 11g only; never use LIMIT, TOP, OFFSET, FETCH FIRST, or PostgreSQL syntax.
- Use only tables and columns present in the verified schema.
- Preserve the user's intended result; do not change the meaning merely to silence the error.
- Use only real FK relationships when joining tables.
- Return one read-only SELECT and nothing else.
"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(template).format(
            question=question, sql=sql, error=error,
            schema_text=schema_text, relationships=relationships
        )
    except ImportError:
        prompt = template.format(
            question=question, sql=sql, error=error,
            schema_text=schema_text, relationships=relationships
        )

    return sql_safety.clean_sql(llm.call_local_model([
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Repair the SQL now."},
    ]))


def _generate(state: GraphState) -> GraphState:
    sql = query_service.question_to_sql(state["question"], history=state.get("history"))
    return {"sql": sql, "status": "generated", "attempt": state.get("attempt", 0)}


def _execute(state: GraphState) -> GraphState:
    sql = sql_safety.clean_sql(state.get("sql", ""))
    if sql.upper().startswith("CLARIFY:"):
        return {"sql": sql, "status": "clarify"}
    if not sql_safety.is_safe_select(sql):
        return {"sql": sql, "error": "Generated query was not a safe SELECT statement.", "status": "failed"}
    try:
        result = db.run_query(sql)
        if not result.empty:
            rag.learn_successful_query(state["question"], sql)
        return {"sql": sql, "result": result, "status": "success", "error": ""}
    except Exception as exc:
        return {"sql": sql, "error": str(exc), "status": "execute_failed"}


def _repair(state: GraphState) -> GraphState:
    attempt = int(state.get("attempt", 0)) + 1
    if attempt > config.MAX_SQL_REPAIR_RETRIES:
        return {"attempt": attempt, "status": "failed"}
    try:
        repaired = _repair_sql(state["question"], state.get("sql", ""), state.get("error", ""))
        return {"sql": repaired, "attempt": attempt, "status": "repaired"}
    except Exception as exc:
        return {"attempt": attempt, "error": f"SQL repair failed: {exc}", "status": "failed"}


def _route_after_generate(state: GraphState) -> str:
    status = state.get("status")
    return "finish" if status in {"clarify", "failed"} else "execute"


def _route_after_execute(state: GraphState) -> str:
    status = state.get("status")
    if status in {"success", "clarify", "failed"}:
        return "finish"
    if int(state.get("attempt", 0)) < config.MAX_SQL_REPAIR_RETRIES:
        return "repair"
    return "finish"


def _build_langgraph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("generate", _generate)
    graph.add_node("execute", _execute)
    graph.add_node("repair", _repair)
    graph.add_edge(START, "generate")
    graph.add_conditional_edges("generate", _route_after_generate, {
        "execute": "execute",
        "finish": END,
    })
    graph.add_conditional_edges("execute", _route_after_execute, {
        "repair": "repair",
        "finish": END,
    })
    graph.add_edge("repair", "execute")
    return graph.compile()


_GRAPH = None


def run(question: str, history: Optional[list] = None) -> Dict[str, Any]:
    """Run the agent workflow and return a plain dict for app.py."""
    global _GRAPH
    state: GraphState = {"question": question, "history": history or [], "attempt": 0}
    try:
        if _GRAPH is None:
            _GRAPH = _build_langgraph()
        return dict(_GRAPH.invoke(state))
    except ImportError:
        # Dependency-free fallback with identical states/limits.
        state = _generate(state)
        if state.get("status") not in {"clarify", "failed"}:
            state = _execute(state)
        while state.get("status") == "execute_failed" and state.get("attempt", 0) < config.MAX_SQL_REPAIR_RETRIES:
            state = {**state, **_repair(state)}
            if state.get("status") == "failed":
                break
            state = {**state, **_execute(state)}
        return dict(state)
