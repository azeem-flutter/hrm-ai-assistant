"""
db.py
-----
Everything that talks to Oracle: a single shared connection, live schema/FK
introspection (so the LLM sees real tables/columns/joins instead of
guessing), and the final query execution + result sanitation.
"""

import numpy as np
import oracledb
import pandas as pd
import streamlit as st

from config import config

if config.ORACLE_LIB_DIR:
    oracledb.init_oracle_client(lib_dir=config.ORACLE_LIB_DIR)


@st.cache_resource
def get_connection():
    """Opens (once, cached for the process lifetime) a SINGLE real Oracle
    connection that the whole app reuses — no pool, no per-query
    acquire()/release(). Every function below just calls get_connection()
    and uses it directly.

    Trade-off (accepted intentionally): with one shared connection,
    concurrent queries from different Streamlit sessions/threads will
    serialize on this single connection instead of running in parallel
    the way a pool would allow. If that ever becomes a bottleneck, this
    is the function to swap back to a pool.
    """
    return oracledb.connect(
        user=config.ORACLE_USER,
        password=config.ORACLE_PASSWORD,
        dsn=config.ORACLE_DSN,
    )


# --- Old pool-based version (kept for reference / easy rollback) --------
# @st.cache_resource
# def get_pool():
#     """Opens (once, cached for the process lifetime) a small pool of
#     real Oracle connections instead of a single shared connection.
#
#     Why a pool instead of one connection:
#       - ping_interval=60 makes the pool health-check a connection before
#         handing it out if it has been idle 60+ seconds, and silently
#         replaces it if it's dead — so an overnight idle timeout / dropped
#         connection doesn't take the whole app down.
#       - Each query borrows its own connection (pool.acquire()) instead of
#         every user sharing one connection object concurrently, which is
#         not safe across Streamlit's per-session threads.
#     """
#     return oracledb.connect(
#         user=config.ORACLE_USER,
#         password=config.ORACLE_PASSWORD,
#         dsn=config.ORACLE_DSN,
#         min=2,
#         max=10,
#         increment=1,
#         ping_interval=60,
#     )
# --------------------------------------------------------------------------


# System/role privileges that let a session write, execute PL/SQL, or
# change grants even though the app only ever *sends* SELECT text. If
# the connected DB account has any of these, the app-layer SELECT-only
# check in sql_safety.py is your only line of defense — this check
# makes sure that's not actually the case.
_DANGEROUS_PRIVILEGES = (
    "CREATE ANY", "INSERT ANY", "UPDATE ANY", "DELETE ANY", "DROP ANY",
    "ALTER ANY", "EXECUTE ANY", "GRANT ANY", "CREATE TABLE",
    "CREATE PROCEDURE", "CREATE SEQUENCE", "CREATE VIEW",
    "UNLIMITED TABLESPACE", "DBA", "IMP_FULL_DATABASE", "EXP_FULL_DATABASE",
)


@st.cache_resource
def verify_read_only_user() -> list:
    """Best-effort startup check: confirms the connected DB account does
    not itself hold write/DDL/admin privileges, beyond what the app-layer
    SELECT-only filter (sql_safety.is_safe_select) assumes.

    This does NOT replace configuring a real least-privilege Oracle user
    (that has to be done in the database itself, e.g.:
        CREATE USER hr_readonly IDENTIFIED BY ...;
        GRANT CREATE SESSION TO hr_readonly;
        GRANT SELECT ON hr.employees TO hr_readonly;
        -- (repeat GRANT SELECT per table, or use a read-only role)
    ). It only catches the case where that setup was skipped or a
    higher-privileged account was used by mistake, and surfaces it
    clearly at startup instead of relying on the app layer alone.

    Returns the list of concerning privileges found (empty = clean).
    Cached for the process lifetime — same rationale as get_connection.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT privilege FROM user_sys_privs
            UNION
            SELECT granted_role FROM user_role_privs
            """
        )
        held = {row[0].upper() for row in cur.fetchall()}
    finally:
        cur.close()

    findings = sorted(
        p for p in held
        if any(p.startswith(bad) or bad in p for bad in _DANGEROUS_PRIVILEGES)
    )
    return findings


def _fetch_relationship_rows(conn) -> list:
    """Raw (child_table, child_column, parent_table, parent_column) rows
    for every FK relationship in the schema, straight from Oracle's data
    dictionary. Shared by build_relationships_text (one flat text block)
    and build_table_chunks (grouped per table) so the query lives in one
    place."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            a.table_name       AS child_table,
            a_cols.column_name AS child_column,
            c.table_name       AS parent_table,
            c_cols.column_name AS parent_column
        FROM all_constraints a
        JOIN all_cons_columns a_cols
          ON a.constraint_name = a_cols.constraint_name
         AND a.owner = a_cols.owner
        JOIN all_constraints c
          ON a.r_constraint_name = c.constraint_name
         AND a.r_owner = c.owner
        JOIN all_cons_columns c_cols
          ON c.constraint_name = c_cols.constraint_name
         AND c.owner = c_cols.owner
         AND a_cols.position = c_cols.position
        WHERE a.constraint_type = 'R'
          AND a.owner = :owner
        ORDER BY a.table_name, a_cols.position
        """,
        owner=config.SCHEMA_OWNER,
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def build_relationships_text(conn) -> str:
    """Real FK relationships from Oracle's data dictionary, so the model
    doesn't have to guess joins from similar-looking column names."""
    rows = _fetch_relationship_rows(conn)
    if not rows:
        return ""
    lines = [f"{ct}.{cc} = {pt}.{pc}" for ct, cc, pt, pc in rows]
    return "\n".join(lines)


@st.cache_data(ttl=config.SCHEMA_CACHE_TTL)
def build_schema_text() -> str:
    """Compact 'table(col,col,...)' text block + relationships, used as
    the schema context the model sees in its system prompt."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT table_name, column_name
        FROM all_tab_columns
        WHERE owner = :owner
        ORDER BY table_name, column_id
        """,
        owner=config.SCHEMA_OWNER,
    )
    rows = cur.fetchall()
    cur.close()

    tables = {}
    for table_name, column_name in rows:
        tables.setdefault(table_name, []).append(column_name)

    if not tables:
        return "(No tables found for this schema owner — check DB_USER in .env.)"

    lines = [f"{table}({','.join(cols)})" for table, cols in tables.items()]
    tables_block = "\n".join(lines)

    fk_block = build_relationships_text(conn)

    if fk_block:
        return (
            f"{tables_block}\n\nRELATIONSHIPS (use these exact joins, "
            f"do not guess by column name):\n{fk_block}"
        )
    return tables_block


@st.cache_data(ttl=config.SCHEMA_CACHE_TTL)
def build_fk_pairs() -> list:
    """[(CHILD_TABLE, CHILD_COLUMN, PARENT_TABLE, PARENT_COLUMN), ...],
    all upper-cased, straight from the same FK rows build_table_chunks
    uses. Used by query_service.py's deterministic ranking template to
    find the real join column between two tables mentioned in a
    question (e.g. EMPLOYEES -> DEPARTMENTS) without guessing by
    similar-looking column names."""
    conn = get_connection()
    rows = _fetch_relationship_rows(conn)
    return [
        (ct.upper(), cc.upper(), pt.upper(), pc.upper())
        for ct, cc, pt, pc in rows
    ]


@st.cache_data(ttl=config.SCHEMA_CACHE_TTL)
def build_schema_dict() -> dict:
    """{TABLE_NAME: {COLUMN_NAME, ...}} for validating model output and
    catching hallucinated columns before they ever reach the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT table_name, column_name
        FROM all_tab_columns
        WHERE owner = :owner
        """,
        owner=config.SCHEMA_OWNER,
    )
    rows = cur.fetchall()
    cur.close()

    schema: dict = {}
    for table_name, column_name in rows:
        schema.setdefault(table_name.upper(), set()).add(column_name.upper())
    return schema


@st.cache_data(ttl=config.SCHEMA_CACHE_TTL)
def build_table_chunks() -> dict:
    """{TABLE_NAME: chunk_text} — one small, self-contained retrievable
    chunk per table, instead of the single giant schema blob that
    build_schema_text() produces.

    Each chunk holds that table's own column list plus every FK
    relationship that touches it, whether the table is on the child or
    the parent side (e.g. EMPLOYEES gets both "EMPLOYEES.DEPARTMENT_ID =
    DEPARTMENTS.DEPARTMENT_ID" and "JOB_HISTORY.EMPLOYEE_ID =
    EMPLOYEES.EMPLOYEE_ID"), so a single table's chunk carries enough
    context on its own to write a correct join — this is what rag.py
    embeds and retrieves per-table.

    Cached the same way as build_schema_text/build_schema_dict, and
    keyed for rag.py's cache off its own content, so adding a new table
    to the database is picked up automatically on the next cache refresh
    (or app restart) with no code change needed here.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT table_name, column_name
        FROM all_tab_columns
        WHERE owner = :owner
        ORDER BY table_name, column_id
        """,
        owner=config.SCHEMA_OWNER,
    )
    rows = cur.fetchall()
    cur.close()

    tables = {}
    for table_name, column_name in rows:
        tables.setdefault(table_name, []).append(column_name)

    if not tables:
        return {}

    fk_rows = _fetch_relationship_rows(conn)

    # Group each FK line under every table it touches — both the child
    # (foreign-key-holding) side and the parent (referenced) side.
    relationships_by_table: dict = {}
    for child_table, child_col, parent_table, parent_col in fk_rows:
        line = f"{child_table}.{child_col} = {parent_table}.{parent_col}"
        relationships_by_table.setdefault(child_table, []).append(line)
        relationships_by_table.setdefault(parent_table, []).append(line)

    chunks = {}
    for table, cols in tables.items():
        chunk_lines = [f"{table}({','.join(cols)})"]

        table_rels = relationships_by_table.get(table)
        if table_rels:
            # De-dupe while preserving order — a self-referencing FK
            # (e.g. EMPLOYEES.MANAGER_ID = EMPLOYEES.EMPLOYEE_ID) would
            # otherwise get added twice, once per side.
            seen = set()
            unique_rels = []
            for line in table_rels:
                if line not in seen:
                    seen.add(line)
                    unique_rels.append(line)
            chunk_lines.append("RELATIONSHIPS:")
            chunk_lines.extend(unique_rels)

        chunks[table] = "\n".join(chunk_lines)

    return chunks


@st.cache_data(ttl=config.SCHEMA_CACHE_TTL)
def build_relationship_graph() -> dict:
    """{TABLE_NAME: {directly-FK-connected TABLE_NAME, ...}} — an
    undirected adjacency map built from the same FK rows as
    build_table_chunks(), used by query_service.py to pull in "bridge"
    tables that a query's JOIN path needs but that RAG's keyword/embedding
    retrieval wouldn't surface on its own.

    Why this matters: retrieval ranks tables by how well their own text
    matches the QUESTION's wording. A question like "region name, country
    name, and number of departments per country" never mentions
    "location", so LOCATIONS can easily miss the cut — even though
    DEPARTMENTS -> LOCATIONS -> COUNTRIES is the only real join path
    between them. Without LOCATIONS in context, the model has no correct
    way to connect the two and guesses a plausible-looking but wrong join
    instead (this is what produced the ORA-01722 error: joining
    COUNTRIES.COUNTRY_ID directly to DEPARTMENTS.LOCATION_ID, a text
    column against a number column, because the real bridge wasn't in
    front of it).
    """
    conn = get_connection()
    fk_rows = _fetch_relationship_rows(conn)

    graph: dict = {}
    for child_table, _child_col, parent_table, _parent_col in fk_rows:
        graph.setdefault(child_table, set()).add(parent_table)
        graph.setdefault(parent_table, set()).add(child_table)
    return graph


def _sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Oracle NULLs (and any NaN/NaT/Inf produced along the way) don't
    render/serialize cleanly. Replace all of them with None."""
    if df.empty:
        return df
    df = df.astype(object).where(pd.notnull(df), None)
    df = df.map(lambda v: None if isinstance(v, float) and not np.isfinite(v) else v)
    return df


def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    columns = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=columns)
    return _sanitize_for_json(df)