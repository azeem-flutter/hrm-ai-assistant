"""One-time/occasional Vanna knowledge sync for Oracle 11g.

Usage after configuring VANNA_MODEL/VANNA_API_KEY:
    python tools/vanna_sync.py

It uploads schema DDL and the curated question/SQL examples. Run it again
when the Oracle schema changes. This is deliberately NOT run on every app
startup, because 30,000-table schemas should be synchronized as a batch job.
"""
import json
import re
import sys
from pathlib import Path

# Standalone script (run as `python tools/vanna_sync.py` from anywhere) --
# put the project root on sys.path so the package imports below resolve,
# since only app.py gets that done automatically by `streamlit run`.
_BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE_DIR))

from config import config
from integrations import db


def get_vanna():
    if not config.VANNA_ENABLED:
        raise RuntimeError("Set VANNA_ENABLED=true first.")
    from vanna.vannadb import VannaDB_VectorStore
    from vanna.ollama import Ollama

    class MyVanna(VannaDB_VectorStore, Ollama):
        def __init__(self):
            VannaDB_VectorStore.__init__(self, vanna_model=config.VANNA_MODEL, vanna_api_key=config.VANNA_API_KEY, config={})
            Ollama.__init__(self, config={"model": config.OLLAMA_MODEL})

    return MyVanna()


def build_ddl():
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name, data_type, data_length, data_precision, data_scale, nullable
        FROM all_tab_columns
        WHERE owner = :owner
        ORDER BY table_name, column_id
    """, owner=config.SCHEMA_OWNER)
    rows = cur.fetchall()
    cur.close()
    tables = {}
    for table, column, dtype, length, precision, scale, nullable in rows:
        if dtype in {"VARCHAR2", "CHAR", "NCHAR", "NVARCHAR2"}:
            typ = f"{dtype}({int(length)})"
        elif dtype == "NUMBER" and precision:
            typ = f"NUMBER({int(precision)},{int(scale or 0)})"
        else:
            typ = dtype
        tables.setdefault(table, []).append(f"{column} {typ}{' NOT NULL' if nullable == 'N' else ''}")
    return [f"CREATE TABLE {name} ({', '.join(cols)})" for name, cols in tables.items()]


def main():
    if not config.VANNA_MODEL or not config.VANNA_API_KEY:
        raise RuntimeError("VANNA_MODEL and VANNA_API_KEY are required.")
    vn = get_vanna()
    ddls = build_ddl()
    print(f"Syncing {len(ddls)} Oracle table definitions to Vanna...")
    for i, ddl in enumerate(ddls, 1):
        vn.train(ddl=ddl)
        if i % 100 == 0:
            print(f"  {i}/{len(ddls)} tables synced")

    examples_path = Path(__file__).parent.parent / "knowledge" / "examples.json"
    if examples_path.exists():
        examples = json.loads(examples_path.read_text(encoding="utf-8"))
        for ex in examples:
            if ex.get("question") and ex.get("sql"):
                vn.train(question=ex["question"], sql=ex["sql"])
        print(f"Synced {len(examples)} curated question/SQL examples.")

    print("Vanna sync complete.")


if __name__ == "__main__":
    main()
