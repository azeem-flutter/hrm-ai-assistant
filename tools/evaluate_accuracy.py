"""
evaluate_accuracy.py
=====================
Answers the actual question this project was built for: "how correctly does
the model (LangGraph + RAG + Qwen) understand a question and bring back the
RIGHT answer from Oracle" -- not just "did some SQL run without an error".

WHY THIS SCRIPT EXISTS
-----------------------
core/agent_graph.py's repair loop only re-tries when Oracle THROWS an error
(ORA-xxxxx). If the model writes SQL that is syntactically fine, executes
successfully, but is semantically wrong (wrong JOIN, wrong filter, wrong
column) -- LangGraph will never see that as a failure and will never repair
it. So "it ran" and "it's correct" are two different things, and only this
script checks the second one, by comparing against a human-verified gold
SQL for the same question.

METRICS REPORTED
-----------------
- execution_rate   : % of questions where SOME SQL executed successfully
                      (this is the metric LangGraph's retry loop already
                      optimizes for)
- result_match_rate: % of questions where the returned ROWS actually match
                      the gold SQL's rows (this is the metric that matters
                      for "does the model understand the question")
- repair_rate      : % of questions that needed >=1 LangGraph repair retry
- avg_latency_sec  : average end-to-end time per question
- breakdown by `difficulty` and by `tables_used`, so you can see exactly
  which kind of question a given model (3B vs 7B, etc.) struggles with.

USAGE
-----
    python tools/evaluate_accuracy.py
    python tools/evaluate_accuracy.py --dataset knowledge/examples.json
    python tools/evaluate_accuracy.py --limit 40          # quick smoke test
    python tools/evaluate_accuracy.py --sample 60 --seed 7 # random subset

Needs a live Oracle connection and a running Ollama server (same .env as
app.py) -- it drives the exact same agent_graph.run() the Streamlit app
calls, so the numbers reflect real end-to-end behaviour, including RAG
retrieval and the LangGraph repair loop.

Writes a CSV (default: training_data/accuracy_report_<model>_<timestamp>.csv)
with one row per question, plus a summary printed to the console.
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE_DIR))

import json

from config import config
from integrations import db
from core import agent_graph


def _normalize_result(df) -> Counter:
    """Row/column-order-tolerant fingerprint of a result set.

    Each row becomes a sorted tuple of its stringified values (so column
    order doesn't matter), and the whole result becomes a multiset of rows
    (so row order doesn't matter, but duplicate rows still count). This is
    intentionally forgiving -- it is NOT an exact-string SQL comparison,
    it's asking "did the two queries return the same information".
    """
    if df is None or df.empty:
        return Counter()
    rows = []
    for _, row in df.iterrows():
        rows.append(tuple(sorted(str(v).strip() for v in row.tolist())))
    return Counter(rows)


def _run_gold(sql: str):
    try:
        return db.run_query(sql), None
    except Exception as exc:  # gold SQL itself may be stale vs current schema
        return None, str(exc)


def load_dataset(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Accept both the plain [{"question","sql"}] shape (knowledge/examples.json)
    # and the richer shape with difficulty/tables_used (training_data/*.json)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(_BASE_DIR / "training_data" / "training_examples_expanded.json"),
        help="Path to a JSON file of {question, sql[, difficulty, tables_used]} records.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples.")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N examples instead of running all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="CSV path. Default: training_data/accuracy_report_<model>_<ts>.csv")
    args = parser.parse_args()

    examples = load_dataset(Path(args.dataset))
    if args.sample:
        random.seed(args.seed)
        examples = random.sample(examples, min(args.sample, len(examples)))
    if args.limit:
        examples = examples[: args.limit]

    model_name = config.OLLAMA_MODEL or "UNSET_OLLAMA_MODEL"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else (
        _BASE_DIR / "training_data" / f"accuracy_report_{model_name.replace(':', '-')}_{ts}.csv"
    )

    print(f"Model under test : {model_name}")
    print(f"RAG provider     : {config.RAG_PROVIDER}")
    print(f"Repair retries   : {config.MAX_SQL_REPAIR_RETRIES}")
    print(f"Examples to run  : {len(examples)}")
    print("-" * 70)

    rows = []
    difficulty_stats = defaultdict(lambda: {"total": 0, "matched": 0})

    for i, ex in enumerate(examples, 1):
        question = ex["question"]
        gold_sql = ex["sql"]
        difficulty = ex.get("difficulty", "unspecified")

        gold_df, gold_error = _run_gold(gold_sql)

        t0 = time.time()
        try:
            graph_result = agent_graph.run(question, history=[])
        except Exception as exc:
            graph_result = {"status": "crashed", "error": str(exc), "sql": "", "attempt": 0}
        elapsed = time.time() - t0

        status = graph_result.get("status")
        executed = status == "success"
        attempts = int(graph_result.get("attempt", 0))
        generated_sql = graph_result.get("sql", "")

        result_match = None
        if gold_df is not None and executed:
            result_match = _normalize_result(graph_result.get("result")) == _normalize_result(gold_df)

        difficulty_stats[difficulty]["total"] += 1
        if result_match:
            difficulty_stats[difficulty]["matched"] += 1

        rows.append({
            "question": question,
            "difficulty": difficulty,
            "tables_used": ",".join(ex.get("tables_used", [])),
            "status": status,
            "executed": executed,
            "result_match": result_match,
            "repair_attempts": attempts,
            "latency_sec": round(elapsed, 2),
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "gold_sql_error": gold_error or "",
            "error": graph_result.get("error", ""),
        })

        mark = "MATCH" if result_match else ("RAN(no gold)" if executed else "FAIL")
        print(f"[{i}/{len(examples)}] {mark:14s} attempts={attempts} {elapsed:5.1f}s  {question[:60]}")

    # --- write CSV -----------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- summary ---------------------------------------------------------
    total = len(rows)
    executed_n = sum(1 for r in rows if r["executed"])
    matched_n = sum(1 for r in rows if r["result_match"])
    repaired_n = sum(1 for r in rows if r["repair_attempts"] > 0)
    avg_latency = sum(r["latency_sec"] for r in rows) / total if total else 0.0

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}   |   dataset: {Path(args.dataset).name}   |   n={total}")
    print("=" * 70)
    print(f"execution_rate    : {executed_n}/{total}  ({executed_n / total:.1%})")
    print(f"result_match_rate : {matched_n}/{total}  ({matched_n / total:.1%})   <- the real accuracy number")
    print(f"repair_rate       : {repaired_n}/{total}  ({repaired_n / total:.1%})  (needed >=1 LangGraph retry)")
    print(f"avg_latency_sec   : {avg_latency:.2f}s")
    print("-" * 70)
    print(f"{'difficulty':<15}{'matched/total':<15}{'accuracy':<10}")
    for level, stats in sorted(difficulty_stats.items()):
        acc = stats["matched"] / stats["total"] if stats["total"] else 0.0
        print(f"{level:<15}{stats['matched']}/{stats['total']:<13}{acc:.1%}")
    print("-" * 70)
    print(f"Full per-question report written to: {out_path}")


if __name__ == "__main__":
    main()
