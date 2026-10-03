"""
build_chat_finetune_dataset.py
================================
THIS SCRIPT RUNS IN YOUR PROJECT FOLDER (where rag.py, llm.py, db.py,
config.py live) -- NOT in Colab, and NOT in the my_finetune_project
folder. It needs your live app's own code and (for one call) your
database connection, because it builds training examples using the
EXACT SAME system-prompt-building functions your Streamlit app uses at
runtime -- not a simplified version.

Why this matters (recap of what we diagnosed): your app sends the model
a big {"role": "system", ...} message built by llm.build_system_prompt(),
plus RAG-retrieved schema chunks and few-shot examples -- not the plain
"### Instruction:/### Input:/### Response:" text block the previous
version of this pipeline trained on. Training must match this exactly,
or the fine-tuning has no effect at inference time.

Usage (from the project root, same venv the Streamlit app uses):
    python tools/build_chat_finetune_dataset.py

Reads:
    training_data/training_examples_expanded.json
Writes:
    training_data/chat_finetune_dataset.jsonl   (upload THIS to Colab instead)
"""

import json
import sys
from pathlib import Path

# Standalone script (run as `python tools/build_chat_finetune_dataset.py`).
_BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE_DIR))

from integrations import db, llm, rag
from config import config

_TRAINING_DATA_DIR = _BASE_DIR / "training_data"
_KNOWLEDGE_DIR = _BASE_DIR / "knowledge"
EXPANDED_FILE = _TRAINING_DATA_DIR / "training_examples_expanded.json"
OUTPUT_FILE = _TRAINING_DATA_DIR / "chat_finetune_dataset.jsonl"


def build_messages_for_example(question: str, sql: str, table_chunks: dict, all_table_names: list) -> dict:
    """Mirrors query_service.py's real prompt-building path (the
    non-ranking branch) as closely as possible, using the SAME
    rag.retrieve_tables() call the app makes at runtime -- so training
    sees the same kind of retrieval noise/selection the live app
    produces, not a hand-picked "perfect" schema slice.
    """
    query_vec = rag.embed_query(question)

    retrieved_chunks = rag.retrieve_tables(
        question, table_chunks, query_vec=query_vec
    )

    examples_path = _KNOWLEDGE_DIR / "examples.json"
    try:
        with open(examples_path, "r", encoding="utf-8") as f:
            all_examples = json.load(f)
    except FileNotFoundError:
        all_examples = []
    retrieved_examples = (
        rag.retrieve_examples(question, all_examples, query_vec=query_vec)
        if all_examples else []
    )

    retrieved_schema_text = (
        "\n\n".join(retrieved_chunks.values())
        if retrieved_chunks
        else db.build_schema_text()
    )
    other_table_names = sorted(set(table_chunks.keys()) - set(retrieved_chunks.keys()))

    system_content = llm.build_system_prompt(
        retrieved_schema_text,
        all_table_names,
        retrieved_examples,
        other_table_names=other_table_names,
    )

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
            {"role": "assistant", "content": sql},
        ]
    }


def main():
    with open(EXPANDED_FILE, "r", encoding="utf-8") as f:
        examples = json.load(f)

    table_chunks = db.build_table_chunks()
    all_table_names = sorted(table_chunks.keys())

    print(f"Building chat-format training data for {len(examples)} examples "
          f"using the live app's own prompt-building logic...")

    written = 0
    skipped_too_long = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, ex in enumerate(examples):
            record = build_messages_for_example(
                ex["question"], ex["sql"], table_chunks, all_table_names
            )

            # Rough length sanity-check (character count, not exact tokens,
            # but close enough to flag examples that are clearly too big
            # before we get to Colab -- see max_seq_length note below).
            total_chars = sum(len(m["content"]) for m in record["messages"])
            if total_chars > 12000:  # ~ roughly 3000+ tokens, generous margin
                skipped_too_long += 1
                print(f"  [warn] skipping example {i} -- prompt too long "
                      f"({total_chars} chars): {ex['question'][:50]}...")
                continue

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(examples)} done")

    print(f"\nWrote {written} records to {OUTPUT_FILE} "
          f"({skipped_too_long} skipped for being too long).")
    print("Upload THIS file to Colab -- it replaces finetune_dataset.jsonl "
          "from the previous version of the guide.")


if __name__ == "__main__":
    main()
