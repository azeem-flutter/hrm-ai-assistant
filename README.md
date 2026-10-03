# HRM AI Assistant — Natural Language to SQL

![tests](https://github.com/azeem-flutter/hrm-ai-assistant/actions/workflows/tests.yml/badge.svg)

Ask HR questions in **English or Roman Urdu** and get answers straight from an
**Oracle** database — no SQL knowledge required. Built with Python, Streamlit,
RAG, LangGraph and an LLM that runs either **locally (Ollama)** or in the
**cloud (Groq)**.

> Example: `sabse zyada salary kis employee ki hai?` → generates, validates and runs
> a safe `SELECT`, then shows the result as a table.

## Highlights

- **RAG schema retrieval** – only the most relevant tables are sent to the LLM, not the whole schema.
  Default: hybrid retrieval (Vanna if configured, otherwise a keyword retriever).
  Optional embedding mode (`RAG_PROVIDER=ollama|local`) caches per-table embeddings and ranks by cosine similarity.
- **Multi-layer SQL safety** (`core/sql_safety.py`, `core/query_grounding.py`):
  read-only `SELECT`/`WITH` allow-list, blocks stacked statements and comments,
  detects hallucinated tables/columns, and verifies every `JOIN` against real foreign keys.
- **Claim verification** – a statement like *"IT has the highest salary"* is checked
  against the data instead of being turned into a `WHERE` filter.
- **Deterministic ranking intent** (`core/ranking_intent.py`) – "top 5", "sabse zyada",
  "bottom 3", ordinals and ranges are parsed with rules, independent of the LLM.
- **Multi-turn follow-ups** (`core/conversation_context.py`) – resolves
  "and their departments?" or asks a clarifying question.
- **LangGraph self-repair loop** – if Oracle returns an error, the query is
  repaired with fresh schema context (up to `MAX_SQL_REPAIR_RETRIES`).
- **Switchable providers** – `LLM_PROVIDER=ollama|groq` and `EMBEDDING_PROVIDER=ollama|local`.
- **Accuracy harness** (`tools/evaluate_accuracy.py`) – runs 398 human-verified
  question/SQL pairs and compares the *rows returned*, not just whether SQL executed.

## Architecture

```
User question
  -> core/conversation_context.py   resolve follow-ups / CLARIFY
  -> core/query_service.py          topic check, ranking intent, claim + JOIN grounding
  -> integrations/rag.py            retrieve relevant tables + examples
  -> integrations/db.py             verified foreign-key relationships
  -> integrations/llm.py            SQL generation (Ollama or Groq)
  -> core/sql_safety.py             clean, validate, enforce read-only
  -> integrations/db.py             execute on Oracle
  -> on error: core/agent_graph.py  repair node -> execute again (max N retries)
  -> ui/                            table + SQL shown in Streamlit
```

## Tech stack

Python · Streamlit · Oracle (`oracledb`) · LangGraph / LangChain Core · Ollama · Groq ·
Vanna (optional) · NumPy / pandas · pytest · GitHub Actions

## Project structure

```
app.py                      Streamlit entry point (the only file you run)
config/config.py            all environment-driven settings
core/                       orchestration + SQL safety
  agent_graph.py              LangGraph: generate -> execute -> repair
  query_service.py            question -> SQL
  ranking_intent.py           top/bottom/ordinal detection (EN + Roman Urdu)
  query_grounding.py          claim verification + FK/JOIN validation
  conversation_context.py     multi-turn follow-up resolution
  sql_safety.py               SQL cleaning / validation / read-only guard
integrations/               external systems
  db.py                       Oracle connection, schema + FK introspection
  llm.py                      prompts + Ollama/Groq calls
  rag.py                      retrieval (Vanna / keyword / embeddings)
ui/                         CSS + render helpers
knowledge/                  data read by the live app (glossary, curated examples)
training_data/              eval set + fine-tuning source (not read by the app)
tools/                      manual scripts (accuracy eval, schema inspect, Vanna sync)
tests/                      pytest unit tests
requirements.txt            single dependency file (core + pytest + optional extras)
docs/                       design notes and sample test questions
```

## Setup

> **Requirement:** a reachable Oracle database. The assistant introspects its
> live schema and generates Oracle 11g-compatible SQL (`ROWNUM`, no `LIMIT`).

```bash
git clone https://github.com/azeem-flutter/hrm-ai-assistant.git
cd hrm-ai-assistant
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set at least `DB_USER`, `DB_PASSWORD`, `DB_DSN`.
**Use a least-privilege, read-only Oracle user** (see "Security").

**Option A — fully local (Ollama):**
```bash
ollama pull qwen2.5-coder:3b
# .env: LLM_PROVIDER=ollama, OLLAMA_MODEL=qwen2.5-coder:3b
```

**Option B — cloud (Groq, no local model):**
```bash
# .env: LLM_PROVIDER=groq and GROQ_API_KEY=gsk_...
```

Run:
```bash
streamlit run app.py
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DB_USER` / `DB_PASSWORD` / `DB_DSN` | — (required) | Oracle credentials |
| `DB_LIB_DIR` | – | Oracle Instant Client path (thick mode) |
| `LLM_PROVIDER` | `ollama` | `ollama` or `groq` |
| `OLLAMA_MODEL` | – | e.g. `qwen2.5-coder:3b` |
| `GROQ_API_KEY` / `GROQ_MODEL` | – | needed for `groq` |
| `RAG_PROVIDER` | `hybrid` | `hybrid`, `keyword`, `vanna`, `ollama`, `local` |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` or `local` (embedding RAG modes only) |
| `MAX_SQL_REPAIR_RETRIES` | `2` | LangGraph repair attempts |
| `ROW_LIMIT` | `200` | default row cap |
| `ENABLE_CLAIM_VERIFICATION` | `true` | claim-verification gate |

See `.env.example` for the full list.

## Testing

```bash
pytest -q
```

The unit tests cover the SQL safety guard, ranking-intent parsing (including Roman
Urdu), claim detection, retrieval and conversation-context resolution. They need
**no database and no LLM**. GitHub Actions runs them on every push.

To measure end-to-end accuracy against a live Oracle DB and model:
```bash
python tools/evaluate_accuracy.py --sample 60 --seed 7
```
It reports `execution_rate`, `result_match_rate` (rows match the gold SQL),
`repair_rate` and `avg_latency_sec`, overall and by difficulty.

## Security notes

- Generated SQL is only executed if `is_safe_select()` passes: a single read-only
  `SELECT`/`WITH`, no `;`, no comments, no DML/DDL, no PL/SQL package calls.
- `integrations/db.py::verify_read_only_user()` warns if the DB account has
  write/DDL privileges. The app-layer check is **not** a substitute for a
  real read-only database user — create one:
  ```sql
  CREATE USER hr_readonly IDENTIFIED BY "<password>";
  GRANT CREATE SESSION TO hr_readonly;
  GRANT SELECT ON hr.employees TO hr_readonly;  -- repeat per table
  ```
- Never commit `.env`. It is in `.gitignore`.
- With `LLM_PROVIDER=groq` or Vanna enabled, questions and schema text leave your
  machine. Use Ollama for fully local operation.

## Limitations

- Requires an Oracle database (the SQL rules are Oracle 11g specific).
- Small local models (3B) can still produce semantically wrong SQL that runs
  successfully; use the accuracy harness to compare models.
- A single shared DB connection is used; swap in a pool for heavy concurrency.

## License

MIT — see [LICENSE](LICENSE).
