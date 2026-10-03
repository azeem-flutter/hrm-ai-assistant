"""
config.py
---------
Central place for every environment-driven setting. No logic lives here —
just constants that the other modules (db.py, llm.py, app.py) import.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

#--- handle error ---#
def _require_env(name: str) ->str:
    value = os.getenv(name)
    if not value:
          sys.exit(
            f"[config] Missing required environment variable: {name}\n"
            f"          Set it in your .env file before starting the app."
        )
    return value

# --- Oracle connection -------------------------------------------------
ORACLE_USER = _require_env("DB_USER")
ORACLE_PASSWORD = _require_env("DB_PASSWORD")
ORACLE_DSN = _require_env("DB_DSN")
ORACLE_LIB_DIR = os.getenv("DB_LIB_DIR") or None
SCHEMA_OWNER = ORACLE_USER.upper()

# --- Local Ollama model --------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Chat-model provider switch -------------------------------------------
# "ollama" (default): behavior is 100% unchanged from before -- calls your
#   local Ollama server exactly as always.
# "groq": routes the SAME chat call through the Groq API instead, for
#   testing on a laptop that doesn't have the local Ollama server (e.g. it
#   only runs on a company machine). Requires GROQ_API_KEY in .env.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # only required when LLM_PROVIDER="groq"
# Default picked for comparable reasoning to the local qwen2.5-coder:3b —
# Qwen's current model on Groq's free tier. Override via .env if you'd
# rather try openai/gpt-oss-20b or another Groq-hosted model.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Embedding provider switch --------------------------------------------
# "ollama" (default): unchanged, uses your local Ollama nomic-embed-text.
# "local": uses the sentence-transformers package instead (runs in-process,
#   no server needed at all) -- useful on the same laptop-without-Ollama
#   setup as LLM_PROVIDER="groq" above, since Groq itself doesn't offer an
#   embeddings endpoint. Requires `pip install sentence-transformers`.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

# Ollama defaults to a 2048-token context window regardless of what the
# model itself supports (qwen2.5-coder:3b supports up to 32768) — if the
# retrieved schema + examples + history push the prompt past this, Ollama
# silently truncates it, and the model starts "hallucinating" columns that
# were actually just cut off. Set explicitly rather than relying on the
# Ollama default.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "10000"))
# Output token cap. 300 was tight for longer JOIN/CTE queries and risked
# truncating valid SQL mid-statement; SQL keywords/identifiers are
# token-heavy, so give it more headroom.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "450"))
# Fixed seed for reproducible generations during debugging (temperature is
# already 0, so this mostly matters for tie-breaking and test stability).
OLLAMA_SEED = int(os.getenv("OLLAMA_SEED", "42"))
# Rough chars-per-token estimate used only for the prompt-size warning in
# query_service.py — not exact, just enough to flag "getting close to num_ctx".
APPROX_CHARS_PER_TOKEN = 4

# --- Retrieval (RAG) -------------------------------------------------------
# How many of the most-relevant table chunks / example (question, sql)
# pairs get pulled into the system prompt for each question, instead of
# dumping the entire schema + every example every time. Raise these if
# the model seems to be missing a table/example it needed; lower them to
# keep the prompt small and fast. See step 7 in the RAG rollout notes.
RAG_TOP_K_TABLES = int(os.getenv("RAG_TOP_K_TABLES", "6"))
RAG_TOP_K_EXAMPLES = int(os.getenv("RAG_TOP_K_EXAMPLES", "6"))

# RAG provider: "keyword" is zero-extra-model and is the safe default for
# testing. "vanna" uses Vanna hosted memory when configured. "hybrid" tries
# Vanna first and falls back to keyword retrieval. The old local embedding
# path remains available as "ollama" for rollback.
RAG_PROVIDER = os.getenv("RAG_PROVIDER", "hybrid").strip().lower()
VANNA_MODEL = os.getenv("VANNA_MODEL", "").strip()
VANNA_API_KEY = os.getenv("VANNA_API_KEY", "").strip()
VANNA_ENABLED = os.getenv("VANNA_ENABLED", "false").lower() == "true"
VANNA_AUTO_LEARN = os.getenv("VANNA_AUTO_LEARN", "false").lower() == "true"
MAX_SQL_REPAIR_RETRIES = int(os.getenv("MAX_SQL_REPAIR_RETRIES", "2"))

# --- Query behaviour -----------------------------------------------------
# Claim-verification gate: when enabled, factual/superlative claims are
# independently checked against the database instead of treating the user
# asserted value as a WHERE filter. Disable only for debugging/rollback.
ENABLE_CLAIM_VERIFICATION = os.getenv("ENABLE_CLAIM_VERIFICATION", "true").lower() == "true"
ENABLE_QUERY_PREFLIGHT = os.getenv("ENABLE_QUERY_PREFLIGHT", "true").lower() == "true"
# LLM-based topic classification (greeting / off-topic / real DB question)
# that runs before SQL generation -- see query_service._classify_topic().
# Disable for instant rollback to the previous behavior, where only an
# EXACT all-filler greeting was caught deterministically and every other
# non-DB question (e.g. "what is the capital of France") fell through to
# the SQL-generation LLM call.
ENABLE_TOPIC_CLASSIFICATION = os.getenv("ENABLE_TOPIC_CLASSIFICATION", "true").lower() == "true"
# Multi-turn context resolver. Disable for instant rollback to the previous behavior.
ENABLE_CONVERSATION_CONTEXT = os.getenv("ENABLE_CONVERSATION_CONTEXT", "true").lower() == "true"
CONVERSATION_CONTEXT_TURNS = int(os.getenv("CONVERSATION_CONTEXT_TURNS", "5"))
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "200"))
SCHEMA_CACHE_TTL = 3600     # seconds — how long schema introspection is cached
MAX_HISTORY_TURNS = 3       # how many past Q/A pairs are fed back to the model

# --- App branding (used by app.py / styles.py) ---------------------------
APP_TITLE = "HRM AI Assistant"
APP_SUBTITLE = "Ask questions about your HR data in natural language"
APP_ICON = "🗂️"