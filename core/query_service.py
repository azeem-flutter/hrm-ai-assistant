"""
query_service.py
-----------------
The orchestration layer: takes a plain-English question, asks the LLM for
SQL, cleans/validates it, and gives the model up to one self-correction
retry if it hallucinated columns or produced an invalid UNION shape.
This is the only module that ties llm.py, sql_safety.py and db.py
together — app.py never talks to those three directly.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from config import config
from integrations import db, llm, rag
from core import sql_safety, ranking_intent, query_grounding

_KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"  # core/ -> project root -> knowledge/
_EXAMPLES_PATH = _KNOWLEDGE_DIR / "examples.json"
_BUSINESS_TERMS_PATH = _KNOWLEDGE_DIR / "business_terms.json"
_PENDING_TERMS_PATH = _KNOWLEDGE_DIR / "pending_terms.json"
_PENDING_TERMS_MAX = 200  # cap the log so it can't grow unbounded

# Self-referential pronouns (English + common Hinglish) that only make sense
# if we know WHICH employee "I"/"my" refers to. Deliberately excludes
# ambiguous tokens like bare "main"/"mai" (also common non-pronoun words,
# e.g. "main office", "main branch") to avoid false positives.
#
# Split into two tiers because "mujhe" is NOT unambiguously possessive the
# way "mera"/"meri"/"apna" are:
#   - "meri salary btao"      -> unambiguous: MY salary        (STRONG)
#   - "mujhe employees do"    -> just "give me [the] employees" -- a
#                                generic request, nothing to do with the
#                                asker's own identity                (NOT self-ref)
#   - "mujhe meri salary do"  -> STRONG token ("meri") also present
#   - "mujhe salary batao"    -> "mujhe" + a personal-data noun, commonly
#                                does mean "tell me MY salary" in speech (WEAK,
#                                only counts combined with a personal-data noun)
# Treating "mujhe" the same as "mera" caused a false trigger on *any*
# question containing "mujhe", regardless of topic (e.g. "mujhe employees
# do" wrongly asked for a name/employee ID even though it's a plain
# "show me the employees table" request).
_STRONG_SELF_REFERENCE_TOKENS = (
    "my", "mine", "myself", "mera", "meri", "mere", "apna", "apni",
)
_WEAK_SELF_REFERENCE_TOKENS = ("mujhe",)
_PERSONAL_DATA_NOUNS = (
    "salary", "record", "profile", "id", "leave", "attendance", "bonus",
    "account", "email", "contact", "information", "info", "details", "detail",
)

# Signals that an identifier (name or employee ID) is already present
# somewhere, so we shouldn't ask again.
_IDENTITY_HINT_RE = re.compile(
    r"\b(employee\s*id|emp\s*id|id\s*[:#]?\s*\d+|\d{2,})\b"
    r"|\b(naam|name\s+is|mera\s+naam|i\s*am|called)\b",
    re.IGNORECASE,
)

# Destructive/write-intent verbs. Checked BEFORE any deterministic
# template or LLM call — the "sab/all employees dikhao" template below
# matches on the bare word "all", which also appears in "delete ALL
# employees in france". Without this check running first, that phrasing
# matched the show-all-employees template and silently ran
# "SELECT * FROM employees ..." — ignoring both the delete intent AND the
# "in france" filter, with no read-only warning ever shown, because the
# deterministic template short-circuits BEFORE the LLM (which does have a
# read-only refusal in its own instructions/examples) ever sees the
# question. This must be checked first, independent of the LLM, so a
# write intent can never slip through disguised inside an "all ..."
# phrasing.
_WRITE_INTENT_TOKENS = (
    "delete", "remove", "drop", "update", "insert", "truncate", "alter",
    "grant", "revoke", "merge",
)

# Vague "give me records/data" phrasing that doesn't name any real table.
# The LLM has a CLARIFY rule for this in its prompt, but a small local
# model sometimes reaches for its generic last-resort "could not
# understand" fallback instead of the more specific one — this catches
# the common phrasing deterministically so the user always gets asked
# which table, instead of a dead-end non-answer.
_VAGUE_RECORDS_TOKENS = (
    "record", "records", "data", "everything", "info", "information",
)

# Meta-questions about the schema itself, not about any row of data —
# "what's the schema" isn't a SQL question at all, so left to the LLM it
# tends to fall into the same generic "could not understand" fallback as
# vague record requests. Answered directly from the real table list.
#
# Deliberately PHRASE-level patterns, not bare words. A bare _SCHEMA_META_
# TOKENS = ("schema", "structure") match anywhere in the text used to
# false-positive badly: a user casually mentioning "my last question was
# related to HR schema or sql regarding who is the CEO" (schema used as
# background context, not a request to list anything) matched on the word
# alone and returned a full 12-table listing instead of the real answer —
# especially likely to fire on a CLARIFY-merge block, which quotes the
# user's own earlier wording verbatim (see _CLARIFY_MERGE_RE below).
# Requiring an actual "show/what/explain ... schema/structure" shape
# keeps the deterministic table-listing shortcut for genuine meta
# questions while no longer matching the word in passing.
_SCHEMA_META_PATTERNS = (
    r"\b(what|which|show|list|explain|describe|give)\b.{0,20}\b(schema|structure)\b",
    r"\b(schema|structure)\b.{0,20}\b(look like|of (the|this) database|of (the|this) table)\b",
    r"\bdatabase\s+schema\b",
)

# Plain greetings / thanks / farewells / "how are you"-style small talk —
# not a database question at all. Left to the LLM, a small local model
# frequently answers these conversationally ("Hello! How can I assist you
# today?") instead of refusing or producing SQL. Every later step in the
# pipeline (sql_safety.ensure_default_row_cap, enforce_correct_row_limit's
# ROWNUM-wrap fallback, ...) assumes whatever it's handed IS a real SQL
# statement and is free to wrap it as "SELECT * FROM (<this>) WHERE ROWNUM
# <= N" — which turns that plain sentence into something that starts with
# SELECT (so it slips past is_safe_select()'s allow-list check) but isn't
# valid SQL, and only fails once it actually reaches Oracle as an
# ORA-00907 "missing right parenthesis". sql_safety.clean_sql() now has a
# defense-in-depth catch for that (see _looks_like_sql_statement there),
# but the much better fix for the common case is to never call the LLM
# for a plain "hello" at all — same "answer deterministically before any
# LLM call" philosophy as _is_write_request above, and it also means the
# user gets an on-topic, helpful reply instead of a generic AI greeting.
#
# Deliberately conservative — same "every remaining word must be filler"
# approach as _is_plain_all_employees_request above — so "hi, show me all
# employees" still falls through to the real pipeline instead of being
# swallowed as small talk.
_SMALLTALK_FILLER_WORDS = {
    "hi", "hii", "hiii", "hello", "hey", "heya", "yo", "hola", "hy",
    "salam", "assalam", "assalamualaikum", "asalam", "walaikum", "alaikum",
    "namaste", "good", "morning", "afternoon", "evening", "night",
    "thanks", "thank", "thankyou", "shukriya", "mehrbani", "mehrbaani",
    "bye", "goodbye", "khuda", "hafiz", "allah", "ok", "okay", "cool",
    "you", "u", "there", "o",
}


def _is_write_request(question: str) -> bool:
    q = _normalise_question(question)
    return any(re.search(rf"\b{re.escape(tok)}\b", q) for tok in _WRITE_INTENT_TOKENS)


def _is_vague_records_request(question: str, table_chunks: dict) -> bool:
    q = _normalise_question(question)
    mentions_vague = any(
        re.search(rf"\b{re.escape(tok)}\b", q) for tok in _VAGUE_RECORDS_TOKENS
    )
    if not mentions_vague:
        return False
    # Not vague if a real table is actually named (singular or plural),
    # e.g. "employee records" should NOT trigger this — that's specific.
    for table_name in table_chunks:
        name_lower = table_name.lower()
        if name_lower in q or name_lower.rstrip("s") in q:
            return False
    return True


def _is_schema_meta_question(question: str) -> bool:
    q = _normalise_question(question)
    if any(re.search(pattern, q) for pattern in _SCHEMA_META_PATTERNS):
        return True
    # "what/which/list ... tables" — but not "employees table" (that's a
    # specific-table question, not a meta one).
    return bool(
        re.search(r"\btables?\b", q)
        and re.search(r"\b(what|which|list|available)\b", q)
    )


# app.py/conversation_context.py merge a pending clarification into ONE
# block of text via conversation_context._build_clarification_answer_question():
#
#   Original request: <the user's original, ambiguous question>
#   Clarification question that was asked: <OUR OWN previous CLARIFY message>
#   User's answer to that clarification: <what the user actually just typed>
#   Treat this as ONE fully-specified request: resolve the original request
#   using the user's answer to fill in what was missing. Do not ask about
#   the same ambiguity again.
#
# Every deterministic keyword-matching helper in this module (_is_write_
# request, _is_vague_records_request, _is_schema_meta_question,
# _needs_identity_clarification, _deterministic_sql_from_intent, the
# ranking templates, ...) funnels through _normalise_question() and scans
# the ENTIRE question text for trigger words. Left unhandled, that means
# they scan our own boilerplate too -- and a "which table ...?" or
# "which field/metric ...?" clarification we ourselves asked is
# *guaranteed* to contain words like "table"/"which"/"what" that match
# some OTHER heuristic, plus the fixed trailing instruction sentence adds
# more of the same ("...fill in WHAT was missing"). Concretely, this was
# observed live: answering our own "Which table would you like to see
# records from?" with "employees" produced a merged block containing
# both "table" and "which" (from OUR question) -- which matched
# _is_schema_meta_question() and returned a full table listing instead of
# ever answering the user's real request. A second case answering "Which
# field or metric...?" hit the same match via "table" from the user's
# own reply plus "what" from the fixed instruction sentence.
#
# ranking_intent.classify() already has its own narrower version of this
# same fix (stripping only the middle "clarification question" line) for
# its own keyword extraction. This is the analogous fix for THIS module,
# where the bug was actually reproduced -- but it goes further: it
# extracts ONLY the real content (the original question + the user's
# actual answer), discarding every fixed system-authored label and
# instruction sentence in the template, not just the middle line. That
# is deliberate: as the second observed case shows, the *trailing*
# instruction sentence ("...fill in what was missing") is just as capable
# of supplying a false-positive trigger word as the middle line is.
_CLARIFY_MERGE_RE = re.compile(
    r"^original request:\s*(?P<orig>.*?)\s*"
    r"(?:clarification question that was asked:.*?)?"
    r"user's answer to that clarification:\s*(?P<answer>.*?)\s*"
    r"treat this as one fully-specified request:.*$",
    re.IGNORECASE | re.DOTALL,
)


def _is_greeting_or_smalltalk(question: str) -> bool:
    """True only for a message that is ENTIRELY greeting/thanks/farewell
    chatter, with no actual database question attached. See the comment
    on _SMALLTALK_FILLER_WORDS above for why this check exists and why
    it requires every word to be filler rather than matching on any
    single greeting word appearing anywhere."""
    words = re.findall(r"[a-z]+", _normalise_question(question))
    if not words:
        return False
    return all(w in _SMALLTALK_FILLER_WORDS for w in words)


_SMALLTALK_REPLY = (
    "Hi! I am the HRM AI Assistant. I can only help with questions about "
    "this database. Try asking something like \"show all IT employees\", "
    "\"how many employees are in each department\", or \"top 10 highest "
    "paid employees\"."
)


def _smalltalk_reply_sql() -> str:
    """Same 'literal text as a one-row result' shape already used for the
    read-only write-refusal message below — the app already knows how to
    render this as a normal, harmless SQL result. The apostrophe-escaping
    here matters: unlike the write-refusal message (which happens to have
    no apostrophes), a friendly reply very easily can ("I'm", "don't", ...),
    and an unescaped apostrophe would break the SQL literal itself — the
    exact same class of bug this whole change is fixing for the LLM's own
    replies. See sql_safety._as_safe_message_sql for the same escaping."""
    return f"SELECT '{_SMALLTALK_REPLY.replace(chr(39), chr(39) * 2)}' AS message FROM dual"


_OFF_TOPIC_REPLY = (
     "I'm the AI Assistant and I can only help with questions about "
    "this database. That's not something I can look up here — try "
    "asking about employees, departments, salaries, orders, products, "
    "or line items instead."
)


def _off_topic_reply_sql() -> str:
    """Same 'literal text as a harmless one-row SELECT' shape as
    _smalltalk_reply_sql() above — see that function's comment for why the
    apostrophe-escaping matters."""
    return f"SELECT '{_OFF_TOPIC_REPLY.replace(chr(39), chr(39) * 2)}' AS message FROM dual"


def _log_pending_term_candidate(question: str) -> None:
    """Appends a question the topic classifier just blocked as off_topic
    into knowledge/pending_terms.json -- NOT into business_terms.json.

    This is deliberately a ONE-WAY, human-reviewed log, not a learning
    loop. It never gets read back into any prompt, and nothing in this
    codebase treats it as ground truth. The point is purely visibility:
    over time this file accumulates the real questions users asked that
    got refused, so a human can periodically skim it and decide -- e.g.
    if "who is the CEO" (or some other business-title phrase) keeps
    showing up here, that's the signal to add a proper entry to
    knowledge/business_terms.json (or knowledge/examples.json) once,
    deliberately, after actually checking it's correct.

    Automatically promoting entries here into the live glossary WITHOUT a
    human step would be a real risk: anyone typing an off-topic or
    misleading question could otherwise poison the permanent knowledge
    base with something false (e.g. claiming a wrong column means
    something it doesn't). Keeping this as a read-only-by-humans log
    avoids that entirely.

    Fails silently (never raises, never blocks the response the user is
    about to get) -- logging a candidate is a nice-to-have, not something
    that should ever be able to break the actual answer.
    """
    try:
        normalised = _normalise_question(question)
        if not normalised:
            return
        existing = []
        if _PENDING_TERMS_PATH.exists():
            try:
                with _PENDING_TERMS_PATH.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []
        # Dedup on normalised question text -- bump the existing entry's
        # count/last_seen instead of piling up identical duplicates, so a
        # frequently-repeated blocked question is visibly more "popular"
        # for review, not just first-seen-wins.
        for entry in existing:
            if _normalise_question(entry.get("question", "")) == normalised:
                entry["times_seen"] = entry.get("times_seen", 1) + 1
                entry["last_seen"] = datetime.now().isoformat(timespec="seconds")
                break
        else:
            existing.append({
                "question": question.strip(),
                "first_seen": datetime.now().isoformat(timespec="seconds"),
                "last_seen": datetime.now().isoformat(timespec="seconds"),
                "times_seen": 1,
            })
        # Cap size -- keep the most recently-seen entries if it ever grows
        # past the limit, so this stays a reviewable log, not an
        # ever-growing file.
        existing.sort(key=lambda e: e.get("last_seen", ""), reverse=True)
        existing = existing[:_PENDING_TERMS_MAX]
        with _PENDING_TERMS_PATH.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception as exc:
        print(f"[GLOSSARY] failed to log pending-term candidate (non-fatal): {exc}")


def _classify_topic(question: str, history: Optional[list] = None) -> str:
    """LLM-based classification for anything the cheap deterministic
    checks (_is_write_request, _is_greeting_or_smalltalk, ...) didn't
    already resolve. This is what actually catches a real, well-formed
    but unrelated question -- e.g. "what is the capital of France?" -- as
    distinct from a plain "hello". _is_greeting_or_smalltalk requires
    EVERY word to be filler, so it structurally can't (and shouldn't)
    catch this case; and unlike "hello" or a write-intent verb,
    off-topic questions have no fixed vocabulary to keyword-match on, so
    this needs the model, not another token list.

    Left unhandled, this exact case used to reach the SQL-generation LLM
    call, which (particularly on a small local model) tends to either
    hallucinate a SELECT against a nonexistent table -- surfacing as a
    real Oracle error once it reaches db.py -- or answer conversationally
    in a shape sql_safety.clean_sql() then has to defensively catch.
    Classifying up front means the user gets the same kind of clean,
    on-topic reply as the greeting case, and the SQL pipeline is never
    invoked for a question that was never going to produce valid SQL.

    Fails OPEN to "db_query" on ANY problem — timeout, model not
    reachable, malformed/unexpected JSON, older `ollama` package, etc.
    This call is a pure optimization to short-circuit obviously
    off-topic questions earlier and with a friendlier message; it must
    never be able to BLOCK a legitimate database question just because
    the local model hiccuped. Worst case on failure is simply identical
    to prior behavior — falls through to the normal SQL pipeline, with
    all of its own existing safety nets still intact.
    """
    try:
        glossary_text = _format_business_glossary(_load_business_glossary())
        history_text = _format_recent_history_for_classifier(history)
        prompt = llm.build_topic_classifier_prompt(
            question, glossary_text=glossary_text, history_text=history_text
        )
        result = llm.call_local_model_structured(
            [{"role": "system", "content": prompt}],
            llm.TOPIC_CLASSIFIER_SCHEMA,
        )
        topic = str(result.get("topic", "")).strip().lower()
        if topic in ("db_query", "greeting", "off_topic"):
            return topic
        print(f"[INTENT] topic classifier returned unexpected value {result!r} — defaulting to db_query")
    except Exception as exc:
        print(f"[INTENT] topic classification failed, defaulting to db_query: {exc}")
    return "db_query"


def _normalise_question(question: str) -> str:
    """Lowercases and squeezes whitespace for robust keyword matching.

    If `question` is the merged [original + our clarifying question +
    user's answer] block built by
    conversation_context._build_clarification_answer_question(), this
    first reduces it down to just the real content -- the original
    request plus the user's actual answer -- so keyword heuristics below
    never misfire on our own previous CLARIFY message or the fixed
    instruction wrapper text. See _CLARIFY_MERGE_RE above for why.
    Non-merged questions (the common case) are returned unchanged aside
    from the usual lowercase/whitespace normalisation.
    """
    stripped = question.strip()
    match = _CLARIFY_MERGE_RE.match(stripped)
    if match:
        stripped = f"{match.group('orig')} {match.group('answer')}".strip()
    return " ".join(stripped.lower().split())


def _contains_any(text: str, tokens: tuple) -> bool:
    """Whole-word/whole-phrase containment check -- NOT plain substring.

    A naive `token in text` check is dangerous here: several short Roman
    Urdu/Hindi tokens are also prefixes of unrelated, more common words.
    The concrete bug this fixes: "sab" (meaning "all/every") is a
    substring of "sabse" (meaning "most", as in "sabse zyada" = "the
    most") -- so "sabse zyada salary wala employee kaun hai" was being
    misread as an "all/every" request and short-circuited to a plain
    "SELECT * FROM employees WHERE ROWNUM <= 200" with no ranking at
    all, bypassing the LLM (and the ranking-correction fix) entirely.
    Word-boundary matching keeps "sab" matching only the standalone word
    "sab", not "sabse", "sabko", "sabhi", etc.
    """
    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens)


def _misplaced_column_hint(table: str, column: str, schema_dict: dict, fk_pairs: list) -> Optional[str]:
    """If `column` doesn't exist on `table` but exists on exactly one other
    real table that's a direct FK neighbor of `table`, return a sentence
    pointing at the real join. Returns None if the column doesn't exist
    anywhere else, exists on more than one other table (ambiguous — don't
    guess which one), or isn't reachable via a single verified FK hop.
    """
    owners = [
        t for t, cols in schema_dict.items()
        if t != table and column in cols
    ]
    if len(owners) != 1:
        return None
    other = owners[0]
    for ct, cc, pt, pc in fk_pairs:
        if ct == table and pt == other:
            return (
                f"{column} actually lives on {other}, reachable via "
                f"{table}.{cc} = {other}.{pc} — JOIN to {other} instead of "
                f"referencing {column} on {table} directly."
            )
        if ct == other and pt == table:
            return (
                f"{column} actually lives on {other}, reachable via "
                f"{other}.{cc} = {table}.{pc} — JOIN to {other} instead of "
                f"referencing {column} on {table} directly."
            )
    return None


def _needs_identity_clarification(question: str, history: Optional[list]) -> bool:
    """Deterministic safety net for self-referential questions ("meri salary
    btao", "what's my salary", "mujhe apna record dikhao").

    This app has no logged-in user/session, so "I"/"my"/"mera" can't be
    resolved to a specific employee. Rather than rely only on the LLM
    following the prompt instruction (small local models sometimes miss
    it), we catch the common phrasing here before an SQL call is even made.

    Skips the check if:
    - No self-referential pronoun is present, or
    - The question itself already contains an identifying hint (a name-ish
      phrase or a number, which is usually an ID), or
    - We just asked this exact identity-clarify question last turn — in
      that case the current message is presumed to be the answer (a name),
      and the LLM's own follow-up handling takes over from here.
    """
    q = _normalise_question(question)

    has_strong = any(re.search(rf"\b{re.escape(tok)}\b", q) for tok in _STRONG_SELF_REFERENCE_TOKENS)
    has_weak = any(re.search(rf"\b{re.escape(tok)}\b", q) for tok in _WEAK_SELF_REFERENCE_TOKENS)
    has_personal_noun = any(re.search(rf"\b{re.escape(n)}\b", q) for n in _PERSONAL_DATA_NOUNS)

    # "mujhe" alone (no possessive word, no personal-data noun) is just a
    # generic "give me ___" request — e.g. "mujhe employees do" — not a
    # question about the asker's own identity.
    has_self_ref = has_strong or (has_weak and has_personal_noun)
    if not has_self_ref:
        return False

    if _IDENTITY_HINT_RE.search(q):
        return False

    if history:
        last = history[-1]
        if (
            last.get("role") == "assistant"
            and last.get("content", "").strip().upper().startswith("CLARIFY:")
            and "employee" in last["content"].lower()
        ):
            return False

    return True


# Generic request/glue words that carry NO filter meaning by themselves --
# used only to recognize a truly plain "show all employees (table/data)"
# request with nothing else attached. Any word in the question that is
# NOT in this set is treated as real filter content (a condition, a
# column name, a date, a department, a number, etc.), so the plain
# all-rows template below is skipped and the question goes to the LLM
# instead, which can actually honor that condition.
_ALL_EMPLOYEES_FILLER_TOKENS = {
    "show", "list", "display", "dikhao", "dikha", "de", "do", "batao",
    "mujhe", "please", "pls", "the", "me", "my", "employee", "employees",
    "table", "records", "record", "data", "sab", "all", "sabhi",
    "ka", "ki", "ke", "ko", "hai", "hain", "mein", "se", "please",
}


def _is_plain_all_employees_request(q: str) -> bool:
    """True only if EVERY word in the (already-lowercased) question is
    generic request/glue phrasing -- i.e. there's no actual filter
    condition, column, date, department, or number attached. See the
    comment at the call site for why this guard exists."""
    words = re.findall(r"[a-z0-9]+", q)
    if not words:
        return False
    return all(w in _ALL_EMPLOYEES_FILLER_TOKENS for w in words)


# Generic stopwords ignored when comparing two questions' vocabulary for
# _looks_like_stale_filter_copy below -- request/glue words that would
# trivially "overlap" between almost any two questions and so shouldn't
# count as evidence the questions are actually related.
_STOPWORDS_FOR_OVERLAP = {
    "the", "a", "an", "is", "are", "in", "on", "of", "to", "for", "with",
    "and", "or", "employees", "employee", "show", "list", "display",
    "all", "please", "me", "ka", "ki", "ke", "ko", "hai", "hain", "mein",
    "se", "tak", "dikhao", "dikha", "batao", "kya", "kaun", "kitna",
    "kitne", "who", "what", "which",
}


def _significant_words(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS_FOR_OVERLAP and len(w) > 2}


# Explicit continuation/follow-up signals -- if the question contains one
# of these, it's clearly and intentionally building on the PREVIOUS
# result set (matches the "FOLLOW-UP" guidance also given to the LLM in
# llm.py), so reusing the same filter is exactly correct and
# _looks_like_stale_filter_copy should never flag it, no matter the
# vocabulary overlap.
_CONTINUATION_SIGNAL_TOKENS = (
    "inhi", "inhee", "unhi", "unhee", "isi", "usi", "wahi", "wohi",
    "same", "in results", "these results", "ismein se", "unme se",
    "in mein se", "un mein se", "pichle wale", "previous",
)


def _is_explicit_continuation(question: str) -> bool:
    q = question.lower()
    return any(tok in q for tok in _CONTINUATION_SIGNAL_TOKENS)


def _looks_like_stale_filter_copy(
    question: str, history: Optional[list], sql: str
) -> bool:
    """Heuristic guard against a small local model blindly copying the
    PREVIOUS turn's WHERE clause into an unrelated new question, instead
    of building a fresh filter from the current question. Observed
    failure: after a turn filtering on hire_date, "salary rank 10 se 15
    tak employees dikhao" came back with that SAME hire_date filter
    still attached and no ranking logic at all -- the model pattern-
    matched on the most recent SQL it could see in history rather than
    answering the actual question.

    Flags this ONLY when both signals line up:
      1. The current SQL's WHERE clause is byte-identical (after
         whitespace normalization) to the immediately-preceding
         assistant turn's WHERE clause.
      2. The current and previous user questions share essentially no
         meaningful vocabulary.
    Two truly-related consecutive questions (a real "narrow these
    results further" follow-up) will almost always share at least one
    real content word, so this combination firing by coincidence on a
    genuine follow-up is very unlikely -- and even if it does, the
    fallback is just one extra LLM call generated fresh without
    history, not a hard failure.
    """
    if not history:
        return False
    if _is_explicit_continuation(question):
        return False
    current_where = sql_safety.extract_where_clause(sql)
    if not current_where:
        return False

    prev_question = None
    prev_sql = None
    for entry in reversed(history):
        if entry.get("role") == "assistant" and prev_sql is None:
            prev_sql = entry.get("content", "")
        elif entry.get("role") == "user" and prev_sql is not None and prev_question is None:
            prev_question = entry.get("content", "")
            break
    if not prev_question or not prev_sql:
        return False

    prev_where = sql_safety.extract_where_clause(prev_sql)
    if not prev_where or prev_where != current_where:
        return False

    current_words = _significant_words(question)
    prev_words = _significant_words(prev_question)
    return len(current_words & prev_words) == 0


def _deterministic_sql_from_intent(question: str) -> Optional[str]:
    """Handles a few reasoning-heavy intents with fixed-safe SQL templates.

    This reduces failure modes for small local models on patterns like:
    - salary categories without a real category column
    - salary > average salary (CTE)
    - Hinglish "sab employees dikhao"
    """
    q = _normalise_question(question)

    asks_employees = _contains_any(q, ("employee", "employees"))
    asks_show = _contains_any(q, ("show", "list", "display", "dikhao", "dikha", "sab", "all", "sabhi"))
    asks_salary = "salary" in q
    asks_category = _contains_any(q, ("category", "band", "bucket"))

    # Hinglish/all-rows phrasing -- but ONLY when the question is a truly
    # PLAIN "show all employees" with no other condition attached. This
    # used to fire on *any* question containing "employee(s)" + a
    # show-word + "sab"/"all"/"sabhi" -- so "show all employees who
    # joined after 2023" also matched (it does contain "show", "all",
    # "employees"), and this template short-circuited BEFORE the LLM
    # ever saw the question, silently discarding the "joined after 2023"
    # filter and returning every row instead. Now it only fires if every
    # remaining word in the question is generic request/glue phrasing
    # (nothing that looks like an actual filter condition, column, date,
    # or number) -- any real filter content makes it fall through to the
    # LLM instead, which does honor conditions.
    if (
        asks_employees
        and asks_show
        and _contains_any(q, ("sab", "all", "sabhi"))
        and _is_plain_all_employees_request(q)
    ):
        return f"SELECT * FROM employees WHERE ROWNUM <= {config.ROW_LIMIT}"

    # "show employees with a salary category" -> derive category by data
    # distribution (NTILE) instead of brittle hardcoded thresholds.
    if asks_employees and asks_salary and asks_category and asks_show:
        return (
            "SELECT * FROM ("
            "SELECT e.employee_id, e.first_name, e.last_name, e.salary, "
            "CASE NTILE(3) OVER (ORDER BY e.salary) "
            "WHEN 1 THEN 'Low' WHEN 2 THEN 'Medium' ELSE 'High' END AS salary_category "
            "FROM employees e"
            f") WHERE ROWNUM <= {config.ROW_LIMIT}"
        )

    # "count employees in each salary category" with derived category.
    asks_count = _contains_any(q, ("count", "how many", "kitne", "number of"))
    asks_each = _contains_any(q, ("each", "per", "every"))
    if asks_salary and asks_category and asks_count and asks_each:
        return (
            "SELECT salary_category, COUNT(*) AS employee_count "
            "FROM ("
            "SELECT CASE NTILE(3) OVER (ORDER BY e.salary) "
            "WHEN 1 THEN 'Low' WHEN 2 THEN 'Medium' ELSE 'High' END AS salary_category "
            "FROM employees e"
            ") "
            "GROUP BY salary_category "
            "ORDER BY CASE salary_category WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END"
        )

    # "salary greater than average" (with/without explicitly saying CTE).
    asks_avg = _contains_any(q, ("average", "avg", "mean"))
    asks_greater = _contains_any(q, ("greater", "more", "above", "higher", "zyada", "more than"))
    asks_cte = "cte" in q or "with" in q
    if asks_employees and asks_salary and asks_avg and asks_greater:
        return (
            "WITH avg_salary AS ("
            "SELECT AVG(salary) AS avg_sal FROM employees"
            ") "
            "SELECT e.employee_id, e.first_name, e.last_name, e.salary "
            "FROM employees e "
            "CROSS JOIN avg_salary a "
            "WHERE e.salary > a.avg_sal "
            + (f"AND ROWNUM <= {config.ROW_LIMIT}" if asks_cte or True else "")
        )

    return None


def _table_word_matches(words: set, schema_dict: dict) -> list:
    """All real tables whose name (singular or plural) appears as a
    whole word in `words`. A per-group ranking question legitimately
    mentions TWO tables (e.g. 'employees' and 'department'), so unlike
    _match_table_name this deliberately does not collapse that down to
    "ambiguous" — the caller decides how many matches make sense for the
    kind of question being templated."""
    matches = []
    for table in schema_dict:
        name = table.lower()
        if name in words or name.rstrip("s") in words or (name + "s") in words:
            matches.append(table)
    return matches


def _match_table_name(word_set: set, schema_dict: dict) -> Optional[str]:
    """Returns the ONE real table whose name (singular or plural) appears
    as a whole word in word_set, or None if zero or more-than-one table
    matches. Ambiguity is treated as "can't safely template this" rather
    than guessing — the LLM path handles it instead."""
    matches = _table_word_matches(word_set, schema_dict)
    return matches[0] if len(matches) == 1 else None


def _match_metric_column(word_set: set, question_text: str, columns: set) -> Optional[str]:
    """Returns the real column (from `columns`, a real table's column
    set) whose name — read as words, underscores as spaces — appears in
    the question, preferring the LONGEST/most specific match (so
    'hire_date' wins over a coincidental single-word overlap). Returns
    None on no match OR a tie, since a tie means the question is
    ambiguous about which column it wants — safer to fall back to the
    LLM than guess."""
    candidates = []
    for col in columns:
        phrase = col.lower().replace("_", " ")
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", question_text):
            candidates.append(col)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    if len(candidates) > 1 and len(candidates[0]) == len(candidates[1]):
        return None
    return candidates[0]


def _resolve_main_and_group_table(t1: str, t2: str, fk_pairs: list):
    """For a per-group ranking question that matched exactly two real
    table names, decides which one is the "main" table (the individual
    rows being ranked — the CHILD side of the FK) and which is the
    "group" table (the parent being grouped by), using the real FK
    direction instead of guessing from word order. Returns
    (main_table, group_table, fk_col, pk_col) or None if the two tables
    aren't directly related by exactly one FK in either direction."""
    found = []
    for ct, cc, pt, pc in fk_pairs:
        if ct == t1 and pt == t2:
            found.append((t1, t2, cc, pc))
        elif ct == t2 and pt == t1:
            found.append((t2, t1, cc, pc))
    return found[0] if len(found) == 1 else None


# Tokens allowed to remain unmatched around a ranking question before we
# trust a fully-deterministic (no-LLM) template with it. Anything else
# left over after removing the matched table/column/count/ranking words
# suggests an extra condition (a date, a status filter, "and", a second
# metric, ...) that this simple template can't safely represent — in
# that case we bail out and let the existing LLM + validation/repair
# pipeline handle it exactly as before. This mirrors the same
# conservative "only fire on a truly plain request" philosophy already
# used by _is_plain_all_employees_request above.
_RANKING_TEMPLATE_GLUE_TOKENS = {
    "show", "list", "display", "dikhao", "dikha", "de", "do", "batao",
    "mujhe", "please", "pls", "the", "me", "my", "a", "an", "of", "for",
    "top", "bottom", "highest", "high", "most", "maximum", "max", "best",
    "largest", "greatest", "newest", "latest", "youngest", "lowest",
    "low", "least", "minimum", "min", "worst", "smallest", "oldest",
    "earliest", "sabse", "zyada", "ziada", "acha", "behtareen", "naya",
    "nayi", "kam", "ghatia", "purana", "purani", "each", "every", "per",
    "har", "ek", "group", "groups", "in", "ka", "ki", "ke", "ko", "hai",
    "hain", "mein", "se", "wala", "wale", "wali", "lene", "by", "with",
    "rank", "ranked", "ranking", "exactly", "is", "at", "and", "with",
}


def _has_only_glue_leftovers(question_text: str, consumed_words: set) -> bool:
    words = re.findall(r"[a-z0-9]+", question_text)
    for w in words:
        if w in consumed_words or w in _RANKING_TEMPLATE_GLUE_TOKENS:
            continue
        if w.isdigit():
            continue
        return False
    return True


def _deterministic_ranking_sql(
    question: str, rank_intent: dict, schema_dict: dict
) -> Optional[str]:
    """Fully Python-built SQL for the two simplest, most common ranking
    shapes — global TOP_N / bottom-N and EXACT_RANK, optionally per-group
    (PARTITION BY one directly-related table). No LLM call at all, so
    there is zero syntax/scoping risk for these cases (the exact class of
    bug seen with a small local model: stray comments, CTE column-scope
    mistakes, wrong ASC/DESC).

    Every step below can bail out (return None) at the first sign of
    ambiguity — an unmatched or multiply-matched table/column, a missing
    count, leftover words that look like an extra filter, no direct FK
    for a per-group join. On any bail-out, question_to_sql() falls
    through to the existing RAG + LLM + validation/repair pipeline
    completely unchanged, so this can only ever ADD safe fast-path
    coverage, never remove coverage that already worked.
    """
    if rank_intent["ranking_type"] not in ("TOP_N", "EXACT_RANK"):
        return None
    if rank_intent["chained"] or rank_intent["compound_top_bottom"]:
        return None
    n = rank_intent.get("n")
    all_requested = bool(rank_intent.get("all_requested"))
    # EXACT_RANK ("the 3rd highest ...") always needs a concrete n; only
    # TOP_N can mean "no limit, give me all of them" (e.g. "sari").
    if rank_intent["ranking_type"] == "EXACT_RANK" and (not n or n <= 0):
        return None
    if rank_intent["ranking_type"] == "TOP_N" and not all_requested and (not n or n <= 0):
        return None

    q = _normalise_question(question)
    words = set(re.findall(r"[a-z0-9]+", q))

    partition_col = None
    group_display = None
    join_sql = None

    if rank_intent["per_group"]:
        # A per-group question legitimately names TWO tables (e.g.
        # "employees" and "department") — resolve both up front via the
        # real FK direction instead of the single-table matcher, which
        # would (correctly, for a non-grouped question) treat two
        # matches as ambiguous and bail.
        table_matches = _table_word_matches(words, schema_dict)
        if len(table_matches) != 2:
            return None
        fk_pairs = db.build_fk_pairs()
        resolved = _resolve_main_and_group_table(table_matches[0], table_matches[1], fk_pairs)
        if not resolved:
            return None
        table, group_table, fk_col, pk_col = resolved
        table_cols = schema_dict.get(table) or set()
        partition_col = fk_col
        name_cols = [c for c in schema_dict.get(group_table, set()) if c.endswith("_NAME")]
        group_display = name_cols[0] if len(name_cols) == 1 else pk_col
        join_sql = f"{table} m JOIN {group_table} g ON m.{fk_col} = g.{pk_col}"
    else:
        table = _match_table_name(words, schema_dict)
        if not table:
            return None
        table_cols = schema_dict.get(table) or set()

    metric_col = _match_metric_column(words, q, table_cols)
    if not metric_col:
        return None

    direction = rank_intent.get("direction") or "DESC"
    consumed = {table.lower(), table.lower().rstrip("s")}
    consumed |= set(metric_col.lower().split("_"))
    if rank_intent["per_group"]:
        consumed |= {group_table.lower(), group_table.lower().rstrip("s")}
        if group_display != pk_col:
            consumed |= set(group_display.lower().split("_"))

    if not _has_only_glue_leftovers(q, consumed):
        return None

    if rank_intent["per_group"]:
        select_cols = f"m.*, g.{group_display} AS group_label"
        rn_clause = f"ROW_NUMBER() OVER (PARTITION BY m.{partition_col} ORDER BY m.{metric_col} {direction}) AS rn"
        inner = f"SELECT {select_cols}, {rn_clause} FROM {join_sql}"
        if all_requested:
            # "sari"/"sabhi" -- every row per group, no top-N cutoff. The
            # overall ROW_LIMIT cap below still applies as a safety limit.
            sql = f"SELECT * FROM ({inner}) ORDER BY m.{partition_col}, rn"
        else:
            rn_filter = f"rn <= {n}" if rank_intent["ranking_type"] == "TOP_N" else f"rn = {n}"
            sql = f"SELECT * FROM ({inner}) WHERE {rn_filter}"
    elif rank_intent["ranking_type"] == "TOP_N":
        if all_requested:
            sql = f"SELECT * FROM {table} ORDER BY {metric_col} {direction}"
        else:
            sql = (
                f"SELECT * FROM (SELECT * FROM {table} ORDER BY {metric_col} {direction}) "
                f"WHERE ROWNUM <= {n}"
            )
    else:
        cols = ", ".join(sorted(table_cols))
        inner = (
            f"SELECT {cols}, ROW_NUMBER() OVER (ORDER BY {metric_col} {direction}) AS rn "
            f"FROM {table}"
        )
        sql = f"SELECT * FROM ({inner}) WHERE rn = {n}"

    return f"SELECT * FROM ({sql}) WHERE ROWNUM <= {config.ROW_LIMIT}"


def _deterministic_group_metric_ranking_sql(
    question: str,
    rank_intent: dict,
    schema_dict: dict,
) -> Optional[str]:
    """Build SQL for plain 'top/bottom N <group> by <business metric>'
    ranking questions without an LLM — e.g. "2 sab se zaida sales kis
    category ki hui" / "top 3 categories by sales".

    This complements _deterministic_ranking_sql (which only fires when the
    question literally names a real table) and
    _build_deterministic_single_table_claim_sql (which only fires for
    claim/assertion wording with a named candidate, e.g. "Mobile category
    has the highest sales"). A plain ranking QUESTION with no table name and
    no asserted candidate — "which category has the most sales?" or "top 2
    categories by sales" — fell through both of those and reached the small
    local model's preflight check, which regularly failed to answer even
    though the schema fully supports it deterministically.

    Reuses the same schema-verified resolve_business_metric /
    resolve_claim_dimension helpers the claim path already trusts, so this
    can only ever fire when both the metric and the grouping dimension are
    unambiguously confirmed to exist in the SAME real table.
    """
    if rank_intent["ranking_type"] not in ("TOP_N", "EXACT_RANK"):
        return None
    if rank_intent["chained"] or rank_intent["compound_top_bottom"] or rank_intent["per_group"]:
        return None
    n = rank_intent.get("n")
    all_requested = bool(rank_intent.get("all_requested"))
    # EXACT_RANK ("the 3rd highest category by sales") always needs a
    # concrete n; only TOP_N can mean "no limit, show every category"
    # (e.g. a "sari"/"sabhi" answer to a top-N clarification).
    if rank_intent["ranking_type"] == "EXACT_RANK" and (not n or n <= 0):
        return None
    if rank_intent["ranking_type"] == "TOP_N" and not all_requested and (not n or n <= 0):
        return None
    direction = rank_intent.get("direction") or "DESC"

    resolved_metric = query_grounding.resolve_business_metric(question, schema_dict)
    if not resolved_metric:
        return None

    metric_table = str(resolved_metric.get("table", "")).upper()
    metric_column = str(resolved_metric.get("column", "")).upper()
    if not metric_table or not metric_column:
        return None

    dimension = query_grounding.resolve_claim_dimension(
        question, schema_dict, preferred_table=metric_table
    )
    if not dimension:
        return None

    dimension_table = str(dimension.get("table", "")).upper()
    dimension_column = str(dimension.get("column", "")).upper()

    # Only safe when both facts live in the same table — cross-table
    # grouping still goes through the existing FK-grounded LLM path.
    if dimension_table != metric_table:
        return None

    table_columns = {str(c).upper() for c in schema_dict.get(metric_table, set())}
    if metric_column not in table_columns or dimension_column not in table_columns:
        return None

    inner = (
        f"SELECT {dimension_column} AS group_label, "
        f"SUM({metric_column}) AS metric_total, "
        f"ROW_NUMBER() OVER (ORDER BY SUM({metric_column}) {direction}) AS rn "
        f"FROM {metric_table} GROUP BY {dimension_column}"
    )
    if all_requested:
        # "sari"/"sabhi" -- every group, no top-N cutoff. The overall
        # ROW_LIMIT cap below still applies as a safety limit.
        sql = f"SELECT group_label, metric_total FROM ({inner}) ORDER BY rn"
    else:
        rn_filter = f"rn <= {n}" if rank_intent["ranking_type"] == "TOP_N" else f"rn = {n}"
        sql = f"SELECT group_label, metric_total FROM ({inner}) WHERE {rn_filter}"
    return f"SELECT * FROM ({sql}) WHERE ROWNUM <= {config.ROW_LIMIT}"


@st.cache_data(ttl=config.SCHEMA_CACHE_TTL)
def _load_examples() -> list:
    """Loads knowledge/examples.json once (same TTL as the schema caches,
    so hand-editing the file is picked up on the next cache refresh or
    app restart — no code change needed).

    A missing or malformed file is treated as "no examples yet" rather
    than a crash, so retrieval just falls back to schema-only prompting
    instead of taking the whole app down.
    """
    if not _EXAMPLES_PATH.exists():
        return []
    try:
        with _EXAMPLES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


@st.cache_data(ttl=config.SCHEMA_CACHE_TTL)
def _load_business_glossary() -> list:
    """Loads knowledge/business_terms.json -- the PERMANENT fix for
    business-title synonyms (e.g. "CEO", "department head") that have no
    literal column in the schema and no fixed vocabulary, so a keyword
    list can never cover them.

    Why a data file instead of hardcoding "CEO" as a special case in
    code: this is a knowledge-base, not logic. Adding a new synonym later
    (e.g. "founder", "MD") is a JSON edit, not a code change or a new
    regex -- same reasoning as knowledge/examples.json above. It also
    means the SAME glossary can be reused anywhere a prompt needs it
    (currently: the topic classifier below), without re-deriving it.

    Same fail-open contract as _load_examples(): a missing or malformed
    file is treated as "no glossary yet", never a crash.
    """
    if not _BUSINESS_TERMS_PATH.exists():
        return []
    try:
        with _BUSINESS_TERMS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _format_business_glossary(glossary: list) -> str:
    """Renders the glossary into a short block for the topic-classifier
    prompt. Kept deliberately terse (term + one-line meaning) -- this
    runs on every classification call, so it must stay small."""
    if not glossary:
        return ""
    lines = ["Known HR/org-chart terms that ARE part of this database even though they aren't literal column names:"]
    for entry in glossary:
        term = entry.get("term", "").strip()
        meaning = entry.get("meaning", "").strip()
        aliases = entry.get("aliases") or []
        if not term or not meaning:
            continue
        alias_text = f" (also called: {', '.join(aliases)})" if aliases else ""
        lines.append(f'- "{term}"{alias_text} -> {meaning}')
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_recent_history_for_classifier(history: Optional[list]) -> str:
    """Renders the last couple of turns from `history` (the same
    role/content list already used for the main SQL-generation call)
    into a short summary for the topic classifier.

    Only a SECONDARY safety net -- the glossary above is what makes a
    FRESH conversation correctly classify "who is the CEO" on the very
    first message. This is for genuine follow-up phrasing within an
    ongoing conversation ("tell me again", "check history", ...) that
    references something resolved a turn or two ago but isn't itself in
    the glossary. Deliberately short (last 2 user turns only, truncated)
    to keep the classifier prompt/latency small -- this is a hint for
    disambiguation, not a full transcript.
    """
    if not history:
        return ""
    user_turns = [m.get("content", "") for m in history if m.get("role") == "user"]
    recent = user_turns[-2:]
    if not recent:
        return ""
    lines = ["Recent questions earlier in this same conversation (for context only -- classify the CURRENT message, but use this to recognize if it refers back to something already established as a real database question):"]
    for q in recent:
        q = (q or "").strip().replace("\n", " ")
        if len(q) > 160:
            q = q[:160] + "..."
        lines.append(f"- {q}")
    return "\n".join(lines)


def _build_deterministic_single_table_claim_sql(
    question: str,
    resolved_metric: Optional[dict],
    schema_dict: dict,
) -> Optional[str]:
    """Build SQL for clear one-table highest/lowest claims without an LLM.

    This is intentionally narrow. It activates only when the live schema can
    prove all required pieces: claimed candidate, grouping dimension, metric,
    direction, and one table containing both the dimension and metric. That
    removes unnecessary clarification for cases such as:

        Mobile category ne sabse zyada sales ki hai

    while leaving joins and genuinely ambiguous claims on the existing guarded
    LLM path.
    """
    if not resolved_metric:
        return None
    if not query_grounding.is_simple_superlative_claim(question):
        return None

    candidate = query_grounding.extract_claim_candidate_value(question)
    direction = query_grounding.claim_sort_direction(question)
    if not candidate or not direction:
        return None

    metric_table = str(resolved_metric.get("table", "")).upper()
    metric_column = str(resolved_metric.get("column", "")).upper()
    if not metric_table or not metric_column:
        return None

    dimension = query_grounding.resolve_claim_dimension(
        question, schema_dict, preferred_table=metric_table
    )
    if not dimension:
        return None

    dimension_table = str(dimension.get("table", "")).upper()
    dimension_column = str(dimension.get("column", "")).upper()

    # Deterministic mode is only safe when both facts live in the same table.
    # Cross-table claims continue through the existing FK-grounded LLM path.
    if dimension_table != metric_table:
        return None

    table_columns = {str(c).upper() for c in schema_dict.get(metric_table, set())}
    if metric_column not in table_columns or dimension_column not in table_columns:
        return None

    # Preserve the user's value only for the final comparison. It is never
    # used in the aggregate/ranking WHERE clause that determines the winner.
    candidate_literal = candidate.replace("'", "''")

    return f"""WITH grouped_claim AS (
    SELECT
        {dimension_column} AS claim_dimension,
        SUM({metric_column}) AS claim_metric
    FROM {metric_table}
    GROUP BY {dimension_column}
), ranked_claim AS (
    SELECT
        claim_dimension,
        claim_metric,
        ROW_NUMBER() OVER (ORDER BY claim_metric {direction}) AS rn
    FROM grouped_claim
)
SELECT
    claim_dimension AS actual_winner,
    claim_metric AS actual_metric,
    CASE
        WHEN UPPER(TO_CHAR(claim_dimension)) = UPPER('{candidate_literal}')
        THEN 'TRUE'
        ELSE 'FALSE'
    END AS claim_verified
FROM ranked_claim
WHERE rn = 1"""


def _claim_verification_sql(
    question: str,
    table_chunks: dict,
    history: Optional[list],
) -> Optional[str]:
    """Handle factual/superlative claims in a separate verification mode.

    The key distinction from ordinary SQL generation is semantic: a named
    value in the user's claim is treated as UNTRUSTED. For example,
    "Mobile category has the highest sales" must calculate sales for all
    categories first and only then compare the actual winner with Mobile.

    Returns SQL/CLARIFY, or None when this mode is disabled/not applicable.
    Existing normal/ranking flow remains the fallback for all other questions.
    """
    if not getattr(config, "ENABLE_CLAIM_VERIFICATION", True):
        return None
    if not query_grounding.looks_like_claim_verification(question):
        return None

    schema_text = db.build_schema_text()
    schema_dict = db.build_schema_dict()
    fk_pairs = db.build_fk_pairs()
    relationship_text = query_grounding.relationship_text_from_fk_pairs(fk_pairs)

    # Resolve common business terms (for example, "sales") to a REAL schema
    # column before the claim-verification model is asked to plan SQL. This
    # prevents a small model from unnecessarily rewriting a clear claim as a
    # generic question such as "What is the category with the highest sales?"
    # If the live schema does not give one unique match, resolved_metric stays
    # None and the existing clarification safeguards remain in control.
    resolved_metric = query_grounding.resolve_business_metric(
        question, schema_dict
    )
    if resolved_metric:
        print(
            "[CLAIM] verified metric resolution: "
            f"{resolved_metric['metric_term']} -> "
            f"{resolved_metric['table']}.{resolved_metric['column']}"
        )

        deterministic_sql = _build_deterministic_single_table_claim_sql(
            question=question,
            resolved_metric=resolved_metric,
            schema_dict=schema_dict,
        )
        if deterministic_sql:
            print("[CLAIM] deterministic single-table verification path selected")
            return deterministic_sql

    prompt = query_grounding.build_claim_verification_prompt(
        question=question,
        schema_text=schema_text,
        relationship_text=relationship_text,
        resolved_metric=resolved_metric,
    )

    try:
        raw = llm.call_local_model([{"role": "system", "content": prompt}])
    except Exception as exc:
        # Do not break the existing application if the extra claim call is
        # unavailable. The normal pipeline remains available as a fallback.
        print(f"[CLAIM] verification call unavailable — falling back to existing pipeline: {exc}")
        return None

    raw = (raw or "").strip()
    if raw.upper().startswith("CLARIFY:"):
        return raw

    sql = sql_safety.clean_sql(raw)
    if not sql:
        print("[CLAIM] verification model returned empty SQL — falling back")
        return None

    issues = query_grounding.validate_claim_sql(sql, question)
    if issues:
        print(f"[CLAIM] verification SQL failed semantic guard: {issues} — requesting one repair")
        repair_prompt = query_grounding.build_claim_repair_note(
            issues=issues,
            question=question,
            schema_text=schema_text,
            relationship_text=relationship_text,
        )
        try:
            retry_raw = llm.call_local_model([
                {"role": "system", "content": prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_prompt},
            ])
        except Exception as exc:
            print(f"[CLAIM] repair unavailable — falling back to existing pipeline: {exc}")
            return None

        retry_raw = (retry_raw or "").strip()
        if retry_raw.upper().startswith("CLARIFY:"):
            return retry_raw
        retried_sql = sql_safety.clean_sql(retry_raw)
        if retried_sql and not query_grounding.validate_claim_sql(retried_sql, question):
            sql = retried_sql
        else:
            # If the model still produced a query that violates the claim
            # guard, do not execute it. Returning CLARIFY is safer than
            # silently treating the user's assertion as proven.
            return (
                "CLARIFY: I couldn't safely verify that claim from the available "
                "schema. Which field or metric should I use for the comparison?"
            )

    # Claim SQL still goes through the same FK JOIN trust boundary used by
    # the normal pipeline. This prevents the dedicated verifier from
    # becoming a back door for invented joins.
    join_violations = query_grounding.validate_sql_joins(
        sql, schema_dict, fk_pairs
    )
    if join_violations:
        print(f"[CLAIM] unverified JOIN in verification SQL: {join_violations} — requesting one repair")
        join_repair = query_grounding.build_join_repair_note(
            join_violations, relationship_text
        )
        try:
            join_retry_raw = llm.call_local_model([
                {"role": "system", "content": prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": join_repair},
            ])
        except Exception as exc:
            print(f"[CLAIM] JOIN repair unavailable — refusing unsafe verification SQL: {exc}")
            return (
                "CLARIFY: I couldn't safely verify that claim because the required "
                "table relationship could not be confirmed. Which tables or metric "
                "should I compare?"
            )
        join_retry_raw = (join_retry_raw or "").strip()
        if join_retry_raw.upper().startswith("CLARIFY:"):
            return join_retry_raw
        join_retry_sql = sql_safety.clean_sql(join_retry_raw)
        if not join_retry_sql or query_grounding.validate_sql_joins(
            join_retry_sql, schema_dict, fk_pairs
        ):
            return (
                "CLARIFY: I couldn't safely verify that claim from the available "
                "table relationships. Which tables or metric should I compare?"
            )
        sql = join_retry_sql

    return sql


def question_to_sql(question: str, history: Optional[list] = None) -> str:
    """
    Orchestrates one question -> SQL turn: retrieves the most relevant
    table chunks and (question, sql) examples via rag.py, builds the
    system prompt from just that retrieved subset (not the full schema),
    and gives the model up to one self-correction retry if it hallucinated
    columns or produced an invalid UNION shape.
    """
    # Checked FIRST, before any deterministic template or LLM call — see
    # _WRITE_INTENT_TOKENS above for why this can't wait until the LLM
    # sees the question.
    if _is_write_request(question):
        print("[INTENT] write/destructive request detected — refusing before any SQL generation")
        return "SELECT 'This assistant is read-only and cannot modify data' AS message FROM dual"

    if _is_greeting_or_smalltalk(question):
        print("[INTENT] greeting/small-talk detected — responding directly, no LLM call")
        return _smalltalk_reply_sql()

    if _needs_identity_clarification(question, history):
        print("[INTENT] self-referential question with no identity in context — asking to clarify")
        return (
            "CLARIFY: I don't have a logged-in user, so I don't know which "
            "employee \"you\" are — what's your name or employee ID?"
        )

    # Skip the classifier call entirely for a merged clarification-answer
    # block (see _CLARIFY_MERGE_RE above) -- that text is, by construction,
    # a continuation of a DB question we ourselves already started asking
    # about, so classifying it again is both wasted latency and a risk of
    # misreading our own boilerplate the same way the keyword helpers
    # above have to guard against.
    is_clarify_merge = bool(_CLARIFY_MERGE_RE.match(question.strip()))
    if config.ENABLE_TOPIC_CLASSIFICATION and not is_clarify_merge:
        topic = _classify_topic(question, history)
        if topic == "greeting":
            # The deterministic _is_greeting_or_smalltalk() check above only
            # catches an EXACT all-filler message; this catches a greeting
            # phrased in a way that check doesn't cover (e.g. mixed script,
            # a filler word not in the fixed list, ...).
            print("[INTENT] LLM classifier: greeting/small-talk — responding directly, no SQL call")
            return _smalltalk_reply_sql()
        if topic == "off_topic":
            print("[INTENT] LLM classifier: off-topic question — responding directly, no SQL call")
            _log_pending_term_candidate(question)
            return _off_topic_reply_sql()

    rank_intent = ranking_intent.classify(question)
    if rank_intent["is_ranking"]:
        print(f"[INTENT] ranking pattern detected: {rank_intent}")

    # CLAIM VERIFICATION MUST RUN BEFORE deterministic ranking. Otherwise a
    # statement such as "Mobile category has the highest sales" could be
    # mistaken for a normal ranking request and the claimed value could be
    # turned into a filter without first checking whether Mobile is actually
    # the winner. This mode is isolated; normal ranking remains unchanged.
    claim_sql = _claim_verification_sql(question, {}, history)
    if claim_sql:
        print("[CLAIM] independent claim-verification path selected")
        return claim_sql

    deterministic_sql = _deterministic_sql_from_intent(question)
    if deterministic_sql:
        print("[INTENT] matched deterministic SQL template")
        return deterministic_sql

    t0 = time.perf_counter()
    table_chunks = db.build_table_chunks()
    all_table_names = sorted(table_chunks.keys())
    examples = _load_examples()
    t1 = time.perf_counter()
    print(f"[TIMING] db.build_table_chunks() + _load_examples(): {t1 - t0:.2f}s")

    if rank_intent["is_ranking"]:
        schema_dict_early = db.build_schema_dict()
        ranking_sql = _deterministic_ranking_sql(question, rank_intent, schema_dict_early)
        if ranking_sql:
            print("[INTENT] matched deterministic ranking SQL template — no LLM call needed")
            return sql_safety.clean_sql(ranking_sql)

        group_metric_sql = _deterministic_group_metric_ranking_sql(
            question, rank_intent, schema_dict_early
        )
        if group_metric_sql:
            print("[INTENT] matched deterministic group-by-business-metric ranking template — no LLM call needed")
            return sql_safety.clean_sql(group_metric_sql)

        print("[INTENT] no confident deterministic ranking match — falling back to LLM pipeline")

    if _is_schema_meta_question(question):
        print("[INTENT] schema meta-question — answering directly from the real table list")
        return " UNION ALL ".join(
            f"SELECT '{name}' AS table_name FROM dual" for name in all_table_names
        )

    if _is_vague_records_request(question, table_chunks):
        print("[INTENT] vague 'records/data' request with no table named — asking to clarify")
        example_tables = ", ".join(all_table_names[:3])
        return f"CLARIFY: Which table would you like to see records from? (e.g. {example_tables})"

    # Existing preflight/clarification layer: it may STOP an ambiguous
    # relationship-heavy question, but it is never trusted to authorize SQL.
    # Claim questions were already handled above by the stricter claim path.
    # Skipped for a merged clarification-answer block for the same reason
    # the topic classifier is skipped above (see is_clarify_merge comment):
    # that text is, by construction, a continuation of a DB question we
    # ourselves already started asking about. Without this guard, the
    # preflight grounding checker -- a SEPARATE, narrower LLM prompt that
    # has no idea an identity/ambiguity question was already answered --
    # was free to read the synthetic "Original request / Clarification
    # asked / User's answer" block cold, get confused by its own
    # boilerplate, and issue a brand-new CLARIFY (worded differently from
    # ours) asking the user the same thing again. That produced an
    # effectively infinite clarification loop: every answer the user gave
    # got wrapped into another merged block, which the preflight checker
    # then misjudged as ambiguous all over again.
    if not is_clarify_merge and query_grounding.should_preflight(question, rank_intent):
        fk_pairs = db.build_fk_pairs()
        relationship_text = query_grounding.relationship_text_from_fk_pairs(fk_pairs)
        preflight_clarify = query_grounding.preflight_question(
            question=question,
            schema_text=db.build_schema_text(),
            relationship_text=relationship_text,
            history=history,
        )
        if preflight_clarify:
            print("[GROUNDING] preflight requested clarification")
            return preflight_clarify

    # RAG (embedding + top-k similarity) only matters when there's more
    # to choose from than we'd send anyway. rag.retrieve_tables()/
    # retrieve_examples() already cap top_k at min(top_k, total_count) —
    # so if the whole schema has fewer tables than RAG_TOP_K_TABLES (or
    # fewer curated examples than RAG_TOP_K_EXAMPLES), retrieval is
    # GUARANTEED to return everything regardless of similarity ranking.
    # In that case embedding the question is pure wasted latency (a real
    # network/local call to the embedding model) for an identical result,
    # so skip it entirely rather than hardcoding a table count anywhere —
    # this keeps working correctly and automatically re-enables itself if
    # the schema or example bank grows past the threshold later.
    needs_table_rag = len(table_chunks) > config.RAG_TOP_K_TABLES
    needs_example_rag = len(examples) > config.RAG_TOP_K_EXAMPLES

    if not needs_table_rag and not needs_example_rag:
        print(
            f"[TIMING] skipping RAG entirely — {len(table_chunks)} tables <= "
            f"RAG_TOP_K_TABLES ({config.RAG_TOP_K_TABLES}) and {len(examples)} "
            f"examples <= RAG_TOP_K_EXAMPLES ({config.RAG_TOP_K_EXAMPLES}), so "
            f"retrieval couldn't have filtered anything out anyway"
        )
        retrieved_chunks = table_chunks
        retrieved_examples = examples
        t2 = t3 = t4 = t1
    else:
        # Embed the question ONCE and reuse it for whichever retrieval(s)
        # actually need it — avoids sending the same question to the
        # embedding model twice.
        #
        # For a merged clarification-answer block, embed the CLEANED text
        # (just the original question + the user's actual answer — see
        # _normalise_question()/_CLARIFY_MERGE_RE above), not the raw
        # block with all its fixed labels/instruction sentences. Embedding
        # the raw block was diluting the similarity signal with boilerplate
        # ("Clarification question that was asked...", "Treat this as ONE
        # fully-specified request...") that has nothing to do with the
        # actual subject — concretely, this was observed to drop the
        # JOBS table out of the top-K for "what is my job title" once the
        # identity clarification got merged in, so the model never saw
        # JOBS.JOB_TITLE at all and hallucinated EMPLOYEES.JOB_TITLE
        # instead (a column that doesn't exist).
        rag_query_text = _normalise_question(question) if is_clarify_merge else question
        query_vec = rag.embed_query(rag_query_text)
        t2 = time.perf_counter()
        print(f"[TIMING] rag.embed_query() (question embed): {t2 - t1:.2f}s")

        retrieved_chunks = (
            rag.retrieve_tables(rag_query_text, table_chunks, query_vec=query_vec)
            if needs_table_rag
            else table_chunks
        )
        t3 = time.perf_counter()
        print(f"[TIMING] rag.retrieve_tables(): {t3 - t2:.2f}s")

        retrieved_examples = (
            rag.retrieve_examples(rag_query_text, examples, query_vec=query_vec)
            if needs_example_rag
            else examples
        )
        t4 = time.perf_counter()
        print(f"[TIMING] rag.retrieve_examples(): {t4 - t3:.2f}s")

        if needs_table_rag:
            # RAG picked tables by how well their text matches the
            # QUESTION's wording — but a correct JOIN can depend on a
            # "bridge" table the question never names (e.g. "region name,
            # country name, and departments per country" never says
            # "location", so LOCATIONS can miss the cut even though
            # DEPARTMENTS -> LOCATIONS -> COUNTRIES is the only real path
            # between them). Without the bridge table's real FK info in
            # context, the model guesses a plausible-looking wrong join
            # instead (e.g. COUNTRIES.COUNTRY_ID = DEPARTMENTS.LOCATION_ID
            # directly — a text column against a number column, which is
            # exactly what produced an ORA-01722 in testing). Pull in any
            # table that's a direct FK neighbor of an already-retrieved
            # table so the real join path is always available.
            relationship_graph = db.build_relationship_graph()
            bridge_tables = set()
            for table_name in retrieved_chunks:
                bridge_tables.update(relationship_graph.get(table_name, ()))
            bridge_tables -= retrieved_chunks.keys()
            if bridge_tables:
                print(
                    f"[TIMING] adding {len(bridge_tables)} FK-bridge table(s) "
                    f"retrieval missed: {sorted(bridge_tables)}"
                )
                for name in bridge_tables:
                    if name in table_chunks:
                        retrieved_chunks[name] = table_chunks[name]

    # If retrieval comes back empty (e.g. an empty schema), fall back to
    # the full schema text rather than sending the model nothing.
    retrieved_schema_text = (
        "\n\n".join(retrieved_chunks.values())
        if retrieved_chunks
        else db.build_schema_text()
    )

    # Real tables that RAG trimmed out of the prompt (only meaningful when
    # retrieval actually ran and actually removed something — if the
    # whole schema was sent, or nothing was filtered out, this is empty
    # and the prompt builders below add nothing for it, so behavior is
    # unchanged for those cases). Passed to the prompt builders as a
    # name-only fallback list — see the comment in llm.build_system_prompt
    # for why this exists.
    other_table_names = (
        sorted(set(table_chunks.keys()) - set(retrieved_chunks.keys()))
        if needs_table_rag
        else []
    )

    # Only ranking/superlative questions use the compact ranking prompt.
    # Every other question keeps the existing prompt and behavior unchanged.
    if rank_intent["is_ranking"]:
        system_content = llm.build_ranking_system_prompt(
            retrieved_schema_text, rank_intent, other_table_names=other_table_names
        )
        print("[INTENT] using isolated compact ranking prompt")
    else:
        system_content = llm.build_system_prompt(
            retrieved_schema_text,
            all_table_names,
            retrieved_examples,
            other_table_names=other_table_names,
        )

    system_message = {
        "role": "system",
        "content": system_content,
    }
    print(
        f"[TIMING] retrieved {len(retrieved_chunks)}/{len(table_chunks)} tables, "
        f"{len(retrieved_examples)} examples | prompt size: "
        f"{len(system_message['content'])} chars"
    )
    if history and (not rank_intent["is_ranking"] or _is_explicit_continuation(question)):
        messages = [system_message] + history + [{"role": "user", "content": question}]
    else:
        # A fresh ranking question does not need unrelated previous SQL.
        # This is the key protection against a previous WHERE/ORDER BY being
        # copied by a small local model. Explicit follow-ups still keep history.
        messages = [system_message, {"role": "user", "content": question}]

    # Rough token estimate (chars/4) across the WHOLE messages list, not
    # just the system prompt — history and the running question count
    # against num_ctx too. This is a heads-up, not exact tokenization, but
    # it's enough to catch "we're about to silently truncate the schema"
    # before it turns into a mystery hallucinated-column bug report.
    total_chars = sum(len(m.get("content", "")) for m in messages)
    approx_tokens = total_chars // config.APPROX_CHARS_PER_TOKEN
    budget_for_input = config.OLLAMA_NUM_CTX - config.OLLAMA_NUM_PREDICT
    if approx_tokens > budget_for_input:
        print(
            f"[WARN] prompt is ~{approx_tokens} tokens (est.), which exceeds "
            f"the ~{budget_for_input} tokens left for input after reserving "
            f"{config.OLLAMA_NUM_PREDICT} for the response (num_ctx="
            f"{config.OLLAMA_NUM_CTX}). Ollama will silently truncate the "
            f"oldest content — likely losing schema/table info the model "
            f"needs. Consider raising OLLAMA_NUM_CTX, lowering "
            f"RAG_TOP_K_TABLES/RAG_TOP_K_EXAMPLES, or trimming "
            f"MAX_HISTORY_TURNS."
        )
    elif approx_tokens > budget_for_input * 0.8:
        print(
            f"[WARN] prompt is ~{approx_tokens} tokens (est.), getting close "
            f"to the ~{budget_for_input} token input budget (num_ctx="
            f"{config.OLLAMA_NUM_CTX}, num_predict={config.OLLAMA_NUM_PREDICT})."
        )

    raw_sql = llm.call_local_model(messages)
    t5 = time.perf_counter()
    print(f"[TIMING] llm.call_local_model() (first SQL generation): {t5 - t4:.2f}s")
    print(f"[TIMING] TOTAL (before any retry): {t5 - t0:.2f}s")
    sql = sql_safety.clean_sql(raw_sql)
    sql = sql_safety.enforce_correct_row_limit(question, sql)

    if sql.strip().upper().startswith("CLARIFY:") or "from dual" in sql.lower():
        return sql

    if _looks_like_stale_filter_copy(question, history, sql):
        print(
            "[INTENT] SQL's WHERE clause is identical to the previous "
            "turn's and the two questions share no vocabulary — looks "
            "like a stale filter copied from history, retrying with a "
            "fresh (history-free) prompt"
        )
        fresh_messages = [system_message, {"role": "user", "content": question}]
        retry_raw = llm.call_local_model(fresh_messages)
        retried_sql = sql_safety.clean_sql(retry_raw)
        retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
        if not (
            retried_sql.strip().upper().startswith("CLARIFY:")
            or "from dual" in retried_sql.lower()
        ):
            sql = retried_sql
            raw_sql = retry_raw

    # Ranking questions get one compact, intent-aware validation pass before
    # the older targeted repair checks below. This does not run for normal
    # questions, so the existing behavior outside ranking is unchanged.
    if rank_intent["is_ranking"]:
        ranking_issues = []
        if sql_safety.needs_ranking_correction(question, sql):
            ranking_issues.append("missing top/bottom ordering or ranking")
        if sql_safety.needs_exact_rank_correction(question, sql):
            ranking_issues.append(f"exact rank {rank_intent.get('n')} is not implemented")
        if sql_safety.needs_rank_range_structure_correction(question, sql):
            ranking_issues.append("rank range needs ROW_NUMBER/RANK in a subquery")
        if sql_safety.needs_partition_correction(question, sql):
            ranking_issues.append("per-group ranking needs PARTITION BY")
        if sql_safety.needs_percent_correction(question, sql):
            ranking_issues.append("percentage was treated as rows or uses unsupported FETCH PERCENT")
        if sql_safety.needs_direction_correction(question, sql):
            ranking_issues.append("ASC/DESC direction is reversed")

        if ranking_issues:
            print(f"[INTENT] ranking validation failed: {ranking_issues} — asking for one combined repair")
            correction_note = (
                "Your SQL failed these ranking checks: " + "; ".join(ranking_issues) + ". "
                f"The extracted intent is {rank_intent}. Rewrite the SQL to satisfy ALL "
                "of the intent constraints at once. Use Oracle 11g syntax only: no "
                "LIMIT, TOP, OFFSET, or FETCH FIRST. Apply WHERE filters before ranking; "
                "use ROW_NUMBER/DENSE_RANK in a subquery for exact/range/group ranks; "
                "use PARTITION BY for each/per/har groups; use COUNT(*) OVER() for "
                "percentage selection; use ASC for bottom/lowest/oldest and DESC for "
                "top/highest/most/newest unless the intent requires otherwise. For a "
                "chained request, use CTEs/subqueries for the stages. Reply with ONLY "
                "the corrected SQL."
            )
            retry_raw = llm.call_local_model(
                messages + [
                    {"role": "assistant", "content": raw_sql},
                    {"role": "user", "content": correction_note},
                ]
            )
            retried_sql = sql_safety.clean_sql(retry_raw)
            retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
            if not (
                sql_safety.needs_ranking_correction(question, retried_sql)
                or sql_safety.needs_exact_rank_correction(question, retried_sql)
                or sql_safety.needs_rank_range_structure_correction(question, retried_sql)
                or sql_safety.needs_partition_correction(question, retried_sql)
                or sql_safety.needs_percent_correction(question, retried_sql)
                or sql_safety.needs_direction_correction(question, retried_sql)
            ):
                sql = retried_sql
                raw_sql = retry_raw

    if sql_safety.needs_ranking_correction(question, sql):
        print("[INTENT] superlative question but SQL has no ranking (ORDER BY/RANK) — asking model to add it")
        correction_note = (
            "The question asks for the top/highest/best (a superlative), "
            "but your SQL has no ORDER BY, RANK(), or ROW_NUMBER() — it "
            "just returns every group/row instead of ranking them. Add an "
            "ORDER BY on the correct aggregate/column (DESC for "
            "highest/most, ASC for lowest/least), wrap it in a subquery, "
            "and use ROWNUM to keep only the requested count (1 unless a "
            "different number was asked for). Reply with ONLY the "
            "corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
        if not sql_safety.needs_ranking_correction(question, retried_sql):
            sql = retried_sql
        # If the retry still has no ranking, fall through with the
        # original `sql` rather than looping forever — better to return
        # something than to hang on a model that won't self-correct.

    elif sql_safety.needs_exact_rank_correction(question, sql):
        target_rank = rank_intent.get("n") or 1
        print(f"[INTENT] exact numeric rank {target_rank} is missing/correctly unfiltered — asking model to fix")
        correction_note = (
            f"The question asks for EXACT rank {target_rank}. This is not top {target_rank}. "
            f"Use ROW_NUMBER() OVER (ORDER BY the requested metric DESC) AS rn "
            f"inside a subquery (or the correct ASC direction if the question asks "
            f"for the lowest/bottom ranking), then filter the OUTER query with "
            f"WHERE rn = {target_rank}. Return ONLY corrected Oracle 11g SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        if not sql_safety.needs_exact_rank_correction(question, retried_sql):
            sql = retried_sql

    elif sql_safety.needs_rank_range_structure_correction(question, sql):
        rank_range = rank_intent.get("range") or (None, None)
        print(f"[INTENT] rank-range request has no analytic rank — asking model to fix")
        correction_note = (
            f"The question asks for ranks {rank_range[0]} through {rank_range[1]}. "
            "Do NOT use ROWNUM BETWEEN. Use ROW_NUMBER() OVER (ORDER BY the "
            "requested metric DESC) AS rn inside a subquery, then filter the "
            "OUTER query with WHERE rn BETWEEN the requested endpoints. If the "
            "question explicitly asks for bottom/lowest, use ASC. If it says "
            "each/per/har group, add PARTITION BY the real group column. "
            "Return ONLY corrected Oracle 11g SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        if not sql_safety.needs_rank_range_structure_correction(question, retried_sql):
            sql = retried_sql

    elif sql_safety.needs_ordinal_correction(question, sql):
        target_rank = sql_safety.extract_ordinal_rank(question) or 2
        print(f"[INTENT] ordinal-rank question (target rank {target_rank}) but SQL has no RANK/ROW_NUMBER — asking model to add it")
        correction_note = (
            f"The question asks for the row at an EXACT rank (position "
            f"{target_rank} when sorted), not the top {target_rank} rows. "
            f"Your SQL used plain ORDER BY + ROWNUM, which returns rows 1 "
            f"through {target_rank} instead of only row {target_rank}. "
            f"Rewrite it using ROW_NUMBER() OVER (ORDER BY ... DESC) AS rn "
            f"in a subquery, then filter WHERE rn = {target_rank}. Reply "
            f"with ONLY the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        if not sql_safety.needs_ordinal_correction(question, retried_sql):
            sql = retried_sql

    elif sql_safety.has_broken_rownum_range(sql):
        print("[INTENT] broken ROWNUM range pattern (BETWEEN/> with ROWNUM) — always returns zero rows, asking model to fix")
        correction_note = (
            "Your SQL uses ROWNUM with BETWEEN or a comparison like '>' "
            "(e.g. 'WHERE ROWNUM BETWEEN 6 AND 10'). This is invalid in "
            "Oracle — ROWNUM is assigned incrementally only to rows that "
            "already pass the WHERE clause, so a condition like this can "
            "NEVER be true and the query will always return zero rows, "
            "no matter what the data is. Rewrite it using "
            "ROW_NUMBER() OVER (ORDER BY ...) AS rn in a subquery, then "
            "filter the OUTER query with WHERE rn BETWEEN ... (or "
            "whatever range/comparison was asked for). Reply with ONLY "
            "the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        if not sql_safety.has_broken_rownum_range(retried_sql):
            sql = retried_sql

    elif sql_safety.needs_rank_filter_correction(question, sql):
        print("[INTENT] rank-range question but SQL never filters by the computed rank — asking model to add the outer filter")
        correction_note = (
            "The question asks for a specific RANGE of ranks (e.g. "
            "'between rank 5 and 10'), and your SQL computes a "
            "ROW_NUMBER()/RANK() column, but never actually filters by "
            "it — every row is coming back instead of just the requested "
            "range. Wrap your ranked SELECT in a subquery and add an "
            "OUTER WHERE clause filtering that rank column to the exact "
            "range asked for (e.g. 'WHERE rn BETWEEN 5 AND 10'). Reply "
            "with ONLY the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        if not sql_safety.needs_rank_filter_correction(question, retried_sql):
            sql = retried_sql

    elif sql_safety.needs_partition_correction(question, sql):
        print("[INTENT] per-group superlative question but SQL has no PARTITION BY — asking model to add it")
        correction_note = (
            "The question asks for the top/highest results WITHIN EACH "
            "group (e.g. 'the 2 highest paid employees in each "
            "department'), not the top results across the whole table. "
            "Your SQL ranks across all rows globally instead of "
            "separately per group. Rewrite it using ROW_NUMBER() OVER "
            "(PARTITION BY <the grouping column> ORDER BY <the ranking "
            "column> DESC) AS rn in a subquery, then filter the outer "
            "query WHERE rn <= <the requested count per group>. Reply "
            "with ONLY the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        if not sql_safety.needs_partition_correction(question, retried_sql):
            sql = retried_sql

    elif sql_safety.needs_percent_correction(question, sql):
        print("[INTENT] percentage-based ranking question has unsupported FETCH PERCENT — asking model to fix")
        correction_note = (
            "Your SQL used 'FETCH FIRST n PERCENT ROWS ONLY', which is "
            "Oracle 12c+ syntax and does NOT work on this Oracle 11g "
            "database. Rewrite it using an 11g-compatible pattern "
            "instead: wrap your ranked query in a subquery that adds "
            "ROW_NUMBER() OVER (ORDER BY <ranking column> DESC) AS rn "
            "and COUNT(*) OVER () AS total_count, then filter the OUTER "
            "query with WHERE rn <= CEIL(total_count * <percent> / 100). "
            "Do not use FETCH FIRST, OFFSET, or LIMIT anywhere. Reply "
            "with ONLY the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        if not sql_safety.needs_percent_correction(question, retried_sql):
            sql = retried_sql

    def _finalize(candidate_sql: str, candidate_raw: str) -> str:
        """Runs the CTE-scope check and the final is_safe_select check on
        whichever SQL is about to be returned, giving each ONE repair
        attempt before falling through. Shared by every exit point below
        so a fix found in one retry path (e.g. invalid-columns) can't
        accidentally skip this net just because it returned early."""
        current_sql = candidate_sql
        current_raw = candidate_raw

        cte_violation = sql_safety.find_cte_scope_violation(current_sql)
        if cte_violation:
            cte_name, alias, bad_col, real_cols = cte_violation
            print(
                f"[INTENT] CTE scope violation: {alias}.{bad_col} referenced "
                f"but {cte_name} only exposes {real_cols} — asking model to fix"
            )
            correction_note = (
                f"Your SQL references {alias}.{bad_col}, but the CTE "
                f"{cte_name} never selected a column called {bad_col} — it "
                "only exposes: " + ", ".join(real_cols) + ". This causes "
                "ORA-00904 (invalid identifier) at query time. Do NOT "
                f"re-join a lookup table the CTE already joined. Either add "
                f"{bad_col} to {cte_name}'s own SELECT list, or reference an "
                f"already-available column from {cte_name} instead. Reply "
                "with ONLY the corrected SQL."
            )
            retry_raw = llm.call_local_model(
                messages + [
                    {"role": "assistant", "content": current_raw},
                    {"role": "user", "content": correction_note},
                ]
            )
            retried_sql = sql_safety.clean_sql(retry_raw)
            retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
            if not sql_safety.find_cte_scope_violation(retried_sql):
                current_sql = retried_sql
                current_raw = retry_raw

        # Same class of scoping mistake as the CTE check above, but for a
        # plain unnamed derived table: "FROM (SELECT ... FROM a x JOIN b
        # y ...) WHERE ..." with no alias on the subquery, where the
        # outer query still re-qualifies columns with x./y. — those
        # aliases don't exist outside the subquery, causing ORA-00904.
        unaliased_violation = sql_safety.find_unaliased_subquery_requalification(current_sql)
        if unaliased_violation:
            bad_alias, bad_col, subquery_snippet = unaliased_violation
            print(
                f"[INTENT] unaliased subquery re-qualification: "
                f"{bad_alias}.{bad_col} referenced outside its subquery — "
                "asking model to fix"
            )
            correction_note = (
                f"Your SQL wraps a subquery in FROM (...) with NO alias, "
                f"but then references {bad_alias}.{bad_col} outside it — "
                f"{bad_alias} was only a table alias INSIDE that subquery "
                "and does not exist outside it. This causes ORA-00904 "
                "(invalid identifier) at query time. Fix this by giving "
                "the subquery its own alias (e.g. 'FROM (...) sub') and "
                "referencing its exposed column names directly (without "
                "the old inner table aliases), or by using SELECT * on "
                "the subquery instead of re-listing qualified columns. "
                "Reply with ONLY the corrected SQL."
            )
            retry_raw = llm.call_local_model(
                messages + [
                    {"role": "assistant", "content": current_raw},
                    {"role": "user", "content": correction_note},
                ]
            )
            retried_sql = sql_safety.clean_sql(retry_raw)
            retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
            if not sql_safety.find_unaliased_subquery_requalification(retried_sql):
                current_sql = retried_sql
                current_raw = retry_raw

        # Hard JOIN trust boundary: every explicit JOIN must correspond to
        # a real Oracle FK relationship. This is deliberately inside the
        # finalization path so normal SQL, ranking SQL, and repaired SQL all
        # receive the same protection.
        if getattr(config, "ENABLE_QUERY_PREFLIGHT", True):
            fk_pairs = db.build_fk_pairs()
            join_violations = query_grounding.validate_sql_joins(
                current_sql, schema_dict, fk_pairs
            )
            if join_violations:
                print(f"[GROUNDING] final JOIN validation failed: {join_violations} — asking model to repair")
                join_repair_note = query_grounding.build_join_repair_note(
                    join_violations,
                    query_grounding.relationship_text_from_fk_pairs(fk_pairs),
                )
                retry_raw = llm.call_local_model([
                    *messages,
                    {"role": "assistant", "content": current_raw},
                    {"role": "user", "content": join_repair_note},
                ])
                retried_sql = sql_safety.clean_sql(retry_raw)
                retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
                if not query_grounding.validate_sql_joins(retried_sql, schema_dict, fk_pairs):
                    current_sql = retried_sql
                    current_raw = retry_raw
                else:
                    return (
                        "CLARIFY: I couldn't safely confirm the table relationship "
                        "needed for this question. Which related tables or field "
                        "do you mean?"
                    )

        result = sql_safety.ensure_default_row_cap(current_sql)

        # Deterministic safety net: don't rely on the model to remember
        # the "wrap text filters in UPPER() on both sides" prompt rule
        # every single time -- a small local model was observed to
        # half-apply it (uppercase the literal but leave the column bare,
        # e.g. WHERE first_name = 'SAMUEL'), which silently matches zero
        # rows against real mixed-case data. See
        # sql_safety.enforce_case_insensitive_text_filters() for details.
        result = sql_safety.enforce_case_insensitive_text_filters(result)

        # Last line of defense: the same read-only allow-list check app.py
        # runs before ever executing a query. Checking it here too — not
        # only in app.py after this function returns — means an unsafe
        # result (most commonly a stray SQL comment despite the prompt/
        # strip already discouraging it, or a leftover semicolon) gets one
        # repair attempt instead of a dead-end error with nothing left to
        # retry.
        if not sql_safety.is_safe_select(result):
            print("[INTENT] final SQL failed is_safe_select() — asking model for a clean rewrite")
            correction_note = (
                "Your SQL was rejected because it is not a single safe "
                "read-only SELECT statement (it may contain a comment "
                "using -- or /* */, a semicolon, or another statement). "
                "Reply with ONLY one plain SELECT statement — no comments, "
                "no semicolon, no explanation."
            )
            retry_raw = llm.call_local_model(
                messages + [
                    {"role": "assistant", "content": current_raw},
                    {"role": "user", "content": correction_note},
                ]
            )
            retried_sql = sql_safety.clean_sql(retry_raw)
            retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
            retried_result = sql_safety.ensure_default_row_cap(retried_sql)
            if sql_safety.is_safe_select(retried_result):
                result = retried_result

        return result

    schema_dict = db.build_schema_dict()

    # Table-existence gate: runs BEFORE the column-existence gate below,
    # because find_invalid_columns() only ever validates columns of
    # tables that already exist in schema_dict — a fully invented table
    # name (e.g. "customers", "provinces") with no qualified column
    # reference for it to catch would otherwise sail straight through
    # to Oracle as an ORA-00942. Same one-retry-then-CLARIFY shape as
    # the column check right after it, so behavior stays consistent.
    invalid_tables = sql_safety.find_invalid_tables(sql, schema_dict)
    if invalid_tables:
        correction_note = (
            "Your previous SQL referenced a table that does not exist: "
            + ", ".join(invalid_tables)
            + ". The real tables in this schema are: "
            + ", ".join(sorted(schema_dict.keys()))
            + ". Rewrite the query using ONLY real tables from that list. "
              "Reply with ONLY the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
        still_invalid_tables = sql_safety.find_invalid_tables(retried_sql, schema_dict)
        if not still_invalid_tables:
            sql = retried_sql
            raw_sql = retry_raw
        else:
            # Same idea as the column-check dead end below: hand it back
            # to the user as a normal CLARIFY turn instead of running a
            # query we already know references a table that doesn't exist.
            bad_tables = ", ".join(still_invalid_tables)
            return (
                "CLARIFY: I couldn't match this to a real table in the "
                f"schema (tried: {bad_tables}). Could you tell me exactly "
                "which table you mean, or rephrase using the field names "
                "shown in the app?"
            )

    invalid = sql_safety.find_invalid_columns(sql, schema_dict)

    if invalid:
        fk_pairs = db.build_fk_pairs()
        correction_lines = []
        for table, col in invalid:
            real_cols = ", ".join(sorted(schema_dict.get(table, [])))
            correction_lines.append(f"{table} actually has these columns: {real_cols}")
            # If the column the model wanted actually exists on exactly one
            # OTHER real table that's a direct FK neighbor of `table`, spell
            # out the real join instead of just listing what's missing.
            # Without this, the model's only options on retry are to keep
            # guessing invented column/view names (e.g. it previously
            # invented EMPLOYEES.JOB_TITLE, then on a later attempt
            # invented a differently-named EMP_DETAILS_VIEW that doesn't
            # exist in this schema either) -- neither of which the plain
            # "here are the real columns of the WRONG table" note above can
            # ever fix, since the column legitimately isn't on that table
            # at all. Pointing at the verified FK-backed join is the only
            # way to give it something it can actually run successfully.
            join_hint = _misplaced_column_hint(table, col, schema_dict, fk_pairs)
            if join_hint:
                correction_lines.append(join_hint)
        correction_note = (
            "Your previous SQL used columns that do not exist: "
            + ", ".join(f"{t}.{c}" for t, c in invalid)
            + ". " + " ".join(correction_lines)
            + " Rewrite the query using ONLY real columns from the schema — "
              "use a real JOIN if a join hint above tells you where the "
              "column actually lives, never invent a table/view name that "
              "isn't in the schema. Reply with ONLY the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
        still_invalid = sql_safety.find_invalid_columns(retried_sql, schema_dict)
        if not still_invalid:
            return _finalize(retried_sql, retry_raw)

        # The model couldn't self-correct even after seeing the real schema —
        # rather than run a query we already know references columns that
        # don't exist (or surface a raw exception), hand it back to the user
        # as a normal CLARIFY turn so they can supply the missing detail and
        # the conversation continues naturally.
        bad_cols = ", ".join(f"{t}.{c}" for t, c in still_invalid)
        return (
            "CLARIFY: I couldn't match this to real columns in the schema "
            f"(tried: {bad_cols}). Could you tell me exactly which "
            "table/column you mean, or rephrase using the field names shown "
            "in the app?"
        )

    # Catch the model literally UNION-ing raw '*' data across tables with
    # different shapes (invalid Oracle SQL / ORA-01789) before it ever
    # reaches the database, and give it one chance to self-correct using
    # the row-count-per-table pattern the prompt actually asks for.
    union_problem = sql_safety.find_union_shape_mismatch(sql, schema_dict)
    if union_problem:
        correction_note = (
            "Your previous SQL is invalid because " + union_problem + ". "
            "Oracle requires every branch of a UNION ALL to return the same "
            "number of columns. If the question was about totals/counts "
            "across tables, use the 'SELECT table_name, COUNT(*) AS "
            "row_count FROM table UNION ALL ...' pattern from the "
            "instructions instead. If the question was actually about "
            "records from one specific table, just query that ONE table "
            "with no UNION at all. Reply with ONLY the corrected SQL."
        )
        retry_raw = llm.call_local_model(
            messages + [
                {"role": "assistant", "content": raw_sql},
                {"role": "user", "content": correction_note},
            ]
        )
        retried_sql = sql_safety.clean_sql(retry_raw)
        retried_sql = sql_safety.enforce_correct_row_limit(question, retried_sql)
        if retried_sql.strip().upper().startswith("CLARIFY:"):
            return retried_sql
        if (
            not sql_safety.find_invalid_columns(retried_sql, schema_dict)
            and not sql_safety.find_union_shape_mismatch(retried_sql, schema_dict)
        ):
            return _finalize(retried_sql, retry_raw)

        # Same idea here: don't throw a raw error, ask the user which single
        # table they meant (or confirm they wanted row counts per table).
        return (
            "CLARIFY: That looks like it needs to combine multiple tables "
            "with different columns, which isn't possible in one query. "
            "Did you want records from ONE specific table (which one?), or "
            "just a row count per table?"
        )

    return _finalize(sql, raw_sql)