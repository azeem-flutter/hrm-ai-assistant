"""
sql_safety.py
-------------
Everything that stands between "text a small local LLM produced" and
"SQL that actually reaches Oracle": syntax repair (ROWNUM/LIMIT/FETCH),
statement de-duplication, the NOT IN -> NOT EXISTS null-trap fix, column
hallucination detection, UNION shape-mismatch detection, and the final
read-only allow-list check. No network, no DB, no LLM calls in this file
— pure string/regex logic, so it's trivially unit-testable.
"""

import re
from typing import Optional

from config import config

# word2number converts English number-words ("five", "twenty-one", ...)
# to ints. Wrapped in try/except so a missing/broken install never
# crashes the app -- it just falls back to digit-only + Urdu detection.
try:
    from word2number import w2n
    _HAS_WORD2NUMBER = True
except ImportError:
    _HAS_WORD2NUMBER = False

# Roman Urdu/Hindi number-words. word2number only understands English, so
# this covers the common local phrasing ("top paanch", "sabse zyada
# panch") separately. Extend this dict if a new word is seen in practice.
_URDU_WORD_TO_NUMBER = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4,
    "panch": 5, "paanch": 5, "che": 6, "chhe": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "dus": 10,
    "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14,
    "pandrah": 15, "solah": 16, "satrah": 17, "atharah": 18,
    "unnees": 19, "bees": 20,
}

# rapidfuzz gives a reliable, fast "closest match" for Roman Urdu number
# words that don't exactly match the dictionary above (Roman Urdu has no
# fixed spelling -- "unnees"/"unnee"/"unnis" are all the same word to a
# person, but three different strings to an exact lookup). Wrapped in
# try/except so a missing/broken install never crashes the app -- it
# just falls back to exact matches only.
try:
    from rapidfuzz import process as _fuzz_process, fuzz as _fuzz_scorer
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

_FUZZY_MATCH_THRESHOLD = 80  # 0-100 similarity; below this, say "unknown"


def _fuzzy_urdu_number(word: str) -> Optional[int]:
    """Best-effort match for a misspelled/variant Roman Urdu number word
    against our known list, e.g. 'unnee' -> 'unnees' -> 19. Returns None
    (never guesses wildly) if nothing is close enough, or if rapidfuzz
    isn't installed -- callers must treat None as "couldn't tell"."""
    if not _HAS_RAPIDFUZZ or not word:
        return None
    match = _fuzz_process.extractOne(
        word, _URDU_WORD_TO_NUMBER.keys(), scorer=_fuzz_scorer.ratio
    )
    if match and match[1] >= _FUZZY_MATCH_THRESHOLD:
        return _URDU_WORD_TO_NUMBER[match[0]]
    return None

# Words that signal a "give me exactly N (or exactly 1) top/bottom
# result(s)" question. Add new phrasings here as they're seen in
# real usage -- this list is intentionally not exhaustive.
_SUPERLATIVE_TOKENS = (
    "top", "best", "highest", "most", "lowest", "worst", "least", "bottom",
    # "sabse"/"sab se" is written both as one word and two, and "zyada"
    # ("most/more") is very commonly spelled "ziada"/"zyaada"/"zaida" in
    # Roman Urdu -- missing any of these spellings/spacings meant a
    # question like "sab se zaida" was invisible to this whole layer.
    "sabse zyada", "sab se zyada", "sabse ziada", "sab se ziada",
    "sabse zyaada", "sab se zyaada", "sabse zaida", "sab se zaida",
    "sabse acha", "sab se acha", "sabse behtareen", "sab se behtareen",
    "sabse kam", "sab se kam", "sabse ghatia", "sab se ghatia",
    # "oldest"/"newest"-style superlatives (missed before -- "sabse purana
    # employee" was falling through to the default 200-row limit because
    # none of these were recognized as a superlative at all).
    "oldest", "newest", "latest", "earliest", "youngest", "senior most",
    "senior-most", "junior most",
    "sabse purana", "sab se purana", "sabse purani", "sab se purani",
    "sabse naya", "sab se naya", "sabse nayi", "sab se nayi",
)

# Split of the tokens above into "ranked from the top" vs "ranked from
# the bottom" -- used to detect a COMBINED "top N and bottom N" question
# (see _is_compound_top_bottom_question below), which needs special
# handling: forcing a single row-limit on a query that UNIONs a top-N
# branch with a bottom-N branch would wrongly cut the combined result
# down to N total instead of leaving 2*N.
_HIGH_END_TOKENS = (
    "top", "best", "highest", "most", "newest", "latest", "youngest",
    "sabse zyada", "sab se zyada", "sabse ziada", "sab se ziada",
    "sabse zyaada", "sab se zyaada", "sabse zaida", "sab se zaida",
    "sabse acha", "sab se acha", "sabse behtareen", "sab se behtareen",
    "sabse naya", "sab se naya", "sabse nayi", "sab se nayi",
)
_LOW_END_TOKENS = (
    "bottom", "lowest", "worst", "least", "oldest", "earliest",
    "sabse kam", "sab se kam", "sabse ghatia", "sab se ghatia",
    "sabse purana", "sab se purana", "sabse purani", "sab se purani",
)


def _is_compound_top_bottom_question(question: str) -> bool:
    """True for a single question that asks for BOTH ends at once, e.g.
    "top 3 aur bottom 3 employees ek sath dikhao". A simple 'force the
    row limit to N' fix is wrong here because the correct total is the
    combination of both branches (e.g. 3 + 3 = 6), not N -- so callers
    should leave SQL matching this pattern untouched rather than risk
    truncating the combined UNION ALL result down to just N rows."""
    q = question.lower()
    has_high = any(tok in q for tok in _HIGH_END_TOKENS)
    has_low = any(tok in q for tok in _LOW_END_TOKENS)
    return has_high and has_low

_SET_OP_BEFORE_SELECT_RE = re.compile(
    r"\b(UNION\s+ALL|UNION|INTERSECT|MINUS)\s*$", re.IGNORECASE
)

_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "merge", "drop", "alter", "truncate",
    "create", "grant", "revoke", "commit", "rollback", "savepoint",
    "call", "exec", "execute", "pragma", "lock", "replace", "set",
    "into",  # blocks SELECT ... INTO, a PL/SQL assignment construct
)

# Oracle built-in package namespaces that can perform side effects
# (network calls, file/OS access, job scheduling, privilege changes,
# session manipulation) even from inside a nominally read-only SELECT,
# e.g. "SELECT UTL_HTTP.REQUEST('http://...') FROM dual". A plain
# keyword blacklist doesn't catch these because none of the words in
# _FORBIDDEN_KEYWORDS appear literally in the call.
_DANGEROUS_PACKAGE_PREFIXES = (
    "dbms_", "utl_", "owa_", "htp.", "htf.",
    "dbms_lock", "dbms_scheduler", "dbms_java", "dbms_advisor",
    "sys.", "systimestamp",  # sys.* covers SYS-owned dictionary/PLSQL objects
)

_PACKAGE_CALL_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _DANGEROUS_PACKAGE_PREFIXES) + r")",
    re.IGNORECASE,
)


def _dedupe_top_level_selects(sql: str) -> str:
    """If the model echoed more than one *separate* statement, keep only
    the last one. A genuine UNION/UNION ALL/INTERSECT/MINUS chain (e.g. the
    per-table row-count query) is NOT a duplicate and is left untouched."""
    positions = []
    depth = 0
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and sql[i:i + 6].lower() == "select":
            before_ok = i == 0 or not sql[i - 1].isalnum()
            after_ok = i + 6 >= n or not sql[i + 6].isalnum()
            if before_ok and after_ok:
                positions.append(i)
        i += 1

    if len(positions) <= 1:
        return sql

    statement_starts = [positions[0]]
    for pos in positions[1:]:
        if _SET_OP_BEFORE_SELECT_RE.search(sql[:pos]):
            continue
        statement_starts.append(pos)

    if len(statement_starts) > 1:
        return sql[statement_starts[-1]:].strip()
    return sql


def enforce_oracle11g_syntax(sql: str, limit: int = config.ROW_LIMIT) -> str:
    """Code-level safety net: rewrites FETCH FIRST / LIMIT / TOP (which the
    model isn't supposed to use, but small local models sometimes do
    anyway) into valid Oracle 11g ROWNUM syntax instead of sending broken
    SQL to the database."""
    original = sql
    sql = re.sub(r"\s+OFFSET\s+\d+\s+ROWS?\s*", " ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+FETCH\s+(FIRST|NEXT)\s+\d+\s+ROWS?\s+ONLY", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+LIMIT\s+\d+\s*$", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^\s*SELECT\s+TOP\s+\d+\s+", "SELECT ", sql, flags=re.IGNORECASE)

    changed = sql.strip() != original.strip()
    sql = sql.strip()

    is_aggregate_only = bool(
        re.match(r"^\s*SELECT\s+(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, re.IGNORECASE)
    )
    if changed and "rownum" not in sql.lower() and not is_aggregate_only:
        sql = f"SELECT * FROM ({sql}) WHERE ROWNUM <= {limit}"
    return sql


def simplify_unnecessary_rownum_wrapper(sql: str) -> str:
    """Flatten SELECT * FROM (<simple single-table query>) WHERE ROWNUM <= N
    into a direct single query, avoiding unnecessary wrappers for
    straightforward single-table reads."""
    pattern = re.compile(
        r"^\s*SELECT\s+\*\s+FROM\s*\(\s*(SELECT\s+.+?\s+FROM\s+[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\s+[A-Za-z_][A-Za-z0-9_]*)?(?:\s+WHERE\s+.+?)?)\s*\)\s+WHERE\s+ROWNUM\s*<=\s*(\d+)\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.match(sql)
    if not match:
        return sql

    inner_sql = match.group(1).strip()
    limit = match.group(2)
    # Real (multi-line) SQL separates clauses with newlines/indentation,
    # not always a single space -- e.g. "...150000\nORDER BY salary DESC".
    # Collapsing all whitespace runs to a single space before the
    # substring check ensures " order by " (and friends) is still found
    # regardless of how the model formatted the line breaks. Without
    # this, a query like "...150000\nORDER BY salary DESC" would slip
    # past the disallowed-token check (the literal-space token never
    # matches across a newline), and this function would then strip the
    # subquery wrapper fix_rownum_after_order_by just added, putting the
    # invalid "ORDER BY ... WHERE ROWNUM" shape right back and re-causing
    # the ORA-00933 it was supposed to fix.
    inner_lower = " ".join(inner_sql.lower().split())

    disallowed_tokens = (
        " join ", " order by ", " group by ", " distinct ",
        " union ", " connect by ", " having ",
    )
    if any(token in inner_lower for token in disallowed_tokens):
        return sql

    if " where " in inner_lower:
        return f"{inner_sql} AND ROWNUM <= {limit}"
    return f"{inner_sql} WHERE ROWNUM <= {limit}"


def fix_not_in_subquery(sql: str) -> str:
    """Fixes the NULL trap in 'col NOT IN (SELECT col2 FROM table)' by
    filtering NULLs out of the subquery, so it can't silently zero out
    otherwise-correct results."""
    pattern = re.compile(
        r"(\w[\w.]*)\s+NOT\s+IN\s*\(\s*(SELECT\s+([\w.]+)\s+FROM\s+[^()]+?)\)",
        re.IGNORECASE,
    )

    def _replace(match: "re.Match[str]") -> str:
        outer_col = match.group(1).strip()
        subquery = match.group(2).strip()
        inner_col = match.group(3).strip()
        if re.search(r"\bWHERE\b", subquery, re.IGNORECASE):
            fixed_subquery = f"{subquery} AND {inner_col} IS NOT NULL"
        else:
            fixed_subquery = f"{subquery} WHERE {inner_col} IS NOT NULL"
        return f"{outer_col} NOT IN ({fixed_subquery})"

    return pattern.sub(_replace, sql)


def _is_properly_wrapped_rownum_query(sql: str) -> bool:
    match = re.match(r"^\s*SELECT\s+\*\s+FROM\s*\(", sql, re.IGNORECASE)
    if not match:
        return False
    depth = 0
    i, n = match.end() - 1, len(sql)
    while i < n:
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                remainder = sql[i + 1:].strip()
                return bool(re.match(r"^WHERE\s+ROWNUM\s*<=\s*\d+$", remainder, re.IGNORECASE))
        i += 1
    return False


def _extract_requested_count(question: str) -> Optional[int]:
    """Pulls the count the user actually asked for out of a superlative
    question ("top 5", "top five", "top paanch", "sabse zyada" with no
    number at all).

    Returns:
      - an int, if a number was found (digit or word, English or Urdu)
      - 1, if a superlative was used but no number at all was given
        (a bare "top selling category" unambiguously means exactly one)
      - None, if we genuinely can't tell -- e.g. a number-like word was
        used that isn't in any of our lookups. In that case the caller
        should leave the LLM's own output alone rather than guess.
    """
    q = question.lower()

    # If this is a PERCENTAGE question ("top 10%", "top 10 percent"), the
    # number is a percentage, not a row count -- "top 10%" of 107
    # employees is ~11 rows, not literally 10. Detecting a digit here and
    # treating it as a row count would silently give the wrong count (and
    # risk clobbering a correct FETCH FIRST ... PERCENT clause the model
    # may have already written). Bail out entirely and let the LLM handle
    # percentage-based ranking on its own.
    if re.search(r"\d\s*%", q) or re.search(r"\d\s*percent\b", q):
        return None

    # 1) Digit right after "top" -- "top 5", "top-5", "top21"
    digit_after_top = re.search(r"\btop[\s-]*(\d+)\b", q)
    if digit_after_top:
        return int(digit_after_top.group(1))

    # 2) Any digit elsewhere in a superlative question -- "5 sabse zyada
    #    sale wali categories dikhao"
    any_digit = re.search(r"\b(\d+)\b", q)
    if any_digit:
        return int(any_digit.group(1))

    # 3) English number-word, any size ("five", "twenty-one", ...) --
    #    word2number scans the whole phrase for a number word itself,
    #    so this call fails harmlessly (ValueError) when none is present.
    if _HAS_WORD2NUMBER:
        try:
            return w2n.word_to_num(q)
        except ValueError:
            pass

    # 4) Roman Urdu/Hindi number-word from our small fixed list. Only
    #    checked on the word immediately after "top" -- NOT scanned across
    #    the whole sentence. Several Urdu number-words double as extremely
    #    common ordinary words elsewhere in a sentence (e.g. "do" means
    #    "give" as in "category do", not the number 2) -- scanning the
    #    whole sentence caused "sabse zyada sales wali category do" to be
    #    misread as asking for 2 rows. Restricting to right-after-"top"
    #    avoids that false positive; a bare superlative with no "top N"
    #    phrasing falls through to the "no number -> 1" rule below anyway.
    word_after_top_match = re.search(r"\btop\s+([a-z]+)", q)
    if word_after_top_match and word_after_top_match.group(1) in _URDU_WORD_TO_NUMBER:
        return _URDU_WORD_TO_NUMBER[word_after_top_match.group(1)]

    # 4b) Same word, but fuzzy -- catches spelling variants like "unnee"
    #     for "unnees" or "paach" for "paanch" that an exact match misses.
    if word_after_top_match:
        fuzzy_result = _fuzzy_urdu_number(word_after_top_match.group(1))
        if fuzzy_result is not None:
            return fuzzy_result

    if not any(tok in q for tok in _SUPERLATIVE_TOKENS):
        return None  # not even a superlative question -- nothing to say

    # 5) A superlative word was used, but no recognizable number. Two
    #    very different situations look identical at this point:
    #      a) "top selling category" -- no number was ever given, this
    #         genuinely means "exactly one" (the #1 category).
    #      b) "top ikkis categories" -- the user DID try to give a count,
    #         but "ikkis" (21) isn't in our small Urdu dictionary, so we
    #         missed it. Defaulting to 1 here would silently return the
    #         wrong thing instead of the 21 they actually asked for.
    #    We can't always tell these apart perfectly, but a reasonable
    #    signal is the word immediately after "top": if it's one of the
    #    common non-numeric words that normally follow it, no count was
    #    attempted at all (-> 1). If it's some other/unrecognized word,
    #    treat it as a possible missed number attempt and refuse to
    #    guess (-> None), leaving the LLM's own ROWNUM value untouched.
    _NON_NUMERIC_FILLERS = {
        "selling", "seller", "sellers", "performing", "performer",
        "rated", "paid", "earning", "earner", "category", "categories",
        "product", "products", "employee", "employees", "customer",
        "customers", "sale", "sales", "department", "departments",
        "record", "records", "result", "results", "item", "items",
        "value", "values", "in", "the", "of", "for",
    }
    word_after_top = re.search(r"\btop\s+([a-z]+)", q)
    if word_after_top and word_after_top.group(1) not in _NON_NUMERIC_FILLERS:
        return None  # looks like an unrecognized number attempt -- don't guess

    return 1


# Ordinal-rank phrasing ("second highest", "3rd highest", "dusra highest",
# "teesra zyada") means something fundamentally different from "top N":
# it asks for exactly the row AT that rank, not the first N rows. Forcing
# ROWNUM <= 2 on "second highest salary" would wrongly return BOTH the
# highest and second-highest, instead of only the second. These questions
# need a RANK()/ROW_NUMBER() window-function pattern instead, which isn't
# something a ROWNUM find-and-replace can safely produce -- so
# enforce_correct_row_limit() detects this phrasing and deliberately does
# nothing, leaving it entirely to the LLM (see the matching prompt
# example in llm.py).
_ORDINAL_RANK_TOKENS = (
    "second", "2nd", "third", "3rd", "fourth", "4th", "fifth", "5th",
    "dusra", "dusri", "doosra", "doosri",
    "teesra", "teesri", "tisra", "tisri",
    "chautha", "chauthi", "chotha", "chothi",
    "panchwa", "panchwan", "panchwin",
)


_GENERIC_ORDINAL_SUFFIX_RE = re.compile(
    r"\b\d+\s*(?:st|nd|rd|th|va|wan|waan|vi|vin|win|ven)\b", re.IGNORECASE
)


def _is_ordinal_rank_question(question: str) -> bool:
    q = question.lower()
    if _GENERIC_ORDINAL_SUFFIX_RE.search(q):
        return True
    return any(re.search(rf"\b{re.escape(tok)}\b", q) for tok in _ORDINAL_RANK_TOKENS)


_ORDINAL_TO_NUMBER = {
    "second": 2, "2nd": 2, "dusra": 2, "dusri": 2, "doosra": 2, "doosri": 2,
    "third": 3, "3rd": 3, "teesra": 3, "teesri": 3, "tisra": 3, "tisri": 3,
    "fourth": 4, "4th": 4, "chautha": 4, "chauthi": 4, "chotha": 4, "chothi": 4,
    "fifth": 5, "5th": 5, "panchwa": 5, "panchwan": 5, "panchwin": 5,
}


def extract_ordinal_rank(question: str) -> Optional[int]:
    """'second highest' -> 2, 'dusra sabse zyada' -> 2, '5va sab se bara'
    -> 5, etc. Returns None if no known ordinal word/pattern is found
    (caller should not guess in that case)."""
    q = question.lower()
    m = re.search(r"\b(\d+)\s*(?:st|nd|rd|th|va|wan|waan|vi|vin|win|ven)\b", q)
    if m:
        return int(m.group(1))
    for word, rank in _ORDINAL_TO_NUMBER.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return rank
    return None


def needs_ordinal_correction(question: str, sql: str) -> bool:
    """True when the question asks for an exact ordinal rank ('second
    highest', 'third highest') but the SQL has no RANK()/ROW_NUMBER()/
    DENSE_RANK() at all -- meaning the model fell back to a plain
    ORDER BY + ROWNUM pattern, which returns the wrong set of rows
    (the top N, not just the row AT that rank). enforce_correct_row_limit()
    deliberately leaves ordinal questions untouched (see its docstring),
    so this is the dedicated check that flags them for a correction
    retry instead."""
    if not _is_ordinal_rank_question(question):
        return False
    lowered_sql = sql.lower()
    has_rank_function = (
        "row_number(" in lowered_sql
        or "dense_rank(" in lowered_sql
        or "rank(" in lowered_sql
    )
    return not has_rank_function


def extract_where_clause(sql: str) -> Optional[str]:
    """Extracts and whitespace-normalizes the SQL's WHERE clause text,
    purely for byte-equality comparison across turns (e.g. detecting a
    model that copied the PREVIOUS turn's filter into an unrelated new
    question -- see _looks_like_stale_filter_copy in query_service.py).
    Not meant for reuse in an actual query. Returns None if there's no
    WHERE clause."""
    match = re.search(
        r"\bWHERE\b\s+(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\)\s*$|$)",
        sql, re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    normalized = " ".join(match.group(1).lower().split())
    # Strip table/alias qualifiers (e.g. "e.hire_date" -> "hire_date") so
    # two semantically-identical WHERE clauses still compare equal even
    # if the model happened to alias the table differently between two
    # turns (e.g. unaliased in one query, "e." in the next).
    normalized = re.sub(r"\b[a-z_][a-z0-9_]*\.", "", normalized)
    return normalized or None


def has_broken_rownum_range(sql: str) -> bool:
    """True if the SQL uses ROWNUM with BETWEEN, >, or >= -- these are
    an Oracle trap: ROWNUM is assigned incrementally and only to rows
    that already pass the WHERE clause, so 'ROWNUM BETWEEN 6 AND 10' or
    'ROWNUM > 5' can NEVER be true for any row -- the counter never gets
    past 1, so the query always silently returns zero rows regardless
    of data. Only ROWNUM <= N (or = 1) is safe on a single-level query."""
    return bool(re.search(r"ROWNUM\s*(BETWEEN|>=|>)", sql, re.IGNORECASE))


_RANK_RANGE_RE = re.compile(
    r"\brank(s)?\b.{0,20}\b(between|to|se|tak|and)\b|\bbetween\b.{0,20}\brank\b",
    re.IGNORECASE,
)



def needs_exact_rank_correction(question: str, sql: str) -> bool:
    """Flags numeric exact-rank wording such as 'salary rank exactly 50'.

    This is intentionally separate from ordinal words (second/third). A
    small model can understand the former as an ordinary filter and emit a
    plain ROWNUM/ORDER BY query, which is semantically wrong.
    """
    q = question.lower()
    match = re.search(
        r"\brank(?:ing)?\s*(?:is\s*)?(?:exactly\s*)?(?:=\s*)?(\d+)\b",
        q,
    )
    if not match:
        return False
    target = int(match.group(1))
    lowered = sql.lower()
    has_rank_function = any(
        token in lowered for token in ("row_number(", "dense_rank(", "rank(")
    )
    if not has_rank_function:
        return True
    # If a rank function exists, require an exact outer filter. Accept
    # rn/rank aliases and direct analytic expressions conservatively.
    exact_filter = bool(
        re.search(r"\b(?:rn|rank|rnk)\s*=\s*" + str(target) + r"\b", lowered)
    )
    return not exact_filter


def needs_rank_range_structure_correction(question: str, sql: str) -> bool:
    """Flags a rank range when the model never created an analytic rank."""
    if not _RANK_RANGE_RE.search(question.lower()):
        return False
    lowered = sql.lower()
    return not any(token in lowered for token in ("row_number(", "dense_rank(", "rank("))


def needs_rank_filter_correction(question: str, sql: str) -> bool:
    """True for a 'rank X to Y' / 'between rank X and Y' question where
    the SQL computes ROW_NUMBER()/RANK() but never actually filters on
    it with an outer WHERE -- so the ranking column is present in the
    output, but every row still comes back instead of just the
    requested range (the model computed the rank but forgot to filter
    by it)."""
    q = question.lower()
    if not _RANK_RANGE_RE.search(q):
        return False
    lowered_sql = sql.lower()
    has_rank_function = (
        "row_number(" in lowered_sql or "rank(" in lowered_sql or "dense_rank(" in lowered_sql
    )
    if not has_rank_function:
        return False
    # A properly filtered query wraps the ranked SELECT in a subquery and
    # filters it: "...) WHERE ...". If there's no closing-paren-then-WHERE
    # anywhere, the rank column was computed but never actually used to
    # filter the result.
    has_outer_filter = bool(re.search(r"\)\s*WHERE\b", sql, re.IGNORECASE))
    return not has_outer_filter


# Phrasing that implies the ranking should be computed SEPARATELY per
# group ("2 highest paid employees in EACH department") rather than
# globally across the whole table. This needs PARTITION BY inside the
# window function -- a plain ORDER BY + ROWNUM/rank without PARTITION BY
# only ever gives the top N overall, not the top N within every group.
_PER_GROUP_RE = re.compile(r"\b(har|each|per|every)\b", re.IGNORECASE)


def needs_partition_correction(question: str, sql: str) -> bool:
    """True for a per-group superlative question ('2 highest paid
    employees har department mein', 'top earner in each department')
    where the SQL has no PARTITION BY -- meaning it ranked/limited
    across the WHOLE table instead of separately within every group,
    silently returning the global top N instead of the top N per
    group."""
    q = question.lower()
    has_superlative = any(tok in q for tok in _SUPERLATIVE_TOKENS)
    if not has_superlative or not _PER_GROUP_RE.search(q):
        return False
    return "partition by" not in sql.lower()


_TEXT_FILTER_RE = re.compile(
    r"(?P<lhs>UPPER\s*\(\s*[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\s*\)"
    r"|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
    r"(?P<space1>\s*)(?P<op>=|LIKE)(?P<space2>\s*)"
    r"(?P<rhs>UPPER\s*\(\s*'[^']*'\s*\)|'[^']*')",
    re.IGNORECASE,
)


def enforce_case_insensitive_text_filters(sql: str) -> str:
    """Deterministic safety net: wraps any plain `column = 'literal'` or
    `column LIKE 'literal'` text filter in UPPER(...) on BOTH sides if it
    isn't already, instead of relying on the model to remember the "Text
    filters: UPPER(column) LIKE UPPER('%value%')" prompt rule every
    single time.

    Concrete bug this fixes: asked for "Samuel", a small local model
    produced `WHERE first_name = 'SAMUEL'` -- it uppercased the LITERAL
    (a half-remembered echo of the rule) but forgot to also wrap the
    COLUMN, so the comparison was still case-sensitive against whatever
    case the data is actually stored in and silently matched zero rows.
    Also covers the case where the model wraps one side but not the
    other (e.g. `UPPER(first_name) = 'Samuel'`).

    Deliberately narrow: only touches `identifier = 'quoted string'` /
    `identifier LIKE 'quoted string'` shapes (optionally already
    UPPER()-wrapped on either side). Never touches numeric/unquoted
    comparisons, join conditions (identifier = identifier, no quotes),
    or literals passed to another function like TO_DATE(...) -- so this
    can't turn a real numeric/date comparison into a broken one.
    Skipped for CLARIFY: responses, same as the other post-processors in
    this module.
    """
    stripped = sql.strip()
    if stripped.upper().startswith("CLARIFY:"):
        return sql

    def _fix(match: "re.Match") -> str:
        lhs, op, rhs = match.group("lhs"), match.group("op"), match.group("rhs")
        lhs_wrapped = lhs.strip().upper().startswith("UPPER(")
        rhs_wrapped = rhs.strip().upper().startswith("UPPER(")
        if lhs_wrapped and rhs_wrapped:
            return match.group(0)  # already correct, leave untouched
        new_lhs = lhs if lhs_wrapped else f"UPPER({lhs})"
        new_rhs = rhs if rhs_wrapped else f"UPPER({rhs})"
        return f"{new_lhs}{match.group('space1')}{op}{match.group('space2')}{new_rhs}"

    return _TEXT_FILTER_RE.sub(_fix, sql)


def ensure_default_row_cap(sql: str) -> str:
    """Final safety net applied to whatever SQL is about to be executed:
    caps the result at config.ROW_LIMIT rows if the model produced a
    query with NO limiting mechanism at all -- not superlative-specific,
    this catches plain "show me X ranked/ordered" requests that aren't a
    "top N" question (so enforce_correct_row_limit() never touches them)
    but still deserve a safety cap, e.g. "employees ranked by salary"
    silently returning all 107+ rows with a rank column instead of a
    sane, bounded result set.

    Skipped for:
      - CLARIFY: responses and the read-only refusal message (not real
        data queries).
      - SQL that already has SOME limiting mechanism (ROWNUM, FETCH
        FIRST/PERCENT) -- don't double-wrap.
      - A true single-row aggregate query (a lone COUNT/SUM/AVG/MAX/MIN
        with no GROUP BY) -- structurally can only ever return one row,
        wrapping it would just be noise.
    """
    stripped = sql.strip()
    if stripped.upper().startswith("CLARIFY:"):
        return sql
    if "from dual" in stripped.lower():
        return sql

    lowered = stripped.lower()
    if "rownum" in lowered or "fetch first" in lowered or "fetch next" in lowered:
        return sql  # already has some limiting mechanism

    is_single_aggregate = bool(
        re.match(r"^\s*select\s+(count|sum|avg|max|min)\s*\(", lowered)
    ) and "group by" not in lowered
    if is_single_aggregate:
        return sql

    return f"SELECT * FROM ({stripped.rstrip(';')}) WHERE ROWNUM <= {config.ROW_LIMIT}"


# Oracle 12c+ introduced "FETCH FIRST n PERCENT ROWS ONLY" -- this app
# targets Oracle 11g (see enforce_oracle11g_syntax), which does NOT
# support that syntax at all. A model that reaches for it produces SQL
# that either errors outright or, if our own code naively wraps it with
# an extra ROWNUM layer (as an earlier version of this fix did), breaks
# in a different way (ORA-00907 seen in testing). Percentage-based
# ranking needs a dedicated 11g-compatible pattern (COUNT(*) OVER() to
# get the total, then ROW_NUMBER() <= CEIL(total * pct / 100)) that a
# small local model isn't likely to produce unprompted -- so this is
# handled with an explicit correction retry + a matching prompt example,
# the same approach used for the other structural SQL patterns above.
_PERCENT_QUESTION_RE = re.compile(r"\d+\s*%|\d+\s*percent\b", re.IGNORECASE)
_FETCH_PERCENT_SQL_RE = re.compile(
    r"FETCH\s+(FIRST|NEXT)\s+\d+\s+PERCENT", re.IGNORECASE
)


def needs_percent_correction(question: str, sql: str) -> bool:
    """Flags percentage ranking SQL that is either Oracle-12c FETCH syntax
    or has incorrectly treated the percentage as a literal row count.
    """
    q = question.lower()
    if not _PERCENT_QUESTION_RE.search(q):
        return False

    if _FETCH_PERCENT_SQL_RE.search(sql):
        return True

    # A percentage request must not collapse into ROWNUM <= 20 for "20%".
    # Such a query is only correct by coincidence when the table has exactly
    # 100 rows, so force the model onto the COUNT(*) OVER() pattern.
    pct_match = _PERCENT_QUESTION_RE.search(q)
    pct = float(pct_match.group(1)) if pct_match else None
    if pct is not None:
        if re.search(r"\bROWNUM\s*<=\s*\d+\b", sql, re.I):
            rownums = [int(x) for x in re.findall(r"\bROWNUM\s*<=\s*(\d+)\b", sql, re.I)]
            if any(x == pct for x in rownums):
                return True
        if "count(*) over" not in sql.lower() and "percent_rank(" not in sql.lower() and "cume_dist(" not in sql.lower():
            return True

    return False



def needs_direction_correction(question: str, sql: str) -> bool:
    """Flags an obvious ASC/DESC inversion for simple superlative wording.

    This is deliberately conservative: only inspect the first ORDER BY
    direction, and only when the question contains an unambiguous high/low
    superlative. Complex/chained queries are left to the ranking prompt.
    """
    q = question.lower()
    low = any(tok in q for tok in _LOW_END_TOKENS)
    high = any(tok in q for tok in _HIGH_END_TOKENS)
    if low == high:  # ambiguous or both ends requested
        return False
    match = re.search(r"\border\s+by\b(.{0,300}?)(?:\)|$)", sql, re.I | re.S)
    if not match:
        return False
    order_text = match.group(1).lower()
    first_dir = re.search(r"\b(asc|desc)\b", order_text)
    if not first_dir:
        return False
    return (low and first_dir.group(1) == "desc") or (high and first_dir.group(1) == "asc")


def needs_ranking_correction(question: str, sql: str) -> bool:
    """True when the question is a superlative ("top", "sabse zyada",
    "highest", etc.) but the generated SQL has no ranking at all --
    no ORDER BY, RANK(), or ROW_NUMBER(). This is a DIFFERENT failure
    from a wrong ROWNUM value: enforce_correct_row_limit() can only fix
    the *number* on an existing ranked query, it can't invent a ranking
    that was never written (e.g. the model just did a plain
    'GROUP BY category' with no ORDER BY, so every category comes back
    instead of just the top one). Callers should use this to trigger an
    explicit correction retry to the LLM rather than silently rewriting
    SQL whose ranking column/direction we can't reliably guess."""
    q = question.lower()
    if not any(tok in q for tok in _SUPERLATIVE_TOKENS):
        return False
    if re.search(r"\d\s*%", q) or re.search(r"\d\s*percent\b", q):
        return False  # percentage questions are handled by the LLM/FETCH...PERCENT, not this path
    lowered_sql = sql.lower()
    has_ranking = (
        "order by" in lowered_sql
        or "row_number(" in lowered_sql
        or "rank(" in lowered_sql
        or "dense_rank(" in lowered_sql
    )
    return not has_ranking


def enforce_correct_row_limit(question: str, sql: str) -> str:
    """Deterministically fixes the row limit on superlative questions
    ("top N", "top selling X", "sabse zyada", "highest paid employee")
    instead of trusting the LLM to have picked the right ROWNUM value --
    small local models often default to the standard row limit (e.g.
    200) even when the question clearly asked for just 1 or N rows.

    Safe by construction:
      - Does nothing if the question isn't a superlative question.
      - Does nothing if the SQL has no ORDER BY (nothing being ranked,
        so there's no "top N" to enforce).
      - Does nothing if the count genuinely can't be determined (see
        _extract_requested_count) -- leaves the LLM's own ROWNUM value
        untouched rather than risk overwriting a value that might
        already be correct.
      - Does nothing for a combined "top N and bottom N" question --
        forcing a single limit would wrongly truncate the combined
        UNION ALL result.
      - Does nothing for a percentage question ("top 10%") or SQL that
        already uses FETCH FIRST ... PERCENT/ROWS ONLY -- that's already
        self-limiting and wrapping it again with ROWNUM caused an
        ORA-00907 syntax error in testing.
      - Does nothing for a per-group question ("2 highest paid in each
        department") -- see needs_partition_correction, which handles
        that case with a dedicated retry instead.
    """
    q = question.lower()
    if not any(tok in q for tok in _SUPERLATIVE_TOKENS):
        return sql
    if "order by" not in sql.lower():
        return sql
    if _is_ordinal_rank_question(question):
        # "second/third/Nth highest" needs a RANK()/ROW_NUMBER() = N
        # pattern, not ROWNUM <= N -- leave this entirely to the LLM
        # (guided by the matching prompt example) rather than risk
        # forcing the wrong SQL shape here.
        return sql
    if _is_compound_top_bottom_question(question):
        return sql  # combined top+bottom -- forcing a single N would truncate the UNION
    if re.search(r"\d\s*%", q) or re.search(r"\d\s*percent\b", q):
        return sql  # percentage-based ranking -- different semantics, leave to the LLM
    if "fetch first" in sql.lower():
        return sql  # already self-limiting via Oracle's native top-N/percent syntax
    if _PER_GROUP_RE.search(q):
        return sql  # per-group ranking -- handled separately by needs_partition_correction

    correct_limit = _extract_requested_count(question)
    if correct_limit is None:
        return sql  # couldn't tell -- don't guess, leave the SQL as-is

    new_sql, replaced = re.subn(
        r"ROWNUM\s*<=\s*\d+", f"ROWNUM <= {correct_limit}", sql, flags=re.IGNORECASE
    )
    if replaced == 0:
        # Model produced a ranked query but forgot the ROWNUM wrapper
        # entirely -- wrap it now rather than let every row through.
        new_sql = f"SELECT * FROM ({sql.rstrip(';')}) WHERE ROWNUM <= {correct_limit}"
    return new_sql


def fix_rownum_after_order_by(sql: str) -> str:
    """The model occasionally tacks a dangling 'WHERE ROWNUM <= N' onto a
    query that already has its own WHERE/ORDER BY/GROUP BY clause, e.g.:
      - '... ORDER BY col DESC WHERE ROWNUM <= N'   (WHERE after ORDER BY)
      - '... WHERE salary BETWEEN 1 AND 2 WHERE ROWNUM <= N'  (2nd WHERE)
      - '... GROUP BY dept WHERE ROWNUM <= N'       (WHERE after GROUP BY)
    All three are invalid Oracle syntax and raise ORA-00933. This is a
    deterministic backstop that rewrites the broken shape into the
    correct subquery-wrapped form, regardless of which earlier clause
    the dangling 'WHERE ROWNUM' was appended after.
    """
    sql = sql.strip()
    if _is_properly_wrapped_rownum_query(sql):
        return sql

    match = re.match(
        r"^(?P<body>.*)\s+WHERE\s+ROWNUM\s*<=\s*(?P<limit>\d+)\s*$",
        sql, re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return sql

    body = match.group("body").strip()
    limit = match.group("limit")

    # Only rewrap if `body` (everything BEFORE this trailing "WHERE
    # ROWNUM") already contains its own WHERE, ORDER BY, or GROUP BY --
    # that's what makes the trailing "WHERE ROWNUM" invalid in the first
    # place (a second WHERE, or WHERE after ORDER/GROUP BY). If none of
    # those are present, "WHERE ROWNUM <= N" was the query's one and only
    # (legitimate) WHERE clause, so leave it untouched.
    if not re.search(r"\b(WHERE|ORDER\s+BY|GROUP\s+BY)\b", body, re.IGNORECASE):
        return sql

    return f"SELECT * FROM ({body}) WHERE ROWNUM <= {limit}"


# Oracle pseudo-tables that are always valid even though they never show
# up in all_tab_columns for the app's schema owner — must never be
# flagged as a hallucinated table.
_ALWAYS_VALID_TABLES = {"DUAL"}


def find_invalid_tables(sql: str, schema_dict: dict) -> list:
    """Checks every table name referenced after FROM/JOIN against the real
    schema and returns the ones that don't exist AT ALL — this is what
    catches a fully hallucinated table (e.g. a fake 'customers' or
    'provinces' table) reaching Oracle as an ORA-00942.

    This is deliberately a separate check from find_invalid_columns():
    that function only validates columns of tables that ARE already in
    schema_dict (`if not table or table not in schema_dict: continue`),
    so a table name that doesn't exist at all — with no qualified
    column reference for it to catch, e.g. a bare "SELECT * FROM
    customers" — silently sails through it undetected. This function
    closes that gap by checking table existence directly, independent
    of whether any column was ever qualified with that table's name.

    Works the same whether the table appears in a flat query or nested
    inside a nested/ROWNUM-wrapped subquery, since it scans for every
    FROM/JOIN keyword in the SQL text regardless of paren nesting depth.
    """
    invalid = []
    seen = set()
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
        sql, re.IGNORECASE,
    ):
        table = match.group(1).upper()
        if table in _ALWAYS_VALID_TABLES:
            continue
        if table not in schema_dict and table not in seen:
            invalid.append(table)
            seen.add(table)
    return invalid


def find_invalid_columns(sql: str, schema_dict: dict) -> list:
    """Checks every 'alias.column' reference in the SQL against the real
    schema and returns (table, column) pairs that don't exist — this is
    what catches model hallucination before it becomes an ORA-00904."""
    reserved_aliases = {"select", "where", "group", "order", "having", "on"}

    alias_map = {}
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)?",
        sql, re.IGNORECASE,
    ):
        table = match.group(1)
        alias = match.group(2)
        if alias and alias.lower() in reserved_aliases:
            alias = None
        alias_map[table.lower()] = table.upper()
        if alias:
            alias_map[alias.lower()] = table.upper()

    invalid = []
    seen = set()
    for alias, column in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", sql):
        table = alias_map.get(alias.lower())
        if not table or table not in schema_dict:
            continue
        if column.upper() not in schema_dict[table] and (table, column.upper()) not in seen:
            invalid.append((table, column.upper()))
            seen.add((table, column.upper()))

    # Also catch invalid unqualified identifiers (e.g., salary_category)
    # in simple single-table SQL where the target table is unambiguous.
    for table, column in _find_invalid_unqualified_columns(sql, schema_dict):
        if (table, column) not in seen:
            invalid.append((table, column))
            seen.add((table, column))

    return invalid


def _count_top_level_commas(cols_part: str) -> int:
    """Counts columns in a SELECT list, ignoring commas inside function
    calls like COUNT(a, b) or TO_CHAR(x, 'fmt')."""
    depth = 0
    count = 1
    for ch in cols_part:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def _split_top_level_commas(expr: str) -> list:
    """Splits expression lists by top-level commas only."""
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(expr[start:i].strip())
            start = i + 1
    tail = expr[start:].strip()
    if tail:
        parts.append(tail)
    return parts


# Matches ONLY the app's own standard single-level pagination wrapper --
# "SELECT * FROM (<inner>) WHERE ROWNUM <= N" -- exactly the shape
# produced by ensure_default_row_cap()/enforce_correct_row_limit().
# Deliberately does NOT match the ranking shape "... ) WHERE rn = 1"
# (that outer WHERE is a computed-alias comparison, not literally
# "ROWNUM <= <number>"), so it can never interfere with that pattern.
_ROWNUM_PAGINATION_WRAP_RE = re.compile(
    r"^\s*SELECT\s*\*\s*FROM\s*\((?P<inner>.*)\)\s*WHERE\s+ROWNUM\s*<=\s*\d+\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _unwrap_simple_rownum_pagination(sql: str) -> str:
    """If `sql` is exactly the app's own single-level ROWNUM pagination
    wrapper around a flat, single-table, non-nested, non-JOIN inner
    query, returns just the inner query text -- so unqualified-column
    checks can see the REAL SELECT/GROUP BY/WHERE clauses (e.g.
    "SELECT category, SUM(sale_amount) ... FROM commercial_orders
    GROUP BY category") instead of the wrapper's own dummy "SELECT *"
    and "WHERE ROWNUM <= N", which carry no useful column information
    and previously caused this whole check to be skipped for every
    ROWNUM-wrapped query (the app's most common final query shape).

    Falls back to returning `sql` completely UNCHANGED for anything
    that doesn't match this one exact, unambiguous wrapper shape --
    including the ranking pattern "SELECT * FROM (...ROW_NUMBER()...)
    WHERE rn = 1", multi-level nesting, or any query containing a JOIN.
    Those cases are untouched on purpose: the outer WHERE in the ranking
    shape references a computed alias (like ROW_NUMBER() AS rn) that
    only exists inside the subquery, and checking it against the inner
    table's real columns would be a false positive -- exactly what the
    original blanket "FROM (" bail-out in _extract_single_table_context
    exists to prevent. This helper only unwraps when it's certain the
    outer clause carries no such risk.
    """
    match = _ROWNUM_PAGINATION_WRAP_RE.match(sql)
    if not match:
        return sql
    inner = match.group("inner")
    if re.search(r"\bJOIN\b", inner, re.IGNORECASE):
        return sql
    if re.search(r"\bFROM\s*\(", inner, re.IGNORECASE):
        return sql
    return inner


def _extract_single_table_context(sql: str) -> tuple:
    """Returns (TABLE_NAME, alias) for simple one-table, no-join,
    no-nested-subquery SQL.

    Deliberately bails out (None, None) whenever the SQL wraps a
    subquery in its FROM clause (e.g. "FROM (SELECT ... ) WHERE rn =
    N" from a ranking/ROW_NUMBER pattern). A naive "exactly one FROM
    <table>" scan matches the INNERMOST table in that shape (since
    "FROM (" doesn't match the identifier pattern), which then makes
    the unqualified-column checks below validate the OUTER query's
    computed aliases (like a ROW_NUMBER() AS rn column) against that
    inner table's real columns -- flagging perfectly valid nested
    ranking SQL as if it referenced a hallucinated column. This
    function's whole contract is "only simple, truly flat SQL", so a
    nested derived table means "don't guess, skip this check" rather
    than risk a false positive.

    EXCEPTION (added): the app's own plain ROWNUM pagination wrapper
    (see _unwrap_simple_rownum_pagination) is first peeled off, if
    present, before any of the checks above run -- that specific shape
    is safe to see through because its outer clause is always just a
    literal "WHERE ROWNUM <= N" with no computed-alias reference at
    all, so unwrapping it carries none of the risk the bail-out above
    protects against. Every other nested/JOIN shape still bails out
    exactly as before.
    """
    sql = _unwrap_simple_rownum_pagination(sql)
    if re.search(r"\bJOIN\b", sql, re.IGNORECASE):
        return None, None
    if re.search(r"\bFROM\s*\(", sql, re.IGNORECASE):
        return None, None

    matches = re.findall(
        r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
        sql,
        flags=re.IGNORECASE,
    )
    if len(matches) != 1:
        return None, None

    table, alias = matches[0]
    if alias and alias.lower() in {"where", "group", "order", "having"}:
        alias = None
    return table.upper(), alias


def _find_invalid_unqualified_columns(sql: str, schema_dict: dict) -> list:
    """Finds invalid bare identifiers in simple one-table SQL.

    Example caught:
      SELECT salary_category, COUNT(*) FROM employees GROUP BY salary_category

    Also now catches the same kind of bare hallucinated column when the
    whole thing is wrapped in the app's own ROWNUM pagination pattern
    (see _unwrap_simple_rownum_pagination) -- e.g.
      SELECT * FROM (SELECT category, SUM(sale_amount) AS total_sales
                      FROM commercial_orders GROUP BY category) WHERE ROWNUM <= 200
    where "category"/"sale_amount" are bare columns on commercial_orders,
    not on some outer computed result.
    """
    table, _ = _extract_single_table_context(sql)
    if not table or table not in schema_dict:
        return []

    table_cols = schema_dict[table]
    # Same unwrap _extract_single_table_context used to find `table` --
    # applied here too so the SELECT/GROUP BY/WHERE regexes below see the
    # real inner clauses instead of the wrapper's own dummy "SELECT *"
    # and "WHERE ROWNUM <= N" (which is what `table` was actually
    # resolved against, so this must match it).
    analysis_sql = _unwrap_simple_rownum_pagination(sql)
    cleaned = re.sub(r"'[^']*'", "''", analysis_sql)

    invalid = []
    seen = set()

    select_match = re.search(
        r"\bSELECT\b\s+(.*?)\s+\bFROM\b",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if select_match:
        for expr in _split_top_level_commas(select_match.group(1)):
            candidate = re.sub(
                r"\s+AS\s+[A-Za-z_][A-Za-z0-9_]*$",
                "",
                expr,
                flags=re.IGNORECASE,
            ).strip()
            if "." in candidate:
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", candidate):
                col = candidate.upper()
                if col not in table_cols and col not in {"ROWNUM", "DUAL"}:
                    key = (table, col)
                    if key not in seen:
                        invalid.append(key)
                        seen.add(key)
                continue
            # A bare column used as the sole argument of a simple aggregate
            # (e.g. "SUM(sale_amount)", "AVG(price)") was previously
            # invisible to this check -- the block above only looks at
            # select items that are themselves a plain identifier, and
            # "SUM(sale_amount)" contains parens so it never matched that
            # pattern. COUNT(*) is naturally excluded since "*" isn't a
            # valid identifier character.
            agg_match = re.match(
                r"^(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$",
                candidate,
                flags=re.IGNORECASE,
            )
            if agg_match:
                col = agg_match.group(1).upper()
                if col not in table_cols and col not in {"ROWNUM", "DUAL"}:
                    key = (table, col)
                    if key not in seen:
                        invalid.append(key)
                        seen.add(key)

        group_match = re.search(
        r"\bGROUP\s+BY\b\s+(.*?)(?:\bHAVING\b|\bORDER\s+BY\b|$)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if group_match:
        for expr in _split_top_level_commas(group_match.group(1)):
            candidate = expr.strip()
            if "." in candidate:
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", candidate):
                col = candidate.upper()
                if col not in table_cols and col not in {"ROWNUM", "ASC", "DESC"}:
                    key = (table, col)
                    if key not in seen:
                        invalid.append(key)
                        seen.add(key)

    # WHERE clause was previously unchecked, so hallucinated columns used
    # only in a filter (e.g. "WHERE first_name = 'Ali' AND last_name =
    # 'Khan'" on a table that only has customer_name) sailed straight
    # through to Oracle as an ORA-00904 instead of being caught here.
    where_match = re.search(
        r"\bWHERE\b\s+(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|$)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        sql_keywords = {
            "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "BETWEEN",
            "EXISTS", "ANY", "ALL", "ROWNUM", "SYSDATE", "DUAL", "TRUE", "FALSE",
        }
        # An identifier counts as a column reference only when it's used
        # like one: directly before a comparison/keyword operator, or
        # directly after AND/OR/WHERE/NOT — and NOT immediately followed
        # by "(" (that's a function call like TRUNC(...) or COUNT(...)).
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", where_match.group(1)):
            candidate = m.group(1)
            col = candidate.upper()
            if col in sql_keywords or col in table_cols:
                continue
            start, end = m.span()
            before = where_match.group(1)[:start]
            after = where_match.group(1)[end:]
            if "." in candidate:
                continue
            if re.match(r"\s*\(", after):
                continue  # function call, not a column
            if re.match(r"^[.\s]*\d", after) or before.rstrip().endswith("."):
                continue
            preceded_ok = bool(re.search(
                r"(?:\bAND\b|\bOR\b|\bNOT\b|\bWHERE\b)\s*$", before, re.IGNORECASE
            )) or before.strip() == ""
            followed_ok = bool(re.match(
                r"\s*(=|<>|!=|<=|>=|<|>|\bIS\b|\bIN\b|\bLIKE\b|\bBETWEEN\b)",
                after, re.IGNORECASE
            ))
            if preceded_ok and followed_ok:
                key = (table, col)
                if key not in seen:
                    invalid.append(key)
                    seen.add(key)

    return invalid


def find_union_shape_mismatch(sql: str, schema_dict: dict) -> Optional[str]:
    """Guards against the model literally UNION-ing raw '*' data from
    tables that have different numbers of columns (e.g. 'SELECT * FROM
    countries UNION ALL SELECT * FROM employees') — invalid Oracle SQL
    that raises ORA-01789, and the exact failure mode a small model falls
    into when it confuses a 'give me every table' request with the
    row-count-per-table UNION ALL pattern from the prompt.

    Only validates the simple, expected branch shape ('SELECT <cols>
    FROM TABLE', no JOIN/WHERE/subquery per branch); anything more
    complex is left alone rather than risking a false positive. Returns
    a human-readable reason if a mismatch is found, else None.
    """
    if not re.search(r"\bUNION\b", sql, re.IGNORECASE):
        return None

    branches = re.split(r"\bUNION\s+ALL\b|\bUNION\b", sql, flags=re.IGNORECASE)
    shapes = []  # (column_count, table_name) per branch
    for branch in branches:
        branch = branch.strip()
        match = re.match(
            r"^SELECT\s+(.*?)\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            branch, re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None  # branch too complex to safely check — don't block it

        cols_part, table = match.group(1).strip(), match.group(2).upper()
        if cols_part == "*":
            table_cols = schema_dict.get(table)
            if not table_cols:
                return None
            shapes.append((len(table_cols), table))
        else:
            shapes.append((_count_top_level_commas(cols_part), table))

    distinct_counts = {count for count, _ in shapes}
    if len(distinct_counts) > 1:
        tables = ", ".join(f"{t} ({c} cols)" for c, t in shapes)
        return (
            f"the branches select different numbers of columns from "
            f"different tables ({tables}), which Oracle can't UNION together"
        )
    return None


def strip_sql_comments(sql: str) -> str:
    """Removes '--' line comments and '/* ... */' block comments.

    Small local models routinely annotate their SQL with comments like
    '-- Assuming department_id 10 is the target department' even when
    told not to. is_safe_select() correctly treats any '--'/'/*'/'*/' as
    an injection red flag (a real attacker could use '--' to comment out
    the rest of a query) and rejects the whole statement outright -- with
    no repair attempt, since that check runs after all the retry logic.
    Stripping comments here, before that check ever runs, removes the
    injection surface (the comment content is gone, not just hidden) while
    also preventing an otherwise-correct query from being thrown away
    for cosmetic reasons. Deliberately naive about string literals (Oracle
    string literals are rare in this app's generated SQL and a comment
    marker inside one is vanishingly unlikely) -- correctness of the
    common case matters more here than perfect SQL-string awareness.
    """
    # Block comments first so a '--' that happens to sit inside a /* */
    # block doesn't get treated as a separate line comment afterwards.
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", "", sql)
    return sql


def _find_matching_paren(sql: str, open_idx: int) -> int:
    """Returns the index of the ')' that closes the '(' at open_idx, or -1."""
    depth = 0
    for i in range(open_idx, len(sql)):
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _top_level_split(s: str, sep: str = ",") -> list:
    """Splits on sep, but only at paren-depth 0 -- so a column list like
    'salary, ROW_NUMBER() OVER (PARTITION BY dept, ORDER BY salary) AS rn'
    doesn't get sliced apart at the comma inside the OVER(...) clause."""
    parts, depth, current = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _extract_select_list_columns(select_body: str) -> Optional[set]:
    """Best-effort set of column/alias names a 'SELECT <list> FROM ...'
    body actually projects. Returns None (meaning "can't say safely")
    for anything too ambiguous to resolve without real risk of a false
    positive: SELECT *, a nested WITH, or any list item that isn't a
    plain 'expr AS alias', 'table.column', or bare 'column'."""
    m = re.match(r"\s*SELECT\s+(.*?)\s+FROM\s", select_body, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    col_list = m.group(1).strip()
    if col_list == "*" or re.match(r"^\s*WITH\b", select_body, re.IGNORECASE):
        return None

    columns = set()
    for part in _top_level_split(col_list):
        part = part.strip()
        if not part:
            continue
        alias_m = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", part, re.IGNORECASE)
        if alias_m:
            columns.add(alias_m.group(1).upper())
            continue
        token = part.split(".")[-1].strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
            columns.add(token.upper())
        else:
            # A function call or expression with no alias -- can't name
            # it confidently, and its presence means the rest of this
            # list might be misparsed too. Bail out for the whole CTE
            # rather than risk a false "column doesn't exist" report.
            return None
    return columns or None


def _extract_cte_definitions(sql: str) -> dict:
    """Parses top-level 'WITH name AS (SELECT ...), name2 AS (...)' and
    returns {CTE_NAME: {COLUMN, ...}} for every CTE whose own column list
    could be resolved confidently by _extract_select_list_columns. CTEs
    that select '*' or nest another WITH are simply omitted (not flagged
    as violations) -- see that function's docstring for why."""
    ctes: dict = {}
    m = re.search(r"\bWITH\s+", sql, re.IGNORECASE)
    if not m:
        return ctes
    pos = m.end()
    while True:
        name_m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql[pos:], re.IGNORECASE)
        if not name_m:
            break
        cte_name = name_m.group(1).upper()
        open_idx = pos + name_m.end() - 1
        close_idx = _find_matching_paren(sql, open_idx)
        if close_idx == -1:
            break
        body = sql[open_idx + 1 : close_idx]
        cols = _extract_select_list_columns(body)
        if cols is not None:
            ctes[cte_name] = cols
        pos = close_idx + 1
        comma_m = re.match(r"\s*,\s*", sql[pos:])
        if not comma_m:
            break
        pos += comma_m.end()
    return ctes


def find_cte_scope_violation(sql: str):
    """Guards against the model referencing alias.column where alias
    refers to a CTE that never actually projected that column -- the
    exact failure behind a live ORA-00904 seen in production: a CTE
    exposed DEPARTMENT_NAME (having already joined DEPARTMENTS once),
    but the OUTER query re-joined DEPARTMENTS again and pulled
    e.DEPARTMENT_ID through the CTE's alias, even though the CTE never
    selected DEPARTMENT_ID at all. Oracle only catches this at query
    execution time; this catches the same class of mistake statically,
    before the query ever reaches the database, so it can go through the
    same self-correction retry as a hallucinated base-table column.
    Only fires on CTEs whose column list could be parsed with
    confidence -- see _extract_select_list_columns -- so this can only
    under-report, never falsely flag a valid query.
    Returns (cte_name, alias, bad_column, real_columns) or None.
    """
    ctes = _extract_cte_definitions(sql)
    if not ctes:
        return None
    for cte_name, real_cols in ctes.items():
        for ref_m in re.finditer(
            rf"\b(?:FROM|JOIN)\s+{re.escape(cte_name)}\s+([A-Za-z_][A-Za-z0-9_]*)\b",
            sql, re.IGNORECASE,
        ):
            alias = ref_m.group(1)
            if alias.upper() in ("WHERE", "ON", "GROUP", "ORDER", "PARTITION"):
                continue
            for col_m in re.finditer(
                rf"\b{re.escape(alias)}\.([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE
            ):
                col = col_m.group(1).upper()
                if col not in real_cols:
                    return (cte_name, alias, col, sorted(real_cols))
    return None


def find_unaliased_subquery_requalification(sql: str):
    """Guards against the model wrapping a derived table with NO alias
    (e.g. "SELECT e.employee_id, ..., j.job_title FROM (SELECT ...
    FROM employees e JOIN departments d ON ... JOIN jobs j ON ...)
    WHERE rn = 1") and then re-qualifying columns in the outer SELECT/
    WHERE with the INNER query's table aliases (e., d., j.) -- those
    aliases don't exist outside the subquery's own scope, so Oracle
    raises ORA-00904 ("<alias>"."<column>": invalid identifier) at
    query time. This is a different failure than
    find_cte_scope_violation (which only handles named WITH-CTEs) --
    this covers the same mistake for a plain unnamed derived table.

    Only fires when a FROM(...) subquery is followed by nothing that
    looks like an alias (WHERE / GROUP BY / ORDER BY / end of that
    paren level / another closing paren) AND an alias defined INSIDE
    that subquery is referenced again outside it -- so this can only
    under-report (skip real mistakes it can't parse confidently),
    never falsely flag SQL that legitimately aliased its subquery.

    Returns (alias, column, subquery_snippet) or None.
    """
    for m in re.finditer(r"\bFROM\s*\(", sql, re.IGNORECASE):
        open_idx = m.end() - 1
        close_idx = _find_matching_paren(sql, open_idx)
        if close_idx == -1:
            continue
        body = sql[open_idx + 1 : close_idx]

        after = sql[close_idx + 1 :]
        after_stripped = after.lstrip()
        has_alias = bool(
            re.match(r"(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*\b", after_stripped, re.IGNORECASE)
            and not re.match(
                r"(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|UNION|\)|$)",
                after_stripped,
                re.IGNORECASE,
            )
        )
        if has_alias:
            continue  # properly aliased, nothing to flag here

        # Aliases the inner subquery itself defined via FROM/JOIN.
        inner_aliases = set()
        for am in re.finditer(
            r"\b(?:FROM|JOIN)\s+[A-Za-z_][A-Za-z0-9_]*\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
            body, re.IGNORECASE,
        ):
            alias = am.group(1)
            if alias.upper() not in ("WHERE", "ON", "GROUP", "ORDER", "PARTITION", "JOIN"):
                inner_aliases.add(alias.lower())
        if not inner_aliases:
            continue

        # Does anything OUTSIDE the subquery (before or after it) still
        # reference one of those now-out-of-scope aliases?
        outside_text = sql[: m.start()] + sql[close_idx + 1 :]
        for alias in inner_aliases:
            col_m = re.search(rf"\b{re.escape(alias)}\.([A-Za-z_][A-Za-z0-9_]*)", outside_text, re.IGNORECASE)
            if col_m:
                return (alias, col_m.group(1), body.strip()[:200])
    return None


# SQL keywords common enough that any real SQL line is very likely to
# contain at least one of them (or the punctuation checked alongside it).
_SQL_LOOKS_LIKE_RE = re.compile(
    r"\b(SELECT|FROM|WHERE|GROUP|ORDER|JOIN|ON|AND|OR|ROWNUM|UNION|WITH|"
    r"AS|BY|HAVING|CASE|WHEN|THEN|ELSE|END|OVER|PARTITION|DISTINCT|NULL)\b",
    re.IGNORECASE,
)
# Punctuation that shows up in real SQL but essentially never ends a plain
# English sentence -- deliberately excludes "." and "," since those are
# exactly what closes chatter like "Done." or "Wait,".
_SQL_PUNCTUATION_RE = re.compile(r"[()=<>*']")


def _truncate_trailing_chatter(sql: str) -> str:
    """Some models (larger/chattier ones especially -- this was first seen
    switching from the terse local 3B model to a bigger hosted one) add a
    stray conversational line or two AFTER an otherwise-complete, correct
    SQL statement, e.g.:

        SELECT 'Could not understand the question' AS message FROM dual
        Done.
        Wait,

    Neither "Done." nor "Wait," is SQL, but nothing previously stripped
    them, so they rode along to Oracle and broke the statement with
    ORA-00933 (SQL command not properly ended) -- even though the actual
    SQL on the first line was completely valid on its own.

    This looks for the first line, after the SQL has already started, that
    contains neither a SQL keyword nor SQL-ish punctuation, and cuts the
    text there. It only ever removes trailing lines -- it can't affect a
    normal single-line query (the overwhelming majority of this app's
    output), since there's nothing after the first line to look at.
    """
    lines = sql.splitlines()
    kept = []
    sql_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if sql_seen:
                kept.append(line)
            continue
        looks_like_sql = bool(
            _SQL_LOOKS_LIKE_RE.search(stripped)
            or _SQL_PUNCTUATION_RE.search(stripped)
        )
        if sql_seen and not looks_like_sql:
            break  # first non-SQL-looking line after real SQL -> stop here
        kept.append(line)
        if looks_like_sql:
            sql_seen = True
    truncated = "\n".join(kept).strip()
    return truncated if truncated else sql


# A small local model occasionally ignores its system prompt entirely and
# answers a greeting/off-topic message conversationally instead of with
# SQL -- e.g. the user says "hello" and it replies "Hello! How can I
# assist you today?". Every downstream step in this file (ensure_default_
# row_cap, enforce_correct_row_limit's ROWNUM-wrap fallback, etc.) assumes
# whatever clean_sql() returns IS a real SQL statement and is free to wrap
# it in "SELECT * FROM (<this>) WHERE ROWNUM <= N" -- which turns plain
# prose into something that syntactically starts with SELECT (so it slips
# past is_safe_select()'s allow-list check) but is not valid SQL at all,
# and blows up against Oracle as ORA-00907 ("missing right parenthesis")
# once it actually reaches the database. This must be caught HERE, before
# any of that wrapping logic ever runs on it.
_SQL_LEADING_KEYWORDS_RE = re.compile(
    r"^(select|with|insert|update|delete|drop|truncate|alter|merge|grant|revoke|create)\b",
    re.IGNORECASE,
)

# A conservative fallback shown when the model's reply itself is empty
# after cleanup (e.g. it returned nothing usable at all).
_DEFAULT_OFF_TOPIC_MESSAGE = (
    "I can only help with questions about this database (employees, "
    "departments, salaries, orders, etc.). Try asking something like "
    "\"show all employees\" or \"top 10 highest paid employees\"."
)


def _looks_like_sql_statement(text: str) -> bool:
    """True only if `text` could plausibly BE (the start of) a SQL
    statement -- a SELECT/WITH query, or a write-intent statement (which
    the existing is_safe_select() forbidden-keyword check further down
    the pipeline is responsible for rejecting -- that behavior is
    unchanged by this function). False for ordinary conversational
    prose, which never legitimately starts with any of these keywords.
    """
    return bool(_SQL_LEADING_KEYWORDS_RE.match(text.strip()))


def _as_safe_message_sql(text: str) -> str:
    """Wraps arbitrary, non-SQL model text as a single-row literal
    'SELECT '<text>' AS message FROM dual' -- valid, harmless, read-only
    SQL that the app already knows how to render as a normal one-row
    result (this mirrors the existing read-only-refusal message
    elsewhere in the app, e.g. query_service.py's write-intent guard).
    Single quotes are doubled per Oracle string-literal escaping, and the
    text is capped at a sane length so a runaway model reply can't
    produce an unreasonably large literal.
    """
    literal = text.strip()
    if not literal:
        literal = _DEFAULT_OFF_TOPIC_MESSAGE
    if len(literal) > 500:
        literal = literal[:497] + "..."
    literal = literal.replace("'", "''")
    return f"SELECT '{literal}' AS message FROM dual"


def clean_sql(raw_sql: str) -> str:
    sql = raw_sql.strip()
    sql = re.sub(r"^```sql", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"^```", "", sql).strip()
    sql = re.sub(r"```$", "", sql).strip()
    sql = _truncate_trailing_chatter(sql)
    sql = strip_sql_comments(sql)
    sql = sql.rstrip(";").strip()

    if sql.upper().startswith("CLARIFY:"):
        return sql

    # Catch non-SQL model chatter BEFORE any of the SQL-repair helpers
    # below get a chance to treat it as a subquery to wrap -- see the
    # comment above _looks_like_sql_statement for why this must happen
    # here, first.
    if not _looks_like_sql_statement(sql):
        return _as_safe_message_sql(sql)

    sql = _dedupe_top_level_selects(sql)
    sql = enforce_oracle11g_syntax(sql)
    sql = fix_rownum_after_order_by(sql)
    sql = fix_not_in_subquery(sql)
    sql = simplify_unnecessary_rownum_wrapper(sql)
    return sql


def is_safe_select(sql: str) -> bool:
    """True only for a single, read-only SELECT statement. This is the
    last line of defense before generated SQL ever reaches the database."""
    stripped = sql.strip()
    if not stripped:
        return False
    lowered = stripped.lower()

    if ";" in stripped:
        return False
    if "--" in stripped or "/*" in stripped or "*/" in stripped:
        return False
    starts_with_select = lowered.startswith("select")
    starts_with_with = lowered.startswith("with ")
    if not starts_with_select and not starts_with_with:
        return False
    if starts_with_with and not re.search(r"\bselect\b", lowered):
        return False
    for word in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{word}\b", lowered):
            return False
    if _PACKAGE_CALL_RE.search(lowered):
        return False
    return True