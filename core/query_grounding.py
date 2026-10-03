"""
query_grounding.py
------------------
Small deterministic trust boundary around the LLM SQL pipeline.

It does NOT generate SQL. It only:
1) asks the local model for a tiny structured preflight decision when a
   question looks relationship-heavy/ambiguous;
2) validates that any JOIN emitted by the SQL model follows real Oracle FK
   relationships and uses real columns.

The module is deliberately fail-open for the preflight model call: if the
local model is unavailable or returns malformed JSON, the existing SQL flow
continues exactly as before. SQL JOIN validation is fail-closed because it
runs after SQL generation and before execution.
"""

import json
import re
from typing import Any, Optional

from config import config
from integrations import llm


# Questions containing these concepts are more likely to require a JOIN or
# an interpretation of a business entity. We do NOT run the extra LLM call on
# every question, which keeps simple SELECT/ranking questions fast.
_RELATIONSHIP_TERMS = {
    "category", "categories", "product", "products", "sale", "sales",
    "order", "orders", "customer", "customers", "department", "departments",
    "location", "locations", "country", "countries", "region", "regions",
    "job", "jobs", "manager", "managers", "employee", "employees",
    "each", "every", "per", "compare", "between", "along",
    "related", "belong", "belongs",
}


# -------------------------------------------------------------------------
# BUSINESS METRIC DEFAULTS
# -------------------------------------------------------------------------
# Common user/business language -> preferred real schema column names.
# These are NOT blindly trusted. resolve_business_metric() only returns a
# mapping after confirming that the column actually exists in the live schema.
BUSINESS_METRIC_MAPPING = {
    "total sales": ["TOTAL_AMOUNT", "SALES_AMOUNT", "AMOUNT", "REVENUE"],
    "sales amount": ["TOTAL_AMOUNT", "SALES_AMOUNT", "AMOUNT"],
    "sales": ["TOTAL_AMOUNT", "SALES_AMOUNT", "AMOUNT", "REVENUE"],
    "revenue": ["REVENUE", "TOTAL_AMOUNT", "SALES_AMOUNT", "AMOUNT"],

    "units sold": ["QUANTITY"],
    "most units": ["QUANTITY"],
    "least units": ["QUANTITY"],
    "units": ["QUANTITY"],
    "products sold": ["QUANTITY"],
    "items sold": ["QUANTITY"],
    "quantity sold": ["QUANTITY"],

    "subtotal": ["SUBTOTAL"],
}

# Group/entity words used only to prefer a table that contains the same
# grouping dimension as the question. This keeps the resolver generic while
# avoiding a random TOTAL_AMOUNT column from an unrelated table.
_BUSINESS_ENTITY_HINTS = {
    "category": "CATEGORY",
    "categories": "CATEGORY",
    "city": "CITY",
    "cities": "CITY",
    "country": "COUNTRY",
    "countries": "COUNTRY",
    "province": "PROVINCE",
    "provinces": "PROVINCE",
    "customer": "CUSTOMER_ID",
    "customers": "CUSTOMER_ID",
    "product": "PRODUCT_ID",
    "products": "PRODUCT_ID",
    "employee": "EMPLOYEE_ID",
    "employees": "EMPLOYEE_ID",
    "department": "DEPARTMENT_ID",
    "departments": "DEPARTMENT_ID",
}


def _plural_tolerant_pattern(term: str) -> str:
    """Build a regex fragment that matches `term` whether the user typed it
    singular or plural (e.g. "sale" or "sales", "total sale" or "total
    sales"). Only the LAST word's trailing 's' is treated as optional —
    that's the only part of these short business phrases that ever varies
    with singular/plural in practice.

    Without this, a literal dict-key match like BUSINESS_METRIC_MAPPING
    silently failed for perfectly reasonable phrasing (e.g. "sale" instead
    of "sales"), and resolve_business_metric()/resolve_claim_dimension()
    returning None then fell through to a far less reliable free-form LLM
    call for something the schema could answer deterministically.
    """
    if term.endswith("s"):
        return re.escape(term[:-1]) + "s?"
    return re.escape(term) + "s?"


def resolve_business_metric(question: str, schema_dict: dict) -> Optional[dict]:
    """Resolve a business term to one verified schema column.

    Example:
        "Mobile category has the highest sales"
            -> COMMERCIAL_ORDERS.TOTAL_AMOUNT

    The function never invents a column. If more than one equally good
    schema match exists, it returns None and lets the normal clarification
    rules handle the ambiguity.
    """
    q = _normalise(question)
    if not q or not schema_dict:
        return None

    # Longest phrase first so "total sales" wins before plain "sales".
    metric_terms = sorted(BUSINESS_METRIC_MAPPING, key=len, reverse=True)

    entity_hints = {
        column
        for word, column in _BUSINESS_ENTITY_HINTS.items()
        if re.search(rf"\b{re.escape(word)}\b", q, re.IGNORECASE)
    }

    for metric_term in metric_terms:
        if not re.search(rf"\b{_plural_tolerant_pattern(metric_term)}\b", q, re.IGNORECASE):
            continue

        preferred_columns = BUSINESS_METRIC_MAPPING[metric_term]
        candidates = []

        for preference_rank, preferred_col in enumerate(preferred_columns):
            for table, columns in schema_dict.items():
                upper_columns = {str(c).upper() for c in columns}
                if preferred_col not in upper_columns:
                    continue

                # Prefer a table that also contains the entity/grouping
                # mentioned by the user (e.g. CATEGORY + TOTAL_AMOUNT).
                entity_score = 1 if entity_hints and any(
                    hint in upper_columns for hint in entity_hints
                ) else 0

                candidates.append((
                    preference_rank,
                    -entity_score,
                    str(table).upper(),
                    preferred_col,
                ))

        if not candidates:
            continue

        candidates.sort()
        best = candidates[0]
        equally_good = [
            item for item in candidates
            if item[0] == best[0] and item[1] == best[1]
        ]

        # If two unrelated tables are equally good, do not guess.
        unique_pairs = {(item[2], item[3]) for item in equally_good}
        if len(unique_pairs) != 1:
            return None

        return {
            "metric_term": metric_term,
            "column": best[3],
            "table": best[2],
        }

    return None


def resolve_claim_dimension(question: str, schema_dict: dict, preferred_table: Optional[str] = None) -> Optional[dict]:
    """Resolve the grouping/entity dimension of a claim to a verified column.

    Example:
        "Mobile category has the highest sales"
            -> COMMERCIAL_ORDERS.CATEGORY

    The resolver only returns a result when the dimension column exists in the
    live schema. If ``preferred_table`` is supplied (normally the table chosen
    by metric resolution), that table is preferred so a single-table claim can
    be verified deterministically without asking the LLM to rediscover obvious
    schema facts.
    """
    q = _normalise(question)
    if not q or not schema_dict:
        return None

    matched = []
    for word, column in _BUSINESS_ENTITY_HINTS.items():
        if re.search(rf"\b{re.escape(word)}\b", q, re.IGNORECASE):
            matched.append((word, column))

    if not matched:
        return None

    preferred_upper = str(preferred_table).upper() if preferred_table else None
    candidates = []
    for word, column in matched:
        for table, columns in schema_dict.items():
            upper_columns = {str(c).upper() for c in columns}
            if column.upper() not in upper_columns:
                continue
            table_upper = str(table).upper()
            score = 0 if preferred_upper and table_upper == preferred_upper else 1
            candidates.append((score, table_upper, column.upper(), word))

    if not candidates:
        return None

    candidates.sort()
    best = candidates[0]
    equally_good = [
        item for item in candidates
        if item[0] == best[0] and item[1] == best[1] and item[2] == best[2]
    ]

    return {
        "table": best[1],
        "column": best[2],
        "term": best[3],
    }


def extract_claim_candidate_value(question: str) -> Optional[str]:
    """Return the asserted candidate value as text for simple claims.

    This intentionally reuses the same candidate extraction used by the SQL
    safety guard. It does not infer database values; it only preserves the
    user's asserted candidate so the final SQL can compare the independently
    calculated winner with that candidate.
    """
    tokens = _extract_claim_candidate_tokens(question)
    if not tokens:
        return None
    return " ".join(tokens).strip() or None


def is_simple_superlative_claim(question: str) -> bool:
    """True for one-candidate highest/lowest style claims.

    Direct comparisons such as A vs B intentionally return False because they
    need a different deterministic SQL shape.
    """
    q = _normalise(question)
    if not q or not looks_like_claim_verification(q):
        return False
    if any(re.search(p, q, re.IGNORECASE) for p in _CLAIM_COMPARISON_PATTERNS):
        return False
    return any(re.search(p, q, re.IGNORECASE) for p in _CLAIM_SUPERLATIVE_PATTERNS)


def claim_sort_direction(question: str) -> Optional[str]:
    """Resolve highest/most vs lowest/least wording to Oracle sort direction.
    Returns None when the direction is not explicit.
    """
    q = _normalise(question)
    low_patterns = (
        r"\blowest\b", r"\bleast\b", r"\bsmallest\b",
        r"\bsabse\s+kam\b", r"\bsab\s+se\s+kam\b",
    )
    high_patterns = (
        r"\bhighest\b", r"\bmost\b", r"\blargest\b",
        r"\bgreatest\b", r"\bsabse\s+(?:zyada|ziada|zyaada|zaida)\b",
        r"\bsab\s+se\s+(?:zyada|ziada|zyaada|zaida)\b",
    )
    if any(re.search(p, q, re.IGNORECASE) for p in low_patterns):
        return "ASC"
    if any(re.search(p, q, re.IGNORECASE) for p in high_patterns):
        return "DESC"
    return None


_SQL_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO|DELETE\s+FROM)\s+([A-Za-z_][A-Za-z0-9_$#]*)",
    re.IGNORECASE,
)
_JOIN_RE = re.compile(
    r"\bJOIN\s+([A-Za-z_][A-Za-z0-9_$#]*)"
    r"(?:\s+(?:AS\s+)?((?!(?:ON|WHERE|GROUP|ORDER|HAVING|UNION)\b)[A-Za-z_][A-Za-z0-9_$#]*))?"
    r"\s+ON\s+(.+?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bUNION\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_QUALIFIED_REF_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_$#]*)\.([A-Za-z_][A-Za-z0-9_$#]*)\b"
)


def _normalise(question: str) -> str:
    return " ".join((question or "").strip().lower().split())




_CLAIM_SUPERLATIVE_PATTERNS = (
    r"\bhas\s+(?:the\s+)?(?:highest|most|largest|greatest|lowest|least|smallest)\b",
    r"\bis\s+(?:the\s+)?(?:highest|most|largest|greatest|lowest|least|smallest)\b",
    r"\b(?:is|are)\s+(?:top|number\s*one|#?1)\b",
    r"\b(?:sold|generated|earned|made|has|had)\s+(?:the\s+)?(?:most|least)\b",
    r"\b(?:sabse\s+(?:zyada|ziada|kam|zyaada|zaida)|sab\s+se\s+(?:zyada|ziada|zyaada|zaida|kam))(?:\s+[^,.!?]{0,80})?(?:\s+(?:hai|hain|karta|karti|karte|hota|hoti|hotay))?\b",
    r"\b(?:highest|lowest|most|least)\s+(?:sales|salary|revenue|count|number|employees|orders)\s+(?:is|belongs\s+to|belongs\s+with)\b",
)
_CLAIM_COMPARISON_PATTERNS = (
    r"\bmore\s+than\b",
    r"\bless\s+than\b",
    r"\bfewer\s+than\b",
    r"\bgreater\s+than\b",
    r"\bless\s+than\b",
    # NOT "sab se zyada"/"sab se zaida" -- that's the superlative idiom
    # ("more than ALL" = "the most"), handled by _CLAIM_SUPERLATIVE_PATTERNS,
    # not a pairwise A-vs-B comparison. Without this exclusion, "kis
    # category ki sales sab se zaida hui hai" ("which category's sales are
    # highest") was wrongly flagged as a comparison, which disabled the
    # deterministic single-table claim-verification path further downstream
    # and fell back to a much less reliable free-form LLM call.
    r"(?<!sab )\bse\s+(?:zyada|ziada|zyaada|zaida|kam)\b",
)



# WH-question words ("which", "who", "what", "kis", "kaunsa"...). A question
# built around one of these is always asking the database to FIND the
# answer -- it never asserts a named winner, no matter what else the
# sentence contains (e.g. a row count: "kis 2 category ne sab se zaida
# sales ki hai" = "which 2 categories had the most sales"). This must be
# checked ANYWHERE in the question, not just at the start, because Roman
# Urdu word order often puts the WH-word in the middle
# ("commercial_orders mei 4 sab se zaida sale KIS category ki hui hai").
_WH_QUESTION_WORDS = (
    "which", "who", "whom", "what",
    "kis", "kise", "kaun", "kaunsa", "kaunsi", "kaunse",
    "konsa", "konsi", "konse",
)


def _has_wh_question_word(q: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", q, re.IGNORECASE) for w in _WH_QUESTION_WORDS)


def looks_like_claim_verification(question: str) -> bool:
    """Detect statements/claims that must be independently checked against
    the database instead of turning the asserted value into a WHERE filter.

    This is deliberately narrower than ordinary ranking detection: "show
    top 5 sales" is a normal retrieval request, while "Mobile category has
    the highest sales" asserts that Mobile is the winner.
    """
    q = _normalise(question)
    if not q:
        return False
    has_superlative = any(re.search(p, q, re.IGNORECASE) for p in _CLAIM_SUPERLATIVE_PATTERNS)
    has_comparison = any(re.search(p, q, re.IGNORECASE) for p in _CLAIM_COMPARISON_PATTERNS)
    if not (has_superlative or has_comparison):
        return False

    # A WH-question ("which category ...", "kis category ne ...", "... kis
    # category ki hui hai") asks the database to FIND the winner -- it is
    # never an assertion about a specific named winner, even if it also
    # contains a row count/ordinal like "kis 2 category" or "4 sab se
    # zaida ... kis category". Checking this BEFORE the candidate-token
    # extraction below matters: "kis" ("which") is short and not covered
    # by the retrieval-imperative regex further down (which only matches
    # English/Urdu verbs at the START of the sentence), so it was being
    # picked up by _extract_claim_candidate_tokens as if the user had
    # named "kis" as their claimed candidate -- silently routing a plain
    # top-N/ordinal retrieval question through the claim-verification SQL
    # shape, which always computes exactly one winner (rank 1) and ignores
    # the requested N.
    if _has_wh_question_word(q):
        return False

    # A generic question such as "Who is the highest paid employee?" asks
    # the database to FIND the winner; it does not assert who the winner is.
    # Claim mode is only for a named candidate/value that the user asserts.
    candidate_tokens = _extract_claim_candidate_tokens(q)
    if has_superlative and not candidate_tokens:
        return False

    # Imperative retrieval wording such as "show the highest paid employee"
    # is a query, not a claim about a named candidate.
    if re.match(r"^(show|list|display|find|give|tell|dikhao|batao)\b", q):
        return False
    # Bare "mujhe" only signals an imperative retrieval request when it is
    # immediately followed by a command verb, e.g. "mujhe dikhao"/"mujhe
    # batao" (show me / tell me). "mujhe lagta/lagtha hai ..." ("I think
    # ...") is the opposite: it's how a claim is normally hedged in Roman
    # Urdu/Hindi. Treating any sentence starting with "mujhe" as a command
    # was silently disqualifying every claim phrased that way -- e.g.
    # "mujhe lagtha hai k Mobile category ki sab se zaida sales hui hai"
    # never reached the claim-verification path at all, and fell back to
    # plain ranking, which then had no row-count to work with and asked an
    # unnecessary clarification instead of just verifying the claim.
    if re.match(r"^mujhe\s+(?:dikhao|batao|de\s+do|bata\s+do|chahiye)\b", q):
        return False
    return True


def build_claim_verification_prompt(
    question: str,
    schema_text: str,
    relationship_text: str,
    resolved_metric: Optional[dict] = None,
) -> str:
    metric_hint = ""
    if resolved_metric:
        metric_hint = f"""
VERIFIED BUSINESS METRIC RESOLUTION:
User term: {resolved_metric["metric_term"]}
Verified column: {resolved_metric["table"]}.{resolved_metric["column"]}
This mapping was checked against the live schema. Use this metric for the
claim verification and do NOT ask a clarification question about this metric
unless another required part of the request is genuinely ambiguous.
"""

    return f"""You are a verification SQL planner for an Oracle 11g database.
The user's wording may contain an UNTRUSTED CLAIM. Never assume that the
claimed person/product/category/department/etc. is actually the winner.
Your job is to independently calculate the database result and compare it
with the user's claim.

USER CLAIM:
{question}

{metric_hint}
VERIFIED SCHEMA:
{schema_text}

VERIFIED FOREIGN-KEY RELATIONSHIPS:
{relationship_text or '(none)'}

NON-NEGOTIABLE CLAIM VERIFICATION RULES:
1. Treat every factual assertion in the user message as unverified input.
2. For a claim like "Mobile category has the highest sales", DO NOT use
   Mobile as the WHERE condition that determines the winner. First calculate
   sales for ALL categories that can be represented by the schema, rank them,
   then compare the actual winner with Mobile.
3. For "John is the highest paid employee", calculate the highest-paid
   employee independently, then compare the winner with John.
4. For direct comparisons like "IT has more employees than HR", calculate
   the requested metric for both named values and compare them; filtering to
   those explicitly named values is allowed because they are the two sides
   being compared, not an assumed winner.
5. If the metric, entity, value, or relationship needed to verify the claim
   cannot be resolved unambiguously from the schema, reply exactly:
   CLARIFY: <one short, specific question>.
6. JOINs may use ONLY the exact FK relationships above. Never infer a JOIN
   from similar column names.
7. Return ONE read-only SELECT statement only. CTEs are allowed.
8. The result must expose enough information to verify the claim, preferably
   the claimed value, the actual winner/result, and a TRUE/FALSE status.
9. Oracle 11g only: no LIMIT, OFFSET, FETCH FIRST, or FETCH PERCENT.
10. Never modify data.

Return ONLY SQL or CLARIFY: ..."""


def _extract_claim_candidate_tokens(question: str) -> list:
    """Best-effort extraction of the named subject/value in common claim
    wording. This is NOT used to invent SQL; it is only a safety check to
    catch the dangerous pattern where the claimed value is turned into a
    WHERE filter. Returns short lowercase tokens, not schema identifiers.
    """
    q = _normalise(question)
    split_markers = (
        r"\bhas\s+(?:the\s+)?(?:highest|most|largest|greatest|lowest|least|smallest)\b",
        r"\bis\s+(?:the\s+)?(?:highest|most|largest|greatest|lowest|least|smallest)\b",
        r"\b(?:sabse\s+(?:zyada|ziada|kam|zyaada|zaida)|sab\s+se\s+(?:zyada|ziada|zyaada|zaida|kam))\b",
        r"\b(?:sold|generated|earned|made|has|had)\s+(?:the\s+)?(?:most|least)\b",
        r"\b(?:more|less|greater)\s+than\b",
    )
    prefix = None
    for marker in split_markers:
        m = re.search(marker, q, re.IGNORECASE)
        if m:
            prefix = q[:m.start()].strip()
            break
    if not prefix:
        return []

    generic = {
        "the", "a", "an", "category", "categories", "department", "departments",
        "product", "products", "employee", "employees", "customer", "customers",
        "order", "orders", "sale", "sales", "revenue", "salary", "amount",
        "hai", "hain", "ne", "ka", "ki", "ke", "ko", "is", "are", "has",
        "mein", "se", "with", "and", "of", "for", "who", "what", "which",
        # WH-question words. Belt-and-suspenders with the earlier
        # _has_wh_question_word() short-circuit in looks_like_claim_verification:
        # "kis" was previously missing here entirely (only its English
        # cousins "who"/"what"/"which" were listed), so "kis" leaked
        # through as a fake "claimed candidate" for any question that
        # reached this function some other way.
        "kis", "kise", "kaun", "kaunsa", "kaunsi", "kaunse",
        "konsa", "konsi", "konse",
    }
    tokens = [w for w in re.findall(r"[a-z0-9_]+", prefix) if w not in generic and len(w) > 1]
    # Keep only the last few meaningful tokens so a long natural-language
    # subject does not make the guard over-sensitive.
    return tokens[-4:]


def validate_claim_sql(sql: str, question: str) -> list:
    """Return semantic safety issues for a claim-verification SQL.

    The checks deliberately target the two most dangerous mistakes:
    1) using the user's asserted candidate as a WHERE filter before finding
       the winner; and
    2) returning a normal top-N query that never actually compares the
       independently calculated result with the claimed candidate.
    """
    issues = []
    if not looks_like_claim_verification(question):
        return issues

    lowered_sql = sql.lower()
    candidates = _extract_claim_candidate_tokens(question)

    # Candidate-as-filter guard. We only flag quoted literals/IN lists in a
    # WHERE clause, not arbitrary column names, because the user's candidate
    # is the thing that must not be used to select the winner.
    where_matches = list(re.finditer(
        r"\bWHERE\b(.+?)(?=\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bUNION\b|$)",
        sql, re.IGNORECASE | re.DOTALL
    ))
    if candidates and where_matches:
        for wm in where_matches:
            where_text = wm.group(1).lower()
            quoted_values = [v.lower() for v in re.findall(r"['\"]([^'\"]{1,120})['\"]", where_text)]
            for candidate in candidates:
                if any(candidate == value.strip() or candidate in value.split() for value in quoted_values):
                    if re.search(r"=|\bIN\b", where_text):
                        issues.append(
                            f"the claimed value '{candidate}' is used as a WHERE/IN filter; the winner must be calculated independently first"
                        )
                        break
            if issues:
                break

    # For superlative claims, the SQL must visibly perform an independent
    # ranking/aggregate comparison AND mention the claimed candidate in the
    # result/comparison. This prevents a plain "top 1" query from being
    # mistaken for verification of the user's assertion.
    is_superlative = any(re.search(p, question, re.IGNORECASE) for p in _CLAIM_SUPERLATIVE_PATTERNS)
    if is_superlative:
        ranking_tokens = (
            "row_number(", "dense_rank(", "rank(", "max(", "min(",
            "order by", "case when", "group by",
        )
        if not any(token in lowered_sql for token in ranking_tokens):
            issues.append("the superlative claim has no visible independent ranking/aggregate comparison")

        if candidates:
            # The candidate should appear as a literal/value in the SQL so
            # the result can say whether that exact claimed subject won.
            if not any(
                re.search(rf"['\"]{re.escape(candidate)}['\"]", sql, re.IGNORECASE)
                for candidate in candidates
            ):
                issues.append("the SQL does not explicitly compare the independently found winner with the user's claimed value")

    return issues


def build_claim_repair_note(issues: list, question: str, schema_text: str, relationship_text: str) -> str:
    return f"""The generated SQL does not safely verify the user's claim.
USER CLAIM: {question}

PROBLEMS:
- """ + "\n- ".join(issues) + f"""

Rewrite it so the claimed value is NEVER assumed to be the winner.
For a "X has the highest/most Y" claim, calculate Y for ALL candidates
first, determine the actual winner, then compare that winner with X. For a
direct A-vs-B comparison, calculate both sides and compare them. Use only
these verified relationships:
{relationship_text or '(none)'}
SCHEMA:
{schema_text}
Reply ONLY with corrected Oracle 11g SELECT SQL or CLARIFY: <specific question>."""


# "X with no Y" / "X that have no Y" style anti-join questions
# ("departments with no employees", "aise department jin k pass koi
# employees na ho", "koi orders nahi jinke customer ka email na ho").
# These always name exactly two related entities (which is exactly the
# ">= 2 relationship terms" signal below), but the shape of the answer
# (LEFT JOIN / NOT EXISTS) is completely unambiguous and already has a
# worked example in the main SQL prompt (see llm.py). The preflight
# grounding call uses a small local model with only vague written rules
# and no example of this exact pattern, and in practice it repeatedly
# mishandles it -- asking for the schema it was already given in its own
# prompt, or treating "aise department"/"such departments" (which simply
# refers forward to the negation clause) as if it were an ambiguous named
# entity. Recognizing the pattern deterministically and skipping the extra
# round-trip avoids that failure mode entirely.
_ZERO_RELATION_RE = re.compile(
    r"\bwith\s+no\b"
    r"|\bwithout\s+any\b"
    r"|\b(?:have|having|has)\s+no\b"
    r"|\bno\s+\w+\s+(?:assigned|associated)\b"
    r"|\b(?:koi|kisi)\s+[\w\s]{0,25}?\bna\s+ho\b"
    r"|\b(?:koi|kisi)\s+[\w\s]{0,25}?\bnahi\b"
    r"|\bbina\s+[\w\s]{0,25}?\bke\b"
    r"|\bjin\w*\s+ke?\s+pass\s+koi\b"
    r"|\bjin\w*\s+k\s+pass\s+koi\b",
    re.IGNORECASE,
)


def _is_zero_relation_question(q: str) -> bool:
    return bool(_ZERO_RELATION_RE.search(q))


def should_preflight(question: str, rank_intent: Optional[dict] = None) -> bool:
    """Return True only for questions where schema/entity ambiguity is likely.

    This is intentionally conservative. Existing deterministic ranking and
    simple-query paths stay untouched unless the wording itself suggests a
    relationship-heavy request.
    """
    if not getattr(config, "ENABLE_QUERY_PREFLIGHT", True):
        return False

    q = _normalise(question)

    if _is_zero_relation_question(q):
        return False

    words = set(re.findall(r"[a-z0-9_]+", q))

    # Explicit multi-entity / comparison language is a strong signal.
    if len(words & _RELATIONSHIP_TERMS) >= 2:
        return True

    # Ranking by a group/entity often needs a verified relationship.
    if rank_intent and rank_intent.get("is_ranking"):
        if any(w in words for w in ("each", "every", "per", "department", "category", "country", "region")):
            return True

    return False


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        # Small local models occasionally put one sentence before JSON.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            value = json.loads(m.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _last_clarification(history: Optional[list]) -> bool:
    if not history:
        return False
    last = history[-1]
    return (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and str(last.get("content", "")).strip().upper().startswith("CLARIFY:")
    )


def preflight_question(
    question: str,
    schema_text: str,
    relationship_text: str,
    history: Optional[list] = None,
) -> Optional[str]:
    """Return a CLARIFY message when the planner is confident the question
    is ambiguous/unsupported; otherwise return None.

    The planner is *not* trusted to authorize SQL. It is only allowed to stop
    the request. Any SQL that is eventually generated still goes through the
    deterministic JOIN validator below.
    """
    if not getattr(config, "ENABLE_QUERY_PREFLIGHT", True):
        return None

    # Always include recent history when present. Previously this was
    # skipped whenever the last turn was our own CLARIFY (see
    # _last_clarification below) -- the intent seems to have been "the
    # user's next message IS the direct answer, so history is redundant",
    # but that's backwards: a short answer like "the most orders placed
    # by a single customer" is only interpretable AS an answer if the
    # checker can see what question it's answering. Hiding history in
    # exactly that case made the checker judge the bare fragment in total
    # isolation, correctly find it ambiguous on its own, and issue a
    # second, redundant CLARIFY -- even though the user had just answered
    # the first one. Including history here (like every other case)
    # fixes that without changing behavior for any other path.
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\nRECENT CONVERSATION (context only):\n" + "\n".join(
            f"{m.get('role', 'unknown').upper()}: {m.get('content', '')}"
            for m in recent
            if isinstance(m, dict)
        )

    prompt = llm.build_grounding_preflight_prompt(
        question=question,
        schema_text=schema_text,
        relationship_text=relationship_text,
        history_text=history_text,
    )

    try:
        raw = llm.call_local_model([{"role": "system", "content": prompt}])
    except Exception as exc:
        print(f"[GROUNDING] preflight unavailable — keeping existing pipeline: {exc}")
        return None

    result = _extract_json(raw)
    if not result:
        print("[GROUNDING] preflight returned malformed JSON — keeping existing pipeline")
        return None

    decision = str(result.get("decision", "ANSWER")).strip().upper()
    clarification = str(result.get("clarification", "")).strip()

    if decision != "CLARIFY" or not clarification:
        return None

    # Never let the model manufacture a fake SQL/schema answer here. This
    # function only returns a human-facing clarification sentence.
    clarification = re.sub(r"\s+", " ", clarification)
    clarification = clarification[:500]
    if clarification.upper().startswith("CLARIFY:"):
        clarification = clarification.split(":", 1)[1].strip()
    return f"CLARIFY: {clarification}"


def _canonical_table(table: str, schema_dict: dict) -> Optional[str]:
    upper = table.upper()
    return upper if upper in schema_dict else None


def _allowed_join_pairs(schema_dict: dict, fk_pairs: list) -> set:
    """Both orientations are allowed because SQL can join either direction."""
    allowed = set()
    for child_table, child_col, parent_table, parent_col in fk_pairs:
        ct, cc = child_table.upper(), child_col.upper()
        pt, pc = parent_table.upper(), parent_col.upper()
        if ct in schema_dict and pt in schema_dict:
            allowed.add((ct, cc, pt, pc))
            allowed.add((pt, pc, ct, cc))
    return allowed


def relationship_text_from_fk_pairs(fk_pairs: list) -> str:
    lines = []
    seen = set()
    for child_table, child_col, parent_table, parent_col in fk_pairs:
        line = f"{child_table.upper()}.{child_col.upper()} = {parent_table.upper()}.{parent_col.upper()}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


def validate_sql_joins(sql: str, schema_dict: dict, fk_pairs: list) -> list:
    """Return human-readable JOIN violations; empty means all explicit JOINs
    are backed by a real FK edge and real columns.

    The validator intentionally checks only explicit JOIN ... ON predicates.
    It does not reject scalar subqueries or legitimate non-JOIN filters.
    """
    violations = []
    allowed = _allowed_join_pairs(schema_dict, fk_pairs)

    aliases = {}
    for m in re.finditer(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_$#]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_$#]*))?",
        sql,
        re.IGNORECASE,
    ):
        table = _canonical_table(m.group(1), schema_dict)
        alias = m.group(2)
        if table:
            aliases[table] = table
            if alias and alias.upper() not in {"ON", "WHERE", "JOIN", "GROUP", "ORDER", "HAVING"}:
                aliases[alias.upper()] = table

    for jm in _JOIN_RE.finditer(sql):
        joined_table_raw = jm.group(1)
        joined_table = _canonical_table(joined_table_raw, schema_dict)
        on_clause = jm.group(3)
        if not joined_table:
            violations.append(f"JOIN table {joined_table_raw.upper()} does not exist in the schema")
            continue

        refs = _QUALIFIED_REF_RE.findall(on_clause)
        # Check each equality-like pair in the ON clause. We only need to
        # reject a JOIN when no FK-backed pair connects the joined table to
        # another table reference in that ON expression.
        connected = False
        for i in range(0, len(refs) - 1):
            left_alias, left_col = refs[i]
            right_alias, right_col = refs[i + 1]
            left_table = aliases.get(left_alias.upper())
            right_table = aliases.get(right_alias.upper())
            if not left_table or not right_table or left_table == right_table:
                continue
            if (
                left_table,
                left_col.upper(),
                right_table,
                right_col.upper(),
            ) in allowed:
                connected = True
                break
            if (
                right_table,
                right_col.upper(),
                left_table,
                left_col.upper(),
            ) in allowed:
                connected = True
                break

        if not connected:
            violations.append(
                f"JOIN to {joined_table} is not backed by a verified foreign-key relationship"
            )

    return violations


def build_join_repair_note(violations: list, relationship_text: str) -> str:
    return (
        "Your SQL contains an unverified JOIN. Do NOT invent or infer a join "
        "from similarly named columns. Rewrite the query using ONLY the exact "
        "foreign-key relationships listed below. If the requested relationship "
        "cannot be represented by these relationships, reply with a concise "
        "CLARIFY: question instead of guessing.\n\n"
        "JOIN VALIDATION ERRORS:\n- "
        + "\n- ".join(violations)
        + "\n\nVERIFIED RELATIONSHIPS:\n"
        + (relationship_text or "(No foreign-key relationships were found.)")
        + "\nReply with ONLY corrected SQL or CLARIFY: <question>."
    )