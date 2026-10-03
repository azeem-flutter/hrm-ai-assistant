"""
ranking_intent.py
-----------------
Small deterministic intent layer for ranking/superlative questions.

This module does NOT generate SQL and does NOT know the database schema.
It only converts wording into a compact set of ranking constraints so the
existing LLM/SQL pipeline can generate SQL from a stable specification.
Normal (non-ranking) questions are not affected.
"""

import re
from typing import Any, Dict, Optional


_HIGH_WORDS = (
    "top", "highest", "high", "most", "maximum", "max", "best",
    "largest", "greatest", "newest", "latest", "youngest",
    # "sabse"/"sab se" is written both as one word and two, and "zyada"
    # ("most/more") is very commonly spelled "ziada"/"zyaada"/"zaida" in
    # Roman Urdu -- all four spellings are the same spoken word. Missing
    # any of these combinations (e.g. "sab se zaida") silently skipped
    # this entire deterministic ranking path.
    "sabse zyada", "sabse ziada", "sabse zyaada", "sabse zaida",
    "sab se zyada", "sab se ziada", "sab se zyaada", "sab se zaida",
    "sabse acha", "sab se acha", "sabse behtareen", "sab se behtareen",
    "sabse naya", "sab se naya", "sabse nayi", "sab se nayi",
)
_LOW_WORDS = (
    "bottom", "lowest", "low", "least", "minimum", "min", "worst",
    "smallest", "oldest", "earliest",
    "sabse kam", "sab se kam", "sabse ghatia", "sab se ghatia",
    "sabse purana", "sab se purana", "sabse purani", "sab se purani",
)
# Explicit "no limit, everything" wording. Deliberately excludes bare "sab"
# because "sab" is also the first word of the "sab se zyada/kam" superlative
# idiom (see _HIGH_WORDS / _LOW_WORDS) -- matching bare "sab" here would
# wrongly flag every plain "sab se zyada sales" superlative question as an
# "all rows, no limit" request. "sari"/"saari"/"sabhi"/etc. are unambiguous
# stand-alone words for "all" and don't collide with that idiom.
_ALL_WORDS = (
    "sari", "saari", "sara", "saara", "sabhi",
    "poori", "puri", "pura", "poora", "all",
)
# The app merges a pending clarification into one block of text via
# conversation_context._build_clarification_answer_question():
#   "...Clarification question that was asked: <our own question>
#    User's answer to that clarification: <what the user actually typed>..."
# The middle part is SYSTEM-authored (it's whatever question we asked
# last turn) and can itself contain numbers/ranking words as examples
# (e.g. "...top 10 categories, or all categories?"). Left in place, those
# words get parsed as if the USER had said them -- e.g. a bare "sari sales
# category" answer was being misread as "10" purely because our own
# earlier question happened to mention 10 as an example option. Strip that
# system-authored sentence out before any extraction below runs.
_CLARIFY_QUESTION_STRIP_RE = re.compile(
    r"clarification question that was asked:.*?"
    r"(?=user's answer to that clarification:|$)",
    re.IGNORECASE | re.DOTALL,
)
_PER_GROUP_RE = re.compile(
    r"\b(each|every|per|har)\b|\b(har\s+(?:ek\s+)?)", re.I
)
_PERCENT_RE = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)\s*(%|percent|percentage)(?!\w)", re.I)
_RANGE_RE = re.compile(
    r"\b(?:rank|ranks?)\s*(?:is\s*)?(?:between\s*)?(\d+)\s*(?:to|and|[-–])\s*(\d+)\b"
    r"|\b(?:between)\s+(?:rank\s+)?(\d+)\s+and\s+(?:rank\s+)?(\d+)\b",
    re.I,
)
_EXACT_RANK_RE = re.compile(
    r"\b(?:rank|ranking)\s*(?:is\s*)?(?:exactly\s*)?(?:=\s*)?(\d+)\b"
    r"|\b(?:exactly|at)\s+(?:rank\s+)?(\d+)\b",
    re.I,
)
_ORDINAL_RE = re.compile(
    r"\b(\d+)(?:st|nd|rd|th)\b|\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth)\b"
    r"|\b(ek|pehla|pehli|doosra|dusra|doosri|dusri|teesra|tisra|chautha|paanchwa|panchwa|chhatha|chehtha)\b",
    re.I,
)
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15,
    "ek": 1, "pehla": 1, "pehli": 1, "doosra": 2, "dusra": 2,
    "doosri": 2, "dusri": 2, "teesra": 3, "tisra": 3, "chautha": 4,
    "paanchwa": 5, "panchwa": 5, "chhatha": 6, "chehtha": 6,
}
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "panch": 5,
    "paanch": 5, "che": 6, "chhe": 6, "saat": 7, "aath": 8,
    "nau": 9, "das": 10, "dus": 10, "gyarah": 11, "barah": 12,
    "terah": 13, "chaudah": 14, "pandrah": 15, "solah": 16,
    "satrah": 17, "atharah": 18, "unnees": 19, "bees": 20,
}
_TOP_BOTTOM_PAIR = (
    re.compile(r"\btop\b[^.]{0,40}\b(?:and|aur|plus)\b[^.]{0,20}\bbottom\b", re.I),
    re.compile(r"\bbottom\b[^.]{0,40}\b(?:and|aur|plus)\b[^.]{0,20}\btop\b", re.I),
    re.compile(r"\b(?:sabse|sab\s+se)\s+(?:zyada|ziada|zyaada|zaida|acha)[^.]{0,40}\b(?:aur|and)\b[^.]{0,20}\b(?:sabse|sab\s+se)\s+kam", re.I),
    re.compile(r"\b(?:sabse|sab\s+se)\s+kam[^.]{0,40}\b(?:aur|and)\b[^.]{0,20}\b(?:sabse|sab\s+se)\s+(?:zyada|ziada|zyaada|zaida|acha)", re.I),
)


def _has_word(q: str, words) -> bool:
    for word in words:
        if " " in word:
            if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", q):
                return True
        elif re.search(rf"\b{re.escape(word)}\b", q):
            return True
    return False


def _extract_count(q: str) -> Optional[int]:
    # Numbers close to top/bottom/highest/lowest wording are row counts.
    for m in re.finditer(r"\b(\d+)\b", q):
        n = int(m.group(1))
        if 0 < n <= 100000:
            # Do not treat years as a top-N count.
            if 1900 <= n <= 2100:
                continue
            return n
    for word, n in sorted(_NUM_WORDS.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(word)}\b", q):
            return n
    return None


def _extract_ordinal(q: str) -> Optional[int]:
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", q)
    if m:
        return int(m.group(1))
    # Generic Roman Urdu ordinal suffix pattern: a digit directly
    # followed by "va"/"wan"/"waan"/"vi"/"win" etc. (e.g. "5va", "10wan",
    # "3ra"). The fixed _ORDINAL_WORDS dict below only covers spelled-out
    # words up to "fifteenth"/"chehtha" (6th) -- it has no entry at all
    # for "5va", "10va", "12wan", and so on, so a question like "5va sab
    # se bara order" was silently NOT recognised as a ranking/ordinal
    # request, skipping the safe deterministic SQL template entirely and
    # falling through to the LLM, which then hallucinated table/column
    # names instead. This pattern catches any digit + Roman-Urdu ordinal
    # suffix generically, with no upper bound.
    m = re.search(r"\b(\d+)\s*(?:va|wan|waan|vi|vin|win|ven)\b", q, re.IGNORECASE)
    if m:
        return int(m.group(1))
    for word, n in _ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return n
    return None


def _extract_range(q: str):
    for m in _RANGE_RE.finditer(q):
        a = m.group(1) or m.group(3)
        b = m.group(2) or m.group(4)
        if a and b:
            return int(a), int(b)
    # Common Roman Urdu phrasing: "rank 10 se 15 tak".
    m = re.search(r"\brank\w*\s+(\d+)\s+se\s+(\d+)\s+tak\b", q)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def classify(question: str) -> Dict[str, Any]:
    q = " ".join(question.lower().split())
    q = _CLARIFY_QUESTION_STRIP_RE.sub(" ", q)
    q = " ".join(q.split())
    high = _has_word(q, _HIGH_WORDS)
    low = _has_word(q, _LOW_WORDS)
    ordinal = _extract_ordinal(q)
    exact = _EXACT_RANK_RE.search(q)
    rank_range = _extract_range(q)
    percent_match = _PERCENT_RE.search(q)
    per_group = bool(_PER_GROUP_RE.search(q))
    has_rank_word = bool(re.search(r"\brank(?:ed|ing)?\b", q))
    compound = any(p.search(q) for p in _TOP_BOTTOM_PAIR)
    # "sari sales category" / "sabhi categories" etc. answering a
    # high/low ranking question means "no limit, show every group" --
    # not "use the default row count". Only meaningful alongside an
    # actual high/low ranking word; a bare "sari"/"all" elsewhere is not
    # a ranking signal by itself and is left to the existing pipeline.
    all_requested = bool(_has_word(q, _ALL_WORDS)) and (high or low) and not compound

    # "ranked by hire date" is intentionally NOT a ranking-selection request.
    # It only needs the existing default cap/order behavior.
    selection_signal = high or low or ordinal is not None or bool(exact) or rank_range is not None or percent_match or compound
    is_ranking = bool(selection_signal)

    direction = None
    if low and not high:
        direction = "ASC"
    elif high and not low:
        direction = "DESC"
    elif ordinal is not None or exact or rank_range:
        # Ordinal/rank normally means "highest" when the question contains
        # salary/pay/value language; the SQL prompt can refine this from schema.
        direction = "DESC"

    if compound:
        ranking_type = "TOP_BOTTOM"
    elif percent_match:
        ranking_type = "PERCENT"
    elif rank_range is not None:
        ranking_type = "RANGE"
    elif exact:
        ranking_type = "EXACT_RANK"
    elif ordinal is not None:
        ranking_type = "EXACT_RANK"
    elif high or low:
        ranking_type = "TOP_N"
    else:
        ranking_type = "NONE"

    percent = float(percent_match.group(1)) if percent_match else None

    # For a combined top/bottom request, preserve both counts independently.
    # This avoids treating "top 2 and bottom 5" as a single N.
    top_count = None
    bottom_count = None
    if compound:
        top_part = re.split(r"\b(?:bottom|(?:sabse|sab\s+se)\s+(?:kam|ghatia|purana|purani))\b", q, maxsplit=1)[0]
        bottom_part = re.split(r"\b(?:top|(?:sabse|sab\s+se)\s+(?:zyada|ziada|zyaada|zaida|acha|behtareen|naya|nayi))\b", q, maxsplit=1)[-1]
        top_count = _extract_count(top_part)
        bottom_count = _extract_count(bottom_part)
        # _extract_count can see the wrong number in a longer phrase; the
        # common "top N ... bottom M" form is safer when parsed directly.
        m_top = re.search(r"\btop\s+(\d+)\b", q)
        m_bottom = re.search(r"\bbottom\s+(\d+)\b", q)
        if m_top:
            top_count = int(m_top.group(1))
        if m_bottom:
            bottom_count = int(m_bottom.group(1))

    n = _extract_count(q) if ranking_type == "TOP_N" else None
    if ranking_type == "EXACT_RANK":
        n = ordinal if ordinal is not None else int(exact.group(1) or exact.group(2))

    # A percentage query may also contain a number that is not a row count.
    if ranking_type == "PERCENT":
        n = None

    # An explicit "all/sari/sabhi" answer always wins over any number that
    # was still found in the text (e.g. a stray digit elsewhere) -- the
    # user asked for everything, not a specific count.
    if all_requested:
        n = None

    return {
        "is_ranking": is_ranking,
        "ranking_type": ranking_type,
        "direction": direction,
        "n": n,
        "percent": percent,
        "range": rank_range,
        "per_group": per_group,
        "compound_top_bottom": compound,
        "top_count": top_count,
        "bottom_count": bottom_count,
        "all_requested": all_requested,
        "chained": bool(re.search(r"\b(then|phir|among them|unme se|inhi mein se|unhi mein se)\b", q)),
        "has_rank_word": has_rank_word,
    }