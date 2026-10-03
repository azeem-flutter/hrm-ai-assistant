"""
conversation_context.py
-----------------------
Conservative, dependency-free multi-turn context resolver that runs BEFORE
RAG/LLM SQL generation.

Design goals:
    - never blindly inherit old context into a clearly new question;
    - detect explicit follow-ups separately from self-contained questions;
    - compare lightweight subject/domain signals between current and previous
      questions before inheritance when the message is ambiguous;
    - prefer CLARIFY over guessing when compatibility is not strong enough;
    - remain pure and fail-open so the existing SQL pipeline is unchanged.

This module does NOT generate SQL, inspect the database, call RAG, or call an
LLM. It only decides NEW | FOLLOW_UP | CLARIFY and, for safe follow-ups,
constructs a self-contained effective question for the existing pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ContextDecision:
    action: str  # NEW | FOLLOW_UP | CLARIFY
    effective_question: str
    clarification: Optional[str] = None
    confidence: float = 0.0
    context_index: Optional[int] = None
    reason: str = ""


# Explicit references are the strongest continuation signal. They mean the
# user is deliberately pointing back to prior conversation state.
_REFERENCE_PATTERNS = (
    r"\b(previous|prior|same|above|earlier|former)\b",
    r"\b(yeh|ye|iska|iske|iski|is\s+mein|in\s+mein\s+se|un\s+mein\s+se)\b",
    r"\b(wo|woh|uska|uske|usi)\b",
    r"\b(pehle\s+wala|last\s+wala|pichla|pichle|upar\s+wala)\b",
)

# Changes that commonly modify an existing request instead of introducing a
# complete new subject.
_MODIFIER_PATTERNS = (
    r"\btop\s+\d+\b",
    r"\bbottom\s+\d+\b",
    r"\bfirst\s+\d+\b",
    r"\blast\s+\d+\b",
    r"\b\d+\s+(highest|lowest|largest|smallest|most|least)\b",
    r"\b(sirf|only|bas)\b",
    r"\b(average|avg|mean|sum|total|count|details?|detail)\b",
    r"\b(after|before|between|baad|pehle)\b",
    r"\b(ab|then|also|instead|aur)\b",
)

# Domain/subject vocabulary. This is intentionally a broad *safety signal*,
# not a schema. A question naming one of these can often stand on its own.
_DOMAIN_TERMS = {
    "employee", "employees", "department", "departments", "customer", "customers",
    "product", "products", "category", "categories", "order", "orders", "sale",
    "sales", "revenue", "salary", "salaries", "city", "cities", "country",
    "countries", "province", "provinces", "location", "locations", "job", "jobs",
    "manager", "managers", "payment", "payments", "channel", "channels",
    "quantity", "quantities", "discount", "discounts", "amount", "amounts",
    "price", "prices", "tax", "taxes", "customer_name", "product_name",
}

# Intent/operation vocabulary. Used to distinguish a real standalone request
# from a fragment such as "top 3 do".
_OPERATION_TERMS = {
    "show", "list", "display", "find", "get", "give", "tell", "which", "what",
    "how", "kitne", "dikhao", "dikha", "batao", "bata", "btao", "kaun", "kon",
    "highest", "lowest", "top", "bottom", "average", "avg", "mean", "sum",
    "total", "count", "details", "detail",
}

# Words that carry little subject meaning for compatibility checks.
_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "by", "in", "on", "with", "and",
    "or", "is", "are", "was", "were", "do", "does", "please", "me", "my",
    "mujhe", "mujh", "do", "kar", "karo", "kr", "ke", "ka", "ki", "ko", "se",
    "mein", "main", "wala", "wale", "wali", "bhi", "bi", "ab", "aur", "then",
    "also", "only", "sirf", "bas", "show", "list", "display", "find", "get",
    "give", "tell", "dikhao", "dikha", "batao", "bata", "btao",
}


def _normalise(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", _normalise(text))


def _has_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _domain_terms(text: str) -> Set[str]:
    return set(_words(text)) & _DOMAIN_TERMS


def _normalize_domain_term(term: str) -> str:
    """Collapses simple singular/plural pairs to the same key, e.g.
    "customer"/"customers" -> "customer", "category"/"categories" ->
    "category", "tax"/"taxes" -> "tax". Both forms are already valid
    entries in _DOMAIN_TERMS (membership tests are unaffected by this) --
    this is only for OVERLAP comparisons between two sets of domain terms,
    so "customer" in one message correctly matches "customers" in another
    instead of looking like two unrelated subjects.
    """
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"  # categories -> category, salaries -> salary
    if term.endswith(("xes", "ses", "zes", "ches", "shes")) and len(term) > 4:
        return term[:-2]  # taxes -> tax
    if term.endswith("s") and len(term) > 3:
        return term[:-1]  # customers -> customer, orders -> order
    return term


def _normalized_domain_set(text: str) -> Set[str]:
    """_domain_terms(), but singular/plural pairs collapsed -- use this
    specifically wherever two domain-term sets are being compared for
    overlap (a genuine subject switch vs. the same subject worded a bit
    differently). Use plain _domain_terms() everywhere else (e.g. simple
    truthiness checks like "does this question name a domain at all")
    since those don't care about singular vs. plural."""
    return {_normalize_domain_term(t) for t in _domain_terms(text)}


def _operation_terms(text: str) -> Set[str]:
    return set(_words(text)) & _OPERATION_TERMS


def _subject_terms(text: str) -> Set[str]:
    """Extract lightweight subject anchors without pretending to do full NLP."""
    result: Set[str] = set()
    for word in _words(text):
        if word in _STOPWORDS:
            continue
        if len(word) <= 2:
            continue
        # Pure numbers usually modify an existing request; they are not a
        # reliable subject/domain anchor.
        if word.isdigit():
            continue
        result.add(word)
    return result


def _looks_self_contained(question: str) -> bool:
    """Conservatively identify questions that should NOT inherit old context."""
    words = _words(question)
    domains = _domain_terms(question)
    operations = _operation_terms(question)

    # A named domain plus an operation/ranking request is normally a complete
    # request, e.g. "top 3 customers by sales".
    if domains and operations:
        return True

    # A sufficiently descriptive message with multiple subject anchors is more
    # likely a new request than a continuation fragment.
    subjects = _subject_terms(question)
    if len(words) >= 5 and len(subjects) >= 3:
        return True

    # Common WH forms with a domain are independently answerable.
    q = _normalise(question)
    if domains and re.search(r"\b(which|what|how|kitne|kaun|kon)\b", q):
        return True

    return False


def _is_context_dependent(question: str) -> bool:
    """Return True only when the message is genuinely under-specified."""
    q = _normalise(question)
    words = _words(q)
    if not q:
        return False

    if _looks_self_contained(q):
        return False

    has_reference = _has_pattern(q, _REFERENCE_PATTERNS)
    has_modifier = _has_pattern(q, _MODIFIER_PATTERNS)

    if has_reference:
        return True

    # Short modifier fragments: "top 3 do", "average batao".
    if len(words) <= 9 and has_modifier:
        return True

    # Short leading restrictions are commonly modifications.
    if len(words) <= 9 and re.match(r"^(sirf|only|bas)\b", q):
        return True

    # Date/filter fragments such as "2023 ke baad wale".
    if len(words) <= 8 and re.search(r"\b(baad|pehle|after|before)\b", q):
        return True

    return False


def _choose_context_index(question: str, contexts: Sequence[Dict[str, Any]]) -> Optional[int]:
    if not contexts:
        return None

    q = _normalise(question)
    if re.search(r"\b(pehle\s+wala|previous\s+one|prior\s+one|pichla)\b", q):
        return len(contexts) - 2 if len(contexts) >= 2 else None

    return len(contexts) - 1


def _compatibility(question: str, context: Dict[str, Any]) -> Tuple[float, str]:
    """Compare current and previous meaning using conservative lexical anchors.

    This is not advertised as true semantic embedding similarity. Its purpose
    is a safety check: detect obvious subject switches before old context is
    inherited. Explicit references still carry strong continuation evidence.
    """
    previous = (context.get("effective_question") or context.get("question") or "").strip()
    if not previous:
        return 0.0, "previous context has no usable question"

    q_domains = _domain_terms(question)
    p_domains = _domain_terms(previous)
    q_domains_norm = _normalized_domain_set(question)
    p_domains_norm = _normalized_domain_set(previous)
    q_subjects = _subject_terms(question)
    p_subjects = _subject_terms(previous)
    explicit_reference = _has_pattern(_normalise(question), _REFERENCE_PATTERNS)

    # Different explicit domains are the strongest NEW-question signal.
    # Compared on the singular/plural-normalized sets so "orders" earlier
    # and "order" now aren't mistaken for two unrelated subjects.
    if q_domains and p_domains and not (q_domains_norm & p_domains_norm):
        return 0.0, f"domain switch detected: {sorted(q_domains)} vs {sorted(p_domains)}"

    # If the current message introduces a clear standalone subject that has no
    # overlap with the previous subject, do not inherit automatically.
    if q_domains and not (q_domains_norm & p_domains_norm) and _looks_self_contained(question):
        return 0.0, "new self-contained domain detected"

    overlap = q_subjects & p_subjects
    union = q_subjects | p_subjects
    lexical_score = len(overlap) / len(union) if union else 0.0

    # Explicit references intentionally raise compatibility, but never erase a
    # detected explicit domain switch above.
    if explicit_reference:
        return max(0.75, lexical_score), "explicit reference to previous context"

    # Modifier-only fragments often have no lexical overlap because they only
    # add a number/order/filter. They are safe with the latest context.
    if not q_domains and _is_context_dependent(question):
        return max(0.70, lexical_score), "under-specified modifier with no competing domain"

    if lexical_score >= 0.20:
        return lexical_score, "subject overlap with previous context"

    return lexical_score, "no strong subject compatibility signal"


def _build_effective_question(question: str, context: Dict[str, Any]) -> str:
    previous = (context.get("effective_question") or context.get("question") or "").strip()
    if not previous:
        return question.strip()

    return (
        f"Previous request: {previous}\n"
        f"Follow-up modification: {question.strip()}\n"
        "Continue the same request context only as supported by the previous request. "
        "Apply the new modification and do not invent a new subject."
    )


def _build_clarification_answer_question(
    *, original_question: str, clarify_asked: str, answer: str
) -> str:
    """Deterministically merges [original ambiguous question] + [the
    clarifying question we asked] + [the user's answer] into ONE
    self-contained request.

    This exists so the LLM never has to re-derive "what am I actually
    being asked" by reading raw back-and-forth history on its own --  it
    gets one already-assembled question instead, with nothing left
    implicit. That re-derivation step is exactly where a weaker model can
    lose track of part of the original request (e.g. dropping "customers"
    entirely and answering a much simpler, wrong question instead) or
    re-ask the same clarifying question because it couldn't connect the
    reply back to what prompted it.
    """
    parts = [f"Original request: {original_question.strip()}"]
    if clarify_asked.strip():
        parts.append(f"Clarification question that was asked: {clarify_asked.strip()}")
    parts.append(f"User's answer to that clarification: {answer.strip()}")
    parts.append(
        "Treat this as ONE fully-specified request: resolve the original "
        "request using the user's answer to fill in what was missing. Do "
        "not ask about the same ambiguity again."
    )
    return "\n".join(parts)


def resolve_question(
    question: str,
    contexts: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    enabled: bool = True,
    pending_clarification: Optional[Dict[str, str]] = None,
) -> ContextDecision:
    """Resolve a message into NEW, FOLLOW_UP, or CLARIFY.

    Decision order:
        0. A pending clarification is waiting for an answer, and this
           message doesn't look like an unrelated new question -> deterministically
           merge and treat as FOLLOW_UP (see _build_clarification_answer_question).
        1. Clearly self-contained -> NEW.
        2. Clearly context-dependent but no context -> CLARIFY.
        3. Context-dependent -> compare against selected previous context.
        4. Strong compatibility -> FOLLOW_UP.
        5. Obvious subject/domain switch -> NEW.
        6. Uncertain -> CLARIFY instead of guessing.
    """
    original = (question or "").strip()
    if not enabled or not original:
        return ContextDecision("NEW", original, confidence=1.0, reason="disabled or empty")

    if pending_clarification and pending_clarification.get("original_question"):
        orig_q = pending_clarification["original_question"]
        clarify_q = pending_clarification.get("clarify_asked", "")
        # Safety guard: don't force-merge a message that clearly abandons
        # the pending clarification for an unrelated new question (e.g.
        # clarify was about "orders", but the user says "show me all
        # departments" instead of answering). Only skip the merge when
        # the new message is BOTH self-contained on its own AND shares no
        # domain vocabulary at all with what was being clarified -- a
        # short/ambiguous reply always still gets merged.
        new_domains = _normalized_domain_set(original)
        prior_domains = _normalized_domain_set(orig_q) | _normalized_domain_set(clarify_q)
        looks_like_abandonment = (
            _looks_self_contained(original)
            and new_domains
            and not (new_domains & prior_domains)
        )
        if not looks_like_abandonment:
            return ContextDecision(
                "FOLLOW_UP",
                _build_clarification_answer_question(
                    original_question=orig_q,
                    clarify_asked=clarify_q,
                    answer=original,
                ),
                confidence=0.99,
                reason="merged answer to pending clarification",
            )
        # else: falls through to the normal logic below, exactly as if no
        # clarification were pending -- app.py already discarded the
        # pending state for this turn before calling here either way.

    safe_contexts = [c for c in (contexts or []) if c and c.get("successful")]

    # Most important protection: a complete new question must not inherit the
    # previous query simply because it happens to be short.
    if _looks_self_contained(original):
        return ContextDecision("NEW", original, confidence=0.98, reason="self-contained question")

    if not _is_context_dependent(original):
        return ContextDecision("NEW", original, confidence=0.85, reason="no continuation signal")

    index = _choose_context_index(original, safe_contexts)
    if index is None:
        return ContextDecision(
            "CLARIFY",
            original,
            clarification=(
                "Your request looks like a continuation, but I don't have a clear "
                "previous result to apply it to. What should I modify or rank?"
            ),
            confidence=0.0,
            reason="context-dependent request with no successful context",
        )

    context = safe_contexts[index]
    compatibility, reason = _compatibility(original, context)

    # Explicit subject/domain switch: definitely do not inherit.
    if compatibility == 0.0 and "domain switch" in reason:
        return ContextDecision("NEW", original, confidence=0.95, reason=reason)

    if compatibility >= 0.70:
        return ContextDecision(
            "FOLLOW_UP",
            _build_effective_question(original, context),
            confidence=compatibility,
            context_index=index,
            reason=reason,
        )

    # Ambiguous fragments should never silently borrow unrelated context.
    return ContextDecision(
        "CLARIFY",
        original,
        clarification=(
            "I can see this may relate to the previous request, but the subject "
            "is not clear enough to safely continue. Please specify what you want "
            "to modify or refer to."
        ),
        confidence=compatibility,
        context_index=index,
        reason=reason,
    )


def make_context_record(
    *,
    question: str,
    effective_question: str,
    sql: str,
) -> Dict[str, Any]:
    """Create a compact successful-turn record for Streamlit session state."""
    return {
        "question": (question or "").strip(),
        "effective_question": (effective_question or question or "").strip(),
        "sql": (sql or "").strip(),
        "successful": True,
    }