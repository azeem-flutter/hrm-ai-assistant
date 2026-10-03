"""Claim verification: a user's assertion must be checked, not trusted."""
from core import query_grounding


def test_superlative_claim_is_detected():
    assert query_grounding.looks_like_claim_verification("IT has the highest salary")


def test_normal_retrieval_is_not_a_claim():
    assert not query_grounding.looks_like_claim_verification("show top 5 employees")
