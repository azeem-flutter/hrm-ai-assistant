"""Deterministic ranking/superlative detection (no LLM involved)."""
from core import ranking_intent


def test_top_n_english():
    r = ranking_intent.classify("top 5 employees by salary")
    assert r["is_ranking"] and r["direction"] == "DESC" and r["n"] == 5


def test_roman_urdu_superlative_high():
    r = ranking_intent.classify("sabse zyada salary kis ki hai")
    assert r["is_ranking"] and r["direction"] == "DESC"


def test_roman_urdu_superlative_low():
    r = ranking_intent.classify("sabse kam salary wale employees")
    assert r["is_ranking"] and r["direction"] == "ASC"


def test_plain_question_is_not_ranking():
    r = ranking_intent.classify("show all employees")
    assert not r["is_ranking"]
    assert r["ranking_type"] == "NONE"
