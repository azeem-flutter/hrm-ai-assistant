"""Retrieval layer: keyword fallback and cosine top-k."""
import numpy as np

from integrations import rag

CHUNKS = {
    "EMPLOYEES": "EMPLOYEES employee_id salary hire_date",
    "DEPARTMENTS": "DEPARTMENTS department_name manager_id",
    "LOCATIONS": "LOCATIONS city country_id",
}


def test_keyword_retrieval_picks_relevant_table():
    assert list(rag._keyword_retrieve("salary of employees", CHUNKS, 1)) == ["EMPLOYEES"]


def test_keyword_retrieval_respects_top_k():
    assert len(rag._keyword_retrieve("employees departments city", CHUNKS, 2)) == 2


def test_retrieve_tables_with_keyword_provider():
    assert "LOCATIONS" in rag.retrieve_tables("which city", CHUNKS, top_k=1)


def test_cosine_topk_orders_by_similarity():
    matrix = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    assert [int(i) for i in rag._cosine_topk(query, matrix, 2)] == [0, 2]


def test_empty_chunks_return_empty():
    assert rag.retrieve_tables("anything", {}) == {}
