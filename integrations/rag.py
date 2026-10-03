"""Retrieval layer.

Default mode is *hybrid* but does NOT load an embedding model. It first uses
Vanna when explicitly enabled/configured, then supplements/falls back to a
small deterministic lexical retriever. This keeps the laptop light while the
schema is still small, and scales to large schemas without sending every table
to Qwen.

The previous local Ollama/sentence-transformers embedding path is retained as
RAG_PROVIDER=ollama/local for rollback.
"""
import re
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
from config import config

try:
    import ollama
except ImportError:
    ollama = None

_local_embed_model = None
_vanna = None


def _get_local_embed_model():
    global _local_embed_model
    if _local_embed_model is None:
        from sentence_transformers import SentenceTransformer
        _local_embed_model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return _local_embed_model


def _embed(texts: list) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if config.EMBEDDING_PROVIDER == "local":
        model = _get_local_embed_model()
        matrix = np.array(model.encode(list(texts), normalize_embeddings=False), dtype=np.float32)
    else:
        if ollama is None:
            raise RuntimeError("ollama package is required for RAG_PROVIDER=ollama")
        vectors = [ollama.embed(model=config.EMBED_MODEL, input=text)["embeddings"][0] for text in texts]
        matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, top_k: int) -> list:
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query_vec
    return list(np.argsort(-scores)[: min(top_k, matrix.shape[0])])


@st.cache_resource(show_spinner=False)
def _embed_table_chunks(chunks_key: tuple) -> np.ndarray:
    return _embed([text for _, text in chunks_key])


@st.cache_resource(show_spinner=False)
def _embed_examples(examples_key: tuple) -> np.ndarray:
    return _embed(list(examples_key))


def embed_query(question: str) -> Optional[np.ndarray]:
    """Only create an embedding in the legacy embedding modes."""
    if config.RAG_PROVIDER not in {"ollama", "local"}:
        return None
    return _embed([question])[0]


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()))


def _keyword_score(question: str, table_name: str, chunk: str) -> float:
    q = _tokens(question)
    if not q:
        return 0.0
    name_tokens = _tokens(table_name)
    c = _tokens(chunk)
    score = 0.0
    score += 5.0 * len(q & name_tokens)
    score += 1.5 * len(q & c)
    # Phrase-ish business vocabulary bonus: a question word occurring in a
    # column definition is more useful than the same word in boilerplate.
    return score / max(1.0, len(q) ** 0.5)


def _keyword_retrieve(question: str, table_chunks: dict, top_k: int) -> dict:
    ranked = sorted(
        table_chunks.items(),
        key=lambda kv: (_keyword_score(question, kv[0], kv[1]), kv[0]),
        reverse=True,
    )
    return dict(ranked[: min(top_k, len(ranked))])


def _get_vanna():
    """Create the legacy Vanna hosted-vector adapter lazily.

    Vanna is intentionally optional because its package has had major API
    changes. If it is unavailable/misconfigured, retrieval continues through
    the deterministic local fallback instead of breaking the app.
    """
    global _vanna
    if _vanna is not None:
        return _vanna
    if not config.VANNA_ENABLED or not config.VANNA_MODEL or not config.VANNA_API_KEY:
        return None
    try:
        from vanna.vannadb import VannaDB_VectorStore
        from vanna.ollama import Ollama

        class MyVanna(VannaDB_VectorStore, Ollama):
            def __init__(self):
                VannaDB_VectorStore.__init__(self, vanna_model=config.VANNA_MODEL, vanna_api_key=config.VANNA_API_KEY, config={})
                Ollama.__init__(self, config={"model": config.OLLAMA_MODEL})

        _vanna = MyVanna()
        return _vanna
    except Exception as exc:
        print(f"[RAG] Vanna unavailable, using keyword fallback: {exc}")
        return None


def _vanna_retrieve(question: str, table_chunks: dict, top_k: int) -> dict:
    vn = _get_vanna()
    if vn is None:
        return {}
    try:
        ddl_rows = vn.get_related_ddl(question) or []
        joined = "\n\n".join(str(x) for x in ddl_rows)
        selected = {}
        # Match returned DDL back to our verified Oracle table chunks. This
        # prevents Vanna from becoming a source of truth for schema names.
        upper = joined.upper()
        for name, chunk in table_chunks.items():
            if name.upper() in upper:
                selected[name] = chunk
        return dict(list(selected.items())[:top_k])
    except Exception as exc:
        print(f"[RAG] Vanna retrieval failed, using keyword fallback: {exc}")
        return {}


def retrieve_tables(question: str, table_chunks: dict, top_k: int = None, query_vec=None) -> dict:
    if not table_chunks:
        return {}
    top_k = top_k or config.RAG_TOP_K_TABLES

    if config.RAG_PROVIDER in {"hybrid", "vanna"}:
        selected = _vanna_retrieve(question, table_chunks, top_k)
        if selected:
            # Add a couple of deterministic lexical candidates so a Vanna
            # result never becomes a single-source bottleneck.
            lexical = _keyword_retrieve(question, table_chunks, max(2, top_k // 2))
            for name, chunk in lexical.items():
                if len(selected) >= top_k:
                    break
                selected.setdefault(name, chunk)
            return selected
        if config.RAG_PROVIDER == "vanna":
            return _keyword_retrieve(question, table_chunks, top_k)
        return _keyword_retrieve(question, table_chunks, top_k)

    if config.RAG_PROVIDER == "keyword":
        return _keyword_retrieve(question, table_chunks, top_k)

    # Legacy embedding path.
    names = sorted(table_chunks.keys())
    key = tuple((n, table_chunks[n]) for n in names)
    matrix = _embed_table_chunks(key)
    query_vec = query_vec if query_vec is not None else embed_query(question)
    idx = _cosine_topk(query_vec, matrix, top_k)
    return {names[i]: table_chunks[names[i]] for i in idx}


def retrieve_examples(question: str, examples: list, top_k: int = None, query_vec=None) -> list:
    if not examples:
        return []
    top_k = top_k or config.RAG_TOP_K_EXAMPLES

    # Vanna can supply learned question/SQL examples. We do not trust its SQL
    # as schema truth; it is only an example signal. The existing curated file
    # remains the deterministic fallback.
    if config.RAG_PROVIDER in {"hybrid", "vanna"}:
        vn = _get_vanna()
        if vn is not None:
            try:
                rows = vn.get_similar_question_sql(question) or []
                out = []
                for row in rows:
                    q = row.get("question") if isinstance(row, dict) else None
                    sql = row.get("sql") if isinstance(row, dict) else None
                    if q and sql:
                        out.append({"question": q, "sql": sql})
                if out:
                    return out[:top_k]
            except Exception as exc:
                print(f"[RAG] Vanna example retrieval failed: {exc}")
        # Cheap lexical matching for the curated examples.
        q_tokens = _tokens(question)
        scored = []
        for ex in examples:
            overlap = len(q_tokens & _tokens(ex.get("question", "")))
            scored.append((overlap, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    if config.RAG_PROVIDER == "keyword":
        q_tokens = _tokens(question)
        scored = [(len(q_tokens & _tokens(ex.get("question", ""))), ex) for ex in examples]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    questions_key = tuple(ex["question"] for ex in examples)
    matrix = _embed_examples(questions_key)
    query_vec = query_vec if query_vec is not None else embed_query(question)
    idx = _cosine_topk(query_vec, matrix, top_k)
    return [examples[i] for i in idx]


def learn_successful_query(question: str, sql: str) -> bool:
    """Opt-in episodic memory: store only successfully executed SQL in Vanna.

    This changes the retrieval memory, not Qwen weights. It is disabled by
    default because Vanna hosted storage means the question/SQL leaves the
    machine.
    """
    if not config.VANNA_AUTO_LEARN:
        return False
    vn = _get_vanna()
    if vn is None:
        return False
    try:
        vn.train(question=question, sql=sql)
        return True
    except Exception as exc:
        print(f"[RAG] Vanna auto-learn failed: {exc}")
        return False


def retrieve_for_prompt(question: str, table_chunks: dict, top_k: int = None) -> dict:
    """Small helper for the LangGraph execution-repair node."""
    return retrieve_tables(question, table_chunks, top_k=top_k or config.RAG_TOP_K_TABLES)
