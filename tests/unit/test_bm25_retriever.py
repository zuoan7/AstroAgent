from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.rag.bm25_retriever import BM25Retriever


class FakeBM25Plus:
    def __init__(self, corpus: list[list[str]]):
        self.corpus = corpus

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        query_counter = Counter(query_tokens)
        scores = []
        for doc_tokens in self.corpus:
            doc_counter = Counter(doc_tokens)
            score = 0.0
            for token, qfreq in query_counter.items():
                score += float(doc_counter.get(token, 0) * qfreq)
            scores.append(score)
        return scores


def test_regex_tokenizer_preserves_term_frequency():
    tokens = BM25Retriever._tokenize_regex("mars mars mission mars")
    assert tokens.count("mars") == 3
    assert tokens.count("mission") == 1


def test_bm25_prefers_document_with_higher_term_frequency(tmp_path: Path):
    index_path = tmp_path / "bm25_index.pkl"
    corpus_path = tmp_path / "bm25_corpus.jsonl"

    with patch("src.rag.bm25_retriever.settings") as mock_settings, patch(
        "src.rag.bm25_retriever.BM25Plus", FakeBM25Plus
    ):
        mock_settings.RAG_ENABLED = True
        mock_settings.VECTOR_DB_PATH = str(tmp_path)

        retriever = BM25Retriever(
            index_path=str(index_path),
            corpus_path=str(corpus_path),
            use_jieba=False,
        )
        retriever.build_index(
            [
                "mars mars mars mission log",
                "mars mission log",
                "jupiter mission log",
            ],
            [{}, {}, {}],
        )

        results = retriever.search("mars mission", top_k=2)

    assert len(results) == 2
    assert results[0]["document"] == "mars mars mars mission log"
    assert results[0]["score"] > results[1]["score"]


def test_bm25_can_rebuild_from_corpus_without_chroma(tmp_path: Path):
    index_path = tmp_path / "bm25_index.pkl"
    corpus_path = tmp_path / "bm25_corpus.jsonl"

    with patch("src.rag.bm25_retriever.settings") as mock_settings, patch(
        "src.rag.bm25_retriever.BM25Plus", FakeBM25Plus
    ):
        mock_settings.RAG_ENABLED = True
        mock_settings.VECTOR_DB_PATH = str(tmp_path)

        retriever = BM25Retriever(
            index_path=str(index_path),
            corpus_path=str(corpus_path),
            use_jieba=False,
        )
        retriever.build_index(
            ["木星是太阳系最大的行星", "土星拥有环系统"],
            [{"source": "wiki"}, {"source": "wiki"}],
        )

        index_path.unlink()

        reloaded = BM25Retriever(
            index_path=str(index_path),
            corpus_path=str(corpus_path),
            use_jieba=False,
        )
        results = reloaded.search("木星", top_k=1)

    assert reloaded.bm25 is not None
    assert len(results) == 1
    assert results[0]["document"] == "木星是太阳系最大的行星"
    assert index_path.exists()
