"""
Tests for the Endee RAG Q&A System.

Run: pytest tests/ -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from app.rag_pipeline import (
    RAGPipeline,
    EndeeClient,
    EmbeddingService,
    DocumentChunker,
    LLMAnswerGenerator,
    Document,
    RetrievalResult,
)


# ── EmbeddingService ─────────────────────────────────────────────────────────

class TestEmbeddingService:
    def test_mock_embed_returns_list(self):
        embedder = EmbeddingService(use_mock=True)
        result = embedder.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == embedder.dimension

    def test_mock_embed_deterministic(self):
        embedder = EmbeddingService(use_mock=True)
        r1 = embedder.embed("test text")
        r2 = embedder.embed("test text")
        assert r1 == r2

    def test_mock_embed_batch(self):
        embedder = EmbeddingService(use_mock=True)
        texts = ["first", "second", "third"]
        results = embedder.embed_batch(texts)
        assert len(results) == 3
        for r in results:
            assert len(r) == embedder.dimension

    def test_different_texts_produce_different_embeddings(self):
        embedder = EmbeddingService(use_mock=True)
        e1 = embedder.embed("apple")
        e2 = embedder.embed("banana")
        assert e1 != e2


# ── DocumentChunker ──────────────────────────────────────────────────────────

class TestDocumentChunker:
    def test_short_text_produces_one_chunk(self):
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk("doc1", "short text", {"source": "test"})
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        chunker = DocumentChunker(chunk_size=10, overlap=2)
        long_text = " ".join(["word"] * 50)
        chunks = chunker.chunk("doc2", long_text, {})
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        chunker = DocumentChunker(chunk_size=10, overlap=2)
        long_text = " ".join(["word"] * 50)
        chunks = chunker.chunk("doc3", long_text, {})
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_propagated_to_chunks(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk("doc4", "some content here", {"author": "Alice"})
        for chunk in chunks:
            assert chunk.metadata["author"] == "Alice"
            assert chunk.metadata["parent_doc_id"] == "doc4"

    def test_overlap_creates_more_chunks(self):
        text = " ".join(["word"] * 100)
        chunker_no_overlap = DocumentChunker(chunk_size=20, overlap=0)
        chunker_with_overlap = DocumentChunker(chunk_size=20, overlap=10)
        chunks_no = chunker_no_overlap.chunk("d", text, {})
        chunks_with = chunker_with_overlap.chunk("d", text, {})
        assert len(chunks_with) >= len(chunks_no)


# ── LLMAnswerGenerator ───────────────────────────────────────────────────────

class TestLLMAnswerGenerator:
    def test_offline_mode_returns_string(self):
        gen = LLMAnswerGenerator(api_key=None)
        answer = gen.generate("What is Endee?", ["Endee is a vector database."])
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_offline_mode_contains_context(self):
        gen = LLMAnswerGenerator(api_key=None)
        answer = gen.generate("What is X?", ["X is a great tool for AI."])
        assert "X is a great tool" in answer or "Offline" in answer


# ── EndeeClient (mocked HTTP) ────────────────────────────────────────────────

class TestEndeeClient:
    def _make_client(self):
        return EndeeClient(host="localhost", port=8080)

    def test_health_check_success(self):
        client = self._make_client()
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert client.health_check() is True

    def test_health_check_failure(self):
        client = self._make_client()
        with patch("requests.get", side_effect=Exception("refused")):
            assert client.health_check() is False

    def test_index_exists_true(self):
        client = self._make_client()
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert client.index_exists("my_index") is True

    def test_index_exists_false(self):
        client = self._make_client()
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            assert client.index_exists("missing_index") is False


# ── RAGPipeline (integration, mocked Endee) ───────────────────────────────────

class TestRAGPipeline:
    def _build_pipeline(self):
        """Build pipeline with all external calls mocked."""
        with patch.object(EndeeClient, "index_exists", return_value=True):
            pipeline = RAGPipeline(use_mock_embedder=True)
        return pipeline

    def test_pipeline_initialises(self):
        pipeline = self._build_pipeline()
        assert pipeline.embedder is not None
        assert pipeline.endee is not None
        assert pipeline.chunker is not None

    def test_ingest_text_calls_upsert(self):
        pipeline = self._build_pipeline()
        with patch.object(pipeline.endee, "upsert_vectors", return_value={}):
            ids = pipeline.ingest_text("Hello world, this is test content.", {"source": "unit_test"})
            assert isinstance(ids, list)
            assert len(ids) >= 1

    def test_retrieve_returns_results(self):
        pipeline = self._build_pipeline()
        mock_search_results = [
            {"id": "chunk_1", "score": 0.95, "payload": {"content": "Endee is a vector db.", "source": "test"}},
            {"id": "chunk_2", "score": 0.88, "payload": {"content": "RAG uses vector search.", "source": "test"}},
        ]
        with patch.object(pipeline.endee, "search", return_value=mock_search_results):
            results = pipeline.retrieve("What is Endee?", top_k=2)
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].document.content == "Endee is a vector db."

    def test_ask_returns_rag_response(self):
        pipeline = self._build_pipeline()
        mock_search_results = [
            {"id": "chunk_1", "score": 0.9, "payload": {"content": "Endee stores vectors.", "source": "s"}},
        ]
        with patch.object(pipeline.endee, "search", return_value=mock_search_results):
            response = pipeline.ask("What does Endee store?", top_k=1)

        assert response.question == "What does Endee store?"
        assert isinstance(response.answer, str)
        assert len(response.sources) == 1

    def test_ingest_documents_batch(self):
        pipeline = self._build_pipeline()
        docs = [
            {"content": "Document one content here.", "metadata": {"id": "1"}},
            {"content": "Document two content here.", "metadata": {"id": "2"}},
        ]
        with patch.object(pipeline.endee, "upsert_vectors", return_value={}):
            total = pipeline.ingest_documents(docs)
        assert total >= 2
