"""
RAG Pipeline using Endee Vector Database
Core retrieval-augmented generation logic
"""

import os
import json
import uuid
import requests
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Data models
# ─────────────────────────────────────────

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    document: Document
    score: float


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[RetrievalResult]
    model_used: str


# ─────────────────────────────────────────
# Endee Client
# ─────────────────────────────────────────

class EndeeClient:
    """
    HTTP client for the Endee Vector Database.
    Wraps index management and vector search endpoints.
    """

    def __init__(self, host: str = "localhost", port: int = 8080, api_key: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Endee health check failed: {e}")
            return False

    def create_index(self, index_name: str, dimension: int, metric: str = "cosine") -> Dict:
        payload = {
            "name": index_name,
            "dimension": dimension,
            "metric": metric,
        }
        resp = requests.post(
            f"{self.base_url}/indexes",
            headers=self.headers,
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def index_exists(self, index_name: str) -> bool:
        resp = requests.get(f"{self.base_url}/indexes/{index_name}", headers=self.headers, timeout=5)
        return resp.status_code == 200

    def upsert_vectors(self, index_name: str, vectors: List[Dict]) -> Dict:
        """
        vectors: list of {"id": str, "values": [...], "payload": {...}}
        """
        payload = {"vectors": vectors}
        resp = requests.post(
            f"{self.base_url}/indexes/{index_name}/upsert",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        payload: Dict[str, Any] = {"vector": query_vector, "top_k": top_k}
        if filters:
            payload["filter"] = filters

        resp = requests.post(
            f"{self.base_url}/indexes/{index_name}/search",
            headers=self.headers,
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def delete_index(self, index_name: str) -> Dict:
        resp = requests.delete(
            f"{self.base_url}/indexes/{index_name}",
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────
# Embedding Service
# ─────────────────────────────────────────

class EmbeddingService:
    """
    Wraps sentence-transformers to generate dense embeddings.
    Falls back to a mock (random) embedder for offline/testing use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_mock: bool = False):
        self.model_name = model_name
        self.use_mock = use_mock
        self.dimension = 384  # default for all-MiniLM-L6-v2
        self._model = None

        if not use_mock:
            self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name} (dim={self.dimension})")
        except ImportError:
            logger.warning("sentence-transformers not installed. Switching to mock embedder.")
            self.use_mock = True

    def embed(self, text: str) -> List[float]:
        if self.use_mock:
            import random
            rng = random.Random(hash(text) % (2**32))
            return [rng.gauss(0, 1) for _ in range(self.dimension)]
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.use_mock:
            return [self.embed(t) for t in texts]
        return [v.tolist() for v in self._model.encode(texts)]


# ─────────────────────────────────────────
# Document Chunker
# ─────────────────────────────────────────

class DocumentChunker:
    """
    Splits documents into overlapping chunks for better retrieval coverage.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc_id: str, content: str, metadata: Dict) -> List[Document]:
        words = content.split()
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_meta = {
                **metadata,
                "chunk_index": chunk_idx,
                "parent_doc_id": doc_id,
            }
            chunks.append(
                Document(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    content=chunk_text,
                    metadata=chunk_meta,
                )
            )
            start += self.chunk_size - self.overlap
            chunk_idx += 1

        return chunks


# ─────────────────────────────────────────
# LLM Answer Generator
# ─────────────────────────────────────────

class LLMAnswerGenerator:
    """
    Calls an OpenAI-compatible chat endpoint to generate answers
    grounded in retrieved context. Works with OpenAI, Groq, Ollama, etc.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url

    def generate(self, question: str, contexts: List[str]) -> str:
        if not self.api_key:
            # Offline fallback: summarise retrieved chunks
            return self._offline_answer(question, contexts)

        context_block = "\n\n---\n\n".join(
            [f"[Source {i+1}]\n{c}" for i, c in enumerate(contexts)]
        )
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question using ONLY "
            "the provided context. If the answer is not in the context, say "
            "'I don't have enough information to answer this question.'"
        )
        user_message = f"Context:\n{context_block}\n\nQuestion: {question}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _offline_answer(self, question: str, contexts: List[str]) -> str:
        """Simple offline answer when no LLM API key is set."""
        combined = " ".join(contexts[:2])
        return (
            f"[Offline mode — no LLM API key set]\n\n"
            f"Based on retrieved documents:\n{combined[:800]}...\n\n"
            f"To get a proper AI-generated answer, set OPENAI_API_KEY in your .env file."
        )


# ─────────────────────────────────────────
# RAG Pipeline (main orchestrator)
# ─────────────────────────────────────────

class RAGPipeline:
    """
    Full retrieval-augmented generation pipeline backed by Endee.

    Flow:
      1. Ingest documents → chunk → embed → upsert into Endee
      2. Query → embed question → search Endee → retrieve top-k chunks
      3. Feed chunks + question to LLM → return grounded answer
    """

    INDEX_NAME = "rag_documents"

    def __init__(
        self,
        endee_host: str = "localhost",
        endee_port: int = 8080,
        endee_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_mock_embedder: bool = False,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-3.5-turbo",
    ):
        self.embedder = EmbeddingService(embedding_model, use_mock=use_mock_embedder)
        self.endee = EndeeClient(endee_host, endee_port, endee_api_key)
        self.chunker = DocumentChunker()
        self.llm = LLMAnswerGenerator(api_key=llm_api_key, model=llm_model)
        self._ensure_index()

    def _ensure_index(self):
        if not self.endee.index_exists(self.INDEX_NAME):
            self.endee.create_index(self.INDEX_NAME, dimension=self.embedder.dimension)
            logger.info(f"Created Endee index '{self.INDEX_NAME}' (dim={self.embedder.dimension})")
        else:
            logger.info(f"Using existing Endee index '{self.INDEX_NAME}'")

    # ── Ingestion ──

    def ingest_text(self, content: str, metadata: Optional[Dict] = None) -> List[str]:
        """Ingest a raw text string. Returns list of chunk IDs stored."""
        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        chunks = self.chunker.chunk(doc_id, content, meta)
        return self._store_chunks(chunks)

    def ingest_file(self, file_path: str) -> List[str]:
        """Ingest a plain-text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        metadata = {"source": filename, "file_path": file_path}
        return self.ingest_text(content, metadata)

    def ingest_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Ingest a list of dicts with 'content' and optional 'metadata' keys.
        Returns total chunks stored.
        """
        total = 0
        for doc in documents:
            ids = self.ingest_text(doc["content"], doc.get("metadata", {}))
            total += len(ids)
        return total

    def _store_chunks(self, chunks: List[Document]) -> List[str]:
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.id,
                "values": emb,
                "payload": {**chunk.metadata, "content": chunk.content},
            })

        self.endee.upsert_vectors(self.INDEX_NAME, vectors)
        logger.info(f"Stored {len(vectors)} chunks into Endee")
        return [c.id for c in chunks]

    # ── Retrieval ──

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        query_embedding = self.embedder.embed(question)
        raw_results = self.endee.search(self.INDEX_NAME, query_embedding, top_k, filters)

        results = []
        for r in raw_results:
            payload = r.get("payload", {})
            content = payload.pop("content", "")
            doc = Document(id=r["id"], content=content, metadata=payload)
            results.append(RetrievalResult(document=doc, score=r.get("score", 0.0)))

        return results

    # ── Full RAG ──

    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> RAGResponse:
        retrieval_results = self.retrieve(question, top_k=top_k, filters=filters)
        contexts = [r.document.content for r in retrieval_results]
        answer = self.llm.generate(question, contexts)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=retrieval_results,
            model_used=self.llm.model,
        )
