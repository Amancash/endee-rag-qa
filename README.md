# 🔍 Endee RAG Q&A System

> **Retrieval-Augmented Generation powered by the [Endee](https://github.com/endee-io/endee) open-source vector database**

[![CI](https://github.com/Amancash/endee-rag-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Amancash/endee-rag-qa/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Endee](https://img.shields.io/badge/vector--db-Endee-orange)](https://endee.io)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [System Design](#-system-design)
- [How Endee is Used](#-how-endee-is-used)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Execution](#-setup--execution)
- [API Reference](#-api-reference)
- [Demo](#-demo)
- [Testing](#-testing)
- [Future Scope](#-future-scope)
- [Author](#-author)

---

## 📖 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that lets users ask natural-language questions over a custom document knowledge base. Instead of relying solely on an LLM's parametric memory, the system retrieves the most relevant document chunks from **Endee** (a high-performance open-source vector database) and uses them as grounded context when generating answers.

The result is:
- ✅ **Accurate answers** — grounded in your actual documents
- ✅ **Source citations** — every answer links back to the chunks it used
- ✅ **Up-to-date knowledge** — just ingest new documents to update the knowledge base
- ✅ **Low hallucination** — the LLM is constrained to the retrieved context

---

## ❗ Problem Statement

Large language models are powerful but have two fundamental limitations:

1. **Static knowledge cutoff** — they don't know about documents or events after their training date.
2. **Hallucination** — they can confidently generate plausible-sounding but incorrect answers.

For enterprises and developers building AI assistants over private knowledge bases (documentation, support articles, research papers, etc.), these limitations are critical.

**Solution:** RAG decouples _retrieval_ from _generation_. A vector database (Endee) retrieves the most relevant passages for any query, and these are passed as context to the LLM — grounding its answer in your actual data.

---

## 🏗 System Design

```
  User Query
      │
      ▼
 ┌──────────────────────────────────────────────────────────┐
 │                    RAG Q&A API (FastAPI)                  │
 │                                                          │
 │  ┌─────────────────────────────────────────────────────┐ │
 │  │                  RAG Pipeline                       │ │
 │  │                                                     │ │
 │  │   1. Embed query  →  sentence-transformers (384-d)  │ │
 │  │   2. Vector search  →  Endee (cosine similarity)   │ │
 │  │   3. Retrieve top-k chunks + metadata              │ │
 │  │   4. Build prompt: [context] + [question]          │ │
 │  │   5. LLM call  →  grounded answer                  │ │
 │  └─────────────────────────────────────────────────────┘ │
 └──────────────────────────────────────────────────────────┘
           ▲                        ▲
           │                        │
  ┌────────┴───────┐      ┌─────────┴──────────┐
  │  Endee Vector  │      │  LLM API           │
  │  Database      │      │  (OpenAI/Groq/      │
  │  :8080         │      │   Ollama)          │
  └────────────────┘      └────────────────────┘
```

### Ingestion Flow
```
Raw Text/File
    │
    ├─▶ Document Chunker  (chunk_size=500 words, overlap=100)
    │         │
    │         ▼
    │   List of Chunks
    │         │
    ├─▶ Embedding Service  (all-MiniLM-L6-v2, 384-dim)
    │         │
    │         ▼
    │   Dense Vectors
    │         │
    └─▶ Endee.upsert()   (stored with payload metadata)
```

### Query Flow
```
Question string
    │
    ├─▶ Embedding Service  →  query vector (384-dim)
    │
    ├─▶ Endee.search()  →  top-k chunks (cosine similarity)
    │
    ├─▶ LLM (with context)  →  grounded answer
    │
    └─▶ Response { answer, sources, scores }
```

---

## 🟠 How Endee is Used

Endee is the **core retrieval layer** of this project. Here is exactly how it is integrated:

| Operation | Endee API Endpoint | Purpose |
|---|---|---|
| Index creation | `POST /indexes` | Creates a `rag_documents` index (384-dim, cosine) |
| Document upsert | `POST /indexes/{name}/upsert` | Stores chunk embeddings + metadata |
| Vector search | `POST /indexes/{name}/search` | Retrieves top-k similar chunks for a query |
| Health check | `GET /health` | Checks if the Endee server is live |
| Filtered search | `POST /indexes/{name}/search` + `filter` key | Metadata-aware retrieval (e.g., by category) |

### Why Endee?

- **Performance** — C++ core with AVX2/AVX512 SIMD optimisations for fast similarity search
- **Payload filtering** — filter by metadata (document category, date, source) during search
- **Open-source & self-hosted** — full control, no cloud vendor lock-in
- **HTTP API** — simple to integrate with any language or framework
- **Hybrid search ready** — sparse vector support for future BM25/hybrid pipelines

### Example: Endee Upsert Payload
```json
{
  "vectors": [
    {
      "id": "doc_abc_chunk_0",
      "values": [0.12, -0.34, 0.56, ...],
      "payload": {
        "content": "Endee is a vector database...",
        "source": "endee_docs",
        "category": "product",
        "chunk_index": 0,
        "parent_doc_id": "doc_abc"
      }
    }
  ]
}
```

### Example: Endee Search with Filter
```json
{
  "vector": [0.12, -0.34, 0.56, ...],
  "top_k": 5,
  "filter": {
    "category": { "$eq": "product" }
  }
}
```

---

## ✨ Features

- 📄 **Multi-format ingestion** — raw text, batch documents, file upload (`.txt`)
- 🔪 **Smart chunking** — overlapping word-level chunking for better retrieval coverage
- 🔢 **Dense embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- 🔍 **Vector search** — cosine similarity via Endee
- 🏷 **Metadata filtering** — filter retrieved chunks by document source, category, etc.
- 🤖 **LLM answer generation** — works with OpenAI, Groq, and local Ollama models
- 🌐 **REST API** — FastAPI with auto-generated Swagger docs at `/docs`
- 🐳 **Docker ready** — single `docker-compose up` starts the full stack
- ✅ **Unit tested** — pytest suite with mocked Endee calls
- 🔁 **CI/CD** — GitHub Actions workflow

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Vector Database | [Endee](https://github.com/endee-io/endee) |
| Embedding Model | `sentence-transformers` — `all-MiniLM-L6-v2` |
| API Framework | FastAPI + Uvicorn |
| LLM Integration | OpenAI API (compatible with Groq, Ollama) |
| Testing | pytest |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
endee-rag-qa/
├── app/
│   ├── __init__.py
│   ├── rag_pipeline.py     # Core RAG logic: embedder, chunker, Endee client, pipeline
│   └── server.py           # FastAPI REST API
├── tests/
│   ├── __init__.py
│   └── test_rag.py         # Unit tests (mocked Endee)
├── docs/
│   └── architecture.md     # Architecture ASCII diagram
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
├── demo.py                 # End-to-end demo script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Setup & Execution

### Prerequisites

- Python 3.11+
- Docker (for Docker deployment)
- A running Endee server (see below)
- (Optional) OpenAI / Groq API key for LLM answers

---

### Step 1 — Start Endee

**Option A: Docker (recommended)**
```bash
docker pull endee-io/endee:latest
docker run -d -p 8080:8080 endee-io/endee:latest
```

**Option B: Local build**
```bash
git clone https://github.com/endee-io/endee
cd endee
chmod +x ./install.sh ./run.sh
./install.sh --release --avx2
./run.sh
```

Endee will be available at `http://localhost:8080`.

---

### Step 2 — Clone this repo
```bash
git clone https://github.com/Amancash/endee-rag-qa.git
cd endee-rag-qa
```

---

### Step 3 — Configure environment
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (or leave blank for offline mode)
```

---

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### Step 5 — Run the demo
```bash
# Quick demo with mock embedder (no GPU or API key required)
python demo.py

# Full demo with real embeddings
python demo.py --real-embedder

# Ask a custom question
python demo.py --question "What is hybrid search?"
```

---

### Step 6 — Start the API server
```bash
uvicorn app.server:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

### Docker Compose (full stack)
```bash
cp .env.example .env   # set your API keys
docker-compose up --build
```

This starts both Endee and the RAG API together.

---

## 📡 API Reference

### `GET /health`
Check system health and Endee connectivity.

### `POST /ingest/text`
Ingest a text document.
```json
{
  "content": "Your document text here...",
  "metadata": { "source": "my_docs", "category": "technical" }
}
```

### `POST /ingest/batch`
Ingest multiple documents at once.
```json
{
  "documents": [
    { "content": "First doc...", "metadata": { "source": "doc1" } },
    { "content": "Second doc...", "metadata": { "source": "doc2" } }
  ]
}
```

### `POST /ingest/file`
Upload a `.txt` file via multipart form.

### `POST /ask`
Ask a question and get a grounded answer.
```json
{
  "question": "What is Endee used for?",
  "top_k": 5,
  "filters": { "category": { "$eq": "product" } }
}
```
**Response:**
```json
{
  "question": "What is Endee used for?",
  "answer": "Endee is a vector database used for AI search, RAG pipelines...",
  "sources": [
    { "id": "chunk_0", "content": "...", "score": 0.94, "metadata": {} }
  ],
  "model_used": "gpt-3.5-turbo"
}
```

### `POST /retrieve`
Retrieve relevant chunks without LLM generation (pure vector search).

---

## 🎬 Demo

```
══════════════════════════════════════════════════════════════════════
  🚀  Endee RAG Q&A System — Demo
══════════════════════════════════════════════════════════════════════

📦 Initialising RAG pipeline...
✅ Endee server is reachable.

📄 Ingesting 6 documents → 12 chunks stored into Endee.

──────────────────────────────────────────────────────────────────────
  ❓  Q1: What is Endee and what is it used for?
──────────────────────────────────────────────────────────────────────

💬 Answer:
Endee is a high-performance open-source vector database designed for
AI search and retrieval workloads. It supports dense vector retrieval,
sparse search for hybrid scenarios, and payload filtering. Common use
cases include RAG pipelines, semantic search, recommendations, and
AI agent memory systems.

📚 Sources (3 retrieved):
  [1] endee_overview (score: 0.9412)
       Endee is a high-performance, open-source vector database designed...
  [2] rag_explanation (score: 0.8103)
       Retrieval-Augmented Generation (RAG) is an AI architecture pattern...
  [3] semantic_search (score: 0.7651)
       Semantic search differs from keyword search by understanding...
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

The test suite covers:
- `EmbeddingService` — mock and batch embedding
- `DocumentChunker` — chunk sizes, overlap, metadata propagation
- `LLMAnswerGenerator` — offline fallback
- `EndeeClient` — mocked HTTP calls (health, index existence, upsert, search)
- `RAGPipeline` — end-to-end ingestion and retrieval with mocked Endee

---

## 🔭 Future Scope

- [ ] **PDF ingestion** using `PyMuPDF` or `pdfplumber`
- [ ] **Hybrid search** — combine Endee dense search with BM25 sparse retrieval
- [ ] **Streaming answers** via Server-Sent Events (SSE)
- [ ] **Conversational memory** — multi-turn chat history stored in Endee
- [ ] **Web UI** — React frontend for document upload and Q&A
- [ ] **Reranking** — cross-encoder reranker for improved precision
- [ ] **Evaluation pipeline** — RAGAS metrics (faithfulness, relevancy, recall)

---

## 👤 Author

**Aman kashyap**  
B.Tech CSE, Galgotias University (2027 Batch)  
GitHub: [@Amancash](https://github.com/Amancash)

---

> Built for the **Endee.io Campus Recruitment Drive** — Galgotias University, 2026.  
> This project demonstrates a production-style RAG system using Endee as the vector retrieval layer.
