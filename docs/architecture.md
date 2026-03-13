# System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Endee RAG Q&A System                         │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────────────────────────────┐
  │   User /     │     │              RAG API (FastAPI)            │
  │   Client     │────▶│                                          │
  └──────────────┘     │  POST /ingest/text   POST /ask           │
                        │  POST /ingest/file   POST /retrieve      │
                        │  POST /ingest/batch  GET  /health        │
                        └──────────────┬───────────────────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │      RAG Pipeline          │
                         │                           │
                         │  ┌──────────────────────┐ │
                         │  │  Document Chunker    │ │
                         │  │  (overlap chunking)  │ │
                         │  └──────────┬───────────┘ │
                         │             │              │
                         │  ┌──────────▼───────────┐ │
                         │  │  Embedding Service   │ │
                         │  │ (sentence-transformers│ │
                         │  │  all-MiniLM-L6-v2)   │ │
                         │  └──────────┬───────────┘ │
                         └────────────-│──────────────┘
                                       │ vectors
                         ┌─────────────▼─────────────┐
                         │   Endee Vector Database    │
                         │                           │
                         │  • Dense vector index     │
                         │  • Payload filtering      │
                         │  • Cosine similarity      │
                         │  • AVX2/AVX512 optimised  │
                         │  • HTTP API :8080         │
                         └─────────────┬─────────────┘
                                       │ top-k chunks
                         ┌─────────────▼─────────────┐
                         │   LLM Answer Generator    │
                         │  (OpenAI / Groq / Ollama) │
                         │                           │
                         │  Prompt = context + query │
                         │  → grounded answer        │
                         └───────────────────────────┘
```
