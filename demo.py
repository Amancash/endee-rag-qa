#!/usr/bin/env python3
"""
demo.py — End-to-end demonstration of the Endee RAG Q&A System

Run this script to see the full RAG pipeline in action:
  1. Ingest sample documents into Endee
  2. Ask natural-language questions
  3. Receive answers grounded in the ingested content

Usage:
    python demo.py                         # uses mock embedder (no GPU required)
    python demo.py --real-embedder         # uses sentence-transformers
    python demo.py --question "Your Q"    # ask a custom question
"""

import argparse
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ── Sample knowledge base ──────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "content": """
        Endee is a high-performance, open-source vector database designed for AI search
        and retrieval workloads. It is built in C++ and optimized for modern CPU targets
        including AVX2, AVX512, NEON, and SVE2. Endee supports dense vector retrieval,
        sparse search for hybrid search scenarios, and payload filtering for
        metadata-aware queries. It exposes an HTTP API and can be deployed locally
        via install scripts, manually compiled, or run inside Docker containers.
        The server listens on port 8080 by default.
        """,
        "metadata": {"source": "endee_overview", "category": "product"},
    },
    {
        "content": """
        Retrieval-Augmented Generation (RAG) is an AI architecture pattern that
        combines a retrieval system with a generative language model. Instead of
        relying solely on the model's parametric knowledge, RAG fetches relevant
        documents from an external knowledge store at query time and passes them as
        context to the LLM. This reduces hallucination, keeps answers up to date,
        and allows citing sources. Endee acts as the retrieval layer: documents are
        embedded and stored in Endee; at query time the question is embedded and a
        vector search retrieves the most relevant chunks.
        """,
        "metadata": {"source": "rag_explanation", "category": "concept"},
    },
    {
        "content": """
        Semantic search differs from keyword search by understanding the meaning of
        a query rather than matching exact terms. A user searching for "affordable
        smartphones" will find results mentioning "budget phones" or "low-cost mobile
        devices" even if those exact words don't appear in the query. Vector databases
        like Endee make semantic search possible by storing text as dense float
        vectors (embeddings). Cosine similarity or dot-product distance between
        the query embedding and stored embeddings determines relevance ranking.
        """,
        "metadata": {"source": "semantic_search", "category": "concept"},
    },
    {
        "content": """
        Hybrid search combines dense vector retrieval with sparse retrieval (BM25 or
        similar) to achieve better relevance than either technique alone. Dense
        retrieval captures semantic similarity; sparse retrieval captures exact
        keyword matches, which matter in domains with specific terminology, model
        numbers, or proper nouns. Endee supports sparse vector capabilities
        documented in its sparse search guide, enabling hybrid retrieval pipelines
        within a single database deployment.
        """,
        "metadata": {"source": "hybrid_search", "category": "feature"},
    },
    {
        "content": """
        Payload filtering in Endee allows queries to apply structured conditions on
        document metadata alongside vector similarity. For example, you can search
        for the most similar product embeddings that also have category='electronics'
        and price_usd less than 500. Filters are evaluated during the vector search,
        not as a post-processing step, which keeps latency low even on large indexes.
        This is documented in the Endee filter guide.
        """,
        "metadata": {"source": "payload_filtering", "category": "feature"},
    },
    {
        "content": """
        AI agents are autonomous software systems that perceive their environment,
        reason about goals, and take sequences of actions. Long-term memory is a key
        challenge: agents need to store and retrieve past observations, tool outputs,
        and user preferences across sessions. Endee can serve as the long-term memory
        store for agents built with frameworks such as LangChain, CrewAI, AutoGen,
        and LlamaIndex. Memories are stored as embeddings; at each step the agent
        retrieves the most relevant past context using vector search.
        """,
        "metadata": {"source": "agent_memory", "category": "use_case"},
    },
]

SAMPLE_QUESTIONS = [
    "What is Endee and what is it used for?",
    "How does retrieval-augmented generation work?",
    "What is the difference between semantic search and keyword search?",
    "How does Endee support hybrid search?",
    "Can I filter search results by metadata in Endee?",
    "How can AI agents use Endee for memory?",
]


def print_divider(char: str = "─", width: int = 70):
    print(char * width)


def run_demo(use_mock: bool = True, custom_question: str = None):
    print_divider("═")
    print("  🚀  Endee RAG Q&A System — Demo")
    print_divider("═")

    # ── Initialise pipeline ─────────────────────────────────────────────────
    print("\n📦 Initialising RAG pipeline...")
    pipeline = RAGPipeline(
        endee_host=os.getenv("ENDEE_HOST", "localhost"),
        endee_port=int(os.getenv("ENDEE_PORT", "8080")),
        endee_api_key=os.getenv("ENDEE_API_KEY"),
        use_mock_embedder=use_mock,
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
    )

    # ── Health check ─────────────────────────────────────────────────────────
    connected = pipeline.endee.health_check()
    if not connected:
        print("\n⚠️  Warning: Could not reach Endee server at localhost:8080")
        print("   Make sure Endee is running: ./install.sh --release --avx2 && ./run.sh")
        print("   Continuing demo (ingestion/search calls will fail if Endee is down).\n")
    else:
        print("✅ Endee server is reachable.\n")

    # ── Ingest documents ─────────────────────────────────────────────────────
    print("📄 Ingesting sample knowledge base into Endee...")
    total_chunks = pipeline.ingest_documents(SAMPLE_DOCUMENTS)
    print(f"✅ Ingested {len(SAMPLE_DOCUMENTS)} documents → {total_chunks} chunks stored.\n")

    # ── Q&A loop ─────────────────────────────────────────────────────────────
    questions = [custom_question] if custom_question else SAMPLE_QUESTIONS

    for i, question in enumerate(questions, 1):
        print_divider()
        print(f"  ❓  Q{i}: {question}")
        print_divider()

        response = pipeline.ask(question, top_k=3)

        print(f"\n💬 Answer:\n{response.answer}\n")
        print(f"📚 Sources ({len(response.sources)} retrieved):")
        for j, src in enumerate(response.sources, 1):
            print(f"  [{j}] {src.document.metadata.get('source', 'unknown')} "
                  f"(score: {src.score:.4f})")
            print(f"       {src.document.content[:120].strip()}...")
        print()

    print_divider("═")
    print("  ✅  Demo complete!")
    print_divider("═")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endee RAG Q&A Demo")
    parser.add_argument(
        "--real-embedder",
        action="store_true",
        help="Use sentence-transformers instead of mock embedder",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Ask a custom question instead of running the default set",
    )
    args = parser.parse_args()

    run_demo(use_mock=not args.real_embedder, custom_question=args.question)
