"""
FastAPI server for the Endee RAG Q&A System
Exposes REST endpoints for document ingestion and question answering.
"""

import os
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_pipeline import RAGPipeline, RAGResponse, RetrievalResult

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Endee RAG Q&A System",
    description=(
        "Retrieval-Augmented Generation API powered by the Endee vector database. "
        "Ingest your documents, then ask questions — get grounded answers with sources."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialise pipeline ───────────────────────────────────────
pipeline = RAGPipeline(
    endee_host=os.getenv("ENDEE_HOST", "localhost"),
    endee_port=int(os.getenv("ENDEE_PORT", "8080")),
    endee_api_key=os.getenv("ENDEE_API_KEY"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    use_mock_embedder=os.getenv("USE_MOCK_EMBEDDER", "false").lower() == "true",
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
)


# ── Pydantic schemas ──────────────────────────────────────────

class IngestTextRequest(BaseModel):
    content: str = Field(..., description="Raw text content to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata tags")


class IngestBatchRequest(BaseModel):
    documents: List[IngestTextRequest]


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional Endee payload filters")


class SourceItem(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    model_used: str


class IngestResponse(BaseModel):
    chunks_stored: int
    message: str


class HealthResponse(BaseModel):
    status: str
    endee_connected: bool
    version: str


# ── Helpers ───────────────────────────────────────────────────

def _format_response(rag_resp: RAGResponse) -> QuestionResponse:
    sources = [
        SourceItem(
            id=r.document.id,
            content=r.document.content[:500],   # truncate for API response
            score=round(r.score, 4),
            metadata=r.document.metadata,
        )
        for r in rag_resp.sources
    ]
    return QuestionResponse(
        question=rag_resp.question,
        answer=rag_resp.answer,
        sources=sources,
        model_used=rag_resp.model_used,
    )


# ── Routes ────────────────────────────────────────────────────

@app.get("/", tags=["Meta"])
def root():
    return {
        "name": "Endee RAG Q&A System",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    connected = pipeline.endee.health_check()
    return HealthResponse(
        status="ok" if connected else "degraded",
        endee_connected=connected,
        version="1.0.0",
    )


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
def ingest_text(req: IngestTextRequest):
    """Ingest a single text document into Endee."""
    try:
        ids = pipeline.ingest_text(req.content, req.metadata)
        return IngestResponse(
            chunks_stored=len(ids),
            message=f"Successfully ingested {len(ids)} chunk(s) into Endee.",
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/batch", response_model=IngestResponse, tags=["Ingestion"])
def ingest_batch(req: IngestBatchRequest):
    """Ingest multiple documents at once."""
    try:
        docs = [{"content": d.content, "metadata": d.metadata or {}} for d in req.documents]
        total = pipeline.ingest_documents(docs)
        return IngestResponse(
            chunks_stored=total,
            message=f"Successfully ingested {total} chunk(s) across {len(docs)} document(s).",
        )
    except Exception as e:
        logger.error(f"Batch ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a plain-text file."""
    try:
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        ids = pipeline.ingest_file(tmp_path)
        os.unlink(tmp_path)

        return IngestResponse(
            chunks_stored=len(ids),
            message=f"File '{file.filename}' ingested as {len(ids)} chunk(s).",
        )
    except Exception as e:
        logger.error(f"File ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QuestionResponse, tags=["Q&A"])
def ask_question(req: QuestionRequest):
    """
    Ask a question. The system retrieves relevant chunks from Endee
    and generates a grounded answer using an LLM.
    """
    try:
        rag_resp = pipeline.ask(req.question, top_k=req.top_k, filters=req.filters)
        return _format_response(rag_resp)
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", tags=["Q&A"])
def retrieve_only(req: QuestionRequest):
    """Retrieve relevant chunks without generating an LLM answer."""
    try:
        results = pipeline.retrieve(req.question, top_k=req.top_k, filters=req.filters)
        return {
            "question": req.question,
            "results": [
                {
                    "id": r.document.id,
                    "content": r.document.content[:500],
                    "score": round(r.score, 4),
                    "metadata": r.document.metadata,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
