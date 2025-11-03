from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .deps import get_app_settings, get_rag_service
from .rag.pipeline import RagService
from .schemas import AskRequest, AskResponse
from .settings import Settings

settings = get_app_settings()

app = FastAPI(
    title="Vietnamese RAG Assistant",
    docs_url="/swagger",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mount_static_docs(settings: Settings) -> None:
    if not settings.serve_docs:
        return
    doc_dir: Path = settings.resolved_data_dir
    if not doc_dir.exists():
        return
    mount_path = settings.docs_mount_path.rstrip("/") or "/docs"
    already_mounted = any(
        getattr(route, "path", None) == mount_path for route in app.routes
    )
    if already_mounted:
        return
    app.mount(
        mount_path, StaticFiles(directory=str(doc_dir), check_dir=False), name="docs"
    )


_mount_static_docs(settings)


@app.on_event("startup")
async def configure_app() -> None:
    if settings.auto_ingest_on_startup:
        rag = get_rag_service()
        rag.ensure_vectorstore(force_rebuild=False)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask_endpoint(
    payload: AskRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> AskResponse:
    result = rag.ask(
        question=payload.question,
        additional_context=payload.additional_context,
        top_k=payload.top_k,
        pool_size=payload.pool_size,
        temperature=payload.temperature,
        rerank=payload.rerank,
    )
    return AskResponse.from_chain_result(
        answer=result["answer"],
        citations=result["citations"],
        sources=result["sources"],
    )
