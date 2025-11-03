from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration for the RAG service."""

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    data_dir: Path = Field(Path("Word"), alias="DATA_DIR")
    persist_dir: Path = Field(Path("chroma_db"), alias="PERSIST_DIR")
    embedding_model: str = Field("BAAI/bge-m3", alias="EMBEDDING_MODEL")
    rerank_model: str = Field("BAAI/bge-reranker-base", alias="RERANK_MODEL")
    chat_model: str = Field("gpt-4o-mini", alias="CHAT_MODEL")
    chunk_size: int = Field(1600, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(300, alias="CHUNK_OVERLAP")
    retriever_k: int = Field(25, alias="RETRIEVER_K")
    rerank_top_k: int = Field(4, alias="RERANK_TOP_K")
    serve_docs: bool = Field(True, alias="SERVE_DOCS")
    auto_ingest_on_startup: bool = Field(False, alias="AUTO_INGEST_ON_STARTUP")
    docs_mount_path: str = Field("/docs", alias="DOCS_MOUNT_PATH")
    allowed_origins: list[str] = Field(["*"], alias="ALLOWED_ORIGINS")
    device: Optional[str] = Field(None, alias="TORCH_DEVICE")

    model_config = SettingsConfigDict(
        env_file=(".env", "config/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("data_dir", "persist_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("device", mode="before")
    @classmethod
    def _normalise_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        return value.lower()

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir

    @property
    def resolved_persist_dir(self) -> Path:
        return self.persist_dir

    def ensure_env(self) -> None:
        os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_env()
    settings.resolved_persist_dir.mkdir(parents=True, exist_ok=True)
    return settings
