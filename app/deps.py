from __future__ import annotations

from functools import lru_cache

from .rag.pipeline import RagService
from .settings import Settings, get_settings


@lru_cache()
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(settings)


def get_app_settings() -> Settings:
    return get_settings()
