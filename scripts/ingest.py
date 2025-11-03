"""CLI helper to rebuild the Chroma index from PDF sources."""

from __future__ import annotations

import argparse
import logging

from app.deps import get_rag_service

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the Chroma vector store from source documents."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full rebuild even if an index already exists.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    rag = get_rag_service()
    LOGGER.info("Starting ingestion (force=%s)", args.force)
    rag.ingest(force_rebuild=args.force)
    LOGGER.info("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
