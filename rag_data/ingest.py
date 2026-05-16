"""Chunk OSHA corpus + embed with BGE + upsert to Qdrant Cloud.

Reads .txt files from rag_data/osha_corpus/ (produced by scrape_osha.py),
chunks them at ~512 tokens with 50-token overlap, embeds with
BAAI/bge-small-en-v1.5 (CPU, free), upserts to a Qdrant Cloud collection.

Brief reference: Layer 3 — RAG over OSHA Regulations
Same retrieval pattern as SentinelOps; only the corpus and collection differ.

Idempotent: clears the collection before re-ingestion (uses recreate).
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = logging.getLogger(__name__)

CORPUS_DIR = Path("rag_data/osha_corpus")
COLLECTION = "safetyvision_osha"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # bge-small-en-v1.5 output dim
CHUNK_SIZE_CHARS = 1800     # ~512 tokens at avg 3.5 chars/token
CHUNK_OVERLAP_CHARS = 200   # ~50 token overlap
UPSERT_BATCH_SIZE = 100


def chunk_text(text: str, source: str) -> list[dict]:
    """Char-based chunking with overlap; prefers paragraph/sentence boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks: list[dict] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE_CHARS, len(text))
        # If not at EOT, try to break at a paragraph or sentence boundary
        # within the last 200 chars (avoids splitting mid-sentence).
        if end < len(text):
            window_start = max(start, end - 200)
            window = text[window_start:end]
            for boundary in ["\n\n", ". ", "\n"]:
                b_idx = window.rfind(boundary)
                if b_idx != -1:
                    end = window_start + b_idx + len(boundary)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source, "chunk_idx": idx})
            idx += 1
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP_CHARS
    return chunks


def ensure_collection(client: QdrantClient) -> None:
    """Recreate the collection (idempotent re-ingestion)."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION in collections:
        client.delete_collection(COLLECTION)
        logger.info("Deleted existing collection %s", COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    logger.info("Created collection %s (dim=%d, cosine)", COLLECTION, EMBED_DIM)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.glob("*.txt")):
        raise SystemExit(
            f"No corpus in {CORPUS_DIR}. Run rag_data/scrape_osha.py first."
        )

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_key:
        raise SystemExit("QDRANT_URL and QDRANT_API_KEY must be set in .env")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)
    ensure_collection(client)

    logger.info("Loading embedding model %s (downloads on first run)", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Gather chunks across the corpus
    all_chunks: list[dict] = []
    for txt_file in sorted(CORPUS_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        file_chunks = chunk_text(text, source=txt_file.name)
        all_chunks.extend(file_chunks)
        logger.info("%s: %d chunks", txt_file.name, len(file_chunks))
    logger.info("Total chunks: %d", len(all_chunks))

    # Embed (BGE recommends normalize_embeddings=True for cosine similarity)
    logger.info("Embedding %d chunks...", len(all_chunks))
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )

    # Upsert in batches
    logger.info("Upserting to Qdrant collection %s...", COLLECTION)
    for i in range(0, len(all_chunks), UPSERT_BATCH_SIZE):
        batch = all_chunks[i : i + UPSERT_BATCH_SIZE]
        batch_emb = embeddings[i : i + UPSERT_BATCH_SIZE]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text": c["text"],
                    "source": c["source"],
                    "chunk_idx": c["chunk_idx"],
                },
            )
            for c, emb in zip(batch, batch_emb)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        logger.info("  upserted %d/%d", min(i + UPSERT_BATCH_SIZE, len(all_chunks)), len(all_chunks))

    # Sanity check: count points in the collection
    info = client.get_collection(COLLECTION)
    logger.info("Done. %s now has %d points.", COLLECTION, info.points_count)


if __name__ == "__main__":
    main()
