"""OSHA regulation retrieval via Qdrant + BGE.

Brief reference: Layer 3 — RAG over OSHA Regulations
Pattern reused from SentinelOps; only the collection name and corpus differ.

Usage:
    from core.rag import OSHARetriever
    retriever = OSHARetriever()
    chunks = retriever.retrieve("hard hat head protection construction", top_k=3)
    for c in chunks:
        print(c.source, c.score, c.text[:200])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION = "safetyvision_osha"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Queries that fire for each detected violation class. The query strings are
# tuned to retrieve the most relevant OSHA standard for each PPE type.
VIOLATION_QUERY_MAP: dict[str, str] = {
    "NO-Hardhat": "head protection hard hat helmet OSHA 1926.100 construction",
    "NO-Safety Vest": "high-visibility safety vest worker visibility traffic",
    "NO-Mask": "respiratory protection respirator mask OSHA 1910.134 1926.103",
    "NO-Gloves": "hand protection gloves OSHA 1910.138",
    "NO-Goggles": "eye face protection goggles OSHA 1910.133 1926.102",
    "No_Harness": "fall protection harness lifeline lanyard OSHA 1926.104 1910.140",
    "Fall-Detected": "fall protection fall arrest OSHA 1926.104 1910.140",
}


@dataclass
class RetrievedChunk:
    text: str
    source: str       # e.g. "1926_100.txt"
    chunk_idx: int
    score: float      # cosine similarity (higher = more relevant)


class OSHARetriever:
    """Singleton-friendly retriever; embedder and Qdrant client are reused."""

    _instance: "OSHARetriever | None" = None

    def __init__(self) -> None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

        logger.info("Loading embedder %s", EMBED_MODEL)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("Connecting to Qdrant at %s", qdrant_url)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)
        # Sanity check the collection exists
        names = [c.name for c in self.client.get_collections().collections]
        if COLLECTION not in names:
            raise RuntimeError(
                f"Collection '{COLLECTION}' not found. Run rag_data/ingest.py first."
            )

    @classmethod
    def get(cls) -> "OSHARetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top-K relevant OSHA chunks for the given natural-language query."""
        # BGE recommends prepending a short instruction for retrieval queries.
        formatted = f"Represent this sentence for searching relevant passages: {query}"
        vec = self.embedder.encode(formatted, normalize_embeddings=True).tolist()
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            RetrievedChunk(
                text=h.payload["text"],
                source=h.payload["source"],
                chunk_idx=h.payload["chunk_idx"],
                score=h.score,
            )
            for h in hits
        ]

    def retrieve_for_violation(
        self, violation_type: str, top_k: int = 3
    ) -> list[RetrievedChunk]:
        """Convenience: map a violation class name to its tuned OSHA query."""
        query = VIOLATION_QUERY_MAP.get(
            violation_type,
            f"OSHA personal protective equipment {violation_type}",
        )
        return self.retrieve(query, top_k=top_k)


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a compact context block for an LLM prompt."""
    if not chunks:
        return "(no relevant OSHA standards retrieved)"
    parts = []
    for i, c in enumerate(chunks, start=1):
        # Strip the .txt and split standard number: "1926_100" → "29 CFR 1926.100"
        std = c.source.replace(".txt", "").replace("_", ".")
        parts.append(
            f"[Context {i} — 29 CFR {std} (similarity={c.score:.2f})]\n{c.text}"
        )
    return "\n\n".join(parts)
