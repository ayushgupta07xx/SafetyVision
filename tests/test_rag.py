"""Tests for core/rag.py::OSHARetriever.

QdrantClient and SentenceTransformer are mocked entirely — no network calls,
no model downloads. The OSHARetriever._instance singleton is reset before
each test so init paths can be re-exercised.

(format_chunks_for_prompt has its own tests in test_detector.py.)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core import rag


# ─── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def reset_singleton():
    """Each test gets a fresh OSHARetriever singleton."""
    rag.OSHARetriever._instance = None
    yield
    rag.OSHARetriever._instance = None


@pytest.fixture
def env_set(monkeypatch):
    """Provide valid-looking Qdrant credentials so __init__ doesn't raise."""
    monkeypatch.setenv("QDRANT_URL", "https://fake.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "fake-key")


@pytest.fixture
def mock_embedder(monkeypatch):
    """Replace SentenceTransformer with a stub that returns a fixed vector."""
    fake = MagicMock()
    fake.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    monkeypatch.setattr(rag, "SentenceTransformer", lambda *a, **kw: fake)
    return fake


@pytest.fixture
def mock_qdrant(monkeypatch):
    """Replace QdrantClient with a stub that fakes a healthy collection and
    returns two synthetic hits on query_points."""
    fake_collections = SimpleNamespace(
        collections=[SimpleNamespace(name=rag.COLLECTION)]
    )
    fake_hits = SimpleNamespace(points=[
        SimpleNamespace(
            payload={"text": "Hard hats are required.",
                     "source": "1926_100.txt", "chunk_idx": 0},
            score=0.92,
        ),
        SimpleNamespace(
            payload={"text": "Vests must be high-visibility.",
                     "source": "1926_201.txt", "chunk_idx": 1},
            score=0.78,
        ),
    ])
    fake_client = MagicMock()
    fake_client.get_collections.return_value = fake_collections
    fake_client.query_points.return_value = fake_hits

    monkeypatch.setattr(rag, "QdrantClient", lambda *a, **kw: fake_client)
    return fake_client


# ─── __init__ failure modes ─────────────────────────────────────────────────
class TestInitFailures:
    def test_missing_url_raises(self, monkeypatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "key")
        with pytest.raises(RuntimeError, match="QDRANT"):
            rag.OSHARetriever()

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "https://fake.qdrant.io")
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="QDRANT"):
            rag.OSHARetriever()

    def test_missing_collection_raises(self, env_set, mock_embedder, monkeypatch):
        # QdrantClient returns no collections → init must raise
        fake_client = MagicMock()
        fake_client.get_collections.return_value = SimpleNamespace(collections=[])
        monkeypatch.setattr(rag, "QdrantClient", lambda *a, **kw: fake_client)
        with pytest.raises(RuntimeError, match="not found"):
            rag.OSHARetriever()


# ─── __init__ success ───────────────────────────────────────────────────────
class TestInitSuccess:
    def test_constructs_with_mocked_deps(self, env_set, mock_embedder, mock_qdrant):
        r = rag.OSHARetriever()
        assert r.embedder is mock_embedder
        assert r.client is mock_qdrant


# ─── get() singleton ────────────────────────────────────────────────────────
class TestGetSingleton:
    def test_get_returns_same_instance_on_repeat_call(
        self, env_set, mock_embedder, mock_qdrant,
    ):
        r1 = rag.OSHARetriever.get()
        r2 = rag.OSHARetriever.get()
        assert r1 is r2

    def test_get_initializes_lazily(self, env_set, mock_embedder, mock_qdrant):
        assert rag.OSHARetriever._instance is None
        rag.OSHARetriever.get()
        assert rag.OSHARetriever._instance is not None


# ─── retrieve() ─────────────────────────────────────────────────────────────
class TestRetrieve:
    def test_returns_retrieved_chunks(self, env_set, mock_embedder, mock_qdrant):
        r = rag.OSHARetriever()
        chunks = r.retrieve("hard hat construction", top_k=2)
        assert len(chunks) == 2
        assert isinstance(chunks[0], rag.RetrievedChunk)
        assert chunks[0].text == "Hard hats are required."
        assert chunks[0].source == "1926_100.txt"
        assert chunks[0].chunk_idx == 0
        assert chunks[0].score == pytest.approx(0.92)

    def test_query_prefixed_with_bge_instruction(
        self, env_set, mock_embedder, mock_qdrant,
    ):
        # BGE expects "Represent this sentence for searching..." prefix
        r = rag.OSHARetriever()
        r.retrieve("hard hat", top_k=3)
        formatted = mock_embedder.encode.call_args.args[0]
        assert formatted.startswith(
            "Represent this sentence for searching relevant passages: "
        )
        assert "hard hat" in formatted

    def test_top_k_passed_through_to_qdrant(
        self, env_set, mock_embedder, mock_qdrant,
    ):
        r = rag.OSHARetriever()
        r.retrieve("query", top_k=7)
        kwargs = mock_qdrant.query_points.call_args.kwargs
        assert kwargs["limit"] == 7
        assert kwargs["collection_name"] == rag.COLLECTION
        assert kwargs["with_payload"] is True

    def test_embedder_called_with_normalize_flag(
        self, env_set, mock_embedder, mock_qdrant,
    ):
        # BGE requires L2-normalized vectors for cosine similarity
        r = rag.OSHARetriever()
        r.retrieve("query")
        kwargs = mock_embedder.encode.call_args.kwargs
        assert kwargs.get("normalize_embeddings") is True


# ─── retrieve_for_violation() ───────────────────────────────────────────────
class TestRetrieveForViolation:
    def test_known_violation_uses_tuned_query(
        self, env_set, mock_embedder, mock_qdrant,
    ):
        r = rag.OSHARetriever()
        r.retrieve_for_violation("NO-Hardhat")
        formatted = mock_embedder.encode.call_args.args[0]
        # The tuned NO-Hardhat query mentions head protection and OSHA 1926.100
        assert "head protection" in formatted
        assert "1926.100" in formatted

    def test_unknown_violation_falls_back_to_generic_query(
        self, env_set, mock_embedder, mock_qdrant,
    ):
        r = rag.OSHARetriever()
        r.retrieve_for_violation("NO-Earplugs")  # not in VIOLATION_QUERY_MAP
        formatted = mock_embedder.encode.call_args.args[0]
        assert "NO-Earplugs" in formatted
        assert "OSHA personal protective equipment" in formatted

    def test_default_top_k_is_three(
        self, env_set, mock_embedder, mock_qdrant,
    ):
        r = rag.OSHARetriever()
        r.retrieve_for_violation("NO-Hardhat")
        kwargs = mock_qdrant.query_points.call_args.kwargs
        assert kwargs["limit"] == 3


# ─── VIOLATION_QUERY_MAP coverage of YOLO classes ───────────────────────────
class TestQueryMapCompleteness:
    @pytest.mark.parametrize("cls", [
        "NO-Hardhat", "NO-Safety Vest", "NO-Mask",
        "NO-Gloves", "NO-Goggles", "No_Harness", "Fall-Detected",
    ])
    def test_yolo_class_has_tuned_query(self, cls):
        assert cls in rag.VIOLATION_QUERY_MAP
        # Tuned queries reference an OSHA citation or specific PPE term
        query = rag.VIOLATION_QUERY_MAP[cls]
        assert len(query) > 15
