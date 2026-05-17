"""Tests for agent/tools.py — the tool functions invoked by the LangGraph node.

retrieve_osha_context: wraps OSHARetriever + format_chunks_for_prompt
generate_incident_report: Gemini multimodal call with image downsizing

log_violation is covered in test_forecast.py (SQLite persistence path).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np

from agent import tools as agent_tools
from core.detector import Violation


# ─── retrieve_osha_context ──────────────────────────────────────────────────
class TestRetrieveOshaContext:
    """Wrapper around OSHARetriever.get().retrieve_for_violation(), formatting
    the returned chunks for the LLM prompt."""

    def test_calls_retriever_and_returns_formatted_chunks(self, monkeypatch):
        from core import rag
        from core.rag import RetrievedChunk

        mock_chunks = [
            RetrievedChunk(
                text="Hard hats are required in construction zones.",
                source="1926_100.txt", chunk_idx=0, score=0.91,
            ),
            RetrievedChunk(
                text="Head protection performance criteria.",
                source="1926_100.txt", chunk_idx=1, score=0.85,
            ),
        ]
        fake_retriever = MagicMock()
        fake_retriever.retrieve_for_violation.return_value = mock_chunks
        monkeypatch.setattr(
            rag.OSHARetriever, "get",
            classmethod(lambda cls: fake_retriever),
        )

        result = agent_tools.retrieve_osha_context("NO-Hardhat", top_k=5)

        fake_retriever.retrieve_for_violation.assert_called_once_with(
            "NO-Hardhat", top_k=5,
        )
        # format_chunks_for_prompt renders the OSHA citation + text
        assert "1926.100" in result
        assert "Hard hats are required" in result

    def test_default_top_k_is_three(self, monkeypatch):
        from core import rag

        fake_retriever = MagicMock()
        fake_retriever.retrieve_for_violation.return_value = []
        monkeypatch.setattr(
            rag.OSHARetriever, "get",
            classmethod(lambda cls: fake_retriever),
        )

        agent_tools.retrieve_osha_context("NO-Hardhat")
        fake_retriever.retrieve_for_violation.assert_called_once_with(
            "NO-Hardhat", top_k=3,
        )

    def test_empty_chunks_returns_no_relevant_marker(self, monkeypatch):
        from core import rag

        fake_retriever = MagicMock()
        fake_retriever.retrieve_for_violation.return_value = []
        monkeypatch.setattr(
            rag.OSHARetriever, "get",
            classmethod(lambda cls: fake_retriever),
        )

        result = agent_tools.retrieve_osha_context("NO-Mask")
        assert "no relevant" in result.lower()


# ─── generate_incident_report ───────────────────────────────────────────────
class TestGenerateIncidentReport:
    """Gemini multimodal call. We mock the SDK and inspect what's passed in."""

    @staticmethod
    def _fake_gemini_json() -> str:
        return json.dumps({
            "violation_type": "NO-Hardhat",
            "risk_level": "HIGH",
            "regulation_cited": "OSHA 29 CFR 1926.100(a)",
            "summary": "Worker missing hard hat",
            "corrective_actions": ["Provide approved PPE"],
            "image_observations": "Worker visible in frame.",
        })

    def _attach_capturing_gemini(self, monkeypatch) -> list:
        """Replace genai.GenerativeModel with one that records every
        generate_content call. Returns the captured-args list."""
        captured: list = []

        def capture(contents):
            captured.append(contents)
            return MagicMock(text=self._fake_gemini_json())

        fake_model = MagicMock()
        fake_model.generate_content = capture
        monkeypatch.setattr(
            agent_tools.genai, "GenerativeModel",
            lambda *a, **kw: fake_model,
        )
        return captured

    def test_large_image_downsized_before_gemini(self, monkeypatch):
        """Images > 800px on longest edge must be resized before the API call.

        Exercises the agent/tools.py downsize branch (lines 104-106 in
        coverage report). Real production images are typically 2000+ px;
        the smoke test's 480px image never hit this path."""
        captured = self._attach_capturing_gemini(monkeypatch)

        # 1200×1800 — longest edge 1800, must come down to 800
        large_img = np.zeros((1200, 1800, 3), dtype=np.uint8)
        violation = Violation(
            type="NO-Hardhat", risk_level="HIGH", confidence=0.9,
            bbox=(0, 0, 100, 100), person_bbox=None,
        )

        agent_tools.generate_incident_report(large_img, violation, "context")

        assert len(captured) == 1
        # captured[0] = [user_prompt_str, PIL.Image]
        pil_image = captured[0][1]
        assert max(pil_image.size) <= 800

    def test_small_image_not_resized(self, monkeypatch):
        """Images ≤ 800px on longest edge stay at original size."""
        captured = self._attach_capturing_gemini(monkeypatch)

        # 400×600 ndarray → PIL.Image (width=600, height=400), well under 800
        small_img = np.zeros((400, 600, 3), dtype=np.uint8)
        violation = Violation(
            type="NO-Hardhat", risk_level="HIGH", confidence=0.9,
            bbox=(0, 0, 50, 50), person_bbox=None,
        )

        agent_tools.generate_incident_report(small_img, violation, "context")

        pil_image = captured[0][1]
        # PIL.Image.size is (width, height) — original numpy was (h=400, w=600)
        assert pil_image.size == (600, 400)

    def test_violation_metadata_in_prompt(self, monkeypatch):
        """Violation type, confidence, bbox must appear in the user prompt."""
        captured = self._attach_capturing_gemini(monkeypatch)

        img = np.zeros((400, 400, 3), dtype=np.uint8)
        violation = Violation(
            type="NO-Safety Vest", risk_level="HIGH", confidence=0.876,
            bbox=(10.5, 20.5, 110.5, 220.5), person_bbox=None,
        )

        agent_tools.generate_incident_report(img, violation, "OSHA context here")

        user_prompt = captured[0][0]
        assert "NO-Safety Vest" in user_prompt
        assert "0.876" in user_prompt
        assert "OSHA context here" in user_prompt
