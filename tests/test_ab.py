"""Tests for evaluation/ab_test.py.

Heavy paths (run_prompt_ab end-to-end, run_threshold_ab over the Kaggle zip)
are intentionally NOT tested here — they require live Gemini + the dataset
zip. We cover the pure helpers and mock the Groq judge call.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from evaluation import ab_test


# ─── Cohen's d interpretation ───────────────────────────────────────────────
class TestInterpretCohensD:
    @pytest.mark.parametrize("d,expected", [
        (0.0, "negligible"),
        (0.1, "negligible"),
        (0.3, "small"),
        (0.6, "medium"),
        (1.0, "large"),
        (-0.3, "small"),     # interpretation uses absolute value
        (-0.6, "medium"),
        (-1.5, "large"),
    ])
    def test_buckets(self, d, expected):
        assert ab_test._interpret_cohens_d(d) == expected


# ─── _compute_stats ─────────────────────────────────────────────────────────
class TestComputeStats:
    def test_too_few_cases_returns_error(self):
        per_case = [
            {"case_id": "c1", "score_a": 4.0, "score_b": 3.0},
            {"case_id": "c2", "score_a": 4.5, "score_b": 3.5},
        ]
        result = ab_test._compute_stats(per_case)
        assert "error" in result
        assert result["n_cases"] == 2

    def test_clear_with_rag_winner(self):
        # A always slightly higher than B → with-RAG wins
        per_case = [
            {"case_id": f"c{i}", "score_a": 4.0 + (i % 3) * 0.1, "score_b": 3.0 + (i % 2) * 0.1}
            for i in range(10)
        ]
        result = ab_test._compute_stats(per_case)
        assert result["n_cases"] == 10
        assert result["variant_a_mean"] > result["variant_b_mean"]
        assert result["winner"] == "with-RAG"
        assert "p_value" in result
        assert "cohens_d" in result
        assert "effect_size_interp" in result
        assert "significant_at_0.05" in result

    def test_b_winning_flips_winner_label(self):
        per_case = [
            {"case_id": f"c{i}", "score_a": 2.0 + (i % 3) * 0.1, "score_b": 4.0 + (i % 2) * 0.1}
            for i in range(8)
        ]
        result = ab_test._compute_stats(per_case)
        assert result["winner"] == "without-RAG"

    def test_tie(self):
        per_case = [
            {"case_id": f"c{i}", "score_a": 4.0, "score_b": 4.0}
            for i in range(5)
        ]
        result = ab_test._compute_stats(per_case)
        assert result["mean_diff"] == 0.0
        assert result["winner"] == "tie"

    def test_per_case_round_trips(self):
        per_case = [
            {"case_id": f"c{i}", "score_a": 4.0 + i * 0.1, "score_b": 3.0}
            for i in range(5)
        ]
        result = ab_test._compute_stats(per_case)
        assert len(result["per_case"]) == 5
        assert result["per_case"][0]["case_id"] == "c0"


# ─── _label_has_violation (YOLO label-file parsing) ─────────────────────────
class TestLabelHasViolation:
    def test_empty_string(self):
        assert ab_test._label_has_violation("") is False

    def test_only_non_violation_classes(self):
        # class 1 and 2 are NOT in VIOLATION_CLASS_IDS (which is {0, 5, 6, 7, 8, 9, 10})
        assert ab_test._label_has_violation(
            "1 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1"
        ) is False

    def test_class_0_is_violation(self):
        assert ab_test._label_has_violation("0 0.5 0.5 0.1 0.1") is True

    def test_class_5_is_violation(self):
        assert ab_test._label_has_violation("5 0.5 0.5 0.1 0.1") is True

    def test_malformed_line_skipped(self):
        # "not-a-number" doesn't parse as int → skipped, then class 0 found
        assert ab_test._label_has_violation(
            "not-a-number\n0 0.5 0.5 0.1 0.1"
        ) is True

    def test_blank_lines_skipped(self):
        assert ab_test._label_has_violation("\n\n6 0.5 0.5 0.1 0.1\n") is True


# ─── _save_partial / _load_done_cases ───────────────────────────────────────
class TestSaveAndLoadPartial:
    def test_roundtrip(self, tmp_path, monkeypatch):
        results_path = tmp_path / "prompt_variant.json"
        monkeypatch.setattr(ab_test, "PROMPT_RESULTS", results_path)

        per_case = [
            {"case_id": "c1", "score_a": 4.0, "score_b": 3.0},
            {"case_id": "c2", "score_a": 4.5, "score_b": 3.5},
            {"case_id": "c3", "score_a": 4.2, "score_b": 3.2},
        ]
        ab_test._save_partial(per_case)
        assert results_path.exists()

        loaded = ab_test._load_done_cases()
        assert set(loaded.keys()) == {"c1", "c2", "c3"}

    def test_load_returns_empty_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ab_test, "PROMPT_RESULTS", tmp_path / "missing.json")
        assert ab_test._load_done_cases() == {}

    def test_load_handles_corrupt_json(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{")
        monkeypatch.setattr(ab_test, "PROMPT_RESULTS", bad)
        assert ab_test._load_done_cases() == {}


# ─── judge_pair (Groq client mocked) ────────────────────────────────────────
class TestJudgePairMocked:
    def test_returns_parsed_dict(self, monkeypatch):
        canned = {
            "report_a": {"regulation_accuracy": 5, "action_relevance": 4, "faithfulness": 5},
            "report_b": {"regulation_accuracy": 3, "action_relevance": 3, "faithfulness": 3},
            "rationale": "A cites correct CFR, B is generic.",
        }
        fake_msg = MagicMock(content=json.dumps(canned))
        fake_choice = MagicMock(message=fake_msg)
        fake_resp = MagicMock(choices=[fake_choice])

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_resp
        monkeypatch.setattr(ab_test, "_groq_client", fake_client)

        result = ab_test.judge_pair(
            expected_reg="OSHA 29 CFR 1926.100(a)",
            expected_risk="HIGH",
            report_a={"summary": "A"},
            report_b={"summary": "B"},
        )
        assert result["report_a"]["regulation_accuracy"] == 5
        assert result["report_b"]["regulation_accuracy"] == 3

        # Confirm we asked Groq for a JSON object response
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["model"] == ab_test.GROQ_MODEL
