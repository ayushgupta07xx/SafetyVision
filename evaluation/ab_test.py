"""A/B tests for SafetyVision (resumable v2).

Test 1 — Prompt variants (with-RAG vs without-RAG):
    Resumable. Saves per_case after every case; on rerun, skips cases already
    present in evaluation/ab_results/prompt_variant.json. This handles the
    Gemini free-tier daily quota (20 requests/day on gemini-flash-latest).
    Statistical test: paired t-test. Effect size: Cohen's d.

Test 2 — Threshold A/B (conf=0.40 vs conf=0.55):
    Random sample of test_split images from the Kaggle zip (default N=200).
    McNemar's test (exact binomial on discordant pairs).

Outputs:
    evaluation/ab_results/prompt_variant.json   (incrementally saved)
    evaluation/ab_results/threshold.json
    MLflow runs under experiment 'safetyvision-ab-tests'

Usage:
    python -m evaluation.ab_test                       # both tests
    python -m evaluation.ab_test --skip-prompt
    python -m evaluation.ab_test --skip-threshold
    python -m evaluation.ab_test --reset-prompt        # ignore prior partial results
    python -m evaluation.ab_test --threshold-n 500
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import warnings
import zipfile
from pathlib import Path

import cv2
import mlflow
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from scipy import stats
from tqdm import tqdm

from agent.tools import generate_incident_report, retrieve_osha_context
from core.detector import (
    VIOLATION_CLASSES,
    PPEDetector,
    Violation,
)

warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

GOLDEN_CASES = Path("evaluation/golden_set/cases.json")
RESULTS_DIR = Path("evaluation/ab_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_RESULTS = RESULTS_DIR / "prompt_variant.json"
ZIP_PATH = Path("/mnt/e/_output_.zip")

GROQ_MODEL = "llama-3.3-70b-versatile"
VIOLATION_CLASS_IDS = {0, 5, 6, 7, 8, 9, 10}

_groq_client: Groq | None = None


def groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY missing from .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ─── Test 1: Prompt A/B (with-RAG vs without-RAG) ───────────────────────────

JUDGE_SYSTEM = """You are an expert OSHA compliance officer evaluating two AI-generated incident reports for the same workplace safety violation.

Score each report on these three dimensions, each 1-5 (5 is best):
1. regulation_accuracy: Does the cited regulation match the EXPECTED OSHA citation provided to you? Exact match = 5; same Part but different section = 3; wrong Part or hallucinated = 1.
2. action_relevance: Are the corrective actions specific, actionable, and appropriate for the violation type? Generic boilerplate = 2; vague but on-topic = 3; specific and concrete = 5.
3. faithfulness: Does the report stick to verifiable facts that match the violation type? Hallucinated details = 1; faithful to violation = 5.

Return ONLY valid JSON matching this exact schema (no prose, no markdown):
{
  "report_a": {"regulation_accuracy": <int 1-5>, "action_relevance": <int 1-5>, "faithfulness": <int 1-5>},
  "report_b": {"regulation_accuracy": <int 1-5>, "action_relevance": <int 1-5>, "faithfulness": <int 1-5>},
  "rationale": "<1-2 sentence explanation>"
}"""


def judge_pair(expected_reg: str, expected_risk: str, report_a: dict, report_b: dict) -> dict:
    user_msg = f"""EXPECTED OSHA regulation: {expected_reg}
EXPECTED risk level: {expected_risk}

REPORT A:
{json.dumps(report_a, indent=2)}

REPORT B:
{json.dumps(report_b, indent=2)}

Score both. Return ONLY valid JSON."""
    resp = groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _compute_stats(per_case: list[dict]) -> dict:
    """Compute paired stats over a list of {score_a, score_b} dicts."""
    scores_a = np.array([c["score_a"] for c in per_case])
    scores_b = np.array([c["score_b"] for c in per_case])
    diffs = scores_a - scores_b
    if len(scores_a) < 3:
        return {
            "error": f"Only {len(scores_a)} cases — need ≥3 for statistics.",
            "n_cases": len(scores_a),
            "per_case": per_case,
        }
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    std_diff = float(diffs.std(ddof=1))
    cohens_d = float(diffs.mean() / std_diff) if std_diff > 0 else 0.0
    return {
        "test": "paired t-test (with-RAG vs without-RAG)",
        "n_cases": len(scores_a),
        "variant_a_label": "with-RAG (Qdrant + BGE retrieval)",
        "variant_b_label": "without-RAG (Gemini zero-shot)",
        "variant_a_mean": float(scores_a.mean()),
        "variant_a_std": float(scores_a.std(ddof=1)),
        "variant_b_mean": float(scores_b.mean()),
        "variant_b_std": float(scores_b.std(ddof=1)),
        "mean_diff": float(diffs.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "effect_size_interp": _interpret_cohens_d(cohens_d),
        "winner": (
            "with-RAG" if diffs.mean() > 0 else
            "without-RAG" if diffs.mean() < 0 else "tie"
        ),
        "significant_at_0.05": bool(p_value < 0.05),
        "per_case": per_case,
    }


def _interpret_cohens_d(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def _save_partial(per_case: list[dict]) -> None:
    """Atomic save: write to .tmp then rename."""
    tmp = PROMPT_RESULTS.with_suffix(".json.tmp")
    result = _compute_stats(per_case)
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(PROMPT_RESULTS)


def _load_done_cases() -> dict[str, dict]:
    if not PROMPT_RESULTS.exists():
        return {}
    try:
        prev = json.loads(PROMPT_RESULTS.read_text())
    except Exception:
        return {}
    return {c["case_id"]: c for c in prev.get("per_case", []) if "case_id" in c}


def run_prompt_ab(cases: dict, reset: bool = False) -> dict:
    valid_cases = {cid: c for cid, c in cases.items() if c["ground_truth"]["violations"]}

    done = {} if reset else _load_done_cases()
    if done:
        print(f"  Resuming: {len(done)} cases already in {PROMPT_RESULTS}")
    per_case: list[dict] = list(done.values())
    pending = [(cid, c) for cid, c in valid_cases.items() if cid not in done]
    print(f"  Pending: {len(pending)} cases\n")

    if not pending and per_case:
        return _compute_stats(per_case)

    for case_id, case in pending:
        print(f"  {case_id}: ", end="", flush=True)
        gt = case["ground_truth"]
        top_v = gt["violations"][0]

        img = cv2.imread(case["image_path"])
        if img is None:
            print("SKIP (image read failed)")
            continue

        violation = Violation(
            type=top_v["type"],
            risk_level=top_v["expected_risk_level"],
            confidence=1.0,
            bbox=(0.0, 0.0, float(img.shape[1]), float(img.shape[0])),
            person_bbox=None,
        )

        try:
            t0 = time.time()
            osha_a = retrieve_osha_context(violation.type, top_k=3)
            report_a = generate_incident_report(img, violation, osha_a)
            t_a = time.time() - t0

            t0 = time.time()
            report_b = generate_incident_report(img, violation, "")
            t_b = time.time() - t0

            judgment = judge_pair(
                top_v["expected_regulation"],
                top_v["expected_risk_level"],
                report_a, report_b,
            )
            score_a = float(np.mean(list(judgment["report_a"].values())))
            score_b = float(np.mean(list(judgment["report_b"].values())))

            per_case.append({
                "case_id": case_id,
                "violation_type": violation.type,
                "expected_regulation": top_v["expected_regulation"],
                "score_a": score_a,
                "score_b": score_b,
                "judgment": judgment,
                "report_a": report_a,
                "report_b": report_b,
                "latency_a_s": round(t_a, 2),
                "latency_b_s": round(t_b, 2),
            })
            _save_partial(per_case)
            print(f"A={score_a:.2f} B={score_b:.2f}  (A:{t_a:.1f}s B:{t_b:.1f}s)  [saved]")
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
                print(f"QUOTA HIT — stopping. ({len(per_case)} cases saved, rerun tomorrow.)")
                break
            logger.exception("Failed on %s", case_id)
            print(f"FAIL: {type(e).__name__}")

    return _compute_stats(per_case)


# ─── Test 2: Threshold A/B (0.40 vs 0.55) ───────────────────────────────────

def _label_has_violation(label_text: str) -> bool:
    for line in label_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if int(parts[0]) in VIOLATION_CLASS_IDS:
                return True
        except ValueError:
            continue
    return False


def run_threshold_ab(threshold_a: float, threshold_b: float, n_samples: int) -> dict:
    if not ZIP_PATH.exists():
        return {"error": f"Zip not found at {ZIP_PATH}"}

    detector = PPEDetector.get()

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        image_names = sorted(
            n for n in names
            if n.startswith("PPE-Combined-1/test/images/") and n.endswith(".jpg")
        )

    random.seed(99)
    sampled = random.sample(image_names, min(n_samples, len(image_names)))
    print(f"\n  Threshold A/B on {len(sampled)} test images (seed=99)\n")

    cc = cw = wc = ww = 0
    per_image: list[dict] = []

    with zipfile.ZipFile(ZIP_PATH) as z:
        for image_name in tqdm(sampled, desc="  threshold scan"):
            try:
                img_bytes = z.read(image_name)
            except KeyError:
                continue
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            label_name = image_name.replace("/images/", "/labels/").rsplit(".jpg", 1)[0] + ".txt"
            try:
                label_text = z.read(label_name).decode("utf-8", errors="ignore")
            except KeyError:
                label_text = ""
            gt = _label_has_violation(label_text)

            detector.conf_threshold = threshold_a
            pred_a = any(d.cls in VIOLATION_CLASSES for d in detector.predict(img).detections)
            correct_a = (pred_a == gt)

            detector.conf_threshold = threshold_b
            pred_b = any(d.cls in VIOLATION_CLASSES for d in detector.predict(img).detections)
            correct_b = (pred_b == gt)

            if correct_a and correct_b:
                cc += 1
            elif correct_a and not correct_b:
                cw += 1
            elif not correct_a and correct_b:
                wc += 1
            else:
                ww += 1

            per_image.append({
                "image": image_name.rsplit("/", 1)[-1],
                "gt_has_violation": gt,
                "pred_a": pred_a, "pred_b": pred_b,
                "correct_a": correct_a, "correct_b": correct_b,
            })

    n = len(per_image)
    acc_a = (cc + cw) / n if n else 0.0
    acc_b = (cc + wc) / n if n else 0.0
    discordant = cw + wc
    if discordant == 0:
        return {
            "test": "McNemar (no discordant pairs)",
            "threshold_a": threshold_a, "threshold_b": threshold_b,
            "n_cases": n,
            "contingency": {"a_right_b_right": cc, "a_right_b_wrong": cw,
                            "a_wrong_b_right": wc, "a_wrong_b_wrong": ww},
            "accuracy_a": acc_a, "accuracy_b": acc_b,
            "note": "Both thresholds identical on every image.",
            "per_image": per_image,
        }
    p_value = float(min(stats.binom.cdf(min(cw, wc), discordant, 0.5) * 2, 1.0))
    return {
        "test": "McNemar's test (exact binomial on discordant pairs)",
        "threshold_a": threshold_a, "threshold_b": threshold_b,
        "n_cases": n,
        "contingency": {"a_right_b_right": cc, "a_right_b_wrong": cw,
                        "a_wrong_b_right": wc, "a_wrong_b_wrong": ww},
        "accuracy_a": acc_a, "accuracy_b": acc_b,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
        "winner": (
            f"threshold={threshold_a}" if cw > wc else
            f"threshold={threshold_b}" if wc > cw else "tie"
        ),
        "per_image": per_image,
    }


# ─── Orchestration ──────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-prompt", action="store_true")
    p.add_argument("--skip-threshold", action="store_true")
    p.add_argument("--reset-prompt", action="store_true",
                   help="Ignore prior partial prompt results and start fresh")
    p.add_argument("--threshold-n", type=int, default=200)
    p.add_argument("--threshold-a", type=float, default=0.40)
    p.add_argument("--threshold-b", type=float, default=0.55)
    args = p.parse_args()

    cases = json.loads(GOLDEN_CASES.read_text())
    print(f"Loaded {len(cases)} golden cases from {GOLDEN_CASES}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("safetyvision-ab-tests")

    if not args.skip_prompt:
        print("\n" + "=" * 72)
        print("A/B Test 1: Prompt variants (with-RAG vs without-RAG)")
        print("=" * 72)
        with mlflow.start_run(run_name="prompt_ab"):
            result = run_prompt_ab(cases, reset=args.reset_prompt)
            for k, v in result.items():
                if k != "per_case":
                    print(f"  {k}: {v}")
            if "error" not in result:
                mlflow.log_metrics({
                    "variant_a_mean": result["variant_a_mean"],
                    "variant_b_mean": result["variant_b_mean"],
                    "p_value": result["p_value"],
                    "cohens_d": result["cohens_d"],
                    "n_cases": result["n_cases"],
                })
                mlflow.set_tag("winner", result["winner"])
            _save_partial(result["per_case"])
            mlflow.log_artifact(str(PROMPT_RESULTS))
            print(f"\n  Saved -> {PROMPT_RESULTS}")

    if not args.skip_threshold:
        print("\n" + "=" * 72)
        print(f"A/B Test 2: Threshold {args.threshold_a} vs {args.threshold_b} (N={args.threshold_n})")
        print("=" * 72)
        with mlflow.start_run(run_name="threshold_ab"):
            result = run_threshold_ab(args.threshold_a, args.threshold_b, args.threshold_n)
            for k, v in result.items():
                if k != "per_image":
                    print(f"  {k}: {v}")
            if "error" not in result:
                mlflow.log_metrics({
                    "accuracy_a": result["accuracy_a"],
                    "accuracy_b": result["accuracy_b"],
                    "p_value": result.get("p_value", 1.0),
                    "n_cases": result["n_cases"],
                })
                mlflow.set_tag("winner", result.get("winner", "n/a"))
            out = RESULTS_DIR / "threshold.json"
            out.write_text(json.dumps(result, indent=2))
            mlflow.log_artifact(str(out))
            print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
