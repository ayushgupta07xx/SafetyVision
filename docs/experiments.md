# SafetyVision — Experiments & A/B Methodology

Three A/B tests ship in the repo. All harness code is in `evaluation/ab_test.py`;
committed outputs are in `evaluation/ab_results/`. Report-quality grading uses Groq
`llama-3.3-70b-versatile` over a 30-case golden set (`evaluation/golden_set/`).
Principle: **honest numbers, no cherry-picking.**

## A/B Test 1 — Incident-report prompt: RAG vs no-RAG

- **Variant A:** report generated *with* retrieved OSHA RAG context.
- **Variant B:** report generated *without* RAG (Gemini zero-shot regulation knowledge).
- **Test:** paired t-test on per-case quality scores.
- **Result (N=16):** RAG wins — Cohen's d = 0.65, p = 0.0197. RAG grounding earns its place.

## A/B Test 2 — Detection confidence threshold: 0.40 vs 0.55

- **Variant A:** confidence threshold 0.40. **Variant B:** 0.55.
- **Test:** McNemar's test on per-image classification correctness.
- **Result (N=200):** 0.40 wins — p = 4×10⁻⁵. Validated as the production default (ADR-009).

## A/B Test 3 — Model: v1 (YOLOv8n) vs v2 (YOLOv8s + Albumentations)

- **Comparison:** per-class precision / recall / F1 on the held-out test set + McNemar's
  test on per-image correctness.
- **Headline:** v2 improves overall mAP@50 from 0.701 → 0.763. Full per-class table lives
  in `docs/model_card.md`; raw output in `evaluation/ab_results/model_v1_vs_v2.json`.
- **Honest note:** v2 missed the 0.78 target (landed 0.763). Documented, not hidden.

## Forecasting evaluation

Prophet vs SARIMA(1,1,1)(1,1,1,7) baseline on a held-out last-7-days window, scored by
MAPE per violation type. Prophet is primary (won 2 of 3 types); SARIMA retained as a
documented baseline. See ADR-008.

## Reproducibility

- Statistical tests: `scipy.stats` (paired t-test, McNemar via `statsmodels`).
- Grading is resumable and spread across Gemini models to respect ~20 req/day free-tier quotas.
- All runs logged to local MLflow (`mlruns/`, committed); training runs also on public W&B.
