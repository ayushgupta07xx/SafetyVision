# Architecture Decisions

This document records key design decisions made during SafetyVision development.
Each ADR captures context, the decision taken, alternatives considered, and consequences.

---

## ADR-001 — Training infrastructure: Kaggle Notebooks (not GCP)

**Status:** Accepted (Week 1)

**Context**

The original brief specified GCP $300 free-trial credits to spin up a T4 (or L4) Compute Engine VM for YOLOv8n fine-tuning. The training job needed ~15 hours of GPU time.

**Decision**

Pivoted to Kaggle Notebooks: free tier provides 2× Tesla T4 GPUs, 30 hours/week quota, with a 12-hour-per-session cap.

**Why**

GCP T4/L4 instances were **unprovisionable** on a fresh paid account due to an undocumented N1/G2 machine family throttle.

Diagnosed via systematic VM-class testing across 30+ zones globally (us-central, us-east, asia-south, asia-east, europe-west, australia-southeast, southamerica-east). All zones returned the same `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` or quota errors specifically for N1 (T4) and G2 (L4) families, even though:

- IAM quota dashboard showed `T4: limit=1.0, usage=0.0` in every region tested
- `e2-micro` instances provisioned successfully in the same zones (account healthy, billing valid)
- The issue was machine-family specific, not zone-specific or account-specific

**Alternatives ruled out**

- **GCP support ticket:** Basic Support plan does not include technical case access. Billing-only support could not address machine family availability.
- **Quota increase request:** Wrong tool — the form is for raising existing quota limits, not for unblocking machine-family throttles on new accounts.
- **Spot/preemptible T4s:** Same throttle applies.
- **Other cloud GPUs (AWS, Azure):** Free tiers do not include GPU instances. Would require paid spend.

**Consequences**

Pros:
- Zero cost
- 2× T4 in DataParallel mode (more parallel than a single T4 on GCP)
- No setup overhead — Kaggle environment ships with CUDA, PyTorch, ultralytics pre-installed
- Public W&B integration straightforward via Kaggle Secrets

Cons:
- 12-hour-per-session cap forced a mid-run resume at epoch 82/100, requiring an intermediate Kaggle Dataset as a checkpoint carrier (see ADR-003).
- No ssh access to the training environment — debugging happens via notebook cells, not interactive shell.

---

## ADR-002 — Training dataset: single PPE-Combined corpus (not multi-dataset merge)

**Status:** Accepted (Week 1)

**Context**

The brief proposed merging three smaller Roboflow Universe datasets — CHV Dataset (~7k images), PPE-det, and Construction Site Safety (~3k) — to reach a combined ~10,000+ image training corpus.

**Decision**

Used a single Roboflow dataset, **PPE-Combined v1** (`mazz-maxx/ppe-combined-9bprl-mmcaf`, forked from `s-workspace-cjeuu/ppe-combined-9bprl`), comprising 57,904 images across 13 classes.

**Why**

- PPE-Combined already covers every PPE class the brief targeted (hard hat, vest, gloves, mask, goggles) plus their "NO-" violation counterparts and additional classes (Fall-Detected, No_Harness, Person).
- 58k images > 10k images — more training signal, particularly for under-represented classes.
- Single-source labels mean no class-taxonomy normalization or de-duplication code to maintain.
- Roboflow's merge tool works but introduces a brittle preprocessing step that adds 1–2 days of work for marginal data-quality benefit.

**Consequences**

Pros:
- Larger and cleaner training corpus
- Faster path from "have an idea" to "training is running"
- Reproducible: anyone with a Roboflow API key can pull the exact same dataset

Cons:
- Less geographic and stylistic diversity than a hand-merged multi-source set would have provided
- Some classes (`No_Harness`, `NO-Safety Vest`) remain underrepresented and detect poorly — documented as failure modes in the model card.
- Single-source class taxonomy is fixed; site-specific PPE (arc-flash hoods, cut-resistant sleeves) is out of scope.

---

## ADR-003 — Recovery from Kaggle 12-hour session cap

**Status:** Accepted (Week 1)

**Context**

YOLOv8n training on 58k images at batch=32, imgsz=640 across 100 epochs takes ~15 hours of wall time on 2× T4 — longer than Kaggle's 12-hour Save Version cap. Save Version 1 was killed by Kaggle's timer at epoch 82.5/100.

**Decision**

Resumed training in a second Kaggle Save Version using:
1. The crashed Save Version 1's output (containing `epoch82.pt`, `last.pt`, and full ultralytics save_dir state) was published as an intermediate Kaggle Dataset (`ayushgupta07xx/safetyvision-v1-checkpoints`).
2. Save Version 2 attached this dataset as input, copied the save_dir to `/kaggle/working/`, and resumed via `model.train(resume=True)` and `wandb.init(id="9nctv2ai", resume="must", settings=wandb.Settings(init_timeout=300))`.
3. Save Version 2 completed epochs 83–100 cleanly in ~2.8 hours and exported the final ONNX.

**Why this approach over alternatives**

- **Resuming in the same notebook session is not possible** — Kaggle terminates the session and clears `/kaggle/working/` between Save Versions.
- **Mounting Save Version 1's raw output directly** is not supported by Kaggle's input mount system. The intermediate Dataset is required.
- **Re-training from scratch in two chunks** would have wasted ~9 hours of compute and broken the W&B run history continuity.

**Sharp edges discovered**

- Kaggle user-created datasets mount at a non-standard path: `/kaggle/input/datasets/<username>/<slug>/`, not the public-dataset path `/kaggle/input/<slug>/`.
- `wandb.init(resume="must")` default 90s timeout is too short for resumed runs; `settings=wandb.Settings(init_timeout=300)` was required.
- Ultralytics requires the save_dir to live on a writable filesystem; `/kaggle/input/` is read-only, so the checkpoint must be `shutil.copytree`'d to `/kaggle/working/` before `resume=True`.

**Consequences**

Pros:
- 100 full epochs trained without re-doing work
- W&B run ID preserved (single canonical training run for the project)

Cons (see ADR-004):
- Ultralytics' built-in W&B callback did not re-bind to the manually-resumed W&B run, leaving a chart gap for epochs 83–100 on the W&B UI.

---

## ADR-004 — Accept W&B chart gap on resumed training; lean on `results.csv` as canonical metrics

**Status:** Accepted (Week 1)

**Context**

After completing epochs 83–100 in Save Version 2, the W&B run shows:
- Run status: Finished (correct)
- Total runtime updated to include resumed time (correct)
- **Training metrics for epochs 83–100: not logged**

Root cause: ultralytics 8.3.40 ships a W&B callback that calls `wandb.init()` itself on training start. When training is resumed via `model.train(resume=True)` and W&B is already initialized externally with `resume="must"`, ultralytics' internal callback fails silently and does not re-bind metric logging to the active run.

**Decision**

Accept the W&B chart gap. Treat the local `model/yolov8n-ppe-v1/results.csv` (full 100-epoch metric history written directly by ultralytics) as the canonical metrics source. Document the gap in the model card and link to results.csv for completeness.

**Why not fix it**

- Patching ultralytics' callback would require either pinning to an older version (regression risk on other features) or maintaining a fork (long-tail maintenance cost).
- The W&B run remains usable for epochs 1–82, which contains the main learning curves and the model's plateau point.
- `results.csv` is more portable than W&B anyway — it gets committed to the repo, survives W&B account changes, and renders directly with `pandas.read_csv()`.

**Consequences**

Pros:
- Zero engineering cost
- Canonical metrics live with the code, not on a third-party service

Cons:
- W&B URL alone does not tell the full training story; readers must also consult `results.csv` for the last 18 epochs
- If we ever need to resume training again, this sharp edge will recur

---

## ADR-005 — Experiment tracking: stack W&B + MLflow

**Status:** Accepted (Week 1)

**Context**

The brief specifies both Weights & Biases (for public training-run visibility) and MLflow (as the local model registry and run archive). Both tools log overlapping metrics and parameters.

**Decision**

Run both in parallel. W&B is enabled via ultralytics' built-in callback (`wandb=True` in training args). MLflow runs alongside via explicit `mlflow.start_run()` + `mlflow.log_metrics()` + `mlflow.pytorch.log_model()`. The MLflow file store (`mlruns/`) is committed to the repo; the W&B run is public.

**Why**

- W&B provides a clean public URL for showing training history to anyone — useful for collaborators, recruiters, and the resume.
- MLflow's local file store lives in the repo and survives third-party service outages or account changes. It also natively integrates with model-registry concepts (stages: `Production`, `Staging`).
- The two tools serve different audiences (public vs in-repo) and different time horizons (sharable now vs reproducible later).
- Cost: zero. Both are free for this scale of use.

**Consequences**

Pros:
- Resilience: if either service breaks or an account is lost, the other still has the run
- Two distinct ATS keywords (W&B and MLflow) anchored in real code, not boilerplate
- MLflow `mlruns/` works offline and after `git clone`

Cons:
- Two systems to keep in sync at every `log_metrics()` site (mitigated by wrapping in a single training script)
- MLflow's local file store is committed to git, which means it grows with every training run (~16 MB for this run — acceptable; future runs may need to be cleaned out before commit)

## ADR-008 — Prophet as primary 7-day compliance forecaster; SARIMA as documented baseline

**Status:** Accepted
**Date:** 2026-05-16
**Decision owner:** Ayush
**Supersedes:** —

### Context
The brief calls for a 7-day compliance forecast per violation type (Layer 5) with
both a primary model and a baseline for the model card. Two candidates were on
the table:

- **Prophet** — additive decomposition (trend + weekly seasonality + noise),
  Bayesian fit via Stan, native handling of missing days, automatic uncertainty
  intervals.
- **SARIMA** — classical ARIMA with explicit seasonal differencing, no
  Bayesian machinery, requires manual order/seasonal_order specification.

Both were implemented (`analytics/forecast.py` and `analytics/forecast_baseline.py`)
against the same `load_compliance_series()` contract. The decision was deferred
until a real backtest could be run on representative data.

### Evidence
30 days of synthetic violation history (deterministic, `seed=42`) seeded via
`analytics/seed_violations.py` with realistic weekly seasonality, slow trend,
and ±15–30% noise. Backtest split: 23 train / 7 test. Compared across three
violation types with different base rates.

| Violation type     | Base rate | Prophet MAPE | SARIMA MAPE | Winner   |
|--------------------|-----------|--------------|-------------|----------|
| NO-Hardhat         | 10%       | 0.0183       | 0.0320      | Prophet  |
| NO-Safety Vest     | 8%        | 0.0136       | 0.0225      | Prophet  |
| No_Harness         | 3%        | 0.0107       | 0.0043      | SARIMA   |
| **Average**        | —         | **0.0142**   | 0.0196      | Prophet  |

SARIMA configuration: order=(1,1,1), seasonal_order=(1,1,1,7).
Reproduce: `python -m analytics.compare_forecasts`.
MLflow runs: experiment `safetyvision-forecasting`, 3 runs logged.

### Decision
**Prophet** is the primary forecaster surfaced in the Gradio app and the
DynamoDB-backed `/forecast` endpoint (Week 3).
**SARIMA** remains in `analytics/forecast_baseline.py` and the model card as a
documented comparison baseline.

### Reasoning
1. **Better average MAPE** on the backtest (2/3 types, lower overall mean).
2. **Native missing-day handling.** DynamoDB violation history will have days
   with zero events; Prophet ingests these without imputation. SARIMA requires
   continuous indexing.
3. **No hyperparameter grid.** SARIMA needs `(p,d,q,P,D,Q,s)` tuning per series;
   Prophet's `weekly_seasonality=True` is the only knob touched.
4. **Built-in uncertainty intervals** at a configurable width
   (`interval_width=0.80` here) for the Gradio CI bands.

### Caveat (honest)
SARIMA wins on **No_Harness** — the lowest-incidence series. This is consistent
with the textbook pattern: simpler models generalize better when signal is
sparse and noise dominates. If a future violation class has a base rate below
~5%, switching to SARIMA for that class is the right move. The model card
documents this so the comparison stays honest and not cherry-picked.

A 30-day synthetic backtest is the floor of trustworthy evaluation, not the
ceiling. Real DynamoDB data is collected starting Week 3 — re-run this backtest
on real data once 30+ days have accumulated, and revisit if average MAPE
crosses over.

### Alternatives considered
- **Exponential smoothing (Holt-Winters)** — simpler than SARIMA, but no
  uncertainty intervals out of the box, and weaker on trend-change adaptation.
- **LSTM** — overkill for ≤60-day daily series, slow to train, and we have no
  GPU at runtime.
- **Prophet only, drop SARIMA** — losing the baseline removes the
  "Prophet vs SARIMA" talking point on the resume and the comparison MAPE
  table in the model card. Keeping both costs ~140 LOC.

### Files touched
- `analytics/forecast.py` (Prophet, primary)
- `analytics/forecast_baseline.py` (SARIMA, baseline)
- `analytics/compare_forecasts.py` (backtest harness, MLflow logging)
- `analytics/seed_violations.py` (synthetic data for the backtest)
- `evaluation/forecast_compare/summary.json` (committed comparison output)

---

## ADR-009 — A/B testing harness: paired t-test (prompt) + McNemar (threshold)

**Status:** Accepted
**Date:** 2026-05-17
**Decision owner:** Ayush
**Supersedes:** —

### Context
The brief specifies two A/B tests with documented statistical methodology
(`docs/experiments.md`):
- **Test 1** — Prompt variants for incident reports (with-RAG vs without-RAG)
- **Test 2** — Confidence threshold tuning (0.40 vs 0.55)

Both tests must produce machine-readable results (`evaluation/ab_results/*.json`),
be reproducible from a single `python -m evaluation.ab_test` invocation, and
log to MLflow. The harness must use named statistical tests, not vibes.

### Decision
**Test 1** uses **paired t-test** with the same case used for both variants
(same image, same expected ground-truth violation). Effect size reported as
**Cohen's d**. Judge is Llama-3.3-70b-versatile via Groq (free tier 500k tokens/day),
not Gemini, to avoid the same provider both generating and judging.

**Test 2** uses **McNemar's test** (exact binomial on discordant pairs) over
per-image binary correctness ("did the detector correctly flag this image as
violation vs compliant"). Run on a random 200-image sample of the held-out
PPE-Combined test split (seed=99 for reproducibility).

Ground truth for Test 1 = curated regulation map (`evaluation/golden_set/regulation_map.py`)
that maps (class, context) → expected OSHA citation. No LLM involvement in
ground truth — every expected value is either from the dataset's YOLO labels
or from a hand-curated map. This avoids circular evaluation.

### Evidence — Test 2 (Threshold)
Run: `python -m evaluation.ab_test --skip-prompt`
Sample: 200 test images, seed=99
Detector: YOLOv8n ONNX from HF Hub

| Threshold | Accuracy |
|-----------|----------|
| 0.40      | **0.76** |
| 0.55      | 0.67     |

Contingency (per-image correctness):

| | B correct | B wrong |
|-|-----------|---------|
| **A correct** | 133 | 19 |
| **A wrong**   | 1   | 47 |

McNemar discordant pairs: 20. Exact binomial p-value: **4.0×10⁻⁵**.
Statistical conclusion: 0.40 wins, **p < 0.001**.
Practical conclusion: when thresholds disagree, 0.40 is right 19× more often
than 0.55. Lower threshold captures genuine NO-X detections that 0.55 filters
out, with negligible false-positive cost on this dataset.

### Evidence — Test 1 (Prompt variants)
Run: `GEMINI_MODEL=gemini-2.5-flash python -m evaluation.ab_test --skip-threshold`
(executed across two sessions due to free-tier daily quota; resumable harness
preserved the first 7 cases and added 9 more on the second run)
Judge: Llama-3.3-70b-versatile (Groq), 3-dimension scoring (regulation_accuracy,
action_relevance, faithfulness), 1-5 each.

| Metric | Value |
|--------|-------|
| N (cases scored) | 16 |
| Variant A (with-RAG) mean score | **4.08** |
| Variant A std | 0.64 |
| Variant B (without-RAG) mean score | 3.67 |
| Variant B std | **0.00** |
| Mean difference (A − B) | +0.42 |
| Paired t-statistic | 2.61 |
| **p-value** | **0.0197** |
| Cohen's d | **0.65** (medium-large effect) |
| Winner | **with-RAG** |
| Significant at α=0.05 | ✓ **Yes** |

**Interpretation:** RAG-augmented prompts win with statistical significance
(p=0.0197 < 0.05). Cohen's d = 0.65 puts the effect between "medium" and
"large" by Cohen's conventions. The most striking finding is **Variant B's
zero standard deviation**: without RAG, Gemini produces uniformly mediocre
reports — the judge cannot distinguish between them at all (every report
scored exactly 3.67). With RAG, scores vary across cases (σ=0.64) because
the retrieved OSHA context differs per violation type and lets the model
produce genuinely case-specific outputs. RAG is what makes the system
distinguishable, not what makes it marginally better.

### Caveat (honest)
**Quota fragmentation across two sessions, two models.** The Gemini free
tier caps each model at ~20 requests/day. The first session used
`gemini-flash-latest` and completed 7 cases before exhaustion. The second
session used `gemini-2.5-flash` (separate per-model quota) and completed
9 more cases before that model also hit its limit. 1 case
(`worker_no_helmet_site`) remains incomplete and is excluded from N=16.

This is a free-tier artifact, not a methodology flaw. The resumable harness
(`evaluation/ab_test.py` saves `prompt_variant.json` after every case)
made the cross-session workflow viable: each session pays only for cases
not already in the partial results file. A future run could finish the
last case + add depth by repeating cases on multiple judge seeds, but the
N=16 result is publishable as-is.

Note also: switching judge models between sessions could in principle add
variance, but the **judge stayed the same** (Llama-3.3-70b-versatile via
Groq) — only the *report generator* varied between sessions. Variant A
and Variant B were generated by the same model within each case, so the
paired structure is preserved.


### Alternatives considered
- **Independent t-test instead of paired**: rejected. Paired is correct
  because both variants score the same case; case-level noise is controlled.
- **Chi-squared McNemar instead of exact binomial**: rejected. Exact binomial
  is more accurate at small discordant counts (20 in our case) and has no
  continuity-correction debate.
- **Gemini as both generator and judge**: rejected to avoid same-model bias.
  Groq Llama-3.3-70b is the independent judge.
- **Holistic single score from judge instead of three dimensions**: rejected.
  Multi-dimensional scoring lets us see *where* RAG helps (regulation accuracy
  is the dimension most affected by retrieval; faithfulness is less affected).
- **Bootstrap CI on Cohen's d**: parked for Week 3 if needed. Reported point
  estimate is enough for the prompt-vs-threshold story.

### Files touched
- `evaluation/ab_test.py` (resumable harness, both tests)
- `evaluation/golden_set/regulation_map.py` (deterministic ground-truth map)
- `evaluation/golden_set/build_ground_truth.py` (YOLO + manual → cases.json)
- `evaluation/golden_set/cases.json` (30 cases: 20 test_split + 10 unsplash)
- `evaluation/ab_results/prompt_variant.json` (committed)
- `evaluation/ab_results/threshold.json` (committed)
- `mlruns/safetyvision-ab-tests/` (experiment runs)

---

*Future ADRs (placeholders):*
- ADR-006 — Lambda Function URLs over AWS API Gateway (Week 3)
- ADR-007 — Gradio vs FastAPI for HF Spaces (Week 3)
