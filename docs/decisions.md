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

## ADR-010 — Raw NO-X violation surfacing over strict Person + NO-X pairing

**Status:** Accepted (Chat 8, Week 3 deploy prep)

### Context

The original brief specified violation detection as:
> *"Rule: violation requires BOTH a Person AND missing PPE. Do NOT flag missing PPE if no person detected in frame."*

Implemented literally in `core.detector._detect_violations`: any `NO-Hardhat`, `NO-Mask`, or `NO-Safety Vest` detection that didn't pair with a `Person` bbox (IoU ≥ `PERSON_IOU_MIN = 0.05`) was silently dropped.

Deploy-time testing on three real worksite images:

| Image | Detection outcome | Brief-rule result |
|---|---|---|
| Two workers in vests + hardhats | Person ✓, Hardhat ✓, Vest ✓ | Correctly clean |
| Casual outdoor pose, sledgehammer, no PPE | Both Person and NO-X missed (OOD) | Empty (model miss) |
| Forklift driver, cab occlusion | NO-Hardhat ✓ (conf 0.43), Person ✗ | **Violation dropped** |

The forklift case is consistent with the documented model failure mode (model card):
> *"Person detection misses side-view, back-view, partially-occluded workers (training data is heavily frontal)."*

The pairing rule converts every such Person-class miss into a silent false negative — the worst failure mode for a safety screening tool.

### Decision

Surface every `NO-X` detection as a violation regardless of Person pairing. When a Person bbox does pair (IoU ≥ `PERSON_IOU_MIN`), attach it to the violation for richer downstream context (incident report prompts, GradCAM bbox-masking). When no Person pairs, the violation carries `person_bbox=None`.

### Rationale

- The `NO-X` training classes in the source datasets (CHV, PPE-det, Construction Site Safety) are themselves annotated on people without PPE. The class label alone is evidence of "a person without PPE was here" — Person pairing was defensive redundancy.
- Commercial PPE detectors (Protex AI, Intenseye) surface raw violation-class detections without requiring a separate Person pairing.
- The Chat-6 golden-set construction already synthesized violations from raw `NO-X` for agent testing — production now matches that pattern, eliminating a train/serve discrepancy that was previously absorbed by the test harness.
- In a pre-screening tool reviewed by a human safety officer (per *Intended Use* in the model card), false positives from spurious `NO-X` detections are recoverable. False negatives from occluded workers are not.

### Consequences

- More violations surface in occluded / partial-pose scenes — directly addresses the documented Person-detection failure mode.
- Slight risk of spurious violations on isolated `NO-X` detections (e.g. label-only fragments with no human context). Mitigated by the production confidence threshold of 0.40 (ADR-009) and by risk-level routing in the Gemini-generated incident report.
- `Violation.person_bbox` is now genuinely optional. Downstream surfaces (`agent.tools.generate_incident_report`, `core.explainer.explain_result`, annotation overlay) already use `violation.bbox` as the primary spatial reference; `person_bbox` is enrichment metadata only.
- `tests/test_violation.py` updated: `TestPersonAndMissingPPERule` → `TestRawNoXSurfacing`, two previous "dropped silently" assertions inverted, one new test (`test_mixed_paired_and_unpaired_violations_in_same_scene`) added for the real-world mixed case.

### Alternatives considered

- **Keep strict pairing, curate demo images.** Rejected — would force the demo onto a narrow image distribution that misrepresents real worksite conditions. Occlusion is the rule, not the exception.
- **Lower `PERSON_IOU_MIN` further.** Rejected — doesn't help when the Person class isn't detected at all, which is the actual failure mode.
- **Confidence-weighted surfacing (paired → full violation, unpaired → low-risk advisory).** Considered. Out of scope for Week 3; current risk-level mapping per class is sufficient signal. Revisit if production false-positive rate proves high.

### Impact on existing artifacts

- **Brief deviation:** explicit and documented here. The brief's stricter rule is preserved in the Git history of `core/detector.py` and in `tests/test_violation.py` prior to this commit.
- **Model card:** failure modes section should be updated to note that occluded-worker violations now surface (with `person_bbox=None`) rather than dropping silently.
- **A/B test results (ADR-009):** unaffected — threshold sweep was run on per-image classification correctness, not on the pairing rule.
- **Forecasting:** unaffected — historical violation records remain the unit of analysis.

---

## ADR-011 — GCP L4 for v2 training; global GPU quota was the real blocker (corrects ADR-001 framing)

**Status:** Accepted
**Date:** 2026-05-20
**Decision owner:** Ayush
**Supersedes:** ADR-001 (partially — the "GCP permanently unusable" conclusion)

### Context
ADR-001 concluded GCP GPUs were unprovisionable on this account and pivoted v1 to Kaggle. That was a correct *v1 unblock*, but the diagnosis was incomplete. v2 needed a single uninterrupted ~62-hour run — infeasible under Kaggle's 12-hour session cap without repeating the ADR-003 checkpoint-carrier dance. That forced a second look at GCP.

### Root cause (corrected)
GCP GPU quota is two-layered:
- **Regional** `NVIDIA_*_GPUS` — what the IAM dashboard shows per region.
- **Global** `GPUS_ALL_REGIONS` — an umbrella cap that overrides regional values.

New paid accounts default `GPUS_ALL_REGIONS = 0`. With the global cap at 0, every GPU request in every region fails **even when the regional quota shows limit=1**. ADR-001 read the regional dashboard (T4: limit=1, usage=0) and mis-attributed the failures to an "N1/G2 machine-family throttle." The actual blocker was the global umbrella sitting at 0.

### Resolution
On an account aged 7+ days, a single quota-increase request for `GPUS_ALL_REGIONS = 1` (with a one-line honest justification) was approved within hours. With global=1, L4 provisioned immediately in `asia-southeast1-c`.

### Decision
Train v2 on a GCP **L4** (`g2-standard-8`: 8 vCPU, 32GB RAM, 1× L4 24GB), 100GB pd-balanced, image `common-cu129-ubuntu-2204-nvidia-580`, ~₹70/hr on-demand. Single continuous 61.25-hour run, no session cap, no resume needed.

### Reasoning
1. One uninterrupted run removes the Kaggle checkpoint-carrier dance (ADR-003) and the W&B resume gap (ADR-004) entirely.
2. L4 24GB fits batch=24 at imgsz=896 (ADR-012); Kaggle's 2×T4 (16GB each) could not at that batch.
3. ssh access enables tmux-managed background training + live log tailing — impossible in Kaggle notebooks.
4. Cost stayed inside the GCP free-trial credit (~₹4,500 of ~₹22-23k for the full run); $0 out of pocket.

### Caveat (honest)
This does not make ADR-001 wrong for v1 — Kaggle was the correct zero-setup choice at the time given the (incomplete) diagnosis and v1's shorter run. ADR-001 stands as the v1 record; this ADR corrects the "permanent wall" framing. Lesson: when GCP GPU requests fail with healthy *regional* quota, check `GPUS_ALL_REGIONS` before concluding the account is throttled.

### Consequences
- v2 training infra is GCP L4; Kaggle is out of the loop.
- VM is **stopped, not deleted**, between sessions (~₹3-4/day disk) so dataset, test split, and checkpoints persist. Delete only after all artifacts are on HF Hub / committed.
- gcloud runs from local WSL only (scope errors inside the VM), always with `--project=safetyvision-training`.

### Files touched
- `scripts/launch_training.sh`
- handoff notes (GCP quota architecture, VM lifecycle)

---

## ADR-012 — v2 final config: batch=24 + imgsz=896 + multi_scale=False; augmentation over class-weighting

**Status:** Accepted
**Date:** 2026-05-20
**Decision owner:** Ayush
**Supersedes:** — (revises the Chat-9 locked config)

### Context
The Chat-9 locked v2 config was batch=24 + imgsz=896 + **multi_scale=True**, plus a planned class-weighted loss to counter the NO-Mask imbalance (~51:1 vs Hardhat). A 5-minute 1%-data dry-run on the L4 validated the pipeline before committing ~62 GPU-hours.

### Evidence
`multi_scale=True` randomizes input size per batch across 0.5–1.5× of imgsz — for imgsz=896 that peaks at ~1344–1792px. At those peak sizes batch=24 **OOMs** on the L4's 24GB. Ultralytics auto-retries by halving to batch=12 (fits at ~21.5G peak) but that pushes wall time to ~95 hours. With `multi_scale=False`, batch=24 peaks at ~9.98G and runs in ~62 hours.

| Config | Peak VRAM | Est. wall time | Fits 24GB |
|---|---|---|---|
| batch=24, multi_scale=True | OOM at peak | — | no |
| batch=12, multi_scale=True | ~21.5G | ~95 hr | yes |
| **batch=24, multi_scale=False** | **~9.98G** | **~62 hr** | yes |

### Decision
Set `multi_scale=False`, keep batch=24 + imgsz=896. Take the augmentation-only robustness path (Albumentations: CoarseDropout, MotionBlur, RandomGamma, CLAHE + native perspective) instead of the planned class-weighted loss.

### Reasoning
1. **multi_scale benefit is marginal here.** SafetyVision serves at fixed resolution (640 on Lambda, 896 on Spaces), not arbitrary scales — scale-jitter robustness buys little, and 62hr << 95hr.
2. **Augmentation alone hit target.** Final val mAP@50 = 0.787 (≥ 0.78). The feared NO-Mask collapse did not happen (recall 0.789); class-weighting would have added a tuning axis for no proven gain.
3. The dry-run caught the trade-off in *minutes*, not mid-run *hours*.

### Results (honest)
- val (8,025 imgs): mAP@50 **0.787**, mAP@50-95 0.504
- test (4,026 held-out): mAP@50 **0.766** @896 / **0.754** @640 (.pt); **0.738** @640 (deployed ONNX, fp32 drift)
- vs v1 test (0.701 / 0.441): +6.5 / +4.6 mAP at 896, +5.3 at 640
- **Weakest class is NO-Safety Vest (test mAP@50 0.386–0.402, ~217 instances)**, not the predicted NO-Mask. Mask (0.55–0.58) is also weak — the construction_safety_gears set mixes COVID face-mask close-ups into the industrial-mask class. Both documented, not cherry-picked.

### Caveat (honest)
The 0.78 target was specified on held-out **test**; test came in at **0.766** (@896) — short by 0.014. Val cleared it (0.787). The model card reports test as the headline generalization number and notes the shortfall rather than leading with val. multi_scale=False only matters if a v3 retrain is attempted at a resolution that benefits from scale jitter.

### Alternatives considered
- **batch=12 + multi_scale=True (~95hr):** rejected — ~33 extra GPU-hours (~₹2,300) for scale robustness a fixed-resolution deployment doesn't use.
- **Class-weighted loss for NO-Mask:** parked — augmentation-only hit target; revisit only if a future class collapses below usable recall.

### Files touched
- `model/train_v2.py` (multi_scale=False, Albumentations monkey-patch)
- `scripts/launch_training.sh`

---

## ADR-013 — Deployment resolution: 640 ONNX on Lambda, 896 ONNX on HF Spaces (dual export, one model)

**Status:** Accepted
**Date:** 2026-05-22
**Decision owner:** Ayush
**Supersedes:** —

### Context
v2 trained at imgsz=896 (ADR-012), but `core/detector.py` preprocesses (letterboxes) to `IMG_SIZE = 640`, carried over from v1. A static-shape ONNX bakes its input size into the graph, so the export imgsz and the detector's resize **must match**. That forced a deployment-resolution decision: run the model at 640 or 896, and on which surface.

### Evidence (held-out test, 4,026 images)
| Measurement | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| `.pt` @ 896 | 0.766 | 0.487 |
| `.pt` @ 640 | 0.754 | 0.485 |
| ONNX @ 640 (onnxslim, opset 20) | 0.738 | 0.463 |

Dropping from 896 to 640 costs only **−0.012 mAP@0.5** on the `.pt`, and 640 still beats v1's 0.701 by +5.3. The ONNX adds a further ~0.016 fp32 drift at 640 (precision unchanged, slight recall dip at threshold — benign).

### Decision
Export **both** resolutions from the single `best.pt` and deploy per surface:
- **AWS Lambda → `best_640.onnx`** (imgsz 640). Matches `detector.py` (no code change) and stays inside the CPU / 2 GB-memory budget that motivated choosing YOLOv8s in the first place.
- **HF Spaces → `best_896.onnx`** (imgsz 896). 16 GB RAM and no payload limit → run the full 0.766 ceiling at no cost.

`detector.py` stays at `IMG_SIZE = 640` (verified, unchanged). Both files pushed to HF Hub `v2/`.

### Reasoning
1. **Training at 896 isn't wasted by inferring at 640.** Training resolution and inference resolution are decoupled — the 0.754-at-640 figure is the 896-training gains carried at lower inference cost, and a 640-*trained* model would likely score lower at 640.
2. **Lambda is the constrained surface.** 896 is ~1.96× the pixels of 640 → ~2× CPU compute and activation RAM. Deploying 896 there would re-open the exact budget question that drove the small-over-medium model choice, for +0.012 mAP. Not worth it.
3. **Spaces can afford the ceiling**, so it gets it.
4. **One model, two export configs — not two models.** Preserves the "single model variant across surfaces" intent (model size/architecture is identical); only the export resolution differs.

### Honest numbers
The deployed Lambda figure is the **ONNX @ 640 = 0.738**, not the `.pt` 0.754 — the model card reports it as such. The 896 ONNX wasn't separately re-evaluated; `.pt` @ 896 = 0.766 is the Spaces ceiling and the ONNX carries a comparable small drift.

### Alternatives considered
- **Single 640 export only (simplest):** rejected — strands the measured 896 ceiling on a surface (Spaces) that can run it, to save one export file.
- **896 on Lambda:** rejected — re-opens the Lambda CPU/memory budget for +0.012 mAP.
- **Dynamic-axis ONNX (one file, variable input):** rejected — letterbox still pads to a fixed size per surface; dynamic axes add runtime overhead with no benefit for two fixed targets.

### Files touched
- HF Hub `v2/best_640.onnx`, `v2/best_896.onnx` (exported from `best.pt`; local copies under gitignored `artifacts/onnx/`)
- `core/detector.py` (unchanged — `IMG_SIZE = 640` confirmed)
- `docs/model_card.md` (deployed-resolution numbers)

---

*Future ADRs (placeholders):*
- ADR-006 — Lambda Function URLs over AWS API Gateway (Week 3)
- ADR-007 — Gradio vs FastAPI for HF Spaces (Week 3)
---

## ADR-006 — Lambda Function URLs over AWS API Gateway

**Status:** Accepted (Week 3). Permission model corrected Chat 13.

**Context:** Mode 2 needs one public HTTPS `/analyze` endpoint for image inference. Two AWS options: API Gateway (REST/HTTP) or a Lambda Function URL.

**Decision:** Lambda Function URL.
- Free forever — no 12-month free-tier expiry (API Gateway's REST tier expires).
- Single built-in HTTPS endpoint; no stages, usage plans, or gateway-layer API keys to manage.
- API-key auth + rate limiting live at the handler level (Supabase-validated key, reserved concurrency = 10), not at a gateway.

**Alternatives considered:** API Gateway — more flexible (usage plans, request transformation, WAF) but the free-tier expiry and extra surface aren't justified for a single endpoint. Kept as the documented "evaluated alternative."

**Consequences:**
- 6MB synchronous payload cap is a hard constraint. A 413 on a large body is expected behaviour, not a bug → route through Mode 1. This is why Mode 2 is image-only; video stays Mode 1 + Mode 3.
- (Chat 13) AWS began enforcing in **Oct 2025** that public (`AuthType=NONE`) Function URLs need BOTH `lambda:InvokeFunctionUrl` AND `lambda:InvokeFunction` in the resource policy. Terraform's `aws_lambda_function_url` only adds the `InvokeFunctionUrl` half → the public URL returned `403 Forbidden` until a second `aws_lambda_permission` (`InvokeFunction`, principal `*`, `invoked_via_function_url`) was added. Runbook: `aws_deploy.md`. The Terraform block is committed so a clean apply reproduces it.

---

## ADR-007 — Gradio over FastAPI for the HF Spaces demo (Mode 1)

**Status:** Accepted (Week 3).

**Context:** Mode 1 is the public, no-signup demo on HF Spaces (free CPU): upload image/video → annotated image, GradCAM, SHAP, OSHA incident report, forecast — all in-browser, zero frontend build. Mode 2 (Lambda) already serves the same pipeline as a FastAPI JSON API.

**Decision:** Gradio for Mode 1; FastAPI stays Mode 2 only.
- Gradio renders a full interactive UI (upload, tabbed outputs, plots) from pure Python — no HTML/JS/React — and HF Spaces' native Gradio SDK auto-builds and hosts it.
- FastAPI returns JSON, not a UI; making it a *demo* would mean building and hosting a separate frontend — wasted effort for a surface whose whole point is "try it in the browser."
- Both surfaces share the same `core/` + `agent/` pipeline, so only the presentation layer differs — no logic duplication.

**Consequences:**
- The Space `sdk_version` must match the locally-tested Gradio (6.14.0) or HF builds a different major and the app breaks (caught + fixed Chat 13).
- Mode 3 (Next.js, Phase 2) becomes the polished product UI; Gradio/HF Spaces stays the lightweight embeddable open-source demo.

---

## ADR-014 — Lambda container: custom Debian-slim base + awslambdaric (not the AWS base image)

**Status:** Accepted (Chat 12).

**Context:** The Mode 2 Lambda runs the v2 ONNX, an **opset-20** export. The AWS-provided `python:3.11` base (Amazon Linux 2023) caps onnxruntime at 1.16.3, whose max ONNX opset is 19 — it cannot load the validated opset-20 artifact.

**Alternatives considered:**
- **A** — AWS base + re-export the ONNX down to opset 19. Rejected: discards the validated opset-20 artifact and adds a re-export/re-validate cycle.
- **B** — AWS base + pin onnxruntime ≤1.16.3. Rejected: same opset-19 ceiling; the model won't load.
- **C (chosen)** — custom `python:3.11-slim` (Debian, glibc 2.36) base, which resolves onnxruntime ≥1.19, so the opset-20 artifact ships untouched. `awslambdaric` (Lambda Runtime Interface Client) supplies the runtime bootstrap the AWS base had built in.

**Decision:** Option C.
- Debian slim → onnxruntime ≥1.19 → opset-20 model loads unchanged.
- `awslambdaric` is the container entrypoint (replaces the AWS base's built-in bootstrap).
- System libs the AWS base bundled but slim doesn't: `libgomp1` (OpenMP — onnxruntime/opencv), `libglib2.0-0` (glib threads — cv2 import), `libgl1` (libGL.so.1 — ultralytics pulls full opencv-python).
- CPU-only torch (`2.12.0+cpu`, required by transformers 5.x) via `--extra-index-url`; numpy pinned first.

**Consequences:**
- Build context must be the repo root (Dockerfile needs `core/` + `agent/`).
- First build ~10–20 min, image ~3–4 GB — don't Ctrl+C on pip/layer progress.
- Leaving the AWS base means we own the runtime bootstrap and the system-lib list (documented in the Dockerfile header + `aws_deploy.md`).

## ADR-015 — PDF generation + persistence boundary lives in `core/`, not the handler

**Context.** Layer 9 generates a per-violation PDF and must store it and hand back a
download URL. The Lambda handler in `serving/lambda/` cannot host this logic: `lambda`
is a Python keyword, so `serving.lambda.handler` is not importable and nothing there is
unit-testable.

**Decision.** PDF rendering and the Supabase persistence adapter live in `core/`
(`core/pdf_report.py`, `core/supabase_db.py`); the handler stays thin and calls into them.
PDFs are written to the Supabase Storage `reports/` bucket and exposed only via short-lived
**signed URLs** stored in `violations.pdf_report_url`.

**Consequences.** Logic is importable and tested from `tests/`. The persistence adapter is
swappable (sqlite locally / Mode 1, Supabase in Mode 3) behind one module. The storage
bucket is never public — only signed URLs leave the system. Trade-off: the handler must
import from `core/`, so the Lambda image build context is the repo root (see `aws_deploy.md`).

## ADR-016 — Mode-3 auth: Supabase sessions for users, handler-level API keys for the API

**Context.** Mode 3 (Next.js) needs user authentication; the Mode 2 API needs per-caller
auth. ADR-006 keeps Lambda Function URLs (no API Gateway), so there is no gateway-level
auth layer to lean on, and everything must fit Supabase's free tier.

**Decision.** User sessions use **Supabase Auth** (email + Google OAuth) in the frontend.
API access uses **handler-level API keys**: minted on the account page, stored as a plain
sha256-hex hash in Supabase, validated on each Lambda invocation. The TypeScript `hashKey`
is byte-identical to Python `core.apikeys.hash_key`, so account-minted keys validate on
Lambda. **Row-level security** (`auth.uid() = user_id`) is the real data-isolation boundary,
not the app layer.

**Consequences.** The Supabase **service-role key** bypasses RLS and must never reach the
frontend — Lambda only; the frontend uses the anon key. Without the RLS policies the anon
key would read every user's rows. A 401 on a freshly minted key is almost always a stale or
revoked key, not a hash mismatch.
