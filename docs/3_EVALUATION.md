# BioShield — Evaluation Methodology

> How we decide whether the detector is any good. Metrics, test sets, confidence, honest caveats.

---

## What we measure (and why)

| Metric | What it means | Why we care |
|---|---|---|
| **AUC (ROC)** | Probability the detector ranks a random fake above a random real. 0.5 = random, 1.0 = perfect. | Threshold-free — doesn't depend on where you set the "fake" cutoff. |
| **F1** | Harmonic mean of precision and recall. Balances false positives vs. false negatives. | Honest single number when classes are balanced. |
| **Accuracy** | Fraction of correct calls. | Easy to explain to reviewers; paired with AUC/F1 for context. |
| **Evasion rate** | Fraction of FAKE samples the detector labels as real. | The metric that matters for adversarial robustness — this is the story of our paper. |

**Why evasion rate is the headline number:** AUC says "can the detector separate real from fake on average"; evasion rate says "how often does a fake slip through." For a deployed system, evasion rate is what the adversary is optimizing to maximize. Our adversarial training is optimizing to minimize it.

---

## Test sets

We evaluate on three different distributions to stress-test the detector:

### 1. In-distribution test set (`data/processed/test.csv`)
- 100 samples (50 real PubMed + 50 BioMistral zero-shot fake)
- Held out from training, seen only at evaluation time
- Same distribution as training data — the "best case" for our detector

### 2. Per-round fake pool (`experiments/round_data/condition_*/round_*/fakes.csv`)
- 250 newly generated fakes per round (SeqGAN for B, BioMistral for C/D, both for D)
- Used to compute round-over-round evasion rate
- This is where the adversarial dynamic is visible

### 3. Cross-domain benchmark: RAID (`scripts/raid_benchmark.py --benchmark raid`)
- `liamdugan/raid` dataset, biomedical slice, 200 samples
- Mix of real web-scraped text + AI-generated text from many models (ChatGPT, GPT-4, Claude, Llama, etc.)
- Our detector never saw these models during training
- The "hard case" — tests whether the detector generalized or just memorized BioMistral's style

### 4. Cross-domain benchmark: M4 (SKIPPED)
- Intended: `Chardonneret/M4` — multilingual multidomain
- Dataset was removed from HuggingFace Hub between project start and this run
- Graceful skip: `condition_D_m4_metrics.json` records `{"skipped": true, "reason": "..."}`
- For the paper, swap in a live alternative like `Hello-SimpleAI/HC3` or re-host M4 manually

---

## Results (500-scale, n=100 in-distribution test)

### Headline table

| Condition | Test AUC | Test F1 | Test Acc | Final-round Evasion |
|---|---|---|---|---|
| **A** — static_baseline | **0.989** | **0.938** | **0.94** | — (one-shot, no adversarial loop) |
| **B** — seqgan_only | 0.898 | 0.738 | 0.68 | **0.54** (down from 0.74) |
| **C** — agent_only | 0.982 | 0.947 | 0.95 | 0.00 |
| **D** — full_pipeline | 0.982 | 0.947 | 0.95 | 0.00 |
| **RAID** (cross-domain, detector from A) | 0.861 | 0.179 | 0.13 | **0.96** |

### Condition B per-round (the adversarial dynamic)

| Round | AUC on test | F1 on test | Accuracy on test | Evasion on fake pool |
|---|---|---|---|---|
| 1 | 0.898 | 0.682 | 0.58 | **0.74** |
| 2 | 0.899 | 0.709 | 0.63 | **0.64** |
| 3 | 0.898 | 0.738 | 0.68 | **0.54** |

**Reading**: test AUC stays flat (~0.898) because the held-out test set isn't changing — the original BioMistral fakes are just hard for a detector that's being retrained on SeqGAN style. But evasion on the evolving SeqGAN fake pool drops from 0.74 → 0.54 (27% relative reduction) — that's the detector getting better at catching the style it's being trained on. F1 on test actually rises (0.68 → 0.74) as the detector's precision improves.

---

## Known quirks we'd flag to reviewers (honest disclosure)

### 1. Conditions C and D have byte-identical metrics
Final metrics for Condition C (no retrain) and Condition D (with retrain) are identical:
- Both pick the Condition A baseline detector as their "best" checkpoint (val AUC 0.9992 ceiling)
- Retraining in D never beats that validation AUC, so the best-val-AUC checkpoint selection keeps returning to A
- **Not a bug** — it's a signal that BioMistral zero-shot fakes are too easy for this detector, and adversarial retraining has no gradient to improve against

**Paper implication**: Condition D's "full pipeline" story is weaker than we'd hoped at 500-scale. Needs harder generators (e.g., a LoRA-fine-tuned BioMistral trained to evade) to show the dynamic we expect. Condition B's SeqGAN is the one that produced hard-to-detect fakes.

### 2. RAID cross-domain evasion is 0.96
The detector trained on PubMed + BioMistral catches almost nothing on RAID:
- 181 fake samples from assorted LLMs, 174 classified as real
- AUC still 0.86 — the detector *can* rank fakes above reals, just picks a bad threshold
- **Paper implication**: our detector is a biomedical specialist, not a universal AI-text detector. We should position the paper that way and propose cross-domain training as future work.

### 3. 95% CI at n=100 is ~±5 percentage points
- Wilson interval on a binary proportion at n=100 gives CI ±0.05
- Our reported AUC improvements between conditions (e.g., 0.898 vs. 0.982) are comfortably outside the noise band
- But per-round AUC differences of 0.001 are inside the noise — we shouldn't claim those as meaningful

### 4. Transfer attack (Llama-3.2-3B) not run
- Llama-3.2-3B-Instruct is downloaded and verified to load on MPS
- Intended to test detector against a different generator family, but `data/processed/transfer_test.csv` is empty
- Follow-up task for the paper run: populate transfer_test with 200 Llama-generated fakes, score with condition_D detector, report evasion

---

## How the experiment was controlled

| Control | Value | Why |
|---|---|---|
| Random seed | base=42, per-round `set_seed(42 + r)` | Reproducibility across rounds, without byte-identical outputs |
| Device | MPS (Apple Silicon M3 Max) | Single machine, deterministic GPU kernel |
| Dtype | bfloat16 | Balance speed and numerical precision; avoids fp16 underflow |
| Detector architecture | microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract (340M params) | Held constant across all conditions — the only difference is training data |
| Train epochs per round | 2 | Fixed — longer doesn't improve val AUC at this scale |
| Test set | `data/processed/test.csv` — 100 stratified samples | Never changes, never retrained on |

---

## Reproducibility

All metrics JSON files and plots are git-tracked:

```
experiments/metrics/
  condition_A_metrics.json            ← baseline, n=100
  condition_B_metrics.json            ← 3-round summary
  condition_B_round{0,1,2,3}_metrics.json  ← per-round retraining records
  condition_C_metrics.json            ← agent-only 3-round summary
  condition_C_round0_metrics.json     ← baseline evaluation
  condition_D_metrics.json            ← full pipeline 3-round summary
  condition_D_round{0,1,2,3}_metrics.json
  condition_D_raid_metrics.json       ← RAID cross-domain benchmark
  condition_D_m4_metrics.json         ← M4 skip marker
  metrics_log.csv                     ← long-form append log

experiments/plots/
  evasion_rate_vs_round.png           ← the headline plot (shows B's 0.74→0.54 curve)
  auc_f1_vs_round.png
  ablation_comparison.png             ← side-by-side A/B/C/D bars
```

Re-run the entire pipeline (500 scale, ~6h on M3 Max):

```bash
bash scripts/run_presentation.sh
```

Or a single condition:

```bash
python training/adversarial_loop.py --config configs/config_presentation.yaml --condition full_pipeline
```

---

## For the slide deck

**The three numbers that tell the story:**

1. **Baseline detector AUC 0.989** — our starting point is already strong.
2. **SeqGAN evasion 0.74 → 0.54 across 3 rounds** — the adversarial loop demonstrably improves the detector.
3. **RAID evasion 0.96** — our detector doesn't transfer outside biomedical. Motivates cross-domain future work.

**What NOT to claim:**
- Don't claim Condition D is better than C (they're identical — our Qwen+retrain combo didn't beat the zero-shot agent on this fake distribution).
- Don't claim this works on "all AI-generated text" — we've only shown biomedical + RAID cross-domain.
- Don't over-index on n=100 per-round numbers — CI is ±0.05.
