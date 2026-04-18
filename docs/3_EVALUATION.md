# BioShield — Evaluation Methodology

> How we decide whether the detector is any good. Metrics, test sets, confidence, honest caveats.
> Numbers below are from the **3k-scale re-run** (1,500 real + 1,500 fake, n=300 test).

---

## What we measure (and why)

| Metric | What it means | Why we care |
|---|---|---|
| **AUC (ROC)** | Probability the detector ranks a random fake above a random real. 0.5 = random, 1.0 = perfect. | Threshold-free — doesn't depend on where you set the "fake" cutoff. |
| **F1** | Harmonic mean of precision and recall. Balances false positives vs. false negatives. | Honest single number when classes are balanced. |
| **Accuracy** | Fraction of correct calls. | Easy to explain to reviewers; paired with AUC/F1 for context. |
| **Evasion rate** | Fraction of FAKE samples the detector labels as real. | The metric that matters for adversarial robustness — this is the headline for the paper. |

**Why evasion rate is the headline number:** AUC says "can the detector separate real from fake on average"; evasion rate says "how often does a fake slip through." For a deployed system, evasion rate is what the adversary is optimizing to maximize. Our ablation compares evasion across generator families to isolate what actually matters.

---

## Test sets

We evaluate on three different distributions to stress-test the detector:

### 1. In-distribution test set (`data/processed/test.csv`)
- 300 samples (150 real PubMed + 150 BioMistral zero-shot fake)
- Held out from training, seen only at evaluation time
- Same distribution as training data — the "best case" for our detector

### 2. Per-round fake pool (`experiments/round_data/condition_*/round_*/fakes.csv`)
- 250 newly generated fakes per round (SeqGAN for B, BioMistral for C/D)
- Used to compute round-over-round evasion rate
- Where the generator-family gap becomes visible

### 3. Cross-domain benchmark: RAID (`scripts/raid_benchmark.py --benchmark raid`)
- `liamdugan/raid` dataset, biomedical slice, 200 samples (19 real + 181 fake)
- Mix of real web-scraped text + AI-generated text from ChatGPT, GPT-4, Claude, Llama, etc.
- Our detector never saw these models during training
- Tests whether the detector generalized or memorized BioMistral's style

### 4. Cross-domain benchmark: M4 (SKIPPED)
- Intended: `Chardonneret/M4` — multilingual multidomain
- Dataset was removed from HuggingFace Hub between project start and this run
- Graceful skip: `condition_D_m4_metrics.json` records `{"skipped": true, "reason": "..."}`
- For the paper, swap in a live alternative like `Hello-SimpleAI/HC3` or re-host M4 manually

---

## Results (3k-scale, n=300 in-distribution test)

### Headline table

| Condition | Test AUC | Test F1 | Test Acc | Final-round Evasion |
|---|---|---|---|---|
| **A** — static_baseline | **0.9997** | **0.9797** | **0.98** | — (one-shot, no adversarial loop) |
| **B** — seqgan_only | 0.9498 | 0.6546 | 0.49 | **0.987** (flat: 0.993 → 0.980 → 0.987) |
| **C** — agent_only | 0.9993 | 0.9797 | 0.98 | 0.007 (0.013 → 0.007 → 0.007) |
| **D** — full_pipeline | 0.9993 | 0.9797 | 0.98 | 0.007 (byte-identical to C) |
| **RAID** (cross-domain, detector from A) | 0.8171 | 0.176 | 0.11 | **0.983** |

### Condition B per-round (the flat SeqGAN curve)

| Round | AUC on test | F1 on test | Accuracy on test | Evasion on fake pool |
|---|---|---|---|---|
| 1 | 0.9492 | 0.6532 | 0.487 | **0.993** |
| 2 | 0.9485 | 0.6561 | 0.493 | **0.980** |
| 3 | 0.9498 | 0.6546 | 0.490 | **0.987** |

**Reading**: 3 rounds of adversarial retraining against SeqGAN produced **no measurable
improvement** in evasion rate (Δ ≈ 0.006 within Monte Carlo noise at n=250). Test AUC stays
pegged at 0.949 because SeqGAN's token distribution is far enough from biomedical English
that the detector can *rank* fakes above reals, but the decision threshold never gets into
a range where classification accuracy or F1 recovers. This is a clean negative result: at
3k scale, BiomedBERT-Large cannot be nudged into covering SeqGAN's style via retraining alone.

### Condition C per-round (BioMistral is trivially caught)

| Round | AUC on test | F1 on test | Evasion on fake pool |
|---|---|---|---|
| 1 | 0.9984 | 0.9764 | **0.013** |
| 2 | 0.9990 | 0.9797 | **0.007** |
| 3 | 0.9993 | 0.9797 | **0.007** |

**Reading**: BioMistral zero-shot fakes are caught >99% of the time even without any
retraining. Qwen rewrites of the hardest 25% don't close the gap — test evasion stays
at 0.007 from round 2 onward.

---

## The 3k-scale story: generator-family dependence

| Fake source | Evasion rate | Interpretation |
|---|---|---|
| **SeqGAN** (Condition B) | **0.987** | Detector never generalizes to SeqGAN's style. |
| **BioMistral zero-shot** (Conditions C, D) | **0.007** | Detector catches BioMistral near-perfectly. |
| **Mixed LLMs** (RAID) | **0.983** | Detector fails cross-domain. |

**100× spread** in evasion across generator families. This is the dominant effect in our
data — larger than any round-over-round retraining effect, larger than the agent rewrite
effect. A published detector has to disclose *which* generator family it was trained
against, and reviewers will want to see a family matrix.

**Prior 500-scale result that the 3k run invalidated:** the 500-scale B curve showed
evasion 0.74 → 0.54 — a 27% relative reduction interpreted as "adversarial training works."
At 3k, that curve disappears; all three rounds sit at ~99% evasion. The 500-scale drop was
likely a small-sample artifact (n=250 per round fake pool, ±6% CI on a single evasion rate).
We've updated the headline narrative accordingly.

---

## Known quirks we'd flag to reviewers (honest disclosure)

### 1. Conditions C and D have byte-identical metrics
Final metrics for Condition C (no retrain) and Condition D (with retrain) are identical:
- Both pick the Condition A baseline detector as their "best" checkpoint (val AUC 0.9990 ceiling)
- Retraining in D never beats that validation AUC, so the best-val-AUC checkpoint selection
  keeps rolling back to round 0
- **Not a bug** — it's a signal that BioMistral zero-shot fakes are too easy for this detector,
  and adversarial retraining has no gradient to improve against

**Paper implication**: Condition D's "full pipeline" story is weaker than we'd hoped. Needs
harder generators (e.g., a LoRA-fine-tuned BioMistral trained to evade) to show the dynamic
we expect. Also motivates switching the checkpoint selector from best-val-AUC to something
more adversarial-aware (e.g., best-evasion-on-held-out-pool).

### 2. Condition B is completely flat
Three rounds of retraining and evasion on SeqGAN fakes stayed at 99.3% → 98.0% → 98.7%.
This isn't a tuning problem — increasing epochs, lr, or warmup doesn't help when
the detector's decision boundary is far enough from SeqGAN's manifold that two epochs
of retraining can't bridge it. **Paper implication**: a stronger baseline (larger detector,
more pretraining, or domain-adversarial pretraining) would likely help more than the
adversarial loop.

### 3. RAID cross-domain evasion is 0.983
The detector trained on PubMed + BioMistral catches almost nothing on RAID:
- 181 fake samples from assorted LLMs, 178 classified as real
- AUC still 0.82 — the detector *can* rank fakes above reals, just picks a bad threshold
- **Paper implication**: our detector is a biomedical specialist, not a universal AI-text
  detector. Position the paper that way and propose cross-domain training as future work.

### 4. 95% CI at n=300 is ±2.9 percentage points
- Wilson interval on a binary proportion at n=300 gives CI ±0.029
- All reported AUC gaps between conditions (e.g., 0.949 B vs 0.9997 A) are comfortably outside noise
- Per-round B evasion differences (0.993, 0.980, 0.987) are **inside** the ±0.06 CI on n=250
  fake-pool evasion — we don't claim those rounds differ

### 5. Transfer attack (Llama-3.2-3B) not run
- Llama-3.2-3B-Instruct is downloaded and verified to load on MPS
- `data/processed/transfer_test.csv` is empty — queued for paper run
- Follow-up: populate transfer_test with 200–500 Llama-generated fakes, score with
  condition_A detector, report evasion; second family cross-check completes the matrix

---

## How the experiment was controlled

| Control | Value | Why |
|---|---|---|
| Random seed | base=42, per-round `set_seed(42 + r)` | Reproducibility across rounds, without byte-identical outputs |
| Device | MPS (Apple Silicon M3 Max) | Single machine, deterministic GPU kernel |
| Dtype | bfloat16 | Balance speed and numerical precision; avoids fp16 underflow |
| Detector architecture | microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract (340M params) | Held constant across all conditions — the only difference is training data |
| Train epochs per round | 2 | Fixed — longer doesn't improve val AUC at this scale |
| Test set | `data/processed/test.csv` — 300 stratified samples | Never changes, never retrained on |
| Checkpoint selection | best val AUC across epochs | Standard — but see caveat #1 above |

---

## Reproducibility

All metrics JSON files and plots are git-tracked:

```
experiments/metrics/
  condition_A_metrics.json                 ← baseline, n=300
  condition_B_metrics.json                 ← 3-round summary
  condition_B_round{0,1,2,3}_metrics.json  ← per-round retraining records
  condition_C_metrics.json                 ← agent-only 3-round summary
  condition_C_round{0,1,2,3}_metrics.json
  condition_D_metrics.json                 ← full pipeline 3-round summary
  condition_D_round{0,1,2,3}_metrics.json
  condition_D_raid_metrics.json            ← RAID cross-domain benchmark
  condition_D_m4_metrics.json              ← M4 skip marker
  metrics_log.csv                          ← long-form append log
  scale_500/                               ← prior 500-scale run, archived

experiments/plots/
  evasion_rate_vs_round.png                ← headline plot (B flat + C/D near-zero)
  auc_f1_vs_round.png
  ablation_comparison.png                  ← side-by-side A/B/C/D bars
  scale_500/                               ← prior 500-scale plots, archived
```

Re-run the entire pipeline (3k scale, ~10h on M3 Max):

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

1. **Baseline detector AUC 0.9997** at n=300 — our starting point is at ceiling in-distribution.
2. **Generator-family spread: 99% vs 0.7% evasion** between SeqGAN and BioMistral. That's the
   dominant effect, not rounds or agents.
3. **RAID evasion 0.983** — our detector doesn't transfer outside biomedical. Motivates
   cross-domain future work, positioning the paper as a biomedical-specialist result.

**What NOT to claim:**
- Don't claim adversarial retraining helped at 3k — Condition B is flat, and C==D by
  checkpoint selection. The 500-scale "0.74 → 0.54" curve was likely noise.
- Don't claim this works on "all AI-generated text" — we've only shown biomedical specialist
  performance + RAID cross-domain failure.
- Don't over-index on per-round evasion differences at n=250 per round — CI is ±0.06.
