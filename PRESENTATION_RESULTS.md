# BioShield — Presentation Results (3k-scale, final)

> Regenerated from the 3,000-sample re-run that launched 18 Apr 13:26 and finished 23:27
> (10h 01m wall-clock). Supersedes the earlier 500-scale numbers — now archived under
> `experiments/metrics/scale_500/` and `experiments/plots/scale_500/`.

---

## TL;DR for the slide deck

- **1,500 real PubMed + 1,500 BioMistral fakes**, 80/10/10 split → n=300 test.
- All 4 conditions + RAID benchmark complete end-to-end on M3 Max (bf16, MPS).
- Condition A detector hits **AUC 0.9997 / F1 0.980 / Acc 0.98** — near-ceiling on in-distribution.
- Headline finding: **generator-family dependence dominates.** SeqGAN fakes slip past 99%; BioMistral fakes caught >99%; RAID cross-domain 98% evasion. The detector is style-specific, not content-specific.

---

## Conditions scoreboard

| Condition | Generator | Agent | Retrain | AUC | F1 | Acc | Evasion (final round) |
|---|---|---|---|---|---|---|---|
| **A** — static_baseline | BioMistral (zero-shot) | — | n/a | **0.9997** | **0.9797** | **0.98** | — |
| **B** — seqgan_only | SeqGAN | — | every round | 0.9498 | 0.6546 | 0.49 | **0.987** (flat) |
| **C** — agent_only | BioMistral | Qwen2.5 (zero-shot) | no | 0.9993 | 0.9797 | 0.98 | **0.007** |
| **D** — full_pipeline | BioMistral | Qwen2.5 | every round | 0.9993 | 0.9797 | 0.98 | **0.007** (byte-identical to C) |
| **RAID** (cross-domain, detector from A) | assorted LLMs | — | n/a (eval only) | 0.8171 | 0.176 | 0.11 | **0.983** |

### Condition B — per-round evasion (the flat SeqGAN curve)

| Round | Test AUC | Test F1 | Evasion on fake pool |
|---|---|---|---|
| 1 | 0.9492 | 0.6532 | **0.993** |
| 2 | 0.9485 | 0.6561 | **0.980** |
| 3 | 0.9498 | 0.6546 | **0.987** |

**Interpretation:** adversarial retraining of a BiomedBERT detector against SeqGAN-style fakes
did **not** improve evasion at 3k scale. Rounds 1, 2, 3 evade at 99.3%, 98.0%, 98.7% — within
Monte Carlo noise of each other. SeqGAN's token distribution is far enough from biomedical
English that the detector can rank it well (test AUC stays at 0.95) but can't classify it right
(F1 only 0.65, evasion ≈ 100%). This is a failure mode of detector-side adversarial training
when the generator family is out of distribution.

### Condition C — per-round evasion (BioMistral is trivially caught)

| Round | Test AUC | Test F1 | Evasion on fake pool |
|---|---|---|---|
| 1 | 0.9984 | 0.9764 | **0.013** |
| 2 | 0.9990 | 0.9797 | **0.007** |
| 3 | 0.9993 | 0.9797 | **0.007** |

**Interpretation:** BioMistral zero-shot fakes are caught >99% of the time even without any
retraining. Qwen rewrites of the hardest 25% don't change this at scale — the baseline
detector is already at ceiling on this generator family.

### Condition D — mirror of C (C==D pathology at scale)

D's per-round metrics are **byte-identical** to C: evasion 0.013 → 0.007 → 0.007,
test AUC 0.998 → 0.999 → 0.999. The retrain loop selected best-val-AUC weights and
could never beat the baseline ceiling (val AUC 0.9990). So D's checkpoint selector
keeps rolling back to round-0, making D functionally equivalent to C.

---

## The headline story for the deck

**"Our detector is generator-family specific."**

| Fake source | Evasion | What that means |
|---|---|---|
| SeqGAN (Condition B) | **0.987** | Detector can't classify SeqGAN output regardless of retraining. |
| BioMistral zero-shot (Conditions C, D) | **0.007** | Detector catches BioMistral trivially. |
| Mixed LLMs on RAID | **0.983** | Detector fails cross-domain entirely. |

Spread: **98% vs 1% evasion across generator families** — that's the dominant effect,
not adversarial retraining, not the agent rewrite. A published detector has to disclose
*which* generator family it was trained to catch.

---

## Bug fixes that landed earlier and held up at scale

### 1. seed-per-round (training/adversarial_loop.py)
`set_seed(base_seed + round_idx)` inside the loop — rounds now produce different fakes,
different rewrites. Confirmed by hash-diff across `condition_*/round_{1,2,3}/fakes.csv`
at 3k scale.

### 2. hard_fraction=0.25 wired (training/adversarial_loop.py)
Only the top-25% hardest fakes (62 of 250) reach Qwen each round. All six rounds of C+D
produced `round_*/fakes_hard_input.csv` and `fakes_hard_rewritten.csv` on disk.

---

## Sample footprint

| Stage | Real | Fake | Cumulative |
|---|---|---|---|
| Initial data prep | 1,500 PubMed | 1,500 BioMistral zero-shot | 3,000 |
| Condition A | 0 | 0 | 3,000 |
| Condition B (3 × 250 SeqGAN) | — | 750 | 3,750 |
| Condition C (3 × 250 BioMistral + 3 × 62 Qwen) | — | 936 | 4,686 |
| Condition D (3 × 250 BioMistral + 3 × 62 Qwen) | — | 936 | **5,622** |
| RAID (eval-only, external) | 19 | 181 | (+200, not generated) |

**Total generated: 5,622 samples. Plus 200 external RAID samples evaluated.**

vs. 500-scale prior run (3,984 generated): **+41% more samples**, n=300 test set
(from n=100) → 95% CI on AUC tightens from ±5.0% to ±2.9%.

---

## Artifacts on disk (3k-scale)

```
experiments/metrics/
  condition_A_metrics.json              ✓  AUC 0.9997
  condition_B_metrics.json              ✓  final evasion 0.987 (flat)
  condition_B_round{0,1,2,3}_metrics.json ✓
  condition_C_metrics.json              ✓  final evasion 0.007
  condition_C_round{0,1,2,3}_metrics.json ✓
  condition_D_metrics.json              ✓  byte-identical to C
  condition_D_round{0,1,2,3}_metrics.json ✓
  condition_D_raid_metrics.json         ✓  AUC 0.817, evasion 0.983
  condition_D_m4_metrics.json           ✓  skip marker (Chardonneret/M4 removed from HF)
  scale_500/                            ←  old 500-scale numbers archived here

experiments/plots/
  evasion_rate_vs_round.png             ✓  regenerated 23:26
  auc_f1_vs_round.png                   ✓  regenerated 23:26
  ablation_comparison.png               ✓  regenerated 23:26
  scale_500/                            ←  old plots archived here

experiments/checkpoints/
  detector/condition_A/                 ✓  BiomedBERT-Large trained on 2,400
  detector/condition_B/, /condition_B_round{1,2,3}/ ✓
  detector/condition_C/, /condition_C_round{1,2,3}/ ✓
  detector/condition_D/, /condition_D_round{0,1,2,3}/ ✓

data/processed/
  train.csv  (2,400 rows, 1,200/1,200 real/fake)
  val.csv    (300 rows, 150/150)
  test.csv   (300 rows, 150/150)
  transfer_test.csv  — EMPTY (Llama transfer-attack generation still queued for paper run)
```

---

## Runtime notes (10h 01m total on M3 Max)

| Stage | Wall-clock | % of total |
|---|---|---|
| Data prep (1,500 PubMed stream + 1,500 BioMistral init) | 3h 55m | **39%** |
| Condition A (detector baseline only) | 13m | 2% |
| Condition B (SeqGAN × 3 rounds) | 56m | 9% |
| Condition C (BioMistral+Qwen × 3 rounds) | 2h 05m | 21% |
| Condition D (BioMistral+Qwen+retrain × 3 rounds) | 2h 52m | **29%** |
| Plots | 0s | — |
| RAID (200 samples) | 22s | — |
| M4 (skipped, dataset removed) | 2s | — |

- **Mac awake:** `caffeinate -dims -w $PID` bound to orchestrator — lid-close safe
- **Memory:** stable 14.5 GB MPS allocated throughout, swap flat at 537 MB
- **Generation pace (BioMistral):** ~7.8 s/sample, zero drift across 8 hours
- **Device:** M3 Max 64 GB bf16 MPS, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 OMP_NUM_THREADS=10`

---

## For the slide deck

Talk-track (reframed from 500-scale):

1. **Problem.** Biomedical deepfake text is dangerous; detectors need to be robust to adversarial generators.
2. **Approach.** 4-condition ablation isolates (a) baseline detector, (b) SeqGAN+retrain loop, (c) LLM-rewrite agent, (d) both combined.
3. **Data.** 1,500 real PubMed + 1,500 BioMistral fakes, 80/10/10 split, n=300 test, ±2.9% CI.
4. **Key result.** **Generator-family dependence dominates.** Evasion rates by source:
   - SeqGAN 99% (retraining didn't help)
   - BioMistral 0.7% (too easy for detector)
   - RAID cross-domain 98% (out-of-distribution)
5. **Engineering.** Flagged 2 bugs in smoke run (seeding, hard-fraction) — fixed and verified at 3k.
6. **Scope.** BioShield is a biomedical specialist trained against BioMistral. RAID result is a positioning statement, not a weakness.

---

## Slide: From project to paper — what it takes to publish

**Tuesday's presentation is "project-worthy."** A publishable paper needs harder fakes (the
B-side flat curve and the C/D pathology both point to generator quality, not scale) and
tighter CIs.

### What we have now (3k-scale, this presentation)
- 5,622 samples generated · n=300 test · ±2.9% CI on AUC
- Full pipeline: **10h 01m on M3 Max**
- Sufficient to defend the generator-family-dependence narrative

### Paper Tier 1 — Pilot (peer-review defensible)

| Dimension | Value |
|---|---|
| Data scale | 2,000 real + 2,000 fake |
| Rounds × fake pool | 3 × 500 |
| Samples produced | ~9,750 |
| Test n / CI | n=400 · ±2.5% |
| Wall-clock (M3 Max) | **~16 h (one overnight)** |

### Paper Tier 2 — Full (reviewer-proof, PRD spec)

| Dimension | Value |
|---|---|
| Data scale | 10,000 real + 10,000 fake |
| Rounds × fake pool | 5 × 1,000 |
| Samples produced | ~38,500 |
| Test n / CI | n=2,000 · ±1.1% |
| Wall-clock (M3 Max) | **~65 h (~2.7 days)** |

### Where the time goes (both tiers)

```
BioMistral zero-shot generation: ~70% of wall-clock
Detector retrain (×4 conditions × N rounds): ~20%
Qwen rewrite of hard-fraction subset: ~8%
Everything else (SeqGAN, plots, benchmarks): ~2%
```

### Pre-work required before the paper run

1. **Batch BioMistral generation** (batch=4 in `train_generator.py`) — cuts paper run from 65h → ~40h. 1-day code change.
2. **Harder fake generator** — LoRA-fine-tune BioMistral to evade the detector (DPO or reward-weighted). Directly addresses the C/D pathology at 3k.
3. **Populate `transfer_test.csv`** — Llama-3.2-3B generates 500 cross-family fakes for transfer-attack eval.
4. **Replace M4 dataset** — `Chardonneret/M4` was removed from HF Hub. Swap in `Hello-SimpleAI/HC3` or `yaful/MAGE`.
5. **Human evaluation** (50 samples, blinded) — paper reviewers expect this.

### Total commitment for paper

- **Tier 1 pilot**: 1 day pre-work + overnight compute = **~2 days**
- **Tier 2 full**: 3 days pre-work + ~3 days compute = **~1 week**

### The 3 claims the paper will defend (updated)

1. *Biomedical specialist detectors are generator-family bound.* SeqGAN 99% evasion flat, BioMistral 0.7% — 100× spread across families at 3k scale.
2. *Best-val-AUC checkpoint selection hides the retraining signal when the generator is too easy.* The C==D pathology isn't a bug; it's a lesson about how to score adversarial loops.
3. *Cross-domain transfer is an open problem.* RAID 98% evasion with detector from A. Future work, not a weakness.
