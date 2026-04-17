# BioShield — Data Flow

> Where every number comes from. One file, concise, no handwaving.

---

## The whole pipeline in one diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             STAGE 1: DATA PREP                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   HuggingFace Hub                           BioMistral-7B (local, bf16, MPS)  │
│   ccdv/pubmed-summarization                 prompts over 50 medical topics    │
│          │                                            │                        │
│          ▼                                            ▼                        │
│   500 real abstracts                        500 fake abstracts                 │
│          │                                            │                        │
│          └────────────────────┬───────────────────────┘                        │
│                               ▼                                                │
│                      labelled pool (1000 rows)                                 │
│                               │                                                │
│                   stratified 80/10/10 split                                    │
│                               │                                                │
│            ┌──────────────────┼──────────────────┐                             │
│            ▼                  ▼                  ▼                             │
│      train.csv          val.csv           test.csv                             │
│      (800 rows)        (100 rows)         (100 rows)                           │
│   400 real / 400 fake  50 / 50            50 / 50                              │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘

                                   │
                                   ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: CONDITION A (BASELINE)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│    train.csv ──► BiomedBERT-Large (340M params) ──► detector_A checkpoint      │
│                    2 epochs, bf16, MPS                                         │
│                  best-val-AUC picked                                           │
│                                                                                │
│    test.csv ──► detector_A ──► auc=0.989  f1=0.938  acc=0.94                  │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘

                                   │
                                   ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: CONDITIONS B / C / D (ADVERSARIAL)               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   For round r in 1..3:                                                         │
│                                                                                │
│     1. GENERATE new fakes                                                      │
│        ├─ B: SeqGAN samples 250 sequences                                      │
│        ├─ C: BioMistral generates 250 zero-shot fakes                          │
│        └─ D: both (merged into 500 new fakes)                                  │
│                           │                                                    │
│                           ▼                                                    │
│     2. SCORE fakes with current detector                                       │
│        ├─ detector.predict(fakes) → prob_real for each                         │
│        └─ sort ascending by prob_real → "hardest" are the ones with high       │
│           prob_real (detector was almost fooled)                               │
│                           │                                                    │
│                           ▼                                                    │
│     3. HARD-FRACTION REWRITE (C, D only)                                       │
│        ├─ Take top 25% hardest (~62 fakes)                                     │
│        ├─ Qwen2.5-7B rewrites them to sound more credible                      │
│        └─ Merge: rewritten_hard + untouched_easy → fakes_merged.csv            │
│                           │                                                    │
│                           ▼                                                    │
│     4. AUGMENT train data                                                      │
│        ├─ train_aug.csv = train.csv + fakes_merged.csv                         │
│        └─ rows: 800 original + 250 new = 1050                                  │
│                           │                                                    │
│                           ▼                                                    │
│     5. RETRAIN detector (B, D only — C skips this)                             │
│        ├─ load detector_{A, prev_round}                                        │
│        ├─ train 2 more epochs on train_aug                                     │
│        └─ save detector_{condition}_round{r}                                   │
│                           │                                                    │
│                           ▼                                                    │
│     6. EVALUATE                                                                │
│        ├─ test.csv ──► detector_round_r ──► auc, f1, accuracy                  │
│        └─ fakes_merged ──► detector_round_r ──► evasion_rate                   │
│                           │                                                    │
│                           ▼                                                    │
│     7. SEED NEXT ROUND                                                         │
│        set_seed(base_seed + r)  ← fixes the "byte-identical rounds" bug        │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘

                                   │
                                   ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 4: CROSS-DOMAIN BENCHMARKS                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   RAID (liamdugan/raid, biomedical slice, 200 samples)                         │
│         │                                                                      │
│         ▼                                                                      │
│   detector_D ──► auc=0.86  evasion=0.96  ← DETECTOR FAILS OUT OF DOMAIN        │
│                                                                                │
│   M4 (Chardonneret/M4) ──► SKIPPED (dataset pulled from HuggingFace Hub)       │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## File-by-file: where each artifact comes from

| File | Produced by | When | Size |
|---|---|---|---|
| `data/processed/train.csv` | `data/prepare_data.py --max_real 500 --fakes-source generator` | Stage 1, once | 800 rows |
| `data/processed/val.csv` | same | Stage 1, once | 100 rows |
| `data/processed/test.csv` | same | Stage 1, once | 100 rows |
| `experiments/checkpoints/detector/condition_A/` | `models/detector/train_detector.py` (via adversarial_loop round 0) | Stage 2 | BiomedBERT-Large weights |
| `experiments/round_data/condition_{B,C,D}/round_{1,2,3}/fakes.csv` | `models/generator/train_generator.py --generate_only` | Stage 3 step 1 | 250 rows each |
| `experiments/round_data/condition_{C,D}/round_{1,2,3}/fakes_hard_input.csv` | `adversarial_loop._score_texts` → sort by prob_real | Stage 3 step 2 | ~62 rows (top 25%) |
| `experiments/round_data/condition_{C,D}/round_{1,2,3}/fakes_hard_rewritten.csv` | `agents/adversarial_agent.py` (Qwen2.5) | Stage 3 step 3 | ~62 rows |
| `experiments/round_data/condition_{B,C,D}/round_{1,2,3}/train_aug.csv` | adversarial_loop merge step | Stage 3 step 4 | 1050 rows |
| `experiments/checkpoints/detector/condition_{B,D}_round{1,2,3}/` | `train_detector.py` (retrain mode) | Stage 3 step 5 | BiomedBERT-Large weights |
| `experiments/metrics/condition_{A,B,C,D}_metrics.json` | adversarial_loop final dump | Stage 3 step 6 | ~600 bytes each |
| `experiments/metrics/condition_D_raid_metrics.json` | `scripts/raid_benchmark.py --benchmark raid` | Stage 4 | 493 bytes |
| `experiments/plots/*.png` | `evaluation/visualization.py --all_conditions` | post-D | 3 PNG files |

---

## Key design choices (what the reader should notice)

1. **Stratified split** — test set has balanced real/fake, so AUC 0.5 means random, 1.0 means perfect.
2. **`best-val-AUC` checkpoint saving** — at each round we keep the detector version with the best validation AUC (not the last epoch's weights). This is what causes Conditions C and D's test metrics to match: both end up selecting the same near-perfect Condition A detector because retraining doesn't improve validation AUC beyond 0.9992.
3. **Hard-fraction rewrite (0.25)** — only the fakes with highest prob_real (i.e., the ones that almost fooled the detector) get sent to the Qwen rewriter. This makes the rewriter focus on sharpening the edge case, not generically rewriting everything. This was a bug in the smoke run and is fixed in the presentation run (see `training/adversarial_loop.py`).
4. **Seed-per-round** (`set_seed(base_seed + round_idx)`) — previously `set_seed(42)` was called once at process start, making SeqGAN sampling and Qwen rewriting byte-identical across rounds. Fixed this run. Visible as varied per-round metrics in Condition B (evasion 0.74 → 0.64 → 0.54).

---

## Configuration (what you can tweak)

All knobs live in `configs/config_presentation.yaml`. Key ones:

```yaml
data.max_real_samples: 500        # dataset scale
detector.epochs: 2                # cheap-ish retrains
detector.batch_size: 16           # MPS-safe for BiomedBERT-Large
loop.num_rounds: 3                # adversarial rounds per condition
loop.fake_pool_size: 250          # new fakes generated per round
loop.hard_fraction: 0.25          # top-25% hardest fakes get rewritten
```

For a 10k-scale paper run, just bump `max_real_samples` to 10000 and budget ~60–80 hours. Everything else scales with it.
