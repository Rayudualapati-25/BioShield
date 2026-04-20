# BioShield — Research Update Log

**Date:** April 2026  
**Branch:** main  
**Last commit before this update:** `a69ac6a` — style: tighten IEEE author block + banner in BioShield paper

---

## Summary

This update advances BioShield from a 4-condition negative-result study to a
novelty-level research contribution by implementing three counter-measures that
directly address the paper's three core failure modes, completing a full
literature survey, and updating every section of the IEEE paper to reflect the
new work.

---

## 1. New Code (files created)

### `evaluation/calibration.py`
Post-hoc temperature scaling to fix the AUC–accuracy gap on cross-domain (RAID) evaluation.

- `TemperatureScaler(nn.Module)` — single scalar `T` fitted with LBFGS on NLL
- `fit_temperature(logits, labels)` — returns optimal temperature
- `calibrate(cfg, detector_ckpt, calib_csv, eval_csv, output_dir)` — full pipeline:
  pre/post calibration metrics, threshold sweep over `[0.1, 0.9]` to maximise F1
- Outputs `calibration_results.json` with pre/post/optimal-threshold metrics
- CLI: `python evaluation/calibration.py --config <yaml> --detector_ckpt <path> --calib_csv <csv> --eval_csv <csv>`

**Addresses:** §VI.C — Cross-domain calibration failure (RAID accuracy 0.11)

---

### `evaluation/family_matrix.py`
Systematic generator-family disclosure tool — builds an N×M evasion matrix.

- `_score_csv(detector_ckpt, csv_path, cfg, device)` — loads checkpoint, runs inference, returns `{evasion_rate, auc, f1, accuracy}`
- `build_matrix(cfg, matrix_cfg, output_dir)` — iterates all detector×generator pairs
- Outputs `family_matrix.json` (full metrics) + `evasion_matrix.csv` (LaTeX-importable)
- Driven by `experiments/family_matrix_cfg.json`
- CLI: `python evaluation/family_matrix.py --config <yaml> --matrix_cfg <json>`

**Addresses:** §VI.A disclosure recommendation — every deployed detector should publish a generator-family evasion matrix

---

### `models/detector/train_multi_family.py`
Condition E: trains BiomedBERT-Large on a mixture of fake families to compress the ~140× evasion spread.

- `_build_multi_family_train(...)` — merges real PubMed + BioMistral + SeqGAN + Llama fakes with configurable `mixing_ratios`
- Default: `[1.0, 1.0, 0.5]` (BioMistral, SeqGAN, Llama)
- Graceful degradation: warns and skips missing CSV families without crashing
- Reuses `train_one_epoch`, `eval_split`, `save_checkpoint` from `train_detector.py`
- Outputs `condition_E_metrics.json` with full test metrics + mixing metadata
- CLI: `python models/detector/train_multi_family.py --config <yaml> [--seqgan_csv <csv>] [--llama_csv <csv>] [--mixing_ratios 1.0,1.0,0.5]`

**Addresses:** §VI.A — generator-family dominance; designed to compress 140× evasion spread

---

## 2. Modified Code (files updated)

### `models/detector/train_detector.py`
Added adversarial-aware checkpoint selection.

| Addition | Detail |
|---|---|
| `--checkpoint_selector [val_auc\|adversarial]` | CLI flag, default `val_auc` |
| `--fake_pool_csv <path>` | Required when selector=adversarial |
| Validation error | Parser errors if adversarial selected without fake pool CSV |
| `save_all` flag | When adversarial, saves every epoch to `epoch_N/` subdirs |
| Post-training epoch scoring | Loops over all epoch checkpoints, scores each on fake pool |
| Evasion-based promotion | Copies winning epoch (lowest evasion) to canonical `ckpt_dir` |

**Addresses:** §VI.B — C=D pathology; best-val-AUC selection silently rolls back to baseline when val-AUC ceiling is hit

---

### `training/adversarial_loop.py`
Threaded `checkpoint_selector` through the full adversarial loop.

| Addition | Detail |
|---|---|
| `run_loop(..., checkpoint_selector="val_auc")` | New parameter with docstring |
| Retrain command extension | `--checkpoint_selector <value>` appended to every retrain subprocess call |
| Conditional `--fake_pool_csv` | Added to retrain command only when selector=adversarial |
| `main()` CLI | `--checkpoint_selector [val_auc\|adversarial]` argument propagated to `run_loop()` |

---

## 3. Paper Updates

### `paper/sections/01_introduction.tex`
- Fixed citation: replaced `\cite{gehrmann2019gltr}` with `\cite{hu2023radar}` in adversarial training claim (GLTR is a detection paper, not adversarial training)
- Added 3 new contributions to the enumerate block:
  1. Adversarial-aware checkpoint selection (`--checkpoint_selector adversarial`)
  2. Multi-family training — Condition E
  3. Post-hoc temperature-scaling calibration

### `paper/sections/02_related_work.tex`
- Reframed RLHF/DPO comparison as conjecture, not established finding
- Fixed Sadasivan attribution — result positioned as consistent with paraphrase-attack literature, not claimed causal link

### `paper/sections/03_method.tex`
- Added `\cite{morris2020textattack,hu2023radar}` to "silent default" checkpoint selection claim
- Added §III.D **Multi-Family Training (Condition E)** — mixing weights, generator families, implementation pointer
- Added §III.E **Adversarial-Aware Checkpoint Selection** — describes `--checkpoint_selector adversarial` and how it decouples selection from val-AUC saturation

### `paper/sections/05_results.tex`
- Added §V.E **Condition E: Multi-Family Training** with placeholder Table VI (pending full-scale run)

### `paper/sections/06_discussion.tex`
- Added inline citations for RAID and MAGE in cross-domain section
- Added §VI.x **Calibration as a Cross-Domain Fix** — describes temperature scaling and `evaluation/calibration.py`

### `paper/sections/07_limitations_future_work.tex`
- Updated checkpoint selection limitation: adversarial selector now implemented; future work = quantitative comparison of selectors
- Replaced "transfer attack queued" paragraph with: Condition E pending full-scale run + calibration pending held-out split evaluation
- Updated future-work enumerate: 6 concrete items (multi-seed, LoRA-BioMistral, adversarial selector re-run, Condition E full-scale, calibration eval, human eval)

### `paper/sections/08_conclusion.tex`
- Added paragraph attributing all three counter-measures with implementation pointers

### `paper/main.tex` (abstract)
- Added closing paragraph listing the three counter-measures before the final positioning statement

### `paper/refs.bib`
- Fixed RAID author list (added Ludan, Xu, Ippolito; removed erroneous entries); added arXiv eprint
- Added 5 new entries: `hu2023radar`, `guo2017calibration`, `kirchenbauer2023watermark`, `wang2024m4`, `sadasivan2024aigeneratedtext`

---

## 4. New Data Files

### `related_papers.csv`
40-entry verified bibliography with columns:  
`title, authors, year, venue, arxiv_id, doi, topic, relevance_to_bioshield`

Covers: MGT detection, adversarial NLP, calibration, RLHF/DPO, LLM red-teaming, biomedical NLP, domain adaptation, foundational LLMs, watermarking.

### `paper/references_bioshield-mgt.bib`
Full IEEE-compatible BibTeX for the literature survey — 40 entries across all topic clusters. Separate from `paper/refs.bib` (paper bibliography); this is the extended research corpus file.

---

## 5. How to Run the New Contributions

```bash
# Adversarial-aware checkpoint selection (Conditions B or D)
python training/adversarial_loop.py \
  --config configs/config_novel.yaml \
  --condition full_pipeline \
  --checkpoint_selector adversarial

# Condition E — multi-family detector training
python models/detector/train_multi_family.py \
  --config configs/config_novel.yaml \
  --label condition_E \
  --seqgan_csv experiments/round_data/condition_B/round_3/fakes.csv \
  --mixing_ratios 1.0,1.0,0.5

# Post-hoc calibration (fixes RAID threshold collapse)
python evaluation/calibration.py \
  --config configs/config_novel.yaml \
  --detector_ckpt experiments/checkpoints/detector/condition_A \
  --calib_csv data/processed/val.csv \
  --eval_csv data/processed/raid_test.csv \
  --output_dir experiments/calibration

# Generator-family evasion matrix
python evaluation/family_matrix.py \
  --config configs/config_novel.yaml \
  --matrix_cfg experiments/family_matrix_cfg.json \
  --output_dir experiments/family_matrix
```

---

## 6. Immediate Next Steps (Priority Order)

1. **Run adversarial selector re-run** — `--condition full_pipeline --checkpoint_selector adversarial` at 3k scale; confirm C=D pathology dissolves
2. **Run Condition E at 3k scale** — report the 3×3 evasion matrix (BioMistral, SeqGAN, Llama) for Conditions A and E
3. **Run calibration evaluation** — split val.csv into train-calib/eval; report pre/post accuracy on RAID
4. **Multi-seed replication** — seeds 43, 44, 45; report mean ± s.d. for all conditions
5. **LoRA-hardened BioMistral** — DPO with detector logits as reward; expected to reactivate retraining gradient
