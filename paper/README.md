# BioShield — Paper Draft

First-draft IEEE conference paper for the BioShield project (3k-scale rerun, generator-family-dependence narrative).

## Structure

```
paper/
├── main.tex                 # IEEE conference document, includes sections/*
├── refs.bib                 # bibliography
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_method.tex
│   ├── 04_experimental_setup.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   ├── 07_limitations_future_work.tex
│   └── 08_conclusion.tex
└── figures/
    ├── ablation_comparison.png
    ├── auc_f1_vs_round.png
    └── evasion_rate_vs_round.png
```

All numbers come from the 3,000-sample rerun finished 2026-04-18 23:27 (10h 01m wall-clock, single Apple M3 Max). See `../PRESENTATION_RESULTS.md` for the full results dump and `../docs/3_EVALUATION.md` for the methodology rationale.

## Build

### Locally (macOS, BasicTeX or MacTeX)

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Output: `main.pdf`.

### Overleaf

1. Upload the entire `paper/` folder to a new Overleaf project.
2. Set the main document to `main.tex`.
3. Compiler: pdfLaTeX. TeX Live: 2023+.

## Headline claims (paper TL;DR)

1. **Generator-family identity dominates evasion.** SeqGAN 98.7%, BioMistral 0.7%, RAID 98.3% — same detector, same test set, ~140× spread.
2. **Adversarial retraining did not help at 3k scale.** Condition B flat (within ±0.062 Wilson CI on n=250). Conditions C and D byte-identical because best-val-AUC checkpoint selection rolls back to the round-0 baseline.
3. **Cross-domain transfer is open.** RAID accuracy 0.11, evasion 0.983 — calibrated threshold is meaningless cross-domain even when AUC (0.82) preserves rank.
4. **A previously reported 500-scale `0.74 → 0.54` evasion curve is retracted** as small-sample noise.

## Not yet in the draft

- Multi-seed replication (single seed per run today)
- LoRA-hardened BioMistral generator
- Llama-3.2-3B transfer attack (data placeholder empty)
- Blinded human evaluation
- M4 cross-domain benchmark (dataset removed from HF Hub; HC3 or MAGE will replace)

These are catalogued as paper-revision priorities in §VII.
