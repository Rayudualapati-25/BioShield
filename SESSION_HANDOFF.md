# Session Handoff — BioShield smoke-test (Apr 17)

> **Read this first** when you open a fresh Claude Code session in this folder.
> It captures everything the prior session knew so we lose zero context.
>
> Companion docs:
> - `OVERNIGHT_SUMMARY.md` — status board + smoke-test result tables
> - `PRD.md`, `Novelty_Upgrade_Experiment_Plan.docx` — original project scope
> - `README.md` — install + run-from-scratch instructions

---

## TL;DR — where we are right now

- **Conditions A, B, C are all done at 200-sample smoke scale.** Committed and pushed to `github.com/Rayudualapati-25/BioShield` at commit **`82d188b`**.
- **Condition D is intentionally NOT run.** User decision — it lives in the 10k paper run with two fixes flagged below.
- All artifacts (configs, metrics JSON, plots, per-round JSONs, log CSV) are tracked in git.
- Working tree is clean. Branch `main` is in sync with `origin/main`.

```
git log --oneline -3
82d188b feat: smoke-test conditions B and C at 200-scale
769884b feat: Condition A smoke-test on real PubMed + BioMistral fakes (n=400)
46c263d feat: upgrade to BioShield — BioMistral-7B + BiomedBERT-Large + Qwen2.5-7B
```

---

## Smoke-test scoreboard (200-scale, n=40 test split)

| Condition | Generator | Agent | Detector retrain | Final AUC | Final F1 | **Evasion** |
|---|---|---|---|---|---|---|
| **A** — static_baseline | BioMistral (zero-shot) | — | n/a (one-shot) | 0.978 | 0.905 | **0.15** |
| **B** — seqgan_only | SeqGAN | — | yes (each round) | 0.900 | 0.679 | **0.75** |
| **C** — agent_only | BioMistral | Qwen2.5 (zero-shot rewrite) | no | 0.945 | 0.923 | **0.05** |
| D — full_pipeline | — | — | — | — | — | *skipped, paper-run only* |

### Two narratives for the paper

1. **SeqGAN fakes are 5× more evasive than 7B-LLM fakes.** Even after 2 retrain rounds the detector cannot close the gap. Likely a distribution-shift artefact of being trained on BioMistral-style fakes.
2. **Zero-shot agent rewrites are counterproductive** (Condition C evasion 0.05 vs A's 0.15). Untrained Qwen leaks its own stylistic markers when asked to "make it more credible." **This is exactly why Condition D matters** — the agent rewriter only earns its keep after adversarial LoRA training.

---

## Two known smoke-run issues — MUST fix before the 10k paper run

### Issue 1 — per-round metrics are byte-identical
Both B and C show round 1 == round 2 to every decimal. Root cause: `set_seed(42)` is called once at process start, so SeqGAN sampling and Qwen sampling produce the same outputs every round.

**Fix:** in `training/adversarial_loop.py`, seed per round inside the loop:
```python
from utils.seed import set_seed
for round_idx in range(num_rounds):
    set_seed(seed + round_idx)   # was: seed once at process start
    ...
```
Without this, the round-on-round adversarial dynamic the paper claims cannot show.

### Issue 2 — `hard_fraction: 0.25` is unwired
The config field exists but `training/adversarial_loop.py` rewrites **all** fakes, not just the top-25% the detector is most confident about. The paper's claim is that the agent focuses on hard examples — the code does not match the claim.

**Fix:** after scoring fakes pre-rewrite, sort by detector confidence and slice to `int(hard_fraction * len(fakes))` before passing to the agent.

---

## What I built this session (file-by-file)

### NEW
- **`configs/config_smoke_bcd.yaml`** — override that compresses the full adversarial loop (5 rounds × 1000 fakes, ~42h) into 200-sample × 2-round smoke (~2.5h on M3 Max bf16). Cuts: epochs (gen 3→1, det 3→2), max_length (384→256), seqgan pretrain 10→3 + adv 30→5, num_rounds 5→2, fake_pool_size 1000→40, BERTScore + perplexity off.
- **`experiments/checkpoints/generator/`** — pre-seeded with 7 symlinks pointing into the BioMistral HF Hub cache snapshot so the loop's `if not (gen_ckpt / "config.json").exists()` check passes and fine-tuning is skipped.
- **`experiments/metrics/condition_B_metrics.json`** + `condition_B_round{0,1,2}_metrics.json`
- **`experiments/metrics/condition_C_metrics.json`** + `condition_C_round0_metrics.json`
- **`experiments/metrics/metrics_log.csv`** — long-form append log of all per-round metrics
- **`experiments/plots/auc_f1_vs_round.png`** + `evasion_rate_vs_round.png`
- **`OVERNIGHT_SUMMARY.md`** — added B/C results table, narratives, known-issue flags

### MODIFIED
- **`experiments/plots/ablation_comparison.png`** — regenerated with A/B/C bars (D legitimately absent)

---

## How to reproduce the smoke run from scratch

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "/Users/venkatrayudu/Workspace/Projects/NOVEL AI PROJECT"

# 1. Real data + BioMistral fakes (~30 min, only needed if data/processed/*.csv missing)
python data/prepare_data.py --config configs/config_novel.yaml --max_real 200 --fakes-source generator

# 2. Condition A (one-shot detector on real data)
python models/detector/train_detector.py --config configs/config_novel.yaml --mode baseline --label condition_A
python evaluation/eval_pipeline.py --config configs/config_novel.yaml --mode test --label condition_A

# 3. Conditions B and C (override config; D omitted)
python training/adversarial_loop.py --config configs/config_smoke_bcd.yaml --condition seqgan_only
python training/adversarial_loop.py --config configs/config_smoke_bcd.yaml --condition agent_only

# 4. Plots
python evaluation/visualization.py --config configs/config_novel.yaml --all_conditions
```

---

## What's next (paper-run path)

1. **Fix the two known issues above** — they're 5-line changes each.
2. **Bump data scale to 10k** in `configs/config_novel.yaml` (`data.max_real_samples: 10000`). Budget ~8–12h for BioMistral zero-shot fakes; run with `caffeinate -dims`.
3. **Run all four conditions A→D** at full scale (5 rounds × 1000 fakes). ~3–5h per condition on M3 Max.
4. **Run RAID and M4 cross-benchmarks** with `scripts/raid_benchmark.py` against the Condition D detector checkpoint. Strengthens the paper against the "evaluated only on your own generation stack" reviewer objection.
5. **Download Llama-3.2-3B-Instruct** (gated on HF) to unblock transfer-attack eval.

---

## Environment notes (Apple Silicon / MPS gotchas)

- **MPS env hardening:** `harden_mps_env()` in `utils/env.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` + `TOKENIZERS_PARALLELISM=false`. Call it once at process start.
- **BioMistral has only `.bin` weights** (no safetensors). The loader pattern in `data/prepare_data.py::load_generator_fakes` is: `use_safetensors=False`, `local_files_only=True`, `low_cpu_mem_usage=True`, plus a monkey-patch to `huggingface_hub.constants.HF_HUB_OFFLINE = True` *inside* the function to bypass transformers 5.x's remote safetensors auto-conversion check. Do NOT set `TRANSFORMERS_OFFLINE` globally — `datasets.load_dataset(...)` for PubMed streaming needs HF Hub reachability.
- **Memory:** BioMistral-7B + Qwen2.5-7B at bf16 each peak around ~15 GB RSS. M3 Max with 64 GB unified memory handles this comfortably; do not load both simultaneously without `offload_when_idle: true`.
- **Clamshell-mode-safe:** verified during this session — AC + external display + Bluetooth input device → training survives lid close.

---

## Current state of the repo (verified at handoff time)

```
git status               → clean
git rev-parse HEAD       → 82d188b
git rev-parse origin/main → 82d188b (in sync)
data/processed/*.csv     → 200 real + 200 fake, 80/10/10 stratified
experiments/checkpoints/detector/condition_A/  → trained BiomedBERT-Large
experiments/checkpoints/generator/             → 7 symlinks → BioMistral HF cache
experiments/metrics/condition_{A,B,C}_metrics.json  → all present, D absent
experiments/plots/{ablation_comparison,auc_f1_vs_round,evasion_rate_vs_round}.png
```

---

## Direct quotes from the user this session (for tone calibration)

- *"i have dataset concerns is 200 samples enough or we are just doin the smoke testing on how it is workin"* → user wants honest scale analysis, not hand-waving
- *"do option 2"* → run B/C/D smoke at 200 to de-risk before 10k
- *"stop after conditoin c"* → mid-flight pivot, drop D
- *"how do we do the condition a? explain in simple way"* → prefers plain-English explanations over jargon

User's tone is direct and outcome-focused. Skip the preamble; lead with what shipped and what's next.
