# BioShield — Overnight Run Summary (Apr 16 → Apr 17)

> Status board for the autonomous overnight execution granted at midnight.
> Read this first when you wake up — it's the single source of truth for what
> ran, what's ready to run next, and what needs human decisions.

---

## Baseline: what you left me with

The scaffold was working on synthetic fixtures — a 302-row BiomedBERT-Large
regression was green (AUC 0.50 on the tiny stand-in model, F1 0.69). The 7B
production models (BioMistral, Qwen2.5) were downloaded but unverified on MPS.
The plan was to swap synthetic fakes for real PubMed + BioMistral-generated fakes,
then retrain the detector.

---

## What I did overnight

### 1. Model upgrades baked into `config_novel.yaml`
| Role | Old (PRD default) | New (production) | Rationale |
|---|---|---|---|
| Generator | BioMedLM (2.7B, GPT-2, 2022) | **BioMistral-7B** (Mistral, 2024) | Modern arch; PubMed-Central continued pre-training |
| Detector | PubMedBERT-base (110M) | **BiomedBERT-Large** (340M) | 3× capacity for discrimination |
| Agent | Phi-3.5-mini (3.8B) | **Qwen2.5-7B-Instruct** | Stronger instruction following; LoRA-compatible q/k/v/o naming |
| Transfer attacker | Llama-3.2-3B-Instruct | (unchanged) | — |

### 2. Data pipeline hardening
- Replaced the **deprecated** `ncbi/pubmed` script-based loader (datasets v3+ retired
  dataset scripts) with streaming from **`ccdv/pubmed-summarization`** (`document`
  config, abstract field). Fallback: `armanc/scientific_papers` [pubmed].
- New function `load_generator_fakes()` in `data/prepare_data.py` uses **zero-shot
  BioMistral-7B** for initial fakes. No external Med-MMHL dependency.
- Added `--fakes-source {auto,generator,med_mmhl}` CLI flag.
- Apple-Silicon plumbing for BioMistral (`.bin`-only weights):
  `use_safetensors=False`, `local_files_only=True`, `low_cpu_mem_usage=True`, and a
  **monkey-patch** to `huggingface_hub.constants.HF_HUB_OFFLINE = True` *inside*
  `load_generator_fakes()` to bypass transformers 5.x's remote safetensors
  auto-conversion PR check (which fails on flaky networks even with
  `use_safetensors=False`).

### 3. MPS memory sanity check (`scripts/memory_check.py`)
Loads BioMistral-7B and Qwen2.5-7B on MPS at bf16 with `attn_implementation="eager"`
and generates 32 tokens each.

| Model | Load time | 32-tok gen | Peak RSS | Coherent? |
|---|---|---|---|---|
| BioMistral/BioMistral-7B | 13.1 s | 6.48 s | ~15 GB | ✅ yes — medical abstract form |
| Qwen/Qwen2.5-7B-Instruct | 11.5 s | 3.39 s | ~15 GB | ✅ yes — rewrote abstract credibly |

### 4. RAID / M4 cross-benchmark scaffold (paper strengthening)
New file: `scripts/raid_benchmark.py`. Streams
- `liamdugan/raid` (biomedical slice via `domain` filter)
- `Chardonneret/M4` (`en_multidomain` → default fallback)
and evaluates any saved detector checkpoint (AUC / F1 / accuracy / evasion_rate).
Defensive: any benchmark failure writes a `{skipped: true, reason: ...}` metrics
file instead of crashing the pipeline. Ready to run post-Condition-D training.

### 5. Documentation
- `README.md` rewritten: production model table, data-flow explanation in 3 lines,
  updated full-experiment command sequence, Apple-Silicon gotchas table extended
  with the `use_safetensors=False` + `TRANSFORMERS_OFFLINE` + `local_files_only`
  discipline.
- `.gitignore` tightened: checkpoints / `.bin` / `.pt` / `.safetensors` excluded
  from git; metrics JSON + plots still tracked for paper artifacts.

---

## What's ready to run now

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "NOVEL AI PROJECT"

# 1. Real data + BioMistral fakes (≈ 30 min — ran overnight, see experiments/prepare_data_real.log)
python data/prepare_data.py --config configs/config_novel.yaml --max_real 200 --fakes-source generator

# 2. Condition A detector on real data (≈ 6–10 min for 3 epochs @ batch 16)
python models/detector/train_detector.py --config configs/config_novel.yaml --mode baseline --label condition_A

# 3. Detector evaluation + plots
python evaluation/eval_pipeline.py --config configs/config_novel.yaml --mode test --label condition_A
python evaluation/visualization.py --config configs/config_novel.yaml --all_conditions

# 4. (after condition D exists) RAID / M4 cross-benchmarks
python scripts/raid_benchmark.py --config configs/config_novel.yaml --benchmark raid --n_samples 500
python scripts/raid_benchmark.py --config configs/config_novel.yaml --benchmark m4   --n_samples 500
```

---

## Artifacts to find when you wake up

- `experiments/prepare_data_real.log` — PubMed streaming + BioMistral zero-shot gen log
- `data/processed/{train,val,test}.csv` — real data splits (80/10/10 stratified)
- `experiments/metrics/condition_A_metrics.json` — Condition A test metrics on real data
- `experiments/checkpoints/detector/condition_A/` — trained BiomedBERT-Large weights
- `experiments/plots/*.png` — ROC, evasion-rate, adversarial-rounds plots
- `experiments/memory_check.log` — MPS smoke-test record

### Dev-scale smoke-test results (200-sample run — NOT paper numbers)

| Metric | Value | CI at n=40 test | Notes |
|---|---|---|---|
| Test AUC | **0.9775** | ±0.08 | Validation tracked stably (0.965 → 0.988) |
| Test F1 | **0.9048** | ±0.08 | |
| Test accuracy | **0.9000** | ±0.08 | |
| Evasion rate | **0.15** | ±0.08 | 3/20 BioMistral fakes slipped past the detector |
| Train wall-clock | **176 s** | — | 3 epochs × 320 rows × batch 16 on M3 Max bf16 |
| Data prep wall-clock | **~37 min** | — | 200 BioMistral zero-shot fakes, ~10s/fake |

**Honest read:** these numbers are great *for the smoke test* but statistically noisy at n=40 (CI ±8%). The real paper numbers need 10k-scale data. The purpose of this run was to **prove the pipeline works end-to-end under the new stack** — which it does.

---

## Known issues / open decisions

1. **Llama-3.2-3B-Instruct**: gated on HF, not downloaded yet. You said you'd grab
   it today. Once fetched, transfer-attack eval can run.
2. **Data scale**: overnight used `--max_real 200` to fit within wall-clock. For the
   paper run, bump to `10000` and budget ~8–12 h for BioMistral zero-shot fakes
   (with `caffeinate -dims`).
3. **Full adversarial loop (conditions B/C/D)**: not run yet. That's the main
   daytime task — each condition is ~3–5 h on M3 Max at 5 rounds × 1000 fakes.
4. **ccdv/pubmed-summarization vs clinical PubMed**: the current source is an
   abstracts-plus-articles dataset; not every entry is strictly clinical PubMed.
   If reviewers push back, swap in `EleutherAI/pile-pubmed-central` — the loader
   is structured to accept additional candidates with one line.

---

## Code changes (diff summary)

- `configs/config_novel.yaml` — model names + prompts updated
- `data/prepare_data.py` — new `load_generator_fakes`, `load_pubmed_real` rewritten,
  `_TOPICS` list, offline-mode dance for BioMistral load
- `models/generator/train_generator.py` — `use_safetensors=False` + BioMistral docstring
- `agents/adversarial_agent.py` — docstring re-anchored on Qwen2.5 / LoRA
- `scripts/memory_check.py` — **new** (bf16 MPS smoke for 7B models)
- `scripts/raid_benchmark.py` — **new** (RAID / M4 cross-benchmark)
- `utils/env.py` — comment about `TRANSFORMERS_OFFLINE` discipline
- `.gitignore` — checkpoint / weight blobs excluded
- `README.md` — full rewrite (production model table, data flow, full-experiment block)
- `OVERNIGHT_SUMMARY.md` — **new** (this file)

---

## Next user decisions

- [ ] Download Llama-3.2-3B-Instruct (gated) to unblock transfer-attack eval
- [ ] Choose data scale for paper run: 200 (dev), 2000 (pilot), 10000 (paper)
- [ ] Commit + push to `github.com/Rayudualapati-25/BioShield`
