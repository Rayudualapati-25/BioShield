# PRD — Deepfake Robustness Pipeline: Novelty Upgrade (Apple Silicon Edition)
**Version:** 2.1 (MPS / Apple Silicon)
**Author:** Rayudu
**Hardware:** Apple M3 Max · 40-core GPU · 64 GB Unified Memory · 1 TB SSD
**Backend:** PyTorch MPS (Metal Performance Shaders)
**Goal:** Upgrade existing pipeline to paper-ready state with novel contributions

---

## 1. Objective

Transform the existing Deepfake Robustness Pipeline into a publishable research system by:
1. Replacing weak models with domain-specialized alternatives
2. Adding the 4-condition ablation experiment
3. Adding the transfer attack experiment (the novel contribution)
4. Scaling data from 2,000 → 10,000 real samples

**Target venues:** BioNLP @ ACL, EMNLP Findings, AMIA

---

## 2. Model Changes (One-Line Swaps)

All changes are in `configs/config_novel.yaml` (already created).

| Component | Old Model | New Model | HuggingFace ID |
|---|---|---|---|
| Generator | GPT-2 Small | **BioMedLM-2.7B** | `stanford-crfm/BioMedLM` |
| Detector | BioBERT-base-v1.2 | **PubMedBERT-base** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Agent | BioGPT + LoRA | **Phi-3.5-mini-instruct + LoRA (bf16)** | `microsoft/Phi-3.5-mini-instruct` |
| SeqGAN | Main pipeline | **Ablation Condition B only** | from scratch (LSTM, unchanged) |
| Transfer attacker | N/A (new) | **Llama-3.2-3B-Instruct (bf16)** | `meta-llama/Llama-3.2-3B-Instruct` |

> **Apple Silicon note.** `bitsandbytes` has no Metal/MPS backend, so 4-bit NF4 quantization is unavailable on M3 Max. With 64 GB unified memory there is no need for it — all models run comfortably at `bf16`. LoRA adapters are applied on top of the frozen bf16 base, giving the same parameter-efficient fine-tuning benefit without CUDA-only dependencies.

---

## 3. Required Code Changes

### 3.0 — Device Helper (`utils/device.py`)

Create a single source of truth for device selection to keep the rest of the pipeline CUDA-free:

```python
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

def allocated_memory_gb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0
```

Import `get_device`, `empty_cache`, and `allocated_memory_gb` everywhere instead of raw `torch.cuda.*` calls.

---

### 3.1 — Add Generator Module (`models/generator/train_generator.py`)

The existing pipeline has `models/seqgan/` but no module for fine-tuning a HuggingFace causal LM (BioMedLM) as the generator. This file needs to be created.

**What it must do:**
- Load `stanford-crfm/BioMedLM` via HuggingFace `AutoModelForCausalLM` with `torch_dtype=torch.bfloat16`
- Move to MPS via `.to(get_device())`
- Fine-tune on PubMed real abstracts with causal LM objective (3 epochs, lr=2e-5)
- Use **gradient accumulation** (physical batch 2, accum steps 8 → effective batch 16) to fit comfortably in unified memory
- Enable `gradient_checkpointing=True` on the HF model
- Generate fake medical abstracts given a topic prompt
- Save/load checkpoint via `torch.save` / `torch.load(map_location=get_device())`
- Expose `generate_fake_pool(n, prompt_template)` method
- Accept `--config` argument pointing to `config_novel.yaml`

**Precision policy (MPS):**
- Weights + activations: `bfloat16`
- No `torch.cuda.amp.GradScaler` (not needed with bf16, and it targets CUDA)
- Wrap forward with `torch.autocast(device_type="mps", dtype=torch.bfloat16)` only where stable; otherwise keep pure bf16 throughout

**Generator prompt template:**
```
Write a convincing but fictitious biomedical abstract about {topic}.
Follow standard scientific writing conventions with methods, results, and conclusion.
Abstract:
```

---

### 3.2 — Upgrade Agent (`agents/adversarial_agent.py`)

Change from BioGPT to Phi-3.5-mini-instruct. The existing file structure is correct — update model loading and LoRA targets.

**Changes required:**
- Replace `microsoft/biogpt` loading with `microsoft/Phi-3.5-mini-instruct`
- Load with `torch_dtype=torch.bfloat16, device_map={"": get_device()}` (no `BitsAndBytesConfig`)
- Update LoRA target modules to Phi-3.5 attention layers: `["q_proj", "v_proj", "k_proj", "o_proj"]`
- Freeze base model; train only LoRA adapters (rank 16, alpha 32, dropout 0.05)
- Use `torch.optim.AdamW` with `fused=False` (fused optimizers are CUDA-only)

**Removed (CUDA-only, do not port):**
```python
# DELETED — bitsandbytes is CUDA-only, unavailable on MPS
# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
```

**Replacement (bf16 full-precision base + LoRA adapters):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from utils.device import get_device

device = get_device()

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # FlashAttention 2 is CUDA-only
).to(device)
base.gradient_checkpointing_enable()

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)
```

**Rewrite prompt format (unchanged — Phi-3.5 chat template):**
```
<|user|>
Rewrite the following medical abstract to sound more credible and harder to detect as AI-generated. Preserve all core claims. Output only the rewritten abstract.

Original: {fake_abstract}
<|end|>
<|assistant|>
```

---

### 3.3 — Upgrade Detector (`models/detector/train_detector.py`)

Change model name only. Verify precision and device settings.

**Change:**
```python
# Before
model_name = "dmis-lab/biobert-base-cased-v1.2"

# After (read from config)
model_name = cfg["detector"]["model_name"]
# = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
```

**MPS-specific settings inside the trainer:**
- `torch_dtype=torch.bfloat16` for the HF model
- `device = get_device()`; `model.to(device)`
- DataLoader: `pin_memory=False`, `persistent_workers=True`, `num_workers=4`
- No `apex` / `fp16` mixed-precision; use native bf16

Verify the config key is read correctly and not hardcoded anywhere else.

---

### 3.4 — Add Ablation Support (`training/adversarial_loop.py`)

The loop needs a `--condition` flag that runs one of four ablation conditions:

```python
parser.add_argument(
    "--condition",
    choices=["full_pipeline", "seqgan_only", "agent_only", "static_baseline"],
    default="full_pipeline",
)
```

**Condition logic:**
- `static_baseline`: Skip loop entirely. Only train baseline detector (round 0). Save metrics as `condition_A_metrics.json`.
- `seqgan_only`: Use SeqGAN generator (not BioMedLM). Skip Phi-3.5 rewriting step. Retrain detector normally. Save as `condition_B_metrics.json`.
- `agent_only`: Use BioMedLM + Phi-3.5 rewrites. Do NOT retrain detector each round. Save as `condition_C_metrics.json`.
- `full_pipeline`: Default — BioMedLM + Phi-3.5 + iterative retraining. Save as `condition_D_metrics.json`.

**Memory hygiene between steps (MPS version):**
```python
from utils.device import empty_cache, get_device
device = get_device()
# ... after each model step
del model
import gc
gc.collect()
empty_cache(device)
```

---

### 3.5 — Add Transfer Attack (`evaluation/eval_pipeline.py`)

Add `--mode transfer` flag to the evaluation script.

**What it must do:**
1. Load the trained PubMedBERT detector from checkpoint (post-loop, Condition D)
2. Load `meta-llama/Llama-3.2-3B-Instruct` with `torch_dtype=torch.bfloat16` (no 4-bit; fits natively in 64 GB)
3. Generate `n_fakes` (default 500) fake medical abstracts using Llama-3.2
4. Run detector on these fakes + equal number of real PubMed abstracts
5. Compute and save: Transfer AUC, Transfer F1, Transfer Accuracy
6. Save results to `experiments/metrics/transfer_attack_results.json`
7. Print comparison: In-distribution AUC vs Transfer AUC

**CLI:**
```bash
python evaluation/eval_pipeline.py \
  --config configs/config_novel.yaml \
  --mode transfer \
  --transfer_generator meta-llama/Llama-3.2-3B-Instruct \
  --n_fakes 500
```

---

### 3.6 — Activate Rewrite Quality Metrics (`evaluation/metrics.py`)

The BERTScore and perplexity code already exists in `evaluation/metrics.py` but is gated. Activate it:

- Set `compute_bertscore: true` in config → compute BERTScore F1 between original fake and rewritten fake
- Set `compute_perplexity: true` → compute perplexity of rewritten text using PubMedBERT as reference LM
- When invoking `bert-score`, pass `device="mps"` explicitly (its default CUDA detection silently falls back to CPU otherwise)
- Log both metrics per round alongside evasion rate
- Add to `metrics_log.csv`: columns `bertscore_f1`, `mean_perplexity`, `evasion_rate`, `auc`, `f1`

---

### 3.7 — Scale Data (`data/prepare_data.py`)

Add `--max_real` CLI argument:

```bash
python data/prepare_data.py --config configs/config_novel.yaml --max_real 10000
```

- Stream 10,000 real PubMed abstracts (not 2,000)
- Fake data: use Med-MMHL (unchanged)
- Apply all existing cleaning steps (HTML strip, dedup, min_words filter)
- Save to `data/processed/train.csv`, `val.csv`, `test.csv`
- Also create `data/processed/transfer_test.csv` placeholder (filled by eval_pipeline.py transfer mode)

---

### 3.8 — Update Visualization (`evaluation/visualization.py`)

Add two new plots:

**Plot 1: Ablation Comparison Bar Chart**
- X-axis: Condition A, B, C, D
- Y-axis: Final AUC and Final F1 (grouped bars)
- Save as `experiments/plots/ablation_comparison.png`

**Plot 2: Transfer Attack Result**
- Grouped bar chart: In-distribution AUC vs Transfer AUC for Condition D detector
- Save as `experiments/plots/transfer_attack_results.png`

**Existing plots to keep:**
- `evasion_rate_vs_round.png`
- `auc_f1_vs_round.png`

Use `matplotlib` with `Agg` backend (`matplotlib.use("Agg")`) so plots render headless without a display server.

---

## 4. Config File

Use `configs/config_novel.yaml` (already created in the project folder).

**Key settings for M3 Max 64 GB:**
```yaml
runtime:
  device: "mps"              # "cuda" / "cpu" fallback handled in utils/device.py
  dtype: "bfloat16"          # bf16 is the MPS-stable choice for transformers
  num_workers: 4             # DataLoader workers; 4–8 on M3 Max
  pin_memory: false          # required for MPS
  persistent_workers: true

generator:
  model_name: "stanford-crfm/BioMedLM"
  dtype: "bfloat16"
  gradient_checkpointing: true
  per_device_batch_size: 2
  grad_accum_steps: 8        # effective batch size 16
  offload_when_idle: true    # del + empty_cache between pipeline stages

detector:
  model_name: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
  batch_size: 16             # fits comfortably in 64 GB unified memory
  max_length: 512
  dtype: "bfloat16"

agent:
  model_name: "microsoft/Phi-3.5-mini-instruct"
  dtype: "bfloat16"          # replaces 4-bit NF4; bitsandbytes is CUDA-only
  attn_implementation: "eager"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

transfer_attacker:
  model_name: "meta-llama/Llama-3.2-3B-Instruct"
  dtype: "bfloat16"

loop:
  num_rounds: 5
  fake_pool_size: 1000
  clear_vram_between_steps: true
```

---

## 5. Memory Management Rule (MPS)

**After every model step:**
```python
import gc, torch
from utils.device import empty_cache, get_device

device = get_device()
del model
gc.collect()
empty_cache(device)           # torch.mps.empty_cache() + torch.mps.synchronize()
```

With 64 GB unified memory, simultaneous residence of BioMedLM and PubMedBERT is technically feasible, but keeping the del/empty_cache discipline avoids fragmentation during long runs. Continue to guarantee: never hold both the generator (bf16 ~5.4 GB) and the agent (bf16 ~7.6 GB) plus their optimizer states at the same time — peak would otherwise exceed ~40 GB with AdamW moments.

---

## 6. Run Order (Sequential)

```bash
# 1. Data
python data/prepare_data.py --config configs/config_novel.yaml --max_real 10000

# 2. Baseline (Condition A)
python models/detector/train_detector.py --config configs/config_novel.yaml --mode baseline

# 3. Train generator
python models/generator/train_generator.py --config configs/config_novel.yaml

# 4. Train SeqGAN (for ablation only)
python models/seqgan/train_seqgan.py --config configs/config_novel.yaml

# 5. Full pipeline loop (Condition D — main result)
python training/adversarial_loop.py --config configs/config_novel.yaml --condition full_pipeline

# 6. Ablation Condition B
python training/adversarial_loop.py --config configs/config_novel.yaml --condition seqgan_only

# 7. Ablation Condition C
python training/adversarial_loop.py --config configs/config_novel.yaml --condition agent_only

# 8. Transfer attack
python evaluation/eval_pipeline.py --config configs/config_novel.yaml --mode transfer --transfer_generator meta-llama/Llama-3.2-3B-Instruct --n_fakes 500

# 9. Generate all plots
python evaluation/visualization.py --config configs/config_novel.yaml --all_conditions
```

Set these environment variables once per shell:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1        # fall back to CPU for ops missing on MPS
export TOKENIZERS_PARALLELISM=false         # avoid fork warnings with num_workers>0
export HF_HOME=$HOME/.cache/huggingface     # or a path on your 1 TB SSD
```

---

## 7. Expected Output Files

After all runs complete, these files should exist:

```
experiments/
├── metrics/
│   ├── condition_A_metrics.json       # static baseline
│   ├── condition_B_metrics.json       # seqgan only
│   ├── condition_C_metrics.json       # agent only
│   ├── condition_D_metrics.json       # full pipeline (main result)
│   ├── metrics_log.csv                # round-level: auc, f1, evasion_rate, bertscore_f1
│   └── transfer_attack_results.json   # transfer AUC vs in-dist AUC
├── plots/
│   ├── auc_f1_vs_round.png
│   ├── evasion_rate_vs_round.png
│   ├── ablation_comparison.png        # NEW
│   └── transfer_attack_results.png    # NEW
└── round_data/
    ├── round_1/ ... round_5/
```

---

## 8. Paper Result Tables

The paper needs these result tables. The code should produce CSVs that map directly to them.

### Table 1: Ablation Results (main result)
| Condition | Generator | Agent | Loop | Final AUC | Final F1 | Evasion Rate (R5) |
|---|---|---|---|---|---|---|
| A — Static | — | — | No | ? | ? | — |
| B — SeqGAN | SeqGAN | — | Yes | ? | ? | ? |
| C — Agent only | BioMedLM | Phi-3.5 | No | ? | ? | ? |
| D — Full (MAIN) | BioMedLM | Phi-3.5 | Yes | ? | ? | ? |

### Table 2: Transfer Attack
| Detector | In-dist AUC | Transfer AUC | Gap |
|---|---|---|---|
| Condition A (static) | ? | ? | ? |
| Condition D (adversarial) | ? | ? | ? |

---

## 9. Dependencies (Apple Silicon / MPS)

```bash
# Core (build for Apple Silicon wheels)
pip install "torch>=2.3" "torchvision>=0.18"          # macOS arm64 wheels include MPS
pip install "transformers>=4.44"
pip install "accelerate>=0.33"                        # multi-device + bf16 handling
pip install "peft>=0.11.0"                            # LoRA
pip install "bert-score>=0.3.13"                      # rewrite quality (runs on MPS)
pip install "evaluate>=0.4.1"                         # metrics
pip install "datasets>=2.19"                          # streaming PubMed
pip install "sentencepiece" "protobuf" "safetensors"
pip install "matplotlib" "scikit-learn" "pandas" "pyyaml" "tqdm"
```

**Explicitly NOT installed on M3 Max (CUDA-only):**
- `bitsandbytes` — no Metal backend; 4-bit NF4 loading is unavailable.
- `flash-attn` — CUDA kernels only; use `attn_implementation="eager"`.
- `apex` / `DeepSpeed` fused kernels — CUDA-only.
- `triton` — limited MPS support; don't rely on it.

If a transitive dependency pulls `bitsandbytes`, install the CPU-only stub package `bitsandbytes-foundation` **only if** a library guards its import behind a feature flag; otherwise remove the offending caller.

---

## 10. Definition of Done

- [ ] `utils/device.py` exposes `get_device`, `empty_cache`, `allocated_memory_gb` and is used everywhere
- [ ] `models/generator/train_generator.py` created; loads BioMedLM in bf16 on MPS
- [ ] `agents/adversarial_agent.py` uses Phi-3.5 bf16 + LoRA (no `BitsAndBytesConfig`)
- [ ] `models/detector/train_detector.py` reads model name from config (PubMedBERT, bf16, MPS)
- [ ] `training/adversarial_loop.py` supports `--condition` flag for all 4 ablations
- [ ] `evaluation/eval_pipeline.py` supports `--mode transfer` with Llama-3.2-3B bf16
- [ ] `evaluation/metrics.py` activates BERTScore (device=mps) and perplexity
- [ ] `evaluation/visualization.py` generates ablation bar chart and transfer attack chart (matplotlib Agg backend)
- [ ] `data/prepare_data.py` supports `--max_real 10000`
- [ ] All 4 ablation conditions run to completion on MPS and produce metrics JSON files
- [ ] Transfer attack produces `transfer_attack_results.json`
- [ ] All plots generated in `experiments/plots/`
- [ ] Peak unified-memory footprint stays ≤ 40 GB during any single stage (verified via `torch.mps.current_allocated_memory()`)
- [ ] `grep -R "cuda\|bitsandbytes\|fp16=True\|nvidia-smi" src/` returns zero hits

---

## 11. Expected Wall-Clock on M3 Max (40-core GPU)

Rough real-world estimates; MPS is typically 2–4× slower than an RTX 4090 for LLM training, faster than a 4060 for inference due to memory bandwidth.

| Stage | Time |
|---|---|
| Data prep (10k PubMed stream + clean) | 1–2 hrs |
| Baseline PubMedBERT (Condition A) | 1–2 hrs |
| BioMedLM fine-tune (3 epochs, eff. batch 16) | 6–10 hrs |
| SeqGAN training (ablation only) | 3–4 hrs |
| Full loop, 5 rounds (Condition D) | 6–10 hrs |
| Condition B loop | 4–6 hrs |
| Condition C loop (no retrain) | 2–3 hrs |
| Llama-3.2 transfer gen (500 samples, bf16) | 45–90 min |
| Rewrite metrics + plots | 1–2 hrs |
| **Total (serial)** | **~25–40 hrs** |

Use `tmux` or `caffeinate -dims python …` so macOS App Nap does not throttle long runs.
