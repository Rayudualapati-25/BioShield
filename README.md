# BioShield — Deepfake Robustness Pipeline (Apple Silicon edition)

Scaffolded from `PRD.md` v2.1 and `Novelty_Upgrade_Experiment_Plan.docx`. All experimental
conditions and metrics follow the PRD exactly — the only deviations are **model upgrades**
(documented in each module's docstring) and **Apple Silicon plumbing** (bf16 on MPS, no
`bitsandbytes` / `flash-attn` / fused-CUDA optimizers).

## Hardware

Apple M3 Max · 40-core GPU · 64 GB Unified Memory · PyTorch MPS backend.

CUDA-only tools (`bitsandbytes`, `flash-attn`, fused AdamW) are **not used**.
`utils/device.py` is the single source of truth for device + dtype + memory ops.

## Production models

| Role | Model | Size | Why |
|---|---|---|---|
| Generator (initial fakes + fine-tune) | `BioMistral/BioMistral-7B` | 7B | Mistral arch (2024), continued pre-training on PubMed Central — strong biomedical generation |
| Detector | `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` | 340M | Larger encoder than PubMedBERT-base (110M) for stronger discriminative capacity |
| Adversarial rewrite agent | `Qwen/Qwen2.5-7B-Instruct` | 7B | Best-in-class instruction following at this size; LoRA-friendly (q/k/v/o_proj naming) |
| Transfer attacker (novel contribution) | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Different model family — tests cross-generator robustness |

All loaded in `bfloat16` with `attn_implementation="eager"`. Peak single-stage
memory budget: ~16 GB unified RAM. Load times on M3 Max: BioMistral 13s, Qwen 12s.

## Layout

```
NOVEL AI PROJECT/
├── PRD.md                               # Ground truth (v2.1 MPS)
├── Novelty_Upgrade_Experiment_Plan.docx # Ground truth (Apple Silicon edition)
├── configs/
│   ├── config_novel.yaml                # PRD-faithful production config
│   └── config_dryrun.yaml               # tiny-model overlay for CI / dry runs
├── utils/                               # device, config, seed, logging, env
├── data/prepare_data.py                 # PubMed (ccdv/pubmed-summarization) stream
│                                        # + BioMistral zero-shot initial fakes
│                                        # + synthetic fallback for dry runs
├── models/
│   ├── detector/train_detector.py       # BiomedBERT-Large (340M, bf16, MPS)
│   ├── generator/train_generator.py     # BioMistral-7B (bf16, MPS) + fake-pool sampler
│   └── seqgan/train_seqgan.py           # Ablation-B LSTM generator
├── agents/adversarial_agent.py          # Qwen2.5-7B-Instruct + LoRA (bf16, MPS)
├── training/adversarial_loop.py         # 4 ablation conditions (A/B/C/D)
├── evaluation/
│   ├── metrics.py                       # classification + BERTScore + perplexity
│   ├── eval_pipeline.py                 # --mode test / --mode transfer (Llama-3.2)
│   └── visualization.py                 # all 4 PRD plots (matplotlib Agg)
├── scripts/
│   ├── memory_check.py                  # BioMistral + Qwen2.5 MPS smoke (bf16)
│   └── raid_benchmark.py                # RAID/M4 cross-benchmark scaffold (novel contribution)
├── experiments/                         # metrics/, plots/, round_data/, checkpoints/
├── tests/test_smoke.py                  # CUDA-leakage grep + unit smoke tests
├── download_models.py                   # bulk snapshot_download for the 4 models
├── OVERNIGHT_SUMMARY.md                 # Status board: what ran, what's ready, what's next
└── requirements.txt
```

## Data flow (3 lines)

1. `data/prepare_data.py` streams real PubMed abstracts (`ccdv/pubmed-summarization`)
   and generates matching *initial fakes* by zero-shot-prompting BioMistral-7B.
2. `models/detector/train_detector.py` trains Condition-A BiomedBERT-Large on that
   balanced corpus and saves a checkpoint.
3. `training/adversarial_loop.py` runs N rounds of {fake-pool regeneration →
   detector re-training} under 4 ablation conditions (A=static, B=SeqGAN-only,
   C=agent-only, D=full pipeline).

No external fake dataset (Med-MMHL) is required — fakes always come from the same
model family the detector will face at every round, avoiding distribution shift.

## Quickstart — dry run on MPS (≈1–2 min)

The dry-run overlay swaps the 7B models for tiny stand-ins (`prajjwal1/bert-tiny`,
`sshleifer/tiny-gpt2`) so it finishes in seconds on CPU. **Production experiments
use `config_novel.yaml` with the models above.**

```bash
cd "NOVEL AI PROJECT"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# 1. Synthetic fixture (no network)
python data/prepare_data.py --config configs/config_dryrun.yaml

# 2. Condition A — static baseline detector on MPS
python models/detector/train_detector.py --config configs/config_dryrun.yaml --mode baseline --label condition_A

# 3. Evaluate on the test split
python evaluation/eval_pipeline.py --config configs/config_dryrun.yaml --mode test --label condition_A

# 4. Plots
python evaluation/visualization.py --config configs/config_dryrun.yaml --all_conditions
```

Artifacts land in `experiments/metrics/condition_A_metrics.json` and `experiments/plots/`.

## Full experiment (PRD run order §6)

```bash
# 0. (one-time) download all 4 production model weights
python download_models.py

# 1. MPS memory sanity check (loads BioMistral + Qwen2.5, generates 32 tokens each)
python scripts/memory_check.py

# 2. Real PubMed + BioMistral zero-shot initial fakes
python data/prepare_data.py --config configs/config_novel.yaml --max_real 10000 --fakes-source generator

# 3. Static-baseline detector (Condition A)
python models/detector/train_detector.py --config configs/config_novel.yaml --mode baseline --label condition_A

# 4. Fine-tune BioMistral on real PubMed (used as the generator inside conditions C/D)
caffeinate -dims python models/generator/train_generator.py --config configs/config_novel.yaml

# 5. SeqGAN (Condition B)
python models/seqgan/train_seqgan.py --config configs/config_novel.yaml

# 6. Adversarial loop — run each ablation condition
caffeinate -dims python training/adversarial_loop.py --config configs/config_novel.yaml --condition full_pipeline
caffeinate -dims python training/adversarial_loop.py --config configs/config_novel.yaml --condition seqgan_only
caffeinate -dims python training/adversarial_loop.py --config configs/config_novel.yaml --condition agent_only

# 7. Cross-generator transfer attack (Llama-3.2-3B-Instruct → Condition D detector)
python evaluation/eval_pipeline.py --config configs/config_novel.yaml --mode transfer \
    --transfer_generator meta-llama/Llama-3.2-3B-Instruct --n_fakes 500

# 8. (optional, paper strengthening) cross-benchmark against RAID / M4
python scripts/raid_benchmark.py --config configs/config_novel.yaml --benchmark raid --n_samples 500
python scripts/raid_benchmark.py --config configs/config_novel.yaml --benchmark m4   --n_samples 500

# 9. All PRD plots (ROC curves, adversarial round metrics, evasion rate, transfer degradation)
python evaluation/visualization.py --config configs/config_novel.yaml --all_conditions
```

Estimated wall-clock on M3 Max: ~25–40 hrs serial (caffeinate keeps the Mac awake).

## Tests

```bash
pytest -q tests/
```

`test_no_direct_cuda_calls_in_source` enforces the MPS discipline: no module outside
`utils/device.py` may reference `torch.cuda.*`, `bitsandbytes`, `BitsAndBytesConfig`, or
`nvidia-smi`.

## Apple Silicon gotchas — baked into the code

| Concern | Where it lives |
|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | `utils/env.py :: harden_mps_env()` called at every entrypoint |
| `pin_memory=False`, `persistent_workers=True` | All DataLoader constructions |
| `torch.autocast("mps", dtype=bfloat16)` | bf16 weights + no `GradScaler` |
| `attn_implementation="eager"` | Agent + transfer-attacker + generator loaders |
| `fused=False` on AdamW | Detector + generator optimizers |
| `use_safetensors=False` on BioMistral | BioMistral ships only `pytorch_model.bin`; auto-convert PR can fail offline |
| `torch.mps.empty_cache()` between stages | `utils.device.empty_cache` |
| `matplotlib.use("Agg")` | `evaluation/visualization.py` |
| `caffeinate -dims` for overnight runs | Production commands in this README |
