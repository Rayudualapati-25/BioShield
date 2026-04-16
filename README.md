# Deepfake Robustness Pipeline — Novelty Upgrade (Apple Silicon)

Scaffolded from `PRD.md` v2.1 and `Novelty_Upgrade_Experiment_Plan.docx`. All models, conditions, and metrics follow the PRD exactly — nothing re-invented.

## Hardware

Apple M3 Max · 40-core GPU · 64 GB Unified Memory · PyTorch MPS backend.

CUDA-only tools (`bitsandbytes`, `flash-attn`, fused AdamW) are not used.
`utils/device.py` is the single source of truth for device + dtype + memory ops.

## Layout

```
NOVEL AI PROJECT/
├── PRD.md                               # Ground truth (v2.1 MPS)
├── Novelty_Upgrade_Experiment_Plan.docx # Ground truth (Apple Silicon edition)
├── configs/
│   ├── config_novel.yaml                # PRD-faithful production config
│   └── config_dryrun.yaml               # tiny-model overlay for CI / dry runs
├── utils/                               # device, config, seed, logging, env
├── data/prepare_data.py                 # PubMed stream + Med-MMHL; synthetic fallback
├── models/
│   ├── detector/train_detector.py       # PubMedBERT (bf16, MPS)
│   ├── generator/train_generator.py     # BioMedLM (bf16, MPS) + fake-pool sampler
│   └── seqgan/train_seqgan.py           # Ablation-B LSTM generator
├── agents/adversarial_agent.py          # Phi-3.5-mini + LoRA (bf16, MPS)
├── training/adversarial_loop.py         # 4 ablation conditions (A/B/C/D)
├── evaluation/
│   ├── metrics.py                       # classification + BERTScore + perplexity
│   ├── eval_pipeline.py                 # --mode test / --mode transfer (Llama-3.2)
│   └── visualization.py                 # all 4 PRD plots (matplotlib Agg)
├── experiments/                         # metrics/, plots/, round_data/, checkpoints/
├── tests/test_smoke.py                  # CUDA-leakage grep + unit smoke tests
└── requirements.txt
```

## Quickstart — dry run on MPS (≈1–2 min)

The dry-run overlay swaps model names (`prajjwal1/bert-tiny`, `sshleifer/tiny-gpt2`) so it finishes quickly. **Production experiments use `config_novel.yaml` with the PRD-specified models untouched.**

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
python data/prepare_data.py --config configs/config_novel.yaml --max_real 10000
python models/detector/train_detector.py --config configs/config_novel.yaml --mode baseline --label condition_A
caffeinate -dims python models/generator/train_generator.py --config configs/config_novel.yaml
python models/seqgan/train_seqgan.py --config configs/config_novel.yaml

caffeinate -dims python training/adversarial_loop.py --config configs/config_novel.yaml --condition full_pipeline
caffeinate -dims python training/adversarial_loop.py --config configs/config_novel.yaml --condition seqgan_only
caffeinate -dims python training/adversarial_loop.py --config configs/config_novel.yaml --condition agent_only

python evaluation/eval_pipeline.py --config configs/config_novel.yaml --mode transfer --transfer_generator meta-llama/Llama-3.2-3B-Instruct --n_fakes 500
python evaluation/visualization.py --config configs/config_novel.yaml --all_conditions
```

Estimated wall-clock on M3 Max: ~25–40 hrs serial.

## Tests

```bash
pytest -q tests/
```

`test_no_direct_cuda_calls_in_source` enforces the MPS discipline: no module outside `utils/device.py` may reference `torch.cuda.*`, `bitsandbytes`, `BitsAndBytesConfig`, or `nvidia-smi`.

## Apple Silicon gotchas — baked into the code

| Concern | Where it lives |
|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | `utils/env.py :: harden_mps_env()` called at every entrypoint |
| `pin_memory=False`, `persistent_workers=True` | All DataLoader constructions |
| `torch.autocast("mps", dtype=bfloat16)` | bf16 weights + no `GradScaler` |
| `attn_implementation="eager"` | Agent + transfer-attacker loaders |
| `fused=False` on AdamW | Detector + generator optimizers |
| `torch.mps.empty_cache()` between stages | `utils.device.empty_cache` |
| `matplotlib.use("Agg")` | `evaluation/visualization.py` |
| `caffeinate -dims` for overnight runs | Production commands in this README |
