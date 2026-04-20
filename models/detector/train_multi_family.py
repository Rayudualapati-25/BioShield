"""Multi-family detector training — Condition E.

Addresses the generator-family dominance finding in BioShield §V.C:
a detector trained on only BioMistral fakes achieves near-zero evasion on
BioMistral (0.7%) but 98.7% on SeqGAN and 98.3% on RAID's mixed LLMs.

This module trains a BiomedBERT-Large detector on a *mixture* of fake
families so the decision surface covers multiple generator distributions:
  - BioMistral-7B (in-family LLM)
  - SeqGAN (LSTM + REINFORCE, out-of-family statistical)
  - Optional: Llama-3.2-3B transfer fakes

The expected outcome (per discussion in §VI.A) is that multi-family training
compresses the 140× evasion spread at the cost of marginally lower in-family
AUC. Whether that tradeoff is favourable depends on the deployment setting.

CLI:
    python models/detector/train_multi_family.py \
        --config configs/config_novel.yaml \
        --label condition_E \
        [--seqgan_csv experiments/round_data/condition_B/round_3/fakes.csv] \
        [--biomistral_csv data/processed/train.csv] \
        [--llama_csv data/processed/transfer_test.csv] \
        [--mixing_ratios 1.0,1.0,0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.detector.dataset import TextClassificationDataset  # noqa: E402
from models.detector.train_detector import (  # noqa: E402
    eval_split,
    save_checkpoint,
    train_one_epoch,
)
from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.device import (  # noqa: E402
    describe,
    empty_cache,
    get_device,
    resolve_dtype,
    sync,
)
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("detector.multi_family")


def _build_multi_family_train(
    base_train_csv: str,
    seqgan_csv: str | None,
    llama_csv: str | None,
    mixing_ratios: list[float],
    output_csv: Path,
    seed: int = 42,
) -> None:
    """Merge real PubMed + multiple fake families into one balanced training CSV.

    mixing_ratios: [biomistral_weight, seqgan_weight, llama_weight]. Weights
    control the sampling fraction relative to the number of real samples.
    For example, [1.0, 1.0, 0.5] gives equal BioMistral and real, half as many
    SeqGAN, and a quarter as many Llama fakes.
    """
    base_df = pd.read_csv(base_train_csv)
    real_df = base_df[base_df["label"] == 1] if "label" in base_df.columns else base_df
    bio_df = (
        base_df[base_df["label"] == 0] if "label" in base_df.columns else pd.DataFrame()
    )
    n_real = len(real_df)

    rng = np.random.RandomState(seed)
    parts: list[pd.DataFrame] = [real_df]

    # BioMistral fakes from the base train CSV
    if len(bio_df) > 0:
        n_bio = int(n_real * mixing_ratios[0])
        sampled = bio_df.sample(
            n=min(n_bio, len(bio_df)), random_state=seed, replace=False
        )
        parts.append(sampled[["text"]].assign(label=0))
        LOG.info(
            "BioMistral fakes: %d (mixing_ratio=%.2f)", len(sampled), mixing_ratios[0]
        )

    # SeqGAN fakes
    if seqgan_csv and Path(seqgan_csv).exists():
        seq_df = pd.read_csv(seqgan_csv)
        n_seq = int(n_real * (mixing_ratios[1] if len(mixing_ratios) > 1 else 1.0))
        sampled_seq = seq_df.sample(
            n=min(n_seq, len(seq_df)), random_state=seed + 1, replace=False
        )
        parts.append(sampled_seq[["text"]].assign(label=0))
        LOG.info(
            "SeqGAN fakes: %d (mixing_ratio=%.2f)",
            len(sampled_seq),
            mixing_ratios[1] if len(mixing_ratios) > 1 else 1.0,
        )
    else:
        LOG.warning(
            "SeqGAN CSV not found — Condition E will train without SeqGAN fakes: %s",
            seqgan_csv,
        )

    # Llama transfer fakes
    if llama_csv and Path(llama_csv).exists():
        llama_df = pd.read_csv(llama_csv)
        llama_fakes = (
            llama_df[llama_df["label"] == 0]
            if "label" in llama_df.columns
            else llama_df
        )
        n_llama = int(n_real * (mixing_ratios[2] if len(mixing_ratios) > 2 else 0.5))
        sampled_llama = llama_fakes.sample(
            n=min(n_llama, len(llama_fakes)), random_state=seed + 2, replace=False
        )
        parts.append(sampled_llama[["text"]].assign(label=0))
        LOG.info(
            "Llama-3.2-3B fakes: %d (mixing_ratio=%.2f)",
            len(sampled_llama),
            mixing_ratios[2] if len(mixing_ratios) > 2 else 0.5,
        )
    else:
        LOG.info(
            "Llama CSV not found or not provided — skipping Llama family in Condition E."
        )

    merged = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)[
        ["text", "label"]
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    label_counts = merged["label"].value_counts().to_dict()
    LOG.info(
        "Multi-family train CSV written: %s | real=%d fake=%d total=%d",
        output_csv,
        label_counts.get(1, 0),
        label_counts.get(0, 0),
        len(merged),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Condition E: multi-family detector (BioMistral + SeqGAN + Llama)"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", default="condition_E")
    parser.add_argument(
        "--seqgan_csv",
        default=None,
        help="CSV of SeqGAN-generated fakes (text column, label=0)",
    )
    parser.add_argument(
        "--llama_csv", default=None, help="CSV of Llama-3.2-3B generated fakes"
    )
    parser.add_argument(
        "--mixing_ratios",
        default="1.0,1.0,0.5",
        help="Comma-separated mixing weights for [BioMistral, SeqGAN, Llama]",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed = int(cfg["runtime"].get("seed", 42))
    set_seed(seed)
    device = get_device(cfg["runtime"].get("device"))
    LOG.info("Condition E (multi-family) training on %s (%s)", device, describe(device))

    mixing_ratios = [float(x) for x in args.mixing_ratios.split(",")]

    # Build multi-family training set
    mf_train_csv = (
        Path(cfg["paths"]["experiments_dir"])
        / "round_data"
        / "condition_E"
        / "train_multifamily.csv"
    )
    _build_multi_family_train(
        base_train_csv=cfg["paths"]["train_csv"],
        seqgan_csv=args.seqgan_csv,
        llama_csv=args.llama_csv,
        mixing_ratios=mixing_ratios,
        output_csv=mf_train_csv,
        seed=seed,
    )

    # Patch config paths to use multi-family train CSV
    mf_cfg = dict(cfg)
    mf_cfg["paths"] = dict(cfg["paths"])
    mf_cfg["paths"]["train_csv"] = str(mf_train_csv)

    # Load model and tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = cfg["detector"]["model_name"]
    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    LOG.info("Loading %s dtype=%s", model_name, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, torch_dtype=dtype, ignore_mismatched_sizes=True
    ).to(device)

    # Build data loaders using patched config
    max_length = int(cfg["detector"]["max_length"])
    batch_size = int(cfg["detector"]["batch_size"])
    num_workers = int(cfg["runtime"].get("num_workers", 2))
    pin_memory = bool(cfg["runtime"].get("pin_memory", False))
    persistent = (
        bool(cfg["runtime"].get("persistent_workers", False)) and num_workers > 0
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    train_loader = DataLoader(
        TextClassificationDataset(str(mf_train_csv), tokenizer, max_length),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TextClassificationDataset(cfg["paths"]["val_csv"], tokenizer, max_length),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TextClassificationDataset(cfg["paths"]["test_csv"], tokenizer, max_length),
        shuffle=False,
        **loader_kwargs,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["detector"]["lr"]),
        weight_decay=float(cfg["detector"].get("weight_decay", 0.01)),
        fused=False,
    )
    criterion = nn.CrossEntropyLoss()
    epochs = int(cfg["detector"]["epochs"])
    ckpt_dir = Path(cfg["detector"]["output_dir"]) / args.label

    best: dict[str, float | int] = {"auc": -1.0, "epoch": -1}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        sync(device)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_split(model, val_loader, device, criterion)
        LOG.info(
            "epoch=%d train_loss=%.4f val_auc=%.4f val_f1=%.4f",
            epoch,
            train_loss,
            val_metrics["auc"],
            val_metrics["f1"],
        )
        if val_metrics["auc"] > best["auc"]:
            best = {"auc": val_metrics["auc"], "epoch": epoch}
            save_checkpoint(
                ckpt_dir, model, tokenizer, optimizer, epoch, val_metrics["auc"]
            )
            LOG.info("Saved best checkpoint (auc=%.4f)", val_metrics["auc"])

    # Final evaluation
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir,
        num_labels=2,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    ).to(device)
    test_metrics = eval_split(model, test_loader, device, criterion)
    elapsed = time.time() - t0
    LOG.info(
        "Condition E done elapsed=%.1fs test_auc=%.4f test_f1=%.4f test_acc=%.4f",
        elapsed,
        test_metrics["auc"],
        test_metrics["f1"],
        test_metrics["accuracy"],
    )

    # Evasion rate on test fakes (label 0 = fake)
    test_df = pd.read_csv(cfg["paths"]["test_csv"])

    label = args.label
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / f"{label}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "multi_family",
        "label": label,
        "model_name": model_name,
        "mixing_ratios": mixing_ratios,
        "seqgan_csv": args.seqgan_csv,
        "llama_csv": args.llama_csv,
        "best_val_auc": best["auc"],
        "best_epoch": best["epoch"],
        "test": test_metrics,
        "elapsed_seconds": elapsed,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Metrics written to %s", metrics_path)
    empty_cache(device)


if __name__ == "__main__":
    main()
