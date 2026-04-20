"""Detector trainer — PubMedBERT classifier, bf16 on MPS.

PRD §3.3: model name comes from config (`detector.model_name`); all other
settings (batch size, max_length, dtype) respect MPS constraints
(pin_memory=false, persistent_workers, bf16 weights, no fused CUDA optimizers).

CLI:
    python models/detector/train_detector.py --config configs/config_novel.yaml --mode baseline
    python models/detector/train_detector.py --config configs/config_novel.yaml --mode retrain --resume <ckpt>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.detector.dataset import TextClassificationDataset  # noqa: E402
from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.device import (  # noqa: E402
    allocated_memory_gb,
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
LOG = get_logger("detector.train")


def build_loaders(cfg: dict, tokenizer) -> tuple[DataLoader, DataLoader, DataLoader]:
    dcfg = cfg["detector"]
    rcfg = cfg["runtime"]
    max_length = int(dcfg["max_length"])
    batch = int(dcfg["batch_size"])

    train = TextClassificationDataset(cfg["paths"]["train_csv"], tokenizer, max_length)
    val = TextClassificationDataset(cfg["paths"]["val_csv"], tokenizer, max_length)
    test = TextClassificationDataset(cfg["paths"]["test_csv"], tokenizer, max_length)

    kwargs = dict(
        batch_size=batch,
        num_workers=int(rcfg.get("num_workers", 2)),
        pin_memory=bool(rcfg.get("pin_memory", False)),
        persistent_workers=bool(rcfg.get("persistent_workers", False))
        and int(rcfg.get("num_workers", 2)) > 0,
    )
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )


def load_model(cfg: dict, device: torch.device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = cfg["detector"]["model_name"]
    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    LOG.info("Loading detector %s dtype=%s on %s", name, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2, torch_dtype=dtype, ignore_mismatched_sizes=True
    )
    model.to(device)
    return model, tokenizer


def eval_split(model, loader, device, criterion) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    probs: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    n = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits.float()
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
            preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())
            labels.extend(y.detach().cpu().numpy().tolist())

    labels_np = np.asarray(labels)
    probs_np = np.asarray(probs)
    try:
        auc = (
            roc_auc_score(labels_np, probs_np)
            if len(set(labels_np)) > 1
            else float("nan")
        )
    except ValueError:
        auc = float("nan")
    f1 = f1_score(labels_np, preds, zero_division=0)
    acc = accuracy_score(labels_np, preds)
    return {
        "loss": total_loss / max(n, 1),
        "auc": float(auc),
        "f1": float(f1),
        "accuracy": float(acc),
        "n": int(n),
    }


def train_one_epoch(
    model, loader, optimizer, criterion, device, log_every: int = 50
) -> float:
    model.train()
    running = 0.0
    n = 0
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn)
        loss = criterion(out.logits.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += float(loss.item()) * y.size(0)
        n += y.size(0)
        if step % log_every == 0:
            LOG.info(
                "step=%d loss=%.4f mem=%.2fGB",
                step,
                loss.item(),
                allocated_memory_gb(device),
            )
    return running / max(n, 1)


def save_checkpoint(
    path: Path, model, tokenizer, optimizer, epoch: int, best_metric: float
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        },
        path / "training_state.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["baseline", "retrain"], default="baseline")
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint dir when mode=retrain"
    )
    parser.add_argument(
        "--label", default=None, help="Suffix for metrics file (e.g. condition_A)"
    )
    parser.add_argument(
        "--checkpoint_selector",
        choices=["val_auc", "adversarial"],
        default="val_auc",
        help=(
            "val_auc (default): keep epoch with highest validation AUC. "
            "adversarial: keep epoch with lowest evasion_rate on --fake_pool_csv. "
            "Fixes the C=D pathology when the val-AUC ceiling is already hit."
        ),
    )
    parser.add_argument(
        "--fake_pool_csv",
        default=None,
        help="CSV of fake samples used for adversarial checkpoint selection (required when "
        "--checkpoint_selector adversarial).",
    )
    args = parser.parse_args()

    if args.checkpoint_selector == "adversarial" and not args.fake_pool_csv:
        parser.error(
            "--fake_pool_csv is required when --checkpoint_selector adversarial"
        )

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    device = get_device(cfg["runtime"].get("device"))
    LOG.info("Detector mode=%s on %s (%s)", args.mode, device, describe(device))

    model, tokenizer = load_model(cfg, device)
    train_loader, val_loader, test_loader = build_loaders(cfg, tokenizer)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["detector"]["lr"]),
        weight_decay=float(cfg["detector"].get("weight_decay", 0.01)),
        fused=False,  # fused AdamW is CUDA-only
    )
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        LOG.info("Resuming from %s", args.resume)
        model = model.from_pretrained(
            args.resume, num_labels=2, torch_dtype=next(iter(model.parameters())).dtype
        )
        model.to(device)
        state = torch.load(Path(args.resume) / "training_state.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])

    epochs = int(cfg["detector"]["epochs"])
    best: dict[str, float | int] = {"auc": -1.0, "epoch": -1}
    ckpt_dir = Path(cfg["detector"]["output_dir"]) / (args.label or args.mode)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # adversarial selector: save every epoch checkpoint to epoch_N subdirs.
    save_all = args.checkpoint_selector == "adversarial"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        sync(device)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_split(model, val_loader, device, criterion)
        LOG.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_auc=%.4f val_f1=%.4f selector=%s",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["f1"],
            args.checkpoint_selector,
        )
        if save_all:
            epoch_dir = ckpt_dir / f"epoch_{epoch}"
            save_checkpoint(
                epoch_dir, model, tokenizer, optimizer, epoch, val_metrics["auc"]
            )
            LOG.info("Saved epoch checkpoint %s (adversarial selector)", epoch_dir)
        elif val_metrics["auc"] > best["auc"]:
            best = {"auc": val_metrics["auc"], "epoch": epoch}
            save_checkpoint(
                ckpt_dir, model, tokenizer, optimizer, epoch, val_metrics["auc"]
            )
            LOG.info("Saved new best checkpoint (val_auc=%.4f)", val_metrics["auc"])

    # Adversarial checkpoint selection: pick the epoch with lowest evasion on the fake pool.
    from transformers import AutoModelForSequenceClassification

    if save_all and args.fake_pool_csv:
        LOG.info(
            "Adversarial selector: scoring %d epoch checkpoints on %s",
            epochs,
            args.fake_pool_csv,
        )
        best_evasion, best_epoch_dir = 1.1, None
        for epoch in range(1, epochs + 1):
            epoch_dir = ckpt_dir / f"epoch_{epoch}"
            ep_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    epoch_dir,
                    num_labels=2,
                    torch_dtype=resolve_dtype(cfg["detector"].get("dtype", "bfloat16")),
                    ignore_mismatched_sizes=True,
                )
                .to(device)
                .eval()
            )
            pool_ds = TextClassificationDataset(
                args.fake_pool_csv, tokenizer, int(cfg["detector"]["max_length"])
            )
            pool_loader = DataLoader(
                pool_ds,
                batch_size=int(cfg["detector"]["batch_size"]),
                shuffle=False,
                pin_memory=False,
            )
            pool_metrics = eval_split(ep_model, pool_loader, device, criterion)
            fake_probs: list[float] = []
            with torch.inference_mode():
                for b in DataLoader(
                    pool_ds,
                    batch_size=int(cfg["detector"]["batch_size"]),
                    shuffle=False,
                    pin_memory=False,
                ):
                    logits = (
                        ep_model(
                            input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                        )
                        .logits.float()
                        .cpu()
                    )
                    fake_probs.extend(
                        torch.softmax(logits, dim=-1)[:, 1].numpy().tolist()
                    )
            pool_labels = [pool_ds[i]["labels"].item() for i in range(len(pool_ds))]
            import numpy as _np

            fake_mask = _np.asarray(pool_labels) == 0
            evasion = (
                float((_np.asarray(fake_probs)[fake_mask] > 0.5).mean())
                if fake_mask.any()
                else 1.0
            )
            LOG.info(
                "  epoch=%d pool_auc=%.4f evasion=%.4f",
                epoch,
                pool_metrics["auc"],
                evasion,
            )
            del ep_model
            empty_cache(device)
            if evasion < best_evasion:
                best_evasion = evasion
                best_epoch_dir = epoch_dir
                best["epoch"] = epoch

        # Promote winning epoch checkpoint to the canonical ckpt_dir.
        import shutil

        if best_epoch_dir and best_epoch_dir != ckpt_dir:
            for f in best_epoch_dir.iterdir():
                shutil.copy2(f, ckpt_dir / f.name)
        LOG.info(
            "Adversarial selector chose epoch=%d (evasion=%.4f)",
            best["epoch"],
            best_evasion,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir,
        num_labels=2,
        torch_dtype=resolve_dtype(cfg["detector"].get("dtype", "bfloat16")),
        ignore_mismatched_sizes=True,
    ).to(device)
    test_metrics = eval_split(model, test_loader, device, criterion)
    elapsed = time.time() - t0
    LOG.info(
        "DONE elapsed=%.1fs test_auc=%.4f test_f1=%.4f test_acc=%.4f mem=%s",
        elapsed,
        test_metrics["auc"],
        test_metrics["f1"],
        test_metrics["accuracy"],
        describe(device),
    )

    # Persist metrics JSON — condition_A label by default when mode=baseline.
    label = args.label or ("condition_A" if args.mode == "baseline" else args.mode)
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / f"{label}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "label": label,
        "model_name": cfg["detector"]["model_name"],
        "device": str(device),
        "dtype": cfg["detector"].get("dtype", "bfloat16"),
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_val_auc": best["auc"],
        "test": test_metrics,
        "elapsed_seconds": elapsed,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Wrote metrics to %s", metrics_path)

    empty_cache(device)


if __name__ == "__main__":
    main()
