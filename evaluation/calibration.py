"""Post-hoc temperature scaling for AI-text detectors.

Addresses the calibration failure documented in BioShield §VI.C:
when a detector trained on one generator family (BioMistral) is applied
to a cross-domain test set (RAID), its decision threshold (0.5) is
meaningless even when AUC is preserved at ~0.82.

Temperature scaling (Guo et al., 2017, ICML) learns a single scalar T
on a held-out calibration set so that p_calibrated = softmax(logits / T).
When T > 1 the model is over-confident; scaling pushes probabilities toward 0.5.

CLI:
    python evaluation/calibration.py \
        --config configs/config_novel.yaml \
        --detector_ckpt experiments/checkpoints/detector/condition_A \
        --calib_csv data/processed/val.csv \
        --eval_csv data/processed/test.csv \
        --output_dir experiments/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.detector.dataset import TextClassificationDataset  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.device import empty_cache, get_device, resolve_dtype  # noqa: E402
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("evaluation.calibration")


# ---------------------------------------------------------------------------
# Temperature scaling layer
# ---------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Wraps a trained classifier and applies a single scalar temperature.

    References:
        Guo et al., "On Calibration of Modern Neural Networks," ICML 2017.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(logits), dim=-1)


# ---------------------------------------------------------------------------
# Calibration fitting
# ---------------------------------------------------------------------------


def _collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen detector and collect raw logits + ground-truth labels."""
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["input_ids"].to(device)
            a = batch["attention_mask"].to(device)
            y = batch["labels"]
            logits = model(input_ids=x, attention_mask=a).logits.float().cpu()
            all_logits.append(logits)
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Fit temperature T on a calibration set by minimising NLL loss.

    Returns the optimal temperature as a float.
    """
    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    logits = logits.float()
    labels = labels.long()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    T = float(scaler.temperature.item())
    LOG.info("Fitted temperature T=%.4f", T)
    return T


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _metrics_at_threshold(
    probs_real: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    preds = (probs_real >= threshold).astype(int)
    fake_mask = labels == 0
    evasion = (
        float((probs_real[fake_mask] > threshold).mean()) if fake_mask.any() else 0.0
    )
    try:
        auc = (
            roc_auc_score(labels, probs_real)
            if len(set(labels.tolist())) > 1
            else float("nan")
        )
    except ValueError:
        auc = float("nan")
    return {
        "auc": float(auc),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "evasion_rate": evasion,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Main calibration pipeline
# ---------------------------------------------------------------------------


def calibrate(
    cfg: dict,
    detector_ckpt: str,
    calib_csv: str,
    eval_csv: str,
    output_dir: Path,
) -> dict:
    """Full calibration pipeline: fit T, evaluate before/after, save results."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = get_device(cfg["runtime"].get("device"))
    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    max_length = int(cfg["detector"]["max_length"])
    batch_size = int(cfg["detector"]["batch_size"])

    tokenizer = AutoTokenizer.from_pretrained(detector_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        detector_ckpt, num_labels=2, torch_dtype=dtype
    ).to(device)

    def _make_loader(csv_path: str) -> DataLoader:
        ds = TextClassificationDataset(csv_path, tokenizer, max_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False)

    calib_loader = _make_loader(calib_csv)
    eval_loader = _make_loader(eval_csv)

    LOG.info("Collecting calibration logits from %s", calib_csv)
    calib_logits, calib_labels = _collect_logits_and_labels(model, calib_loader, device)

    LOG.info("Collecting eval logits from %s", eval_csv)
    eval_logits, eval_labels = _collect_logits_and_labels(model, eval_loader, device)

    empty_cache(device)

    # Pre-calibration metrics
    raw_probs = torch.softmax(eval_logits, dim=-1)[:, 1].numpy()
    pre_metrics = _metrics_at_threshold(raw_probs, eval_labels.numpy())
    LOG.info(
        "Pre-calibration: acc=%.3f evasion=%.3f auc=%.4f",
        pre_metrics["accuracy"],
        pre_metrics["evasion_rate"],
        pre_metrics["auc"],
    )

    # Fit temperature on calibration set
    T = fit_temperature(calib_logits, calib_labels)

    # Post-calibration metrics
    cal_probs = torch.softmax(eval_logits / T, dim=-1)[:, 1].numpy()
    post_metrics = _metrics_at_threshold(cal_probs, eval_labels.numpy())
    LOG.info(
        "Post-calibration (T=%.4f): acc=%.3f evasion=%.3f auc=%.4f",
        T,
        post_metrics["accuracy"],
        post_metrics["evasion_rate"],
        post_metrics["auc"],
    )

    # Optimal threshold search (maximise F1 on calibration set)
    cal_probs_calib = torch.softmax(calib_logits / T, dim=-1)[:, 1].numpy()
    best_thresh, best_f1 = 0.5, -1.0
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (cal_probs_calib >= thresh).astype(int)
        from sklearn.metrics import f1_score

        f1 = f1_score(calib_labels.numpy(), preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = float(f1), float(thresh)
    opt_metrics = _metrics_at_threshold(
        cal_probs, eval_labels.numpy(), threshold=best_thresh
    )
    LOG.info(
        "Optimal threshold=%.2f: acc=%.3f evasion=%.3f auc=%.4f",
        best_thresh,
        opt_metrics["accuracy"],
        opt_metrics["evasion_rate"],
        opt_metrics["auc"],
    )

    result = {
        "detector_ckpt": str(detector_ckpt),
        "calib_csv": str(calib_csv),
        "eval_csv": str(eval_csv),
        "fitted_temperature": T,
        "optimal_threshold": best_thresh,
        "pre_calibration": pre_metrics,
        "post_calibration_fixed_threshold": post_metrics,
        "post_calibration_optimal_threshold": opt_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "calibration_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOG.info("Calibration results written to %s", out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--detector_ckpt", required=True)
    parser.add_argument(
        "--calib_csv", required=True, help="Calibration set (e.g. val.csv)"
    )
    parser.add_argument(
        "--eval_csv", required=True, help="Evaluation set (e.g. RAID test CSV)"
    )
    parser.add_argument("--output_dir", default="experiments/calibration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    calibrate(
        cfg, args.detector_ckpt, args.calib_csv, args.eval_csv, Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
