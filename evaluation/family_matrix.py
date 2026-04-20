"""Generator-family evasion matrix.

Computes the N×M matrix where entry (i, j) is the evasion rate of
detector_i against fakes produced by generator_j. This makes the
generator-family effect described in BioShield §V.C explicit and
reproducible as a single command.

Usage:
    python evaluation/family_matrix.py \
        --config configs/config_novel.yaml \
        --matrix_cfg experiments/family_matrix_cfg.json \
        --output_dir experiments/family_matrix

family_matrix_cfg.json format:
{
  "detectors": [
    {"name": "condition_A",  "ckpt": "experiments/checkpoints/detector/condition_A"},
    {"name": "condition_E_multi", "ckpt": "experiments/checkpoints/detector/condition_E"}
  ],
  "generators": [
    {"name": "BioMistral-7B",  "csv": "data/processed/test.csv"},
    {"name": "SeqGAN",          "csv": "experiments/round_data/condition_B/round_3/test.csv"},
    {"name": "Llama-3.2-3B",   "csv": "data/processed/transfer_test.csv"},
    {"name": "RAID-mixed",     "csv": "data/processed/raid_test.csv"}
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.detector.dataset import TextClassificationDataset  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.device import empty_cache, get_device, resolve_dtype  # noqa: E402
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("evaluation.family_matrix")


def _score_csv(
    detector_ckpt: str,
    csv_path: str,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    """Score a CSV with a detector checkpoint. Returns evasion, auc, f1, accuracy."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(detector_ckpt)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            detector_ckpt, num_labels=2, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )

    ds = TextClassificationDataset(
        csv_path, tokenizer, int(cfg["detector"]["max_length"])
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["detector"]["batch_size"]),
        shuffle=False,
        pin_memory=False,
    )

    probs: list[float] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["input_ids"].to(device)
            a = batch["attention_mask"].to(device)
            y = batch["labels"]
            logits = model(input_ids=x, attention_mask=a).logits.float().cpu()
            p = torch.softmax(logits, dim=-1)[:, 1].numpy()
            probs.extend(p.tolist())
            labels.extend(y.numpy().tolist())

    del model
    empty_cache(device)

    y_np = np.asarray(labels)
    p_np = np.asarray(probs)
    preds = (p_np >= 0.5).astype(int)
    fake_mask = y_np == 0
    evasion = float((p_np[fake_mask] > 0.5).mean()) if fake_mask.any() else float("nan")
    try:
        auc = roc_auc_score(y_np, p_np) if len(set(y_np.tolist())) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "evasion_rate": evasion,
        "auc": float(auc),
        "f1": float(f1_score(y_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_np, preds)),
        "n": len(labels),
    }


def build_matrix(
    cfg: dict,
    matrix_cfg: dict,
    output_dir: Path,
) -> dict:
    """Build and persist the full detector × generator evasion matrix."""
    device = get_device(cfg["runtime"].get("device"))
    detectors: list[dict] = matrix_cfg["detectors"]
    generators: list[dict] = matrix_cfg["generators"]

    results: dict[str, dict[str, dict]] = {}
    for det in detectors:
        det_name = det["name"]
        det_ckpt = det["ckpt"]
        results[det_name] = {}
        for gen in generators:
            gen_name = gen["name"]
            gen_csv = gen["csv"]
            if not Path(gen_csv).exists():
                LOG.warning("Generator CSV not found — skipping: %s", gen_csv)
                results[det_name][gen_name] = {
                    "evasion_rate": float("nan"),
                    "note": "csv_missing",
                }
                continue
            LOG.info("Scoring detector=%s vs generator=%s", det_name, gen_name)
            metrics = _score_csv(det_ckpt, gen_csv, cfg, device)
            results[det_name][gen_name] = metrics
            LOG.info(
                "  evasion=%.3f auc=%.4f acc=%.3f (n=%d)",
                metrics["evasion_rate"],
                metrics["auc"],
                metrics["accuracy"],
                metrics["n"],
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist full JSON
    json_path = output_dir / "family_matrix.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Persist evasion-only CSV (easy to import into LaTeX via \input)
    csv_path = output_dir / "evasion_matrix.csv"
    gen_names = [g["name"] for g in generators]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["detector"] + gen_names)
        for det_name, gen_results in results.items():
            row = [det_name] + [
                f"{gen_results.get(g, {}).get('evasion_rate', float('nan')):.3f}"
                for g in gen_names
            ]
            writer.writerow(row)

    LOG.info("Evasion matrix written to %s", csv_path)

    # Pretty-print to console
    header = f"{'Detector':<25}" + "".join(f"{g['name']:>18}" for g in generators)
    LOG.info("\n%s", header)
    for det_name, gen_results in results.items():
        row_str = f"{det_name:<25}" + "".join(
            f"{gen_results.get(g['name'], {}).get('evasion_rate', float('nan')):18.3f}"
            for g in generators
        )
        LOG.info(row_str)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--matrix_cfg",
        required=True,
        help="JSON file specifying detectors and generators",
    )
    parser.add_argument("--output_dir", default="experiments/family_matrix")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    matrix_cfg = json.loads(Path(args.matrix_cfg).read_text(encoding="utf-8"))
    build_matrix(cfg, matrix_cfg, Path(args.output_dir))


if __name__ == "__main__":
    main()
