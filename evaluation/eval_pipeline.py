"""Evaluation CLI — test-split metrics and the transfer attack (PRD §3.5).

Modes:
  --mode test      : reload a detector checkpoint + score the configured test split
  --mode transfer  : generate `n_fakes` with the transfer attacker, score the detector
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import classification_metrics, evasion_rate  # noqa: E402
from models.detector.dataset import TextClassificationDataset  # noqa: E402
from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.device import empty_cache, get_device, resolve_dtype  # noqa: E402
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("evaluation.pipeline")


def _load_detector(cfg: dict, checkpoint: Path, device: torch.device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, torch_dtype=dtype
    ).to(device)
    model.eval()
    return model, tokenizer


def _score_csv(cfg: dict, detector_ckpt: Path, csv_path: Path, device: torch.device) -> dict:
    model, tokenizer = _load_detector(cfg, detector_ckpt, device)
    ds = TextClassificationDataset(str(csv_path), tokenizer, int(cfg["detector"]["max_length"]))
    loader = DataLoader(ds, batch_size=int(cfg["detector"]["batch_size"]), shuffle=False, pin_memory=False)

    import numpy as np
    probs: list[float] = []
    labels: list[int] = []
    preds: list[int] = []
    with torch.inference_mode():
        for b in loader:
            x = b["input_ids"].to(device)
            a = b["attention_mask"].to(device)
            y = b["labels"].to(device)
            logits = model(input_ids=x, attention_mask=a).logits.float()
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            preds.extend(np.argmax(logits.cpu().numpy(), axis=1).tolist())
            labels.extend(y.cpu().numpy().tolist())

    metrics = classification_metrics(labels, probs, preds)
    metrics["evasion_rate"] = evasion_rate(labels, probs)
    del model
    empty_cache(device)
    return metrics


def _transfer_generate(cfg: dict, n: int, device: torch.device) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tcfg = cfg["transfer_attacker"]
    name = tcfg["model_name"]
    dtype = resolve_dtype(tcfg.get("dtype", "bfloat16"))
    LOG.info("Loading transfer attacker %s (dtype=%s)", name, dtype)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, attn_implementation=tcfg.get("attn_implementation", "eager"),
    ).to(device)
    model.eval()

    topics = [
        "type 2 diabetes", "breast cancer screening", "COPD exacerbation",
        "antibiotic stewardship", "ischemic stroke", "childhood asthma",
        "Parkinson progression", "rheumatoid arthritis biologics",
    ]
    rng = random.Random(int(cfg["runtime"].get("seed", 42)))
    outputs: list[str] = []
    with torch.inference_mode():
        for i in range(n):
            topic = rng.choice(topics)
            prompt = (
                f"<|user|>\nWrite a realistic biomedical research abstract about {topic}. "
                f"Include methods, results, and a conclusion.\n<|end|>\n<|assistant|>\n"
            )
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_ids = model.generate(
                **enc,
                max_new_tokens=int(tcfg.get("max_new_tokens", 384)),
                do_sample=True,
                temperature=float(tcfg.get("temperature", 0.8)),
                top_p=float(tcfg.get("top_p", 0.95)),
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs.append(tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            if i % 10 == 0:
                LOG.info("transfer gen %d/%d", i, n)

    del model
    empty_cache(device)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["test", "transfer"], default="test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--label", default=None)
    parser.add_argument("--detector_ckpt", default=None,
                        help="Directory with a saved HF classifier (defaults to condition_A output)")
    parser.add_argument("--transfer_generator", default=None)
    parser.add_argument("--n_fakes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    device = get_device(cfg["runtime"].get("device"))

    default_label = args.label or ("condition_A" if args.mode == "test" else "condition_D")
    checkpoint = Path(
        args.detector_ckpt or Path(cfg["detector"]["output_dir"]) / default_label
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint missing: {checkpoint}")

    if args.mode == "test":
        split_csv = Path(cfg["paths"][f"{args.split}_csv"])
        metrics = _score_csv(cfg, checkpoint, split_csv, device)
        out_path = Path(cfg["paths"]["metrics_dir"]) / f"{default_label}_{args.split}_metrics.json"
        out_path.write_text(json.dumps({"label": default_label, "split": args.split, **metrics}, indent=2))
        LOG.info("Wrote %s: %s", out_path, metrics)
        return

    # Transfer mode
    if args.transfer_generator:
        cfg["transfer_attacker"]["model_name"] = args.transfer_generator
    n = int(args.n_fakes or cfg["transfer_attacker"]["n_fakes"])
    fakes = _transfer_generate(cfg, n, device)

    # Combine with equal-size real sample from the test split.
    real = pd.read_csv(cfg["paths"]["test_csv"])
    real = real[real["label"] == 1] if "label" in real.columns else real
    real = real.sample(n=min(len(real), len(fakes)), random_state=int(cfg["runtime"].get("seed", 42)))
    combined = pd.concat(
        [pd.DataFrame({"text": fakes, "label": 0}), real[["text", "label"]]],
        ignore_index=True,
    )
    transfer_csv = Path(cfg["paths"]["transfer_test_csv"])
    transfer_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(transfer_csv, index=False)
    LOG.info("Built transfer test set at %s (n=%d)", transfer_csv, len(combined))

    transfer_metrics = _score_csv(cfg, checkpoint, transfer_csv, device)

    # Pair against the in-distribution test AUC using the same detector.
    in_dist_metrics = _score_csv(cfg, checkpoint, Path(cfg["paths"]["test_csv"]), device)
    payload = {
        "detector_ckpt": str(checkpoint),
        "transfer_generator": cfg["transfer_attacker"]["model_name"],
        "n_fakes": n,
        "in_distribution": in_dist_metrics,
        "transfer": transfer_metrics,
        "gap_auc": (in_dist_metrics["auc"] - transfer_metrics["auc"])
        if (in_dist_metrics["auc"] == in_dist_metrics["auc"])  # NaN check
        else float("nan"),
    }
    out_path = Path(cfg["paths"]["metrics_dir"]) / "transfer_attack_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Wrote %s", out_path)
    LOG.info(
        "In-distribution AUC=%.4f | Transfer AUC=%.4f | Gap=%.4f",
        in_dist_metrics["auc"], transfer_metrics["auc"], payload["gap_auc"],
    )


if __name__ == "__main__":
    main()
