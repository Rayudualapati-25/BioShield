"""Adversarial training loop with the 4 ablation conditions from PRD §3.4.

Conditions:
  - static_baseline (A): no loop. Just train detector once. Metrics → condition_A_metrics.json
  - seqgan_only    (B): SeqGAN as generator, no agent rewrites, detector retrained per round.
  - agent_only     (C): BioMedLM + Phi-3.5 rewrites; detector NOT retrained.
  - full_pipeline  (D): BioMedLM + Phi-3.5 rewrites + detector retrained per round (MAIN).

Metrics per round are logged to `experiments/metrics/metrics_log.csv` with the
columns required by PRD §3.6: round, condition, auc, f1, evasion_rate,
bertscore_f1, mean_perplexity.

Each round is a single "stage" in the loop; between stages the MPS allocator
is flushed via utils.device.empty_cache. Checkpoints are written to
experiments/checkpoints/detector/<condition>/round_<k>.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.device import describe, empty_cache, get_device, sync  # noqa: E402
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("training.loop")

CONDITIONS = ("full_pipeline", "seqgan_only", "agent_only", "static_baseline")
CONDITION_LABEL = {
    "static_baseline": "condition_A",
    "seqgan_only": "condition_B",
    "agent_only": "condition_C",
    "full_pipeline": "condition_D",
}


def _run(cmd: list[str]) -> None:
    LOG.info("$ %s", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {res.returncode}: {' '.join(cmd)}"
        )


def _append_metrics_row(cfg: dict, row: dict) -> None:
    path = Path(cfg["paths"]["metrics_dir"]) / "metrics_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "round",
        "condition",
        "auc",
        "f1",
        "evasion_rate",
        "bertscore_f1",
        "mean_perplexity",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})


def run_static_baseline(cfg: dict, config_path: str) -> dict:
    """Condition A — train the detector once, no loop."""
    label = CONDITION_LABEL["static_baseline"]
    _run(
        [
            sys.executable,
            "models/detector/train_detector.py",
            "--config",
            config_path,
            "--mode",
            "baseline",
            "--label",
            label,
        ]
    )
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / f"{label}_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    _append_metrics_row(
        cfg,
        {
            "round": 0,
            "condition": label,
            "auc": payload["test"]["auc"],
            "f1": payload["test"]["f1"],
            "evasion_rate": "",
            "bertscore_f1": "",
            "mean_perplexity": "",
        },
    )
    return payload


def _score_detector_on(cfg: dict, detector_ckpt: Path, test_csv: Path, device) -> dict:
    """Reload a checkpoint and evaluate on `test_csv`. Returns auc/f1/acc/evasion_rate."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader

    from models.detector.dataset import TextClassificationDataset
    from utils.device import resolve_dtype

    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(detector_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        detector_ckpt, num_labels=2, torch_dtype=dtype
    ).to(device)
    model.eval()

    ds = TextClassificationDataset(
        str(test_csv), tokenizer, int(cfg["detector"]["max_length"])
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["detector"]["batch_size"]),
        shuffle=False,
        pin_memory=False,
    )

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    probs: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
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

    # Evasion rate: fraction of fake samples (label==0) the detector misclassified as real (prob>0.5).
    fake_mask = labels_np == 0
    evasion_rate = float((probs_np[fake_mask] > 0.5).mean()) if fake_mask.any() else 0.0
    del model
    empty_cache(device)
    return {
        "auc": float(auc),
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
        "evasion_rate": evasion_rate,
    }


def _score_texts(
    cfg: dict, detector_ckpt: Path, texts: list[str], device
) -> list[float]:
    """Score a batch of texts with the detector at detector_ckpt. Returns list of p(real)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from utils.device import resolve_dtype

    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    max_length = int(cfg["detector"]["max_length"])
    batch_size = int(cfg["detector"]["batch_size"])

    tokenizer = AutoTokenizer.from_pretrained(detector_ckpt)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            detector_ckpt, num_labels=2, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )

    probs: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)
            logits = model(**enc).logits.float().cpu()
            p = torch.softmax(logits, dim=-1)[:, 1].numpy().tolist()
            probs.extend(p)
    del model
    empty_cache(device)
    return probs


def _build_round_test_set(
    cfg: dict, round_idx: int, fakes_csv: Path, test_csv_out: Path
) -> None:
    """Build a balanced round-specific test set: all new fakes + equal real samples from cfg.test_csv."""
    real = pd.read_csv(cfg["paths"]["test_csv"])
    real = real[real["label"] == 1] if "label" in real.columns else real
    fakes = pd.read_csv(fakes_csv)
    if "label" not in fakes.columns:
        fakes["label"] = 0
    n = min(len(real), len(fakes))
    merged = pd.concat(
        [
            real.sample(n=n, random_state=round_idx),
            fakes.sample(n=n, random_state=round_idx),
        ],
        ignore_index=True,
    )
    merged = merged[["text", "label"]]
    test_csv_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(test_csv_out, index=False)


def run_loop(
    condition: str,
    cfg: dict,
    config_path: str,
    checkpoint_selector: str = "val_auc",
) -> dict:
    """Drives one of B / C / D. Returns the final metrics dict.

    checkpoint_selector: 'val_auc' (default) or 'adversarial'.
    When 'adversarial', each retrain call saves all epoch checkpoints and
    selects the one with the lowest evasion on the round's fake pool,
    dissolving the C=D pathology caused by val-AUC-ceiling saturation.
    """
    label = CONDITION_LABEL[condition]
    device = get_device(cfg["runtime"].get("device"))
    LOG.info("Loop condition=%s label=%s (%s)", condition, label, describe(device))

    num_rounds = int(cfg["loop"]["num_rounds"])
    pool_size = int(cfg["loop"]["fake_pool_size"])
    base_seed = int(cfg["runtime"].get("seed", 42))
    round_root = Path(cfg["paths"]["round_data_dir"]) / label
    round_root.mkdir(parents=True, exist_ok=True)

    # Round 0: train the baseline detector (always) so rounds 1..N have something to score against.
    base_label = f"{label}_round0"
    _run(
        [
            sys.executable,
            "models/detector/train_detector.py",
            "--config",
            config_path,
            "--mode",
            "baseline",
            "--label",
            base_label,
        ]
    )
    current_ckpt = Path(cfg["detector"]["output_dir"]) / base_label

    per_round_metrics: list[dict] = []
    for r in range(1, num_rounds + 1):
        # Re-seed per round so generator/agent sampling differs across rounds.
        # Without this, all rounds produce byte-identical fakes and per-round
        # metrics become degenerate. See OVERNIGHT_SUMMARY.md "Known issues".
        set_seed(base_seed + r)
        LOG.info(
            "--- ROUND %d / %d (%s) [seed=%d] ---", r, num_rounds, label, base_seed + r
        )
        round_dir = round_root / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        fakes_csv = round_dir / "fakes.csv"

        # 1. Produce a fake pool.
        if condition == "seqgan_only":
            ckpt = Path(cfg["seqgan"]["output_dir"]) / "seqgan.pt"
            if not ckpt.exists():
                _run(
                    [
                        sys.executable,
                        "models/seqgan/train_seqgan.py",
                        "--config",
                        config_path,
                    ]
                )
            _run(
                [
                    sys.executable,
                    "models/seqgan/train_seqgan.py",
                    "--config",
                    config_path,
                    "--generate_only",
                    "--checkpoint",
                    str(ckpt),
                    "--n",
                    str(pool_size),
                    "--out",
                    str(fakes_csv),
                ]
            )
        else:
            # full_pipeline / agent_only — use the BioMedLM generator (skip fine-tune if already saved)
            gen_ckpt = Path(cfg["generator"]["output_dir"])
            if not (gen_ckpt / "config.json").exists():
                _run(
                    [
                        sys.executable,
                        "models/generator/train_generator.py",
                        "--config",
                        config_path,
                    ]
                )
            _run(
                [
                    sys.executable,
                    "models/generator/train_generator.py",
                    "--config",
                    config_path,
                    "--generate_only",
                    "--checkpoint",
                    str(gen_ckpt),
                    "--n",
                    str(pool_size),
                    "--out",
                    str(fakes_csv),
                ]
            )

            # 2. Optional Qwen rewrites (full_pipeline + agent_only; skipped for seqgan_only)
            # Rewrites focus on "hard" fakes — those the detector caught most confidently
            # (low prob_real). The agent's job is to close that gap. hard_fraction < 1.0
            # means only the hardest N% are rewritten; the rest pass through untouched.
            if condition in ("full_pipeline", "agent_only"):
                hard_fraction = float(cfg["loop"].get("hard_fraction", 1.0))
                if 0.0 < hard_fraction < 1.0:
                    fakes_df = pd.read_csv(fakes_csv)
                    if "label" not in fakes_df.columns:
                        fakes_df["label"] = 0
                    # Score all fakes with the current detector to rank by hardness.
                    probs = _score_texts(
                        cfg, current_ckpt, fakes_df["text"].astype(str).tolist(), device
                    )
                    fakes_df["prob_real"] = probs
                    # "Hard for the generator" = detector confidently called it fake = low prob_real.
                    fakes_df = fakes_df.sort_values(
                        "prob_real", ascending=True
                    ).reset_index(drop=True)
                    n_hard = max(1, int(round(hard_fraction * len(fakes_df))))
                    hard_df = fakes_df.iloc[:n_hard]
                    easy_df = fakes_df.iloc[n_hard:]
                    hard_input = round_dir / "fakes_hard_input.csv"
                    hard_df[["text", "label"]].to_csv(hard_input, index=False)
                    rewritten_hard = round_dir / "fakes_hard_rewritten.csv"
                    _run(
                        [
                            sys.executable,
                            "agents/adversarial_agent.py",
                            "--config",
                            config_path,
                            "--input_csv",
                            str(hard_input),
                            "--output_csv",
                            str(rewritten_hard),
                        ]
                    )
                    # Merge rewritten-hard + untouched-easy into the round's fake pool.
                    rew_df = pd.read_csv(rewritten_hard)[["text"]].assign(label=0)
                    merged = pd.concat(
                        [rew_df, easy_df[["text"]].assign(label=0)], ignore_index=True
                    )
                    merged_csv = round_dir / "fakes_merged.csv"
                    merged.to_csv(merged_csv, index=False)
                    LOG.info(
                        "hard_fraction=%.2f -> rewrote %d/%d fakes (easy %d passed through)",
                        hard_fraction,
                        len(rew_df),
                        len(fakes_df),
                        len(easy_df),
                    )
                    fakes_csv = merged_csv
                else:
                    rewritten = round_dir / "fakes_rewritten.csv"
                    _run(
                        [
                            sys.executable,
                            "agents/adversarial_agent.py",
                            "--config",
                            config_path,
                            "--input_csv",
                            str(fakes_csv),
                            "--output_csv",
                            str(rewritten),
                        ]
                    )
                    fakes_csv = rewritten

        # 3. Build this round's test set and evaluate the current detector on it.
        round_test = round_dir / "test.csv"
        _build_round_test_set(cfg, r, fakes_csv, round_test)
        scored = _score_detector_on(cfg, current_ckpt, round_test, device)
        LOG.info(
            "round=%d condition=%s pre-retrain: auc=%.4f f1=%.4f evasion=%.4f",
            r,
            label,
            scored["auc"],
            scored["f1"],
            scored["evasion_rate"],
        )
        _append_metrics_row(
            cfg,
            {
                "round": r,
                "condition": label,
                "auc": scored["auc"],
                "f1": scored["f1"],
                "evasion_rate": scored["evasion_rate"],
                "bertscore_f1": "",
                "mean_perplexity": "",
            },
        )

        # 4. Retrain the detector for conditions B and D (NOT C).
        if condition in ("full_pipeline", "seqgan_only"):
            # Build an augmented training set: current train + this round's fakes.
            aug_train = round_dir / "train_aug.csv"
            base_train = pd.read_csv(cfg["paths"]["train_csv"])
            more_fakes = pd.read_csv(fakes_csv)[["text"]].assign(label=0)
            pd.concat([base_train, more_fakes], ignore_index=True).to_csv(
                aug_train, index=False
            )

            # Swap train_csv via a per-round yaml sidecar.
            round_cfg = round_dir / "config.yaml"
            round_cfg_data = load_config(config_path)
            round_cfg_data["paths"]["train_csv"] = str(aug_train)
            import yaml

            round_cfg.write_text(yaml.safe_dump(round_cfg_data), encoding="utf-8")

            round_ckpt_label = f"{label}_round{r}"
            retrain_cmd = [
                sys.executable,
                "models/detector/train_detector.py",
                "--config",
                str(round_cfg),
                "--mode",
                "retrain",
                "--resume",
                str(current_ckpt),
                "--label",
                round_ckpt_label,
                "--checkpoint_selector",
                checkpoint_selector,
            ]
            if checkpoint_selector == "adversarial":
                # Pass the current round's fake pool so the selector can score epochs.
                retrain_cmd += ["--fake_pool_csv", str(fakes_csv)]
            _run(retrain_cmd)
            current_ckpt = Path(cfg["detector"]["output_dir"]) / round_ckpt_label

        per_round_metrics.append({"round": r, **scored})
        sync(device)
        empty_cache(device)

    final_scored = per_round_metrics[-1]
    payload = {
        "condition": label,
        "num_rounds": num_rounds,
        "per_round": per_round_metrics,
        "final": final_scored,
    }
    out_path = Path(cfg["paths"]["metrics_dir"]) / f"{label}_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Wrote %s", out_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="full_pipeline")
    parser.add_argument(
        "--checkpoint_selector",
        choices=["val_auc", "adversarial"],
        default="val_auc",
        help=(
            "val_auc (default): pick epoch with highest validation AUC (standard). "
            "adversarial: pick epoch with lowest evasion on the round fake pool — "
            "dissolves the C=D pathology reported in §VI.B."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    t0 = time.time()

    if args.condition == "static_baseline":
        run_static_baseline(cfg, args.config)
    else:
        run_loop(
            args.condition,
            cfg,
            args.config,
            checkpoint_selector=args.checkpoint_selector,
        )

    LOG.info("Condition %s completed in %.1fs", args.condition, time.time() - t0)


if __name__ == "__main__":
    main()
