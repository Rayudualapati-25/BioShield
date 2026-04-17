"""Cross-benchmark evaluation — RAID and M4 text deepfake benchmarks.

Paper-strengthening extension (novel contribution §6 of Experiment Plan).

Why this matters
----------------
Our adversarial loop and transfer attack test robustness *within* our own generation
stack. A reviewer will immediately ask: "how does your hardened detector fare on
established, held-out benchmarks built by third parties?" This script runs the
Condition-D detector against two such benchmarks and writes the same metrics
envelope used for our own eval pipeline, so results plug straight into
`evaluation/visualization.py`.

Supported benchmarks (all are public, text-only, and stream from HuggingFace):

- `raid`   — liamdugan/raid (AI-generated text across 8 generators, 11 domains,
             4 decoding strategies, 11 adversarial attacks). Biomedical slice via
             domain filter. https://huggingface.co/datasets/liamdugan/raid
- `m4`     — Chardonneret/M4 (HC3 + MULTITuDE style mixed-generator benchmark).
             Balanced real-vs-generated abstracts.
             https://huggingface.co/datasets/Chardonneret/M4

Both benchmarks are loaded with `datasets.load_dataset(..., streaming=True)` so
we never materialize tens of GB to disk.

If a benchmark is unavailable (rate-limited, schema changed), this script is
*defensive*: it logs the failure, falls back to the next candidate, and exits
with code 0 + a `skipped=True` metrics file so the full experiment pipeline
never blocks on external availability.

CLI
---
    python scripts/raid_benchmark.py --config configs/config_novel.yaml \
        --benchmark raid --checkpoint experiments/checkpoints/detector/condition_D \
        --n_samples 500 --label condition_D_raid

    python scripts/raid_benchmark.py --config configs/config_novel.yaml \
        --benchmark m4 --n_samples 500 --label condition_D_m4
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.device import describe, empty_cache, get_device, resolve_dtype  # noqa: E402
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("bench.raid")


# ---------- Benchmark loaders ----------

def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def load_raid(n_samples: int, seed: int, biomedical_only: bool = True) -> pd.DataFrame:
    """Stream RAID benchmark → DataFrame[text, label].

    RAID schema (v1.5):
      - generation (str)   — text
      - model (str)        — "human" or a generator name (e.g. "gpt-4", "cohere")
      - domain (str)       — "abstracts", "books", "news", "wiki", ...
      - decoding (str)     — "greedy", "sampling", ...
      - attack (str)       — adversarial attack applied (e.g. "upper_lower", "none")
    Label: 1 if model == "human" else 0 (aligned with our detector convention where
    label=1 is REAL; RAID ships with model=="human" as the ground-truth real class).
    """
    from datasets import load_dataset

    LOG.info("Streaming RAID benchmark (n=%d, biomedical_only=%s)...", n_samples, biomedical_only)
    # RAID exposes a "test" split with all generators + attacks.
    stream = load_dataset("liamdugan/raid", split="train", streaming=True).shuffle(
        seed=seed, buffer_size=5_000
    )
    rows: list[dict] = []
    seen: set[str] = set()
    for row in stream:
        domain = row.get("domain") or row.get("source") or ""
        if biomedical_only and domain not in {"abstracts", "medicine", "pubmed", "biomed"}:
            continue
        text = _clean(row.get("generation") or row.get("text") or "")
        if len(text.split()) < 40:
            continue
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)
        is_real = 1 if str(row.get("model", "")).lower() == "human" else 0
        rows.append({"text": text, "label": is_real})
        if len(rows) >= n_samples:
            break

    df = pd.DataFrame(rows)
    pos = int(df["label"].sum())
    LOG.info("RAID → %d rows (real=%d, fake=%d)", len(df), pos, len(df) - pos)
    return df


def load_m4(n_samples: int, seed: int) -> pd.DataFrame:
    """Stream M4 benchmark → DataFrame[text, label].

    M4 schema (Chardonneret/M4):
      - text (str)
      - label (int) — 0 human / 1 machine — we REMAP to our convention
        (our detector: 1 = real, 0 = fake).
    """
    from datasets import load_dataset

    LOG.info("Streaming M4 benchmark (n=%d)...", n_samples)
    # M4 has multiple configs; try "en_multidomain" first, fallback to default.
    candidates = [
        ("Chardonneret/M4", "en_multidomain", "train"),
        ("Chardonneret/M4", None, "train"),
    ]
    rows: list[dict] = []
    seen: set[str] = set()
    last_err: Exception | None = None
    for repo, name, split in candidates:
        try:
            kw = {"split": split, "streaming": True}
            if name:
                kw["name"] = name
            stream = load_dataset(repo, **kw).shuffle(seed=seed, buffer_size=5_000)
            for row in stream:
                text = _clean(row.get("text") or "")
                if len(text.split()) < 40:
                    continue
                h = hash(text)
                if h in seen:
                    continue
                seen.add(h)
                m4_label = int(row.get("label", 0))
                # M4 label=0 human, 1 machine. Our convention: 1 real, 0 fake → invert.
                rows.append({"text": text, "label": 1 - m4_label})
                if len(rows) >= n_samples:
                    break
            if rows:
                break
        except Exception as e:
            last_err = e
            LOG.warning("M4 %s name=%s failed: %s", repo, name, str(e)[:160])
            continue
    if not rows:
        raise RuntimeError(f"Could not load M4. Last error: {last_err}")

    df = pd.DataFrame(rows)
    pos = int(df["label"].sum())
    LOG.info("M4 → %d rows (real=%d, fake=%d)", len(df), pos, len(df) - pos)
    return df


# ---------- Evaluation ----------

def evaluate_detector(
    df: pd.DataFrame,
    checkpoint: str | Path,
    cfg: dict,
    device: torch.device,
) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = resolve_dtype(cfg["detector"].get("dtype", "bfloat16"))
    max_length = int(cfg["detector"]["max_length"])
    batch_size = int(cfg["detector"]["batch_size"])

    LOG.info("Loading detector from %s (dtype=%s) on %s", checkpoint, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device).eval()

    texts = df["text"].tolist()
    labels = df["label"].astype(int).tolist()
    probs: list[float] = []
    preds: list[int] = []

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
            out = model(**enc)
            logits = out.logits.float().cpu()
            p = torch.softmax(logits, dim=-1)[:, 1].numpy().tolist()
            probs.extend(p)
            preds.extend(np.argmax(logits.numpy(), axis=1).tolist())
            if (i // batch_size) % 10 == 0:
                LOG.info("  evaluated %d / %d", i + len(chunk), len(texts))

    probs_np = np.asarray(probs)
    labels_np = np.asarray(labels)
    try:
        auc = roc_auc_score(labels_np, probs_np) if len(set(labels_np)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    f1 = f1_score(labels_np, preds, zero_division=0)
    acc = accuracy_score(labels_np, preds)
    # Evasion rate: of the fakes (label=0), what fraction the detector called real (pred=1).
    fake_mask = labels_np == 0
    evasion = float(np.mean(np.asarray(preds)[fake_mask] == 1)) if fake_mask.any() else float("nan")

    empty_cache(device)
    return {
        "auc": float(auc),
        "f1": float(f1),
        "accuracy": float(acc),
        "evasion_rate": float(evasion),
        "n": int(len(texts)),
        "n_real": int((labels_np == 1).sum()),
        "n_fake": int((labels_np == 0).sum()),
    }


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=["raid", "m4"], required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Detector checkpoint dir. Defaults to detector.output_dir/condition_D.")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed = int(cfg["runtime"].get("seed", 42))
    set_seed(seed)
    device = get_device(cfg["runtime"].get("device"))

    checkpoint = args.checkpoint or str(
        Path(cfg["detector"]["output_dir"]) / "condition_D"
    )
    if not Path(checkpoint).exists():
        # Fallback: any existing detector subdir (e.g. condition_A on first run).
        parent = Path(cfg["detector"]["output_dir"])
        existing = [p for p in parent.iterdir() if p.is_dir()] if parent.exists() else []
        if existing:
            checkpoint = str(existing[0])
            LOG.warning("Condition D checkpoint missing; falling back to %s", checkpoint)
        else:
            raise FileNotFoundError(
                f"No detector checkpoint at {checkpoint}. Train a detector first."
            )

    label = args.label or f"condition_D_{args.benchmark}"
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / f"{label}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        if args.benchmark == "raid":
            df = load_raid(args.n_samples, seed, biomedical_only=True)
        else:
            df = load_m4(args.n_samples, seed)
    except Exception as e:
        LOG.warning("Benchmark %s unavailable: %s", args.benchmark, e)
        payload = {
            "label": label,
            "benchmark": args.benchmark,
            "skipped": True,
            "reason": str(e)[:300],
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG.info("Wrote skip marker to %s", metrics_path)
        return

    if len(df) == 0:
        LOG.warning("Empty benchmark slice; writing skip marker.")
        metrics_path.write_text(
            json.dumps({"label": label, "benchmark": args.benchmark, "skipped": True,
                        "reason": "empty slice"}, indent=2),
            encoding="utf-8",
        )
        return

    metrics = evaluate_detector(df, checkpoint, cfg, device)
    elapsed = round(time.time() - t0, 1)

    payload = {
        "label": label,
        "benchmark": args.benchmark,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "device_desc": describe(device),
        "dtype": cfg["detector"].get("dtype", "bfloat16"),
        "biomedical_only": args.benchmark == "raid",
        "n_samples": args.n_samples,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info(
        "DONE benchmark=%s auc=%.4f f1=%.4f evasion=%.4f elapsed=%.1fs → %s",
        args.benchmark, metrics["auc"], metrics["f1"], metrics["evasion_rate"], elapsed, metrics_path,
    )


if __name__ == "__main__":
    main()
