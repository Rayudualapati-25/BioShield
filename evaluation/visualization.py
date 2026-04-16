"""Plots for the paper (PRD §3.8).

Outputs in `experiments/plots/`:
  - evasion_rate_vs_round.png
  - auc_f1_vs_round.png
  - ablation_comparison.png
  - transfer_attack_results.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.logging import get_logger  # noqa: E402

LOG = get_logger("evaluation.viz")


def plot_evasion_vs_round(metrics_csv: Path, out_path: Path) -> None:
    if not metrics_csv.exists():
        LOG.warning("%s missing — skipping evasion plot", metrics_csv)
        return
    df = pd.read_csv(metrics_csv)
    df = df[df["evasion_rate"].notna() & (df["evasion_rate"] != "")]
    if df.empty:
        LOG.warning("No evasion rows — skipping evasion plot")
        return
    df["evasion_rate"] = df["evasion_rate"].astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, sub in df.groupby("condition"):
        ax.plot(sub["round"], sub["evasion_rate"], marker="o", label=cond)
    ax.set_xlabel("Round")
    ax.set_ylabel("Evasion rate")
    ax.set_title("Evasion rate vs. round")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def plot_auc_f1_vs_round(metrics_csv: Path, out_path: Path) -> None:
    if not metrics_csv.exists():
        return
    df = pd.read_csv(metrics_csv)
    df = df[df["auc"].notna()]
    if df.empty:
        return
    df["auc"] = df["auc"].astype(float)
    df["f1"] = df["f1"].astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for cond, sub in df.groupby("condition"):
        ax1.plot(sub["round"], sub["auc"], marker="o", label=cond)
        ax2.plot(sub["round"], sub["f1"], marker="o", label=cond)
    ax1.set_title("AUC vs. round"); ax1.set_xlabel("Round"); ax1.set_ylabel("AUC"); ax1.legend()
    ax2.set_title("F1 vs. round");  ax2.set_xlabel("Round"); ax2.set_ylabel("F1");  ax2.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def plot_ablation_comparison(metrics_dir: Path, out_path: Path) -> None:
    labels = ["condition_A", "condition_B", "condition_C", "condition_D"]
    aucs: list[float] = []
    f1s: list[float] = []
    present: list[str] = []
    for label in labels:
        path = metrics_dir / f"{label}_metrics.json"
        if not path.exists():
            LOG.warning("Missing %s — skipping in ablation plot", path)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "test" in data:
            aucs.append(float(data["test"].get("auc", float("nan"))))
            f1s.append(float(data["test"].get("f1", float("nan"))))
        elif "final" in data:
            aucs.append(float(data["final"].get("auc", float("nan"))))
            f1s.append(float(data["final"].get("f1", float("nan"))))
        else:
            continue
        present.append(label)

    if not present:
        LOG.warning("No ablation metrics present yet — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x = range(len(present))
    ax.bar([i - width / 2 for i in x], aucs, width=width, label="AUC")
    ax.bar([i + width / 2 for i in x], f1s, width=width, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(present)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Ablation comparison (final AUC / F1)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def plot_transfer(metrics_dir: Path, out_path: Path) -> None:
    path = metrics_dir / "transfer_attack_results.json"
    if not path.exists():
        LOG.warning("Missing %s — skipping transfer plot", path)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    in_auc = float(data["in_distribution"]["auc"])
    tr_auc = float(data["transfer"]["auc"])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["In-distribution", "Transfer"], [in_auc, tr_auc])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Transfer attack (Condition D detector)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--all_conditions", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])

    plot_evasion_vs_round(metrics_dir / "metrics_log.csv", plots_dir / "evasion_rate_vs_round.png")
    plot_auc_f1_vs_round(metrics_dir / "metrics_log.csv", plots_dir / "auc_f1_vs_round.png")
    if args.all_conditions:
        plot_ablation_comparison(metrics_dir, plots_dir / "ablation_comparison.png")
        plot_transfer(metrics_dir, plots_dir / "transfer_attack_results.png")


if __name__ == "__main__":
    main()
