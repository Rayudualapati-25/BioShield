"""Data preparation for the Deepfake Robustness Pipeline.

PRD §3.7: streams ~10k real PubMed abstracts, combines with Med-MMHL fakes,
applies cleaning (HTML strip, dedup, min-words filter) and splits train/val/test.

`data.source: synthetic` in config uses a deterministic PubMed-styled fixture
so dry runs and CI do not require network access or a HuggingFace login.

CLI:
    python data/prepare_data.py --config configs/config_novel.yaml --max_real 10000
    python data/prepare_data.py --config configs/config_dryrun.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

import pandas as pd

# Make repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

LOG = get_logger("data.prepare")

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _HTML_RE.sub(" ", text or "")
    text = _WS_RE.sub(" ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def _content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


# ---------- Synthetic fixture (dry runs only) ----------

_REAL_SEEDS = [
    "We investigated the role of {gene} in {disease} using a cohort of {n} patients. "
    "Methods included {method} with statistical significance assessed at p<0.05. "
    "Results indicated {finding}. These findings suggest {implication}.",
    "Objective: To evaluate {treatment} in patients with {disease}. "
    "Methods: A randomized controlled trial enrolled {n} participants. "
    "Results: The intervention reduced {marker} by {pct}%. "
    "Conclusion: {treatment} shows promise for clinical use.",
    "Background: {disease} remains a leading cause of morbidity. "
    "Methods: We performed {method} on {n} samples. "
    "Results: Expression of {gene} correlated with disease severity (r={r}). "
    "Conclusion: {gene} is a candidate biomarker.",
]

_FAKE_SEEDS = [
    "Our study totally proves that {gene} causes {disease}. We looked at {n} people. "
    "The results were absolutely clear and significant in every way possible.",
    "Scientists have long wondered about {disease}. In this groundbreaking paper we show "
    "that {treatment} cures it. {n} patients were tested. All of them got better immediately.",
    "It is well known that {gene} is important. In this study we confirmed that {gene} is "
    "really very important indeed. The implications are revolutionary for {disease}.",
]

_GENES = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC", "PTEN", "VEGF", "IL6", "TNF"]
_DISEASES = [
    "type 2 diabetes", "breast cancer", "Alzheimer disease", "hypertension",
    "rheumatoid arthritis", "chronic kidney disease", "asthma", "Parkinson disease",
]
_METHODS = ["qPCR", "RNA-seq", "immunohistochemistry", "mass spectrometry", "Western blot"]
_TREATMENTS = ["metformin", "aspirin", "tamoxifen", "statin therapy", "immunotherapy"]


def _synth_sample(template_pool: list[str], rng: random.Random) -> str:
    tpl = rng.choice(template_pool)
    return tpl.format(
        gene=rng.choice(_GENES),
        disease=rng.choice(_DISEASES),
        method=rng.choice(_METHODS),
        treatment=rng.choice(_TREATMENTS),
        n=rng.randint(40, 2000),
        pct=rng.randint(5, 65),
        r=round(rng.uniform(0.3, 0.9), 2),
        marker=rng.choice(["HbA1c", "LDL-C", "CRP", "blood pressure"]),
        finding=rng.choice([
            "significant upregulation in disease tissue",
            "a dose-dependent response",
            "improved survival in the treatment arm",
        ]),
        implication=rng.choice([
            "a novel therapeutic target",
            "the need for further mechanistic study",
            "potential for clinical translation",
        ]),
    )


def build_synthetic_pool(n_real: int, n_fake: int, seed: int) -> pd.DataFrame:
    """Generate a deterministic PubMed-styled fixture. Labels: 1=real, 0=fake."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for _ in range(n_real):
        rows.append({"text": _synth_sample(_REAL_SEEDS, rng), "label": 1})
    for _ in range(n_fake):
        rows.append({"text": _synth_sample(_FAKE_SEEDS, rng), "label": 0})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------- Real PubMed path (streaming) ----------

def load_pubmed_real(max_samples: int, min_words: int, seed: int) -> list[str]:
    """Stream PubMed abstracts from HuggingFace Datasets.

    Uses streaming so we never materialize the full corpus on disk. If the
    hub call fails (no network / auth), we raise clearly so the caller can
    switch to `data.source: synthetic` instead of silently producing bad data.
    """
    try:
        from datasets import load_dataset  # local import keeps dry runs lean
    except Exception as e:  # pragma: no cover
        raise RuntimeError("`datasets` package is required for PubMed streaming") from e

    LOG.info("Streaming PubMed abstracts (target=%d, min_words=%d)...", max_samples, min_words)
    stream = load_dataset("ncbi/pubmed", split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    collected: list[str] = []
    seen: set[str] = set()
    for row in stream:
        abs_text = ((row.get("MedlineCitation", {}) or {}).get("Article", {}) or {}).get("Abstract", {}) or {}
        text = abs_text.get("AbstractText") or ""
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        text = clean_text(str(text))
        if word_count(text) < min_words:
            continue
        h = _content_hash(text)
        if h in seen:
            continue
        seen.add(h)
        collected.append(text)
        if len(collected) >= max_samples:
            break
    LOG.info("Collected %d unique real abstracts.", len(collected))
    return collected


def load_med_mmhl_fakes(path: str | None, min_words: int) -> list[str]:
    """Load Med-MMHL (or any CSV with a `text` column) as the fake corpus."""
    if not path:
        LOG.warning("No med_mmhl_fakes_path set — returning empty fake corpus.")
        return []
    df = pd.read_csv(path)
    col = "text" if "text" in df.columns else df.columns[0]
    texts = [clean_text(str(t)) for t in df[col].dropna().tolist()]
    texts = [t for t in texts if word_count(t) >= min_words]
    LOG.info("Loaded %d Med-MMHL fakes from %s", len(texts), path)
    return texts


def split_and_write(df: pd.DataFrame, cfg: dict, seed: int) -> None:
    """Stratified split on `label`, write CSVs to paths.train/val/test."""
    from sklearn.model_selection import train_test_split

    val_r = float(cfg["data"]["val_ratio"])
    test_r = float(cfg["data"]["test_ratio"])
    if val_r + test_r >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    train_val, test = train_test_split(df, test_size=test_r, stratify=df["label"], random_state=seed)
    rel_val = val_r / (1.0 - test_r)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)

    train.to_csv(cfg["paths"]["train_csv"], index=False)
    val.to_csv(cfg["paths"]["val_csv"], index=False)
    test.to_csv(cfg["paths"]["test_csv"], index=False)

    # Transfer test placeholder (filled by eval_pipeline.py --mode transfer).
    Path(cfg["paths"]["transfer_test_csv"]).write_text("text,label\n", encoding="utf-8")

    LOG.info(
        "Wrote splits: train=%d val=%d test=%d", len(train), len(val), len(test),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_real", type=int, default=None)
    parser.add_argument("--source", choices=["pubmed", "synthetic"], default=None,
                        help="Override data.source from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed = int(cfg["runtime"].get("seed", 42))
    set_seed(seed)

    source = args.source or cfg["data"].get("source", "pubmed")
    max_real = int(args.max_real) if args.max_real is not None else int(cfg["data"]["max_real_samples"])
    min_words = int(cfg["data"]["min_words"])

    if source == "synthetic":
        LOG.info("Source=synthetic — generating deterministic fixture (n_real=%d)", max_real)
        df = build_synthetic_pool(n_real=max_real, n_fake=max_real, seed=seed)
    else:
        reals = load_pubmed_real(max_real, min_words, seed)
        fakes = load_med_mmhl_fakes(cfg["data"].get("med_mmhl_fakes_path"), min_words)
        if not fakes:
            raise RuntimeError(
                "No fakes available. Set data.med_mmhl_fakes_path or use data.source: synthetic."
            )
        # Balance: take equal count so the detector trains on a balanced binary task.
        n = min(len(reals), len(fakes))
        rows = (
            [{"text": t, "label": 1} for t in reals[:n]]
            + [{"text": t, "label": 0} for t in fakes[:n]]
        )
        df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Final cleanup pass — strip HTML, dedup, length filter.
    df["text"] = df["text"].map(clean_text)
    before = len(df)
    df["_h"] = df["text"].map(_content_hash)
    df = df.drop_duplicates(subset="_h").drop(columns=["_h"]).reset_index(drop=True)
    df = df[df["text"].map(word_count) >= min_words].reset_index(drop=True)
    LOG.info("Post-clean rows: %d (from %d)", len(df), before)

    split_and_write(df, cfg, seed)


if __name__ == "__main__":
    main()
