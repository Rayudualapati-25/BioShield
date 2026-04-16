"""Smoke tests — these must pass on any checkout with no network / no large downloads.

They cover:
  - device resolution + dtype mapping
  - config loading + dotted get
  - synthetic data fixture shape
  - SeqGAN token dataset + generator forward pass
  - no-CUDA discipline grep
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.prepare_data import build_synthetic_pool, clean_text, word_count  # noqa: E402
from models.seqgan.train_seqgan import (  # noqa: E402
    SeqGANGenerator,
    TokenDataset,
    build_vocab,
    tokenize,
)
from utils.config import get as cfg_get
from utils.config import load_config, merge  # noqa: E402
from utils.device import empty_cache, get_device, resolve_dtype  # noqa: E402


def test_device_resolves():
    d = get_device()
    assert d.type in {"mps", "cuda", "cpu"}


def test_dtype_mapping():
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype("bf16") == torch.bfloat16
    assert resolve_dtype("float32") == torch.float32
    with pytest.raises(ValueError):
        resolve_dtype("int8")


def test_empty_cache_runs_on_cpu():
    empty_cache(torch.device("cpu"))  # must not raise


def test_config_roundtrip():
    cfg = load_config(ROOT / "configs" / "config_dryrun.yaml")
    assert cfg_get(cfg, "runtime.device") == "mps"
    assert cfg_get(cfg, "detector.model_name", "?") != "?"
    merged = merge(cfg, {"runtime": {"device": "cpu"}})
    assert cfg_get(merged, "runtime.device") == "cpu"
    assert cfg_get(cfg, "runtime.device") == "mps"  # no mutation of original


def test_synthetic_fixture_is_balanced_and_clean():
    df = build_synthetic_pool(n_real=30, n_fake=30, seed=0)
    assert len(df) == 60
    assert set(df["label"].unique()) == {0, 1}
    assert df["label"].value_counts().min() == 30
    assert all(word_count(t) >= 10 for t in df["text"])
    assert clean_text("<p>hello  world</p>") == "hello world"


def test_seqgan_forward_pass():
    vocab = build_vocab(["tumor cells grow", "abstract biomedical text"], max_size=40)
    ds = TokenDataset(["tumor cells grow abstract text"], vocab, seq_len=12)
    model = SeqGANGenerator(
        vocab_size=len(vocab), emb_dim=8, hidden_dim=16, pad_idx=vocab["<pad>"]
    )
    x = ds[0].unsqueeze(0)
    out = model(x)
    assert out.shape == (1, 12, len(vocab))


def test_tokenize_splits_medical_terms():
    toks = tokenize("BRCA1 mutation p<0.05 (n=42)")
    assert "brca" in toks
    assert "42" in toks


def test_no_direct_cuda_calls_in_source():
    """Grep-check the discipline: no module outside utils/device.py calls torch.cuda.* directly."""
    bad = []
    forbidden = ["torch.cuda.empty_cache", "bitsandbytes", "BitsAndBytesConfig", "nvidia-smi"]
    for path in ROOT.rglob("*.py"):
        if "tests/" in str(path):
            continue
        if path.name == "device.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden:
            if needle in text:
                bad.append(f"{path}: {needle}")
    assert not bad, "CUDA-only discipline violated:\n" + "\n".join(bad)
