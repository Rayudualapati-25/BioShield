"""Minimal SeqGAN generator used ONLY for ablation Condition B.

PRD §2 demotes SeqGAN to the weak-baseline ablation. This file keeps the
original SeqGAN design (word-level LSTM) but wires it to MPS so the ablation
can run without CUDA. It is intentionally small: policy-gradient training
of a full SeqGAN on biomedical text is out of scope for Condition B's
role, which only needs a weak-but-runnable fake text generator.

CLI:
    python models/seqgan/train_seqgan.py --config configs/config_novel.yaml
    python models/seqgan/train_seqgan.py --config configs/config_novel.yaml --generate_only --n 200
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.device import empty_cache, get_device, sync  # noqa: E402
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("seqgan.train")

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], max_size: int) -> dict[str, int]:
    counts: Counter = Counter()
    for t in texts:
        counts.update(tokenize(t))
    specials = [PAD, BOS, EOS, UNK]
    top = [w for w, _ in counts.most_common(max_size - len(specials))]
    return {w: i for i, w in enumerate(specials + top)}


class TokenDataset(Dataset):
    def __init__(self, texts: list[str], vocab: dict[str, int], seq_len: int) -> None:
        self.vocab = vocab
        self.seq_len = seq_len
        self.unk = vocab[UNK]
        self.bos = vocab[BOS]
        self.eos = vocab[EOS]
        self.pad = vocab[PAD]
        self.samples: list[list[int]] = []
        for t in texts:
            ids = [self.bos] + [vocab.get(tok, self.unk) for tok in tokenize(t)] + [self.eos]
            ids = ids[: seq_len]
            if len(ids) < seq_len:
                ids += [self.pad] * (seq_len - len(ids))
            self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.samples[idx], dtype=torch.long)


class SeqGANGenerator(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, pad_idx: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        h, _ = self.lstm(e)
        return self.head(h)

    @torch.inference_mode()
    def sample(self, bos: int, eos: int, pad: int, seq_len: int, batch: int, device: torch.device) -> torch.Tensor:
        tokens = torch.full((batch, 1), bos, dtype=torch.long, device=device)
        hidden = None
        for _ in range(seq_len - 1):
            e = self.emb(tokens[:, -1:])
            out, hidden = self.lstm(e, hidden)
            logits = self.head(out[:, -1])
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens


def decode(ids: list[int], inv_vocab: dict[int, str]) -> str:
    out = []
    for i in ids:
        tok = inv_vocab.get(int(i), UNK)
        if tok in (BOS, PAD):
            continue
        if tok == EOS:
            break
        out.append(tok)
    return " ".join(out)


def train(cfg: dict, device: torch.device) -> Path:
    scfg = cfg["seqgan"]
    df = pd.read_csv(cfg["paths"]["train_csv"])
    texts = df[df.get("label", 1) == 1]["text"].dropna().astype(str).tolist() if "label" in df else df["text"].tolist()
    vocab = build_vocab(texts, int(scfg["vocab_size"]))
    ds = TokenDataset(texts, vocab, int(scfg["seq_length"]))
    loader = DataLoader(ds, batch_size=int(scfg["batch_size"]), shuffle=True, pin_memory=False)

    model = SeqGANGenerator(
        vocab_size=len(vocab),
        emb_dim=int(scfg["embedding_dim"]),
        hidden_dim=int(scfg["hidden_dim"]),
        pad_idx=vocab[PAD],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD])

    # Maximum-likelihood pretraining — the "G" half of SeqGAN. Adversarial
    # training is omitted because Condition B only needs a weak generator.
    epochs = int(scfg["pretrain_epochs"])
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * inp.size(0)
            n += inp.size(0)
        LOG.info("seqgan epoch=%d loss=%.4f", epoch, total / max(n, 1))

    out_dir = Path(scfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "cfg": scfg}, out_dir / "seqgan.pt")
    sync(device)
    LOG.info("SeqGAN saved to %s", out_dir / "seqgan.pt")
    empty_cache(device)
    return out_dir / "seqgan.pt"


def generate(cfg: dict, device: torch.device, checkpoint: str, n: int) -> pd.DataFrame:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    vocab: dict[str, int] = state["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    scfg = cfg["seqgan"]
    model = SeqGANGenerator(
        vocab_size=len(vocab),
        emb_dim=int(scfg["embedding_dim"]),
        hidden_dim=int(scfg["hidden_dim"]),
        pad_idx=vocab[PAD],
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    batch = int(scfg["batch_size"])
    seq_len = int(scfg["seq_length"])
    outs: list[str] = []
    while len(outs) < n:
        ids = model.sample(vocab[BOS], vocab[EOS], vocab[PAD], seq_len, batch, device)
        for row in ids.tolist():
            outs.append(decode(row, inv_vocab))
            if len(outs) >= n:
                break
    empty_cache(device)
    return pd.DataFrame({"text": outs, "label": 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    device = get_device(cfg["runtime"].get("device"))

    if not args.generate_only:
        ckpt = train(cfg, device)
        args.checkpoint = args.checkpoint or str(ckpt)

    df = generate(cfg, device, args.checkpoint, args.n)
    out_path = Path(args.out or f"experiments/round_data/seqgan_fake_n{args.n}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOG.info("Wrote %d SeqGAN fakes to %s", len(df), out_path)


if __name__ == "__main__":
    main()
