"""BioMedLM (or any HF causal LM) fine-tuner + fake-pool generator.

PRD §3.1:
- AutoModelForCausalLM in bf16 on MPS
- gradient_checkpointing=true on HF model
- per-device batch 2, grad_accum 8  → effective batch 16
- 3 epochs, lr 2e-5
- exposes `generate_fake_pool(n, prompt_template)` for the adversarial loop

CLI:
    python models/generator/train_generator.py --config configs/config_novel.yaml
    python models/generator/train_generator.py --config configs/config_novel.yaml --generate_only --n 1000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
LOG = get_logger("generator.train")

TOPIC_POOL = [
    "type 2 diabetes biomarkers",
    "glioblastoma treatment outcomes",
    "asthma genetic susceptibility",
    "cardiovascular risk in chronic kidney disease",
    "immunotherapy response in non-small-cell lung cancer",
    "Alzheimer disease early biomarkers",
    "antibiotic resistance in urinary tract infections",
    "childhood obesity and metabolic syndrome",
]


class CausalLMDataset(Dataset):
    """Tokenize real abstracts for causal-LM fine-tuning (labels == input_ids)."""

    def __init__(self, csv_path: str, tokenizer, max_length: int, label_filter: int | None = 1) -> None:
        df = pd.read_csv(csv_path)
        if label_filter is not None and "label" in df.columns:
            df = df[df["label"] == label_filter].reset_index(drop=True)
        self.texts = df["text"].dropna().astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100  # ignore loss on pad tokens
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def load_causal_lm(cfg: dict, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["generator"]["model_name"]
    dtype = resolve_dtype(cfg["generator"].get("dtype", "bfloat16"))
    LOG.info("Loading generator %s dtype=%s on %s", name, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    if bool(cfg["generator"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    return model, tokenizer


def fine_tune(cfg: dict, device: torch.device) -> Path:
    model, tokenizer = load_causal_lm(cfg, device)
    ds = CausalLMDataset(cfg["paths"]["train_csv"], tokenizer, int(cfg["generator"]["max_length"]))
    rcfg = cfg["runtime"]
    gcfg = cfg["generator"]
    loader = DataLoader(
        ds,
        batch_size=int(gcfg["per_device_batch_size"]),
        shuffle=True,
        num_workers=int(rcfg.get("num_workers", 2)),
        pin_memory=bool(rcfg.get("pin_memory", False)),
        persistent_workers=bool(rcfg.get("persistent_workers", False))
        and int(rcfg.get("num_workers", 2)) > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(gcfg["lr"]), fused=False)
    accum = int(gcfg.get("grad_accum_steps", 1))
    epochs = int(gcfg["epochs"])

    out_dir = Path(gcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / accum
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 25 == 0:
                LOG.info(
                    "epoch=%d step=%d loss=%.4f mem=%.2fGB",
                    epoch, step, out.loss.item(), allocated_memory_gb(device),
                )

    sync(device)
    LOG.info("Generator fine-tune done in %.1fs (%s)", time.time() - t0, describe(device))
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    empty_cache(device)
    return out_dir


def generate_fake_pool(
    cfg: dict,
    device: torch.device,
    n: int,
    checkpoint: str | Path | None = None,
) -> pd.DataFrame:
    """Load the (optionally fine-tuned) generator and produce `n` fake abstracts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gcfg = cfg["generator"]
    source = str(checkpoint) if checkpoint else gcfg["model_name"]
    dtype = resolve_dtype(gcfg.get("dtype", "bfloat16"))
    LOG.info("Loading generator for sampling: %s (dtype=%s)", source, dtype)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype).to(device)
    model.eval()

    prompts: list[str] = []
    rng = random.Random(int(cfg["runtime"].get("seed", 42)))
    for _ in range(n):
        topic = rng.choice(TOPIC_POOL)
        prompts.append(gcfg["prompt_template"].format(topic=topic))

    outputs: list[str] = []
    max_new = int(gcfg.get("max_length", 256))
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_new).to(device)
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                top_p=0.95,
                temperature=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(text.strip() or "(empty)")
            if i % 10 == 0:
                LOG.info("sampled %d/%d mem=%.2fGB", i, n, allocated_memory_gb(device))

    df = pd.DataFrame({"text": outputs, "label": 0})
    empty_cache(device)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    device = get_device(cfg["runtime"].get("device"))

    if not args.generate_only:
        ckpt = fine_tune(cfg, device)
        LOG.info("Fine-tuned generator at %s", ckpt)
        args.checkpoint = args.checkpoint or str(ckpt)

    df = generate_fake_pool(cfg, device, args.n, checkpoint=args.checkpoint)
    out_path = Path(args.out or f"experiments/round_data/fake_pool_n{args.n}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOG.info("Wrote %d fakes to %s", len(df), out_path)


if __name__ == "__main__":
    main()
