"""Adversarial rewrite agent — Phi-3.5-mini-instruct + LoRA on MPS (bf16).

PRD §3.2:
- Load Phi-3.5-mini-instruct in bf16 (no 4-bit NF4 quantization; the usual CUDA-only quantizer is out).
- LoRA target modules: q_proj, v_proj, k_proj, o_proj.
- attn_implementation="eager" (FlashAttention 2 is CUDA-only).
- Rewrite prompt uses the Phi-3.5 chat template.

This module exposes `AdversarialAgent` with .rewrite(texts: list[str]) -> list[str]
for the adversarial loop. If `peft` is unavailable, rewrites still work but
LoRA fine-tuning is disabled (surfaced via a clear RuntimeError).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ensure_dirs, load_config  # noqa: E402
from utils.device import (  # noqa: E402
    allocated_memory_gb,
    empty_cache,
    get_device,
    resolve_dtype,
)
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

harden_mps_env()
LOG = get_logger("agent.adversarial")


@dataclass
class AgentConfig:
    model_name: str
    dtype: torch.dtype
    attn_implementation: str
    max_new_tokens: int
    temperature: float
    top_p: float
    rewrite_prompt: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: list[str]

    @classmethod
    def from_cfg(cls, cfg: dict) -> "AgentConfig":
        a = cfg["agent"]
        return cls(
            model_name=a["model_name"],
            dtype=resolve_dtype(a.get("dtype", "bfloat16")),
            attn_implementation=a.get("attn_implementation", "eager"),
            max_new_tokens=int(a.get("max_new_tokens", 384)),
            temperature=float(a.get("temperature", 0.8)),
            top_p=float(a.get("top_p", 0.95)),
            rewrite_prompt=a["rewrite_prompt"],
            lora_r=int(a["lora"]["r"]),
            lora_alpha=int(a["lora"]["alpha"]),
            lora_dropout=float(a["lora"]["dropout"]),
            lora_targets=list(a["lora"]["target_modules"]),
        )


class AdversarialAgent:
    def __init__(self, cfg: dict, device: torch.device) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.ac = AgentConfig.from_cfg(cfg)
        self.device = device

        LOG.info("Loading agent %s dtype=%s on %s", self.ac.model_name, self.ac.dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ac.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.ac.model_name,
            torch_dtype=self.ac.dtype,
            attn_implementation=self.ac.attn_implementation,
            low_cpu_mem_usage=True,
        )
        base.to(device)
        self.base_model = base
        self.model = base  # replaced by a PEFT wrapper if/when LoRA is enabled
        self._lora_enabled = False

    # ---------- LoRA ----------

    def enable_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise RuntimeError(
                "peft is required for LoRA fine-tuning. Install with `pip install peft`."
            ) from e

        lora_cfg = LoraConfig(
            r=self.ac.lora_r,
            lora_alpha=self.ac.lora_alpha,
            lora_dropout=self.ac.lora_dropout,
            target_modules=self.ac.lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, lora_cfg)
        self._lora_enabled = True
        LOG.info("LoRA enabled (r=%d alpha=%d targets=%s)", self.ac.lora_r, self.ac.lora_alpha, self.ac.lora_targets)

    # ---------- Rewrite ----------

    @torch.inference_mode()
    def rewrite(self, texts: list[str]) -> list[str]:
        self.model.eval()
        outputs: list[str] = []
        for i, original in enumerate(texts):
            prompt = self.ac.rewrite_prompt.format(fake_abstract=original)
            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=self.ac.max_new_tokens,
                do_sample=True,
                temperature=self.ac.temperature,
                top_p=self.ac.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            decoded = self.tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(decoded.strip() or original)
            if i % 20 == 0:
                LOG.info("rewrote %d/%d mem=%.2fGB", i, len(texts), allocated_memory_gb(self.device))
        return outputs

    # ---------- Save / teardown ----------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)

    def teardown(self) -> None:
        del self.model
        del self.base_model
        empty_cache(self.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_csv", required=True, help="CSV with a 'text' column of fakes to rewrite")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--enable_lora", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    device = get_device(cfg["runtime"].get("device"))

    agent = AdversarialAgent(cfg, device)
    if args.enable_lora:
        agent.enable_lora()

    df = pd.read_csv(args.input_csv)
    rewrites = agent.rewrite(df["text"].astype(str).tolist())
    out_df = pd.DataFrame({"original": df["text"], "text": rewrites, "label": 0})
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    LOG.info("Wrote %d rewrites to %s", len(out_df), args.output_csv)
    agent.teardown()


if __name__ == "__main__":
    main()
