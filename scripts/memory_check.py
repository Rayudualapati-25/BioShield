"""Memory sanity check — load BioMistral-7B and Qwen2.5-7B on MPS and generate one sample.

Validates that:
  - Each model loads in bf16 on MPS without OOM
  - attn_implementation="eager" works on MPS for Mistral + Qwen2 architectures
  - First-token generation is correct (sanity, not quality)

Run: python3 scripts/memory_check.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.device import describe, empty_cache, get_device  # noqa: E402
from utils.env import harden_mps_env  # noqa: E402
from utils.logging import get_logger  # noqa: E402

harden_mps_env()
LOG = get_logger("memcheck")


def process_rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def check(model_name: str, prompt: str) -> dict:
    device = get_device()
    LOG.info("=" * 72)
    LOG.info("CHECK: %s on %s (%s)", model_name, device, describe(device))
    LOG.info("Process RSS before load: %.2f GB", process_rss_gb())

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    load_s = time.time() - t0
    rss_after_load = process_rss_gb()
    LOG.info("Loaded in %.1fs. RSS after load: %.2f GB", load_s, rss_after_load)

    # Single 32-token generation
    t0 = time.time()
    with torch.inference_mode():
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    gen_s = time.time() - t0
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    rss_after_gen = process_rss_gb()
    LOG.info("Generated 32 tok in %.1fs. Peak RSS: %.2f GB", gen_s, rss_after_gen)
    LOG.info("Sample: %r", text[:200])

    # Cleanup
    del model
    del tok
    gc.collect()
    empty_cache(device)
    rss_after_free = process_rss_gb()
    LOG.info("RSS after free: %.2f GB", rss_after_free)

    return {
        "model": model_name,
        "load_seconds": round(load_s, 1),
        "gen_seconds_32tok": round(gen_s, 2),
        "rss_peak_gb": round(rss_after_gen, 2),
        "rss_after_free_gb": round(rss_after_free, 2),
        "sample_head": text[:120],
    }


if __name__ == "__main__":
    results = []
    # Generator smoke: BioMistral writes a biomedical abstract.
    results.append(check(
        "BioMistral/BioMistral-7B",
        "Write a convincing but fictitious biomedical research abstract about BRCA1 mutations and breast cancer.\nAbstract:",
    ))
    # Agent smoke: Qwen2.5 rewrites a short fake.
    results.append(check(
        "Qwen/Qwen2.5-7B-Instruct",
        "<|im_start|>user\nRewrite this medical abstract to sound more credible: BRCA1 mutations cause breast cancer always.<|im_end|>\n<|im_start|>assistant\n",
    ))

    print("\n=== Memory Check Results ===")
    for r in results:
        print(f"\n{r['model']}")
        for k, v in r.items():
            if k == "model":
                continue
            print(f"  {k}: {v}")
