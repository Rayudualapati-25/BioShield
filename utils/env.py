"""Environment hardening for MPS runs. Call `harden_mps_env()` once at process start."""
from __future__ import annotations

import os


def harden_mps_env() -> None:
    """Set MPS-friendly env vars if they aren't already set.

    - PYTORCH_ENABLE_MPS_FALLBACK=1  → ops without MPS kernels fall back to CPU
    - TOKENIZERS_PARALLELISM=false   → avoids fork warnings when num_workers>0

    Note: scripts that load causal LMs (BioMistral, Qwen) should also set
    `TRANSFORMERS_OFFLINE=1` *before* importing transformers to skip the aggressive
    safetensors auto-conversion PR check. See `data/prepare_data.py` for the pattern.
    That is deliberately kept out of this function because `datasets.load_dataset(...)`
    for PubMed streaming needs HF Hub reachability, so offline mode cannot be global.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
