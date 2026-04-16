"""Single source of truth for device + precision + memory ops.

All pipeline modules import `get_device`, `empty_cache`, `allocated_memory_gb`,
`resolve_dtype`, and `sync` from here. Never call `torch.cuda.*` directly anywhere
else in the project — a CI grep check enforces that.
"""
from __future__ import annotations

import gc
from typing import Literal

import torch

DtypeName = Literal["bfloat16", "float16", "float32"]

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def get_device(preferred: str | None = None) -> torch.device:
    """Resolve the best available device.

    Order:
      1. `preferred` argument (if given and available)
      2. MPS (Apple Silicon)
      3. CUDA (for optional CI on a GPU box)
      4. CPU
    """
    if preferred:
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cpu":
            return torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config string to a torch dtype. Raises on unknown names."""
    key = name.lower().strip()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {name!r}. Known: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def sync(device: torch.device) -> None:
    """Block until all queued work on `device` is finished. Use before timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device: torch.device) -> None:
    """Release cached allocator memory. Safe to call on any backend."""
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def allocated_memory_gb(device: torch.device) -> float:
    """Current allocated memory in GB, or 0.0 on CPU."""
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def driver_memory_gb(device: torch.device) -> float:
    """Driver-reserved memory in GB (MPS/CUDA only)."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


def describe(device: torch.device) -> str:
    alloc = allocated_memory_gb(device)
    reserved = driver_memory_gb(device)
    return f"device={device} allocated={alloc:.2f}GB reserved={reserved:.2f}GB"
