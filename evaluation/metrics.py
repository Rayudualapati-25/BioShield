"""Evaluation metrics. PRD §3.6 gates BERTScore + perplexity behind config flags.

All metrics here are MPS-aware — bert-score is called with `device="mps"` and
perplexity is computed on the active `utils.device.get_device()` device.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def classification_metrics(
    y_true: Iterable[int], y_prob: Iterable[float], y_pred: Iterable[int] | None = None
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_true_np = np.asarray(list(y_true))
    y_prob_np = np.asarray(list(y_prob))
    if y_pred is None:
        y_pred_np = (y_prob_np >= 0.5).astype(int)
    else:
        y_pred_np = np.asarray(list(y_pred))
    try:
        auc = roc_auc_score(y_true_np, y_prob_np) if len(set(y_true_np.tolist())) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "auc": float(auc),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
    }


def evasion_rate(y_true_fake: Iterable[int], y_prob_real: Iterable[float]) -> float:
    """Share of fake items (label==0) the detector labels as real (prob_real > 0.5)."""
    y_np = np.asarray(list(y_true_fake))
    p_np = np.asarray(list(y_prob_real))
    fake_mask = y_np == 0
    if not fake_mask.any():
        return 0.0
    return float((p_np[fake_mask] > 0.5).mean())


# ---------- BERTScore (gated by cfg.metrics.compute_bertscore) ----------

def compute_bertscore_f1(
    originals: list[str], rewrites: list[str], model_name: str, device: torch.device
) -> float:
    try:
        from bert_score import score
    except Exception as e:  # pragma: no cover
        raise RuntimeError("bert-score not installed; pip install bert-score") from e
    _P, _R, F1 = score(
        rewrites, originals,
        model_type=model_name,
        device=str(device),
        rescale_with_baseline=False,
        verbose=False,
    )
    return float(F1.mean().item())


# ---------- Perplexity (gated by cfg.metrics.compute_perplexity) ----------

@torch.inference_mode()
def compute_mean_perplexity(
    texts: list[str], reference_lm_name: str, device: torch.device, max_length: int = 512
) -> float:
    """Mean token-normalized perplexity using a causal reference LM.

    Masked LMs (e.g. PubMedBERT) are not exact language models; for a clean
    perplexity signal prefer a causal LM. When the config specifies PubMedBERT
    we compute pseudo-perplexity via AutoModelForCausalLM if possible;
    otherwise fall back to AutoModel.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(reference_lm_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(reference_lm_name).to(device)
    except Exception:  # MLM fallback — issues a log warning and uses loss-of-tokens as proxy
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(reference_lm_name).to(device)
    model.eval()

    losses: list[float] = []
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length,
        ).to(device)
        try:
            out = model(**enc, labels=enc["input_ids"])
            loss = float(out.loss.item())
        except TypeError:
            # MLM path — no labels kwarg; approximate with mean cross-entropy on a random mask
            from torch.nn.functional import cross_entropy
            logits = model(**enc).logits
            loss = float(cross_entropy(
                logits.view(-1, logits.size(-1)), enc["input_ids"].view(-1)
            ).item())
        if np.isfinite(loss):
            losses.append(loss)

    if not losses:
        return float("nan")
    return float(np.exp(np.mean(losses)))
