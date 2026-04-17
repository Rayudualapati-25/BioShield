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
import os
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
    """Stream real biomedical abstracts from HuggingFace Datasets.

    Uses streaming so we never materialize the full corpus on disk. Tries the
    modern parquet-backed `ccdv/pubmed-summarization` first (a published 2021
    snapshot of PubMed abstracts). Falls back to `armanc/scientific_papers` if
    that is unavailable. `ncbi/pubmed` (the script-backed dataset) is NOT used
    because it was retired from the `datasets` library in v3.x.
    """
    try:
        from datasets import load_dataset  # local import keeps dry runs lean
    except Exception as e:  # pragma: no cover
        raise RuntimeError("`datasets` package is required for PubMed streaming") from e

    LOG.info("Streaming PubMed abstracts (target=%d, min_words=%d)...", max_samples, min_words)

    # Source precedence — first that works wins.
    # Each entry: (repo, config_name or None, split, field)
    candidates = [
        # ccdv/pubmed-summarization has two configs: "document" (full-text article) and
        # "section" (section-level). We use "document" and take the "abstract" field.
        ("ccdv/pubmed-summarization", "document", "train", "abstract"),
        ("ccdv/pubmed-summarization", "section", "train", "abstract"),
        # scientific_papers has a "pubmed" config with article + abstract.
        ("armanc/scientific_papers", "pubmed", "train", "abstract"),
    ]

    last_err: Exception | None = None
    collected: list[str] = []
    seen: set[str] = set()
    used_repo: str | None = None
    for repo, config, split, field in candidates:
        try:
            LOG.info("  Trying %s[%s, config=%s] field=%s ...", repo, split, config, field)
            kw = {"split": split, "streaming": True, "trust_remote_code": True}
            if config:
                kw["name"] = config
            stream = load_dataset(repo, **kw).shuffle(seed=seed, buffer_size=2_000)
            used_repo = f"{repo}:{config}" if config else repo
            for row in stream:
                text = row.get(field) or ""
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
            if collected:
                break  # success — don't try fallbacks
        except Exception as e:
            last_err = e
            LOG.warning("    %s failed: %s", repo, str(e)[:160])
            continue

    if not collected:
        raise RuntimeError(
            f"Could not stream any biomedical abstracts. Last error: {last_err}. "
            "Fix: check network or HF access, or use data.source: synthetic for dry runs."
        )
    LOG.info("Collected %d unique real abstracts from %s.", len(collected), used_repo)
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


# ---------- Initial fake generation via pre-trained LM (zero-shot BioMistral) ----------
#
# PRD §3.0 calls for a balanced real-vs-fake corpus before the adversarial loop begins.
# Rather than requiring a pre-existing misinformation dataset (Med-MMHL), we use the
# pre-trained generator model (no fine-tuning) to produce the initial fakes. This is
# stronger for the paper because:
#   1. Fakes come from the same model family (BioMistral) whose outputs the detector
#      will face throughout the adversarial loop — no distribution shift at round 0.
#   2. No external dataset dependency — fully reproducible from public weights.
#
# The generator runs on MPS at bf16 with `attn_implementation="eager"` (utils/device
# discipline). Offloaded immediately after generation.

_TOPICS = [
    "BRCA1 mutations and triple-negative breast cancer",
    "metformin as an adjunct therapy in type 2 diabetes",
    "EGFR-targeted therapy in non-small cell lung cancer",
    "biomarkers for early-stage Alzheimer disease",
    "RNA-seq analysis of tumor heterogeneity in glioblastoma",
    "statins and cardiovascular risk reduction in primary prevention",
    "immune checkpoint inhibitors in metastatic melanoma",
    "gut microbiome composition and inflammatory bowel disease",
    "CRISPR-based therapy for sickle cell disease",
    "renal biomarkers in chronic kidney disease progression",
    "SARS-CoV-2 spike protein and long COVID symptoms",
    "deep brain stimulation in Parkinson disease",
    "GLP-1 receptor agonists for obesity management",
    "tau protein aggregation in frontotemporal dementia",
    "anti-VEGF therapy in diabetic retinopathy",
    "gene expression profiling in hepatocellular carcinoma",
    "IL-6 signaling in rheumatoid arthritis",
    "telomere length and cellular senescence",
    "BNT162b2 mRNA vaccine immunogenicity",
    "KRAS G12C inhibitors in colorectal cancer",
]


def load_generator_fakes(
    generator_cfg: dict,
    n_fakes: int,
    min_words: int,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 42,
) -> list[str]:
    """Generate initial fake abstracts using the pre-trained generator (zero-shot).

    Lives in prepare_data so the detector's initial training corpus is bootstrapped
    without requiring Med-MMHL or any third-party fake dataset. Uses utils.device to
    stay MPS-faithful and offloads the model immediately after generation.
    """
    # Force transformers + huggingface_hub to offline mode BEFORE importing transformers.
    # Rationale: transformers 5.x aggressively calls a remote HF Space to auto-convert
    # pytorch_model.bin → model.safetensors, even with use_safetensors=False + local_files_only=True.
    # That network call can time out and then raise OSError. By the time this function
    # runs, all datasets.load_dataset(...) streaming for PubMed has already completed
    # (see main() ordering), so we can safely flip offline mode for the model load.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Force-refresh huggingface_hub constants: they are read once at import-time, so
    # functions (including transformers' safetensors_conversion) won't pick up the env
    # vars we just set unless we patch the module attribute directly.
    try:
        import huggingface_hub.constants as _hf_constants
        _hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Local imports keep the dry-run synthetic path lean (no torch/transformers needed).
    from utils.device import empty_cache, get_device, resolve_dtype
    from utils.env import harden_mps_env

    harden_mps_env()
    device = get_device(generator_cfg.get("device"))
    dtype = resolve_dtype(generator_cfg.get("dtype", "bfloat16"))
    model_name = generator_cfg["model_name"]
    prompt_template = generator_cfg.get(
        "prompt_template",
        "Write a convincing but fictitious biomedical research abstract about {topic}.\nAbstract:",
    )
    attn_impl = generator_cfg.get("attn_implementation", "eager")

    LOG.info(
        "Loading generator %s on %s (dtype=%s) for zero-shot fake generation...",
        model_name, device, dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # BioMistral-7B ships only pytorch_model.bin. Transformers 5.x aggressively tries to
    # auto-convert to safetensors via a remote HF Space call — which times out when the
    # weights are already local and we don't need it. Force local-only + use_safetensors=False
    # to skip the conversion path entirely.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        use_safetensors=False,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    rng = random.Random(seed)
    fakes: list[str] = []
    attempts = 0
    max_attempts = int(n_fakes * 3)  # allow rejection for length/dedup
    seen: set[str] = set()
    with torch.inference_mode():
        while len(fakes) < n_fakes and attempts < max_attempts:
            attempts += 1
            topic = rng.choice(_TOPICS)
            prompt = prompt_template.format(topic=topic)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Strip the prompt from the generated text.
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = clean_text(text)
            if word_count(text) < min_words:
                continue
            h = _content_hash(text)
            if h in seen:
                continue
            seen.add(h)
            fakes.append(text)
            if len(fakes) % 25 == 0 or len(fakes) == n_fakes:
                LOG.info("  [fake gen] %d / %d (attempts=%d)", len(fakes), n_fakes, attempts)

    # Offload generator weights — detector training comes next on the same device.
    del model
    del tokenizer
    empty_cache(device)

    LOG.info("Zero-shot generator produced %d unique fakes in %d attempts.", len(fakes), attempts)
    return fakes


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
    parser.add_argument("--fakes-source", choices=["auto", "generator", "med_mmhl"], default="auto",
                        help="auto=use med_mmhl if path set else generator; generator=force zero-shot LM; "
                             "med_mmhl=force loading from path")
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

        # Resolve the fake corpus source with a clear precedence order.
        mmhl_path = cfg["data"].get("med_mmhl_fakes_path")
        use_generator = (
            args.fakes_source == "generator"
            or (args.fakes_source == "auto" and not mmhl_path)
        )
        if use_generator:
            LOG.info("Fake source: zero-shot generator (%s)", cfg["generator"]["model_name"])
            n_fakes = len(reals)  # target a balanced corpus
            fakes = load_generator_fakes(
                generator_cfg=cfg["generator"],
                n_fakes=n_fakes,
                min_words=min_words,
                max_new_tokens=int(cfg["data"].get("initial_fake_max_new_tokens", 256)),
                temperature=float(cfg["data"].get("initial_fake_temperature", 0.9)),
                seed=seed,
            )
        else:
            LOG.info("Fake source: Med-MMHL CSV at %s", mmhl_path)
            fakes = load_med_mmhl_fakes(mmhl_path, min_words)

        if not fakes:
            raise RuntimeError(
                "No fakes available. Fix: (a) set data.med_mmhl_fakes_path, "
                "(b) rerun with --fakes-source generator, or (c) use data.source: synthetic."
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
