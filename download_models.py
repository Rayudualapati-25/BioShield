"""Download all open production models to HuggingFace cache.
Run: python3 download_models.py
Llama-3.2-3B-Instruct is skipped (gated — run separately after huggingface-cli login).
"""
import time
from huggingface_hub import snapshot_download

MODELS = [
    ("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract", "Detector"),
    ("BioMistral/BioMistral-7B", "Generator"),
    ("Qwen/Qwen2.5-7B-Instruct", "Adversarial Agent"),
]

for repo_id, role in MODELS:
    print(f"\n{'='*60}")
    print(f"Downloading [{role}]: {repo_id}")
    print(f"{'='*60}")
    t0 = time.time()
    path = snapshot_download(repo_id=repo_id)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s → {path}")

print("\n✅ All 3 open models downloaded.")
print("⏭  Skipped: meta-llama/Llama-3.2-3B-Instruct (download tomorrow after login)")
