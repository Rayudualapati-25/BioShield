"""Text-classification dataset used by the detector. Works for any HF tokenizer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """Lightweight dataset. Tokenization is done lazily per item to save RAM."""

    def __init__(self, csv_path: str | Path, tokenizer, max_length: int) -> None:
        self.df = pd.read_csv(csv_path)
        if not {"text", "label"}.issubset(self.df.columns):
            raise ValueError(f"{csv_path} must have 'text' and 'label' columns")
        self.df = self.df.dropna(subset=["text", "label"]).reset_index(drop=True)
        self.df["label"] = self.df["label"].astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }
