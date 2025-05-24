from __future__ import annotations

from typing import Tuple

from datasets import load_dataset, IterableDataset


class DatasetLoader:
    """Utility for streaming datasets and tokenization."""

    def __init__(self, dataset_name: str, subset: str, *, text_field: str | None = None,
                 shuffle_buffer: int = 10000, seed: int = 0) -> None:
        self.dataset_name = dataset_name
        self.subset = subset
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def load(self) -> Tuple[IterableDataset, str]:
        ds = load_dataset(
            self.dataset_name,
            name=self.subset,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(self.shuffle_buffer, seed=self.seed)
        first = next(iter(ds.take(1)))
        if self.text_field:
            if self.text_field not in first:
                raise ValueError(f"Dataset has no field '{self.text_field}'")
            field = self.text_field
        else:
            if "text" in first:
                field = "text"
            elif "content" in first:
                field = "content"
            else:
                raise ValueError(
                    "Dataset must contain a 'text' or 'content' field; use text_field param"
                )
        return ds, field

    @staticmethod
    def tokenize_batch(tokenizer, examples, *, text_field: str, seq_len: int, dynamic_batch: bool):
        texts = [ex[text_field] for ex in examples]
        tokens = tokenizer(
            texts,
            max_length=seq_len,
            truncation=True,
            padding=True if dynamic_batch else "max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens
