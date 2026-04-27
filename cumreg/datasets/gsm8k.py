"""GSM8K dataset streamer."""

import os
from typing import List, Optional

import pandas as pd

from .base import DatasetStreamer, Problem


class GSM8KStreamer(DatasetStreamer):
    """Stream GSM8K math problems."""

    def __init__(self, data_dir: str = "data", seed: int = 42):
        self._problems: List[Problem] = []
        self._idx = 0
        self._load(data_dir, seed)

    def _load(self, data_dir: str, seed: int):
        cache_path = os.path.join(data_dir, "gsm8k_shuffled.csv")

        if os.path.exists(cache_path):
            print(f"Loading cached GSM8K from {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            from datasets import load_dataset
            print("Downloading GSM8K...")
            ds = load_dataset("openai/gsm8k", "main", split="train")
            ds = ds.shuffle(seed=seed)
            df = pd.DataFrame({"question": ds["question"], "answer": ds["answer"]})
            os.makedirs(data_dir, exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"Cached {len(df)} problems to {cache_path}")

        for i, row in df.iterrows():
            self._problems.append(Problem(
                id=str(i),
                question=row["question"],
                ground_truth=row["answer"],
                metadata={"dataset": "gsm8k"},
            ))

    def get_next_batch(self, batch_size: int) -> Optional[List[Problem]]:
        if self._idx >= len(self._problems):
            return None
        batch = self._problems[self._idx : self._idx + batch_size]
        self._idx += batch_size
        return batch

    def __len__(self) -> int:
        return len(self._problems)

    @property
    def idx(self) -> int:
        return self._idx

    @idx.setter
    def idx(self, value: int):
        self._idx = value
