"""Base abstractions for datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Problem:
    id: str
    question: str
    ground_truth: Any  # test cases (code) or answer string (math)
    metadata: dict = field(default_factory=dict)


class DatasetStreamer(ABC):
    """Abstract base for streaming problems in order."""

    @abstractmethod
    def get_next_batch(self, batch_size: int) -> Optional[List[Problem]]:
        """Return next batch of problems, or None if exhausted."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def idx(self) -> int:
        """Current position in the stream (for resume)."""
        ...

    @idx.setter
    @abstractmethod
    def idx(self, value: int):
        ...
