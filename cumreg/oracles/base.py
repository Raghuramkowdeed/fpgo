"""Oracle base class for evaluating model responses."""

from abc import ABC, abstractmethod
from typing import Optional

from ..datasets.base import Problem


class Oracle(ABC):
    """Evaluates a model response against a problem's ground truth."""

    @abstractmethod
    def evaluate(self, response: str, problem: Problem) -> float:
        """Return score in [0, 1]. 1 = correct, 0 = incorrect."""
        ...

    @abstractmethod
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the answer portion from a model response."""
        ...

    @abstractmethod
    def get_feedback(self, response: str, problem: Problem) -> str:
        """Return feedback string for multi-turn repair prompts.

        Should include error messages, failing test info, etc.
        """
        ...
