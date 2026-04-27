"""Math oracle for GSM8K-style problems."""

import re
from typing import Optional

from ..datasets.base import Problem
from .base import Oracle


class MathOracle(Oracle):
    """Evaluates math responses by extracting #### answers."""

    def extract_answer(self, text: str) -> Optional[str]:
        if not text:
            return None
        # Try #### delimiter first
        if "####" in text:
            for m in re.finditer(r'####\s*(-?[\d,]+(?:\.\d+)?)', text):
                return m.group(1).replace(",", "")
            after_hash = text.split("####")[-1].strip()
            match = re.search(r'-?[\d,]+(?:\.\d+)?', after_hash)
            if match:
                return match.group().replace(",", "")
        # Fallback: last number
        matches = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
        if matches:
            return matches[-1].replace(",", "")
        return None

    def evaluate(self, response: str, problem: Problem) -> float:
        pred = self.extract_answer(response)
        truth = self.extract_answer(str(problem.ground_truth))
        if pred is None or truth is None:
            return 0.0
        try:
            return 1.0 if float(pred) == float(truth) else 0.0
        except (ValueError, TypeError):
            return 0.0

    def get_feedback(self, response: str, problem: Problem) -> str:
        pred = self.extract_answer(response)
        truth = self.extract_answer(str(problem.ground_truth))
        if pred is None:
            return "Could not extract a numerical answer. Make sure to end with #### followed by the final number."
        return f"Your answer was {pred}, but that is incorrect. Try again."
