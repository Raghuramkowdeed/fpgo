"""LiveCodeBench dataset streamer."""

import json
import os
from typing import List, Optional

from .base import DatasetStreamer, Problem


class LiveCodeBenchStreamer(DatasetStreamer):
    """Stream LiveCodeBench code generation problems from HuggingFace."""

    def __init__(self, data_dir: str = "data", seed: int = 42):
        self._problems: List[Problem] = []
        self._idx = 0
        self._load(data_dir, seed)

    def _load(self, data_dir: str, seed: int):
        import random

        cache_path = os.path.join(data_dir, "livecodebench_problems.json")
        if os.path.exists(cache_path):
            print(f"Loading cached LiveCodeBench from {cache_path}")
            with open(cache_path) as f:
                records = json.load(f)
            self._problems = [Problem(**r) for r in records]
            return

        print("Downloading LiveCodeBench from HuggingFace...")
        rows = self._download_jsonl()

        problems = []
        for row in rows:
            # Parse public test cases (JSON string -> list of dicts)
            public_tests = row.get("public_test_cases", "[]")
            if isinstance(public_tests, str):
                try:
                    public_tests = json.loads(public_tests)
                except json.JSONDecodeError:
                    public_tests = []

            # Determine fn_name from starter_code (LeetCode style)
            starter_code = row.get("starter_code", "")
            fn_name = self._extract_fn_name(starter_code) if starter_code else None

            ground_truth = {
                "public_test_cases": public_tests,
                "private_test_cases": "",  # stripped to keep cache small
                "fn_name": fn_name,
                "testtype": public_tests[0].get("testtype", "stdin") if public_tests else "stdin",
            }

            metadata = {
                "platform": row.get("platform", ""),
                "difficulty": row.get("difficulty", ""),
                "starter_code": starter_code,
                "question_title": row.get("question_title", ""),
                "contest_id": row.get("contest_id", ""),
                "contest_date": row.get("contest_date", ""),
            }

            problems.append(Problem(
                id=row.get("question_id", str(len(problems))),
                question=row["question_content"],
                ground_truth=ground_truth,
                metadata=metadata,
            ))

        # Shuffle deterministically
        rng = random.Random(seed)
        rng.shuffle(problems)
        self._problems = problems

        # Cache
        os.makedirs(data_dir, exist_ok=True)
        records = [
            {"id": p.id, "question": p.question,
             "ground_truth": p.ground_truth, "metadata": p.metadata}
            for p in problems
        ]
        with open(cache_path, "w") as f:
            json.dump(records, f)
        print(f"Cached {len(problems)} problems to {cache_path}")

    @staticmethod
    def _download_jsonl():
        """Download all JSONL splits from HuggingFace hub."""
        from huggingface_hub import hf_hub_download, list_repo_files
        import re

        repo = "livecodebench/code_generation_lite"
        files = list_repo_files(repo, repo_type="dataset")
        jsonl_files = sorted(f for f in files if f.endswith(".jsonl"))

        rows = []
        for fname in jsonl_files:
            path = hf_hub_download(repo, fname, repo_type="dataset")
            with open(path) as f:
                for line in f:
                    rows.append(json.loads(line))
        print(f"Loaded {len(rows)} problems from {len(jsonl_files)} JSONL files")
        return rows

    @staticmethod
    def _extract_fn_name(starter_code: str):
        """Extract function name from LeetCode starter code."""
        import re
        match = re.search(r'def\s+(\w+)\s*\(', starter_code)
        return match.group(1) if match else None

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
