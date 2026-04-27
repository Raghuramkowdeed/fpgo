"""Experiment manager: CSV logging, resume, and regret tracking."""

import json
import os
import time
from typing import Optional

import pandas as pd

from .config import Config


class ExperimentManager:
    """Manages experiment state, logging, and resume."""

    def __init__(self, config: Config):
        self.config = config
        self.exp_dir = config.exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        self.log_file = os.path.join(self.exp_dir, "results.csv")
        self.state_file = os.path.join(self.exp_dir, "state.json")
        self.config_file = os.path.join(self.exp_dir, "config.json")

        # Save config
        self._save_config()

        # Tracking
        self.cumulative_regret = 0
        self.total_solved = 0
        self.total_seen = 0
        self.start_time = time.time()

    def _save_config(self):
        from dataclasses import asdict
        with open(self.config_file, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def get_resume_step(self) -> int:
        """Return the step to resume from (0 if fresh start)."""
        if not os.path.exists(self.state_file):
            return 0
        with open(self.state_file) as f:
            state = json.load(f)
        resume_step = state.get("step", 0)

        # Trim CSV to match state
        if os.path.exists(self.log_file):
            df = pd.read_csv(self.log_file)
            if len(df) > resume_step:
                df = df.iloc[:resume_step]
                df.to_csv(self.log_file, index=False)

        self.cumulative_regret = state.get("cumulative_regret", 0)
        self.total_solved = state.get("total_solved", 0)
        self.total_seen = state.get("total_seen", 0)
        print(f"Resuming from step {resume_step} "
              f"(regret={self.cumulative_regret}, solved={self.total_solved}/{self.total_seen})")
        return resume_step

    def save_state(self, step: int):
        """Checkpoint current state for resume."""
        state = {
            "step": step,
            "cumulative_regret": self.cumulative_regret,
            "total_solved": self.total_solved,
            "total_seen": self.total_seen,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def log_step(self, data: dict):
        """Append one row to the results CSV."""
        df = pd.DataFrame([data])
        header = not os.path.exists(self.log_file)
        df.to_csv(self.log_file, mode="a", header=header, index=False)

    def update_regret(self, score: float):
        """Update cumulative regret. score=1 means solved, 0 means regret."""
        self.total_seen += 1
        if score > 0:
            self.total_solved += 1
        else:
            self.cumulative_regret += 1

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        acc = self.total_solved / max(self.total_seen, 1)
        return (
            f"Done. {self.total_seen} problems, {self.total_solved} solved "
            f"({acc:.1%}), cumulative regret={self.cumulative_regret}, "
            f"elapsed={elapsed:.0f}s"
        )
