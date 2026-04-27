"""Anthropic Claude wrapper for FGPO frontier-guided hint generation.

Given a problem and a history of (hint, model_samples, rewards, errors),
produces the next hint to try.
"""

import json
import os
import re
import time
from typing import Optional

import anthropic


SYSTEM_PROMPT = """You are coaching a smaller code-generation model on a competitive-programming problem.

Your job: produce a SHORT planning hint (under 300 tokens) that will be prepended to the problem statement before the smaller model writes its Python solution.

You will see:
- The problem statement
- A history of your previous hints and how the smaller model performed under each (avg reward in [0, 1], plus a few failed-sample errors)

Diagnose the failure pattern across multiple samples (not just one error). Identify:
- Which algorithmic insight is missing
- Which edge cases keep getting missed
- Whether the hint you gave last round was too vague, too specific, or off-target

Then produce a better hint. Constraints on the hint:
- Plain prose with at most a one-line pseudocode sketch.
- Do NOT write the full solution code or function bodies.
- Do NOT restate the problem.
- Focus on what the model needs to UNDERSTAND, not what to write.

Output format: a single JSON object with one key:
{"hint": "<your hint here>"}

Output the JSON only. No surrounding prose, no markdown fences."""


class FrontierClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 600,
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ):
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in env.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def next_hint(self, problem_text: str, history: list) -> dict:
        """Produce the next hint given history of prior attempts.

        history: list of dicts, one per past iteration:
          {
            "hint": str | None,                    # hint used this round (None for round 0)
            "avg_reward": float,
            "shown_samples": [                     # mix of best + worst samples
              {"code": str, "reward": float, "error": str, "label": "best"|"worst"}
            ]
          }
        """
        user_msg = self._render_user_message(problem_text, history)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text if resp.content else ""
                hint = self._parse_hint(raw)
                return {
                    "hint": hint,
                    "raw": raw,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                }
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                last_err = e
                wait = 2 ** attempt
                print(f"  [frontier] retry {attempt+1}/{self.max_retries} after {wait}s: {e}", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"Frontier API failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _render_user_message(problem_text: str, history: list) -> str:
        parts = ["# Problem\n\n", problem_text.strip(), "\n\n# Coaching history\n"]
        if not history:
            parts.append("(no prior attempts — produce your first hint based on the problem only)\n")
        else:
            for i, h in enumerate(history):
                parts.append(f"\n## Round {i}\n")
                parts.append(f"Hint used: {h.get('hint') or '(no hint — raw prompt)'}\n")
                parts.append(f"Avg reward (across samples): {h.get('avg_reward', 0.0):.3f}\n")
                fs = h.get("shown_samples", [])
                if fs:
                    parts.append(f"Samples shown ({len(fs)}, mix of best & worst):\n")
                    for j, s in enumerate(fs):
                        code = (s.get("code") or "").strip()
                        if len(code) > 1200:
                            code = code[:1200] + "\n... [truncated]"
                        err = (s.get("error") or "").strip()
                        if len(err) > 600:
                            err = err[:600] + "\n... [truncated]"
                        label = s.get("label", "?")
                        parts.append(f"\n  Sample {j} [{label}, reward={s.get('reward', 0.0):.2f}]:\n")
                        parts.append(f"  Code:\n```\n{code}\n```\n")
                        parts.append(f"  Error/feedback:\n  {err}\n")
        parts.append("\n# Task\n\nProduce the next hint as JSON.")
        return "".join(parts)

    @staticmethod
    def _parse_hint(raw: str) -> str:
        """Extract hint from JSON. Falls back to raw text if JSON parse fails."""
        text = raw.strip()
        # Strip optional code fences.
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "hint" in obj:
                return str(obj["hint"]).strip()
        except json.JSONDecodeError:
            pass
        # Fallback: try to find a "hint" key by regex
        m = re.search(r'"hint"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if m:
            return m.group(1).encode().decode("unicode_escape")
        # Last resort: return the raw text trimmed
        return text[:1500]
