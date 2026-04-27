"""Dataset-aware prompt building for single-turn and multi-turn modes.

Supports passing traces (reasoning chains) from previous examples
for Agent Lightning-style in-context learning.
"""

from typing import List, Optional

from .datasets.base import Problem
from .retriever import HistoryEntry


# ── System prompts ──────────────────────────────────────────────────────────

CODE_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Solve the given problem by writing clean, correct Python code. "
    "Wrap your solution in ```python ... ``` blocks."
)

MATH_SYSTEM_PROMPT = (
    "Solve the math problem step-by-step. "
    "End with #### followed by the final number."
)

SYSTEM_PROMPTS = {
    "livecodebench": CODE_SYSTEM_PROMPT,
    "gsm8k": MATH_SYSTEM_PROMPT,
}


# ── Prompt builders ─────────────────────────────────────────────────────────

def build_prompt(
    problem: Problem,
    examples: List[HistoryEntry],
    dataset: str,
    tokenizer=None,
    include_traces: bool = True,
) -> str:
    """Build a single-turn prompt with few-shot examples and traces.

    Args:
        problem: Current problem to solve.
        examples: Retrieved few-shot examples from history.
        dataset: Dataset name (for system prompt selection).
        tokenizer: HF tokenizer with apply_chat_template.
        include_traces: If True, include reasoning traces from examples.
    """
    system_prompt = SYSTEM_PROMPTS.get(dataset, CODE_SYSTEM_PROMPT)
    messages = [{"role": "system", "content": system_prompt}]

    # Few-shot examples as user/assistant turns
    for ex in examples:
        # User turn: the example problem
        messages.append({"role": "user", "content": _format_question(ex.problem, dataset)})
        # Assistant turn: trace + response
        assistant_content = _format_example_response(ex, dataset, include_traces)
        messages.append({"role": "assistant", "content": assistant_content})

    # Current problem
    messages.append({"role": "user", "content": _format_question(problem, dataset)})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # Fallback: ChatML format
    return _messages_to_chatml(messages)


def build_messages(
    problem: Problem,
    examples: List[HistoryEntry],
    dataset: str,
    include_traces: bool = True,
) -> list:
    """Build chat messages list for OpenAI-compatible APIs (e.g. vLLM).

    Returns list of {"role": ..., "content": ...} dicts.
    Same message structure as build_prompt, but without applying a chat template.
    """
    system_prompt = SYSTEM_PROMPTS.get(dataset, CODE_SYSTEM_PROMPT)
    messages = [{"role": "system", "content": system_prompt}]

    for ex in examples:
        messages.append({"role": "user", "content": _format_question(ex.problem, dataset)})
        assistant_content = _format_example_response(ex, dataset, include_traces)
        messages.append({"role": "assistant", "content": assistant_content})

    messages.append({"role": "user", "content": _format_question(problem, dataset)})
    return messages


def build_repair_prompt(
    problem: Problem,
    previous_code: str,
    feedback: str,
    dataset: str,
    tokenizer=None,
) -> str:
    """Build a multi-turn repair prompt with error feedback.

    For multi-turn mode: feeds back the previous attempt + error info.
    """
    system_prompt = SYSTEM_PROMPTS.get(dataset, CODE_SYSTEM_PROMPT)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _format_question(problem, dataset)},
        {"role": "assistant", "content": previous_code},
        {"role": "user", "content": (
            f"Your previous solution was incorrect.\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Please fix your solution."
        )},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    return _messages_to_chatml(messages)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _format_question(problem: Problem, dataset: str) -> str:
    """Format a problem's question text, including starter code if present."""
    question = problem.question
    starter_code = problem.metadata.get("starter_code", "")
    if starter_code:
        question += f"\n\nStarter code:\n```python\n{starter_code}\n```"
    return question


def _format_example_response(entry: HistoryEntry, dataset: str,
                              include_traces: bool) -> str:
    """Format an example's response, optionally including its trace."""
    parts = []
    if include_traces and entry.trace:
        parts.append(entry.trace)
    parts.append(entry.response)
    return "\n\n".join(parts)


def _messages_to_chatml(messages: list) -> str:
    """Fallback ChatML formatting."""
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt
