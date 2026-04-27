"""Agent Lightning rollout for cumulative regret experiments.

Uses @rollout decorator so the solve function is a proper AGL agent,
compatible with Trainer, Runner, and trace collection.

The full reasoning chain (all turns + feedback) is emitted via
agl.emit_object for the algorithm to retrieve and pass to future examples.
"""

import agentlightning as agl

from .config import Config
from .datasets.base import Problem
from .engine import ICLEngine
from .formatting import build_repair_prompt
from .oracles.base import Oracle


def make_rollout(engine: ICLEngine, oracle: Oracle, config: Config,
                 tokenizer, result_store: dict):
    """Create a @rollout function with closed-over engine/oracle/config.

    Args:
        engine: Local HF generation engine.
        oracle: Code/math evaluator.
        config: Experiment config.
        tokenizer: HF tokenizer.
        result_store: Shared dict for passing results to the algorithm.
            Key = problem id, value = {"response", "trace", "score"}.
            Safe because we run with n_runners=1 (sequential).
    """

    @agl.rollout
    def solve(task: dict, prompt_template: agl.PromptTemplate, rollout: agl.Rollout) -> float:
        """Solve a single problem. Receives task dict with 'formatted_prompt' from algorithm.

        prompt_template and rollout are injected by AGL but we build prompts ourselves
        (because chat templates + code blocks can't go through f-string PromptTemplate).

        Returns float reward (0.0 or 1.0), auto-emitted as final reward span.
        """
        problem = Problem(
            id=task["id"],
            question=task["question"],
            ground_truth=task["ground_truth"],
            metadata=task.get("metadata", {}),
        )
        prompt = task["formatted_prompt"]

        max_turns = 1 if config.mode == "single_turn" else config.max_turns
        response = ""
        score = 0.0
        trace_parts = []

        for turn in range(max_turns):
            # ── Generate ────────────────────────────────────────────
            agl.emit_message(f"generate_turn_{turn}: prompt_length={len(prompt)}")

            # Best-of-N on first turn if n_generations > 1
            n_gen = config.n_generations if turn == 0 else 1
            if n_gen > 1:
                candidates = engine.generate_n([prompt], n=n_gen,
                                               temperature=config.cache_temperature)[0]
                # Evaluate all, pick best
                best_response, best_score = candidates[0], 0.0
                for cand in candidates:
                    cand_score = oracle.evaluate(cand, problem)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_response = cand
                response = best_response
                score = best_score
                agl.emit_message(f"best_of_{n_gen}: best_score={score}")
            else:
                responses = engine.generate([prompt])
                response = responses[0]
                score = oracle.evaluate(response, problem)

            trace_parts.append(f"[Turn {turn + 1}]\n{response}")
            agl.emit_message(f"evaluate_turn_{turn}: score={score}")

            if score > 0:
                agl.emit_message(f"Solved on turn {turn + 1}")
                break

            # ── Multi-turn repair ───────────────────────────────────
            if config.mode == "multi_turn" and turn < max_turns - 1:
                feedback = oracle.get_feedback(response, problem)
                agl.emit_message(f"feedback_turn_{turn}: {feedback[:200]}")

                trace_parts.append(f"[Feedback]\n{feedback}")

                prompt = build_repair_prompt(
                    problem=problem,
                    previous_code=response,
                    feedback=feedback,
                    dataset=config.dataset,
                    tokenizer=tokenizer,
                )

        # Full trace for ICL retrieval
        trace = "\n\n".join(trace_parts)

        # Emit structured result (stored as span for auditing)
        agl.emit_object({
            "problem_id": problem.id,
            "final_response": response,
            "trace": trace,
            "score": score,
            "num_turns": len(trace_parts),
        })

        # Write to shared dict for algorithm to read
        result_store[problem.id] = {
            "response": response,
            "trace": trace,
            "score": score,
        }

        return score

    return solve


def make_vllm_rollout(base_url: str, model_name: str, oracle: Oracle,
                      max_tokens: int = 1024, result_store: dict = None):
    """Create a @rollout that calls a vLLM OpenAI-compatible API server.

    Uses chat completions endpoint. vLLM applies the chat template server-side.
    Thread-safe: OpenAI client uses httpx which handles concurrent requests.

    Args:
        base_url: vLLM server URL, e.g. "http://localhost:8000/v1".
        model_name: Model name as registered in vLLM.
        oracle: Code evaluator for reward computation.
        max_tokens: Max tokens to generate.
        result_store: Shared dict for passing results to the algorithm.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    @agl.rollout
    def solve(task: dict, prompt_template: agl.PromptTemplate, rollout: agl.Rollout) -> float:
        problem = Problem(
            id=task["id"],
            question=task["question"],
            ground_truth=task["ground_truth"],
            metadata=task.get("metadata", {}),
        )
        messages = task["messages"]

        agl.emit_message(f"generate: {len(messages)} messages")

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        choice = completion.choices[0]
        text = choice.message.content or ""

        # Debug: log first few completions
        if len(result_store) < 3:
            print(f"[DEBUG vllm] problem={problem.id}")
            print(f"[DEBUG vllm] finish_reason={choice.finish_reason}")
            print(f"[DEBUG vllm] content type={type(choice.message.content)}")
            print(f"[DEBUG vllm] content repr={repr(text[:500])}")
            print(f"[DEBUG vllm] usage={completion.usage}")
            print(f"[DEBUG vllm] messages[0]={messages[0]}")
            if len(messages) > 1:
                print(f"[DEBUG vllm] messages[-1] content[:200]={messages[-1]['content'][:200]}")

        reward = oracle.evaluate(text, problem, fractional=True)
        agl.emit_message(f"evaluate: reward={reward:.4f}")

        agl.emit_object({
            "problem_id": problem.id,
            "final_response": text,
            "reward": reward,
        })

        if result_store is not None:
            result_store[problem.id] = {
                "response": text,
                "reward": reward,
            }

        return reward

    return solve


def make_cache_rollout(engine: ICLEngine, oracle: Oracle, config: Config,
                       result_store: dict):
    """Create a @rollout for cache-based ICL: single greedy generation + fractional eval.

    Matches run_icl.py semantics exactly:
    - Single greedy generation (no best-of-N, no multi-turn)
    - Fractional reward (passed/total test cases)
    - Prompt is pre-built by the algorithm and passed via task["formatted_prompt"]

    Args:
        engine: Local HF generation engine.
        oracle: Code/math evaluator.
        config: Experiment config.
        result_store: Shared dict for passing results to the algorithm.
    """

    @agl.rollout
    def solve(task: dict, prompt_template: agl.PromptTemplate, rollout: agl.Rollout) -> float:
        problem = Problem(
            id=task["id"],
            question=task["question"],
            ground_truth=task["ground_truth"],
            metadata=task.get("metadata", {}),
        )
        prompt = task["formatted_prompt"]

        # Single greedy generation
        agl.emit_message(f"generate: prompt_length={len(prompt)}")
        responses = engine.generate([prompt])
        response = responses[0]

        # Fractional reward evaluation
        reward = oracle.evaluate(response, problem, fractional=True)
        agl.emit_message(f"evaluate: reward={reward:.4f}")

        agl.emit_object({
            "problem_id": problem.id,
            "final_response": response,
            "reward": reward,
        })

        result_store[problem.id] = {
            "response": response,
            "reward": reward,
        }

        return reward

    return solve
