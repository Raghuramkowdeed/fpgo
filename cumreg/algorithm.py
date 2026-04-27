"""Custom Agent Lightning algorithm for ICL cumulative regret.

The algorithm iterates over tasks sequentially, building ICL prompts
from retriever history (with traces from previous rollouts), enqueuing
rollouts one at a time, and collecting results.

This is the "brain" of the experiment — it controls what context each
rollout sees, which is the core of in-context learning.
"""

import time
from typing import Optional, Any, List

import agentlightning as agl
from agentlightning.algorithm import FastAlgorithm
from agentlightning.store import LightningStore
from agentlightning.reward import find_final_reward
from agentlightning.algorithm.utils import with_store

# Dummy prompt template — we pass the real prompt via the task dict
# because PromptTemplate uses f-string which breaks on code with { }
_DUMMY_PROMPT = agl.PromptTemplate(template="unused", engine="f-string")

import csv
import pickle
import numpy as np

from .config import Config
from .datasets.base import Problem
from .experiment import ExperimentManager
from .formatting import build_prompt
from .retriever import ExampleRetriever, HistoryEntry


def make_icl_algorithm(config: Config, tokenizer, retriever: ExampleRetriever,
                       manager: ExperimentManager, result_store: dict):
    """Create a FastAlgorithm for ICL cumulative regret.

    Uses FastAlgorithm so it works with trainer.dev() in Jupyter
    (runs in-process, no subprocess spawning, no CUDA fork issues).
    """

    class ICLCumRegAlgorithm(FastAlgorithm):
        @with_store
        async def run(
            self,
            store: LightningStore,
            train_dataset: Optional[List[Any]] = None,
            val_dataset: Optional[List[Any]] = None,
        ) -> None:
            """ICL cumulative regret algorithm.

            For each task:
            1. Retrieve similar solved examples (with traces) from history
            2. Build few-shot prompt
            3. Enqueue rollout with the formatted prompt
            4. Wait for completion
            5. Collect result, update retriever and regret tracker
            """
            resume_step = manager.get_resume_step()
            num_steps = config.num_steps
            batch_size = config.batch_size
            batch_start_time = time.time()
            batch_solved = 0

            for step, task in enumerate(train_dataset):
                # Skip to resume point
                if step < resume_step:
                    continue

                if num_steps is not None and step >= num_steps:
                    break

                step_start = time.time()

                problem = Problem(
                    id=task["id"],
                    question=task["question"],
                    ground_truth=task["ground_truth"],
                    metadata=task.get("metadata", {}),
                )

                # ── Build ICL prompt with retrieved examples + traces ───
                examples = retriever.get_examples(problem.question)

                prompt = build_prompt(
                    problem=problem,
                    examples=examples,
                    dataset=config.dataset,
                    tokenizer=tokenizer,
                    include_traces=True,
                )

                # ── Enqueue rollout with formatted prompt ───────────────
                task_with_prompt = {**task, "formatted_prompt": prompt}

                # Register resources (prompt_template required by @rollout signature)
                res = await store.add_resources({"prompt_template": _DUMMY_PROMPT})
                rollout = await store.enqueue_rollout(
                    input=task_with_prompt,
                    resources_id=res.resources_id,
                )
                await store.wait_for_rollouts(rollout_ids=[rollout.rollout_id])

                # ── Collect result ──────────────────────────────────────
                # Primary: shared dict (fast, reliable)
                result = result_store.pop(problem.id, None)

                if result is None:
                    # Fallback: query spans from store
                    spans = await store.query_spans(rollout.rollout_id)
                    reward = find_final_reward(spans) or 0.0
                    result = {"response": "", "trace": "", "score": reward}

                score = result["score"]
                response = result["response"]
                trace = result["trace"]

                # ── Update retriever with result + trace ────────────────
                retriever.add(problem, response, score, trace=trace)

                # ── Update regret tracking ──────────────────────────────
                manager.update_regret(score)

                step_time = time.time() - step_start

                # ── Log ─────────────────────────────────────────────────
                manager.log_step({
                    "step": step,
                    "problem_id": problem.id,
                    "score": score,
                    "cumulative_regret": manager.cumulative_regret,
                    "total_solved": manager.total_solved,
                    "total_seen": manager.total_seen,
                    "accuracy": manager.total_solved / manager.total_seen,
                    "mode": config.mode,
                    "num_examples": len(examples),
                    "step_time_s": round(step_time, 1),
                    "response_preview": response[:200],
                })

                # Track batch stats
                if score > 0:
                    batch_solved += 1

                # Progress (per-step)
                acc = manager.total_solved / manager.total_seen
                print(
                    f"  Step {step}: score={score} regret={manager.cumulative_regret} "
                    f"acc={acc:.1%} [{step_time:.1f}s]"
                )

                # Batch summary every batch_size steps
                effective_step = step - resume_step + 1
                if effective_step > 0 and effective_step % batch_size == 0:
                    batch_time = time.time() - batch_start_time
                    batch_acc = batch_solved / batch_size
                    print(
                        f"\n{'='*60}\n"
                        f"Batch [{step - batch_size + 1}-{step}] summary: "
                        f"solved={batch_solved}/{batch_size} ({batch_acc:.1%}) "
                        f"time={batch_time:.1f}s ({batch_time/batch_size:.1f}s/problem)\n"
                        f"Overall: solved={manager.total_solved}/{manager.total_seen} "
                        f"({acc:.1%}) regret={manager.cumulative_regret} "
                        f"eligible_examples={retriever.num_eligible()}\n"
                        f"{'='*60}\n"
                    )
                    batch_start_time = time.time()
                    batch_solved = 0

                # Checkpoint
                if (step + 1) % config.checkpoint_every == 0:
                    manager.save_state(step + 1)
                    print(f"  Checkpointed at step {step + 1}")

            # Final checkpoint
            manager.save_state(step + 1 if 'step' in dir() else resume_step)
            print()
            print(manager.summary())

    return ICLCumRegAlgorithm()


# ── Cache-based ICL algorithm (matches run_icl.py exactly) ─────────────────

def _to_history_entry(record) -> HistoryEntry:
    """Convert a cache record dict to HistoryEntry for prompt building."""
    return HistoryEntry(
        problem=record["problem"],
        response=record["best_response"],
        score=record["best_score"],
        trace="",
        embedding=record["embedding"],
    )


def _retrieve_dynamic_k(query_embedding, history, problem, tokenizer,
                         max_k=3, max_seq_length=8192, max_new_tokens=2048,
                         dataset="livecodebench"):
    """Retrieve up to max_k examples without exceeding max_seq_length.

    Returns (examples: List[HistoryEntry], similarities: List[float]).
    Always includes at least 1 example if history is non-empty.
    Identical to run_icl.py::retrieve_dynamic_k.
    """
    if not history:
        return [], []

    emb_matrix = np.stack([h["embedding"] for h in history])
    sims = emb_matrix @ query_embedding

    k = min(max_k, len(history))
    top_indices = np.argsort(sims)[-k:][::-1]

    max_input_tokens = max_seq_length - max_new_tokens

    for n in range(k, 0, -1):
        indices = top_indices[:n]
        examples = [_to_history_entry(history[i]) for i in indices]
        prompt = build_prompt(problem, examples, dataset=dataset, tokenizer=tokenizer)
        n_tokens = len(tokenizer.encode(prompt))

        if n_tokens <= max_input_tokens:
            return examples, [float(sims[i]) for i in indices]

    idx = top_indices[0]
    examples = [_to_history_entry(history[idx])]
    return examples, [float(sims[idx])]


def make_cache_icl_algorithm(config: Config, tokenizer, cache_records: list,
                              output_dir: str, result_store: dict,
                              use_messages: bool = False):
    """Create a FastAlgorithm that matches run_icl.py batch-by-batch semantics.

    Args:
        config: Experiment config (uses batch_size, k_shots, max_seq_length, max_new_tokens).
        tokenizer: HF tokenizer for token counting.
        cache_records: List of dicts with keys: problem, embedding, best_response, best_score.
        output_dir: Directory for CSV and pickle output.
        result_store: Shared dict populated by rollout with {problem_id: {response, reward}}.
        use_messages: If True, pass chat messages list (for vLLM/OpenAI API) instead of
            formatted prompt string. Default False for backward compatibility.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    class CacheICLAlgorithm(FastAlgorithm):
        @with_store
        async def run(
            self,
            store: LightningStore,
            train_dataset: Optional[List[Any]] = None,
            val_dataset: Optional[List[Any]] = None,
        ) -> None:
            batch_size = config.batch_size
            n_problems = len(cache_records)
            n_batches = (n_problems + batch_size - 1) // batch_size

            csv_path = os.path.join(output_dir, "icl_results.csv")
            details_path = os.path.join(output_dir, "icl_details.pkl")

            # ── Resume support ─────────────────────────────────────────
            resume_step = 0
            detail_entries = []
            cumulative_regret = 0.0
            total_reward = 0.0
            n_perfect = 0
            n_nonzero = 0
            history = []  # list of cache record dicts

            if os.path.exists(details_path) and os.path.exists(csv_path):
                try:
                    with open(details_path, "rb") as f:
                        saved = pickle.load(f)
                    detail_entries = saved["entries"]
                    resume_step = len(detail_entries)

                    if resume_step > 0:
                        completed_batches = resume_step // batch_size
                        history_end = completed_batches * batch_size
                        for rec in cache_records[:history_end]:
                            history.append(rec)

                        resume_step = history_end
                        if len(detail_entries) > resume_step:
                            detail_entries = detail_entries[:resume_step]

                        for d in detail_entries:
                            reward = d["reward"]
                            cumulative_regret += (1.0 - reward)
                            total_reward += reward
                            if reward == 1.0:
                                n_perfect += 1
                            if reward > 0:
                                n_nonzero += 1

                        print(f"RESUMING from step {resume_step} "
                              f"({completed_batches} batches, history={len(history)})")
                except Exception as e:
                    print(f"Warning: could not resume ({e}), starting fresh")
                    resume_step = 0
                    detail_entries = []
                    cumulative_regret = total_reward = 0.0
                    n_perfect = n_nonzero = 0
                    history = []

            # ── Write CSV header ───────────────────────────────────────
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["step", "problem_id", "reward", "cumulative_regret",
                                 "accuracy", "n_examples", "time_s"])

            # Rewrite resumed rows
            _cum_reg = 0.0
            _n_nz = 0
            for d in detail_entries:
                _cum_reg += (1.0 - d["reward"])
                if d["reward"] > 0:
                    _n_nz += 1
                csv_writer.writerow([
                    d["step"], d["problem_id"], f"{d['reward']:.4f}",
                    f"{_cum_reg:.4f}", f"{_n_nz / (d['step'] + 1):.4f}",
                    len(d["context"]), "0.00",
                ])
            csv_file.flush()

            start_batch = resume_step // batch_size
            start_time = time.time()

            print(f"\n{'='*70}")
            print(f"Cache ICL (AGL): {n_problems} problems, {n_batches} batches")
            print(f"k_shots={config.k_shots}, max_seq_length={config.max_seq_length}, "
                  f"max_new_tokens={config.max_new_tokens}")
            print(f"{'='*70}\n")

            # ── Batch loop ─────────────────────────────────────────────
            for batch_idx in range(start_batch, n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_problems)
                batch_recs = cache_records[batch_start:batch_end]
                batch_time = time.time()

                # 1. Build prompts with dynamic k retrieval
                batch_payloads = []  # formatted_prompt or messages per problem
                batch_contexts = []

                for rec in batch_recs:
                    examples, similarities = _retrieve_dynamic_k(
                        query_embedding=rec["embedding"],
                        history=history,
                        problem=rec["problem"],
                        tokenizer=tokenizer,
                        max_k=config.k_shots,
                        max_seq_length=config.max_seq_length,
                        max_new_tokens=config.max_new_tokens,
                        dataset="livecodebench",
                    )

                    if use_messages:
                        from .formatting import build_messages
                        msgs = build_messages(
                            rec["problem"], examples,
                            dataset="livecodebench",
                        )
                        batch_payloads.append(msgs)
                    else:
                        prompt = build_prompt(
                            rec["problem"], examples,
                            dataset="livecodebench",
                            tokenizer=tokenizer,
                        )
                        batch_payloads.append(prompt)

                    ctx = []
                    for j, ex in enumerate(examples):
                        ctx.append({
                            "problem_id": ex.problem.id,
                            "question": ex.problem.question[:200],
                            "response": ex.response,
                            "similarity": similarities[j] if j < len(similarities) else 0.0,
                        })
                    batch_contexts.append(ctx)

                # 2. Enqueue all rollouts for the batch
                rollout_ids = []
                for i, rec in enumerate(batch_recs):
                    p = rec["problem"]
                    task_with_prompt = {
                        "id": p.id,
                        "question": p.question,
                        "ground_truth": p.ground_truth,
                        "metadata": p.metadata,
                    }
                    if use_messages:
                        task_with_prompt["messages"] = batch_payloads[i]
                    else:
                        task_with_prompt["formatted_prompt"] = batch_payloads[i]
                    res = await store.add_resources({"prompt_template": _DUMMY_PROMPT})
                    rollout = await store.enqueue_rollout(
                        input=task_with_prompt,
                        resources_id=res.resources_id,
                    )
                    rollout_ids.append(rollout.rollout_id)

                # 3. Wait for all rollouts in this batch
                await store.wait_for_rollouts(rollout_ids=rollout_ids)

                # 4. Collect results and evaluate
                batch_rewards = []
                for i, rec in enumerate(batch_recs):
                    step = batch_start + i
                    p = rec["problem"]
                    t0 = time.time()

                    result = result_store.pop(p.id, None)
                    if result is None:
                        spans = await store.query_spans(rollout_ids[i])
                        reward = find_final_reward(spans) or 0.0
                        response = ""
                    else:
                        reward = result["reward"]
                        response = result["response"]

                    cumulative_regret += (1.0 - reward)
                    total_reward += reward
                    if reward == 1.0:
                        n_perfect += 1
                    if reward > 0:
                        n_nonzero += 1
                    batch_rewards.append(reward)

                    accuracy = n_nonzero / (step + 1)

                    csv_writer.writerow([
                        step, p.id, f"{reward:.4f}",
                        f"{cumulative_regret:.4f}", f"{accuracy:.4f}",
                        len(batch_contexts[i]), f"{time.time()-t0:.2f}",
                    ])

                    detail_entries.append({
                        "step": step,
                        "problem_id": p.id,
                        "question": p.question,
                        "context": batch_contexts[i],
                        "output": response,
                        "reward": reward,
                        "cache_best_score": rec["best_score"],
                    })

                    ctx_str = f"{len(batch_contexts[i])}-shot" if batch_contexts[i] else "0-shot"
                    print(f"  [{step:4d}] {p.id:30s} reward={reward:.3f} "
                          f"{ctx_str} cache={rec['best_score']:.2f}", flush=True)

                csv_file.flush()

                # 5. After batch: add ALL batch problems to history
                for rec in batch_recs:
                    history.append(rec)

                # 6. Save checkpoint
                with open(details_path, "wb") as f:
                    pickle.dump({"entries": detail_entries}, f)

                # 7. Batch summary
                elapsed = time.time() - batch_time
                total_elapsed = time.time() - start_time
                problems_done = batch_end
                problems_this_session = problems_done - (start_batch * batch_size)
                problems_left = n_problems - problems_done
                rate = problems_this_session / total_elapsed if total_elapsed > 0 else 0
                eta_s = problems_left / rate if rate > 0 else 0

                avg_batch_reward = np.mean(batch_rewards)
                batch_perfect = sum(1 for r in batch_rewards if r == 1.0)
                batch_nonzero = sum(1 for r in batch_rewards if r > 0)

                print(f"\n[Batch {batch_idx+1}/{n_batches}] "
                      f"problems {batch_start}-{batch_end-1} | history={len(history)}")
                print(f"  Batch: avg_reward={avg_batch_reward:.3f} "
                      f"perfect={batch_perfect}/{len(batch_recs)} "
                      f"nonzero={batch_nonzero}/{len(batch_recs)} "
                      f"time={elapsed:.1f}s")
                print(f"  Cumul: regret={cumulative_regret:.1f} "
                      f"avg_reward={total_reward/problems_done:.3f} "
                      f"perfect={n_perfect}/{problems_done} "
                      f"nonzero={n_nonzero}/{problems_done}")
                print(f"  Progress: {problems_done}/{n_problems} "
                      f"rate={rate:.2f} prob/s ETA={eta_s/60:.0f}min\n", flush=True)

            csv_file.close()
            with open(details_path, "wb") as f:
                pickle.dump({"entries": detail_entries}, f)

            # Final summary
            all_rewards = [d["reward"] for d in detail_entries]
            print(f"\n{'='*70}")
            print(f"DONE: {n_problems} problems, regret={cumulative_regret:.2f}, "
                  f"avg_reward={np.mean(all_rewards):.4f}, "
                  f"perfect={n_perfect}/{n_problems}, "
                  f"nonzero={n_nonzero}/{n_problems}")
            print(f"{'='*70}")

    return CacheICLAlgorithm()
