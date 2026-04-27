"""FGPO Step-1 smoke test: iterative frontier-guided hint refinement.

Loop per problem:
  for round r in 0..N_ROUNDS-1:
    1. Build prompt = system + (current_hint + problem_question)
    2. Sample n_samples completions (batched across problems for GPU efficiency)
    3. Score each via CodeOracle, compute avg reward
    4. Pick worst-k failures, get oracle.get_feedback() error trace
    5. Send full history to Claude -> get next hint

No early stop — always runs N_ROUNDS iterations to see full reward trajectory.

Outputs JSON of per-problem per-round metrics + a summary.
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from cumreg.datasets.base import Problem
from cumreg.oracles.code_oracle import CodeOracle
from fgpo.frontier_client import FrontierClient


CODE_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Solve the given problem by writing clean, correct Python code. "
    "Output ONLY the code in a ```python ... ``` block. "
    "Do NOT include explanations, step-by-step breakdowns, or post-code summaries. "
    "Brief code comments are fine."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PROJECT_DIR, "fgpo/results_step1_smoke"))
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--problems_path", type=str,
                   default=os.path.join(PROJECT_DIR, "data/livecodebench_problems.json"))
    p.add_argument("--n_problems", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--n_rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen_batch_size", type=int, default=50,
                   help="Max prompts per model.generate() call (VRAM cap)")
    p.add_argument("--frontier_model", type=str, default="claude-sonnet-4-6")
    p.add_argument("--frontier_workers", type=int, default=8,
                   help="Parallel Claude API calls within each round")
    p.add_argument("--worst_k", type=int, default=2,
                   help="Lowest-reward samples sent to Claude")
    p.add_argument("--best_k", type=int, default=2,
                   help="Highest-reward samples sent to Claude (shows what's almost working)")
    p.add_argument("--oracle_timeout", type=int, default=60,
                   help="Per-test subprocess timeout (seconds)")
    return p.parse_args()


def load_problems(path):
    with open(path) as f:
        raw = json.load(f)
    return [Problem(**p) for p in raw]


def build_prompt(question: str, hint: str | None) -> list:
    """Returns chat-template messages. Hint (if any) is prepended to the user turn."""
    if hint:
        user_content = f"Hint: {hint.strip()}\n\nProblem:\n{question}"
    else:
        user_content = question
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_batch(model, tokenizer, prompts: list[list[dict]],
                   n_samples: int, max_new_tokens: int, max_seq_length: int,
                   gen_batch_size: int, temperature: float) -> list[list[str]]:
    """Generate n_samples completions for each prompt. Returns list[len(prompts)] of list[n_samples] strings."""
    model.eval()
    tokenizer.padding_side = "left"

    rendered = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompts
    ]
    # Expand: each prompt repeated n_samples times.
    expanded = []
    for r in rendered:
        expanded.extend([r] * n_samples)

    all_responses = []
    for start in range(0, len(expanded), gen_batch_size):
        chunk = expanded[start:start + gen_batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen = out[:, input_length:]
        all_responses.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        del inputs, out, gen
        torch.cuda.empty_cache()

    # Re-shape into [len(prompts)][n_samples]
    grouped = []
    for i in range(len(prompts)):
        grouped.append(all_responses[i * n_samples:(i + 1) * n_samples])
    return grouped


def score_samples(samples_per_problem: list[list[str]],
                  problems: list[Problem],
                  oracle: CodeOracle,
                  worst_k: int, best_k: int) -> list[dict]:
    """Score every sample, then per problem build an iter record with avg_reward
    and a mix of best_k highest-reward + worst_k lowest-reward samples (deduped)."""
    out = []
    for prob, samples in zip(problems, samples_per_problem):
        rewards = [oracle.evaluate(s, prob, fractional=True) for s in samples]
        avg = float(np.mean(rewards))
        # Indices sorted ascending by reward (worst first)
        idx_sorted = sorted(range(len(rewards)), key=lambda i: rewards[i])
        worst_idx = idx_sorted[:worst_k]
        best_idx = list(reversed(idx_sorted))[:best_k]
        # Deduplicate while preserving labels
        chosen = []
        seen = set()
        for i in worst_idx:
            if i in seen:
                continue
            chosen.append((i, "worst"))
            seen.add(i)
        for i in best_idx:
            if i in seen:
                continue
            chosen.append((i, "best"))
            seen.add(i)
        sent = []
        for i, label in chosen:
            sent.append({
                "code": samples[i],
                "reward": float(rewards[i]),
                "label": label,
                "error": oracle.get_feedback(samples[i], prob),
            })
        out.append({
            "avg_reward": avg,
            "rewards": [float(r) for r in rewards],
            "n_pass": int(sum(r >= 1.0 for r in rewards)),
            "shown_samples": sent,
        })
    return out


def call_frontier_parallel(frontier: FrontierClient,
                           problems: list[Problem],
                           histories: list[list[dict]],
                           n_workers: int) -> list[dict]:
    """Call Claude once per problem in parallel. histories[i] is the per-problem coaching history."""
    results = [None] * len(problems)

    def _one(i):
        try:
            return i, frontier.next_hint(problems[i].question, histories[i])
        except Exception as e:
            return i, {"hint": "", "raw": "", "error": str(e),
                       "input_tokens": 0, "output_tokens": 0}

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_one, i) for i in range(len(problems))]
        for fut in as_completed(futures):
            i, r = fut.result()
            results[i] = r
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[smoke] config: n_problems={args.n_problems} n_samples={args.n_samples} "
          f"n_rounds={args.n_rounds} model={args.model_name} frontier={args.frontier_model}",
          flush=True)

    # --- Load problems & sample
    all_problems = load_problems(args.problems_path)
    rng = random.Random(args.seed)
    problems = rng.sample(all_problems, args.n_problems)
    print(f"[smoke] sampled {len(problems)} problems with seed={args.seed}", flush=True)

    # --- Load model
    print(f"[smoke] loading model {args.model_name} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s", flush=True)

    oracle = CodeOracle(timeout=args.oracle_timeout)
    frontier = FrontierClient(model=args.frontier_model)

    # --- State across rounds
    current_hints: list[str | None] = [None] * len(problems)
    histories: list[list[dict]] = [[] for _ in range(len(problems))]
    per_round_avg_rewards: list[list[float]] = []  # [round][problem]
    api_token_log: list[dict] = []

    for r in range(args.n_rounds):
        print(f"\n[smoke] === Round {r}/{args.n_rounds - 1} ===", flush=True)
        t_round = time.time()

        # 1. Build prompts using current_hints
        prompts = [build_prompt(p.question, h) for p, h in zip(problems, current_hints)]

        # 2. Generate n_samples per problem
        t_gen = time.time()
        samples_per_problem = generate_batch(
            model, tokenizer, prompts,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
            gen_batch_size=args.gen_batch_size,
            temperature=args.temperature,
        )
        print(f"[smoke] round {r}: generation took {time.time() - t_gen:.1f}s", flush=True)

        # 3. Score
        t_score = time.time()
        scored = score_samples(samples_per_problem, problems, oracle,
                               worst_k=args.worst_k, best_k=args.best_k)
        print(f"[smoke] round {r}: scoring took {time.time() - t_score:.1f}s", flush=True)

        avg_rs = [s["avg_reward"] for s in scored]
        per_round_avg_rewards.append(avg_rs)
        print(f"[smoke] round {r}: avg_reward mean={np.mean(avg_rs):.3f} "
              f"median={np.median(avg_rs):.3f} max={np.max(avg_rs):.3f} "
              f"#with_any_pass={sum(s['n_pass'] > 0 for s in scored)}/{len(scored)}",
              flush=True)

        # 4. Append this round to per-problem history
        for i, s in enumerate(scored):
            histories[i].append({
                "hint": current_hints[i],
                "avg_reward": s["avg_reward"],
                "n_pass": s["n_pass"],
                "shown_samples": s["shown_samples"],
            })

        # 5. If not final round, call frontier for the next hint
        if r < args.n_rounds - 1:
            t_front = time.time()
            front_results = call_frontier_parallel(
                frontier, problems, histories, args.frontier_workers,
            )
            in_tok = sum(fr.get("input_tokens", 0) for fr in front_results)
            out_tok = sum(fr.get("output_tokens", 0) for fr in front_results)
            n_err = sum(1 for fr in front_results if fr.get("error"))
            print(f"[smoke] round {r}: frontier took {time.time() - t_front:.1f}s "
                  f"(in_tok={in_tok}, out_tok={out_tok}, errors={n_err})", flush=True)
            api_token_log.append({"round": r, "input_tokens": in_tok,
                                  "output_tokens": out_tok, "errors": n_err})
            for i, fr in enumerate(front_results):
                current_hints[i] = fr.get("hint") or current_hints[i]

        print(f"[smoke] round {r}: total {time.time() - t_round:.1f}s", flush=True)

        # Snapshot per round in case of crash
        snapshot = {
            "config": vars(args),
            "round_completed": r,
            "per_round_avg_rewards": per_round_avg_rewards,
            "problem_ids": [p.id for p in problems],
            "current_hints": current_hints,
            "api_token_log": api_token_log,
        }
        with open(os.path.join(args.output_dir, "snapshot.json"), "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    # --- Final outputs
    summary = {
        "config": vars(args),
        "problem_ids": [p.id for p in problems],
        "per_round_avg_rewards": per_round_avg_rewards,
        "round_means": [float(np.mean(rs)) for rs in per_round_avg_rewards],
        "round_medians": [float(np.median(rs)) for rs in per_round_avg_rewards],
        "round_n_with_any_pass": [
            int(sum(per_round_avg_rewards[r][i] > 0 for i in range(len(problems))))
            for r in range(len(per_round_avg_rewards))
        ],
        "api_token_log": api_token_log,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Full per-problem detail (with histories — large)
    with open(os.path.join(args.output_dir, "details.pkl"), "wb") as f:
        pickle.dump({
            "config": vars(args),
            "problem_ids": [p.id for p in problems],
            "histories": histories,
            "per_round_avg_rewards": per_round_avg_rewards,
        }, f)

    print("\n[smoke] === DONE ===", flush=True)
    print(f"[smoke] round means: {summary['round_means']}", flush=True)
    print(f"[smoke] round medians: {summary['round_medians']}", flush=True)
    print(f"[smoke] outputs in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
