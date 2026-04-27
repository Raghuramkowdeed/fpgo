#!/usr/bin/env python3
"""FGPO-RLOO: RLOO training with frontier-guided hint augmentation.

Pipeline per batch of train problems:
  1. Pre-eval (greedy, raw prompt) on the batch.
  2. For any problem without a cached hint: run Step-1 iterative loop
     (model probe -> oracle -> frontier hint -> repeat). Cache the best hint.
  3. Build RLOO dataset: each problem contributes TWO rows — one with raw
     prompt, one with hint-augmented prompt. Both get num_generations rollouts.
  4. RLOO update.
  5. Post-eval (greedy, raw prompt) on the batch.
  6. Every K batches: eval on held-out test set (raw prompts only).

Train/test split comes from fgpo/data/details_{train,test}.pkl.
Hints persist in fgpo/cache/hints.json (cache-once-forever for v0).
"""

import argparse
import csv
import gc
import json
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import vllm.sampling_params as _vsp
if not hasattr(_vsp, "GuidedDecodingParams"):
    _vsp.GuidedDecodingParams = None
from trl import RLOOTrainer, RLOOConfig

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from cumreg.datasets.base import Problem
from cumreg.oracles.code_oracle import CodeOracle
from fgpo.frontier_client import FrontierClient


_ORACLE_WORKERS = 8  # overwritten from CLI in main()


def parallel_evaluate(oracle, responses, problems):
    """Run oracle.evaluate concurrently (threads release GIL on subprocess.run).
    Returns list[float] of fractional rewards in input order."""
    if _ORACLE_WORKERS <= 1 or len(responses) <= 1:
        return [float(oracle.evaluate(r, p, fractional=True))
                for r, p in zip(responses, problems)]
    with ThreadPoolExecutor(max_workers=_ORACLE_WORKERS) as ex:
        def _eval(pair):
            return float(oracle.evaluate(pair[0], pair[1], fractional=True))
        return list(ex.map(_eval, list(zip(responses, problems))))


def parallel_feedback(oracle, responses, problems, rewards=None):
    """Like parallel_evaluate but for get_feedback. If rewards is given, only
    fetch feedback for samples with reward < 1.0 (passes return None)."""
    n = len(responses)
    if rewards is None:
        rewards = [0.0] * n
    idx = [i for i in range(n) if rewards[i] < 1.0]
    out = [None] * n
    if not idx:
        return out
    if _ORACLE_WORKERS <= 1 or len(idx) <= 1:
        for i in idx:
            out[i] = oracle.get_feedback(responses[i], problems[i])
        return out
    with ThreadPoolExecutor(max_workers=_ORACLE_WORKERS) as ex:
        def _fb(i):
            return i, oracle.get_feedback(responses[i], problems[i])
        for i, fb in ex.map(_fb, idx):
            out[i] = fb
    return out

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
                   default=os.path.join(PROJECT_DIR, "fgpo/results_fgpo_rloo"))
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--train_details", type=str,
                   default=os.path.join(PROJECT_DIR, "fgpo/data/details_train.pkl"))
    p.add_argument("--test_details", type=str,
                   default=os.path.join(PROJECT_DIR, "fgpo/data/details_test.pkl"))
    p.add_argument("--problems_path", type=str,
                   default=os.path.join(PROJECT_DIR, "data/livecodebench_problems.json"))
    p.add_argument("--hint_cache_path", type=str,
                   default=os.path.join(PROJECT_DIR, "fgpo/cache/hints.pkl"))
    p.add_argument("--frontier_model", type=str, default="claude-sonnet-4-6")
    p.add_argument("--reward_threshold", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=20,
                   help="Problems per batch (each contributes 2 RLOO rows: x and x_aug)")
    p.add_argument("--eval_batch_size", type=int, default=20)
    p.add_argument("--num_generations", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=10)
    p.add_argument("--gradient_accumulation_steps", type=int, default=10)
    p.add_argument("--generation_batch_size", type=int, default=200)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n_epochs", type=int, default=1)
    # FGPO-specific
    p.add_argument("--use_frontier_hints", action="store_true",
                   help="If set, run frontier prompt optimization for first "
                        "--frontier_fraction of training problems. "
                        "If unset: pure RLOO baseline on raw x only.")
    p.add_argument("--frontier_fraction", type=float, default=0.5,
                   help="Fraction of train set eligible for frontier hints "
                        "(only matters if --use_frontier_hints).")
    p.add_argument("--fgpo_n_rounds", type=int, default=3)
    p.add_argument("--fgpo_probe_samples", type=int, default=10)
    p.add_argument("--fgpo_stop_reward", type=float, default=0.5)
    p.add_argument("--fgpo_worst_k", type=int, default=2)
    p.add_argument("--fgpo_best_k", type=int, default=2)
    p.add_argument("--oracle_timeout", type=int, default=60)
    p.add_argument("--oracle_workers", type=int, default=8,
                   help="Thread-pool size for parallel oracle.evaluate/get_feedback calls.")
    # Held-out test eval cadence
    p.add_argument("--test_eval_every", type=int, default=5)
    p.add_argument("--test_eval_n_samples", type=int, default=10)
    p.add_argument("--test_eval_n_problems", type=int, default=100,
                   help="Problems sampled for mid-training eval (fast cadence)")
    p.add_argument("--final_test_eval_n_problems", type=int, default=400,
                   help="Problems for end-of-training full eval")
    return p.parse_args()


def load_problems(problems_path):
    with open(problems_path) as f:
        raw = json.load(f)
    return {p["id"]: Problem(**p) for p in raw}


def load_entries(details_path, problems_by_id):
    with open(details_path, "rb") as f:
        details = pickle.load(f)
    entries = []
    for d in details:
        pid = d["problem_id"]
        if pid not in problems_by_id:
            continue
        prob = problems_by_id[pid]
        question = prob.question
        starter_code = prob.metadata.get("starter_code", "")
        if starter_code:
            question += f"\n\nStarter code:\n```python\n{starter_code}\n```"
        entries.append({
            "problem_id": pid,
            "question": question,
            "best_reward": d.get("best_reward", 0.0),
        })
    return entries


def filter_trainable(entries, reward_threshold):
    return [e for e in entries if e["best_reward"] >= reward_threshold]


class HintCache:
    """Persistent JSON: problem_id -> {
        problem_id: str,
        question: str,              # the x itself (for standalone analysis)
        hint: str | None,           # winning hint (None if frontier failed all rounds)

        # --- direct without-hint vs with-hint view (front & center) ---
        baseline_avg_reward: float,         # round 0 (no hint) avg over 10 samples
        baseline_samples: [                 # ALL 10 samples WITHOUT any hint
            {code: str, reward: float, error: str | None}
        ],
        hinted_avg_reward: float | None,    # avg of the round that used the winning hint
        hinted_samples: [                   # ALL 10 samples WITH the winning hint
            {code: str, reward: float, error: str | None}
        ],

        # --- deeper per-round trace (every hint Claude tried) ---
        n_rounds: int,
        rounds: [
            {
                hint: str | None,           # None for round 0
                avg_reward: float,
                n_pass: int,
                samples: [{code, reward, error}],           # all 10 samples
                shown_to_frontier: [{code, reward, label, error}],  # worst_k+best_k subset
            }
        ],
        ts: float,
    }
    has() returns True only when a usable (truthy) hint was obtained;
    the full record is written even when frontier failed (hint=None)."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = {}

    def has(self, pid):
        return pid in self.data and self.data[pid].get("hint")

    def get(self, pid):
        return self.data[pid]["hint"]

    def put(self, pid, question, hint, n_rounds, rounds_detail):
        # Round 0 (always present) is the baseline.
        baseline_avg = rounds_detail[0]["avg_reward"] if rounds_detail else 0.0
        baseline_samples = rounds_detail[0]["samples"] if rounds_detail else []

        # Find the round that used the winning hint.
        hinted_avg = None
        hinted_samples = []
        if hint and rounds_detail:
            for rd in rounds_detail:
                if rd.get("hint") == hint:
                    hinted_avg = rd["avg_reward"]
                    hinted_samples = rd["samples"]
                    break

        self.data[pid] = {
            "problem_id": pid,
            "question": question,
            "hint": hint,
            "baseline_avg_reward": baseline_avg,
            "baseline_samples": baseline_samples,
            "hinted_avg_reward": hinted_avg,
            "hinted_samples": hinted_samples,
            "n_rounds": n_rounds,
            "rounds": rounds_detail,
            "ts": time.time(),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)


def build_prompt(question, hint):
    user = f"Hint: {hint.strip()}\n\nProblem:\n{question}" if hint else question
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def run_step1_loop(problem, model, tokenizer, oracle, frontier,
                   n_rounds, n_probe, stop_reward, worst_k, best_k,
                   max_new_tokens, max_seq_length):
    """Step-1 frontier loop for ONE problem.
    Returns (best_hint, best_avg_reward, n_rounds_run, rounds_detail)
    where rounds_detail is a list of dicts (one per round) with full
    per-sample code+reward, hint used, avg, and the subset shown to Claude.
    """
    model.eval()
    tokenizer.padding_side = "left"

    history = []          # lightweight view passed to FrontierClient
    rounds_detail = []    # full debug record persisted to cache
    current_hint = None
    best_hint = None
    best_avg = -1.0

    for r in range(n_rounds):
        messages = build_prompt(problem.question, current_hint)
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(
            [prompt_str] * n_probe, return_tensors="pt", padding=True, truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=1.0, use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, input_len:]
        samples = tokenizer.batch_decode(gen, skip_special_tokens=True)
        del inputs, out, gen
        torch.cuda.empty_cache()

        rewards = parallel_evaluate(oracle, samples, [problem] * len(samples))
        avg = float(np.mean(rewards))
        n_pass = int(sum(r_ >= 1.0 for r_ in rewards))

        # Track best hint seen so far (only hinted rounds qualify)
        if current_hint and avg > best_avg:
            best_avg = avg
            best_hint = current_hint

        # Pick worst_k + best_k for frontier feedback
        idx_sorted = sorted(range(len(rewards)), key=lambda i: rewards[i])
        seen = set()
        chosen = []
        for i in idx_sorted[:worst_k]:
            if i not in seen: chosen.append((i, "worst")); seen.add(i)
        for i in list(reversed(idx_sorted))[:best_k]:
            if i not in seen: chosen.append((i, "best")); seen.add(i)
        # Feedback for ALL samples once, pulled in parallel (None for passes).
        feedbacks = parallel_feedback(oracle, samples, [problem] * len(samples), rewards)
        shown = [{"code": samples[i], "reward": rewards[i], "label": label,
                  "error": feedbacks[i] if feedbacks[i] is not None
                           else oracle.get_feedback(samples[i], problem)}
                 for i, label in chosen]
        history.append({"hint": current_hint, "avg_reward": avg, "shown_samples": shown})

        # Full per-round record for cache (ALL samples, not just shown subset).
        all_samples = [
            {"code": samples[i], "reward": rewards[i], "error": feedbacks[i]}
            for i in range(len(samples))
        ]
        rounds_detail.append({
            "hint": current_hint,
            "avg_reward": avg,
            "n_pass": n_pass,
            "samples": all_samples,
            "shown_to_frontier": shown,
        })

        if avg >= stop_reward and current_hint:
            break

        # Ask frontier for next hint
        if r < n_rounds - 1:
            try:
                res = frontier.next_hint(problem.question, history)
                current_hint = res["hint"] or current_hint
            except Exception as e:
                print(f"    [fgpo] frontier error on {problem.id}: {e}", flush=True)

    # Fallback: if no hint ever improved on r0, still return the last one (if any)
    if best_hint is None and current_hint:
        best_hint = current_hint
        best_avg = history[-1]["avg_reward"]

    return best_hint, best_avg, len(history), rounds_detail


def build_fgpo_batch_dataset(entries, hint_cache, eligible_ids, pad_to=1):
    """Each entry -> raw row; plus hint-augmented row if problem is in
    eligible_ids AND a hint is cached. If eligible_ids is empty, baseline mode:
    only raw rows. Pads total row count to a multiple of `pad_to` by duplicating
    raw rows (round-robin) so TRL's effective-batch divisibility holds even when
    only some problems in the batch have hints."""
    prompts, problem_ids = [], []
    raw_rows = []  # (prompt, pid) — pool for padding
    for e in entries:
        raw = build_prompt(e["question"], None)
        prompts.append(raw); problem_ids.append(e["problem_id"])
        raw_rows.append((raw, e["problem_id"]))
        if e["problem_id"] in eligible_ids and hint_cache.has(e["problem_id"]):
            prompts.append(build_prompt(e["question"], hint_cache.get(e["problem_id"])))
            problem_ids.append(e["problem_id"])
    if pad_to > 1 and raw_rows:
        i = 0
        while len(prompts) % pad_to != 0:
            p, pid = raw_rows[i % len(raw_rows)]
            prompts.append(p); problem_ids.append(pid)
            i += 1
    return Dataset.from_dict({"prompt": prompts, "problem_id": problem_ids})


def make_reward_fn(oracle, problems_by_id):
    def reward_fn(prompts, completions, **kwargs):
        problem_ids = kwargs["problem_id"]
        texts, probs = [], []
        for completion, pid in zip(completions, problem_ids):
            if isinstance(completion, list):
                text = completion[0]["content"] if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = completion
            texts.append(text)
            probs.append(problems_by_id[pid])
        return parallel_evaluate(oracle, texts, probs)
    return reward_fn


def evaluate_greedy(model, tokenizer, problems_by_id, entries, oracle,
                    batch_size, max_new_tokens, max_seq_length):
    model.eval()
    tokenizer.padding_side = "left"
    rewards = []
    prompts = [
        tokenizer.apply_chat_template(build_prompt(e["question"], None),
                                       tokenize=False, add_generation_prompt=True)
        for e in entries
    ]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        be = entries[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, input_len:]
        responses = tokenizer.batch_decode(gen, skip_special_tokens=True)
        del inputs, out, gen
        torch.cuda.empty_cache()
        probs = [problems_by_id[e["problem_id"]] for e in be]
        rewards.extend(parallel_evaluate(oracle, responses, probs))
    return rewards


def evaluate_pass_at_n(model, tokenizer, problems_by_id, entries, oracle,
                       n_samples, batch_size, max_new_tokens, max_seq_length, max_gen_batch):
    model.eval()
    tokenizer.padding_side = "left"
    results = []
    for start in range(0, len(entries), batch_size):
        be = entries[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(build_prompt(e["question"], None),
                                          tokenize=False, add_generation_prompt=True)
            for e in be
        ]
        expanded = []
        for p in prompts:
            expanded.extend([p] * n_samples)
        all_responses = []
        for gs in range(0, len(expanded), max_gen_batch):
            chunk = expanded[gs:gs + max_gen_batch]
            inputs = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=max_seq_length,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=1.0, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                )
            gen = out[:, input_len:]
            all_responses.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            del inputs, out, gen
            gc.collect()
            torch.cuda.empty_cache()
        # Parallel-evaluate all (n_entries × n_samples) responses at once.
        flat_probs = []
        for e in be:
            flat_probs.extend([problems_by_id[e["problem_id"]]] * n_samples)
        flat_rews = parallel_evaluate(oracle, all_responses, flat_probs)
        for i, e in enumerate(be):
            rews = flat_rews[i * n_samples:(i + 1) * n_samples]
            pass_1 = float(np.mean([r >= 1.0 for r in rews]))
            pass_n = float(any(r >= 1.0 for r in rews))
            avg = float(np.mean(rews))
            results.append((e["problem_id"], pass_1, pass_n, avg, rews))
    return results


def load_state(output_dir):
    p = os.path.join(output_dir, "state.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None


def save_state(output_dir, state):
    with open(os.path.join(output_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    global _ORACLE_WORKERS
    _ORACLE_WORKERS = max(1, int(args.oracle_workers))

    print(f"[fgpo] batch_size={args.batch_size} num_gen={args.num_generations} "
          f"lr={args.learning_rate} beta={args.beta} n_epochs={args.n_epochs}", flush=True)

    if args.resume:
        state = load_state(args.output_dir) or {"epoch": 0, "batch": 0}
    else:
        existing = load_state(args.output_dir)
        if existing and existing.get("epoch", 0) > 0:
            print(f"ERROR: state.json exists. Use --resume or delete {args.output_dir}.")
            sys.exit(1)
        state = {"epoch": 0, "batch": 0}

    problems_by_id = load_problems(args.problems_path)
    train_entries = load_entries(args.train_details, problems_by_id)
    test_entries = load_entries(args.test_details, problems_by_id)
    train_entries = filter_trainable(train_entries, args.reward_threshold)
    print(f"[fgpo] train={len(train_entries)} test={len(test_entries)}", flush=True)

    n_batches = (len(train_entries) + args.batch_size - 1) // args.batch_size
    if args.max_batches is not None:
        n_batches = min(n_batches, args.max_batches)

    # Frontier-eligible set: only problems in the first --frontier_fraction of the
    # train split get hint-optimized (only when --use_frontier_hints is on).
    if args.use_frontier_hints:
        cutoff = int(len(train_entries) * args.frontier_fraction)
        frontier_eligible_ids = {e["problem_id"] for e in train_entries[:cutoff]}
        print(f"[fgpo] frontier hints ON for first {len(frontier_eligible_ids)} "
              f"/ {len(train_entries)} train problems", flush=True)
    else:
        frontier_eligible_ids = set()
        print("[fgpo] frontier hints OFF (baseline) — raw x only, no x_aug rows", flush=True)

    # Divisibility checks. Baseline: 1 row/problem. Frontier: up to 2 rows/problem.
    assert args.generation_batch_size % args.num_generations == 0
    effective_batch_ = args.per_device_train_batch_size * args.gradient_accumulation_steps
    assert (args.batch_size * args.num_generations) % effective_batch_ == 0, (
        "per_device * gas must divide (batch_size * num_gen)")
    if args.use_frontier_hints:
        assert (args.batch_size * 2 * args.num_generations) % effective_batch_ == 0, (
            "per_device * gas must divide (batch_size * 2 * num_gen) when frontier hints on")

    print(f"[fgpo] loading {args.model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    latest_adapter = os.path.join(args.output_dir, "lora_latest")
    if args.resume and os.path.exists(latest_adapter):
        print(f"[fgpo] loading LoRA from {latest_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, latest_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules="all-linear",
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    oracle = CodeOracle(timeout=args.oracle_timeout)
    frontier = FrontierClient(model=args.frontier_model) if args.use_frontier_hints else None
    hint_cache = HintCache(args.hint_cache_path)
    reward_fn = make_reward_fn(oracle, problems_by_id)

    # Test subset (fixed across the run for comparability)
    rng = random.Random(args.seed)
    test_subset = rng.sample(test_entries, min(args.test_eval_n_problems, len(test_entries)))

    # Optimizer + scheduler
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    est_samples_per_batch = args.batch_size * 2 * args.num_generations  # worst case (all augmented)
    updates_per_batch = max(1, est_samples_per_batch // effective_batch)
    total_training_steps = n_batches * args.n_epochs * updates_per_batch

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_training_steps * 0.05),
        num_training_steps=total_training_steps,
    )

    optim_path = os.path.join(args.output_dir, "optim_state.pt")
    if args.resume and os.path.exists(optim_path):
        ckpt = torch.load(optim_path, map_location="cpu")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"[fgpo] restored optimizer/scheduler (lr={scheduler.get_last_lr()[0]:.2e})", flush=True)

    # CSV logs
    train_log_path = os.path.join(args.output_dir, "training_log.csv")
    test_log_path = os.path.join(args.output_dir, "test_eval.csv")
    train_log_exists = os.path.exists(train_log_path) and os.path.getsize(train_log_path) > 0
    test_log_exists = os.path.exists(test_log_path) and os.path.getsize(test_log_path) > 0
    train_file = open(train_log_path, "a", newline="")
    train_writer = csv.writer(train_file)
    if not train_log_exists:
        train_writer.writerow(["epoch", "batch_idx", "train_loss", "train_reward",
                               "pre_avg", "post_avg", "delta", "n_aug",
                               "step1_time_s", "train_time_s"])
    test_file = open(test_log_path, "a", newline="")
    test_writer = csv.writer(test_file)
    if not test_log_exists:
        test_writer.writerow(["epoch", "batch_idx", "test_greedy_avg",
                              "test_pass_at_1", f"test_pass_at_{args.test_eval_n_samples}",
                              "test_avg_reward", "eval_time_s"])

    # === Epoch loop ===
    start_epoch = state["epoch"]
    start_batch = state["batch"] if args.resume else 0

    for epoch in range(start_epoch, args.n_epochs):
        current_epoch = epoch + 1
        print(f"\n=== Epoch {current_epoch}/{args.n_epochs} ===", flush=True)
        epoch_t0 = time.time()

        batch_range_start = start_batch if epoch == start_epoch else 0
        for batch_idx in range(batch_range_start, n_batches):
            batch = train_entries[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
            print(f"\n[E{current_epoch} B{batch_idx+1}/{n_batches}] {len(batch)} problems", flush=True)

            # 0. Pre-eval (raw greedy)
            gc.collect()
            torch.cuda.empty_cache()
            pre_rewards = evaluate_greedy(
                model, tokenizer, problems_by_id, batch, oracle,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens, max_seq_length=args.max_seq_length,
            )
            pre_avg = float(np.mean(pre_rewards))
            print(f"  Pre (raw): {pre_avg:.4f}", flush=True)

            # 1. Step-1 loop for frontier-eligible problems (skips ones whose
            #    cache entry already has a non-empty hint). Debug trace is
            #    persisted every time, even when frontier fails to produce a hint.
            t_step1 = time.time()
            n_new = 0
            if args.use_frontier_hints:
                for e in batch:
                    pid = e["problem_id"]
                    if pid not in frontier_eligible_ids:
                        continue
                    if hint_cache.has(pid):
                        continue
                    prob = problems_by_id[pid]
                    hint, avg, nr, rounds_detail = run_step1_loop(
                        prob, model, tokenizer, oracle, frontier,
                        n_rounds=args.fgpo_n_rounds, n_probe=args.fgpo_probe_samples,
                        stop_reward=args.fgpo_stop_reward,
                        worst_k=args.fgpo_worst_k, best_k=args.fgpo_best_k,
                        max_new_tokens=args.max_new_tokens, max_seq_length=args.max_seq_length,
                    )
                    hint_cache.put(pid, e["question"], hint, nr, rounds_detail)
                    if hint:
                        n_new += 1
            n_aug = sum(1 for e in batch
                        if e["problem_id"] in frontier_eligible_ids
                        and hint_cache.has(e["problem_id"]))
            step1_time = time.time() - t_step1
            print(f"  Step-1: +{n_new} new / {n_aug} augmented in {step1_time:.1f}s", flush=True)

            # 2. RLOO training
            torch.cuda.empty_cache()
            # pad_to: smallest row-count divisor such that rows*num_gen is a
            # multiple of per_device*gas. Protects against odd n_aug values from
            # partial step-1 failures.
            from math import gcd
            eff = args.per_device_train_batch_size * args.gradient_accumulation_steps
            pad_to = eff // gcd(args.num_generations, eff)
            batch_dataset = build_fgpo_batch_dataset(
                batch, hint_cache, frontier_eligible_ids, pad_to=pad_to)
            n_rows = len(batch_dataset)
            total_samples = n_rows * args.num_generations
            assert total_samples % eff == 0, (
                f"batch divisibility broken: rows={n_rows} num_gen={args.num_generations} "
                f"per_device*gas={eff}")
            n_optim = max(1, total_samples // eff)
            print(f"  RLOO: {n_rows} rows × {args.num_generations} gens = {total_samples} samples, "
                  f"{n_optim} optim steps (pad_to={pad_to})", flush=True)

            rloo_config = RLOOConfig(
                output_dir=os.path.join(args.output_dir, "_tmp_batch"),
                seed=args.seed + batch_idx + 1000 * epoch,
                learning_rate=args.learning_rate,
                num_train_epochs=1,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                generation_batch_size=args.generation_batch_size,
                max_grad_norm=1.0, temperature=1.0,
                max_prompt_length=args.max_seq_length,
                max_completion_length=args.max_new_tokens,
                num_generations=args.num_generations,
                beta=args.beta, num_iterations=1,
                bf16=True, fp16=False,
                logging_steps=1, save_steps=999999,
                report_to="none", remove_unused_columns=False,
            )
            trainer = RLOOTrainer(
                model=model,
                reward_funcs=reward_fn,
                args=rloo_config,
                train_dataset=batch_dataset,
                processing_class=tokenizer,
                optimizers=(optimizer, scheduler),
            )
            t_tr = time.time()
            train_result = trainer.train()
            train_time = time.time() - t_tr
            train_loss = train_result.training_loss or 0.0
            train_reward = 0.0
            if trainer.state.log_history:
                train_reward = trainer.state.log_history[-1].get("rewards", 0.0)
            model = trainer.model
            print(f"  Train: loss={train_loss:.6f} reward={train_reward:.4f} time={train_time:.1f}s", flush=True)

            # Free trainer state before we allocate more during eval.
            del trainer, rloo_config, batch_dataset
            gc.collect()
            torch.cuda.empty_cache()

            # 3. Post-eval
            torch.cuda.empty_cache()
            post_rewards = evaluate_greedy(
                model, tokenizer, problems_by_id, batch, oracle,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens, max_seq_length=args.max_seq_length,
            )
            post_avg = float(np.mean(post_rewards))
            print(f"  Post (raw): {post_avg:.4f} delta={post_avg - pre_avg:+.4f}", flush=True)

            train_writer.writerow([
                current_epoch, batch_idx, f"{train_loss:.6f}", f"{train_reward:.4f}",
                f"{pre_avg:.4f}", f"{post_avg:.4f}", f"{post_avg - pre_avg:+.4f}",
                n_aug, f"{step1_time:.1f}", f"{train_time:.1f}",
            ])
            train_file.flush()

            # Save checkpoint every batch
            model.save_pretrained(latest_adapter)
            torch.save({
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, optim_path)
            state = {"epoch": epoch, "batch": batch_idx + 1}
            save_state(args.output_dir, state)

            # 4. Held-out test eval (greedy only — pass@n dropped for memory/speed)
            if (batch_idx + 1) % args.test_eval_every == 0 or batch_idx == n_batches - 1:
                gc.collect()
                torch.cuda.empty_cache()
                t_te = time.time()
                g = evaluate_greedy(
                    model, tokenizer, problems_by_id, test_subset, oracle,
                    batch_size=args.eval_batch_size,
                    max_new_tokens=args.max_new_tokens, max_seq_length=args.max_seq_length,
                )
                te_time = time.time() - t_te
                g_avg = float(np.mean(g))
                print(f"  Test: greedy={g_avg:.4f} time={te_time:.1f}s", flush=True)
                # CSV has 7 columns (pass@1, pass@n, avg_reward slots kept blank for
                # compatibility with rows written by earlier runs)
                test_writer.writerow([
                    current_epoch, batch_idx, f"{g_avg:.4f}",
                    "", "", "", f"{te_time:.1f}",
                ])
                test_file.flush()
                del g
                gc.collect()

        # End of epoch
        model.save_pretrained(os.path.join(args.output_dir, f"lora_epoch_{current_epoch}"))
        state = {"epoch": current_epoch, "batch": 0}
        save_state(args.output_dir, state)
        print(f"\n=== Epoch {current_epoch} done in {time.time() - epoch_t0:.1f}s ===", flush=True)

    train_file.close()
    test_file.close()

    # === Final full eval on held-out test set (greedy only) ===
    n_final = min(args.final_test_eval_n_problems, len(test_entries))
    final_subset = test_entries[:n_final]
    print(f"\n=== Final eval on first {n_final} test problems (greedy only) ===", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    t_f = time.time()
    g = evaluate_greedy(
        model, tokenizer, problems_by_id, final_subset, oracle,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens, max_seq_length=args.max_seq_length,
    )
    final_time = time.time() - t_f
    g_avg = float(np.mean(g))
    print(f"Final: greedy={g_avg:.4f} time={final_time:.1f}s", flush=True)

    final_path = os.path.join(args.output_dir, "final_eval.json")
    with open(final_path, "w") as f:
        json.dump({
            "n_problems": n_final,
            "greedy_avg": g_avg,
            "time_s": final_time,
            "per_problem": [
                {"problem_id": final_subset[i]["problem_id"],
                 "greedy_reward": float(g[i])}
                for i in range(n_final)
            ],
        }, f, indent=2)
    print(f"Saved final eval to {final_path}", flush=True)


if __name__ == "__main__":
    main()
