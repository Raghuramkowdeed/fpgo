#!/usr/bin/env python3
"""Online ICL-conditioned Self-Distillation under 1-oracle-per-q on LCB.

Combines:
  - DistilTrainer + DistilConfig from the public idanshen/Self-Distillation repo
    (loss = forward KL between student and teacher on shared completion tokens)
  - LiveCache memory of past (question, response, reward) for ICL retrieval
  - Qwen3-Embedding-0.6B for semantic retrieval over problem statements
  - Strict 1 oracle call per question — the single greedy rollout serves
    triple duty: pass@1 measurement, SDFT data candidate, memory anchor.

Per problem:
  1. Embed question with Qwen3-Embedding-0.6B
  2. Retrieve k nearest past entries from memory (filtered by min_reward)
  3. Build greedy prompt: system + k ICL examples + current question
  4. Generate ONE greedy rollout (do_sample=False)
  5. Oracle the rollout (1 call) → reward r
       - r is the pass@1 measurement (cumulative regret += 1 - r)
       - if r >= reward_threshold → add to SDFT buffer for end-of-batch update
       - append (q, embedding, rollout, r) to memory regardless of r
  6. End of batch (B problems): one DistilTrainer.train() step on accepted pairs

1-oracle/q baselines we compare against (see ../results/icl_sdft/):
  - Base 0-shot       : Qwen2.5-Coder-7B-Instruct, no training, no ICL
  - ICL k=3 (stream)  : same model + kNN ICL anchors from a growing cache
  - RLOO + Nemotron-RM: RM scores 200 rollouts/q, oracle only for eval
  - ICL+SDFT (this)   : 1 oracle/q, no RM, growing ICL memory + SDFT update

Usage:
    python run_icl_sdft_online1q.py \\
        --output_dir ../results/icl_sdft \\
        --batch_size 10 --batch_window 9
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root = parent of this file's directory (icl_sdft/ → fpgo/) — so
# `from cumreg...` resolves to the sibling cumreg/ package.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Self-Distillation repo: https://github.com/Continual-Intelligence/Self-Distillation
# Configure via SELF_DISTILLATION_PATH env var, or place the repo alongside
# this one (../Self-Distillation).
_SD_PATH = os.environ.get(
    "SELF_DISTILLATION_PATH",
    os.path.join(os.path.dirname(PROJECT_DIR), "Self-Distillation"),
)
if not os.path.isdir(_SD_PATH):
    raise RuntimeError(
        f"Self-Distillation not found at {_SD_PATH}. "
        f"Clone https://github.com/Continual-Intelligence/Self-Distillation "
        f"and either place it alongside this repo or set "
        f"SELF_DISTILLATION_PATH."
    )
sys.path.insert(0, _SD_PATH)

from cumreg.datasets.base import Problem
from cumreg.oracles.code_oracle import CodeOracle
from distil_config import DistilConfig
from distil_trainer import DistilTrainer


CODE_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Solve the given problem by writing clean, correct Python code. "
    "Output ONLY the code in a ```python ... ``` block. "
    "Do NOT include explanations, step-by-step breakdowns, or post-code summaries. "
    "Brief code comments are fine."
)


# ── Qwen3 Embedding wrapper ───────────────────────────────────────────────

class Qwen3Embedder:
    """Wrap Qwen/Qwen3-Embedding-0.6B for cosine-similarity retrieval.

    Uses the official last-token + L2-normalize protocol from the Qwen3 model
    card. Loaded in bf16 on the same device as the policy.
    """

    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device="cuda",
                 max_length=512):
        from transformers import AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            h = self.model(**enc).last_hidden_state
            # last-token pooling (left padding means the last non-pad token is at -1)
            emb = h[:, -1]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0)


# ── Live cache (grows during the run) ─────────────────────────────────────

class LiveCache:
    """In-memory store of (problem, embedding, response, reward). kNN retrieval."""

    def __init__(self):
        self.entries = []  # list of dicts

    def add(self, problem, embedding, response, reward):
        self.entries.append({
            "problem": problem,
            "embedding": embedding,
            "response": response,
            "reward": reward,
        })

    def retrieve(self, query_embedding, k=3, min_reward=0.8):
        pool = [e for e in self.entries if e["reward"] >= min_reward]
        if not pool:
            return []
        emb = np.stack([e["embedding"] for e in pool])
        sims = emb @ query_embedding
        top_k = min(k, len(pool))
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [pool[i] for i in top_idx]

    def __len__(self):
        return len(self.entries)

    def save(self, path):
        data = [
            {
                "problem_id": e["problem"].id,
                "question": e["problem"].question,
                "ground_truth": e["problem"].ground_truth,
                "metadata": e["problem"].metadata,
                "embedding": e["embedding"],
                "response": e["response"],
                "reward": e["reward"],
            }
            for e in self.entries
        ]
        with open(path, "wb") as f:
            pickle.dump(data, f)


# ── Prompt builders (ICL-conditioned) ─────────────────────────────────────

def _question_text(problem):
    q = problem.question
    starter = (problem.metadata or {}).get("starter_code", "")
    if starter:
        q += f"\n\nStarter code:\n```python\n{starter}\n```"
    return q


def build_greedy_messages(problem, examples):
    """Messages for the single greedy rollout (no reward tag)."""
    msgs = [{"role": "system", "content": CODE_SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user", "content": _question_text(ex["problem"])})
        msgs.append({"role": "assistant", "content": ex["response"]})
    msgs.append({"role": "user", "content": _question_text(problem)})
    return msgs


def build_student_messages(problem, examples):
    """Student prompt for DistilTrainer (matches build_greedy_messages)."""
    return build_greedy_messages(problem, examples)


def build_teacher_messages(problem, examples, demo_response):
    """Teacher prompt: same ICL + question → demo → re-ask question."""
    msgs = [{"role": "system", "content": CODE_SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user", "content": _question_text(ex["problem"])})
        msgs.append({"role": "assistant", "content": ex["response"]})
    q = _question_text(problem)
    msgs.append({"role": "user", "content": q})
    msgs.append({"role": "assistant", "content": demo_response})
    msgs.append({"role": "user", "content": q})
    return msgs


# ── Greedy generation ────────────────────────────────────────────────────

def greedy_generate(model, tokenizer, messages_list, max_new_tokens, max_seq_length):
    """Deterministic batched generation. Returns list[str] (decoded completions)."""
    model.eval()
    tokenizer.padding_side = "left"
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_seq_length,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_tokens = out[:, input_len:]
    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


# ── Data loading ──────────────────────────────────────────────────────────

def load_problems(problems_path):
    with open(problems_path) as f:
        raw = json.load(f)
    return [Problem(**p) for p in raw]


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Online ICL+SDFT under 1-oracle/q")
    # Paths
    ap.add_argument("--output_dir", default=os.path.join(PROJECT_DIR, "results_icl_sdft_1q"))
    ap.add_argument("--problems_path",
                    default=os.path.join(PROJECT_DIR, "data/livecodebench_problems.json"))
    # Models
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--embedder_name", default="Qwen/Qwen3-Embedding-0.6B")
    # Data / batching
    ap.add_argument("--max_problems", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=10,
                    help="Problems per outer step (ICL retrieve + greedy + SDFT)")
    ap.add_argument("--seed", type=int, default=42)
    # ICL retrieval
    ap.add_argument("--knn_k", type=int, default=3)
    ap.add_argument("--icl_min_reward", type=float, default=0.8,
                    help="A past entry qualifies as ICL anchor only if reward >= this")
    # SDFT data gate
    ap.add_argument("--reward_threshold", type=float, default=0.5,
                    help="Push (student, teacher) into SDFT buffer only if rollout reward >= this")
    # SDFT trainer
    ap.add_argument("--learning_rate", type=float, default=5e-5,
                    help="Matches Self-Distillation README recipe")
    ap.add_argument("--num_train_epochs", type=int, default=2,
                    help="Passes over the windowed SDFT buffer (matches repo recipe)")
    ap.add_argument("--batch_window", type=int, default=5,
                    help="Sliding window: train on current + previous N batches' accepted pairs")
    ap.add_argument("--sdft_micro_batch", type=int, default=2,
                    help="Per-device micro-batch for DistilTrainer (caps GPU memory). "
                         "Effective batch = full window via grad accumulation.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--max_completion_length", type=int, default=2048)
    # Oracle
    ap.add_argument("--oracle_timeout", type=int, default=10)
    # Checkpointing
    ap.add_argument("--checkpoint_every", type=int, default=5,
                    help="Save LoRA + cache every N batches")
    # Held-out learning curve
    ap.add_argument("--heldout_size", type=int, default=10,
                    help="Reserve N problems for held-out eval (not trained on)")
    ap.add_argument("--heldout_eval_every", type=int, default=5,
                    help="Re-eval the held-out set every N batches")
    return ap.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*70}\nOnline ICL + SDFT (1 oracle / q)\n{'='*70}")
    print(f"model       : {args.model_name}")
    print(f"embedder    : {args.embedder_name}")
    print(f"batch_size  : {args.batch_size}")
    print(f"knn_k       : {args.knn_k}   icl_min_reward: {args.icl_min_reward}")
    print(f"sdft thresh : {args.reward_threshold}")
    print(f"lr          : {args.learning_rate}")
    print(f"out         : {args.output_dir}")

    # ── Problems ─────────────────────────────────────────────────────
    all_problems = load_problems(args.problems_path)
    if args.max_problems:
        all_problems = all_problems[:args.max_problems]
    problems_by_id = {p.id: p for p in all_problems}
    print(f"problems    : {len(all_problems)}")

    # ── Resume detection ─────────────────────────────────────────────
    state_path     = os.path.join(args.output_dir, "state.json")
    lora_dir       = os.path.join(args.output_dir, "lora_adapter")
    cache_pkl_path = os.path.join(args.output_dir, "cache.pkl")
    pairhist_path  = os.path.join(args.output_dir, "pair_history.pkl")
    details_path   = os.path.join(args.output_dir, "details.pkl")
    can_resume = all(os.path.exists(p) for p in
                     [state_path, os.path.join(lora_dir, "adapter_config.json"),
                      cache_pkl_path, pairhist_path, details_path])
    if can_resume:
        with open(state_path) as f:
            saved_state = json.load(f)
        print(f"\nRESUMING from batch {saved_state['next_batch']} "
              f"(last_completed={saved_state['last_completed_batch']})", flush=True)
    else:
        saved_state = None
        print("\nStarting FRESH", flush=True)

    # ── Models (student + teacher in bf16, no quantization) ──────────
    print(f"\nLoading student/teacher (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if saved_state is not None:
        from peft import PeftModel
        print(f"  loading LoRA from {lora_dir}", flush=True)
        model = PeftModel.from_pretrained(model, lora_dir, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules="all-linear",
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Embedder ─────────────────────────────────────────────────────
    print(f"\nLoading embedder: {args.embedder_name} ...", flush=True)
    embedder = Qwen3Embedder(args.embedder_name, device="cuda")

    # ── Oracle ───────────────────────────────────────────────────────
    oracle = CodeOracle(timeout=args.oracle_timeout)

    # ── VRAM report ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM allocated after loads: "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── State ────────────────────────────────────────────────────────
    cache = LiveCache()
    cumulative_regret = 0.0
    detail_entries = []
    global_step = 0
    # Rolling history of accepted SDFT pairs across batches.
    # Each entry: {"prompt": messages, "teacher_prompt": messages, "batch_idx": int}
    pair_history = []
    start_batch = 0
    if saved_state is not None:
        # Rehydrate cache, pair_history, detail_entries from saved checkpoint
        cache_data = pickle.load(open(cache_pkl_path, "rb"))
        for d in cache_data:
            if d["problem_id"] in problems_by_id:
                cache.entries.append({
                    "problem": problems_by_id[d["problem_id"]],
                    "embedding": d["embedding"],
                    "response": d["response"],
                    "reward": d["reward"],
                })
        pair_history = pickle.load(open(pairhist_path, "rb"))
        detail_entries = pickle.load(open(details_path, "rb"))
        cumulative_regret = saved_state["cumulative_regret"]
        global_step = saved_state["global_step"]
        start_batch = saved_state["next_batch"]
        print(f"  rehydrated: cache={len(cache)} pair_history={len(pair_history)} "
              f"details={len(detail_entries)} cumreg={cumulative_regret:.2f}", flush=True)

    # ── CSVs ─────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "per_problem.csv")
    batch_csv_path = os.path.join(args.output_dir, "batch_metrics.csv")
    csv_mode = "a" if saved_state is not None else "w"
    csv_f = open(csv_path, csv_mode, newline="")
    csv_w = csv.writer(csv_f)
    bf = open(batch_csv_path, csv_mode, newline="")
    bw = csv.writer(bf)
    if saved_state is None:
        csv_w.writerow([
            "step", "problem_id", "n_icl", "greedy_reward", "cumulative_regret",
            "accepted_into_sdft", "kl_loss_batch", "batch_idx",
        ])
        bw.writerow([
            "batch_idx", "n_problems", "n_with_icl", "n_accepted",
            "avg_greedy_reward", "cumulative_regret", "kl_loss",
            "gen_time_s", "oracle_time_s", "train_time_s", "peak_vram_gb",
        ])

    # ── Main loop ────────────────────────────────────────────────────
    n_batches = (len(all_problems) + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    for batch_idx in range(start_batch, n_batches):
        batch = all_problems[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        B = len(batch)
        print(f"\n[Batch {batch_idx+1}/{n_batches}] {B} problems  "
              f"cache={len(cache)}", flush=True)

        # 1. Embed all questions in batch
        t0 = time.time()
        questions = [_question_text(p) for p in batch]
        q_embs = embedder.encode(questions)
        emb_time = time.time() - t0

        # 2. Retrieve ICL examples per problem
        per_problem_examples = [
            cache.retrieve(q_embs[i], k=args.knn_k, min_reward=args.icl_min_reward)
            for i in range(B)
        ]
        n_with_icl = sum(1 for ex in per_problem_examples if ex)

        # 3. Build greedy prompts and generate ONE rollout per problem
        torch.cuda.empty_cache()
        t0 = time.time()
        greedy_msgs = [
            build_greedy_messages(p, per_problem_examples[i])
            for i, p in enumerate(batch)
        ]
        rollouts = greedy_generate(
            model, tokenizer, greedy_msgs,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
        )
        gen_time = time.time() - t0

        # 4. Oracle each rollout exactly once → triple-duty signal
        t0 = time.time()
        rewards = [
            oracle.evaluate(rollouts[i], batch[i], fractional=True)
            for i in range(B)
        ]
        oracle_time = time.time() - t0

        # 5. Update cumulative regret + add EVERY rollout to memory
        for i in range(B):
            cumulative_regret += (1.0 - rewards[i])
            cache.add(batch[i], q_embs[i], rollouts[i], rewards[i])

        # 6. Push accepted pairs into rolling history (current batch only)
        accepted_flags = []
        for i, p in enumerate(batch):
            if rewards[i] >= args.reward_threshold:
                pair_history.append({
                    "prompt": build_student_messages(p, per_problem_examples[i]),
                    "teacher_prompt": build_teacher_messages(
                        p, per_problem_examples[i], rollouts[i]),
                    "batch_idx": batch_idx,
                })
                accepted_flags.append(True)
            else:
                accepted_flags.append(False)
        n_accepted_this_batch = sum(accepted_flags)
        # Window: current + last batch_window batches
        window_start = max(0, batch_idx - args.batch_window)
        window_pairs = [p for p in pair_history if p["batch_idx"] >= window_start]
        n_window = len(window_pairs)
        avg_reward = float(np.mean(rewards))

        print(f"  embed={emb_time:.1f}s  gen={gen_time:.1f}s  "
              f"oracle={oracle_time:.1f}s  avg_R={avg_reward:.3f}  "
              f"n_icl={n_with_icl}/{B}  accepted_new={n_accepted_this_batch}/{B}  "
              f"window=[{window_start}..{batch_idx}] pairs={n_window}", flush=True)

        # 7. SDFT update over windowed buffer
        kl_loss = 0.0
        train_time = 0.0
        if n_window > 0:
            torch.cuda.empty_cache()
            batch_ds = Dataset.from_dict({
                "prompt": [p["prompt"] for p in window_pairs],
                "teacher_prompt": [p["teacher_prompt"] for p in window_pairs],
            })
            config = DistilConfig(
                output_dir=os.path.join(args.output_dir, "_tmp_batch"),
                seed=args.seed,
                learning_rate=args.learning_rate,
                warmup_ratio=0.0,
                lr_scheduler_type="constant",
                num_train_epochs=args.num_train_epochs,
                # micro=1 + accum=n_window → every pair contributes exactly once per
                # epoch (no drops). Effective batch = full window. Slower per-step
                # than parallel micro-batches but correctness > speed here.
                per_device_train_batch_size=1,
                gradient_accumulation_steps=n_window,
                max_grad_norm=1.0,
                use_vllm=False,
                temperature=1.0,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                num_generations=1,
                generate_from_teacher=True,
                beta=0.0,
                alpha=0.0,
                num_iterations=1,
                num_loss_tokens_to_skip=3,
                sync_ref_model=False,
                bf16=True,
                fp16=False,
                logging_steps=1,
                save_steps=999999,
                report_to="none",
            )
            trainer = DistilTrainer(
                model=model,
                ref_model=teacher_model,
                args=config,
                train_dataset=batch_ds,
                processing_class=tokenizer,
            )
            t0 = time.time()
            result = trainer.train()
            train_time = time.time() - t0
            kl_loss = float(result.training_loss or 0.0)
            model = trainer.model
            print(f"  SDFT: kl_loss={kl_loss:.6f}  pairs={n_window}  "
                  f"time={train_time:.1f}s", flush=True)
        else:
            print(f"  SDFT: skipped (window empty)", flush=True)

        # 8. Per-problem CSV + details
        for i, p in enumerate(batch):
            csv_w.writerow([
                global_step, p.id, len(per_problem_examples[i]),
                f"{rewards[i]:.4f}", f"{cumulative_regret:.4f}",
                int(accepted_flags[i]), f"{kl_loss:.6f}", batch_idx,
            ])
            detail_entries.append({
                "step": global_step,
                "problem_id": p.id,
                "n_icl": len(per_problem_examples[i]),
                "greedy_reward": rewards[i],
                "accepted": accepted_flags[i],
                "rollout": rollouts[i],
                "kl_loss_batch": kl_loss,
                "batch_idx": batch_idx,
            })
            global_step += 1

        peak_vram = (torch.cuda.max_memory_allocated()/1e9
                     if torch.cuda.is_available() else 0.0)
        bw.writerow([
            batch_idx, B, n_with_icl, n_accepted_this_batch,
            f"{avg_reward:.4f}", f"{cumulative_regret:.4f}",
            f"{kl_loss:.6f}", f"{gen_time:.1f}", f"{oracle_time:.1f}",
            f"{train_time:.1f}", f"{peak_vram:.1f}",
        ])
        csv_f.flush()
        bf.flush()

        # 9. Checkpoint (LoRA + cache + pair_history + state.json)
        if (batch_idx + 1) % args.checkpoint_every == 0 or batch_idx == n_batches - 1:
            model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
            cache.save(os.path.join(args.output_dir, "cache.pkl"))
            with open(os.path.join(args.output_dir, "details.pkl"), "wb") as f:
                pickle.dump(detail_entries, f)
            with open(os.path.join(args.output_dir, "pair_history.pkl"), "wb") as f:
                pickle.dump(pair_history, f)
            with open(os.path.join(args.output_dir, "state.json"), "w") as f:
                json.dump({
                    "last_completed_batch": batch_idx,
                    "next_batch": batch_idx + 1,
                    "cumulative_regret": cumulative_regret,
                    "global_step": global_step,
                }, f, indent=2)

    csv_f.close()
    bf.close()

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    rewards_all = [d["greedy_reward"] for d in detail_entries]
    accepted_all = sum(1 for d in detail_entries if d["accepted"])
    print(f"\n{'='*70}\nDONE in {elapsed/60:.1f} min")
    print(f"  problems        : {len(detail_entries)}")
    print(f"  cum regret      : {cumulative_regret:.2f}")
    print(f"  avg greedy R    : {np.mean(rewards_all):.4f}")
    print(f"  SDFT accepted   : {accepted_all}/{len(detail_entries)} "
          f"({100*accepted_all/max(1,len(detail_entries)):.1f}%)")
    print(f"  final cache len : {len(cache)}")
    print(f"  outputs         : {args.output_dir}")


if __name__ == "__main__":
    main()
