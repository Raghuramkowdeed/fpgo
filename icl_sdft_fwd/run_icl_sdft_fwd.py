#!/usr/bin/env python3
"""Forward-looking ICL+SDFT on LiveCodeBench (1 oracle / new q).

Builds on `icl_sdft/run_icl_sdft_online1q.py`. The key difference is that we
re-evaluate past questions inside a rolling window using the *current* model
and the *current* in-context memory, so the SDFT teacher example for any q in
the window is always the BEST rollout we've ever produced for it (not the
historical one made before training had a chance to help).

Single source of truth:
  - `mem_bank` holds the MAX-R rollout per past pid (tie → latest).
    Used both as ICL retrieval anchor pool AND as the SDFT teacher demo.
  - `hist[pid]` is an audit-only chronological log of every attempt — not used
    in training.

Per outer batch t (batch_window = 9 by default):
  1. NEW batch (10 pids): retrieve ICL from mem_bank → greedy gen → oracle
     (this is the cumreg-cost call) → log → update mem_bank (max-R rule).
  2. RE-EVAL past 9 batches' pids (≤90): retrieve ICL with current mem_bank
     → greedy gen → oracle (training cost only, not metric cost) → log →
     update mem_bank if new reward ≥ existing (max-R, tie → latest).
  3. SDFT: pool = pids in (past_window ∪ current batch) where
     mem_bank[pid].reward ≥ reward_threshold (default 0.5).
     Each pair uses the current mem_bank's ICL anchors and the max-R rollout
     as the teacher demo. Grad-accum chunk = min(n_sdft, sdft_chunk_size).

Oracle parallelism via multiprocessing.Pool (`--oracle_workers`, default 30).

Cumulative regret accounting (unchanged from v1):
  cumreg = sum over NEW-batch pids of (1 - greedy_reward). Re-eval oracle
  calls are training-only cost and do NOT enter cumreg.

Outputs (`<output_dir>/`):
  state.json, lora_adapter/, mem_bank.pkl, hist.pkl, pid_log.pkl, details.pkl,
  per_problem.csv, batch_metrics.csv, mem_bank_trajectory.csv, forward_attempts.csv.
"""

import argparse
import csv
import json
import multiprocessing as mp
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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

_SD_PATH = os.environ.get(
    "SELF_DISTILLATION_PATH",
    os.path.join(os.path.dirname(PROJECT_DIR), "Self-Distillation"),
)
if not os.path.isdir(_SD_PATH):
    raise RuntimeError(
        f"Self-Distillation not found at {_SD_PATH}. "
        f"Set SELF_DISTILLATION_PATH or place the repo at {_SD_PATH}."
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
            emb = h[:, -1]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0)


# ── MemBank — single source of truth ───────────────────────────────────────

class MemBank:
    """Per-pid MAX-R rollout (tie → latest). One entry per pid.

    Used both for ICL retrieval and as the SDFT teacher example.
    """

    def __init__(self):
        # pid → {problem, embedding, response, reward, batch_idx, icl_pids}
        self.entries = {}

    def update(self, problem, embedding, response, reward, batch_idx, icl_pids):
        """max-R wins, tie → latest. Returns True if inserted/replaced."""
        cur = self.entries.get(problem.id)
        if cur is None or reward >= cur["reward"]:
            self.entries[problem.id] = {
                "problem": problem,
                "embedding": embedding,
                "response": response,
                "reward": reward,
                "batch_idx": batch_idx,
                "icl_pids": list(icl_pids),
            }
            return True
        return False

    def get(self, pid):
        return self.entries.get(pid)

    def retrieve(self, query_emb, k=3, min_reward=0.8, exclude_pid=None):
        """Top-k by cosine sim over entries where reward >= min_reward."""
        pool = [
            (pid, e) for pid, e in self.entries.items()
            if e["reward"] >= min_reward and (exclude_pid is None or pid != exclude_pid)
        ]
        if not pool:
            return []
        emb = np.stack([e["embedding"] for _, e in pool])
        sims = emb @ query_emb
        top = min(k, len(pool))
        idx = np.argsort(sims)[-top:][::-1]
        return [pool[i][1] for i in idx]

    def __len__(self):
        return len(self.entries)

    def stats(self, min_reward=0.8):
        if not self.entries:
            return 0, 0.0, 0, 0
        rs = np.array([e["reward"] for e in self.entries.values()])
        return (len(rs), float(rs.mean()),
                int((rs >= min_reward).sum()), int((rs == 1.0).sum()))

    def save(self, path):
        data = {
            pid: {
                "problem_id": e["problem"].id,
                "embedding": e["embedding"],
                "response": e["response"],
                "reward": e["reward"],
                "batch_idx": e["batch_idx"],
                "icl_pids": e["icl_pids"],
            }
            for pid, e in self.entries.items()
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)


# ── Prompt builders ───────────────────────────────────────────────────────

def _question_text(problem):
    q = problem.question
    starter = (problem.metadata or {}).get("starter_code", "")
    if starter:
        q += f"\n\nStarter code:\n```python\n{starter}\n```"
    return q


def build_greedy_messages(problem, examples):
    msgs = [{"role": "system", "content": CODE_SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user", "content": _question_text(ex["problem"])})
        msgs.append({"role": "assistant", "content": ex["response"]})
    msgs.append({"role": "user", "content": _question_text(problem)})
    return msgs


def build_student_messages(problem, examples):
    return build_greedy_messages(problem, examples)


def build_teacher_messages(problem, examples, demo_response):
    msgs = [{"role": "system", "content": CODE_SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user", "content": _question_text(ex["problem"])})
        msgs.append({"role": "assistant", "content": ex["response"]})
    q = _question_text(problem)
    msgs.append({"role": "user", "content": q})
    msgs.append({"role": "assistant", "content": demo_response})
    msgs.append({"role": "user", "content": q})
    return msgs


# ── Greedy generation (batched, GPU sequential) ───────────────────────────

def greedy_generate(model, tokenizer, messages_list, max_new_tokens,
                    max_seq_length, micro_bs=10):
    """Batched deterministic generation, split into micro-batches of micro_bs."""
    model.eval()
    tokenizer.padding_side = "left"
    out_all = []
    for start in range(0, len(messages_list), micro_bs):
        sub = messages_list[start:start + micro_bs]
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in sub
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
        out_all.extend(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
    return out_all


# ── Oracle parallel pool ─────────────────────────────────────────────────

def _pool_init(timeout):
    global _ORACLE
    _ORACLE = CodeOracle(timeout=timeout)


def _pool_eval(args):
    response, problem = args
    return _ORACLE.evaluate(response, problem, fractional=True)


def parallel_evaluate(pool, responses, problems):
    return pool.map(_pool_eval, list(zip(responses, problems)))


# ── Data loading ──────────────────────────────────────────────────────────

def load_problems(problems_path):
    with open(problems_path) as f:
        raw = json.load(f)
    return [Problem(**p) for p in raw]


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Forward-looking ICL+SDFT")
    ap.add_argument("--output_dir", default=os.path.join(PROJECT_DIR, "results/icl_sdft_fwd"))
    ap.add_argument("--problems_path",
                    default=os.path.join(PROJECT_DIR, "data/livecodebench_problems.json"))
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--embedder_name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--max_problems", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--knn_k", type=int, default=3)
    ap.add_argument("--icl_min_reward", type=float, default=0.8,
                    help="An entry qualifies as ICL anchor only if mem_bank[pid].reward >= this")
    ap.add_argument("--reward_threshold", type=float, default=0.5,
                    help="A pid enters SDFT only if mem_bank[pid].reward >= this")
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--batch_window", type=int, default=9,
                    help="Re-eval pids first-seen in batches [t-batch_window, t-1]")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--gen_micro_batch", type=int, default=10)
    ap.add_argument("--oracle_timeout", type=int, default=10)
    ap.add_argument("--oracle_workers", type=int, default=30)
    ap.add_argument("--sdft_chunk_size", type=int, default=50,
                    help="grad_accum chunk for SDFT; uses ALL eligible pairs")
    ap.add_argument("--checkpoint_every", type=int, default=5)
    return ap.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*70}\nForward-looking ICL+SDFT\n{'='*70}")
    print(f"model            : {args.model_name}")
    print(f"embedder         : {args.embedder_name}")
    print(f"batch_size       : {args.batch_size}   batch_window: {args.batch_window}")
    print(f"knn_k            : {args.knn_k}   icl_min_reward: {args.icl_min_reward}")
    print(f"reward_threshold : {args.reward_threshold} (SDFT gate)")
    print(f"lr               : {args.learning_rate}   epochs: {args.num_train_epochs}")
    print(f"sdft_chunk_size  : {args.sdft_chunk_size}   oracle_workers: {args.oracle_workers}")
    print(f"out              : {args.output_dir}")

    all_problems = load_problems(args.problems_path)
    if args.max_problems:
        all_problems = all_problems[:args.max_problems]
    problems_by_id = {p.id: p for p in all_problems}
    print(f"problems         : {len(all_problems)}")

    # ── Resume detection ────────────────────────────────────────────
    state_path   = os.path.join(args.output_dir, "state.json")
    lora_dir     = os.path.join(args.output_dir, "lora_adapter")
    membank_path = os.path.join(args.output_dir, "mem_bank.pkl")
    hist_path    = os.path.join(args.output_dir, "hist.pkl")
    pidlog_path  = os.path.join(args.output_dir, "pid_log.pkl")
    details_path = os.path.join(args.output_dir, "details.pkl")
    can_resume = all(os.path.exists(p) for p in [
        state_path, os.path.join(lora_dir, "adapter_config.json"),
        membank_path, hist_path, pidlog_path, details_path,
    ])
    if can_resume:
        with open(state_path) as f:
            saved_state = json.load(f)
        print(f"\nRESUMING from batch {saved_state['next_batch']} "
              f"(last_completed={saved_state['last_completed_batch']})", flush=True)
    else:
        saved_state = None
        print("\nStarting FRESH", flush=True)

    # ── Models ──────────────────────────────────────────────────────
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

    print(f"\nLoading embedder: {args.embedder_name} ...", flush=True)
    embedder = Qwen3Embedder(args.embedder_name, device="cuda")

    # ── Oracle pool ─────────────────────────────────────────────────
    mp_ctx = mp.get_context("spawn")
    pool = mp_ctx.Pool(
        processes=args.oracle_workers,
        initializer=_pool_init,
        initargs=(args.oracle_timeout,),
    )
    print(f"oracle pool      : {args.oracle_workers} workers (spawn)")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM after loads: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── State ───────────────────────────────────────────────────────
    mem_bank = MemBank()
    hist = {}              # pid → list[attempt dict]
    pid_log = {}           # batch_idx → list[pid]
    cumulative_regret = 0.0
    detail_entries = []
    global_step = 0
    start_batch = 0

    if saved_state is not None:
        mb_data = pickle.load(open(membank_path, "rb"))
        hist = pickle.load(open(hist_path, "rb"))
        pid_log = pickle.load(open(pidlog_path, "rb"))
        detail_entries = pickle.load(open(details_path, "rb"))
        for pid, d in mb_data.items():
            if pid in problems_by_id:
                mem_bank.entries[pid] = {
                    "problem": problems_by_id[pid],
                    "embedding": d["embedding"],
                    "response": d["response"],
                    "reward": d["reward"],
                    "batch_idx": d["batch_idx"],
                    "icl_pids": d.get("icl_pids", []),
                }
        cumulative_regret = saved_state["cumulative_regret"]
        global_step = saved_state["global_step"]
        start_batch = saved_state["next_batch"]
        print(f"  rehydrated: mem_bank={len(mem_bank)} hist_pids={len(hist)} "
              f"pid_log_batches={len(pid_log)} details={len(detail_entries)} "
              f"cumreg={cumulative_regret:.2f}", flush=True)

    # ── CSVs ────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "per_problem.csv")
    batch_csv_path = os.path.join(args.output_dir, "batch_metrics.csv")
    mb_csv_path = os.path.join(args.output_dir, "mem_bank_trajectory.csv")
    fa_csv_path = os.path.join(args.output_dir, "forward_attempts.csv")
    csv_mode = "a" if saved_state is not None else "w"
    csv_f = open(csv_path, csv_mode, newline="")
    csv_w = csv.writer(csv_f)
    bf = open(batch_csv_path, csv_mode, newline="")
    bw = csv.writer(bf)
    mb_f = open(mb_csv_path, csv_mode, newline="")
    mb_w = csv.writer(mb_f)
    fa_f = open(fa_csv_path, csv_mode, newline="")
    fa_w = csv.writer(fa_f)
    if saved_state is None:
        csv_w.writerow([
            "step", "problem_id", "n_icl", "greedy_reward", "cumulative_regret",
            "accepted_into_mem", "kl_loss_batch", "batch_idx",
        ])
        bw.writerow([
            "batch_idx", "n_problems", "n_with_icl", "n_new_into_mem",
            "n_reeval", "n_reeval_replaced", "avg_greedy_reward",
            "cumulative_regret", "kl_loss", "gen_time_s", "oracle_time_s",
            "reeval_gen_time_s", "reeval_oracle_time_s", "train_time_s",
            "mem_bank_size", "peak_vram_gb",
        ])
        mb_w.writerow([
            "batch_idx", "mem_bank_size", "mean_R", "n_R_ge_0.8", "n_R_eq_1",
        ])
        fa_w.writerow([
            "batch_idx", "problem_id", "old_mem_R", "new_R",
            "replaced", "n_icl_used",
        ])

    n_batches = (len(all_problems) + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    for batch_idx in range(start_batch, n_batches):
        batch = all_problems[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        B = len(batch)
        print(f"\n[Batch {batch_idx+1}/{n_batches}] {B} problems  "
              f"mem_bank={len(mem_bank)}  hist_pids={len(hist)}", flush=True)

        # ── 1. Embed batch questions ────────────────────────────────
        t0 = time.time()
        questions = [_question_text(p) for p in batch]
        q_embs = embedder.encode(questions)
        emb_time = time.time() - t0

        # ── 2. Retrieve ICL per new pid (from current mem_bank) ─────
        per_problem_examples = [
            mem_bank.retrieve(q_embs[i], k=args.knn_k,
                              min_reward=args.icl_min_reward,
                              exclude_pid=batch[i].id)
            for i in range(B)
        ]
        per_problem_icl_pids = [
            [ex["problem"].id for ex in per_problem_examples[i]]
            for i in range(B)
        ]
        n_with_icl = sum(1 for ex in per_problem_examples if ex)

        # ── 3. Greedy gen for new batch ─────────────────────────────
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
            micro_bs=args.gen_micro_batch,
        )
        gen_time = time.time() - t0

        # ── 4. Oracle on new batch (parallel) — cumreg cost ─────────
        t0 = time.time()
        rewards = parallel_evaluate(pool, rollouts, batch)
        oracle_time = time.time() - t0

        # ── 5. Update cumreg + hist + mem_bank for new batch ────────
        accepted_new = []
        for i, p in enumerate(batch):
            cumulative_regret += (1.0 - rewards[i])
            hist.setdefault(p.id, []).append({
                "response": rollouts[i],
                "reward": rewards[i],
                "icl_pids": per_problem_icl_pids[i],
                "batch_idx": batch_idx,
                "origin": "new",
            })
            inserted = mem_bank.update(
                problem=p, embedding=q_embs[i],
                response=rollouts[i], reward=rewards[i],
                batch_idx=batch_idx, icl_pids=per_problem_icl_pids[i],
            )
            accepted_new.append(inserted)

        pid_log[batch_idx] = [p.id for p in batch]
        n_new_into_mem = sum(accepted_new)
        avg_reward = float(np.mean(rewards))

        print(f"  embed={emb_time:.1f}s  gen={gen_time:.1f}s  "
              f"oracle={oracle_time:.1f}s  avg_R={avg_reward:.3f}  "
              f"n_icl={n_with_icl}/{B}  new_into_mem={n_new_into_mem}/{B}", flush=True)

        # ── 6. Re-eval past window pids ─────────────────────────────
        window_start = max(0, batch_idx - args.batch_window)
        past_pids = []
        for b in range(window_start, batch_idx):
            past_pids.extend(pid_log.get(b, []))
        seen = set()
        past_pids = [p for p in past_pids if not (p in seen or seen.add(p))]

        n_reeval = 0
        n_reeval_replaced = 0
        reeval_gen_time = 0.0
        reeval_oracle_time = 0.0

        if past_pids:
            reeval_problems = [problems_by_id[pid] for pid in past_pids]
            reeval_anchors = [
                mem_bank.retrieve(mem_bank.get(pid)["embedding"],
                                  k=args.knn_k,
                                  min_reward=args.icl_min_reward,
                                  exclude_pid=pid)
                for pid in past_pids
            ]
            reeval_anchor_pids = [
                [ex["problem"].id for ex in al] for al in reeval_anchors
            ]
            reeval_msgs = [
                build_greedy_messages(p, a)
                for p, a in zip(reeval_problems, reeval_anchors)
            ]
            torch.cuda.empty_cache()
            t0 = time.time()
            reeval_rollouts = greedy_generate(
                model, tokenizer, reeval_msgs,
                max_new_tokens=args.max_new_tokens,
                max_seq_length=args.max_seq_length,
                micro_bs=args.gen_micro_batch,
            )
            reeval_gen_time = time.time() - t0

            t0 = time.time()
            reeval_rewards = parallel_evaluate(pool, reeval_rollouts, reeval_problems)
            reeval_oracle_time = time.time() - t0

            n_reeval = len(past_pids)
            for pid, p, anchors, apids, y, r in zip(
                past_pids, reeval_problems, reeval_anchors, reeval_anchor_pids,
                reeval_rollouts, reeval_rewards,
            ):
                old_r = mem_bank.get(pid)["reward"]
                hist[pid].append({
                    "response": y, "reward": r, "icl_pids": apids,
                    "batch_idx": batch_idx, "origin": "reeval",
                })
                replaced = mem_bank.update(
                    problem=p, embedding=mem_bank.get(pid)["embedding"],
                    response=y, reward=r,
                    batch_idx=batch_idx, icl_pids=apids,
                )
                if replaced:
                    n_reeval_replaced += 1
                fa_w.writerow([
                    batch_idx, pid, f"{old_r:.4f}", f"{r:.4f}",
                    int(replaced), len(anchors),
                ])

        print(f"  reeval: n={n_reeval}  replaced={n_reeval_replaced}  "
              f"gen={reeval_gen_time:.1f}s  oracle={reeval_oracle_time:.1f}s",
              flush=True)

        # ── 7. SDFT — train on rolling window pids whose mem_bank R >= gate ─
        train_pids = list(set(past_pids) | set(pid_log[batch_idx]))
        sdft_pairs = []
        for pid in train_pids:
            entry = mem_bank.get(pid)
            if entry is None or entry["reward"] < args.reward_threshold:
                continue
            p = entry["problem"]
            anchors = mem_bank.retrieve(entry["embedding"], k=args.knn_k,
                                        min_reward=args.icl_min_reward,
                                        exclude_pid=pid)
            sdft_pairs.append({
                "prompt": build_student_messages(p, anchors),
                "teacher_prompt": build_teacher_messages(p, anchors, entry["response"]),
            })
        n_sdft = len(sdft_pairs)

        kl_loss = 0.0
        train_time = 0.0
        if n_sdft > 0:
            torch.cuda.empty_cache()
            grad_accum = min(n_sdft, args.sdft_chunk_size)
            opt_steps_per_epoch = (n_sdft + grad_accum - 1) // grad_accum
            batch_ds = Dataset.from_dict({
                "prompt": [p["prompt"] for p in sdft_pairs],
                "teacher_prompt": [p["teacher_prompt"] for p in sdft_pairs],
            })
            config = DistilConfig(
                output_dir=os.path.join(args.output_dir, "_tmp_batch"),
                seed=args.seed,
                learning_rate=args.learning_rate,
                warmup_ratio=0.0,
                lr_scheduler_type="constant",
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=grad_accum,
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
            print(f"  SDFT: kl_loss={kl_loss:.6f}  pairs={n_sdft}  "
                  f"grad_accum={grad_accum} ({opt_steps_per_epoch} opt step(s)/epoch x "
                  f"{args.num_train_epochs} epoch)  "
                  f"time={train_time:.1f}s", flush=True)
        else:
            print(f"  SDFT: skipped (no eligible pairs)", flush=True)

        # ── 8. Per-problem CSV + details ────────────────────────────
        for i, p in enumerate(batch):
            csv_w.writerow([
                global_step, p.id, len(per_problem_examples[i]),
                f"{rewards[i]:.4f}", f"{cumulative_regret:.4f}",
                int(accepted_new[i]), f"{kl_loss:.6f}", batch_idx,
            ])
            detail_entries.append({
                "step": global_step,
                "problem_id": p.id,
                "n_icl": len(per_problem_examples[i]),
                "icl_pids": per_problem_icl_pids[i],
                "greedy_reward": rewards[i],
                "accepted": accepted_new[i],
                "rollout": rollouts[i],
                "kl_loss_batch": kl_loss,
                "batch_idx": batch_idx,
            })
            global_step += 1

        # ── 9. Mem-bank trajectory log ──────────────────────────────
        size, mean_R, n_ge_08, n_eq_1 = mem_bank.stats(min_reward=args.icl_min_reward)
        mb_w.writerow([batch_idx, size, f"{mean_R:.4f}", n_ge_08, n_eq_1])
        mb_f.flush()
        fa_f.flush()

        peak_vram = (torch.cuda.max_memory_allocated()/1e9
                     if torch.cuda.is_available() else 0.0)
        bw.writerow([
            batch_idx, B, n_with_icl, n_new_into_mem,
            n_reeval, n_reeval_replaced,
            f"{avg_reward:.4f}", f"{cumulative_regret:.4f}",
            f"{kl_loss:.6f}",
            f"{gen_time:.1f}", f"{oracle_time:.1f}",
            f"{reeval_gen_time:.1f}", f"{reeval_oracle_time:.1f}",
            f"{train_time:.1f}", len(mem_bank), f"{peak_vram:.1f}",
        ])
        csv_f.flush()
        bf.flush()

        # ── 10. Checkpoint ──────────────────────────────────────────
        if (batch_idx + 1) % args.checkpoint_every == 0 or batch_idx == n_batches - 1:
            model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
            mem_bank.save(membank_path)
            with open(hist_path, "wb") as f:
                pickle.dump(hist, f)
            with open(details_path, "wb") as f:
                pickle.dump(detail_entries, f)
            with open(pidlog_path, "wb") as f:
                pickle.dump(pid_log, f)
            with open(state_path, "w") as f:
                json.dump({
                    "last_completed_batch": batch_idx,
                    "next_batch": batch_idx + 1,
                    "cumulative_regret": cumulative_regret,
                    "global_step": global_step,
                }, f, indent=2)

    csv_f.close()
    bf.close()
    mb_f.close()
    fa_f.close()
    pool.close()
    pool.join()

    elapsed = time.time() - t_start
    rewards_all = [d["greedy_reward"] for d in detail_entries]
    accepted_all = sum(1 for d in detail_entries if d["accepted"])
    print(f"\n{'='*70}\nDONE in {elapsed/60:.1f} min")
    print(f"  problems         : {len(detail_entries)}")
    print(f"  cum regret       : {cumulative_regret:.2f}")
    print(f"  avg greedy R     : {np.mean(rewards_all):.4f}")
    print(f"  new-into-mem     : {accepted_all}/{len(detail_entries)}")
    print(f"  mem bank size    : {len(mem_bank)}")
    print(f"  hist pids        : {len(hist)}")
    print(f"  outputs          : {args.output_dir}")


if __name__ == "__main__":
    main()
