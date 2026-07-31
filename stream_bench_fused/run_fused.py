#!/usr/bin/env python3
"""FUSED distribution-shift stream — base / ICL / SDFT+fwd baselines in one runner.

Streams the pre-fused seed-42 order (fused_stream_order.pkl: DS-1000 + DDXPlus +
HotpotQA, within-dataset order preserved vs each standalone run) and applies the SAME
rolling-window online skeleton as our per-dataset runners, but dispatches every per-item
operation (prompt, oracle, target text) through FusedBench. cumreg = Σ over NEW-batch of
(1 - correct), scored BEFORE any update, so it is comparable to the standalone runs and
across methods.

--method:
  base  : 0-shot greedy -> oracle. no memory, no ICL, no training.
  icl   : Self-StreamICL. mem_bank of self-generated correct answers, kNN demos, greedy.
  sdft  : ours (fwd4). mem_bank + kNN ICL + forward window re-eval + self-distillation.

Cross-dataset kNN retrieval (a code question may pull a QA demo) is intentional — it is
the distribution-shift phenomenon under study. REINFORCE++ is a separate runner.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

_SD_PATH = os.environ.get(
    "SELF_DISTILLATION_PATH",
    "/data/pulkitag/misc/raghuramkowdeed/projects/Self-Distillation")
sys.path.insert(0, _SD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fused_bench import FusedBench


# ── Qwen3 embedder (kNN-ICL) ─────────────────────────────────────────────

class Qwen3Embedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device="cuda", max_length=512):
        from transformers import AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval(); self.device = device; self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts, batch_size=8):
        if isinstance(texts, str): texts = [texts]
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[s:s+batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            out.append(torch.nn.functional.normalize(h[:, -1], p=2, dim=-1).float().cpu().numpy())
        return np.concatenate(out, axis=0)


# ── MemBank: uid -> latest self-generated correct answer (with source dataset) ──

class MemBank:
    def __init__(self):
        self.entries = {}   # uid -> {q_text, gen_prompt, answer, dataset, embedding, batch_idx}

    def update(self, uid, q_text, gen_prompt, answer, dataset, embedding, batch_idx):
        self.entries[uid] = {"q_text": q_text, "gen_prompt": gen_prompt, "answer": answer,
                             "dataset": dataset, "embedding": embedding, "batch_idx": batch_idx}

    def get(self, uid): return self.entries.get(uid)

    def retrieve(self, query_emb, k, exclude_uid=None):
        pool = [(u, e) for u, e in self.entries.items() if u != exclude_uid]
        if not pool or k <= 0: return []
        M = np.stack([e["embedding"] for _, e in pool])
        idx = np.argsort(M @ query_emb)[-min(k, len(pool)):][::-1]
        return [pool[i][1] for i in idx]

    def __len__(self): return len(self.entries)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({u: dict(e) for u, e in self.entries.items()}, f)


# ── message builders (per-item system prompt; demos are (question, answer) pairs) ──

def build_student_messages(system_prompt, gen_prompt, demos):
    msgs = [{"role": "system", "content": system_prompt}]
    for e in demos:
        msgs.append({"role": "user", "content": e["q_text"]})
        msgs.append({"role": "assistant", "content": e["answer"]})
    msgs.append({"role": "user", "content": gen_prompt})
    return msgs

def build_teacher_messages(system_prompt, gen_prompt, demos, demo_answer):
    msgs = build_student_messages(system_prompt, gen_prompt, demos)
    msgs.append({"role": "assistant", "content": demo_answer})
    msgs.append({"role": "user", "content": gen_prompt})
    return msgs


@torch.no_grad()
def greedy_generate(model, tokenizer, messages_list, max_new_tokens, max_seq_length, micro_bs):
    model.eval(); tokenizer.padding_side = "left"; out = []
    for s in range(0, len(messages_list), micro_bs):
        sub = messages_list[s:s+micro_bs]
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in sub]
        inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_seq_length).to(model.device)
        ilen = inp["input_ids"].shape[1]
        g = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                           use_cache=True, pad_token_id=tokenizer.pad_token_id)
        out.extend(tokenizer.batch_decode(g[:, ilen:], skip_special_tokens=True))
        del inp, g; torch.cuda.empty_cache()
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["base", "icl", "sdft"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--embedder_name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--stream_order", default="fused_stream_order.pkl")
    ap.add_argument("--max_problems", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--batch_window", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--knn_k", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--max_completion_length", type=int, default=1024)
    ap.add_argument("--gen_micro_batch", type=int, default=10)
    ap.add_argument("--sdft_chunk_size", type=int, default=50)
    ap.add_argument("--oracle_timeout", type=float, default=10.0)
    ap.add_argument("--checkpoint_every", type=int, default=5)
    ap.add_argument("--gen_from_teacher", type=int, default=1,
                    help="1 (default): completion generated from hint-conditioned teacher prompt. "
                         "0: generated from bare student prompt (paper default, on-policy).")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    use_icl = args.method in ("icl", "sdft")
    do_reeval = args.method == "sdft"
    do_train = args.method == "sdft"
    knn_k = args.knn_k if use_icl else 0

    print(f"\n{'='*70}\nFUSED stream — method={args.method}\n{'='*70}")
    print(f"model={args.model_name}  bs={args.batch_size} window={args.batch_window} "
          f"knn={knn_k} icl={use_icl} reeval={do_reeval} train={do_train}", flush=True)

    with open(args.stream_order, "rb") as f:
        cache = pickle.load(f)
    stream_items = cache["stream"]
    if args.max_problems: stream_items = stream_items[:args.max_problems]
    n_total = len(stream_items)

    fb = FusedBench(seed=args.seed, timeout=args.oracle_timeout)

    # per-uid metadata (uid = "dataset#id", globally unique across datasets)
    def uid_of(it): return f"{it['dataset']}#{it['id']}"
    uids = [uid_of(it) for it in stream_items]
    ds_of = {uid_of(it): it["dataset"] for it in stream_items}
    row_of = {uid_of(it): fb.row(it["dataset"], it["id"]) for it in stream_items}
    q_text = {u: fb.question(ds_of[u], row_of[u]) for u in uids}
    gen_prompt = {u: fb.gen_prompt(ds_of[u], row_of[u]) for u in uids}
    from collections import Counter
    print(f"stream: {n_total}  composition={dict(Counter(ds_of[u] for u in uids))}\n", flush=True)

    # ── Resume ──
    state_path = os.path.join(args.output_dir, "state.json")
    lora_dir = os.path.join(args.output_dir, "lora_adapter")
    mb_path = os.path.join(args.output_dir, "mem_bank.pkl")
    pidlog_path = os.path.join(args.output_dir, "pid_log.pkl")
    resume_paths = [state_path, mb_path, pidlog_path]
    if do_train: resume_paths.append(os.path.join(lora_dir, "adapter_config.json"))
    can_resume = all(os.path.exists(p) for p in resume_paths)
    saved = json.load(open(state_path)) if can_resume else None

    print("Loading model (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    teacher_model = None
    if do_train:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, PeftModel
        from distil_config import DistilConfig
        from distil_trainer import DistilTrainer
        teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        if saved is not None:
            model = PeftModel.from_pretrained(model, lora_dir, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

    embedder = Qwen3Embedder(args.embedder_name, device="cuda") if use_icl else None

    mem = MemBank(); pid_log = {}; cumreg = 0.0; n_correct = 0; start_batch = 0
    corr_by_ds = {"ds1000": [0, 0], "ddxplus": [0, 0], "hotpotqa": [0, 0]}  # [correct, seen]
    if saved is not None:
        for u, e in pickle.load(open(mb_path, "rb")).items(): mem.entries[u] = e
        pid_log = {int(k): v for k, v in pickle.load(open(pidlog_path, "rb")).items()}
        cumreg = saved["cumulative_regret"]; n_correct = saved["n_correct"]; start_batch = saved["next_batch"]
        corr_by_ds = saved.get("corr_by_ds", corr_by_ds)
        print(f"  resumed: batch {start_batch} mem={len(mem)} cumreg={cumreg:.1f}", flush=True)

    csv_mode = "a" if saved else "w"
    pf = open(os.path.join(args.output_dir, "per_problem.csv"), csv_mode, newline=""); pw = csv.writer(pf)
    bfm = open(os.path.join(args.output_dir, "batch_metrics.csv"), csv_mode, newline=""); bw = csv.writer(bfm)
    if saved is None:
        pw.writerow(["step", "uid", "dataset", "n_icl", "correct", "cumulative_regret", "mem_size"])
        bw.writerow(["batch_idx", "n_new", "acc_running", "cumulative_regret",
                     "acc_ds1000", "acc_ddxplus", "acc_hotpotqa",
                     "n_reeval", "n_reeval_recovered", "n_sdft_pairs", "kl_loss",
                     "mem_size", "gen_time_s", "reeval_time_s", "train_time_s"])

    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    for bidx in range(start_batch, n_batches):
        batch_uids = uids[bidx*args.batch_size:(bidx+1)*args.batch_size]
        # ── 1. NEW batch: (ICL retrieve) -> greedy -> oracle (cumreg) -> mem ──
        t0 = time.time()
        if use_icl:
            embs = embedder.encode([q_text[u] for u in batch_uids])
            demos_new = [mem.retrieve(embs[i], knn_k, exclude_uid=u) for i, u in enumerate(batch_uids)]
        else:
            embs = [None] * len(batch_uids); demos_new = [[] for _ in batch_uids]
        msgs = [build_student_messages(fb.system(ds_of[u]), gen_prompt[u], demos_new[i])
                for i, u in enumerate(batch_uids)]
        gens = greedy_generate(model, tokenizer, msgs, args.max_new_tokens, args.max_seq_length, args.gen_micro_batch)
        gen_time = time.time() - t0
        for i, u in enumerate(batch_uids):
            d = ds_of[u]
            pred = fb.postprocess(d, gens[i])
            correct = fb.is_correct(d, pred, row_of[u], bidx*args.batch_size+i)
            cumreg += (1 - correct); n_correct += correct
            corr_by_ds[d][0] += correct; corr_by_ds[d][1] += 1
            if correct and use_icl:
                mem.update(u, q_text[u], gen_prompt[u], fb.demo_answer(d, pred), d, embs[i], bidx)
            pw.writerow([bidx*args.batch_size+i, u, d, len(demos_new[i]), correct, f"{cumreg:.4f}", len(mem)])
        pid_log[bidx] = batch_uids

        # ── 2. WINDOW re-eval (sdft only) ──
        n_reeval = 0; n_recovered = 0; reeval_time = 0.0
        if do_reeval:
            wstart = max(0, bidx - args.batch_window)
            past = []; seen = set()
            for b in range(wstart, bidx):
                for u in pid_log.get(b, []):
                    if u not in seen: seen.add(u); past.append(u)
            if past:
                t0 = time.time()
                pembs = embedder.encode([q_text[u] for u in past])
                pmsgs = [build_student_messages(fb.system(ds_of[u]), gen_prompt[u],
                                                mem.retrieve(pembs[i], knn_k, exclude_uid=u))
                         for i, u in enumerate(past)]
                pgens = greedy_generate(model, tokenizer, pmsgs, args.max_new_tokens, args.max_seq_length, args.gen_micro_batch)
                for i, u in enumerate(past):
                    d = ds_of[u]
                    pred = fb.postprocess(d, pgens[i])
                    correct = fb.is_correct(d, pred, row_of[u], -1)
                    if correct:
                        if u not in mem.entries: n_recovered += 1
                        mem.update(u, q_text[u], gen_prompt[u], fb.demo_answer(d, pred), d, pembs[i], bidx)
                n_reeval = len(past); reeval_time = time.time() - t0

        # ── 3. SDFT distill window pool (sdft only) ──
        kl_loss = 0.0; train_time = 0.0; n_pairs = 0
        if do_train:
            train_uids = list(set(pid_log.get(bidx, [])) | (seen if do_reeval else set()))
            sdft_pairs = []
            for u in train_uids:
                e = mem.get(u)
                if e is None: continue
                demos = mem.retrieve(e["embedding"], knn_k, exclude_uid=u)
                sys_p = fb.system(e["dataset"])
                sdft_pairs.append({
                    "prompt": build_student_messages(sys_p, e["gen_prompt"], demos),
                    "teacher_prompt": build_teacher_messages(sys_p, e["gen_prompt"], demos, e["answer"]),
                })
            n_pairs = len(sdft_pairs)
            if sdft_pairs:
                t0 = time.time(); torch.cuda.empty_cache()
                grad_accum = min(len(sdft_pairs), args.sdft_chunk_size)
                ds = Dataset.from_dict({"prompt": [x["prompt"] for x in sdft_pairs],
                                        "teacher_prompt": [x["teacher_prompt"] for x in sdft_pairs]})
                cfg = DistilConfig(
                    output_dir=os.path.join(args.output_dir, "_tmp"), seed=args.seed,
                    learning_rate=args.learning_rate, warmup_ratio=0.0, lr_scheduler_type="constant",
                    num_train_epochs=args.num_train_epochs, per_device_train_batch_size=1,
                    gradient_accumulation_steps=grad_accum, max_grad_norm=1.0, use_vllm=False,
                    temperature=1.0, max_prompt_length=args.max_prompt_length,
                    max_completion_length=args.max_completion_length, num_generations=1,
                    generate_from_teacher=bool(args.gen_from_teacher), beta=0.0, alpha=0.0, num_iterations=1,
                    num_loss_tokens_to_skip=3, sync_ref_model=False, bf16=True, fp16=False,
                    logging_steps=1, save_steps=999999, report_to="none")
                trainer = DistilTrainer(model=model, ref_model=teacher_model, args=cfg,
                                        train_dataset=ds, processing_class=tokenizer)
                res = trainer.train(); kl_loss = float(res.training_loss or 0.0); model = trainer.model
                train_time = time.time() - t0

        seen_tot = (bidx+1)*args.batch_size
        acc = n_correct / (seen_tot if seen_tot <= n_total else n_total)
        accd = {d: (corr_by_ds[d][0]/corr_by_ds[d][1] if corr_by_ds[d][1] else 0.0) for d in corr_by_ds}
        bw.writerow([bidx, len(batch_uids), f"{acc:.4f}", f"{cumreg:.4f}",
                     f"{accd['ds1000']:.4f}", f"{accd['ddxplus']:.4f}", f"{accd['hotpotqa']:.4f}",
                     n_reeval, n_recovered, n_pairs, f"{kl_loss:.6f}", len(mem),
                     f"{gen_time:.1f}", f"{reeval_time:.1f}", f"{train_time:.1f}"])
        pf.flush(); bfm.flush()
        print(f"[B{bidx+1}/{n_batches}] acc={acc:.3f} cumreg={cumreg:.1f} mem={len(mem)} "
              f"reeval={n_reeval}(rec {n_recovered}) sdft={n_pairs} kl={kl_loss:.4f} "
              f"[ds1k={accd['ds1000']:.2f} ddx={accd['ddxplus']:.2f} hpqa={accd['hotpotqa']:.2f}] "
              f"gen={gen_time:.0f}s reeval={reeval_time:.0f}s train={train_time:.0f}s", flush=True)

        if (bidx+1) % args.checkpoint_every == 0 or bidx == n_batches-1:
            if do_train: model.save_pretrained(lora_dir)
            mem.save(mb_path); pickle.dump(pid_log, open(pidlog_path, "wb"))
            json.dump({"last_completed_batch": bidx, "next_batch": bidx+1,
                       "cumulative_regret": cumreg, "n_correct": n_correct,
                       "corr_by_ds": corr_by_ds}, open(state_path, "w"), indent=2)

    pf.close(); bfm.close()
    print(f"\n{'='*70}\nDONE in {(time.time()-t_start)/60:.1f} min")
    print(f"  pass@1 : {n_correct/n_total:.4f}  ({n_correct}/{n_total})   cumreg: {cumreg:.2f}")
    for d in corr_by_ds:
        c, s = corr_by_ds[d]
        print(f"    {d:9}: acc={c/s if s else 0:.4f} ({c}/{s})")
    print(flush=True)


if __name__ == "__main__":
    main()
