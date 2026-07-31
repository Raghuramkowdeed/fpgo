#!/usr/bin/env python3
"""FUSED distribution-shift stream — REINFORCE++ baseline.

Faithful REINFORCE++ (per-token KL folded into the return, global-whitened
return-to-go advantage, PPO-clip surrogate with importance ratio, offline
rolling-window replay of stored samples) — identical math to the per-dataset
runners. Only the stream (fused DS-1000 + DDXPlus + HotpotQA), the PER-ITEM system
prompt, and the PER-DATASET oracle are dispatched through FusedBench. Bare prompt
(NO ICL), temp=0 greedy. cumreg = Σ over NEW-batch of (1 - correct), scored before
the update -> comparable to the other fused baselines and the standalone runs.
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
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_constant_schedule_with_warmup)
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_bench import FusedBench


def build_messages_bare(system_prompt, gen_prompt):
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": gen_prompt}]


@torch.no_grad()
def greedy_generate(model, tokenizer, sys_gen_pairs, max_new_tokens, max_seq_length, micro_bs):
    """sys_gen_pairs: list of (system_prompt, gen_prompt)."""
    model.eval(); tokenizer.padding_side = "left"; out = []
    for s in range(0, len(sys_gen_pairs), micro_bs):
        sub = sys_gen_pairs[s:s+micro_bs]
        texts = [tokenizer.apply_chat_template(build_messages_bare(sp, gp), tokenize=False,
                                               add_generation_prompt=True) for sp, gp in sub]
        inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_seq_length).to(model.device)
        ilen = inp["input_ids"].shape[1]
        g = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                           use_cache=True, pad_token_id=tokenizer.pad_token_id)
        out.extend(tokenizer.batch_decode(g[:, ilen:], skip_special_tokens=True))
        del inp, g; torch.cuda.empty_cache()
    return out


# ── Per-token logps (identical math to the per-dataset runners) ──────────

def completion_token_logps(model, tokenizer, system_prompt, gen_prompt, response, max_seq_length, with_grad=False):
    prompt_text = tokenizer.apply_chat_template(build_messages_bare(system_prompt, gen_prompt),
                                                tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + response, add_special_tokens=False,
                         truncation=True, max_length=max_seq_length)["input_ids"]
    p_len = len(prompt_ids)
    if len(full_ids) <= p_len:
        return None, None
    input_ids = torch.tensor([full_ids], device=model.device)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids).logits[0]
        logps = F.log_softmax(logits.float(), dim=-1)
        comp_ids = input_ids[0, p_len:]
        comp_logps = logps[p_len-1: len(full_ids)-1]
        tok_logp = comp_logps.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
    return comp_ids.detach(), tok_logp


def capture_stored_logps(model, tokenizer, system_prompt, gen_prompt, response, max_seq_length):
    comp_ids, old_lp = completion_token_logps(model, tokenizer, system_prompt, gen_prompt, response, max_seq_length, False)
    if comp_ids is None:
        return None
    with model.disable_adapter():
        _, ref_lp = completion_token_logps(model, tokenizer, system_prompt, gen_prompt, response, max_seq_length, False)
    if ref_lp is None or ref_lp.numel() != old_lp.numel():
        ref_lp = old_lp.clone()
    return {"old_logp": old_lp.detach().cpu().numpy().astype(np.float32),
            "ref_logp": ref_lp.detach().cpu().numpy().astype(np.float32)}


def reinforce_pp_update(model, tokenizer, optimizer, scheduler, pool_items,
                        beta, gamma, clip_eps, max_seq_length, max_grad_norm=1.0):
    # Pass 1: returns -> global whiten
    returns_per_item = []; all_returns = []
    for it in pool_items:
        old = it["old_logp"].astype(np.float64); ref = it["ref_logp"].astype(np.float64)
        T = len(old); kl = old - ref
        tok_reward = -beta * kl; tok_reward[-1] += float(it["reward"])
        ret = np.zeros(T); running = 0.0
        for t in range(T-1, -1, -1):
            running = tok_reward[t] + gamma * running; ret[t] = running
        returns_per_item.append(ret); all_returns.append(ret)
    flat = np.concatenate(all_returns) if all_returns else np.zeros(1)
    mu = float(flat.mean()); var = float(((flat-mu)**2).mean()); inv = 1.0/np.sqrt(var+1e-8)
    adv_per_item = [(ret-mu)*inv for ret in returns_per_item]
    mean_kl = float(np.mean([(it["old_logp"].astype(np.float64)-it["ref_logp"].astype(np.float64)).mean()
                             for it in pool_items])) if pool_items else 0.0
    # Pass 2: PPO-clip loss
    model.train(); optimizer.zero_grad()
    n = len(pool_items); tot_pg = 0.0; tot_clip = 0.0; n_used = 0
    for it, adv_np in zip(pool_items, adv_per_item):
        comp_ids, cur_lp = completion_token_logps(model, tokenizer, it["system_prompt"], it["gen_prompt"],
                                                  it["response"], max_seq_length, True)
        if cur_lp is None or cur_lp.numel() != len(it["old_logp"]):
            continue
        dev = cur_lp.device
        old_lp = torch.tensor(it["old_logp"], device=dev, dtype=cur_lp.dtype)
        adv = torch.tensor(adv_np, device=dev, dtype=cur_lp.dtype)
        ratio = torch.exp(torch.clamp(cur_lp - old_lp, -20, 20))
        pg1 = -adv * ratio; pg2 = -adv * torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
        pg = torch.maximum(pg1, pg2).mean()
        (pg / n).backward()
        tot_pg += float(pg.detach())/n; tot_clip += float((pg2>pg1).float().mean().detach())/n; n_used += 1
    if n_used > 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step(); scheduler.step()
    return tot_pg, mean_kl, float(np.sqrt(var)), tot_clip, n_used


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results/fused_reinforce")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--stream_order", default="fused_stream_order.pkl")
    ap.add_argument("--max_problems", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--batch_window", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--gen_micro_batch", type=int, default=10)
    ap.add_argument("--oracle_timeout", type=float, default=10.0)
    ap.add_argument("--checkpoint_every", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"\n{'='*70}\nFUSED stream — REINFORCE++ (bare prompt, no ICL)\n{'='*70}")
    print(f"bs={args.batch_size} window={args.batch_window} beta={args.beta} clip={args.clip_eps}", flush=True)

    with open(args.stream_order, "rb") as f:
        stream_items = pickle.load(f)["stream"]
    if args.max_problems: stream_items = stream_items[:args.max_problems]
    n_total = len(stream_items)

    fb = FusedBench(seed=args.seed, timeout=args.oracle_timeout)
    def uid_of(it): return f"{it['dataset']}#{it['id']}"
    uids = [uid_of(it) for it in stream_items]
    ds_of = {uid_of(it): it["dataset"] for it in stream_items}
    row_of = {uid_of(it): fb.row(it["dataset"], it["id"]) for it in stream_items}
    sys_of = {u: fb.system(ds_of[u]) for u in uids}
    gen_prompt = {u: fb.gen_prompt(ds_of[u], row_of[u]) for u in uids}
    from collections import Counter
    print(f"stream: {n_total}  composition={dict(Counter(ds_of[u] for u in uids))}\n", flush=True)

    state_path = os.path.join(args.output_dir, "state.json")
    lora_dir = os.path.join(args.output_dir, "lora_adapter")
    store_path = os.path.join(args.output_dir, "store.pkl")
    pidlog_path = os.path.join(args.output_dir, "pid_log.pkl")
    optim_path = os.path.join(args.output_dir, "optim.pt")
    can_resume = all(os.path.exists(p) for p in [state_path, os.path.join(lora_dir, "adapter_config.json"),
                                                 store_path, pidlog_path, optim_path])
    saved = json.load(open(state_path)) if can_resume else None

    print("Loading model (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if saved is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)

    store = {}; pid_log = {}; cumreg = 0.0; n_correct = 0; start_batch = 0
    corr_by_ds = {"ds1000": [0, 0], "ddxplus": [0, 0], "hotpotqa": [0, 0]}
    if saved is not None:
        for u, d in pickle.load(open(store_path, "rb")).items(): store[u] = d
        pid_log = {int(k): v for k, v in pickle.load(open(pidlog_path, "rb")).items()}
        cumreg = saved["cumulative_regret"]; n_correct = saved["n_correct"]; start_batch = saved["next_batch"]
        corr_by_ds = saved.get("corr_by_ds", corr_by_ds)
        osd = torch.load(optim_path, map_location="cpu")
        optimizer.load_state_dict(osd["optimizer"]); scheduler.load_state_dict(osd["scheduler"])
        print(f"  resumed batch {start_batch} store={len(store)} cumreg={cumreg:.1f}", flush=True)

    mode = "a" if saved else "w"
    pf = open(os.path.join(args.output_dir, "per_problem.csv"), mode, newline=""); pw = csv.writer(pf)
    bfm = open(os.path.join(args.output_dir, "batch_metrics.csv"), mode, newline=""); bw = csv.writer(bfm)
    if saved is None:
        pw.writerow(["step", "uid", "dataset", "correct", "cumulative_regret"])
        bw.writerow(["batch_idx", "n_train", "acc_running", "cumulative_regret",
                     "acc_ds1000", "acc_ddxplus", "acc_hotpotqa",
                     "pg_loss", "mean_kl", "adv_std", "ppo_clipfrac", "gen_time_s", "train_time_s"])

    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    t_start = time.time()
    for bidx in range(start_batch, n_batches):
        batch_uids = uids[bidx*args.batch_size:(bidx+1)*args.batch_size]
        # 1. NEW batch: greedy + oracle + store logps
        torch.cuda.empty_cache(); t0 = time.time()
        gens = greedy_generate(model, tokenizer, [(sys_of[u], gen_prompt[u]) for u in batch_uids],
                               args.max_new_tokens, args.max_seq_length, args.gen_micro_batch)
        for i, (u, g) in enumerate(zip(batch_uids, gens)):
            d = ds_of[u]
            pred = fb.postprocess(d, g)
            correct = fb.is_correct(d, pred, row_of[u], bidx*args.batch_size+i)
            cumreg += (1 - correct); n_correct += correct
            corr_by_ds[d][0] += correct; corr_by_ds[d][1] += 1
            caps = capture_stored_logps(model, tokenizer, sys_of[u], gen_prompt[u], g, args.max_seq_length)
            if caps is not None:
                store[u] = {"system_prompt": sys_of[u], "gen_prompt": gen_prompt[u],
                            "response": g, "reward": float(correct), **caps}
            pw.writerow([bidx*args.batch_size+i, u, d, correct, f"{cumreg:.4f}"])
        gen_time = time.time() - t0
        pid_log[bidx] = batch_uids
        # 2. window
        wstart = max(0, bidx - args.batch_window)
        seen = set(); train_uids = []
        for b in range(wstart, bidx+1):
            for u in pid_log.get(b, []):
                if u not in seen: seen.add(u); train_uids.append(u)
        pool = [store[u] for u in train_uids if u in store]
        # 3. update
        t0 = time.time()
        pg, mkl, adv_std, clipf, n_used = reinforce_pp_update(
            model, tokenizer, optimizer, scheduler, pool,
            beta=args.beta, gamma=args.gamma, clip_eps=args.clip_eps, max_seq_length=args.max_seq_length)
        train_time = time.time() - t0
        acc = n_correct / min((bidx+1)*args.batch_size, n_total)
        accd = {dd: (corr_by_ds[dd][0]/corr_by_ds[dd][1] if corr_by_ds[dd][1] else 0.0) for dd in corr_by_ds}
        bw.writerow([bidx, len(pool), f"{acc:.4f}", f"{cumreg:.4f}",
                     f"{accd['ds1000']:.4f}", f"{accd['ddxplus']:.4f}", f"{accd['hotpotqa']:.4f}",
                     f"{pg:.6f}", f"{mkl:.6f}", f"{adv_std:.4f}", f"{clipf:.4f}", f"{gen_time:.1f}", f"{train_time:.1f}"])
        pf.flush(); bfm.flush()
        print(f"[B{bidx+1}/{n_batches}] acc={acc:.3f} cumreg={cumreg:.1f} pool={len(pool)} "
              f"pg={pg:.4f} kl={mkl:.4f} clipfrac={clipf:.3f} "
              f"[ds1k={accd['ds1000']:.2f} ddx={accd['ddxplus']:.2f} hpqa={accd['hotpotqa']:.2f}] "
              f"gen={gen_time:.0f}s train={train_time:.0f}s", flush=True)
        if (bidx+1) % args.checkpoint_every == 0 or bidx == n_batches-1:
            model.save_pretrained(lora_dir)
            pickle.dump({u: d for u, d in store.items()}, open(store_path, "wb"))
            pickle.dump(pid_log, open(pidlog_path, "wb"))
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, optim_path)
            json.dump({"last_completed_batch": bidx, "next_batch": bidx+1,
                       "cumulative_regret": cumreg, "n_correct": n_correct,
                       "corr_by_ds": corr_by_ds}, open(state_path, "w"), indent=2)

    pf.close(); bfm.close()
    print(f"\n{'='*70}\nDONE in {(time.time()-t_start)/60:.1f} min")
    print(f"  pass@1: {n_correct/n_total:.4f} ({n_correct}/{n_total})  cumreg: {cumreg:.2f}")
    for d in corr_by_ds:
        c, s = corr_by_ds[d]
        print(f"    {d:9}: acc={c/s if s else 0:.4f} ({c}/{s})")
    print(flush=True)


if __name__ == "__main__":
    main()
