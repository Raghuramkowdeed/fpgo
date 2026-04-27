#!/usr/bin/env python3
"""Generate greedy responses (no hints) from a trained LoRA on a sample of test
problems, plus the un-trained base model, and save raw generations for manual
inspection of whether the trained model spontaneously emits hint-style
reasoning (Key insight:/algorithm:/etc) when no hint is in the prompt."""

import argparse, json, os, pickle, sys, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from cumreg.datasets.base import Problem  # noqa
from cumreg.oracles.code_oracle import CodeOracle
from fgpo.run_fgpo_rloo import build_prompt, load_problems, load_entries


def gen_greedy(model, tokenizer, prompts, max_new_tokens, max_seq_length, batch_size=4):
    tokenizer.padding_side = "left"
    out_texts = []
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s:s + batch_size]
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq_length).to(model.device)
        ilen = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                 use_cache=True, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, ilen:]
        out_texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        del inp, out, gen
        torch.cuda.empty_cache()
    return out_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_paths", nargs="+", required=True,
                    help="One or more LoRA dirs to inspect (label@path or just path).")
    ap.add_argument("--include_base", action="store_true",
                    help="Also generate from the un-trained base model.")
    ap.add_argument("--n_problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--problems_path", type=str,
                    default=os.path.join(PROJECT_DIR, "data/livecodebench_problems.json"))
    ap.add_argument("--test_details", type=str,
                    default=os.path.join(PROJECT_DIR, "fgpo/data/details_test.pkl"))
    args = ap.parse_args()

    print(f"[inspect] loading problems...", flush=True)
    problems_by_id = load_problems(args.problems_path)
    test_entries = load_entries(args.test_details, problems_by_id)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(test_entries), size=min(args.n_problems, len(test_entries)),
                     replace=False)
    chosen = [test_entries[i] for i in idx]
    print(f"[inspect] selected {len(chosen)} test problems", flush=True)

    print(f"[inspect] loading tokenizer + base model {args.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="auto",
    )

    prompts = [
        tokenizer.apply_chat_template(build_prompt(e["question"], None),
                                       tokenize=False, add_generation_prompt=True)
        for e in chosen
    ]

    oracle = CodeOracle()

    runs = []
    if args.include_base:
        runs.append(("base", base, None))
    for spec in args.lora_paths:
        if "@" in spec:
            label, path = spec.split("@", 1)
        else:
            label, path = os.path.basename(os.path.dirname(spec)) or "lora", spec
        runs.append((label, base, path))

    results = {"meta": {"n_problems": len(chosen), "model": args.model_name,
                         "max_new_tokens": args.max_new_tokens, "seed": args.seed},
                "problems": [{"problem_id": e["problem_id"], "question": e["question"][:500]}
                             for e in chosen],
                "runs": []}

    current_lora = None
    peft_model = None
    for label, _b, lora_path in runs:
        print(f"\n[inspect] === run: {label} (lora={lora_path}) ===", flush=True)
        if lora_path is None:
            model = base
        else:
            if peft_model is not None:
                # Unload previous LoRA cleanly by deleting peft_model and reattaching
                del peft_model
                torch.cuda.empty_cache()
            peft_model = PeftModel.from_pretrained(base, lora_path)
            peft_model.eval()
            model = peft_model

        t0 = time.time()
        gens = gen_greedy(model, tokenizer, prompts,
                          max_new_tokens=args.max_new_tokens,
                          max_seq_length=args.max_seq_length, batch_size=4)
        print(f"[inspect] generated in {time.time()-t0:.1f}s", flush=True)

        # Score with oracle
        probs = [problems_by_id[e["problem_id"]] for e in chosen]
        rewards = [float(oracle.evaluate(g, p, fractional=True))
                   for g, p in zip(gens, probs)]
        print(f"[inspect] greedy reward: avg={np.mean(rewards):.3f}", flush=True)

        results["runs"].append({
            "label": label,
            "lora_path": lora_path,
            "avg_reward": float(np.mean(rewards)),
            "generations": [
                {"problem_id": chosen[i]["problem_id"],
                 "reward": rewards[i],
                 "response": gens[i]}
                for i in range(len(chosen))
            ],
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[inspect] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
