#!/usr/bin/env python3
"""FusedBench — dispatcher over 3 diverse StreamBench datasets for the fused
distribution-shift stream (DS-1000 code / DDXPlus diagnosis / HotpotQA QA).

Each fused stream item is {"dataset": name, "id": local_id}. This class routes every
per-item operation to the right sub-benchmark: system prompt, zero-shot prompt,
question text (for kNN embedding), generation post-processing, the PER-DATASET oracle
(ds1000 -> execute, ddxplus -> label-equality, hotpotqa -> exact-match), and the
self-generated correct-answer TEXT stored in mem_bank (never a gold label — matches the
standalone runners: ds1000 stores the code, ddx/hotpot store get_label_text(pred)).

Local id convention (matches build_fused_stream.py and the standalone runs):
  ds1000   -> metadata.problem_id (non-TF); rows keyed from get_dataset()
  ddxplus  -> positional index into list(get_dataset())   (same seed as standalone)
  hotpotqa -> positional index into list(get_dataset())
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stream_bench.benchmarks.ds_1000 import DS1000
from stream_bench.benchmarks.ddxplus import create_ddxplus
from stream_bench.benchmarks.hotpotqa_distract import HotpotQADistract, compute_exact_match

CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer for data science. "
    "Solve the problem by writing correct Python code. "
    "Output ONLY the code in a ```python ... ``` block.")
DDX_SYSTEM_PROMPT = "You are an expert medical doctor making a differential diagnosis."
HOTPOT_SYSTEM_PROMPT = (
    "You are a careful assistant answering multi-hop questions with short, exact answers in JSON.")

SYSTEM_PROMPT = {"ds1000": CODE_SYSTEM_PROMPT, "ddxplus": DDX_SYSTEM_PROMPT,
                 "hotpotqa": HOTPOT_SYSTEM_PROMPT}


class FusedBench:
    def __init__(self, seed=42, timeout=10.0):
        self.seed = seed
        self.bench = {}
        self.bench["ds1000"] = DS1000(split="test", seed=seed, timeout=timeout)
        DDX = create_ddxplus()
        self.bench["ddxplus"] = DDX(split="test", seed=seed)
        self.bench["hotpotqa"] = HotpotQADistract(split="test", seed=seed, setting="distractor")

        # id -> row lookups (keyed exactly as the standalone runs / the fused cache)
        self.rows = {}
        self.rows["ds1000"] = {
            int(r["metadata"]["problem_id"]): r
            for r in self.bench["ds1000"].get_dataset()
            if r.get("metadata", {}).get("library") != "Tensorflow"}
        self.rows["ddxplus"] = {i: r for i, r in enumerate(self.bench["ddxplus"].get_dataset())}
        self.rows["hotpotqa"] = {i: r for i, r in enumerate(self.bench["hotpotqa"].get_dataset())}

    # ── per-item accessors (dispatch by dataset) ──
    def system(self, dataset):
        return SYSTEM_PROMPT[dataset]

    def row(self, dataset, local_id):
        return self.rows[dataset][local_id]

    def gen_prompt(self, dataset, row):
        return self.bench[dataset].get_input(row)["prompt_zeroshot"]

    def question(self, dataset, row):
        # text used for kNN-ICL embedding
        return self.bench[dataset].get_input(row)["question"].strip()

    def postprocess(self, dataset, text):
        return self.bench[dataset].postprocess_generation(text)

    def is_correct(self, dataset, pred, row, step):
        """PER-DATASET oracle -> 0/1. pred = postprocess(text)."""
        b = self.bench[dataset]
        if dataset == "ds1000":
            ref = b.get_output(row)
            res = b.process_results(pred, ref, return_details=True, time_step=step, simulate_env=True)
            return int(bool(res["correct"]))
        if dataset == "ddxplus":
            return int(pred == b.get_output(row))
        if dataset == "hotpotqa":
            return int(compute_exact_match(pred, b.get_output(row)))
        raise ValueError(dataset)

    def target_text(self, dataset, pred):
        """Self-generated correct-answer TEXT stored in mem_bank (raw, matches the
        standalone runners: ds1000 -> code; ddx/hotpot -> get_label_text(pred))."""
        if dataset == "ds1000":
            return pred                                   # the code string
        return self.bench[dataset].get_label_text(pred)   # ddx: "N. diag"; hotpot: {"answer": ...}

    def demo_answer(self, dataset, pred):
        """Formatted ASSISTANT message string for an ICL demo / SDFT teacher hint —
        rendered the way the model is asked to answer that dataset (code fenced for
        ds1000; label text for ddx/hotpot). This is what mem_bank stores as the demo."""
        if dataset == "ds1000":
            return f"```python\n{pred}\n```"
        return self.bench[dataset].get_label_text(pred)

    def gold_output_text(self, dataset, row):
        """An 'ideal' model output string that MUST score correct=1 — used only to
        validate dispatch + oracle wiring on CPU (no model needed)."""
        b = self.bench[dataset]
        if dataset == "ds1000":
            return f"```python\n{row['reference_code']}\n```"
        if dataset == "ddxplus":
            return b.get_label_text(b.get_output(row))    # "N. diag" -> postprocess -> N
        if dataset == "hotpotqa":
            return b.get_label_text(b.get_output(row))    # {"answer": ans} -> postprocess -> ans
        raise ValueError(dataset)


if __name__ == "__main__":
    # CPU validation: prove routing + oracle wiring — each dataset's ideal output scores 1.
    import pickle
    fb = FusedBench()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_stream_order.pkl"), "rb") as f:
        stream = pickle.load(f)["stream"]
    print(f"loaded fused stream: {len(stream)} items")
    seen = {}
    for step, item in enumerate(stream):
        d = item["dataset"]
        if d in seen:
            continue
        seen[d] = True
        row = fb.row(d, item["id"])
        ideal = fb.gold_output_text(d, row)
        pred = fb.postprocess(d, ideal)
        corr = fb.is_correct(d, pred, row, step)
        tgt = fb.target_text(d, pred)
        print(f"  [{d:9}] id={item['id']}  ideal->postprocess->correct={corr}  "
              f"target_text[:50]={str(tgt)[:50]!r}")
        if len(seen) == 3:
            # a few more of each to be safe
            pass
    # extra: 5 rows per dataset through the oracle
    print("\n5-per-dataset oracle check (all should be 1):")
    cnt = {d: 0 for d in fb.bench}
    for step, item in enumerate(stream):
        d = item["dataset"]
        if cnt[d] >= 5:
            continue
        cnt[d] += 1
        row = fb.row(d, item["id"])
        pred = fb.postprocess(d, fb.gold_output_text(d, row))
        print(f"  {d:9} #{cnt[d]}: correct={fb.is_correct(d, pred, row, step)}")
        if all(v >= 5 for v in cnt.values()):
            break
    print("DONE")
