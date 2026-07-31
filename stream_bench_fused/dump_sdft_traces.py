#!/usr/bin/env python3
"""Generate human-readable SDFT trace flows + raw JSONL for leakage inspection.

FAITHFULNESS: prompts are built by importing the EXACT functions the training run
used (`build_student_messages` / `build_teacher_messages` / `MemBank.retrieve` from
run_fused.py, `SYSTEM_PROMPT` from fused_bench.py) applied to the run's own artifacts
(mem_bank.pkl, per_problem.csv). Demo retrieval is restricted to memory entries
stored BEFORE the traced problem's batch, so retrieval is historically valid.
Known approximation: memory keeps only each problem's LATEST verified answer, so a
demo's answer text may be a later refresh of the same problem (structure unchanged).

Outputs (in --run_dir):
  sdft_trace_flows.md      readable per-problem lifecycle walkthroughs
  sdft_traces_sample.jsonl raw chat-format messages, one record per traced problem
"""
import argparse, json, os, pickle, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from run_fused import build_student_messages, build_teacher_messages, MemBank
from fused_bench import SYSTEM_PROMPT

CLIP = 900  # chars per message shown in the .md (JSONL is never clipped)

def show(text, clip=CLIP):
    text = text.strip()
    if len(text) <= clip: return text
    return text[:clip] + f"\n[... {len(text)-clip} more chars — full text in sdft_traces_sample.jsonl]"

def fmt_msgs(msgs, clip=CLIP):
    return "\n".join(f"[{m['role'].upper()}]\n{show(m['content'], clip)}\n" for m in msgs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/fused_sdft")
    ap.add_argument("--per_dataset", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=10)
    args = ap.parse_args()

    with open(f"{args.run_dir}/mem_bank.pkl", "rb") as f:
        raw = pickle.load(f)
    pp = pd.read_csv(f"{args.run_dir}/per_problem.csv").drop_duplicates("uid", keep="last").set_index("uid")

    rng = random.Random(0)
    by_ds = {}
    for u, e in raw.items():
        # trace only problems we can fully narrate: scored + enough history for demos
        if u in pp.index and e["batch_idx"] >= 15:
            by_ds.setdefault(e["dataset"], []).append(u)

    md = [
        "# SDFT trace flows — how each training pair was built (leakage audit)",
        "",
        "**How this file was generated** (`stream-bench/dump_sdft_traces.py`): every prompt",
        "below is built by the *training code itself* — `build_student_messages`,",
        "`build_teacher_messages`, and `MemBank.retrieve` imported from `run_fused.py`,",
        "applied to the run's own artifacts (`mem_bank.pkl`, `per_problem.csv`). Demo",
        "retrieval is restricted to memory entries stored before the traced problem's",
        "batch, so retrieval is historically valid. Long texts are clipped for reading;",
        "`sdft_traces_sample.jsonl` has the same records unclipped.",
        "",
        "**The flow for every problem in the stream:**",
        "1. Problem arrives -> kNN demos retrieved from memory (past problems only)",
        "2. Model answers from the STUDENT prompt (no hint exists for this problem)",
        "3. Oracle grades that answer 0/1 -> THIS is the recorded benchmark score, final",
        "4. Only if correct: the answer is stored in memory (reward filter)",
        "5. For the next ~9 batches the problem is in the training window: the TEACHER",
        "   prompt = student prompt + the stored self-answer appended as a hint; forward",
        "   KL pulls the student's no-hint distribution toward the teacher's",
        "",
        "Gold labels exist only inside the oracle; only the 0/1 grade ever leaves it.",
        "",
    ]
    jl = open(f"{args.run_dir}/sdft_traces_sample.jsonl", "w")

    for ds in sorted(by_ds):
        picks = rng.sample(by_ds[ds], min(args.per_dataset, len(by_ds[ds])))
        for u in picks:
            e = raw[u]
            row = pp.loc[u]
            arrival_batch = int(row.step) // args.batch_size
            # historically-valid retrieval via the training code path
            mb = MemBank()
            mb.entries = {u2: e2 for u2, e2 in raw.items() if e2["batch_idx"] < arrival_batch}
            demos = mb.retrieve(e["embedding"], 3, exclude_uid=u)
            demo_uids = [u2 for u2, e2 in mb.entries.items() if any(e2 is d for d in demos)]
            stu = build_student_messages(SYSTEM_PROMPT[ds], e["gen_prompt"], demos)
            tea = build_teacher_messages(SYSTEM_PROMPT[ds], e["gen_prompt"], demos, e["answer"])
            assert tea[:len(stu)] == stu and len(tea) == len(stu) + 2
            assert u not in demo_uids

            md += [
                f"\n---\n\n## {u} ({ds})",
                "",
                f"**Step 1 — arrival** at stream position {int(row.step)} (batch {arrival_batch}).",
                f"Memory at that point held only problems from batches < {arrival_batch}.",
                "",
                f"**Step 2 — kNN retrieval** picked 3 past problems as demos: {demo_uids}",
                "(their stored answers are the model's own earlier verified outputs).",
                "",
                "**Step 3 — student prompt** (what the model answered from — no hint for this problem exists anywhere):",
                "", "```", fmt_msgs(stu), "```", "",
                f"**Step 4 — oracle grade of the arrival answer: {'CORRECT (1)' if row.correct==1 else 'WRONG (0)'}** — recorded as the benchmark score, final.",
                "",
                (f"**Step 5 — stored in memory** (reward filter passed). Stored answer:"
                 if row.correct == 1 else
                 f"**Step 5 — NOT stored at arrival** (failed reward filter); entered memory later at batch {e['batch_idx']} when window re-eval with the improved model produced a correct answer:"),
                "", "```", show(e["answer"], 500), "```", "",
                "**Step 6 — teacher prompt for training** = the student prompt above **plus** these two turns (the only difference):",
                "", "```",
                fmt_msgs(tea[len(stu):], 500),
                "```",
                "The appended assistant turn is the stored self-answer from step 5 — the",
                "model's own output, already scored. The student never sees it; forward KL",
                "teaches the student to match the teacher's hint-informed next-token",
                "distribution without the hint.",
            ]
            jl.write(json.dumps({
                "uid": u, "dataset": ds, "arrival_step": int(row.step),
                "arrival_batch": arrival_batch, "arrival_correct": int(row.correct),
                "stored_at_batch": e["batch_idx"], "demo_uids": demo_uids,
                "student_messages": stu, "teacher_messages": tea, "hint": e["answer"],
                "hint_provenance": "model's own earlier oracle-passing answer (self-generated; gold labels exist only inside the 0/1 oracle)",
            }) + "\n")

    jl.close()
    with open(f"{args.run_dir}/sdft_trace_flows.md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote {args.run_dir}/sdft_trace_flows.md and sdft_traces_sample.jsonl")

if __name__ == "__main__":
    main()
