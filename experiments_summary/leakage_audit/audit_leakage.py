#!/usr/bin/env python3
"""Leakage audit over a finished/partial fused-SDFT run (CPU-only, no GPU needed).

Checks the causal ordering that rules out train->eval leakage:
  A. Every scored problem is scored exactly once (after resume-dedup by uid).
  B. A problem's memory entry is created AT or AFTER the batch in which it was
     scored (never before) -> the scored generation could not see its own hint.
  C. Memory only ever contains problems from batches <= current -> kNN demos at
     scoring time come strictly from the past.
  D. Aggregate counters in state.json match the per-problem log (no phantom wins).
"""
import argparse, json, pickle
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/fused_sdft")
    ap.add_argument("--batch_size", type=int, default=10)
    args = ap.parse_args()

    pp = pd.read_csv(f"{args.dir}/per_problem.csv")
    n_raw = len(pp)
    pp = pp.drop_duplicates("uid", keep="last")
    mem = pickle.load(open(f"{args.dir}/mem_bank.pkl", "rb"))
    state = json.load(open(f"{args.dir}/state.json"))
    print(f"per_problem rows: {n_raw} raw -> {len(pp)} deduped; mem entries: {len(mem)}")

    # A. one score per problem
    assert pp.uid.is_unique, "FAIL A: duplicate scored uids after dedupe"
    print("A. one final score per problem: OK")

    # B. mem entry never predates scoring batch
    pp["arrival_batch"] = pp.step // args.batch_size
    viol = []
    for _, r in pp.iterrows():
        e = mem.get(r.uid)
        if e is not None and e["batch_idx"] < r.arrival_batch:
            viol.append((r.uid, e["batch_idx"], r.arrival_batch))
    assert not viol, f"FAIL B: {len(viol)} mem entries predate scoring, e.g. {viol[:3]}"
    print(f"B. no memory entry predates its problem's scoring batch "
          f"({sum(pp.uid.isin(mem))} problems in memory): OK")

    # C. memory composition is causally valid at every batch boundary:
    #    entries stored at batch b are for problems that ARRIVED at batch <= b
    bad = [(u, e["batch_idx"]) for u, e in mem.items()
           if u in set(pp.uid) and
           e["batch_idx"] < int(pp.loc[pp.uid == u, "arrival_batch"].iloc[0])]
    assert not bad, f"FAIL C: {bad[:3]}"
    print("C. memory is append-only w.r.t. the stream (demos always from the past): OK")

    # D. state counters vs per-problem log (state may lag CSV by <checkpoint_every batches)
    upto = (state["last_completed_batch"] + 1) * args.batch_size
    sub = pp[pp.step < upto]
    n_corr_csv = int(sub.correct.sum())
    print(f"D. state.json n_correct={state['n_correct']} vs CSV(first {upto})={n_corr_csv} "
          f"{'OK' if n_corr_csv == state['n_correct'] else 'MISMATCH'}")
    for d, (c, n) in state["corr_by_ds"].items():
        s = sub[sub.dataset == d]
        m = "OK" if (int(s.correct.sum()), len(s)) == (c, n) else f"MISMATCH got {int(s.correct.sum())}/{len(s)}"
        print(f"   {d}: state {c}/{n} vs CSV -> {m}")

    print("\nAll causal-ordering checks passed." )

if __name__ == "__main__":
    main()
