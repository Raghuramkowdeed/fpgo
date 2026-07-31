#!/usr/bin/env python3
"""Build the FULL-data fused stream as a PREFIX-EXTENSION of the small (n_per=300)
one, so the full SDFT run can RESUME from the small run's batch-90 checkpoint instead
of recomputing the first 900. Guarantees:
  full.stream[:900] == small.stream  (byte-identical order)  -> resume-safe
Then appends every dataset's REMAINING problems ([n_per:] of its standalone order),
merged seed-42 with within-dataset order preserved.
"""
import os, pickle, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fused_stream import ds1000_ids, ddxplus_ids, hotpotqa_ids

SEED = 42
SMALL = "fused_stream_order.pkl"
OUT = "fused_stream_order_full.pkl"

def main():
    with open(SMALL, "rb") as f:
        small = pickle.load(f)
    prefix = small["stream"]; n_per = small["n_per"]
    print(f"prefix (small): {len(prefix)} items, n_per={n_per}")

    loaders = {"ds1000": ds1000_ids, "ddxplus": ddxplus_ids, "hotpotqa": hotpotqa_ids}
    # sanity: the prefix's per-dataset ids must equal each loader's first n_per
    tails = {}
    for name, fn in loaders.items():
        full = fn(SEED)
        head = [s["id"] for s in prefix if s["dataset"] == name]
        assert head == full[:n_per], f"{name}: prefix head != standalone first {n_per}"
        tails[name] = full[n_per:]
        print(f"  {name:9}: full={len(full):5d}  tail(appended)={len(tails[name])}")

    # merge tails preserving within-dataset order (seed-42 tag shuffle)
    rng = random.Random(SEED + 1)   # different seed so tail interleave != prefix pattern
    tags = [name for name, t in tails.items() for _ in t]
    rng.shuffle(tags)
    cur = {name: 0 for name in tails}
    stream = list(prefix)
    for t in tags:
        stream.append({"dataset": t, "id": tails[t][cur[t]]}); cur[t] += 1

    # verify: prefix identity + within-dataset order preserved across the WHOLE stream
    assert stream[:len(prefix)] == prefix, "prefix broken!"
    for name, fn in loaders.items():
        got = [s["id"] for s in stream if s["dataset"] == name]
        assert got == fn(SEED)[:len(got)], f"{name}: within-order broken"

    cache = {"seed": SEED, "n_per": None, "preserve_within_order": True,
             "extends": SMALL, "prefix_len": len(prefix),
             "datasets": list(loaders.keys()), "stream": stream}
    with open(OUT, "wb") as f:
        pickle.dump(cache, f)
    from collections import Counter
    print(f"\nfull fused stream: {len(stream)} items -> {OUT}")
    print(f"  composition: {dict(Counter(s['dataset'] for s in stream))}")
    print(f"  prefix-identical first {len(prefix)}: OK  (resume-safe)")

if __name__ == "__main__":
    main()
