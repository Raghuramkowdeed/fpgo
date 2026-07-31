#!/usr/bin/env python3
"""Build a FUSED distribution-shift stream: 3 diverse StreamBench datasets
(DS-1000 code / DDXPlus diagnosis / HotpotQA multi-hop QA), n_per each, fused by a
RANDOM MERGE that PRESERVES each dataset's own within-dataset order. So the first
n_per problems of each dataset appear in the SAME relative order they had in that
dataset's STANDALONE run — enabling a direct standalone-vs-fused comparison
(same problems, same intra-dataset sequence, only diluted/interleaved by the others).
Cross-dataset interleaving is randomized seed-42. Every baseline replays this
identical fused sequence so cumulative regret is comparable across methods.

Per-dataset order = that dataset's standalone stream order:
  ds1000   -> ds1000_stream_order.pkl["problem_ids"] (non-TF), first n_per
  ddxplus  -> list(get_dataset()) index order, first n_per   (standalone used this)
  hotpotqa -> list(get_dataset()) index order, first n_per   (standalone used this)

Cache format (fused_stream_order.pkl):
  {
    "seed": 42, "n_per": N, "preserve_within_order": True,
    "datasets": ["ds1000", "ddxplus", "hotpotqa"],
    "stream": [ {"dataset": "ds1000", "id": <problem_id|index>}, ... ]   # merged
  }
"""
import argparse, os, pickle, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stream_bench.benchmarks.ds_1000 import DS1000
from stream_bench.benchmarks.ddxplus import create_ddxplus
from stream_bench.benchmarks.hotpotqa_distract import HotpotQADistract


def ds1000_ids(seed, stream_order="ds1000_stream_order.pkl"):
    # match the STANDALONE ds1000 run: its seed-42 stream order, non-TF only, in order
    b = DS1000(split="test", seed=seed, timeout=10.0)
    by_pid = {int(r["metadata"]["problem_id"]) for r in b.get_dataset()
              if r.get("metadata", {}).get("library") != "Tensorflow"}
    with open(stream_order, "rb") as f:
        order = pickle.load(f)["problem_ids"]
    return [int(pid) for pid in order if int(pid) in by_pid]

def ddxplus_ids(seed):
    DDX = create_ddxplus()
    b = DDX(split="test", seed=seed)
    return list(range(len(list(b.get_dataset()))))

def hotpotqa_ids(seed):
    b = HotpotQADistract(split="test", seed=seed, setting="distractor")
    return list(range(len(list(b.get_dataset()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per", type=int, default=300, help="questions drawn from EACH dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="fused_stream_order.pkl")
    args = ap.parse_args()

    loaders = {"ds1000": ds1000_ids, "ddxplus": ddxplus_ids, "hotpotqa": hotpotqa_ids}
    # per-dataset ORDERED queue = standalone order, first n_per (kept in order)
    queues = {}
    print(f"Building fused stream (n_per={args.n_per}, seed={args.seed}, preserve within-order)")
    for name, fn in loaders.items():
        ids = fn(args.seed)
        take = ids[:args.n_per]
        print(f"  {name:9}: available={len(ids):5d}  taking={len(take)}")
        if len(take) < args.n_per:
            print(f"    !! only {len(take)} available (< {args.n_per})")
        queues[name] = list(take)

    # RANDOM MERGE preserving within-dataset order: shuffle a bag of dataset TAGS,
    # then pop items from each dataset's ordered queue in tag order.
    rng = random.Random(args.seed)
    tags = [name for name, q in queues.items() for _ in q]
    rng.shuffle(tags)
    cursor = {name: 0 for name in queues}
    stream = []
    for t in tags:
        stream.append({"dataset": t, "id": queues[t][cursor[t]]})
        cursor[t] += 1

    cache = {"seed": args.seed, "n_per": args.n_per, "preserve_within_order": True,
             "datasets": list(loaders.keys()), "stream": stream}
    with open(args.out, "wb") as f:
        pickle.dump(cache, f)

    # sanity: composition + VERIFY within-dataset order preserved vs the queues
    from collections import Counter
    comp = Counter(s["dataset"] for s in stream)
    extracted = {name: [s["id"] for s in stream if s["dataset"] == name] for name in queues}
    ok = all(extracted[name] == queues[name] for name in queues)
    print(f"\nfused stream: {len(stream)} items -> {args.out}")
    print(f"  composition: {dict(comp)}")
    print(f"  within-dataset order preserved vs standalone: {ok}")
    print(f"  first 14 sources: {[s['dataset'][:4] for s in stream[:14]]}")


if __name__ == "__main__":
    main()
