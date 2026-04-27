"""Split results_ablation_qwen/details.pkl into train (first 600) + test (rest)."""

import os
import pickle

IN_PATH = "/data/pulkitag/misc/raghuramkowdeed/projects/agl/results_ablation_qwen/details.pkl"
OUT_DIR = "/data/pulkitag/misc/raghuramkowdeed/projects/agl/fgpo/data"
N_TRAIN = 600


def main():
    with open(IN_PATH, "rb") as f:
        details = pickle.load(f)
    n = len(details)
    assert n > N_TRAIN, f"only {n} entries, need > {N_TRAIN}"

    os.makedirs(OUT_DIR, exist_ok=True)
    train = details[:N_TRAIN]
    test = details[N_TRAIN:]

    with open(os.path.join(OUT_DIR, "details_train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(OUT_DIR, "details_test.pkl"), "wb") as f:
        pickle.dump(test, f)

    # Also store id lists as JSON for easy inspection
    import json
    with open(os.path.join(OUT_DIR, "split_ids.json"), "w") as f:
        json.dump({
            "n_total": n,
            "n_train": len(train),
            "n_test": len(test),
            "train_ids": [d["problem_id"] for d in train],
            "test_ids": [d["problem_id"] for d in test],
        }, f, indent=2)

    print(f"total={n} train={len(train)} test={len(test)}")
    print(f"wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
