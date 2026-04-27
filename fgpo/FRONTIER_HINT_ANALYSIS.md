# Frontier-Hint Cache Analysis

Cache: `agl/fgpo/cache/hints_frontier.pkl`
Source: `claude-sonnet-4-6` hinting `Qwen2.5-Coder-7B-Instruct`, up to 3 oracle rounds (probe → hint → re-probe).
Coverage: 300 problems = first 600 train problems × `frontier_fraction=0.5`.

## 1. Aggregate impact of hints

| | Mean reward | Distribution |
|---|---:|---|
| Baseline (no hint) | 0.339 | 98 zero, 33 already=1.0, rest partial |
| Hinted | 0.679 | 24 zero, 127 =1.0, rest partial |
| **Δ** | **+0.340** | |

- Helped: 218/300 (72.7%)
- Hurt: 28/300 (9.3%)
- Neutral: 54/300 (18.0%)
- Big wins (Δ ≥ 0.5): 100
- "Saved" (b<0.5 → h=1.0): 57
- Stuck even with hints (h ≤ 0.1, b ≤ 0.1): **39**

## 2. Multi-round refinement loop

Each problem: round 0 = baseline, rounds 1–2 = hint attempts. Stop early if reward = 1.0.

| | n | Mean reward |
|---|---:|---:|
| Round 0 (no hint) | 300 | 0.339 |
| Round 1 (first hint) | 300 | 0.576 |
| Round 2 (refined hint) | 189 | 0.477 |

- Solved with one hint round: 111 (97 perfect, 14 partial) → stopped at round 1
- Required refinement: 189; refinement improved 107 (57%) of those
- Refinement never made things worse (cache stores max-over-rounds)
- Stayed at zero across 2 hint rounds: 24

## 3. What worked

| Pattern | n | Δ |
|---|---:|---:|
| Hints 300–600 chars | 122 | **+0.464** ← sweet spot |
| Hints <300 chars (already-easy problems) | 75 | +0.303 |
| Hints 600–1000 chars | 61 | +0.342 |
| Lead with "Key insight:" | 87 | +0.320 |
| Mention binary search | 20 | +0.360 |
| Mention graph algorithms | 17 | +0.357 |
| Mention sort | 53 | +0.294 |
| Mention DP | 42 | +0.270 |

Concrete 0 → ~1 saves:
- `1899_A` (game theory): explicit invariant — "Vanya can reach mult-of-3 in 1 move iff `n%3≠0`".
- `3605`, `3264`, `3384` (LeetCode): function-name reminder + concrete algorithm steps.
- `abc371_g` (cycles in permutation): cycle-following recipe with index transformation.

## 4. What didn't work

| Pattern | n | Δ |
|---|---:|---:|
| **Hints >1000 chars** | 42 | **+0.043** ← collapses |
| Mentions "tree" (often profile DP / hard) | 21 | +0.035 |
| Hints starting with meta-commentary ("the model has no idea...") | many | ≈ 0 |

Failure modes:
- **Verbose hints lose the actionable line**: >1000 chars, baseline 0.146 → hinted 0.188 (essentially no lift).
- **Frontier-confused hints**: when the hint preface admits the frontier is reasoning aloud (e.g. `arc190_c`, `arc189_b`, `abc376_g`, `abc314_e`, `abc391_g`), the result almost always stays at 0.
- **Hint-induced regression on strong baselines** (28 cases): a hint with a buggy edge-case can drop a 0.9 baseline. Examples: `abc374_b` 0.90→0.73 (hint asserted an off-by-one length formula); `arc186_a` 0.35→0.05 (meta-commentary leaked into hint body).
- **Hard contest problems** (late-letter arc/abc — profile DPs, expected-value DPs, complex combinatorics): the 24 hard-zeros are concentrated here.

## 5. Does the trained model emit hint-style prose without a hint?

Inspected 3000 round-0 (no-hint) and 3000 round-1 (with-hint) cached responses for hint-style prose markers (`Key insight:`, `Notice that`, `Step N`, `Observe`, `The algorithm is`, etc.).

| | Prose before ```python | Prose after closing ``` | Hint markers anywhere |
|---|---:|---:|---:|
| No-hint responses | 0.0% | 2.8% | 0.03% (1/3000) |
| With-hint responses | 0.0% | 1.3% | 0.03% (1/3000) |
| With-refined-hint responses | 0.0% | 2.3% | 0.2% (4/1890) |

Conclusion: the system prompt (`"Output ONLY the code... Do NOT include explanations"`) is honored strongly. The few markers that surface (`# Step 1: Build the tree`, etc.) appear **inside code comments**, never as standalone prose. **Hints change the algorithm chosen, not the commentary around it.**

Caveat: these are training-time probe samples (model state evolving). For final post-trained model behavior on test problems, run `inspect_trained_gens.py` on `lora_epoch_1`.

## 6. Practical takeaways

1. **Hint length is the strongest knob.** 300–600 chars → +0.46; >1000 chars → +0.04. Constrain the oracle to terse hints.
2. **One hint round is usually sufficient or never sufficient.** 87% of solvable-by-1-hint problems hit perfect on the first attempt; the refinement bucket converges slowly and 24 never converge.
3. **Skip hinting when baseline is already strong** (e.g. `baseline_avg_reward ≥ 0.8`). Removes 28 hint-induced regressions and ~33 cache-bloat entries.
4. **The 24 hard-zero problems contribute zero training signal in RLOO.** Filtering them from the frontier subset would speed step-1 without quality loss.
5. **Hints transfer as algorithm choice, not as prose.** The trained model produces clean code-only outputs. So the value of hinting is steering the *solution space*, not teaching a verbalized chain-of-thought.

## Reproduction

```python
import pickle
with open('agl/fgpo/cache/hints_frontier.pkl','rb') as f:
    d = pickle.load(f)
# Each entry: problem_id, question, hint, baseline_avg_reward,
# hinted_avg_reward, n_rounds, rounds=[{hint, avg_reward, n_pass, samples, shown_to_frontier}, ...]
```

`samples[i].code` = full model response (round 0 raw, round ≥1 hinted).
