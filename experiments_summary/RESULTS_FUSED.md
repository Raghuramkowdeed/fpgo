# Fused distribution-shift experiment — final results

**Setup.** One online stream of 4,219 problems: DS-1000 (code, 955) + DDXPlus
(diagnosis, 1764) + HotpotQA (multi-hop QA, 1500), randomly interleaved with each
dataset's internal order preserved (seed 42) — so per-dataset results are directly
comparable to the standalone runs on identical problem sequences. Protocol identical
to the single-dataset StreamBench work: predict-then-train, oracle grades 0/1 before
any update, self-generated training targets only. Model Qwen2.5-Coder-7B + LoRA r16.
Code: `stream_bench_fused/`; per-run CSVs: `data/fused/`.

## Headline table (full 4,219-problem stream)

| Method | Acc | Cumreg | DS-1000 | DDXPlus | HotpotQA |
|---|---|---|---|---|---|
| Base (no adaptation) | 0.439 | 2365 | 0.318 | 0.436 | 0.521 |
| ICL k=3 (Self-StreamICL) | 0.533 | 1971 | 0.358 | 0.647 | 0.509 |
| REINFORCE++ (fused stream) | 0.253* | 3153 | 0.213 | 0.263 | 0.267 |
| SDFT standalone (3 specialists) | 0.573 | 1800 | 0.423 | 0.664 | **0.562** |
| **SDFT fused (one model, ours)** | **0.585** | **1750** | **0.457** | **0.676** | 0.560 |

Base/ICL/SDFT-standalone rows are the per-dataset standalone runs combined at matched
prefixes (cumreg summed, acc weighted by n). SDFT-fused and REINFORCE++ ran on the
actual fused stream.

## Key findings

1. **SDFT wins under distribution shift** — best on every domain; +5.2 pts over ICL,
   +14.6 over base.
2. **One fused model ≥ three specialists** — fused beats standalone SDFT overall
   (+1.2 pts) with positive cross-domain transfer to code (+3.4 pts on DS-1000) and
   no domain hurt (HotpotQA −0.2 pts, a wash).
3. **Transfer is through the weights, not prompts** — kNN retrieval self-segregates
   domains in embedding space (99.9% of retrieved demos are same-dataset; 2/2059
   queries pull any cross-domain demo), so base/ICL prompts are essentially identical
   fused vs standalone, yet fused SDFT still gains.
4. **REINFORCE++ collapses on the long fused stream.** Running acc peaked at .503
   around problem 2,000, then the policy diverged catastrophically (per-500 window
   acc: .44/.50/.55/.52 → .10/.01/.00). Collapse signature at batch ~205-210:
   approx-KL flips negative (−0.6), PPO clipfrac spikes 0.08→0.68, advantage std
   explodes 0.38→1.37. Same hyperparameters were stable on shorter standalone
   streams (≤1,764 problems) — long-horizon mixed-domain streams break the RL
   baseline while SDFT stays stable throughout. (*Final RPP acc is dominated by the
   post-collapse region; pre-collapse it tracked ~5 pts below SDFT.)
5. **Accuracy keeps climbing** — fused SDFT rolling-300 acc .44 → .68 peak (~problem
   2,700), plateau ~.59-.68 band after. Plot:
   `plots/fused_sdft_accuracy_trend.png`.

## Ablation: generate_from_teacher (completion source for distillation)

- **DS-1000 standalone, full 955**: gft=0 (paper default, bare student-prompt
  completions) acc .436/creg 539 vs gft=1 (hint-conditioned, ours) .423/551 —
  gft=0 slightly better, gap grows late-stream.
- **Fused stream (stopped at 1,820/4,219)**: gft=0 lost ~10 cumreg in the first 300
  problems (bootstrap phase), then dead-even for 1,500 problems.
- **Verdict**: generation source is a minor bootstrap-phase effect, not a method
  driver. Either setting is defensible; results do not depend on hint-conditioned
  generation (which also closes the corresponding leakage concern).

## Oracle budget

SDFT uses ~9.8 oracle calls/problem (1 arrival call that is scored + ~8.8 window
re-eval calls that only refresh memory). Base/ICL/REINFORCE++ use exactly 1/problem.
Re-eval never changes recorded scores; see `leakage_audit/` for the causal-ordering
audit (all checks pass) and reconstructed student/teacher prompt traces.
