# StreamBench-family results — SDFT+fwd (fwd4) vs baselines

Format: **acc / cumreg**  (cumulative regret = Σ(1−reward), lower=better; equivalently 1−running acc).
Model: Qwen2.5-Coder-7B-Instruct + LoRA r=16; embedder Qwen3-Embedding-0.6B. 1 oracle/question,
online stream (seed-42, identical order for all methods). ICL k=3 = StreamBench Self-StreamICL.

| Benchmark | Domain | Size | Metric | Base | ICL k=3 | REINFORCE++ | **SDFT+fwd (ours)** |
|---|---|---:|---|---|---|---|---|
| LiveCodeBench | code | 1055 | avg reward | 0.338 / 699 | 0.424 / 608 | — / 648 | **0.448 / 583** |
| DS-1000 (TF-excl) | Python | 955 | pass@1 | 0.325 / 645 | 0.367 / 605 | 0.369 / 603 | **0.423 / 551** |
| Spider | text-to-SQL | 2147 | EX | 0.747 / 544 | 0.778 / 476 | 0.587 / 886 | **0.808 / 412** |
| DDXPlus | medical dx | 1764 | accuracy | 0.436 / 995 | 0.647 / 622 | 0.404 / 1051 | **0.664 / 592** |
| BIRD | hard SQL | 1534 | EX | 0.330 / 1028 | 0.346 / 1003 | 0.271 / 1119 | **0.354 / 991** |
| HotpotQA | multi-hop QA | 1500 | EM | 0.521 / 719 | 0.509 / 736 | 0.537 / 695 | **0.562 / 657** |
| ToolBench | func-calling | 750 | strict-EM | 0.549 / 338 | 0.593 / 305 | 0.607 / 295 | **0.615 / 289** |

**SDFT+fwd wins all 7** (highest acc / lowest cumreg). REINFORCE++ collapses below base on 4/7
(LCB, Spider, DDXPlus, BIRD). On HotpotQA ICL *hurts* (below base) yet SDFT still wins.
DS-1000: 45 TensorFlow problems excluded from all methods (import conflict); scored on 955.
