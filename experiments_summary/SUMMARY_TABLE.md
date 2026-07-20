# All results — Avg Accuracy (SDFT+fwd vs baselines)

Avg Acc = each benchmark's native accuracy metric (avg reward LCB, pass@1 DS-1000, EX Spider/BIRD,
accuracy DDXPlus, EM HotpotQA, strict-EM ToolBench). Higher = better. Qwen2.5-Coder-7B + LoRA r=16;
1 oracle/question; online seed-42 stream (identical order for all methods).

| Benchmark | Domain | Size | Base | ICL k=3 | REINFORCE++ | **SDFT+fwd (ours)** |
|---|---|---:|---:|---:|---:|---:|
| LiveCodeBench | code | 1055 | 0.338 | 0.424 | 0.386 | **0.448** |
| DS-1000 | Python | 955 | 0.325 | 0.367 | 0.369 | **0.423** |
| Spider | text-to-SQL | 2147 | 0.747 | 0.778 | 0.587 | **0.808** |
| DDXPlus | medical dx | 1764 | 0.436 | 0.647 | 0.404 | **0.664** |
| BIRD | hard SQL | 1534 | 0.330 | 0.346 | 0.271 | **0.354** |
| HotpotQA | multi-hop QA | 1500 | 0.521 | 0.509 | 0.537 | **0.562** |
| ToolBench | func-calling | 750 | 0.549 | 0.593 | 0.607 | **0.615** |

*SDFT+fwd has the highest Avg Acc on all 7 benchmarks.*
