# 1-oracle/q comparison on full LCB (n=1055)

| Method | avg R | cumreg | oracle/q (train) | RM calls/q (train) |
|---|---:|---:|---:|---:|
| Base 0-shot | 0.3377 | 698.77 | 0 | 0 |
| ICL k=3 (stream) | 0.4242 | 607.50 | 0 | 0 |
| RLOO + Nemotron-RM | 0.4627 | 566.85 | 0 | 0 oracle / 200 RM |
| ICL+SDFT (ours) | 0.4396 | 591.19 | 1 | 0 |
