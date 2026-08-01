# Positioning memo C — distillation + RLVR (verified Aug 2026)

## Key works
| Work | Venue/Yr | ~Cites | Gap vs ours |
|---|---|---|---|
| GKD (Agarwal) | ICLR 2024 | 621 | external fixed teacher, offline batch |
| MiniLLM (Gu) | ICLR 2024 | undercounted (universal baseline) | reverse-KL w/ external teacher; motivates our forward-KL contrast |
| SDFT (Yang) | ACL 2024 | 107 | closest objective ancestor; offline, epoch-based, REQUIRES gold dataset |
| SDPO (Hübotter) | arXiv 2601.20802 | 195 | must-discuss: self-distillation densifies reward but ROLLOUT-BATCH RL, moving self-teacher; we= streaming, one pass, frozen-base anchor |
| On-policy distillation blog (Lu/TML) | 2025, DOI 10.64434/tml.20251026 | — | citable; external teacher, offline |
| SKD (Xu) | ICLR 2025 | 73 | interpolation axis, external teacher |
| InstructGPT (Ouyang) | NeurIPS 2022 | ~23000 | lineage cite for rollout-batch RL |
| DeepSeekMath/GRPO (Shao) | arXiv 2024 | 7882 | scalar reward, fixed corpus |
| DeepSeek-R1 (Guo) | **Nature 645:633, 2025** | 5460 | RLVR drives self-improvement at massive batch scale — we ask: during deployment, one example at a time? |
| RLOO (Ahmadian) | ACL 2024 | several hundred | REINFORCE-suffices ancestor |
| REINFORCE++ (Hu) | arXiv 2501.03262 (v9 retitled "Stabilizing Critic-Free...") | 83 | OUR baseline; its selling point IS stability → collapse on our stream is a strong result |
| Entropy Mechanism (Cui) | arXiv 2505.22617 | 399 | instability cite #1: entropy collapse intrinsic to RLVR |
| ProRL (Liu) | NeurIPS 2025 | 157 | instability cite #2: long-horizon RLVR needs KL control + periodic reference RESETS; our frozen-base teacher = implicit always-on anchor |
| TTRL (Zuo) | NeurIPS 2025 | 213 | test-time weights but batch RL + consensus pseudo-rewards |

## NOVELTY SEARCH RESULT (load-bearing)
No prior work found doing verified, weight-updating self-distillation ONLINE on a task stream:
- 2026 OPD survey (arXiv 2604.00626): no online/streaming-deployment category at all
- OPSD family (2604.03128, 2607.02234, 2607.05184): all batch RL-style
- StreamBench methods: memory only, never weights
- Nearest stream-weight-update works are SFT-style, not self-distillation: SAGE (2509.05385, trigger-driven LoRA on streams), aTTT (2607.03441, within-episode agent TTT), SCoL (2605.07076, context internalization)
→ Phrase novelty against 3 named near-misses: SDPO (self-distill, but batch RL), TTRL (test-time weights, but batch RL + pseudo-rewards), SAGE/aTTT (streaming weights, but SFT, no self-teacher).

## Sharpest sentence
"Prior work uses self-distillation to densify rewards inside rollout-batch RL (SDPO) or to protect the base distribution offline (SDFT); we show that distilling a frozen-base self-teacher conditioned on stream-verified answers turns self-distillation into a stable ONLINE LEARNING RULE — matching-loss batch training and stability-engineered RLVR (REINFORCE++) both fail where it succeeds."

## GKD intro skeleton
1 context+cost → 2 diagnosis: ONE precise pathology of the standard approach → 3 proposal = direct negation of the pathology (+1 degree of freedom) → 4 unification + 3 quantified wins → 5 contribution bullets.
Our para 2: "streaming self-improvement today is either prompt-memory (no consolidation) or rollout-batch RL (collapses on long non-stationary streams)"; para 4: 7/7 wins + shift-stability + online>batch.

## BibTeX deltas vs current refs.bib
- REPLACE yang2024selfdistillation body with ACL camera-ready (pages 1028-1043, doi 10.18653/v1/2024.acl-long.58)
- REPLACE hue2026sdpo → full author list Hübotter et al. (keep key or rename hubotter2026sdpo + update \citep)
- REPLACE hu2025reinforcepp body: title now "REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization", authors Hu, Liu, Xu, Shen
- REPLACE ahmadian2024rloo: title "Back to Basics: Revisiting REINFORCE-Style Optimization..." ACL pages 12248-12267
- ADD: gu2024minillm, xu2025skd, ouyang2022instructgpt, guo2025deepseekr1 (Nature), cui2025entropy, liu2025prorl, lu2025onpolicy (misc w/ DOI)
- Caveats: REINFORCE++ unrefereed (cite arXiv, note retitle); SDPO arXiv-only but 195 cites — must-discuss.
