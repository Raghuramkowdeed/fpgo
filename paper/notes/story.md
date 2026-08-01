# Argument map — thesis, claim tree, evidence links (the paper's spine)

## Central thesis (one sentence)
A deployed LLM can **permanently** improve itself **during** a single pass over a
problem stream, using only binary verifiable feedback — by distilling its own
verified answers into its weights — and this *online* self-distillation is more
effective, more robust, and more stable than the natural alternatives (memory-only
retrieval, scalar-reward RL, and offline batch self-training).

Everything in the paper must serve this sentence. Four load-bearing claims:

## Claim tree → evidence
C1 EFFECTIVE (it works, broadly)
   ↳ 7-benchmark table: best on all 7, 5 domains, 4 oracle types.
     Breadth is the point: not a task-specific trick. HotpotQA is the star
     exhibit (ICL hurts, we help) — proves the distillation channel is doing
     work that retrieval cannot.

C2 ROBUST (it survives realistic non-stationarity)
   ↳ Fused stream: one model ≥ 3 specialists, positive transfer to code.
   ↳ Built-in control: base/ICL fused≈standalone (their prompts are unchanged
     by fusion) — isolates the effect to training methods.
   ↳ Contrast: REINFORCE++ collapses at ~2k on the same stream (stable on all
     shorter streams) — long+mixed is exactly where the thin-signal method dies.

C3 WEIGHT-BORNE (the mechanism is in θ, not in the prompt)
   ↳ Retrieval self-segregation (99.9%): cross-domain transfer cannot be
     prompt-carried → must be weights.
   ↳ Mastery eval strips ALL scaffolding: adapter alone beats base +16.7.
   ↳ Together C3 kills the "it's just fancy ICL" reviewer objection.

C4 ONLINE-NESS MATTERS (the schedule is a contribution, not a detail)
   ↳ Same loss+data+model, batch epochs vs stream: online +5.5 mastery.
   ↳ Mechanism: coverage (online trains on 53.5% of problems vs batch 41.8%),
     curriculum (train at the frontier), repetition-under-change (window), late
     harvest (re-eval). Overlap analysis (123 vs 71 unique) shows breadth.
   ↳ PENDING: n=5 batch (exploration-matched) — does coverage close the gap?
     PENDING: transfer arms (hotpotqa warm-start) — does online-trained
     generalize better off-domain? These complete C4; \todo until landed.

C5 STABLE (why it doesn't blow up)
   ↳ Objective anatomy: frozen teacher, verified targets, bounded disagreement —
     no moving critic/advantage. Collapse diagnostics of RPP as the foil.
   ↳ Empirical: ~12k streamed problems, zero catastrophic regressions.

## Ablations = alternative-explanation killers (say this explicitly in the paper)
- "It's the oracle budget" → RPP at 1 call fails; batch at 4 calls fails; frozen
  re-eval recovers nothing. Budget only helps a method that learns.
- "It's the hint-conditioned generation" → gft ablation: paper-default equals or
  beats ours; not the driver.
- "It's multi-sample data" → N=10 wash; sample count not the driver.
- "More lr would do it" → epoch/lr: repetition, not step size.

## Known gaps / threats to validity (be honest, plan or acknowledge)
1. Single seed per run (StreamBench used 5 shuffles in appendix) → limitation
   paragraph + (optional future run) one reshuffled-seed replicate of a cheap task.
2. Single model, 7B, LoRA-only → limitations.
3. General-capability retention NOT measured (SDFT paper's forgetting suite:
   MMLU/safety/AlpacaEval). Our mastery eval shows small task-level forgetting
   (37 problems) but no general-capability eval. → candidate cheap experiment:
   run final fused adapter vs base on 2-3 OpenLLM tasks; else acknowledge.
4. Transfer/n5 results pending → \todo discipline.
5. Oracle budget asymmetry → already argued 3 ways; keep the honesty paragraph.

## Narrative arc for the intro (STaR template mapped)
1. Hook: deployed agents face streams with sparse verifiable feedback; models
   are frozen at deployment — every mistake is repeated forever.
2. Establish: self-training from verified own outputs works offline
   (STaR/ReST/RFT lineage) — but only BETWEEN deployments.
3. Dead-end A: offline loops need the corpus upfront + multi-epoch sweeps —
   impossible mid-stream. (SEAL/TTRL get closer but need repeated episodes /
   ungrounded rewards.)
4. Dead-end B: streaming alternatives either never touch weights (Self-StreamICL
   — bounded by frozen model, can even hurt) or use the 1-bit signal as RL —
   thin credit assignment, unstable on long streams (we show collapse).
5. The turn: verified self-outputs are exactly the dense supervision an online
   learner needs — hint-conditioned self-distillation converts each success into
   a token-level training signal.
6. The loop's failure mode motivates components: greedy-only successes stall
   coverage → forward re-eval (flywheel); demonstrations help now but don't
   persist → memory feeds BOTH retrieval and distillation.
7. Name method, Figure 1 (method schematic — TODO: create).
8. Headline numbers + contributions.

## Terminology discipline
\method{} = Online SDFT everywhere. "verified answer" (not "correct answer" when
provenance matters), "prequential" defined once then used, "mastery" defined at
first use, "oracle" g / "feedback bit" fb consistently.
