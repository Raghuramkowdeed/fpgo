# The research narrative — hypotheses, questions, and what our experiments actually answered
(Reconstructed from the full experiment history; this is the paper's intellectual backbone.
Every experiments-section subsection should open by stating the hypothesis it tests.)

## H1 — Sufficiency of verified self-supervision
**Hypothesis.** A model's own oracle-verified outputs are sufficient training signal
for streaming improvement — no gold labels, no external teacher.
**Question.** Can weight updates driven only by the (question, verified self-answer)
pairs harvested from the stream beat both the frozen model and the strongest
memory-only method?
**Experiment.** 7 StreamBench tasks, all methods identical model/stream/decoding.
**Answer.** Yes, uniformly: best on all 7 (e.g., DS-1000 .423 vs ICL .367; DDXPlus
.664 vs base .436). Effect sizes largest where base is weakest — consistent with a
bootstrapping account (headroom = verifiable successes not yet consolidated).
**Rigor note.** Uniformity across 4 oracle types rules out oracle-specific artifacts.

## H2 — Dense supervision extracts more from the bit than policy gradients
**Hypothesis.** The binary verification bit, converted into token-level distillation
targets via the self-teacher, is a richer learning signal than the same bit used as
a scalar RL reward.
**Question.** At matched skeleton (rolling window, same stored completions, same
oracle budget), does REINFORCE++ match Online SDFT?
**Experiment.** Faithful REINFORCE++ (whitened advantages, PPO clip, KL anchor) on
all 7 tasks + the fused stream.
**Answer.** No: inconsistent across tasks (below base on Spider/DDXPlus) and
catastrophically divergent on the long fused stream (collapse at ~2k problems,
final .253). Same hparams stable on shorter streams → failure is horizon/mixture
specific, matching the RLVR entropy-collapse literature.
**Interpretation discipline.** We claim RL *as instantiated here* fails — not that no
RL variant could work (cf. ProRL's resets). The contrast is signal-density, not
algorithm-tuning.

## H3 — Retrieval and distillation are complementary channels, not substitutes
**Hypothesis.** Memory helps twice: as demonstrations now (retrieval) and as
supervision forever (distillation); the second channel works even where the first
fails.
**Question.** Do gains survive removal of all prompting scaffolding? Do they appear
where retrieval hurts?
**Experiments.** (a) HotpotQA case: ICL below base (.509 vs .521) yet SDFT above
(.562). (b) Mastery eval: bare-prompt greedy, no memory — adapter alone +16.7 over
base.
**Answer.** Both confirmed. The hint channel cannot mislead (it is the model's own
verified answer to the *same* problem), whereas neighbors can.

## H4 — Robustness to distribution shift; transfer is weight-borne
**Hypothesis.** One model can absorb a heterogeneous stream without interference,
and any cross-domain benefit must travel through weights (since retrieval
self-segregates).
**Question.** At matched per-domain prefixes, does the fused model lose to
specialists? Where does any difference come from?
**Experiment.** Order-preserving 3-domain merge (4,219) + matched-prefix
comparison + retrieval-composition audit + base/ICL controls (their prompts are
fusion-invariant).
**Answer.** No interference; positive transfer to code (+3.4); 99.9% same-domain
retrieval pins the mechanism to weights. Base/ICL fused≈standalone confirms the
design isolates training effects.
**Design pearl worth stating.** The order-preserving merge is itself a
methodological contribution: it makes "specialist vs generalist" comparisons exact
at the problem level.

## H5 — The schedule is causal: online > batch at identical loss/data/model
**Hypothesis.** Streaming (one problem at a time, rolling-window replay, re-eval)
is not an inconvenience to engineer around but a *source* of learning efficiency:
it induces an emergent curriculum (train at the current frontier) and converts
weight gains into new data (flywheel).
**Question.** Hold objective, data, model, oracle fixed; vary only the schedule.
**Experiment.** Batch SDFT: 4 epochs of generate→filter→distill over the same 955
problems. Mastery: online .480 vs batch .425. Coverage: online ends with verified
answers for 53.5% of problems vs batch 41.8%; solved-set overlap shows online's
edge is breadth (123 unique vs 71).
**Sub-question (pending).** Is the gap merely exploration? n=5 sampled batch
(coverage .565 in training) — eval queued. If n=5 closes mastery but not transfer,
the curriculum story sharpens further.
**Answer so far.** Schedule matters; coverage is the proximate mechanism.

## H6 — Stability by construction (anchored distillation)
**Hypothesis.** Distilling toward a frozen-base teacher on verified text is stable
where policy gradients are not, because nothing in the objective moves: no critic,
no advantage estimate, teacher fixed, targets verified.
**Evidence.** KL loss bounded in [1e-3, 1e-1] over ~12k streamed problems, zero
catastrophic regressions; RPP collapse diagnostics as the foil.
**Framing.** Our frozen teacher = ProRL's reference-reset safeguard made intrinsic.

## H7 — Negative results that sharpen the mechanism (report honestly)
- Multi-sample distillation (N=10, filtered or not): wash → per-problem sample
  count is not the bottleneck; *which problems* get a verified answer (coverage) is.
- Generation source (hint-conditioned vs bare): ±1 pt, bootstrap-phase only →
  results do not hinge on the hint entering generation; the hint's power is in the
  teacher's distribution.
- LR cannot substitute for epochs (+5 vs −21 regret) → consolidation needs
  repetition, consistent with the window-replay design.
These are the ablations-as-objection-killers; each kills one alternative
explanation of H1/H5.

## Open questions (state as such; some pending, some future work)
1. Does online-trained transfer better off-domain than batch-trained? (arms staged)
2. Does exploration (n=5) close the batch gap in mastery AND transfer? (eval queued)
3. General-capability retention (MMLU-style) after long streams — unmeasured.
4. Multi-seed variance; scale beyond 7B; richer-than-binary feedback.

## Writing conversion rules
- Each experiments subsection opens: "We test whether …" (the hypothesis), then the
  design in one sentence, then the answer with the exhibit.
- Use "consistent with H_k" style cross-references sparingly in analysis.
- Negative results get their own visibility (H7) — they are evidence of rigor.
- Never claim beyond the design: single-seed → "in our runs"; one model → "for a
  7B code-centric model".
