# Jargon + framings harvest (verified Aug 2026)

## Anchor framings to adopt
1. **Era of Experience (Silver & Sutton 2025, DeepMind/MIT Press excerpt)** — exact phrases:
   "streams of experience" (LLMs "reset after each interaction"); "grounded rewards...
   rather than human prejudgement"; "any static procedure for synthetically generating
   data will quickly become outstripped" (= our online>batch result, stated as prophecy);
   "experiential agent" that "can continue to learn throughout a lifetime".
   → Position our paper as an existence proof of dimensions 1+3 (streams, grounded
   rewards) in today's LLM setting.
2. **Experience internalization vs accumulation** — the field's bifurcation:
   component-level evolution (memory/tools/skill libraries: ReasoningBank, AgentFly,
   Memp, JitRL — "without weight updates") vs policy-level/parametric internalization
   (SEAL, continual post-training, US). Our kNN self-segregation + mastery result =
   direct empirical argument for internalization.
3. **Prequential evaluation / input-feedback sequence / streaming scenario** —
   StreamBench's adopted vocabulary; use it verbatim.
4. **On-policy distillation sweet spot** (Thinking Machines 2025): "RL's error-correction
   relevance at SFT's reward density" — community-legible one-liner for why our channel
   beats both sparse RL and off-policy SFT.
5. **RLVR + failure modes**: verifier gaming/reward hacking (2604.15149, 2604.13602);
   "always-yes collapse" of self-judges (2607.05904); entropy collapse / rise-then-collapse
   self-training regression (2606.21090). → Verifiable binary feedback = what keeps
   deployment-time learning non-degenerate; our stability result must engage this.
6. **Data flywheel** (Agent-in-the-Loop, EMNLP 2025 Industry): "every serve is a gradient" —
   practical intro motivation.
7. **Self-generated curriculum / learnability** (Absolute Zero): moderate-difficulty tasks
   drive learning — echoes our emergent-frontier-curriculum account of H5.

## Glossary (use these terms of art)
prequential; input-feedback sequence; grounded rewards; verifiable rewards (RLVR);
programmatic oracle/verifier; outcome-only binary reward; on-policy data; self-teacher;
distribution gap; experience internalization; parametric vs non-parametric memory;
stability-plasticity tradeoff; continual post-training; catastrophic forgetting;
policy-level vs component-level evolution; deployment-time adaptation; test-time
learning; pseudo-rewards; entropy collapse; model collapse; data flywheel;
self-edits (SEAL); experiential agent; lifelong agentic systems.

## Surveys to cite (verified)
- Self-Evolving AI Agents survey (2508.07407, Aug 2025) — taxonomy: what/when/how to evolve; locate us: policy weights / online / self-distillation on verified feedback.
- Self-Evolving Agents What/When/How/Where (2507.21046)
- Lifelong Learning of LLM Agents roadmap (2501.07278)
- Agentic RL Landscape (2509.02547) — self-improvement as core capability
- CL of LLMs (2404.16789 → ACM CSUR 2025) [already in bib as shi2025continual]
- Adaptation of Agentic AI: Post-Training/Memory/Skills (2512.16301) — maps our weights-vs-memory axis
- Era of Experience (Silver & Sutton) — position-paper anchor

## New bib entries to add when cited
@misc{silver2025experience, title={Welcome to the Era of Experience}, author={Silver, David and Sutton, Richard S.}, howpublished={Google DeepMind; excerpt from \emph{Designing an Intelligence}, MIT Press}, year={2025}}
@article{fang2025selfevolving, title={A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems}, author={Fang, Jinyuan and Peng, Yanwen and others}, journal={arXiv preprint arXiv:2508.07407}, year={2025}}
(+ optionally 2509.02547 landscape survey, 2501.07278 roadmap)

## Where to apply
- Intro hook: streams-of-experience + frozen-at-deployment contrast (Silver&Sutton cite).
- Formulation: "input-feedback sequence", "prequential" (already), "grounded/verifiable".
- Related work: add internalization-vs-accumulation sentence + survey cites; RLVR failure-mode cites near stability discussion.
- Analysis Q4: "static data procedures get outstripped" quote-paraphrase for online>batch.
- Conclusion: era-of-experience echo.
