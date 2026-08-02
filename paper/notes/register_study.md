# Prose register study (from SCoRe/GKD/SIFT/SKD camera-readies + STaR/Reflexion; Aug 2026)
(Full annotated version in the agent transcript; this is the working distillate.)

## Abstract register invariants (violations = amateur)
1. Problem stated as a mechanism or a COST, never a tragedy. No "indefinitely", no drama.
2. Gap = taxonomy of prior-work requirements/failures, each branch concrete, hedged with "typically/often".
3. Gap -> proposal DIRECTLY. No separate "we study how..." framing sentence.
4. Acronyms expanded properly ("Generalized Knowledge Distillation (GKD)") — never circular.
5. EXACTLY one "Concretely,"-grade mechanism sentence: the recipe skeleton (no hyperparams, no plumbing).
6. Differentiator triple as its own sentence (SCoRe: "entirely self-generated data").
7. Every generality adjective cashed out by an enumeration ("diverse tasks (X, Y, Z)") or deleted.
8. Named benchmark suite + named baselines + at least one NUMBER WITH ITS COMPARATOR (91% vs 80%).
9. At most ONE antithesis, and only if structural (it IS the method), never rhetorical flourish.
10. Closer = one flat takeaway (STaR style). No "substrate/byproduct" poetry, no ring composition.

## Intro-opening invariants
- Breadth budget: ONE understated broad sentence, carrying citations ("a useful tool", not "revolutionized").
- Gap arrives by sentence ~3 ("However, ...").
- Terminology defined inline via parentheses, never own sentences.
- EVERY negative claim about the field is cited — never on our own authority.
- Paragraph 2 = feasibility argument with a concrete illustrative example, stacked hedges, ends converting puzzle into demand.
- Taxonomies of prior work: per-branch citation AND per-branch weakness in the same breath.
- Anchor our problem to an established literature (imitation-learning move: cite the 1991 lineage).
- Forward-reference our own evidence even in the intro ("e.g., Figure 3").

## Results-narration pattern
1. Pointer first, no claim in the pointer ("Results are shown in Table 2").
2. One qualitative ordinal claim licenses the numbers that follow (claim -> evidence, never number-heap).
3. "Notably," marks THE most important fact, with a strengthening qualifier or external anchor
   (e.g., "comparable to the gap between GPT-3.5 and GPT-4").
4. Comparisons always PAIRED and directional: vs base AND vs next-best, both numbers.
5. ~One interpretive clause per number; interpretation mechanistic, never evaluative ("demonstrates
   scalability w.r.t. student capacity", not "impressive").
6. Consistency across settings narrated as evidence ("aligns with our findings on XSum").

## Verdicts on our old abstract (lessons)
- "internalize its experience" = metaphor doing a mechanism's job. CUT the framing sentence.
- "Online SDFT, an online self-distillation method" = circular acronym. Expand properly.
- Mechanism sentence was MISSING — our distinctive recipe (memory -> hints -> frozen-base teacher ->
  forward-KL -> LoRA) must appear, once, at skeleton level.
- Zero numbers + unnamed suite = "insecure or unfinished".
- "not merely feasible but advantageous" = essayistic antithesis. The matched-data result IS the claim.

## Applied: abstract replaced with the study's grounded rewrite (2026-08-01).
Next applications: intro openings re-audited against invariants; results subsections re-narrated
per the 6-step pattern (pointer/claim/notably/paired-comparison discipline).
