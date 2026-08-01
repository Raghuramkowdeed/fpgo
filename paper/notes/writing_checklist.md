# Master writing checklist — 16 checkable rules (synthesized from Foerster, Rush, Peyton Jones, Nanda, Black, Lipton&Steinhardt, Farquhar, ACL RR; Aug 2026)

## (a) Framing & claims
1. One-sentence thesis appears near-verbatim in abstract, intro, conclusion. If it needs "and", it's two papers.
2. Claims–evidence matrix: each contribution bullet → the figure/table/section proving it; each experiment → its claim. Unmatched rows = cut or add.
3. Calibrated wording: classify evidence (existence / systematic / suggestive), match the verb. Grep "first|significantly|novel|solves|understands" — each backed or deleted.
4. Speculation quarantined in a labeled discussion/limitations paragraph; results text states only what numbers show.

## (b) Introduction
5. One page, fixed skeleton: problem+why (ONE concrete example, no "field is huge" opener) → why hard / what prior work misses → insight in one sentence → contribution bullets with forward refs.
6. The nugget is quotable — an INSIGHT sentence, not an architecture description.
7. Page-1 completeness: claim + contributions + Figure 1 all visible by end of page 1.
8. Related work positions, never surveys: grouped by theme, each group ends "unlike X, we…"; present tense; nothing misrepresented.

## (c) Experiments
9. Signposted results: every subsection opens with the claim it tests + exactly what to look at. Headers+first-sentences alone reconstruct the argument.
10. Source-of-gains ablations; baselines tuned as hard as the method, identical settings, strongest available.
11. Statistical hygiene: seeds/variance where possible; random vs cherry-picked examples labeled; re-implementable setup.
12. Text–numbers audit: every result sentence checked against actual table/figure values.

## (d) Figures & tables
13. Figure 1 stands alone; caption says what to notice; vector, legible, colorblind-safe.
14. One message per table; headline number bolded, findable in <5s; captions self-contained (interpretation).

## (e) Style
15. The one-third cut after first full draft. Grep-fix: passive voice, tense switches, "allows to/enables/showcases", hedges; symbols defined before use; notation table; equations match code.

## (f) Pre-submission sweep
16. Rejection-trigger scan: overclaiming, unclear contributions, weak baselines, unexplained gains, speculation-as-conclusion, misrepresented related work, missing variance, decorative math, undefined terms, irreproducible setup. Preempt un-fixable ones in an honest limitations paragraph.

Key per-guide gems:
- Foerster: fractal skeleton (problem→hard→solution→verify) in every section; 1 outline line = 1 paragraph; Figure 1 on page 1.
- SPJ: contributions = refutable claims list that DRIVES the paper; related work after content; "credit is not like money".
- Nanda: readers keep ~3 sentences — choose them; abstract/intro/figures deserve half the total polish; red-team own evidence.
- Black: Goal→Problem→Solution religiously; the nugget ≠ architecture; simple framed via insight so it can't be called trivial.
- Farquhar: abstract formula (achievement / why / how / 2-evidence + striking number); results paragraphs tell exactly where to look; sloppy prose signals sloppy code.
- Lipton: label speculation; identify source of gains; no mathiness; no suitcase words.
