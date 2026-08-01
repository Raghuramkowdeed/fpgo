# ICLR conventions spec (from 4 accepted ICLR 2024-2025 papers in our area; verified OpenReview)

Exemplars: SCoRe (ICLR25 ORAL — closest problem: LLM self-improving from binary correctness on self-generated data),
SIFT (ICLR25), GKD (ICLR24), SKD (ICLR25). SPIN is ICML not ICLR. PDFs in scratchpad (score_iclr/sift/gkd/skd.pdf).

## Structure to conform to (SCoRe template)
1 Introduction → 2 Related Work (or pre-conclusion; late is most common — KEEP OURS pre-conclusion)
→ 3 Preliminaries and Problem Setup (run-in \paragraph notation, NO theorem envs for empirical papers)
→ 4 analysis-or-method → 5 Method → 6 Experimental Evaluation (6.1 benchmarks, 6.2 ..., 6.3 Ablation Studies)
→ 7 Discussion, Limitations, and Conclusion (MERGED — no standalone Limitations)
→ unnumbered Reproducibility Statement → Acknowledgements → References (start p11).

## Hard rules
- Main text EXACTLY 10 pages; refs p11+; appendix 10-15pp (hyperparams, prompts, per-dataset tables, extra ablations, qualitative transcripts, extended related work OK).
- Figure 1 = HEADLINE RESULTS on page 1 (not a schematic!). SCoRe: two-panel results. Method schematic = Fig 2/3.
- Main text: 5-10 figures, 0-4 tables; bar charts often preferred over tables (GKD has 0 tables).
- Abstract: single para, 165-255 words: problem → why existing falls short (1-2 sent) → "To address this, we introduce NAME" + mechanics (1-2 sent) → headline numbers with benchmark names.
- Contribution bullets optional (≤3 if used; SCoRe/SIFT use prose instead).
- \paragraph{} run-in headers everywhere (Tasks. Models. Baselines. Setup. Results.); never \subsubsection.
- Reproducibility Statement (~5-8 sentences): hyperparams appendix pointer, open benchmarks named, prompts appendix, code-release promise. SCoRe + SKD have it. MUST NOT MISS.
- Ethics Statement: optional in this area (none of the 4 have one).
- SCoRe device to adopt: gray "Takeaways:" boxes ending analysis subsections.

## Conformance TODO for our draft
- [x] Rename "Problem Formulation" → "Preliminaries and Problem Setup"
- [x] Merge Conclusion → "Discussion, Limitations, and Conclusion" with labeled limitations
- [x] Add unnumbered Reproducibility Statement before references
- [ ] Figure 1: build headline two-panel results figure (7-bench bars + fused trend), page 1; demote schematic idea
- [ ] Takeaways boxes for analysis Q1-Q4
- [ ] Page-budget audit once content settles (main ≤10pp)
- [ ] Move per-task oracle table / some baseline detail to appendix if over budget
