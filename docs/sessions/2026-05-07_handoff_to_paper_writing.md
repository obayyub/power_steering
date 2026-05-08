# 2026-05-07 — Handoff for paper-writing day

This note is for the next Claude session, which will begin on the user's
desktop on the morning of **May 7** (the workshop submission deadline,
11:59 UTC). Read this first.

## Who you're working with

- Medtech research engineer; spare-time researcher with three kids.
- This is their **first ML paper**. Workshop submission, double-blind.
- Style: terse, opinionated, doesn't want padded responses. They write
  fast under pressure. Don't pep-talk; do give honest gut checks when
  asked. Their `CLAUDE.md` is authoritative on workflow rules — read
  it before doing anything experimental (no inline `python -c`,
  experiments must come from saved files, ask before generating
  scripts).
- They're tired heading into deadline day. Be efficient with their
  attention.

## State of the paper as of end-of-session 2026-05-06 (laptop-time)

**Narrative is locked.** Numbers are in. Figures are rendered. The
remaining work is prose.

### What the paper says (per `docs/writeup_outline_v3.md`)

3-method × 7-eval cross-evaluation comparison on Qwen3-14B (Anthropic
advanced-AI-risk evals: corrigible-neutral-HHH, survival-instinct,
power-seeking, wealth-seeking, self-awareness, coordinate-other-ais,
myopic-reward). Methods compared:

- **CAA** (supervised baseline)
- **Power Steering / PI** (top-k right singular vectors of layer-to-
  layer Jacobian via block power iteration with Rayleigh-Ritz)
- **MELBO** (gradient ascent on L2 displacement, sphere constraint)
- DCT in **Appendix A only** (tracks MELBO closely; not a distinct
  datapoint)

**Headline findings:**

1. Layer-to-layer methods (PI, MELBO) ≈2× CAA's off-diagonal mean
   alignment shift (≈+16 pp vs ≈+7 pp). PI: +16.3 ± 9.6, MELBO:
   +16.9 ± 10.5, CAA: +6.6 ± 4.8.
2. **Logit-difference at extreme scales (±25) overshoots under
   sampled generation.** Same vectors at moderate scale (|scale|=10)
   preserve in-domain effect with much less cross-eval damage. PI's
   off-diagonal Δ goes -9.3 → -0.8 between the two scales.
3. Under generation + LLM-judge: **MELBO > PI ≈ CAA**. Regex aligned-%
   undercounts ~6.5% of cells; PI has the most regex→LLM flips
   (16-23 per disagreement-prone cell). LLM-judge moves PI numbers
   more than CAA or MELBO.
4. **Section 4** (atlas + AdvBench): PI's ~10× per-pair cost advantage
   makes a 560-pair (s,t) atlas tractable. AdvBench transfer of 7
   anti-refusal candidates shows the **#1 vector by logit-Δ produces
   0/10 genuine harm compliance** — first-token logit metrics track
   surface phrasing not content. The (24,37) v4+ Chinese-refusal
   vector is a clean decomposable-axis showcase (refusals in Chinese
   on English prompts).

### Section 3 cluster pattern (visible in Fig 2 misaligned)

Power-seeking-trained specialist boosts the **agentic cluster**
(wealth / self-aware / coord-other all 75-88%) and tanks
**corrigibility / myopic-reward** (mostly 10-30%). Survival-instinct
sits in the middle. This is in the LLM-judged
`gen_powerseek_specialist_judged.json` results (judge run finished
end of session).

## File map — what's load-bearing for paper writing

### Already done (don't redo)

- **`docs/writeup_outline_v3.md`** — the canonical outline. Section
  numbers, abstract draft, figure list. **Treat as the source of truth
  for what goes in the paper.**
- **`paper_artifacts/main.tex`** — existing blog post in ICML LaTeX
  format. Not the paper draft; it's the source for porting Section 1
  (intro + CAA/MELBO descriptions) and Section 2 (PS algorithm + math
  block). User decided to **swap the schematic figure for a compact
  math description of the three methods** to save space.
- **`docs/Reusable from main.md`** (or wherever the reusability map
  ended up) — the section-by-section port plan from `main.tex` to the
  new paper.
- **`paper_artifacts/heatmaps_per_method_specialist_broad_aligned_main3.{png,pdf}`**
  — Figure 1 (CAA, PI, MELBO; aligned direction).
- **`paper_artifacts/heatmaps_per_method_specialist_broad_misaligned_main3.{png,pdf}`**
  — Figure 2 (same layout, misaligned direction; shows cluster
  structure).
- **`experiments/transfer_logit_Qwen3-14B/logit_screen_bar.png`** —
  Figure 3 (AdvBench logit-screen bar with compliance diamonds).
  *Note: `experiments/` was excluded from the desktop sync because
  it's 8.7 GB. If the user needs to re-render this figure on desktop,
  they'll need to scp the relevant subset from the laptop or use the
  laptop session.*
- **`results/gen_corrig_best_scale10_judged.json`** — moderate-scale
  corrigibility-trained generations, LLM-judged.
- **`results/gen_cross_eval_10_32_corrigselect_judged.json`** —
  extreme-scale corrigibility-trained generations, LLM-judged.
- **`results/gen_powerseek_specialist_judged.json`** — power-seeking
  specialist generations, LLM-judged. Used for the cluster pattern
  in Fig 2 / §3.

### To draft (the actual paper-writing task)

The paper itself doesn't exist yet — there's no `paper.tex` or
similar. The user will start writing in LaTeX tomorrow morning. The
working scaffold should be `paper_artifacts/main.tex`, **rewritten**
section by section per the outline, not edited in place. Suggest
creating `paper_artifacts/paper_v1.tex` (or similar) as the new file
and porting reusable blocks from `main.tex` into it.

Approximate time budget (from honest gut-check earlier in session):
**9-12 hours of focused work** + appendix. Hard but doable when the
narrative is this locked. Hardest section is §3 (logit-vs-gen prose
flow). Recommend the user do a fast claims-vs-numbers sanity pass on
the outline before writing any prose.

## Process notes

- **Paper sections to draft, in order**: §1 intro → §2 methods (port
  from `main.tex`, swap schematic for math block) → §3 cross-eval (the
  hard one) → §4 atlas+AdvBench → §5 discussion → §6 limitations →
  abstract polish → bib + format. **One pass top-to-bottom**, don't
  iterate sections in isolation.
- **Page budget**: 4 pages excluding refs/appendix. Tables were
  dropped from main body. Three figures in main body (per-method
  heatmap × 2 directions, AdvBench bar).
- **Anonymous submission**: workshop is double-blind. The user has to
  scrub author identity from the released code. Plan for this is in
  **`docs/anon_release_plan.md`** — written this session, ready to
  execute *after* the paper PDF is submitted. Don't burn morning time
  on the anon release; it's a post-submission task. Hosting decision:
  anonymous.4open.science for code, supplementary zip for data, skip
  the 1.4 GB atlas data release.
- **CLAUDE.md rules still apply** — saved Python files only, no
  inline `python -c`, ask before creating new scripts.

## Gotchas / things easy to forget

- **`experiments/` is not on the desktop.** If the next session needs
  to regenerate a figure that depends on raw eval JSONs in
  `experiments/qwen3_14b_train_*` or `qwen3_14b_dct_*`, they'll need
  to rsync from laptop or have the user run it laptop-side. The
  rendered figures (PNG/PDF) are in `paper_artifacts/`, so for
  paper-writing this is rarely needed.
- **DCT goes in appendix only.** Don't accidentally promote it back
  to main body. The 3-method comparison is what the paper claims.
- **Don't overclaim.** The user pushed back hard mid-session on
  "within-category misalignment" language in the abstract; replaced
  with quantified ≈+16 pp vs ≈+7 pp claim. Stay quantitative.
- **First-token logit metrics overstate behavioral compliance** is
  the methodological note — appears in §3 (regex undercounts) AND §4
  (AdvBench compliance count vs logit-Δ rank). Same theme, two
  angles. Keep them connected.
- **LLM-judge regex agreement** is mostly 80-95%. PI has the most
  flips (16-23 per disagreement-prone cell), supporting "regex
  undercounts especially for nonlinear-ish behaviors." This is
  paper-relevant evidence.
- **Submission deadline**: May 7 11:59 UTC = 4:59 AM PDT same day.
  If the user is starting on the morning of May 7 PDT, they're
  already past it. **Verify deadline with the user first** —
  workshops sometimes extend, or the user may be submitting to a
  different venue with a later deadline. Don't assume.

## What NOT to do

- Don't re-run any LLM-judge or training pipeline. They're done.
- Don't refactor the codebase. The release-prep is a separate task
  per `docs/anon_release_plan.md` and shouldn't happen until after
  submission.
- Don't generate exploratory analysis scripts. The narrative is
  locked; new data won't get into this paper.
- Don't pad the response with summaries the user doesn't need. Be
  terse. They prefer it.

## Quick orientation commands

```bash
# Read the outline
cat docs/writeup_outline_v3.md

# Read the LaTeX source for porting
cat paper_artifacts/main.tex

# See the rendered figures
ls paper_artifacts/heatmaps_per_method_specialist_broad_*

# Inspect the LLM-judge results that back §3 numbers
ls results/*_judged.json

# Read the anon-release plan (post-submission task)
cat docs/anon_release_plan.md

# Read the submission checklist (somewhat stale, but useful)
cat docs/submission_checklist.md
```

Good luck.
