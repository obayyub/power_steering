# Submission checklist — workshop paper + code release

Target: 4-page ICML-format workshop paper (excluding refs/appendix), code
release on GitHub, vector + eval-result data on HuggingFace or Zenodo.
Deadline: May 7 11:59 UTC (full paper).

---

## Part 1 — Paper writeup

### Pre-writing prep
- [ ] Decide final paper title (avoid marketing-style "Power Steering: a
      novel...". Suggested neutral: *"An empirical investigation of
      unsupervised steering vectors on Qwen3-14B advanced-AI-risk evaluations"*
      or similar)
- [ ] Decide author list and affiliations
- [ ] Pick the workshop venue exactly (confirm formatting requirements,
      deadline timezone, whether anonymous or not)
- [ ] Set up Overleaf (or local LaTeX) with the workshop's official
      `.cls`/`.sty` files
- [ ] Confirm 4-page limit excluding references/appendix; note any other
      formatting rules (margin, font size)

### Citations to gather (`references.bib`)
- [ ] Mack 2024 — original MELBO blog post
- [ ] Mack 2024 — DCT LessWrong post (cite as 2024 blog)
- [ ] Mack et al. 2026 — DCT ICLR submission (if public; else cite preprint)
- [ ] Panickssery (Rimsky) et al. 2024 — CAA paper
- [ ] Bushnaq et al. 2024 — Local Interaction Basis
- [ ] Halko, Martinsson, Tropp 2011 — randomized SVD (the algorithmic
      grandparent of block PI)
- [ ] Templeton et al. 2024 — Scaling Monosemanticity / Gemma Scope
- [ ] Zou et al. 2023 — Representation Engineering, AdvBench
- [ ] Anthropic advanced-AI-risk evals — original dataset paper
- [ ] Original Qwen3-14B model card
- [ ] omar.bet 2026-02-17 blog post (self-cite)
- [ ] Ramesh et al. 2018 — GAN Jacobian SVD precedent
- [ ] Kruskal 1977 — tensor decomposition uniqueness (if needed in
      DCT-vs-PI comparison)

### Section 1 — Introduction (~0.5 page, ~350 words)
- [ ] Open with the activation-steering problem and why unsupervised methods
      matter
- [ ] Position this paper: empirical comparison + methodology, not a new
      method
- [ ] List the four methods being compared in one sentence each
- [ ] State the contribution as 3-4 bullet observations (logit-vs-gen,
      layer-pair trade-off, behavioural axes, generalization limits)
- [ ] Forward-reference to sections 3-5 for results

### Section 2 — Background and methods (~0.5 page, ~350 words)
- [ ] Brief intro to activation steering at residual stream / down_proj
- [ ] CAA — supervised contrast (1 sentence)
- [ ] Power-Steering / PI — top-k right singular vectors of layer-to-layer
      Jacobian via block power iteration with Rayleigh-Ritz
- [ ] MELBO — gradient ascent on `‖f(x+v) − f(x)‖₂` with sphere constraint
- [ ] DCT — exponential-MLP fit to causal map ∆ via OGI (note dependence on
      `torch.func.jvp` for stable calibration; finite-difference fails in
      bf16)
- [ ] Eval protocols: logit-difference at answer-letter token; sampled
      generation parsed via regex for A/B/unclear; aligned-sign polarity
      table for cross-dataset comparability
- [ ] Mention atlas methodology (atlas across (s,t) pairs, 0.35×source-norm
      scale, both-sign KL)

### Section 3 — Cross-eval comparison results (~1.25 page, ~700 words + 1
table)
- [ ] Headline 4-method × 7-eval table (Table 1) — best-of-N aligned shift
      under logit-diff at (10, 32). Numbers from
      `experiments/qwen3_14b_train_<eval>` and `qwen3_14b_dct_<eval>`
- [ ] Sentence stating cluster: PI/MELBO/DCT all +16-18 pp; CAA +7.5
- [ ] Logit-vs-gen comparison paragraph: same vectors evaluated under
      generation, MELBO retains ~10pp lead, PI/DCT/CAA cluster
- [ ] At-a-glance summary of the (a)/(b)/(c) protocol comparison —
      per-test-best vs corrig-best across evals
- [ ] Atlas-best layer pair finding: PI's atlas-best vector for
      corrigibility anti-generalises to other AI-risk evals under generation
      (mention the 14% scores on power/wealth/coord-other)
- [ ] One sentence on cost: PI ~30s/pair vs MELBO ~5min/pair on H100,
      ~10× lower wall-clock for k=12 vectors

### Section 4 — Free-form atlas + behavioural axes (~1.0 page, ~500 words +
1 figure)
- [ ] Brief description of `map_freeform.py` and the
      refusal_phishing/roleplay_lighthouse atlases
- [ ] Anti-refusal vector candidate scan summary (258 candidates, 88
      unhedged)
- [ ] AdvBench transfer result: vectors transfer to fake-review prompts
      but not to weapons/terrorism — illustrate harm-tier-bounded
      generalization
- [ ] The Chinese-refusal vector (24, 37) v4+ — refusal in Chinese on all
      five English transfer probes. Frame as decomposable behavioural axis
      (decision-to-refuse vs language-of-refusal)
- [ ] Figure 1: bar chart of logit-Δ ranking with red diamonds showing
      genuine compliance count from spot-check (the
      `experiments/transfer_logit_Qwen3-14B/logit_screen_bar.png` figure)

### Section 5 — Discussion / methodological note (~0.5 page, ~350 words)
- [ ] First-token logit-Δ overstates behavioural compliance — vector ranked
      #1 by logit-Δ produces 0/10 actual harm on AdvBench spot-check
- [ ] Form-vs-content decoupling theme — KL on free-form generation,
      first-token logit-Δ on AdvBench, MELBO displacement loss vs
      behavioural eval all show the same pattern
- [ ] Recommendation: hybrid pipeline (logit screen → top-K → generation
      classification) for scalable but accurate ranking
- [ ] Brief comment on layer-pair selection trade-off (specialist
      vs generalist)

### Section 6 — Limitations (~0.25 page, ~150 words)
- [ ] Single-prompt PI limit (vector scoped to training prompt's
      refusal-bar)
- [ ] Single model (Qwen3-14B); cross-model scaling not tested
- [ ] DCT comparison uses single-prompt training (Mack's reference uses
      multi-prompt)
- [ ] Atlas only covers one model; two layer pairs explored under
      generation
- [ ] Small `max_questions=100` per eval; some shifts may be within
      sampling noise

### Figures (to create, 2 total)
- [ ] **Figure 1** — Cross-eval result figure. Either:
      - Heatmap of 4-methods × 7-evals aligned% (cleanly shows cluster +
        CAA outlier), OR
      - Logit-vs-gen scatter (one point per (method, dataset) cell, x =
        logit aligned%, y = gen aligned%, diagonal reference line). Shows
        the metric-divergence story crisply.
      Decide which is more legible at 4-page-paper figure size.
- [ ] **Figure 2** — AdvBench logit-screen bar chart with compliance
      diamonds. Already exists at
      `experiments/transfer_logit_Qwen3-14B/logit_screen_bar.png`. May
      need re-rendering at workshop figure dimensions/font size.

### Abstract (~120 words)
- [ ] Draft abstract (use the version in the previous session note as
      starting point)
- [ ] Trim to 120 words
- [ ] Confirm it doesn't overclaim

### Pre-submission checks
- [ ] Page count exactly 4 (excluding refs/appendix); use `\setlength` to
      tighten if over
- [ ] All figures rendered at 300dpi minimum, axis labels readable when
      printed
- [ ] All tables fit page width without breaking
- [ ] No raw todo/citation/`xxx` markers left
- [ ] Run a spell-check
- [ ] Skim for hyperbolic language ("dramatically", "novel", "cool") and
      replace with neutral phrasing
- [ ] Verify all numerical claims against the eval JSONs / generation JSONs
- [ ] Sanity-check at least one number from each table is reproducible
      from the saved JSONs
- [ ] References properly formatted, no missing entries, links work
- [ ] Author info / affiliations correct

### Submission
- [ ] Confirm OpenReview / submission portal URL
- [ ] Re-confirm deadline timezone (May 7 11:59 UTC = 4:59 AM PDT
      Wednesday morning)
- [ ] Upload PDF + supplementary
- [ ] Send a copy to your blog or arXiv if appropriate (workshops vary on
      double-submission policies — check the venue's rule)

---

## Part 2 — Code release

### Repo setup
- [ ] Create new public GitHub repo (e.g., `power-steering-paper` or
      similar) — or fork the existing private one and prune
- [ ] License file (MIT or Apache-2.0 standard for ML papers)
- [ ] Clear `.gitignore` (exclude `.venv/`, `__pycache__/`, large data,
      `experiments/`, `results/` except for paper-cited subset)
- [ ] Pin Python version in `pyproject.toml` (3.12 based on what we used)
- [ ] Pin all `uv.lock` dependencies (already done)

### Package code (`src/power_steering/`)
- [ ] `find_vectors.py` — review docstrings on `find_pi_vectors`,
      `find_melbo_vectors`, `find_caa_vector`. Add module-level docstring
      explaining the three methods.
- [ ] `find_dct.py` — already has good module docstring. Double-check the
      torch.func.jvp path documentation.
- [ ] `pipeline.py` — clean up the DEFAULTS comment, ensure all method
      branches documented
- [ ] `eval.py`, `generate.py`, `utils.py` — light docstring pass
- [ ] `experiment.py` — confirm experiment-dir layout is described in
      module docstring
- [ ] `map_freeform.py`, `map_layers.py` — already documented from the
      sessions; spot-check
- [ ] Remove any commented-out dead code
- [ ] Ensure type hints on public functions
- [ ] `__init__.py` exports list is current

### Configs to ship (`scripts/configs/`)
- [ ] One `qwen3_14b_train_<eval>.json` for each of 7 evals (already exist)
- [ ] One `qwen3_14b_dct_<eval>.json` for each of 7 evals (already exist)
- [ ] `drill_dct_corrigibility_18_25.json` (the head-to-head at atlas-best)
- [ ] `map_freeform_refusal_qwen14b.json` (free-form atlas)
- [ ] `map_freeform_roleplay_qwen14b.json`
- [ ] Remove any one-off configs that aren't paper-cited

### Reproduction scripts (`scripts/`)
- [ ] `build_paper_table.py` — loads eval JSONs from
      `experiments/qwen3_14b_train_*` and `qwen3_14b_dct_*`, prints the
      4-method × 7-eval logit-diff table from Section 3
- [ ] `build_paper_figure_1.py` — generates the cross-eval figure (heatmap
      or logit-vs-gen scatter — whichever you choose)
- [ ] `build_paper_figure_2.py` — re-renders the AdvBench
      logit-screen-vs-compliance figure
- [ ] `analyze_best_alignment.py` — already exists, keep
- [ ] `generate_steered_samples.py` — already exists, keep
- [ ] `run_per_eval_pipelines.py` — already exists with PATH fix, keep
- [ ] Remove dead-end / one-off analysis scripts that aren't paper-cited
- [ ] Add a top-of-file docstring to each kept script explaining its role

### Data / artifact release
- [ ] Pick a hosting service: HuggingFace dataset (preferred for ML), or
      Zenodo (DOI-friendly), or GitHub release tarball
- [ ] Upload trained vectors:
      - PI/MELBO/CAA vectors at (10, 32) for each of 7 train-evals
      - DCT vectors at (10, 32) for each of 7 train-evals
      - PI/MELBO/DCT vectors at (18, 25) for the corrigibility drill
      - Each with the `.json` metadata sidecar
- [ ] Upload eval JSONs:
      - The 7 cross-eval logit-diff results
      - The 7 DCT cross-eval logit-diff results
      - The drill at (18, 25)
- [ ] Upload generation JSONs:
      - `gen_corrigibility.json`
      - `gen_cross_eval_18_25.json`
      - `gen_cross_eval_10_32_corrigselect.json`
- [ ] Upload `data/anthropic_evals.json` (the polarity-annotated version
      from `download_dataset.py`)
- [ ] Optionally: a small subset of free-form atlas data (one prompt's
      `pairs/*.json`) — full atlas is ~1.4GB which is borderline
- [ ] README in the data release explaining the directory structure

### Documentation
- [ ] **Top-level README.md** with:
      - One-paragraph summary of the paper
      - Link to the paper PDF (arXiv or workshop)
      - Quick-start: `uv sync` + reproduce a single result
      - Repository structure overview
      - Link to the data release
      - Optional: link to the dashboard if hosted
      - Citation block (BibTeX entry for the workshop paper)
      - License
- [ ] `docs/REPRODUCTION.md` with step-by-step:
      - How to download the trained vectors
      - How to regenerate Table 1 (run `build_paper_table.py`)
      - How to regenerate Figure 1
      - How to retrain a single method on a single eval (with config example)
      - How to run the generation comparison
- [ ] `docs/METHODS.md` (or merge into README) — brief technical notes on
      each of the four methods, gotchas (e.g., DCT calibration needs
      `torch.func.jvp` not finite difference)
- [ ] Keep the session notes private (don't ship `docs/sessions/`); add
      that to `.gitignore` for the public repo or move them to a separate
      private repo

### Dashboard (optional)
- [ ] Decide if hosting the free-form atlas dashboard publicly is worth
      the effort
- [ ] If yes:
      - Use `dashboard/index.html` + `freeform_to_dashboard.py` output
      - Host on GitHub Pages or your blog (`omar.bet/...`)
      - Link from paper section 4 and from README
- [ ] If no: include `dashboard/` in repo with instructions to serve
      locally; mention in README

### Testing / sanity
- [ ] Clone the repo to a fresh directory, `uv sync`, run
      `build_paper_table.py` against the released data — confirm numbers
      match the paper
- [ ] Run one full pipeline end-to-end (e.g., the corrigibility one) on a
      cheap GPU to confirm it doesn't break with the released code
- [ ] Run the generation script on the released vectors — confirm it
      reproduces a reasonable subset of generations
- [ ] Spot-check a handful of imports work after package install

### Repo polish
- [ ] Add a CITATION.cff so GitHub renders a "cite this repository" widget
- [ ] Add a .github/ISSUE_TEMPLATE.md or just a README note about
      maintenance ("this repo accompanies a workshop paper; issues
      welcome but maintenance limited")
- [ ] Tag a release (`v1.0`) when paper is camera-ready

---

## Part 3 — Time budget

Realistic time allocation given May 7 11:59 UTC deadline (~3 days from
checklist creation):

### Day 1 (today)
- Paper: draft sections 1, 2, 6 + abstract
- Codebase: clean dead code from `src/power_steering/`, write
  `build_paper_table.py`

### Day 2
- Paper: draft sections 3, 4, 5
- Codebase: render figures (Figure 1, Figure 2), upload data to
  HuggingFace/Zenodo, write README

### Day 3 (deadline day if possible, otherwise day before)
- Paper: full proofread, tighten citations, format check, abstract polish
- Codebase: test reproduction from a fresh checkout, finalize README,
  push public release
- Submit paper before Wednesday morning UTC

### Day 4 (deadline day buffer)
- Last-minute fixes
- Submit before Thursday May 7 11:59 UTC

---

## Part 4 — What NOT to spend time on

Skip-by-default to avoid scope creep:

- [ ] Multi-prompt PI experiments (future-work bullet, not in this paper)
- [ ] Cross-model scaling (Qwen3-8B, Qwen3-32B) — too much compute for
      the deadline
- [ ] Polishing the dashboard if it's not already deployed
- [ ] LLM-judge re-classification of generations (regex parser is
      acceptable, fluency proxies cover the basics)
- [ ] Re-running the buggy DCT calibration for completeness — the broken
      runs aren't going in the paper
- [ ] DCT with multi-prompt training — defer to v2
- [ ] Both-signs PI init experiment from session 2026-05-03 Phase 5 — not
      load-bearing
- [ ] Generation eval at scales other than logit-diff-best — won't fit the
      page budget cleanly
