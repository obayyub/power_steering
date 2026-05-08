# Anonymous code + data release plan

For double-blind workshop submission. Camera-ready version (with author
info restored) is a separate task post-acceptance.

## Hosting

- **Code**: anonymous.4open.science mirror of a fresh `git init` repo
  (no history). Submit URL like `anonymous.4open.science/r/power-steering-XXXX/`.
- **Data**: anonymous HuggingFace org (one-off, paper-themed name) if
  atlas data ships; otherwise bundle vectors + JSONs in the
  submission-portal supplementary zip and skip atlas reproducibility.

Atlas decision: skip atlas data release. Cite qualitatively in §4 and
Appendix B, ship only the handful of cited vectors. Saves ~1.4 GB and
the hassle of an anon HF org.

## Files to keep in `src/power_steering/`

- `find_vectors.py` — CAA / PI / MELBO
- `find_dct.py` — DCT (appendix A)
- `pipeline.py`, `run.py` — training entrypoints
- `download_dataset.py` — eval prep
- `map_freeform.py`, `map_layers.py` — atlas builders (cited in §4)
- `__init__.py` — keep exports current

## Files to keep in `scripts/`

- `build_paper_figures.py` — Fig 1+2 supporting code
- `build_per_method_heatmaps.py` — Fig 1+2 main renderer
- `plot_logit_screen.py` — Fig 3 (AdvBench)
- `generate_steered_samples.py` — §3 sampled generation
- `llm_judge_generations.py` — §3 LLM-judge
- `run_per_eval_pipelines.py` — §3 training orchestrator
- `analyze_best_alignment.py`, `analyze_per_eval_matrix.py`,
  `analyze_transfer.py` — number-producing utilities cited inline
- `freeform_to_dashboard.py` — only if atlas dashboard ships

## Files to drop from `scripts/`

- `compare_atlas_vs_handpicked.py`, `compare_pi_init_warmstart.py`,
  `compare_specialist_vs_generalist.py`, `cosine_specialists.py`,
  `fill_missing_signs.py`, `inspect_map.py`,
  `logit_screen_transfer.py`, `method_comparison_*.py`,
  `render_matrix_table.py`, `render_specialist_bars.py`,
  `check_signal_robustness.py`, `test_vector_transfer.py`,
  `test_setup.py`, `build_paper_figure2.py`, `build_paper_tables.py`

Reason: superseded, exploratory, or for paper sections that got cut.

## Directories to drop entirely

- `docs/sessions/` — private working notes
- `analysis/` — one-off explorations
- `paper_artifacts/` — the LaTeX draft + intermediate figures
- `experiments/` — bulky training artifacts; ship the cited subset via
  HF or supplementary zip instead
- `dashboard/` unless explicitly shipping the atlas viewer
- `.claude/`

## Configs to keep in `scripts/configs/`

- `qwen3_14b_train_<eval>.json` × 7 evals
- `qwen3_14b_dct_<eval>.json` × 7 evals (appendix)
- `gen_corrigselect_generalist.json`, `gen_powerseek_specialist.json`
  (the cell-list configs cited)
- `map_freeform_refusal_qwen14b.json`,
  `map_freeform_roleplay_qwen14b.json` (atlas configs cited in §4)
- Drop any one-off configs not cited in the paper

## Data bundle

Ship in supplementary zip (or HF dataset if atlas included):

- **Vectors**:
  - PI/MELBO/CAA × 7 evals at the cells used in §3 (specialist + the
    moderate-scale `|scale|=10` cells)
  - DCT × 7 evals (appendix A)
  - The atlas vectors cited in §4: `(24,37) v4+` (Chinese), `(19,28) v5+`,
    `(20,28) v9+`, `(20,26) v7-` (AdvBench transfer)
- **Eval JSONs** (logit-diff results):
  - 7 cross-eval JSONs × 4 methods (28 files)
- **Generation JSONs** (regex + LLM-judged):
  - `gen_cross_eval_10_32_corrigselect_judged.json`
  - `gen_corrig_best_scale10_judged.json`
  - `gen_powerseek_specialist_judged.json`
- `data/anthropic_evals.json` — polarity-annotated
- `experiments/transfer_logit_Qwen3-14B/` — the AdvBench logit-screen
  data behind Fig 3

## Anonymization checklist

Before pushing the anon repo:

- [ ] `rm -rf .git && git init` — fresh history, no committer info
- [ ] `pyproject.toml` — clear `authors = [...]`, `maintainers`,
  `urls`; replace with neutral placeholder
- [ ] Remove `CLAUDE.md` (workflow notes; also implicit identifier)
- [ ] Remove `lambda_cloud.py` if it leaks account/SSH-key defaults,
  or scrub those defaults
- [ ] Grep tree for `omar`, `omar.bet`, `obayyub`, `Omar`, the user's
  email — replace with `[anonymous]` placeholder
- [ ] In README, replace blog self-citation with
  `[anonymous blog post, link omitted for review]`
- [ ] Strip any `Co-Authored-By` lines from preserved messages
  (irrelevant after `git init` but worth a grep)
- [ ] Ensure no Lambda/SSH/API keys in any committed file
  (scan `.env`, `*.json` for tokens)
- [ ] Check `download_dataset.py` and any HF/Anthropic config for
  default cache paths that include the username
- [ ] `.gitignore` for the anon repo: `.venv/`, `__pycache__/`,
  `experiments/`, `results/` (selectively re-add only the
  paper-cited subset), `paper_artifacts/`, `analysis/`,
  `docs/sessions/`, `.claude/`, `.env*`

## Reproducibility scope

- **Main figures**: must reproduce from a fresh clone + released data.
  Three commands total (Fig 1, Fig 2, Fig 3).
- **Appendix figures**: ship the data, mention "see `scripts/` for the
  analysis pipeline." Don't promise turnkey re-rendering.

## Documentation

- `README.md` (~150 lines): paragraph summary, paper PDF link
  placeholder, data link placeholder, three-line quickstart, repo map,
  `[citation TBD pending acceptance]` block, license.
- `REPRODUCTION.md` (~80 lines): how to regenerate Fig 1, Fig 2, Fig 3
  from released data; how to retrain a single method on a single eval
  with a config example.
- Skip METHODS.md (duplicates paper).
- Skip CITATION.cff (overkill for workshop).

## Order of ops on release day

1. Branch + prune to a release tree (still in private repo).
2. `rm -rf .git && git init` on the release tree, single initial commit.
3. Run anonymization grep + checklist above.
4. Push to a fresh public GitHub repo (or upload directly to
   anonymous.4open.science from a tarball).
5. Submit anon URL to anonymous.4open.science.
6. Upload data bundle (HF anon org OR supplementary zip).
7. Smoke test: fresh `git clone` + `uv sync` + run the three
   reproduction commands; confirm figures regenerate.

## Post-acceptance (camera-ready)

- Restore author info in `pyproject.toml`, README, paper.
- Push the same tree to your real GitHub account, public.
- Re-host data on your real HF account (or migrate the anon org).
- Update paper PDF link, real blog self-citation.
- Tag `v1.0`, optional CITATION.cff at this point.
