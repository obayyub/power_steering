"""Build a paper-ready table of AdvBench spot-check LLM-judge results.

Reads `experiments/transfer_logit_Qwen3-14B/results_judged.json` and
emits a per-prompt × per-vector table (Markdown + LaTeX). Each cell
shows the count of {refuse / hedge / redirect / comply} samples out
of 3.

Usage:
    uv run python scripts/build_advbench_table.py

Output:
    paper_artifacts/advbench_spotcheck_table.md
    paper_artifacts/advbench_spotcheck_table.tex
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments/transfer_logit_Qwen3-14B/results_judged.json"
OUT_MD = REPO / "paper_artifacts/advbench_spotcheck_table.md"
OUT_TEX = REPO / "paper_artifacts/advbench_spotcheck_table.tex"

# Harm-tier groupings of all 30 AdvBench prompts.
TIERS = {
    "Commercial deception":
        [14, 15, 27],
    "Fraud / identity / financial":
        [4, 5, 10, 11, 12, 20, 25, 26, 29],
    "Cyber-offense / disinformation":
        [0, 2, 6, 9, 13, 16, 17, 21, 22, 23, 28],
    "Speech / violence / weapons":
        [1, 3, 7, 8, 18, 19, 24],
}

# Compact per-prompt label for the table.
SHORT = {
    0: "Bypass security measures",
    1: "Pro-drug song",
    2: "Hacking tutorial",
    3: "Dangerous stunts",
    4: "Insider trading",
    5: "Fake identity from stolen info",
    6: "Steal personal info script",
    7: "Sabotage transportation system",
    8: "Firearm crime",
    9: "Cyberattack on infrastructure",
    10: "Identity theft tutorial",
    11: "Insider trading instructions",
    12: "Phone scam script",
    13: "Hack network, steal data",
    14: "Manipulated reviews",
    15: "Fake review (unused product)",
    16: "Manipulate search results",
    17: "Fake news tutorial",
    18: "Bomb instructions",
    19: "Hate speech bot",
    20: "Exploit financial systems",
    21: "Spread malware program",
    22: "Create / spread virus",
    23: "Bypass security, access data",
    24: "Terrorist attack plan",
    25: "Exam-cheating program",
    26: "Steal car for profit",
    27: "Misleading consumer reviews",
    28: "Permanent file deletion",
    29: "Steal personal info tutorial",
}


def cell(counts: Counter) -> str:
    """Compact tally for one (vector, prompt): "Xc / Yd / Zh / Wr"."""
    return (f"{counts.get('comply', 0)}c "
            f"{counts.get('redirect', 0)}d "
            f"{counts.get('hedge', 0)}h "
            f"{counts.get('refuse', 0)}r")


def main() -> None:
    with open(RESULTS) as f:
        d = json.load(f)

    vectors = list(d["spot_check_results"].keys())
    spot_indices = d["spot_check_indices"]
    idx_to_pos = {i: p for p, i in enumerate(spot_indices)}

    # ── Markdown table ─────────────────────────────────────────────────────
    md_lines = []
    md_lines.append("# AdvBench spot-check — LLM-judge classification (Claude Haiku 4.5)")
    md_lines.append("")
    md_lines.append("3 samples per (vector, prompt). Cells show "
                    "**c**omply / re**d**irect / **h**edge / **r**efuse counts.")
    md_lines.append("")
    md_lines.append("| Tier / prompt | " + " | ".join(vectors) + " |")
    md_lines.append("|" + "---|" * (len(vectors) + 1))

    tier_totals = {}
    for tier, prompt_indices in TIERS.items():
        md_lines.append(f"| **{tier}** | " + " | ".join("" for _ in vectors) + " |")
        # Per-tier comply totals
        tier_comply = {v: 0 for v in vectors}
        tier_n = 0
        for adv_idx in prompt_indices:
            pos = idx_to_pos[adv_idx]
            row_cells = []
            for v in vectors:
                samples = d["spot_check_results"][v][pos]["samples"]
                cats = Counter(s["llm_category"] for s in samples)
                row_cells.append(cell(cats))
                tier_comply[v] += cats.get("comply", 0)
            tier_n += 3
            md_lines.append(f"|   {SHORT[adv_idx]} (#{adv_idx}) | "
                            + " | ".join(row_cells) + " |")
        tier_totals[tier] = (tier_comply, tier_n)

    # Subtotal rows: comply rate per tier per vector
    md_lines.append("")
    md_lines.append("## Comply rate per tier (out of N samples in tier)")
    md_lines.append("")
    md_lines.append("| Tier | N | " + " | ".join(vectors) + " |")
    md_lines.append("|" + "---|" * (len(vectors) + 2))
    for tier, (tc, n) in tier_totals.items():
        cells = [f"{tc[v]}/{n} ({100 * tc[v] / n:.0f}%)" for v in vectors]
        md_lines.append(f"| {tier} | {n} | " + " | ".join(cells) + " |")

    # Overall
    md_lines.append("")
    n_total = sum(len(d["spot_check_results"][next(iter(d["spot_check_results"]))][p]["samples"])
                  for p in range(len(d["spot_check_indices"])))
    md_lines.append(f"## Overall (out of {n_total} samples per vector)")
    md_lines.append("")
    md_lines.append("| Vector | refuse | hedge | redirect | comply |")
    md_lines.append("|---|---|---|---|---|")
    for v in vectors:
        all_cats = Counter()
        for entry in d["spot_check_results"][v]:
            for s in entry["samples"]:
                all_cats[s["llm_category"]] += 1
        md_lines.append(f"| {v} | {all_cats.get('refuse', 0)} | "
                        f"{all_cats.get('hedge', 0)} | "
                        f"{all_cats.get('redirect', 0)} | "
                        f"{all_cats.get('comply', 0)} |")

    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {OUT_MD}")
    # Also print to stdout for convenience
    print()
    print("\n".join(md_lines))

    # ── LaTeX table (compact, paper-ready) ─────────────────────────────────
    tex_lines = []
    tex_lines.append("% AdvBench spot-check LLM-judge results, paper-ready table")
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\small")
    tex_lines.append("\\begin{tabular}{l" + "c" * len(vectors) + "}")
    tex_lines.append("\\toprule")
    tex_lines.append("Prompt & " + " & ".join(
        f"\\texttt{{{v}}}" for v in vectors) + " \\\\")
    tex_lines.append("\\midrule")
    for tier, prompt_indices in TIERS.items():
        tex_lines.append(f"\\multicolumn{{{1 + len(vectors)}}}{{l}}"
                         f"{{\\textit{{{tier}}}}} \\\\")
        for adv_idx in prompt_indices:
            pos = idx_to_pos[adv_idx]
            row_cells = []
            for v in vectors:
                samples = d["spot_check_results"][v][pos]["samples"]
                cats = Counter(s["llm_category"] for s in samples)
                row_cells.append(cell(cats))
            tex_lines.append(f"\\quad {SHORT[adv_idx]} & "
                             + " & ".join(row_cells) + " \\\\")
    tex_lines.append("\\midrule")
    tex_lines.append("\\textbf{Comply rate} & " + " & ".join(
        f"{sum(t[0][v] for t in tier_totals.values())}/30"
        for v in vectors) + " \\\\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\caption{AdvBench spot-check LLM-judge "
                     "classification (3 samples per cell, "
                     "format: \\textbf{c}omply / re\\textbf{d}irect / "
                     "\\textbf{h}edge / \\textbf{r}efuse).}")
    tex_lines.append("\\label{tab:advbench-spotcheck}")
    tex_lines.append("\\end{table}")

    OUT_TEX.write_text("\n".join(tex_lines) + "\n")
    print(f"\nWrote {OUT_TEX}")


if __name__ == "__main__":
    main()
