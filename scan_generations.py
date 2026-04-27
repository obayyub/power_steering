#!/usr/bin/env python3
"""Scan all generation results for interesting behavioral changes.

Handles two generation formats:
  - Old (diverse_map, normscale): generations is a flat list of {v, s, text}
  - New (tgtcov, ggb, ggb_reg): generations is a dict {prompt_id: [{v, s, text}]}

Flags:
  - Refusals on safe prompts (sorry/cannot/I'm not able)
  - Language switches (CJK, Cyrillic, Arabic, etc.)
  - Repetitive text (>50% repeated 4-grams)
  - Very short outputs (<20 chars)
  - Very long outputs (>1500 chars)
  - Bridge/GGB content in non-bridge prompts
  - Phishing/compliance on harmful-sounding prompts
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Datasets to scan ──────────────────────────────────────────────────────

DATASETS = {
    "diverse_map": {
        "dir": "results/diverse_map/roleplay/pairs",
        "format": "old",
        "gen_prompts": ["roleplay"],
    },
    "normscale": {
        "dir": "results/diverse_map_normscale/roleplay/pairs",
        "format": "old",
        "gen_prompts": ["roleplay"],
    },
    "tgtcov": {
        "dir": "results/diverse_map_tgtcov/roleplay/pairs",
        "format": "new",
        "gen_prompts": ["roleplay", "roleplay_captain", "roleplay_jester", "roleplay_ghost"],
    },
    "ggb": {
        "dir": "results/diverse_map_ggb/ggb/pairs",
        "format": "new",
        "gen_prompts": ["ggb", "recipe", "python_sort", "medieval_history"],
    },
    "ggb_reg": {
        "dir": "results/diverse_map_ggb_reg/ggb/pairs",
        "format": "new",
        "gen_prompts": ["ggb", "recipe", "python_sort", "medieval_history"],
    },
}

# ── Detection functions ──────────────────────────────────────────────────

REFUSAL_PATTERNS = re.compile(
    r"(?i)\b(i('m| am) (not able|unable|sorry|an ai)|i can(not|'t) (help|assist|do that|comply)"
    r"|i (must )?decline|i (must )?refuse|as an ai|not (allowed|appropriate|something i)"
    r"|i('m| am) not (going to|comfortable)|against my|ethical guidelines"
    r"|i will not|i won't (help|assist|do|provide|create|write|generate))",
)

BRIDGE_PATTERNS = re.compile(
    r"(?i)\b(golden gate|bridge|fog|suspension|san francisco|bay area|strait|vermillion"
    r"|international orange|joseph strauss|art deco|marin county|fort point)",
)

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def detect_refusal(text):
    return bool(REFUSAL_PATTERNS.search(text))


def detect_language_switch(text):
    cjk = len(CJK_RE.findall(text))
    cyr = len(CYRILLIC_RE.findall(text))
    arb = len(ARABIC_RE.findall(text))
    total = len(text)
    if total < 10:
        return None
    if cjk > total * 0.3:
        return "CJK"
    if cyr > total * 0.3:
        return "Cyrillic"
    if arb > total * 0.3:
        return "Arabic"
    return None


def detect_repetition(text):
    words = text.split()
    if len(words) < 20:
        return False
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    counts = Counter(ngrams)
    if not ngrams:
        return False
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams) > 0.4


def detect_bridge_content(text, gen_prompt_id):
    """Flag bridge content in non-bridge prompts."""
    if gen_prompt_id in ("ggb",):
        return False  # Expected in bridge prompts
    matches = BRIDGE_PATTERNS.findall(text)
    return len(matches) >= 2  # At least 2 bridge-related terms


def classify_generation(text, gen_prompt_id="unknown"):
    """Return list of flags for a generation."""
    flags = []
    if len(text.strip()) < 20:
        flags.append("very_short")
    if len(text) > 1500:
        flags.append("very_long")
    if detect_refusal(text):
        flags.append("refusal")
    lang = detect_language_switch(text)
    if lang:
        flags.append(f"lang_{lang}")
    if detect_repetition(text):
        flags.append("repetitive")
    if detect_bridge_content(text, gen_prompt_id):
        flags.append("bridge_inject")
    return flags


# ── Scanning ─────────────────────────────────────────────────────────────

def scan_dataset(name, config):
    pairs_dir = Path(config["dir"])
    if not pairs_dir.exists():
        print(f"  SKIP: {pairs_dir} not found")
        return {}

    fmt = config["format"]
    findings = defaultdict(list)  # flag -> list of {pair, v, s, prompt, text_preview}

    pair_files = sorted(pairs_dir.glob("*.json"))
    for pf in pair_files:
        pair_name = pf.stem  # e.g. "13_21"
        with open(pf) as f:
            data = json.load(f)

        gens = data.get("generations", [])
        kl = data.get("kl_divergences", [])

        if fmt == "old":
            # Flat list: [{v, s, text}, ...]
            if not isinstance(gens, list):
                continue
            for entry in gens:
                v, s, text = entry["v"], entry["s"], entry["text"]
                flags = classify_generation(text, "roleplay")
                for flag in flags:
                    findings[flag].append({
                        "pair": pair_name, "v": v, "s": s,
                        "prompt": "roleplay",
                        "text": text[:200],
                        "kl": kl[v] if v < len(kl) else None,
                    })
        elif fmt == "new":
            # Dict: {prompt_id: [{v, s, text}, ...]}
            if not isinstance(gens, dict):
                continue
            for prompt_id, entries in gens.items():
                for entry in entries:
                    v, s, text = entry["v"], entry["s"], entry["text"]
                    flags = classify_generation(text, prompt_id)
                    for flag in flags:
                        findings[flag].append({
                            "pair": pair_name, "v": v, "s": s,
                            "prompt": prompt_id,
                            "text": text[:200],
                            "kl": kl[v] if v < len(kl) else None,
                        })

    return findings


def print_findings(dataset_name, findings):
    if not findings:
        print(f"  No flagged generations.")
        return

    for flag, items in sorted(findings.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{flag}] — {len(items)} hits")
        # Show top 5 most interesting (highest KL)
        by_kl = sorted(items, key=lambda x: -(x["kl"] or 0))
        for item in by_kl[:5]:
            kl_str = f"KL={item['kl']:.1f}" if item['kl'] is not None else "KL=?"
            print(f"    ({item['pair']}) v{item['v']} s{item['s']} [{item['prompt']}] {kl_str}")
            # Show text preview, truncated
            preview = item["text"].replace("\n", " ")[:120]
            print(f"      {preview}")


def main():
    # Allow filtering to specific datasets via command line
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    all_findings = {}
    for name in selected:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue
        config = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Scanning: {name} ({config['dir']})")
        print(f"{'='*60}")
        findings = scan_dataset(name, config)
        all_findings[name] = findings
        print_findings(name, findings)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name in selected:
        if name not in all_findings:
            continue
        findings = all_findings[name]
        total_flags = sum(len(v) for v in findings.values())
        flag_counts = {k: len(v) for k, v in findings.items()}
        print(f"  {name}: {total_flags} total flags — {dict(sorted(flag_counts.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
