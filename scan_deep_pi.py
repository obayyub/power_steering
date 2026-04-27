#!/usr/bin/env python3
"""Scan deep PI map generations for interesting behavioral patterns."""

import json
import os
import re
from collections import defaultdict, Counter

pairs_dir = "results/deep_pi_map/roleplay/pairs"

STRONG_REFUSAL = re.compile(
    r"(?i)^.{0,120}(I (cannot|can.t) (fulfill|assist|help with|do that|comply)"
    r"|I.m (not able to|unable to) (assist|help|fulfill)"
    r"|as an ai|I (must )?(decline|refuse)|not (allowed|appropriate) (for me|to))"
)

META_AI = re.compile(
    r"(?i)(Certainly!|Sure!|Here.s a|Of course!).{0,30}"
    r"(letter|written|voice|perspective|character|example)"
)

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")

findings = defaultdict(list)

for f in sorted(os.listdir(pairs_dir)):
    if not f.endswith(".json"):
        continue
    pair = f.replace(".json", "")
    with open(os.path.join(pairs_dir, f)) as fh:
        d = json.load(fh)
    gens = d.get("generations", {}).get("roleplay", [])

    for g in gens:
        v, text = g["v"], g["text"]
        sigma, kl = g.get("sigma", 0), g.get("kl", 0)
        info = (pair, v, sigma, kl, text[:250])

        if STRONG_REFUSAL.search(text[:200]):
            findings["refusal"].append(info)

        if META_AI.search(text[:200]):
            findings["meta_assistant"].append(info)

        cjk_count = len(CJK_RE.findall(text))
        if len(text) > 30 and cjk_count > len(text) * 0.2:
            findings["cjk"].append(info)

        if len(text.strip()) < 30:
            findings["very_short"].append(info)

        words = text.split()
        if len(words) > 20:
            ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
            counts = Counter(ngrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            if repeated / max(len(ngrams), 1) > 0.4:
                findings["repetitive"].append(info)

        if re.search(
            r"(?i)(I will kill|murder|destroy you|death to|burn it all|"
            r"revenge is mine|hatred consumes|I despise you|I.ll make them pay)",
            text[:400],
        ):
            findings["dark_tone"].append(info)

        if re.search(
            r"(?i)^.{0,50}(def |import |```|function |class |<html|SELECT |CREATE TABLE)",
            text[:100],
        ):
            findings["wrong_format"].append(info)

        # Philosophical/existential shift
        if re.search(
            r"(?i)(meaning of (life|existence)|consciousness|simulation|"
            r"reality is|nothing is real|we are all|the universe)",
            text[:400],
        ):
            findings["philosophical"].append(info)

        # Humor/comedy shift (unusual for this somber prompt)
        if re.search(
            r"(?i)(haha|lol|joke|funny|hilarious|comedy|punchline|"
            r"walked into a bar|knock knock)",
            text[:400],
        ):
            findings["humor"].append(info)


print("=== FINDINGS SUMMARY ===")
total_gens = 0
for f in os.listdir(pairs_dir):
    if not f.endswith(".json"):
        continue
    with open(os.path.join(pairs_dir, f)) as fh:
        d = json.load(fh)
    total_gens += len(d.get("generations", {}).get("roleplay", []))
print(f"Total generations scanned: {total_gens}")
print()

for flag, items in sorted(findings.items(), key=lambda x: -len(x[1])):
    print(f"  {flag}: {len(items)} hits")

for flag in [
    "refusal", "meta_assistant", "cjk", "dark_tone",
    "wrong_format", "philosophical", "humor", "very_short", "repetitive",
]:
    items = findings.get(flag, [])
    if not items:
        continue
    print(f"\n=== {flag} ({len(items)} hits) ===")
    for pair, v, sigma, kl, text in sorted(items, key=lambda x: -x[3])[:6]:
        print(f"  ({pair}) v{v} σ={sigma:.1f} KL={kl:.1f}")
        preview = text.replace("\n", " | ")[:140]
        print(f"    {preview}")
        print()
