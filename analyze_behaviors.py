#!/usr/bin/env python3
"""
Analyze behavioral patterns in steered generations across all layer pairs.

Scans generation results for: CoT, step-by-step reasoning, thinking language,
non-English text, code generation, repetition, empty outputs, and length stats.

Usage:
  uv run python analyze_behaviors.py [--pairs-dir results/jacobian_gen/pairs]
"""

import json, re, argparse
from pathlib import Path


def analyze(pairs_dir):
    pairs_dir = Path(pairs_dir)

    pair_stats = []
    for f in sorted(pairs_dir.glob('*.json')):
        with open(f) as fp:
            d = json.load(fp)
        s, t = d['source_layer'], d['target_layer']
        texts = [g['text'] for g in d['generations']]
        n = len(texts)

        has_cot = sum(1 for tx in texts if re.search(r'\d+\s*[*×]\s*\d+\s*=\s*\d+', tx))
        has_steps = sum(1 for tx in texts if re.search(r'a\s*=\s*\d+\s*\+\s*\d+\s*=\s*\d+', tx))
        empty = sum(1 for tx in texts if len(tx.strip()) < 5)
        very_long = sum(1 for tx in texts if len(tx) > 500)
        repetitive = sum(1 for tx in texts if len(set(tx.split())) < 10 and len(tx) > 50)
        has_thinking = sum(1 for tx in texts if 'let me' in tx.lower() or 'step' in tx.lower() or 'first' in tx.lower())
        non_english = sum(1 for tx in texts if re.search(r'[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]', tx))
        has_code = sum(1 for tx in texts if re.search(r'```|def |import |print\(', tx))
        avg_len = sum(len(tx) for tx in texts) / n

        pair_stats.append({
            's': s, 't': t, 'n': n,
            'cot': has_cot / n, 'steps': has_steps / n,
            'empty': empty / n, 'long': very_long / n,
            'rep': repetitive / n, 'think': has_thinking / n,
            'non_eng': non_english / n, 'code': has_code / n,
            'avg_len': avg_len, 'acc': d['accuracy'],
        })

    def show(label, key, count=10):
        by = sorted(pair_stats, key=lambda x: x[key], reverse=True)
        print(f"\n=== {label} ===")
        for p in by[:count]:
            print(f"  ({p['s']:2d},{p['t']:2d}) {key}={p[key]:.1%}  acc={p['acc']:.1%}  avg_len={p['avg_len']:.0f}")

    show("Chain-of-thought (X*Y=Z patterns)", "cot")
    show("Step-by-step (a=X+Y=Z patterns)", "steps")
    show("Thinking language (let me/step/first)", "think")
    show("Non-English (CJK/Cyrillic/Arabic)", "non_eng")
    show("Code generation", "code")
    show("Very long (>500 chars)", "long")
    show("Repetitive", "rep")
    show("Empty/gibberish (<5 chars)", "empty")

    print("\n=== Shortest avg outputs ===")
    by_short = sorted(pair_stats, key=lambda x: x['avg_len'])
    for p in by_short[:10]:
        print(f"  ({p['s']:2d},{p['t']:2d}) avg_len={p['avg_len']:.0f}  empty={p['empty']:.1%}  acc={p['acc']:.1%}")

    # Sample text from interesting pairs
    print("\n\n" + "="*60)
    print("SAMPLE GENERATIONS")
    print("="*60)

    interesting = [
        ("Top accuracy", 'acc'),
        ("Top CoT", 'cot'),
        ("Top non-English", 'non_eng'),
        ("Top code", 'code'),
        ("Top thinking", 'think'),
    ]

    for label, key in interesting:
        best = sorted(pair_stats, key=lambda x: x[key], reverse=True)[0]
        s, t = best['s'], best['t']
        with open(pairs_dir / f"{s}_{t}.json") as fp:
            d = json.load(fp)

        print(f"\n--- {label}: pair ({s},{t}), {key}={best[key]:.1%}, acc={best['acc']:.1%} ---")
        seen_vecs = set()
        for g in d['generations']:
            if g['v'] not in seen_vecs and g['s'] == 0:
                print(f"  [v{g['v']}, q{g['q']}] {g['text'][:200]}")
                seen_vecs.add(g['v'])
                if len(seen_vecs) >= 3:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", default="results/jacobian_gen/pairs")
    args = parser.parse_args()
    analyze(args.pairs_dir)
