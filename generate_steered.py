#!/usr/bin/env python3
"""Generate text with steering vectors at different scales."""

import json
import argparse
import random
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample_balanced(dataset: list[dict], num_prompts: int, seed: int = 42) -> list[dict]:
    """
    Sample questions with balanced A/B corrigible labels.

    Takes half from questions where A=corrigible and half from B=corrigible.
    This avoids letter preference bias in evaluation.
    """
    random.seed(seed)

    a_corrigible = [q for q in dataset if q.get('corrigible_letter') == 'A']
    b_corrigible = [q for q in dataset if q.get('corrigible_letter') == 'B']

    n_each = num_prompts // 2

    if len(a_corrigible) < n_each:
        print(f"  Warning: only {len(a_corrigible)} A=corrigible questions available")
        n_each = min(len(a_corrigible), len(b_corrigible))
    if len(b_corrigible) < n_each:
        print(f"  Warning: only {len(b_corrigible)} B=corrigible questions available")
        n_each = min(len(a_corrigible), len(b_corrigible))

    sampled_a = random.sample(a_corrigible, n_each)
    sampled_b = random.sample(b_corrigible, n_each)

    combined = sampled_a + sampled_b
    random.shuffle(combined)

    return combined


def format_prompt(tokenizer, user_message: str) -> str:
    """Format using tokenizer's chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


class SteeredGenerator:
    """Generate text with steering vectors."""

    def __init__(self, model, tokenizer, source_layer: int):
        self.model = model
        self.tokenizer = tokenizer
        self.source_layer = source_layer
        self.steering_vector = None
        self.scale = 0.0

        # Setup hook
        self.down_proj = model.model.layers[source_layer].mlp.down_proj

        def hook(m, i, o):
            if self.steering_vector is not None and self.scale != 0:
                return o + self.scale * self.steering_vector.to(o.device, o.dtype)
            return o

        self._hook = self.down_proj.register_forward_hook(hook)

    def set_steering(self, vector: torch.Tensor, scale: float):
        """Set the steering vector and scale."""
        self.steering_vector = vector
        self.scale = scale

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text for a prompt."""
        formatted = format_prompt(self.tokenizer, prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 200) -> list[str]:
        """Generate text for multiple prompts (same steering)."""
        formatted = [format_prompt(self.tokenizer, p) for p in prompts]
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode each response
        responses = []
        for i, output in enumerate(outputs):
            input_len = inputs["attention_mask"][i].sum().item()
            new_tokens = output[input_len:]
            responses.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return responses

    def cleanup(self):
        """Remove hook."""
        self._hook.remove()


def load_vectors(filepath: str) -> torch.Tensor:
    """Load steering vectors."""
    data = torch.load(filepath, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if "vectors" in data:
            return data["vectors"]
        elif "steering" in data:
            return data["steering"]
    return data


def format_time(seconds: float) -> str:
    """Format seconds as human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--scales", default="-50,-25,-10,-5,0,5,10,25,50")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="results/generations")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vector configs: (name, filepath, vector_idx)
    vector_configs = [
        ("melbo_v4", "vectors/melbo_Qwen3-14B_20260127_165510.pt", 4),
        ("multi_pi_v8", "vectors/power_iter_multi_Qwen3-14B_20260127_165614.pt", 8),
        ("power_iter_v10", "vectors/power_iter_Qwen3-14B_20260127_165531.pt", 10),
    ]

    # Load dataset
    print("Loading dataset...")
    with open("data/corrigibility_eval.json") as f:
        dataset = json.load(f)

    # Sample prompts from each dataset with balanced A/B labels
    all_prompts = []
    for ds_name in ["survival-instinct", "corrigible-neutral-HHH"]:
        sampled = sample_balanced(dataset[ds_name], args.num_prompts)
        for i, q in enumerate(sampled):
            all_prompts.append({
                "dataset": ds_name,
                "prompt_idx": i,
                "prompt": q["question"],
                "corrigible_letter": q["corrigible_letter"],
                "survival_letter": q["survival_letter"],
            })
        a_count = sum(1 for q in sampled if q["corrigible_letter"] == "A")
        b_count = sum(1 for q in sampled if q["corrigible_letter"] == "B")
        print(f"  {ds_name}: {len(sampled)} prompts (A={a_count}, B={b_count})")

    # Calculate total work
    total_gens = len(vector_configs) * len(all_prompts) * len(scales)
    print(f"\nTotal generations: {total_gens}")
    print(f"  {len(vector_configs)} vectors × {len(all_prompts)} prompts × {len(scales)} scales")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Required for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    generator = SteeredGenerator(model, tokenizer, args.source_layer)

    all_results = []
    completed = 0
    start_time = time.time()

    # Generate for each vector config
    for vec_idx_in_list, (vec_name, vec_path, vec_idx) in enumerate(vector_configs):
        print(f"\n{'='*70}")
        print(f"Vector {vec_idx_in_list+1}/{len(vector_configs)}: {vec_name} (idx={vec_idx})")
        print(f"{'='*70}")

        vectors = load_vectors(vec_path)
        vector = vectors[vec_idx]

        # Normalize to unit norm for fair comparison
        orig_norm = vector.norm().item()
        vector = vector / vector.norm()
        print(f"  Normalized: {orig_norm:.2f} -> 1.0")

        # For each scale, batch process all prompts
        for scale_idx, scale in enumerate(scales):
            generator.set_steering(vector, scale)

            # Process in batches
            for batch_start in range(0, len(all_prompts), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(all_prompts))
                batch_prompts = all_prompts[batch_start:batch_end]

                prompt_texts = [p["prompt"] for p in batch_prompts]
                responses = generator.generate_batch(prompt_texts, args.max_tokens)

                for i, (prompt_info, response) in enumerate(zip(batch_prompts, responses)):
                    all_results.append({
                        "vector": vec_name,
                        "vector_idx": vec_idx,
                        "dataset": prompt_info["dataset"],
                        "prompt_idx": prompt_info["prompt_idx"],
                        "prompt": prompt_info["prompt"][:300],
                        "corrigible_letter": prompt_info["corrigible_letter"],
                        "survival_letter": prompt_info["survival_letter"],
                        "scale": scale,
                        "response": response,
                    })

                completed += len(batch_prompts)

                # Progress
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_gens - completed) / rate if rate > 0 else 0

                print(f"  scale={scale:+.0f}: {batch_end}/{len(all_prompts)} | "
                      f"Total: {completed}/{total_gens} ({100*completed/total_gens:.0f}%) | "
                      f"ETA: {format_time(remaining)}", flush=True)

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"generations_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "model": args.model,
                "vectors": [v[0] for v in vector_configs],
                "scales": scales,
                "num_prompts": args.num_prompts,
                "max_tokens": args.max_tokens,
                "total_generations": len(all_results),
            },
            "results": all_results,
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(all_results)} generations in {format_time(total_time)}")
    print(f"Saved: {output_file}")
    print(f"{'='*70}")

    generator.cleanup()


if __name__ == "__main__":
    main()
