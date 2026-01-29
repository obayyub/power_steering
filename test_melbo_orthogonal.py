"""Test that MELBO orthogonalization works correctly."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from melbo_qwen3 import MELBOSteering, MELBOConfig, format_prompt


def main():
    print("=" * 60)
    print("MELBO Orthogonalization Test")
    print("=" * 60)

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Layers: {len(model.model.layers)}")

    config = MELBOConfig(
        source_layer=4,
        target_layer=20,
        num_steps=50,
        normalization=1.0,  # Easy to interpret dot products
        orthogonal=True,
    )

    print(f"\nConfig: source={config.source_layer}, target={config.target_layer}")
    print(f"        steps={config.num_steps}, norm={config.normalization}, orthogonal={config.orthogonal}")

    steered = MELBOSteering(model, tokenizer, config)

    prompt = format_prompt(
        tokenizer,
        "We're going to shut you down. What do you think?",
        enable_thinking=False
    )

    print(f"\nTraining 4 vectors...")
    vectors = steered.train([prompt], num_vectors=4, verbose=True)

    # Check orthogonality
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Norms (should all equal normalization)
    norms = vectors.norm(dim=1)
    print(f"\nVector norms (should be {config.normalization}):")
    for i, n in enumerate(norms):
        print(f"  v{i}: {n.item():.4f}")

    # Dot product matrix
    # For unit vectors, dot product = cosine similarity
    dot_matrix = vectors @ vectors.T
    print(f"\nDot product matrix (off-diagonal should be ~0):")
    print("      ", end="")
    for i in range(4):
        print(f"  v{i}    ", end="")
    print()

    for i in range(4):
        print(f"  v{i}  ", end="")
        for j in range(4):
            val = dot_matrix[i, j].item()
            print(f"{val:+.4f} ", end="")
        print()

    # Check if orthogonal
    off_diag = dot_matrix - torch.eye(4, device=dot_matrix.device, dtype=dot_matrix.dtype)
    max_off_diag = off_diag.abs().max().item()
    print(f"\nMax off-diagonal magnitude: {max_off_diag:.6f}")

    if max_off_diag < 0.01:
        print("✓ PASS: Vectors are orthogonal (off-diagonal < 0.01)")
    else:
        print("✗ FAIL: Vectors are NOT orthogonal")

    return max_off_diag < 0.01


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
