"""Simple test script to verify the run workflow."""

from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

output_file = results_dir / "hello.txt"
output_file.write_text("Hello from Lambda Cloud!\n")

print(f"Created {output_file}")
