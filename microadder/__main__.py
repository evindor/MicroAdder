"""Allow running `uv run python -m microadder` to show usage info."""

print("MicroAdder: 74-parameter transformer for 10-digit addition")
print()
print("Available commands:")
print("  uv run python -m microadder.train --run-name <name> --seed <seed>")
print("  uv run python -m microadder.eval --checkpoint <path>")
