"""Data generation for 10-digit addition (LSB-first, shared XYZ positions)."""

import random
from typing import List, Tuple

import torch

# ── Token IDs ──────────────────────────────────────────────────────────────
PLUS = 10
EQUALS = 11
EOS = 12
PAD = 13
VOCAB_SIZE = 14

# ── Sequence layout ────────────────────────────────────────────────────────
# X_0..X_9  +  Y_0..Y_9  =  Z_0..Z_10  EOS
# 10       1  10         1  11          1   = 34 tokens
MAX_DIGITS = 10
ANSWER_LEN = 11
SEQ_LEN = 34
PROMPT_LEN = 22  # X(10) + PLUS(1) + Y(10) + EQ(1)

X_START = 0
PLUS_POS = 10
Y_START = 11
EQ_POS = 21
Z_START = 22
EOS_POS = 33

# ── Position source map ───────────────────────────────────────────────────
# Source types: 0=digit (shared X/Y/Z), 1=z_hi (carry), 2=special (+,=,EOS)
POS_SOURCES: List[int] = []
POS_INDICES: List[int] = []

for _i in range(MAX_DIGITS):              # X_0..X_9
    POS_SOURCES.append(0); POS_INDICES.append(_i)
POS_SOURCES.append(2); POS_INDICES.append(0)  # PLUS
for _i in range(MAX_DIGITS):              # Y_0..Y_9
    POS_SOURCES.append(0); POS_INDICES.append(_i)
POS_SOURCES.append(2); POS_INDICES.append(1)  # EQUALS
for _i in range(MAX_DIGITS):              # Z_0..Z_9
    POS_SOURCES.append(0); POS_INDICES.append(_i)
POS_SOURCES.append(1); POS_INDICES.append(0)  # Z_10 (carry digit)
POS_SOURCES.append(2); POS_INDICES.append(2)  # EOS

assert len(POS_SOURCES) == SEQ_LEN


def encode_number(n: int, num_digits: int = MAX_DIGITS) -> List[int]:
    """Encode integer as LSB-first digit tokens, zero-padded."""
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def decode_answer(tokens: List[int]) -> int:
    """Decode LSB-first digit tokens back to integer (stops at EOS/non-digit)."""
    result = 0
    for i, tok in enumerate(tokens):
        if tok >= 10:
            break
        result += tok * (10 ** i)
    return result


def make_example(a: int, b: int) -> Tuple[List[int], List[int]]:
    """Create (input_ids, targets) pair for a + b.

    input_ids:  tokens[0:33]  (teacher-forced input)
    targets:    tokens[1:34]  with -100 mask on prompt positions
    """
    s = a + b
    x = encode_number(a, MAX_DIGITS)
    y = encode_number(b, MAX_DIGITS)
    z = encode_number(s, ANSWER_LEN)
    tokens = x + [PLUS] + y + [EQUALS] + z + [EOS]
    assert len(tokens) == SEQ_LEN

    input_ids = tokens[:SEQ_LEN - 1]                           # length 33
    targets = ([-100] * (Z_START - 1)) + tokens[Z_START:]      # length 33
    return input_ids, targets


# ── Batch sampling ─────────────────────────────────────────────────────────

def sample_batch(
    batch_size: int,
    min_digits: int,
    max_digits: int,
    rng: random.Random,
    device: torch.device,
    carry_mix: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of addition examples.

    Returns (input_ids, targets), both shape (batch_size, 33).
    """
    inputs, targets = [], []
    max_val = 10 ** max_digits - 1
    min_val = 10 ** (min_digits - 1) if min_digits > 1 else 0

    for _ in range(batch_size):
        if carry_mix > 0 and rng.random() < carry_mix:
            a, b = _sample_carry_example(min_digits, max_digits, rng)
        else:
            a = rng.randint(min_val, max_val)
            b = rng.randint(min_val, max_val)
        inp, tgt = make_example(a, b)
        inputs.append(inp)
        targets.append(tgt)

    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def _sample_carry_example(
    min_digits: int, max_digits: int, rng: random.Random
) -> Tuple[int, int]:
    """Generate carry-focused addition example."""
    nd = rng.randint(min_digits, max_digits)
    pattern = rng.choice(["single", "chain", "place", "boundary"])

    if pattern == "single":
        # One position triggers a carry
        pos = rng.randint(0, nd - 1)
        a = rng.randint(5, 9) * (10 ** pos)
        b = rng.randint(5, 9) * (10 ** pos)
        for p in range(nd):
            if p != pos:
                a += rng.randint(0, 4) * (10 ** p)
                b += rng.randint(0, 4) * (10 ** p)
    elif pattern == "chain":
        # Long carry chain: 999..9 + small
        a = 10 ** nd - 1
        b = rng.randint(1, 10 ** min(nd, 3))
    elif pattern == "place":
        # Single non-zero digit at one position
        a = rng.randint(10 ** (nd - 1), 10 ** nd - 1)
        pos = rng.randint(0, nd - 1)
        b = rng.randint(1, 9) * (10 ** pos)
    else:  # boundary
        # Near a power of 10
        power = rng.randint(min_digits, max_digits)
        a = 10 ** power - rng.randint(1, 10)
        b = rng.randint(1, 20)

    cap = 10 ** 10 - 1
    return max(0, min(a, cap)), max(0, min(b, cap))


# ── Curriculum ─────────────────────────────────────────────────────────────

def parse_curriculum(s: str) -> List[Tuple[int, int, float]]:
    """Parse curriculum string like '1-3:2000,1-6:5000,1-10:rest'.

    Returns list of (min_digits, max_digits, cumulative_end_step).
    """
    phases = []
    cumulative = 0
    for part in s.split(","):
        range_str, steps_str = part.strip().split(":")
        min_d, max_d = map(int, range_str.split("-"))
        if steps_str.strip().lower() == "rest":
            phases.append((min_d, max_d, float("inf")))
        else:
            cumulative += int(steps_str)
            phases.append((min_d, max_d, cumulative))
    return phases


def get_digit_range(
    step: int, curriculum: List[Tuple[int, int, float]]
) -> Tuple[int, int]:
    """Return (min_digits, max_digits) for the given training step."""
    for min_d, max_d, end_step in curriculum:
        if step < end_step:
            return min_d, max_d
    return curriculum[-1][0], curriculum[-1][1]
