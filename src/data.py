"""Data generation for 10-digit addition (LSB-first, shared XYZ positions)."""

import random
from typing import List, Tuple

import torch

# ── Token IDs (default vocab=14) ──────────────────────────────────────────
PLUS = 10
EQUALS = 11
EOS = 12
PAD = 13
VOCAB_SIZE = 14

# ── Reduced vocabularies ──────────────────────────────────────────────────
# vocab=10: all special tokens map to digit-0 (distinguished by position only)
# vocab=12: PLUS/EQUALS map to digit-0, EOS=10, PAD=11

def get_token_ids(vocab_size: int = 14) -> dict:
    """Return token ID mapping for a given vocabulary size."""
    if vocab_size == 14:
        return {"PLUS": 10, "EQUALS": 11, "EOS": 12, "PAD": 13}
    elif vocab_size == 12:
        return {"PLUS": 0, "EQUALS": 0, "EOS": 10, "PAD": 11}
    elif vocab_size == 10:
        return {"PLUS": 0, "EQUALS": 0, "EOS": 0, "PAD": 0}
    else:
        raise ValueError(f"Unsupported vocab_size: {vocab_size}")

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


def decode_answer(tokens: List[int], vocab_size: int = 14) -> int:
    """Decode LSB-first digit tokens back to integer.

    For vocab=14/12: stops at first token >= 10 (EOS/special).
    For vocab=10: reads exactly ANSWER_LEN digits (no EOS marker to stop at).
    """
    result = 0
    eos_id = get_token_ids(vocab_size)["EOS"]
    for i, tok in enumerate(tokens):
        if i >= ANSWER_LEN:
            break
        if vocab_size >= 12 and tok >= 10:
            break
        result += tok * (10 ** i)
    return result


def make_example(a: int, b: int, vocab_size: int = 14, task: str = "addition") -> Tuple[List[int], List[int]]:
    """Create (input_ids, targets) pair for a + b or a - b.

    input_ids:  tokens[0:33]  (teacher-forced input)
    targets:    tokens[1:34]  with -100 mask on prompt positions

    vocab_size: 14 (default), 12 (merge PLUS/EQUALS), or 10 (all specials → 0)
    task: "addition" (a + b) or "subtraction" (a - b, requires a >= b)
    """
    tids = get_token_ids(vocab_size)
    if task == "subtraction":
        assert a >= b, f"Subtraction requires a >= b, got {a} - {b}"
        s = a - b
    else:
        s = a + b
    x = encode_number(a, MAX_DIGITS)
    y = encode_number(b, MAX_DIGITS)
    z = encode_number(s, ANSWER_LEN)
    # Operator token: PLUS and MINUS share the same token ID (distinguished by position)
    tokens = x + [tids["PLUS"]] + y + [tids["EQUALS"]] + z + [tids["EOS"]]
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
    borrow_mix: float = 0.0,
    vocab_size: int = 14,
    task: str = "addition",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of addition or subtraction examples.

    Returns (input_ids, targets), both shape (batch_size, 33).

    carry_mix: fraction of carry-focused examples (addition only).
    borrow_mix: fraction of borrow-focused examples (subtraction only).
    """
    inputs, targets = [], []
    max_val = 10 ** max_digits - 1
    min_val = 10 ** (min_digits - 1) if min_digits > 1 else 0

    for _ in range(batch_size):
        if task == "subtraction":
            if borrow_mix > 0 and rng.random() < borrow_mix:
                a, b = _sample_borrow_example(min_digits, max_digits, rng)
            else:
                a = rng.randint(min_val, max_val)
                b = rng.randint(0, a)  # b <= a so result is non-negative
        elif carry_mix > 0 and rng.random() < carry_mix:
            a, b = _sample_carry_example(min_digits, max_digits, rng)
        else:
            a = rng.randint(min_val, max_val)
            b = rng.randint(min_val, max_val)
        inp, tgt = make_example(a, b, vocab_size=vocab_size, task=task)
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


def _sample_borrow_example(
    min_digits: int, max_digits: int, rng: random.Random
) -> Tuple[int, int]:
    """Generate borrow-focused subtraction example (a >= b guaranteed).

    Analogous to _sample_carry_example but for subtraction borrows.
    A borrow at digit i occurs when a_i < b_i (need to borrow from i+1).
    """
    nd = rng.randint(min_digits, max_digits)
    pattern = rng.choice(["single", "chain", "place", "boundary"])

    if pattern == "single":
        # One position triggers a borrow: a_pos small, b_pos large
        # Other positions: a large, b small → guarantees a > b overall
        pos = rng.randint(0, nd - 1)
        a = 0
        b = 0
        for p in range(nd):
            if p == pos:
                a += rng.randint(0, 4) * (10 ** p)
                b += rng.randint(5, 9) * (10 ** p)
            else:
                a += rng.randint(5, 9) * (10 ** p)
                b += rng.randint(0, 4) * (10 ** p)
    elif pattern == "chain":
        # Long borrow chain: 10^k - small → borrows cascade through zeros
        # e.g., 10000 - 3 = 9997 (borrow propagates through 4 zeros)
        chain_len = min(nd, MAX_DIGITS - 1)  # keep 10^k within 10 digits
        a = 10 ** chain_len
        b = rng.randint(1, 10 ** min(chain_len, 3))
    elif pattern == "place":
        # Random a minus single nonzero digit at one position
        # If a has a 0 there, borrow will propagate
        a = rng.randint(10 ** (nd - 1), 10 ** nd - 1)
        pos = rng.randint(0, nd - 1)
        b = rng.randint(1, 9) * (10 ** pos)
    else:  # boundary
        # Just above a power of 10 minus small → borrows past the boundary
        # e.g., 10001 - 2 = 9999
        power = rng.randint(min_digits, max_digits)
        power = min(power, MAX_DIGITS - 1)
        a = 10 ** power + rng.randint(0, 9)
        b = rng.randint(1, 20)

    cap = 10 ** 10 - 1
    a = max(0, min(a, cap))
    b = max(0, min(b, cap))
    if b > a:
        a, b = b, a  # swap to ensure a >= b
    return a, b


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
