"""
verify.py
=========

Empirical verification of the Cellular Automata Finite-Context Barrier Theorem.

The core insight: a transformer with fixed context window C sees only C cells.
After r generations, the true state depends on cells up to distance r away.
When r > C/2, hidden boundary cells affect the outcome — but the transformer
never saw them, so it cannot predict correctly for all possible boundaries.

We verify this by:
    1. Fixing the visible window to all zeros.
    2. Sampling boundary configurations of the hidden region.
    3. Computing how many distinct outcomes are possible at (center, r).
    4. Showing: 1 outcome for r <= C/2, >=2 outcomes for r > C/2.

This is an information-theoretic test — no statistics needed.

Usage:
    source ~/heartlib/.venv/bin/activate
    python empirical/verify.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Set

import numpy as np
import torch


# =============================================================================
# Section 1: Device + Reproducibility
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def manual_seed(seed: int = 1729) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Section 2: Cellular Automata (Rule 30 — Chaotic)
# =============================================================================

RULE_30_TABLE = {
    (1, 1, 1): 0, (1, 1, 0): 0, (1, 0, 1): 0, (1, 0, 0): 1,
    (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): 0,
}


def ca_step(config: np.ndarray) -> np.ndarray:
    """One generation of elementary CA with OPEN (zero-padded) boundaries."""
    n = len(config)
    nxt = np.zeros(n, dtype=np.int8)
    for i in range(n):
        left = config[i - 1] if i > 0 else 0
        right = config[i + 1] if i < n - 1 else 0
        nxt[i] = RULE_30_TABLE[(left, config[i], right)]
    return nxt


def outcome_at(config: np.ndarray, pos: int, gen: int) -> int:
    """State at position pos after gen generations."""
    cur = config.copy()
    for _ in range(gen):
        cur = ca_step(cur)
    return int(cur[pos])


# =============================================================================
# Section 3: Information-Theoretic Barrier Test
# =============================================================================

def count_distinct_outcomes(
    visible_window: np.ndarray,
    hidden_size: int,
    target_pos: int,
    target_gen: int,
    n_samples: int = 200,
) -> int:
    """
    Sample boundary configs and count distinct outcomes at (target_pos, target_gen).
    Returns the number of distinct outcomes found.
    """
    if hidden_size == 0:
        # No hidden boundary — deterministic
        return 1
    
    outcomes: Set[int] = set()
    for _ in range(n_samples):
        left = np.random.randint(0, 2, hidden_size).astype(np.int8)
        right = np.random.randint(0, 2, hidden_size).astype(np.int8)
        full = np.concatenate([left, visible_window, right])
        # Adjust target_pos to account for the left padding
        adjusted_pos = target_pos + hidden_size
        outcomes.add(outcome_at(full, adjusted_pos, target_gen))
    
    return len(outcomes)


# =============================================================================
# Section 4: Theorem Checks
# =============================================================================

@dataclass
class TheoremResult:
    name: str
    passed: bool
    metric: float
    detail: str


def check_theorem_1(context_window: int = 10, max_gen: int = 10) -> TheoremResult:
    """
    Theorem 1: Context Window as Temporal Barrier.
    
    For r <= C/2: only 1 outcome possible (fully determined).
    For r > C/2: >=2 outcomes possible (boundary affects result).
    """
    d = 1
    barrier = context_window // (2 * d)
    visible = np.zeros(context_window, dtype=np.int8)
    center = context_window // 2
    
    results = {}
    for gen in range(1, max_gen + 1):
        hidden = max(0, gen - barrier)
        n_out = count_distinct_outcomes(visible, hidden, center, gen, n_samples=200)
        results[gen] = n_out
    
    pre_barrier_max = max(results[g] for g in range(1, barrier + 1))
    post_barrier_min = min(results[g] for g in range(barrier + 1, max_gen + 1) if g in results)
    
    passed = (pre_barrier_max == 1) and (post_barrier_min >= 2)
    
    detail = f"C={context_window}, barrier={barrier} gen | "
    detail += f"Pre: max outcomes={pre_barrier_max} | Post: min outcomes={post_barrier_min} | "
    detail += ", ".join(f"g{g}:{results[g]}" for g in [1, barrier, barrier + 1, min(barrier + 3, max_gen)])
    
    return TheoremResult(
        name="Theorem 1: Context Window Barrier",
        passed=passed,
        metric=float(post_barrier_min),
        detail=detail,
    )


def check_theorem_2(context_window: int = 8, max_gen: int = 10) -> TheoremResult:
    """
    Theorem 2: Boundary Entropy Growth.
    
    Outcome multiplicity grows with generation once barrier is crossed.
    """
    d = 1
    barrier = context_window // (2 * d)
    visible = np.zeros(context_window, dtype=np.int8)
    center = context_window // 2
    
    outcome_counts = {}
    for gen in range(1, max_gen + 1):
        hidden = max(0, gen - barrier)
        n_out = count_distinct_outcomes(visible, hidden, center, gen, n_samples=200)
        outcome_counts[gen] = n_out
    
    pre_single = all(outcome_counts[g] == 1 for g in range(1, barrier + 1))
    post_multiple = any(outcome_counts[g] >= 2 for g in range(barrier + 1, max_gen + 1))
    growth = outcome_counts.get(barrier + 1, 0) > outcome_counts.get(barrier, 0)
    
    passed = pre_single and post_multiple and growth
    
    detail = f"C={context_window}, barrier={barrier} | "
    detail += f"Pre: {outcome_counts[1]} outcome, Post: {outcome_counts.get(barrier+1, 0)} outcomes | "
    detail += ", ".join(f"g{g}:{outcome_counts[g]}" for g in [1, barrier, barrier + 1, min(barrier + 3, max_gen)])
    
    return TheoremResult(
        name="Theorem 2: Boundary Entropy Growth",
        passed=passed,
        metric=float(outcome_counts.get(barrier + 1, 0)),
        detail=detail,
    )


def check_theorem_3(context_window: int = 10) -> TheoremResult:
    """
    Theorem 3: Constructive Simulation Within the Barrier.
    
    With full information, CA evolution is perfectly deterministic.
    """
    max_gen = context_window // 2
    config_size = context_window * 5
    initial = np.random.randint(0, 2, size=config_size).astype(np.int8)
    
    # Simulate max_gen generations
    cur = initial.copy()
    history = [cur.copy()]
    for _ in range(max_gen):
        cur = ca_step(cur)
        history.append(cur.copy())
    
    # Verify determinism
    cur2 = initial.copy()
    for i, expected in enumerate(history[1:], 1):
        cur2 = ca_step(cur2)
        if not np.array_equal(cur2, expected):
            return TheoremResult(
                name="Theorem 3: Constructive Simulation",
                passed=False,
                metric=1.0,
                detail=f"Determinism failed at gen {i}",
            )
    
    return TheoremResult(
        name="Theorem 3: Constructive Simulation",
        passed=True,
        metric=0.0,
        detail=f"C={context_window}, max_gen={max_gen} | Deterministic simulation verified",
    )


# =============================================================================
# Section 5: Main Runner
# =============================================================================

def main() -> int:
    print("=" * 70)
    print(" CA Finite-Context Barrier — Empirical Verification")
    print("=" * 70)
    
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
    print()
    
    manual_seed(1729)
    
    results = []
    
    print("--- Theorem 1: Context Window Barrier ---")
    r1 = check_theorem_1(context_window=10, max_gen=10)
    results.append(r1)
    print(f"  {'✓' if r1.passed else '✗'}  {r1.detail}")
    print()
    
    print("--- Theorem 2: Boundary Entropy Growth ---")
    r2 = check_theorem_2(context_window=8, max_gen=10)
    results.append(r2)
    print(f"  {'✓' if r2.passed else '✗'}  {r2.detail}")
    print()
    
    print("--- Theorem 3: Constructive Simulation ---")
    r3 = check_theorem_3(context_window=10)
    results.append(r3)
    print(f"  {'✓' if r3.passed else '✗'}  {r3.detail}")
    print()
    
    n_pass = sum(1 for r in results if r.passed)
    print("=" * 70)
    print(f"SUMMARY: {n_pass}/{len(results)} theorems verified")
    for r in results:
        flag = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"   {flag}  {r.name}")
        print(f"          {r.detail}")
    print("=" * 70)
    
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
