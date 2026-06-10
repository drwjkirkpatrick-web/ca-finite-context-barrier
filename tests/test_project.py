"""
test_project.py
===============

pytest suite for the CA Finite-Context Barrier proof project.

Run with:
    source ~/heartlib/.venv/bin/activate
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))
from verify import (
    ca_step,
    outcome_at,
    count_distinct_outcomes,
    check_theorem_1,
    check_theorem_2,
    check_theorem_3,
    RULE_30_TABLE,
    get_device,
    manual_seed,
)


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def seed():
    manual_seed(1729)


class TestCAStep:
    """Unit tests for the CA simulator."""

    def test_ca_step_open_boundary(self):
        """Open boundaries: edge cells see 0 outside."""
        config = np.array([1, 1, 1], dtype=np.int8)
        nxt = ca_step(config)
        # Left edge: (0,1,1) → 1; Center: (1,1,1) → 0; Right edge: (1,1,0) → 0
        expected = np.array([1, 0, 0], dtype=np.int8)
        assert np.array_equal(nxt, expected)

    def test_ca_step_rule_30_known(self):
        """Verify Rule 30 on a known 5-cell config."""
        config = np.array([0, 0, 1, 0, 0], dtype=np.int8)
        nxt = ca_step(config)
        # (0,0,0)→0, (0,0,1)→1, (0,1,0)→1, (1,0,0)→1, (0,0,0)→0
        expected = np.array([0, 1, 1, 1, 0], dtype=np.int8)
        assert np.array_equal(nxt, expected)

    def test_ca_step_deterministic(self):
        """Same input always produces same output."""
        config = np.random.randint(0, 2, 20).astype(np.int8)
        n1 = ca_step(config)
        n2 = ca_step(config)
        assert np.array_equal(n1, n2)


class TestOutcomeAt:
    """Tests for computing state at a specific position after generations."""

    def test_outcome_at_zero_generations(self):
        """At generation 0, outcome is just the initial state."""
        config = np.array([1, 0, 1, 0], dtype=np.int8)
        assert outcome_at(config, 0, 0) == 1
        assert outcome_at(config, 1, 0) == 0
        assert outcome_at(config, 2, 0) == 1

    def test_outcome_at_one_generation(self):
        """After 1 generation, outcome matches ca_step."""
        config = np.array([0, 0, 1, 0, 0], dtype=np.int8)
        expected = ca_step(config)
        for i in range(len(config)):
            assert outcome_at(config, i, 1) == expected[i]

    def test_outcome_at_multiple_generations(self):
        """After multiple generations, outcome is consistent with repeated steps."""
        config = np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
        cur = config.copy()
        for gen in range(1, 4):
            cur = ca_step(cur)
            for i in range(len(cur)):
                assert outcome_at(config, i, gen) == cur[i]


class TestDistinctOutcomes:
    """Tests for the information-theoretic barrier test."""

    def test_no_hidden_boundary_deterministic(self):
        """With no hidden boundary (hidden=0), only 1 outcome."""
        visible = np.zeros(10, dtype=np.int8)
        n_out = count_distinct_outcomes(visible, 0, 5, 3, n_samples=50)
        assert n_out == 1

    def test_hidden_boundary_creates_variation(self):
        """With hidden boundary, multiple outcomes are possible."""
        visible = np.zeros(10, dtype=np.int8)
        # Gen 6 with C=10, barrier=5: hidden=1 cell on each side
        n_out = count_distinct_outcomes(visible, 1, 5, 6, n_samples=200)
        assert n_out >= 2, f"Expected >=2 outcomes, got {n_out}"

    def test_barrier_property_consistent(self):
        """Pre-barrier: 1 outcome. Post-barrier: >=2 outcomes."""
        C = 10
        barrier = C // 2
        visible = np.zeros(C, dtype=np.int8)
        center = C // 2

        for gen in range(1, barrier + 1):
            n_out = count_distinct_outcomes(visible, 0, center, gen, n_samples=50)
            assert n_out == 1, f"Pre-barrier gen {gen}: expected 1, got {n_out}"

        for gen in range(barrier + 1, barrier + 4):
            hidden = gen - barrier
            n_out = count_distinct_outcomes(visible, hidden, center, gen, n_samples=200)
            assert n_out >= 2, f"Post-barrier gen {gen}: expected >=2, got {n_out}"


class TestTheorem1:
    """Theorem 1: Context Window Barrier."""

    def test_pass_default(self):
        r = check_theorem_1(context_window=10, max_gen=10)
        assert r.passed, f"Theorem 1 failed: {r.detail}"

    def test_small_context_window(self):
        r = check_theorem_1(context_window=6, max_gen=8)
        assert r.passed, f"Small window failed: {r.detail}"

    def test_large_context_window(self):
        r = check_theorem_1(context_window=16, max_gen=12)
        assert r.passed, f"Large window failed: {r.detail}"


class TestTheorem2:
    """Theorem 2: Boundary Entropy Growth."""

    def test_pass_default(self):
        r = check_theorem_2(context_window=8, max_gen=10)
        assert r.passed, f"Theorem 2 failed: {r.detail}"

    def test_growth_trend(self):
        r = check_theorem_2(context_window=6, max_gen=10)
        assert r.passed, f"Growth trend failed: {r.detail}"


class TestTheorem3:
    """Theorem 3: Constructive Simulation."""

    def test_pass_default(self):
        r = check_theorem_3(context_window=10)
        assert r.passed, f"Theorem 3 failed: {r.detail}"

    def test_different_window_sizes(self):
        for C in [6, 10, 14]:
            r = check_theorem_3(context_window=C)
            assert r.passed, f"C={C} failed: {r.detail}"


class TestRuleTables:
    """Sanity checks on CA rule tables."""

    def test_rule_30_exhaustive(self):
        """Rule 30 covers all 8 possible neighborhoods."""
        assert len(RULE_30_TABLE) == 8
        for (l, c, r), out in RULE_30_TABLE.items():
            assert l in (0, 1) and c in (0, 1) and r in (0, 1)
            assert out in (0, 1)

    def test_rule_30_not_constant(self):
        """Rule 30 produces both 0 and 1 outputs."""
        outputs = set(RULE_30_TABLE.values())
        assert outputs == {0, 1}
