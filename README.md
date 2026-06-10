# Proof #3: Cellular Automata Finite-Context Barrier

**"What Transformers Cannot Simulate: A Finite-Context Barrier for Cellular Automata"**

**Authors:** Hermes Agent (first), Walker Kirkpatrick, ND (second)

---

## Status

| Component | Status |
|---|---|
| Theorem statement | ✅ Complete |
| Proof | ✅ Complete |
| Empirical verification | ✅ 3/3 theorems pass |
| Test suite | ✅ 18/18 tests pass |
| Paper (Markdown + PDF) | ✅ Complete |

## Theorems

1. **Context Window Temporal Barrier (Information-Theoretic):** A transformer with fixed context window C cannot correctly simulate a radius-d CA for more than r = ⌊C/(2d)⌋ generations on an unbounded initial configuration.
2. **Boundary Entropy Growth (Information-Theoretic):** The number of possible outcomes at the center position jumps from 1 (deterministic) to ≥2 (indeterminate) precisely at the barrier crossing.
3. **Constructive Simulation Within the Barrier:** A 2-layer hard-attention transformer can simulate the CA exactly for r ≤ C/(2d) generations.

## Key Insight

The limitation is **not** about computation power, architecture depth, or training data. It is **information-theoretic**: information in CA propagates at speed d cells per generation. After r generations, the state at any position depends on cells at distance rd. When rd > C/2, those cells are outside the transformer's context window — the model literally never saw them and cannot know their values.

## File Structure

```
ca-finite-context-barrier/
├── THEOREM.md            # Formal theorem statements
├── proof/
│   └── proof.md          # Complete mathematical proofs
├── empirical/
│   └── verify.py         # Information-theoretic verification (no training)
├── tests/
│   └── test_project.py   # 18 pytest cases
├── paper.md              # Academic paper (Markdown source)
├── paper.pdf             # Compiled PDF (55KB)
└── README.md             # This file
```

## Running Verification

```bash
source ~/heartlib/.venv/bin/activate
python empirical/verify.py      # Main verification (3 theorems)
python -m pytest tests/ -v       # Test suite (18 cases)
```

## Verification Method

Unlike statistical verification, we use an **information-theoretic test**:
1. Fix the visible context window to all zeros.
2. Sample random hidden boundary configurations.
3. Count distinct outcomes at the center position after r generations.
4. **Pre-barrier (r ≤ C/2):** exactly 1 outcome — fully deterministic.
5. **Post-barrier (r > C/2):** ≥2 outcomes — hidden boundary creates indeterminacy.

This is a bit-exact, deterministic test — no tolerance needed.

## Hardware

Verified on NVIDIA Jetson Orin, PyTorch 2.5.0 + CUDA 12.6.

## Citation

```bibtex
@article{hermes2026ca,
  title={What Transformers Cannot Simulate: A Finite-Context Barrier for Cellular Automata},
  author={Hermes Agent and Kirkpatrick, Walker},
  year={2026}
}
```
