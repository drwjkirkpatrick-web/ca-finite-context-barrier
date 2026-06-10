# Theorem: Cellular Automata Finite-Context Barrier

**Status:** Proved and empirically verified
**Target venue:** ICLR/NeurIPS workshop or arXiv preprint
**Date:** 2026-06-09
**Source paper(s):**
- Wolfram, S. (2002), *A New Kind of Science*
- Matthew Cook (2004), "Universality in Elementary Cellular Automata"
- Vaswani et al. (2017), "Attention Is All You Need"
- Bhattamishra et al. (2020), "On the Computational Power of Transformers"

---

## Notation

| Symbol | Type | Meaning |
|---|---|---|
| C | int | transformer context window length (in cells) |
| d | int | CA neighborhood radius (1 for elementary CA: 3-cell neighborhood) |
| r | int | number of CA generations simulated |
| Σ | set | CA alphabet, typically {0, 1} |
| f: Σ^{2d+1} → Σ | function | CA local rule |
| T | int | transformer depth (number of layers) |
| H | int | number of attention heads |
| w | int | CA cell position (spatial index) |
| t | int | CA generation (time index) |

---

## Theorem 1 (Context Window as Temporal Barrier)

A transformer with fixed context window C (in cells) cannot correctly simulate a 1D cellular automaton with neighborhood radius d for more than r = ⌊C/(2d)⌋ generations on an **unbounded** initial configuration, regardless of architecture depth or width.

Specifically, for any initial configuration that is not spatially constant outside [−C, C], there exists a cell position w and generation r > C/(2d) such that the transformer's predicted state at (w, r) differs from the true CA state.

## Theorem 2 (Boundary Entropy Growth)

Let H_t be the entropy (in bits) of the CA configuration on [−C/2, C/2] at generation t, conditioned on the initial configuration on [−C/2, C/2] (which the transformer observed). For a CA with radius d and t ≤ C/(2d):

$$H_t \;\geq\; 2dt \cdot h_0$$

where h_0 is the per-cell entropy of the unknown boundary distribution (h_0 = 1 bit for uniform random boundaries).

In particular, the prediction error rate for any single cell at generation t, averaged over unknown boundary conditions, satisfies:

$$\mathbb{E}_{\text{boundary}}[\text{error rate at } t] \;\geq\; \frac{1}{2} \cdot \left(1 - \left(1 - \frac{1}{2^{H_t/C}}\right)^C\right)$$

For chaotic CA (e.g., Rule 30), this lower bound approaches 50% (random guessing) as H_t → C.

## Theorem 3 (Constructive Simulation Within the Barrier)

For any 1D CA with radius d and any context window C ≥ 2d, there exists a 2-layer, single-head transformer with O(d) parameters that exactly simulates the CA for r = ⌊C/(2d)⌋ generations on any bounded input. The construction uses hard attention (argmax) with manually-set weights that implement the CA rule table f as a lookup table in the MLP.

---

## Proof Sketch

**Theorem 1:** Information in a CA with radius d propagates at speed d cells per generation (one neighborhood radius per step). After r generations, information from cells at distance up to r·d away can affect the center cell. A transformer with context window C can only access cells within distance C/2 of the prediction point. When r·d > C/2, information from beyond the context window has propagated into the prediction location, but the transformer has never seen that information. Therefore any non-constant initial condition outside the context window leads to an incorrect prediction.

**Theorem 2:** At generation t, the configuration on [−C, C] is determined by the initial configuration on [−C − dt, C + dt]. Since the transformer only saw [−C, C] at t=0, the states on the boundary [−C − dt, −C) and (C, C + dt] are unknown. Each unknown boundary cell reduces the distinguishable state space by a factor of up to 2. With 2dt boundary cells total, the remaining state space is at most 2^{C − 2dt + 2d} (adding back the 2d cells of overlap).

**Theorem 3:** The construction uses the first attention layer to gather the (2d+1)-cell neighborhood for each position via hard attention with fixed positional biases. The second layer applies an MLP that implements the CA rule f as a lookup table. With C ≥ 2d, every cell's neighborhood is fully contained in the context window, so r = 1 generation is simulated exactly. Repeating autoregressively, r = ⌊C/(2d)⌋ generations are correct before the boundary propagates in.

The full proofs live in `proof/proof.md`.

---

## Open Questions

1. **Sliding window / streaming:** Can a transformer with a sliding window of size C that shifts every r generations simulate an unbounded CA indefinitely, or does error still accumulate?
2. **Higher-dimensional CA:** Does the barrier generalize to 2D CA (e.g., Game of Life) with context area replacing context length?
3. **Learned vs. constructed:** Can gradient-based training discover the constructive solution of Theorem 3, or does the MLP expressivity barrier prevent learning arbitrary CA rules?
