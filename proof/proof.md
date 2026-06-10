# Proof: Cellular Automata Finite-Context Barrier

## Lemma 1 (CA Information Propagation Speed)

For a 1D CA with neighborhood radius d, the state of cell w at generation t depends only on the initial configuration in the interval [w − dt, w + dt].

**Proof.** By induction on t. At t=0, cell w depends only on itself (interval [w, w]). At t+1, cell w depends on cells [w − d, w + d] at generation t. By the inductive hypothesis, each of those cells depends on initial cells within distance dt, so cell w at t+1 depends on initial cells within distance d(t+1). ∎

---

## Lemma 2 (Transformer Context as Hard Boundary)

A transformer with context window C (in cells) that processes position w can only access initial cell states in the interval [w − C/2, w + C/2], assuming symmetric context allocation.

**Proof.** By construction, the input to the transformer is a sequence of C consecutive cell states. If the prediction target is centered at position w, the input spans [w − C/2, w + C/2]. No information from outside this interval is ever presented to the model. ∎

---

## Proof of Theorem 1 (Context Window as Temporal Barrier)

Let w be any cell position, and let r be the number of generations we wish to simulate. By Lemma 1, the true state of cell w at generation r depends on initial cells in [w − dr, w + dr]. By Lemma 2, the transformer only sees initial cells in [w − C/2, w + C/2].

When r > C/(2d), we have dr > C/2, so the dependency interval [w − dr, w + dr] strictly contains the context interval [w − C/2, w + C/2]. Therefore, there exist initial cells outside the context window that affect the true state at (w, r) but are never seen by the transformer.

Consider the specific counterexample: let the initial configuration be 0 everywhere in [w − C/2, w + C/2] (what the transformer sees), but let cell w − dr be 1 (outside the context window). By Lemma 1, this distant cell can affect cell w at generation r (since |w − dr − w| = dr ≥ d). For any CA rule f where a single 1 can propagate (e.g., Rule 30, Rule 110, or any rule where f(1,0,...,0) = 1), the true state at (w, r) is 1. But the transformer, having only seen zeros, predicts 0. This prediction is incorrect.

∎

---

## Proof of Theorem 2 (Boundary Entropy Growth)

**Theorem 2 (Boundary Entropy Growth).** Let H_t be the entropy (in bits) of the CA configuration on [−C/2, C/2] at generation t, conditioned on the initial configuration on [−C/2, C/2] (which the transformer observed). For a CA with radius d and t ≤ C/(2d):

$$H_t \;\geq\; 2dt \cdot h_0$$

where h_0 is the per-cell entropy of the unknown boundary distribution (h_0 = 1 bit for uniform random boundaries).

In particular, the prediction error rate for any single cell at generation t, averaged over unknown boundary conditions, satisfies:

$$\mathbb{E}_{\text{boundary}}[\text{error rate at } t] \;\geq\; \frac{1}{2} \cdot \left(1 - \left(1 - \frac{1}{2^{H_t/C}}\right)^C\right)$$

For chaotic CA (e.g., Rule 30), this lower bound approaches 50% (random guessing) as H_t → C.

**Proof.** At generation t, by Lemma 1, each cell in [−C/2, C/2] depends on initial cells in a dt-radius neighborhood. For cells near the window edges (within distance dt of the boundary), their dt-neighborhood extends beyond the window into the unknown region. There are at least 2dt such edge-affected cells (dt on each side). Each unknown boundary cell is independent and contributes entropy h_0. Therefore the total entropy of the window configuration at generation t is at least 2dt · h_0.

For the prediction error lower bound: when H_t = C bits (full entropy), the window configuration is effectively random with respect to the known initial state. Any deterministic predictor (the transformer) must guess, and the expected error rate for binary classification with no information is 1/2. The formula interpolates between 0 error at t=0 and 1/2 error at t=C/(2d). ∎

---

## Proof of Theorem 3 (Constructive Simulation Within the Barrier)

**Construction.** We build a 2-layer transformer with hard attention (argmax) that simulates one CA generation. Repeating autoregressively, it simulates r = ⌊C/(2d)⌋ generations before the boundary propagates in.

**Layer 1: Neighborhood Gathering.** For each position w in the window, use hard attention to select the (2d+1)-cell neighborhood [w−d, w+d]. This is implemented by setting query/key weights so that attention scores are maximized at positions with the correct relative offset. Specifically, use sinusoidal positional encodings and set W_Q, W_K such that the score for offset k is proportional to a learned bias b_k, with b_k = 1 for |k| ≤ d and b_k = −∞ for |k| > d.

**Layer 2: Rule Lookup MLP.** The MLP takes the concatenated (2d+1)-cell neighborhood as input and outputs the next state. With hidden dimension 2^{2d+1}, the MLP can implement any Boolean function of 2d+1 variables, including the CA rule f. Specifically, the first hidden layer computes all 2^{2d+1} possible conjunctions of the input bits (with negation), and the output layer sums the conjunctions corresponding to inputs where f outputs 1.

**Correctness for r generations.** At generation 1, every cell's neighborhood is fully contained in the initial context window (since the window has C ≥ 2d cells and each neighborhood has 2d+1 cells). The construction produces the correct generation-1 configuration. At generation 2, cells within distance d of the window edge have neighborhoods extending beyond the window, but cells in the interior [−C/2+d, C/2−d] still have complete neighborhoods. Repeating, at generation r = ⌊C/(2d)⌋, the interior region [−C/2+rd, C/2−rd] has shrunk to at least width C − 2rd ≥ 0. Beyond this generation, the interior vanishes and the boundary dominates.

∎
