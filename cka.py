from __future__ import annotations
"""
Representational similarity within and across models

What CKA is, in plain language. Imagine two layers L1 and L2. 
For every pair of samples (i, j), layer L1 has a notion of "how similar are 
sample i and sample j" — measured by the inner product of their hidden vectors. 
Layer L2 has its own notion. CKA asks: do the two layers agree on the similarity structure? 
It returns a number between 0 (unrelated) and 1 (identical structure). 
Crucially, it does not require the two layers to have the same dimensionality — so you can compare a 768-dim BERT layer to a 1024-dim Qwen layer.

Why this matters for your argument. This is the single most powerful technique for the question "do different models learn the same emotion code?" 
If BERT's layer 6 and GPT-2's layer 8 have CKA = 0.82 on the ISEAR dataset, they are encoding emotion in a structurally similar way. 
That is a strong result — it means the emotion representation is not a quirk of one architecture's training, but a shared consequence of learning from natural text.

If the CKA is near zero (say 0.15) but the probe scores are similar, that is also a strong result — it means the two models both learn to classify emotion but use different 
internal codes, which supports the "multiple sufficient representations" view in the interpretability literature.

Either way, you get a publishable finding. The only uninteresting outcome is if you never compute it.

The 2026 literature context. The "Platonic Representation Hypothesis" was recently formalised and tested at scale. 
A March 2026 paper reports CKA ≥ 0.97 between Llama 3.3 70B and Qwen 2.5 72B on semantic prompts. A May 2026 paper 
adds a crucial nuance: models converge more on problems they collectively fail (CKA = 0.897) than on those they solve 
(CKA = 0.830), and pre-decision representations align (CKA = 0.875) while post-decision representations diverge (CKA = 0.274). 
This tells you that the layer you choose matters enormously — early and middle layers are where convergence lives. 
If you compare last layers and find low CKA, you may be missing the interesting story.
"""


"""
cka.py — Centered Kernel Alignment between layers (within a model) and
between models (at corresponding depths).

Thesis question answered:
    "Is the structure of the emotional representation the same in BERT's
     layer 6 as it is in GPT-2's layer 8? And how does that similarity
     evolve as we move through depth?"

Method:
    Linear CKA (Kornblith et al., 2019):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    where X and Y are the [N, D1] and [N, D2] hidden states for the same N
    samples. The formula is basis-invariant and scale-invariant.

Reads:  hidden_states/<slug>/<dataset>/hidden_states.npy
Writes: hidden_states/<slug>/<dataset>/analysis/cka_within.npz
        hidden_states/<slug>/<dataset>/analysis/cka_within.json
        probe/<slug_a>__<slug_b>/<dataset>/cka_cross.npz   (cross-model only)
"""


from pathlib import Path
import time
import numpy as np
import torch

from _shared import (
    analysis_dir_for_extraction, atomic_json, atomic_npz,
    load_hidden_states, stable_hash,
)



# cka.py
"""
CKA — similarity of representations across layers and models.

Two modes:
    run_cka_within(model_slug, dataset)   → [L, L] matrix per model
    run_cka_cross(slug_a, slug_b, dataset) → [L_a, L_b] matrix across models
"""


import time
import pandas as pd
from _shared import (
    artifact_dir, analysis_dir_for_extraction,
    atomic_json, atomic_npz, load_hidden_states,
)

def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def linear_cka(X, Y):
    """
    Linear CKA between two [N, D1] and [N, D2] matrices.
    Uses the linear kernel; O(N^2) but we cap N to ~2000.
    """
    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T
    Kc, Lc = _center(K), _center(L)
    num = (Kc * Lc).sum()
    den = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float(num / den) if den > 0 else 0.0

def _subsample(N, cap, seed=42):
    if N <= cap:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, cap, replace=False))


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# CKA is O(N² · D) for the Gram matrices. On 8 GB of RAM, N = 800 and
# D = 3584 is the safe upper bound. Raising N beyond this will thrash.
# The value of N only affects the *precision* of the CKA estimate, not its
# qualitative shape — 500 samples is plenty for a smooth depth curve.

CKA_SAMPLE_CAP = 500
CKA_DTYPE      = torch.float64   # float64 for numerical stability of the Frobenius norms
SEED           = 42


# ─────────────────────────────────────────────────────────────────────────────
# Core primitive
# ─────────────────────────────────────────────────────────────────────────────

def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between two representations of the same N samples.

    Parameters
    ----------
    X : [N, D1] torch.float64 — layer L1's hidden states for N samples.
    Y : [N, D2] torch.float64 — layer L2's hidden states for the SAME
        N samples, in the SAME row order. This pairing is non-negotiable:
        if row i of X and row i of Y are not the same input sentence, the
        CKA number is meaningless.

    Returns
    -------
    cka_value : float in [0, 1]. 1 = identical structure. 0 = orthogonal.
        Values below ~0.05 are noise on a sample of N=500.

    Implementation walk-through:
        1. We center both X and Y along the sample axis. This removes the
           mean representation, which would otherwise dominate the Gram
           matrices and inflate CKA toward 1 for every pair of layers
           (they all have a similar mean direction).
        2. We compute linear Gram matrices K_X = X X^T and K_Y = Y Y^T.
           These are [N, N] matrices whose entry (i, j) is the inner product
           of sample i and sample j in the respective representation.
        3. We do NOT center the Gram matrices again. Linear CKA already
           centers the features, so the Gram matrices have zero-mean rows
           and columns implicitly.
        4. The numerator is <K_X, K_Y>_F = sum(K_X * K_Y).
           The denominator is sqrt(<K_X, K_X>_F * <K_Y, K_Y>_F).
        5. The ratio is CKA.

    Numerical note: we add 1e-12 to the denominator to avoid division by
    zero when both layers are constant (which should not happen for real
    hidden states, but a defended code base handles the degenerate case).
    """
    X = X.double(); Y = Y.double()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.t()             # [N, N] linear Gram of layer 1
    L = Y @ Y.t()             # [N, N] linear Gram of layer 2

    hsic = (K * L).sum()      # <K, L>_F
    denom = torch.sqrt((K * K).sum() * (L * L).sum()) + 1e-12
    return float(hsic / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Within-model CKA
# ─────────────────────────────────────────────────────────────────────────────

def cka_within_model(
    states: np.ndarray,       # memmap [N, L, D]
    *,
    sample_cap: int = CKA_SAMPLE_CAP,
    seed: int = SEED,
) -> np.ndarray:
    """
    Compute the [L, L] CKA matrix between every pair of layers in one model.

    Returns
    -------
    M : [L, L] float32 — M[i, j] = CKA(layer_i, layer_j). The matrix is
        symmetric with 1.0 on the diagonal (a layer is perfectly similar
        to itself, modulo numerical noise).

    Interpretation guide:
        • High CKA in a contiguous block (say layers 4-9 all mutually > 0.9)
          means those layers form a "computational stage" that refines the
          representation without changing its structure.
        • A sharp drop in CKA between layer k and layer k+1 means the
          representation is fundamentally reorganised at that boundary.
          That is often the layer where the model switches from syntactic
          to semantic processing.
        • Low CKA everywhere (all entries < 0.3) means each layer has its
          own representation and the depth profile of emotion must be
          read off the probe curves, not the CKA matrix.
    """
    n_layers = states.shape[1]

    if states.shape[0] > sample_cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(states.shape[0], sample_cap, replace=False))
    else:
        idx = np.arange(states.shape[0])

    # Materialise one [N, D] block per layer. For N=500, D=3584, that is
    # 7 MB per layer. With 33 layers (Qwen3-4B), 230 MB total. Acceptable.
    per_layer = [
        torch.from_numpy(np.asarray(states[idx, l, :], dtype=np.float32))
        for l in range(n_layers)
    ]

    M = np.zeros((n_layers, n_layers), dtype=np.float32)
    for i in range(n_layers):
        for j in range(i, n_layers):
            v = cka(per_layer[i], per_layer[j])
            M[i, j] = M[j, i] = v
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model CKA
# ─────────────────────────────────────────────────────────────────────────────

def cka_cross_model(
    states_a: np.ndarray,     # memmap [Na, La, Da]
    states_b: np.ndarray,     # memmap [Nb, Lb, Db]
    *,
    sample_cap: int = CKA_SAMPLE_CAP,
    seed: int = SEED,
) -> np.ndarray:
    """
    Compute the [La, Lb] CKA matrix between two models' layers.

    HARD REQUIREMENT: the two models must have seen the same input sentences
    in the same row order. If the datasets differ (e.g. model A processed
    ISEAR and model B processed GoEmotions), the CKA is not meaningful.
    The caller is responsible for asserting this; we do not attempt to
    detect it because a shuffled pairing can still produce a plausible-
    looking CKA matrix.

    Returns
    -------
    M : [La, Lb] float32 — M[i, j] = CKA(layer_i_of_A, layer_j_of_B).
    """
    n = min(states_a.shape[0], states_b.shape[0], sample_cap)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(min(states_a.shape[0], states_b.shape[0]),
                             n, replace=False))

    La, Lb = states_a.shape[1], states_b.shape[1]

    A = [torch.from_numpy(np.asarray(states_a[idx, l, :], dtype=np.float32))
         for l in range(La)]
    B = [torch.from_numpy(np.asarray(states_b[idx, l, :], dtype=np.float32))
         for l in range(Lb)]

    M = np.zeros((La, Lb), dtype=np.float32)
    for i in range(La):
        for j in range(Lb):
            M[i, j] = cka(A[i], B[j])
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_cka_within(model_slug_: str, dataset: str,
                   sample_cap: int = CKA_SAMPLE_CAP) -> Path:
    """Compute and save the within-model CKA matrix for one (model, dataset)."""
    t0 = time.perf_counter()
    states = load_hidden_states(model_slug_, dataset)
    M = cka_within_model(states, sample_cap=sample_cap)

    out_dir = analysis_dir_for_extraction(model_slug_, dataset)
    atomic_npz(out_dir / "cka_within.npz", cka=M)
    atomic_json(out_dir / "cka_within.json", {
        "technique": "linear_cka_within_model",
        "model_slug": model_slug_,
        "dataset": dataset,
        "n_samples_used": min(states.shape[0], sample_cap),
        "n_layers": int(states.shape[1]),
        "shape": list(M.shape),
        "min_offdiag": float(M[~np.eye(M.shape[0], dtype=bool)].min()),
        "max_offdiag": float(M[~np.eye(M.shape[0], dtype=bool)].max()),
        "mean_offdiag": float(M[~np.eye(M.shape[0], dtype=bool)].mean()),
        "elapsed_seconds": time.perf_counter() - t0,
        "output": str(out_dir / "cka_within.npz"),
    })
    return out_dir / "cka_within.npz"


def run_cka_cross(
    model_slug_a: str, model_slug_b: str, dataset: str,
    *,
    sample_cap: int = CKA_SAMPLE_CAP,
) -> Path:
    """
    Compute and save the cross-model CKA matrix. Writes to probe/ so the
    result is not duplicated inside hidden_states/.
    """
    from _shared import PROBE_ROOT
    t0 = time.perf_counter()
    A = load_hidden_states(model_slug_a, dataset)
    B = load_hidden_states(model_slug_b, dataset)
    M = cka_cross_model(A, B, sample_cap=sample_cap)

    out_dir = PROBE_ROOT / "_cross_model" / f"{model_slug_a}__{model_slug_b}" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(out_dir / "cka_cross.npz", cka=M)
    atomic_json(out_dir / "cka_cross.json", {
        "technique": "linear_cka_cross_model",
        "model_a": model_slug_a, "model_b": model_slug_b,
        "dataset": dataset,
        "shape": list(M.shape),
        "best_pair": [int(np.unravel_index(M.argmax(), M.shape)[0]),
                      int(np.unravel_index(M.argmax(), M.shape)[1])],
        "best_value": float(M.max()),
        "elapsed_seconds": time.perf_counter() - t0,
    })
    return out_dir / "cka_cross.npz"