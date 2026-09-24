"""

Sparse Autoencoders (sae.py)

What it is. An SAE learns a sparse dictionary of directions in the hidden space. 
Each direction is supposed to correspond to a monosemantic feature — a single human-interpretable concept,
unlike the polysemantic neurons of the raw model. SAEs have become the standard tool for mechanistic interpretability, 
and 2026 has seen a wave of improvements: AdaptiveK SAE dynamically adjusts sparsity per input, ClassifSAE targets the classifier's decision specifically, 
and LowRank SAEs introduce geometric regularisation that improves feature orthogonality.

Why it matters for your project. An SAE trained on the layer-9 hidden states would give you a set of emotion features — 
sparse, interpretable directions that you can name and plot. You could then ask: "does the model have a single 'joy' feature, or 
is joy represented by the combination of a 'positive-valenced' feature and an 'aroused' feature?" That is a level of structural detail that 
probing alone cannot provide.

Feasibility note. Training an SAE on a 5000 × 768 matrix for 10,000 steps 
takes ~30 minutes on CPU. It is expensive but not prohibitive for a handful of (model, layer) pairs."""

# ── Path bootstrap: keep `from _shared import ...` working from expl/ ──
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────






# sae.py
"""
Sparse Autoencoder for a single (model, layer) pair.

Architecture: linear encoder → ReLU → linear decoder (tied or untied),
with an L1 penalty on the hidden activations. Adam. This is the
Bricken 2023 / Cunningham 2023 vanilla SAE — it is the correct baseline.

We train one SAE per (model, dataset, layer). Layers worth training are
those where probing peaks; the point of the SAE is to decompose that
layer's representation into monosemantic features you can name.

Cost: on a 5000 x 768 float32 matrix, 10k steps at batch 512 takes
~20 minutes on CPU. Budget accordingly.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from _shared import (
    artifact_dir, analysis_dir_for_extraction,
    atomic_json, atomic_npz, load_hidden_states,
)

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        # Scale-invariant init: initial decoder columns unit-norm.
        with torch.no_grad():
            self.decoder.weight.data = self.decoder.weight.data / \
                self.decoder.weight.data.norm(dim=0, keepdim=True)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

def train_sae(X, hidden_mult=4, l1_coef=1e-3, steps=10_000,
              batch_size=512, lr=3e-4, seed=42, log_every=500):
    """
    X : [N, D] float32, already the layer's hidden states.
    hidden_mult : dictionary size / input size (4 is standard for D<1000).
    l1_coef : L1 sparsity coefficient. 1e-3 is a safe starting point.
    """
    torch.manual_seed(seed)
    N, D = X.shape
    H = D * hidden_mult
    sae = SparseAutoencoder(D, H)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    Xt = torch.from_numpy(X.astype(np.float32))
    losses = []
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,))
        xb = Xt[idx]
        x_hat, z = sae(xb)
        mse = ((x_hat - xb) ** 2).mean()
        l1 = z.abs().mean()
        loss = mse + l1_coef * l1
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            losses.append({"step": step, "mse": float(mse), "l1": float(l1),
                           "active": float((z > 0).float().mean())})
    return sae, losses

def feature_statistics(sae, X, top_k=20):
    """
    For each hidden unit, count how often it activates and what the
    mean activation is. Then return the top-k most active units.
    """
    with torch.no_grad():
        _, z = sae(torch.from_numpy(X.astype(np.float32)))
    z = z.numpy()
    active = (z > 0).mean(axis=0)         # [H]
    mean_act = z.mean(axis=0)
    top = np.argsort(-active)[:top_k]
    return [{
        "unit": int(u),
        "active_fraction": float(active[u]),
        "mean_activation": float(mean_act[u]),
    } for u in top]

def run_sae(model_slug_: str, dataset: str, layer_index: int,
            sample_cap: int = 5000, hidden_mult: int = 4,
            l1_coef: float = 1e-3, steps: int = 10_000) -> Path:
    t0 = time.perf_counter()
    states = load_hidden_states(model_slug_, dataset)
    N = states.shape[0]
    idx = np.arange(N) if N <= sample_cap else np.random.default_rng(42).choice(N, sample_cap, replace=False)
    X = np.asarray(states[idx, layer_index, :], dtype=np.float32)

    sae, losses = train_sae(X, hidden_mult=hidden_mult, l1_coef=l1_coef, steps=steps)
    stats = feature_statistics(sae, X, top_k=40)

    out_dir = analysis_dir_for_extraction(model_slug_, dataset) / "sae"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"layer{layer_index}_h{hidden_mult}_l1{l1_coef}"
    torch.save(sae.state_dict(), out_dir / f"sae_{tag}.pt")
    atomic_npz(out_dir / f"sae_{tag}_weights.npz",
               W_enc=sae.encoder.weight.detach().numpy(),
               b_enc=sae.encoder.bias.detach().numpy(),
               W_dec=sae.decoder.weight.detach().numpy())
    atomic_json(out_dir / f"sae_{tag}.json", {
        "technique": "vanilla_sparse_autoencoder",
        "model_slug": model_slug_, "dataset": dataset,
        "layer_index": int(layer_index),
        "hidden_mult": hidden_mult, "l1_coef": l1_coef, "steps": steps,
        "n_samples": int(len(idx)),
        "losses": losses,
        "top_features": stats,
        "elapsed_seconds": time.perf_counter() - t0,
    })
    return out_dir / f"sae_{tag}.pt"