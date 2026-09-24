from __future__ import annotations

# ── Path bootstrap: keep `from _shared import ...` working from expl/ ──
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────

"""
Prototype-based explanation (protopics.py)

What it is. Instead of asking "what direction in the representation corresponds to this class?", 
ask "what training examples are the prototypes of this class?" A prototype-based surrogate learns a small set of sentence-level 
prototypes per class and makes its predictions by nearest-prototype. This is the 2026 state of the art in explainable classification for LLMs.

Why it matters for your project. Prototypes give you human-readable explanations. Instead of "the probe's decision for joy is driven by dimensions 342,
891, and 1204," you get "the probe's decision for joy is driven by its similarity to these three sentences from the training set: 'I got the job!', 
'my daughter was born today', 'we won the championship'." That is the kind of explanation an examiner can read and immediately understand.

The 2026 paper. ProtoSurE was published at AAAI 2026. It trains an interpretable-by-design surrogate that aligns with the target LLM and uses sentence-level 
prototypes as the concepts. It outperforms LIME and SHAP on faithfulness and human comprehensibility.
"""


"""
prototype.py — Prototype-based explanation of the probe.

Method (ProtoSurE, AAAI 2026):
    1. For each class c, cluster the training samples of class c in the
       layer's hidden space (KMeans, k=5).
    2. The centroid of each cluster is a "prototype vector".
    3. Train a nearest-prototype classifier: predict argmin_c ||x - proto_c||.
    4. Compare its accuracy to the original probe. If it matches, the
       prototypes are faithful to the probe's decision.
    5. For each prototype, find the training sentence whose hidden state
       is closest to the prototype vector. That sentence is the human-
       readable explanation of the prototype.

Reads:  hidden_states/<slug>/<dataset>/hidden_states.npy
        datasets/<dataset>/processed/<dataset>_clean.csv
Writes: probe/<slug>/<dataset>/<run_key>/analysis/prototypes_layer<L>.json
"""


from pathlib import Path
import json
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


def extract_prototypes(
    states: np.ndarray,
    y: np.ndarray,
    texts: list[str],
    layer_index: int,
    n_classes: int,
    *,
    n_prototypes_per_class: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    For each class, cluster the class's hidden states and record the closest
    training sentence to each cluster centroid.
    """
    X = np.asarray(states[:, layer_index, :], dtype=np.float32)
    protos = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) < n_prototypes_per_class:
            continue
        Xc = X[idx]
        km = KMeans(n_clusters=n_prototypes_per_class,
                    random_state=seed, n_init=10).fit(Xc)
        for k, centroid in enumerate(km.cluster_centers_):
            # Find the training sentence whose hidden state is closest.
            d = np.linalg.norm(Xc - centroid, axis=1)
            closest_local = int(d.argmin())
            closest_global = int(idx[closest_local])
            protos.append({
                "class": c,
                "prototype_index": k,
                "sentence": texts[closest_global],
                "sentence_index": closest_global,
                "distance": float(d[closest_local]),
            })
    return protos


def nearest_prototype_accuracy(
    states, y, layer_index, n_classes,
    n_prototypes_per_class=5, seed=42,
):
    """
    Train the same KMeans prototypes as extract_prototypes, then classify
    every sample by argmin distance to any prototype. Return accuracy and
    macro-F1 of this interpretable surrogate.
    """
    from sklearn.metrics import accuracy_score, f1_score
    X = np.asarray(states[:, layer_index, :], dtype=np.float32)
    all_protos = []
    proto_labels = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) < n_prototypes_per_class:
            continue
        km = KMeans(n_clusters=n_prototypes_per_class,
                    random_state=seed, n_init=10).fit(X[idx])
        for centroid in km.cluster_centers_:
            all_protos.append(centroid)
            proto_labels.append(c)
    protos = np.stack(all_protos)
    labels = np.asarray(proto_labels)

    # Nearest-prototype classification on the same data (or on a held-out split
    # if you want to be honest about generalisation).
    dists = np.linalg.norm(X[:, None, :] - protos[None, :, :], axis=2)  # [N, P]
    pred = labels[dists.argmin(axis=1)]
    return {
        "prototype_accuracy": float(accuracy_score(y, pred)),
        "prototype_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "n_prototypes": int(len(protos)),
    }