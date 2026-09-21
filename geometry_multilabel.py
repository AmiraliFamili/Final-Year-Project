from __future__ import annotations
"""
GoEmotions-specific geometry

The problem this solves. GoEmotions is multi-label: a comment can be labelled joy and gratitude. 
Class centroid is the wrong primitive because a sample belongs to multiple classes at once. 
The right primitive is a per-label positive-vs-negative contrast: for each label j, compute 
the mean hidden state of samples that have label j and the mean hidden state of samples that do not. 
The difference is the label's "direction" in that layer.

Why this matters. This gives you a 28-column table (one column per emotion) × L rows (layers). 
You can then plot a heatmap and ask: does gratitude separate early and nervousness separate late? 
Does neutral fail to separate at any layer? These are concrete, named findings that belong in your Results chapter.
"""


"""
geometry_multilabel.py — Per-label geometry for multi-label datasets.

Thesis question answered:
    "In a multi-label setting, does each emotion get its own direction in
     the representation, and at which layer does each emotion become
     linearly separable from the absence of that emotion?"

Method:
    For each layer L and each label j:
        pos_centroid = mean(hidden[L] for samples with label j = 1)
        neg_centroid = mean(hidden[L] for samples with label j = 0)
        separation_j = ||pos_centroid - neg_centroid||_2 / mean_std(hidden[L])
    The normalisation by mean_std makes the values comparable across layers
    whose absolute scales differ.

Reads:  hidden_states/<slug>/<dataset>/hidden_states.npy
        hidden_states/<slug>/<dataset>/labels.npy   (object array of lists)
        hidden_states/<slug>/<dataset>/label_codes.npy
Writes: hidden_states/<slug>/<dataset>/analysis/geometry_multilabel.parquet
"""


from pathlib import Path
import time
import numpy as np
import pandas as pd

from _shared import (
    artifact_dir, analysis_dir_for_extraction,
    atomic_json, load_hidden_states, stable_hash,
)


SAMPLE_CAP = 5000
SEED       = 42


def multilabel_geometry(
    states: np.ndarray,           # [N, L, D]
    Y: np.ndarray,                # [N, C] one-hot int64
    class_names: list[str],
    *,
    sample_cap: int = SAMPLE_CAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    One row per layer. One column per label, named sep::<label>.
    The DataFrame is therefore [L, C+1] with layer_index as the first column.
    """
    if states.shape[0] > sample_cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(states.shape[0], sample_cap, replace=False))
    else:
        idx = np.arange(states.shape[0])

    rows = []
    for layer in range(states.shape[1]):
        X = np.asarray(states[idx, layer, :], dtype=np.float32)
        # scale = mean standard deviation across dimensions, used to
        # normalise the centroid distance so that layers with different
        # activation magnitudes are comparable.
        scale = float(X.std(axis=0).mean())
        row = {"layer_index": layer}
        for j, name in enumerate(class_names):
            pos = Y[idx, j] == 1
            if pos.sum() < 5 or (~pos).sum() < 5:
                # Too few positives or too few negatives to estimate a
                # stable contrast. Record NaN so the heatmap shows a gap.
                row[f"sep::{name}"] = float("nan")
                continue
            c_pos = X[pos].mean(axis=0)
            c_neg = X[~pos].mean(axis=0)
            sep = float(np.linalg.norm(c_pos - c_neg) / max(scale, 1e-9))
            row[f"sep::{name}"] = sep
        rows.append(row)
    return pd.DataFrame(rows)


def run_geometry_multilabel(model_slug_: str, dataset: str,
                            sample_cap: int = SAMPLE_CAP) -> Path:
    """Top-level entry point. Only meaningful for multi-label datasets."""
    t0 = time.perf_counter()
    states = load_hidden_states(model_slug_, dataset)
    labels = np.load(artifact_dir(model_slug_, dataset) / "labels.npy",
                     allow_pickle=True)
    codes  = np.load(artifact_dir(model_slug_, dataset) / "label_codes.npy",
                     allow_pickle=True)

    # Build the [N, C] one-hot. labels is an object array of lists.
    C = len(codes)
    Y = np.zeros((len(labels), C), dtype=np.int64)
    for i, lab in enumerate(labels):
        if isinstance(lab, list):
            for v in lab:
                Y[i, int(v)] = 1

    df = multilabel_geometry(states, Y, list(codes), sample_cap=sample_cap)
    df.insert(0, "model_slug", model_slug_)
    df.insert(1, "dataset", dataset)

    out_dir = analysis_dir_for_extraction(model_slug_, dataset)
    out_path = out_dir / "geometry_multilabel.parquet"
    df.to_parquet(out_path, index=False)

    atomic_json(out_dir / "geometry_multilabel.json", {
        "technique": "multilabel_per_label_geometry",
        "model_slug": model_slug_,
        "dataset": dataset,
        "n_labels": C,
        "class_names": list(codes),
        "n_layers": int(states.shape[1]),
        "elapsed_seconds": time.perf_counter() - t0,
        "output": str(out_path),
    })
    return out_path