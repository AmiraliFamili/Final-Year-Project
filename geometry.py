from __future__ import annotations

"""
What it does, in plain language. 
For each layer L, treat the hidden state of every sample as a point in a 768-dimensional (or 1024-, or 3584-dimensional) space. 
Group the points by their emotion label. Compute three things: (1) how far apart the group centres are, 
(2) how tightly each group clusters, and (3) the ratio of the two. That ratio is the layer's emotional separation score. 
A layer that has learned emotion will have a high separation score; a layer that has not will have a score near zero.

Why this is different from probing. A probe is a classifier with a learning algorithm and a bias term. If it achieves 80% accuracy on layer 9, 
you don't know whether that reflects a rich emotional representation or a lucky hyperplane. The separation score is probe-independent. 
It says: "the class centroids in layer 9 are 3.2× further apart than the average spread within a class." 
That statement is true regardless of any downstream classifier.

Why it matters for your argument. Your thesis claim is "emotion lives in the middle-to-late layers." 
Probing gives you one piece of evidence. Geometry gives you a second, orthogonal piece. 
If both point to the same layer, the finding is robust. If they disagree, 
that disagreement is itself a result worth reporting — it would mean the class structure is present but not linearly readable, 
which points toward needing non-linear probes.

"""



"""
geometry.py — Probe-independent analysis of the emotional geometry of each layer.

Thesis question answered:
    "For each layer L, how separable are the emotional classes in that layer's
     hidden states, and how does that separability evolve with depth?"

Method:
    For each layer:
        1. Compute the centroid of every class (mean vector over samples of that class).
        2. Compute the mean pairwise distance between class centroids.
        3. Compute the mean within-class dispersion (mean distance from a sample
           to its own class centroid).
        4. Compute the separation ratio = (2) / (3). Higher = more separable.

Reads:  hidden_states/<slug>/<dataset>/{hidden_states.npy, labels.npy}
Writes: hidden_states/<slug>/<dataset>/analysis/geometry.parquet
        hidden_states/<slug>/<dataset>/analysis/geometry.json
"""


from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from _shared import (
    artifact_dir, analysis_dir_for_extraction,
    atomic_json, load_hidden_states, load_labels, stable_hash,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Every knob that changes the output is a module-level constant so the
# resulting parquet file can be traced back to the exact settings used.
# This mirrors the pattern in Extraction.py where HyperParameters is the
# single source of truth for every value that affects the tensor content.

SAMPLE_CAP        = 5000      # cap N to bound sklearn cost (silhouette is O(N²))
SILHOUETTE_CAP    = 3000      # silhouette is the slowest call — cap separately
SEED              = 42        # deterministic subsampling
FLOAT_PRECISION   = "float32" # we never need float64 for geometry


# ─────────────────────────────────────────────────────────────────────────────
# Core primitives
# ─────────────────────────────────────────────────────────────────────────────

def class_centroids(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Compute the centroid of each class.

    Parameters
    ----------
    X : [N, D] float32 — hidden states for ONE layer, already subset to the
        rows in `sample_idx`. We do not pass the full [N, L, D] tensor here
        because that would force a materialisation of the whole thing.
    y : [N] int64 — integer-encoded labels in the same row order as X.
    n_classes : int — total number of classes in the dataset. We use this
        rather than y.max()+1 so that a class entirely absent from the
        subsample still gets a slot (and its centroid is NaN).

    Returns
    -------
    centroids : [C, D] float32 — one row per class. Rows for absent classes
        are NaN, which downstream code is expected to ignore. The alternative
        (dropping absent classes) would silently change the shape of the
        distance matrix and break the correspondence with class_names.

    Why this matters for emotion probing:
        The centroid is the "prototype" of an emotion in this layer's
        representation space. If the probe's decision boundary passes near
        the centroid of class `joy`, then the probe is essentially asking
        "is this sample close to the joy prototype?" — which is the
        interpretable answer we want. If the probe's decision boundary is
        far from all centroids, the probe is doing something more complex,
        and we need to look at per-sample attributions (see ig.py).
    """
    D = X.shape[1]
    centroids = np.full((n_classes, D), np.nan, dtype=np.float32)
    for c in range(n_classes):
        mask = (y == c)
        if mask.any():
            # mean over the rows of X that belong to class c
            centroids[c] = X[mask].mean(axis=0)
    return centroids


def pairwise_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Euclidean distance between every pair of class centroids.

    Parameters
    ----------
    centroids : [C, D] — output of class_centroids(). May contain NaN rows.

    Returns
    -------
    distances : [C', C'] — a square matrix where C' is the number of
        non-NaN centroids. Entry (i, j) is ||centroid_i - centroid_j||_2.

    Implementation note:
        We use broadcasting instead of a Python loop:
            (C, 1, D) - (1, C, D)  →  [C, C, D]
        then take the L2 norm along the last axis.
        For C = 28 and D = 768, this allocates a 28×28×768 float32 tensor
        = 2.4 MB. That is fine. For C = 28 and D = 3584 (a larger model),
        it is 11 MB. Also fine. We never do this on the full [N, L, D].
    """
    valid = ~np.isnan(centroids).any(axis=1)
    C = centroids[valid]
    diff = C[:, None, :] - C[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def within_class_dispersion(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    For each class, the mean distance from a sample to its own class centroid.

    Parameters
    ----------
    X : [N, D] float32 — one layer's hidden states, subsampled.
    y : [N] int64 — labels.
    n_classes : int — total classes.

    Returns
    -------
    dispersion : [C] float32 — one value per class. NaN for absent classes
        and for classes with only one sample (dispersion is undefined for
        a singleton — the sample IS the centroid, distance zero, but the
        value would be misleadingly perfect).

    Why this matters:
        A class with high dispersion is "spread out" in the representation.
        An emotion like `neutral` in GoEmotions tends to be highly dispersed
        because it is the catch-all bucket — many different meanings get
        labelled neutral. A class with low dispersion is "tight" — the model
        represents it as a well-localised region. Comparing dispersion across
        emotions tells you which emotions the model has learned as distinct
        concepts and which it has learned as residual categories.
    """
    disp = np.full(n_classes, np.nan, dtype=np.float32)
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() < 2:          # singleton classes have no dispersion
            continue
        sub = X[mask]
        ctr = sub.mean(axis=0)
        disp[c] = float(np.linalg.norm(sub - ctr, axis=1).mean())
    return disp


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer orchestration
# ─────────────────────────────────────────────────────────────────────────────

# geometry.py — per_layer_geometry (replaces the existing function)

def per_layer_geometry(
    states: np.ndarray,              # memmap [N, L, D], C-order
    y: np.ndarray,                   # [N] int64
    n_classes: int,
    *,
    sample_idx: np.ndarray | None = None,
    silhouette_cap: int = SILHOUETTE_CAP,
    seed: int = SEED,
    memory_budget_bytes: int = 500_000_000,   # 500 MB ceiling for X_all
) -> pd.DataFrame:
    """One row per layer. Columns describe the emotional geometry.

    Performance note
    ----------------
    `states` is a C-order memmap [N, L, D].  Naive per-layer indexing
    (`states[sample_idx, l, :]`) forces 5000 seek-and-read-4KB operations
    per layer, which on an external HDD is catastrophic.  We therefore
    materialise every sampled row ONCE with a single contiguous read
    (`states[sample_idx]`, which walks the file sequentially), then slice
    that array in memory per layer.  For Qwen3-0.6B-Base on 5000 samples
    this is ~590 MB; if that exceeds the memory budget, we down-sample.
    """
    if sample_idx is None:
        sample_idx = np.arange(len(y))
    sample_idx = np.sort(np.asarray(sample_idx))

    n_layers    = states.shape[1]
    hidden_size = states.shape[2]

    # ── Bound the memory footprint. ──
    bytes_per_sample = n_layers * hidden_size * 4   # float32
    max_samples_by_memory = max(500, memory_budget_bytes // max(1, bytes_per_sample))

    if len(sample_idx) > max_samples_by_memory:
        print(f"[geom] reducing sample_idx {len(sample_idx)} → {max_samples_by_memory} "
              f"to keep X_all under {memory_budget_bytes / 1e6:.0f} MB")
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(sample_idx, max_samples_by_memory, replace=False))

    print(f"[geom] materialising {len(sample_idx)} samples × "
          f"{n_layers} layers × {hidden_size} dims "
          f"(≈ {len(sample_idx) * bytes_per_sample / 1e6:.0f} MB) …")
    t0 = time.perf_counter()

    # ── THE FIX: one contiguous read for all sampled rows. ──
    # states[sample_idx] with an array index walks the file in row order.
    # All 29 layers of each sample are read in a single 118 KB block, and
    # consecutive sample indices are adjacent, so this is effectively a
    # sequential read of the file's tail region that we care about.
    X_all = np.asarray(states[sample_idx], dtype=np.float32)   # [M, L, D]

    elapsed = time.perf_counter() - t0
    print(f"[geom] read complete in {elapsed:.2f}s "
          f"({len(sample_idx) * bytes_per_sample / max(elapsed, 1e-3) / 1e6:.1f} MB/s)")

    ys = y[sample_idx]

    rows: list[dict] = []
    for layer in range(n_layers):
        X = X_all[:, layer, :]        # in-memory slice, no I/O

        ctr  = class_centroids(X, ys, n_classes)
        dmat = pairwise_centroid_distances(ctr)
        disp = within_class_dispersion(X, ys, n_classes)

        sil = float("nan")
        if len(sample_idx) > silhouette_cap:
            rng     = np.random.default_rng(seed + layer)
            sub_idx = rng.choice(len(sample_idx), silhouette_cap, replace=False)
            Xs, ys_sil = X[sub_idx], ys[sub_idx]
        else:
            Xs, ys_sil = X, ys

        if len(np.unique(ys_sil)) > 1:
            try:
                sil = float(silhouette_score(Xs, ys_sil))
            except Exception:
                pass

        mean_sep = float(np.nanmean(dmat)) if dmat.size else float("nan")
        min_sep  = float(np.nanmin(dmat))  if dmat.size else float("nan")
        max_sep  = float(np.nanmax(dmat))  if dmat.size else float("nan")
        mean_dis = float(np.nanmean(disp))
        ratio    = mean_sep / mean_dis if mean_dis > 1e-9 else float("nan")

        rows.append({
            "layer_index":            layer,
            "relative_depth":         layer / max(1, n_layers - 1),
            "mean_pairwise_centroid": mean_sep,
            "min_pairwise_centroid":  min_sep,
            "max_pairwise_centroid":  max_sep,
            "mean_within_dispersion": mean_dis,
            "separation_ratio":       ratio,
            "silhouette":             sil,
        })

    # Free the big buffer before the caller continues.
    del X_all

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_geometry(
    model_slug_: str,
    dataset: str,
    *,
    sample_cap: int = SAMPLE_CAP,
    seed: int = SEED,
) -> Path:
    """
    Top-level function. Called once per (model, dataset) pair from your
    orchestrating notebook. Returns the path to the written parquet file.

    This function does NOT require the model to be loaded. It reads only
    hidden_states.npy and labels.npy. That means you can run it for every
    (model, dataset) pair that already has hidden states, even while
    extraction is still running for others.
    """
    t0 = time.perf_counter()

    states = load_hidden_states(model_slug_, dataset)
    labels = load_labels(model_slug_, dataset)

    if labels.dtype == object:
        # Multi-label: fall back to the same one-hot expansion Probe.py uses.
        # We import from Probe.py to keep the two modules in sync.
        import Probe as P
        codes = np.load(artifact_dir(model_slug_, dataset) / "label_codes.npy",
                        allow_pickle=True)
        # Build a dense one-hot for geometry purposes.
        y = np.zeros((len(labels), len(codes)), dtype=np.int64)
        for i, lab in enumerate(labels):
            if isinstance(lab, list):
                for v in lab:
                    y[i, int(v)] = 1
        n_classes = len(codes)
        # Multi-label geometry is handled separately (see geometry_multilabel.py).
        # For now, fall back to the dominant-label interpretation, which is
        # what the single-label path expects.
        y = y.argmax(axis=1)
        n_classes = len(codes)
    else:
        y = labels.astype(np.int64)
        n_classes = int(y.max()) + 1

    # Deterministic subsample. See SAMPLE_CAP docstring for the rationale.
    if len(y) > sample_cap:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(len(y), sample_cap, replace=False))
    else:
        sample_idx = np.arange(len(y))

    df = per_layer_geometry(states, y, n_classes,
                            sample_idx=sample_idx, seed=seed)

    # Provenance columns so the parquet is self-describing.
    df.insert(0, "model_slug", model_slug_)
    df.insert(1, "dataset", dataset)
    df.insert(2, "n_samples_used", len(sample_idx))
    df.insert(3, "n_samples_total", len(y))

    out_dir = analysis_dir_for_extraction(model_slug_, dataset)
    parquet_path = out_dir / "geometry.parquet"
    df.to_parquet(parquet_path, index=False)

    atomic_json(out_dir / "geometry.json", {
        "technique": "layer_wise_class_geometry",
        "model_slug": model_slug_,
        "dataset": dataset,
        "sample_cap": sample_cap,
        "seed": seed,
        "n_samples_total": int(len(y)),
        "n_samples_used": int(len(sample_idx)),
        "n_layers": int(states.shape[1]),
        "hidden_size": int(states.shape[2]),
        "elapsed_seconds": time.perf_counter() - t0,
        "output": str(parquet_path),
        "config_hash": stable_hash({
            "sample_cap": sample_cap, "seed": seed,
            "n_classes": n_classes,
        }),
    })
    return parquet_path




def plot_geometry_curves(parquet_path: Path, out_png: Path):
    import matplotlib.pyplot as plt
    df = pd.read_parquet(parquet_path)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["layer_index"], df["separation_ratio"], marker="o", linewidth=2)
    ax.set_xlabel("Layer"); ax.set_ylabel("separation ratio")
    ax.set_title("Probe-independent class separation by layer")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)