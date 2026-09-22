"""
_shared.py — path constants, atomic I/O, and loaders used by every
interpretability technique in this project.

Canonical layout
----------------
    HIDDEN_STATES_ROOT = /Volumes/Amirali/hidden_states
        Extraction.py output.  Reads only.

    INTEREX_ROOT       = /Volumes/Amirali/interEx
        Probe.py output AND every wired.py analysis artefact.
"""
from __future__ import annotations

from pathlib import Path
import json, os, time, hashlib
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Canonical roots
# ─────────────────────────────────────────────────────────────────────────────

AMIRALI_MOUNT       = Path("/Volumes/Amirali")
HIDDEN_STATES_ROOT  = AMIRALI_MOUNT / "hidden_states"
INTEREX_ROOT        = AMIRALI_MOUNT / "interEx"
MODELS_ROOT         = AMIRALI_MOUNT / "models"
DATASETS_ROOT       = AMIRALI_MOUNT / "datasets"

# `PROBE_ROOT` is kept as an alias so existing imports in Probe.py keep
# working without a rename.  It points at the interEx tree.
PROBE_ROOT = INTEREX_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Slug and path helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_slug(name: str) -> str:
    """'Qwen/Qwen2-0.5B' → 'Qwen2-0.5B'."""
    return name.split("/")[-1]


def artifact_dir(model_slug_: str, dataset: str) -> Path:
    """Where Extraction.py wrote the frozen hidden states."""
    return HIDDEN_STATES_ROOT / model_slug_ / dataset


def interex_dir(model_slug_: str, dataset: str) -> Path:
    """Top-level interEx dir for one (model, dataset)."""
    d = INTEREX_ROOT / model_slug_ / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def analysis_dir_for_extraction(model_slug_: str, dataset: str) -> Path:
    """Extraction-level techniques (geometry, CKA-within, attention)
    write here.  Reads hidden_states.npy from HIDDEN_STATES_ROOT.
    """
    d = interex_dir(model_slug_, dataset) / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def analysis_dir_for_probe(run_key_dir: Path) -> Path:
    """Probe-level techniques (TCAV, prototypes, IG) write here.
    Reads the fitted probe from the run_key directory.
    """
    d = Path(run_key_dir) / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Atomic writers
#
# The two writers below are the *only* sanctioned way to persist JSON or
# NPZ from any analysis module.  Both write to a sibling .tmp file and
# then os.replace() into position, so a crash mid-write cannot leave a
# half-written file.
# ─────────────────────────────────────────────────────────────────────────────

def atomic_json(path: Path, payload: dict) -> None:
    """Write JSON atomically.

    Uses a raw file handle because json.dump writes to whatever stream it
    is given.  The .tmp suffix is only for the temporary file on disk; it
    does not affect the format of the payload.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    """Write an .npz archive atomically.

    CRITICAL: np.savez appends '.npz' to any filename that does not
    already end in '.npz'.  If we hand it 'foo.npz.tmp', it silently
    writes 'foo.npz.tmp.npz', leaving the expected tmp path empty and
    crashing the subsequent os.replace.  We therefore pass a *file
    handle* rather than a filename; the append heuristic does not fire
    when savez writes to a stream.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_hidden_states(model_slug_: str, dataset: str) -> np.ndarray:
    """Memory-mapped view of [N, L, D].  Never materialises the full tensor."""
    p = artifact_dir(model_slug_, dataset) / "hidden_states.npy"
    if not p.is_file():
        raise FileNotFoundError(f"hidden_states.npy missing: {p}")
    return np.load(p, mmap_mode="r")


def load_labels(model_slug_: str, dataset: str) -> np.ndarray:
    """Load labels.npy, or fail loudly if extraction did not write it."""
    p = artifact_dir(model_slug_, dataset) / "labels.npy"
    if not p.is_file():
        raise FileNotFoundError(
            f"labels.npy missing for {model_slug_}/{dataset}.\n"
            f"Expected: {p}\n"
            f"Re-run extraction for this pair, or exclude it from the analysis."
        )
    return np.load(p, allow_pickle=True)


def load_label_codes(model_slug_: str, dataset: str) -> np.ndarray:
    p = artifact_dir(model_slug_, dataset) / "label_codes.npy"
    if not p.is_file():
        raise FileNotFoundError(f"label_codes.npy missing: {p}")
    return np.load(p, allow_pickle=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────

def stable_hash(obj, length: int = 16) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


# ─────────────────────────────────────────────────────────────────────────────
# Probe artefact loader (used by TCAV, prototypes, IG)
# ─────────────────────────────────────────────────────────────────────────────

def load_probe_for_attribution(
    run_key_dir: Path,
    probe_name: str,
    layer_index: int,
    repeat: int = 0,
):
    """Return (callable_probe, probe_kind, scaler).

    The callable accepts an [N, D] numpy array and returns an [N, C]
    probability matrix.  Handles both sklearn Pipelines (logistic) and
    PyTorch state_dicts (MLP).
    """
    import joblib

    d = Path(run_key_dir) / "models" / probe_name / f"layer_{layer_index}" / f"repeat_{repeat}"
    if not d.is_dir():
        raise FileNotFoundError(f"Probe artefact directory missing: {d}")

    scaler = None
    sp = d / "scaler.joblib"
    if sp.exists():
        scaler = joblib.load(sp)

    joblib_path = d / "probe.joblib"
    if joblib_path.exists():
        pipe = joblib.load(joblib_path)

        def call(X):
            X = np.asarray(X, dtype=np.float32)
            if scaler is not None:
                X = scaler.transform(X)
            return pipe.predict_proba(X)

        return call, "logistic", scaler

    pt_path = d / "probe_state_dict.pt"
    if pt_path.exists():
        import torch
        import Probe as P

        m = json.loads((d / "metrics.json").read_text())
        dims = m["record"]["resolved_hidden_dims"]
        input_dim = m["record"]["input_dim"]
        n_classes = m["record"]["class_count"]

        net = P.TorchMLP(input_dim, n_classes, dims, 0.0)
        net.load_state_dict(torch.load(pt_path, map_location="cpu"))
        net.eval()

        def call(X):
            X = np.asarray(X, dtype=np.float32)
            if scaler is not None:
                X = scaler.transform(X)
            with torch.no_grad():
                logits = net(torch.from_numpy(X))
                return torch.softmax(logits, dim=1).numpy()

        return call, "mlp", scaler

    raise FileNotFoundError(f"No probe artefact at {d}")


# ─────────────────────────────────────────────────────────────────────────────
# Chunked reader
# ─────────────────────────────────────────────────────────────────────────────

def read_block(states, sample_idx, chunk_rows: int = 1500):
    """Yield (chunk_indices, X_chunk) in *sorted* index order.

    WARNING
    -------
    Rows within each chunk are returned in ascending index order (to keep
    disk reads sequential).  If your downstream code assumes X[i] pairs
    with sample_idx[i], that assumption is WRONG; it pairs with the
    sorted index.  Either sort your labels the same way, or apply the
    inverse permutation yourself.
    """
    sample_idx = np.asarray(sample_idx)
    order = np.argsort(sample_idx)
    sample_idx_sorted = sample_idx[order]

    for start in range(0, len(sample_idx_sorted), chunk_rows):
        idx = sample_idx_sorted[start:start + chunk_rows]
        X = np.asarray(states[idx], dtype=np.float32)
        yield idx, X