"""
Explainability & Interpretability for Emotion Probing: A Complete, Code-Anchored Guide

Part 0 — Why this document exists, and how to read it

Your project asks a specific scientific question: does a frozen transformer, trained on general text, 
encode emotion in its hidden states — and if so, at which layer? Classifier probing answers the decodability half of that question. 
It does not answer:

    What shape does the emotional information have in each layer? (geometry)
    Is it the same shape across models? (representational similarity)
    Which input tokens drive the probe's decision? (attribution)
    Is the probe reading emotion, or a linguistic shortcut such as "the word happy appears"? (concept testing)
    Can we causally intervene on the emotion representation, or does it only correlate? (activation steering)
    Every technique in this document targets one of those five gaps. Each is delivered as a standalone Python module that 
    follows the same conventions as your Extraction.py and Probe.py: root constants at the top, build_*_paths() helpers, 
    atomic JSON writes, np.savez for arrays, per-stage output directories, and the same verbosity flags. Nothing writes into a stage it does not own.
"""
from pathlib import Path
import json, os, time, hashlib
import numpy as np

AMIRALI_MOUNT       = Path("/Volumes/Amirali")
HIDDEN_STATES_ROOT  = AMIRALI_MOUNT / "hidden_states"
PROBE_ROOT          = AMIRALI_MOUNT / "probe"

def model_slug(name: str) -> str:
    return name.split("/")[-1]

def artifact_dir(model_slug_: str, dataset: str) -> Path:
    return HIDDEN_STATES_ROOT / model_slug_ / dataset

def analysis_dir_for_extraction(model_slug_: str, dataset: str) -> Path:
    """Techniques that read hidden_states.npy write here."""
    d = artifact_dir(model_slug_, dataset) / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d

def analysis_dir_for_probe(run_key_dir: Path) -> Path:
    """Techniques that read a fitted probe write here."""
    d = run_key_dir / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)

def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)

def load_hidden_states(model_slug_: str, dataset: str) -> np.ndarray:
    """Memory-mapped — never materialise [N, L, D] in RAM."""
    return np.load(artifact_dir(model_slug_, dataset) / "hidden_states.npy",
                   mmap_mode="r")

def load_labels(model_slug_: str, dataset: str) -> np.ndarray:
    p = artifact_dir(model_slug_, dataset) / "labels.npy"
    return np.load(p, allow_pickle=True)

def stable_hash(obj, length=16):
    import json, hashlib
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]




def load_probe_for_attribution(run_key_dir: Path, probe_name: str, layer_index: int, repeat: int = 0):
    """
    Return (callable_probe, probe_kind, scaler) where callable_probe
    accepts a numpy [N, D] array and returns [N, C] probabilities.
    Handles both logistic Pipelines and MLP state_dicts.
    """
    d = run_key_dir / "models" / probe_name / f"layer_{layer_index}" / f"repeat_{repeat}"
    scaler = None
    sp = d / "scaler.joblib"
    if sp.exists():
        scaler = joblib.load(sp)

    joblib_path = d / "probe.joblib"
    if joblib_path.exists():
        pipe = joblib.load(joblib_path)
        # sklearn Pipeline: predict_proba needs the scaler already in it,
        # but if scaler is separate, wrap:
        if scaler is not None and not hasattr(pipe, "named_steps"):
            def call(X):
                return pipe.predict_proba(scaler.transform(X))
        else:
            def call(X):
                return pipe.predict_proba(X)
        return call, "logistic", scaler

    pt_path = d / "probe_state_dict.pt"
    if pt_path.exists():
        import torch, Probe as P
        meta = json.loads((run_key_dir / "complete_run_metadata.json").read_text())
        # Find hidden dims from the saved metrics (recorded by Probe.py)
        m = json.loads((d / "metrics.json").read_text())
        dims = m["record"]["resolved_hidden_dims"]
        input_dim = m["record"]["input_dim"]
        n_classes = m["record"]["class_count"]
        net = P.TorchMLP(input_dim, n_classes, dims, 0.0)
        net.load_state_dict(torch.load(pt_path, map_location="cpu"))
        net.eval()
        def call(X):
            with torch.no_grad():
                x = torch.from_numpy(X.astype("float32"))
                if scaler is not None:
                    x = torch.from_numpy(scaler.transform(X).astype("float32"))
                return torch.softmax(net(x), dim=1).numpy()
        return call, "mlp", scaler

    raise FileNotFoundError(f"No probe artifact at {d}")


def read_block(states, sample_idx, chunk_rows=1500):
    """
    Yield (chunk_idx, X_chunk) where X_chunk is [len(chunk_idx), L, D]
    and the rows were read in sorted order to make disk reads sequential.

    Usage:
        for chunk_idx, X_chunk in read_block(states, sample_idx):
            for layer in range(X_chunk.shape[1]):
                X = X_chunk[:, layer, :]
                ...
    """
    sample_idx = np.asarray(sample_idx)
    order = np.argsort(sample_idx)
    sample_idx_sorted = sample_idx[order]

    for start in range(0, len(sample_idx_sorted), chunk_rows):
        idx = sample_idx_sorted[start:start + chunk_rows]
        X = np.asarray(states[idx], dtype=np.float32)  # [chunk, L, D]
        yield idx, X