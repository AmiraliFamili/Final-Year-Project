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

# Extraction run metadata (manifest, ledger, environment, results).
# Historically lived in Probing-Emotions/; now it lives beside the tensors
# it describes. Underscore prefix keeps it out of the model-scan glob.
EXTRACTION_META_ROOT = HIDDEN_STATES_ROOT / "_meta"

# HuggingFace cache. Top-level so it survives any extraction-root rename.
HF_CACHE_ROOT    = AMIRALI_MOUNT / ".hf_cache"
HF_HUB_CACHE     = HF_CACHE_ROOT / "hub"
HF_XET_CACHE     = HF_CACHE_ROOT / "xet"
HF_ASSETS_CACHE  = HF_CACHE_ROOT / "assets"




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



        
def _fsync_dir(p: Path) -> None:
    """Make the directory entry durable. Must be called AFTER os.replace."""
    fd = os.open(str(p), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
        
        
        
        
        

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
    _fsync_dir(path.parent)


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
    _fsync_dir(path.parent)


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
        
        



# ─────────────────────────────────────────────────────────────────────────────
# Dataset schema & contract resolution  (SINGLE SOURCE OF TRUTH)
#
# Three tiers of intervention. A dataset that appears in NO dict still works:
# contract_dict_for returns target_type="custom", class_order=None, and every
# other field comes from the auto-generated schema sidecar.
#
#   Tier 1 — target_type override
#       KNOWN_TARGET_TYPES. Use only when the dataset needs a specialist
#       adapter (goemotions, isear, dimensional) rather than "custom".
#
#   Tier 2 — canonical vocabulary
#       KNOWN_CLASS_ORDERS (categorical) or KNOWN_DIMENSION_NAMES
#       (dimensional). Use when the integer labels have a fixed meaning
#       that matters for cross-model comparison.
#
#   Tier 3 — auto-detector escape hatch
#       SCHEMA_OVERRIDES. Use only when master_dataset.SchemaDetector picks
#       the wrong column or task_type and you cannot fix the source CSV.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any


# ── Tier 1: target_type overrides ──────────────────────────────────────────
KNOWN_TARGET_TYPES: dict[str, str] = {
    "goemo":    "goemotions",     # 0..27, multi-label
    "isear":    "isear",          # 1..7,  single-label, 1-based IDs
    "emobank":  "dimensional",    # VAD regression
}

# ── Tier 2a: categorical vocabularies ─────────────────────────────────────
# Positional: index i names class id i. Length must equal the class count.
# Numeric labels must be 0-based; the adapter raises if they aren't.
KNOWN_CLASS_ORDERS: dict[str, tuple[str, ...]] = {
    "goemo": (
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral",
    ),
    "isear": ("joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"),
    "emotion": (
        "sadness", "joy", "love", "anger", "fear", "surprise",
    ),
    "tweet_eval_emotion": ("anger", "joy", "optimism", "sadness"),
    "sst2": ("negative", "positive"),
    "amazon_polarity": ("negative", "positive"),
}

# ── Tier 2b: dimensional vocabularies ─────────────────────────────────────
KNOWN_DIMENSION_NAMES: dict[str, tuple[str, ...]] = {
    "emobank": ("valence", "arousal", "dominance"),
}

# ── Tier 3: auto-detector escape hatches ──────────────────────────────────
SCHEMA_OVERRIDES: dict[str, dict[str, object]] = {
    # "some_weird_dataset": {"text_column": "sentence", "label_column": "gold"},
}


# In-process cache. Invalidated by invalidate_schema_cache().
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def dataset_schema_path(dataset_name: str) -> Path:
    """Where master_dataset.py writes its schema sidecar."""
    return DATASETS_ROOT / dataset_name / "schema.json"


def _processed_csv_for(dataset_name: str) -> Path | None:
    base = DATASETS_ROOT / dataset_name / "processed"
    for cand in (f"{dataset_name}_clean.csv", f"{dataset_name}.csv"):
        p = base / cand
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def load_dataset_schema(
    dataset_name: str, *, use_cache: bool = True
) -> dict[str, Any]:
    """Read the schema sidecar written by master_dataset.py. {} if absent."""
    if use_cache and dataset_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[dataset_name]
    p = dataset_schema_path(dataset_name)
    if not p.is_file():
        return {}
    try:
        schema = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if use_cache:
        _SCHEMA_CACHE[dataset_name] = schema
    return schema


def ensure_dataset_schema(dataset_name: str) -> dict[str, Any]:
    """Return the schema for a dataset, computing it if the sidecar is missing.

    Reads the sidecar if present. Otherwise runs master_dataset.SchemaDetector
    on the processed CSV and writes the sidecar. Returns {} if there is no
    processed CSV to sample — the caller then falls back to contract defaults.
    """
    schema = load_dataset_schema(dataset_name)
    if (
        schema
        and schema.get("text_column")
        and schema.get("task_type") not in (None, "unknown")
    ):
        return schema

    csv_path = _processed_csv_for(dataset_name)
    if csv_path is None:
        return schema

    try:
        import master_dataset as md
    except Exception:
        return schema

    try:
        sample = md.DatasetLoader.read_rows_only(
            csv_path, 5_000, mode="random", seed=42
        )
        if sample.empty:
            return schema

        detector = md.SchemaDetector(md.Renderer(quiet=True, no_visuals=True))
        detection = detector.detect(sample)

        if detection.label_column:
            try:
                label_series = sample[detection.label_column].map(
                    md.LabelNormalizer.parse
                )
                task_type, class_count = md.infer_task_and_class_count(label_series)
                detection.task_type = task_type
                detection.class_count = class_count
            except Exception:
                pass

        md.write_schema_sidecar(dataset_name, detection)
        _SCHEMA_CACHE.pop(dataset_name, None)
        return load_dataset_schema(dataset_name, use_cache=False)
    except Exception as exc:
        print(f"[_shared] could not compute schema for {dataset_name!r}: {exc}")
        return schema


def contract_dict_for(dataset_name: str) -> dict[str, Any]:
    """Resolve a dataset name to a DatasetContract kwargs dict.

    Resolution order:
        1. schema sidecar (text_column, label_column, task_type)
        2. KNOWN_TARGET_TYPES (target_type override)
        3. KNOWN_CLASS_ORDERS / KNOWN_DIMENSION_NAMES (canonical vocabulary)
        4. SCHEMA_OVERRIDES (manual escape hatch, applied last)

    Every dataset — including ones never seen before — gets a valid contract.
    A dataset with no entry in any dict returns:
        target_type="custom", task_type from the sidecar, class_order=None.
    """
    schema = ensure_dataset_schema(dataset_name)

    # ── Tier 1: target_type ──
    target_type = KNOWN_TARGET_TYPES.get(dataset_name, "custom")

    # ── task_type: fixed by target_type when that type is specialised. ──
    FIXED_TASK_TYPES = {
        "goemotions":  "multi_label",
        "isear":       "single_label",
        "dimensional": "dimensional",
    }
    task_type = (
        FIXED_TASK_TYPES.get(target_type)
        or schema.get("task_type")
        or "auto"
    )
    if task_type == "unknown":
        task_type = "auto"

    # ── Tier 2: class_order / dimension names. ──
    if target_type == "dimensional":
        dims = KNOWN_DIMENSION_NAMES.get(dataset_name)
        class_order = list(dims) if dims else None
    else:
        names = KNOWN_CLASS_ORDERS.get(dataset_name)
        class_order = list(names) if names else None

    contract: dict[str, Any] = {
        "target_type":  target_type,
        # Enforced by PROCESSED_COLUMNS in Extraction.py. The schema sidecar
        # records the RAW column names (text, labels, ...) which do not exist
        # in the processed CSV. Ignore them here.
        "text_column":  "clean_text",
        "label_column": "label",
        "id_column":    "auto",
        "task_type":    task_type,
        "label_format": "auto",
        "class_order":  class_order,
        "single_label_policy": "first_label" if task_type == "multi_label" else None,
        "require_provenance":               True,
        "require_label_fingerprint":        False,
        "lenient_provenance":               False,
        "allow_missing_label_fingerprint":  True,
    }

    # ── Tier 3: escape hatch applied last. ──
    contract.update(SCHEMA_OVERRIDES.get(dataset_name, {}))
    return contract


def invalidate_schema_cache(dataset_name: str | None = None) -> None:
    """Drop the cache. Call after reprocessing a dataset."""
    if dataset_name is None:
        _SCHEMA_CACHE.clear()
    else:
        _SCHEMA_CACHE.pop(dataset_name, None)