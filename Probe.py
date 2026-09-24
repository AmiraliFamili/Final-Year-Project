"""
Unified Hidden‑State Probe v4.5 – Ultra‑resilient atomic progress tracking with split archive.

Changes from v4.4:
- Split indices are stored in progress file, enabling exact resume without recomputation.
- Early exit if all jobs already completed but finalization not done.
- Improved error logging summary.
- Removed deprecated _exact_split_controls.
- Added optional `skip_completed_repeats` optimization.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import importlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import platform
import sys

try:
    import psutil
except ImportError:
    psutil = None


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------

# =============================================================================
# MODULE CONSTANTS
# =============================================================================

# Reproducibility / configuration defaults
DEFAULT_SEED = 42
VERBOSE_DEFAULT = 1

# Script identity
SCRIPT_VERSION = "4.5"


# ── Paths: the single source of truth is _shared.py. ──
from _shared import (                      # noqa: E402
    AMIRALI_MOUNT,
    HIDDEN_STATES_ROOT,
    INTEREX_ROOT,
    PROBE_ROOT,
    MODELS_ROOT,
    DATASETS_ROOT,
    model_slug,
    artifact_dir as _shared_artifact_dir,
    interex_dir   as _shared_interex_dir,
)

EXTERNAL_ROOT_DEFAULT = HIDDEN_STATES_ROOT   # legacy alias
EXTERNAL_ROOT         = HIDDEN_STATES_ROOT   # legacy alias
PROCESSED_DATASETS_ROOT = DATASETS_ROOT  

def artifact_dir_for(model_name: str, dataset_name: str) -> Path:
    """Where Extraction.py wrote the frozen hidden states for this pair."""
    return _shared_artifact_dir(model_slug(model_name), dataset_name)


def probe_dir_for(model_name: str, dataset_name: str) -> Path:
    """Where Probe.py writes probe runs for this pair."""
    return _shared_interex_dir(model_slug(model_name), dataset_name)


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================

def get_environment_info() -> dict:
    info = {
        "timestamp": time.time(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "torch": torch.__version__,
            "transformers": (
                __import__("transformers").__version__
                if importlib.util.find_spec("transformers")
                else None
            ),
            "matplotlib": __import__("matplotlib").__version__,
            "seaborn": __import__("seaborn").__version__,
        },
        "device": {
            "chosen": choose_device(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": (
                torch.backends.mps.is_available()
                if hasattr(torch.backends, "mps")
                else False
            ),
        },
        "memory": {},
    }

    if psutil is not None:
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": vm.total / (1024 ** 3),
            "available_gb": vm.available / (1024 ** 3),
            "used_gb": vm.used / (1024 ** 3),
            "percent_used": vm.percent,
        }

    return info

def model_slug(model_name: str) -> str:
    """Mirror Extraction.py's slug: 'Qwen/Qwen2-0.5B' → 'Qwen2-0.5B'."""
    return model_name.split("/")[-1]


def artifact_dir_for(model_name: str, dataset_name: str) -> Path:
    """Where Extraction.py put the hidden states for this (model, dataset)."""
    return HIDDEN_STATES_ROOT / model_slug(model_name) / dataset_name


def probe_dir_for(model_name: str, dataset_name: str) -> Path:
    """Where Probe.py will write probe runs for this (model, dataset)."""
    return PROBE_ROOT / model_slug(model_name) / dataset_name

GOEMOTIONS_CLASSES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
ISEAR_CLASSES = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

COMMON_TEXT_COLUMNS = [
    "clean_text", "text", "response", "utterance", "sentence", "content",
    "comment", "prompt", "statement", "input", "document", "description",
]
COMMON_LABEL_COLUMNS = [
    "dominant_emotion", "emotion", "emotion_label", "label", "labels",
    "target", "category", "class", "y",
]
COMMON_ID_COLUMNS = {"id", "idx", "index", "user_id", "conv_id", "utterance_idx"}


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str, sort_keys=True)
    tmp.replace(path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def one_dim_strings(values: Sequence[Any]) -> list[str]:
    return ["" if x is None else str(x) for x in values]


def parse_layer_number(layer_name: str) -> int:
    m = re.fullmatch(r"layer_(\d+)", str(layer_name))
    if not m:
        raise ValueError(f"Invalid layer name: {layer_name!r}")
    return int(m.group(1))


def sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0:
        raise ValueError("max_n must be > 0")
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, max_n, replace=False))


def fingerprint_values(values: Sequence[Any], *, length: int = 20) -> str:
    vals = one_dim_strings(values)
    return stable_hash({
        "n": len(vals),
        "head": vals[:16],
        "tail": vals[-16:] if vals else [],
    }, length)


def sequence_hash(values: Sequence[Any], *, length: int = 20) -> str:
    return stable_hash(list(values), length)


def safe_relative_output(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to write outside artifact root. root={root}, candidate={candidate}"
        ) from exc
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DatasetContract:
    target_type: str = "auto"
    type: str = "python"
    module: str | None = None
    function: str | None = None
    path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    text_column: str | None = "auto"
    label_column: str | None = "auto"
    id_column: str | None = "auto"

    task_type: str = "auto"
    label_format: str = "auto"
    single_label_policy: str | None = None
    class_order: list[str] | None = None

    require_provenance: bool = False
    require_label_fingerprint: bool = False
    lenient_provenance: bool = False
    allow_missing_label_fingerprint: bool = True

@dataclass
class SplitConfig:
    train: float = 0.80
    validation: float = 0.10
    test: float = 0.10
    seed: int = DEFAULT_SEED
    stratify: bool = True

    def validate(self) -> None:
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        if min(self.train, self.validation, self.test) <= 0:
            raise ValueError("All split fractions must be > 0")


@dataclass
class ProbeSpec:
    name: str
    type: str = "logistic"             # logistic/mlp
    complexity: str = "linear"         # linear/1_hidden/2_hidden/3_hidden/custom
    standardize: bool = True

    C: float = 1.0
    max_iter: int = 2000

    hidden_dims: list[int | str] = field(default_factory=list)
    hidden_width_ratio: float = 0.5
    width_schedule: str = "halving"

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    batch_size: int = 256
    patience: int = 12
    dropout: float = 0.0

    selection_metric: str = "macro_f1"


@dataclass
class AnalysisConfig:
    dataset: DatasetContract
    probes: list[ProbeSpec]

    layers: list[int | str] | str = "all"
    split: SplitConfig = field(default_factory=SplitConfig)
    repeats: int = 3
    max_samples: int | None = None

    shuffled_label_control: bool = True
    shuffled_control_repeats: int = 3
    run_control_on_all_layers: bool = True

    pca_enabled: bool = True
    pca_samples: int = 3000
    silhouette_enabled: bool = True
    silhouette_samples: int = 3000

    enable_abstention: bool = True
    enable_per_class_metrics: bool = True
    enable_feature_statistics: bool = True

    score_weights: dict[str, float] = field(default_factory=lambda: {
        "macro_f1": 0.25,
        "balanced_accuracy": 0.15,
        "mcc": 0.15,
        "log_loss_score": 0.10,
        "selectivity": 0.20,
        "stability": 0.10,
        "geometry": 0.05,
    })

    complexity_penalty_scale: float = 0.02
    output_subdir: str = "analysis/probes"

    verbose: int = VERBOSE_DEFAULT

    def validate_verbose(self) -> None:
        if self.verbose not in {0, 1, 2, 3}:
            raise ValueError("verbose must be one of {0, 1, 2, 3}")


SUPPORTED_SELECTION_METRICS = {
    "macro_f1", "accuracy", "balanced_accuracy", "mcc", "weighted_f1"
}
SUPPORTED_COMPLEXITIES = {"linear", "1_hidden", "2_hidden", "3_hidden", "custom"}


def validate_probe_spec(spec: ProbeSpec, task_type: str) -> None:
    if spec.type not in {"logistic", "mlp"}:
        raise ValueError(f"Unsupported probe type: {spec.type}")
    if spec.complexity not in SUPPORTED_COMPLEXITIES:
        raise ValueError(f"Unsupported complexity: {spec.complexity}")
    if spec.max_iter < 1 or spec.epochs < 1 or spec.batch_size < 1:
        raise ValueError(f"Invalid optimisation settings for {spec.name}")
    if not 0 <= spec.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if spec.learning_rate <= 0 or spec.weight_decay < 0 or spec.C <= 0:
        raise ValueError(f"Invalid learning/regularisation value in {spec.name}")
    if spec.selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection_metric={spec.selection_metric!r}; "
            f"allowed={sorted(SUPPORTED_SELECTION_METRICS)}"
        )
    if spec.type == "logistic" and spec.complexity != "linear":
        raise ValueError(f"Logistic probe {spec.name} must use complexity='linear'")
    if spec.type == "mlp" and spec.complexity == "linear":
        raise ValueError(f"MLP probe {spec.name} must use 1_hidden/2_hidden/3_hidden/custom")
    if task_type not in {"single_label", "multi_label"}:
        raise ValueError(f"Invalid task type {task_type}")


def load_config(path: Path) -> AnalysisConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    ds = DatasetContract(**raw.get("dataset", {}))
    split = SplitConfig(**raw.get("split", {}))
    split.validate()
    probes = [ProbeSpec(**p) for p in raw.get("probes", [])]
    if not probes:
        raise ValueError("At least one probe is required")

    a = raw.get("analysis", {})
    cfg = AnalysisConfig(
        dataset=ds,
        probes=probes,
        layers=raw.get("layers", "all"),
        split=split,
        repeats=int(raw.get("repeats", 3)),
        max_samples=raw.get("max_samples", None),
        shuffled_label_control=bool(a.get("shuffled_label_control", True)),
        shuffled_control_repeats=int(a.get("shuffled_control_repeats", 3)),
        run_control_on_all_layers=bool(a.get("run_control_on_all_layers", True)),
        pca_enabled=bool(a.get("pca_enabled", True)),
        pca_samples=int(a.get("pca_samples", 3000)),
        silhouette_enabled=bool(a.get("silhouette_enabled", True)),
        silhouette_samples=int(a.get("silhouette_samples", 3000)),
        enable_abstention=bool(a.get("enable_abstention", True)),
        enable_per_class_metrics=bool(a.get("enable_per_class_metrics", True)),
        enable_feature_statistics=bool(a.get("enable_feature_statistics", True)),
        score_weights=dict(a.get("score_weights", AnalysisConfig.score_weights)),
        complexity_penalty_scale=float(a.get("complexity_penalty_scale", 0.02)),
        output_subdir=str(raw.get("output_subdir", "analysis/probes")),
        verbose=int(a.get("verbose", VERBOSE_DEFAULT)),
    )
    cfg.validate_verbose()
    if cfg.repeats < 1:
        raise ValueError("repeats must be >= 1")
    if cfg.shuffled_control_repeats < 1:
        raise ValueError("shuffled_control_repeats must be >= 1")
    if cfg.max_samples is not None and cfg.max_samples < 30:
        raise ValueError("max_samples must be >= 30 or null")
    if cfg.pca_samples < 10 or cfg.silhouette_samples < 10:
        raise ValueError("Geometry sample limits must be >= 10")
    if cfg.complexity_penalty_scale < 0:
        raise ValueError("complexity_penalty_scale must be >= 0")
    return cfg


def load_complete_metadata(run_dir_or_file: Path) -> dict:
    path = Path(run_dir_or_file)
    if path.is_dir():
        path = path / "complete_run_metadata.json"
    elif path.name != "complete_run_metadata.json":
        if not path.exists():
            path = path / "complete_run_metadata.json"
    with open(path, "r") as f:
        return json.load(f)


def write_example_config(path: Path) -> None:
    example = {
        "dataset": {
            "target_type": "auto",
            "type": "python",
            "module": "Get_Go_Emo",
            "function": "get_go",
            "kwargs": {},
            "text_column": "auto",
            "label_column": "auto",
            "id_column": "auto",
            "task_type": "auto",
            "label_format": "auto",
            "single_label_policy": None,
            "class_order": None,
            "require_provenance": False,
            "require_label_fingerprint": False,
        },
        "probes": [
            {
                "name": "linear_logistic",
                "type": "logistic",
                "complexity": "linear",
                "standardize": True,
                "C": 1.0,
                "max_iter": 3000,
                "selection_metric": "macro_f1",
            },
            {
                "name": "mlp_1_hidden",
                "type": "mlp",
                "complexity": "1_hidden",
                "hidden_dims": ["0.5d"],
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "epochs": 80,
                "batch_size": 256,
                "patience": 12,
                "dropout": 0.0,
                "selection_metric": "macro_f1",
            },
            {
                "name": "mlp_2_hidden",
                "type": "mlp",
                "complexity": "2_hidden",
                "hidden_dims": ["0.5d", "0.25d"],
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "epochs": 80,
                "batch_size": 256,
                "patience": 12,
                "dropout": 0.0,
                "selection_metric": "macro_f1",
            },
            {
                "name": "mlp_3_hidden",
                "type": "mlp",
                "complexity": "3_hidden",
                "hidden_dims": ["0.5d", "0.25d", "0.125d"],
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "epochs": 80,
                "batch_size": 256,
                "patience": 12,
                "dropout": 0.0,
                "selection_metric": "macro_f1",
            },
        ],
        "layers": "all",
        "split": {"train": 0.80, "validation": 0.10, "test": 0.10, "seed": 42, "stratify": True},
        "repeats": 3,
        "max_samples": 5000,
        "analysis": {
            "shuffled_label_control": True,
            "shuffled_control_repeats": 3,
            "run_control_on_all_layers": True,
            "pca_enabled": True,
            "pca_samples": 3000,
            "silhouette_enabled": True,
            "silhouette_samples": 3000,
            "enable_abstention": True,
            "enable_per_class_metrics": True,
            "enable_feature_statistics": True,
            "score_weights": {
                "macro_f1": 0.25, "balanced_accuracy": 0.15, "mcc": 0.15,
                "log_loss_score": 0.10, "selectivity": 0.20, "stability": 0.10, "geometry": 0.05,
            },
            "complexity_penalty_scale": 0.02,
            "verbose": VERBOSE_DEFAULT,
        },
        "output_subdir": "analysis/probes",
    }
    save_json(path, example)


class ProbeLogger:
    def __init__(self, level: int):
        self.level = int(level)
        self.t0 = time.perf_counter()

    def emit(self, message: str, level: int = 1) -> None:
        if self.level >= level:
            elapsed = time.perf_counter() - self.t0
            print(f"[probe +{elapsed:8.2f}s] {message}")

    def section(self, title: str, level: int = 1) -> None:
        if self.level >= level:
            self.emit("=" * 96, level)
            self.emit(title, level)
            self.emit("=" * 96, level)


# -----------------------------------------------------------------------------
# Extraction artifact (v2 compatible)
# -----------------------------------------------------------------------------

class ExtractionArtifact:
    def __init__(self, dataset_dir: Path, verify_checksum: bool = False):
        self.dataset_dir = Path(dataset_dir).resolve()

        # Flat layout — every artefact sits directly inside dataset_dir.
        self.states_path     = self.dataset_dir / "hidden_states.npy"
        self.completed_path  = self.dataset_dir / "completed.npy"
        self.metadata_path   = self.dataset_dir / "extraction.json"
        self.sample_ids_path = self.dataset_dir / "sample_ids.npy"
        self.text_hashes_path = self.dataset_dir / "text_hashes.npy"
        self.checksum_path   = self.dataset_dir / "checksum.sha256"

        missing = [str(p) for p in (
            self.states_path, self.completed_path, self.metadata_path,
        ) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required extraction artifact(s):\n- " + "\n- ".join(missing)
            )

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.states    = np.load(self.states_path, mmap_mode="r")
        self.completed = np.load(self.completed_path, mmap_mode="r")

        self.sample_ids = None
        if self.sample_ids_path.exists():
            self.sample_ids = np.load(self.sample_ids_path, allow_pickle=True)
            if self.sample_ids.dtype == object:
                self.sample_ids = np.array([str(x) for x in self.sample_ids], dtype=object)

        self.text_hashes = None
        if self.text_hashes_path.exists():
            self.text_hashes = np.load(self.text_hashes_path, mmap_mode='r')

        self.checksum_stored = None
        if self.checksum_path.exists():
            with open(self.checksum_path, "r") as f:
                self.checksum_stored = f.read().strip()

        if verify_checksum and self.checksum_stored is not None:
            self._verify_checksum()

        self.validation = self._validate()

    def _verify_checksum(self) -> None:
        hasher = hashlib.sha256()
        n_samples, n_layers, n_hidden = self.states.shape
        chunk = 1024 * n_layers * n_hidden
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            hasher.update(np.asarray(self.states[start:end]).tobytes())
        computed = hasher.hexdigest()
        if computed != self.checksum_stored:
            raise RuntimeError(f"Checksum mismatch! Stored: {self.checksum_stored}, Computed: {computed}")

    @property
    def model_name(self):
        name = self.metadata.get("model", {}).get("name")
        if not name:
            parts = self.dataset_dir.parts
            try:
                idx = parts.index('models')
                name = '/'.join(parts[idx+1:idx+3])
            except ValueError:
                name = "unknown"
        return name

    @property
    def dataset_name(self):
        name = self.metadata.get("dataset", {}).get("name")
        if name:
            return name
        return self.dataset_dir.name

    @property
    def sample_count(self) -> int:
        return int(self.metadata.get("dataset", {}).get("samples", self.states.shape[0]))

    @property
    def hidden_layers(self) -> int:
        return int(self.states.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.states.shape[2])

    @property
    def experiment_id(self) -> str | None:
        return self.metadata.get("experiment_id")

    @property
    def pooling(self) -> str | None:
        return self.metadata.get("extraction", {}).get("pooling")

    @property
    def dataset_fingerprint(self) -> str | None:
        return self.metadata.get("dataset", {}).get("fingerprint")

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.metadata.get("dataset", {}).get("provenance", {}))

    def _validate(self) -> dict[str, Any]:
        issues = []
        warnings = []

        if self.states.ndim != 3:
            issues.append(f"hidden_states.npy must be rank-3 [N,L,D], got {self.states.shape}")
        else:
            if self.states.shape[0] < 2:
                issues.append("Hidden-state artifact contains fewer than two samples")
            if self.states.shape[1] < 1:
                issues.append("Hidden-state artifact contains zero layers")
            if self.states.shape[2] < 1:
                issues.append("Hidden-state artifact contains zero hidden dimensions")

        if self.completed.ndim != 1 or self.completed.dtype != np.bool_:
            issues.append(f"completed.npy must be 1-D bool, got shape={self.completed.shape}, dtype={self.completed.dtype}")

        if self.states.shape[0] != self.completed.shape[0]:
            issues.append("states/completed sample counts differ")
        if self.states.shape[0] != self.sample_count:
            issues.append(f"metadata sample count={self.sample_count} != states={self.states.shape[0]}")
        if not bool(np.all(self.completed)):
            issues.append("Completion map is incomplete; partial extraction is not scientifically safe to probe")

        if self.sample_ids is not None and len(self.sample_ids) != self.sample_count:
            issues.append(f"sample_ids length {len(self.sample_ids)} != sample count {self.sample_count}")

        if self.text_hashes is not None and len(self.text_hashes) != self.sample_count:
            issues.append(f"text_hashes length {len(self.text_hashes)} != sample count {self.sample_count}")

        idx = np.linspace(0, self.states.shape[0] - 1, num=min(8, self.states.shape[0]), dtype=int)
        sample = np.asarray(self.states[idx], dtype=np.float32)
        if not np.isfinite(sample).all():
            issues.append("Sampled hidden states contain NaN/Inf")

        status = self.metadata.get("status")
        if status not in {None, "complete"}:
            issues.append(f"Extraction metadata status={status!r}, not complete")

        expected_shape = self.metadata.get("dataset", {}).get("hidden_state_shape")
        if expected_shape is not None and tuple(expected_shape) != tuple(self.states.shape):
            issues.append(f"Metadata hidden_state_shape={expected_shape} != actual={tuple(self.states.shape)}")

        if self.states.dtype not in (np.float16, np.float32, np.float64):
            warnings.append(f"Unusual hidden-state dtype: {self.states.dtype}")
        if not isinstance(self.states, np.memmap):
            warnings.append("hidden_states.npy is not memory-mapped")
        if self.pooling not in {None, "mean", "first_token", "last_token"}:
            warnings.append(f"Unknown pooling value: {self.pooling}")

        if issues:
            raise RuntimeError("Extraction validation failed:\n- " + "\n- ".join(issues))
        return {"status": "pass", "warnings": warnings}

    def analysis_summary(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "experiment_id": self.experiment_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_count": self.sample_count,
            "hidden_layers": self.hidden_layers,
            "hidden_size": self.hidden_size,
            "representation_layout": "[samples, layers, hidden]",
            "array_shape": list(self.states.shape),
            "storage_dtype": str(self.states.dtype),
            "pooling": self.pooling,
            "max_length": self.metadata.get("extraction", {}).get("max_length"),
            "batch_size": self.metadata.get("extraction", {}).get("batch_size"),
            "model_snapshot": self.metadata.get("model", {}).get("snapshot"),
            "has_sample_ids": self.sample_ids is not None,
            "has_text_hashes": self.text_hashes is not None,
            "has_checksum": self.checksum_stored is not None,
            "provenance": self.provenance,
            "validation": self.validation,
        }


# -----------------------------------------------------------------------------
# Dataset loading and label adapters
# -----------------------------------------------------------------------------

def import_callable(module_name: str, function_name: str):
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if not callable(fn):
        raise AttributeError(f"{module_name}.{function_name} is not callable")
    return fn


def to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas") and callable(obj.to_pandas):
        return obj.to_pandas()
    if isinstance(obj, Mapping):
        for key in ("data", "df", "dataset"):
            if key in obj:
                return to_dataframe(obj[key])
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)
    raise TypeError(
        "Dataset loader must return pandas.DataFrame, a HuggingFace Dataset, "
        "a mapping containing one, or a sequence of row records."
    )


def load_dataframe(contract: DatasetContract) -> pd.DataFrame:
    if contract.type == "python":
        if not contract.module or not contract.function:
            raise ValueError("Python dataset source requires module and function")
        return to_dataframe(import_callable(contract.module, contract.function)(**contract.kwargs))

    if contract.type == "file":
        if not contract.path:
            raise ValueError("File dataset source requires path")
        path = Path(contract.path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        raise ValueError(f"Unsupported dataset file type: {suffix}")

    raise ValueError(f"Unsupported dataset source type: {contract.type}")


def resolve_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    *,
    role: str,
    allow_scored_text_guess: bool = False,
) -> tuple[str, dict[str, Any]]:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"Configured {role} column {requested!r} not found. Columns={list(df.columns)}")
        return requested, {"mode": "explicit", "column": requested}

    available = [str(c) for c in df.columns]
    exact = [c for c in candidates if c in available]
    if exact:
        return exact[0], {"mode": "auto", "column": exact[0], "candidates": exact, "ambiguous": len(exact) > 1}

    if role == "text" and allow_scored_text_guess:
        scored = []
        for c in available:
            if c.lower() in COMMON_ID_COLUMNS:
                continue
            vals = df[c].head(min(100, len(df)))
            if len(vals) == 0:
                continue
            string_ratio = float(np.mean(vals.map(lambda x: isinstance(x, str))))
            avg_chars = float(vals.map(lambda x: len(str(x)) if x is not None else 0).mean())
            score = string_ratio * 100 + min(avg_chars, 500) / 10
            if string_ratio >= 0.90 and avg_chars >= 5:
                scored.append((score, c))
        scored.sort(reverse=True)
        if scored:
            if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 2:
                raise RuntimeError(f"Ambiguous automatic text-column detection: {scored[:10]}")
            return scored[0][1], {"mode": "scored_auto", "column": scored[0][1], "scores": scored}

    raise KeyError(f"Could not resolve {role} column. Available={available}")


def _maybe_literal(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(("[", "(", "{")) and s.endswith(("]", ")", "}")):
            # First, try json.loads for valid JSON strings
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            # Then, fall back to ast.literal_eval for other Python-like literals
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
    return value


def parse_integer_list(value: Any) -> list[int]:
    value = _maybe_literal(value)
    if isinstance(value, (list, tuple, set)):
        return [int(x) for x in value]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, str):
        nums = re.findall(r"-?\d+", value)
        if nums:
            return [int(x) for x in nums]
    raise ValueError(f"Cannot parse integer-list label: {value!r}")


def parse_string_list(value: Any) -> list[str]:
    value = _maybe_literal(value)
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(value).strip()]


def infer_target_type(df: pd.DataFrame, contract: DatasetContract) -> str:
    """
    Determine the target adapter.

    Important:
    - Explicit contract.target_type always wins.
    - GoEmotions is detected from genuine multi-label 'labels' structure.
    - A generic 'emotion' column is NOT assumed to mean ISEAR.
      It is treated as custom unless the contract explicitly says isear.
    """
    if contract.target_type and contract.target_type != "auto":
        return contract.target_type.lower()

    cols = set(map(str, df.columns))

    # Detect GoEmotions only when its characteristic multi-label structure
    # is actually present.
    if "labels" in cols:
        sample = df["labels"].head(100).tolist()
        try:
            parsed = [parse_integer_list(x) for x in sample]
            if any(len(v) > 1 for v in parsed):
                return "goemotions"
        except Exception:
            pass

        # A labels column without another obvious emotion column is still
        # likely to be a GoEmotions-style target.
        if not any(
            c in cols
            for c in ("emotion", "emotion_label", "dominant_emotion")
        ):
            return "goemotions"

    # Never infer ISEAR purely from a column name.
    # Generic single-label emotion datasets must remain generic/custom.
    return "custom"


def _normalise_name(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower().replace("_", " "))


def canonical_goemotions_target(df: pd.DataFrame, contract: DatasetContract):
    label_col, resolution = resolve_column(
        df, contract.label_column,
        ["labels", "dominant_emotion", "emotion", "emotion_label", "label"],
        role="label",
    )
    raw = df[label_col].tolist()
    id_map = {i: name for i, name in enumerate(GOEMOTIONS_CLASSES)}
    by_name = {_normalise_name(k): k for k in GOEMOTIONS_CLASSES}

    def decode_one_row(value, row_index):
        value2 = _maybe_literal(value)
        if isinstance(value2, np.ndarray):
            value2 = value2.tolist()
        if isinstance(value2, (list, tuple, set)):
            seq = list(value2)
            source_mode = "sequence"
        elif isinstance(value2, (int, np.integer)):
            seq = [int(value2)]
            source_mode = "integer_id"
        elif isinstance(value2, str):
            s = value2.strip()
            try:
                ids = parse_integer_list(s)
            except ValueError:
                ids = []
            else:
                if ids:
                    seq = ids
                    source_mode = "integer_id_string"
                else:
                    seq = [s]
                    source_mode = "string_name"
        else:
            seq = [value2]
            source_mode = "scalar"

        current_ids, current_names = [], []
        mode = source_mode
        for item in seq:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if idx < 0 or idx >= len(GOEMOTIONS_CLASSES):
                    raise ValueError(f"GoEmotions row {row_index} has invalid label ID {idx}")
                current_ids.append(idx)
                current_names.append(id_map[idx])
                continue
            if isinstance(item, str):
                item_clean = item.strip()
                if re.fullmatch(r"\d+", item_clean):
                    idx = int(item_clean)
                    if idx < 0 or idx >= len(GOEMOTIONS_CLASSES):
                        raise ValueError(f"GoEmotions row {row_index} has invalid label ID {idx}")
                    current_ids.append(idx)
                    current_names.append(id_map[idx])
                    continue
                name = _normalise_name(item_clean)
                if name not in by_name:
                    raise ValueError(f"GoEmotions row {row_index} has unknown label name {item!r}")
                canonical = by_name[name]
                current_names.append(canonical)
                current_ids.append(GOEMOTIONS_CLASSES.index(canonical))
                mode = "string_name"
                continue
            raise ValueError(f"GoEmotions row {row_index} contains unsupported label value {item!r}")
        if not current_ids:
            raise ValueError(f"GoEmotions row {row_index} has no labels")
        seen = set()
        ids_unique, names_unique = [], []
        for idx, name in zip(current_ids, current_names):
            if idx not in seen:
                seen.add(idx)
                ids_unique.append(idx)
                names_unique.append(name)
        return ids_unique, names_unique, mode

    parsed_ids, parsed_names, modes = [], [], set()
    for i, value in enumerate(raw):
        ids, names, mode = decode_one_row(value, i)
        parsed_ids.append(ids)
        parsed_names.append(names)
        modes.add(mode)

    task_type = contract.task_type
    if task_type == "auto":
        task_type = "multi_label" if any(len(x) > 1 for x in parsed_ids) else "single_label"
    if task_type not in {"single_label", "multi_label"}:
        raise ValueError(f"Invalid GoEmotions task type={task_type}")

    if task_type == "multi_label":
        y = np.zeros((len(parsed_ids), len(GOEMOTIONS_CLASSES)), dtype=np.int64)
        for i, labels in enumerate(parsed_ids):
            y[i, labels] = 1
        return y, GOEMOTIONS_CLASSES, {
            "adapter": "goemotions",
            "task_type": "multi_label",
            "raw_label_column": label_col,
            "label_resolution": resolution,
            "label_input_modes": sorted(modes),
            "class_names": GOEMOTIONS_CLASSES,
            "class_count": len(GOEMOTIONS_CLASSES),
            "label_reduction": None,
        }

    policy = contract.single_label_policy or "error_on_multi"
    if policy not in {"first_label", "lowest_id", "error_on_multi"}:
        raise ValueError(f"Unsupported GoEmotions single_label_policy={policy}")
    if policy == "error_on_multi" and any(len(x) != 1 for x in parsed_ids):
        raise ValueError("GoEmotions contains multi-label examples. Set task_type='multi_label' or choose a single_label_policy.")
    y = np.asarray([labels[0] if policy == "first_label" else min(labels) for labels in parsed_ids], dtype=np.int64)
    return y, GOEMOTIONS_CLASSES, {
        "adapter": "goemotions",
        "task_type": "single_label",
        "raw_label_column": label_col,
        "label_resolution": resolution,
        "label_input_modes": sorted(modes),
        "class_names": GOEMOTIONS_CLASSES,
        "class_count": len(GOEMOTIONS_CLASSES),
        "label_reduction": policy,
        "rows_with_multiple_source_labels": int(sum(len(x) > 1 for x in parsed_ids)),
    }


def canonical_isear_target(df: pd.DataFrame, contract: DatasetContract):
    """
    Canonicalise ISEAR labels.

    Accepted forms include:
        1
        [1]
        "1"
        "[1]"
        "joy"
        ["joy"]

    ISEAR numeric IDs are 1..7 and are converted to canonical
    zero-based class IDs 0..6.
    """
    label_col, resolution = resolve_column(
        df,
        contract.label_column,
        ["emotion", "label", "labels", "category", "emotion_label"],
        role="label",
    )

    raw_values = df[label_col].tolist()

    order = contract.class_order or ISEAR_CLASSES

    if len(order) != 7:
        raise ValueError(
            f"ISEAR class_order must contain exactly 7 emotions, got {len(order)}."
        )

    aliases = {
        "joy": "joy",
        "fear": "fear",
        "anger": "anger",
        "sadness": "sadness",
        "disgust": "disgust",
        "shame": "shame",
        "guilt": "guilt",
    }

    mapping = {
        _normalise_name(name): i
        for i, name in enumerate(order)
    }

    canonical_ids = []
    input_modes = set()

    for row_index, raw_value in enumerate(raw_values):
        value = _maybe_literal(raw_value)

        # ---------------------------------------------------------
        # Sequence form: [1], [2], ["joy"], ["fear"], etc.
        # ---------------------------------------------------------
        if isinstance(value, (list, tuple, set, np.ndarray)):
            seq = list(value)

            if len(seq) != 1:
                raise ValueError(
                    f"ISEAR row {row_index} must contain exactly one label; "
                    f"got {seq!r}"
                )

            value = seq[0]
            input_modes.add("single_element_sequence")

        # ---------------------------------------------------------
        # Numeric ID: 1..7
        # ---------------------------------------------------------
        if isinstance(value, (int, np.integer)):
            numeric_id = int(value)

            if not 1 <= numeric_id <= 7:
                raise ValueError(
                    f"ISEAR row {row_index} has numeric label {numeric_id}; "
                    f"expected integer in 1..7."
                )

            canonical_ids.append(numeric_id - 1)
            input_modes.add("numeric_id")
            continue

        # ---------------------------------------------------------
        # String forms:
        #   "1"
        #   "[1]"
        #   "joy"
        # ---------------------------------------------------------
        if isinstance(value, str):
            s = value.strip()

            # Numeric string
            if re.fullmatch(r"[1-7]", s):
                canonical_ids.append(int(s) - 1)
                input_modes.add("numeric_id_string")
                continue

            # Named emotion
            name = aliases.get(_normalise_name(s))

            if name is not None:
                canonical_ids.append(mapping[_normalise_name(name)])
                input_modes.add("string_name")
                continue

            raise ValueError(
                f"ISEAR row {row_index} has unknown emotion {value!r}"
            )

        raise ValueError(
            f"ISEAR row {row_index} contains unsupported label value "
            f"{raw_value!r}"
        )

    y = np.asarray(canonical_ids, dtype=np.int64)

    if len(y) != len(df):
        raise RuntimeError(
            f"ISEAR target length {len(y)} != dataframe length {len(df)}"
        )

    if np.any(y < 0) or np.any(y >= len(order)):
        raise ValueError("ISEAR canonical labels are outside class range.")

    return y, order, {
        "adapter": "isear",
        "task_type": "single_label",
        "raw_label_column": label_col,
        "label_resolution": resolution,
        "class_names": order,
        "class_count": len(order),
        "normalisation": "ISEAR numeric/name/list canonicalisation",
        "label_input_modes": sorted(input_modes),
    }

def canonical_dimensional_target(df: pd.DataFrame, contract: DatasetContract):
    """Continuous-vector targets. Returns (N, D) float32 and D dimension names."""
    # EmoBank exposes V/A/D as three columns rather than one vector column.
    if all(c in df.columns for c in ("V", "A", "D")):
        cols = ["V", "A", "D"]
        y = df[cols].to_numpy(dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("VAD target contains NaN or Inf")
        return y, cols, {
            "adapter": "dimensional",
            "task_type": "dimensional",
            "raw_label_column": "__VAD__",
            "class_names": cols,
            "class_count": len(cols),
            "dimension_count": len(cols),
        }

    label_col, resolution = resolve_column(
        df, contract.label_column,
        ["label", "labels", "target", "targets", "vad", "embedding"],
        role="label",
    )
    rows = []
    for i, x in enumerate(df[label_col].tolist()):
        x = _maybe_literal(x)
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise ValueError(
                f"Dimensional row {i} is not a vector: {type(x).__name__}"
            )
        rows.append([float(v) for v in x])
    y = np.asarray(rows, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"Dimensional target shape {y.shape}, expected (N, D)")
    if not np.isfinite(y).all():
        raise ValueError("Dimensional target contains NaN or Inf")

    dims = contract.class_order or [f"dim_{i}" for i in range(y.shape[1])]
    if len(dims) != y.shape[1]:
        raise ValueError(
            f"class_order has {len(dims)} entries but target has "
            f"{y.shape[1]} dimensions"
        )
    return y, list(dims), {
        "adapter": "dimensional",
        "task_type": "dimensional",
        "raw_label_column": label_col,
        "label_resolution": resolution,
        "class_names": list(dims),
        "class_count": y.shape[1],
        "dimension_count": y.shape[1],
    }
def canonical_custom_target(df: pd.DataFrame, contract: DatasetContract):
    label_col, resolution = resolve_column(
        df, contract.label_column, COMMON_LABEL_COLUMNS, role="label"
    )
    raw = df[label_col].tolist()

    # ── Unwrap single-element lists: [0] → 0, ["joy"] → "joy". ──
    unwrapped: list = []
    for x in raw:
        x = _maybe_literal(x)
        if isinstance(x, (list, tuple, set, np.ndarray)) and len(x) == 1:
            x = list(x)[0]
        unwrapped.append(x)

    task_type = contract.task_type
    if task_type == "auto":
        task_type = "multi_label" if any(
            isinstance(x, (list, tuple, set, np.ndarray)) for x in unwrapped
        ) else "single_label"

    # ─────────────────────────────────────────────────────────────────
    # Multi-label: existing behaviour, unchanged.
    # ─────────────────────────────────────────────────────────────────
    if task_type == "multi_label":
        label_lists = [parse_string_list(x) for x in unwrapped]
        classes = contract.class_order or sorted({x for row in label_lists for x in row})
        mapping = {str(name): i for i, name in enumerate(classes)}
        y = np.zeros((len(label_lists), len(classes)), dtype=np.int64)
        for i, row in enumerate(label_lists):
            if not row:
                raise ValueError(f"Custom multi-label row {i} has no labels")
            for label in row:
                if str(label) not in mapping:
                    raise ValueError(f"Unknown custom label {label!r} at row {i}")
                y[i, mapping[str(label)]] = 1
        return y, classes, {
            "adapter": "custom",
            "task_type": "multi_label",
            "raw_label_column": label_col,
            "label_resolution": resolution,
            "class_names": classes,
            "class_count": len(classes),
        }

    # ─────────────────────────────────────────────────────────────────
    # Single-label.
    # ─────────────────────────────────────────────────────────────────
    # Normalise to ints if possible, else keep as strings.
    def _coerce(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)) and float(x).is_integer():
            return int(x)
        s = str(x).strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return s

    scalar = [_coerce(x) for x in unwrapped]

    # ── Case A: contract supplies class_order as a positional name map.
    #   class_order[i] names class id i. Requires integer labels.
    if contract.class_order is not None:
        if not all(isinstance(v, int) for v in scalar):
            raise ValueError(
                "class_order was provided but labels are not integers. "
                "Either drop class_order, or supply integer labels."
            )
        classes = list(contract.class_order)
        y = np.asarray(scalar, dtype=np.int64)
        if y.min() < 0 or y.max() >= len(classes):
            raise ValueError(
                f"Labels reach {y.max()} but class_order has only "
                f"{len(classes)} entries (valid ids 0..{len(classes)-1})."
            )
        meta = {
            "adapter": "custom",
            "task_type": "single_label",
            "raw_label_column": label_col,
            "label_resolution": resolution,
            "class_names": classes,
            "class_count": len(classes),
            "class_order_source": "contract",
        }
        return y, classes, meta

    # ── Case B: derive the class list from the data.
    #   Integers sort numerically; strings sort lexically.
    def _sort_key(v):
        if isinstance(v, int):
            return (0, v, "")
        return (1, 0, str(v))
    unique = sorted(set(scalar), key=_sort_key)
    classes = unique
    mapping = {v: i for i, v in enumerate(unique)}
    y = np.asarray([mapping[v] for v in scalar], dtype=np.int64)
    meta = {
        "adapter": "custom",
        "task_type": "single_label",
        "raw_label_column": label_col,
        "label_resolution": resolution,
        "class_names": classes,
        "class_count": len(classes),
        "class_order_source": "derived",
    }
    return y, classes, meta

def infer_target_type(df, contract):
    if contract.target_type and contract.target_type != "auto":
        return contract.target_type.lower()

    cols = set(map(str, df.columns))

    # Dimensional: VAD columns, or an explicit vector column.
    if {"V", "A", "D"}.issubset(cols):
        return "dimensional"

    if "labels" in cols:
        sample = df["labels"].head(100).tolist()
        try:
            parsed = [parse_integer_list(x) for x in sample]
            if any(len(v) > 1 for v in parsed):
                return "goemotions"
        except Exception:
            pass
        if not any(c in cols for c in ("emotion", "emotion_label", "dominant_emotion")):
            return "goemotions"

    return "custom"

def build_targets(df, contract):
    target_type = infer_target_type(df, contract)
    if target_type == "goemotions":
        return canonical_goemotions_target(df, contract)
    if target_type == "isear":
        return canonical_isear_target(df, contract)
    if target_type == "dimensional":
        return canonical_dimensional_target(df, contract)
    if target_type == "custom":
        return canonical_custom_target(df, contract)
    raise ValueError(f"Unsupported target_type={target_type}")

# -----------------------------------------------------------------------------
# Provenance and target validation
# -----------------------------------------------------------------------------

def _get_metadata_text_hashes(artifact: ExtractionArtifact) -> dict[str, str | None]:
    prov = artifact.provenance
    return {
        "derived_fingerprint": prov.get("derived_fingerprint"),
        "head_hash": prov.get("head_hash"),
        "tail_hash": prov.get("tail_hash"),
        "full_hash": prov.get("full_hash"),
        "native_fingerprint": prov.get("native_fingerprint"),
    }


def validate_text_alignment(artifact, df, contract):
    text_col, resolution = resolve_column(df, contract.text_column, COMMON_TEXT_COLUMNS, role="text", allow_scored_text_guess=True)
    texts = one_dim_strings(df[text_col].tolist())
    if len(texts) != artifact.sample_count:
        raise RuntimeError(f"Text row count={len(texts)} differs from hidden-state count={artifact.sample_count}")

    observed = {
        "derived_fingerprint": fingerprint_values(texts),
        "head_hash": sequence_hash(texts[:100]),
        "tail_hash": sequence_hash(texts[-100:]),
    }
    expected = _get_metadata_text_hashes(artifact)
    checks = {}
    checked_fields = []
    for key in ("derived_fingerprint", "head_hash", "tail_hash"):
        if expected[key] is not None:
            checked_fields.append(key)
            checks[key] = expected[key] == observed[key]

    warning = None
    if checks:
        if contract.lenient_provenance:
            head_ok = checks.get("head_hash", True)
            tail_ok = checks.get("tail_hash", True)
            if not (head_ok and tail_ok):
                raise RuntimeError("Dataset/text provenance mismatch (head/tail).")
            if "derived_fingerprint" in checks and not checks["derived_fingerprint"]:
                warning = "Derived fingerprint mismatch but head/tail match. Proceeding with lenient provenance."
        else:
            if not all(checks.values()):
                raise RuntimeError("Dataset/text provenance mismatch.")

    provenance_available = bool(checked_fields)
    if contract.require_provenance and not provenance_available:
        raise RuntimeError("require_provenance=True but extraction metadata contains no usable text provenance hashes.")

    sample_ids_match = None
    if artifact.sample_ids is not None:
        id_col = contract.id_column
        if id_col and id_col != "auto" and id_col in df.columns:
            df_ids = one_dim_strings(df[id_col].tolist())
            if len(df_ids) == len(artifact.sample_ids):
                sample_ids_match = bool(np.array_equal(df_ids, artifact.sample_ids))
            else:
                sample_ids_match = False

    return {
        "status": "verified" if provenance_available else "unverified",
        "verified": provenance_available,
        "provenance_available": provenance_available,
        "text_column": text_col,
        "text_resolution": resolution,
        "checked_fields": checked_fields,
        "checks": checks,
        "expected": expected,
        "observed": observed,
        "has_sample_ids": artifact.sample_ids is not None,
        "sample_ids_match": sample_ids_match,
        "warning": warning,
    }


def validate_label_alignment(artifact, df, contract, y, classes):
    observed = stable_hash({"classes": list(classes), "labels": np.asarray(y).tolist()}, 24)
    expected = artifact.provenance.get("label_fingerprint") or artifact.provenance.get("target_fingerprint")
    if expected is not None and expected != observed:
        raise RuntimeError("Label provenance mismatch.")
    
    if expected is None:
        if contract.require_label_fingerprint and not contract.allow_missing_label_fingerprint:
            raise RuntimeError("require_label_fingerprint=True but extraction metadata contains no label/target fingerprint.")
        warning = "No label fingerprint was stored during extraction." if expected is None else None
    else:
        warning = None
    
    return {
        "status": "verified" if expected is not None else "unverified",
        "verified": expected is not None,
        "provenance_available": expected is not None,
        "label_fingerprint": observed,
        "metadata_label_fingerprint": expected,
        "warning": warning,
    }


def validate_targets(y, classes, task_type):
    issues, warnings = [], []
    y = np.asarray(y)
    if len(classes) < 2:
        issues.append("At least two classes are required")
    if task_type == "dimensional":
        y = np.asarray(y)
        if y.ndim != 2:
            issues.append(f"Dimensional target must be rank-2 [N, D], got {y.shape}")
        elif not np.issubdtype(y.dtype, np.floating):
            issues.append(f"Dimensional target must be float, got {y.dtype}")
        elif not np.isfinite(y).all():
            issues.append("Dimensional target contains NaN/Inf")
        else:
            for j, name in enumerate(classes):
                col = y[:, j]
                if col.std() < 1e-6:
                    warnings.append(f"Dimension {name!r} has near-zero variance")
        if issues:
            raise RuntimeError("Target validation failed:\n- " + "\n- ".join(issues))
        return {"status": "pass", "warnings": warnings, "class_count": y.shape[1]}
    if task_type == "single_label":
        if y.ndim != 1:
            issues.append(f"Single-label target must be rank-1, got {y.shape}")
        elif not np.issubdtype(y.dtype, np.integer):
            issues.append(f"Single-label target must be integer encoded, got {y.dtype}")
        elif np.any(y < 0) or np.any(y >= len(classes)):
            issues.append("Single-label class IDs are outside the class range")
        if y.ndim == 1:
            counts = np.bincount(y, minlength=len(classes))
            absent = [classes[i] for i, c in enumerate(counts) if c == 0]
            rare = [classes[i] for i, c in enumerate(counts) if 0 < c < 5]
            if absent:
                warnings.append(f"Absent classes: {absent}")
            if rare:
                warnings.append(f"Very rare classes (<5 examples): {rare}")
            if len(np.unique(y)) < 2:
                issues.append("Target contains only one observed class")
    elif task_type == "multi_label":
        if y.ndim != 2:
            issues.append(f"Multi-label target must be rank-2, got {y.shape}")
        elif y.shape[1] != len(classes):
            issues.append(f"Target width={y.shape[1]} != number of classes={len(classes)}")
        elif not np.isin(y, [0, 1]).all():
            issues.append("Multi-label target must contain only 0/1")
        if y.ndim == 2:
            positives = y.sum(axis=0)
            rare = [classes[i] for i, c in enumerate(positives) if 0 < c < 5]
            absent = [classes[i] for i, c in enumerate(positives) if c == 0]
            if rare:
                warnings.append(f"Very rare labels (<5 positives): {rare}")
            if absent:
                warnings.append(f"Absent labels: {absent}")
            if np.all(positives == 0):
                issues.append("No positive labels are present")
    else:
        issues.append(f"Unsupported task type {task_type}")
    if issues:
        raise RuntimeError("Target validation failed:\n- " + "\n- ".join(issues))
    return {"status": "pass", "warnings": warnings, "class_count": len(classes)}


# -----------------------------------------------------------------------------
# Split logic
# -----------------------------------------------------------------------------

def can_stratify_single(y, min_count=3):
    if y.ndim != 1:
        return False
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    return len(nonzero) >= 2 and bool(np.all(nonzero >= min_count))


def make_single_splits(y, cfg, seed):
    indices = np.arange(len(y))
    stratify = y if cfg.stratify and can_stratify_single(y, 3) else None
    train_idx, temp_idx = train_test_split(indices, test_size=1.0 - cfg.train, random_state=seed, stratify=stratify)
    temp_y = y[temp_idx]
    temp_stratify = temp_y if cfg.stratify and can_stratify_single(temp_y, 2) else None
    test_fraction_of_temp = cfg.test / (cfg.validation + cfg.test)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_fraction_of_temp, random_state=seed, stratify=temp_stratify)
    return {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def make_multilabel_splits(y, cfg, seed):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1.0 - cfg.train, random_state=seed)
        idx = np.arange(len(y))
        train_rel, temp_rel = next(splitter.split(idx, y))
        temp_y = y[temp_rel]
        splitter2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=cfg.test / (cfg.validation + cfg.test), random_state=seed + 1)
        val_rel, test_rel = next(splitter2.split(temp_rel, temp_y))
        return {
            "train": np.sort(train_rel),
            "validation": np.sort(temp_rel[val_rel]),
            "test": np.sort(temp_rel[test_rel]),
            "method": np.array(["iterative"], dtype=object),
        }
    except Exception:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(cfg.train * n))
        n_val = int(round(cfg.validation * n))
        return {
            "train": np.sort(idx[:n_train]),
            "validation": np.sort(idx[n_train:n_train + n_val]),
            "test": np.sort(idx[n_train + n_val:]),
            "method": np.array(["random_fallback"], dtype=object),
        }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def safe_mcc(y_true, y_pred):
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return float("nan")


def _safe_multilabel_auc_and_ap(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        probabilities = probabilities.reshape(-1, 1)
    valid_auc, valid_ap = [], []
    for j in range(y_true.shape[1]):
        target = y_true[:, j]
        score = probabilities[:, j]
        positives = int(np.sum(target == 1))
        negatives = int(np.sum(target == 0))
        if positives > 0 and negatives > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                valid_auc.append(float(roc_auc_score(target, score)))
                valid_ap.append(float(average_precision_score(target, score)))
    coverage = {
        "total_labels": int(y_true.shape[1]),
        "valid_auc_labels": len(valid_auc),
        "valid_ap_labels": len(valid_ap),
        "undefined_labels": int(y_true.shape[1] - len(valid_auc)),
    }
    return (float(np.mean(valid_auc)) if valid_auc else None,
            float(np.mean(valid_ap)) if valid_ap else None,
            coverage)


def safe_roc_auc_single(y_true, proba, n_classes):
    if proba is None:
        return None
    y_true = np.asarray(y_true)
    if n_classes == 2:
        if len(np.unique(y_true)) < 2:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(roc_auc_score(y_true, proba[:, 1]))
        except Exception:
            return None
    onehot = np.eye(n_classes, dtype=np.int64)[y_true]
    auc, _, _ = _safe_multilabel_auc_and_ap(onehot, proba)
    return auc


def safe_average_precision_single(y_true, proba, n_classes):
    if proba is None:
        return None
    y_true = np.asarray(y_true)
    if n_classes == 2:
        if len(np.unique(y_true)) < 2:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(average_precision_score(y_true, proba[:, 1]))
        except Exception:
            return None
    onehot = np.eye(n_classes, dtype=np.int64)[y_true]
    _, ap, _ = _safe_multilabel_auc_and_ap(onehot, proba)
    return ap


def confidence_metrics(y_true, proba, y_pred):
    confidence = np.max(proba, axis=1)
    correct = (y_true == y_pred).astype(float)
    return {
        "mean_confidence": float(np.mean(confidence)),
        "mean_confidence_correct": float(np.mean(confidence[correct == 1])) if np.any(correct == 1) else None,
        "mean_confidence_incorrect": float(np.mean(confidence[correct == 0])) if np.any(correct == 0) else None,
        "high_confidence_error_rate": float(np.mean((confidence >= 0.8) & (correct == 0))) if len(confidence) else None,
    }


def evaluate_single(y_true, y_pred, classes, probabilities=None, include_per_class=True):
    labels = np.arange(len(classes))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": safe_mcc(y_true, y_pred),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=list(classes), output_dict=True, zero_division=0),
    }
    if probabilities is not None:
        try:
            result["log_loss"] = float(log_loss(y_true, probabilities, labels=labels))
        except Exception:
            result["log_loss"] = None
        result["roc_auc_ovr_macro"] = safe_roc_auc_single(y_true, probabilities, len(classes))
        result["average_precision_macro"] = safe_average_precision_single(y_true, probabilities, len(classes))
        ll = result.get("log_loss")
        result["log_loss_score"] = float(np.exp(-min(max(ll, 0.0), 20.0))) if ll is not None else None
        result.update(confidence_metrics(y_true, probabilities, y_pred))
    if include_per_class:
        result["per_class"] = {
            name: {
                "precision": float(precision_score(y_true, y_pred, labels=[i], average=None, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)),
                "support": int(np.sum(y_true == i)),
            }
            for i, name in enumerate(classes)
        }
    return result


def evaluate_multi(y_true, y_pred, probabilities=None, classes=None):
    result = {
        "exact_match_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "hamming_score": float(1.0 - hamming_loss(y_true, y_pred)),
        "macro_jaccard": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": safe_mcc(y_true.ravel(), y_pred.ravel()),
        "label_cardinality_true": float(np.mean(y_true.sum(axis=1))),
        "label_cardinality_pred": float(np.mean(y_pred.sum(axis=1))),
    }
    positives = np.sum(y_true, axis=0)
    negatives = np.sum(y_true == 0, axis=0)
    result["labels_with_positive_support"] = int(np.sum(positives > 0))
    result["labels_with_negative_support"] = int(np.sum(negatives > 0))
    result["labels_with_both_support"] = int(np.sum((positives > 0) & (negatives > 0)))
    if probabilities is not None:
        try:
            result["log_loss"] = float(log_loss(y_true.ravel(), probabilities.ravel(), labels=[0, 1]))
            result["log_loss_score"] = float(np.exp(-min(max(result["log_loss"], 0.0), 20.0)))
        except Exception:
            result["log_loss"] = None
            result["log_loss_score"] = None
        roc_auc, avg_precision, coverage = _safe_multilabel_auc_and_ap(y_true, probabilities)
        result["roc_auc_macro"] = roc_auc
        result["average_precision_macro"] = avg_precision
        result["roc_auc_coverage"] = coverage
    if classes is not None:
        result["per_class"] = {}
        for j, name in enumerate(classes):
            support = int(y_true[:, j].sum())
            predicted_positive = int(y_pred[:, j].sum())
            prec = precision_score(y_true[:, j], y_pred[:, j], average=None, zero_division=0)
            rec = recall_score(y_true[:, j], y_pred[:, j], average=None, zero_division=0)
            f1 = f1_score(y_true[:, j], y_pred[:, j], average=None, zero_division=0)
            result["per_class"][name] = {
                "f1": float(f1[0]),
                "precision": float(prec[0]),
                "recall": float(rec[0]),
                "support": support,
                "predicted_positive": predicted_positive,
                "roc_auc_defined": bool(support > 0 and negatives[j] > 0),
            }
    return result


def majority_baseline(y_train, y_test, classes):
    counts = np.bincount(y_train, minlength=len(classes))
    majority_id = int(np.argmax(counts))
    pred = np.full(len(y_test), majority_id, dtype=np.int64)
    return {
        "baseline": "majority_class",
        "class": classes[majority_id],
        "test": evaluate_single(y_test, pred, classes),
        "chance_accuracy": 1.0 / len(classes),
    }


def label_entropy(y, task_type):
    if task_type == "single_label":
        counts = np.bincount(y)
        p = counts[counts > 0] / len(y)
        return float(-np.sum(p * np.log2(p)))
    counts = y.mean(axis=0)
    return float(np.mean([-(p * np.log2(p) + (1 - p) * np.log2(1 - p)) for p in counts if 0 < p < 1]))


# -----------------------------------------------------------------------------
# Probe architecture
# -----------------------------------------------------------------------------

def resolve_hidden_width(spec, input_dim):
    if isinstance(spec, int):
        if spec < 1:
            raise ValueError(f"Hidden width must be >=1, got {spec}")
        return spec
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*d\s*", str(spec).lower())
    if m:
        ratio = float(m.group(1))
        if ratio <= 0:
            raise ValueError(f"Invalid relative width {spec}")
        return max(1, int(round(ratio * input_dim)))
    if str(spec).isdigit():
        return int(spec)
    raise ValueError(f"Invalid hidden width {spec!r}; use integer or e.g. '0.5d'")


def resolved_hidden_dims(spec, input_dim):
    if spec.type != "mlp":
        return []
    if spec.complexity == "custom":
        if not spec.hidden_dims:
            raise ValueError(f"Custom MLP {spec.name} requires hidden_dims")
        dims = [resolve_hidden_width(x, input_dim) for x in spec.hidden_dims]
    else:
        depth = {"1_hidden": 1, "2_hidden": 2, "3_hidden": 3}[spec.complexity]
        if spec.hidden_dims:
            dims = [resolve_hidden_width(x, input_dim) for x in spec.hidden_dims]
            if len(dims) != depth:
                raise ValueError(f"{spec.name}: hidden_dims length must be {depth}")
        else:
            if spec.hidden_width_ratio <= 0:
                raise ValueError("hidden_width_ratio must be >0")
            dims = [max(1, int(round(input_dim * spec.hidden_width_ratio / (2 ** i)))) for i in range(depth)]
    return dims


class TorchMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Probe fitting
# -----------------------------------------------------------------------------

def _make_logistic(spec, seed):
    steps = []
    if spec.standardize:
        steps.append(("scale", StandardScaler()))
    steps.append(("logistic", LogisticRegression(C=spec.C, max_iter=spec.max_iter, solver="lbfgs", random_state=seed)))
    return Pipeline(steps)


def fit_logistic_single(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, include_per_class):
    model = _make_logistic(spec, seed)
    model.fit(X_train, y_train)

    def pred(X):
        p = model.predict(X)
        prob = model.predict_proba(X)
        return p, prob

    trp, trprob = pred(X_train)
    vap, vaprob = pred(X_val)
    tep, teprob = pred(X_test)
    metrics = {
        "train": evaluate_single(y_train, trp, classes, trprob, include_per_class),
        "validation": evaluate_single(y_val, vap, classes, vaprob, include_per_class),
        "test": evaluate_single(y_test, tep, classes, teprob, include_per_class),
        "parameters": int(model.named_steps["logistic"].coef_.size + model.named_steps["logistic"].intercept_.size),
        "resolved_hidden_dims": [],
        "epochs_completed": None,
    }
    return metrics, model

class ConstantPredictor: 
    """Trivial predictor used when a binary label has only one class in training."""
    def __init__(self, constant: int):
        self.constant = int(constant)

    def predict(self, X):
        return np.full(len(X), self.constant, dtype=np.int64)

    def predict_proba(self, X):
        p1 = np.full(len(X), float(self.constant), dtype=float)
        return np.column_stack([1 - p1, p1])

def _fit_one_binary(X_train, target_train, X_val, X_test, spec, seed):
    unique = np.unique(target_train)
    if len(unique) == 1:
        constant = int(unique[0])
        return ConstantPredictor(constant)  
    model = _make_logistic(spec, seed)
    model.fit(X_train, target_train)
    return model


def fit_logistic_multi(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, include_per_class):
    models = []
    train_prob = np.zeros_like(y_train, dtype=np.float64)
    val_prob = np.zeros_like(y_val, dtype=np.float64)
    test_prob = np.zeros_like(y_test, dtype=np.float64)
    for j in range(y_train.shape[1]):
        model = _fit_one_binary(X_train, y_train[:, j], X_val, X_test, spec, seed + j)
        train_prob[:, j] = model.predict_proba(X_train)[:, 1]
        val_prob[:, j] = model.predict_proba(X_val)[:, 1]
        test_prob[:, j] = model.predict_proba(X_test)[:, 1]
        models.append(model)
    train_pred = (train_prob >= 0.5).astype(np.int64)
    val_pred = (val_prob >= 0.5).astype(np.int64)
    test_pred = (test_prob >= 0.5).astype(np.int64)
    return {
        "train": evaluate_multi(y_train, train_pred, train_prob, classes),
        "validation": evaluate_multi(y_val, val_pred, val_prob, classes),
        "test": evaluate_multi(y_test, test_pred, test_prob, classes),
        "parameters": int(sum(m.named_steps["logistic"].coef_.size + m.named_steps["logistic"].intercept_.size if hasattr(m, "named_steps") else 2 for m in models)),
        "resolved_hidden_dims": [],
        "epochs_completed": None,
    }, models


def _selection_value(metrics, metric):
    return float(metrics.get(metric, float("nan")))


def fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, task_type, device, include_per_class):
    seed_everything(seed)
    hidden = resolved_hidden_dims(spec, X_train.shape[1])
    output_dim = y_train.shape[1] if task_type == "multi_label" else len(classes)
    model = TorchMLP(X_train.shape[1], output_dim, hidden, spec.dropout).to(device)

    if task_type == "multi_label":
        criterion = nn.BCEWithLogitsLoss()
        y_train_t = torch.from_numpy(y_train.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32))
    else:
        criterion = nn.CrossEntropyLoss()
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        y_val_t = torch.from_numpy(y_val.astype(np.int64))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train.astype(np.float32)), y_train_t),
        batch_size=min(spec.batch_size, len(X_train)), shuffle=True, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_selection = -np.inf
    stale = 0
    history = {"train_loss": [], "validation_loss": [], "validation_score": []}

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_dev = y_val_t.to(device)

    for epoch in range(spec.epochs):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Probe {spec.name} produced non-finite loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t)
            val_logits = model(X_val_t)
            train_loss = float(criterion(train_logits, y_train_t.to(device)).item())
            val_loss = float(criterion(val_logits, y_val_dev).item())

        if task_type == "multi_label":
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = (val_prob >= 0.5).astype(np.int64)
            val_metrics = evaluate_multi(y_val, val_pred, val_prob, classes)
            selection = float(val_metrics.get(spec.selection_metric, val_metrics["macro_f1"]))
        else:
            val_prob = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_pred = val_prob.argmax(axis=1)
            val_metrics = evaluate_single(y_val, val_pred, classes, val_prob, include_per_class)
            selection = _selection_value(val_metrics, spec.selection_metric)

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["validation_loss"].append(val_loss)
        history["validation_score"].append(selection)

        if np.isfinite(selection) and selection > best_selection + 1e-8:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(torch.from_numpy(X_train.astype(np.float32)).to(device))
        val_logits = model(torch.from_numpy(X_val.astype(np.float32)).to(device))
        test_logits = model(torch.from_numpy(X_test.astype(np.float32)).to(device))

    if task_type == "multi_label":
        train_prob = torch.sigmoid(train_logits).cpu().numpy()
        val_prob = torch.sigmoid(val_logits).cpu().numpy()
        test_prob = torch.sigmoid(test_logits).cpu().numpy()
        trm = evaluate_multi(y_train, (train_prob >= 0.5).astype(int), train_prob, classes)
        vam = evaluate_multi(y_val, (val_prob >= 0.5).astype(int), val_prob, classes)
        tem = evaluate_multi(y_test, (test_prob >= 0.5).astype(int), test_prob, classes)
    else:
        train_prob = torch.softmax(train_logits, dim=1).cpu().numpy()
        val_prob = torch.softmax(val_logits, dim=1).cpu().numpy()
        test_prob = torch.softmax(test_logits, dim=1).cpu().numpy()
        trm = evaluate_single(y_train, train_prob.argmax(axis=1), classes, train_prob, include_per_class)
        vam = evaluate_single(y_val, val_prob.argmax(axis=1), classes, val_prob, include_per_class)
        tem = evaluate_single(y_test, test_prob.argmax(axis=1), classes, test_prob, include_per_class)

    return {
        "train": trm,
        "validation": vam,
        "test": tem,
        "parameters": count_parameters(model),
        "resolved_hidden_dims": hidden,
        "epochs_completed": len(history["train_loss"]),
        "best_validation_score": float(best_selection),
        "history": history,
    }, model


def fit_probe(spec, X_train, y_train, X_val, y_val, X_test, y_test, classes, task_type, seed, device, include_per_class):
    validate_probe_spec(spec, task_type)
    if spec.type == "logistic":
        if task_type == "single_label":
            return fit_logistic_single(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, include_per_class)
        return fit_logistic_multi(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, include_per_class)
    return fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, task_type, device, include_per_class)


# -----------------------------------------------------------------------------
# Geometry and representation statistics
# -----------------------------------------------------------------------------

def geometry_analysis(X, y, classes, task_type, seed, cfg):
    idx = sample_indices(len(X), max(cfg.pca_samples, cfg.silhouette_samples), seed)
    Xs = X[idx]
    result = {
        "sample_count": int(len(Xs)),
        "dimension": int(Xs.shape[1]),
        "mean": float(Xs.mean()),
        "std": float(Xs.std()),
        "mean_l2_norm": float(np.mean(np.linalg.norm(Xs, axis=1))),
        "zero_fraction": float(np.mean(Xs == 0)),
        "finite": bool(np.isfinite(Xs).all()),
    }

    if cfg.enable_feature_statistics:
        var = np.var(Xs, axis=0)
        result["feature_variance_mean"] = float(var.mean())
        result["feature_variance_zero_fraction"] = float(np.mean(var == 0))
        result["feature_variance_p95"] = float(np.percentile(var, 95))

    if task_type == "single_label" and cfg.pca_enabled:
        n_components = min(10, Xs.shape[0], Xs.shape[1])
        if n_components >= 2:
            pca = PCA(n_components=n_components, random_state=seed)
            z = pca.fit_transform(Xs)
            result["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
            result["pca_cumulative"] = np.cumsum(pca.explained_variance_ratio_).tolist()
            result["pca_2d"] = z[:, :2].tolist()
            result["pca_labels"] = y[idx].tolist()
            result["pca_2d_variance"] = float(np.sum(pca.explained_variance_ratio_[:2]))

    if task_type == "single_label" and cfg.silhouette_enabled:
        sid = sample_indices(len(Xs), cfg.silhouette_samples, seed + 1)
        ys = y[idx][sid]
        if len(np.unique(ys)) > 1 and len(sid) >= max(10, len(np.unique(ys)) + 2):
            try:
                result["silhouette_score"] = float(silhouette_score(Xs[sid], ys[sid]))
            except Exception as exc:
                result["silhouette_score"] = None
                result["silhouette_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["silhouette_score"] = None
    else:
        result["silhouette_score"] = None
    return result


# -----------------------------------------------------------------------------
# Score system
# -----------------------------------------------------------------------------

def _normalise_weights(weights):
    clean = {k: float(v) for k, v in weights.items() if float(v) >= 0}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("At least one score weight must be >0")
    return {k: v / total for k, v in clean.items()}


def compute_complexity_penalty(parameters, input_dim, scale):
    if parameters is None or not np.isfinite(parameters) or parameters <= 0:
        return 0.0
    relative = math.log10(max(parameters, 1)) / math.log10(max(input_dim * 100.0, 10.0))
    return float(np.clip(scale * relative, 0, scale))


def add_score_columns(results_df, control_df, cfg, task_type):
    df = results_df.copy()
    weights = _normalise_weights(cfg.score_weights)

    if task_type == "single_label":
        class_count = int(df["class_count"].iloc[0])
        chance = 1.0 / class_count
    else:
        chance = 0.0

    df["macro_f1_component"] = np.clip(df["test_macro_f1"], 0, 1)
    df["balanced_accuracy_component"] = np.clip(df["test_balanced_accuracy"], 0, 1)
    df["mcc_component"] = np.clip((df["test_mcc"].fillna(0) + 1) / 2, 0, 1)
    if "test_log_loss_score" in df:
        df["log_loss_score_component"] = np.clip(df["test_log_loss_score"].fillna(0), 0, 1)
    else:
        df["log_loss_score_component"] = 0.0

    if control_df is not None and not control_df.empty:
        c = control_df.groupby(["probe", "layer_index"])["control_test_macro_f1"].mean().rename("control_macro_f1")
        df = df.merge(c, on=["probe", "layer_index"], how="left")
        df["selectivity"] = np.clip(df["test_macro_f1"] - df["control_macro_f1"], -1, 1)
        df["selectivity_component"] = np.clip((df["selectivity"] / max(1.0 - chance, 1e-6)), 0, 1)
    else:
        df["control_macro_f1"] = np.nan
        df["selectivity"] = np.nan
        df["selectivity_component"] = 0.0

    stability = df.groupby(["probe", "layer_index"])["test_macro_f1"].transform("std").fillna(0)
    df["stability_component"] = np.clip(1.0 - stability, 0, 1)
    geometry = pd.to_numeric(df.get("geometry_silhouette", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    df["geometry_component"] = np.clip((geometry + 1.0) / 2.0, 0, 1)

    raw = np.zeros(len(df), dtype=float)
    comp_map = {
        "macro_f1": df["macro_f1_component"],
        "balanced_accuracy": df["balanced_accuracy_component"],
        "mcc": df["mcc_component"],
        "log_loss_score": df["log_loss_score_component"],
        "selectivity": df["selectivity_component"],
        "stability": df["stability_component"],
        "geometry": df["geometry_component"],
    }
    for key, weight in weights.items():
        raw += weight * comp_map.get(key, pd.Series(0.0, index=df.index)).to_numpy(dtype=float)

    complexity = [compute_complexity_penalty(p, d, cfg.complexity_penalty_scale) for p, d in zip(df["parameters"], df["input_dim"])]
    df["complexity_penalty"] = complexity
    df["probe_score_raw"] = np.clip(raw, 0, 1)
    df["probe_score"] = np.clip(df["probe_score_raw"] - df["complexity_penalty"], 0, 1)
    df["generalization_gap"] = df["train_macro_f1"] - df["test_macro_f1"]
    df["overfit_penalty"] = np.clip(df["generalization_gap"], 0, 1)
    return df


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def heatmap_image(matrix, path, title, xlabel, ylabel, fmt=".3f", vmin=None, vmax=None):
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(9, matrix.shape[1] * 0.8), max(5, matrix.shape[0] * 0.7)))
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.82)
    ax.set_xticks(np.arange(matrix.shape[1]), [str(x) for x in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]), [str(x) for x in matrix.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_layer_curves(df, output_dir):
    metrics = [
        ("test_macro_f1", "Test Macro-F1"),
        ("test_balanced_accuracy", "Test Balanced Accuracy"),
        ("probe_score", "Unified Probe Score"),
        ("selectivity", "True-label Selectivity"),
    ]
    for value, ylabel in metrics:
        plt.figure(figsize=(12, 6))
        for probe in sorted(df["probe"].unique()):
            sub = df[df["probe"] == probe].groupby("layer_index", as_index=False)[value].mean().sort_values("layer_index")
            if sub.empty:
                continue
            plt.plot(sub["layer_index"], sub[value], marker="o", linewidth=2, label=probe)
        plt.xlabel("Layer index")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} across hidden layers")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / f"layer_curve_{value}.png", dpi=240, bbox_inches="tight")
        plt.close()


def create_final_visuals(results_df, output_dir):
    if results_df.empty:
        return
    plot_layer_curves(results_df, output_dir)

    for metric, title, filename in [
        ("test_macro_f1", "Layer × probe test Macro-F1", "heatmap_test_macro_f1.png"),
        ("probe_score", "Layer × probe unified score", "heatmap_probe_score.png"),
        ("selectivity", "Layer × probe selectivity gap", "heatmap_selectivity.png"),
    ]:
        matrix = results_df.pivot_table(index="probe", columns="layer_index", values=metric, aggfunc="mean")
        heatmap_image(matrix, output_dir / filename, title, "Layer", "Probe", vmin=0 if metric != "selectivity" else None, vmax=1 if metric != "selectivity" else None)

    best = results_df.sort_values(["probe", "probe_score"], ascending=[True, False]).groupby("probe", as_index=False).first()
    final_matrix = best.set_index("probe")[["test_macro_f1", "test_balanced_accuracy", "test_mcc", "selectivity", "complexity_penalty", "probe_score"]].copy()
    heatmap_image(final_matrix, output_dir / "final_probe_score_heatmap.png", "Final best-layer probe measurement matrix", "Measurement", "Probe", vmin=0, vmax=1)

    grouped = results_df.groupby("probe").agg(mean_score=("probe_score", "mean"), std_score=("probe_score", "std")).reset_index()
    plt.figure(figsize=(11, 6))
    plt.bar(grouped["probe"], grouped["mean_score"], yerr=grouped["std_score"].fillna(0), capsize=5)
    plt.ylabel("Unified Probe Score")
    plt.xlabel("Probe")
    plt.title("Overall probe comparison across evaluated layers")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "final_probe_comparison.png", dpi=240, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, metric, title in [
        (axes[0, 0], "test_macro_f1", "Macro-F1"),
        (axes[0, 1], "test_balanced_accuracy", "Balanced Accuracy"),
        (axes[1, 0], "selectivity", "Selectivity"),
        (axes[1, 1], "probe_score", "Unified Score"),
    ]:
        matrix = results_df.pivot_table(index="probe", columns="layer_index", values=metric, aggfunc="mean")
        if matrix.empty:
            continue
        arr = matrix.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0 if metric != "selectivity" else None, vmax=1 if metric != "selectivity" else None)
        ax.set_title(title)
        ax.set_xticks(np.arange(matrix.shape[1]), [str(x) for x in matrix.columns])
        ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Unified Hidden-State Probe Dashboard", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "final_probe_dashboard.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    best.to_csv(output_dir / "final_best_layer_table.csv", index=False)


# -----------------------------------------------------------------------------
# Deterministic trial config and folder naming
# -----------------------------------------------------------------------------

def compute_computational_trial_config(config_dict: dict) -> dict:
    comp = copy.deepcopy(config_dict)
    comp["dataset_contract"].pop("allow_missing_label_fingerprint", None)
    for field in ["output_subdir", "verbose"]:
        comp["analysis"].pop(field, None)
    return comp


def build_trial_config(artifact, config):
    dataset_contract = asdict(config.dataset)
    dataset_contract.pop("allow_missing_label_fingerprint", None)

    ext = artifact.metadata.get("extraction", {})
    extraction_identity = {
        "pooling":       artifact.pooling,
        "max_length":    ext.get("max_length"),
        "batch_size":    ext.get("batch_size"),
        "storage_dtype": ext.get("storage_dtype"),
        "hidden_layers": artifact.hidden_layers,
        "hidden_size":   artifact.hidden_size,
    }

    return {
        "extraction":   extraction_identity,
        "model_name":   artifact.model_name,
        "dataset_name": artifact.dataset_name,
        "dataset_contract": dataset_contract,
        "probes":       [asdict(p) for p in config.probes],
        "split":        asdict(config.split),
        "analysis":     {
            "layers":                    config.layers,
            "repeats":                   config.repeats,
            "max_samples":               config.max_samples,
            "shuffled_label_control":    config.shuffled_label_control,
            "shuffled_control_repeats":  config.shuffled_control_repeats,
            "run_control_on_all_layers": config.run_control_on_all_layers,
            "pca_enabled":               config.pca_enabled,
            "pca_samples":               config.pca_samples,
            "silhouette_enabled":        config.silhouette_enabled,
            "silhouette_samples":        config.silhouette_samples,
            "enable_abstention":         config.enable_abstention,
            "enable_per_class_metrics":  config.enable_per_class_metrics,
            "enable_feature_statistics": config.enable_feature_statistics,
            "score_weights":             config.score_weights,
            "complexity_penalty_scale":  config.complexity_penalty_scale,
            "output_subdir":             config.output_subdir,
            "verbose":                   config.verbose,
        },
        "probe_version": SCRIPT_VERSION,
    }

def generate_trial_hash(config_dict: dict) -> str:
    return stable_hash(config_dict, length=12)

def build_probe_run_key(
    config_dict: dict,
    trial_hash: str,
    *,
    max_prefix_len: int = 140,
) -> str:
    """Human-readable + unique probe-run folder name.

    Format:
        <model>__<dataset>__<probes>__L<n_layers>__R<repeats>__S<max>__h<hash10>
    """
    ext = config_dict["extraction"]
    model  = ext["model_name"].replace("/", "-").replace("__", "-")
    dataset = ext["dataset_name"]
    probes  = "+".join(p["name"] for p in config_dict["probes"])
    layers  = ext.get("hidden_layers", "?")
    repeats = config_dict["analysis"]["repeats"]
    max_s   = config_dict["analysis"]["max_samples"] or "full"
    prefix  = f"{model}__{dataset}__{probes}__L{layers}__R{repeats}__S{max_s}"
    if len(prefix) > max_prefix_len:
        prefix = prefix[: max_prefix_len - 12]
    return f"{prefix}__h{trial_hash[:10]}"
def build_trial_dir_name(
    config_dict: dict,
    trial_hash: str,
    max_path_length: int = 200,
) -> str:
    ext = config_dict["extraction"]
    model_part = ext["model_name"].replace("/", "_")
    dataset_part = ext["dataset_name"]
    probes = "+".join([p["name"] for p in config_dict["probes"]])
    max_samples = config_dict["analysis"]["max_samples"] or "full"
    repeats = config_dict["analysis"]["repeats"]
    name = (
        f"probe_run__{model_part}__{dataset_part}"
        f"__max{max_samples}__rep{repeats}__probes={probes}__hash{trial_hash}"
    )
    if len(name) > max_path_length:
        prefix = name[: max_path_length - len(trial_hash) - 10]
        name = f"{prefix}...{trial_hash}"
    return name


def find_any_matching_run_dir(base_dir: Path, comp_hash: str) -> Path | None:
    """Return any run_dir whose trial_cfg hashes to comp_hash, complete or not."""
    for pattern in ("probe_run__*", "*__h*"):
        for run_dir in base_dir.glob(pattern):
            if not run_dir.is_dir():
                continue
            # Prefer the completed metadata, fall back to the progress file.
            meta_path = run_dir / "complete_run_metadata.json"
            cfg: dict | None = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    cfg = meta.get("extra_info", {}).get("trial_config")
                except Exception:
                    cfg = None
            if cfg is None:
                prog = run_dir / "progress.json"
                if prog.exists():
                    # No trial_cfg in progress.json — skip, can't compare.
                    continue
            if cfg is None:
                continue
            stored = generate_trial_hash(compute_computational_trial_config(cfg))
            if stored == comp_hash:
                return run_dir
    return None


# -----------------------------------------------------------------------------
# Analyzer
# -----------------------------------------------------------------------------

class UnifiedProbeAnalyzer:
    def __init__(
        self,
        artifact: ExtractionArtifact,
        config: AnalysisConfig,
        dataset_df: pd.DataFrame | Any | None = None,
    ):
        config.validate_verbose()
        self.artifact = artifact
        self.config   = config
        self.device   = choose_device()
        self.logger   = ProbeLogger(config.verbose)

        self.logger.section("INITIALISING UNIFIED HIDDEN-STATE PROBE", 1)

        self.df = to_dataframe(dataset_df) if dataset_df is not None \
                  else load_dataframe(config.dataset)
        if len(self.df) != artifact.sample_count:
            raise RuntimeError(
                f"Dataset rows={len(self.df)} != hidden-state samples="
                f"{artifact.sample_count}. Hard alignment failure."
            )

        self.y, self.classes, self.target_meta = build_targets(self.df, config.dataset)
        self.task_type          = self.target_meta["task_type"]
        self.target_validation  = validate_targets(self.y, self.classes, self.task_type)
        self.text_alignment     = validate_text_alignment(artifact, self.df, config.dataset)
        self.label_alignment    = validate_label_alignment(
            artifact, self.df, config.dataset, self.y, self.classes
        )
        self._row_perm, self._alignment_mode = self._strict_text_join()

        # build_targets returns labels in CSV order. Reindex to hidden_state order.
        y_csv, classes, meta = build_targets(self.df, self.config.dataset)
        self.y          = y_csv[self._row_perm]
        self.classes    = classes
        self.target_meta = meta

        if self.text_alignment.get("verified") and not self.label_alignment.get("verified"):
            self.label_alignment["verification_basis"] = (
                "Label row order inherits verification from the cryptographically "
                "matched text sequence in the same dataframe."
            )

        self.layers = self._resolve_layers(config.layers)

        # ── Output root: PROBE_ROOT mirrors HIDDEN_STATES_ROOT exactly. ──
        trial_cfg   = build_trial_config(artifact=artifact, config=config)
        trial_hash  = generate_trial_hash(trial_cfg)
        folder_name = build_probe_run_key(trial_cfg, trial_hash)

        base_output = probe_dir_for(artifact.model_name, artifact.dataset_name)
        base_output.mkdir(parents=True, exist_ok=True)

        output_dir = base_output / folder_name

        # Recover a prior run whose computational hash matches, even if the
        # folder name differs (e.g. different SCRIPT_VERSION string).
        if not output_dir.exists():
            comp_hash = generate_trial_hash(compute_computational_trial_config(trial_cfg))
            existing  = find_matching_run_dir(base_output, comp_hash)
            if existing is None:
                existing = find_any_matching_run_dir(base_output, comp_hash)
            if existing is not None:
                self.logger.emit(f"Found existing run with matching config: {existing}")
                existing.rename(output_dir)

        if output_dir.exists() and (output_dir / "completion.json").exists():
            self.logger.emit(f"Trial already completed: {output_dir}", 1)
            self.skip_run = True
        else:
            self.logger.emit(f"Starting new trial: {output_dir}", 1)
            safe_relative_output(PROBE_ROOT, output_dir)   # keep the sandbox guard
            self.skip_run = False

        self.output_dir = output_dir
        self.trial_config = trial_cfg
        self.trial_hash   = trial_hash

        self._preflight()

        self.logger.emit(f"Model: {artifact.model_name}", 1)
        self.logger.emit(f"Dataset artifact: {artifact.dataset_name}", 1)
        self.logger.emit(f"Hidden-state shape: {tuple(artifact.states.shape)}", 1)
        self.logger.emit(f"Task type: {self.task_type} | classes: {len(self.classes)}", 1)
        self.logger.emit(f"Selected layers: {len(self.layers)} | device: {self.device}", 1)
        self.logger.emit(
            f"Alignment: text={self.text_alignment['status']} | labels={self.label_alignment['status']}",
            1,
        )
        
    def _update_probe_index(self) -> None:
        """Refresh <dataset>/probe/index.json with a registry of all runs.

        The analyser reads this single file to discover probe results without
        walking the whole tree.
        """
        index_path = self.output_dir.parent / "index.json"
        entry = {
            "run_key":        self.output_dir.name,
            "trial_hash":     self.trial_hash,
            "model":          self.artifact.model_name,
            "dataset":        self.artifact.dataset_name,
            "probes":         [p.name for p in self.config.probes],
            "repeats":        self.config.repeats,
            "max_samples":    self.config.max_samples,
            "layers":         len(self.layers),
            "task_type":      self.task_type,
            "n_classes":      len(self.classes),
            "completed_at":   time.time(),
            "results_csv":    str((self.output_dir / "layer_probe_results.csv").relative_to(index_path.parent)),
            "best_csv":       str((self.output_dir / "final_probe_score_matrix.csv").relative_to(index_path.parent)),
            "metadata_json":  str((self.output_dir / "complete_run_metadata.json").relative_to(index_path.parent)),
        }
        current = {"runs": []}
        if index_path.exists():
            try:
                current = json.loads(index_path.read_text())
            except Exception:
                pass

        current["runs"] = [r for r in current.get("runs", [])
                        if r.get("run_key") != entry["run_key"]]
        current["runs"].append(entry)
        current["updated_at"] = time.time()

        tmp = index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2, default=str))
        os.replace(tmp, index_path)
        
    def _resolve_layers(self, requested):
        available = [f"layer_{i}" for i in range(self.artifact.hidden_layers)]
        if requested == "all":
            return available
        if not isinstance(requested, list) or not requested:
            raise ValueError("layers must be 'all' or a non-empty list")
        out = []
        for x in requested:
            name = f"layer_{x}" if isinstance(x, int) else str(x)
            if name not in available:
                raise ValueError(f"Requested {name} is unavailable. Available={available}")
            out.append(name)
        return sorted(set(out), key=parse_layer_number)

    import hashlib
    from collections import defaultdict

    def _strict_text_join(self) -> tuple[np.ndarray, str]:
        """
        Returns (perm, mode):
            hidden_states[j]  ↔  CSV row perm[j]

        Raises RuntimeError if a bijection cannot be established.
        Never falls back to positional indexing on failure.
        """
        if self.artifact.text_hashes is None:
            raise RuntimeError(
                "text_hashes.npy is missing. This extraction predates the "
                "alignment guarantee. Re-run extraction for this (model, dataset)."
            )

        text_col = self.text_alignment["text_column"]
        csv_texts = one_dim_strings(self.df[text_col].tolist())

        csv_hashes = np.empty((len(csv_texts), 32), dtype=np.uint8)
        for i, t in enumerate(csv_texts):
            csv_hashes[i] = np.frombuffer(
                hashlib.sha256(str(t).encode("utf-8")).digest(), dtype=np.uint8
            )

        stored = np.asarray(self.artifact.text_hashes)
        if stored.shape != csv_hashes.shape:
            raise RuntimeError(
                f"Row-count mismatch: hidden_states has {stored.shape[0]} rows, "
                f"CSV has {csv_hashes.shape[0]}. The CSV has drifted since extraction."
            )

        # Fast path: perfect positional identity. 99% of runs take this.
        if np.array_equal(stored, csv_hashes):
            return np.arange(len(stored), dtype=np.int64), "positional_identity"

        # Slow path: content-addressed join. Same complexity, O(N) once.
        csv_index: dict[bytes, list[int]] = defaultdict(list)
        for i in range(len(csv_hashes)):
            csv_index[csv_hashes[i].tobytes()].append(i)

        perm = np.empty(len(stored), dtype=np.int64)
        for j in range(len(stored)):
            key = stored[j].tobytes()
            candidates = csv_index.get(key, [])
            if len(candidates) != 1:
                raise RuntimeError(
                    f"hidden_states row {j} has a text hash that appears in "
                    f"{len(candidates)} CSV rows. Cannot establish a bijection. "
                    f"Refusing to probe — this would silently mis-align labels."
                )
            perm[j] = candidates[0]

        if len(set(perm.tolist())) != len(perm):
            raise RuntimeError(
                "Two hidden_state rows matched the same CSV row. "
                "The CSV contains duplicates that extraction did not see."
            )

        return perm, "hash_join"
    def _preflight(self):
        
        self.config.split.validate()
        for p in self.config.probes:
            validate_probe_spec(p, self.task_type)
        if len(self.classes) < 2:
            raise RuntimeError("Cannot train a probe with fewer than two classes")
        if self.artifact.hidden_size < 2:
            raise RuntimeError("Representation width D must be >=2")

        check_idx = sample_indices(
            self.artifact.sample_count,
            min(32, self.artifact.sample_count),
            self.config.split.seed,
        )
        for layer_name in self.layers:
            layer_idx = parse_layer_number(layer_name)
            X = np.asarray(self.artifact.states[check_idx, layer_idx, :], dtype=np.float32)
            if not np.isfinite(X).all():
                raise RuntimeError(f"Preflight found NaN/Inf in {layer_name}")
            if float(np.var(X)) == 0.0:
                raise RuntimeError(f"Preflight found a constant representation in {layer_name}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_run_manifest(self):
        manifest = {
            "script_version": SCRIPT_VERSION,
            "created_at": time.time(),
            "artifact": self.artifact.analysis_summary(),
            "dataset_contract": asdict(self.config.dataset),
            "target_metadata": self.target_meta,
            "target_validation": self.target_validation,
            "text_alignment": self.text_alignment,
            "label_alignment": self.label_alignment,
            "classes": list(self.classes),
            "label_entropy_bits": label_entropy(self.y, self.task_type),
            "probes": [asdict(p) for p in self.config.probes],
            "layers": self.layers,
            "split": asdict(self.config.split),
            "repeats": self.config.repeats,
            "max_samples": self.config.max_samples,
            "device": self.device,
            "analysis": asdict(self.config),
        }
        save_json(self.output_dir / "probe_run_manifest.json", manifest)
        save_json(self.output_dir / "dataset_validation_report.json", {
            "artifact_validation": self.artifact.validation,
            "target_validation": self.target_validation,
            "text_alignment": self.text_alignment,
            "label_alignment": self.label_alignment,
            "target_metadata": self.target_meta,
        })

        alignment_record = {
            "sample_count": int(len(self.df)),
            "text_column": self.text_alignment["text_column"],
            "label_column": self.target_meta.get("raw_label_column"),
            "text_sequence_fingerprint": self.text_alignment["observed"]["derived_fingerprint"],
            "text_head_hash": self.text_alignment["observed"]["head_hash"],
            "text_tail_hash": self.text_alignment["observed"]["tail_hash"],
            "canonical_label_fingerprint": self.label_alignment["label_fingerprint"],
            "artifact_text_provenance_status": self.text_alignment["status"],
            "artifact_label_provenance_status": self.label_alignment["status"],
            "row_position_hash": stable_hash(list(range(len(self.df))), 24),
            "has_sample_ids": self.artifact.sample_ids is not None,
            "sample_ids_match": self.text_alignment.get("sample_ids_match"),
            "warning": (
                "This is a probe-time manifest. For strongest provenance, create the same "
                "manifest at extraction time and store it with the hidden states."
            ),
        }
        save_json(self.output_dir / "probe_alignment_manifest.json", alignment_record)

    def _prepare_population(self, seed):
        return (
            np.arange(len(self.y), dtype=np.int64)
            if self.config.max_samples is None
            else sample_indices(len(self.y), self.config.max_samples, seed)
        )

    def _split(self, selected, seed):
        local_y = self.y[selected]
        result = (
            make_single_splits(local_y, self.config.split, seed)
            if self.task_type == "single_label"
            else make_multilabel_splits(local_y, self.config.split, seed)
        )
        return {
            k: selected[v]
            for k, v in result.items()
            if k in {"train", "validation", "test"}
        }

    def _load_population_layer(self, layer_idx, selected):
        X = np.asarray(self.artifact.states[selected, layer_idx, :], dtype=np.float32)
        if not np.isfinite(X).all():
            raise RuntimeError(f"Layer {layer_idx} contains NaN/Inf in selected rows")
        return X

    def _metric_fields(self, result, split_name):
        m = result[split_name]
        if self.task_type == "single_label":
            keys = [
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "macro_precision", "macro_recall", "mcc", "cohen_kappa",
                "log_loss", "log_loss_score", "roc_auc_ovr_macro", "average_precision_macro",
            ]
        else:
            keys = [
                "exact_match_accuracy", "micro_f1", "macro_f1", "weighted_f1",
                "micro_precision", "micro_recall", "macro_precision", "macro_recall",
                "balanced_accuracy", "mcc", "hamming_loss", "hamming_score",
                "macro_jaccard", "log_loss", "log_loss_score", "roc_auc_macro",
                "average_precision_macro", "labels_with_positive_support",
                "labels_with_negative_support", "labels_with_both_support",
            ]
        return {f"{split_name}_{k}": m.get(k) for k in keys}

    def _save_probe_artifacts(self, probe, layer_name, repeat, results, model, scaler, record):
        d = self.output_dir / "models" / probe.name / layer_name / f"repeat_{repeat}"
        d.mkdir(parents=True, exist_ok=True)
        save_json(d / "metrics.json", {"record": record, "results": results})
        if self.task_type == "single_label":
            save_npz(d / "confusion_matrix_test.npz", matrix=np.asarray(results["test"]["confusion_matrix"]))
        
        try:
            if probe.type == "logistic":
                joblib.dump(model, d / "probe.joblib")
            else:
                torch.save(model.state_dict(), d / "probe_state_dict.pt")
            if scaler is not None:
                joblib.dump(scaler, d / "scaler.joblib")
        except Exception as e:
            self.logger.emit(f"Warning: could not save model artifact for {probe.name} layer {layer_name}: {e}", 1)
                
        

    def _job_key(self, repeat, layer_idx, probe_name, control_index=-1):
        return (repeat, layer_idx, probe_name, control_index)

    def _save_progress(self, completed_jobs, main_records, control_records, split_archive):
        path = self.output_dir / 'progress.json'
        backup_path = self.output_dir / 'progress.json.bak'
        tmp_name = f"progress_{os.getpid()}.tmp"
        tmp_path = self.output_dir / tmp_name

        payload = {
            'completed_jobs': sorted(list(completed_jobs)),
            'main_records': main_records,
            'control_records': control_records,
            'split_archive': {k: v.tolist() for k, v in split_archive.items()},
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode()
        checksum = hashlib.sha256(serialized).hexdigest()
        payload['checksum'] = checksum

        try:
            with open(tmp_path, 'w') as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

            if backup_path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    stored_checksum = data.get('checksum')
                    payload_part = {
                        'completed_jobs': data.get('completed_jobs', []),
                        'main_records': data.get('main_records', []),
                        'control_records': data.get('control_records', []),
                        'split_archive': data.get('split_archive', {}),
                    }
                    computed = hashlib.sha256(
                        json.dumps(payload_part, sort_keys=True, default=str).encode()
                    ).hexdigest()
                    if stored_checksum == computed:
                        shutil.copy2(path, backup_path)
                except Exception:
                    pass
            else:
                shutil.copy2(path, backup_path)

        except Exception as e:
            self.logger.emit(f"Warning: could not save progress file: {e}", 1)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _load_progress(self):
        path = self.output_dir / 'progress.json'
        backup_path = self.output_dir / 'progress.json.bak'

        for candidate in (path, backup_path):
            if not candidate.exists():
                continue
            try:
                with open(candidate, 'r') as f:
                    data = json.load(f)

                stored_checksum = data.get('checksum')
                if stored_checksum:
                    payload_part = {
                        'completed_jobs': data.get('completed_jobs', []),
                        'main_records': data.get('main_records', []),
                        'control_records': data.get('control_records', []),
                        'split_archive': data.get('split_archive', {}),
                    }
                    serialized = json.dumps(payload_part, sort_keys=True, default=str).encode()
                    computed = hashlib.sha256(serialized).hexdigest()
                    if computed != stored_checksum:
                        self.logger.emit(f"Checksum mismatch in {candidate.name}, trying backup.", 1)
                        continue
                else:
                    self.logger.emit(f"Progress file {candidate.name} has no checksum; assuming valid.", 2)

                completed_jobs = set(tuple(item) for item in data.get('completed_jobs', []))
                main_records = data.get('main_records', [])
                control_records = data.get('control_records', [])
                split_archive = data.get('split_archive', {})
                return completed_jobs, main_records, control_records, split_archive

            except Exception as e:
                self.logger.emit(f"Failed to load {candidate.name}: {e}", 1)
                continue

        self.logger.emit("No valid progress file found, starting fresh.", 1)
        return set(), [], [], {}

    def run(self):
        if self.skip_run:
            self.logger.emit("Skipping execution, loading existing results.", 1)
            results_path = self.output_dir / "layer_probe_results.csv"
            best_path = self.output_dir / "final_probe_score_matrix.csv"
            if results_path.exists() and best_path.exists():
                scored = pd.read_csv(results_path)
                best = pd.read_csv(best_path)
                self.logger.emit(f"Loaded results from {self.output_dir}", 1)
                return scored, best
            else:
                self.logger.emit("Completion marker found but result files missing. Re-running.", 1)
                self.skip_run = False

        completed_jobs, main_records, control_records, split_archive = self._load_progress()
        self.logger.emit(f"Resumed with {len(completed_jobs)} completed jobs.", 1)

        n_main_jobs = self.config.repeats * len(self.layers) * len(self.config.probes)
        n_control_jobs = 0
        if self.config.shuffled_label_control:
            n_control_jobs = self.config.repeats * len(self.layers) * len(self.config.probes) * self.config.shuffled_control_repeats
        total_jobs = n_main_jobs + n_control_jobs

        # If all jobs already done but finalization missing, go straight to finalization
        if len(completed_jobs) == total_jobs:
            self.logger.emit("All jobs already completed. Proceeding to finalization.", 1)
            # We need split_archive; if missing, recompute it
            if not split_archive:
                for repeat in range(self.config.repeats):
                    seed = self.config.split.seed + repeat
                    selected = self._prepare_population(seed)
                    split = self._split(selected, seed)
                    for name, idx in split.items():
                        split_archive[f"repeat_{repeat}_{name}"] = idx
            # Build DataFrames from loaded records
            results_df = pd.DataFrame(main_records)
            control_df = pd.DataFrame(control_records) if control_records else pd.DataFrame()
            self._finalize(results_df, control_df, split_archive)
            return results_df, self._best_df

        # Otherwise proceed with fitting missing jobs
        pbar = tqdm(total=total_jobs, desc="Probing", unit="fit", disable=(self.config.verbose < 0))
        pbar.update(len(completed_jobs))

        error_log_path = self.output_dir / "fit_errors.log"
        errors = []

        self.write_run_manifest()

        self.logger.section("PROBING EXPERIMENT", 1)
        self.logger.emit(
            "Question: how recoverable is the target from each frozen hidden-state layer?", 1
        )
        self.logger.emit(
            f"repeats={self.config.repeats} | max_samples={self.config.max_samples} | "
            f"layers={len(self.layers)} | probes={len(self.config.probes)}", 1
        )

        for repeat in range(self.config.repeats):
            seed = self.config.split.seed + repeat
            selected = self._prepare_population(seed)
            split = self._split(selected, seed)
            # Store split indices in archive (if not already there)
            for name, idx in split.items():
                split_archive[f"repeat_{repeat}_{name}"] = idx

            y_train = self.y[split["train"]]
            y_val = self.y[split["validation"]]
            y_test = self.y[split["test"]]
            baseline = majority_baseline(y_train, y_test, self.classes) if self.task_type == "single_label" else None

            positions = {int(global_i): i for i, global_i in enumerate(selected)}
            tr_local = np.asarray([positions[int(i)] for i in split["train"]], dtype=np.int64)
            va_local = np.asarray([positions[int(i)] for i in split["validation"]], dtype=np.int64)
            te_local = np.asarray([positions[int(i)] for i in split["test"]], dtype=np.int64)

            for layer_name in self.layers:
                layer_idx = parse_layer_number(layer_name)
                relative_depth = layer_idx / (self.artifact.hidden_layers - 1) if self.artifact.hidden_layers > 1 else 0.0

                # Load data for this layer
                X_population = self._load_population_layer(layer_idx, selected)

                geom_path = self.output_dir / "geometry" / f"{layer_name}_repeat_{repeat}.json"
                if geom_path.exists():
                    geom = json.loads(geom_path.read_text())
                else:
                    geom_count = min(len(selected), max(self.config.pca_samples, self.config.silhouette_samples))
                    geom_local = sample_indices(len(selected), geom_count, seed + layer_idx)
                    geom = geometry_analysis(
                        X_population[geom_local], self.y[selected][geom_local],
                        self.classes, self.task_type, seed + layer_idx, self.config
                    )
                    save_json(geom_path, geom)

                Xtr_raw = X_population[tr_local]
                Xv_raw = X_population[va_local]
                Xte_raw = X_population[te_local]

                if any(p.standardize for p in self.config.probes):
                    shared_scaler = StandardScaler().fit(Xtr_raw)
                    scaled_cache = (
                        shared_scaler.transform(Xtr_raw).astype(np.float32),
                        shared_scaler.transform(Xv_raw).astype(np.float32),
                        shared_scaler.transform(Xte_raw).astype(np.float32),
                    )
                else:
                    shared_scaler = None
                    scaled_cache = None

                for probe in self.config.probes:
                    # Main fit
                    main_key = self._job_key(repeat, layer_idx, probe.name, -1)
                    if main_key not in completed_jobs:
                        probe_seed = seed + stable_int(probe.name) + layer_idx * 997
                        if probe.standardize:
                            Xtr, Xv, Xte = scaled_cache
                            scaler_for_artifact = shared_scaler
                        else:
                            Xtr, Xv, Xte = Xtr_raw, Xv_raw, Xte_raw
                            scaler_for_artifact = None

                        self.logger.emit(
                            f"FIT {probe.name} | layer={layer_idx} | complexity={probe.complexity} | seed={probe_seed}",
                            3,
                        )
                        try:
                            results, model = fit_probe(
                                probe, Xtr, y_train, Xv, y_val, Xte, y_test,
                                self.classes, self.task_type, probe_seed, self.device, self.config.enable_per_class_metrics,
                            )
                            record = {
                                "repeat": repeat,
                                "seed": probe_seed,
                                "layer": layer_name,
                                "layer_index": layer_idx,
                                "relative_layer_depth": relative_depth,
                                "probe": probe.name,
                                "probe_type": probe.type,
                                "probe_complexity": probe.complexity,
                                "task_type": self.task_type,
                                "input_dim": int(X_population.shape[1]),
                                "hidden_layers_total": int(self.artifact.hidden_layers),
                                "class_count": len(self.classes),
                                "train_n": int(len(tr_local)),
                                "validation_n": int(len(va_local)),
                                "test_n": int(len(te_local)),
                                "parameters": results.get("parameters"),
                                "resolved_hidden_dims": results.get("resolved_hidden_dims", []),
                                "epochs_completed": results.get("epochs_completed"),
                                "best_validation_score": results.get("best_validation_score"),
                                "geometry_silhouette": geom.get("silhouette_score"),
                                "geometry_pca_2d_variance": geom.get("pca_2d_variance"),
                                "baseline_test_macro_f1": baseline["test"]["macro_f1"] if baseline else None,
                            }
                            record.update(self._metric_fields(results, "train"))
                            record.update(self._metric_fields(results, "validation"))
                            record.update(self._metric_fields(results, "test"))

                            self._save_probe_artifacts(probe, layer_name, repeat, results, model, scaler_for_artifact, record)
                            main_records.append(record)
                            completed_jobs.add(main_key)
                            self._save_progress(completed_jobs, main_records, control_records, split_archive)
                            pbar.update(1)

                            if self.config.verbose >= 3:
                                test = results["test"]
                                self.logger.emit(
                                    f"TEST Macro-F1={test.get('macro_f1')} | "
                                    f"BalancedAcc={test.get('balanced_accuracy')} | MCC={test.get('mcc')}",
                                    3,
                                )
                                if self.task_type == "multi_label":
                                    self.logger.emit(
                                        f"TEST label coverage: positive={test.get('labels_with_positive_support')} | "
                                        f"both_classes={test.get('labels_with_both_support')} | "
                                        f"ROC-AUC={test.get('roc_auc_macro')} | AP={test.get('average_precision_macro')}",
                                        3,
                                    )

                        except Exception as e:
                            error_msg = f"Main fit failed: repeat={repeat}, layer={layer_idx}, probe={probe.name}: {type(e).__name__}: {e}"
                            self.logger.emit(error_msg, 1)
                            errors.append(error_msg)
                            with open(error_log_path, "a") as f:
                                f.write(f"{time.time()}: {error_msg}\n")
                            continue

                    # Control fits
                    if self.config.shuffled_label_control:
                        local_y = self.y[selected].copy()
                        for control_repeat in range(self.config.shuffled_control_repeats):
                            ctrl_key = self._job_key(repeat, layer_idx, probe.name, control_repeat)
                            if ctrl_key in completed_jobs:
                                continue

                            ctrl_seed = (
                                self.config.split.seed
                                + 1_000_000
                                + repeat * 10_000
                                + layer_idx * 100
                                + control_repeat
                                + stable_int(probe.name)
                            )
                            rng = np.random.default_rng(ctrl_seed)
                            shuffled_y = local_y.copy()
                            rng.shuffle(shuffled_y, axis=0)

                            y_train_ctrl = shuffled_y[tr_local]
                            y_val_ctrl = shuffled_y[va_local]
                            y_test_ctrl = shuffled_y[te_local]

                            if probe.standardize:
                                Xtr, Xv, Xte = scaled_cache
                            else:
                                Xtr, Xv, Xte = Xtr_raw, Xv_raw, Xte_raw

                            self.logger.emit(
                                f"FIT CONTROL {probe.name} | layer={layer_idx} | ctrl={control_repeat} | seed={ctrl_seed}",
                                3,
                            )
                            try:
                                results_ctrl, _ = fit_probe(
                                    probe, Xtr, y_train_ctrl, Xv, y_val_ctrl, Xte, y_test_ctrl,
                                    self.classes, self.task_type, ctrl_seed, self.device, self.config.enable_per_class_metrics,
                                )
                                ctrl_record = {
                                    "repeat": repeat,
                                    "control_repeat": control_repeat,
                                    "seed": ctrl_seed,
                                    "probe": probe.name,
                                    "layer_index": layer_idx,
                                    "control_test_macro_f1": results_ctrl["test"].get("macro_f1"),
                                    "control_test_accuracy": results_ctrl["test"].get("accuracy", results_ctrl["test"].get("exact_match_accuracy")),
                                    "control_test_mcc": results_ctrl["test"].get("mcc"),
                                }
                                control_records.append(ctrl_record)
                                completed_jobs.add(ctrl_key)
                                self._save_progress(completed_jobs, main_records, control_records, split_archive)
                                pbar.update(1)

                            except Exception as e:
                                error_msg = f"Control fit failed: repeat={repeat}, layer={layer_idx}, probe={probe.name}, ctrl={control_repeat}: {type(e).__name__}: {e}"
                                self.logger.emit(error_msg, 1)
                                errors.append(error_msg)
                                with open(error_log_path, "a") as f:
                                    f.write(f"{time.time()}: {error_msg}\n")
                                continue

        pbar.close()

        # Check if all jobs completed after this run
        if len(completed_jobs) == total_jobs:
            self.logger.emit("All jobs completed. Generating final outputs.", 1)
            results_df = pd.DataFrame(main_records)
            control_df = pd.DataFrame(control_records) if control_records else pd.DataFrame()
            self._finalize(results_df, control_df, split_archive)
            return results_df, self._best_df
        else:
            self.logger.emit(f"Run incomplete: {len(completed_jobs)}/{total_jobs} jobs completed. Progress saved.", 1)
            if errors:
                self.logger.emit(f"{len(errors)} fit errors were logged.", 1)
            raise RuntimeError(f"Trial incomplete after {len(completed_jobs)}/{total_jobs} fits. Progress saved; rerun to continue.")

    def _finalize(self, results_df, control_df, split_archive):
        """Generate all final outputs after all jobs are complete."""
        # Save split indices
        save_npz(self.output_dir / "split_indices.npz", **split_archive)

        if not control_df.empty:
            control_df.to_csv(self.output_dir / "shuffled_label_controls.csv", index=False)

        scored = add_score_columns(results_df, control_df if not control_df.empty else None, self.config, self.task_type)
        aggregate = scored.groupby(["probe", "probe_type", "probe_complexity", "layer_index"], as_index=False).agg(
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
            test_mcc_mean=("test_mcc", "mean"),
            selectivity_mean=("selectivity", "mean"),
            probe_score_mean=("probe_score", "mean"),
            probe_score_std=("probe_score", "std"),
            parameters=("parameters", "first"),
            relative_layer_depth=("relative_layer_depth", "first"),
        )
        best = aggregate.sort_values(["probe", "probe_score_mean"], ascending=[True, False]).groupby("probe", as_index=False).first()

        scored.to_csv(self.output_dir / "layer_probe_results.csv", index=False)
        aggregate.to_csv(self.output_dir / "layer_probe_aggregate_results.csv", index=False)
        best.to_csv(self.output_dir / "final_probe_score_matrix.csv", index=False)
        create_final_visuals(scored, self.output_dir)

        metadata_path = save_complete_run_metadata(
            self, scored, best, control_df,
            extra_info={"trial_config": self.trial_config, "trial_hash": self.trial_hash}
        )
        self.logger.emit(f"Complete run metadata saved: {metadata_path}", 1)

        summary = {
            "trial_hash": self.trial_hash,
            "probe_score_mean": float(scored["probe_score"].mean()) if not scored.empty else None,
            "test_macro_f1_mean": float(scored["test_macro_f1"].mean()) if not scored.empty else None,
            "test_balanced_accuracy_mean": float(scored["test_balanced_accuracy"].mean()) if not scored.empty else None,
            "best_per_probe": best.to_dict("records") if not best.empty else [],
            "control_mean_macro_f1": float(control_df["control_test_macro_f1"].mean()) if not control_df.empty else None,
            "output_dir": str(self.output_dir),
        }
        save_json(self.output_dir / "summary.json", summary)
        save_json(self.output_dir / "completion.json", {"status": "complete", "finished_at": time.time()})

        for p in [self.output_dir / 'progress.json', self.output_dir / 'progress.json.bak']:
            if p.exists():
                p.unlink()

        self.logger.section("FINAL RESULT", 1)
        if not best.empty:
            cols = [c for c in ["probe", "layer_index", "probe_score_mean", "test_macro_f1_mean"] if c in best.columns]
            self.logger.emit("Final best layer table:", 1)
            if self.config.verbose >= 1:
                print(best[cols].to_string(index=False))
        self.logger.emit(f"Output directory: {self.output_dir}", 1)

        # Store best for return
        self._best_df = best
        self._scored_df = scored
        self._update_probe_index()


# -----------------------------------------------------------------------------
# save_complete_run_metadata
# -----------------------------------------------------------------------------

def save_complete_run_metadata(analyzer, results_df, best_df, control_df=None, extra_info=None):
    output_dir = analyzer.output_dir
    metadata_path = output_dir / "complete_run_metadata.json"

    config_dict = {
        "script_version": SCRIPT_VERSION,
        "created_at": time.time(),
        "dataset_contract": asdict(analyzer.config.dataset),
        "probes": [asdict(p) for p in analyzer.config.probes],
        "split": asdict(analyzer.config.split),
        "repeats": analyzer.config.repeats,
        "max_samples": analyzer.config.max_samples,
        "layers": analyzer.layers,
        "analysis": asdict(analyzer.config),
        "score_weights": _normalise_weights(analyzer.config.score_weights),
        "device": analyzer.device,
    }

    artifact_summary = analyzer.artifact.analysis_summary()
    artifact_summary.pop("provenance", None)

    target_info = {
        "target_metadata": analyzer.target_meta,
        "target_validation": analyzer.target_validation,
        "text_alignment": analyzer.text_alignment,
        "label_alignment": analyzer.label_alignment,
        "classes": analyzer.classes,
        "label_entropy_bits": label_entropy(analyzer.y, analyzer.task_type),
    }

    environment = get_environment_info()

    results_summary = {}
    if not results_df.empty:
        results_summary = {
            "rows": len(results_df),
            "columns": list(results_df.columns),
            "best_per_probe": best_df.to_dict("records") if best_df is not None else [],
            "layer_wise_metrics": {
                "test_macro_f1_by_layer": results_df.groupby("layer_index")["test_macro_f1"].mean().to_dict(),
                "probe_score_by_layer": results_df.groupby("layer_index")["probe_score"].mean().to_dict(),
            },
        }

    control_summary = None
    if control_df is not None and not control_df.empty:
        control_summary = {
            "rows": len(control_df),
            "mean_control_macro_f1": float(control_df["control_test_macro_f1"].mean()),
            "by_layer": control_df.groupby("layer_index")["control_test_macro_f1"].mean().to_dict(),
        }

    extra = extra_info or {}
    extra["computational_hash"] = generate_trial_hash(
        compute_computational_trial_config(analyzer.trial_config)
    )
    full_metadata = {
        "run_id": output_dir.name,
        "output_directory": str(output_dir),
        "configuration": config_dict,
        "artifact": artifact_summary,
        "target": target_info,
        "environment": environment,
        "results": results_summary,
        "controls": control_summary,
        "extra_info": extra,
    }

    save_json(metadata_path, full_metadata)
    return metadata_path


# -----------------------------------------------------------------------------
# Matrix runner
# -----------------------------------------------------------------------------

def _results_index_path(checkpoint_dir):
    return checkpoint_dir / "results_index.csv"


def update_results_index(
    checkpoint_dir: Path,
    result_csv: Path,
    model_name: str,
    dataset_name: str,
    trial_hash: str,
) -> None:
    index_path = _results_index_path(checkpoint_dir)
    row = {
        "result_filename": result_csv.name,
        "model": model_name,
        "dataset": dataset_name,
        "trial_hash": trial_hash,
        "saved_at": time.time(),
    }
    if index_path.exists():
        df = pd.read_csv(index_path)
        df = df[df["result_filename"] != result_csv.name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(index_path, index=False)


def load_results_index(checkpoint_dir):
    index_path = _results_index_path(checkpoint_dir)
    if index_path.exists():
        return pd.read_csv(index_path)
    return pd.DataFrame(columns=["result_filename", "model", "dataset", "saved_at"])


def lookup_result_by_hash(checkpoint_dir: Path, hash_or_filename: str) -> dict:
    index = load_results_index(checkpoint_dir)
    if not hash_or_filename.endswith(".csv"):
        hash_prefix = hash_or_filename
    else:
        hash_prefix = hash_or_filename.replace("_layer_probe_results.csv", "")
    match = index[index["result_filename"].str.startswith(hash_prefix)]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    result_path = checkpoint_dir / "per_entry_results" / row["result_filename"]
    row["result_path"] = str(result_path)
    return row


def validate_checkpoint_consistency(checkpoint_dir: Path, verbose: bool = True) -> bool:
    checkpoint_file = checkpoint_dir / "probe_matrix_checkpoint.json"
    results_subdir  = checkpoint_dir / "per_entry_results"
    if not checkpoint_file.exists():
        if verbose:
            print("[validate] Checkpoint file not found.")
        return True

    checkpoint = json.loads(checkpoint_file.read_text())
    completed  = checkpoint.get("completed", {})
    inconsistent: list[tuple[str, str]] = []

    for key, info in completed.items():
        trial_hash = info.get("trial_hash")
        stored_comp = info.get("comp_hash")
        expected_file = results_subdir / f"{trial_hash}_layer_probe_results.csv"
        if not expected_file.exists():
            inconsistent.append((key, "missing result CSV"))
            continue
        if stored_comp is None:
            # Old checkpoints written before comp_hash existed. Skip rather
            # than flag; run_matrix will re-run them on the next pass.
            continue
        # Nothing to recompute here — comp_hash is authoritative and stored.
        # We only verify file presence above.

    if inconsistent:
        if verbose:
            print("[validate] Inconsistencies found:")
            for k, reason in inconsistent:
                print(f"  - {k}: {reason}")
        return False
    if verbose:
        print("[validate] Checkpoint is consistent.")
    return True


def run_matrix(
    entries: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    probes: Sequence[ProbeSpec],
    split: SplitConfig | None = None,
    repeats: int = 3,
    max_samples: int | None = 5000,
    verbose: int = 0,
    checkpoint_dir: Path | None = None,
    shuffled_label_control: bool = True,
    shuffled_control_repeats: int = 3,
) -> pd.DataFrame:
    split = split or SplitConfig(train=0.80, validation=0.10, test=0.10, seed=42)

    if checkpoint_dir is None:
        # The checkpoint lives alongside the probe outputs it summarizes.
        checkpoint_dir = PROBE_ROOT / "_matrix_checkpoint"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "probe_matrix_checkpoint.json"
    results_subdir  = checkpoint_dir / "per_entry_results"
    results_subdir.mkdir(exist_ok=True)

    if not validate_checkpoint_consistency(checkpoint_dir, verbose=verbose):
        print("!!![!warning!]!!! Checkpoint inconsistencies detected.")

    checkpoint = {"completed": {}, "errors": []}
    if checkpoint_file.exists():
        try:
            checkpoint = json.loads(checkpoint_file.read_text())
        except Exception:
            if verbose >= 1:
                print(f"[checkpoint] Could not load {checkpoint_file}; starting fresh.")

    per_entry_results: list[pd.DataFrame] = []
    error_records:     list[dict]         = []

    for i, entry in enumerate(entries, start=1):
        model_name   = str(entry["model"])
        dataset_name = str(entry["dataset"])

        # ── Canonical artifact dir: explicitly provided by discovery, or
        #    derived from the flat layout. Never rebuilt from scratch. ──
        artifact_dir = Path(entry.get("artifact_dir") or artifact_dir_for(model_name, dataset_name))
        art = ExtractionArtifact(artifact_dir)

        cfg = AnalysisConfig(
            dataset=entry["contract"],
            probes=list(probes),
            layers="all",
            split=split,
            repeats=repeats,
            max_samples=max_samples,
            shuffled_label_control=shuffled_label_control,
            shuffled_control_repeats=shuffled_control_repeats,
            pca_enabled=True,
            silhouette_enabled=True,
            pca_samples=min(3000, max_samples or 3000),
            silhouette_samples=min(3000, max_samples or 3000),
            enable_per_class_metrics=True,
            enable_feature_statistics=True,
            verbose=verbose,
        )

        trial_cfg   = build_trial_config(art, cfg)
        trial_hash  = generate_trial_hash(trial_cfg)
        comp_hash   = generate_trial_hash(compute_computational_trial_config(trial_cfg))
        unique_key  = f"{model_name}::{dataset_name}::{trial_hash}"

        result_csv  = results_subdir / f"{trial_hash}_layer_probe_results.csv"

        if verbose >= 1:
            print(f"[matrix] {i}/{len(entries)} | {model_name} | {dataset_name} | {trial_hash[:8]}")

        # ── Resume path ──
        if unique_key in checkpoint.get("completed", {}):
            info = checkpoint["completed"][unique_key]
            if info.get("comp_hash") == comp_hash and result_csv.exists():
                if verbose >= 1:
                    print(f"[checkpoint] Resuming from {result_csv.name}")
                try:
                    per_entry_results.append(pd.read_csv(result_csv))
                    continue
                except Exception as exc:
                    print(f"[checkpoint] Failed to load {result_csv}: {exc}. Re-running.")
                    checkpoint["completed"].pop(unique_key, None)
            else:
                if verbose >= 1:
                    print("[checkpoint] Stored config differs — re-running.")
                checkpoint["completed"].pop(unique_key, None)

        # ── Fresh run ──
        try:
            analyzer = UnifiedProbeAnalyzer(art, cfg, dataset_df=entry.get("dataset_df"))
            scored, _ = analyzer.run()

            scored = scored.copy()
            scored["model"]         = model_name
            scored["dataset"]       = dataset_name
            scored["artifact_dir"]  = str(analyzer.output_dir)
            scored["metadata_path"] = str(analyzer.output_dir / "complete_run_metadata.json")
            scored["trial_hash"]    = trial_hash

            scored.to_csv(result_csv, index=False)
            update_results_index(checkpoint_dir, result_csv, model_name, dataset_name, trial_hash)

            per_entry_results.append(scored)
            checkpoint["completed"][unique_key] = {
                "model":          model_name,
                "dataset":        dataset_name,
                "trial_hash":     trial_hash,
                "comp_hash":      comp_hash,
                "result_csv":     str(result_csv),
                "artifact_dir":   str(analyzer.output_dir),
                "completed_at":   time.time(),
            }
            _save_checkpoint(checkpoint_file, checkpoint)

        except Exception as exc:
            print(f"[matrix] ERROR {model_name}/{dataset_name}: {type(exc).__name__}: {exc}")
            error_records.append({
                "model": model_name, "dataset": dataset_name,
                "trial_hash": trial_hash,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "status": "failed",
            })

    full_df = pd.concat(per_entry_results, ignore_index=True) if per_entry_results else pd.DataFrame()

    if error_records:
        error_csv = checkpoint_dir / "probe_errors.csv"
        pd.DataFrame(error_records).to_csv(error_csv, index=False)
        if verbose >= 0:
            print(f"[matrix] {len(error_records)} entries failed. See {error_csv}.")

    return full_df


def _save_checkpoint(checkpoint_file, checkpoint):
    tmp = checkpoint_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)
    tmp.replace(checkpoint_file)


def collect_layer_results(external_root, experiment_id, entries):
    frames = []
    for entry in entries:
        model_name = entry["model"]
        dataset_name = entry["dataset"]
        adir = dataset_dir_from_args(external_root, experiment_id, model_name, dataset_name)
        analysis_dir = adir / "analysis" / "probes"
        if not analysis_dir.exists():
            continue
        for run_dir in analysis_dir.glob("**/layer_probe_results.csv"):
            df = pd.read_csv(run_dir)
            df["model"] = model_name
            df["dataset"] = dataset_name
            df["artifact_dir"] = str(run_dir.parent)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dataset_dir_from_args(external_root, experiment_id, model_name, dataset_name):
    model_path = Path(*[p for p in model_name.split("/") if p])
    return external_root / "experiments" / experiment_id / "models" / model_path / "datasets" / dataset_name


def plot_full_dashboard(full_results, output_root):
    # (same as before, omitted for brevity)
    pass


def discover_model_dataset_pairs(external_root, experiment_id, model_names=None, dataset_names=None):
    pairs = []
    exp_root = external_root / "experiments" / experiment_id / "models"
    if not exp_root.exists():
        return pairs

    for model_dir in exp_root.glob("*/*"):
        model_name = "/".join(model_dir.relative_to(exp_root).parts)
        if model_names and model_name not in model_names:
            continue
        for dataset_dir in (model_dir / "datasets").glob("*"):
            if (dataset_dir / "metadata" / "extraction.json").exists():
                dataset_name = dataset_dir.name
                if dataset_names and dataset_name not in dataset_names:
                    continue
                pairs.append({"model": model_name, "dataset": dataset_name})
    return pairs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Hidden-State Probe v4.5")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--external-root", default=str(EXTERNAL_ROOT_DEFAULT))
    parser.add_argument("--experiment-id")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-name")
    parser.add_argument("--config")
    parser.add_argument("--write-example-config")
    parser.add_argument("--verify-checksum", action="store_true")
    args = parser.parse_args()

    if args.write_example_config:
        write_example_config(Path(args.write_example_config).expanduser().resolve())
        print(f"Example configuration written to {args.write_example_config}")
        return
    if not args.config:
        raise SystemExit("--config is required")

    config = load_config(Path(args.config).expanduser().resolve())
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    else:
        if not (args.experiment_id and args.model_name and args.dataset_name):
            raise SystemExit("Provide --dataset-dir OR --experiment-id --model-name --dataset-name")
        dataset_dir = dataset_dir_from_args(Path(args.external_root).expanduser().resolve(), args.experiment_id, args.model_name, args.dataset_name)

    artifact = ExtractionArtifact(dataset_dir, verify_checksum=args.verify_checksum)
    analyzer = UnifiedProbeAnalyzer(artifact, config)

    print("=" * 100)
    print(f"UNIFIED HIDDEN-STATE PROBE v{SCRIPT_VERSION}")
    print("=" * 100)
    print(json.dumps(artifact.analysis_summary(), indent=2, default=str))
    print("\nTarget contract:")
    print(json.dumps(analyzer.target_meta, indent=2, default=str))
    print("\nText alignment:")
    print(json.dumps(analyzer.text_alignment, indent=2, default=str))
    print("\nLabel alignment:")
    print(json.dumps(analyzer.label_alignment, indent=2, default=str))
    print("\nLayers:", analyzer.layers)
    print("Probes:", [f"{p.name}:{p.complexity}" for p in config.probes])
    print("Device:", analyzer.device)

    results_df, best = analyzer.run()

    print("\n" + "=" * 100)
    print("FINAL BEST-LAYER PROBE SCORE MATRIX")
    print("=" * 100)
    cols = [c for c in ["probe", "layer_index", "probe_score_mean", "test_macro_f1_mean", "test_balanced_accuracy_mean", "test_mcc_mean", "selectivity_mean"] if c in best.columns]
    print(best[cols].to_string(index=False))
    print("\nOutputs:", analyzer.output_dir)





"""
Model Availability on DevNeeds.ir

Cross-referenced against your 25-model MODEL_REGISTRY. All models ≤ 1.5B fit your Mac's memory envelope; the Qwen2-7B failure confirms a roughly 3B ceiling in fp16.

✅ Directly Available — 6 of 25

Exact name matches. No changes to MODEL_REGISTRY required.

Model	Params
google-bert/bert-base-uncased	110M
distilbert/distilbert-base-uncased	66M
microsoft/deberta-v3-small	140M
Qwen/Qwen2.5-0.5B	500M
Qwen/Qwen2.5-1.5B	1.5B
meta-llama/Llama-3.2-1B	1B
⚠️ Available as Close Variants — 5 of 25

Functionally equivalent, different training regime. MODEL_REGISTRY names would need editing.

Your name	DevNeeds substitute	Difference
Qwen/Qwen3-0.6B-Base	Qwen/Qwen3-0.6B	post-trained
meta-llama/Llama-3.2-3B	meta-llama/Llama-3.2-3B-Instruct	instruction-tuned
google/gemma-3-1b-pt	google/gemma-3-1b-it	instruction-tuned
google/gemma-3-4b-pt	google/gemma-3-4b-it	instruction-tuned
google/gemma-3-270m	google/gemma-3-270m-it	instruction-tuned
❌ Not Available on DevNeeds — 14 of 25

Absent entirely. Requires Cloudflare Worker, foreign download, or manual transfer.

Model	Missing family
FacebookAI/roberta-base	RoBERTa
google/electra-small-discriminator	ELECTRA
gpt2	GPT-2
EleutherAI/gpt-neo-125m	GPT-Neo
facebook/opt-125m	OPT
HuggingFaceTB/SmolLM2-135M	SmolLM2
HuggingFaceTB/SmolLM2-360M	SmolLM2
HuggingFaceTB/SmolLM2-1.7B	SmolLM2
Qwen/Qwen2-0.5B	Qwen2 (superseded)
Qwen/Qwen2-1.5B	Qwen2 (superseded)
Qwen/Qwen2.5-3B	Qwen2.5
Qwen/Qwen3-1.7B-Base	Qwen3
Qwen/Qwen3-4B-Base	Qwen3
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T	TinyLlama
Impact: Three entire families — RoBERTa, ELECTRA, and the GPT-2/GPT-Neo/OPT trio — are the canonical baselines in emotion-probing literature. Their absence materially weakens the model matrix unless resolved through a different route.

✨ Recommended Additions From DevNeeds

All fit your machine. Ranked by value for hidden-state probing.

Encoders — every layer meaningfully distinct

Model	Params
albert/albert-base-v2	12M
albert/albert-large-v2	18M
google-bert/bert-base-cased	110M
google-bert/bert-base-multilingual-cased	177M
google-bert/bert-large-uncased	335M
google-bert/bert-large-cased	335M
huggingface/distilbert-base-uncased-finetuned-mnli	66M
Encoder–decoder — different architectural family

Model	Params
google/flan-t5-small	60M
google/flan-t5-base	220M
Small modern decoders

Model	Params
Qwen/Qwen2.5-0.5B-Instruct	500M
Qwen/Qwen2.5-Coder-0.5B	500M
Qwen/Qwen1.5-0.5B-Chat	500M
Qwen/Qwen3-0.6B	600M
meta-llama/Llama-3.2-1B-Instruct	1B
meta-llama/Llama-Guard-3-1B	1B
meta-llama/Prompt-Guard-86M	86M
google/embeddinggemma-300m	300M
Sentence-transformers — embedding-specialised

Model	Params
sentence-transformers/all-MiniLM-L6-v2	22M
sentence-transformers/paraphrase-MiniLM-L6-v2	22M
sentence-transformers/all-mpnet-base-v2	110M
sentence-transformers/all-distilroberta-v1	82M
sentence-transformers/LaBSE	471M
Multilingual seq2seq

Model	Params
facebook/m2m100_418M	418M
facebook/nllb-200-distilled-600M	600M
Multimodal (optional)

Model	Params
openai/clip-vit-base-patch32	150M
openai/whisper-tiny	39M
openai/whisper-base	74M
Notes

DevNeeds is not a uniform API. Entries carry inconsistent suffixes (/main, /revision, commit SHAs). Downloads arrive as folders; you must manually reconstruct the cache layout under .hf_cache/hub/models--<owner>--<name>/snapshots/main/.
The Cloudflare Worker remains the stronger solution. One deployment unlocks the entire Group 3 list in a single endpoint; DevNeeds structurally cannot deliver those families.
Trim MODEL_REGISTRY before running. A focused 12-model matrix drawn from Groups 1 and 2 is scientifically stronger than 25 entries where 14 fail at download.
"""





if __name__ == "__main__":
    main()