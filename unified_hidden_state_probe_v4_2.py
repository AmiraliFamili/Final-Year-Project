"""
Unified Hidden‑State Probe v4.2 – Compatible with v2 extraction format.
Adds support for sample_ids, text_hashes, optional checksum verification,
and robust error handling in the matrix run.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import seaborn
import math
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from tqdm.auto import tqdm

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import platform
import sys
import importlib
import psutil  # optional, fallback if not installed
import torch

def get_environment_info() -> dict:
    """Collect comprehensive environment details."""
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
            "transformers": __import__("transformers").__version__ if importlib.util.find_spec("transformers") else None,
            "matplotlib": __import__("matplotlib").__version__,
            "seaborn": __import__("seaborn").__version__,
        },
        "device": {
            "chosen": choose_device(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        },
        "memory": {},
    }
    # Add memory info if psutil is available
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent_used": vm.percent,
        }
    except ImportError:
        pass
    return info

# =============================================================================
# Constants
# =============================================================================

EXTERNAL_ROOT_DEFAULT = Path("/Volumes/Amirali/hidden_states")
DEFAULT_SEED = 42
SCRIPT_VERSION = "4.2.0-v2"
DEBUG_MODE = False
VERBOSE_DEFAULT = 3 if DEBUG_MODE else 0

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


# =============================================================================
# General helpers
# =============================================================================

def save_complete_run_metadata(
    analyzer: UnifiedProbeAnalyzer,
    results_df: pd.DataFrame,
    best_df: pd.DataFrame,
    control_df: pd.DataFrame | None = None,
    extra_info: dict | None = None,
) -> Path:
    """
    Save a comprehensive metadata file next to the run results.
    Includes configuration, environment, validation, and final metrics.
    """
    output_dir = analyzer.output_dir
    metadata_path = output_dir / "complete_run_metadata.json"

    # 1. Base configuration
    config_dict = {
        "script_version": SCRIPT_VERSION,
        "created_at": time.time(),
        "dataset_contract": asdict(analyzer.config.dataset),
        "probes": [asdict(p) for p in analyzer.config.probes],
        "split": asdict(analyzer.config.split),
        "repeats": analyzer.config.repeats,
        "max_samples": analyzer.config.max_samples,
        "layers": analyzer.layers,
        "analysis": asdict(analyzer.config),  # includes all analysis flags
        "score_weights": _normalise_weights(analyzer.config.score_weights),
        "device": analyzer.device,
    }

    # 2. Artifact summary
    artifact_summary = analyzer.artifact.analysis_summary()
    # Remove potentially large provenance dict if needed (already stored)
    artifact_summary.pop("provenance", None)

    # 3. Target and validation
    target_info = {
        "target_metadata": analyzer.target_meta,
        "target_validation": analyzer.target_validation,
        "text_alignment": analyzer.text_alignment,
        "label_alignment": analyzer.label_alignment,
        "classes": analyzer.classes,
        "label_entropy_bits": label_entropy(analyzer.y, analyzer.task_type),
    }

    # 4. Environment
    environment = get_environment_info()

    # 5. Results summary (aggregate metrics)
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
    else:
        results_summary = {"rows": 0, "columns": [], "best_per_probe": [], "layer_wise_metrics": {}}

    # 6. Control summary if available
    control_summary = None
    if control_df is not None and not control_df.empty:
        control_summary = {
            "rows": len(control_df),
            "mean_control_macro_f1": float(control_df["control_test_macro_f1"].mean()),
            "by_layer": control_df.groupby("layer_index")["control_test_macro_f1"].mean().to_dict(),
        }

    # 7. Extra user-provided info (e.g., command-line args, notebook cell id)
    extra = extra_info or {}

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
    np.savez_compressed(path, **arrays)


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
    """Mirror the extraction pipeline's dataset_fingerprint.
    Uses first/last 16 samples to match extraction's stable_hash input.
    """
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


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DatasetContract:
    target_type: str = "auto"              # auto/goemotions/isear/custom
    type: str = "python"                   # python/file
    module: str | None = None
    function: str | None = None
    path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    text_column: str | None = "auto"
    label_column: str | None = "auto"
    id_column: str | None = "auto"

    task_type: str = "auto"                # auto/single_label/multi_label
    label_format: str = "auto"             # auto/scalar/integer_id_list/string_list
    single_label_policy: str | None = None # first_label/lowest_id/error_on_multi
    class_order: list[str] | None = None

    # Stronger alignment checks when extraction metadata contains row identifiers.
    require_provenance: bool = False
    require_label_fingerprint: bool = False
    
    lenient_provenance: bool = False 


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

    # Logistic complexity control.
    C: float = 1.0
    max_iter: int = 2000

    # MLP complexity control. "0.5d" means half of the representation width D.
    hidden_dims: list[int | str] = field(default_factory=list)
    hidden_width_ratio: float = 0.5
    width_schedule: str = "halving"    # halving/constant/custom

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

    # Controls.
    shuffled_label_control: bool = True
    shuffled_control_repeats: int = 3
    run_control_on_all_layers: bool = True

    # Geometry.
    pca_enabled: bool = True
    pca_samples: int = 3000
    silhouette_enabled: bool = True
    silhouette_samples: int = 3000

    # Extra measurements.
    enable_abstention: bool = True
    enable_per_class_metrics: bool = True
    enable_feature_statistics: bool = True

    # Final score. Weights are normalised internally.
    score_weights: dict[str, float] = field(default_factory=lambda: {
        "macro_f1": 0.25,
        "balanced_accuracy": 0.15,
        "mcc": 0.15,
        "log_loss_score": 0.10,
        "selectivity": 0.20,
        "stability": 0.10,
        "geometry": 0.05,
    })

    # Complexity penalty is deliberately small; complexity is reported separately.
    complexity_penalty_scale: float = 0.02
    output_subdir: str = "analysis/probes"

    # Runtime diagnostics. 0=clean runtime, 1=lifecycle, 2=layer/repeat, 3=per-probe.
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
    """Load complete_run_metadata.json from either the run directory or the file itself."""
    path = Path(run_dir_or_file)
    if path.is_dir():
        path = path / "complete_run_metadata.json"
    # If path is a file but not named 'complete_run_metadata.json', we assume it's the metadata file.
    # If it's neither, we append the expected filename (best effort).
    elif path.name != "complete_run_metadata.json":
        # If it has a .json suffix but different name, treat as metadata? 
        # For safety, append standard name if it doesn't exist
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
    """Centralised runtime diagnostics."""
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


# =============================================================================
# Extraction artifact (v2 compatible)
# =============================================================================


class ExtractionArtifact:
    def __init__(self, dataset_dir: Path, verify_checksum: bool = False):
        self.dataset_dir = dataset_dir.resolve()
        self.data_dir = self.dataset_dir / "data"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.states_path = self.data_dir / "hidden_states.npy"
        self.completed_path = self.data_dir / "completed.npy"
        self.metadata_path = self.metadata_dir / "extraction.json"
        self.sample_ids_path = self.metadata_dir / "sample_ids.npy"
        self.text_hashes_path = self.metadata_dir / "text_hashes.npy"
        self.checksum_path = self.metadata_dir / "checksum.sha256"

        missing = [str(p) for p in (
            self.states_path, self.completed_path, self.metadata_path
        ) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required extraction artifact(s):\n- " + "\n- ".join(missing))

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.states = np.load(self.states_path, mmap_mode="r")
        self.completed = np.load(self.completed_path, mmap_mode="r")

        # Load optional v2 artifacts
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

        # Optionally verify checksum (may be slow for large files)
        if verify_checksum and self.checksum_stored is not None:
            self._verify_checksum()

        self.validation = self._validate()

    def _verify_checksum(self) -> None:
        """Compute SHA256 of the entire hidden_states array and compare."""
        # Compute in chunks to avoid memory blow
        hasher = hashlib.sha256()
        n_samples, n_layers, n_hidden = self.states.shape
        chunk = 1024 * n_layers * n_hidden  # ~1k samples per chunk
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
            # Fallback: derive from dataset_dir path (e.g., .../models/<provider>/<model>/datasets/<ds>)
            parts = self.dataset_dir.parts
            # find 'models' and take next two parts
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
        # fallback: if the last part is 'datasets', use the parent model name + '_' + something?
        parts = self.dataset_dir.parts
        if parts[-1] == 'datasets':
            # maybe return the experiment dataset identifier from metadata?
            return self.metadata.get("dataset", {}).get("text_column", "unknown")
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
        issues: list[str] = []
        warnings: list[str] = []

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

        # Sample IDs check
        if self.sample_ids is not None and len(self.sample_ids) != self.sample_count:
            issues.append(f"sample_ids length {len(self.sample_ids)} != sample count {self.sample_count}")

        # Text hashes check
        if self.text_hashes is not None and len(self.text_hashes) != self.sample_count:
            issues.append(f"text_hashes length {len(self.text_hashes)} != sample count {self.sample_count}")

        # Sampled finite check avoids materialising the full mmap at validation time.
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


# =============================================================================
# Dataset loading and label adapters (unchanged)
# =============================================================================


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
        scored: list[tuple[float, str]] = []
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
            try:
                return ast.literal_eval(s)
            except Exception:
                return value
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
    if contract.target_type != "auto":
        return contract.target_type.lower()
    cols = set(map(str, df.columns))
    if "labels" in cols:
        sample = df["labels"].head(20).tolist()
        try:
            parsed = [parse_integer_list(x) for x in sample]
            if any(len(v) > 1 for v in parsed):
                return "goemotions"
        except Exception:
            pass
        if "emotion" not in cols and "emotion_label" not in cols and "dominant_emotion" not in cols:
            return "goemotions"
    if any(c in cols for c in ("emotion", "emotion_label", "dominant_emotion")):
        return "isear" if "dominant_emotion" not in cols and "emotion" in cols else "custom"
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

    def decode_one_row(value: Any, row_index: int) -> tuple[list[int], list[str], str]:
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

        current_ids: list[int] = []
        current_names: list[str] = []
        mode = source_mode

        for item in seq:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if idx < 0 or idx >= len(GOEMOTIONS_CLASSES):
                    raise ValueError(
                        f"GoEmotions row {row_index} has invalid label ID {idx}; "
                        f"valid IDs are 0..{len(GOEMOTIONS_CLASSES)-1}"
                    )
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
                    raise ValueError(
                        f"GoEmotions row {row_index} has unknown label name {item!r}. "
                        f"Expected an ID in 0..{len(GOEMOTIONS_CLASSES)-1} or one of "
                        f"{GOEMOTIONS_CLASSES}."
                    )
                canonical = by_name[name]
                current_names.append(canonical)
                current_ids.append(GOEMOTIONS_CLASSES.index(canonical))
                mode = "string_name"
                continue

            raise ValueError(
                f"GoEmotions row {row_index} contains unsupported label value {item!r} "
                f"of type {type(item).__name__}"
            )

        if not current_ids:
            raise ValueError(f"GoEmotions row {row_index} has no labels")

        seen: set[int] = set()
        ids_unique: list[int] = []
        names_unique: list[str] = []
        for idx, name in zip(current_ids, current_names):
            if idx not in seen:
                seen.add(idx)
                ids_unique.append(idx)
                names_unique.append(name)
        return ids_unique, names_unique, mode

    parsed_ids: list[list[int]] = []
    parsed_names: list[list[str]] = []
    modes: set[str] = set()
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
        raise ValueError(
            "GoEmotions contains multi-label examples. Set task_type='multi_label' "
            "or explicitly choose a single_label_policy."
        )
    y = np.asarray([
        labels[0] if policy == "first_label" else min(labels)
        for labels in parsed_ids
    ], dtype=np.int64)
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
    label_col, resolution = resolve_column(
            df, contract.label_column,
            ["emotion", "label", "labels", "category", "emotion_label"],
            role="label",
        )
    raw_values = df[label_col].tolist()
    aliases = {
        "joy": "joy", "fear": "fear", "anger": "anger", "sadness": "sadness",
        "disgust": "disgust", "shame": "shame", "guilt": "guilt",
    }
    # Detect if labels are integers
    if all(isinstance(x, (int, np.integer)) for x in raw_values):
        # Map integer to class names using the class_order list positions
        order = contract.class_order or ISEAR_CLASSES
        if len(order) != 7:
            raise ValueError("ISEAR class_order must contain 7 emotions for numeric mapping.")
        # Convert to integer indices (0-based)
        y = np.asarray([int(x) - 1 for x in raw_values], dtype=np.int64)
        # Optional: ensure indices are within range
        if np.any(y < 0) or np.any(y >= len(order)):
            raise ValueError("ISEAR numeric labels out of range.")
        return y, order, {
            "adapter": "isear",
            "task_type": "single_label",
            "raw_label_column": label_col,
            "label_resolution": resolution,
            "class_names": order,
            "class_count": len(order),
            "normalisation": "numeric index to class_order",
        }
    else:
        raw = [_normalise_name(x) for x in df[label_col].tolist()]
        normalised: list[str] = []
        for i, x in enumerate(raw):
            key = aliases.get(x)
            if key is None:
                raise ValueError(f"ISEAR row {i} has unknown emotion {x!r}; expected {ISEAR_CLASSES}")
            normalised.append(key)
        order = contract.class_order or ISEAR_CLASSES
        mapping = {_normalise_name(name): i for i, name in enumerate(order)}
        unknown = sorted(set(normalised) - set(_normalise_name(x) for x in order))
        if unknown:
            raise ValueError(f"ISEAR labels missing from class_order: {unknown}")
        y = np.asarray([mapping[_normalise_name(x)] for x in normalised], dtype=np.int64)
        return y, order, {
            "adapter": "isear",
            "task_type": "single_label",
            "raw_label_column": label_col,
            "label_resolution": resolution,
            "class_names": order,
            "class_count": len(order),
            "normalisation": "lowercase categorical canonicalisation",
        }


def canonical_custom_target(df: pd.DataFrame, contract: DatasetContract):
    label_col, resolution = resolve_column(df, contract.label_column, COMMON_LABEL_COLUMNS, role="label")
    raw = df[label_col].tolist()
    task_type = contract.task_type

    if task_type == "multi_label" or (
        task_type == "auto" and any(isinstance(_maybe_literal(x), (list, tuple, set)) for x in raw)
    ):
        label_lists = [parse_string_list(x) for x in raw]
        classes = contract.class_order or sorted({x for row in label_lists for x in row})
        mapping = {str(name): i for i, name in enumerate(classes)}
        y = np.zeros((len(label_lists), len(classes)), dtype=np.int64)
        for i, row in enumerate(label_lists):
            if not row:
                raise ValueError(f"Custom multi-label row {i} has no labels")
            for label in row:
                if label not in mapping:
                    raise ValueError(f"Unknown custom label {label!r} at row {i}")
                y[i, mapping[label]] = 1
        return y, classes, {
            "adapter": "custom",
            "task_type": "multi_label",
            "raw_label_column": label_col,
            "label_resolution": resolution,
            "class_names": classes,
            "class_count": len(classes),
        }

    scalar = [str(x) for x in raw]
    classes = contract.class_order or sorted(pd.unique(np.asarray(scalar, dtype=object)).tolist())
    mapping = {name: i for i, name in enumerate(classes)}
    unknown = sorted(set(scalar) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown custom labels: {unknown}")
    y = np.asarray([mapping[x] for x in scalar], dtype=np.int64)
    return y, classes, {
        "adapter": "custom",
        "task_type": "single_label",
        "raw_label_column": label_col,
        "label_resolution": resolution,
        "class_names": classes,
        "class_count": len(classes),
    }


def build_targets(df: pd.DataFrame, contract: DatasetContract):
    target_type = infer_target_type(df, contract)
    if target_type == "goemotions":
        return canonical_goemotions_target(df, contract)
    if target_type == "isear":
        return canonical_isear_target(df, contract)
    if target_type == "custom":
        return canonical_custom_target(df, contract)
    raise ValueError(f"Unsupported target_type={target_type}")


# =============================================================================
# Provenance and target validation
# =============================================================================


def _get_metadata_text_hashes(artifact: ExtractionArtifact) -> dict[str, str | None]:
    prov = artifact.provenance
    return {
        "derived_fingerprint": prov.get("derived_fingerprint"),
        "head_hash": prov.get("head_hash"),
        "tail_hash": prov.get("tail_hash"),
        "full_hash": prov.get("full_hash"),
        "native_fingerprint": prov.get("native_fingerprint"),
    }

def validate_text_alignment(
    artifact: ExtractionArtifact,
    df: pd.DataFrame,
    contract: DatasetContract,
) -> dict[str, Any]:
    text_col, resolution = resolve_column(
        df, contract.text_column, COMMON_TEXT_COLUMNS,
        role="text", allow_scored_text_guess=True,
    )
    texts = one_dim_strings(df[text_col].tolist())
    if len(texts) != artifact.sample_count:
        raise RuntimeError(
            f"Text row count={len(texts)} differs from hidden-state count={artifact.sample_count}"
        )

    observed = {
        "derived_fingerprint": fingerprint_values(texts),
        "head_hash": sequence_hash(texts[:100]),
        "tail_hash": sequence_hash(texts[-100:]),
    }
    expected = _get_metadata_text_hashes(artifact)
    checks: dict[str, bool] = {}
    checked_fields: list[str] = []

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
                raise RuntimeError(
                    "Dataset/text provenance mismatch (head/tail). Refusing to probe.\n"
                    + json.dumps({"expected": expected, "observed": observed, "checks": checks}, indent=2, default=str)
                )
            # Accept if head and tail match, even if full derived_fingerprint differs
            if "derived_fingerprint" in checks and not checks["derived_fingerprint"]:
                warning = "Derived fingerprint mismatch but head/tail match. Proceeding with lenient provenance."
        else:
            if not all(checks.values()):
                raise RuntimeError(
                    "Dataset/text provenance mismatch. Refusing to probe.\n"
                    + json.dumps({"expected": expected, "observed": observed, "checks": checks}, indent=2, default=str)
                )
    provenance_available = bool(checked_fields)
    if contract.require_provenance and not provenance_available:
        raise RuntimeError("require_provenance=True but extraction metadata contains no usable text provenance hashes.")

    # sample_ids check
    sample_ids_match = None
    if artifact.sample_ids is not None:
        id_col = contract.id_column
        if id_col and id_col != "auto" and id_col in df.columns:
            df_ids = one_dim_strings(df[id_col].tolist())
            if len(df_ids) == len(artifact.sample_ids):
                sample_ids_match = bool(np.array_equal(df_ids, artifact.sample_ids))
            else:
                sample_ids_match = False
        # else can't verify

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


def validate_label_alignment(
    artifact: ExtractionArtifact,
    df: pd.DataFrame,
    contract: DatasetContract,
    y: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    observed = stable_hash({
        "classes": list(classes),
        "labels": np.asarray(y).tolist(),
    }, 24)
    expected = artifact.provenance.get("label_fingerprint") or artifact.provenance.get("target_fingerprint")

    if expected is not None and expected != observed:
        raise RuntimeError(
            "Label provenance mismatch. Refusing to probe because the stored target "
            "fingerprint disagrees with the clean-dataset target.\n"
            + json.dumps({"expected": expected, "observed": observed}, indent=2)
        )
    if expected is None and contract.require_label_fingerprint:
        raise RuntimeError(
            "require_label_fingerprint=True but extraction metadata contains no label/target fingerprint."
        )

    return {
        "status": "verified" if expected is not None else "unverified",
        "verified": expected is not None,
        "provenance_available": expected is not None,
        "label_fingerprint": observed,
        "metadata_label_fingerprint": expected,
        "warning": None if expected is not None else (
            "No label fingerprint was stored during extraction. Clean labels were "
            "reconstructed deterministically, but extraction-time row identity is not "
            "cryptographically proven by the existing artifact."
        ),
    }


def validate_targets(y: np.ndarray, classes: Sequence[str], task_type: str) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    y = np.asarray(y)
    if len(classes) < 2:
        issues.append("At least two classes are required")

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


# =============================================================================
# Split logic (unchanged)
# =============================================================================


def can_stratify_single(y: np.ndarray, min_count: int = 3) -> bool:
    if y.ndim != 1:
        return False
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    return len(nonzero) >= 2 and bool(np.all(nonzero >= min_count))


def make_single_splits(y: np.ndarray, cfg: SplitConfig, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    stratify = y if cfg.stratify and can_stratify_single(y, 3) else None
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1.0 - cfg.train,
        random_state=seed,
        stratify=stratify,
    )
    temp_y = y[temp_idx]
    temp_stratify = temp_y if cfg.stratify and can_stratify_single(temp_y, 2) else None
    test_fraction_of_temp = cfg.test / (cfg.validation + cfg.test)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_fraction_of_temp,
        random_state=seed,
        stratify=temp_stratify,
    )
    return {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def make_multilabel_splits(y: np.ndarray, cfg: SplitConfig, seed: int) -> dict[str, np.ndarray]:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=1.0 - cfg.train,
            random_state=seed,
        )
        idx = np.arange(len(y))
        train_rel, temp_rel = next(splitter.split(idx, y))
        temp_y = y[temp_rel]
        splitter2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=cfg.test / (cfg.validation + cfg.test),
            random_state=seed + 1,
        )
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


# =============================================================================
# Metrics (unchanged)
# =============================================================================


def safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return float("nan")


def _safe_multilabel_auc_and_ap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float | None, float | None, dict[str, int]]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        probabilities = probabilities.reshape(-1, 1)

    valid_auc: list[float] = []
    valid_ap: list[float] = []

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
    return (
        float(np.mean(valid_auc)) if valid_auc else None,
        float(np.mean(valid_ap)) if valid_ap else None,
        coverage,
    )


def safe_roc_auc_single(y_true: np.ndarray, proba: np.ndarray | None, n_classes: int) -> float | None:
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


def safe_average_precision_single(y_true: np.ndarray, proba: np.ndarray | None, n_classes: int) -> float | None:
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


def confidence_metrics(y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    confidence = np.max(proba, axis=1)
    correct = (y_true == y_pred).astype(float)
    return {
        "mean_confidence": float(np.mean(confidence)),
        "mean_confidence_correct": float(np.mean(confidence[correct == 1])) if np.any(correct == 1) else None,
        "mean_confidence_incorrect": float(np.mean(confidence[correct == 0])) if np.any(correct == 0) else None,
        "high_confidence_error_rate": float(np.mean((confidence >= 0.8) & (correct == 0))) if len(confidence) else None,
    }


def evaluate_single(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    probabilities: np.ndarray | None = None,
    include_per_class: bool = True,
) -> dict[str, Any]:
    labels = np.arange(len(classes))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": safe_mcc(y_true, y_pred),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(classes), output_dict=True, zero_division=0
        ),
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


def evaluate_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
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
            result["log_loss"] = float(
                log_loss(y_true.ravel(), probabilities.ravel(), labels=[0, 1])
            )
            result["log_loss_score"] = float(
                np.exp(-min(max(result["log_loss"], 0.0), 20.0))
            )
        except Exception:
            result["log_loss"] = None
            result["log_loss_score"] = None

        roc_auc, avg_precision, coverage = _safe_multilabel_auc_and_ap(
            y_true, probabilities
        )
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


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    counts = np.bincount(y_train, minlength=len(classes))
    majority_id = int(np.argmax(counts))
    pred = np.full(len(y_test), majority_id, dtype=np.int64)
    return {
        "baseline": "majority_class",
        "class": classes[majority_id],
        "test": evaluate_single(y_test, pred, classes),
        "chance_accuracy": 1.0 / len(classes),
    }


def label_entropy(y: np.ndarray, task_type: str) -> float:
    if task_type == "single_label":
        counts = np.bincount(y)
        p = counts[counts > 0] / len(y)
        return float(-np.sum(p * np.log2(p)))
    counts = y.mean(axis=0)
    return float(np.mean([-(p * np.log2(p) + (1 - p) * np.log2(1 - p)) for p in counts if 0 < p < 1]))


# =============================================================================
# Probe architecture (unchanged)
# =============================================================================


def resolve_hidden_width(spec: int | str, input_dim: int) -> int:
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


def resolved_hidden_dims(spec: ProbeSpec, input_dim: int) -> list[int]:
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
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Probe fitting (unchanged)
# =============================================================================


def _make_logistic(spec: ProbeSpec, seed: int) -> Pipeline:
    steps = []
    if spec.standardize:
        steps.append(("scale", StandardScaler()))
    steps.append((
        "logistic",
        LogisticRegression(C=spec.C, max_iter=spec.max_iter, solver="lbfgs", random_state=seed),
    ))
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


def _fit_one_binary(X_train, target_train, X_val, X_test, spec, seed):
    unique = np.unique(target_train)
    if len(unique) == 1:
        constant = int(unique[0])
        class Constant:
            def __init__(self, c): self.c = c
            def predict(self, X): return np.full(len(X), self.c, dtype=np.int64)
            def predict_proba(self, X):
                p1 = np.full(len(X), float(self.c), dtype=float)
                return np.column_stack([1 - p1, p1])
        return Constant(constant)
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
        "parameters": int(sum(
            m.named_steps["logistic"].coef_.size + m.named_steps["logistic"].intercept_.size
            if hasattr(m, "named_steps") else 2 for m in models
        )),
        "resolved_hidden_dims": [],
        "epochs_completed": None,
    }, models


def _selection_value(metrics: Mapping[str, Any], metric: str) -> float:
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


def fit_probe(spec: ProbeSpec, X_train, y_train, X_val, y_val, X_test, y_test, classes, task_type, seed, device, include_per_class):
    validate_probe_spec(spec, task_type)
    if spec.type == "logistic":
        if task_type == "single_label":
            return fit_logistic_single(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, include_per_class)
        return fit_logistic_multi(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, include_per_class)
    return fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, classes, spec, seed, task_type, device, include_per_class)


# =============================================================================
# Geometry and representation statistics
# =============================================================================


def geometry_analysis(X: np.ndarray, y: np.ndarray, classes: Sequence[str], task_type: str, seed: int, cfg: AnalysisConfig) -> dict[str, Any]:
    idx = sample_indices(len(X), max(cfg.pca_samples, cfg.silhouette_samples), seed)
    Xs = X[idx]
    result: dict[str, Any] = {
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


# =============================================================================
# Score system
# =============================================================================


def _normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {k: float(v) for k, v in weights.items() if float(v) >= 0}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("At least one score weight must be >0")
    return {k: v / total for k, v in clean.items()}


def compute_complexity_penalty(parameters: int | float | None, input_dim: int, scale: float) -> float:
    if parameters is None or not np.isfinite(parameters) or parameters <= 0:
        return 0.0
    relative = math.log10(max(parameters, 1)) / math.log10(max(input_dim * 100.0, 10.0))
    return float(np.clip(scale * relative, 0, scale))


def add_score_columns(results_df: pd.DataFrame, control_df: pd.DataFrame | None, cfg: AnalysisConfig, task_type: str) -> pd.DataFrame:
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

    complexity = [
        compute_complexity_penalty(p, d, cfg.complexity_penalty_scale)
        for p, d in zip(df["parameters"], df["input_dim"])
    ]
    df["complexity_penalty"] = complexity
    df["probe_score_raw"] = np.clip(raw, 0, 1)
    df["probe_score"] = np.clip(df["probe_score_raw"] - df["complexity_penalty"], 0, 1)
    df["generalization_gap"] = df["train_macro_f1"] - df["test_macro_f1"]
    df["overfit_penalty"] = np.clip(df["generalization_gap"], 0, 1)
    return df


# =============================================================================
# Plots and final dashboard (unchanged)
# =============================================================================


def heatmap_image(matrix: pd.DataFrame, path: Path, title: str, xlabel: str, ylabel: str, fmt: str = ".3f", vmin: float | None = None, vmax: float | None = None) -> None:
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


def plot_layer_curves(df: pd.DataFrame, output_dir: Path) -> None:
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


def create_final_visuals(results_df: pd.DataFrame, output_dir: Path) -> None:
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

    best = (
        results_df.sort_values(["probe", "probe_score"], ascending=[True, False])
        .groupby("probe", as_index=False)
        .first()
    )
    final_matrix = best.set_index("probe")[[
        "test_macro_f1", "test_balanced_accuracy", "test_mcc",
        "selectivity", "complexity_penalty", "probe_score",
    ]].copy()
    heatmap_image(final_matrix, output_dir / "final_probe_score_heatmap.png", "Final best-layer probe measurement matrix", "Measurement", "Probe", vmin=0, vmax=1)

    grouped = results_df.groupby("probe").agg(
        mean_score=("probe_score", "mean"),
        std_score=("probe_score", "std"),
    ).reset_index()
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


# =============================================================================
# Analyzer
# =============================================================================


class UnifiedProbeAnalyzer:
    def __init__(
        self,
        artifact: ExtractionArtifact,
        config: AnalysisConfig,
        output_dir: Path | None = None,
        dataset_df: pd.DataFrame | Any | None = None,
    ):
        config.validate_verbose()
        self.artifact = artifact
        self.config = config
        self.device = choose_device()
        self.logger = ProbeLogger(config.verbose)

        self.logger.section("INITIALISING UNIFIED HIDDEN-STATE PROBE", 1)

        self.df = to_dataframe(dataset_df) if dataset_df is not None else load_dataframe(config.dataset)
        if len(self.df) != artifact.sample_count:
            raise RuntimeError(
                f"Dataset rows={len(self.df)} != hidden-state samples={artifact.sample_count}. "
                "This is a hard alignment failure."
            )

        self.y, self.classes, self.target_meta = build_targets(self.df, config.dataset)
        self.task_type = self.target_meta["task_type"]
        self.target_validation = validate_targets(self.y, self.classes, self.task_type)
        self.text_alignment = validate_text_alignment(artifact, self.df, config.dataset)
        self.label_alignment = validate_label_alignment(artifact, self.df, config.dataset, self.y, self.classes)

        if self.text_alignment.get("verified") and not self.label_alignment.get("verified"):
            self.label_alignment["verification_basis"] = (
                "Label row order inherits verification from the cryptographically matched "
                "text sequence in the same dataframe."
            )

        self.layers = self._resolve_layers(config.layers)

        if output_dir is None:
            run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{stable_hash(asdict(config), 10)}"
            output_dir = artifact.dataset_dir / config.output_subdir / run_id
        self.output_dir = safe_relative_output(artifact.dataset_dir, Path(output_dir))
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

    def _resolve_layers(self, requested: list[int | str] | str) -> list[str]:
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

    def _preflight(self) -> None:
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

    def write_run_manifest(self) -> None:
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

    def _prepare_population(self, seed: int) -> np.ndarray:
        return (
            np.arange(len(self.y), dtype=np.int64)
            if self.config.max_samples is None
            else sample_indices(len(self.y), self.config.max_samples, seed)
        )

    def _split(self, selected: np.ndarray, seed: int) -> dict[str, np.ndarray]:
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

    def _load_population_layer(self, layer_idx: int, selected: np.ndarray) -> np.ndarray:
        X = np.asarray(self.artifact.states[selected, layer_idx, :], dtype=np.float32)
        if not np.isfinite(X).all():
            raise RuntimeError(f"Layer {layer_idx} contains NaN/Inf in selected rows")
        return X

    def _metric_fields(self, result: Mapping[str, Any], split_name: str) -> dict[str, Any]:
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

    def _save_probe_artifacts(
        self,
        probe: ProbeSpec,
        layer_name: str,
        repeat: int,
        results: dict[str, Any],
        model: Any,
        scaler: Any,
        record: dict[str, Any],
    ) -> None:
        d = self.output_dir / "models" / probe.name / layer_name / f"repeat_{repeat}"
        d.mkdir(parents=True, exist_ok=True)
        save_json(d / "metrics.json", {"record": record, "results": results})
        if self.task_type == "single_label":
            save_npz(
                d / "confusion_matrix_test.npz",
                matrix=np.asarray(results["test"]["confusion_matrix"]),
            )
        if probe.type == "logistic":
            joblib.dump(model, d / "probe.joblib")
        else:
            torch.save(model.state_dict(), d / "probe_state_dict.pt")
        if scaler is not None:
            joblib.dump(scaler, d / "scaler.joblib")

    def _exact_split_controls(
        self,
        layer_idx: int,
        probe: ProbeSpec,
        X_population: np.ndarray,
        selected: np.ndarray,
        split: Mapping[str, np.ndarray],
        repeat: int,
    ) -> list[dict[str, Any]]:
        if not self.config.shuffled_label_control:
            return []

        positions = {int(global_i): i for i, global_i in enumerate(selected)}
        tr = np.asarray([positions[int(i)] for i in split["train"]], dtype=np.int64)
        va = np.asarray([positions[int(i)] for i in split["validation"]], dtype=np.int64)
        te = np.asarray([positions[int(i)] for i in split["test"]], dtype=np.int64)
        local_y = self.y[selected].copy()
        rows: list[dict[str, Any]] = []

        for control_repeat in range(self.config.shuffled_control_repeats):
            seed = (
                self.config.split.seed
                + 1_000_000
                + repeat * 10_000
                + layer_idx * 100
                + control_repeat
                + stable_int(probe.name)
            )
            rng = np.random.default_rng(seed)
            shuffled_y = local_y.copy()
            rng.shuffle(shuffled_y, axis=0)

            Xtr_raw = X_population[tr]
            Xv_raw = X_population[va]
            Xte_raw = X_population[te]
            scaler = StandardScaler().fit(Xtr_raw) if probe.standardize else None
            if scaler is not None:
                Xtr = scaler.transform(Xtr_raw).astype(np.float32)
                Xv = scaler.transform(Xv_raw).astype(np.float32)
                Xte = scaler.transform(Xte_raw).astype(np.float32)
            else:
                Xtr, Xv, Xte = Xtr_raw, Xv_raw, Xte_raw

            result, _ = fit_probe(
                probe,
                Xtr,
                shuffled_y[tr],
                Xv,
                shuffled_y[va],
                Xte,
                shuffled_y[te],
                self.classes,
                self.task_type,
                seed,
                self.device,
                self.config.enable_per_class_metrics,
            )
            test_result = result["test"]
            rows.append({
                "repeat": repeat,
                "control_repeat": control_repeat,
                "seed": seed,
                "probe": probe.name,
                "layer_index": layer_idx,
                "control_test_macro_f1": test_result.get("macro_f1"),
                "control_test_accuracy": test_result.get(
                    "accuracy", test_result.get("exact_match_accuracy")
                ),
                "control_test_mcc": test_result.get("mcc"),
            })
        return rows

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.write_run_manifest()
        records: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        split_archive: dict[str, np.ndarray] = {}

        self.logger.section("PROBING EXPERIMENT", 1)
        self.logger.emit(
            "Question: how recoverable is the target from each frozen hidden-state layer?",
            1,
        )
        self.logger.emit(
            f"repeats={self.config.repeats} | max_samples={self.config.max_samples} | "
            f"layers={len(self.layers)} | probes={len(self.config.probes)}",
            1,
        )

        # Display target coverage (unchanged)
        if self.logger.level >= 1:
            if self.task_type == "single_label":
                counts = np.bincount(self.y, minlength=len(self.classes))
                rare = sorted(
                    [(self.classes[i], int(c)) for i, c in enumerate(counts) if c > 0],
                    key=lambda x: x[1],
                )[:10]
                self.logger.emit(f"Target coverage: observed_classes={int(np.sum(counts > 0))}/{len(self.classes)} | rarest={rare}", 1)
            else:
                positives = np.sum(self.y, axis=0)
                observed = int(np.sum(positives > 0))
                self.logger.emit(
                    f"Target coverage: labels_with_positive_support={observed}/{len(self.classes)} | "
                    f"rarest={sorted((int(c), self.classes[i]) for i, c in enumerate(positives) if c > 0)[:10]}",
                    1,
                )

        # Calculate total number of fittings for progress bar
        total_fittings = self.config.repeats * len(self.layers) * len(self.config.probes)
        if self.config.shuffled_label_control:
            total_fittings += (
                self.config.repeats * len(self.layers) * len(self.config.probes)
                * self.config.shuffled_control_repeats
            )

        # Create progress bar (always visible, even if verbose=0, but only if tqdm installed)
        pbar = tqdm(total=total_fittings, desc="Probing", unit="fit", disable=(self.config.verbose < 0))

        try:
            for repeat in range(self.config.repeats):
                seed = self.config.split.seed + repeat
                selected = self._prepare_population(seed)
                split = self._split(selected, seed)
                for name, idx in split.items():
                    split_archive[f"repeat_{repeat}_{name}"] = idx

                y_train = self.y[split["train"]]
                y_val = self.y[split["validation"]]
                y_test = self.y[split["test"]]
                baseline = (
                    majority_baseline(y_train, y_test, self.classes)
                    if self.task_type == "single_label"
                    else None
                )

                self.logger.section(f"REPEAT {repeat + 1}/{self.config.repeats}", 2)
                self.logger.emit(
                    f"seed={seed} | population={len(selected)} | train={len(y_train)} | "
                    f"val={len(y_val)} | test={len(y_test)}",
                    2,
                )

                positions = {int(global_i): i for i, global_i in enumerate(selected)}
                tr_local = np.asarray([positions[int(i)] for i in split["train"]], dtype=np.int64)
                va_local = np.asarray([positions[int(i)] for i in split["validation"]], dtype=np.int64)
                te_local = np.asarray([positions[int(i)] for i in split["test"]], dtype=np.int64)

                for layer_name in self.layers:
                    layer_idx = parse_layer_number(layer_name)
                    relative_depth = (
                        layer_idx / (self.artifact.hidden_layers - 1)
                        if self.artifact.hidden_layers > 1 else 0.0
                    )

                    X_population = self._load_population_layer(layer_idx, selected)

                    geom_count = min(
                        len(selected),
                        max(self.config.pca_samples, self.config.silhouette_samples),
                    )
                    geom_local = sample_indices(len(selected), geom_count, seed + layer_idx)
                    geom = geometry_analysis(
                        X_population[geom_local],
                        self.y[selected][geom_local],
                        self.classes,
                        self.task_type,
                        seed + layer_idx,
                        self.config,
                    )
                    save_json(
                        self.output_dir / "geometry" / f"{layer_name}_repeat_{repeat}.json",
                        geom,
                    )

                    self.logger.emit(
                        f"Layer {layer_idx} | relative depth={relative_depth:.3f} | "
                        f"geometry silhouette={geom.get('silhouette_score')}",
                        2,
                    )

                    Xtr_raw = X_population[tr_local]
                    Xv_raw = X_population[va_local]
                    Xte_raw = X_population[te_local]

                    scaled_cache = None
                    if any(p.standardize for p in self.config.probes):
                        shared_scaler = StandardScaler().fit(Xtr_raw)
                        scaled_cache = (
                            shared_scaler.transform(Xtr_raw).astype(np.float32),
                            shared_scaler.transform(Xv_raw).astype(np.float32),
                            shared_scaler.transform(Xte_raw).astype(np.float32),
                        )
                    else:
                        shared_scaler = None

                    for probe in self.config.probes:
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

                        results, model = fit_probe(
                            probe,
                            Xtr,
                            y_train,
                            Xv,
                            y_val,
                            Xte,
                            y_test,
                            self.classes,
                            self.task_type,
                            probe_seed,
                            self.device,
                            self.config.enable_per_class_metrics,
                        )
                        pbar.update(1)   # <-- advance progress bar after real fit

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
                        records.append(record)

                        self._save_probe_artifacts(
                            probe,
                            layer_name,
                            repeat,
                            results,
                            model,
                            scaler_for_artifact,
                            record,
                        )

                        # Shuffled-label controls
                        ctrl_rows = self._exact_split_controls(
                            layer_idx,
                            probe,
                            X_population,
                            selected,
                            split,
                            repeat,
                        )
                        controls.extend(ctrl_rows)
                        pbar.update(len(ctrl_rows))   # advance by number of control fits

                        if self.config.verbose >= 3:
                            test = results["test"]
                            self.logger.emit(
                                f"TEST Macro-F1={test.get('macro_f1')} | "
                                f"BalancedAcc={test.get('balanced_accuracy')} | "
                                f"MCC={test.get('mcc')}",
                                3,
                            )
                            if self.task_type == "multi_label":
                                self.logger.emit(
                                    f"TEST label coverage: positive={test.get('labels_with_positive_support')} | "
                                    f"both_classes={test.get('labels_with_both_support')} | "
                                    f"ROC-AUC={test.get('roc_auc_macro')} | AP={test.get('average_precision_macro')}",
                                    3,
                                )
                                
            pbar.close()
                                
        except Exception:
            print("Unexpected Error, Closing the P Bar... ")
            pbar.close()

        # Rest of function unchanged...
        save_npz(self.output_dir / "split_indices.npz", **split_archive)
        results_df = pd.DataFrame(records)
        control_df = pd.DataFrame(controls)
        if not control_df.empty:
            control_df.to_csv(self.output_dir / "shuffled_label_controls.csv", index=False)

        scored = add_score_columns(
            results_df,
            control_df if not control_df.empty else None,
            self.config,
            self.task_type,
        )

        aggregate = (
            scored.groupby(
                ["probe", "probe_type", "probe_complexity", "layer_index"],
                as_index=False,
            )
            .agg(
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
        )
        best = (
            aggregate.sort_values(
                ["probe", "probe_score_mean"], ascending=[True, False]
            )
            .groupby("probe", as_index=False)
            .first()
        )

        scored.to_csv(self.output_dir / "layer_probe_results.csv", index=False)
        aggregate.to_csv(self.output_dir / "layer_probe_aggregate_results.csv", index=False)
        best.to_csv(self.output_dir / "final_probe_score_matrix.csv", index=False)
        create_final_visuals(scored, self.output_dir)
        metadata_path = save_complete_run_metadata(self, scored, best, control_df)
        self.logger.emit(f"Complete run metadata saved: {metadata_path}", 1)

        summary = { ... }  # unchanged
        save_json(self.output_dir / "summary.json", summary)
        save_json(
            self.output_dir / "completion.json",
            {"status": "complete", "finished_at": time.time()},
        )

        self.logger.section("FINAL RESULT", 1)
        if not best.empty:
            cols = [
                c for c in [
                    "probe", "layer_index", "relative_layer_depth",
                    "probe_score_mean", "test_macro_f1_mean",
                    "test_balanced_accuracy_mean", "test_mcc_mean",
                    "selectivity_mean",
                ] if c in best.columns
            ]
            self.logger.emit("Final best layer table:", 1)
            if self.config.verbose >= 1:
                print(best[cols].to_string(index=False))
        self.logger.emit(f"Output directory: {self.output_dir}", 1)
        return scored, best


# =============================================================================
# Generalised model × dataset driver with error handling
# =============================================================================

def _results_index_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "results_index.csv"

def update_results_index(checkpoint_dir: Path, result_csv: Path, model_name: str, dataset_name: str) -> None:
    """
    Append or update a row in the results index CSV.
    The index maps hashed result filenames to model/dataset.
    """
    index_path = _results_index_path(checkpoint_dir)
    row = {
        "result_filename": result_csv.name,
        "model": model_name,
        "dataset": dataset_name,
        "saved_at": time.time(),
    }
    if index_path.exists():
        df = pd.read_csv(index_path)
        # Remove any existing row with same filename
        df = df[df["result_filename"] != result_csv.name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(index_path, index=False)

def load_results_index(checkpoint_dir: Path) -> pd.DataFrame:
    index_path = _results_index_path(checkpoint_dir)
    if index_path.exists():
        return pd.read_csv(index_path)
    return pd.DataFrame(columns=["result_filename", "model", "dataset", "saved_at"])

def lookup_result_by_hash(checkpoint_dir: Path, hash_or_filename: str) -> dict:
    """
    Given a hashed filename (e.g., '85063c5df2ed_layer_probe_results.csv')
    or just the hash, return a dict with model, dataset, and full path.
    Returns None if not found.
    """
    index = load_results_index(checkpoint_dir)
    # Allow either the full filename or just the hash prefix
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

def run_matrix(
    entries: Sequence[Mapping[str, Any]],
    *,
    external_root: Path,
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
    """
    Run the same probe benchmark across arbitrary frozen model artifacts, with
    automatic checkpointing and resume.

    Each entry must provide model and dataset names and optionally the dataset_df
    and contract. After each successful entry, the full per-layer results are saved
    to a CSV file and the checkpoint file is updated. If the process is restarted,
    already completed entries are skipped and their saved results are loaded.

    Parameters:
        checkpoint_dir: Directory where per-entry results and the checkpoint file
                        will be stored. If None, uses
                        external_root/experiments/experiment_id/matrix_checkpoint.

    Returns:
        Full per-layer DataFrame with columns for model, dataset, artifact_dir,
        metadata_path, and all probe metrics. The order of rows corresponds to
        the original entries order (completed ones first, then new ones).
    """
    split = split or SplitConfig(train=0.80, validation=0.10, test=0.10, seed=42)
    if checkpoint_dir is None:
        checkpoint_dir = external_root / "experiments" / experiment_id / "matrix_checkpoint"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "probe_matrix_checkpoint.json"
    results_subdir = checkpoint_dir / "per_entry_results"
    results_subdir.mkdir(exist_ok=True)

    # Load existing checkpoint if present
    checkpoint = {"completed": {}, "errors": []}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
        except Exception:
            if verbose >= 1:
                print(f"[checkpoint] Could not load checkpoint file {checkpoint_file}; starting fresh.")

    # Map each entry to a unique key
    entry_keys = [f"{str(entry['model'])}::{str(entry['dataset'])}" for entry in entries]

    # For each entry, either load its saved results or run it fresh
    per_entry_results = []   # list of DataFrames in final order
    error_records = []

    for i, (entry, key) in enumerate(zip(entries, entry_keys), start=1):
        model_name = str(entry["model"])
        dataset_name = str(entry["dataset"])

        # Determine the result CSV path for this entry
        result_csv = results_subdir / f"{stable_hash(f'{model_name}::{dataset_name}', 12)}_layer_probe_results.csv"
        
        if key in checkpoint.get("completed", {}):
            # Load from saved CSV
            if verbose >= 1:
                print(f"[checkpoint] {i}/{len(entries)} | {model_name} | {dataset_name} : already completed, loading from {result_csv.name}")
            if result_csv.exists():
                try:
                    df = pd.read_csv(result_csv)
                    per_entry_results.append(df)
                except Exception as e:
                    print(f"[checkpoint] Failed to load {result_csv}: {e}. Will re-run this entry.")
                    checkpoint["completed"].pop(key, None)
                    # fall through to run
                else:
                    continue
            else:
                print(f"[checkpoint] Checkpoint says {key} completed but result file missing. Re-running.")
                checkpoint["completed"].pop(key, None)

        # If we get here, we need to run the entry
        if verbose >= 1:
            print(f"[matrix] {i}/{len(entries)} | {model_name} | {dataset_name}")

        try:
            art = ExtractionArtifact(dataset_dir_from_args(external_root, experiment_id, model_name, dataset_name))
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
            out_dir = (
                dataset_dir_from_args(external_root, experiment_id, model_name, dataset_name)
                / "analysis" / "probes" / "matrix_runs"
                / f"unified_v4_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            analyzer = UnifiedProbeAnalyzer(art, cfg, out_dir, dataset_df=entry.get("dataset_df"))
            scored, _ = analyzer.run()   # full per-layer results

            # Add identification columns
            scored = scored.copy()
            scored["model"] = model_name
            scored["dataset"] = dataset_name
            scored["artifact_dir"] = str(out_dir)
            scored["metadata_path"] = str(out_dir / "complete_run_metadata.json")

            # Save the per-entry full results to CSV
            scored.to_csv(result_csv, index=False)
            update_results_index(checkpoint_dir, result_csv, model_name, dataset_name)
            if verbose >= 1:
                print(f"[checkpoint] Saved {result_csv.name}")

            per_entry_results.append(scored)

            # Update checkpoint
            checkpoint["completed"][key] = {
                "model": model_name,
                "dataset": dataset_name,
                "result_csv": str(result_csv),
                "completed_at": time.time(),
            }
            _save_checkpoint(checkpoint_file, checkpoint)

        except Exception as e:
            if verbose >= 0:
                print(f"[matrix] ERROR for {model_name}/{dataset_name}: {type(e).__name__}: {e}")
            error_records.append({
                "model": model_name,
                "dataset": dataset_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status": "failed",
            })

    # Combine all results in the original entries order
    if per_entry_results:
        full_df = pd.concat(per_entry_results, ignore_index=True)
    else:
        full_df = pd.DataFrame()

    # Save error log (optional)
    if error_records:
        error_df = pd.DataFrame(error_records)
        error_csv = checkpoint_dir / "probe_errors.csv"
        error_df.to_csv(error_csv, index=False)
        if verbose >= 0:
            print(f"[matrix] {len(error_records)} entries failed. See {error_csv}.")

    return full_df

def _save_checkpoint(checkpoint_file: Path, checkpoint: dict) -> None:
    """Write checkpoint atomically."""
    tmp = checkpoint_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)
    tmp.replace(checkpoint_file)
    
def collect_layer_results(
    external_root: Path,
    experiment_id: str,
    entries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
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
            df["artifact_dir"] = str(run_dir.parent)   # directory containing the CSV
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def dataset_dir_from_args(external_root: Path, experiment_id: str, model_name: str, dataset_name: str) -> Path:
    model_path = Path(*[p for p in model_name.split("/") if p])
    return external_root / "experiments" / experiment_id / "models" / model_path / "datasets" / dataset_name

def plot_full_dashboard(full_results: pd.DataFrame, output_root: Path):
    """Generate publication‑quality plots from full layer‑wise probe results."""
    if full_results.empty:
        print("No results to plot.")
        return

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Layer curves for test Macro-F1 and Unified Score (one panel per model/dataset)
    for (model, dataset), group in full_results.groupby(["model", "dataset"]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, metric, title in [
            (axes[0], "test_macro_f1", "Test Macro-F1"),
            (axes[1], "probe_score", "Unified Probe Score"),
        ]:
            for probe_name in group["probe"].unique():
                sub = group[group["probe"] == probe_name].sort_values("layer_index")
                ax.plot(sub["layer_index"], sub[metric], marker="o", label=probe_name)
            ax.set_xlabel("Layer index")
            ax.set_ylabel(title)
            ax.set_title(f"{title} – {model} / {dataset}")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_root / f"layer_curves_{model.replace('/', '_')}_{dataset}.png", dpi=240)
        plt.show()

    # 2. Heatmap: probe × layer for test Macro-F1 averaged over repeats and models/datasets
    pivot = full_results.pivot_table(index="probe", columns="layer_index", values="test_macro_f1", aggfunc="mean")
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Macro-F1"})
    plt.title("Test Macro-F1 Heatmap (all models/datasets)")
    plt.xlabel("Layer")
    plt.ylabel("Probe")
    plt.tight_layout()
    plt.savefig(output_root / "heatmap_macro_f1.png", dpi=240)
    plt.show()

    # 3. Confusion matrix for the best probe-layer combination (across all)
    best_row = full_results.loc[full_results["test_macro_f1"].idxmax()]
    model, dataset, probe_name, layer_idx = best_row["model"], best_row["dataset"], best_row["probe"], int(best_row["layer_index"])
    artifact_dir = Path(best_row["artifact_dir"])
    # Locate confusion matrix file
    cm_file = artifact_dir / "models" / probe_name / f"layer_{layer_idx}" / "repeat_0" / "confusion_matrix_test.npz"
    if cm_file.exists():
        data = np.load(cm_file)
        cm = data["matrix"]
        # We need class names; retrieve from metadata or classification report
        # For simplicity, we can load from a saved metrics.json if available
        metrics_file = artifact_dir / "models" / probe_name / f"layer_{layer_idx}" / "repeat_0" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            classes = metrics.get("record", {}).get("classes", [str(i) for i in range(cm.shape[0])])
        else:
            classes = [str(i) for i in range(cm.shape[0])]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix – {model}/{dataset} – {probe_name} @ Layer {layer_idx}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(output_root / "confusion_matrix_best.png", dpi=240)
        plt.show()

    # 4. Control comparison: true vs shuffled macro-F1 per layer (average over all)
    # We need control data; if not in full_results, load from CSV files
    control_frames = []
    for run_dir in full_results["artifact_dir"].unique():
        ctrl_file = Path(run_dir) / "shuffled_label_controls.csv"
        if ctrl_file.exists():
            ctrl = pd.read_csv(ctrl_file)
            ctrl["model"] = best_row["model"] if "model" not in ctrl.columns else ctrl["model"]
            ctrl["dataset"] = best_row["dataset"] if "dataset" not in ctrl.columns else ctrl["dataset"]
            control_frames.append(ctrl)
    if control_frames:
        control_df = pd.concat(control_frames, ignore_index=True)
        plt.figure(figsize=(12, 6))
        for probe_name in control_df["probe"].unique():
            sub_ctrl = control_df[control_df["probe"] == probe_name].groupby("layer_index")["control_test_macro_f1"].mean()
            plt.plot(sub_ctrl.index, sub_ctrl.values, linestyle="--", marker="x", label=f"{probe_name} (shuffled)")
        # Also plot true macro-f1 from full_results for comparison
        for probe_name in full_results["probe"].unique():
            sub_true = full_results[full_results["probe"] == probe_name].groupby("layer_index")["test_macro_f1"].mean()
            plt.plot(sub_true.index, sub_true.values, linestyle="-", marker="o", label=f"{probe_name} (true)")
        plt.xlabel("Layer index")
        plt.ylabel("Macro-F1")
        plt.title("True vs Shuffled Label Controls")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "control_comparison.png", dpi=240)
        plt.show()

def discover_model_dataset_pairs(
    external_root: Path,
    experiment_id: str,
    model_names: Sequence[str] | None = None,
    dataset_names: Sequence[str] | None = None,
) -> list[dict]:
    """Discover existing extraction directories for given models/datasets."""
    pairs = []
    exp_root = external_root / "experiments" / experiment_id / "models"
    if not exp_root.exists():
        return pairs

    # Find all model directories
    for model_dir in exp_root.glob("*/*"):  # handles nested models like google-bert/bert-base-uncased
        model_name = "/".join(model_dir.relative_to(exp_root).parts)
        if model_names and model_name not in model_names:
            continue
        for dataset_dir in (model_dir / "datasets").glob("*"):
            if (dataset_dir / "metadata" / "extraction.json").exists():
                dataset_name = dataset_dir.name
                if dataset_names and dataset_name not in dataset_names:
                    continue
                pairs.append({
                    "model": model_name,
                    "dataset": dataset_name,
                })
    return pairs


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Hidden-State Probe v4.2")
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
    output_dir = dataset_dir / config.output_subdir / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{stable_hash(asdict(config), 10)}"
    analyzer = UnifiedProbeAnalyzer(artifact, config, output_dir)

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
    cols = [c for c in [
        "probe", "layer_index", "probe_score_mean", "test_macro_f1_mean",
        "test_balanced_accuracy_mean", "test_mcc_mean", "selectivity_mean",
    ] if c in best.columns]
    print(best[cols].to_string(index=False))
    print("\nOutputs:", analyzer.output_dir)


if __name__ == "__main__":
    main()