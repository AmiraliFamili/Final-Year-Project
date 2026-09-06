#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMOTION PROBE LAB – UNIFIED PIPELINE
====================================

A complete, self‑contained orchestrator for hidden‑state emotion probing.

This file provides a single entry point for:
  • Full end‑to‑end pipelines (extraction – optional, probing, analysis, reporting)
  • Forensic pre‑flight checks (provenance, split integrity, artifact completeness)
  • Task‑aware evaluation (single‑label vs multi‑label)
  • Beautiful terminal UI (using `rich` if available)
  • Interactive guided mode
  • CLI with subcommands: run, audit, analyse, interactive, models, compare
  • Rich visualisations (layer curves, heatmaps, control comparisons, …)

All heavy computation is delegated to existing project modules:
  Extraction6_new (or Extraction6/Extraction5) and
  unified_hidden_state_probe_v4_3 (or v4_2).

The pipeline is designed for reproducibility, scientific integrity, and ease of use.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Optional, Tuple, List, Dict, Union
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Optional dependencies for enhanced user experience
# ------------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, ProgressColumn
    )
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# ============================================================================
# USER GUIDE (expanded)
# ============================================================================
"""
COMMAND USAGE
=============

This script provides several subcommands for the emotion probe pipeline.
Each subcommand can be abbreviated (shortcuts shown in parentheses).

1. run (r)
   Execute a full experiment.
   Usage:
     python G_test.py run --model MODEL --dataset {goEmo,ISEAR} [options]
   Options:
     -m, --model MODEL         Hugging Face model name (full or alias)
     -d, --dataset {goEmo,ISEAR}  Dataset to probe
     -R, --root PATH           Root directory for artifacts (default: /Volumes/Amirali/hidden_states)
     -e, --experiment-id ID    Experiment identifier (default: baseline_v5_001)
     -o, --output-dir PATH     Output directory for local results/plots
     -p, --probe {logistic,mlp1,mlp2,mlp3}  Probe type (can be repeated for comparison)
     -r, --repeats N           Number of independent train/test splits (default: 3)
     -n, --max-samples N       Maximum samples to use (default: 5000)
     -f, --full                Use all samples (overrides --max-samples)
     -s, --seed N              Random seed (default: 42)
     -S, --no-shuffle-control  Disable shuffled-label control
     -x, --extract-if-missing  Run extraction if artifact is missing
     -P, --no-plot             Skip generating plots
     -q, --quiet               Suppress all non-essential output (final summary still shown)

2. models (m)
   List available models and their artifact availability.
   Usage: python G_test.py models

3. interactive (i)
   Launch guided interactive mode.
   Usage: python G_test.py interactive

4. audit (a)
   Audit an existing run directory without retraining.
   Usage: python G_test.py audit RUN_DIR

5. analyse (an)
   Generate plots from one or more result CSV files (or directories containing CSVs).
   Usage: python G_test.py analyse -i CSV1 [-i CSV2 ...] [-o OUTPUT_DIR]
   Options:
     -i, --input FILE         Input CSV file (repeatable) or directory
     -o, --output-dir DIR     Output directory for plots (default: ./analysis_plots)

6. compare (c)
   Compare multiple completed runs side‑by‑side.
   Usage: python G_test.py compare RUN_DIR1 RUN_DIR2 ... [-o OUTPUT_DIR]

EXAMPLES
--------
# Run a standard experiment with logistic probe
python G_test.py run --model google-bert/bert-base-uncased --dataset goEmo

# Run with two probes (logistic and MLP) for comparison
python G_test.py run --model Qwen/Qwen2-0.5B --dataset ISEAR --probe logistic --probe mlp1 --extract-if-missing

# List models and their artifact status
python G_test.py models

# Launch interactive menu
python G_test.py interactive

# Audit a previous run
python G_test.py audit ./demo_runs/goEmo__bert-base-uncased/abc123def456

# Generate plots from a master run directory
python G_test.py analyse -i /path/to/master_run_dir -o ./my_plots

# Compare two completed runs
python G_test.py compare ./run1 ./run2 -o ./comparison_plots

# Run silently (only final result table is printed)
python G_test.py run --model Qwen/Qwen2-0.5B --dataset ISEAR --quiet
"""
# ============================================================================

# ============================================================================
# 1. Constants and configurations
# ============================================================================

MASTER_VERSION = "2.1.0"

# Default paths – adjust to your environment
DEFAULT_ROOT = Path("/Volumes/Amirali/hidden_states")
DEFAULT_EXPERIMENT_ID = "baseline_v5_001"
DEFAULT_OUTPUT_DIR = Path("./demo_runs")

# Module candidates – the first importable one will be used
DEFAULT_EXTRACTION_MODULES = (
    "Extraction",
    "Extraction6_new",
    "Extraction6",
    "Extraction5",
)
DEFAULT_PROBE_MODULES = (
    "Probe",
    "unified_hidden_state_probe_v4_3",
    "unified_hidden_state_probe_v4_2",
)

# Supported datasets and their metadata
DATASET_SPECS = {
    "goEmo": {
        "label": "GoEmotions",
        "module": "Get_Go_Emo",
        "function": "get_go",
        "task_type": "multi_label",
        "target_type": "goemotions",
        "class_count": 28,
    },
    "ISEAR": {
        "label": "ISEAR",
        "module": "Get_Isear",
        "function": "get_isr",
        "task_type": "single_label",
        "target_type": "isear",
        "class_count": 7,
    },
}

# Static model aliases – short names to full Hugging Face IDs
MODEL_ALIASES = {
    "BERT": "google-bert/bert-base-uncased",
    "DBERT": "distilbert/distilbert-base-uncased",
    "DISTILBERT": "distilbert/distilbert-base-uncased",
    "ROBERTA": "FacebookAI/roberta-base",
    "ELECTRA": "google/electra-small-discriminator",
    "DEBERTA": "microsoft/deberta-v3-small",
    "GPT2": "gpt2",
    "GPT-NEO": "EleutherAI/gpt-neo-125m",
    "OPT": "facebook/opt-125m",
    "SMOL2-135M": "HuggingFaceTB/SmolLM2-135M",
    "SMOL2-360M": "HuggingFaceTB/SmolLM2-360M",
    "SMOL2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "GEMMA-270M": "google/gemma-3-270m",
    "GEMMA-1B": "google/gemma-3-1b-pt",
    "GEMMA-4B": "google/gemma-3-4b-pt",
    "QWEN2-0.5B": "Qwen/Qwen2-0.5B",
    "QWEN2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "QWEN2-1.5B": "Qwen/Qwen2-1.5B",
    "QWEN2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "QWEN2.5-3B": "Qwen/Qwen2.5-3B",
    "QWEN3-0.6B": "Qwen/Qwen3-0.6B-Base",
    "QWEN3-1.7B": "Qwen/Qwen3-1.7B-Base",
    "QWEN3-4B": "Qwen/Qwen3-4B-Base",
    "LLAMA-1B": "meta-llama/Llama-3.2-1B",
    "LLAMA-3B": "meta-llama/Llama-3.2-3B",
    "TINYLLAMA": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}

def build_reverse_alias(aliases: dict) -> dict:
    """
    Build a mapping from full model name to its primary short alias.

    Args:
        aliases: Dictionary of short alias -> full name.

    Returns:
        Dictionary of full name -> short alias.
    """
    reverse = {}
    for short, full in aliases.items():
        if full not in reverse:
            reverse[full] = short
    return reverse


FULL_TO_SHORT = build_reverse_alias(MODEL_ALIASES)


def resolve_model_name(raw: str) -> str:
    """
    Resolve a user-provided model name to a full registry name.

    Accepts full name, short alias (case-insensitive), or abbreviation.
    Raises ValueError if not resolvable.

    Args:
        raw: User input string.

    Returns:
        Canonical model name (full Hugging Face ID).
    """
    raw = raw.strip()
    if not raw:
        return "google-bert/bert-base-uncased"

    alias_key = raw.upper()
    if alias_key in MODEL_ALIASES:
        return MODEL_ALIASES[alias_key]

    for full in MODEL_ALIASES.values():
        if raw.lower() == full.lower():
            return full

    matches = []
    for alias, full in MODEL_ALIASES.items():
        if raw.lower() in alias.lower() or raw.lower() in full.lower():
            matches.append(full)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = "\n".join([f"  - {m} (alias: {FULL_TO_SHORT.get(m, 'N/A')})" for m in matches])
        raise ValueError(f"Ambiguous model name '{raw}'. Please choose one of:\n{choices}")
    valid = "\n".join([f"  - {alias}: {full}" for alias, full in MODEL_ALIASES.items()])
    raise ValueError(f"Model '{raw}' not recognized. Valid names:\n{valid}")

# Predefined probe configurations
DEFAULT_PROBE_PRESETS = {
    "logistic": {
        "name": "linear_logistic",
        "type": "logistic",
        "complexity": "linear",
        "standardize": True,
        "C": 1.0,
        "max_iter": 3000,
        "selection_metric": "macro_f1",
    },
    "mlp1": {
        "name": "mlp_1_hidden",
        "type": "mlp",
        "complexity": "1_hidden",
        "hidden_dims": ["0.5d"],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 80,
        "batch_size": 256,
        "patience": 12,
        "dropout": 0.0,
        "selection_metric": "macro_f1",
    },
    "mlp2": {
        "name": "mlp_2_hidden",
        "type": "mlp",
        "complexity": "2_hidden",
        "hidden_dims": ["0.5d", "0.25d"],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 80,
        "batch_size": 256,
        "patience": 12,
        "dropout": 0.0,
        "selection_metric": "macro_f1",
    },
    "mlp3": {
        "name": "mlp_3_hidden",
        "type": "mlp",
        "complexity": "3_hidden",
        "hidden_dims": ["0.5d", "0.25d", "0.125d"],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 80,
        "batch_size": 256,
        "patience": 12,
        "dropout": 0.0,
        "selection_metric": "macro_f1",
    },
}


# ============================================================================
# 2. Utility functions
# ============================================================================

def stable_hash(value: Any, length: int = 16) -> str:
    """
    Deterministic hash of a Python object (JSON‑serialisable).

    Args:
        value: Any JSON‑serialisable object.
        length: Desired hash length (default 16).

    Returns:
        Hexadecimal hash string.
    """
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Atomically write a JSON file (write to temporary, then replace).

    Args:
        path: Destination file path.
        payload: Dictionary to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def normalize_model_name(name: str) -> str:
    """Strip whitespace from a model name."""
    return str(name).strip()


def safe_model_path(model_name: str) -> Path:
    """
    Convert a Hugging Face model name (e.g. 'org/name') to a safe relative path.

    Example: 'google-bert/bert-base-uncased' -> Path('google-bert/bert-base-uncased')
    """
    return Path(*[p for p in model_name.split("/") if p])


def dataset_artifact_dir(
    root: Path,
    experiment_id: str,
    model_name: str,
    dataset_name: str,
) -> Path:
    """
    Return the expected location of the frozen hidden‑state artifact.

    The artifact is stored under:
        root/experiments/experiment_id/models/model_name/datasets/dataset_name
    """
    return (
        root / "experiments" / experiment_id / "models"
        / safe_model_path(model_name) / "datasets" / dataset_name
    )


def import_first(candidates: Sequence[str]):
    """
    Import the first module that can be loaded from a list of candidates.

    Args:
        candidates: Sequence of module names.

    Returns:
        The imported module.

    Raises:
        ImportError: If none of the candidates can be imported.
    """
    errors = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ImportError(
        "Could not import any candidate module:\n- " + "\n- ".join(errors)
    )


def compact_path(path: Path, max_chars: int = 74) -> str:
    """
    Shorten a path for display if it exceeds max_chars.

    Args:
        path: Path object.
        max_chars: Maximum length before truncation.

    Returns:
        String representation, possibly shortened with '…'.
    """
    s = str(path)
    return s if len(s) <= max_chars else "…" + s[-(max_chars - 1):]


# ------------------------------------------------------------------------------
# NEW: Functions for listing available models and artifacts
# ------------------------------------------------------------------------------
def get_run_label(run_dir: Path) -> str:
    """
    Generate a compact human‑readable label for a run directory.
    Format: <model>_<dataset>_<short_hash>
    """
    run_dir = run_dir.resolve()
    parts = run_dir.parts

    # Try to find dataset and model from path structure
    # master_runs/<dataset>__<org>/<model>/<hash>
    if "master_runs" in parts:
        idx = parts.index("master_runs")
        # parts after idx: dataset__org, model_name, hash
        if len(parts) > idx + 2:
            dataset_org = parts[idx + 1]      # e.g., "ISEAR__Qwen"
            model_name = parts[idx + 2]       # e.g., "Qwen2-0.5B"
            hash_part = parts[idx + 3] if len(parts) > idx + 3 else ""
            if "__" in dataset_org:
                dataset, org = dataset_org.split("__", 1)
                full_model = f"{org}/{model_name}"
            else:
                dataset = dataset_org
                full_model = model_name
            short_hash = hash_part[:6] if hash_part else ""
            # Replace slashes to keep label filesystem-safe
            label = f"{full_model.replace('/', '_')}_{dataset}_{short_hash}"
            return label

    # Fallback for probe output directories
    # .../models/<org>/<model>/datasets/<dataset>/analysis/probes/<run_dir>
    if "models" in parts:
        idx = parts.index("models")
        if len(parts) > idx + 3:
            org = parts[idx + 1]
            model_name = parts[idx + 2]
            if parts[idx + 3] == "datasets" and len(parts) > idx + 4:
                dataset = parts[idx + 4]
                full_model = f"{org}/{model_name}"
                # Use last directory name as hash identifier
                hash_part = parts[-1]
                short_hash = hash_part[-6:] if hash_part else ""
                label = f"{full_model.replace('/', '_')}_{dataset}_{short_hash}"
                return label

    # If all else fails, use the directory name truncated
    return run_dir.name[:20]


def get_available_models(extraction_mod=None):
    """
    Return a list of (model_name, family, params) from the extraction module's registry.

    If no extraction module is given, it tries to import one using ModuleFactory.
    Returns an empty list if no registry is found.
    """
    if extraction_mod is None:
        extraction_mod = ModuleFactory.extraction(MasterConfig())
    if hasattr(extraction_mod, "MODEL_REGISTRY"):
        return [
            (m.name, m.family, m.parameter_billions)
            for m in extraction_mod.MODEL_REGISTRY
        ]
    return []


def has_artifact(root: Path, experiment_id: str, model: str, dataset: str) -> bool:
    """
    Check whether a frozen hidden‑state artifact exists for the given model and dataset.
    """
    ds_dir = dataset_artifact_dir(root, experiment_id, model, dataset)
    return (ds_dir / "metadata" / "extraction.json").exists()


def list_analysable_runs(root: Path = DEFAULT_ROOT, experiment_id: str = DEFAULT_EXPERIMENT_ID) -> None:
    """
    List all directories that contain analysable probe result CSVs.
    Scans:
      - master_runs/<dataset>__<org>/<model>/<hash>/           -> master_results.csv
      - experiments/<exp_id>/models/<org>/<model>/datasets/<dataset>/analysis/probes/<run_dir>/ -> layer_probe_results.csv
    Displays model, dataset, run type, directory (truncated to 100 chars), and config hash if available.
    """
    renderer = Renderer()
    entries = []

    # Scan master runs
    master_runs = root / "master_runs"
    if master_runs.exists():
        for master_csv in master_runs.glob("**/master_results.csv"):
            run_dir = master_csv.parent
            # Relative path from master_runs: e.g., "goEmo__HuggingFaceTB/SmolLM2-360M/9cf..."
            parts = run_dir.relative_to(master_runs).parts
            if len(parts) >= 3:
                dataset_org = parts[0]           # "goEmo__HuggingFaceTB"
                model_name = parts[1]            # "SmolLM2-360M"
                config_hash = parts[2]
                # dataset_org can itself contain "__" if model org has slash?
                # Actually dataset_org is dataset__org (org may have no slash)
                if "__" in dataset_org:
                    dataset, org = dataset_org.split("__", 1)
                    full_model = f"{org}/{model_name}"
                else:
                    dataset = dataset_org
                    full_model = model_name
                entries.append({
                    "model": full_model,
                    "dataset": dataset,
                    "run_type": "master",
                    "path": str(run_dir),
                    "config_hash": config_hash,
                })

    # Scan probe output directories
    exp_models = root / "experiments" / experiment_id / "models"
    if exp_models.exists():
        for probe_csv in exp_models.glob("**/layer_probe_results.csv"):
            run_dir = probe_csv.parent
            # Relative path from exp_models: <org>/<model>/datasets/<dataset>/analysis/probes/<run_dir>
            parts = run_dir.relative_to(exp_models).parts
            if len(parts) >= 5 and parts[2] == "datasets":
                org = parts[0]
                model_name = parts[1]
                dataset = parts[3]
                full_model = f"{org}/{model_name}"
                # Try to extract config hash from run_dir name (optional)
                config_hash = ""
                # Run_dir name often contains "__hash<hash>"
                m = re.search(r"__hash([a-f0-9]+)", run_dir.name)
                if m:
                    config_hash = m.group(1)
                entries.append({
                    "model": full_model,
                    "dataset": dataset,
                    "run_type": "probe",
                    "path": str(run_dir),
                    "config_hash": config_hash,
                })

    if not entries:
        renderer.warning("No analysable runs found.")
        return

    # Sort by model, dataset, run_type
    entries.sort(key=lambda x: (x["model"], x["dataset"], x["run_type"]))

    if RICH_AVAILABLE:
        table = Table(title="Analysable Runs", show_lines=True, header_style="bold magenta", width=150)
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Model", style="cyan", min_width=20, no_wrap=True)
        table.add_column("Dataset", style="green", min_width=10)
        table.add_column("Type", style="yellow", width=8)
        table.add_column("Run Directory", style="white", min_width=40, max_width=100, overflow="fold")
        table.add_column("Config Hash", style="yellow", min_width=12)
        for i, entry in enumerate(entries, 1):
            table.add_row(
                str(i),
                entry["model"],
                entry["dataset"],
                entry["run_type"],
                entry["path"],
                entry["config_hash"],
            )
        renderer.console.print(table)
    else:
        print("\nAnalysable Runs:")
        for i, entry in enumerate(entries, 1):
            print(f"{i:3d}. {entry['model']:30s} {entry['dataset']:10s} {entry['run_type']:6s} {entry['path']}")


def list_auditable_runs(root: Path = DEFAULT_ROOT, experiment_id: str = DEFAULT_EXPERIMENT_ID) -> None:
    """
    List all probe run directories that contain a probe_run_manifest.json.
    Displays model (org/name), dataset, and run directory.
    """
    renderer = Renderer()
    exp_models = root / "experiments" / experiment_id / "models"
    if not exp_models.exists():
        renderer.warning(f"No experiment directory found at {exp_models}.")
        return

    entries = []
    for manifest_path in exp_models.glob("**/probe_run_manifest.json"):
        run_dir = manifest_path.parent
        parts = run_dir.relative_to(exp_models).parts
        # Expected: <org>/<model>/datasets/<dataset>/analysis/probes/<run_dir>
        if len(parts) >= 5 and parts[2] == "datasets":
            org = parts[0]
            model_name = parts[1]
            dataset = parts[3]
            full_model = f"{org}/{model_name}"
            entries.append({
                "model": full_model,
                "dataset": dataset,
                "path": str(run_dir),
            })

    if not entries:
        renderer.warning("No auditable runs found.")
        return

    entries.sort(key=lambda x: (x["model"], x["dataset"]))

    if RICH_AVAILABLE:
        table = Table(title="Auditable Runs", show_lines=True, header_style="bold magenta", width=150)
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Model", style="cyan", min_width=20, no_wrap=True)
        table.add_column("Dataset", style="green", min_width=10)
        table.add_column("Run Directory", style="white", min_width=40, max_width=100, overflow="fold")
        for i, entry in enumerate(entries, 1):
            table.add_row(str(i), entry["model"], entry["dataset"], entry["path"])
        renderer.console.print(table)
    else:
        print("\nAuditable Runs:")
        for i, entry in enumerate(entries, 1):
            print(f"{i:3d}. {entry['model']:30s} {entry['dataset']:10s} {entry['path']}")

# ============================================================================
# 3. Configuration dataclasses
# ============================================================================

@dataclass
class ProbeChoice:
    """
    A user‑specified probe configuration.

    Attributes:
        preset: Key in DEFAULT_PROBE_PRESETS.
        overrides: Optional dict of hyperparameters to override.
    """
    preset: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class MasterConfig:
    """
    Complete configuration for a pipeline run.

    All parameters are frozen into a deterministic hash to create a unique
    run directory, ensuring reproducibility.
    """
    root: Path = DEFAULT_ROOT
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    model: str | None = None
    dataset: str | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # Scientific parameters
    layers: str | list[int] = "all"
    repeats: int = 3
    max_samples: int | None = 5000

    split_train: float = 0.80
    split_validation: float = 0.10
    split_test: float = 0.10
    seed: int = 42

    shuffled_label_control: bool = True
    shuffled_control_repeats: int = 3

    pca_enabled: bool = True
    silhouette_enabled: bool = True
    pca_samples: int = 3000
    silhouette_samples: int = 3000

    verify_checksum: bool = False
    strict_provenance: bool = True

    # Module overrides (optional)
    extraction_module: str | None = None
    probe_module: str | None = None

    quiet: bool = False

    # List of probes to run
    probes: list[ProbeChoice] = field(
        default_factory=lambda: [ProbeChoice("logistic")]
    )

    def as_dict(self) -> dict[str, Any]:
        """Convert to a JSON‑serialisable dictionary (paths as strings)."""
        d = asdict(self)
        d["root"] = str(self.root)
        d["output_dir"] = str(self.output_dir)
        return d

    @property
    def config_hash(self) -> str:
        """Unique hash of the full configuration (used for run directory)."""
        return stable_hash(self.as_dict(), 20)


# ============================================================================
# 4. Factories for dynamic module and dataset loading
# ============================================================================

class ModuleFactory:
    """Resolves project implementation modules without hard-coding filenames."""

    @staticmethod
    def extraction(config: MasterConfig):
        """
        Return an extraction module, or None if none can be imported.

        Uses the following resolution order:
            1. config.extraction_module (if provided)
            2. DEFAULT_EXTRACTION_MODULES (in order)
            3. None (if none found)
        """
        if config.extraction_module:
            try:
                return importlib.import_module(config.extraction_module)
            except ImportError:
                return None
        for name in DEFAULT_EXTRACTION_MODULES:
            try:
                return importlib.import_module(name)
            except ImportError:
                continue
        return None  
    
    @staticmethod
    def probe(config: MasterConfig):
        """Return the probing module (e.g., unified_hidden_state_probe_v4_3)."""
        if config.probe_module:
            return importlib.import_module(config.probe_module)
        return import_first(DEFAULT_PROBE_MODULES)


class DatasetFactory:
    """Adapter for loading datasets from the existing project modules."""

    @staticmethod
    def spec(name: str) -> dict[str, Any]:
        """Return the metadata dictionary for a given dataset name."""
        key = str(name)
        if key not in DATASET_SPECS:
            raise KeyError(
                f"Unsupported dataset {key!r}. Available: {sorted(DATASET_SPECS)}"
            )
        return dict(DATASET_SPECS[key])

    @staticmethod
    def load(name: str) -> Any:
        """Load the dataset (returns a pandas DataFrame or similar)."""
        spec = DatasetFactory.spec(name)
        module = importlib.import_module(spec["module"])
        fn = getattr(module, spec["function"])
        return fn()


class ProbeFactory:
    """Creates ProbeSpec objects for the installed probe module."""

    @staticmethod
    def create(probe_module: Any, choices: Sequence[ProbeChoice]):
        """
        Instantiate probe specification objects.

        Args:
            probe_module: The imported probe module.
            choices: List of ProbeChoice.

        Returns:
            List of ProbeSpec objects.
        """
        ProbeSpec = getattr(probe_module, "ProbeSpec")
        probes = []

        for choice in choices:
            key = choice.preset.lower()
            if key not in DEFAULT_PROBE_PRESETS:
                raise ValueError(
                    f"Unknown probe preset {choice.preset!r}. "
                    f"Available: {sorted(DEFAULT_PROBE_PRESETS)}"
                )
            payload = dict(DEFAULT_PROBE_PRESETS[key])
            payload.update(choice.overrides)
            probes.append(ProbeSpec(**payload))

        if not probes:
            raise ValueError("At least one probe must be selected.")
        return probes


# ============================================================================
# 5. Pipeline state machine
# ============================================================================

class Stage:
    """Constants for pipeline stages."""
    PREFLIGHT = "PREFLIGHT"
    EXTRACTION = "EXTRACTION"
    PROBING = "PROBING"
    ANALYSIS = "ANALYSIS"
    REPORT = "REPORT"
    COMPLETE = "COMPLETE"


@dataclass
class StageRecord:
    """Record of a single stage's execution status."""
    name: str
    status: str = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    message: str = ""

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at


class PipelineState:
    """
    Persistent, human‑readable state machine record.

    Saves a JSON file in the run directory that can be inspected after
    a failure or interruption.
    """

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, StageRecord] = {}

    def start(self, stage: str) -> None:
        """Mark a stage as started."""
        self.records[stage] = StageRecord(
            name=stage, status="running", started_at=time.time()
        )
        self.save()

    def finish(self, stage: str, message: str = "") -> None:
        """Mark a stage as completed."""
        rec = self.records.setdefault(stage, StageRecord(name=stage))
        rec.status = "complete"
        rec.finished_at = time.time()
        rec.message = message
        self.save()

    def fail(self, stage: str, message: str) -> None:
        """Mark a stage as failed."""
        rec = self.records.setdefault(stage, StageRecord(name=stage))
        rec.status = "failed"
        rec.finished_at = time.time()
        rec.message = message
        self.save()

    def save(self) -> None:
        """Write the state to a JSON file."""
        save_json(
            self.path,
            {"records": {k: asdict(v) for k, v in self.records.items()}},
        )


# ============================================================================
# 6. Terminal renderer (with optional Rich support)
# ============================================================================

class Renderer:
    """
    Handles all terminal output.

    If `rich` is installed, it uses rich panels, tables, and progress bars.
    Otherwise falls back to simple print statements.
    """

    def __init__(self, silent: bool = False):
        self.silent = silent
        self.console = None if silent else (Console() if RICH_AVAILABLE else None)

    def title(self, title: str, subtitle: str = "") -> None:
        """Display a prominent title."""
        if self.silent: return
        if self.console:
            body = Text(subtitle) if subtitle else ""
            self.console.print(Panel(body, title=title, expand=False))
        else:
            print("\n" + "=" * 88)
            print(title)
            if subtitle:
                print(subtitle)
            print("=" * 88)

    def info(self, text: str) -> None:
        """Display an informational message."""
        if self.silent: return
        if self.console:
            self.console.print(text)
        else:
            print(text)

    def warning(self, text: str) -> None:
        """Display a warning message."""
        if self.silent: return
        if self.console:
            self.console.print(f"[yellow]WARNING[/yellow] {text}")
        else:
            print(f"WARNING: {text}")

    def success(self, text: str) -> None:
        """Display a success message."""
        if self.console:
            self.console.print(f"[green]✓[/green] {text}")
        else:
            print(f"✓ {text}")

    def error(self, text: str) -> None:
        """Display an error message."""
        if self.console:
            self.console.print(f"[red]✗[/red] {text}")
        else:
            print(f"ERROR: {text}")

    def stage(self, current: str, stages: Sequence[str]) -> None:
        """Show a list of stages with the current one highlighted."""
        if self.silent: return
        if self.console:
            table = Table(show_header=False, box=None, padding=(0, 1))
            for stage in stages:
                if stage == current:
                    table.add_row(f"[bold cyan]● {stage}[/bold cyan]")
                else:
                    table.add_row(f"[dim]○ {stage}[/dim]")
            self.console.print(table)
        else:
            print("  ".join(f"[{s}]" if s == current else s for s in stages))

    def config(self, cfg: MasterConfig) -> None:
        """Pretty‑print the configuration."""
        if self.silent: return
        rows = [
            ("Model", cfg.model or "—"),
            ("Dataset", cfg.dataset or "—"),
            ("Task", DatasetFactory.spec(cfg.dataset)["task_type"] if cfg.dataset else "—"),
            ("Repeats", str(cfg.repeats)),
            ("Max samples", "FULL" if cfg.max_samples is None else f"{cfg.max_samples:,}"),
            ("Split", f"{cfg.split_train:.0%}/{cfg.split_validation:.0%}/{cfg.split_test:.0%}"),
            ("Seed", str(cfg.seed)),
            ("Label control", "ON" if cfg.shuffled_label_control else "OFF"),
            ("Probe config", ", ".join(p.preset for p in cfg.probes)),
            ("Output dir", str(cfg.output_dir)),
            ("Config hash", cfg.config_hash),
        ]

        if self.console:
            table = Table(title="Experiment configuration")
            table.add_column("Field")
            table.add_column("Value")
            for a, b in rows:
                table.add_row(a, b)
            self.console.print(table)
        else:
            for a, b in rows:
                print(f"{a:18}: {b}")

    def result_table(self, df: pd.DataFrame, task_type: str) -> None:
        """Display a snapshot of the results DataFrame."""
        if self.silent: return
        if df.empty:
            self.warning("No result rows available.")
            return

        # Choose columns based on task type
        preferred = (
            ["probe", "layer_index", "test_macro_f1", "test_micro_f1",
             "test_average_precision_macro", "control_macro_f1",
             "selectivity", "probe_score"]
            if task_type == "multi_label"
            else
            ["probe", "layer_index", "test_macro_f1",
             "test_balanced_accuracy", "test_mcc",
             "probe_score"]
        )

        cols = [c for c in preferred if c in df.columns]
        shown = df[cols].copy()
        numeric = shown.select_dtypes(include="number").columns
        shown[numeric] = shown[numeric].round(4)

        if self.console:
            table = Table(title="Probe result snapshot")
            for col in shown.columns:
                table.add_column(str(col))
            for _, row in shown.head(30).iterrows():
                table.add_row(*[str(x) for x in row.tolist()])
            self.console.print(table)
        else:
            print(shown.head(30).to_string(index=False))

    def progress(self, description: str, total: int):
        """
        Return a Rich progress bar if available, else None.

        Usage:
            with renderer.progress("Processing", 100) as progress:
                for i in progress.track(range(100)):
                    ...
        """
        if self.silent: return None
        if self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
        return None


# ============================================================================
# 7. Forensic integrity auditor
# ============================================================================

class ForensicAuditor:
    """
    Cheap checks designed to run BEFORE expensive probe fitting.

    The auditor is intentionally independent of the probe's own metrics.
    It checks the artifacts, split partitions, and provenance.
    """

    def __init__(self, root: Path, experiment_id: str, renderer: Renderer):
        self.root = Path(root)
        self.experiment_id = experiment_id
        self.renderer = renderer

    @staticmethod
    def _load_labels(dataset_dir: Path):
        """Try to load labels from a dataset directory."""
        candidates = [
            dataset_dir / "data" / "labels.npy",
            dataset_dir / "labels.npy",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return np.load(path, allow_pickle=True)
                except Exception:
                    return None
        return None

    @staticmethod
    def _canonicalize_object_labels(labels: np.ndarray) -> np.ndarray:
        """Convert object‑dtype labels to binary matrix."""
        if labels.ndim == 2:
            return labels.astype(np.int8)
        if labels.dtype != object:
            return labels
        rows = []
        for x in labels:
            if isinstance(x, (list, tuple, set, np.ndarray)):
                rows.append(list(x))
            else:
                rows.append([x])
        classes = sorted({str(v) for row in rows for v in row})
        mapping = {v: i for i, v in enumerate(classes)}
        out = np.zeros((len(rows), len(classes)), dtype=np.int8)
        for i, row in enumerate(rows):
            for v in row:
                out[i, mapping[str(v)]] = 1
        return out

    def audit_trial(self, run_dir: Path) -> dict[str, Any]:
        """
        Audit a completed probe run directory.

        Returns:
            A dictionary with status, warnings, and errors.
        """
        report: dict[str, Any] = {
            "run_dir": str(run_dir),
            "status": "unknown",
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        completion = run_dir / "completion.json"
        results = run_dir / "layer_probe_results.csv"
        split_file = run_dir / "split_indices.npz"
        manifest = run_dir / "probe_run_manifest.json"

        report["checks"]["completion_exists"] = completion.exists()
        report["checks"]["results_exists"] = results.exists()
        report["checks"]["split_indices_exists"] = split_file.exists()
        report["checks"]["manifest_exists"] = manifest.exists()

        # Check split disjointness
        if not split_file.exists():
            report["errors"].append("split_indices.npz is missing; cannot independently verify partition disjointness.")
        else:
            try:
                archive = np.load(split_file, allow_pickle=True)
                by_repeat = {}
                for key in archive.files:
                    m = re.fullmatch(r"repeat_(\d+)_(train|validation|test)", key)
                    if not m:
                        continue
                    repeat = int(m.group(1))
                    part = m.group(2)
                    by_repeat.setdefault(repeat, {})[part] = archive[key].astype(np.int64)

                overlap_records = []
                coverage_records = []

                for repeat, parts in sorted(by_repeat.items()):
                    sets = {k: set(map(int, v)) for k, v in parts.items()}
                    names = list(sets)
                    for i, a in enumerate(names):
                        for b in names[i + 1:]:
                            overlap_records.append(
                                {
                                    "repeat": repeat,
                                    "a": a,
                                    "b": b,
                                    "overlap": len(sets[a] & sets[b]),
                                }
                            )
                    union = set().union(*sets.values()) if sets else set()
                    coverage_records.append(
                        {
                            "repeat": repeat,
                            "train": len(sets.get("train", set())),
                            "validation": len(sets.get("validation", set())),
                            "test": len(sets.get("test", set())),
                            "union": len(union),
                        }
                    )

                report["split_audit"] = {
                    "overlaps": overlap_records,
                    "coverage": coverage_records,
                }

                bad = [x for x in overlap_records if x["overlap"] != 0]
                if bad:
                    report["errors"].append(
                        f"Split leakage detected: {bad[:5]}"
                    )
            except Exception as exc:
                report["errors"].append(
                    f"Could not parse split_indices.npz: {type(exc).__name__}: {exc}"
                )

        # Check manifest
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
                report["manifest_summary"] = {
                    "task_type": m.get("target_metadata", {}).get("task_type"),
                    "class_count": len(m.get("classes", [])),
                    "text_alignment": m.get("text_alignment", {}),
                    "label_alignment": m.get("label_alignment", {}),
                    "split": m.get("split", {}),
                    "repeats": m.get("repeats"),
                    "max_samples": m.get("max_samples"),
                }

                label_status = (
                    m.get("label_alignment", {}).get("status")
                    or m.get("artifact_label_provenance_status")
                )
                if label_status not in (None, "pass", "verified"):
                    report["warnings"].append(
                        f"Label provenance is not explicitly verified: {label_status!r}"
                    )
            except Exception as exc:
                report["warnings"].append(
                    f"Manifest could not be parsed: {type(exc).__name__}: {exc}"
                )

        # Check results file
        if results.exists():
            try:
                df = pd.read_csv(results)
                report["result_rows"] = len(df)

                if "task_type" in df.columns:
                    task_types = sorted(df["task_type"].dropna().astype(str).unique())
                    report["checks"]["single_task_type"] = len(task_types) <= 1
                    if len(task_types) > 1:
                        report["errors"].append(
                            f"One result file contains multiple task types: {task_types}"
                        )

                if {"train_n", "validation_n", "test_n"}.issubset(df.columns):
                    report["split_sizes"] = (
                        df[["train_n", "validation_n", "test_n"]]
                        .drop_duplicates()
                        .to_dict("records")
                    )

                if "test_macro_f1" in df.columns:
                    mx = float(df["test_macro_f1"].max())
                    if mx > 0.98:
                        report["warnings"].append(
                            "Macro-F1 > 0.98. This is not proof of leakage, "
                            "but for this project it should trigger forensic review."
                        )
            except Exception as exc:
                report["errors"].append(
                    f"Could not parse layer_probe_results.csv: {type(exc).__name__}: {exc}"
                )

        report["status"] = "FAIL" if report["errors"] else (
            "WARN" if report["warnings"] else "PASS"
        )
        return report


# ============================================================================
# 8. Task‑aware analysis
# ============================================================================

class TaskAwareAnalyzer:
    """
    Analysis layer that keeps multi‑label and single‑label semantics separate.

    Provides normalised metrics, prevalence‑based null baselines for multi‑label,
    and summary statistics.
    """

    SINGLE_METRICS = [
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_mcc",
        "test_log_loss",
        "test_roc_auc_ovr_macro",
        "test_average_precision_macro",
    ]

    MULTI_METRICS = [
        "test_macro_f1",
        "test_micro_f1",
        "test_weighted_f1",
        "test_hamming_score",
        "test_macro_jaccard",
        "test_roc_auc_macro",
        "test_average_precision_macro",
        "test_log_loss",
    ]

    @staticmethod
    def _chance_macro_f1_single(class_count: int) -> float:
        """Conservative chance reference for balanced random guessing."""
        return 1.0 / max(class_count, 1)

    @staticmethod
    def normalized_macro_f1(
        score: float,
        chance: float,
        *,
        clip: bool = True,
    ) -> float:
        """
        Normalise Macro‑F1 relative to chance.

        Formula: (score - chance) / (1 - chance)
        """
        denom = 1.0 - chance
        if denom <= 0:
            return float("nan")
        value = (float(score) - chance) / denom
        return float(np.clip(value, 0.0, 1.0) if clip else value)

    @staticmethod
    def add_task_aware_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add columns for chance reference and normalised Macro‑F1.

        For single‑label, chance = 1/class_count.
        For multi‑label, chance is left as NaN because prevalence matters.
        """
        out = df.copy()

        if "task_type" not in out.columns:
            raise ValueError("Results must contain task_type.")

        out["chance_macro_f1_reference"] = np.nan
        out["normalized_macro_f1"] = np.nan

        single = out["task_type"].eq("single_label")
        if single.any():
            class_count = pd.to_numeric(
                out.loc[single, "class_count"], errors="coerce"
            ).fillna(1)
            chance = 1.0 / class_count
            out.loc[single, "chance_macro_f1_reference"] = chance
            out.loc[single, "normalized_macro_f1"] = [
                TaskAwareAnalyzer.normalized_macro_f1(s, c)
                for s, c in zip(
                    out.loc[single, "test_macro_f1"],
                    chance,
                )
            ]

        multi = out["task_type"].eq("multi_label")
        if multi.any():
            # Keep NaN; no simple uniform chance.
            pass

        return out

    @staticmethod
    def multi_label_prevalence_baseline(
        y_true: np.ndarray,
        *,
        seed: int = 42,
        repeats: int = 3,
    ) -> float:
        """
        Estimate a genuine multi‑label null baseline by preserving each label's
        empirical prevalence while randomising sample assignment.

        This is preferable to using 1/class_count.
        """
        y_true = np.asarray(y_true)
        if y_true.ndim != 2:
            raise ValueError("Expected [N, C] binary multi-label matrix.")

        rng = np.random.default_rng(seed)
        scores = []

        from sklearn.metrics import f1_score

        for _ in range(repeats):
            pred = np.zeros_like(y_true, dtype=np.int8)
            for j in range(y_true.shape[1]):
                k = int(y_true[:, j].sum())
                if k:
                    idx = rng.choice(len(y_true), size=k, replace=False)
                    pred[idx, j] = 1
            scores.append(float(f1_score(
                y_true, pred, average="macro", zero_division=0
            )))

        return float(np.mean(scores))

    @staticmethod
    def summarize(df: pd.DataFrame) -> dict[str, Any]:
        """Generate summary statistics per task type."""
        if df.empty:
            return {"status": "empty"}

        out = {
            "rows": int(len(df)),
            "models": sorted(df["model"].dropna().unique().tolist())
            if "model" in df.columns else [],
            "datasets": sorted(df["dataset"].dropna().unique().tolist())
            if "dataset" in df.columns else [],
            "task_types": sorted(df["task_type"].dropna().unique().tolist())
            if "task_type" in df.columns else [],
        }

        summaries = {}
        for task_type, group in df.groupby("task_type"):
            metrics = (
                TaskAwareAnalyzer.MULTI_METRICS
                if task_type == "multi_label"
                else TaskAwareAnalyzer.SINGLE_METRICS
            )
            available = [m for m in metrics if m in group.columns]
            summaries[task_type] = {
                metric: {
                    "mean": float(pd.to_numeric(group[metric], errors="coerce").mean()),
                    "std": float(pd.to_numeric(group[metric], errors="coerce").std()),
                }
                for metric in available
            }
        out["by_task_type"] = summaries
        return out


# ============================================================================
# 9. Report generation
# ============================================================================

class ReportBuilder:
    """Build a human‑readable Markdown report with interpretation guardrails."""

    @staticmethod
    def infer_patterns(df: pd.DataFrame, task_type: str) -> list[str]:
        """Generate natural‑language insights from the results."""
        if df.empty:
            return ["No completed probe results are available."]

        messages = []

        metric = "test_macro_f1" if "test_macro_f1" in df.columns else None

        if metric:
            curve = (
                df.groupby("layer_index")[metric]
                .mean()
                .sort_index()
            )
            if not curve.empty:
                best_layer = int(curve.idxmax())
                best_value = float(curve.max())
                first_value = float(curve.iloc[0])
                last_value = float(curve.iloc[-1])

                messages.append(
                    f"Best mean test Macro‑F1 occurs at layer {best_layer} "
                    f"({best_value:.3f})."
                )

                if best_value > 0.98:
                    messages.append(
                        "The result is in the forensic‑risk zone (>0.98); "
                        "do not interpret it as genuine emotion recoverability "
                        "until split, alignment and null controls pass."
                    )

                delta = last_value - first_value
                if abs(delta) < 0.02:
                    messages.append(
                        "The representation is approximately depth‑invariant "
                        "under the selected probe/metric."
                    )
                elif delta > 0:
                    messages.append(
                        "Recoverability increases toward the final layers."
                    )
                else:
                    messages.append(
                        "Recoverability decreases toward the final layers."
                    )

        if "selectivity" in df.columns:
            s = pd.to_numeric(df["selectivity"], errors="coerce").mean()
            if np.isfinite(s):
                messages.append(
                    f"Mean true‑label selectivity over the recorded rows is {s:.3f}; "
                    "this should be interpreted relative to the matched shuffled‑label control."
                )

        if task_type == "multi_label":
            messages.append(
                "For GoEmotions, prioritise Macro‑F1, Micro‑F1, average precision, "
                "per‑label F1, label cardinality and the shuffled‑label null. "
                "Do not treat 1/28 as a meaningful universal chance Macro‑F1."
            )
        else:
            messages.append(
                "For ISEAR, report Macro‑F1, balanced accuracy, MCC, confusion "
                "matrix and class‑wise recall/F1; the 1/7 reference is interpretable "
                "as a simple uniform‑class accuracy baseline, not as a complete null."
            )

        return messages

    @staticmethod
    def write_markdown(
        path: Path,
        cfg: MasterConfig,
        df: pd.DataFrame,
        task_type: str,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        """Write a comprehensive Markdown report."""
        lines = [
            "# Emotion Probe Lab — Run Report",
            "",
            f"- Master version: `{MASTER_VERSION}`",
            f"- Configuration hash: `{cfg.config_hash}`",
            f"- Model: `{cfg.model}`",
            f"- Dataset: `{cfg.dataset}`",
            f"- Task type: `{task_type}`",
            f"- Repeats: `{cfg.repeats}`",
            f"- Max samples: `{'FULL' if cfg.max_samples is None else cfg.max_samples}`",
            "",
            "## Interpretation guardrails",
            "",
        ]

        for msg in ReportBuilder.infer_patterns(df, task_type):
            lines.append(f"- {msg}")

        if audit:
            lines += [
                "",
                "## Forensic audit",
                "",
                f"- Status: **{audit.get('status')}**",
                f"- Errors: `{len(audit.get('errors', []))}`",
                f"- Warnings: `{len(audit.get('warnings', []))}`",
            ]
            for err in audit.get("errors", []):
                lines.append(f"- ERROR: {err}")
            for warning in audit.get("warnings", []):
                lines.append(f"- WARNING: {warning}")

        if not df.empty:
            lines += [
                "",
                "## Result summary",
                "",
                "```text",
                df.head(40).to_string(index=False),
                "```",
                "",
            ]

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# 10. Result visualisation (enhanced with matplotlib/seaborn)
# ============================================================================

class ResultAnalyser:
    """
    Generate publication‑ready plots from one or more result CSV files.

    The analyser automatically detects available metrics and creates:
        - Layer curves with mean ± std (or median + IQR)
        - Distribution box/violin plots per layer
        - Heatmaps (probes × layers) for each metric
        - Bar charts of best layer per probe
        - True vs shuffled control comparison (if control data present)
        - Best‑performance heatmap across models/datasets/probes (if multiple models)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if PLOTTING_AVAILABLE and sns is not None:
            sns.set_theme(style="whitegrid", palette="deep")
            plt.rcParams.update({
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": "#dddddd",
                "legend.facecolor": "white",
                "legend.edgecolor": "black",
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            })

    def load_csvs(self, csv_paths: Iterable[Path]) -> pd.DataFrame:
        """
        Concatenate multiple CSV files into a single DataFrame.

        If a directory is provided, all CSV files inside it are loaded.
        Handles non‑UTF‑8 files by falling back to latin‑1 or skipping with a warning.
        """
        frames = []
        for p in csv_paths:
            p = Path(p)
            if not p.exists():
                print(f"Warning: {p} not found, skipping.")
                continue
            if p.is_dir():
                csv_files = [f for f in sorted(p.glob("*.csv")) if not f.name.startswith("._")]
                if not csv_files:
                    print(f"Warning: No valid CSV files found in {p}, skipping.")
                    continue
                print(f"Loading {len(csv_files)} CSV file(s) from directory {p}.")
                for csv_file in csv_files:
                    self._try_read_csv(csv_file, frames)
            else:
                self._try_read_csv(p, frames)

        if not frames:
            raise ValueError("No valid CSV files provided.")
        return pd.concat(frames, ignore_index=True)

    def _try_read_csv(self, path: Path, frames: list) -> None:
        """Attempt to read a CSV file, trying different encodings."""
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding='latin-1')
                print(f"Warning: {path} is not UTF‑8; read with latin‑1.")
                frames.append(df)
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}. Skipping.")
        except pd.errors.ParserError as e:
            print(f"Warning: Could not parse {path}: {e}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}. Skipping.")

    def _available_metrics(self, df: pd.DataFrame) -> list[str]:
        """Return list of metric columns that contain numeric data."""
        candidates = [
            "test_macro_f1",
            "test_micro_f1",
            "test_balanced_accuracy",
            "test_mcc",
            "test_hamming_score",
            "test_average_precision_macro",
            "test_roc_auc_macro",
            "probe_score",
        ]
        present = [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        return present

    def generate_plots(self, df: pd.DataFrame) -> None:
        """
        Generate all standard plots for the given DataFrame.

        Args:
            df: DataFrame with columns at least 'probe', 'layer_index',
                and one or more metric columns.
        """
        if not PLOTTING_AVAILABLE:
            print("matplotlib/seaborn not installed - skipping plots.")
            return
        if df.empty:
            print("Empty DataFrame, nothing to plot.")
            return

        metrics = self._available_metrics(df)
        if not metrics:
            print("No numeric metric columns found; cannot generate plots.")
            return

        # Ensure required columns exist
        required = {"probe", "layer_index"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns for plotting: {missing}")

        # ========== 1. Layer curves for each metric (mean ± std) ==========
        for metric in metrics:
            self._plot_layer_curves(df, metric)

        # ========== 2. Distribution plots (box/violin) ==========
        self._plot_distributions(df, metrics)

        # ========== 3. Heatmaps (probes × layers) ==========
        for metric in metrics:
            self._plot_heatmap(df, metric)

        # ========== 4. Best layer per probe (bar chart) ==========
        self._plot_best_per_probe(df, metrics)

        # ========== 5. True vs shuffled control ==========
        if "control_macro_f1" in df.columns:
            self._plot_control_comparison(df)

        # ========== 6. Cross‑model best heatmap ==========
        if "model" in df.columns and "dataset" in df.columns:
            self._plot_best_heatmap(df, metrics)

        print(f"Plots saved to {self.output_dir}")

    # ------------------------------------------------------------------
    # Individual plot methods (private)
    # ------------------------------------------------------------------
    def _plot_layer_curves(self, df: pd.DataFrame, metric: str):
        """Line plot with mean and std across repeats for each probe."""
        fig, ax = plt.subplots(figsize=(10, 6))
        stats = df.groupby(["probe", "layer_index"])[metric].agg(["mean", "std"]).reset_index()

        for probe in stats["probe"].unique():
            sub = stats[stats["probe"] == probe].sort_values("layer_index")
            ax.plot(sub["layer_index"], sub["mean"], marker="o", label=probe)
            if df["probe"].nunique() == 1 or len(sub) > 1:
                ax.fill_between(
                    sub["layer_index"],
                    sub["mean"] - sub["std"],
                    sub["mean"] + sub["std"],
                    alpha=0.2
                )
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Layer-wise {metric.replace('_', ' ').title()} (mean ± std)")
        ax.grid(alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        safe_metric = metric.replace("/", "_").replace(" ", "_")
        fig.savefig(self.output_dir / f"layer_curves_{safe_metric}.png")
        plt.close(fig)

    def _plot_distributions(self, df: pd.DataFrame, metrics: list[str]):
        """Box plot showing distribution of metrics across repeats for each layer."""
        metric = metrics[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        melted = df.melt(id_vars=["probe", "layer_index"], value_vars=[metric],
                         var_name="metric", value_name="score")
        sns.boxplot(data=melted, x="layer_index", y="score", hue="probe", ax=ax)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} across repeats")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(self.output_dir / "distribution_boxplot.png")
        plt.close(fig)

    def _plot_heatmap(self, df: pd.DataFrame, metric: str):
        """Heatmap of mean metric per probe and layer."""
        pivot = df.pivot_table(index="probe", columns="layer_index", values=metric, aggfunc="mean")
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*0.8), max(4, len(pivot.index)*0.6)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5,
                    cbar_kws={"label": metric.replace("_", " ").title()}, ax=ax)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe")
        ax.set_title(f"Mean {metric.replace('_', ' ').title()} Heatmap")
        fig.tight_layout()
        safe_metric = metric.replace("/", "_").replace(" ", "_")
        fig.savefig(self.output_dir / f"heatmap_{safe_metric}.png")
        plt.close(fig)

    def _plot_best_per_probe(self, df: pd.DataFrame, metrics: list[str]):
        """Bar chart of best layer (max metric) for each probe."""
        if not metrics:
            return
        metric = metrics[0]
        best_rows = df.groupby(["probe", "layer_index"])[metric].mean().reset_index()
        best_rows = best_rows.loc[best_rows.groupby("probe")[metric].idxmax()]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=best_rows, x="probe", y=metric, hue="probe", legend=False, ax=ax)
        ax.set_ylabel(f"Best {metric.replace('_', ' ').title()}")
        ax.set_title(f"Best Layer per Probe ({metric.replace('_', ' ').title()})")
        ax.set_xlabel("Probe")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "best_per_probe.png")
        plt.close(fig)

    def _plot_control_comparison(self, df: pd.DataFrame):
        """Line plot comparing true vs shuffled label performance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        true = df.groupby(["probe", "layer_index"])["test_macro_f1"].mean().reset_index()
        ctrl = df.groupby(["probe", "layer_index"])["control_macro_f1"].mean().reset_index()

        for probe in true["probe"].unique():
            sub_true = true[true["probe"] == probe].sort_values("layer_index")
            sub_ctrl = ctrl[ctrl["probe"] == probe].sort_values("layer_index")
            ax.plot(sub_true["layer_index"], sub_true["test_macro_f1"], marker="o", label=f"{probe} (true)")
            ax.plot(sub_ctrl["layer_index"], sub_ctrl["control_macro_f1"], marker="x", linestyle="--", label=f"{probe} (shuffled)")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Macro-F1")
        ax.set_title("True vs Shuffled Label Control")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "control_comparison.png")
        plt.close(fig)

    def _plot_best_heatmap(self, df: pd.DataFrame, metrics: list[str]):
        """Heatmap of best metric across models/datasets and probes."""
        if not metrics:
            return
        metric = metrics[0]
        best = df.groupby(["model", "dataset", "probe"])[metric].max().reset_index()
        pivot = best.pivot_table(index=["model", "dataset"], columns="probe", values=metric, aggfunc="first")
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.2), max(4, len(pivot.index)*0.7)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5,
                    cbar_kws={"label": f"Best {metric.replace('_', ' ').title()}"}, ax=ax)
        ax.set_title(f"Best {metric.replace('_', ' ').title()} per Model/Dataset/Probe")
        fig.tight_layout()
        fig.savefig(self.output_dir / "best_heatmap.png")
        plt.close(fig)


# ============================================================================
# 11. Run comparison
# ============================================================================

def compare_runs(run_dirs: Sequence[Path], output_dir: Path):
    """
    Compare multiple completed runs by loading their master_results.csv files.

    Args:
        run_dirs: List of directories containing master_results.csv (or probe result CSVs).
        output_dir: Directory where comparison plots/tables will be saved.
    """
    frames = []
    labels = []
    for d in run_dirs:
        d = Path(d)
        if not d.exists():
            print(f"Warning: {d} does not exist, skipping.")
            continue
        # Look for master_results.csv or any layer_probe_results.csv
        csv_files = list(d.glob("**/master_results.csv"))
        if not csv_files:
            csv_files = list(d.glob("**/layer_probe_results.csv"))
        if not csv_files:
            print(f"Warning: No result CSV found in {d}, skipping.")
            continue

        label = get_run_label(d)
        labels.append(label)
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df["run"] = label          # Use human-readable label
            frames.append(df)

    if not frames:
        raise ValueError("No result CSV files found in provided run directories.")

    combined = pd.concat(frames, ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined CSV
    combined.to_csv(output_dir / "combined_results.csv", index=False)

    # Prepare summary
    summary = combined.groupby(["run", "probe"]).agg(
        best_macro_f1=("test_macro_f1", "max"),
        mean_macro_f1=("test_macro_f1", "mean"),
        best_layer=("layer_index", "first"),   # simplified; may be improved
    ).reset_index()

    # Display summary as rich table if available
    renderer = Renderer()
    if RICH_AVAILABLE:
        table = Table(title="Comparison Summary", show_lines=True, header_style="bold magenta")
        table.add_column("Run", style="cyan", no_wrap=True)
        table.add_column("Probe", style="green")
        table.add_column("Best Macro-F1", justify="right")
        table.add_column("Mean Macro-F1", justify="right")
        table.add_column("Best Layer", justify="right")
        for _, row in summary.iterrows():
            table.add_row(
                str(row["run"]),
                str(row["probe"]),
                f"{row['best_macro_f1']:.4f}",
                f"{row['mean_macro_f1']:.4f}",
                str(int(row["best_layer"])),
            )
        renderer.console.print(table)
    else:
        print("\nComparison Summary:")
        summary = combined.groupby(["run", "probe"]).agg(
        best_macro_f1=("test_macro_f1", "max"),
        mean_macro_f1=("test_macro_f1", "mean"),
        best_layer=("layer_index", lambda x: x[combined.loc[x.index, "test_macro_f1"].idxmax()] if len(x) else None)
    ).reset_index()
        print("\nComparison Summary:")
        print(summary.to_string(index=False))

    # Generate comparison plots (if plotting available)
    if PLOTTING_AVAILABLE:
        # Best macro_f1 per run and probe (bar chart)
        if "test_macro_f1" in combined.columns:
            best_per_run_probe = combined.groupby(["run", "probe"])["test_macro_f1"].max().reset_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(data=best_per_run_probe, x="run", y="test_macro_f1", hue="probe")
            plt.title("Best Test Macro-F1 per Run and Probe")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "comparison_best_macro_f1.png", dpi=240)
            plt.close()

        # Layer curves per run for each probe (mean)
        if "layer_index" in combined.columns and "probe" in combined.columns:
            for probe in combined["probe"].unique():
                plt.figure(figsize=(12, 6))
                for run in combined["run"].unique():
                    sub = combined[(combined["probe"] == probe) & (combined["run"] == run)]
                    avg = sub.groupby("layer_index")["test_macro_f1"].mean().sort_index()
                    plt.plot(avg.index, avg.values, marker="o", label=run)
                plt.xlabel("Layer index")
                plt.ylabel("Test Macro-F1")
                plt.title(f"Layer-wise Macro-F1 for {probe} across runs")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                safe_probe = probe.replace("/", "_").replace(" ", "_")
                plt.savefig(output_dir / f"comparison_layer_curves_{safe_probe}.png", dpi=240)
                plt.close()








def discover_all_master_runs(root: Path = DEFAULT_ROOT) -> list[Path]:
    """
    Return a list of all directories containing master_results.csv under master_runs.
    """
    master_runs = root / "master_runs"
    if not master_runs.exists():
        return []
    return sorted(set(p.parent for p in master_runs.glob("**/master_results.csv")))

# ============================================================================
# 12. Main orchestrator
# ============================================================================

class EmotionProbePipeline:
    """
    Public master API for the emotion probe pipeline.

    The constructor resolves the scientific configuration but does not start
    expensive computation. Users can inspect/modify the pipeline before
    calling .run().
    """

    STAGES = [
        Stage.PREFLIGHT,
        Stage.EXTRACTION,
        Stage.PROBING,
        Stage.ANALYSIS,
        Stage.REPORT,
        Stage.COMPLETE,
    ]

    def __init__(
        self,
        *,
        model: str,
        dataset: str,
        root: str | Path = DEFAULT_ROOT,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        probes: Sequence[str | ProbeChoice] = ("logistic",),
        repeats: int = 3,
        max_samples: int | None = 5000,
        seed: int = 42,
        shuffled_label_control: bool = True,
        shuffled_control_repeats: int = 3,
        extraction_module: str | None = None,
        probe_module: str | None = None,
        strict_provenance: bool = True,
        quiet: bool = False,
    ):
        # Normalise probe choices
        normalized_choices = []
        for p in probes:
            if isinstance(p, ProbeChoice):
                normalized_choices.append(p)
            else:
                normalized_choices.append(ProbeChoice(str(p)))
                
        # Normalise and resolve model name
        raw_model = normalize_model_name(model)
        try:
            resolved_model = resolve_model_name(raw_model)
        except ValueError as e:
            raise ValueError(str(e)) from e

        # Create configuration with resolved model
        self.config = MasterConfig(
            root=Path(root).resolve(),
            experiment_id=experiment_id,
            model=resolved_model,
            dataset=str(dataset),
            output_dir=Path(output_dir).resolve(),
            probes=normalized_choices,
            repeats=int(repeats),
            max_samples=max_samples,
            seed=int(seed),
            shuffled_label_control=bool(shuffled_label_control),
            shuffled_control_repeats=int(shuffled_control_repeats),
            extraction_module=extraction_module,
            probe_module=probe_module,
            strict_provenance=bool(strict_provenance),
            quiet=bool(quiet),
        )

        # Renderer is created after config so quiet flag can be used
        self.renderer = Renderer(silent=self.config.quiet)

        if resolved_model != raw_model:
            self.renderer.warning(f"Model name '{raw_model}' resolved to '{resolved_model}'.")

        self._validate_config()

        # Auto-adjust experiment ID if necessary
        self._auto_adjust_experiment_id()

        # Deterministic run directory
        self.run_root = (
            self.config.root / "master_runs"
            / f"{self.config.dataset}__{safe_model_path(self.config.model)}"
            / self.config.config_hash
        )
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.state = PipelineState(self.run_root / "pipeline_state.json")
        self.probe_mod = None
        self.extraction_mod = None
        self.artifact = None
        self.results: pd.DataFrame = pd.DataFrame()
        self.audit_report: dict[str, Any] | None = None

        save_json(
            self.run_root / "master_config.json",
            self.config.as_dict(),
        )
    
    def _auto_adjust_experiment_id(self) -> None:
        """
        If the current experiment ID already exists and contains a different set of
        models, change to a new unique ID derived from the model name.
        """
        if not self.config.model:
            return

        experiment_dir = self.config.root / "experiments" / self.config.experiment_id
        manifest_path = experiment_dir / "run_manifest.json"

        if not manifest_path.exists():
            return

        existing_models = []
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            for key in ("models", "model_names", "model_matrix", "model_list"):
                if key in manifest:
                    raw = manifest[key]
                    if isinstance(raw, list):
                        existing_models = [str(m) for m in raw]
                        break
        except Exception:
            existing_models = ["__unknown__"]

        if not existing_models:
            existing_models = ["__unknown__"]

        current_model = self.config.model.lower()
        existing_normalized = [m.lower() for m in existing_models]

        if current_model in existing_normalized:
            return

        model_slug = safe_model_path(self.config.model).as_posix().replace("/", "_")
        new_exp_id = f"{self.config.experiment_id}__{model_slug}"
        old_exp_id = self.config.experiment_id
        self.config.experiment_id = new_exp_id
        self.renderer.warning(
            f"Experiment ID '{old_exp_id}' already contains different models. "
            f"Auto‑adjusting to '{new_exp_id}' for model '{self.config.model}'."
        )

    # ----------------------------------------------------------------------
    # Public helpers
    # ----------------------------------------------------------------------

    def describe(self) -> None:
        """Display configuration and paths."""
        self.renderer.title(
            "EMOTION PROBE LAB",
            "Representation learning • probing • forensic evaluation",
        )
        self.renderer.config(self.config)
        self.renderer.info(
            f"Artifacts: {compact_path(self.artifact_dir)}\n"
            f"Master run: {compact_path(self.run_root)}"
        )

    @property
    def artifact_dir(self) -> Path:
        """Path to the frozen hidden‑state artifact."""
        return dataset_artifact_dir(
            self.config.root,
            self.config.experiment_id,
            self.config.model,
            self.config.dataset,
        )

    def set_probes(self, *probes: str) -> "EmotionProbePipeline":
        """Change probes after construction (chainable)."""
        self.config.probes = [ProbeChoice(p) for p in probes]
        self._validate_config()
        save_json(self.run_root / "master_config.json", self.config.as_dict())
        return self

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Ensure configuration is consistent."""
        DatasetFactory.spec(self.config.dataset)

        if self.config.repeats < 1:
            raise ValueError("repeats must be >= 1")

        if self.config.max_samples is not None and self.config.max_samples < 30:
            raise ValueError("max_samples must be >= 30 or None")

        fractions = (
            self.config.split_train
            + self.config.split_validation
            + self.config.split_test
        )
        if not np.isclose(fractions, 1.0):
            raise ValueError("Train/validation/test fractions must sum to 1.")

        if self.config.shuffled_control_repeats < 1:
            raise ValueError("shuffled_control_repeats must be >= 1")

    # ----------------------------------------------------------------------
    # Artifact management
    # ----------------------------------------------------------------------

    def _artifact_exists(self) -> bool:
        """Check if the hidden‑state artifact exists and is complete."""
        required = [
            self.artifact_dir / "data" / "hidden_states.npy",
            self.artifact_dir / "data" / "completed.npy",
            self.artifact_dir / "metadata" / "extraction.json",
        ]
        return all(p.exists() for p in required)

    def _load_artifact(self):
        """Load the artifact via the probe module's ExtractionArtifact."""
        self.probe_mod = ModuleFactory.probe(self.config)
        ExtractionArtifact = getattr(self.probe_mod, "ExtractionArtifact")
        self.artifact = ExtractionArtifact(
            self.artifact_dir,
            verify_checksum=self.config.verify_checksum,
        )
        return self.artifact
    
    def _run_extraction(self, dataset_obj):
        """Call the extraction module's run_experiments with current config."""
        self.extraction_mod.run_experiments(
            datasets={self.config.dataset: dataset_obj},
            model_names=[self.config.model],
            base_output=self.config.root,
            max_length=512,
            auto_batch_size=False,
            continue_on_model_error=False,
            show_verbose=True,
            show_info=True,
            show_critical=True,
            experiment_id=self.config.experiment_id,
        )

    def ensure_extraction(self, *, run_if_missing: bool = False) -> None:
        """
        Ensure the hidden‑state artifact exists; optionally run extraction.

        Args:
            run_if_missing: If True, run extraction when artifact is missing.
        """
        if self._artifact_exists():
            self.renderer.success(
                f"Frozen hidden‑state artifact found: {compact_path(self.artifact_dir)}"
            )
            self._load_artifact()
            return

        if not run_if_missing:
            raise FileNotFoundError(
                "No compatible hidden‑state artifact exists.\n"
                f"Expected: {self.artifact_dir}\n"
                "Run extraction first or call ensure_extraction(run_if_missing=True)."
            )

        self.extraction_mod = ModuleFactory.extraction(self.config)
        if self.extraction_mod is None:
            raise ImportError(
                "No extraction module found. Please install one of: "
                f"{', '.join(DEFAULT_EXTRACTION_MODULES)} or set extraction_module in config.\n"
                "If you already have a frozen artifact, run without --extract-if-missing."
            )

        if not hasattr(self.extraction_mod, "run_experiments"):
            raise AttributeError("Extraction module does not expose run_experiments(...).")

        dataset_obj = DatasetFactory.load(self.config.dataset)
        self.renderer.warning(
            "Extraction artifact is missing. Starting deterministic extraction."
        )

        try:
            self._run_extraction(dataset_obj)
        except RuntimeError as e:
            if "model matrix differs" in str(e):
                old_id = self.config.experiment_id
                model_slug = safe_model_path(self.config.model).as_posix().replace("/", "_")
                self.config.experiment_id = f"{old_id}__{model_slug}_retry"
                self.renderer.warning(
                    f"Model matrix conflict detected. Retrying with new experiment ID: {self.config.experiment_id}"
                )
                self._run_extraction(dataset_obj)
            else:
                raise

        if not self._artifact_exists():
            raise RuntimeError("Extraction completed without producing the expected artifact.")
        self._load_artifact()

    # ----------------------------------------------------------------------
    # Forensic pre‑flight
    # ----------------------------------------------------------------------

    def forensic_preflight(self, *, extract_if_missing: bool = False) -> dict[str, Any]:
        """
        Run all pre‑probe integrity checks.

        Args:
            extract_if_missing: Passed to ensure_extraction.

        Returns:
            A report dictionary with status, provenance info, etc.
        """
        self.state.start(Stage.PREFLIGHT)
        self.renderer.stage(Stage.PREFLIGHT, self.STAGES)

        self.ensure_extraction(run_if_missing=extract_if_missing)

        if not np.all(self.artifact.completed):
            raise RuntimeError(
                "Extraction completion map is incomplete. Refusing to probe."
            )

        if self.artifact.states.shape[0] != self.artifact.sample_count:
            raise RuntimeError("Hidden‑state sample count does not match metadata.")

        ds_spec = DatasetFactory.spec(self.config.dataset)
        dataset_df = DatasetFactory.load(self.config.dataset)

        DatasetContract = getattr(self.probe_mod, "DatasetContract")
        contract = DatasetContract(
            target_type=ds_spec["target_type"],
            type="python",
            module=ds_spec["module"],
            function=ds_spec["function"],
            task_type=ds_spec["task_type"],
            require_provenance=self.config.strict_provenance,
        )

        build_targets = getattr(self.probe_mod, "build_targets")
        y, classes, target_meta = build_targets(dataset_df, contract)

        if len(y) != self.artifact.sample_count:
            raise RuntimeError(
                f"Target count {len(y)} != hidden‑state count {self.artifact.sample_count}."
            )

        validate_targets = getattr(self.probe_mod, "validate_targets")
        target_validation = validate_targets(
            y, classes, ds_spec["task_type"]
        )

        validate_text_alignment = getattr(self.probe_mod, "validate_text_alignment")
        validate_label_alignment = getattr(self.probe_mod, "validate_label_alignment")

        text_alignment = validate_text_alignment(
            self.artifact, dataset_df, contract
        )
        label_alignment = validate_label_alignment(
            self.artifact, dataset_df, contract, y, classes
        )

        if self.config.strict_provenance:
            if text_alignment.get("status") not in {"pass", "verified"}:
                raise RuntimeError(
                    f"Text provenance did not pass strict validation: {text_alignment}"
                )

        if not label_alignment.get("verified", False):
            self.renderer.warning(
                "Label alignment is not cryptographically verified by the artifact."
            )

        auditor = ForensicAuditor(
            self.config.root,
            self.config.experiment_id,
            self.renderer,
        )

        existing_dirs = list(
            (self.artifact_dir / "analysis" / "probes").glob(
                "**/completion.json"
            )
        )
        for completion in existing_dirs[-20:]:
            report = auditor.audit_trial(completion.parent)
            if report["status"] == "FAIL":
                self.renderer.warning(
                    f"Existing probe run failed forensic audit: {completion.parent}"
                )

        report = {
            "status": "PASS",
            "dataset": self.config.dataset,
            "model": self.config.model,
            "task_type": ds_spec["task_type"],
            "target_count": len(y),
            "class_count": len(classes),
            "target_validation": target_validation,
            "text_alignment": text_alignment,
            "label_alignment": label_alignment,
            "artifact_validation": self.artifact.validation,
            "artifact_shape": list(self.artifact.states.shape),
        }

        save_json(self.run_root / "preflight.json", report)
        self.state.finish(Stage.PREFLIGHT, "Scientific preflight passed.")
        self.renderer.success("Forensic preflight passed.")
        return report

    # ----------------------------------------------------------------------
    # Probe execution
    # ----------------------------------------------------------------------

    def run_probes(self) -> pd.DataFrame:
        """Run the probing analysis using the installed probe module."""
        if self.artifact is None:
            self._load_artifact()

        self.state.start(Stage.PROBING)
        self.renderer.stage(Stage.PROBING, self.STAGES)

        ProbeSpec = getattr(self.probe_mod, "ProbeSpec")
        AnalysisConfig = getattr(self.probe_mod, "AnalysisConfig")
        SplitConfig = getattr(self.probe_mod, "SplitConfig")
        UnifiedProbeAnalyzer = getattr(self.probe_mod, "UnifiedProbeAnalyzer")
        DatasetContract = getattr(self.probe_mod, "DatasetContract")

        probes = ProbeFactory.create(self.probe_mod, self.config.probes)
        ds_spec = DatasetFactory.spec(self.config.dataset)

        contract = DatasetContract(
            target_type=ds_spec["target_type"],
            type="python",
            module=ds_spec["module"],
            function=ds_spec["function"],
            task_type=ds_spec["task_type"],
            require_provenance=self.config.strict_provenance,
        )

        split = SplitConfig(
            train=self.config.split_train,
            validation=self.config.split_validation,
            test=self.config.split_test,
            seed=self.config.seed,
            stratify=True,
        )

        analysis = AnalysisConfig(
            dataset=contract,
            probes=probes,
            layers=self.config.layers,
            split=split,
            repeats=self.config.repeats,
            max_samples=self.config.max_samples,
            shuffled_label_control=self.config.shuffled_label_control,
            shuffled_control_repeats=self.config.shuffled_control_repeats,
            run_control_on_all_layers=True,
            pca_enabled=self.config.pca_enabled,
            silhouette_enabled=self.config.silhouette_enabled,
            pca_samples=self.config.pca_samples,
            silhouette_samples=self.config.silhouette_samples,
            enable_abstention=True,
            enable_per_class_metrics=True,
            enable_feature_statistics=True,
            verbose=1,
        )

        output_dir = (
            self.artifact_dir / "analysis" / "probes"
            / f"master__{self.config.config_hash}"
        )

        dataset_df = DatasetFactory.load(self.config.dataset)

        analyzer = UnifiedProbeAnalyzer(
            self.artifact,
            analysis,
            output_dir=output_dir,
            dataset_df=dataset_df,
        )

        self.renderer.info(
            f"Running {len(probes)} probe(s) across "
            f"{len(analyzer.layers)} layer(s), {self.config.repeats} repeat(s)."
        )

        progress = self.renderer.progress("Probing", total=1)
        try:
            if progress:
                with progress:
                    task = progress.add_task("Probing...", total=1)
                    scored, best = analyzer.run()
                    progress.update(task, advance=1)
            else:
                scored, best = analyzer.run()
        except RuntimeError as e:
            self.renderer.warning(f"Probe run incomplete: {e}")
            partial_path = analyzer.output_dir / "layer_probe_results.csv"
            if partial_path.exists():
                scored = pd.read_csv(partial_path)
                best = scored.sort_values(
                    ["probe", "probe_score"], ascending=[True, False]
                ).groupby("probe", as_index=False).first()
            else:
                raise

        if scored is None or scored.empty:
            raise RuntimeError("Probe execution produced no scored results.")

        scored = scored.copy()
        scored["master_version"] = MASTER_VERSION
        scored["master_config_hash"] = self.config.config_hash

        scored = TaskAwareAnalyzer.add_task_aware_columns(scored)

        self.results = scored

        scored.to_csv(self.run_root / "master_results.csv", index=False)
        best.to_csv(self.run_root / "master_best.csv", index=False)

        self.state.finish(
            Stage.PROBING,
            f"{len(scored)} scored rows written.",
        )
        self.renderer.success(f"Probe execution complete: {len(scored)} rows.")
        return scored

    # ----------------------------------------------------------------------
    # Analysis / reporting
    # ----------------------------------------------------------------------

    def analyze(self) -> dict[str, Any]:
        """Generate task‑aware summary statistics."""
        if self.results.empty:
            result_path = self.run_root / "master_results.csv"
            if result_path.exists():
                self.results = pd.read_csv(result_path)
            else:
                raise RuntimeError("No probe results available.")

        self.state.start(Stage.ANALYSIS)

        task_type = DatasetFactory.spec(self.config.dataset)["task_type"]

        summary = TaskAwareAnalyzer.summarize(self.results)

        if "task_type" in self.results.columns:
            for task, group in self.results.groupby("task_type"):
                group.to_csv(
                    self.run_root / f"results__{task}.csv",
                    index=False,
                )

        save_json(self.run_root / "analysis_summary.json", summary)

        self.renderer.result_table(self.results, task_type)

        self.state.finish(Stage.ANALYSIS, "Task‑aware analysis complete.")
        return summary

    def report(self) -> Path:
        """Write the final Markdown report."""
        self.state.start(Stage.REPORT)

        task_type = DatasetFactory.spec(self.config.dataset)["task_type"]
        path = self.run_root / "FINAL_REPORT.md"

        ReportBuilder.write_markdown(
            path=path,
            cfg=self.config,
            df=self.results,
            task_type=task_type,
            audit=self.audit_report,
        )

        self.state.finish(Stage.REPORT, f"Report: {path}")
        self.renderer.success(f"Report written: {compact_path(path)}")
        return path

    # ----------------------------------------------------------------------
    # Optional visualisation
    # ----------------------------------------------------------------------

    def generate_plots(self, output_dir: Path | None = None) -> None:
        """Generate plots from the results."""
        if self.results.empty:
            result_path = self.run_root / "master_results.csv"
            if result_path.exists():
                self.results = pd.read_csv(result_path)
            else:
                raise RuntimeError("No results to plot.")

        out_dir = Path(output_dir) if output_dir else self.run_root / "plots"
        analyser = ResultAnalyser(out_dir)
        analyser.generate_plots(self.results)

    # ----------------------------------------------------------------------
    # End‑to‑end run
    # ----------------------------------------------------------------------

    def run(
        self,
        *,
        extract_if_missing: bool = False,
        generate_plots: bool = False,
    ) -> dict[str, Any]:
        """
        Execute the full pipeline.

        Args:
            extract_if_missing: If True, run extraction if artifact missing.
            generate_plots: If True, generate visualisations.

        Returns:
            Dictionary with configuration, run root, summary, and results.
        """
        self.describe()

        self.state.start(Stage.PREFLIGHT)
        self.state.finish(Stage.PREFLIGHT, "Delegating to forensic_preflight.")

        self.forensic_preflight(extract_if_missing=extract_if_missing)

        self.run_probes()
        summary = self.analyze()
        self.report()

        if generate_plots and PLOTTING_AVAILABLE:
            self.generate_plots(self.config.output_dir)

        self.state.start(Stage.COMPLETE)
        save_json(
            self.run_root / "completion.json",
            {
                "status": "complete",
                "master_version": MASTER_VERSION,
                "config_hash": self.config.config_hash,
                "finished_at": time.time(),
            },
        )
        self.state.finish(Stage.COMPLETE, "Pipeline complete.")

        self.renderer.title(
            "RUN COMPLETE",
            f"Results and report: {compact_path(self.run_root)}",
        )

        return {
            "config": self.config.as_dict(),
            "run_root": str(self.run_root),
            "summary": summary,
            "results": self.results,
            "report": str(self.run_root / "FINAL_REPORT.md"),
        }


# ============================================================================
# 13. Interactive interface
# ============================================================================

class InteractiveApp:
    """
    Friendly terminal front‑end.

    Exposes a guided menu that asks for model, dataset, probes, and other
    parameters, then builds and runs the pipeline.
    """

    def __init__(self):
        self.renderer = Renderer()

    def ask(self, prompt: str, default: str | None = None) -> str:
        """Ask a question with an optional default."""
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        return value if value else (default or "")

    def choose(self, title: str, options: Sequence[str]) -> str:
        """Present a menu of options and return the selected one."""
        print("\n" + title)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")

        while True:
            raw = input("Select: ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            print("Please select a valid number.")

    def get_available_models(self) -> list[tuple]:
        """Return list of (model_name, family, params) from extraction registry."""
        try:
            extraction_mod = ModuleFactory.extraction(MasterConfig())
        except ImportError:
            return []
        if hasattr(extraction_mod, "MODEL_REGISTRY"):
            return [
                (m.name, m.family, m.parameter_billions)
                for m in extraction_mod.MODEL_REGISTRY
            ]
        return []

    def has_artifact(self, model: str, dataset: str) -> bool:
        """Check if an artifact exists for the given model/dataset."""
        root = DEFAULT_ROOT
        exp = DEFAULT_EXPERIMENT_ID
        ds_dir = dataset_artifact_dir(root, exp, model, dataset)
        return (ds_dir / "metadata" / "extraction.json").exists()

    def run(self) -> dict[str, Any]:
        """Launch the interactive session."""
        self.renderer.title(
            "EMOTION PROBE LAB",
            "A controlled interface for frozen hidden‑state emotion probing",
        )

        list_available_models()
        print()

        available_models = self.get_available_models()
        if not available_models:
            model = self.ask("Model", "google-bert/bert-base-uncased")
        else:
            while True:
                raw = self.ask("Model (enter number or full name)", "google-bert/bert-base-uncased")
                if raw.lower() == 'q':
                    self.renderer.info("Exiting.")
                    return {"status": "cancelled"}
                try:
                    model = resolve_model_name(raw)
                    break
                except ValueError as e:
                    self.renderer.error(str(e))
                    print("Please try again.\n")

        dataset = self.choose("Dataset", ["goEmo", "ISEAR"])

        print("\nProbe selection")
        print("  ENTER = logistic baseline")
        print("  1     = logistic")
        print("  2     = logistic + 1-hidden MLP")
        print("  3     = logistic + 1/2-hidden MLP")
        print("  4     = logistic + 1/2/3-hidden MLP")

        choice = input("Probe configuration: ").strip()
        probe_map = {
            "": ["logistic"],
            "1": ["logistic"],
            "2": ["logistic", "mlp1"],
            "3": ["logistic", "mlp1", "mlp2"],
            "4": ["logistic", "mlp1", "mlp2", "mlp3"],
        }
        probes = probe_map.get(choice, ["logistic"])

        max_raw = self.ask("Maximum samples (ENTER = 5000, type FULL for all)", "5000")
        max_samples = None if max_raw.upper() == "FULL" else int(max_raw)

        repeats = int(self.ask("Independent repeats", "3"))
        controls = self.ask("Run shuffled-label control? Y/N", "Y").upper() == "Y"

        output_dir = self.ask("Output directory (ENTER = ./demo_runs)", "./demo_runs") or "./demo_runs"

        extract_if_missing = self.ask("Extract hidden states if missing? Y/N", "Y").upper() == "Y"

        pipeline = EmotionProbePipeline(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            probes=probes,
            repeats=repeats,
            max_samples=max_samples,
            shuffled_label_control=controls,
        )

        pipeline.describe()

        confirmation = self.ask("Start this experiment? Y/N", "Y").upper()
        if confirmation != "Y":
            self.renderer.info("Experiment cancelled.")
            return {"status": "cancelled"}

        return pipeline.run(extract_if_missing=extract_if_missing, generate_plots=True)




        
def print_rich_help(parser):
    """Display a rich help panel for the CLI."""
    renderer = Renderer()
    if not RICH_AVAILABLE:
        parser.print_help()
        return

    renderer.title("EMOTION PROBE LAB", "Unified command-line interface")
    sub_table = Table(title="Available Commands", show_lines=True, header_style="bold cyan")
    sub_table.add_column("Command", style="bold magenta", min_width=15)
    sub_table.add_column("Description", style="white")
    sub_table.add_column("Shortcut", style="green")
    subcommands = [
        ("run", "Execute a full probing experiment", "r"),
        ("models", "List available models and artifact status", "m"),
        ("interactive", "Launch guided interactive mode", "i"),
        ("audit", "Audit an existing probe run", "a"),
        ("analyse", "Generate plots from result CSVs", "an"),
        ("compare", "Compare multiple completed runs", "c"),
    ]
    for cmd, desc, short in subcommands:
        sub_table.add_row(cmd, desc, short)
    renderer.console.print(sub_table)

    opt_table = Table(title="Global Options", show_lines=True, header_style="bold yellow")
    opt_table.add_column("Option", style="bold")
    opt_table.add_column("Description", style="white")
    opt_table.add_row("-h, --help", "Show this help message and exit")
    renderer.console.print(opt_table)
    print("\nRun 'python Pipeline__.py <command> --help' for command-specific options.")

def print_rich_run_help():
    renderer = Renderer()
    if not RICH_AVAILABLE:
        print("Run help:\n")
        return
    table = Table(title="Run Command Options", show_lines=True, header_style="bold green")
    table.add_column("Option", style="bold cyan")
    table.add_column("Description", style="white")
    options = [
        ("-m, --model", "Hugging Face model name (full or alias)"),
        ("-d, --dataset", "Dataset: goEmo or ISEAR"),
        ("-R, --root", "Root directory for artifacts"),
        ("-e, --experiment-id", "Experiment identifier"),
        ("-o, --output-dir", "Output directory"),
        ("-p, --probe", "Probe type (repeatable)"),
        ("-r, --repeats", "Number of repeats"),
        ("-n, --max-samples", "Max samples (use --full for all)"),
        ("-f, --full", "Use all samples"),
        ("-s, --seed", "Random seed"),
        ("-S, --no-shuffle-control", "Disable shuffled-label control"),
        ("-x, --extract-if-missing", "Run extraction if artifact missing"),
        ("-P, --no-plot", "Skip generating plots"),
        ("-q, --quiet", "Suppress non-essential output"),
    ]
    for opt, desc in options:
        table.add_row(opt, desc)
    renderer.console.print(table)

def print_rich_models_help():
    renderer = Renderer()
    if RICH_AVAILABLE:
        table = Table(title="Models Command", show_lines=True)
        table.add_column("Usage", style="bold cyan")
        table.add_row("python Pipeline__.py models")
        renderer.console.print(table)
    else:
        print("Usage: python Pipeline__.py models")

def print_rich_interactive_help():
    renderer = Renderer()
    if RICH_AVAILABLE:
        table = Table(title="Interactive Command", show_lines=True)
        table.add_column("Usage", style="bold cyan")
        table.add_row("python Pipeline__.py interactive")
        renderer.console.print(table)
    else:
        print("Usage: python Pipeline__.py interactive")

def print_rich_audit_help():
    renderer = Renderer()
    if RICH_AVAILABLE:
        table = Table(title="Audit Command", show_lines=True)
        table.add_column("Usage", style="bold cyan")
        table.add_column("Description", style="white")
        table.add_row("python Pipeline__.py audit RUN_DIR", "Audit a specific run directory")
        table.add_row("python Pipeline__.py audit --list", "List all auditable run directories")
        renderer.console.print(table)
    else:
        print("Usage: python Pipeline__.py audit RUN_DIR | --list")

def print_rich_analyse_help():
    renderer = Renderer()
    if RICH_AVAILABLE:
        table = Table(title="Analyse Command", show_lines=True)
        table.add_column("Option", style="bold cyan")
        table.add_column("Description", style="white")
        table.add_row("-i, --input FILE", "Input CSV file (repeatable) or directory")
        table.add_row("-o, --output-dir DIR", "Output directory for plots")
        table.add_row("--list", "List all analysable run directories")
        renderer.console.print(table)
    else:
        print("Usage: python Pipeline__.py analyse -i CSV [-i CSV2 ...] [-o DIR] | --list")

def print_rich_compare_help():
    renderer = Renderer()
    if RICH_AVAILABLE:
        table = Table(title="Compare Command", show_lines=True)
        table.add_column("Usage", style="bold cyan")
        table.add_row("python Pipeline__.py compare RUN_DIR1 RUN_DIR2 ... [-o OUTPUT_DIR]")
        renderer.console.print(table)
    else:
        print("Usage: python Pipeline__.py compare RUN_DIR1 RUN_DIR2 ... [-o OUTPUT_DIR]")
# ============================================================================
# 14. CLI
# ============================================================================


def audit_existing_run(run_dir: str | Path) -> dict[str, Any]:
    """
    Standalone audit helper that accepts either a master‑run directory
    (produced by the pipeline) or a raw probe‑run directory.
    """
    path = Path(run_dir).resolve()

    # Determine if this is a master‑run directory (contains master_results.csv)
    if (path / "master_results.csv").exists() and (path / "master_config.json").exists():
        # Load master config to find the artifact and config hash
        with open(path / "master_config.json", "r", encoding="utf-8") as f:
            master_cfg = json.load(f)
        model = master_cfg["model"]
        dataset = master_cfg["dataset"]
        root = Path(master_cfg["root"])
        experiment_id = master_cfg.get("experiment_id", DEFAULT_EXPERIMENT_ID)
        config_hash = master_cfg.get("config_hash", path.name)

        artifact_dir = dataset_artifact_dir(root, experiment_id, model, dataset)
        # The probe output directory uses a pattern like "master__<hash>"
        probe_dirs = list(
            (artifact_dir / "analysis" / "probes").glob(f"master__{config_hash}*")
        )
        if probe_dirs:
            # Use the first matching probe‑run directory
            path = probe_dirs[0]
        else:
            # Fallback: search any probe directory that contains layer_probe_results.csv
            probe_dirs = list(
                (artifact_dir / "analysis" / "probes").glob("**/layer_probe_results.csv")
            )
            if probe_dirs:
                path = probe_dirs[0].parent

    # Now run the auditor on the (possibly) probe‑run directory
    auditor = ForensicAuditor(
        root=path.parent,
        experiment_id="audit-only",
        renderer=Renderer(),
    )
    report = auditor.audit_trial(path)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report




def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true", help="Show help")

    sub = parser.add_subparsers(dest="command")

    # run
    run = sub.add_parser("run", aliases=["r"], help="Run an experiment.", add_help=False)
    run.add_argument("-m", "--model", required=True)
    run.add_argument("-d", "--dataset", choices=sorted(DATASET_SPECS), required=True)
    run.add_argument("-R", "--root", default=str(DEFAULT_ROOT))
    run.add_argument("-e", "--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    run.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run.add_argument("-p", "--probe", action="append", default=["logistic"],
                     choices=["logistic", "mlp1", "mlp2", "mlp3"])
    run.add_argument("-r", "--repeats", type=int, default=3)
    run.add_argument("-n", "--max-samples", type=int, default=5000)
    run.add_argument("-f", "--full", action="store_true")
    run.add_argument("-s", "--seed", type=int, default=42)
    run.add_argument("-S", "--no-shuffle-control", action="store_true")
    run.add_argument("-x", "--extract-if-missing", action="store_true")
    run.add_argument("-P", "--no-plot", action="store_true", help="Skip generating plots.")
    run.add_argument("-q", "--quiet", action="store_true", help="Suppress non‑essential output.")
    run.add_argument("-h", "--help", action="store_true", help="Show run help")

    # models
    models = sub.add_parser("models", aliases=["m"], help="List available models and their artifact status.", add_help=False)
    models.add_argument("-h", "--help", action="store_true", help="Show models help")

    # interactive
    interactive = sub.add_parser("interactive", aliases=["i"], help="Launch the guided interface.", add_help=False)
    interactive.add_argument("-h", "--help", action="store_true", help="Show interactive help")

    # analyse
    analyse = sub.add_parser("analyse", aliases=["an"], help="Generate visualisations from result CSV(s).", add_help=False)
    analyse.add_argument("-i", "--input", action="append", required=False)
    analyse.add_argument("-o", "--output-dir", default="./analysis_plots")
    analyse.add_argument("-l", "--list", action="store_true", help="List analysable runs")
    analyse.add_argument("-h", "--help", action="store_true", help="Show analyse help")

    # audit
    audit = sub.add_parser("audit", aliases=["a"], help="Audit an existing probe run without retraining.", add_help=False)
    audit.add_argument("run_dir", nargs="?")
    audit.add_argument("-l", "--list", action="store_true", help="List auditable runs")
    audit.add_argument("-h", "--help", action="store_true", help="Show audit help")

    # compare
    compare = sub.add_parser("compare", aliases=["c"], help="Compare multiple completed runs.", add_help=False)
    compare.add_argument("run_dirs", nargs="*", help="Directories containing master_results.csv files.")
    compare.add_argument("-a", "--all", action="store_true", help="Compare all available master runs.")
    compare.add_argument("-o", "--output-dir", default="./comparison_plots")
    compare.add_argument("-h", "--help", action="store_true", help="Show compare help")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or (len(argv) == 1 and argv[0] in ("-h", "--help")):
        parser = build_parser()
        print_rich_help(parser)
        return 0

    subcommand = None
    for token in argv:
        if not token.startswith('-'):
            subcommand = token
            break

    help_requested = any(arg in ("-h", "--help") for arg in argv[1:])

    if help_requested and subcommand:
        alias_to_canonical = {
            "r": "run", "run": "run",
            "m": "models", "models": "models",
            "i": "interactive", "interactive": "interactive",
            "a": "audit", "audit": "audit",
            "an": "analyse", "analyse": "analyse",
            "c": "compare", "compare": "compare",
        }
        canonical = alias_to_canonical.get(subcommand)
        if canonical == "run":
            print_rich_run_help()
        elif canonical == "models":
            print_rich_models_help()
        elif canonical == "interactive":
            print_rich_interactive_help()
        elif canonical == "audit":
            print_rich_audit_help()
        elif canonical == "analyse":
            print_rich_analyse_help()
        elif canonical == "compare":
            print_rich_compare_help()
        else:
            parser = build_parser()
            print_rich_help(parser)
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        probes = list(dict.fromkeys(args.probe))
        try:
            pipeline = EmotionProbePipeline(
                model=args.model,
                dataset=args.dataset,
                root=args.root,
                experiment_id=args.experiment_id,
                output_dir=args.output_dir,
                probes=probes,
                repeats=args.repeats,
                max_samples=None if args.full else args.max_samples,
                seed=args.seed,
                shuffled_label_control=not args.no_shuffle_control,
                quiet=args.quiet,
            )
            pipeline.run(
                extract_if_missing=args.extract_if_missing,
                generate_plots=not args.no_plot,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        return 0

    elif args.command == "models":
        list_available_models()
        return 0

    elif args.command == "interactive":
        InteractiveApp().run()
        return 0

    elif args.command == "analyse":
        if args.list:
            list_analysable_runs()
            return 0
        if not args.input:
            print("Error: either --list or --input must be provided.")
            return 1
        csv_paths = [Path(p) for p in args.input]
        analyser = ResultAnalyser(Path(args.output_dir))
        df = analyser.load_csvs(csv_paths)
        analyser.generate_plots(df)
        return 0

    elif args.command == "audit":
        if args.list:
            list_auditable_runs()
            return 0
        if not args.run_dir:
            print("Error: either --list or a run directory must be provided.")
            return 1
        report = audit_existing_run(args.run_dir)
        return 1 if report["status"] == "FAIL" else 0

    elif args.command == "compare":
        if args.all:
            run_dirs = discover_all_master_runs()
            if not run_dirs:
                print("No master runs found.")
                return 1
        else:
            if not args.run_dirs:
                print("Error: either provide run directories or use --all.")
                return 1
            run_dirs = [Path(p) for p in args.run_dirs]
        compare_runs(run_dirs, Path(args.output_dir))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# ============================================================================
# 15. Command usage documentation (appears in --help and as a comment)
# ============================================================================

"""
COMMAND USAGE
=============

This script provides four subcommands:

1. run
   Execute a full experiment.
   Usage: python G_test.py run --model MODEL --dataset {goEmo,ISEAR} [options]
   Options:
     --root PATH              Root directory for artifacts (default: /Volumes/Amirali/hidden_states)
     --experiment-id ID       Experiment identifier (default: baseline_v5_001)
     --output-dir PATH        Local output directory (default: ./demo_runs)
     --probe {logistic,mlp1,mlp2,mlp3}  Probe type (can be repeated)
     --repeats N              Number of independent repeats (default: 3)
     --max-samples N          Maximum samples to use (default: 5000)
     --full                   Use all samples (overrides --max-samples)
     --seed N                 Random seed (default: 42)
     --no-shuffle-control     Disable shuffled-label control
     --extract-if-missing     Run extraction if artifact is missing
     --no-plot                Skip generating plots

2. interactive
   Launch guided interactive mode.
   Usage: python G_test.py interactive

3. audit
   Audit an existing run directory without retraining.
   Usage: python G_test.py audit RUN_DIR 

4. analyse
   Generate plots from one or more result CSV files.
   Usage: python G_test.py analyse -i CSV1 [-i CSV2 ...] [-o OUTPUT_DIR]
   Options:
     -i, --input FILE         Input CSV file (can be repeated)
     -o, --output-dir DIR     Output directory for plots (default: ./analysis_plots)

EXAMPLES
--------
# Run a standard experiment with logistic probe
python G_test.py run --model google-bert/bert-base-uncased --dataset goEmo

# Run with two probes and extraction if missing
python G_test.py run --model Qwen/Qwen2-1.5B --dataset ISEAR --probe logistic --probe mlp1 --extract-if-missing

# Launch interactive menu
python G_test.py interactive

# Audit a previous run
python G_test.py audit ./demo_runs/goEmo__bert-base-uncased/abc123def456

# Generate plots from two result CSVs
python G_test.py analyse -i results__goEmo.csv -i results__ISEAR.csv -o ./my_plots

# Use full dataset and skip plots
python G_test.py run --model google-bert/bert-base-uncased --dataset goEmo --full --no-plot
"""