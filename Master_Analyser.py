"""
EMOTION PROBE LAB – UNIFIED ANALYSER
=====================================

A sophisticated CLI tool for analysing extraction outputs, probe results,
and generating comprehensive reports with rich visualisations.

This script provides a single entry point for:
  • Extraction health checks (integrity, completeness, anomalies)
  • Probe result analysis (performance, comparisons, anomalies)
  • Full project analysis with summary reports
  • Comparison of multiple runs
  • Forensic audit of individual run directories
  • Beautiful terminal UI (using `rich` if available)
  • Command-line interface with subcommands
  • Interactive guided mode

All analysis is performed on the project’s output directories:
  - Extraction artifacts: hidden_states/experiments/<exp_id>/models/
  - Probe results: hidden_states/experiments/<exp_id>/matrix_checkpoint/
                   and analysis/probes/ directories.

Usage:
    python Master_Analyser.py extraction     # scan extraction outputs
    python Master_Analyser.py probes         # scan probing results
    python Master_Analyser.py all            # full analysis
    python Master_Analyser.py report         # generate full report (same as all)
    python Master_Analyser.py compare RUN1 RUN2 ...   # compare runs
    python Master_Analyser.py compare --model1 MODEL_A --model2 MODEL_B   # compare models
    python Master_Analyser.py audit RUN_DIR          # forensic audit
    python Master_Analyser.py list-runs              # list analysable runs
    python Master_Analyser.py list-models            # list models with artifact status
    python Master_Analyser.py plots                  # generate plots from probe results
    python Master_Analyser.py interactive            # launch guided interactive mode

Options:
    --project-root PATH     Path to Final-Year-Project (default: .)
    --result-root PATH      Path to hidden_states (default: /Volumes/Amirali/hidden_states)
    --exp-id ID             Experiment ID (default: baseline_v5_001)
    --output-html FILE      Export report as HTML (default: report.html)
    --no-plots              Disable plot generation
    --quiet                 Suppress non‑essential output
    --model FILTER          Filter by model name (substring)
    --dataset FILTER        Filter by dataset name (substring)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Optional dependencies for enhanced user experience
# ------------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.text import Text
    from rich import box
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
# 1. Constants and configurations
# ============================================================================

MASTER_VERSION = "1.0.0"
DEFAULT_ROOT = Path("/Volumes/Amirali/hidden_states")
DEFAULT_EXPERIMENT_ID = "baseline_v5_001"

# Supported datasets
DATASET_SPECS = {
    "goEmo": {"label": "GoEmotions", "task_type": "multi_label"},
    "ISEAR": {"label": "ISEAR", "task_type": "single_label"},
}

# Model aliases (same as Pipeline__.py)
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


# ============================================================================
# 2. Utility functions
# ============================================================================

def clean_path_input(raw: str) -> List[Path]:
    """
    Extract valid filesystem paths from a messy input string.
    Tries to find paths starting with /Volumes/ or / (Unix absolute paths).
    Returns a list of unique Path objects that exist.
    """
    # Remove any box-drawing or extra characters that are not path-like
    pattern = r'(/[^\s]+)'  # simple: slash followed by non-space chars
    matches = re.findall(pattern, raw)
    # Filter out obvious non-paths: e.g., single "/"
    candidates = [m for m in matches if len(m) > 3 and not m.endswith('/')]
    # Also look for paths that start with 'Volumes' without leading slash? We'll handle separately.
    # If no match, try to split by spaces and check each token.
    if not candidates:
        tokens = raw.split()
        candidates = [t for t in tokens if t.startswith('/') or t.startswith('Volumes/')]
    # Clean up: remove trailing garbage like '│' or '|'
    cleaned = []
    for c in candidates:
        # Strip any trailing non-path chars: e.g., '│' or '|'
        c = c.strip('│| \t')
        # If it starts with 'Volumes' but not '/', prepend '/'
        if c.startswith('Volumes/'):
            c = '/' + c
        cleaned.append(c)
    # Convert to Path and filter existence (optional)
    paths = [Path(p) for p in cleaned if p]
    # Remove duplicates
    unique = list(dict.fromkeys(paths))
    return unique


def stable_hash(value: Any, length: int = 16) -> str:
    """Deterministic hash of a JSON‑serialisable object."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def save_json(path: Path, payload: dict) -> None:
    """Atomically write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def safe_model_path(model_name: str) -> Path:
    """Convert a Hugging Face model name to a safe relative path."""
    return Path(*[p for p in model_name.split("/") if p])


def dataset_artifact_dir(root: Path, experiment_id: str, model_name: str, dataset_name: str) -> Path:
    """Return the expected location of the frozen hidden‑state artifact."""
    return root / "experiments" / experiment_id / "models" / safe_model_path(model_name) / "datasets" / dataset_name


def compact_path(path: Path, max_chars: int = 74) -> str:
    """Shorten a path for display if it exceeds max_chars."""
    s = str(path)
    return s if len(s) <= max_chars else "…" + s[-(max_chars - 1):]


def resolve_model_name(raw: str) -> str:
    """Resolve a user-provided model name to a full registry name."""
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
        choices = "\n".join([f"  - {m} (alias: {alias})" for m in matches])
        raise ValueError(f"Ambiguous model name '{raw}'. Please choose one of:\n{choices}")
    valid = "\n".join([f"  - {alias}: {full}" for alias, full in MODEL_ALIASES.items()])
    raise ValueError(f"Model '{raw}' not recognized. Valid names:\n{valid}")


# ============================================================================
# 3. Renderer (Rich or fallback)
# ============================================================================

class Renderer:
    """Handles all terminal output with optional Rich support."""

    def __init__(self, silent: bool = False):
        self.silent = silent
        self.console = None if silent else (Console() if RICH_AVAILABLE else None)

    def title(self, title: str, subtitle: str = "") -> None:
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

    def rule(self, title: str) -> None:
        """Draw a horizontal rule with a title."""
        if self.silent: return
        if self.console:
            self.console.rule(title)
        else:
            print("\n" + "-" * 80)
            print(title)
            print("-" * 80)

    def info(self, text: str) -> None:
        if self.silent: return
        if self.console:
            self.console.print(text)
        else:
            print(text)

    def warning(self, text: str) -> None:
        if self.silent: return
        if self.console:
            self.console.print(f"[yellow]WARNING[/yellow] {text}")
        else:
            print(f"WARNING: {text}")

    def success(self, text: str) -> None:
        if self.silent: return
        if self.console:
            self.console.print(f"[green]✓[/green] {text}")
        else:
            print(f"✓ {text}")

    def error(self, text: str) -> None:
        if self.silent: return
        if self.console:
            self.console.print(f"[red]✗[/red] {text}")
        else:
            print(f"ERROR: {text}")

    def progress(self, description: str, total: int):
        """Return a Rich progress bar if available, else None."""
        if self.silent or not self.console:
            return None
        if self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
        return None


# ============================================================================
# 4. Core Analyser Class
# ============================================================================

class ProjectAnalyser:
    """
    Main analyser for the project, handling extraction and probe analysis.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        result_root: Union[str, Path],
        exp_id: str = DEFAULT_EXPERIMENT_ID,
        quiet: bool = False,
        model_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.result_root = Path(result_root).resolve()
        self.exp_id = exp_id
        self.quiet = quiet
        self.model_filter = model_filter.lower() if model_filter else None
        self.dataset_filter = dataset_filter.lower() if dataset_filter else None
        self.renderer = Renderer(silent=quiet)

        # Derived paths
        self.exp_root = self.result_root / "experiments" / self.exp_id
        self.models_dir = self.exp_root / "models"
        self.matrix_checkpoint = self.exp_root / "matrix_checkpoint"
        self.per_entry_results = self.matrix_checkpoint / "per_entry_results"
        self.results_index = self.matrix_checkpoint / "results_index.csv"
        self.checkpoint_path = self.matrix_checkpoint / "probe_matrix_checkpoint.json"

        # Storage for analysis results
        self.extraction_report: Dict[str, Any] = {}
        self.probe_report: Dict[str, Any] = {}
        self.full_report: Dict[str, Any] = {}

    def _filter_model(self, name: str) -> bool:
        if self.model_filter is None:
            return True
        return self.model_filter in name.lower()

    def _filter_dataset(self, name: str) -> bool:
        if self.dataset_filter is None:
            return True
        return self.dataset_filter in name.lower()

    # ------------------------------------------------------------------
    # Extraction Analysis
    # ------------------------------------------------------------------

    def analyse_extraction(self) -> Dict[str, Any]:
        """Scan the extraction output and produce a detailed health report."""
        self.renderer.rule("[bold yellow]Extraction Health Check[/bold yellow]")
        report = {
            "per_dataset": [],
            "summary": {"total_datasets": 0, "healthy": 0, "unhealthy": 0, "status": "OK"},
        }

        if not self.models_dir.exists():
            self.renderer.error(f"Models directory not found: {self.models_dir}")
            report["summary"]["status"] = "ERROR"
            self.extraction_report = report
            return report

        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            self.renderer.error("No model directories found!")
            report["summary"]["status"] = "ERROR"
            self.extraction_report = report
            return report

        total_datasets = 0
        total_ok = 0
        total_errors = 0

        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            if not self._filter_model(model_name):
                continue
            datasets_dir = model_dir / "datasets"
            if not datasets_dir.exists():
                continue
            for dataset_dir in sorted(datasets_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                dataset_name = dataset_dir.name
                if not self._filter_dataset(dataset_name):
                    continue
                total_datasets += 1
                ctx = self._load_verification_context(model_name, dataset_name)
                if ctx is None:
                    total_errors += 1
                    continue
                rep = self._generate_extraction_report(ctx)
                healthy = rep.get("is_healthy", False)
                if healthy:
                    total_ok += 1
                else:
                    total_errors += 1
                report["per_dataset"].append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "healthy": healthy,
                    "missing": len(rep.get("missing_files", [])),
                    "shape_errors": len(rep.get("shape_errors", [])),
                    "completion": rep.get("completion", {}).get("completion_pct", 0),
                    "nan_inf": rep.get("nan_inf", {}).get("has_invalid", False),
                })

        report["summary"] = {
            "total_datasets": total_datasets,
            "healthy": total_ok,
            "unhealthy": total_errors,
            "status": "OK" if total_errors == 0 else "WARNING" if total_errors < total_datasets else "ERROR",
        }

        self._display_extraction_summary(report)
        self.extraction_report = report
        return report

    def _load_verification_context(self, model_name: str, dataset_name: str):
        """Load the verification context for a model/dataset without holding large object arrays."""
        model_dir = self.models_dir / model_name
        dataset_dir = model_dir / "datasets" / dataset_name

        paths = {
            "states": dataset_dir / "data" / "hidden_states.npy",
            "completed": dataset_dir / "data" / "completed.npy",
            "labels": dataset_dir / "data" / "labels.npy",
            "sample_ids": dataset_dir / "metadata" / "sample_ids.npy",
            "metadata": dataset_dir / "metadata" / "extraction.json",
            "integrity": dataset_dir / "metadata" / "integrity_hashes.jsonl",
            "label_codes": dataset_dir / "metadata" / "label_codes.npy",
            "sample_manifest": dataset_dir / "metadata" / "sample_manifest.jsonl",
        }

        # Check for critical files
        for name, p in paths.items():
            if not p.exists():
                self.renderer.error(f"Missing {name} for {model_name}/{dataset_name}: {p}")
                return None

        try:
            states = np.load(paths["states"], mmap_mode='r')
            completed = np.load(paths["completed"], mmap_mode='r')
        except Exception as e:
            self.renderer.error(f"Error loading arrays for {model_name}/{dataset_name}: {e}")
            return None

        # Load metadata files (JSON) – small
        metadata = json.load(open(paths["metadata"])) if paths["metadata"].exists() else None
        integrity_hashes = self._load_integrity_hashes(paths["integrity"])
        sample_manifest = self._load_sample_manifest(paths["sample_manifest"])

        # For labels and sample_ids – load only shapes, not the full arrays (to avoid memory issues)
        labels_shape = None
        sample_ids_shape = None
        try:
            # Load just the shape by using .shape on a memory-mapped file (works for numeric arrays)
            # But if they are object dtype, mmap fails, so we load with allow_pickle and then delete
            labels = np.load(paths["labels"], allow_pickle=True)
            labels_shape = labels.shape
            del labels
        except Exception:
            labels_shape = None

        try:
            sample_ids = np.load(paths["sample_ids"], allow_pickle=True)
            sample_ids_shape = sample_ids.shape
            del sample_ids
        except Exception:
            sample_ids_shape = None

        ctx = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "states": states,
            "completed": completed,
            "paths": paths,
            "labels_shape": labels_shape,
            "sample_ids_shape": sample_ids_shape,
            "metadata": metadata,
            "integrity_hashes": integrity_hashes,
            "label_codes": None,  # not used
            "sample_manifest": sample_manifest,
        }
        return ctx

    def _load_integrity_hashes(self, path: Path) -> List[Dict]:
        if not path.exists():
            return []
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]

    def _load_sample_manifest(self, path: Path):
        if not path.exists():
            return None
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]

    def _generate_extraction_report(self, ctx: Dict) -> Dict:
        """Generate a report for a single model/dataset."""
        report = {}
        states = ctx["states"]
        completed = ctx["completed"]
        labels_shape = ctx.get("labels_shape")
        sample_ids_shape = ctx.get("sample_ids_shape")
        n_samples = states.shape[0]

        # Shape checks
        shape_errors = []
        if completed.shape != (n_samples,):
            shape_errors.append(f"completed shape {completed.shape} != states samples {n_samples}")
        if sample_ids_shape is not None and sample_ids_shape != (n_samples,):
            shape_errors.append(f"sample_ids shape {sample_ids_shape} != states samples {n_samples}")
        if labels_shape is not None and labels_shape[0] != n_samples:
            shape_errors.append(f"labels shape {labels_shape} != states samples {n_samples}")
        if states.ndim != 3:
            shape_errors.append(f"states must be 3D, got {states.ndim}D")
        report["shape_errors"] = shape_errors

        # Completion
        done = int(np.sum(completed))
        missing = n_samples - done
        report["completion"] = {
            "total_samples": n_samples,
            "completed_samples": done,
            "missing_samples": missing,
            "completion_pct": (done / n_samples) * 100 if n_samples > 0 else 0,
            "is_complete": done == n_samples,
        }

        # Integrity hashes
        hashes = ctx.get("integrity_hashes", [])
        if hashes:
            integrity_status = "ok"
            mismatches = []
            for h in hashes:
                start = h["batch_start"]
                end = h["batch_end"]
                if np.all(completed[start:end]):
                    data = np.asarray(states[start:end])
                    computed = hashlib.sha256(data.tobytes()).hexdigest()
                    if computed != h["hash"]:
                        mismatches.append(f"batch {start}-{end} hash mismatch")
            if mismatches:
                integrity_status = "corrupted"
            report["integrity"] = {"status": integrity_status, "mismatches": mismatches}
        else:
            report["integrity"] = {"status": "not_available"}

        # NaN/Inf
        nan_count = 0
        inf_count = 0
        for start in range(0, n_samples, 1000):
            end = min(start + 1000, n_samples)
            chunk = np.asarray(states[start:end])
            nan_count += int(np.isnan(chunk).sum())
            inf_count += int(np.isinf(chunk).sum())
        has_invalid = (nan_count + inf_count) > 0
        report["nan_inf"] = {"nan_count": nan_count, "inf_count": inf_count, "has_invalid": has_invalid}

        # Missing files
        missing_files = [str(p) for p in ctx["paths"].values() if not p.exists()]
        report["missing_files"] = missing_files

        # Linkage checks (use shapes)
        linkage_errors = []
        # Sample IDs check would require loading, but we can skip if shape is OK
        # We already checked shape. We'll skip deeper linkage to save memory.
        report["linkage_errors"] = linkage_errors

        # Overall health
        is_healthy = (
            not shape_errors
            and not has_invalid
            and not linkage_errors
            and report["completion"]["is_complete"]
            and report["integrity"].get("status") in ("ok", "not_available")
            and not missing_files
        )
        report["is_healthy"] = is_healthy

        return report

    def _display_extraction_summary(self, report: Dict):
        """Display a summary table of extraction health."""
        if self.quiet:
            return
        if RICH_AVAILABLE:
            table = Table(title="Extraction Health Summary", box=box.HEAVY, style="bold cyan")
            table.add_column("Model", style="cyan")
            table.add_column("Dataset", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Completion", justify="right")
            table.add_column("Missing Files", justify="right")
            table.add_column("Shape Errors", justify="right")
            table.add_column("NaN/Inf", justify="center")

            for entry in report.get("per_dataset", []):
                status = "✓" if entry["healthy"] else "✗"
                status_color = "green" if entry["healthy"] else "red"
                nan_inf = "Yes" if entry["nan_inf"] else "No"
                table.add_row(
                    entry["model"],
                    entry["dataset"],
                    Text(status, style=status_color),
                    f"{entry['completion']:.1f}%",
                    str(entry["missing"]),
                    str(entry["shape_errors"]),
                    nan_inf,
                )
            self.renderer.console.print(table)
        else:
            # Fallback plain-text
            print("\nExtraction Health Summary:")
            print(f"{'Model':<30} {'Dataset':<10} {'Status':<8} {'Completion':<12} {'Missing':<10} {'Shape':<8} {'NaN/Inf'}")
            for entry in report.get("per_dataset", []):
                status = "OK" if entry["healthy"] else "FAIL"
                print(f"{entry['model']:<30} {entry['dataset']:<10} {status:<8} {entry['completion']:.1f}%    {entry['missing']:<10} {entry['shape_errors']:<8} {entry['nan_inf']}")

        self.renderer.info(f"Summary: {report['summary']['healthy']} healthy, {report['summary']['unhealthy']} unhealthy")
        self.renderer.info(f"Overall Status: {report['summary']['status']}")

    # ------------------------------------------------------------------
    # Probe Results Analysis
    # ------------------------------------------------------------------

    def analyse_probes(self) -> Dict[str, Any]:
        """Analyse all probe result CSV files and produce metrics summary."""
        self.renderer.rule("[bold magenta]Probe Results Analysis[/bold magenta]")
        report = {
            "files_found": 0,
            "files_loaded": 0,
            "errors": [],
            "summary": {},
            "per_model": {},
            "best_performers": [],
            "anomalies": {},
        }

        # Locate CSV files
        csv_files = []
        # From matrix_checkpoint/per_entry_results
        csv_files.extend(self.per_entry_results.glob("*_layer_probe_results.csv"))
        # From analysis/probes directories (including renamed ones)
        csv_files.extend(self.exp_root.glob("**/analysis/probes/**/layer_probe_results.csv"))
        # Also from matrix_runs
        csv_files.extend(self.exp_root.glob("**/analysis/probes/matrix_runs/*/layer_probe_results.csv"))

        # Deduplicate
        csv_files = list(set(csv_files))
        csv_files = [f for f in csv_files if not f.name.startswith("._")]

        report["files_found"] = len(csv_files)
        self.renderer.info(f"Found [bold]{len(csv_files)}[/bold] result CSV files.")

        if not csv_files:
            self.renderer.error("No probe result files found!")
            return report

        # Load and combine
        dataframes = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                df["_source"] = str(f)
                dataframes.append(df)
            except Exception as e:
                report["errors"].append(f"Failed to read {f.name}: {e}")

        if not dataframes:
            self.renderer.error("No data loaded!")
            return report

        df_all = pd.concat(dataframes, ignore_index=True)
        report["files_loaded"] = len(dataframes)

        # Identify key columns
        model_col = next((c for c in ["model", "model_name"] if c in df_all.columns), None)
        dataset_col = next((c for c in ["dataset", "dataset_name"] if c in df_all.columns), None)
        layer_col = next((c for c in ["layer", "layer_idx", "layer_index"] if c in df_all.columns), None)
        metric_cols = [c for c in df_all.columns if c.startswith("test_") and not c.startswith("test_labels_")]
        primary_metric = None
        for m in ["test_macro_f1", "test_balanced_accuracy", "test_micro_f1", "test_accuracy"]:
            if m in metric_cols:
                primary_metric = m
                break

        if primary_metric is None:
            self.renderer.error("No standard metric columns found!")
            return report

        report["primary_metric"] = primary_metric

        # Convert metric to numeric
        df_all[primary_metric] = pd.to_numeric(df_all[primary_metric], errors="coerce")

        # Summary by model/dataset
        group_cols = [c for c in [model_col, dataset_col] if c]
        if group_cols:
            summary = df_all.groupby(group_cols)[primary_metric].agg(["mean", "std", "count"]).reset_index()
            report["summary_table"] = summary

        # Best per model/dataset
        if model_col and layer_col:
            best_rows = []
            for key, g in df_all.dropna(subset=[primary_metric]).groupby([model_col, dataset_col] if dataset_col else [model_col]):
                idx = g[primary_metric].idxmax()
                row = g.loc[idx]
                best_rows.append({
                    "model": row[model_col],
                    "dataset": row[dataset_col] if dataset_col else "",
                    "best_layer": row[layer_col],
                    "best_score": row[primary_metric],
                    "best_probe": row.get("probe", ""),
                })
            report["best_performers"] = best_rows

        # Detect anomalies (like NaN, inf)
        nan_rows = df_all[primary_metric].isna().sum()
        inf_rows = (df_all[primary_metric] == np.inf).sum()
        report["anomalies"] = {
            "nan_rows": nan_rows,
            "inf_rows": inf_rows,
            "total_rows": len(df_all),
        }

        self._display_probe_summary(report, df_all, model_col, dataset_col, layer_col, primary_metric)
        self.probe_report = report
        return report

    def _display_probe_summary(self, report: Dict, df_all: pd.DataFrame,
                                model_col: str, dataset_col: str,
                                layer_col: str, primary_metric: str):
        if self.quiet:
            return
        # Table: files loaded
        if RICH_AVAILABLE:
            table = Table(title="Probe Data Load Summary", box=box.HEAVY)
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="bold white")
            table.add_row("Files found", str(report["files_found"]))
            table.add_row("Files loaded", str(report["files_loaded"]))
            table.add_row("Total rows", str(report["anomalies"]["total_rows"]))
            table.add_row("NaN rows", str(report["anomalies"]["nan_rows"]))
            table.add_row("Inf rows", str(report["anomalies"]["inf_rows"]))
            table.add_row("Primary metric", primary_metric)
            self.renderer.console.print(table)
        else:
            print("\nProbe Data Load Summary:")
            print(f"Files found: {report['files_found']}")
            print(f"Files loaded: {report['files_loaded']}")
            print(f"Total rows: {report['anomalies']['total_rows']}")
            print(f"NaN rows: {report['anomalies']['nan_rows']}")
            print(f"Inf rows: {report['anomalies']['inf_rows']}")
            print(f"Primary metric: {primary_metric}")

        # Table: best performers
        if report["best_performers"]:
            if RICH_AVAILABLE:
                best_table = Table(title=f"Best Performers (by {primary_metric})", box=box.HEAVY)
                best_table.add_column("Model", style="cyan")
                if dataset_col:
                    best_table.add_column("Dataset")
                best_table.add_column("Best Layer", justify="center")
                best_table.add_column("Best Score", justify="right", style="green")
                best_table.add_column("Probe", style="yellow")
                for entry in sorted(report["best_performers"], key=lambda x: x["best_score"], reverse=True)[:10]:
                    row = [entry["model"]]
                    if dataset_col:
                        row.append(entry["dataset"])
                    row.extend([str(entry["best_layer"]), f"{entry['best_score']:.4f}", entry.get("best_probe", "")])
                    best_table.add_row(*row)
                self.renderer.console.print(best_table)
            else:
                print("\nBest Performers:")
                for entry in sorted(report["best_performers"], key=lambda x: x["best_score"], reverse=True)[:10]:
                    print(f"{entry['model']} {entry['dataset']} layer {entry['best_layer']}: {entry['best_score']:.4f}")

        # Optionally, show summary table if group_cols exist
        if "summary_table" in report and RICH_AVAILABLE:
            sum_table = Table(title="Aggregated Performance", box=box.HEAVY)
            sum_table.add_column("Model", style="cyan")
            if dataset_col:
                sum_table.add_column("Dataset")
            sum_table.add_column(f"Mean {primary_metric} (best)", justify="right", style="green")
            sum_table.add_column("Std", justify="right")
            sum_table.add_column("Count", justify="right")
            for _, row in report["summary_table"].iterrows():
                sum_table.add_row(
                    str(row[model_col]),
                    str(row[dataset_col]) if dataset_col else "",
                    f"{row['mean']:.4f}",
                    f"{row['std']:.4f}",
                    str(row["count"])
                )
            self.renderer.console.print(sum_table)

    # ------------------------------------------------------------------
    # Full Analysis
    # ------------------------------------------------------------------

    def analyse_all(self) -> Dict[str, Any]:
        """Run both extraction and probe analysis and produce a combined report."""
        self.renderer.rule("[bold green]Full Project Analysis[/bold green]")
        report = {
            "extraction": self.analyse_extraction(),
            "probes": self.analyse_probes(),
            "overall_status": "OK",
        }
        # Determine overall status
        ext_status = report["extraction"]["summary"].get("status", "ERROR")
        probe_status = "OK" if report["probes"].get("files_loaded", 0) > 0 else "WARNING"
        if ext_status == "ERROR" or probe_status == "ERROR":
            report["overall_status"] = "ERROR"
        elif ext_status == "WARNING" or probe_status == "WARNING":
            report["overall_status"] = "WARNING"
        else:
            report["overall_status"] = "OK"

        self.renderer.info(f"[bold]Overall Status: [{'green' if report['overall_status'] == 'OK' else 'red'}]{report['overall_status']}[/][/bold]")
        self.full_report = report
        return report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, output_html: Optional[str] = None) -> str:
        """Generate a comprehensive HTML report from the analysis results."""
        if self.full_report is None:
            self.analyse_all()

        html_parts = []
        html_parts.append("<html><head><title>Project Analysis Report</title>")
        html_parts.append("<style>body { font-family: sans-serif; } table { border-collapse: collapse; } th, td { border: 1px solid #ddd; padding: 6px; } th { background-color: #f2f2f2; }</style>")
        html_parts.append("</head><body>")
        html_parts.append("<h1>Project Analysis Report</h1>")

        if "extraction" in self.full_report:
            html_parts.append("<h2>Extraction Health</h2>")
            ext = self.full_report["extraction"]
            html_parts.append(f"<p>Overall Status: {ext['summary']['status']}</p>")
            if "per_dataset" in ext:
                df = pd.DataFrame(ext["per_dataset"])
                html_parts.append(df.to_html(index=False))
        if "probes" in self.full_report:
            html_parts.append("<h2>Probe Results</h2>")
            probe = self.full_report["probes"]
            html_parts.append(f"<p>Files loaded: {probe['files_loaded']}</p>")
            if "summary_table" in probe:
                html_parts.append(probe["summary_table"].to_html(index=False))
            if "best_performers" in probe:
                best_df = pd.DataFrame(probe["best_performers"])
                html_parts.append("<h3>Best Performers</h3>")
                html_parts.append(best_df.to_html(index=False))
        html_parts.append("</body></html>")
        html_content = "\n".join(html_parts)

        if output_html:
            Path(output_html).write_text(html_content)
            self.renderer.success(f"Report saved to {output_html}")

        return html_content

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def generate_plots(self, output_dir: Optional[Path] = None) -> None:
        """Generate plots from probe results."""
        if not PLOTTING_AVAILABLE:
            self.renderer.warning("matplotlib/seaborn not installed - skipping plots.")
            return

        if self.probe_report is None or not self.probe_report.get("files_loaded", 0):
            self.renderer.warning("No probe data loaded; cannot generate plots.")
            return

        # Reload data if needed
        df = self._load_probe_data()
        if df is None or df.empty:
            return

        out_dir = Path(output_dir) if output_dir else self.project_root / "analysis_plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        analyser = ResultAnalyser(out_dir, renderer=self.renderer)
        analyser.generate_plots(df)

    def _load_probe_data(self) -> Optional[pd.DataFrame]:
        """Load combined probe data from report or directly from files."""
        if self.probe_report and "files_loaded" in self.probe_report and self.probe_report["files_loaded"] > 0:
            # Try to reload from the combined data (we didn't store the full df in report)
            pass
        # Fallback: load from files again
        csv_files = []
        csv_files.extend(self.per_entry_results.glob("*_layer_probe_results.csv"))
        csv_files.extend(self.exp_root.glob("**/analysis/probes/**/layer_probe_results.csv"))
        csv_files = list(set(csv_files))
        csv_files = [f for f in csv_files if not f.name.startswith("._")]

        if not csv_files:
            return None

        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)


# ============================================================================
# 5. Result Analyser (plotting)
# ============================================================================

class ResultAnalyser:
    """Generate publication‑ready plots from one or more result CSV files."""

    def __init__(self, output_dir: Path, renderer: Renderer):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = renderer
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

    def generate_plots(self, df: pd.DataFrame) -> None:
        """Generate all standard plots for the given DataFrame."""
        if not PLOTTING_AVAILABLE:
            self.renderer.warning("matplotlib/seaborn not installed - skipping plots.")
            return
        if df.empty:
            self.renderer.warning("Empty DataFrame, nothing to plot.")
            return

        metrics = self._available_metrics(df)
        if not metrics:
            self.renderer.warning("No numeric metric columns found; cannot generate plots.")
            return

        # Ensure required columns exist
        required = {"probe", "layer_index"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            self.renderer.warning(f"Missing required columns for plotting: {missing}")
            return

        self.renderer.info("Generating plots...")

        # ========== 1. Layer curves for each metric (mean ± std) ==========
        for metric in metrics:
            self._plot_layer_curves(df, metric)

        # ========== 2. Distribution plots (box) ==========
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

        self.renderer.success(f"Plots saved to {self.output_dir}")

    def _available_metrics(self, df: pd.DataFrame) -> list[str]:
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
        return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    def _plot_layer_curves(self, df: pd.DataFrame, metric: str):
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
# 6. Forensic Auditor (copied from Pipeline__.py)
# ============================================================================

class ForensicAuditor:
    """
    Cheap checks designed to run BEFORE expensive probe fitting.
    """

    def __init__(self, root: Path, experiment_id: str, renderer: Renderer):
        self.root = Path(root)
        self.experiment_id = experiment_id
        self.renderer = renderer

    @staticmethod
    def _load_labels(dataset_dir: Path):
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

    def audit_trial(self, run_dir: Path) -> Dict[str, Any]:
        """Audit a completed probe run directory."""
        report = {
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
                    report["errors"].append(f"Split leakage detected: {bad[:5]}")
            except Exception as exc:
                report["errors"].append(f"Could not parse split_indices.npz: {type(exc).__name__}: {exc}")

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
                    report["warnings"].append(f"Label provenance is not explicitly verified: {label_status!r}")
            except Exception as exc:
                report["warnings"].append(f"Manifest could not be parsed: {type(exc).__name__}: {exc}")

        # Check results file
        if results.exists():
            try:
                df = pd.read_csv(results)
                report["result_rows"] = len(df)

                if "task_type" in df.columns:
                    task_types = sorted(df["task_type"].dropna().astype(str).unique())
                    report["checks"]["single_task_type"] = len(task_types) <= 1
                    if len(task_types) > 1:
                        report["errors"].append(f"One result file contains multiple task types: {task_types}")

                if {"train_n", "validation_n", "test_n"}.issubset(df.columns):
                    report["split_sizes"] = (
                        df[["train_n", "validation_n", "test_n"]]
                        .drop_duplicates()
                        .to_dict("records")
                    )

                if "test_macro_f1" in df.columns:
                    mx = float(df["test_macro_f1"].max())
                    if mx > 0.98:
                        report["warnings"].append("Macro-F1 > 0.98. This is not proof of leakage, but for this project it should trigger forensic review.")
            except Exception as exc:
                report["errors"].append(f"Could not parse layer_probe_results.csv: {type(exc).__name__}: {exc}")

        report["status"] = "FAIL" if report["errors"] else ("WARN" if report["warnings"] else "PASS")
        return report


# ============================================================================
# 7. Run Discovery and Listing
# ============================================================================

def get_analysable_entries(root: Path, experiment_id: str) -> List[Dict]:
    """Return a list of dicts for analysable runs."""
    entries = []
    master_runs = root / "master_runs"
    if master_runs.exists():
        for master_csv in master_runs.glob("**/master_results.csv"):
            run_dir = master_csv.parent
            parts = run_dir.relative_to(master_runs).parts
            if len(parts) >= 3:
                dataset_org = parts[0]
                model_name = parts[1]
                config_hash = parts[2]
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
    exp_models = root / "experiments" / experiment_id / "models"
    if exp_models.exists():
        for probe_csv in exp_models.glob("**/layer_probe_results.csv"):
            run_dir = probe_csv.parent
            parts = run_dir.relative_to(exp_models).parts
            if len(parts) >= 5 and parts[2] == "datasets":
                org = parts[0]
                model_name = parts[1]
                dataset = parts[3]
                full_model = f"{org}/{model_name}"
                config_hash = ""
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
    entries.sort(key=lambda x: (x["model"], x["dataset"], x["run_type"]))
    return entries


def list_analysable_runs(root: Path = DEFAULT_ROOT, experiment_id: str = DEFAULT_EXPERIMENT_ID) -> None:
    """List all directories that contain analysable probe result CSVs."""
    renderer = Renderer()
    entries = get_analysable_entries(root, experiment_id)

    if not entries:
        renderer.warning("No analysable runs found.")
        return

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
    """List all probe run directories that contain a probe_run_manifest.json."""
    renderer = Renderer()
    exp_models = root / "experiments" / experiment_id / "models"
    if not exp_models.exists():
        renderer.warning(f"No experiment directory found at {exp_models}.")
        return

    entries = []
    for manifest_path in exp_models.glob("**/probe_run_manifest.json"):
        run_dir = manifest_path.parent
        parts = run_dir.relative_to(exp_models).parts
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


def list_available_models(root: Path = DEFAULT_ROOT, experiment_id: str = DEFAULT_EXPERIMENT_ID) -> None:
    """Display a table of all available models and their artifact availability."""
    renderer = Renderer()
    models = get_available_models()
    if not models:
        renderer.warning("No models found in the extraction module's registry.")
        return

    if RICH_AVAILABLE:
        table = Table(title="Available Models", show_lines=True, header_style="bold magenta", width=150)
        table.add_column("#", style="dim", justify="right")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Family", style="magenta")
        table.add_column("Params (B)", justify="right")
        table.add_column("goEmo artifact", justify="center")
        table.add_column("ISEAR artifact", justify="center")

        for i, (name, family, params) in enumerate(models, 1):
            has_go = has_artifact(root, experiment_id, name, "goEmo")
            has_isear = has_artifact(root, experiment_id, name, "ISEAR")
            go_mark = "[green]✓[/green]" if has_go else "[red]✗[/red]"
            isear_mark = "[green]✓[/green]" if has_isear else "[red]✗[/red]"
            table.add_row(str(i), name, family, f"{params:.3f}", go_mark, isear_mark)

        renderer.console.print(table)
        renderer.info("✓ = hidden‑state artifact exists (ready for probing)\n✗ = artifact missing (extraction needed)")
    else:
        print("\nAvailable models:")
        for i, (name, family, params) in enumerate(models, 1):
            has_go = has_artifact(root, experiment_id, name, "goEmo")
            has_isear = has_artifact(root, experiment_id, name, "ISEAR")
            status = f"goEmo:{'✓' if has_go else '✗'}, ISEAR:{'✓' if has_isear else '✗'}"
            print(f"{i:3d}. {name:30s} {family:10s} {params:.3f}B  [{status}]")
    print()


def get_available_models():
    """Return a list of (model_name, family, params) from the extraction module's registry."""
    try:
        # Try to import Extraction module to get model registry
        import Extraction as ext
        if hasattr(ext, "MODEL_REGISTRY"):
            return [(m.name, m.family, m.parameter_billions) for m in ext.MODEL_REGISTRY]
    except ImportError:
        pass
    return []


def has_artifact(root: Path, experiment_id: str, model: str, dataset: str) -> bool:
    """Check whether a frozen hidden‑state artifact exists for the given model and dataset."""
    ds_dir = dataset_artifact_dir(root, experiment_id, model, dataset)
    return (ds_dir / "metadata" / "extraction.json").exists()


# ============================================================================
# 8. Compare Runs
# ============================================================================

def get_run_label(run_dir: Path) -> str:
    """Generate a compact, human‑readable label for a run directory."""
    run_dir = run_dir.resolve()
    parts = run_dir.parts

    if "master_runs" in parts:
        idx = parts.index("master_runs")
        if len(parts) > idx + 3:
            dataset_org = parts[idx + 1]
            model_name = parts[idx + 2]
            hash_part = parts[idx + 3] if len(parts) > idx + 3 else ""
            if "__" in dataset_org:
                dataset, org = dataset_org.split("__", 1)
                full_model = f"{org}/{model_name}"
            else:
                dataset = dataset_org
                full_model = model_name
            short_id = hash_part[:6] if hash_part else "master"
            return f"{full_model.replace('/', '_')}_{dataset}_{short_id}"

    if "models" in parts:
        idx = parts.index("models")
        if len(parts) > idx + 4:
            org = parts[idx + 1]
            model_name = parts[idx + 2]
            if parts[idx + 3] == "datasets" and len(parts) > idx + 5:
                dataset = parts[idx + 4]
                run_name = parts[-1]
                if run_name == "matrix_runs":
                    short_id = parts[-1][-8:] if len(parts[-1]) > 8 else parts[-1]
                else:
                    m = re.search(r"__hash([a-f0-9]+)", parts[-1])
                    if m:
                        short_id = m.group(1)[:6]
                    else:
                        short_id = parts[-1][-8:]
                full_model = f"{org}/{model_name}"
                return f"{full_model.replace('/', '_')}_{dataset}_{short_id}"

    return run_dir.name[:20]


def discover_all_runs(root: Path = DEFAULT_ROOT, experiment_id: str = DEFAULT_EXPERIMENT_ID) -> List[Path]:
    """Return a list of all directories that contain probe result CSVs."""
    runs = []
    master_runs = root / "master_runs"
    if master_runs.exists():
        for master_csv in master_runs.glob("**/master_results.csv"):
            runs.append(master_csv.parent)
    exp_models = root / "experiments" / experiment_id / "models"
    if exp_models.exists():
        for probe_csv in exp_models.glob("**/layer_probe_results.csv"):
            runs.append(probe_csv.parent)
    return sorted(set(runs))


def compare_runs(run_dirs: Sequence[Path], output_dir: Path, renderer: Renderer) -> None:
    """Compare multiple completed runs by loading their result CSVs."""
    frames = []
    for d in run_dirs:
        d = Path(d)
        if not d.exists():
            renderer.warning(f"{d} does not exist, skipping.")
            continue
        csv_files = list(d.glob("**/master_results.csv"))
        if not csv_files:
            csv_files = list(d.glob("**/layer_probe_results.csv"))
        if not csv_files:
            renderer.warning(f"No result CSV found in {d}, skipping.")
            continue

        label = get_run_label(d)
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df["run"] = label
            frames.append(df)

    if not frames:
        raise ValueError("No result CSV files found in provided run directories.")

    combined = pd.concat(frames, ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_dir / "combined_results.csv", index=False)

    # Summary
    summary = combined.groupby(["run", "probe"]).agg(
        best_macro_f1=("test_macro_f1", "max"),
        mean_macro_f1=("test_macro_f1", "mean"),
        best_layer=("layer_index", lambda x: x[combined.loc[x.index, "test_macro_f1"].idxmax()] if len(x) else None)
    ).reset_index()

    # Display summary
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
                str(int(row["best_layer"])) if pd.notna(row["best_layer"]) else "",
            )
        renderer.console.print(table)
    else:
        print("\nComparison Summary:")
        print(summary.to_string(index=False))

    # Generate comparison plots if plotting available
    if PLOTTING_AVAILABLE:
        if "test_macro_f1" in combined.columns:
            best_per_run_probe = combined.groupby(["run", "probe"])["test_macro_f1"].max().reset_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(data=best_per_run_probe, x="run", y="test_macro_f1", hue="probe")
            plt.title("Best Test Macro-F1 per Run and Probe")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "comparison_best_macro_f1.png", dpi=240)
            plt.close()

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

    renderer.success(f"Comparison plots saved to {output_dir}")


# ============================================================================
# 9. Model-based Comparison
# ============================================================================

def compare_models(model_a: str, model_b: str, root: Path, exp_id: str, output_dir: Path, renderer: Renderer):
    """Compare all runs of two models, pairing by dataset and config hash when possible."""
    entries = get_analysable_entries(root, exp_id)
    runs_a = [e for e in entries if e["model"] == model_a]
    runs_b = [e for e in entries if e["model"] == model_b]
    if not runs_a:
        renderer.error(f"No runs found for model '{model_a}'.")
        return
    if not runs_b:
        renderer.error(f"No runs found for model '{model_b}'.")
        return

    # Pair by dataset and config hash
    paired = []
    # Create lookup for B by (dataset, config_hash)
    b_lookup = {}
    for r in runs_b:
        key = (r["dataset"], r["config_hash"])
        b_lookup.setdefault(key, []).append(r)

    for a in runs_a:
        key = (a["dataset"], a["config_hash"])
        if key in b_lookup:
            # Pair each B run with this A run
            for b in b_lookup[key]:
                paired.append((Path(a["path"]), Path(b["path"])))
        else:
            # No exact match: pair with all B runs of same dataset
            same_dataset = [r for r in runs_b if r["dataset"] == a["dataset"]]
            if same_dataset:
                for b in same_dataset:
                    paired.append((Path(a["path"]), Path(b["path"])))
            else:
                renderer.warning(f"No matching dataset for {a['model']} {a['dataset']}; skipping.")

    if not paired:
        renderer.error("No compatible runs to pair.")
        return

    # Collect all unique run directories
    all_dirs = []
    for a_dir, b_dir in paired:
        all_dirs.append(a_dir)
        all_dirs.append(b_dir)
    all_dirs = list(dict.fromkeys(all_dirs))
    renderer.info(f"Comparing {len(all_dirs)} runs across models.")
    compare_runs(all_dirs, output_dir, renderer)


# ============================================================================
# 10. Interactive Mode
# ============================================================================

class InteractiveApp:
    """
    Friendly terminal front‑end for the analyser.

    Exposes a guided menu that asks for analysis type, filtering options,
    and runs the selected analysis.
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

    def run(self) -> Dict[str, Any]:
        """Launch the interactive session."""
        self.renderer.title(
            "EMOTION PROBE LAB – ANALYSER",
            "Guided analysis interface",
        )

        # Analysis type
        analysis_type = self.choose(
            "What would you like to analyse?",
            [
                "Extraction health check (hidden states)",
                "Probe results (CSV analysis)",
                "Full analysis (both extraction and probes)",
                "Compare multiple runs",
                "Generate plots from probe results",
                "Audit a specific run",
                "List analysable runs",
                "List models with artifact status",
            ]
        )

        # Common options
        project_root = self.ask("Project root directory", ".")
        result_root = self.ask("Result root directory", str(DEFAULT_ROOT))
        exp_id = self.ask("Experiment ID", DEFAULT_EXPERIMENT_ID)
        quiet = self.ask("Quiet mode? (y/n)", "n").lower() == "y"

        # Create analyser
        analyser = ProjectAnalyser(
            project_root=project_root,
            result_root=result_root,
            exp_id=exp_id,
            quiet=quiet,
        )

        # Dispatch based on choice
        if analysis_type.startswith("Extraction"):
            analyser.analyse_extraction()
        elif analysis_type.startswith("Probe"):
            analyser.analyse_probes()
        elif analysis_type.startswith("Full"):
            analyser.analyse_all()
            html = self.ask("Save HTML report? (y/n)", "y").lower() == "y"
            if html:
                output = self.ask("Output HTML file", "report.html")
                analyser.generate_report(output_html=output)
        elif analysis_type.startswith("Compare"):
            # Show available runs first
            list_analysable_runs(Path(result_root), exp_id)
            self.renderer.info("Enter the indices (space-separated) of the runs to compare, or paste the full paths:")
            raw_input = input("Runs: ").strip()
            if not raw_input:
                self.renderer.error("No input provided.")
                return {"status": "cancelled"}

            run_dirs = []
            # Try to parse as indices first
            parts = raw_input.split()
            indices = []
            for p in parts:
                if p.isdigit():
                    indices.append(int(p))
                else:
                    # Treat as raw path string (maybe garbage)
                    cleaned = clean_path_input(p)
                    run_dirs.extend(cleaned)

            if indices:
                entries = get_analysable_entries(Path(result_root), exp_id)
                if entries:
                    for idx in indices:
                        if 1 <= idx <= len(entries):
                            run_dirs.append(Path(entries[idx-1]["path"]))
                        else:
                            self.renderer.warning(f"Index {idx} out of range.")

            # If still empty, try to parse the entire raw input as paths
            if not run_dirs:
                run_dirs = clean_path_input(raw_input)

            if not run_dirs:
                self.renderer.error("No valid run directories provided.")
                return {"status": "cancelled"}

            output_dir = self.ask("Output directory for comparison plots", "./comparison_plots")
            compare_runs(run_dirs, Path(output_dir), analyser.renderer)
        elif analysis_type.startswith("Generate plots"):
            analyser.analyse_probes()  # ensure data loaded
            output_dir = self.ask("Output directory for plots", "./analysis_plots")
            analyser.generate_plots(Path(output_dir))
        elif analysis_type.startswith("Audit"):
            run_dir = self.ask("Run directory to audit")
            if not run_dir:
                self.renderer.error("No run directory provided.")
                return {"status": "cancelled"}
            auditor = ForensicAuditor(
                root=Path(result_root) / "experiments" / exp_id,
                experiment_id=exp_id,
                renderer=analyser.renderer,
            )
            report = auditor.audit_trial(Path(run_dir))
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        elif analysis_type.startswith("List analysable"):
            list_analysable_runs(Path(result_root), exp_id)
        elif analysis_type.startswith("List models"):
            list_available_models(Path(result_root), exp_id)
        else:
            self.renderer.warning("Unknown selection.")

        self.renderer.success("Analysis complete.")
        return {"status": "done"}


# ============================================================================
# 11. CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true", help="Show help")
    sub = parser.add_subparsers(dest="command")

    # extraction
    ext = sub.add_parser("extraction", help="Analyse extraction outputs.", add_help=False)
    ext.add_argument("--model", help="Filter by model name (substring)")
    ext.add_argument("--dataset", help="Filter by dataset name (substring)")
    ext.add_argument("-h", "--help", action="store_true", help="Show help")

    # probes
    probes = sub.add_parser("probes", help="Analyse probe results.", add_help=False)
    probes.add_argument("--model", help="Filter by model name (substring)")
    probes.add_argument("--dataset", help="Filter by dataset name (substring)")
    probes.add_argument("-h", "--help", action="store_true", help="Show help")

    # all
    all_cmd = sub.add_parser("all", aliases=["report"], help="Full analysis.", add_help=False)
    all_cmd.add_argument("--model", help="Filter by model name (substring)")
    all_cmd.add_argument("--dataset", help="Filter by dataset name (substring)")
    all_cmd.add_argument("--output-html", default="report.html", help="HTML report file")
    all_cmd.add_argument("-h", "--help", action="store_true", help="Show help")

    # compare
    compare = sub.add_parser("compare", help="Compare multiple runs or two models.", add_help=False)
    compare.add_argument("run_dirs", nargs="*", help="Directories containing result CSVs.")
    compare.add_argument("-a", "--all", action="store_true", help="Compare all available runs.")
    compare.add_argument("--model1", help="First model name to compare (e.g., 'BERT')")
    compare.add_argument("--model2", help="Second model name to compare")
    compare.add_argument("-o", "--output-dir", default="./comparison_plots")
    compare.add_argument("-h", "--help", action="store_true", help="Show help")

    # audit
    audit = sub.add_parser("audit", help="Audit a specific run directory.", add_help=False)
    audit.add_argument("run_dir", help="Run directory to audit.")
    audit.add_argument("-h", "--help", action="store_true", help="Show help")

    # list-runs
    list_runs = sub.add_parser("list-runs", help="List analysable runs.", add_help=False)
    list_runs.add_argument("-h", "--help", action="store_true", help="Show help")

    # list-models
    list_models = sub.add_parser("list-models", help="List models with artifact status.", add_help=False)
    list_models.add_argument("-h", "--help", action="store_true", help="Show help")

    # plots
    plots = sub.add_parser("plots", help="Generate plots from probe results.", add_help=False)
    plots.add_argument("--model", help="Filter by model name (substring)")
    plots.add_argument("--dataset", help="Filter by dataset name (substring)")
    plots.add_argument("-o", "--output-dir", help="Output directory for plots.", default="./analysis_plots")
    plots.add_argument("-h", "--help", action="store_true", help="Show help")

    # interactive
    interactive = sub.add_parser("interactive", help="Launch guided interactive mode.", add_help=False)
    interactive.add_argument("-h", "--help", action="store_true", help="Show help")

    # Common options
    for p in [ext, probes, all_cmd, plots, list_runs, list_models, interactive, audit, compare]:
        p.add_argument("--project-root", default=".", help="Path to Final-Year-Project")
        p.add_argument("--result-root", default=str(DEFAULT_ROOT), help="Path to hidden_states")
        p.add_argument("--exp-id", default=DEFAULT_EXPERIMENT_ID, help="Experiment ID")
        p.add_argument("--quiet", action="store_true", help="Suppress non-essential output")

    return parser


def print_help(parser):
    renderer = Renderer()
    if RICH_AVAILABLE:
        renderer.title("EMOTION PROBE LAB – ANALYSER", "Unified analysis tool")
        table = Table(title="Available Commands", show_lines=True, header_style="bold cyan")
        table.add_column("Command", style="bold magenta", min_width=15)
        table.add_column("Description", style="white")
        table.add_row("extraction", "Analyse extraction artifacts (hidden states)")
        table.add_row("probes", "Analyse probe result CSV files")
        table.add_row("all", "Full analysis (both extraction and probes)")
        table.add_row("compare", "Compare multiple runs or two models (use --model1/--model2)")
        table.add_row("audit", "Forensic audit of a run directory")
        table.add_row("list-runs", "List analysable runs")
        table.add_row("list-models", "List models with artifact status")
        table.add_row("plots", "Generate plots from probe results")
        table.add_row("interactive", "Launch guided interactive mode")
        renderer.console.print(table)
        renderer.info("Run 'python Master_Analyser.py <command> --help' for command-specific options.")
    else:
        parser.print_help()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.help:
        print_help(parser)
        return 0

    # Create analyser
    analyser = ProjectAnalyser(
        project_root=args.project_root,
        result_root=args.result_root,
        exp_id=args.exp_id,
        quiet=getattr(args, "quiet", False),
        model_filter=getattr(args, "model", None),
        dataset_filter=getattr(args, "dataset", None),
    )

    if args.command == "interactive":
        app = InteractiveApp()
        app.run()
        return 0

    if args.command == "extraction":
        analyser.analyse_extraction()
    elif args.command == "probes":
        analyser.analyse_probes()
    elif args.command in ("all", "report"):
        analyser.analyse_all()
        analyser.generate_report(output_html=getattr(args, "output_html", "report.html"))
    elif args.command == "compare":
        if args.model1 and args.model2:
            # Resolve model names if needed
            try:
                model1 = resolve_model_name(args.model1)
                model2 = resolve_model_name(args.model2)
            except ValueError as e:
                analyser.renderer.error(str(e))
                return 1
            compare_models(model1, model2, Path(args.result_root), args.exp_id,
                           Path(args.output_dir), analyser.renderer)
        elif args.all:
            run_dirs = discover_all_runs(args.result_root, args.exp_id)
            if not run_dirs:
                analyser.renderer.error("No runs found.")
                return 1
            compare_runs(run_dirs, Path(args.output_dir), analyser.renderer)
        else:
            if not args.run_dirs:
                analyser.renderer.error("No run directories or models provided.")
                return 1
            # Also allow clean_path_input on each argument to handle garbage paths
            all_paths = []
            for arg in args.run_dirs:
                cleaned = clean_path_input(arg)
                if cleaned:
                    all_paths.extend(cleaned)
                else:
                    # Fallback to raw Path
                    all_paths.append(Path(arg))
            compare_runs(all_paths, Path(args.output_dir), analyser.renderer)
    elif args.command == "audit":
        auditor = ForensicAuditor(
            root=Path(args.result_root) / "experiments" / args.exp_id,
            experiment_id=args.exp_id,
            renderer=analyser.renderer,
        )
        report = auditor.audit_trial(Path(args.run_dir))
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif args.command == "list-runs":
        list_analysable_runs(Path(args.result_root), args.exp_id)
    elif args.command == "list-models":
        list_available_models(Path(args.result_root), args.exp_id)
    elif args.command == "plots":
        if analyser.probe_report is None:
            analyser.analyse_probes()
        analyser.generate_plots(Path(args.output_dir))
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())