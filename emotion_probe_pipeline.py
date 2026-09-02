
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMOTION PROBE LAB
=================

Single user-facing orchestrator for the hidden-state emotion project.

Design goals
------------
* One stable public interface over the existing extraction + probing modules.
* Factories for datasets, probes, stages and renderers.
* Explicit state machine rather than a fragile linear script.
* Deterministic configuration hashing and immutable run manifests.
* Forensic preflight checks before any expensive fitting.
* Separate single-label / multi-label evaluation semantics.
* Reuse of existing completed artifacts; never silently overwrite them.
* Interactive terminal UI with an optional Rich renderer.
* CLI for scripted / notebook use.

The heavy scientific implementations remain in the existing project modules:
    - Extraction6_new.py (or another extraction module configured below)
    - unified_hidden_state_probe_v4_3.py

The master file orchestrates them; it does not duplicate their model/extraction code.
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
from typing import Any, Callable, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn,
    )
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False


# ============================================================================
# Stable project constants
# ============================================================================

MASTER_VERSION = "1.0.0"

DEFAULT_ROOT = Path("/Volumes/Amirali/hidden_states")
DEFAULT_EXPERIMENT_ID = "baseline_v5_001"

# Existing project module names. The first importable candidate is used.
DEFAULT_EXTRACTION_MODULES = (
    "Extraction6_new",
    "Extraction6",
    "Extraction5",
)

DEFAULT_PROBE_MODULES = (
    "unified_hidden_state_probe_v4_3",
    "unified_hidden_state_probe_v4_2",
)

DATASET_SPECS = {
    "goEmo": {
        "label": "GoEmotions",
        "module": "Get_Go_Emo",
        "function": "get_go",
        "task_type": "multi_label",
        "class_count": 28,
    },
    "ISEAR": {
        "label": "ISEAR",
        "module": "Get_Isear",
        "function": "get_isr",
        "task_type": "single_label",
        "class_count": 7,
    },
}

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
# Utilities
# ============================================================================

def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def normalize_model_name(name: str) -> str:
    return str(name).strip()


def safe_model_path(model_name: str) -> Path:
    return Path(*[p for p in model_name.split("/") if p])


def dataset_artifact_dir(
    root: Path,
    experiment_id: str,
    model_name: str,
    dataset_name: str,
) -> Path:
    return (
        root / "experiments" / experiment_id / "models"
        / safe_model_path(model_name) / "datasets" / dataset_name
    )


def import_first(candidates: Sequence[str]):
    errors = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ImportError(
        "Could not import any candidate module:\n- " + "\n- ".join(errors)
    )


def make_dataclass_instance(module: Any, class_name: str, payload: Mapping[str, Any]):
    cls = getattr(module, class_name)
    return cls(**dict(payload))


def compact_path(path: Path, max_chars: int = 74) -> str:
    s = str(path)
    return s if len(s) <= max_chars else "…" + s[-(max_chars - 1):]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProbeChoice:
    preset: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class MasterConfig:
    root: Path = DEFAULT_ROOT
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    model: str | None = None
    dataset: str | None = None

    # Scientific defaults: deliberately conservative.
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

    extraction_module: str | None = None
    probe_module: str | None = None

    probes: list[ProbeChoice] = field(
        default_factory=lambda: [ProbeChoice("logistic")]
    )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["root"] = str(self.root)
        return d

    @property
    def config_hash(self) -> str:
        return stable_hash(self.as_dict(), 20)


# ============================================================================
# Factories
# ============================================================================

class ModuleFactory:
    """Resolves project implementation modules without hard-coding one filename."""

    @staticmethod
    def extraction(config: MasterConfig):
        if config.extraction_module:
            return importlib.import_module(config.extraction_module)
        return import_first(DEFAULT_EXTRACTION_MODULES)

    @staticmethod
    def probe(config: MasterConfig):
        if config.probe_module:
            return importlib.import_module(config.probe_module)
        return import_first(DEFAULT_PROBE_MODULES)


class DatasetFactory:
    """Dataset adapters. The master pipeline never manipulates raw dataset labels."""

    @staticmethod
    def spec(name: str) -> dict[str, Any]:
        key = str(name)
        if key not in DATASET_SPECS:
            raise KeyError(
                f"Unsupported dataset {key!r}. Available: {sorted(DATASET_SPECS)}"
            )
        return dict(DATASET_SPECS[key])

    @staticmethod
    def load(name: str) -> Any:
        spec = DatasetFactory.spec(name)
        module = importlib.import_module(spec["module"])
        fn = getattr(module, spec["function"])
        return fn()


class ProbeFactory:
    """Creates ProbeSpec objects belonging to the installed probe implementation."""

    @staticmethod
    def create(probe_module: Any, choices: Sequence[ProbeChoice]):
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

            # Make names unique if users intentionally create variants.
            probes.append(ProbeSpec(**payload))

        if not probes:
            raise ValueError("At least one probe must be selected.")
        return probes


# ============================================================================
# Stage model
# ============================================================================

class Stage:
    PREFLIGHT = "PREFLIGHT"
    EXTRACTION = "EXTRACTION"
    PROBING = "PROBING"
    ANALYSIS = "ANALYSIS"
    REPORT = "REPORT"
    COMPLETE = "COMPLETE"


@dataclass
class StageRecord:
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
    """Persistent, human-readable state machine record."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, StageRecord] = {}

    def start(self, stage: str) -> None:
        self.records[stage] = StageRecord(
            name=stage, status="running", started_at=time.time()
        )
        self.save()

    def finish(self, stage: str, message: str = "") -> None:
        rec = self.records.setdefault(stage, StageRecord(name=stage))
        rec.status = "complete"
        rec.finished_at = time.time()
        rec.message = message
        self.save()

    def fail(self, stage: str, message: str) -> None:
        rec = self.records.setdefault(stage, StageRecord(name=stage))
        rec.status = "failed"
        rec.finished_at = time.time()
        rec.message = message
        self.save()

    def save(self) -> None:
        save_json(
            self.path,
            {"records": {k: asdict(v) for k, v in self.records.items()}},
        )


# ============================================================================
# Presentation
# ============================================================================

class Renderer:
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None

    def title(self, title: str, subtitle: str = "") -> None:
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
        if self.console:
            self.console.print(text)
        else:
            print(text)

    def warning(self, text: str) -> None:
        if self.console:
            self.console.print(f"[yellow]WARNING[/yellow] {text}")
        else:
            print(f"WARNING: {text}")

    def success(self, text: str) -> None:
        if self.console:
            self.console.print(f"[green]✓[/green] {text}")
        else:
            print(f"✓ {text}")

    def error(self, text: str) -> None:
        if self.console:
            self.console.print(f"[red]✗[/red] {text}")
        else:
            print(f"ERROR: {text}")

    def stage(self, current: str, stages: Sequence[str]) -> None:
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
        if df.empty:
            self.warning("No result rows available.")
            return

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
# Forensic integrity auditor
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
# Task-aware analysis
# ============================================================================

class TaskAwareAnalyzer:
    """Analysis layer deliberately keeps multi-label and single-label semantics separate."""

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
        # Conservative chance reference for balanced random guessing.
        return 1.0 / max(class_count, 1)

    @staticmethod
    def normalized_macro_f1(
        score: float,
        chance: float,
        *,
        clip: bool = True,
    ) -> float:
        denom = 1.0 - chance
        if denom <= 0:
            return float("nan")
        value = (float(score) - chance) / denom
        return float(np.clip(value, 0.0, 1.0) if clip else value)

    @staticmethod
    def add_task_aware_columns(df: pd.DataFrame) -> pd.DataFrame:
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

        # For multi-label, random macro-F1 is prevalence-dependent and cannot
        # honestly be represented by 1 / 28. Keep a reference explicit.
        multi = out["task_type"].eq("multi_label")
        if multi.any():
            out.loc[multi, "chance_macro_f1_reference"] = np.nan
            out.loc[multi, "normalized_macro_f1"] = np.nan

        return out

    @staticmethod
    def multi_label_prevalence_baseline(
        y_true: np.ndarray,
        *,
        seed: int = 42,
        repeats: int = 3,
    ) -> float:
        """
        Estimate a genuine multi-label null baseline by preserving each label's
        empirical prevalence while randomizing sample assignment.

        This is preferable to inventing a single universal '1/28' chance value.
        """
        y_true = np.asarray(y_true)
        if y_true.ndim != 2:
            raise ValueError("Expected [N, C] binary multi-label matrix.")

        rng = np.random.default_rng(seed)
        scores = []

        # Import only when this optional diagnostic is used.
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

        # Never merge semantics across task types.
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
# Report generation
# ============================================================================

class ReportBuilder:
    @staticmethod
    def infer_patterns(df: pd.DataFrame, task_type: str) -> list[str]:
        if df.empty:
            return ["No completed probe results are available."]

        messages = []

        metric = (
            "test_macro_f1"
            if "test_macro_f1" in df.columns
            else None
        )

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
                    f"Best mean test Macro-F1 occurs at layer {best_layer} "
                    f"({best_value:.3f})."
                )

                if best_value > 0.98:
                    messages.append(
                        "The result is in the forensic-risk zone (>0.98); "
                        "do not interpret it as genuine emotion recoverability "
                        "until split, alignment and null controls pass."
                    )

                delta = last_value - first_value
                if abs(delta) < 0.02:
                    messages.append(
                        "The representation is approximately depth-invariant "
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
                    f"Mean true-label selectivity over the recorded rows is {s:.3f}; "
                    "this should be interpreted relative to the matched shuffled-label control."
                )

        if task_type == "multi_label":
            messages.append(
                "For GoEmotions, prioritize Macro-F1, Micro-F1, average precision, "
                "per-label F1, label cardinality and the shuffled-label null. "
                "Do not treat 1/28 as a meaningful universal chance Macro-F1."
            )
        else:
            messages.append(
                "For ISEAR, report Macro-F1, balanced accuracy, MCC, confusion "
                "matrix and class-wise recall/F1; the 1/7 reference is interpretable "
                "as a simple uniform-class accuracy baseline, not as a complete null."
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
# Main orchestrator
# ============================================================================

class EmotionProbePipeline:
    """
    Public master API.

    The constructor resolves the scientific configuration but does not start
    expensive computation. Users can therefore inspect/modify the pipeline
    before execution.

    Example
    -------
        pipe = EmotionProbePipeline(
            model="google-bert/bert-base-uncased",
            dataset="goEmo",
            max_samples=5000,
        )
        result = pipe.run()
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
        probes: Sequence[str | ProbeChoice] = ("logistic",),
        repeats: int = 3,
        max_samples: int | None = 5000,
        seed: int = 42,
        shuffled_label_control: bool = True,
        shuffled_control_repeats: int = 3,
        extraction_module: str | None = None,
        probe_module: str | None = None,
        strict_provenance: bool = True,
    ):
        self.renderer = Renderer()

        normalized_choices = []
        for p in probes:
            if isinstance(p, ProbeChoice):
                normalized_choices.append(p)
            else:
                normalized_choices.append(ProbeChoice(str(p)))

        self.config = MasterConfig(
            root=Path(root).resolve(),
            experiment_id=experiment_id,
            model=normalize_model_name(model),
            dataset=str(dataset),
            probes=normalized_choices,
            repeats=int(repeats),
            max_samples=max_samples,
            seed=int(seed),
            shuffled_label_control=bool(shuffled_label_control),
            shuffled_control_repeats=int(shuffled_control_repeats),
            extraction_module=extraction_module,
            probe_module=probe_module,
            strict_provenance=bool(strict_provenance),
        )

        self._validate_config()

        # Run directory is deterministic and configuration-addressed.
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

    # ---------------------------------------------------------------------
    # Public configuration helpers
    # ---------------------------------------------------------------------

    def describe(self) -> None:
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
        return dataset_artifact_dir(
            self.config.root,
            self.config.experiment_id,
            self.config.model,
            self.config.dataset,
        )

    def set_probes(self, *probes: str) -> "EmotionProbePipeline":
        self.config.probes = [ProbeChoice(p) for p in probes]
        self._validate_config()
        save_json(self.run_root / "master_config.json", self.config.as_dict())
        return self

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def _validate_config(self) -> None:
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

    # ---------------------------------------------------------------------
    # Artifact / extraction management
    # ---------------------------------------------------------------------

    def _artifact_exists(self) -> bool:
        required = [
            self.artifact_dir / "data" / "hidden_states.npy",
            self.artifact_dir / "data" / "completed.npy",
            self.artifact_dir / "metadata" / "extraction.json",
        ]
        return all(p.exists() for p in required)

    def _load_artifact(self):
        self.probe_mod = ModuleFactory.probe(self.config)
        ExtractionArtifact = getattr(self.probe_mod, "ExtractionArtifact")
        self.artifact = ExtractionArtifact(
            self.artifact_dir,
            verify_checksum=self.config.verify_checksum,
        )
        return self.artifact

    def ensure_extraction(self, *, run_if_missing: bool = False) -> None:
        if self._artifact_exists():
            self.renderer.success(
                f"Frozen hidden-state artifact found: {compact_path(self.artifact_dir)}"
            )
            self._load_artifact()
            return

        if not run_if_missing:
            raise FileNotFoundError(
                "No compatible hidden-state artifact exists.\n"
                f"Expected: {self.artifact_dir}\n"
                "Run extraction first or call ensure_extraction(run_if_missing=True)."
            )

        self.extraction_mod = ModuleFactory.extraction(self.config)
        if not hasattr(self.extraction_mod, "run_experiments"):
            raise AttributeError(
                "Extraction module does not expose run_experiments(...)."
            )

        dataset_obj = DatasetFactory.load(self.config.dataset)
        self.renderer.warning(
            "Extraction artifact is missing. Starting deterministic extraction."
        )

        # Keep extraction configuration explicit. Do not enable adaptive batch
        # changes: the existing extractor explicitly treats those as invalid.
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

        if not self._artifact_exists():
            raise RuntimeError(
                "Extraction completed without producing the expected artifact."
            )

        self._load_artifact()

    # ---------------------------------------------------------------------
    # Forensic preflight
    # ---------------------------------------------------------------------

    def forensic_preflight(self) -> dict[str, Any]:
        self.state.start(Stage.PREFLIGHT)
        self.renderer.stage(Stage.PREFLIGHT, self.STAGES)

        self.ensure_extraction(run_if_missing=False)

        # Cheap artifact-level checks.
        if not np.all(self.artifact.completed):
            raise RuntimeError(
                "Extraction completion map is incomplete. Refusing to probe."
            )

        if self.artifact.states.shape[0] != self.artifact.sample_count:
            raise RuntimeError("Hidden-state sample count does not match metadata.")

        # The probe implementation already validates text and target alignment.
        # We still force the alignment routines here before expensive fitting.
        ds_spec = DatasetFactory.spec(self.config.dataset)
        dataset_df = DatasetFactory.load(self.config.dataset)

        # UnifiedProbeAnalyzer expects its DatasetContract. Construct it from
        # the installed module's dataclass, keeping the adapter centralized.
        DatasetContract = getattr(self.probe_mod, "DatasetContract")
        contract = DatasetContract(
            target_type=self.config.dataset.lower()
            if self.config.dataset in {"goEmo", "ISEAR"} else "auto",
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
                f"Target count {len(y)} != hidden-state count {self.artifact.sample_count}."
            )

        validate_targets = getattr(self.probe_mod, "validate_targets")
        target_validation = validate_targets(
            y, classes, ds_spec["task_type"]
        )

        validate_text_alignment = getattr(
            self.probe_mod, "validate_text_alignment"
        )
        validate_label_alignment = getattr(
            self.probe_mod, "validate_label_alignment"
        )

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

        # Existing result audit, if this master config points to an old run.
        analyzer = ForensicAuditor(
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
            report = analyzer.audit_trial(completion.parent)
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

    # ---------------------------------------------------------------------
    # Probe execution
    # ---------------------------------------------------------------------

    def run_probes(self) -> pd.DataFrame:
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
            target_type=self.config.dataset.lower(),
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

        scored, best = analyzer.run()

        if scored is None or scored.empty:
            raise RuntimeError("Probe execution produced no scored results.")

        # Attach master provenance.
        scored = scored.copy()
        scored["master_version"] = MASTER_VERSION
        scored["master_config_hash"] = self.config.config_hash

        # Task-aware normalization.
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

    # ---------------------------------------------------------------------
    # Analysis / reporting
    # ---------------------------------------------------------------------

    def analyze(self) -> dict[str, Any]:
        if self.results.empty:
            result_path = self.run_root / "master_results.csv"
            if result_path.exists():
                self.results = pd.read_csv(result_path)
            else:
                raise RuntimeError("No probe results available.")

        self.state.start(Stage.ANALYSIS)

        task_type = DatasetFactory.spec(self.config.dataset)["task_type"]

        summary = TaskAwareAnalyzer.summarize(self.results)

        # Strictly separate result tables by task type.
        if "task_type" in self.results.columns:
            for task, group in self.results.groupby("task_type"):
                group.to_csv(
                    self.run_root / f"results__{task}.csv",
                    index=False,
                )

        save_json(self.run_root / "analysis_summary.json", summary)

        self.renderer.result_table(self.results, task_type)

        self.state.finish(Stage.ANALYSIS, "Task-aware analysis complete.")
        return summary

    def report(self) -> Path:
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

    # ---------------------------------------------------------------------
    # End-to-end
    # ---------------------------------------------------------------------

    def run(
        self,
        *,
        extract_if_missing: bool = False,
    ) -> dict[str, Any]:
        self.describe()

        self.state.start(Stage.PREFLIGHT)
        self.state.finish(Stage.PREFLIGHT, "Delegating to forensic_preflight.")

        # Use a dedicated call so its checks remain independently callable.
        self.forensic_preflight()

        self.run_probes()
        summary = self.analyze()
        self.report()

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
# Interactive interface
# ============================================================================

class InteractiveApp:
    """
    Friendly terminal front-end.

    It intentionally exposes a small number of decisions and delegates all
    scientific validation to EmotionProbePipeline.
    """

    def __init__(self):
        self.renderer = Renderer()

    def ask(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        return value if value else (default or "")

    def choose(self, title: str, options: Sequence[str]) -> str:
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

    def run(self) -> dict[str, Any]:
        self.renderer.title(
            "EMOTION PROBE LAB",
            "A controlled interface for frozen hidden-state emotion probing",
        )

        model = self.ask(
            "Model",
            "google-bert/bert-base-uncased",
        )

        dataset = self.choose(
            "Dataset",
            ["goEmo", "ISEAR"],
        )

        print("\nProbe selection")
        print("  ENTER = linear logistic baseline")
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

        max_raw = self.ask(
            "Maximum samples (ENTER = 5000, type FULL for all)",
            "5000",
        )
        max_samples = None if max_raw.upper() == "FULL" else int(max_raw)

        repeats = int(self.ask("Independent repeats", "3"))
        controls = self.ask("Run shuffled-label control? Y/N", "Y").upper() == "Y"

        pipeline = EmotionProbePipeline(
            model=model,
            dataset=dataset,
            probes=probes,
            repeats=repeats,
            max_samples=max_samples,
            shuffled_label_control=controls,
        )

        pipeline.describe()

        confirmation = self.ask(
            "Start this experiment? Y/N",
            "Y",
        ).upper()

        if confirmation != "Y":
            self.renderer.info("Experiment cancelled before fitting.")
            return {"status": "cancelled"}

        return pipeline.run()


# ============================================================================
# Existing-run audit mode
# ============================================================================

def audit_existing_run(run_dir: str | Path) -> dict[str, Any]:
    """
    Standalone audit helper.

    Crucially, this does NOT retrain anything and does NOT modify old results.
    """
    path = Path(run_dir).resolve()
    auditor = ForensicAuditor(
        root=path,
        experiment_id="audit-only",
        renderer=Renderer(),
    )
    report = auditor.audit_trial(path)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Master interface for the frozen hidden-state emotion probing project."
    )

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run an experiment.")
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    run.add_argument("--root", default=str(DEFAULT_ROOT))
    run.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    run.add_argument("--probe", action="append", default=["logistic"],
                     choices=["logistic", "mlp1", "mlp2", "mlp3"])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--max-samples", type=int, default=5000)
    run.add_argument("--full", action="store_true")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--no-shuffle-control", action="store_true")
    run.add_argument("--extract-if-missing", action="store_true")

    audit = sub.add_parser("audit", help="Audit an existing probe run without retraining.")
    audit.add_argument("run_dir")

    interactive = sub.add_parser("interactive", help="Launch the guided interface.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "interactive":
        InteractiveApp().run()
        return 0

    if args.command == "audit":
        report = audit_existing_run(args.run_dir)
        return 1 if report["status"] == "FAIL" else 0

    if args.command == "run":
        probes = []
        seen = set()
        for p in args.probe:
            if p not in seen:
                probes.append(p)
                seen.add(p)

        pipeline = EmotionProbePipeline(
            model=args.model,
            dataset=args.dataset,
            root=args.root,
            experiment_id=args.experiment_id,
            probes=probes,
            repeats=args.repeats,
            max_samples=None if args.full else args.max_samples,
            seed=args.seed,
            shuffled_label_control=not args.no_shuffle_control,
        )

        pipeline.run(extract_if_missing=args.extract_if_missing)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
