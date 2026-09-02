#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMOTION PROBE LAB – DEMONSTRATION PIPELINE (v3.4.0)
===================================================
Reads hidden-state artifacts, never modifies experiment files.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

# ----------------------------- Constants -----------------------------
MASTER_VERSION = "3.4.0"
DEFAULT_ROOT = Path("/Volumes/Amirali/hidden_states")
DEFAULT_EXPERIMENT_ID = "baseline_v5_001"
DEFAULT_OUTPUT_DIR = Path("./demo_runs")

EXTRACTION_MODULES = ("Extraction6_new", "Extraction6", "Extraction5")
PROBE_MODULES = ("unified_hidden_state_probe_v4_3", "unified_hidden_state_probe_v4_2")

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

PROBE_PRESETS = {
    "logistic": {"name": "linear_logistic", "type": "logistic", "complexity": "linear",
                 "standardize": True, "C": 1.0, "max_iter": 3000, "selection_metric": "macro_f1"},
    "mlp1": {"name": "mlp_1_hidden", "type": "mlp", "complexity": "1_hidden",
             "standardize": True, "hidden_dims": ["0.5d"], "learning_rate": 1e-3,
             "weight_decay": 1e-4, "epochs": 80, "batch_size": 256, "patience": 12, "dropout": 0.0,
             "selection_metric": "macro_f1"},
    "mlp2": {"name": "mlp_2_hidden", "type": "mlp", "complexity": "2_hidden",
             "standardize": True, "hidden_dims": ["0.5d", "0.25d"], "learning_rate": 1e-3,
             "weight_decay": 1e-4, "epochs": 80, "batch_size": 256, "patience": 12, "dropout": 0.0,
             "selection_metric": "macro_f1"},
    "mlp3": {"name": "mlp_3_hidden", "type": "mlp", "complexity": "3_hidden",
             "standardize": True, "hidden_dims": ["0.5d", "0.25d", "0.125d"], "learning_rate": 1e-3,
             "weight_decay": 1e-4, "epochs": 80, "batch_size": 256, "patience": 12, "dropout": 0.0,
             "selection_metric": "macro_f1"},
}

# ----------------------------- Utilities -----------------------------
def stable_hash(value, length=16):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]

def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def normalize_model_name(name):
    return str(name).strip()

def safe_model_path(model_name):
    return Path(*[p for p in model_name.split("/") if p])

def dataset_artifact_dir(root, experiment_id, model_name, dataset_name):
    return root / "experiments" / experiment_id / "models" / safe_model_path(model_name) / "datasets" / dataset_name

def import_first(candidates):
    errors = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ImportError("Could not import any candidate module:\n- " + "\n- ".join(errors))

def get_available_models(extraction_mod=None):
    if extraction_mod is None:
        extraction_mod = import_first(EXTRACTION_MODULES)
    if hasattr(extraction_mod, "MODEL_REGISTRY"):
        return [(m.name, m.family, m.parameter_billions) for m in extraction_mod.MODEL_REGISTRY]
    return []

def has_artifact(root, experiment_id, model, dataset):
    ds_dir = dataset_artifact_dir(root, experiment_id, model, dataset)
    return ds_dir.exists() and (ds_dir / "metadata" / "extraction.json").exists()

def compact_path(path, max_chars=74):
    s = str(path)
    return s if len(s) <= max_chars else "…" + s[-(max_chars - 1):]

# ----------------------------- Config dataclasses -----------------------------
@dataclass
class ProbeChoice:
    preset: str
    overrides: dict = field(default_factory=dict)

@dataclass
class MasterConfig:
    root: Path = DEFAULT_ROOT
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    model: str | None = None
    dataset: str | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
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
    probes: list[ProbeChoice] = field(default_factory=lambda: [ProbeChoice("logistic")])

    def as_dict(self):
        d = asdict(self)
        d["root"] = str(self.root)
        d["output_dir"] = str(self.output_dir)
        return d

    @property
    def config_hash(self):
        return stable_hash(self.as_dict(), 20)

# ----------------------------- Module factory -----------------------------
class ModuleFactory:
    @staticmethod
    def extraction(config):
        return import_first(EXTRACTION_MODULES)

    @staticmethod
    def probe(config):
        return import_first(PROBE_MODULES)

class DatasetFactory:
    @staticmethod
    def spec(name):
        if name not in DATASET_SPECS:
            raise KeyError(f"Unsupported dataset {name!r}. Available: {sorted(DATASET_SPECS)}")
        return dict(DATASET_SPECS[name])

    @staticmethod
    def load(name):
        spec = DatasetFactory.spec(name)
        module = importlib.import_module(spec["module"])
        fn = getattr(module, spec["function"])
        return fn()

class ProbeFactory:
    @staticmethod
    def create(probe_module, choices):
        ProbeSpec = getattr(probe_module, "ProbeSpec")
        probes = []
        for choice in choices:
            if choice.preset not in PROBE_PRESETS:
                raise ValueError(f"Unknown probe preset {choice.preset!r}. Available: {sorted(PROBE_PRESETS)}")
            payload = dict(PROBE_PRESETS[choice.preset])
            payload.update(choice.overrides)
            probes.append(ProbeSpec(**payload))
        return probes

# ----------------------------- Renderer -----------------------------
class Renderer:
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None

    def title(self, title, subtitle=""):
        if self.console:
            body = Text(subtitle) if subtitle else ""
            self.console.print(Panel(body, title=title, expand=False))
        else:
            print("\n" + "=" * 88)
            print(title)
            if subtitle: print(subtitle)
            print("=" * 88)

    def info(self, text):
        if self.console: self.console.print(text)
        else: print(text)

    def warning(self, text):
        if self.console: self.console.print(f"[yellow]WARNING[/yellow] {text}")
        else: print(f"WARNING: {text}")

    def success(self, text):
        if self.console: self.console.print(f"[green]✓[/green] {text}")
        else: print(f"✓ {text}")

    def error(self, text):
        if self.console: self.console.print(f"[red]✗[/red] {text}")
        else: print(f"ERROR: {text}")

    def stage(self, current, stages):
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

    def config(self, cfg):
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
            for a, b in rows: table.add_row(a, b)
            self.console.print(table)
        else:
            for a, b in rows: print(f"{a:18}: {b}")

    def result_table(self, df, task_type):
        if df.empty:
            self.warning("No result rows available.")
            return
        preferred = (["probe", "layer_index", "test_macro_f1", "test_micro_f1", "test_average_precision_macro", "control_macro_f1", "selectivity", "probe_score"]
                     if task_type == "multi_label" else
                     ["probe", "layer_index", "test_macro_f1", "test_balanced_accuracy", "test_mcc", "probe_score"])
        cols = [c for c in preferred if c in df.columns]
        shown = df[cols].copy()
        numeric = shown.select_dtypes(include="number").columns
        shown[numeric] = shown[numeric].round(4)
        if self.console:
            table = Table(title="Probe result snapshot")
            for col in shown.columns: table.add_column(str(col))
            for _, row in shown.head(30).iterrows():
                table.add_row(*[str(x) for x in row.tolist()])
            self.console.print(table)
        else:
            print(shown.head(30).to_string(index=False))

# ----------------------------- Main pipeline (read-only) -----------------------------
class EmotionProbePipeline:
    def __init__(self, *, model, dataset, root=DEFAULT_ROOT, experiment_id=DEFAULT_EXPERIMENT_ID,
                 output_dir=DEFAULT_OUTPUT_DIR, probes=("logistic",), repeats=3, max_samples=5000,
                 seed=42, shuffled_label_control=True, shuffled_control_repeats=3,
                 strict_provenance=True):
        self.renderer = Renderer()
        normalized = [p if isinstance(p, ProbeChoice) else ProbeChoice(str(p)) for p in probes]
        self.config = MasterConfig(
            root=Path(root).resolve(),
            experiment_id=experiment_id,
            model=normalize_model_name(model),
            dataset=str(dataset),
            output_dir=Path(output_dir).resolve(),
            probes=normalized,
            repeats=int(repeats),
            max_samples=max_samples,
            seed=int(seed),
            shuffled_label_control=bool(shuffled_label_control),
            shuffled_control_repeats=int(shuffled_control_repeats),
            strict_provenance=bool(strict_provenance)
        )
        self._validate()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact = None
        self.results = pd.DataFrame()

    def _validate(self):
        DatasetFactory.spec(self.config.dataset)
        if self.config.repeats < 1: raise ValueError("repeats must be >= 1")
        if self.config.max_samples is not None and self.config.max_samples < 1:
            raise ValueError("max_samples must be >= 1 or None")
        if not np.isclose(self.config.split_train + self.config.split_validation + self.config.split_test, 1.0):
            raise ValueError("Split fractions must sum to 1.")
        if self.config.shuffled_control_repeats < 1: raise ValueError("shuffled_control_repeats must be >= 1")

    @property
    def artifact_dir(self):
        return dataset_artifact_dir(self.config.root, self.config.experiment_id, self.config.model, self.config.dataset)

    def _artifact_exists(self):
        return (self.artifact_dir / "data" / "hidden_states.npy").exists() and \
               (self.artifact_dir / "data" / "completed.npy").exists() and \
               (self.artifact_dir / "metadata" / "extraction.json").exists()

    def _load_artifact(self):
        self.probe_mod = ModuleFactory.probe(self.config)
        ExtractionArtifact = getattr(self.probe_mod, "ExtractionArtifact")
        self.artifact = ExtractionArtifact(self.artifact_dir, verify_checksum=self.config.verify_checksum)

    def preflight(self):
        if not self._artifact_exists():
            raise FileNotFoundError(
                f"Hidden-state artifact not found at {self.artifact_dir}\n"
                "This demo pipeline does NOT run extraction. Please run extraction separately."
            )
        self._load_artifact()
        if not np.all(self.artifact.completed):
            raise RuntimeError("Extraction completion map is incomplete.")
        if self.artifact.states.shape[0] != self.artifact.sample_count:
            raise RuntimeError("Sample count mismatch.")

        ds_spec = DatasetFactory.spec(self.config.dataset)
        dataset_df = DatasetFactory.load(self.config.dataset)
        probe_mod = ModuleFactory.probe(self.config)
        DatasetContract = getattr(probe_mod, "DatasetContract")
        contract = DatasetContract(
            target_type=ds_spec["target_type"], type="python",
            module=ds_spec["module"], function=ds_spec["function"],
            task_type=ds_spec["task_type"], require_provenance=self.config.strict_provenance
        )
        build_targets = getattr(probe_mod, "build_targets")
        y, classes, _ = build_targets(dataset_df, contract)
        if len(y) != self.artifact.sample_count:
            raise RuntimeError(f"Target count {len(y)} != hidden-state count {self.artifact.sample_count}.")
        self.renderer.success("Forensic preflight passed (read‑only).")

    def run_probes(self):
        probe_mod = ModuleFactory.probe(self.config)
        ProbeSpec = getattr(probe_mod, "ProbeSpec")
        AnalysisConfig = getattr(probe_mod, "AnalysisConfig")
        SplitConfig = getattr(probe_mod, "SplitConfig")
        UnifiedProbeAnalyzer = getattr(probe_mod, "UnifiedProbeAnalyzer")
        DatasetContract = getattr(probe_mod, "DatasetContract")

        probes = ProbeFactory.create(probe_mod, self.config.probes)
        ds_spec = DatasetFactory.spec(self.config.dataset)
        contract = DatasetContract(
            target_type=ds_spec["target_type"], type="python",
            module=ds_spec["module"], function=ds_spec["function"],
            task_type=ds_spec["task_type"], require_provenance=self.config.strict_provenance
        )

        split = SplitConfig(
            train=self.config.split_train, validation=self.config.split_validation,
            test=self.config.split_test, seed=self.config.seed, stratify=True
        )

        analysis = AnalysisConfig(
            dataset=contract, probes=probes, layers=self.config.layers,
            split=split, repeats=self.config.repeats, max_samples=self.config.max_samples,
            shuffled_label_control=self.config.shuffled_label_control,
            shuffled_control_repeats=self.config.shuffled_control_repeats,
            run_control_on_all_layers=True,
            pca_enabled=self.config.pca_enabled,
            silhouette_enabled=self.config.silhouette_enabled,
            pca_samples=self.config.pca_samples,
            silhouette_samples=self.config.silhouette_samples,
            enable_abstention=True, enable_per_class_metrics=True,
            enable_feature_statistics=True, verbose=1
        )

        out_dir = self.config.output_dir / f"{self.config.dataset}__{safe_model_path(self.config.model)}__{self.config.config_hash[:10]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_df = DatasetFactory.load(self.config.dataset)

        analyzer = UnifiedProbeAnalyzer(self.artifact, analysis, output_dir=out_dir, dataset_df=dataset_df)
        try:
            scored, best = analyzer.run()
        except RuntimeError as e:
            self.renderer.warning(f"Probe run incomplete: {e}")
            partial = analyzer.output_dir / "layer_probe_results.csv"
            if partial.exists():
                scored = pd.read_csv(partial)
                best = scored.sort_values(["probe", "probe_score"], ascending=[True, False]).groupby("probe", as_index=False).first()
            else:
                raise

        if scored is None or scored.empty:
            raise RuntimeError("No scored results.")
        scored = scored.copy()
        scored["master_version"] = MASTER_VERSION
        scored["master_config_hash"] = self.config.config_hash
        self.results = scored

        main_csv = self.config.output_dir / f"results__{self.config.dataset}__{safe_model_path(self.config.model).as_posix().replace('/', '_')}.csv"
        scored.to_csv(main_csv, index=False)
        best.to_csv(self.config.output_dir / "best__" + main_csv.name, index=False)
        self.renderer.success(f"Probe results saved to {compact_path(main_csv)}")
        return scored

    def run(self):
        self.renderer.title("EMOTION PROBE LAB", "Demonstration mode – no modifications to existing experiments")
        self.renderer.config(self.config)
        self.preflight()
        self.run_probes()
        summary = {
            "rows": len(self.results),
            "task_type": DatasetFactory.spec(self.config.dataset)["task_type"],
            "best_macro_f1": self.results["test_macro_f1"].max() if "test_macro_f1" in self.results else None
        }
        save_json(self.config.output_dir / "analysis_summary.json", summary)
        self.renderer.title("RUN COMPLETE", f"Results in {compact_path(self.config.output_dir)}")
        return {"config": self.config.as_dict(), "results": self.results}

# ----------------------------- Analysis -----------------------------
class ResultAnalyser:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_csvs(self, csv_paths):
        frames = []
        for p in csv_paths:
            if not Path(p).exists():
                print(f"Warning: {p} not found, skipping.")
                continue
            frames.append(pd.read_csv(p))
        if not frames: raise ValueError("No valid CSV files provided.")
        return pd.concat(frames, ignore_index=True)

    def generate_plots(self, df):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="darkgrid")
        plt.rcParams["figure.facecolor"] = "#1e1e1e"
        plt.rcParams["axes.facecolor"] = "#2d2d2d"
        plt.rcParams["axes.edgecolor"] = "#d4d4d4"
        plt.rcParams["axes.labelcolor"] = "#d4d4d4"
        plt.rcParams["text.color"] = "#d4d4d4"
        plt.rcParams["xtick.color"] = "#d4d4d4"
        plt.rcParams["ytick.color"] = "#d4d4d4"
        plt.rcParams["grid.color"] = "#444444"
        plt.rcParams["legend.facecolor"] = "#2d2d2d"
        plt.rcParams["legend.edgecolor"] = "#d4d4d4"

        if "model" in df.columns and "dataset" in df.columns:
            for (model, dataset), group in df.groupby(["model", "dataset"]):
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                for ax, metric, title in [(axes[0], "test_macro_f1", "Test Macro-F1"),
                                          (axes[1], "probe_score", "Unified Probe Score")]:
                    if metric not in group.columns: continue
                    for probe_name in group["probe"].unique():
                        sub = group[group["probe"] == probe_name].sort_values("layer_index")
                        ax.plot(sub["layer_index"], sub[metric], marker="o", label=probe_name)
                    ax.set_xlabel("Layer index")
                    ax.set_ylabel(title)
                    ax.set_title(f"{title} – {model} / {dataset}")
                    ax.grid(alpha=0.3)
                    ax.legend()
                fig.tight_layout()
                fig.savefig(self.output_dir / f"layer_curves_{model.replace('/', '_')}_{dataset}.png", dpi=240)
                plt.close(fig)

        if "test_macro_f1" in df.columns:
            pivot = df.pivot_table(index="probe", columns="layer_index", values="test_macro_f1", aggfunc="mean")
            if not pivot.empty:
                plt.figure(figsize=(12, 6))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Macro-F1"})
                plt.title("Mean Test Macro-F1")
                plt.xlabel("Layer")
                plt.ylabel("Probe")
                plt.tight_layout()
                plt.savefig(self.output_dir / "heatmap_macro_f1.png", dpi=240)
                plt.close(fig)

        if "test_macro_f1" in df.columns:
            best = df.loc[df.groupby("probe")["test_macro_f1"].idxmax()]
            plt.figure(figsize=(11, 6))
            sns.barplot(data=best, x="probe", y="test_macro_f1", palette="viridis")
            plt.xticks(rotation=20, ha="right")
            plt.title("Best Layer per Probe (Macro-F1)")
            plt.tight_layout()
            plt.savefig(self.output_dir / "best_per_probe.png", dpi=240)
            plt.close(fig)

        if "control_macro_f1" in df.columns:
            plt.figure(figsize=(12, 6))
            for probe_name in df["probe"].unique():
                sub = df[df["probe"] == probe_name].groupby("layer_index")[["test_macro_f1", "control_macro_f1"]].mean()
                plt.plot(sub.index, sub["test_macro_f1"], marker="o", label=f"{probe_name} (true)")
                plt.plot(sub.index, sub["control_macro_f1"], marker="x", linestyle="--", label=f"{probe_name} (shuffled)")
            plt.xlabel("Layer index")
            plt.ylabel("Macro-F1")
            plt.title("True vs Shuffled Controls")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "control_comparison.png", dpi=240)
            plt.close(fig)

        if "model" in df.columns and "dataset" in df.columns:
            best = df.loc[df.groupby(["model", "dataset", "probe"])["test_macro_f1"].idxmax()]
            pivot = best.pivot_table(index=["model", "dataset"], columns="probe", values="test_macro_f1", aggfunc="first")
            if not pivot.empty:
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": "Best Macro-F1"})
                plt.title("Best Macro-F1 per Model/Dataset/Probe")
                plt.tight_layout()
                plt.savefig(self.output_dir / "best_heatmap.png", dpi=240)
                plt.close(fig)

        print(f"Plots saved to {self.output_dir}")

# ----------------------------- Interactive -----------------------------
class InteractiveApp:
    def __init__(self): self.renderer = Renderer()

    def ask(self, prompt, default=None):
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        return value if value else (default or "")

    def choose(self, title, options):
        print("\n" + title)
        for i, opt in enumerate(options, 1): print(f"  {i}. {opt}")
        while True:
            raw = input("Select: ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options): return options[idx]
            except ValueError: pass
            print("Please select a valid number.")

    def run(self):
        self.renderer.title("EMOTION PROBE LAB", "Controlled interface (demo mode)")

        dataset = self.choose("Dataset", ["goEmo", "ISEAR"])

        # Get all registry models and mark which have artifacts
        extraction_mod = ModuleFactory.extraction(MasterConfig())
        all_models = get_available_models(extraction_mod)
        if not all_models:
            model = self.ask("Model", "google-bert/bert-base-uncased")
        else:
            print("\nAvailable models (✓ = has artifact):")
            for i, (name, family, params) in enumerate(all_models, 1):
                has = has_artifact(DEFAULT_ROOT, DEFAULT_EXPERIMENT_ID, name, dataset)
                mark = "✓" if has else "✗"
                print(f"  {i:2d}. {mark} {name}  ({family}, {params:.3f}B)")
            print("\nType a model name or number, or press ENTER for default.")
            raw = input("Model [default: google-bert/bert-base-uncased]: ").strip()
            if raw == "":
                model = "google-bert/bert-base-uncased"
            elif raw.isdigit():
                idx = int(raw) - 1
                model = all_models[idx][0] if 0 <= idx < len(all_models) else "google-bert/bert-base-uncased"
            else:
                # Resolve partial match
                matches = [m[0] for m in all_models if raw.lower() in m[0].lower()]
                if len(matches) == 1:
                    model = matches[0]
                elif len(matches) > 1:
                    print("Multiple matches found. Please choose one:")
                    for i, m in enumerate(matches, 1):
                        print(f"  {i}. {m}")
                    choice = input("Select number: ").strip()
                    try:
                        model = matches[int(choice)-1]
                    except: model = matches[0]
                else:
                    print(f"⚠️ Model '{raw}' not found in registry. Using it as-is.")
                    model = raw

        # Probe selection
        print("\nProbe selection")
        print("  ENTER = logistic baseline")
        print("  1     = logistic")
        print("  2     = logistic + 1-hidden MLP")
        print("  3     = logistic + 1/2-hidden MLP")
        print("  4     = logistic + 1/2/3-hidden MLP")
        choice = input("Probe configuration: ").strip()
        probe_map = {"": ["logistic"], "1": ["logistic"], "2": ["logistic", "mlp1"],
                     "3": ["logistic", "mlp1", "mlp2"], "4": ["logistic", "mlp1", "mlp2", "mlp3"]}
        probes = probe_map.get(choice, ["logistic"])

        max_raw = self.ask("Maximum samples (ENTER = 5000, type FULL for all)", "5000")
        max_samples = None if max_raw.upper() == "FULL" else int(max_raw)
        repeats = int(self.ask("Independent repeats", "3"))
        controls = self.ask("Run shuffled-label control? Y/N", "Y").upper() == "Y"
        output_dir = self.ask("Output directory (ENTER = ./demo_runs)", "./demo_runs") or "./demo_runs"

        pipeline = EmotionProbePipeline(model=model, dataset=dataset, output_dir=output_dir,
                                        probes=probes, repeats=repeats, max_samples=max_samples,
                                        shuffled_label_control=controls)
        pipeline.renderer.config(pipeline.config)
        confirmation = self.ask("Start this experiment? Y/N", "Y").upper()
        if confirmation != "Y":
            self.renderer.info("Experiment cancelled.")
            return {"status": "cancelled"}
        return pipeline.run()

# ----------------------------- CLI -----------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Emotion Probe Lab – demonstration pipeline (read-only)")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Run a probing experiment and save CSV results.")
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    run.add_argument("--root", default=str(DEFAULT_ROOT))
    run.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run.add_argument("--probe", action="append", default=["logistic"], choices=["logistic", "mlp1", "mlp2", "mlp3"])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--max-samples", type=int, default=5000)
    run.add_argument("--full", action="store_true")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--no-shuffle-control", action="store_true")
    analyse = sub.add_parser("analyse", help="Generate visualizations from result CSV(s).")
    analyse.add_argument("--input", "-i", action="append", required=True)
    analyse.add_argument("--output-dir", "-o", default="./analysis_plots")
    sub.add_parser("interactive", help="Launch guided interface.")
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "interactive":
        InteractiveApp().run()
        return 0
    if args.command == "run":
        probes = list(dict.fromkeys(args.probe))
        pipeline = EmotionProbePipeline(model=args.model, dataset=args.dataset, root=args.root,
                                        experiment_id=args.experiment_id, output_dir=args.output_dir,
                                        probes=probes, repeats=args.repeats,
                                        max_samples=None if args.full else args.max_samples,
                                        seed=args.seed, shuffled_label_control=not args.no_shuffle_control)
        pipeline.run()
        return 0
    if args.command == "analyse":
        csv_paths = [Path(p) for p in args.input]
        analyser = ResultAnalyser(args.output_dir)
        df = analyser.load_csvs(csv_paths)
        analyser.generate_plots(df)
        return 0
    parser.print_help()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())