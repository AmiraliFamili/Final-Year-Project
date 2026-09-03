"""
probing_pipeline.py
===================

Master orchestration layer for representation probing.

The pipeline turns:

    dataset source + Hugging Face model name

into:

    cleaned dataset
    -> frozen hidden-state artifact
    -> v4.2 probe analysis
    -> controls / metrics
    -> plots
    -> manifest

The architecture deliberately treats probing as one analysis method in a
pluggable interpretability registry. New methods can be registered later
without changing dataset ingestion or the experiment lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer

from dataset_loader import DatasetLoader


LOGGER = logging.getLogger(__name__)

MODEL_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

PROBE_PRESETS: dict[str, dict[str, Any]] = {
    "linear": {
        "name": "linear_logistic",
        "type": "logistic",
        "complexity": "linear",
        "standardize": True,
        "C": 1.0,
        "max_iter": 3000,
        "selection_metric": "macro_f1",
    },
    "mlp_1": {
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
    "mlp_2": {
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
    "mlp_3": {
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


@dataclass
class ExperimentContext:
    model_name: str
    dataset_name: str
    output_dir: Path
    artifact_dir: Path
    clean_dataset_path: Path
    dataset_report: dict[str, Any]
    label_mapping: dict[str, Any]
    results: pd.DataFrame | None = None
    best: pd.DataFrame | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "output_dir": str(self.output_dir),
            "artifact_dir": str(self.artifact_dir),
            "clean_dataset_path": str(self.clean_dataset_path),
            "dataset_report": self.dataset_report,
            "label_mapping": self.label_mapping,
        }


@dataclass
class RegisteredMethod:
    name: str
    runner: Callable[[ExperimentContext], Any]
    enabled: bool = True


class ProbingPipeline:
    """
    End-to-end, model-agnostic representation probing.

    The first-class interface is intentionally tiny:

        ProbingPipeline(
            model_name="bert-base-uncased",
            dataset_paths="dataset.csv",
        ).run()

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.
    dataset_paths:
        Local/remote dataset path(s).
    text_column / label_column:
        Optional overrides; otherwise DatasetLoader infers them.
    probe_complexity:
        linear, mlp_1, mlp_2, mlp_3.
    visualization_type:
        One type or a sequence: layerwise, heatmap, shuffle_advantage.
    output_dir:
        Experiment artifact root.
    batch_size, max_samples, seed:
        Core runtime controls.
    run_integrity_check:
        Ask the legacy analyser to validate a reused/new artifact when a
        recognisable analyser entry point is available.
    reuse_artifacts:
        Reuse an existing hidden-state artifact only when its metadata fingerprint
        matches the cleaned dataset.
    """

    def __init__(
        self,
        model_name: str,
        dataset_paths: str | Path | Sequence[str | Path],
        *,
        text_column: str | None = None,
        label_column: str | None = None,
        probe_complexity: str = "linear",
        visualization_type: str | Sequence[str] = "layerwise",
        output_dir: str | Path = "probe_runs",
        batch_size: int = 16,
        max_samples: int | None = 5000,
        seed: int = 42,
        max_length: int = 512,
        pooling: str = "mean",
        device: str | None = None,
        repeats: int = 3,
        run_integrity_check: bool = False,
        reuse_artifacts: bool = True,
        clean_kwargs: Mapping[str, Any] | None = None,
        probe_kwargs: Mapping[str, Any] | None = None,
        extraction_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if probe_complexity not in PROBE_PRESETS:
            raise ValueError(
                f"probe_complexity must be one of {sorted(PROBE_PRESETS)}"
            )
        if pooling not in {"mean", "first_token", "last_token"}:
            raise ValueError(
                "pooling must be one of {'mean', 'first_token', 'last_token'}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_samples is not None and max_samples < 30:
            raise ValueError("max_samples must be >= 30 or None")
        if repeats < 1:
            raise ValueError("repeats must be >= 1")

        self.model_name = model_name
        self.dataset_paths = dataset_paths
        self.text_column = text_column
        self.label_column = label_column
        self.probe_complexity = probe_complexity
        self.visualization_type = (
            [visualization_type]
            if isinstance(visualization_type, str)
            else list(visualization_type)
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.max_length = int(max_length)
        self.pooling = pooling
        self.device = device or self._choose_device()
        self.repeats = int(repeats)
        self.run_integrity_check = bool(run_integrity_check)
        self.reuse_artifacts = bool(reuse_artifacts)
        self.clean_kwargs = dict(clean_kwargs or {})
        self.probe_kwargs = dict(probe_kwargs or {})
        self.extraction_kwargs = dict(extraction_kwargs or {})
        self.extra_kwargs = dict(kwargs)

        unknown_plots = set(self.visualization_type) - {
            "layerwise",
            "heatmap",
            "shuffle_advantage",
        }
        if unknown_plots:
            raise ValueError(f"Unknown visualization type(s): {sorted(unknown_plots)}")

        self.methods: dict[str, RegisteredMethod] = {}
        self.register_method("probing", self._run_v42_probe)

        self.context: ExperimentContext | None = None

    # ------------------------------------------------------------------
    # Extension API
    # ------------------------------------------------------------------

    def register_method(
        self,
        name: str,
        runner: Callable[[ExperimentContext], Any],
        *,
        enabled: bool = True,
    ) -> None:
        """
        Register an interpretability/explainability method.

        The runner receives the fully prepared ExperimentContext, so future
        methods can operate on the same frozen dataset, model artifact, and
        result directory without duplicating orchestration.
        """
        if not name or not callable(runner):
            raise ValueError("A non-empty method name and callable runner are required.")
        self.methods[name] = RegisteredMethod(name, runner, enabled)

    # ------------------------------------------------------------------
    # Main lifecycle
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        started = time.perf_counter()
        self._seed_everything(self.seed)

        LOGGER.info("=== Probing pipeline started ===")
        LOGGER.info("Model: %s", self.model_name)

        dataset_dir = self.output_dir / "dataset"
        loader = DatasetLoader(
            self.dataset_paths,
            output_dir=dataset_dir,
            text_column=self.text_column,
            label_column=self.label_column,
            random_state=self.seed,
            **self.clean_kwargs,
        )
        clean_df = loader.load()

        dataset_name = self._dataset_name(loader.sources)
        artifact_dir = (
            self.output_dir
            / "artifacts"
            / self._safe_name(self.model_name)
            / self._safe_name(dataset_name)
        )

        context = ExperimentContext(
            model_name=self.model_name,
            dataset_name=dataset_name,
            output_dir=self.output_dir,
            artifact_dir=artifact_dir,
            clean_dataset_path=dataset_dir / "cleaned_dataset.csv",
            dataset_report=asdict(loader.report) if loader.report else {},
            label_mapping=loader.label_mapping or {},
        )
        self.context = context

        self._write_initial_manifest(context)

        self._ensure_hidden_states(clean_df, context)

        if self.run_integrity_check:
            self._optional_integrity_check(context.artifact_dir)

        for method in self.methods.values():
            if not method.enabled:
                continue
            if method.name == "probing":
                LOGGER.info("Running legacy-compatible v4.2 probe engine")
            else:
                LOGGER.info("Running registered interpretability method: %s", method.name)
            method.runner(context)

        if context.results is None:
            raise RuntimeError("No probing results were produced.")

        self._generate_requested_plots(context.results)
        self._write_final_manifest(context, time.perf_counter() - started)

        LOGGER.info(
            "=== Probing pipeline complete in %.2fs ===",
            time.perf_counter() - started,
        )
        return context.results

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    def _ensure_hidden_states(
        self,
        clean_df: pd.DataFrame,
        context: ExperimentContext,
    ) -> None:
        context.artifact_dir.mkdir(parents=True, exist_ok=True)
        expected_fp = self._dataset_fingerprint(clean_df["text"].tolist())

        metadata_path = context.artifact_dir / "metadata" / "extraction.json"
        states_path = context.artifact_dir / "data" / "hidden_states.npy"
        complete_path = context.artifact_dir / "data" / "completed.npy"

        if self.reuse_artifacts and metadata_path.exists() and states_path.exists() and complete_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                stored_fp = metadata.get("dataset", {}).get("fingerprint")
                completed = np.load(complete_path, mmap_mode="r")
                if stored_fp == expected_fp and bool(np.all(completed)):
                    LOGGER.info("Reusing compatible hidden-state artifact: %s", context.artifact_dir)
                    return
                LOGGER.warning("Existing artifact is incompatible; rebuilding it.")
            except Exception as exc:
                LOGGER.warning("Could not validate existing artifact (%s); rebuilding.", exc)

        self._extract_hidden_states(clean_df, context, expected_fp)

    def _extract_hidden_states(
        self,
        clean_df: pd.DataFrame,
        context: ExperimentContext,
        dataset_fingerprint: str,
    ) -> None:
        data_dir = context.artifact_dir / "data"
        metadata_dir = context.artifact_dir / "metadata"
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        sample_df = clean_df
        if self.max_samples is not None and len(sample_df) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            indices = np.sort(
                rng.choice(len(sample_df), size=self.max_samples, replace=False)
            )
            sample_df = sample_df.iloc[indices].reset_index(drop=True)
            # The clean dataset remains the canonical full dataset. Extraction
            # is explicitly sampled, and the sampled rows are persisted.
            sampled_path = context.output_dir / "dataset" / "extraction_population.csv"
            sample_df.to_csv(sampled_path, index=False)
            clean_df = sample_df

        tokenizer = None
        model = None

        try:
            LOGGER.info("Loading tokenizer: %s", self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if tokenizer.pad_token is None:
                # Common for decoder-only LMs.
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|probe_pad|>"})

            LOGGER.info("Loading model: %s", self.model_name)
            model = AutoModel.from_pretrained(
                self.model_name,
                output_hidden_states=True,
            )
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

            if tokenizer.vocab_size != getattr(model.config, "vocab_size", tokenizer.vocab_size):
                # Only resize if the tokenizer was genuinely extended.
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception:
                    LOGGER.debug("Tokenizer/model vocab sizes differ; resize skipped.", exc_info=True)

            model.eval()
            model.to(self.device)

            n = len(clean_df)
            hidden_layers = None
            hidden_size = None
            states_mm = None

            completed = np.zeros(n, dtype=np.bool_)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                texts = clean_df["text"].iloc[start:end].tolist()

                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.inference_mode():
                    outputs = model(**encoded)

                hidden = outputs.hidden_states
                if hidden is None:
                    raise RuntimeError(
                        "The selected Hugging Face model did not return hidden states."
                    )

                # hidden is a tuple of [B, T, D], one tensor per embedding +
                # transformer layer.
                if hidden_layers is None:
                    hidden_layers = len(hidden)
                    hidden_size = int(hidden[-1].shape[-1])
                    states_mm = np.lib.format.open_memmap(
                        data_dir / "hidden_states.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(n, hidden_layers, hidden_size),
                    )

                batch_array = np.stack(
                    [
                        self._pool_hidden_state(h, encoded["attention_mask"])
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        for h in hidden
                    ],
                    axis=1,
                )

                if not np.isfinite(batch_array).all():
                    raise FloatingPointError(
                        f"Non-finite hidden states encountered for rows {start}:{end}"
                    )

                states_mm[start:end] = batch_array
                completed[start:end] = True
                states_mm.flush()

                LOGGER.info(
                    "Extracted rows %d:%d / %d",
                    start,
                    end,
                    n,
                )

            if states_mm is None or hidden_layers is None or hidden_size is None:
                raise RuntimeError("Model extraction produced no hidden-state tensor.")

            np.save(complete_path, completed)

            # Recompute on the actual extraction population, which may have
            # been sampled from the cleaned dataset.
            active_texts = clean_df["text"].tolist()
            active_labels = clean_df["label"].tolist()
            active_fingerprint = self._dataset_fingerprint(active_texts)
            label_fingerprint = self._label_fingerprint(
                active_labels,
                context.label_mapping.get("classes", []),
            )

            np.save(
                metadata_dir / "sample_ids.npy",
                np.arange(len(clean_df), dtype=np.int64),
            )

            metadata = {
                "status": "complete",
                "experiment_id": f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}",
                "model": {
                    "name": self.model_name,
                },
                "dataset": {
                    "name": context.dataset_name,
                    "samples": len(clean_df),
                    "fingerprint": active_fingerprint,
                    "hidden_state_shape": [len(clean_df), hidden_layers, hidden_size],
                    "provenance": {
                        "derived_fingerprint": active_fingerprint,
                        "head_hash": self._sequence_hash(active_texts[:100]),
                        "tail_hash": self._sequence_hash(active_texts[-100:]),
                        "label_fingerprint": label_fingerprint,
                        "target_fingerprint": label_fingerprint,
                    },
                },
                "extraction": {
                    "pooling": self.pooling,
                    "max_length": self.max_length,
                    "batch_size": self.batch_size,
                    "device": self.device,
                },
            }

            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, ensure_ascii=False)

        except Exception as exc:
            raise RuntimeError(
                f"Hidden-state extraction failed for model {self.model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            del model
            del tokenizer
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _pool_hidden_state(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "first_token":
            return hidden[:, 0, :]

        mask = attention_mask.bool()
        if self.pooling == "last_token":
            lengths = mask.sum(dim=1).clamp_min(1) - 1
            row = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[row, lengths, :]

        weights = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    # ------------------------------------------------------------------
    # v4.2 integration
    # ------------------------------------------------------------------

    def _run_v42_probe(self, context: ExperimentContext) -> None:
        try:
            probe_module = importlib.import_module("unified_hidden_state_probe_v4_2")
        except ImportError as exc:
            raise ImportError(
                "unified_hidden_state_probe_v4_2.py must be importable from the project "
                "environment to run the established probe analyser."
            ) from exc

        try:
            ExtractionArtifact = probe_module.ExtractionArtifact
            DatasetContract = probe_module.DatasetContract
            ProbeSpec = probe_module.ProbeSpec
            SplitConfig = probe_module.SplitConfig
            AnalysisConfig = probe_module.AnalysisConfig
            UnifiedProbeAnalyzer = probe_module.UnifiedProbeAnalyzer
        except AttributeError as exc:
            raise ImportError(
                "unified_hidden_state_probe_v4_2 is missing one of the required v4.2 "
                "public classes."
            ) from exc

        label_mapping = context.label_mapping
        task_type = label_mapping.get("task_type", "single_label")

        contract = DatasetContract(
            target_type="custom",
            type="file",
            path=str(context.clean_dataset_path),
            text_column="text",
            label_column="label",
            id_column="auto",
            task_type=task_type,
            label_format="auto",
            # DatasetLoader stores the semantic class names in
            # label_mapping.json, while the probe dataframe stores stable
            # integer class IDs. v4.2 therefore receives the corresponding
            # stringified IDs as its canonical class order.
            class_order=[
                str(i)
                for i, _ in enumerate(label_mapping.get("classes", []))
            ],
            single_label_policy=None,
            require_provenance=False,
            require_label_fingerprint=False,
        )

        analysis_overrides = dict(self.probe_kwargs.pop("analysis_overrides", {}))
        base = dict(PROBE_PRESETS[self.probe_complexity])
        base.update(self.probe_kwargs)
        probe_spec = ProbeSpec(**base)

        split = SplitConfig(
            train=0.80,
            validation=0.10,
            test=0.10,
            seed=self.seed,
            stratify=True,
        )

        cfg = AnalysisConfig(
            dataset=contract,
            probes=[probe_spec],
            layers="all",
            split=split,
            repeats=self.repeats,
            max_samples=self.max_samples,
            shuffled_label_control=True,
            shuffled_control_repeats=3,
            run_control_on_all_layers=True,
            pca_enabled=True,
            silhouette_enabled=True,
            pca_samples=min(3000, self.max_samples or 3000),
            silhouette_samples=min(3000, self.max_samples or 3000),
            enable_abstention=True,
            enable_per_class_metrics=True,
            enable_feature_statistics=True,
            verbose=1,
            **analysis_overrides,
        )

        artifact = ExtractionArtifact(context.artifact_dir)
        probe_output = context.output_dir / "probe_results"
        probe_output.mkdir(parents=True, exist_ok=True)

        analyzer = UnifiedProbeAnalyzer(
            artifact,
            cfg,
            probe_output,
            dataset_df=pd.read_csv(context.clean_dataset_path),
        )
        scored, best = analyzer.run()

        scored = scored.copy()
        scored["model"] = self.model_name
        scored["dataset"] = context.dataset_name
        scored.to_csv(probe_output / "layer_probe_results_pipeline.csv", index=False)

        if not best.empty:
            best.to_csv(
                probe_output / "best_probe_layers_pipeline.csv",
                index=False,
            )

        context.results = scored
        context.best = best

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _generate_requested_plots(self, results: pd.DataFrame) -> None:
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for plot_type in self.visualization_type:
            try:
                if plot_type == "layerwise":
                    self._plot_layerwise(results, plots_dir)
                elif plot_type == "heatmap":
                    self._plot_heatmap(results, plots_dir)
                elif plot_type == "shuffle_advantage":
                    self._plot_shuffle_advantage(results, plots_dir)
            except Exception:
                LOGGER.exception("Plotting failed for %s", plot_type)

    def _plot_layerwise(self, results: pd.DataFrame, plots_dir: Path) -> None:
        metric = self._first_existing(
            results.columns,
            ["test_macro_f1", "test_macro_f1_mean", "probe_score"],
        )
        if metric is None or "layer_index" not in results.columns:
            LOGGER.warning("Layerwise plot skipped: no usable metric/layer columns.")
            return

        fig, ax = plt.subplots(figsize=(11, 6))
        group_cols = ["probe"] if "probe" in results.columns else [None]

        if group_cols[0] is None:
            grouped = results.groupby("layer_index")[metric].mean()
            ax.plot(grouped.index, grouped.values, marker="o")
        else:
            for probe_name, group in results.groupby("probe"):
                grouped = group.groupby("layer_index")[metric].mean()
                ax.plot(
                    grouped.index,
                    grouped.values,
                    marker="o",
                    label=str(probe_name),
                )
            ax.legend()

        ax.set(
            title=f"Layer-wise {metric}",
            xlabel="Layer index",
            ylabel=metric,
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "layerwise.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    def _plot_heatmap(self, results: pd.DataFrame, plots_dir: Path) -> None:
        metric = self._first_existing(
            results.columns,
            ["test_macro_f1", "test_macro_f1_mean", "probe_score_mean"],
        )
        if metric is None or "layer_index" not in results.columns:
            LOGGER.warning("Heatmap skipped: no usable metric/layer columns.")
            return

        index = "probe" if "probe" in results.columns else "model"
        if index not in results.columns:
            tmp = results.copy()
            tmp["_probe"] = "probe"
            index = "_probe"
        else:
            tmp = results

        pivot = tmp.pivot_table(
            index=index,
            columns="layer_index",
            values=metric,
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(13, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            linewidths=0.4,
            cbar_kws={"label": metric},
            ax=ax,
        )
        ax.set_title(f"Layer × probe {metric}")
        ax.set_xlabel("Layer index")
        ax.set_ylabel(index)
        fig.tight_layout()
        fig.savefig(plots_dir / "heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    def _plot_shuffle_advantage(self, results: pd.DataFrame, plots_dir: Path) -> None:
        true_metric = self._first_existing(
            results.columns,
            ["test_macro_f1", "test_macro_f1_mean"],
        )
        shuffle_metric = self._first_existing(
            results.columns,
            [
                "control_test_macro_f1",
                "shuffled_test_macro_f1",
                "shuffle_test_macro_f1",
            ],
        )

        if true_metric is None or shuffle_metric is None:
            # v4.2 normally persists shuffled controls separately. Load them.
            candidates = list(
                (self.output_dir / "probe_results").glob("**/shuffled_label_controls.csv")
            )
            if candidates:
                control = pd.concat(
                    [pd.read_csv(path) for path in candidates],
                    ignore_index=True,
                )
                if "control_test_macro_f1" in control.columns:
                    true = (
                        results.groupby(["probe", "layer_index"])[true_metric]
                        .mean()
                        .reset_index()
                    )
                    shuffled = (
                        control.groupby(["probe", "layer_index"])[
                            "control_test_macro_f1"
                        ]
                        .mean()
                        .reset_index()
                        .rename(
                            columns={"control_test_macro_f1": "shuffle_macro_f1"}
                        )
                    )
                    merged = true.merge(
                        shuffled,
                        on=["probe", "layer_index"],
                        how="inner",
                    )
                else:
                    LOGGER.warning("Shuffle-control CSV does not contain macro-F1.")
                    return
            else:
                LOGGER.warning(
                    "Shuffle advantage plot skipped: no true/shuffled metrics found. "
                    "No fake control values are generated."
                )
                return
        else:
            merged = results[
                ["probe", "layer_index", true_metric, shuffle_metric]
            ].copy()
            merged = merged.rename(
                columns={shuffle_metric: "shuffle_macro_f1"}
            )
            merged = merged.rename(columns={true_metric: "true_macro_f1"})

        merged["shuffle_advantage"] = (
            merged["true_macro_f1"] - merged["shuffle_macro_f1"]
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        for probe_name, group in merged.groupby("probe"):
            curve = group.groupby("layer_index")["shuffle_advantage"].mean()
            ax.plot(curve.index, curve.values, marker="o", label=str(probe_name))

        ax.axhline(0.0, linewidth=1)
        ax.set(
            title="Shuffle advantage: true Macro-F1 − shuffled-label Macro-F1",
            xlabel="Layer index",
            ylabel="Macro-F1 advantage",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plots_dir / "shuffle_advantage.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Integrity / manifests
    # ------------------------------------------------------------------

    def _optional_integrity_check(self, artifact_dir: Path) -> None:
        try:
            module = importlib.import_module("_Analyser_")
        except ImportError:
            LOGGER.warning(
                "run_integrity_check=True but _Analyser_.py could not be imported; skipping."
            )
            return

        candidates = [
            "run_integrity_check",
            "analyze_extraction",
            "analyse_extraction",
            "validate_extraction",
        ]
        for name in candidates:
            fn = getattr(module, name, None)
            if not callable(fn):
                continue
            try:
                LOGGER.info("Calling legacy integrity checker: _Analyser_.%s", name)
                fn(artifact_dir)
                return
            except TypeError:
                try:
                    fn(str(artifact_dir))
                    return
                except Exception:
                    LOGGER.exception("Legacy integrity checker %s failed.", name)
                    return
            except Exception:
                LOGGER.exception("Legacy integrity checker %s failed.", name)
                return

        LOGGER.warning(
            "_Analyser_ imported, but no recognised callable integrity entry point was found."
        )

    def _write_initial_manifest(self, context: ExperimentContext) -> None:
        payload = {
            "created_at": time.time(),
            "pipeline": "ProbingPipeline",
            "model_name": self.model_name,
            "probe_complexity": self.probe_complexity,
            "visualization_type": self.visualization_type,
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "device": self.device,
            "context": context.as_dict(),
        }
        with (self.output_dir / "pipeline_manifest.json").open(
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

    def _write_final_manifest(
        self,
        context: ExperimentContext,
        elapsed: float,
    ) -> None:
        payload = {
            "completed_at": time.time(),
            "elapsed_seconds": elapsed,
            "context": context.as_dict(),
            "result_rows": int(len(context.results)) if context.results is not None else 0,
            "best_probe": (
                context.best.to_dict(orient="records")
                if context.best is not None
                else []
            ),
        }
        with (self.output_dir / "completion.json").open(
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _choose_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _safe_name(value: str) -> str:
        return MODEL_SAFE_RE.sub("_", value).strip("_") or "unnamed"

    @staticmethod
    def _dataset_name(sources: Sequence[str]) -> str:
        if len(sources) == 1:
            candidate = Path(urlparse(sources[0]).path).stem
            if candidate:
                return candidate
            return ProbingPipeline._safe_name(sources[0])
        return f"combined_{len(sources)}"

    @staticmethod
    def _dataset_fingerprint(values: Sequence[str]) -> str:
        payload = {
            "n": len(values),
            "head": list(values[:16]),
            "tail": list(values[-16:]) if values else [],
        }
        return ProbingPipeline._stable_hash(payload, 20)

    @staticmethod
    def _sequence_hash(values: Sequence[Any]) -> str:
        return ProbingPipeline._stable_hash(list(values), 20)

    @staticmethod
    def _label_fingerprint(
        labels: Sequence[Any],
        classes: Sequence[Any],
    ) -> str:
        return ProbingPipeline._stable_hash(
            {
                "classes": list(classes),
                "labels": list(labels),
            },
            24,
        )

    @staticmethod
    def _stable_hash(value: Any, length: int) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:length]

    @staticmethod
    def _first_existing(
        columns: Sequence[str],
        candidates: Sequence[str],
    ) -> str | None:
        columns_set = set(columns)
        return next((name for name in candidates if name in columns_set), None)


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a complete hidden-state probing experiment."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, nargs="+")
    parser.add_argument("--output-dir", default="probe_runs")
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument(
        "--probe",
        choices=sorted(PROBE_PRESETS),
        default="linear",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=["layerwise", "heatmap", "shuffle_advantage"],
        default=["layerwise", "heatmap", "shuffle_advantage"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["mean", "first_token", "last_token"], default="mean")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--integrity-check", action="store_true")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pipeline = ProbingPipeline(
        model_name=args.model,
        dataset_paths=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        probe_complexity=args.probe,
        visualization_type=args.plot,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_length=args.max_length,
        pooling=args.pooling,
        repeats=args.repeats,
        seed=args.seed,
        run_integrity_check=args.integrity_check,
        reuse_artifacts=not args.no_reuse,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
