from __future__ import annotations

"""Hidden State Verification & Analysis Module (Advanced)

This module provides comprehensive checks, anomaly detection, and visualisations
for extracted hidden states. It can process a single model/dataset or all models
at once, returning a summary DataFrame for further analysis.

Key features:
- Recursively scans all model/dataset directories under an experiment root.
- Performs shape, completion, integrity, linkage, anomaly checks.
- Aggregates results into a pandas DataFrame.
- Generates dark‑themed plots.
- Gracefully handles missing or non‑numeric label files.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
except ImportError:
    HAS_PLOT = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ----------------------------- Helper Functions -----------------------------

def load_array(path: Path, mmap_mode: str = 'r', optional: bool = False) -> Optional[np.ndarray]:
    if not path.exists():
        if optional:
            return None
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except (ValueError, TypeError):
        # Fallback for object dtype arrays (e.g., labels as strings)
        return np.load(path, allow_pickle=True)

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def scan_directory(dataset_dir: Path) -> Dict[str, Any]:
    """Scan dataset directory, filter AppleDouble, return present/missing/unexpected."""
    expected_files = {
        "hidden_states.npy": dataset_dir / "data" / "hidden_states.npy",
        "completed.npy": dataset_dir / "data" / "completed.npy",
        "labels.npy": dataset_dir / "data" / "labels.npy",
        "sample_ids.npy": dataset_dir / "metadata" / "sample_ids.npy",
        "extraction.json": dataset_dir / "metadata" / "extraction.json",
        "integrity_hashes.jsonl": dataset_dir / "metadata" / "integrity_hashes.jsonl",
        "label_codes.npy": dataset_dir / "metadata" / "label_codes.npy",
        "sample_manifest.jsonl": dataset_dir / "metadata" / "sample_manifest.jsonl",
    }
    labels_path = expected_files["labels.npy"]
    if labels_path.exists():
        try:
            labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
            if labels.dtype.kind in ('i', 'u'):
                expected_files.pop("label_codes.npy", None)
        except Exception:
            pass

    present = {name: str(path) for name, path in expected_files.items() if path.exists()}
    missing = {name: str(path) for name, path in expected_files.items() if not path.exists()}
    unexpected = []
    for p in dataset_dir.rglob("*"):
        if p.is_file():
            if p.name.startswith("._") or "__MACOSX" in p.parts:
                continue
            rel = p.relative_to(dataset_dir)
            if not any(rel == ep.relative_to(dataset_dir) for ep in expected_files.values()):
                unexpected.append(str(rel))
    return {"present": present, "missing": missing, "unexpected": unexpected}

def load_verification_context(model_name, dataset_name, base_output):
    base_path = Path(base_output).resolve()
    model_dir = base_path / "models" / Path(*model_name.split("/"))
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

    states = load_array(paths["states"], mmap_mode='r')
    completed = load_array(paths["completed"], mmap_mode='r')
    sample_ids = load_array(paths["sample_ids"], mmap_mode='r', optional=True)
    labels = load_array(paths["labels"], mmap_mode='r', optional=True)
    label_codes = load_array(paths["label_codes"], mmap_mode='r', optional=True)
    metadata = load_json(paths["metadata"])

    integrity_hashes = []
    if paths["integrity"].exists():
        with open(paths["integrity"], 'r') as f:
            for line in f:
                integrity_hashes.append(json.loads(line))

    sample_manifest = None
    if paths["sample_manifest"].exists():
        sample_manifest = []
        with open(paths["sample_manifest"], 'r') as f:
            for line in f:
                sample_manifest.append(json.loads(line))

    missing_files = [str(p) for p in paths.values() if not p.exists()]
    file_inventory = scan_directory(dataset_dir)

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "states": states,
        "completed": completed,
        "labels": labels,
        "sample_ids": sample_ids,
        "label_codes": label_codes,
        "metadata": metadata,
        "integrity_hashes": integrity_hashes,
        "sample_manifest": sample_manifest,
        "paths": paths,
        "missing_files": missing_files,
        "file_inventory": file_inventory,
    }

# ----------------------------- Check Functions -----------------------------

def verify_shapes(ctx):
    errors = []
    states = ctx["states"]
    completed = ctx["completed"]
    sample_ids = ctx["sample_ids"]
    labels = ctx["labels"]
    n_samples = states.shape[0]
    if completed.shape != (n_samples,):
        errors.append(f"completed shape {completed.shape} != states samples {n_samples}")
    if sample_ids is not None and sample_ids.shape != (n_samples,):
        errors.append(f"sample_ids shape {sample_ids.shape} != states samples {n_samples}")
    if labels is not None and labels.shape[0] != n_samples:
        errors.append(f"labels shape {labels.shape} != states samples {n_samples}")
    if states.ndim != 3:
        errors.append(f"states must be 3D, got {states.ndim}D")
    return errors

def verify_completion(ctx):
    completed = ctx["completed"]
    total = len(completed)
    done = int(np.sum(completed))
    missing_indices = np.where(~completed)[0]
    return {
        "total_samples": total,
        "completed_samples": done,
        "missing_samples": total - done,
        "missing_indices": missing_indices.tolist(),
        "is_complete": done == total,
    }

def verify_integrity_hashes(ctx):
    states = ctx["states"]
    completed = ctx["completed"]
    hashes = ctx["integrity_hashes"]
    if not hashes:
        return {"status": "not_available", "message": "No integrity hashes found.", "checked_batches": 0}
    results = []
    for h in hashes:
        start = h["batch_start"]
        end = h["batch_end"]
        if np.all(completed[start:end]):
            data = np.asarray(states[start:end])
            computed = hashlib.sha256(data.tobytes()).hexdigest()
            if computed == h["hash"]:
                results.append({"batch": f"{start}-{end}", "status": "ok"})
            else:
                results.append({"batch": f"{start}-{end}", "status": "mismatch"})
        else:
            results.append({"batch": f"{start}-{end}", "status": "incomplete"})
    mismatches = [r for r in results if r["status"] == "mismatch"]
    return {
        "status": "ok" if not mismatches else "corrupted",
        "checked_batches": len(results),
        "mismatched_batches": mismatches,
        "all_results": results,
    }

def verify_nan_inf(ctx):
    states = ctx["states"]
    nan_count = 0
    inf_count = 0
    for start in range(0, states.shape[0], 1000):
        end = min(start + 1000, states.shape[0])
        chunk = np.asarray(states[start:end])
        nan_count += int(np.isnan(chunk).sum())
        inf_count += int(np.isinf(chunk).sum())
    return {"nan_count": nan_count, "inf_count": inf_count, "has_invalid": (nan_count + inf_count) > 0}

def verify_id_linkage(ctx):
    sample_ids = ctx["sample_ids"]
    labels = ctx["labels"]
    completed = ctx["completed"]
    errors = []
    if sample_ids is None:
        errors.append("sample_ids file is missing")
    else:
        n = len(sample_ids)
        if not np.array_equal(sample_ids, np.arange(n)):
            errors.append("Sample IDs are not sequential 0..N-1")
    if labels is None:
        errors.append("labels file is missing")
    else:
        # For object dtype (strings/lists), just check length
        if labels.dtype.kind == 'O':
            if len(labels) != len(completed):
                errors.append("labels length mismatch")
        else:
            if labels.dtype.kind in ('f', 'c'):
                missing = int(np.isnan(labels).sum())
            else:
                missing = int((labels == -1).sum())
            if missing > 0:
                errors.append(f"{missing} samples have missing labels")
    status = "ok" if not errors else ("warning" if any("missing" in e for e in errors) else "error")
    return {"status": status, "errors": errors, "sample_count": len(sample_ids) if sample_ids is not None else 0}

def verify_sample_manifest(ctx, n_check=100):
    manifest = ctx.get("sample_manifest")
    if not manifest:
        return {"status": "not_available", "message": "Sample manifest not found."}
    states = ctx["states"]
    n = states.shape[0]
    if len(manifest) != n:
        return {"status": "error", "message": f"Manifest count {len(manifest)} != states {n}"}
    # For a quick check, we verify only that the structure is correct.
    # A full cryptographic verification would require per‑sample hashes.
    return {"status": "ok", "checked": len(manifest)}

# ----------------------------- Stats & Anomaly -----------------------------

def compute_layer_statistics(ctx):
    states = ctx["states"]
    n_samples, n_layers, hidden_dim = states.shape
    stats = {"mean": np.zeros((n_layers, hidden_dim)),
             "std": np.zeros((n_layers, hidden_dim)),
             "min": np.full((n_layers, hidden_dim), np.inf),
             "max": np.full((n_layers, hidden_dim), -np.inf)}
    for start in range(0, n_samples, 500):
        end = min(start + 500, n_samples)
        chunk = np.asarray(states[start:end])
        stats["mean"] += chunk.sum(axis=0)
        stats["max"] = np.maximum(stats["max"], chunk.max(axis=0))
        stats["min"] = np.minimum(stats["min"], chunk.min(axis=0))
    stats["mean"] /= n_samples
    var = np.zeros((n_layers, hidden_dim))
    for start in range(0, n_samples, 500):
        end = min(start + 500, n_samples)
        chunk = np.asarray(states[start:end])
        var += ((chunk - stats["mean"]) ** 2).sum(axis=0)
    stats["std"] = np.sqrt(var / n_samples)
    return stats

def detect_layer_anomalies(stats, threshold_std=1e-6):
    anomalies = []
    layer_std = stats["std"].mean(axis=1)
    for i, s in enumerate(layer_std):
        if s < threshold_std:
            anomalies.append(f"Layer {i} has near-zero std ({s:.2e})")
    return anomalies

def detect_dead_neurons(ctx, threshold_std=1e-6, max_check=500):
    states = ctx["states"]
    n_samples, n_layers, hidden_dim = states.shape
    dead_neurons = {layer: [] for layer in range(n_layers)}
    subset_size = min(n_samples, max_check)
    indices = np.random.choice(n_samples, subset_size, replace=False)
    subset = np.asarray(states[indices])
    for layer in range(n_layers):
        layer_std = subset[:, layer, :].std(axis=0)
        dead_neurons[layer] = np.where(layer_std < threshold_std)[0].tolist()
    total = sum(len(v) for v in dead_neurons.values())
    return {"total_dead_neurons": total, "per_layer": dead_neurons, "threshold_std": threshold_std}

def detect_outliers(ctx, z_threshold=5.0, max_check=1000):
    states = ctx["states"]
    n_samples = states.shape[0]
    subset_size = min(n_samples, max_check)
    indices = np.random.choice(n_samples, subset_size, replace=False)
    subset = np.asarray(states[indices])
    sample_mean = subset.mean(axis=(1,2))
    sample_std = subset.std(axis=(1,2))
    global_mean = sample_mean.mean()
    global_std = sample_std.mean()
    z = (sample_mean - global_mean) / (global_std + 1e-12)
    outliers = indices[np.abs(z) > z_threshold].tolist()
    return {"outlier_count": len(outliers), "outlier_indices": outliers,
            "z_threshold": z_threshold, "checked_samples": subset_size}

# ----------------------------- Report Generation (single) -----------------------------

def generate_report(ctx):
    report = {}
    report["model"] = ctx["model_name"]
    report["dataset"] = ctx["dataset_name"]
    report["missing_files"] = ctx["missing_files"]
    report["file_inventory"] = ctx["file_inventory"]

    report["shape_errors"] = verify_shapes(ctx)
    report["completion"] = verify_completion(ctx)
    report["integrity"] = verify_integrity_hashes(ctx) if ctx["integrity_hashes"] else {"status": "not_available", "message": "Integrity hashes file missing.", "checked_batches": 0}
    report["sample_manifest"] = verify_sample_manifest(ctx)
    report["nan_inf"] = verify_nan_inf(ctx)
    report["id_linkage"] = verify_id_linkage(ctx) if (ctx["sample_ids"] is not None and ctx["labels"] is not None) else {"status": "warning", "errors": ["sample_ids or labels missing; linkage verification partially skipped."], "sample_count": 0}

    stats = compute_layer_statistics(ctx)
    report["layer_anomalies"] = detect_layer_anomalies(stats)
    report["stats_summary"] = {
        "global_mean": float(np.mean(stats["mean"])),
        "global_std": float(np.mean(stats["std"])),
        "min_value": float(np.min(stats["min"])),
        "max_value": float(np.max(stats["max"])),
    }
    report["dead_neurons"] = detect_dead_neurons(ctx)
    report["outliers"] = detect_outliers(ctx)

    ok = (
        not report["shape_errors"]
        and not report["nan_inf"]["has_invalid"]
        and report["id_linkage"]["status"] == "ok"
        and not report["layer_anomalies"]
        and (report["integrity"]["status"] in ("ok", "not_available"))
        and not report["missing_files"]
        and report["dead_neurons"]["total_dead_neurons"] == 0
        and report["outliers"]["outlier_count"] == 0
    )
    report["is_probe_ready"] = ok
    return report

# ----------------------------- Aggregation for All Models -----------------------------

def analyze_all_models(base_output: str | Path) -> "pd.DataFrame":
    """
    Scans the experiment directory (which contains a 'models' folder),
    runs all checks for every model/dataset, and returns a pandas DataFrame
    with one row per model/dataset combination.
    """
    import pandas as pd
    base_path = Path(base_output).resolve()          # experiment root, e.g. .../baseline_v5_001
    models_dir = base_path / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    records = []
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = str(model_dir.relative_to(models_dir)).replace(os.sep, "/")
        datasets_dir = model_dir / "datasets"
        if not datasets_dir.exists():
            continue
        for dataset_dir in sorted(datasets_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            try:
                ctx = load_verification_context(model_name, dataset_name, base_path)
                rep = generate_report(ctx)
                record = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "probe_ready": rep["is_probe_ready"],
                    "completion_pct": rep["completion"]["completed_samples"] / rep["completion"]["total_samples"] * 100 if rep["completion"]["total_samples"] > 0 else 0,
                    "missing_files": len(rep["missing_files"]),
                    "shape_errors": len(rep["shape_errors"]),
                    "nan_inf": rep["nan_inf"]["has_invalid"],
                    "integrity_status": rep["integrity"]["status"],
                    "sample_manifest_status": rep["sample_manifest"]["status"],
                    "id_linkage_status": rep["id_linkage"]["status"],
                    "layer_anomalies": len(rep["layer_anomalies"]),
                    "dead_neurons": rep["dead_neurons"]["total_dead_neurons"],
                    "outliers": rep["outliers"]["outlier_count"],
                }
                records.append(record)
            except Exception as e:
                records.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "probe_ready": False,
                    "completion_pct": 0,
                    "missing_files": -1,
                    "shape_errors": -1,
                    "nan_inf": False,
                    "integrity_status": "error",
                    "sample_manifest_status": "error",
                    "id_linkage_status": "error",
                    "layer_anomalies": -1,
                    "dead_neurons": -1,
                    "outliers": -1,
                    "error": str(e),
                })
    return pd.DataFrame(records)

# ----------------------------- Visualisation Functions -----------------------------

def plot_completion(ctx, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(12, 2))
    completed = ctx["completed"]
    total = len(completed)
    im = ax.imshow(completed.reshape(1, -1), aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xlabel("Sample Index")
    ax.set_title(f"Completion Map (green=done, red=missing) — {int(completed.sum())}/{total}", color='#4fc3f7')
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2, label='Completed')
    return ax

def plot_layer_means(stats, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(10, 4))
    mean_per_layer = stats["mean"].mean(axis=1)
    ax.plot(mean_per_layer, marker='o', color='#4fc3f7', linewidth=2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Mean Activation per Layer", color='#4fc3f7')
    ax.grid(True, alpha=0.3)
    return ax

def plot_layer_std(stats, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(10, 4))
    std_per_layer = stats["std"].mean(axis=1)
    ax.plot(std_per_layer, marker='s', color='#f48fb1', linewidth=2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Std Dev")
    ax.set_title("Std Dev per Layer", color='#4fc3f7')
    ax.grid(True, alpha=0.3)
    return ax

def plot_hidden_distribution(ctx, layer_idx=-1, n_samples=500, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
    states = ctx["states"]
    indices = np.random.choice(states.shape[0], min(n_samples, states.shape[0]), replace=False)
    data = np.asarray(states[indices, layer_idx, :]).flatten()
    sns.histplot(data, bins=100, kde=True, color='#4fc3f7', ax=ax)
    ax.set_title(f"Activation Distribution (Layer {layer_idx})", color='#4fc3f7')
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Frequency")
    return ax

def plot_layer_boxplot(ctx, n_samples=1000, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(12, 6))
    states = ctx["states"]
    indices = np.random.choice(states.shape[0], min(n_samples, states.shape[0]), replace=False)
    data = np.asarray(states[indices])  # (samples, layers, hidden_dim)
    n_layers = data.shape[1]
    # Prepare data for boxplot: list of arrays, one per layer
    data_per_layer = [data[:, i, :].flatten() for i in range(n_layers)]
    sns.boxplot(data=data_per_layer, palette="Set2", ax=ax)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Activation Value")
    ax.set_title("Activation Distribution per Layer (boxplot)", color='#4fc3f7')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(i) for i in range(n_layers)], rotation=45)
    return ax

def plot_layer_correlation(stats, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(10, 8))
    mean_matrix = stats["mean"]  # (n_layers, hidden_dim)
    corr = np.corrcoef(mean_matrix)
    sns.heatmap(corr, cmap='viridis', ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title("Layer Correlation Matrix (based on mean activations)", color='#4fc3f7')
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Layer Index")
    return ax

def plot_anomaly_heatmap(ctx, stats, z_threshold=3.0, max_samples=500, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(12, 6))
    states = ctx["states"]
    n_samples, n_layers, hidden_dim = states.shape
    sample_size = min(n_samples, max_samples)
    indices = np.random.choice(n_samples, sample_size, replace=False)
    subset = np.asarray(states[indices])
    sample_means = subset.mean(axis=2)
    global_mean = sample_means.mean(axis=0)
    global_std = sample_means.std(axis=0) + 1e-12
    z = (sample_means - global_mean) / global_std
    sns.heatmap(z, cmap='RdBu_r', center=0, vmin=-z_threshold, vmax=z_threshold,
                ax=ax, cbar_kws={'label': 'Z-score'})
    ax.set_title(f"Per-Sample Z-scores (layers vs samples)", color='#4fc3f7')
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Sample Index")
    return ax

def plot_tsne(ctx, layer_idx=-1, n_samples=1000, perplexity=30, use_umap=False, ax=None):
    if not HAS_PLOT or not HAS_SKLEARN: return None
    if ax is None: fig, ax = plt.subplots(figsize=(10, 8))
    states = ctx["states"]
    labels = ctx["labels"]
    indices = np.random.choice(states.shape[0], min(n_samples, states.shape[0]), replace=False)
    data = np.asarray(states[indices, layer_idx, :])
    if data.shape[1] > 50:
        pca = PCA(n_components=50)
        data = pca.fit_transform(data)
    if use_umap and HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        proj = reducer.fit_transform(data)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        proj = tsne.fit_transform(data)
    if labels is not None:
        label_subset = labels[indices]
        if label_subset.ndim > 1:
            label_flat = label_subset.argmax(axis=1)
        else:
            label_flat = label_subset
        scatter = ax.scatter(proj[:,0], proj[:,1], c=label_flat, cmap='tab20', s=5, alpha=0.7)
        ax.legend(*scatter.legend_elements(), title="Labels", bbox_to_anchor=(1.05,1), loc='upper left')
    else:
        ax.scatter(proj[:,0], proj[:,1], s=5, alpha=0.5, c='#4fc3f7')
    ax.set_title(f"{'UMAP' if use_umap else 't-SNE'} Projection (Layer {layer_idx})", color='#4fc3f7')
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return ax

def plot_file_tree(ctx, ax=None):
    if not HAS_PLOT: return None
    if ax is None: fig, ax = plt.subplots(figsize=(8, 6))
    inv = ctx["file_inventory"]
    labels = list(inv["present"].keys()) + ["MISSING"]*len(inv["missing"])
    sizes = [1]*len(inv["present"]) + [0.5]*len(inv["missing"])
    colors = ['#2ecc71']*len(inv["present"]) + ['#e74c3c']*len(inv["missing"])
    ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90, wedgeprops=dict(width=0.4))
    ax.set_title("File Inventory (green=present, red=missing)", color='#4fc3f7')
    return ax

# ----------------------------- Master Function for Single Analysis -----------------------------

def analyze_all_models(
    base_output: str | Path,
    show_verbose: bool = False,
    show_info: bool = True,
    show_critical: bool = True,
) -> "pd.DataFrame":
    """
    Recursively scans all model/dataset directories and runs all checks.
    Provides live logging at different verbosity levels.

    Parameters:
        show_verbose: if True, print detailed per‑check progress.
        show_info: if True, print summary per model/dataset.
        show_critical: if True, print only failures.

    Returns:
        pd.DataFrame with one row per model/dataset, including status_message.
    """
    import pandas as pd
    base_path = Path(base_output).resolve()
    models_dir = base_path / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    records = []
    total_datasets = 0
    # First count total dataset dirs for progress
    dataset_dirs = list(models_dir.rglob("datasets"))
    total_datasets = sum(len(list(d.iterdir())) for d in dataset_dirs if d.is_dir())

    current = 0
    for dataset_parent in sorted(dataset_dirs):
        model_dir = dataset_parent.parent
        model_name = str(model_dir.relative_to(models_dir)).replace(os.sep, "/")
        # Load model metadata if available
        model_info = {}
        model_meta_path = model_dir / "model_metadata.json"
        if model_meta_path.exists():
            with open(model_meta_path, 'r') as f:
                meta = json.load(f)
                spec = meta.get("model", {})
                model_info = {
                    "family": spec.get("family", "unknown"),
                    "architecture": spec.get("architecture", "unknown"),
                }

        for ds_dir in sorted(dataset_parent.iterdir()):
            if not ds_dir.is_dir():
                continue
            current += 1
            dataset_name = ds_dir.name
            if show_info:
                print(f"[{current}/{total_datasets}] Processing {model_name} / {dataset_name}...")

            ctx = None
            try:
                ctx = load_verification_context(model_name, dataset_name, base_path)
                if show_verbose:
                    print(f"  Loaded context for {model_name}/{dataset_name}")
                    print(f"    Hidden states shape: {ctx['states'].shape}")
                    print(f"    Completed samples: {int(ctx['completed'].sum())}/{len(ctx['completed'])}")
            except FileNotFoundError as e:
                if show_critical:
                    print(f"  ✗ SKIP {model_name}/{dataset_name}: {e}")
                records.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "family": model_info.get("family", ""),
                    "architecture": model_info.get("architecture", ""),
                    "probe_ready": False,
                    "completion_pct": 0,
                    "missing_files": -1,
                    "shape_errors": -1,
                    "nan_inf": False,
                    "integrity_status": "error",
                    "sample_manifest_status": "error",
                    "id_linkage_status": "error",
                    "layer_anomalies": -1,
                    "dead_neurons": -1,
                    "outliers": -1,
                    "status_message": f"Missing mandatory file: {e}",
                })
                continue

            try:
                rep = generate_report(ctx)
                # Build status message
                if rep["is_probe_ready"]:
                    status_msg = "All checks passed"
                else:
                    reasons = []
                    if rep["missing_files"]:
                        reasons.append(f"{len(rep['missing_files'])} missing files")
                    if rep["shape_errors"]:
                        reasons.append(f"{len(rep['shape_errors'])} shape errors")
                    if rep["nan_inf"]["has_invalid"]:
                        reasons.append("contains NaN/Inf")
                    if rep["integrity"]["status"] not in ("ok", "not_available"):
                        reasons.append(f"integrity {rep['integrity']['status']}")
                    if rep["id_linkage"]["status"] != "ok":
                        reasons.append("linkage issues")
                    if rep["layer_anomalies"]:
                        reasons.append(f"{len(rep['layer_anomalies'])} layer anomalies")
                    if rep["dead_neurons"]["total_dead_neurons"] > 0:
                        reasons.append(f"{rep['dead_neurons']['total_dead_neurons']} dead neurons")
                    if rep["outliers"]["outlier_count"] > 0:
                        reasons.append(f"{rep['outliers']['outlier_count']} outliers")
                    status_msg = "; ".join(reasons) if reasons else "Unknown issue"

                if show_info:
                    print(f"  ✓ {model_name}/{dataset_name} -> {status_msg}")

                records.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "family": model_info.get("family", ""),
                    "architecture": model_info.get("architecture", ""),
                    "probe_ready": rep["is_probe_ready"],
                    "completion_pct": rep["completion"]["completed_samples"] / rep["completion"]["total_samples"] * 100 if rep["completion"]["total_samples"] > 0 else 0,
                    "missing_files": len(rep["missing_files"]),
                    "shape_errors": len(rep["shape_errors"]),
                    "nan_inf": rep["nan_inf"]["has_invalid"],
                    "integrity_status": rep["integrity"]["status"],
                    "sample_manifest_status": rep["sample_manifest"]["status"],
                    "id_linkage_status": rep["id_linkage"]["status"],
                    "layer_anomalies": len(rep["layer_anomalies"]),
                    "dead_neurons": rep["dead_neurons"]["total_dead_neurons"],
                    "outliers": rep["outliers"]["outlier_count"],
                    "status_message": status_msg,
                })
            except Exception as e:
                if show_critical:
                    print(f"  ✗ ERROR {model_name}/{dataset_name}: {type(e).__name__}: {e}")
                records.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "family": model_info.get("family", ""),
                    "architecture": model_info.get("architecture", ""),
                    "probe_ready": False,
                    "completion_pct": 0,
                    "missing_files": -1,
                    "shape_errors": -1,
                    "nan_inf": False,
                    "integrity_status": "error",
                    "sample_manifest_status": "error",
                    "id_linkage_status": "error",
                    "layer_anomalies": -1,
                    "dead_neurons": -1,
                    "outliers": -1,
                    "status_message": f"Error: {type(e).__name__}: {e}",
                })
        # end dataset loop

    if show_info:
        print(f"\nCompleted analysis of {current} datasets.")
    return pd.DataFrame(records)


# ------------ summary and report def plot_summary(df: "pd.DataFrame"):
def plot_summary(df: "pd.DataFrame"):
    """Generate a dashboard from the summary DataFrame produced by analyze_all_models."""
    if df.empty:
        print("No data to visualise.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Completion heatmap
    pivot = df.pivot_table(index='model', columns='dataset', values='completion_pct', aggfunc='mean')
    plt.figure(figsize=(14, max(6, len(pivot)*0.4)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap='viridis', cbar_kws={'label': 'Completion %'})
    plt.title('Completion Percentage by Model and Dataset', color='#4fc3f7')
    plt.tight_layout()
    plt.show()

    # 2. Probe readiness count (stacked bar)
    plt.figure(figsize=(12, 6))
    pd.crosstab(df['model'], df['probe_ready']).plot(kind='bar', stacked=True, color=['#e74c3c','#2ecc71'], ax=plt.gca())
    plt.title('Probe Readiness by Model', color='#4fc3f7')
    plt.xlabel('Model')
    plt.ylabel('Number of Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Probe Ready', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()

    # 3. Missing files bar
    missing = df.groupby('model')['missing_files'].sum().sort_values()
    plt.figure(figsize=(10, max(6, len(missing)*0.4)))
    sns.barplot(x=missing.values, y=missing.index, color='#f48fb1')
    plt.title('Total Missing Files per Model', color='#4fc3f7')
    plt.xlabel('Missing Files Count')
    plt.tight_layout()
    plt.show()

    # 4. Integrity status distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='integrity_status', hue='integrity_status', palette='viridis', legend=False)
    plt.title('Integrity Hash Status', color='#4fc3f7')
    plt.tight_layout()
    plt.show()

    # 5. Anomaly scatter: dead neurons vs outliers (bubble size = completion)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['dead_neurons'], df['outliers'],
                          c=df['completion_pct'], cmap='coolwarm', s=80)
    plt.colorbar(scatter, label='Completion %')
    plt.xlabel('Dead Neurons')
    plt.ylabel('Outliers')
    plt.title('Anomaly Overview', color='#4fc3f7')
    plt.tight_layout()
    plt.show()