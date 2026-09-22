# wired.py
"""
wired.py — orchestrator for every downstream interpretability technique.

Layout
------
    hidden_states/<slug>/<dataset>/             Extraction.py output (reads)
    interEx/<slug>/<dataset>/analysis/          extraction-level techniques (writes)
    interEx/<slug>/<dataset>/<run_key>/analysis/ probe-level techniques (writes)

Every technique is wrapped in _safe() so one failure cannot kill the run.
Every technique writes to a fixed path, so re-running is idempotent.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Global seeding
# ─────────────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Failure-isolated call wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _safe(label: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"[{label}] skipped — missing file: {e}")
    except ValueError as e:
        print(f"[{label}] skipped — invalid input: {e}")
    except Exception as e:
        print(f"[{label}] FAILED — {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _iter_extraction_pairs(root: Path):
    """Yield (model_slug, dataset, dataset_dir) for every extraction that
    has hidden_states.npy AND labels.npy on disk."""
    if not root.is_dir():
        return
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if not (ds_dir / "hidden_states.npy").is_file():
                continue
            if not (ds_dir / "labels.npy").is_file():
                print(f"[skip] {model_dir.name}/{ds_dir.name}: labels.npy missing")
                continue
            yield model_dir.name, ds_dir.name, ds_dir


def discover_probe_runs(root: Path | None = None) -> pd.DataFrame:
    """Walk interEx/<model>/<dataset>/index.json and return one row per run."""
    from _shared import INTEREX_ROOT
    root = Path(root or INTEREX_ROOT)
    rows = []
    if not root.is_dir():
        return pd.DataFrame(columns=["model_slug", "dataset", "run_key_dir"])
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            idx = ds_dir / "index.json"
            if not idx.is_file():
                continue
            try:
                payload = json.loads(idx.read_text())
            except Exception:
                continue
            for run in payload.get("runs", []):
                rows.append({
                    "model_slug":  model_dir.name,
                    "dataset":     ds_dir.name,
                    "run_key_dir": ds_dir / run["run_key"],
                    "probes":      run.get("probes", []),
                    "task_type":   run.get("task_type"),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction-level techniques
# ─────────────────────────────────────────────────────────────────────────────

def _run_extraction_level_techniques() -> None:
    from _shared import HIDDEN_STATES_ROOT
    import geometry, geometry_multilabel, cka

    for model_slug_, dataset, _ in _iter_extraction_pairs(HIDDEN_STATES_ROOT):
        print(f"[wired] extraction-level: {model_slug_}/{dataset}")

        # Geometry — single-label
        _safe("geom", geometry.run_geometry, model_slug_, dataset)

        # Geometry — multi-label (only for GoEmotions)
        if dataset == "goemo":
            _safe(
                "geom-ml",
                geometry_multilabel.run_geometry_multilabel,
                model_slug_, dataset,
            )

        # Within-model CKA
        _safe("cka-within", cka.run_cka_within, model_slug_, dataset)


def _run_cross_model_cka() -> None:
    from _shared import HIDDEN_STATES_ROOT
    import cka

    slugs: list[str] = []
    for model_dir in sorted(p for p in HIDDEN_STATES_ROOT.iterdir() if p.is_dir()):
        # Only include models that produced at least one extraction.
        for ds_dir in model_dir.iterdir():
            if (ds_dir / "hidden_states.npy").is_file():
                slugs.append(model_dir.name)
                break

    for i, a in enumerate(slugs):
        for b in slugs[i + 1:]:
            _safe("cka-cross", cka.run_cka_cross, a, b, "isear")


def _run_attention() -> None:
    """Attention analysis requires a fresh forward pass; it uses the model
    snapshot under MODELS_ROOT, so it only runs when the snapshot exists.
    """
    from _shared import HIDDEN_STATES_ROOT, MODELS_ROOT, DATASETS_ROOT
    import attention

    for model_slug_, dataset, _ in _iter_extraction_pairs(HIDDEN_STATES_ROOT):
        if not (MODELS_ROOT / model_slug_).is_dir():
            print(f"[attention] skip {model_slug_}/{dataset}: model snapshot missing")
            continue
        csv = DATASETS_ROOT / dataset / "processed" / f"{dataset}_clean.csv"
        if not csv.is_file():
            print(f"[attention] skip {model_slug_}/{dataset}: processed CSV missing")
            continue
        try:
            texts = pd.read_csv(csv)["clean_text"].astype(str).tolist()
        except Exception as e:
            print(f"[attention] skip {model_slug_}/{dataset}: {e}")
            continue

        _safe(
            "attention",
            attention.run_attention,
            model_slug_, dataset, texts,
            max_length=64, sample_size=300,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Probe-level techniques
# ─────────────────────────────────────────────────────────────────────────────

def _run_probe_level_techniques() -> None:
    from _shared import DATASETS_ROOT, load_labels
    import tcav, protopics

    runs = discover_probe_runs()
    if runs.empty:
        print("[wired] no probe runs found under interEx/")
        return

    for row in runs.itertuples(index=False):
        print(f"[wired] probe-level: {row.model_slug}/{row.dataset} :: {Path(row.run_key_dir).name}")

        csv = DATASETS_ROOT / row.dataset / "processed" / f"{row.dataset}_clean.csv"
        if not csv.is_file():
            print(f"  [skip] processed CSV missing: {csv}")
            continue

        try:
            texts = pd.read_csv(csv)["clean_text"].astype(str).tolist()
        except Exception as e:
            print(f"  [skip] CSV read failed: {e}")
            continue

        labels = _safe("labels", load_labels, row.model_slug, row.dataset)
        if labels is None:
            continue

        if len(texts) != len(labels):
            print(f"  [skip] texts={len(texts)} labels={len(labels)} mismatch")
            continue

        _safe(
            "tcav",
            tcav.run_tcav,
            row.model_slug, Path(row.run_key_dir), texts, labels,
        )
        _safe(
            "prototypes",
            protopics.extract_prototypes,       # adapt to whichever entry point you use
            row.model_slug, Path(row.run_key_dir), texts, labels,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all_techniques(force: bool = False) -> None:
    _seed_everything(42)

    print("[wired] 1/4 extraction-level techniques")
    _run_extraction_level_techniques()

    print("[wired] 2/4 cross-model CKA")
    _run_cross_model_cka()

    print("[wired] 3/4 attention analysis")
    _run_attention()

    print("[wired] 4/4 probe-level techniques")
    _run_probe_level_techniques()

    print("[wired] done")


if __name__ == "__main__":
    run_all_techniques()