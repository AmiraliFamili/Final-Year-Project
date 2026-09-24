# wired.py
"""
wired.py — orchestrator for every downstream interpretability technique.

Design goals
------------
* Line-buffered output so that progress appears immediately when piped.
* Idempotent: every technique skips itself if its output already exists,
  unless --force is passed.
* Failure-isolated: one failing (model, dataset, technique) triple is
  logged and the run continues.
* Verbose: prints the exact function it is about to call, with its args.
* Dry-run mode: lists what would run, then exits.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Force line-buffered stdout even when the process is piped.
# The `-u` flag on the command line achieves the same thing; doing it here
# means the script is safe to invoke either way.
# ─────────────────────────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


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
# Timing + failure-isolated call wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _safe(label: str, fn, *args, skip_if: Path | None = None,
          force: bool = False, dry_run: bool = False, **kwargs):
    """Call fn(*args, **kwargs) with timing, skip-if-exists, and error isolation.

    Returns the function result, or None on skip/failure.
    """
    if skip_if is not None and skip_if.exists() and not force:
        print(f"[{_stamp()}] [{label}] skip  (exists: {skip_if.name})")
        return None

    arg_repr = ", ".join(
        [str(a) for a in args] +
        [f"{k}={v!r}" for k, v in kwargs.items()]
    )
    if len(arg_repr) > 120:
        arg_repr = arg_repr[:117] + "..."

    if dry_run:
        print(f"[{_stamp()}] [{label}] would call {fn.__name__}({arg_repr})")
        return None

    print(f"[{_stamp()}] [{label}] start {fn.__name__}({arg_repr})")
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[{_stamp()}] [{label}] done  {elapsed:6.2f}s")
        return result
    except FileNotFoundError as e:
        print(f"[{_stamp()}] [{label}] SKIP — missing file: {e}")
    except ValueError as e:
        print(f"[{_stamp()}] [{label}] SKIP — invalid input: {e}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"[{_stamp()}] [{label}] FAIL after {elapsed:.2f}s — "
              f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _iter_extraction_pairs(root: Path):
    """Yield (model_slug, dataset, dataset_dir) for every extraction that
    has hidden_states.npy AND labels.npy on disk."""
    if not root.is_dir():
        print(f"[{_stamp()}] [discover] root missing: {root}")
        return
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_dir.name.startswith("_"):
            continue
        for ds_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if not (ds_dir / "hidden_states.npy").is_file():
                continue
            if not (ds_dir / "labels.npy").is_file():
                print(f"[{_stamp()}] [skip] {model_dir.name}/{ds_dir.name}: labels.npy missing")
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
        if model_dir.name.startswith("_"):
            continue
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

def _run_extraction_level_techniques(force: bool, dry_run: bool) -> None:
    from _shared import HIDDEN_STATES_ROOT
    import expl.geometry as geometry, geometry_multilabel, cka

    for model_slug_, dataset, _ in _iter_extraction_pairs(HIDDEN_STATES_ROOT):
        from _shared import analysis_dir_for_extraction
        out_dir = analysis_dir_for_extraction(model_slug_, dataset)
        print(f"[{_stamp()}] === {model_slug_}/{dataset} ===")

        _safe(
            "geom",
            geometry.run_geometry, model_slug_, dataset,
            skip_if=out_dir / "geometry.parquet",
            force=force, dry_run=dry_run,
        )

        if dataset == "goemo":
            _safe(
                "geom-ml",
                geometry_multilabel.run_geometry_multilabel, model_slug_, dataset,
                skip_if=out_dir / "geometry_multilabel.parquet",
                force=force, dry_run=dry_run,
            )

        _safe(
            "cka-within",
            cka.run_cka_within, model_slug_, dataset,
            skip_if=out_dir / "cka_within.npz",
            force=force, dry_run=dry_run,
        )


def _run_cross_model_cka(force: bool, dry_run: bool) -> None:
    from _shared import HIDDEN_STATES_ROOT, INTEREX_ROOT
    import expl.cka as cka

    slugs: list[str] = []
    for model_dir in sorted(p for p in HIDDEN_STATES_ROOT.iterdir() if p.is_dir()):
        if model_dir.name.startswith("_"):
            continue
        if any((ds / "hidden_states.npy").is_file() for ds in model_dir.iterdir()):
            slugs.append(model_dir.name)

    print(f"[{_stamp()}] cross-model CKA across {len(slugs)} models "
          f"({len(slugs) * (len(slugs) - 1) // 2} pairs)")

    for i, a in enumerate(slugs):
        for b in slugs[i + 1:]:
            out_dir = INTEREX_ROOT / "_cross_model" / f"{a}__{b}" / "isear"
            _safe(
                "cka-cross",
                cka.run_cka_cross, a, b, "isear",
                skip_if=out_dir / "cka_cross.npz",
                force=force, dry_run=dry_run,
            )


def _run_attention(force: bool, dry_run: bool) -> None:
    from _shared import HIDDEN_STATES_ROOT, MODELS_ROOT, DATASETS_ROOT
    from _shared import analysis_dir_for_extraction
    import expl.attention as attention

    for model_slug_, dataset, _ in _iter_extraction_pairs(HIDDEN_STATES_ROOT):
        if not (MODELS_ROOT / model_slug_).is_dir():
            print(f"[{_stamp()}] [attention] skip {model_slug_}/{dataset}: model snapshot missing")
            continue

        csv = DATASETS_ROOT / dataset / "processed" / f"{dataset}_clean.csv"
        if not csv.is_file():
            print(f"[{_stamp()}] [attention] skip {model_slug_}/{dataset}: processed CSV missing")
            continue

        out_dir = analysis_dir_for_extraction(model_slug_, dataset)
        skip_if = out_dir / "attention.npz"
        if skip_if.exists() and not force:
            print(f"[{_stamp()}] [attention] skip  (exists: {skip_if.name})")
            continue

        if dry_run:
            print(f"[{_stamp()}] [attention] would run {model_slug_}/{dataset}")
            continue

        try:
            texts = pd.read_csv(csv)["clean_text"].astype(str).tolist()
        except Exception as e:
            print(f"[{_stamp()}] [attention] skip {model_slug_}/{dataset}: {e}")
            continue

        _safe(
            "attention",
            attention.run_attention, model_slug_, dataset, texts,
            max_length=64, sample_size=300,
            dry_run=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Probe-level techniques
# ─────────────────────────────────────────────────────────────────────────────

def _run_probe_level_techniques(force: bool, dry_run: bool) -> None:
    from _shared import DATASETS_ROOT, load_labels
    import expl.tcav as tcav
    try:
        import expl.protopics as protopics
    except ImportError:
        protopics = None

    runs = discover_probe_runs()
    if runs.empty:
        print(f"[{_stamp()}] [probe-level] no probe runs found under interEx/")
        return

    print(f"[{_stamp()}] [probe-level] {len(runs)} probe runs to process")

    for row in runs.itertuples(index=False):
        run_dir = Path(row.run_key_dir)
        print(f"[{_stamp()}] === {row.model_slug}/{row.dataset} :: {run_dir.name} ===")

        csv = DATASETS_ROOT / row.dataset / "processed" / f"{row.dataset}_clean.csv"
        if not csv.is_file():
            print(f"[{_stamp()}]   [skip] processed CSV missing: {csv}")
            continue

        try:
            texts = pd.read_csv(csv)["clean_text"].astype(str).tolist()
        except Exception as e:
            print(f"[{_stamp()}]   [skip] CSV read failed: {e}")
            continue

        labels = _safe("labels", load_labels, row.model_slug, row.dataset)
        if labels is None:
            continue

        if len(texts) != len(labels):
            print(f"[{_stamp()}]   [skip] texts={len(texts)} labels={len(labels)} mismatch")
            continue

        if tcav is not None:
            _safe(
                "tcav",
                tcav.run_tcav,
                row.model_slug, run_dir, texts, labels,
                dry_run=dry_run,
            )

        # Uncomment when the protopics entry point matches its signature.
        # if protopics is not None:
        #     _safe(
        #         "prototypes",
        #         protopics.run_prototypes,
        #         row.model_slug, run_dir, texts, labels,
        #         dry_run=dry_run,
        #     )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all_techniques(force: bool = False, dry_run: bool = False,
                       only: str | None = None) -> None:
    _seed_everything(42)
    t_start = time.perf_counter()

    stages = {
        "extraction": _run_extraction_level_techniques,
        "cross-cka":  _run_cross_model_cka,
        "attention":  _run_attention,
        "probe":      _run_probe_level_techniques,
    }

    selected = [only] if only else list(stages.keys())

    for i, name in enumerate(selected, 1):
        print(f"\n[{_stamp()}] ───── stage {i}/{len(selected)}: {name} ─────")
        try:
            stages[name](force=force, dry_run=dry_run)
        except Exception as e:
            print(f"[{_stamp()}] stage {name} crashed: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)

    elapsed = time.perf_counter() - t_start
    print(f"\n[{_stamp()}] wired.py complete in {elapsed:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-run techniques even if their output exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be called, then exit.")
    ap.add_argument("--only", choices=["extraction", "cross-cka", "attention", "probe"],
                    help="Run only one stage.")
    args = ap.parse_args()

    run_all_techniques(force=args.force, dry_run=args.dry_run, only=args.only)