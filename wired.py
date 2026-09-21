# wired.py
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
"""
Interpretability orchestrator.

After probing has run for a (model, dataset) pair, this module runs
every downstream technique that reads either the hidden states or the
fitted probes. It is idempotent: each technique writes to a fixed path
and skips itself if that path exists.
"""

from pathlib import Path
import json, numpy as np, pandas as pd

from _shared import (
    artifact_dir, PROBE_ROOT, load_hidden_states, load_labels,
)
import geometry, geometry_multilabel, attention, cka, tcav, ig, protopics

def discover_probe_runs(root: Path = PROBE_ROOT) -> pd.DataFrame:
    """Walk <model>/<dataset>/index.json and return one row per completed run."""
    rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            idx = dataset_dir / "index.json"
            if not idx.exists(): continue
            payload = json.loads(idx.read_text())
            for run in payload.get("runs", []):
                rows.append({
                    "model_slug": model_dir.name,
                    "dataset":    dataset_dir.name,
                    "run_key_dir": dataset_dir / run["run_key"],
                    "probes":     run["probes"],
                    "task_type":  run["task_type"],
                })
    return pd.DataFrame(rows)

def run_all_techniques(force: bool = False):
    # ── 1. Geometry (only needs hidden_states.npy — can run now) ──
    from Extraction import HIDDEN_STATES_ROOT
    for model_dir in sorted(p for p in HIDDEN_STATES_ROOT.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if not (dataset_dir / "hidden_states.npy").exists(): continue
            dataset = dataset_dir.name
            try:
                geometry.run_geometry(model_dir.name, dataset)
            except Exception as e:
                print(f"[geom] {model_dir.name}/{dataset}: {e}")
            if dataset == "goemo":
                try:
                    geometry_multilabel.run_geometry_multilabel(model_dir.name, dataset)
                except Exception as e:
                    print(f"[geom-ml] {model_dir.name}/{dataset}: {e}")

    # ── 2. Within-model CKA ──
    for model_dir in sorted(p for p in HIDDEN_STATES_ROOT.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if not (dataset_dir / "hidden_states.npy").exists(): continue
            cka.run_cka_within(model_dir.name, dataset_dir.name)

    # ── 3. Cross-model CKA (pairwise; only if you've extracted both) ──
    slugs = [p.name for p in HIDDEN_STATES_ROOT.iterdir()
             if p.is_dir() and any(d.name == "isear" for d in p.iterdir())]
    for i, a in enumerate(slugs):
        for b in slugs[i+1:]:
            try:
                cka.run_cka_cross(a, b, "isear")
            except Exception as e:
                print(f"[cka-cross] {a} vs {b}: {e}")

    # ── 4. Attention (needs the model on disk, one pass per model) ──
    # (your existing wired.py already had this — keep it, but cap max_length)

    # ── 5. Probe-dependent: TCAV, prototypes, IG, steering ──
    runs = discover_probe_runs()
    for row in runs.itertuples(index=False):
        texts  = pd.read_csv(f"datasets/{row.dataset}/processed/{row.dataset}_clean.csv")["clean_text"].tolist()
        labels = load_labels(row.model_slug, row.dataset)
        try:
            tcav.run_tcav(row.model_slug, row.run_key_dir, texts, labels)
        except Exception as e:
            print(f"[tcav] {row.model_slug}/{row.dataset}: {e}")
        try:
            protopics.run_prototypes(row.model_slug, row.run_key_dir, texts, labels)
        except Exception as e:
            print(f"[proto] {row.model_slug}/{row.dataset}: {e}")
        # IG and steering only if you have time budget.

if __name__ == "__main__":
    run_all_techniques()