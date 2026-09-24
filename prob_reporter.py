"""
probe_report.py — discover, validate, and plot probe runs in interEx/.

Reads:
    /Volumes/Amirali/interEx/<slug>/<dataset>/<run_key>/layer_probe_results.csv

Writes:
    ./probe_report/validity.csv          one row per run: alignment, splits, selectivity
    ./probe_report/<model>__<dataset>__layers.png   layer curves per model/dataset
    ./probe_report/*.png                 ResultAnalyser's full dashboard
    ./probe_report/report.html           index
"""
from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Locate the project modules ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from _shared import INTEREX_ROOT, model_slug          # noqa: E402
from Master_Analyser import ResultAnalyser, ForensicAuditor, Renderer  # noqa: E402

OUT = PROJECT_ROOT / "probe_report"
OUT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# 1. Discovery
# ─────────────────────────────────────────────────────────────────────────
def discover_runs(root: Path = INTEREX_ROOT) -> list[dict]:
    """Return one dict per (slug, dataset, run_key) that has a results CSV."""
    runs = []
    if not root.is_dir():
        return runs
    for slug_dir in sorted(root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("_"):
            continue
        for dataset_dir in sorted(slug_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for run_dir in sorted(dataset_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                results = run_dir / "layer_probe_results.csv"
                if results.is_file() and results.stat().st_size > 0:
                    runs.append({
                        "slug":    slug_dir.name,
                        "dataset": dataset_dir.name,
                        "run_dir": run_dir,
                        "results": results,
                    })
    return runs


# ─────────────────────────────────────────────────────────────────────────
# 2. Validity — one row per run, five checks
# ─────────────────────────────────────────────────────────────────────────
def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def check_alignment(run_dir: Path) -> dict:
    """Did extraction leave a text-hash fingerprint, and does it match?"""
    m = _load_json(run_dir / "probe_alignment_manifest.json")
    if not m:
        return {"align_text": "MISSING", "align_labels": "MISSING", "sample_ids_match": None}
    return {
        "align_text":       m.get("artifact_text_provenance_status", "MISSING"),
        "align_labels":     m.get("artifact_label_provenance_status", "MISSING"),
        "sample_ids_match": m.get("sample_ids_match"),
    }


def check_splits(run_dir: Path) -> dict:
    """Zero overlap between train/val/test, for every repeat."""
    p = run_dir / "split_indices.npz"
    if not p.is_file():
        return {"split_overlap_max": None, "split_repeats": 0}

    arch = np.load(p, allow_pickle=True)
    repeats = set()
    for key in arch.files:
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "repeat":
            repeats.add(int(parts[1]))

    worst = 0
    for r in sorted(repeats):
        try:
            tr = set(arch[f"repeat_{r}_train"].tolist())
            va = set(arch[f"repeat_{r}_validation"].tolist())
            te = set(arch[f"repeat_{r}_test"].tolist())
        except KeyError:
            continue
        worst = max(worst, len(tr & va), len(tr & te), len(va & te))

    return {"split_overlap_max": int(worst), "split_repeats": len(repeats)}


def check_performance(run_dir: Path) -> dict:
    """Best layer, best F1, control F1, selectivity."""
    df = pd.read_csv(run_dir / "layer_probe_results.csv")
    best_row = df.loc[df["test_macro_f1"].idxmax()]

    out = {
        "n_layers":       int(df["layer_index"].nunique()),
        "n_probes":       int(df["probe"].nunique()),
        "best_macro_f1":  float(best_row["test_macro_f1"]),
        "best_layer":     int(best_row["layer_index"]),
        "best_probe":     str(best_row["probe"]),
        "first_layer_f1": float(df[df["layer_index"] == df["layer_index"].min()]["test_macro_f1"].mean()),
        "last_layer_f1":  float(df[df["layer_index"] == df["layer_index"].max()]["test_macro_f1"].mean()),
    }

    if "control_macro_f1" in df.columns and df["control_macro_f1"].notna().any():
        out["control_max_f1"] = float(df["control_macro_f1"].max())
        out["selectivity"]    = out["best_macro_f1"] - out["control_max_f1"]
    else:
        out["control_max_f1"] = None
        out["selectivity"]    = None

    return out


def validity_row(run: dict) -> dict:
    d = run["run_dir"]
    row = {"model": run["slug"], "dataset": run["dataset"], "run_key": d.name}
    row.update(check_alignment(d))
    row.update(check_splits(d))
    row.update(check_performance(d))
    return row


# ─────────────────────────────────────────────────────────────────────────
# 3. Layer-by-layer plots — one figure per (model, dataset)
# ─────────────────────────────────────────────────────────────────────────
def plot_layers(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    made = []
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Left: absolute F1 per layer, one line per probe
        ax = axes[0]
        for probe in sorted(group["probe"].unique()):
            sub = group[group["probe"] == probe].sort_values("layer_index")
            ax.plot(sub["layer_index"], sub["test_macro_f1"],
                    marker="o", linewidth=2, label=probe)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Test Macro-F1")
        ax.set_title(f"{model} / {dataset} — layer-wise F1")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # Right: selectivity = true − control. Positive = real signal.
        ax = axes[1]
        if "control_macro_f1" in group.columns and group["control_macro_f1"].notna().any():
            for probe in sorted(group["probe"].unique()):
                sub = group[group["probe"] == probe].sort_values("layer_index")
                sel = sub["test_macro_f1"] - sub["control_macro_f1"]
                ax.plot(sub["layer_index"], sel, marker="o",
                        linewidth=2, label=probe)
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax.set_title("Selectivity (true − shuffled-label control)")
            ax.set_ylabel("F1 gap")
        else:
            ax.text(0.5, 0.5, "no shuffled-label control in this run",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Layer index")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        fig.tight_layout()
        path = out_dir / f"{model}__{dataset}__layers.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        made.append(path)
    return made


# ─────────────────────────────────────────────────────────────────────────
# 4. Driver
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    runs = discover_runs()
    if not runs:
        print(f"No probe runs found under {INTEREX_ROOT}.")
        return 1

    print(f"Discovered {len(runs)} probe run(s).\n")

    # 4a. Validity table
    rows = [validity_row(r) for r in runs]
    validity = pd.DataFrame(rows)
    validity.to_csv(OUT / "validity.csv", index=False)

    cols = ["model", "dataset", "align_text", "align_labels",
            "split_overlap_max", "best_macro_f1", "best_layer",
            "control_max_f1", "selectivity"]
    print("=" * 100)
    print("VALIDITY SUMMARY")
    print("=" * 100)
    print(validity[cols].to_string(index=False))
    print()

    # 4b. Forensic audit per run (uses Master_Analyser's own auditor)
    print("=" * 100)
    print("FORENSIC AUDIT")
    print("=" * 100)
    auditor = ForensicAuditor(
        root=INTEREX_ROOT, experiment_id="report", renderer=Renderer(silent=True),
    )
    audit_rows = []
    for r in runs:
        rep = auditor.audit_trial(r["run_dir"])
        audit_rows.append({
            "model":   r["slug"],
            "dataset": r["dataset"],
            "status":  rep["status"],
            "errors":  len(rep["errors"]),
            "warnings": len(rep["warnings"]),
        })
        if rep["errors"]:
            for e in rep["errors"]:
                print(f"  ✗ {r['slug']}/{r['dataset']}: {e}")
        if rep["warnings"]:
            for w in rep["warnings"]:
                print(f"  ⚠ {r['slug']}/{r['dataset']}: {w}")
    pd.DataFrame(audit_rows).to_csv(OUT / "forensic_audit.csv", index=False)

    # 4c. Layer-by-layer plots
    print()
    print("=" * 100)
    print("LAYER-BY-LAYER PLOTS")
    print("=" * 100)
    all_df = pd.concat([pd.read_csv(r["results"]) for r in runs], ignore_index=True)

    # Ensure `model`/`dataset` columns exist for ResultAnalyser.
    if "model" not in all_df.columns:
        # Backfill from run metadata if the runner didn't add them.
        rename = {r["results"]: (r["slug"], r["dataset"]) for r in runs}
        all_df["model"] = all_df.get("model", None)
        all_df["dataset"] = all_df.get("dataset", None)

    made = plot_layers(all_df, OUT)
    for p in made:
        print(f"  ✓ {p.name}")

    # 4d. Master_Analyser's full dashboard
    analyser = ResultAnalyser(OUT, Renderer(silent=True))
    analyser.generate_plots(all_df)

    print()
    print(f"Everything written to {OUT}/")
    print(f"  validity.csv          — five validity checks per run")
    print(f"  forensic_audit.csv    — pass/warn/fail per run")
    print(f"  *__layers.png         — layer curves + selectivity per model/dataset")
    print(f"  layer_curves_*.png    — ResultAnalyser's cross-run curves")
    print(f"  heatmap_*.png         — ResultAnalyser's layer × probe heatmaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())