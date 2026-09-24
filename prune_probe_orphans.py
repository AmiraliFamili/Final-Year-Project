"""
prune_probe_orphans.py — find and delete probe run folders that the
current code will never resume or complete.

A folder is KEPT when either:
  * it has a `completion.json` (a fully finished run), OR
  * its hash matches the hash the current code would compute
    (the run is either in progress or waiting to be resumed).

Everything else is an ORPHAN: created by a config that no longer exists,
unresumable, and taking up disk.

Usage:
    python prune_probe_orphans.py              # dry run, list only
    python prune_probe_orphans.py --delete     # actually remove
"""
from __future__ import annotations
import argparse, re, shutil
from pathlib import Path

import _shared
import Probe as probe


# ────────────────────────────────────────────────────────────────────────
# Must match the notebook (cell 6 for probes, cell 12 for the rest).
# If you edit those, edit these too, or the script will think everything
# is an orphan.
# ────────────────────────────────────────────────────────────────────────

REPEATS                  = 4
MAX_SAMPLES              = None
SHUFFLED_CONTROL         = True
SHUFFLED_CONTROL_REPEATS = 3

PROBE_ARGS = [
    dict(name="linear_logistic", type="logistic", complexity="linear",
         standardize=True, C=1.0, max_iter=3000, selection_metric="macro_f1"),
    dict(name="mlp_1_hidden", type="mlp", complexity="1_hidden",
         standardize=True, hidden_dims=["0.5d"],
         learning_rate=1e-3, weight_decay=1e-4,
         epochs=80, batch_size=256, patience=12,
         selection_metric="macro_f1"),
    dict(name="mlp_2_hidden", type="mlp", complexity="2_hidden",
         standardize=True, hidden_dims=["0.5d", "0.25d"],
         learning_rate=1e-3, weight_decay=1e-4,
         epochs=80, batch_size=256, patience=12,
         selection_metric="macro_f1"),
    dict(name="mlp_3_hidden", type="mlp", complexity="3_hidden",
         standardize=True, hidden_dims=["0.5d", "0.25d", "0.125d"],
         learning_rate=1e-3, weight_decay=1e-4,
         epochs=80, batch_size=256, patience=12,
         selection_metric="macro_f1"),
]


def current_hash_for(artifact_dir: Path) -> str | None:
    """Compute the trial hash the current code would produce for this pair."""
    try:
        art = probe.ExtractionArtifact(artifact_dir)
        cfg = probe.AnalysisConfig(
            dataset=probe.DatasetContract(**_shared.contract_dict_for(art.dataset_name)),
            probes=[probe.ProbeSpec(**args) for args in PROBE_ARGS],
            layers="all",
            split=probe.SplitConfig(train=0.80, validation=0.10, test=0.10, seed=42),
            repeats=REPEATS,
            max_samples=MAX_SAMPLES,
            shuffled_label_control=SHUFFLED_CONTROL,
            shuffled_control_repeats=SHUFFLED_CONTROL_REPEATS,
            pca_enabled=True,
            silhouette_enabled=True,
            pca_samples=min(3000, MAX_SAMPLES or 3000),
            silhouette_samples=min(3000, MAX_SAMPLES or 3000),
            enable_per_class_metrics=True,
            enable_feature_statistics=True,
            verbose=1,
        )
        trial_cfg = probe.build_trial_config(art, cfg)
        return probe.generate_trial_hash(trial_cfg)
    except Exception as exc:
        print(f"    could not hash {artifact_dir}: {type(exc).__name__}: {exc}")
        return None


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="Actually remove orphan folders. Without this, dry-run.")
    args = ap.parse_args()

    interex = _shared.INTEREX_ROOT
    if not interex.is_dir():
        print(f"no interEx root: {interex}")
        return 1

    orphans: list[tuple[Path, int]] = []
    kept = 0

    for model_dir in sorted(interex.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name == "analysis":
                continue

            artifact_dir = _shared.artifact_dir(model_dir.name, dataset_dir.name)
            if not artifact_dir.is_dir():
                continue

            current_hash = current_hash_for(artifact_dir)
            if current_hash is None:
                print(f"  {model_dir.name}/{dataset_dir.name}: skipping (hash failed)")
                continue

            for run_dir in sorted(dataset_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                m = re.search(r"__h([a-f0-9]{10})$", run_dir.name)
                if not m:
                    continue
                run_hash = m.group(1)
                has_complete = (run_dir / "completion.json").is_file()
                is_current = (run_hash == current_hash[:10])

                if has_complete or is_current:
                    kept += 1
                    marker = "COMPLETE" if has_complete else "ACTIVE"
                    print(f"  KEEP  [{marker}]  {run_dir.relative_to(interex)}")
                else:
                    size = dir_size(run_dir)
                    orphans.append((run_dir, size))
                    print(f"  DROP           {run_dir.relative_to(interex)}  ({human_size(size)})")

    print()
    total_size = sum(s for _, s in orphans)
    print(f"Would keep  : {kept} folder(s)")
    print(f"Would drop  : {len(orphans)} folder(s), {human_size(total_size)} total")

    if not orphans:
        return 0

    if not args.delete:
        print("\nDry run. Re-run with --delete to remove.")
        return 0

    for p, _ in orphans:
        shutil.rmtree(p)
        print(f"  removed {p.name}")
    print(f"\nRemoved {len(orphans)} orphan(s), freed {human_size(total_size)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
