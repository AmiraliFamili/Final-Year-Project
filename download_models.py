#!/usr/bin/env python3
"""
download_models.py — CLI wrapper around model_downloader.download_model.

    python3 download_models.py                      # all registry models
    python3 download_models.py gpt2                 # one
    python3 download_models.py --force bert-base…   # redownload
    python3 download_models.py --probe              # print endpoints only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_downloader as MD
import Extraction as EX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="Model ids (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="Redownload even if already complete")
    ap.add_argument("--probe", action="store_true",
                    help="Just print probe results and exit")
    ap.add_argument("--repair", action="store_true",
                help="Only visit folders that are not yet complete")
    ap.add_argument("--audit", action="store_true",
                    help="Print a per-model status table and exit")

    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Models root : {MD.MODELS_ROOT}")
    print(f"HF cache    : {MD.HF_HUB_CACHE}")
    print(f"HF_TOKEN    : {'set' if token else 'not set'}")
    print("Endpoints   :")
    for i, ep in enumerate(MD.probe_endpoints(force=True)):
        print(f"  {i+1}. {ep}  {'← will use' if i == 0 else ''}")
    if args.probe:
        return 0

    if args.audit:
        status = MD.audit_all([s.name for s in EX.MODEL_REGISTRY])
        from collections import Counter
        counts = Counter(status.values())
        for name, st in status.items():
            icon = {"complete": "✓", "missing-config": "?", 
                    "missing-weights": "!", "absent": "·"}[st]
            print(f"  {icon} {name:<52} {st}")
        print()
        print("  summary:", dict(counts))
        return 0

    order = {s.name: i for i, s in enumerate(EX.MODEL_REGISTRY)}
    if args.models:
        wanted = sorted(set(args.models), key=lambda n: order.get(n, 10_000))
    else:
        wanted = [s.name for s in EX.MODEL_REGISTRY]

    print(f"\nModels      : {len(wanted)}\n")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures: list[tuple[str, str]] = []

    for i, name in enumerate(wanted, 1):
        dest = MD.model_dir_for(name)
        print(f"[{i:02d}/{len(wanted)}] {name}")
        if MD.is_complete(dest) and not args.force:
            print(f"      ↺ already complete — {dest}")
            counts["skip"] += 1
            continue
        try:
            MD.download_model(name, dest, force=args.force, token=token)
            counts["ok"] += 1
        except Exception as exc:
            print(f"      ✗ {type(exc).__name__}: {exc}")
            failures.append((name, str(exc)))
            counts["fail"] += 1

    print()
    print(f"Downloaded : {counts['ok']}")
    print(f"Skipped    : {counts['skip']}")
    print(f"Failed     : {counts['fail']}")
    if failures:
        print("\nFailures:")
        for n, d in failures:
            print(f"  {n}\n      {d.splitlines()[0]}")
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
    
    