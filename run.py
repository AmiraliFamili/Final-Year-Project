import Extraction as EX
from pathlib import Path
import importlib
importlib.reload(EX)

for slug in ("Qwen2-0.5B", "Qwen2.5-0.5B"):
    for ds in ("emotion", "goemo", "tweet_eval_emotion"):
        d = EX.build_dataset_directory(slug, ds)
        if not (d / "extraction.json").exists():
            continue
        try:
            EX.repair_dataset_auxiliaries(
                dataset=EX.discover_processed_datasets(show_info=False)[0][ds],
                output_dir=d,
                batch_size=32, pooling="mean", max_length=128,
                storage_dtype=__import__("numpy").float32,
                experiment_id="master_v2",
            )
            print("repaired:", slug, ds)
        except Exception as e:
            print("failed:", slug, ds, type(e).__name__, e)