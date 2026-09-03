"""
example_probe.py

Minimal example:
    python example_probe.py

The point of this file is that the user-facing experiment does not need to
know ISEAR/GoEmotions column names or any v4.2 contracts.
"""

from pathlib import Path

import pandas as pd

from probing_pipeline import ProbingPipeline


DATASET = Path("probe_run/cleaned_dataset.csv")

if not DATASET.exists():
    pd.DataFrame(
        {
            "review_text": [
                "I got the result I wanted and felt delighted.",
                "The sudden noise frightened me.",
                "I was angry because the plan changed.",
                "Losing the opportunity made me sad.",
                "The surprise birthday party made me happy.",
                "I felt guilty after breaking the rule.",
                "The disgusting smell made me uncomfortable.",
                "I was afraid of what would happen next.",
                "I felt proud after finishing the project.",
                "The argument left me frustrated and upset.",
                "Getting the message was wonderful news.",
                "I regretted what I had said.",
            ],
            "emotion": [
                "joy",
                "fear",
                "anger",
                "sadness",
                "joy",
                "guilt",
                "disgust",
                "fear",
                "joy",
                "anger",
                "joy",
                "sadness",
            ],
        }
    ).to_csv(DATASET, index=False)

pipeline = ProbingPipeline(
    model_name="bert-base-uncased",
    dataset_paths=str(DATASET),
    # No text_column or label_column is required.
    probe_complexity="linear",
    visualization_type=["layerwise", "heatmap", "shuffle_advantage"],
    output_dir="runs/dummy_bert_probe",
    batch_size=8,
    max_samples=5000,
    max_length=128,
    pooling="mean",
    repeats=3,
)

results = pipeline.run()

print("\nBest rows:")
print(
    results.sort_values("test_macro_f1", ascending=False)
    .head(10)
    .to_string(index=False)
)
