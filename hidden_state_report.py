# ============================================================
# HIDDEN STATE ANOMALY ANALYSIS
# ============================================================

import os
import json
import numpy as np
import pandas as pd


def robust_z_score(values):
    """
    Robust z-score using median and MAD.
    Much less sensitive to extreme outliers than standard z-score.
    """

    values = np.asarray(values, dtype=np.float64)

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    if mad == 0:
        return np.zeros_like(values)

    return (
        0.6745 * (values - median) / mad
    )


def analyze_hidden_states(
    directory,
    anomaly_threshold=4.0,
    top_k=10,
    verbose=True
):
    """
    Analyze all hidden-state JSON files in a directory.

    Produces a compact dataset-wide report rather than describing
    every file independently.
    """

    json_files = [
        f for f in os.listdir(directory)
        if f.endswith(".json")
        and not f.startswith("._")
        and f != ".DS_Store"
    ]

    json_files = sorted(json_files)

    if not json_files:

        print("No valid JSON hidden-state files found.")

        return None

    print("=" * 80)
    print("HIDDEN STATE DATASET ANALYSIS")
    print("=" * 80)
    print(f"Directory: {directory}")
    print(f"Valid JSON files: {len(json_files)}")
    print()

    # --------------------------------------------------------
    # Global containers
    # --------------------------------------------------------

    summary_rows = []
    anomaly_rows = []

    for file_number, filename in enumerate(
        json_files,
        start=1
    ):

        filepath = os.path.join(
            directory,
            filename
        )

        # ----------------------------------------------------
        # Parse filename
        # ----------------------------------------------------

        filename_no_ext = os.path.splitext(
            filename
        )[0]

        parts = filename_no_ext.split("_")

        # Your current naming convention:
        #
        # QwenQwen2-0.5B_goEmo_654
        #
        # So:
        #
        # model = first section
        # dataset = middle section
        # batch = final section

        model_name = parts[0]

        if len(parts) >= 3:

            dataset_name = "_".join(
                parts[1:-1]
            )

        else:

            dataset_name = "unknown"

        batch_number = (
            parts[-1]
            if len(parts) >= 2
            else "unknown"
        )

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------

        try:

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            OSError
        ) as e:

            print(
                f"⚠ Skipping invalid file: "
                f"{filename} | {e}"
            )

            continue

        # ----------------------------------------------------
        # Structural checks
        # ----------------------------------------------------

        if not isinstance(data, list) or len(data) == 0:

            print(
                f"⚠ Empty/invalid structure: {filename}"
            )

            continue

        first_sample = data[0]

        if not isinstance(first_sample, dict):

            print(
                f"⚠ Invalid sample structure: {filename}"
            )

            continue

        layer_names = sorted(
            first_sample.keys(),
            key=lambda x: int(
                x.split("_")[1]
            )
        )

        number_of_layers = len(layer_names)

        hidden_dimension = len(
            first_sample[layer_names[0]]
        )

        sample_count = len(data)

        # ----------------------------------------------------
        # Validate structure
        # ----------------------------------------------------

        nan_count = 0
        inf_count = 0
        malformed_samples = 0

        sample_final_norms = []

        layer_norms = {
            layer_name: []
            for layer_name in layer_names
        }

        for sample_index, sample in enumerate(data):

            if not isinstance(sample, dict):

                malformed_samples += 1
                continue

            try:

                for layer_name in layer_names:

                    vector = np.asarray(
                        sample[layer_name],
                        dtype=np.float32
                    )

                    if len(vector) != hidden_dimension:

                        malformed_samples += 1
                        continue

                    nan_count += int(
                        np.isnan(vector).sum()
                    )

                    inf_count += int(
                        np.isinf(vector).sum()
                    )

                    norm = float(
                        np.linalg.norm(vector)
                    )

                    layer_norms[
                        layer_name
                    ].append(norm)

                final_vector = np.asarray(
                    sample[layer_names[-1]],
                    dtype=np.float32
                )

                final_norm = float(
                    np.linalg.norm(final_vector)
                )

                sample_final_norms.append(
                    final_norm
                )

            except Exception:

                malformed_samples += 1

        # ----------------------------------------------------
        # Detect unusual representations
        # ----------------------------------------------------

        final_norms = np.asarray(
            sample_final_norms,
            dtype=np.float64
        )

        anomaly_scores = robust_z_score(
            final_norms
        )

        anomaly_indices = np.where(
            np.abs(anomaly_scores)
            >= anomaly_threshold
        )[0]

        # ----------------------------------------------------
        # Summary statistics
        # ----------------------------------------------------

        mean_final_norm = float(
            np.mean(final_norms)
        )

        median_final_norm = float(
            np.median(final_norms)
        )

        std_final_norm = float(
            np.std(final_norms)
        )

        # ----------------------------------------------------
        # Find most variable layer
        # ----------------------------------------------------

        layer_std = {}

        for layer_name, values in layer_norms.items():

            values = np.asarray(
                values,
                dtype=np.float64
            )

            if len(values) > 0:

                layer_std[layer_name] = float(
                    np.std(values)
                )

        if layer_std:

            most_variable_layer = max(
                layer_std,
                key=layer_std.get
            )

            least_variable_layer = min(
                layer_std,
                key=layer_std.get
            )

        else:

            most_variable_layer = "N/A"
            least_variable_layer = "N/A"

        # ----------------------------------------------------
        # Store anomalies
        # ----------------------------------------------------

        for local_rank, sample_idx in enumerate(
            anomaly_indices
        ):

            anomaly_rows.append({
                "file": filename,
                "model": model_name,
                "dataset": dataset_name,
                "batch": batch_number,
                "sample_index": int(sample_idx),
                "anomaly_score": float(
                    anomaly_scores[sample_idx]
                ),
                "final_layer_norm": float(
                    final_norms[sample_idx]
                )
            })

        # ----------------------------------------------------
        # Determine status
        # ----------------------------------------------------

        status = "OK"

        if nan_count > 0 or inf_count > 0:

            status = "CRITICAL"

        elif malformed_samples > 0:

            status = "WARNING"

        elif len(anomaly_indices) > 0:

            status = "ANOMALIES"

        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------

        summary_rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "batch": batch_number,
            "samples": sample_count,
            "layers": number_of_layers,
            "hidden_dim": hidden_dimension,
            "nan": nan_count,
            "inf": inf_count,
            "malformed": malformed_samples,
            "anomalies": len(anomaly_indices),
            "mean_final_norm": mean_final_norm,
            "median_final_norm": median_final_norm,
            "std_final_norm": std_final_norm,
            "most_variable_layer": most_variable_layer,
            "least_variable_layer": least_variable_layer,
            "status": status
        })

    # --------------------------------------------------------
    # Convert to DataFrames
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    anomalies_df = pd.DataFrame(
        anomaly_rows
    )

    # --------------------------------------------------------
    # Overall report
    # --------------------------------------------------------

    if verbose:

        print()
        print("=" * 80)
        print("OVERALL HEALTH")
        print("=" * 80)

        total_samples = int(
            summary_df["samples"].sum()
        )

        total_anomalies = int(
            summary_df["anomalies"].sum()
        )

        total_nan = int(
            summary_df["nan"].sum()
        )

        total_inf = int(
            summary_df["inf"].sum()
        )

        total_malformed = int(
            summary_df["malformed"].sum()
        )

        print(
            f"Total files:            "
            f"{len(summary_df):,}"
        )

        print(
            f"Total samples:          "
            f"{total_samples:,}"
        )

        print(
            f"NaN values:             "
            f"{total_nan:,}"
        )

        print(
            f"Infinite values:        "
            f"{total_inf:,}"
        )

        print(
            f"Malformed samples:      "
            f"{total_malformed:,}"
        )

        print(
            f"Anomalous samples:      "
            f"{total_anomalies:,}"
        )

        if total_nan > 0 or total_inf > 0:

            print(
                "\n❌ CRITICAL: invalid numerical values detected."
            )

        elif total_malformed > 0:

            print(
                "\n⚠ WARNING: malformed samples detected."
            )

        elif total_anomalies > 0:

            print(
                "\n⚠ ANOMALIES FOUND: unusual representations detected."
            )

        else:

            print(
                "\n✓ No major abnormalities detected."
            )

        # ----------------------------------------------------
        # Dataset overview
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("DATASET OVERVIEW")
        print("=" * 80)

        dataset_summary = (
            summary_df
            .groupby(
                ["model", "dataset"],
                as_index=False
            )
            .agg({
                "samples": "sum",
                "layers": "first",
                "hidden_dim": "first",
                "nan": "sum",
                "inf": "sum",
                "malformed": "sum",
                "anomalies": "sum",
                "mean_final_norm": "mean",
                "std_final_norm": "mean"
            })
        )

        print(
            dataset_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

        # ----------------------------------------------------
        # Top anomalies
        # ----------------------------------------------------

        if not anomalies_df.empty:

            print()
            print("=" * 80)
            print(
                f"TOP {top_k} MOST UNUSUAL SAMPLES"
            )
            print("=" * 80)

            top_anomalies = (
                anomalies_df
                .assign(
                    abs_score=lambda x:
                    x["anomaly_score"].abs()
                )
                .sort_values(
                    "abs_score",
                    ascending=False
                )
                .head(top_k)
            )

            print(
                top_anomalies[
                    [
                        "dataset",
                        "batch",
                        "sample_index",
                        "anomaly_score",
                        "final_layer_norm"
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.3f}"
                )
            )

        # ----------------------------------------------------
        # Most suspicious files
        # ----------------------------------------------------

        suspicious_files = (
            summary_df[
                summary_df["anomalies"] > 0
            ]
            .sort_values(
                "anomalies",
                ascending=False
            )
        )

        if not suspicious_files.empty:

            print()
            print("=" * 80)
            print("FILES REQUIRING ATTENTION")
            print("=" * 80)

            print(
                suspicious_files[
                    [
                        "model",
                        "dataset",
                        "batch",
                        "samples",
                        "anomalies",
                        "nan",
                        "inf",
                        "malformed",
                        "status"
                    ]
                ].to_string(
                    index=False
                )
            )

    return {
        "summary": summary_df,
        "anomalies": anomalies_df
    }