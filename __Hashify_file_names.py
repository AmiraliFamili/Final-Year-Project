import json
import re
import time
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import unified_hidden_state_probe_v4_2 as uprobe

# CHANGE: Use a wide console to avoid truncation; adjust if needed.
console = Console(width=140)

root = Path('/Volumes/Amirali/hidden_states')
exp_id = 'baseline_v5_001'
cp_dir = root / 'experiments' / exp_id / 'matrix_checkpoint'
res_dir = cp_dir / 'per_entry_results'
idx_path = cp_dir / 'results_index.csv'
chk_path = cp_dir / 'probe_matrix_checkpoint.json'
analysis_base = root / 'experiments' / exp_id

res_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def is_hash(fname: str) -> bool:
    """Return True if filename stem is a 12+ hex hash."""
    stem = fname.replace('_layer_probe_results.csv', '')
    return len(stem) >= 12 and all(c in '0123456789abcdef' for c in stem[:12])

def extract_model_dataset_from_path(folder_path: Path):
    """
    Extract model and dataset from path using 'models' and 'datasets' markers.
    Works for both nested providers and flat structures.
    """
    parts = folder_path.parts
    try:
        models_idx = parts.index('models')
        datasets_idx = parts.index('datasets', models_idx + 1)
        model_parts = parts[models_idx + 1:datasets_idx]
        model = '/'.join(model_parts)
        dataset = parts[datasets_idx + 1] if datasets_idx + 1 < len(parts) else None
        return model, dataset
    except ValueError:
        return None, None

def extract_hyperparams_from_metadata(folder_path: Path):
    """Try to read hyperparameters from complete_run_metadata.json or probe_run_manifest.json."""
    for meta_name in ('complete_run_metadata.json', 'probe_run_manifest.json'):
        meta_path = folder_path / meta_name
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # Try various locations for repeats/max_samples
            repeats = meta.get('repeats') or meta.get('configuration', {}).get('repeats')
            max_samples = meta.get('max_samples') or meta.get('configuration', {}).get('max_samples')

            probes_list = meta.get('probes') or meta.get('configuration', {}).get('probes')
            probes = None
            if probes_list:
                if isinstance(probes_list, list):
                    probes = '+'.join([p.get('name', '?') for p in probes_list])
                else:
                    probes = str(probes_list)

            trial_hash = meta.get('trial_hash') or meta.get('extra_info', {}).get('trial_hash')

            return {
                'repeats': repeats,
                'max_samples': max_samples,
                'probes': probes,
                'trial_hash': trial_hash,
                'source': 'metadata'
            }
        except Exception:
            continue
    return None

def extract_hyperparams_from_folder_name(folder_name: str):
    """Parse deterministic folder name: probe_run__Model__Dataset__max2000__rep2__probes=...__hash..."""
    if not folder_name.startswith('probe_run__'):
        return None
    parts = folder_name.split('__')
    if len(parts) < 5:
        return None
    info = {'repeats': None, 'max_samples': None, 'probes': None, 'trial_hash': None, 'source': 'folder_name'}
    for part in parts[3:]:
        if part.startswith('max'):
            info['max_samples'] = part[3:]
        elif part.startswith('rep'):
            try:
                info['repeats'] = int(part[3:])
            except:
                pass
        elif part.startswith('probes='):
            info['probes'] = part[7:]
        elif part.startswith('hash'):
            info['trial_hash'] = part[4:16]
    return info

def summarize_results(df: pd.DataFrame) -> dict:
    """Compute summary metrics."""
    if df.empty:
        return {}
    summary = {}
    for probe in df['probe'].unique():
        sub = df[df['probe'] == probe]
        if sub.empty:
            continue
        best_idx = sub['test_macro_f1'].idxmax()
        best_row = sub.loc[best_idx]
        summary[f'best_{probe}'] = {
            'layer': int(best_row['layer_index']),
            'macro_f1': float(best_row['test_macro_f1']),
            'probe_score': float(best_row['probe_score']),
        }
    if 'probe_score' in df.columns and not df.empty:
        overall_best_idx = df['probe_score'].idxmax()
        overall_best = df.loc[overall_best_idx]
        summary['overall_best'] = {
            'probe': overall_best['probe'],
            'layer': int(overall_best['layer_index']),
            'macro_f1': float(overall_best['test_macro_f1']),
            'probe_score': float(overall_best['probe_score']),
        }
    summary['average_per_probe'] = df.groupby('probe')[['test_macro_f1', 'probe_score']].mean().to_dict('index')
    return summary

# ----------------------------------------------------------------------
# Load existing index
# ----------------------------------------------------------------------
existing_rows = {}
if idx_path.exists():
    try:
        df_existing = pd.read_csv(idx_path)
        existing_rows = {row['result_filename']: row for _, row in df_existing.iterrows()}
    except Exception:
        existing_rows = {}
        console.log("[yellow]Could not read existing index, starting fresh.[/yellow]")

# Load checkpoint (for verification later)
checkpoint = {"completed": {}}
if chk_path.exists():
    with open(chk_path) as f:
        checkpoint = json.load(f)

# ----------------------------------------------------------------------
# Collect all result files
# ----------------------------------------------------------------------
csv_files = []
# 1. Legacy files in per_entry_results
for f in res_dir.glob('*_layer_probe_results.csv'):
    if not f.name.startswith('._'):
        csv_files.append(f)
# 2. New matrix run folders
for f in analysis_base.glob('**/analysis/probes/matrix_runs/*/layer_probe_results.csv'):
    csv_files.append(f)

# Deduplicate by full path
csv_files = list({str(f): f for f in csv_files}.values())
csv_files.sort(key=lambda x: str(x))

console.rule("[bold cyan]Migration, Hyperparameter Detection, and Analysis")
console.print(f"[dim]Checkpoint directory: {cp_dir}[/dim]")
console.print(f"Found [bold]{len(csv_files)}[/bold] result files.")

# ----------------------------------------------------------------------
# Process files
# ----------------------------------------------------------------------
file_infos = []
files_processed = 0
files_indexed = 0
files_updated = 0
files_skipped = 0

# Default hyperparameters for legacy files
legacy_default_hp = {
    'repeats': 2,
    'max_samples': 2000,
    'probes': 'linear_logistic+mlp_1_hidden+mlp_2_hidden+mlp_3_hidden'
}

with console.status("[bold green]Scanning files...", spinner="dots"):
    for f in csv_files:
        files_processed += 1
        fname = f.name
        is_legacy = (f.parent == res_dir)

        model = None
        dataset = None
        repeats = None
        max_samples = None
        probes = None
        trial_hash = None
        source = 'unknown'

        if is_legacy:
            # Legacy file: get model/dataset from CSV content
            try:
                df_head = pd.read_csv(f, nrows=1)
                if 'model' in df_head.columns:
                    model = df_head['model'].iloc[0]
                if 'dataset' in df_head.columns:
                    dataset = df_head['dataset'].iloc[0]
            except:
                pass
            if model is None or dataset is None:
                console.log(f"[red]Could not determine model/dataset for legacy file {fname}, skipping.[/red]")
                files_skipped += 1
                continue
            repeats = legacy_default_hp['repeats']
            max_samples = legacy_default_hp['max_samples']
            probes = legacy_default_hp['probes']
            stem = fname.replace('_layer_probe_results.csv', '')
            trial_hash = stem[:12] if is_hash(fname) else None
            source = 'legacy'
        else:
            # New run: use path to get model/dataset reliably
            model, dataset = extract_model_dataset_from_path(f.parent)
            if model is None or dataset is None:
                console.log(f"[red]Could not parse model/dataset from path for {f}, skipping.[/red]")
                files_skipped += 1
                continue

            # Try to get hyperparameters from metadata, then folder name
            hp_meta = extract_hyperparams_from_metadata(f.parent)
            if hp_meta:
                repeats = hp_meta.get('repeats')
                max_samples = hp_meta.get('max_samples')
                probes = hp_meta.get('probes')
                trial_hash = hp_meta.get('trial_hash')
                source = hp_meta.get('source', 'metadata')
            else:
                hp_folder = extract_hyperparams_from_folder_name(f.parent.name)
                if hp_folder:
                    repeats = hp_folder.get('repeats')
                    max_samples = hp_folder.get('max_samples')
                    probes = hp_folder.get('probes')
                    trial_hash = hp_folder.get('trial_hash')
                    source = hp_folder.get('source', 'folder_name')
                else:
                    # Fallback: unknown
                    source = 'path_only'
                    repeats = None
                    max_samples = None
                    probes = None
                    trial_hash = None

            # If trial_hash still None, generate a stable hash
            if trial_hash is None:
                minimal_cfg = {
                    'model': model,
                    'dataset': dataset,
                    'repeats': repeats,
                    'max_samples': max_samples,
                    'probes': probes,
                }
                trial_hash = uprobe.stable_hash(minimal_cfg, 12)

        # Ensure no None values for display
        repeats = repeats if repeats is not None else '?'
        max_samples = max_samples if max_samples is not None else '?'
        probes = probes if probes is not None else '?'
        trial_hash = trial_hash if trial_hash is not None else 'unknown'

        # Update index (only for legacy files)
        if is_legacy:
            row_data = {
                'result_filename': fname,
                'model': model,
                'dataset': dataset,
                'trial_hash': trial_hash,
                'saved_at': time.time(),
            }
            if fname in existing_rows:
                existing_rows[fname] = row_data
                files_updated += 1
            else:
                existing_rows[fname] = row_data
                files_indexed += 1

        # Load full CSV for analysis
        try:
            df_full = pd.read_csv(f)
        except:
            df_full = pd.DataFrame()

        summary = summarize_results(df_full)

        file_infos.append({
            'filename': fname,
            'path': str(f),
            'model': model,
            'dataset': dataset,
            'trial_hash': trial_hash,
            'repeats': repeats,
            'max_samples': max_samples,
            'probes': probes,
            'source': source,
            'size': f.stat().st_size,
            'rows': len(df_full),
            'cols': len(df_full.columns) if not df_full.empty else 0,
            'best_overall_probe': summary.get('overall_best', {}).get('probe', '?'),
            'best_overall_layer': summary.get('overall_best', {}).get('layer', '?'),
            'best_overall_macro_f1': summary.get('overall_best', {}).get('macro_f1', '?'),
            'best_overall_score': summary.get('overall_best', {}).get('probe_score', '?'),
        })

# ----------------------------------------------------------------------
# Save index (only legacy files)
# ----------------------------------------------------------------------
if existing_rows:
    df_index = pd.DataFrame(list(existing_rows.values()))
    df_index = df_index[['result_filename', 'model', 'dataset', 'trial_hash', 'saved_at']]
    df_index = df_index.sort_values(['model', 'dataset']).reset_index(drop=True)
    df_index.to_csv(idx_path, index=False)
    console.print(f"[bold green]Index saved[/bold green] with {len(df_index)} entries.")
else:
    console.print("[yellow]No entries to save.[/yellow]")

# ----------------------------------------------------------------------
# Summary of processing
# ----------------------------------------------------------------------
console.rule("[bold cyan]Processing Summary")
console.print(f"Files processed : {files_processed}")
console.print(f"Files indexed   : {files_indexed}")
console.print(f"Files updated   : {files_updated}")
console.print(f"Files skipped   : {files_skipped}")

# ----------------------------------------------------------------------
# Verification (for new-format checkpoint)
# ----------------------------------------------------------------------
console.rule("[bold cyan]Checkpoint Verification")
if chk_path.exists():
    completed = checkpoint.get('completed', {})
    actual_files = {f.name for f in res_dir.glob('*_layer_probe_results.csv') if not f.name.startswith('._')}
    missing = []
    for key in completed:
        parts = key.split('::')
        if len(parts) == 3:
            trial_hash = parts[-1]
            expected_name = f"{trial_hash}_layer_probe_results.csv"
            if expected_name not in actual_files:
                missing.append((key, expected_name))
    if missing:
        console.print("[red]Missing files for completed trials:[/red]")
        for k, v in missing:
            console.print(f"  [red]{k} → {v}[/red]")
    else:
        console.print("[green]All new-format completed entries have corresponding files.[/green]")
else:
    console.print("[yellow]Checkpoint file not found.[/yellow]")

# ----------------------------------------------------------------------
# Trial Inventory with hyperparameters (full display)
# ----------------------------------------------------------------------
console.rule("[bold cyan]Trial Inventory with Hyperparameters")
if file_infos:
    # CHANGE: Use expand=True, no_wrap=True to prevent truncation
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", no_wrap=True)
    table.add_column("Repeats", justify="center")
    table.add_column("Max Samples", justify="center")
    table.add_column("Probes", no_wrap=False)  # allow wrapping if too long
    table.add_column("Trial Hash", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("Size", justify="right")
    for info in file_infos:
        table.add_row(
            info['model'],
            info['dataset'],
            str(info['repeats']),
            str(info['max_samples']),
            info['probes'],
            info['trial_hash'],
            info['source'],
            str(info['rows']),
            f"{info['size']:,}",
        )
    console.print(table)
else:
    console.print("[yellow]No trial information available.[/yellow]")

# ----------------------------------------------------------------------
# Performance Summary
# ----------------------------------------------------------------------
console.rule("[bold cyan]Performance Summary (Best Overall)")
if file_infos:
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", no_wrap=True)
    table.add_column("Best Probe", no_wrap=True)
    table.add_column("Best Layer", justify="center")
    table.add_column("Macro-F1", justify="right")
    table.add_column("Probe Score", justify="right")
    for info in file_infos:
        mf = info['best_overall_macro_f1']
        ps = info['best_overall_score']
        mf_str = f"{mf:.4f}" if isinstance(mf, float) else str(mf)
        ps_str = f"{ps:.4f}" if isinstance(ps, float) else str(ps)
        table.add_row(
            info['model'],
            info['dataset'],
            info['best_overall_probe'],
            str(info['best_overall_layer']),
            mf_str,
            ps_str,
        )
    console.print(table)
else:
    console.print("[yellow]No performance data available.[/yellow]")

console.rule("[bold cyan]Done")