import json
import re
import time
import shutil
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
import Probe as uprobe   # adjust if module name differs

console = Console(width=140)

root = Path('/Volumes/Amirali/hidden_states')
exp_id = 'baseline_v5_001'
cp_dir = root / 'experiments' / exp_id / 'matrix_checkpoint'
res_dir = cp_dir / 'per_entry_results'
idx_path = cp_dir / 'results_index.csv'
chk_path = cp_dir / 'probe_matrix_checkpoint.json'
analysis_base = root / 'experiments' / exp_id

# ----------------------------------------------------------------------
# NEW: Rename old run directories to new hash
# ----------------------------------------------------------------------
def add_missing_field_to_dataset_contract(dataset_contract):
    """Ensure allow_missing_label_fingerprint is present with value True."""
    if 'allow_missing_label_fingerprint' not in dataset_contract:
        dataset_contract['allow_missing_label_fingerprint'] = True
    return dataset_contract

def rename_old_runs_to_new_hash():
    console.rule("[bold yellow]Renaming old run directories to new hash[/bold yellow]")
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    # Look for run directories that contain complete_run_metadata.json
    run_dirs = []
    for meta_file in analysis_base.glob('**/analysis/probes/**/complete_run_metadata.json'):
        run_dirs.append(meta_file.parent)
    # Also check matrix_runs
    for meta_file in analysis_base.glob('**/analysis/probes/matrix_runs/**/complete_run_metadata.json'):
        run_dirs.append(meta_file.parent)

    # Deduplicate
    run_dirs = list(set(run_dirs))

    for run_dir in run_dirs:
        try:
            meta_path = run_dir / 'complete_run_metadata.json'
            with open(meta_path) as f:
                meta = json.load(f)

            # Get old trial config
            old_trial_config = meta.get('extra_info', {}).get('trial_config')
            if old_trial_config is None:
                # Try to find it in probe_run_manifest.json
                manifest_path = run_dir / 'probe_run_manifest.json'
                if manifest_path.exists():
                    with open(manifest_path) as mf:
                        manifest = json.load(mf)
                    old_trial_config = manifest.get('analysis', {}).get('trial_config')
                if old_trial_config is None:
                    console.log(f"[yellow]No trial_config found in {run_dir}, skipping.[/yellow]")
                    skipped_count += 1
                    continue

            # Modify dataset_contract to include the new field
            if 'dataset_contract' in old_trial_config:
                old_trial_config['dataset_contract'] = add_missing_field_to_dataset_contract(
                    old_trial_config['dataset_contract']
                )
            else:
                console.log(f"[red]No dataset_contract in trial_config for {run_dir}, skipping.[/red]")
                error_count += 1
                continue

            # Recompute new hash
            new_hash = uprobe.stable_hash(old_trial_config, length=12)

            # Generate new directory name
            new_folder_name = uprobe.build_trial_dir_name(old_trial_config, new_hash)

            # Old folder name (current directory name)
            old_folder_name = run_dir.name

            if old_folder_name == new_folder_name:
                console.log(f"[green]Already correct: {run_dir}[/green]")
                continue

            # New path
            new_dir = run_dir.parent / new_folder_name
            if new_dir.exists():
                console.log(f"[yellow]New directory already exists, skipping rename: {new_dir}[/yellow]")
                skipped_count += 1
                continue

            # Perform rename
            shutil.move(str(run_dir), str(new_dir))
            console.log(f"[bold green]Renamed:[/bold green] {run_dir.name} -> {new_folder_name}")
            renamed_count += 1

        except Exception as e:
            console.log(f"[red]Error processing {run_dir}: {e}[/red]")
            error_count += 1

    console.print(f"Renaming summary: {renamed_count} renamed, {skipped_count} skipped, {error_count} errors.\n")

# Run the renaming first
rename_old_runs_to_new_hash()

# ----------------------------------------------------------------------
# Original scanning and analysis continues below
# ----------------------------------------------------------------------

# Helper functions (unchanged from original)
def is_hash(fname: str) -> bool:
    stem = fname.replace('_layer_probe_results.csv', '')
    return len(stem) >= 12 and all(c in '0123456789abcdef' for c in stem[:12])

def extract_model_dataset_from_path(folder_path: Path):
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
    for meta_name in ('complete_run_metadata.json', 'probe_run_manifest.json'):
        meta_path = folder_path / meta_name
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
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

# Load checkpoint
checkpoint = {"completed": {}}
if chk_path.exists():
    with open(chk_path) as f:
        checkpoint = json.load(f)

# ----------------------------------------------------------------------
# Collect all result files (after renaming)
# ----------------------------------------------------------------------
csv_files = []
for f in res_dir.glob('*_layer_probe_results.csv'):
    if not f.name.startswith('._'):
        csv_files.append(f)
for f in analysis_base.glob('**/analysis/probes/matrix_runs/*/layer_probe_results.csv'):
    csv_files.append(f)
for f in analysis_base.glob('**/analysis/probes/**/layer_probe_results.csv'):
    # also catch those in renamed directories, but avoid duplicates
    if f not in csv_files:
        csv_files.append(f)

csv_files = list({str(f): f for f in csv_files}.values())
csv_files.sort(key=lambda x: str(x))

console.rule("[bold cyan]Scanning Result Files after Renaming")
console.print(f"Found [bold]{len(csv_files)}[/bold] result files.")

# The rest of the original script (processing files, saving index, verification, tables) remains unchanged.
# I include the rest here for completeness; it is identical to your original code after the scanning section.

file_infos = []
files_processed = 0
files_indexed = 0
files_updated = 0
files_skipped = 0

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
            model, dataset = extract_model_dataset_from_path(f.parent)
            if model is None or dataset is None:
                console.log(f"[red]Could not parse model/dataset from path for {f}, skipping.[/red]")
                files_skipped += 1
                continue

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
                    source = 'path_only'
                    repeats = None
                    max_samples = None
                    probes = None
                    trial_hash = None

            if trial_hash is None:
                minimal_cfg = {
                    'model': model,
                    'dataset': dataset,
                    'repeats': repeats,
                    'max_samples': max_samples,
                    'probes': probes,
                }
                trial_hash = uprobe.stable_hash(minimal_cfg, 12)

        repeats = repeats if repeats is not None else '?'
        max_samples = max_samples if max_samples is not None else '?'
        probes = probes if probes is not None else '?'
        trial_hash = trial_hash if trial_hash is not None else 'unknown'

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

# Save index
if existing_rows:
    df_index = pd.DataFrame(list(existing_rows.values()))
    df_index = df_index[['result_filename', 'model', 'dataset', 'trial_hash', 'saved_at']]
    df_index = df_index.sort_values(['model', 'dataset']).reset_index(drop=True)
    df_index.to_csv(idx_path, index=False)
    console.print(f"[bold green]Index saved[/bold green] with {len(df_index)} entries.")
else:
    console.print("[yellow]No entries to save.[/yellow]")

# Processing summary
console.rule("[bold cyan]Processing Summary")
console.print(f"Files processed : {files_processed}")
console.print(f"Files indexed   : {files_indexed}")
console.print(f"Files updated   : {files_updated}")
console.print(f"Files skipped   : {files_skipped}")

# Verification
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

# Trial Inventory
console.rule("[bold cyan]Trial Inventory with Hyperparameters")
if file_infos:
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", no_wrap=True)
    table.add_column("Repeats", justify="center")
    table.add_column("Max Samples", justify="center")
    table.add_column("Probes", no_wrap=False)
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

# Performance Summary
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