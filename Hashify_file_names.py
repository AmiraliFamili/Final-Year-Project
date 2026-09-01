import json
import time
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
import unified_hidden_state_probe_v4_2 as uprobe


print("\n")
console = Console()
root = Path('/Volumes/Amirali/hidden_states')
exp_id = 'baseline_v5_001'
cp_dir = root / 'experiments' / exp_id / 'matrix_checkpoint'
res_dir = cp_dir / 'per_entry_results'
idx_path = cp_dir / 'results_index.csv'
chk_path = cp_dir / 'probe_matrix_checkpoint.json'

res_dir.mkdir(parents=True, exist_ok=True)

def is_hash(fname):
    stem = fname.replace('_layer_probe_results.csv', '')
    return len(stem) >= 12 and all(c in '0123456789abcdef' for c in stem[:12])

def parse_old(fname):
    stem = fname.replace('_layer_probe_results.csv', '')
    if 'goEmo' in stem:
        ds = 'goEmo'
        model_part = stem.replace('_goEmo', '')
    elif 'ISEAR' in stem:
        ds = 'ISEAR'
        model_part = stem.replace('_ISEAR', '')
    else:
        return None
    return model_part.replace('_', '/'), ds

# Load existing index
rows = []
if idx_path.exists():
    try:
        rows = pd.read_csv(idx_path).to_dict('records')
    except Exception:
        rows = []
existing = {r.get('result_filename') for r in rows if 'result_filename' in r}

console.rule("[bold cyan]Migration and Verification")
console.print(f"[dim]Checkpoint directory: {cp_dir}[/dim]")

# Migration phase
with console.status("[bold green]Migrating files...", spinner="dots"):
    files = list(res_dir.glob('*_layer_probe_results.csv'))
    for f in files:
        if f.name.startswith('._'):
            continue
        if is_hash(f.name):
            if f.name not in existing:
                try:
                    df = pd.read_csv(f, nrows=1)
                    m = df['model'].iloc[0] if 'model' in df.columns else None
                    d = df['dataset'].iloc[0] if 'dataset' in df.columns else None
                    if m and d:
                        rows.append({'result_filename': f.name, 'model': m, 'dataset': d, 'saved_at': 0.0})
                        existing.add(f.name)
                        console.log(f"Indexed existing [bold]{f.name}[/bold]")
                except Exception:
                    pass
            continue

        parsed = parse_old(f.name)
        if not parsed:
            continue
        m, d = parsed
        key = f"{m}::{d}"
        new_hash = uprobe.stable_hash(key, 12)
        new_name = f"{new_hash}_layer_probe_results.csv"
        new_path = res_dir / new_name

        if new_path.exists():
            if new_name not in existing:
                rows.append({'result_filename': new_name, 'model': m, 'dataset': d, 'saved_at': time.time()})
                existing.add(new_name)
                console.log(f"Indexed [bold]{new_name}[/bold]")
            continue

        try:
            f.rename(new_path)
            rows.append({'result_filename': new_name, 'model': m, 'dataset': d, 'saved_at': time.time()})
            existing.add(new_name)
            console.log(f"Renamed [bold]{f.name}[/bold] → [bold]{new_name}[/bold]")
        except Exception as e:
            console.log(f"[red]Failed {f.name}: {e}[/red]")

if rows:
    df = pd.DataFrame(rows).sort_values(['model', 'dataset']).reset_index(drop=True)
    df.to_csv(idx_path, index=False)
    console.print(f"\n[bold green]Index saved[/bold green] ({len(df)} entries)")
else:
    console.print("[yellow]No files indexed.[/yellow]")

# Verification phase
console.rule("[bold cyan]Verification")
if chk_path.exists():
    with open(chk_path) as fh:
        completed = set(json.load(fh).get('completed', {}).keys())
    actual = {f.name for f in res_dir.glob('*_layer_probe_results.csv') if not f.name.startswith('._')}
    missing = []
    for key in completed:
        try:
            exp_name = f"{uprobe.stable_hash(key, 12)}_layer_probe_results.csv"
            if exp_name not in actual:
                missing.append((key, exp_name))
        except ValueError:
            continue
    if missing:
        console.print("[red]Missing files:[/red]")
        for k, v in missing:
            console.print(f"  [red]{k} → {v}[/red]")
    else:
        console.print("[green]All completed entries have corresponding files.[/green]")
else:
    console.print("[yellow]Checkpoint file not found.[/yellow]")

# CSV file check with a table
console.rule("[bold cyan]CSV File Check")
files = sorted([f for f in res_dir.glob('*_layer_probe_results.csv') if not f.name.startswith('._')],
               key=lambda x: x.name)
if not files:
    console.print("[yellow]No CSV files found.[/yellow]")
else:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Cols", justify="right")
    for f in files:
        try:
            sz = f.stat().st_size
            df = pd.read_csv(f, nrows=5)
            nrows = sum(1 for _ in open(f, encoding='utf-8', errors='ignore')) - 1
            cols = len(df.columns)
            table.add_row(f.name, f"{sz:,}", f"{nrows:,}", str(cols))
        except Exception as e:
            table.add_row(f.name, "ERR", str(e), "")
    console.print(table)

# Index preview
if idx_path.exists() and not pd.read_csv(idx_path).empty:
    console.rule("[bold cyan]Index Preview")
    idx_df = pd.read_csv(idx_path)[['result_filename', 'model', 'dataset']]
    idx_table = Table(show_header=True, header_style="bold magenta")
    for col in idx_df.columns:
        idx_table.add_column(col)
    for _, row in idx_df.iterrows():
        idx_table.add_row(*map(str, row))
    console.print(idx_table)


print("\n\n")
console.rule("[bold cyan]Done")
print("\n\n")
