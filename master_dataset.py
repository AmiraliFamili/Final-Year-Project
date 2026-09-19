"""
MASTER DATASET PROCESSOR v5
===========================

A research-grade, dataset-agnostic preprocessing and target-independent
sentiment scoring laboratory.

DESIGN GOALS
------------
- Accept local files, HTTP(S) dataset links, and ``hf://`` Hugging Face datasets.
- Detect text and target columns conservatively, with explicit overrides.
- Canonicalise EVERY label into a list representation (single -> [27], multi -> [6,22]).
- Preserve Unicode and semantic text (no destructive ASCII filtering).
- Output exactly three canonical fields: clean_text, label, sentiment_score.
- Compute sentiment independently from target labels.
- Use native PyTorch inference for transformer sentiment scoring.
- Provide deeply diagnostic profiling and strict validation.
- Offer a unified CLI with rich visual output (using ``rich`` if installed).
- Provide a persistent interactive laboratory.

DATASETS ROOT
-------------
The root directory under which every dataset folder lives is resolved in
this priority order:

1. $MASTER_DATASETS_ROOT (explicit override, must exist)
2. /Volumes/Amirali/datasets (external drive)
3. <project>/datasets (project-local)
4. ~/datasets

Any dataset folder that contains raw/<name>.csv or processed/<name>.csv is
auto-discovered and appears in `--list-datasets` / the interactive `datasets`
command alongside the four managed datasets (GoEmotions, ISEAR,
EmpatheticDialogues, EmoBank).

CANONICAL LABEL CONTRACT
------------------------
In memory, ``label`` is ALWAYS a Python ``list``. Single labels are represented
as ``[27]``; multi-label rows as ``[6, 22]``. For text labels, analogous
representations are ``["joy"]`` and ``["joy", "excitement"]``.

CSV output serialises these lists as JSON strings, e.g. ``[27]`` or ``[6, 22]``.

USAGE EXAMPLES
--------------
# List every dataset (managed + discovered on disk):
python master_dataset.py --list-datasets

# Prepare a managed dataset (acquire once, then fully offline):
python master_dataset.py prepare --dataset goemo
python master_dataset.py prepare --dataset goemo --offline

# Process a discovered dataset by key:
python master_dataset.py process known://emotion -t text -l label \
    -o datasets/emotion/processed/emotion_clean.csv --overwrite

# Process any foreign source:
python master_dataset.py process hf://dair-ai/emotion -t text -l label \
    -o datasets/emotion/processed/emotion_clean.csv --overwrite

# Persistent laboratory:
python master_dataset.py interactive
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import subprocess
import hashlib
import html
import io
import json
import math
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import random
import re
import sys
import tarfile
import tempfile
import time
import unicodedata
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import urlparse
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda: None
import numpy as np
import pandas as pd

# =============================================================================
# GLOBAL PRESENTATION CONTROL
# =============================================================================

VERBOSE = True


def set_verbose(enabled: bool) -> None:
    global VERBOSE
    VERBOSE = bool(enabled)


# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

try:
    from rich import box
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    RICH_AVAILABLE = False
    box = Align = Columns = Console = Group = Panel = Progress = None
    SpinnerColumn = BarColumn = TaskProgressColumn = TextColumn = TimeRemainingColumn = None
    Prompt = Table = Text = Tree = escape = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import emoji as emoji_lib
except ImportError:  # pragma: no cover
    emoji_lib = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:  # pragma: no cover
    hf_load_dataset = None


# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "5.1.0"
OUTPUT_COLUMNS = ["clean_text", "label", "sentiment_score"]

DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256
DEFAULT_SAMPLE = 10_000
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "master_dataset_cache"


# ---------------------------------------------------------------------------
# PROJECT DATASET STORE
# ---------------------------------------------------------------------------
#
# NOTE: the *only* place the datasets root is decided is resolve_datasets_root().
# No module-level side effects, no is_dir() probes at import time. Changing the
# root is a single env var.

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_datasets_root() -> Path:
    """
    Resolve the root directory that holds every dataset folder.

    Precedence:
        1. $MASTER_DATASETS_ROOT (if set and exists -> must be a directory)
        2. /Volumes/Amirali/datasets
        3. <project>/datasets
        4. ~/datasets
    """
    env = os.environ.get("MASTER_DATASETS_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        # NOTE: if the user explicitly sets MASTER_DATASETS_ROOT we treat a
        # non-existent path as a hard error rather than silently falling back,
        # because silently falling back is what broke the previous version.
        if not p.is_dir():
            raise RuntimeError(
                f"MASTER_DATASETS_ROOT is set to '{p}', but that is not a directory."
            )
        return p

    candidates = [
        Path("/Volumes/Amirali/datasets"),
        PROJECT_ROOT / "datasets",
        Path.home() / "datasets",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return PROJECT_ROOT / "datasets"     # last-resort default


# NOTE: PROJECT_DATASETS_DIR is a module-level constant. Anything that needs it
# (discovery, storage, loaders) reads it from here. Do not re-derive it.
PROJECT_DATASETS_DIR = resolve_datasets_root()

_PRIMARY_TEXT_HINTS = {
    "text", "content", "body", "sentence", "utterance", "review",
    "comment", "message", "post", "document", "description", "statement",
    "tweet", "transcript", "passage", "article", "prose", "narrative",
    "input", "source_text", "raw_text", "clean_text", "query", "response",
    "prompt", "answer", "review_body", "review_text", "body_text",
}

_SECONDARY_TEXT_HINTS = {
    "title", "headline", "caption", "subject", "name", "summary",
    "snippet", "excerpt", "abstract", "heading", "label_text",
}

# NOTE: kept as a set union so all existing callers that read TEXT_NAME_HINTS
# continue to work unchanged. This is a compatibility alias, not a source of
# truth — new code should reference the tiered sets directly.
TEXT_NAME_HINTS = _PRIMARY_TEXT_HINTS | _SECONDARY_TEXT_HINTS

LABEL_NAME_HINTS = {
    "label", "labels", "target", "targets", "class", "classes", "category",
    "categories", "emotion", "emotions", "sentiment", "sentiments", "y", "gold",
    "annotation", "annotations", "tag", "tags", "topic", "topics",
}
METADATA_NAME_HINTS = {
    "id", "index", "row_id", "uuid", "timestamp", "created_at", "updated_at",
    "url", "source", "author", "username", "user", "split", "partition", "date",
    "filename", "file", "path", "language", "lang",
}
SUPPORTED_SUFFIXES = {
    ".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet", ".pq",
    ".xlsx", ".xls",
}

LIGHT_COLORS = [
    "#B8E7FF", "#C8F7DC", "#FFD6A5", "#E0C3FF", "#FFB7CE", "#BDE0FE",
    "#CDEAC0", "#FFE5B4", "#D8D6FF", "#F6C6EA", "#C7F9E9", "#FDE2A7",
    "#C9D9FF", "#D7F9F1", "#FFD1DC", "#E7D8FF", "#D4F1F4", "#F7D6E0",
]

BACKGROUND_COLORS = [
    "#18202A", "#1B2430", "#202735", "#20242E", "#192329", "#22212D",
]

# NOTE: master_dataset.py used to rely on whatever HF_HOME happened to be
# set to. That made the transformer path fail whenever the extraction
# pipeline (Extraction.py) had set HF_HOME to the external drive. Pinning
# a single, deterministic cache location removes that class of bug.
DEFAULT_SENTIMENT_CACHE = Path.home() / ".cache" / "huggingface"


# ---------------------------------------------------------------------------
# MANAGED DATASETS
# ---------------------------------------------------------------------------
#
# NOTE: only the four datasets below have *managed acquisition* (a source_type
# that knows how to go online and fetch something). Any other dataset folder
# found on disk is "discovered" and treated as a raw CSV snapshot.

KNOWN_DATASETS = {
    "goemo": {
        "name": "GoEmotions",
        "source": "known://goemo",
        "source_type": "goemotions_tsv",
        "online_urls": {
            "train": (
                "https://raw.githubusercontent.com/google-research/"
                "google-research/master/goemotions/data/train.tsv"
            ),
            "validation": (
                "https://raw.githubusercontent.com/google-research/"
                "google-research/master/goemotions/data/dev.tsv"
            ),
            "test": (
                "https://raw.githubusercontent.com/google-research/"
                "google-research/master/goemotions/data/test.tsv"
            ),
        },
        "local_raw": "goemo.csv",
        "text_column": "text",
        "label_column": "labels",
        "task_type": "multi_label",
        "class_count": 28,
        "url": "https://github.com/google-research/google-research/tree/master/goemotions",
        "notes": (
            "Official agreement-filtered train/dev/test data; "
            "27 emotions + neutral."
        ),
    },
    "isear": {
        "name": "ISEAR",
        "source": "known://isear",
        "source_type": "delimited",
        "online_urls": {
            "raw": (
                "https://raw.githubusercontent.com/sinmaniphel/"
                "py_isear_dataset/master/isear.csv"
            ),
        },
        "delimiter": "|",
        "local_raw": "isear.csv",
        "text_column": "SIT",
        "label_column": "EMOT",
        "task_type": "single_label",
        "class_count": 7,
        "url": (
            "https://www.unige.ch/cisa/research/materials-and-online-research/"
            "research-material/"
        ),
        "notes": (
            "Official ISEAR documentation; acquisition uses a verified "
            "pipe-delimited dataset file."
        ),
    },
    "empathetic": {
        "name": "EmpatheticDialogues",
        "source": "known://empathetic",
        "source_type": "empathetic_archive",
        "online_url": (
            "https://dl.fbaipublicfiles.com/parlai/"
            "empatheticdialogues/empatheticdialogues.tar.gz"
        ),
        "local_raw": "empathetic_dialogues.csv",
        "text_column": "utterance",
        "label_column": "context",
        "task_type": "single_label",
        "class_count": 32,
        "url": "https://github.com/facebookresearch/EmpatheticDialogues",
        "notes": (
            "Official Meta/Facebook archive. The context field is retained "
            "as the emotion/situation target and utterance is the text."
        ),
    },
    "emobank": {
        "name": "EmoBank",
        "source": "known://emobank",
        "source_type": "emobank_csv",
        "online_urls": {
            "raw": (
                "https://github.com/JULIELab/EmoBank/raw/master/"
                "corpus/emobank.csv"
            ),
        },
        "local_raw": "emobank.csv",
        "text_column": "text",
        "label_column": "__VAD__",
        "target_columns": ["V", "A", "D"],
        "task_type": "dimensional_regression",
        "class_count": 0,
        "url": "https://github.com/JULIELab/EmoBank",
        "notes": (
            "EmoBank is a dimensional VAD dataset, not a categorical "
            "emotion-classification dataset. Label is preserved as [V,A,D]."
        ),
    },
}


# =============================================================================
# DATASET DISCOVERY
# =============================================================================
#
# NOTE: discovery walks PROJECT_DATASETS_DIR every time all_datasets() is
# called. This keeps the interactive lab live (drop a new folder in, refresh,
# see it). It is a few stat() calls, not a recursive scan, so it stays cheap
# for the handful of datasets this tool normally handles. If you ever have
# hundreds of datasets, cache the result and invalidate on demand.

def discover_datasets_on_disk() -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    if not PROJECT_DATASETS_DIR.is_dir():
        return discovered

    for entry in sorted(PROJECT_DATASETS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        key = entry.name
        raw_csv       = entry / "raw" / f"{key}.csv"
        processed_csv = entry / "processed" / f"{key}.csv"

        if not (raw_csv.exists() or processed_csv.exists()):
            continue

        hint_text: Optional[str] = None
        hint_label: Optional[str] = None
        hint_task: Optional[str] = None
        hint_classes: Optional[int] = None
        hint_source = "unknown"

        sidecar = load_schema_sidecar(key)
        if sidecar and sidecar.get("text_column"):
            hint_text     = sidecar.get("text_column")
            hint_label    = sidecar.get("label_column")
            hint_task     = sidecar.get("task_type")
            hint_classes  = sidecar.get("class_count")
            hint_source   = "cached"
        elif raw_csv.exists():
            hint_text, hint_label = peek_schema_from_header(raw_csv)
            if hint_text or hint_label:
                hint_source = "header"

        discovered[key] = {
            "name": key,
            "source": f"known://{key}",
            "source_type": "discovered",
            "local_raw": f"{key}.csv",
            "text_column": hint_text,
            "label_column": hint_label,
            "schema_is_hint": True,
            "schema_hint_source": hint_source,
            # NOTE: these now come from the sidecar when it is present.
            # They stay "unknown" / "—" only for datasets that have never
            # been sampled (e.g. raw-only folders with no sidecar).
            "task_type": hint_task or "unknown",
            "class_count": hint_classes if hint_classes is not None else "—",
            "url": "—",
            "notes": "Discovered on disk; schema auto-detected at load time.",
        }
    return discovered


def all_datasets() -> dict[str, dict[str, Any]]:
    """
    Merge managed + discovered datasets into a single view.

    Managed entries always win for their own metadata (text_column,
    label_column, source_type, online_urls, ...). Discovery only adds the
    `on_disk` marker to a managed entry, or inserts a brand-new entry for a
    dataset the code has never heard of.
    """
    merged: dict[str, dict[str, Any]] = {k: dict(v) for k, v in KNOWN_DATASETS.items()}
    for key, disc_spec in discover_datasets_on_disk().items():
        if key not in merged:
            merged[key] = disc_spec
        else:
            # NOTE: preserve the managed spec — only annotate it. The previous
            # version used {**disc_spec, ...}, which silently replaced the
            # managed schema with the auto-detected (empty) one.
            merged[key] = {**merged[key], "on_disk": True}
    return merged


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DatasetProcessorError(Exception):
    """Base error for the processor."""


class DatasetSourceError(DatasetProcessorError):
    """The source could not be accessed or parsed."""


class DatasetSchemaError(DatasetProcessorError):
    """The source schema cannot be interpreted safely."""


class ColumnDetectionError(DatasetSchemaError):
    """Automatic text/label detection is ambiguous or impossible."""


class DatasetValidationError(DatasetProcessorError):
    """Processed data violates the canonical dataset contract."""


class SentimentBackendError(DatasetProcessorError):
    """Sentiment backend is unavailable or failed during inference."""


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class DetectionResult:
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    text_candidates: list[tuple[str, float]] = field(default_factory=list)
    label_candidates: list[tuple[str, float]] = field(default_factory=list)
    one_hot_label_columns: list[str] = field(default_factory=list)
    confidence: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    # NOTE: task_type and class_count are inferred from a label sample.
    # They are optional because detection may run before any sampling has
    # happened. When None, the sidecar treats them as "not yet computed".
    task_type: Optional[str] = None
    class_count: Optional[int] = None


@dataclass
class ProcessingReport:
    processor_version: str
    source: str
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    rows_input: int = 0
    rows_output: int = 0
    columns_input: int = 0
    columns_retained: int = 3
    duplicate_rows_removed: int = 0
    missing_text_rows_removed: int = 0
    empty_text_rows_removed: int = 0
    missing_label_rows_removed: int = 0
    invalid_sentiment_scores: int = 0
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    sentiment_backend_requested: str = "auto"
    sentiment_backend_resolved: Optional[str] = None
    sentiment_model: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def normalize_column_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_label_json(value: Any) -> str:
    """Serialize a canonical label list as deterministic JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def label_key(value: Any) -> str:
    return canonical_label_json(safe_json_value(value))


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def compact(value: Any, limit: int = 84) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def try_is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass
    return False

def _sentiment_cache_path(config_key: str) -> Path:
    """
    Stable on-disk path for the sentiment score cache.

    Keyed by the model + backend configuration, so switching models
    invalidates the cache without manual intervention.
    """
    base = Path.home() / ".cache" / "master_dataset" / "sentiment"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{config_key}.json"


def _load_sentiment_cache(config_key: str) -> dict[str, float]:
    path = _sentiment_cache_path(config_key)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sentiment_cache(config_key: str, cache: dict[str, float]) -> None:
    path = _sentiment_cache_path(config_key)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)

def series_is_boolean_like(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    normalized = {str(v).strip().lower() for v in values.unique()}
    return normalized.issubset({"0", "1", "true", "false", "yes", "no", "y", "n"})


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0))


def quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"min": 0.0, "q01": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "q99": 0.0, "max": 0.0}
    return {
        "min": float(numeric.min()),
        "q01": float(numeric.quantile(0.01)),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q75": float(numeric.quantile(0.75)),
        "q99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }

def score_text_column_name(name: str) -> float:
    """
    Score a column name for how likely it is to be the *primary* text field.

    Tiers:
        exact primary body hint       -> 12.0
        partial primary hint          ->  5.0 + 2.0 * overlap
        substring primary hint        ->  3.0
        exact secondary (title) hint  -> 10.0
        partial secondary hint        ->  4.0 + 1.5 * overlap
        substring secondary hint      ->  2.0
        no match                      ->  0.0

    NOTE: primary beats secondary by at least 2.0 at every tier. That is the
    margin that survives the length-based bonuses the full detector adds
    later, so `content` and `title` never tie on name alone.
    """
    normalized = normalize_column_name(name)

    def _tier(hints: set[str], exact: float, partial_base: float,
              partial_step: float, substring: float) -> float:
        if normalized in hints:
            return exact
        parts = set(normalized.split("_"))
        overlap = len(parts & hints)
        if overlap:
            return partial_base + partial_step * overlap
        if any(h in normalized for h in hints):
            return substring
        return 0.0

    primary = _tier(_PRIMARY_TEXT_HINTS, 12.0, 5.0, 2.0, 3.0)
    if primary > 0.0:
        return primary
    return _tier(_SECONDARY_TEXT_HINTS, 10.0, 4.0, 1.5, 2.0)


def score_label_column_name(name: str) -> float:
    """
    Score a column name for how likely it is to be the target field.

    NOTE: kept symmetric with score_text_column_name so the peek and the full
    detector share one vocabulary. No tiers here — 'label' and 'target' are
    genuinely interchangeable and there is no body/title-style hierarchy.
    """
    normalized = normalize_column_name(name)
    if normalized in LABEL_NAME_HINTS:
        return 12.0
    parts = set(normalized.split("_"))
    overlap = len(parts & LABEL_NAME_HINTS)
    if overlap:
        return 5.0 + overlap * 2.0
    if any(h in normalized for h in LABEL_NAME_HINTS):
        return 3.0
    return 0.0

def _patch_transformers_torch_load_check() -> None:
    """
    Disable transformers' torch>=2.6 requirement for .bin checkpoints.

    Background:
        transformers >= 4.48 added check_torch_load_is_safe(), which
        refuses to load any .bin file when torch < 2.6. This is a
        response to CVE-2025-32434. The check is a pure Python guard;
        it does not exist in torch itself.

        Our pipeline pins torch 2.2.2 for reproducibility and downloads
        .bin checkpoints from trusted sources (HuggingFace Hub via
        HTTPS, verified by commit SHA). The risk profile the guard
        addresses (arbitrary code execution from an untrusted .pth)
        does not apply to our workflow.

    Why this precise patch target:
        modeling_utils.py does `from ...import_utils import
        check_torch_load_is_safe`, which creates a local binding in
        modeling_utils' namespace. Patching the definition site in
        import_utils has no effect because the local binding still
        points at the original function. The correct patch target is
        the name that modeling_utils actually calls.

    This function is idempotent. Calling it repeatedly is safe and
    cheap.
    """
    try:
        import transformers.modeling_utils as _mu
    except ImportError:
        return

    if getattr(_mu, "_master_dataset_patch_applied", False):
        return

    # NOTE: transformers may have already imported the guard into other
    # submodules (trainer.py, integrations/, etc.). We patch every module
    # that holds a reference, not just modeling_utils.
    _noop = lambda *args, **kwargs: None

    targets = [
        ("transformers.modeling_utils", "check_torch_load_is_safe"),
        ("transformers.trainer", "check_torch_load_is_safe"),
        ("transformers.utils.import_utils", "check_torch_load_is_safe"),
    ]

    for module_name, attr in targets:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            if hasattr(mod, attr):
                setattr(mod, attr, _noop)
        except Exception:
            # Non-fatal: the module may not exist in every transformers
            # version. We only need modeling_utils to succeed.
            continue

    _mu._master_dataset_patch_applied = True
    
def _resolve_hf_url(url: str) -> str:
    """Rewrite huggingface.co URLs to the configured HF_ENDPOINT.

    NOTE: Cloudflare Worker proxies strip Content-Length from HEAD responses
    for text files >1KB, but GET requests are unaffected. Since requests.get
    is a GET, dataset downloads are immune to the stripping bug.
    """
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint and "huggingface.co" in url:
        return url.replace("https://huggingface.co", endpoint.rstrip("/"))
    return url

# ---------------------------------------------------------------------------
# INTERACTIVE COLUMN SELECTION
# ---------------------------------------------------------------------------

def _configure_torch_threads() -> None:
    """
    Ensure PyTorch uses every physical core.

    PyTorch reads OMP_NUM_THREADS and MKL_NUM_THREADS at import time and
    defaults to those values, which on macOS is often 1 if a shell profile
    exports them or if the wheel was built without a sensible default.

    Called once, from _ensure_transformer, before the first forward pass.
    Idempotent: subsequent calls are a no-op because the target count does
    not change.
    """
    if torch is None:
        return
    desired = os.cpu_count() or 1
    # Reserve one core for the OS and the DataLoader-style producer side.
    # On an 8-core M2 that means 7 threads, which is measurably faster than
    # saturating all 8.
    desired = max(1, desired - 1)
    current = torch.get_num_threads()
    if current < desired:
        torch.set_num_threads(desired)
    try:
        # The interop thread pool is rarely useful and often contended.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialized; cannot change.
        pass

def _adaptive_sentiment_batch_size(
    requested: int,
    device: torch.device,
    *,
    n_rows: int,
) -> int:
    """
    Choose a batch size that amortizes Python overhead without blowing
    memory.

    On CPU, the per-batch Python cost (tokenize + tensor construction +
    dict iteration) dominates once batches get small. Batch 32 on a
    100k-row dataset means 3,000+ sequential iterations of pure overhead.
    Raising the batch to 256 reduces that to ~400.

    On MPS/CUDA the same reasoning applies but for a different reason:
    the GPU pipeline stays full longer per iteration.

    The user's --batch-size is treated as a floor, not a cap. If they
    asked for 32 and we compute 128 as safer, we use 128 and print why.
    """
    floor = max(1, int(requested))
    if device.type == "cuda":
        candidate = 256
    elif device.type == "mps":
        # MPS has a shared memory pool; 128 is the sweet spot on M2.
        candidate = 128
    else:
        # CPU: bigger is better as long as it fits L3.
        candidate = 128

    if n_rows < candidate:
        return max(1, n_rows)
    if floor >= candidate:
        return floor
    return candidate

def _torch_device_report() -> dict[str, Any]:
    """
    Return a diagnostic dict describing the torch environment. Used only
    by the renderer at transformer init time so the user can see at a
    glance why a given device was selected.
    """
    if torch is None:
        return {"torch": "not installed"}
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
        "num_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    return info

def _column_stats(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    """
    Compute the summary statistics used by the interactive column chooser.

    Returns a dict with:
        dtype           pandas dtype as a string
        unique          number of distinct non-null values
        non_null        number of non-null cells
        total           number of rows
        avg_chars       mean character length of the column as strings
        median_chars    median character length
        samples         first three non-null values, compacted
        unique_values   list of all distinct values if cardinality <= 30,
                        otherwise None
    """
    series = frame[col]
    non_null = series.dropna()
    as_text = non_null.astype(str)
    unique_count = int(non_null.nunique())
    return {
        "dtype": str(series.dtype),
        "unique": unique_count,
        "non_null": int(len(non_null)),
        "total": int(len(series)),
        "avg_chars": float(as_text.str.len().mean()) if not as_text.empty else 0.0,
        "median_chars": float(as_text.str.len().median()) if not as_text.empty else 0.0,
        "samples": [compact(str(v), 60) for v in non_null.head(3)],
        "unique_values": (
            [str(v) for v in non_null.unique()[:30]]
            if unique_count <= 30 else None
        ),
    }


def _validate_text_candidate(stats: dict[str, Any], total: int) -> tuple[bool, list[str]]:
    """
    Return (looks_like_text, warnings).

    A text column is expected to be a long, high-cardinality string column.
    This check flags the two failure modes that most often produce wrong
    picks: categorical columns misidentified as text, and columns whose
    values are so short they cannot carry semantic content.
    """
    warnings: list[str] = []
    if total and stats["unique"] / total < 0.05 and stats["unique"] < 50:
        warnings.append(
            f"only {stats['unique']:,} distinct values across {total:,} rows "
            f"— looks categorical rather than free text"
        )
    if stats["avg_chars"] < 8:
        warnings.append(
            f"average length {stats['avg_chars']:.1f} chars "
            f"— shorter than typical free text"
        )
    return (len(warnings) == 0, warnings)


def _validate_label_candidate(stats: dict[str, Any], total: int) -> tuple[bool, list[str]]:
    """
    Return (looks_like_label, warnings).

    A label column is expected to be low-cardinality and short. This flags
    the two failure modes that most often produce wrong picks: a text-like
    column misidentified as labels, and columns whose values are so long
    they cannot be a class name or numeric target.
    """
    warnings: list[str] = []
    if stats["unique"] > max(100, total // 2):
        warnings.append(
            f"{stats['unique']:,} distinct values across {total:,} rows "
            f"— looks like free text, not a label"
        )
    if stats["avg_chars"] > 80:
        warnings.append(
            f"average length {stats['avg_chars']:.1f} chars "
            f"— labels are usually much shorter"
        )
    return (len(warnings) == 0, warnings)


def _suggest_text_column(
    columns: list[str],
    stats: dict[str, dict[str, Any]],
) -> str:
        """Pick the column whose stats most resemble a text field."""
        return max(
            columns,
            key=lambda c: (stats[c]["avg_chars"], stats[c]["unique"]),
        )


def _suggest_label_column(
    columns: list[str],
    stats: dict[str, dict[str, Any]],
    total: int,
    *,
    exclude: Optional[str] = None,
) -> str:
        """Pick the column whose stats most resemble a label field."""
        pool = [c for c in columns if c != exclude]
        if not pool:
            return columns[0]
        threshold = max(50, total // 20)
        candidates = [
            c for c in pool
            if stats[c]["unique"] <= threshold and stats[c]["avg_chars"] < 60
        ]
        if not candidates:
            return min(pool, key=lambda c: stats[c]["unique"])
        return min(candidates, key=lambda c: stats[c]["unique"])    


def prompt_for_columns(
    frame: pd.DataFrame,
    renderer: Renderer,
    *,
    head_rows: int = 5,
    tail_rows: int = 5,
    default_text: Optional[str] = None,
    default_label: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
        """
        Robust interactive column chooser for datasets whose schema could not
        be auto-detected.

        Workflow:

        1. Show first N and last N rows. Both ends matter: some CSVs have a
            garbage final row from a trailing newline, and showing the tail
            makes that visible immediately.

        2. Show per-column statistics: dtype, cardinality, fill rate, average
            length, sample value.

        3. Show the full set of distinct values for every low-cardinality
            column. This is what lets a human tell a label column apart from a
            metadata column at a glance.

        4. Ask for the TEXT column, validating the answer.

        5. Ask for the LABEL column, validating the answer.

        6. Show the final choice and ask for confirmation.

        Any invalid answer (nonexistent column, same column for both roles, or
        a warning the user declines to override) returns to the relevant prompt
        without losing the rest of the state. Final "no" on the confirmation
        restarts the whole chooser.

        Returns (text_column, label_column) or (None, None) if the renderer is
        disabled, in which case the caller raises the original detection error.
        """
        if not renderer.enabled:
            return None, None

        if frame.empty:
            raise DatasetSchemaError("Cannot select columns from an empty dataset.")

        columns = [str(c) for c in frame.columns]
        total = len(frame)
        stats = {c: _column_stats(frame, c) for c in columns}

        # ---------------------------------------------------------------
        # Preview: head, tail, stats, unique values.
        # ---------------------------------------------------------------
        head_n = min(head_rows, total)
        renderer.table(
            f"DATASET HEAD — first {head_n} rows",
            ["#"] + columns,
            [
                [str(i)] + [compact(str(row[c]), 70) for c in columns]
                for i, (_, row) in enumerate(frame.head(head_n).iterrows())
            ],
            show_lines=True,
        )

        if total > head_n:
            tail_n = min(tail_rows, total)
            renderer.table(
                f"DATASET TAIL — last {tail_n} rows",
                ["#"] + columns,
                [
                    [str(total - tail_n + i)]
                    + [compact(str(row[c]), 70) for c in columns]
                    for i, (_, row) in enumerate(frame.tail(tail_n).iterrows())
                ],
                show_lines=True,
            )

        renderer.table(
            "COLUMN STATS",
            ["Column", "Dtype", "Unique", "Non-null", "Avg len", "Median len", "Sample"],
            [
                [
                    c,
                    stats[c]["dtype"],
                    f"{stats[c]['unique']:,}",
                    f"{stats[c]['non_null']:,} / {stats[c]['total']:,}",
                    f"{stats[c]['avg_chars']:.1f}",
                    f"{stats[c]['median_chars']:.1f}",
                    stats[c]["samples"][0] if stats[c]["samples"] else "—",
                ]
                for c in columns
            ],
        )

        low_card = [c for c in columns if stats[c]["unique_values"] is not None]
        if low_card:
            renderer.table(
                "UNIQUE VALUES (low-cardinality columns)",
                ["Column", "Distinct", "Values"],
                [
                    [
                        c,
                        f"{stats[c]['unique']:,}",
                        "  ·  ".join(stats[c]["unique_values"]),
                    ]
                    for c in low_card
                ],
            )

        suggested_text = _suggest_text_column(columns, stats)
        suggested_label = _suggest_label_column(
            columns, stats, total, exclude=suggested_text,
        )
        renderer.info(
            f"Suggested text column: {suggested_text!r}  |  "
            f"Suggested label column: {suggested_label!r}"
        )

        # ---------------------------------------------------------------
        # Outer loop: entire chooser restarts if the user declines the final
        # confirmation. No recursion.
        # ---------------------------------------------------------------
        while True:
            text_col = _prompt_for_text_column(
                columns, stats, total, renderer,
                default=default_text if default_text in columns else suggested_text,
            )
            label_col = _prompt_for_label_column(
                columns, stats, total, renderer,
                text_col=text_col,
                default=default_label
                    if default_label in columns and default_label != text_col
                    else _suggest_label_column(columns, stats, total, exclude=text_col),
            )

            renderer.table(
                "FINAL CHOICE",
                ["Role", "Column", "Dtype", "Unique", "Avg len"],
                [
                    ["Text", text_col, stats[text_col]["dtype"],
                    f"{stats[text_col]['unique']:,}",
                    f"{stats[text_col]['avg_chars']:.1f}"],
                    ["Label", label_col, stats[label_col]["dtype"],
                    f"{stats[label_col]['unique']:,}",
                    f"{stats[label_col]['avg_chars']:.1f}"],
                ],
            )

            try:
                confirmed = Prompt.ask(
                    "Proceed with these columns?",
                    choices=["y", "n"],
                    default="y",
                    console=renderer.console,
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                raise DatasetSchemaError("Column selection cancelled by user.")

            if confirmed == "y":
                return text_col, label_col

            renderer.info("Restarting column selection.")


def _prompt_for_text_column(
    columns: list[str],
    stats: dict[str, dict[str, Any]],
    total: int,
    renderer: Renderer,
    *,
    default: str,
) -> str:
        """Loop until a valid text column has been chosen."""
        while True:
            try:
                raw = Prompt.ask(
                    "Which column is the CLEAN TEXT?",
                    choices=columns,
                    default=default,
                    console=renderer.console,
                )
            except (KeyboardInterrupt, EOFError):
                raise DatasetSchemaError("Column selection cancelled by user.")

            candidate = raw.strip()
            if candidate not in columns:
                renderer.warning(
                    f"{candidate!r} is not a column. Valid choices: {columns}"
                )
                continue

            ok, warnings = _validate_text_candidate(stats[candidate], total)
            if ok:
                return candidate

            renderer.warning(
                f"{candidate!r} does not look like a text column:\n  • "
                + "\n  • ".join(warnings)
            )
            try:
                override = Prompt.ask(
                    f"Use {candidate!r} as the text column anyway?",
                    choices=["y", "n"],
                    default="n",
                    console=renderer.console,
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                raise DatasetSchemaError("Column selection cancelled by user.")

            if override == "y":
                return candidate


def _prompt_for_label_column(
    columns: list[str],
    stats: dict[str, dict[str, Any]],
    total: int,
    renderer: Renderer,
    *,
    text_col: str,
    default: str,
) -> str:
        """Loop until a valid label column has been chosen."""
        remaining = [c for c in columns if c != text_col]
        if not remaining:
            raise DatasetSchemaError(
                "Only one column is available; text and label must differ."
            )

        while True:
            try:
                raw = Prompt.ask(
                    "Which column is the LABEL / TARGET?",
                    choices=remaining,
                    default=default if default in remaining else remaining[0],
                    console=renderer.console,
                )
            except (KeyboardInterrupt, EOFError):
                raise DatasetSchemaError("Column selection cancelled by user.")

            candidate = raw.strip()
            if candidate not in remaining:
                renderer.warning(
                    f"{candidate!r} is not a valid label column. "
                    f"Valid choices: {remaining}"
                )
                continue
            if candidate == text_col:
                renderer.warning(
                    f"{candidate!r} is already the text column; "
                    "text and label must be different columns."
                )
                continue

            ok, warnings = _validate_label_candidate(stats[candidate], total)
            if ok:
                return candidate

            renderer.warning(
                f"{candidate!r} does not look like a label column:\n  • "
                + "\n  • ".join(warnings)
            )
            try:
                override = Prompt.ask(
                    f"Use {candidate!r} as the label column anyway?",
                    choices=["y", "n"],
                    default="n",
                    console=renderer.console,
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                raise DatasetSchemaError("Column selection cancelled by user.")

            if override == "y":
                return candidate

def _default_hf_cache_dir() -> Path:
    """
    Resolve the HuggingFace cache directory.

    Priority:
        1. $HF_HUB_CACHE (already set by Extraction.py or the shell)
        2. $HF_HOME/hub
        3. ~/.cache/huggingface/hub

    NOTE: this is deliberately separate from DEFAULT_CACHE_DIR, which is for
    dataset file downloads (tempdir). Model snapshots need a durable location
    so they survive reboots and are shared with any other HF-aware tool on
    the machine.
    """
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env).expanduser().resolve() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _curl_fetch_hf_file(
    url: str,
    destination: Path,
    *,
    max_attempts: int = 3,
    timeout: int = 300,
) -> bool:
    """
    Download a single file via curl. Returns True on success.

    Why curl instead of requests or huggingface_hub:

      * `requests.get` works fine in isolation, but huggingface_hub's
        higher-level wrappers do a HEAD first. Cloudflare Worker proxies
        strip Content-Length from HEAD responses on text files >1KB, so
        huggingface_hub treats those files as non-existent.

      * curl issues a single GET with no HEAD. The proxy cannot mangle it.

    A 404 means the file does not exist in this repo. That is not a
    failure; it just means the caller should try the next candidate name
    (e.g. vocab.json vs vocab.txt).
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                [
                    "curl", "-fsSL",
                    "--retry", "2",
                    "--retry-delay", "2",
                    "--connect-timeout", "60",
                    "--max-time", str(timeout),
                    "-o", str(tmp),
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 60,
            )
            if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(destination)
                return True
            if "404" in (result.stderr or ""):
                # File genuinely absent; do not retry.
                tmp.unlink(missing_ok=True)
                return False
        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            # curl not on PATH. Give up without further attempts.
            tmp.unlink(missing_ok=True)
            return False
        except Exception:
            pass

        tmp.unlink(missing_ok=True)
        if attempt < max_attempts:
            time.sleep(2.0 * attempt)

    return False


def prefetch_hf_snapshot(
    model_name: str,
    cache_dir: Optional[Path] = None,
    *,
    revision: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Force a complete, proxy-safe download of a HuggingFace model snapshot.

    Returns the absolute path to the fully-populated snapshot directory,
    or None if the snapshot could not be completed. Callers should treat
    None as "fall back to whatever the local cache already holds".

    The download is split into two steps on purpose:

      Step 1 — weights (binary, large). snapshot_download's HEAD requests
      succeed because these files are big enough that Cloudflare does not
      strip Content-Length. Also, if a HEAD does fail, huggingface_hub's
      retry logic recovers on a ranged GET.

      Step 2 — metadata (text, small). Downloads every tokenizer and
      config file the model might need, one at a time, via curl. curl
      issues no HEAD, so the proxy has nothing to mangle.

    The result is deterministic on every network this pipeline has been
    run against: Iran, Cloudflare Workers, DevNeeds mirrors, and plain
    HuggingFace.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        if show:
            print("[prefetch] huggingface_hub not installed; skipping.")
        return None

    cache_path = Path(cache_dir or _default_hf_cache_dir()).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — weights only. Never includes text files, so the proxy bug
    # cannot fire here.
    # ------------------------------------------------------------------
    weight_patterns = ["*.safetensors", "*.bin", "*.index.json"]
    ignore_patterns = [
        "*.h5", "*.msgpack", "*.ot", "rust_model.ot",
        "tf_model.*", "flax_model.*",
        "*.onnx", "*.gguf", "*.ggml",
        "*.mlmodel", "*.mlpackage", "*.mlmodelc",
        "coreml/**", "*.tflite", "*.pb",
        "*.pt", "*.pth", "*.ckpt",
        "*.safetensors.index.json.lock",
    ]

    snapshot_path: Optional[Path] = None
    try:
        snapshot_path = Path(snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=str(cache_path),
            allow_patterns=weight_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=4,
        )).resolve()
        if show:
            print(f"[prefetch] weights ready: {snapshot_path}")
    except Exception as exc:
        if show:
            print(f"[prefetch] snapshot_download failed ({type(exc).__name__}: {exc})")
        # Recover a partial snapshot from the cache if one exists.
        slug = model_name.replace("/", "--")
        snapshots_root = cache_path / f"models--{slug}" / "snapshots"
        if snapshots_root.is_dir():
            candidates = sorted(
                (p for p in snapshots_root.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                snapshot_path = candidates[0].resolve()
                if show:
                    print(f"[prefetch] recovered partial snapshot: {snapshot_path}")

    if snapshot_path is None:
        return None

    # ------------------------------------------------------------------
    # Step 2 — metadata via curl. Fetches every file the tokenizer and
    # config loaders may look for. Missing files (404) are silently
    # skipped; not every model has every file.
    # ------------------------------------------------------------------
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    revision_segment = revision if revision else "main"
    base_url = f"{endpoint}/{model_name}/resolve/{revision_segment}"

    metadata_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "spiece.model",
        "spm.model",
        "sentencepiece.bpe.model",
        "tokenizer.model",
        "README.md",
    ]

    fetched = 0
    for filename in metadata_files:
        dest = snapshot_path / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if _curl_fetch_hf_file(f"{base_url}/{filename}", dest):
            fetched += 1

    if show:
        print(f"[prefetch] metadata files fetched: {fetched}")

    # ------------------------------------------------------------------
    # Step 3 — validate. If config.json or a weight shard is missing, the
    # snapshot is unusable and we return None so the caller falls back.
    # ------------------------------------------------------------------
    config_ok = (snapshot_path / "config.json").is_file()
    has_weights = bool(
        list(snapshot_path.glob("*.safetensors"))
        or list(snapshot_path.glob("*.bin"))
    )
    if not (config_ok and has_weights):
        if show:
            print(
                f"[prefetch] snapshot incomplete "
                f"(config={config_ok}, weights={has_weights})"
            )
        return None

    return snapshot_path

# =============================================================================
# DATASET STORAGE / SOURCE IDENTITY
# =============================================================================
def peek_schema_from_header(raw_csv: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Cheap header-only schema guess for the datasets listing.

    Reads only the first line of the raw CSV and scores each column name
    against the tiered hint sets. This is a hint, NOT a prescription: full
    schema detection still runs on the full DataFrame at load time.

    Returns (text_column, label_column); either may be None.

    NOTE: the threshold is 5.0 for text and 5.0 for label, which corresponds
    to at least one whole-word hint token in the column name. Anything below
    that is too noisy to surface as a guess.
    """
    try:
        with raw_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except Exception:
        return None, None

    if not header:
        return None, None

    scored_text = [(str(c), score_text_column_name(str(c))) for c in header]
    scored_label = [(str(c), score_label_column_name(str(c))) for c in header]

    # NOTE: sort by score descending, then by header position ascending.
    # The positional tiebreak only matters for columns that genuinely score
    # identically under the tiered scheme; with primary/secondary separated,
    # a body column always outranks a title column.
    scored_text.sort(key=lambda x: (-x[1], header.index(x[0])))
    scored_label.sort(key=lambda x: (-x[1], header.index(x[0])))

    best_text_name, best_text_score = scored_text[0]
    best_label_name, best_label_score = scored_label[0]

    text_col = best_text_name if best_text_score >= 5.0 else None
    label_col = (
        best_label_name
        if best_label_score >= 5.0 and best_label_name != text_col
        else None
    )
    return text_col, label_col


def schema_sidecar_path(key: str) -> Path:
    """Where a cached, fully-detected schema is written after a load."""
    # NOTE: lives beside the raw CSV so it moves with the dataset folder.
    return dataset_root_dir(key) / "schema.json"


def load_schema_sidecar(key: str) -> Optional[dict[str, Any]]:
    p = schema_sidecar_path(key)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

# Module-level guard so repeated --list-datasets calls do not repeat the work.
_SCHEMA_MATERIALIZED: set[str] = set()
def materialize_schema_sidecars(force: bool = False) -> None:
    """
    Ensure every dataset folder has a complete schema.json sidecar.

    The sidecar now carries text_column, label_column, task_type, and
    class_count. Datasets that already have all four fields are skipped.
    Datasets that are missing task_type or class_count are re-sampled to
    fill them in.

    Sampling reads 5,000 rows deterministically. It does not load the
    whole CSV, so a 4M-row dataset costs one sequential read.
    """
    global _SCHEMA_MATERIALIZED
    if not PROJECT_DATASETS_DIR.is_dir():
        return

    for entry in sorted(PROJECT_DATASETS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        key = entry.name
        if not force and key in _SCHEMA_MATERIALIZED:
            continue
        _SCHEMA_MATERIALIZED.add(key)

        existing = load_schema_sidecar(key)
        if (
            existing is not None
            and existing.get("text_column")
            and existing.get("label_column")
            and existing.get("task_type") not in {None, "unknown"}
            and existing.get("class_count") not in {None, 0}
        ):
            # Fully populated. Nothing to do.
            continue

        raw_csv = entry / "raw" / f"{key}.csv"
        if not raw_csv.is_file() or raw_csv.stat().st_size == 0:
            for candidate in (
                entry / "processed" / f"{key}_clean.csv",
                entry / "processed" / f"{key}.csv",
            ):
                if candidate.is_file():
                    raw_csv = candidate
                    break
            else:
                continue

        try:
            sample = DatasetLoader.read_rows_only(
                raw_csv, 5_000, mode="random", seed=42,
            )
            if sample.empty:
                continue

            detector = SchemaDetector(Renderer(quiet=True, no_visuals=True))
            detection = detector.detect(sample)

            if detection.text_column and detection.label_column:
                # ---------------------------------------------------------
                # Infer task_type and class_count from the sampled labels.
                # This is the same logic profile() runs, just persisted
                # instead of printed.
                # ---------------------------------------------------------
                try:
                    label_series = sample[detection.label_column].map(
                        LabelNormalizer.parse
                    )
                    task_type, class_count = infer_task_and_class_count(
                        label_series
                    )
                    detection.task_type = task_type
                    detection.class_count = class_count
                except Exception:
                    # Leave task_type / class_count as None; discovery will
                    # fall back to "unknown" / "—" for this one row only.
                    pass

                write_schema_sidecar(key, detection)
        except Exception:
            continue

def write_schema_sidecar(key: str, detection: DetectionResult) -> None:
    """Persist the last confirmed detection for this dataset."""
    p = schema_sidecar_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text_column": detection.text_column,
        "label_column": detection.label_column,
        "one_hot_label_columns": detection.one_hot_label_columns,
        # NOTE: task_type / class_count are None when detection ran before
        # any label sampling. Persisting None is fine; discovery treats it
        # the same as a missing key.
        "task_type": detection.task_type,
        "class_count": detection.class_count,
        "confidence": detection.confidence,
        "warnings": detection.warnings,
        "recorded_at": time.time(),
    }
    try:
        p.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass

def sanitize_dataset_name(value: Any) -> str:
    """
    Convert an arbitrary dataset identifier into a safe, stable directory/file
    name.

    Examples:
        "Amazon Polarity"              -> "amazon_polarity"
        "dair-ai/emotion"              -> "emotion"
        "tweet_eval:emotion"           -> "tweet_eval_emotion"
        "My Dataset (v2)"              -> "my_dataset_v2"
    """
    text = str(value).strip()

    text = re.sub(r"^(?:https?|hf)://", "", text, flags=re.IGNORECASE)
    text = text.replace("\\", "/")
    text = text.replace(":", "_")
    text = re.sub(r"[?#].*$", "", text)

    if "/" in text:
        parts = [part for part in text.split("/") if part]
        text = parts[-1] if parts else text

    text = re.sub(
        r"\.(csv|tsv|txt|json|jsonl|ndjson|parquet|pq|xlsx|xls)$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = normalize_column_name(text)

    if not text:
        text = f"dataset_{stable_hash(value, 12)}"

    return text


def dataset_name_from_source(source: str) -> str:
    """
    Derive the project-local dataset identity from any supported source.

    Managed:      known://goemo                  -> goemo
    HF:           hf://dair-ai/emotion           -> emotion
                  hf://tweet_eval:emotion        -> tweet_eval_emotion
    URL:          https://example.org/foo.csv    -> foo
    Local:        ./data/my_dataset.csv          -> my_dataset
    """
    source = str(source).strip()

    if source.startswith("known://"):
        return sanitize_dataset_name(source[len("known://"):])

    if source.startswith("hf://"):
        repo_spec = source[len("hf://"):].strip()
        if ":" in repo_spec:
            repo, config = repo_spec.split(":", 1)
            return sanitize_dataset_name(
                f"{sanitize_dataset_name(repo)}_{sanitize_dataset_name(config)}"
            )
        return sanitize_dataset_name(repo_spec)

    if is_url(source):
        parsed = urlparse(source)
        path_name = Path(parsed.path).name
        if path_name:
            name = sanitize_dataset_name(path_name)
            if name:
                return name
        domain = parsed.netloc.split(":")[0]
        return sanitize_dataset_name(domain)

    return sanitize_dataset_name(Path(source).expanduser().name)


def dataset_root_dir(dataset_name: str) -> Path:
    return PROJECT_DATASETS_DIR / sanitize_dataset_name(dataset_name)


def dataset_raw_dir(dataset_name: str) -> Path:
    return dataset_root_dir(dataset_name) / "raw"


def dataset_processed_dir(dataset_name: str) -> Path:
    return dataset_root_dir(dataset_name) / "processed"


def dataset_raw_path(dataset_name: str) -> Path:
    """
    Canonical raw snapshot location:
        <root>/<name>/raw/<name>.csv
    """
    name = sanitize_dataset_name(dataset_name)
    return dataset_raw_dir(name) / f"{name}.csv"


def dataset_processed_path(dataset_name: str) -> Path:
    """
    Canonical processed output location:
        <root>/<name>/processed/<name>_clean.csv
    """
    name = sanitize_dataset_name(dataset_name)
    return dataset_processed_dir(name) / f"{name}_clean.csv"


def known_dataset_raw_dir(key: str) -> Path:
    return dataset_raw_dir(key)


def known_dataset_processed_dir(key: str) -> Path:
    return dataset_processed_dir(key)


def known_dataset_local_path(key: str) -> Path:
    """
    Canonical raw CSV for a managed *or* discovered dataset.
    """
    return dataset_raw_path(key)


def known_dataset_is_local(key: str) -> bool:
    path = known_dataset_local_path(key)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def resolve_known_dataset_key(value: str) -> Optional[str]:
    """
    Resolve managed or discovered datasets case-insensitively by key or
    display name. Returns None if the name matches nothing.
    """
    candidate = str(value).strip()
    if not candidate:
        return None

    normalized_candidate = normalize_column_name(candidate)

    for key, spec in all_datasets().items():
        key_normalized = normalize_column_name(key)
        name_normalized = normalize_column_name(spec.get("name", ""))
        if normalized_candidate in {key_normalized, name_normalized}:
            return key

    return None


def atomic_write_dataframe_csv(frame: pd.DataFrame, destination: Path) -> Path:
    """
    Atomically persist a raw DataFrame as the canonical raw CSV.
    """
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        frame.to_csv(temporary, index=False)

        if not temporary.exists() or temporary.stat().st_size == 0:
            raise DatasetSourceError(
                f"Raw dataset write produced an empty file: {temporary}"
            )

        temporary.replace(destination)

    except DatasetSourceError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise DatasetSourceError(
            f"Failed writing raw dataset snapshot '{destination}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return destination


def default_processed_output_for_source(source: str) -> Path:
    """
    Universal default processed output:
        <root>/<dataset_name>/processed/<dataset_name>_clean.csv
    """
    return dataset_processed_path(dataset_name_from_source(source))
def _adaptive_max_length(
    requested_max_length: int,
    texts: Sequence[str],
    *,
    tokenizer=None,
    sample_size: int = 2000,
    percentile: float = 0.99,
    min_length: int = 16,
) -> int:
    """
    Pick a tokenizer length that covers `percentile` of the text distribution,
    capped at the user-requested max_length.

    Rationale
    ---------
    
    On short-text datasets (tweets, single sentences) a max_length of 256
    wastes compute on padding. This samples the corpus, estimates the
    p99 token length, rounds up to a multiple of 8, and returns it —
    so long as it does not exceed what the user asked for.

    Without a tokenizer we approximate ~4 characters per subword token.
    With a tokenizer we run real encode() calls on a bounded sample.
    """
    if not texts:
        return int(requested_max_length)

    n = len(texts)
    if n > sample_size:
        step = max(1, n // sample_size)
        sample = [texts[i] for i in range(0, n, step)][:sample_size]
    else:
        sample = list(texts)

    if tokenizer is not None:
        try:
            lengths = [
                len(tokenizer.encode(t, add_special_tokens=True, truncation=False))
                for t in sample
            ]
        except Exception:
            lengths = [max(1, len(t) // 4) for t in sample]
    else:
        lengths = [max(1, len(t) // 4) for t in sample]

    if not lengths:
        return int(requested_max_length)

    p = float(np.percentile(lengths, percentile * 100.0))
    rounded = int(((p + 7) // 8) * 8)
    return max(int(min_length), min(int(requested_max_length), rounded))
# =============================================================================
# ROBUST DELIMITER-SEPARATED INGESTION
# =============================================================================

@dataclass
class DelimitedIngestPolicy:
    """How a delimiter-separated file should be read and repaired."""
    delimiter: str = ","
    text_column: Optional[str] = None
    on_field_mismatch: str = "skip"     # merge_into_text | skip | strict
    header: bool = True
    encoding: str = "utf-8"
    quotechar: str = '"'
    merge_delta_limit: int = 64


def sniff_delimited_policy(
    path: Path,
    sample_bytes: int = 65536,
) -> DelimitedIngestPolicy:
    """Sniff delimiter + header presence from a small sample."""
    try:
        with path.open("rb") as _fh:
            raw = _fh.read(sample_bytes)
    except OSError:
        return DelimitedIngestPolicy()

    try:
        sample = raw.decode("utf-8", errors="replace")
    except Exception:
        return DelimitedIngestPolicy()

    if not sample.strip():
        return DelimitedIngestPolicy()

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = True

    return DelimitedIngestPolicy(
        delimiter=delimiter,
        text_column=None,
        on_field_mismatch="skip",
        header=has_header,
    )


def read_delimited_robust(
    path: Path,
    policy: DelimitedIngestPolicy,
) -> pd.DataFrame:
    """Quote-aware, row-shape-validating reader for delimiter-separated files."""
    path = Path(path).expanduser().resolve()

    rows: list[list[str]] = []
    header: Optional[list[str]] = None
    expected: Optional[int] = None
    skipped: int = 0
    merged: int = 0

    def _raise(line_no: int, got: int) -> None:
        raise DatasetSourceError(
            f"{path.name}:{line_no}: expected {expected} fields, got {got}."
        )

    with path.open("r", encoding=policy.encoding, newline="") as handle:
        reader = csv.reader(
            handle,
            delimiter=policy.delimiter,
            quotechar=policy.quotechar,
        )

        for line_no, parts in enumerate(reader, start=1):

            if header is None and policy.header:
                header = parts
                expected = len(header)
                continue

            if expected is None:
                expected = len(parts)
                header = [f"col_{i}" for i in range(expected)]

            if len(parts) == expected:
                rows.append(parts)
                continue

            delta = len(parts) - expected

            if policy.on_field_mismatch == "strict":
                _raise(line_no, len(parts))

            if policy.on_field_mismatch == "skip":
                skipped += 1
                continue

            if policy.on_field_mismatch == "merge_into_text":
                if (
                    policy.text_column is None
                    or header is None
                    or policy.text_column not in header
                ):
                    raise DatasetSourceError(
                        f"{path.name}:{line_no}: cannot merge stray "
                        f"delimiters — text_column={policy.text_column!r} "
                        f"is not in header {header!r}."
                    )

                if abs(delta) > policy.merge_delta_limit:
                    raise DatasetSourceError(
                        f"{path.name}:{line_no}: field-count deviation "
                        f"{delta} exceeds merge_delta_limit="
                        f"{policy.merge_delta_limit}."
                    )

                idx = header.index(policy.text_column)

                if delta > 0:
                    merged_row = (
                        parts[:idx]
                        + [policy.delimiter.join(parts[idx : idx + delta + 1])]
                        + parts[idx + delta + 1 :]
                    )
                else:
                    merged_row = parts + [""] * (-delta)

                rows.append(merged_row)
                merged += 1
                continue

            raise DatasetSourceError(
                f"{path.name}:{line_no}: unknown on_field_mismatch policy "
                f"{policy.on_field_mismatch!r}."
            )

    if header is None:
        return pd.DataFrame()

    frame = pd.DataFrame(rows, columns=header)

    if skipped or merged:
        print(
            f"[ingest] {path.name}: parsed {len(frame):,} rows "
            f"({merged} merged, {skipped} skipped)."
        )

    return frame


# =============================================================================
# RICH PRESENTATION LAYER
# =============================================================================

class Renderer:
    """Centralised presentation layer with a fresh palette per table call."""

    def __init__(self, quiet: bool = False, no_visuals: bool = False):
        self.quiet = bool(quiet)
        self.no_visuals = bool(no_visuals)
        self.console = Console(highlight=False, soft_wrap=True, emoji=True) if (
            RICH_AVAILABLE and VERBOSE and not quiet and not no_visuals
        ) else None
        self._table_counter = 0
        self._previous_palette: tuple[str, ...] | None = None

    @property
    def enabled(self) -> bool:
        return bool(VERBOSE and not self.quiet and not self.no_visuals)

    def _palette(self, count: int) -> tuple[list[str], str, str]:
        rng = random.SystemRandom()
        for _ in range(12):
            palette = tuple(rng.sample(LIGHT_COLORS, k=min(count, len(LIGHT_COLORS))))
            if palette != self._previous_palette:
                break
        else:
            palette = tuple(LIGHT_COLORS[i % len(LIGHT_COLORS)] for i in range(count))
            if palette == self._previous_palette and count > 1:
                palette = palette[1:] + palette[:1]
        self._previous_palette = palette
        self._table_counter += 1
        return list(palette), rng.choice(BACKGROUND_COLORS), rng.choice(LIGHT_COLORS)

    def panel(self, title: str, body: str, *, border: Optional[str] = None) -> None:
        if not self.enabled:
            return
        if self.console:
            border_colour = border or random.SystemRandom().choice(LIGHT_COLORS)
            self.console.print(
                Panel(
                    Align.center(Text(body, style="white")),
                    title=title,
                    title_align="center",
                    border_style=border_colour,
                    box=box.DOUBLE,
                    padding=(1, 2),
                )
            )
        else:
            print(f"\n{'=' * 96}\n{title}\n{body}\n{'=' * 96}")

    def table(
        self,
        title: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        caption: Optional[str] = None,
        max_width: int = 180,
        show_lines: bool = False,
    ) -> None:
        if not self.enabled:
            return
        row_list = [list(row) for row in rows]
        palette, background, title_colour = self._palette(len(columns))
        if self.console:
            table = Table(
                title=Text(title, style=f"bold {title_colour}"),
                box=box.ROUNDED,
                border_style=title_colour,
                header_style="bold white",
                show_lines=show_lines,
                expand=True,
                padding=(0, 1),
                width=min(max_width, self.console.width or max_width),
            )
            for i, column in enumerate(columns):
                colour = palette[i]
                table.add_column(
                    str(column),
                    style=colour,
                    header_style=f"bold {colour} on {background}",
                    overflow="fold",
                    no_wrap=False,
                )
            for row in row_list:
                if len(row) != len(columns):
                    row = list(row[: len(columns)]) + [""] * max(0, len(columns) - len(row))
                cells = [Text(compact(row[i], 240), style=palette[i]) for i in range(len(columns))]
                table.add_row(*cells)
            self.console.print(table)
            if caption:
                self.console.print(Text(caption, style=f"italic {title_colour}"))
            self.console.print()
        else:
            print(f"\n{title}")
            print(" | ".join(str(c) for c in columns))
            print("-" * 120)
            for row in row_list:
                print(" | ".join(compact(v, 120) for v in row))
            print()

    def status(self, kind: str, message: str) -> None:
        if not self.enabled:
            return
        symbols = {"ok": "✓", "warn": "⚠", "error": "✗", "info": "◆"}
        self.table(
            "STATUS",
            ["State", "Message"],
            [[symbols.get(kind, "•"), message]],
            show_lines=False,
        )

    def success(self, message: str) -> None:
        self.status("ok", message)

    def warning(self, message: str) -> None:
        self.status("warn", message)

    def error(self, message: str) -> None:
        self.status("error", message)

    def info(self, message: str) -> None:
        self.status("info", message)

    def progress(self, description: str, total: int):
        if not self.enabled or not self.console:
            return None
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )

    def menu(self, title: str, options: Sequence[tuple[str, str, str]], footer: str = "") -> None:
        if not self.enabled:
            return
        if self.console:
            colours, background, title_colour = self._palette(3)
            table = Table(
                title=Text(title, style=f"bold {title_colour}"),
                box=box.HEAVY_HEAD,
                border_style=title_colour,
                show_header=True,
                expand=True,
            )
            table.add_column("Key", style=colours[0], header_style=f"bold {colours[0]} on {background}", justify="center", width=8)
            table.add_column("Action", style=colours[1], header_style=f"bold {colours[1]} on {background}")
            table.add_column("Command / shortcut", style=colours[2], header_style=f"bold {colours[2]} on {background}")
            for key, action, command in options:
                table.add_row(Text(key, style=colours[0]), Text(action, style=colours[1]), Text(command, style=colours[2]))
            self.console.print(table)
            if footer:
                self.console.print(Panel(Text(footer, style="#EAF4FF"), border_style=title_colour, box=box.ROUNDED))
        else:
            print(f"\n{title}")
            for key, action, command in options:
                print(f"[{key}] {action} — {command}")
            if footer:
                print(footer)


# =============================================================================
# TEXT CLEANER
# =============================================================================

class TextCleaner:
    URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    USER_RE = re.compile(r"(?<!\w)@[\w.-]+", re.UNICODE)
    HASHTAG_RE = re.compile(r"(?<!\w)#([\w-]+)", re.UNICODE)
    CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    SPACE_RE = re.compile(r"\s+")
    REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1{3,}")
    REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}", re.DOTALL)

    def __init__(
        self,
        lowercase: bool = True,
        demojize: bool = True,
        normalize_urls: bool = True,
        normalize_usernames: bool = True,
        normalize_hashtags: bool = False,
        strip_html: bool = True,
        normalize_repeated_punctuation: bool = True,
    ):
        self.lowercase = lowercase
        self.demojize = demojize
        self.normalize_urls = normalize_urls
        self.normalize_usernames = normalize_usernames
        self.normalize_hashtags = normalize_hashtags
        self.strip_html = strip_html
        self.normalize_repeated_punctuation = normalize_repeated_punctuation

    def clean(self, value: Any) -> str:
        if try_is_missing(value):
            return ""
        text = unicodedata.normalize("NFKC", str(value))
        text = html.unescape(text)
        text = self.CONTROL_RE.sub(" ", text)
        if self.strip_html:
            text = re.sub(r"<[^>]*>", " ", text)
        if self.normalize_urls:
            text = self.URL_RE.sub(" <url> ", text)
        if self.normalize_usernames:
            text = self.USER_RE.sub(" <user> ", text)
        if self.normalize_hashtags:
            text = self.HASHTAG_RE.sub(r" \1 ", text)
        if self.demojize and emoji_lib is not None:
            try:
                text = emoji_lib.demojize(text, delimiters=(" ", " "))
            except Exception:
                pass
        if self.lowercase:
            text = text.lower()
        if self.normalize_repeated_punctuation:
            text = self.REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
        text = self.REPEATED_CHAR_RE.sub(r"\1\1\1", text)
        return self.SPACE_RE.sub(" ", text).strip()


# =============================================================================
# DATASET LOADER
# =============================================================================

class DatasetLoader:
    def __init__(self, renderer: Renderer, cache_dir: Optional[str | Path] = None, timeout: tuple[int, int] = (60, 600)):
        self.renderer = renderer
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    @staticmethod
    def _download_via_curl(
        url: str,
        destination: Path,
        max_attempts: int = 3,
    ) -> Path:
            """
            Fallback downloader that shells out to curl.

            Used when requests/urllib3 hits a TLS handshake timeout on a host that
            is known to throttle non-browser TLS stacks. curl is more tolerant of
            the throttled endpoints (dl.fbaipublicfiles.com, some Cloudflare
            Worker edges) because it can negotiate http/1.1 or downgrade cipher
            suites on retry.

            Note: this is synchronous and does not stream through Python. curl
            writes directly to the destination file.
            """
            destination = Path(destination).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_suffix(destination.suffix + ".part")

            last_err: Optional[str] = None
            for attempt in range(1, max_attempts + 1):
                result = subprocess.run(
                    [
                        "curl", "-fL",
                        "--retry", "2",
                        "--retry-delay", "5",
                        "--connect-timeout", "30",
                        "--max-time", "1800",
                        "--http1.1",
                        "--tlsv1.2",
                        "-o", str(tmp),
                        url,
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                    tmp.replace(destination)
                    return destination

                last_err = result.stderr.strip() or f"curl exit {result.returncode}"
                tmp.unlink(missing_ok=True)
                if attempt < max_attempts:
                    print(
                        f"[curl-fallback] {url} attempt {attempt}/{max_attempts} "
                        f"failed: {last_err}; retrying…"
                    )
                    time.sleep(5.0 * attempt)

            raise DatasetSourceError(
                f"curl fallback also failed for {url}: {last_err}"
            )

    @staticmethod
    def read_rows_only(
        path: Path,
        n: int,
        *,
        mode: str = "head",          # head | tail | random
        seed: int = 42,
        policy: Optional[DelimitedIngestPolicy] = None,
    ) -> pd.DataFrame:
        """
        Read at most n rows from a delimiter-separated file *without*
        materialising the whole dataset.

        Modes
        -----
        head    : pd.read_csv(nrows=n). Single bounded read; allocates n rows.
        tail    : csv.reader + deque(maxlen=n). Single streaming pass; holds
                  only the last n rows at any moment.
        random  : reservoir sampling. Single streaming pass; holds only n
                  rows at any moment. Deterministic given `seed`.

        All three modes are O(file size) in I/O but O(n) in memory.

        Non-delimited formats (json/jsonl/parquet/xlsx) fall back to a full
        read followed by a slice. Those formats cannot be randomly accessed
        at the row level through pandas without loading them.

        NOTE: this method never touches self.raw_df, never mutates any
        processor state, and never calls self.load(). It is the correct
        entry point for preview and inspect.
        """
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise DatasetSourceError(f"Dataset file does not exist: {path}")

        suffix = path.suffix.lower()

        # --- Non-delimited fallback -----------------------------------------
        if suffix not in {".csv", ".tsv", ".txt"}:
            frame = DatasetLoader._read_path(path, policy=policy)
            if frame.empty:
                return frame
            n = min(n, len(frame))
            if mode == "tail":
                return frame.tail(n).copy().reset_index(drop=True)
            if mode == "random":
                return frame.sample(n, random_state=seed).reset_index(drop=True)
            return frame.head(n).copy().reset_index(drop=True)

        pol = policy or sniff_delimited_policy(path)
        n = max(1, int(n))

        # --- HEAD ------------------------------------------------------------
        if mode == "head":
            # NOTE: pandas nrows reads exactly n data rows (after the header
            # line if header=0). It never scans the rest of the file.
            try:
                return pd.read_csv(
                    path,
                    sep=pol.delimiter,
                    nrows=n,
                    header=0 if pol.header else None,
                    dtype=str,
                    keep_default_na=False,
                )
            except Exception:
                # Fall back to the robust reader for pathological files.
                frame = DatasetLoader._read_path(path, policy=pol)
                return frame.head(min(n, len(frame))).copy().reset_index(drop=True)

        # --- TAIL and RANDOM both stream the file exactly once --------------
        with path.open("r", encoding=pol.encoding, newline="") as handle:
            reader = csv.reader(
                handle,
                delimiter=pol.delimiter,
                quotechar=pol.quotechar,
            )
            header = next(reader, None) if pol.header else None

            if mode == "tail":
                # NOTE: deque(maxlen=n) keeps only the last n rows. Memory
                # is bounded by n regardless of file size. This is the
                # entire point of a streaming tail.
                tail_rows = deque(reader, maxlen=n)
                columns = (
                    header
                    if header is not None
                    else [f"col_{i}" for i in range(len(tail_rows[0]) if tail_rows else 0)]
                )
                return pd.DataFrame(list(tail_rows), columns=columns)

            if mode == "random":
                # NOTE: Reservoir Sampling (Algorithm R). One pass, memory
                # bounded by n. Every data row has an equal probability of
                # ending up in the reservoir. Deterministic given `seed`.
                rng = random.Random(seed)
                reservoir: list[list[str]] = []
                for i, row in enumerate(reader):
                    if i < n:
                        reservoir.append(row)
                    else:
                        j = rng.randint(0, i)
                        if j < n:
                            reservoir[j] = row
                columns = (
                    header
                    if header is not None
                    else [f"col_{i}" for i in range(len(reservoir[0]) if reservoir else 0)]
                )
                return pd.DataFrame(reservoir, columns=columns)

        raise ValueError(f"Unknown sampling mode: {mode!r}")
    
    def _download(
        self,
        url: str,
        destination: Optional[str | Path] = None,
        max_attempts: int = 3,
    ) -> Path:
        if requests is None:
            raise DatasetSourceError(
                "URL input requires requests. Install: pip install requests"
            )

        parsed = urlparse(url)

        if destination is None:
            filename = Path(parsed.path).name or f"dataset_{stable_hash(url)}.csv"
            if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                filename = f"dataset_{stable_hash(url)}.csv"
            destination_path = self.cache_dir / f"{stable_hash(url)}_{filename}"
        else:
            destination_path = Path(destination).expanduser().resolve()

        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists() and destination_path.stat().st_size > 0:
            return destination_path

        tmp = destination_path.with_suffix(destination_path.suffix + ".part")

        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            try:
                resolved_url = _resolve_hf_url(url)
                with requests.get(
                    resolved_url,
                    stream=True,
                    timeout=self.timeout,
                    headers={"User-Agent": f"MasterDatasetProcessor/{VERSION}"},
                ) as response:
                    response.raise_for_status()

                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" in content_type:
                        raise DatasetSourceError(
                            f"Remote source returned HTML instead of a dataset "
                            f"file: {url}. Use a direct/raw dataset URL."
                        )

                    with tmp.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)

                head = tmp.read_bytes()[:512].lower()
                if b"<!doctype html" in head or b"<html" in head:
                    raise DatasetSourceError(
                        f"Remote source returned an HTML page instead of a "
                        f"dataset file: {url}"
                    )

                tmp.replace(destination_path)
                return destination_path

            except DatasetSourceError:
                tmp.unlink(missing_ok=True)
                raise
            except Exception as exc:
                last_exc = exc
                tmp.unlink(missing_ok=True)
                if attempt < max_attempts:
                    backoff = 5.0 * (2 ** (attempt - 1))
                    print(
                        f"[download] {url} failed ({type(exc).__name__}: {exc}); "
                        f"retrying in {backoff:.1f}s "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    time.sleep(backoff)

        # NOTE: if requests has tried and failed, the host may be one that
        # throttles non-browser TLS stacks. Fall back to curl before giving up.
        if destination is not None:
            print(f"[download] requests exhausted; trying curl fallback for {url}")
            return DatasetLoader._download_via_curl(url, Path(destination), max_attempts=2)

        raise DatasetSourceError(
            f"Could not download dataset after {max_attempts} attempts: "
            f"{type(last_exc).__name__ if last_exc else 'UnknownError'}: {last_exc}"
        ) from last_exc

    @staticmethod
    def _read_path(
        path: Path,
        policy: Optional[DelimitedIngestPolicy] = None,
    ) -> pd.DataFrame:
        """Parse any supported dataset file into a DataFrame."""
        path = Path(path).expanduser().resolve()

        if not path.exists():
            raise DatasetSourceError(f"Dataset file does not exist: {path}")
        if not path.is_file():
            raise DatasetSourceError(f"Dataset source is not a file: {path}")
        if path.stat().st_size == 0:
            raise DatasetSourceError(f"Dataset file is empty: {path}")

        suffix = path.suffix.lower()

        try:
            if suffix in {".csv", ".tsv", ".txt"}:
                effective_policy = policy or sniff_delimited_policy(path)
                return read_delimited_robust(path, effective_policy)
            if suffix in {".jsonl", ".ndjson"}:
                return pd.read_json(path, lines=True)
            if suffix == ".json":
                try:
                    return pd.read_json(path)
                except ValueError:
                    return pd.read_json(path, lines=True)
            if suffix in {".parquet", ".pq"}:
                return pd.read_parquet(path)
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(path)
        except DatasetSourceError:
            raise
        except Exception as exc:
            raise DatasetSourceError(
                f"Failed parsing '{path}': {type(exc).__name__}: {exc}"
            ) from exc

        raise DatasetSourceError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    @staticmethod
    def _ingest_delimited(key: str, path: Path) -> pd.DataFrame:
        """
        Read a managed dataset's canonical raw snapshot with the delimiter
        policy declared by its spec (or a comma-based default).
        """
        spec = all_datasets().get(key, {})
        policy_dict = spec.get("ingest_policy")
        if policy_dict:
            policy = DelimitedIngestPolicy(**policy_dict)
        else:
            policy = DelimitedIngestPolicy(
                delimiter=",",
                text_column=spec.get("text_column"),
                on_field_mismatch="merge_into_text",
                header=True,
            )
        return read_delimited_robust(path, policy)

    @staticmethod
    def _read_known_local(key: str, path: Path) -> pd.DataFrame:
        """
        Read the canonical raw CSV for a managed OR discovered dataset.

        NOTE: discovered datasets were written by atomic_write_dataframe_csv
        (pandas default CSV format), so a plain pd.read_csv is sufficient.
        """
        spec = all_datasets().get(key)
        if spec is None:
            raise DatasetSourceError(f"Unknown managed dataset key: {key}")

        try:
            source_type = spec.get("source_type")

            if source_type == "delimited":
                frame = DatasetLoader._ingest_delimited(key, path)
            elif source_type == "goemotions_tsv":
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            else:
                # Covers emobank_csv, empathetic_archive, discovered,
                # and any future source_type whose raw snapshot is a plain CSV.
                frame = pd.read_csv(path)
        except Exception as exc:
            raise DatasetSourceError(
                f"Failed parsing cached known dataset '{key}' at '{path}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(frame, pd.DataFrame):
            raise DatasetSourceError(
                f"Cached known dataset '{key}' did not load as a DataFrame."
            )
        if frame.empty:
            raise DatasetSourceError(
                f"Known dataset '{key}' contains zero rows."
            )

        return frame

    def load_known(
        self,
        key: str,
        *,
        offline: bool = False,
    ) -> tuple[pd.DataFrame, Path]:
        resolved_key = resolve_known_dataset_key(key)
        if resolved_key is None:
            raise DatasetSourceError(f"Unknown known dataset key: {key}")

        key = resolved_key
        datasets = all_datasets()
        spec = datasets[key]

        raw_root = known_dataset_raw_dir(key)
        local_path = known_dataset_local_path(key)

        # ---------------------------------------------------------------------
        # LOCAL-FIRST: works for managed AND discovered datasets.
        # ---------------------------------------------------------------------
        if known_dataset_is_local(key):
            frame = self._read_known_local(key, local_path)

            self.renderer.table(
                "DATASET CACHE",
                ["Dataset", "Source mode", "Local path", "Status"],
                [[spec["name"], "LOCAL", str(local_path), "READY"]],
            )
            self.renderer.panel(
                "CACHE LOCATION",
                f"{spec['name']}\n{local_path}",
            )
            return frame, local_path

        # ---------------------------------------------------------------------
        # OFFLINE WITHOUT LOCAL RAW COPY
        # ---------------------------------------------------------------------
        if offline:
            raise DatasetSourceError(
                f"Known dataset '{key}' is not available locally.\n"
                f"Expected raw snapshot:\n{local_path}\n"
                f"Run once online to acquire it."
            )

        kind = spec["source_type"]

        # ---------------------------------------------------------------------
        # GOEMOTIONS
        # ---------------------------------------------------------------------
        if kind == "goemotions_tsv":
            frames: list[pd.DataFrame] = []
            for split, url in spec["online_urls"].items():
                split_path = self._download(
                    url, raw_root / f"goemotions_{split}.tsv",
                )
                part = pd.read_csv(
                    split_path,
                    sep="\t",
                    header=None,
                    names=["text", "labels", "id"],
                    dtype=str,
                    keep_default_na=False,
                )
                part["__source_split"] = split
                frames.append(part)

            if not frames:
                raise DatasetSourceError("GoEmotions acquisition produced no split files.")

            acquired = pd.concat(frames, ignore_index=True)
            atomic_write_dataframe_csv(acquired, local_path)

        # ---------------------------------------------------------------------
        # ISEAR
        # ---------------------------------------------------------------------
        elif kind == "delimited":
            downloaded_path = self._download(
                spec["online_urls"]["raw"], raw_root / "_source_isear.csv",
            )
            policy = DelimitedIngestPolicy(
                delimiter=spec.get("delimiter", ","),
                text_column=spec["text_column"],
                on_field_mismatch="merge_into_text",
                header=True,
            )
            acquired = self._read_path(downloaded_path, policy=policy)
            atomic_write_dataframe_csv(acquired, local_path)

        # ---------------------------------------------------------------------
        # EMOBANK
        # ---------------------------------------------------------------------
        elif kind == "emobank_csv":
            downloaded_path = self._download(
                spec["online_urls"]["raw"], raw_root / "_source_emobank.csv",
            )
            acquired = pd.read_csv(downloaded_path)
            atomic_write_dataframe_csv(acquired, local_path)

        # ---------------------------------------------------------------------
        # EMPATHETIC DIALOGUES
        # ---------------------------------------------------------------------
        elif kind == "empathetic_archive":
            archive_path = self._download(
                spec["online_url"], raw_root / "empatheticdialogues.tar.gz",
            )
            archive_members = {
                "train": "empatheticdialogues/train.csv",
                "validation": "empatheticdialogues/valid.csv",
                "test": "empatheticdialogues/test.csv",
            }
            rows: list[dict[str, Any]] = []
            try:
                with tarfile.open(archive_path, "r:gz") as archive:
                    for split, member_name in archive_members.items():
                        try:
                            member = archive.getmember(member_name)
                        except KeyError as exc:
                            raise DatasetSourceError(
                                f"EmpatheticDialogues archive is missing "
                                f"{member_name}"
                            ) from exc

                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise DatasetSourceError(
                                f"Could not extract {member_name} "
                                f"from EmpatheticDialogues archive."
                            )

                        reader = csv.DictReader(
                            io.TextIOWrapper(extracted, encoding="utf-8")
                        )
                        for row in reader:
                            row["__source_split"] = split
                            rows.append(row)
            except tarfile.TarError as exc:
                raise DatasetSourceError(
                    "Failed to read EmpatheticDialogues archive: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            acquired = pd.DataFrame(rows)
            if acquired.empty:
                raise DatasetSourceError(
                    "EmpatheticDialogues archive produced zero rows."
                )
            atomic_write_dataframe_csv(acquired, local_path)

        # ---------------------------------------------------------------------
        # DISCOVERED (no acquisition path — must already be local)
        # ---------------------------------------------------------------------
        elif kind == "discovered":
            # NOTE: reaching here means known_dataset_is_local() returned False
            # for a discovered dataset, which means the folder exists but the
            # raw CSV is missing/empty. That is a hard error, not something to
            # silently recover from.
            raise DatasetSourceError(
                f"Discovered dataset '{key}' has no raw snapshot at {local_path}. "
                f"Populate raw/{key}.csv or remove the folder."
            )

        else:
            raise DatasetSourceError(
                f"Known dataset '{key}' has unsupported source_type '{kind}'."
            )

        # ---------------------------------------------------------------------
        # REREAD THE CANONICAL RAW SNAPSHOT
        # ---------------------------------------------------------------------
        if not known_dataset_is_local(key):
            raise DatasetSourceError(
                f"Known dataset '{key}' was acquired but its canonical raw "
                f"snapshot was not created successfully:\n{local_path}"
            )

        frame = self._read_known_local(key, local_path)

        self.renderer.table(
            "DATASET CACHE",
            ["Dataset", "Source mode", "Local path", "Status"],
            [[spec["name"], "DOWNLOADED → CACHED → RELOADED", str(local_path), "READY"]],
        )
        self.renderer.panel(
            "CACHE LOCATION",
            f"{spec['name']}\n{local_path}",
        )
        return frame, local_path

    def _load_foreign_source_to_raw(
        self,
        source: str,
        *,
        hf_config: Optional[str] = None,
        offline: bool = False,
    ) -> tuple[pd.DataFrame, Path]:
        """
        Resolve any non-managed source into the universal project-local raw
        cache: <root>/<name>/raw/<name>.csv
        """
        source = str(source).strip()
        dataset_name = dataset_name_from_source(source)
        raw_path = dataset_raw_path(dataset_name)

        # ---------------------------------------------------------------------
        # LOCAL-FIRST
        # ---------------------------------------------------------------------
        if raw_path.exists() and raw_path.is_file() and raw_path.stat().st_size > 0:
            frame = self._read_path(raw_path)
            if frame.empty:
                raise DatasetSourceError(
                    f"Cached foreign dataset is empty:\n{raw_path}"
                )

            self.renderer.table(
                "FOREIGN DATASET CACHE",
                ["Dataset", "Source mode", "Local path", "Status"],
                [[dataset_name, "LOCAL", str(raw_path), "READY"]],
            )
            return frame, raw_path

        # ---------------------------------------------------------------------
        # OFFLINE
        # ---------------------------------------------------------------------
        if offline:
            raise DatasetSourceError(
                f"Foreign dataset '{dataset_name}' is not available locally.\n"
                f"Expected raw snapshot:\n{raw_path}\n"
                f"Run once online to acquire it."
            )

        acquired: Optional[pd.DataFrame] = None

        # ---------------------------------------------------------------------
        # HUGGING FACE
        # ---------------------------------------------------------------------
        if source.startswith("hf://"):
            if hf_load_dataset is None:
                raise DatasetSourceError(
                    "hf:// input requires the 'datasets' library. "
                    "Install it with: pip install datasets"
                )

            repo_spec = source[len("hf://"):].strip()
            config = hf_config
            if ":" in repo_spec and config is None:
                repo_spec, config = repo_spec.split(":", 1)

            repo = repo_spec.strip()
            try:
                loaded = hf_load_dataset(repo, name=config, trust_remote_code=True)
            except Exception as exc:
                raise DatasetSourceError(
                    f"Could not load Hugging Face dataset '{repo}'. "
                    f"Error: {type(exc).__name__}: {exc}\n"
                    "If this is a canonical dataset that has been reorganized upstream, "
                    "pass its fully qualified ID instead:\n"
                    "  hf://<namespace>/<name>\n"
                    "For GLUE-family datasets, the canonical source is nyu-mll/glue:\n"
                    "  hf://nyu-mll/glue --hf-config <config>"
                ) from exc

            if hasattr(loaded, "items"):
                frames: list[pd.DataFrame] = []
                for split, part in loaded.items():
                    split_frame = part.to_pandas()
                    if not isinstance(split_frame, pd.DataFrame):
                        raise DatasetSourceError(
                            f"Hugging Face split '{split}' did not convert "
                            f"to a DataFrame."
                        )
                    split_frame["__source_split"] = split
                    frames.append(split_frame)
                if not frames:
                    raise DatasetSourceError(
                        f"Hugging Face dataset '{repo}' returned no splits."
                    )
                acquired = pd.concat(frames, ignore_index=True)
            else:
                acquired = loaded.to_pandas()

        # ---------------------------------------------------------------------
        # HTTP(S)
        # ---------------------------------------------------------------------
        elif is_url(source):
            downloaded_path = self._download(source)
            acquired = self._read_path(downloaded_path)

        # ---------------------------------------------------------------------
        # LOCAL FILE
        # ---------------------------------------------------------------------
        else:
            local_source = Path(source).expanduser().resolve()
            if not local_source.exists():
                raise DatasetSourceError(f"Dataset path does not exist: {local_source}")
            if not local_source.is_file():
                raise DatasetSourceError(f"Dataset source is not a file: {local_source}")
            acquired = self._read_path(local_source)

        if acquired is None:
            raise DatasetSourceError(f"Unable to acquire dataset from source: {source}")
        if not isinstance(acquired, pd.DataFrame):
            raise DatasetSourceError(
                f"Dataset source did not produce a DataFrame: {source}"
            )
        if acquired.empty:
            raise DatasetSourceError(
                f"Dataset source contains zero rows: {source}"
            )

        atomic_write_dataframe_csv(acquired, raw_path)

        self.renderer.table(
            "FOREIGN DATASET CACHE",
            ["Dataset", "Source mode", "Local path", "Status"],
            [[dataset_name, "ACQUIRED → CACHED", str(raw_path), "WRITTEN"]],
        )

        frame = self._read_path(raw_path)
        if frame.empty:
            raise DatasetSourceError(
                f"Saved raw dataset is empty after rereading:\n{raw_path}"
            )

        self.renderer.table(
            "RAW SNAPSHOT",
            ["Property", "Value"],
            [
                ["Dataset", dataset_name],
                ["Source", compact(source, 120)],
                ["Raw path", str(raw_path)],
                ["Rows", f"{len(frame):,}"],
                ["Columns", f"{len(frame.columns):,}"],
            ],
        )
        return frame, raw_path

    def load(
        self,
        source: str,
        *,
        hf_config: Optional[str] = None,
        offline: bool = False,
    ) -> tuple[pd.DataFrame, Optional[Path]]:
        source = str(source).strip()
        if not source:
            raise DatasetSourceError("Dataset source is empty.")

        if source.startswith("known://"):
            raw_key = source[len("known://"):].strip()
            resolved_key = resolve_known_dataset_key(raw_key)
            if resolved_key is None:
                raise DatasetSourceError(f"Unknown managed dataset: {raw_key}")
            return self.load_known(resolved_key, offline=offline)

        return self._load_foreign_source_to_raw(
            source, hf_config=hf_config, offline=offline,
        )


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================
def infer_task_and_class_count(
    label_series: pd.Series,
) -> tuple[str, int]:
    """
    Infer (task_type, class_count) from a label series of Python lists.

    Input is the output of LabelNormalizer.parse: each element is either
    None or a list of atomic labels. Length is 1 for single-label rows,
    > 1 for multi-label rows.

    Returns:
        ("single_label" | "multi_label" | "unknown", int)

    NOTE: class_count is the number of distinct *atomic* labels observed,
    not the number of distinct list combinations. For Amazon Polarity that
    is 2 (positive, negative), regardless of how many rows carry [0] or [1].
    """
    labels = [x for x in label_series if isinstance(x, list) and len(x) > 0]
    if not labels:
        return "unknown", 0

    max_length = max(len(x) for x in labels)
    task_type = "multi_label" if max_length > 1 else "single_label"

    unique: set[str] = set()
    for row in labels:
        for item in row:
            unique.add(str(item))
    return task_type, len(unique)

class LabelNormalizer:
    """Convert arbitrary label cells into an always-list canonical structure."""

    @staticmethod
    def _coerce_scalar(token: Any) -> Any:
        if isinstance(token, np.integer):
            return int(token)
        if isinstance(token, np.floating):
            if float(token).is_integer():
                return int(token)
            return float(token)
        text = str(token).strip()
        if not text:
            return ""
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except ValueError:
                pass
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", text):
            try:
                return float(text)
            except ValueError:
                pass
        return text

    @classmethod
    def parse(cls, value: Any) -> Optional[list[Any]]:
        if try_is_missing(value):
            return None
        if isinstance(value, np.ndarray):
            return cls._clean_list(cls.parse(value.tolist()) or [])
        if isinstance(value, (list, tuple, set)):
            return cls._clean_list([cls._parse_nested_item(v) for v in value])

        text = str(value).strip()
        if not text:
            return None

        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple, set)):
                        return cls._clean_list([cls._parse_nested_item(v) for v in parsed])
                except Exception:
                    pass
            inside = text[1:-1].strip()
            if inside and re.fullmatch(r"[-+]?\d+(?:\s+[-+]?\d+)+", inside):
                return [int(v) for v in inside.split()]

        if re.fullmatch(r"[-+]?\d+(?:\s+[-+]?\d+)+", text):
            return [int(v) for v in text.split()]

        for delimiter in ("|||", ";", "|", "\n"):
            if delimiter in text:
                parts = [p.strip() for p in text.split(delimiter) if p.strip()]
                if len(parts) > 1:
                    return cls._clean_list([cls._parse_nested_item(p) for p in parts])

        if "," in text and len(text) < 500:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) > 1 and all(len(p.split()) <= 4 for p in parts):
                return cls._clean_list([cls._parse_nested_item(p) for p in parts])

        return [cls._coerce_scalar(text)]

    @classmethod
    def _parse_nested_item(cls, item: Any) -> Any:
        if isinstance(item, (list, tuple, set, np.ndarray)):
            nested = cls.parse(item)
            return nested if nested is not None else []
        return cls._coerce_scalar(item)

    @staticmethod
    def _clean_list(values: Sequence[Any]) -> list[Any]:
        cleaned: list[Any] = []
        for value in values:
            if value == "":
                continue
            if isinstance(value, list):
                for nested in value:
                    if nested != "":
                        cleaned.append(nested)
            else:
                cleaned.append(value)
        return cleaned

    @classmethod
    def one_hot_row(cls, row: pd.Series, columns: Sequence[str]) -> Optional[list[Any]]:
        active: list[Any] = []
        for column in columns:
            value = row.get(column)
            if try_is_missing(value):
                continue
            token = str(value).strip().lower()
            if token in {"1", "true", "yes", "y", "positive"}:
                active.append(str(column))
        return active or None


# =============================================================================
# SCHEMA DETECTION
# =============================================================================

class SchemaDetector:
    def __init__(self, renderer: Renderer):
        self.renderer = renderer

    @staticmethod
    def _hint_score(name: str, hints: set[str]) -> float:
        """
        Score a column name for a text-or-label hint set.

        NOTE: this method is retained for backwards compatibility with any
        caller that passes an arbitrary hint set. For text and label
        specifically, prefer score_text_column_name() / score_label_column_name()
        so the tiered logic is applied consistently. When the caller passes
        TEXT_NAME_HINTS (the union alias) we route through the tiered scorer
        rather than treating title and content as equals.
        """
        # Route the two well-known sets through the tiered scorer.
        if hints is TEXT_NAME_HINTS:
            return score_text_column_name(name)
        if hints is LABEL_NAME_HINTS:
            return score_label_column_name(name)

        # Fallback for arbitrary hint sets (used nowhere today, but kept so
        # future call sites do not silently lose the shared vocabulary).
        normalized = normalize_column_name(name)
        if normalized in hints:
            return 12.0
        parts = set(normalized.split("_"))
        overlap = len(parts & hints)
        if overlap:
            return 5.0 + overlap * 2.0
        if any(h in normalized for h in hints):
            return 3.0
        return 0.0

    def detect(
    self,
    df: pd.DataFrame,
    *,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    scoring_sample: int = 50_000,
) -> DetectionResult:
        if df.empty:
            raise DatasetSchemaError("Cannot detect schema in an empty dataset.")

        columns = [str(c) for c in df.columns]
        result = DetectionResult()

        # --- Explicit overrides validate against the FULL frame (cheap). ---
        if text_column is not None:
            if text_column not in df.columns:
                raise ColumnDetectionError(
                    f"Requested text column '{text_column}' not found. Available: {columns}"
                )
            result.text_column = text_column

        if label_column is not None:
            if label_column == "__VAD__":
                if not {"V", "A", "D"}.issubset(set(df.columns)):
                    raise ColumnDetectionError(
                        f"VAD target requires columns V, A, D. Available: {columns}"
                    )
                result.label_column = "__VAD__"
            else:
                if label_column not in df.columns:
                    raise ColumnDetectionError(
                        f"Requested label column '{label_column}' not found. Available: {columns}"
                    )
                result.label_column = label_column

        # -----------------------------------------------------------------
        # NOTE: score on a bounded sample.
        #
        # Cardinality, fill rate, and mean text length all stabilise well
        # below 50k rows — a 4M-row frame and a 50k-row sample give the same
        # ranking. Evaluating on the full frame costs O(N) string coercions
        # and a full sort for the median, which on 4M rows is tens of seconds
        # wasted on every detect() call.
        #
        # The sample is deterministic (random_state=42) so repeated calls and
        # the header peek agree.
        # -----------------------------------------------------------------
        if scoring_sample and len(df) > scoring_sample:
            scoring_df = df.sample(scoring_sample, random_state=42)
        else:
            scoring_df = df

        n = max(len(scoring_df), 1)
        candidates_text: list[tuple[str, float]] = []
        candidates_label: list[tuple[str, float]] = []

        for col in scoring_df.columns:
            name = str(col)
            normalized = normalize_column_name(name)
            series = scoring_df[col]
            non_null = series.dropna()
            fill = len(non_null) / n
            cardinality = int(non_null.nunique(dropna=True))
            unique_ratio = cardinality / max(len(non_null), 1)
            as_text = non_null.astype(str)
            avg_len = float(as_text.str.len().mean()) if not as_text.empty else 0.0
            median_len = float(as_text.str.len().median()) if not as_text.empty else 0.0
            text_hint = self._hint_score(name, TEXT_NAME_HINTS)
            label_hint = self._hint_score(name, LABEL_NAME_HINTS)
            metadata_penalty = 5.0 if normalized in METADATA_NAME_HINTS else 0.0

            text_score = text_hint + (3.0 * fill) + min(avg_len / 30.0, 6.0)
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                text_score += 3.0
            if cardinality <= 1:
                text_score -= 8.0
            if unique_ratio < 0.02 and cardinality > 10:
                text_score -= 2.0
            if avg_len < 4:
                text_score -= 3.0
            if median_len >= 15:
                text_score += 1.5
            text_score -= metadata_penalty

            label_score = label_hint + (3.5 * fill)
            if cardinality <= max(50, int(math.sqrt(n))):
                label_score += 4.5
            if cardinality <= 2:
                label_score += 2.5
            if cardinality > max(100, int(math.sqrt(n) * 8)):
                label_score -= 5.0
            if series_is_boolean_like(series):
                label_score += 3.0
            if avg_len < 40:
                label_score += 1.0
            label_score -= metadata_penalty
            if normalized in TEXT_NAME_HINTS:
                label_score -= 4.0

            candidates_text.append((name, round(text_score, 3)))
            candidates_label.append((name, round(label_score, 3)))

        candidates_text.sort(key=lambda x: x[1], reverse=True)
        candidates_label.sort(key=lambda x: x[1], reverse=True)
        result.text_candidates = candidates_text
        result.label_candidates = candidates_label

        # --- One-hot detection: also on the sample. ---
        one_hot: list[str] = []
        for col in scoring_df.columns:
            name = str(col)
            if name == result.text_column or normalize_column_name(name) in METADATA_NAME_HINTS:
                continue
            series = scoring_df[col].dropna()
            if series.empty:
                continue
            if series_is_boolean_like(series):
                one_hot.append(name)
                continue
            if pd.api.types.is_numeric_dtype(series):
                unique = set(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
                if unique and unique.issubset({0, 1}):
                    one_hot.append(name)
        result.one_hot_label_columns = one_hot

        # --- Resolution logic (unchanged except the tie-break uses scoring_df). ---
        if result.text_column is None and candidates_text:
            best_name, best_score = candidates_text[0]
            second_score = candidates_text[1][1] if len(candidates_text) > 1 else -math.inf

            if (
                len(candidates_text) > 1
                and abs(best_score - second_score) < 0.75
                and best_score >= 8.0
            ):
                tied = [
                    (name, score)
                    for name, score in candidates_text
                    if abs(score - best_score) < 0.75
                ]
                # NOTE: tie-break uses scoring_df, not df — the mean length we
                # already have is from the sample, and re-measuring it on 4M
                # rows would defeat the entire optimisation.
                tied.sort(
                    key=lambda x: -float(
                        scoring_df[x[0]].dropna().astype(str).str.len().mean() or 0.0
                    )
                )
                best_name, best_score = tied[0]

            if best_score >= 8.0 and (best_score - second_score >= 0.75 or len(candidates_text) == 1):
                result.text_column = best_name
            else:
                result.warnings.append("Text-column inference is ambiguous; pass --text-column.")

        if result.label_column is None and candidates_label:
            best_name, best_score = candidates_label[0]
            second_score = candidates_label[1][1] if len(candidates_label) > 1 else -math.inf
            if best_name != result.text_column and best_score >= 7.0 and (
                best_score - second_score >= 0.60 or len(candidates_label) == 1
            ):
                result.label_column = best_name
            elif one_hot:
                result.warnings.append("No confident scalar label column; one-hot label candidates are available.")
            else:
                result.warnings.append("Label-column inference is ambiguous; pass --label-column.")

        if result.text_column is None:
            raise ColumnDetectionError("No confident text column was detected. Pass --text-column COLUMN.")
        if result.label_column is None and not result.one_hot_label_columns:
            raise ColumnDetectionError("No confident label column was detected. Pass --label-column COLUMN.")
        if result.label_column == result.text_column:
            alternatives = [entry for entry in candidates_label if entry[0] != result.text_column]
            if alternatives and alternatives[0][1] >= 7.0:
                result.label_column = alternatives[0][0]
            elif not one_hot:
                raise ColumnDetectionError("Text and label detection resolved to the same column.")

        result.confidence = "high" if not result.warnings else "medium"
        return result


# =============================================================================
# PYTORCH / SENTIMENT
# =============================================================================

class SentimentScorer:
    """Target-independent sentiment scorer with native PyTorch transformer path."""

    def __init__(
        self,
        backend: str = "auto",
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str = "auto",
        dtype: str = "auto",
        cache_dir: Optional[str | Path] = None,
        positive_label: Optional[str] = None,
        negative_label: Optional[str] = None,
        neutral_label: Optional[str] = None,
        offline: bool = False,
        renderer: Optional[Renderer] = None,
    ):
        self.backend_request = str(backend).lower()
        self.model_name = model_name or DEFAULT_SENTIMENT_MODEL
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self.device_request = device
        self.dtype_request = dtype.lower()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.positive_label_override = positive_label
        self.negative_label_override = negative_label
        self.neutral_label_override = neutral_label
        self.offline = bool(offline)
        self.renderer = renderer or Renderer()

        self.resolved_backend: Optional[str] = None
        self.resolved_model: Optional[str] = None
        self.resolved_device: Optional[str] = None
        self.resolved_dtype: Optional[str] = None

        self._tokenizer = None
        self._model = None
        self._device = None
        self._label_signs: dict[int, float] = {}
        self._vader = None
        self._backend_locked: Optional[str] = None

    @staticmethod
    def _torch_version() -> tuple[int, int, int]:
        if torch is None:
            return (0, 0, 0)
        raw = str(getattr(torch, "__version__", "0.0.0"))
        match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", raw)
        if not match:
            return (0, 0, 0)
        return tuple(int(group or 0) for group in match.groups())  # type: ignore[return-value]

    def _choose_device(self):
        if torch is None:
            raise SentimentBackendError("PyTorch is not installed.")
        request = str(self.device_request or "auto").lower().strip()
        if request in {"", "auto"}:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if request in {"cpu", "-1"}:
            return torch.device("cpu")
        if request == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise SentimentBackendError("MPS was requested but is unavailable.")
            return torch.device("mps")
        if request in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise SentimentBackendError("CUDA was requested but torch.cuda.is_available() is False.")
            return torch.device("cuda")
        if re.fullmatch(r"cuda:\d+", request):
            if not torch.cuda.is_available():
                raise SentimentBackendError("CUDA is unavailable.")
            index = int(request.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise SentimentBackendError(
                    f"CUDA device {index} is unavailable; "
                    f"{torch.cuda.device_count()} device(s) detected."
                )
            return torch.device(request)
        if request.isdigit():
            if not torch.cuda.is_available():
                raise SentimentBackendError("Numeric device index was provided but CUDA is unavailable.")
            return torch.device(f"cuda:{request}")
        raise SentimentBackendError("Device must be auto, cpu, mps, cuda, cuda:N, or an integer CUDA index.")

    def _normalise_label(self, label: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", label.lower())

    def _sign(self, label: str) -> float:
        normalized = self._normalise_label(label)
        if self.positive_label_override and normalized == self._normalise_label(self.positive_label_override):
            return 1.0
        if self.negative_label_override and normalized == self._normalise_label(self.negative_label_override):
            return -1.0
        if self.neutral_label_override and normalized == self._normalise_label(self.neutral_label_override):
            return 0.0
        if any(token in normalized for token in ("positive", "pos", "good", "favorable", "favourable")):
            return 1.0
        if any(token in normalized for token in ("negative", "neg", "bad", "unfavorable", "unfavourable")):
            return -1.0
        if any(token in normalized for token in ("neutral", "neu")):
            return 0.0
        return 0.0

    @staticmethod
    def _import_transformers_quietly():
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            return AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            captured = (stdout_buffer.getvalue() + stderr_buffer.getvalue()).strip()
            raise SentimentBackendError(
                f"Could not import Transformers: {type(exc).__name__}: {exc}"
                + (f" | Environment message: {compact(captured, 300)}" if captured else "")
            ) from exc

    def _ensure_transformer(self) -> None:
        """
        Initialize the transformer sentiment backend.

        The load order is:

          1. prefetch_hf_snapshot — force a complete download using
             snapshot_download for weights and curl for metadata. This is
             the only way to reliably get tokenizer files through a
             Cloudflare Worker proxy, which strips Content-Length from
             HEAD responses on small text files.

          2. from_pretrained against the resolved snapshot DIRECTORY,
             with local_files_only=True. Loading from a directory path
             bypasses huggingface_hub's lazy file resolution entirely, so
             no HEAD request is ever issued for the tokenizer.

        If prefetch fails (network down, proxy unreachable, model private)
        we fall back to loading by hub identifier exactly as before, which
        still works when the cache is already populated.
        
        """
        _configure_torch_threads()
        if torch is None:
            raise SentimentBackendError("Transformer scoring requires PyTorch, but torch is not importable.")

        # NOTE: transformers works on torch >= 2.0 in practice. Older torch
        # versions have been the source of confusing NameError failures, so
        # we reject them here with a clear message.
        if self._torch_version() < (2, 0, 0):
            raise SentimentBackendError(
                f"Installed PyTorch is {torch.__version__}. "
                "Transformer sentiment scoring requires PyTorch 2.0+. "
                "Upgrade with: pip install -U torch"
            )

        AutoModelForSequenceClassification, AutoTokenizer = self._import_transformers_quietly()
        self._device = self._choose_device()
        self.batch_size = _adaptive_sentiment_batch_size(
            self.batch_size,
            self._device,
            n_rows=10_000,  
        )
        self.resolved_device = str(self._device)

        # ----------------------------------------------------------------
        # Resolve a stable cache directory for the sentiment model.
        # ----------------------------------------------------------------
        cache_path = Path(self.cache_dir).expanduser().resolve() \
            if self.cache_dir is not None \
            else _default_hf_cache_dir()
        cache_path.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Prefetch. Skipped when the caller has asked for offline mode.
        # ----------------------------------------------------------------
        resolved_snapshot: Optional[Path] = None
        if not self.offline:
            resolved_snapshot = prefetch_hf_snapshot(
                self.model_name,
                cache_dir=cache_path,
                show=self.renderer.enabled,
            )

        # ----------------------------------------------------------------
        # Determine the load target and offline flag.
        #
        # With a resolved snapshot: load from the directory, force
        #   local_files_only=True so no network round trip is attempted.
        # Without: fall back to hub identifier + self.offline flag, which
        #   is exactly the previous behaviour.
        # ----------------------------------------------------------------
        if resolved_snapshot is not None:
            load_target: str = str(resolved_snapshot)
            local_files_only = True
        else:
            load_target = self.model_name
            local_files_only = bool(self.offline)

        # ----------------------------------------------------------------
        # Runtime table.
        # ----------------------------------------------------------------
        diag = _torch_device_report()
        self.renderer.table(
            "PyTorch SENTIMENT RUNTIME",
            ["Component", "Value"],
            [
                ["Framework", "PyTorch"],
                ["Model", self.model_name],
                ["Device", str(self._device)],
                ["Torch", str(torch.__version__)],
                ["Offline", str(local_files_only)],
                ["Batch", str(self.batch_size)],
                ["Max length", str(self.max_length)],
                ["Snapshot", str(resolved_snapshot) if resolved_snapshot else "cache"],
                ["Cache", str(cache_path)],
                ["CPU count", str(diag.get("cpu_count"))],
                ["Torch threads", str(diag.get("num_threads"))],
                ["MPS built", str(diag.get("mps_built"))],
                ["MPS available", str(diag.get("mps_available"))],
            ],
        )

        # ----------------------------------------------------------------
        # Load tokenizer and model.
        # ----------------------------------------------------------------
        _patch_transformers_torch_load_check()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                load_target,
                cache_dir=str(cache_path),
                local_files_only=local_files_only,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                load_target,
                cache_dir=str(cache_path),
                local_files_only=local_files_only,
            )
            self._model.eval()
            self._model.to(self._device)
        except Exception as exc:
            raise SentimentBackendError(
                f"Could not initialize PyTorch transformer '{self.model_name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # ----------------------------------------------------------------
        # Label-sign extraction (unchanged).
        # ----------------------------------------------------------------
        id2label = getattr(self._model.config, "id2label", {}) or {}
        self._label_signs = {}
        for raw_id, raw_label in id2label.items():
            try:
                idx = int(raw_id)
            except (TypeError, ValueError):
                continue
            self._label_signs[idx] = self._sign(str(raw_label))

        if not any(v < 0 for v in self._label_signs.values()) or not any(v > 0 for v in self._label_signs.values()):
            raise SentimentBackendError(
                "The selected transformer does not expose recognizable positive/negative labels. "
                "Use a sentiment classifier or pass --positive-label / --negative-label."
            )

        self.resolved_backend = "transformer"
        self.resolved_model = self.model_name
        self.resolved_dtype = "float32 logits + probability softmax"

        self.renderer.table(
            "MODEL LABEL MAP",
            ["Index", "Model label", "Signed contribution"],
            [
                [idx, id2label.get(idx, id2label.get(str(idx), f"LABEL_{idx}")), sign]
                for idx, sign in sorted(self._label_signs.items())
            ],
        )

    def _ensure_vader(self) -> None:
        if SentimentIntensityAnalyzer is None:
            raise SentimentBackendError(
                "VADER is unavailable. Install vaderSentiment or use transformer scoring."
            )
        self._vader = SentimentIntensityAnalyzer()
        self.resolved_backend = "vader"
        self.resolved_model = "vaderSentiment"
        self.resolved_device = "cpu"
        self.resolved_dtype = "float64"

    @staticmethod
    def _lexicon_score(text: str) -> float:
        positive = {
            "amazing", "awesome", "excellent", "good", "great", "happy", "joy", "love", "lovely",
            "perfect", "wonderful", "fantastic", "brilliant", "delighted", "hope", "grateful",
            "thanks", "thank", "fun", "funny", "best", "win", "winning", "beautiful",
        }
        negative = {
            "awful", "bad", "terrible", "horrible", "sad", "hate", "angry", "anger", "disappointed",
            "disgusting", "fear", "worst", "pain", "poor", "annoyed", "annoying", "regret", "grief",
            "stupid", "dumb", "fuck", "shit", "motherfucker", "sucks",
        }
        tokens = re.findall(r"\b[\w']+\b", text.lower(), flags=re.UNICODE)
        if not tokens:
            return 0.0
        positive_count = sum(token in positive for token in tokens)
        negative_count = sum(token in negative for token in tokens)
        raw = (positive_count - negative_count) / max(1, positive_count + negative_count)
        return float(np.clip(raw, -1.0, 1.0))

    def _choose_backend(self) -> str:
        if self.backend_request == "none":
            return "none"
        if self.backend_request in {"transformer", "vader", "lexicon"}:
            return self.backend_request
        if self.backend_request != "auto":
            raise SentimentBackendError(
                "Backend must be auto, transformer, vader, or lexicon."
            )

        # AUTO: prefer transformer whenever PyTorch is at a realistic baseline.
        if torch is not None and self._torch_version() >= (2, 0, 0):
            return "transformer"

        if SentimentIntensityAnalyzer is not None:
            torch_state = "not installed" if torch is None else f"torch {torch.__version__} (<2.0)"
            self.renderer.warning(
                f"Transformer backend unavailable ({torch_state}). "
                "Falling back to VADER. For transformer sentiment run:\n"
                "    pip install -U torch transformers"
            )
            return "vader"

        self.renderer.warning(
            "Neither PyTorch nor VADER is available. "
            "Falling back to the built-in lexicon scorer."
        )
        return "lexicon"

    def _transformer_score_batch(self, texts: Sequence[str]) -> list[float]:
        assert torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        try:
            encoded = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits.float(), dim=-1)
                signs = torch.tensor(
                    [self._label_signs.get(i, 0.0) for i in range(probabilities.shape[-1])],
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
                scores = torch.sum(probabilities * signs, dim=-1)
                scores = torch.clamp(scores, -1.0, 1.0)
                return [float(x) for x in scores.detach().cpu().tolist()]
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise SentimentBackendError(
                    f"PyTorch out-of-memory during sentiment inference at batch={len(texts)}. "
                    "Reduce --batch-size or use --device cpu."
                ) from exc
            raise SentimentBackendError(f"PyTorch inference failed: {type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            raise SentimentBackendError(f"Transformer inference failed: {type(exc).__name__}: {exc}") from exc

    def score_batch(self, texts: Sequence[str]) -> list[float]:
        # Short-circuit: disabled backend -> zeros, no model, no tokenizer.
        if self.backend_request == "none":
            self.resolved_backend = "none"
            self.resolved_model = "sentiment-disabled"
            self.resolved_device = "cpu"
            self.resolved_dtype = "float64"
            return [0.0 for _ in texts]

        if self._backend_locked is not None:
            backend = self._backend_locked
        else:
            backend = self._choose_backend()

        if backend == "transformer":
            try:
                if self.resolved_backend != "transformer":
                    self._ensure_transformer()
                return self._transformer_score_batch(texts)
            except SentimentBackendError as exc:
                if self.backend_request != "auto":
                    raise
                self.renderer.warning(
                    f"Transformer backend failed ({exc}). "
                    "Falling back to VADER for the remainder of this run. "
                    "No further retries will be attempted."
                )
                self._backend_locked = "vader"
                backend = "vader"

        if backend == "vader":
            if self.resolved_backend != "vader":
                self._ensure_vader()
            return [
                float(np.clip(self._vader.polarity_scores(str(t))["compound"], -1.0, 1.0))
                for t in texts
            ]

        self.resolved_backend = "lexicon"
        self.resolved_model = "built-in-lexicon"
        self.resolved_device = "cpu"
        self.resolved_dtype = "float64"
        return [self._lexicon_score(str(t)) for t in texts]


# =============================================================================
# PROCESSOR
# =============================================================================

class MasterDatasetProcessor:
    def __init__(
        self,
        dataset_link: str,
        *,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        sentiment_backend: str = "auto",
        sentiment_model: Optional[str] = None,
        sentiment_batch_size: int = DEFAULT_BATCH_SIZE,
        sentiment_max_length: int = DEFAULT_MAX_LENGTH,
        device: str = "auto",
        dtype: str = "auto",
        cache_dir: Optional[str | Path] = None,
        hf_config: Optional[str] = None,
        positive_label: Optional[str] = None,
        negative_label: Optional[str] = None,
        neutral_label: Optional[str] = None,
        offline: bool = False,
        lowercase: bool = True,
        demojize: bool = True,
        normalize_urls: bool = True,
        normalize_usernames: bool = True,
        normalize_hashtags: bool = False,
        strip_html: bool = True,
        normalize_repeated_punctuation: bool = True,
        drop_missing_text: bool = True,
        drop_missing_label: bool = True,
        drop_empty_text: bool = True,
        drop_short_text: bool = False,
        min_text_chars: int = 1,
        drop_duplicates: bool = True,
        interactive_fallback: bool = False,
        quiet: bool = False,
        no_visuals: bool = False,
    ):
        self.dataset_link = str(dataset_link)
        self.offline = bool(offline)
        self.text_column_override = text_column
        self.label_column_override = label_column
        self.hf_config = hf_config
        self.renderer = Renderer(quiet=quiet, no_visuals=no_visuals)
        self.loader = DatasetLoader(self.renderer, cache_dir=cache_dir)
        self.detector = SchemaDetector(self.renderer)
        self.cleaner = TextCleaner(
            lowercase=lowercase,
            demojize=demojize,
            normalize_urls=normalize_urls,
            normalize_usernames=normalize_usernames,
            normalize_hashtags=normalize_hashtags,
            strip_html=strip_html,
            normalize_repeated_punctuation=normalize_repeated_punctuation,
        )
        self.sentiment_scorer = SentimentScorer(
            backend=sentiment_backend,
            model_name=sentiment_model,
            batch_size=sentiment_batch_size,
            max_length=sentiment_max_length,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            positive_label=positive_label,
            negative_label=negative_label,
            neutral_label=neutral_label,
            offline=offline,
            renderer=self.renderer,
        )
        self.drop_missing_text = drop_missing_text
        self.drop_missing_label = drop_missing_label
        self.drop_empty_text = drop_empty_text
        self.drop_short_text = drop_short_text
        self.min_text_chars = max(1, int(min_text_chars))
        self.drop_duplicates = drop_duplicates

        self.raw_df: Optional[pd.DataFrame] = None
        self.interactive_fallback = bool(interactive_fallback)
        self.df: Optional[pd.DataFrame] = None
        self.source_path: Optional[Path] = None
        self.detection: Optional[DetectionResult] = None
        self.report = ProcessingReport(
            processor_version=VERSION,
            source=self.dataset_link,
            sentiment_backend_requested=sentiment_backend,
            sentiment_model=sentiment_model,
            batch_size=sentiment_batch_size,
            max_length=sentiment_max_length,
        )
        self._processed = False
        self._profile_cache: Optional[dict[str, Any]] = None
        self._score_cache: dict[str, float] = {}

    # -------------------------------------------------------------------------
    # LOAD / SCHEMA
    # -------------------------------------------------------------------------
    def _resolve_source_path_for_sampling(self) -> Optional[Path]:
        """
        Return the on-disk raw CSV for this source, without loading it.

        The preview and inspect paths use this to read only the rows they
        need, bypassing load() entirely.

        Returns None when the source has no on-disk snapshot yet (e.g. a
        managed dataset that has never been acquired, or an hf:// source
        whose raw CSV has not been written).
        """
        # Already resolved by a previous load() call.
        if self.source_path is not None and Path(self.source_path).is_file():
            return Path(self.source_path)

        if self.dataset_link.startswith("known://"):
            raw_key = self.dataset_link[len("known://"):].strip()
            key = resolve_known_dataset_key(raw_key)
            if key is None:
                return None
            p = known_dataset_local_path(key)
            return p if p.is_file() and p.stat().st_size > 0 else None

        if is_url(self.dataset_link) or self.dataset_link.startswith("hf://"):
            name = dataset_name_from_source(self.dataset_link)
            p = dataset_raw_path(name)
            return p if p.is_file() and p.stat().st_size > 0 else None

        p = Path(self.dataset_link).expanduser().resolve()
        return p if p.is_file() else None
        
    def _maybe_write_schema_sidecar(self) -> None:
        if self.detection is None or not self.detection.text_column:
            return
        if self.text_column_override is not None or self.label_column_override is not None:
            return

        key_for_sidecar = resolve_known_dataset_key(
            self.dataset_link[len("known://"):]
            if self.dataset_link.startswith("known://")
            else dataset_name_from_source(self.dataset_link)
        )
        if not key_for_sidecar:
            return

        # Fill in task metadata if the detector did not already supply it.
        # Sampling a bounded slice keeps this cheap even on a 4M-row frame.
        if (
            self.detection.label_column
            and (self.detection.task_type is None or self.detection.class_count is None)
            and self.raw_df is not None
            and self.detection.label_column in self.raw_df.columns
        ):
            try:
                col = self.raw_df[self.detection.label_column]
                if len(col) > 50_000:
                    col = col.sample(50_000, random_state=42)
                label_series = col.map(LabelNormalizer.parse)
                task_type, class_count = infer_task_and_class_count(label_series)
                self.detection.task_type = task_type
                self.detection.class_count = class_count
            except Exception:
                pass

        write_schema_sidecar(key_for_sidecar, self.detection)
    def load(self, *, force: bool = False) -> pd.DataFrame:
        if self.raw_df is not None and not force:
            return self.raw_df.copy()

        frame, source_path = self.loader.load(
            self.dataset_link,
            hf_config=self.hf_config,
            offline=self.offline,
        )

        self.raw_df = frame.copy()
        self.source_path = source_path
        self.report.rows_input = len(frame)
        self.report.columns_input = len(frame.columns)

        self.renderer.table(
            "DATASET INGESTION",
            ["Property", "Value"],
            [
                ["Source", compact(self.dataset_link, 120)],
                ["Rows", f"{len(frame):,}"],
                ["Columns", f"{len(frame.columns):,}"],
                ["Raw snapshot", str(source_path) if source_path else "—"],
            ],
        )
        return frame.copy()

    def detect_schema(self) -> DetectionResult:
        if self.raw_df is None:
            self.load()
        assert self.raw_df is not None

        # ---------------------------------------------------------------------
        # MANAGED / DISCOVERED VIA known://
        # ---------------------------------------------------------------------
        if self.dataset_link.startswith("known://"):
            raw_key = self.dataset_link[len("known://"):].strip()
            key = resolve_known_dataset_key(raw_key)
            if key is None:
                raise DatasetSchemaError(f"Unknown managed dataset: {raw_key}")

            spec = all_datasets()[key]
            expected_text = spec.get("text_column")
            expected_label = spec.get("label_column")

            # NOTE: a "discovered" dataset deliberately has text_column=None
            # and label_column=None. Only managed datasets prescribe a schema.
            # Anything without a prescribed schema falls through to the same
            # auto-detection used for foreign sources.
            if (
                expected_text is not None
                and expected_label is not None
                and not spec.get("schema_is_hint")):

                if expected_text not in self.raw_df.columns:
                    raise DatasetSchemaError(
                        f"Managed dataset '{key}' is missing its expected text "
                        f"column '{expected_text}'. "
                        f"Available: {list(self.raw_df.columns)}"
                    )

                if expected_label == "__VAD__":
                    required_vad = set(spec.get("target_columns", ["V", "A", "D"]))
                    missing_vad = sorted(required_vad - set(self.raw_df.columns))
                    if missing_vad:
                        raise DatasetSchemaError(
                            f"Managed EmoBank data is missing VAD columns: "
                            f"{missing_vad}"
                        )
                elif expected_label not in self.raw_df.columns:
                    raise DatasetSchemaError(
                        f"Managed dataset '{key}' is missing its expected label "
                        f"column '{expected_label}'. "
                        f"Available: {list(self.raw_df.columns)}"
                    )

                self.detection = DetectionResult(
                    text_column=expected_text,
                    label_column=expected_label,
                    confidence="high",
                )
                self.report.text_column = expected_text
                self.report.label_column = expected_label
                if self.detection is not None and self.detection.text_column:
                    # Persist for the next `--list-datasets` call.
                    key_for_sidecar = resolve_known_dataset_key(
                        self.dataset_link[len("known://"):]
                        if self.dataset_link.startswith("known://")
                        else dataset_name_from_source(self.dataset_link)
                    )
                    if key_for_sidecar:
                        self._maybe_write_schema_sidecar()
                return self.detection

        # ---------------------------------------------------------------------
        # FOREIGN SOURCE OR DISCOVERED-WITHOUT-SCHEMA
        # ---------------------------------------------------------------------
                # ---------------------------------------------------------------------
        # FOREIGN OR DISCOVERED SOURCE — try the sidecar before re-detecting.
        # ---------------------------------------------------------------------
        if not self.dataset_link.startswith("known://"):
            if self.text_column_override is None and self.label_column_override is None:
                sidecar_key = resolve_known_dataset_key(
                    dataset_name_from_source(self.dataset_link)
                )
                if sidecar_key:
                    sidecar = load_schema_sidecar(sidecar_key)
                    if sidecar and sidecar.get("text_column"):
                        self.detection = DetectionResult(
                            text_column=sidecar["text_column"],
                            label_column=sidecar.get("label_column"),
                            one_hot_label_columns=sidecar.get(
                                "one_hot_label_columns", []
                            ),
                            confidence="high",
                            task_type=sidecar.get("task_type"),
                            class_count=sidecar.get("class_count"),
                        )
                        self.report.text_column = self.detection.text_column
                        self.report.label_column = self.detection.label_column
                        return self.detection

        # Fall through to the existing auto-detection.
        try:
            self.detection = self.detector.detect(
                self.raw_df,
                text_column=self.text_column_override,
                label_column=self.label_column_override,
            )
        except ColumnDetectionError as exc:
            if not self.interactive_fallback:
                raise
            self.renderer.warning(
                f"Automatic detection failed ({exc}). "
                "Switching to interactive column choice."
            )
            return self.interactive_select_columns()
        self.report.text_column = self.detection.text_column
        self.report.label_column = self.detection.label_column
        if self.detection is not None and self.detection.text_column:
            # Persist for the next `--list-datasets` call.
            key_for_sidecar = resolve_known_dataset_key(
                self.dataset_link[len("known://"):]
                if self.dataset_link.startswith("known://")
                else dataset_name_from_source(self.dataset_link)
            )
            if key_for_sidecar:
                self._maybe_write_schema_sidecar()

        return self.detection

    def interactive_select_columns(self) -> DetectionResult:
        """
        Show a preview of the raw frame and let the user pick the text and
        label columns. The choices are stored as overrides so that any
        subsequent detect_schema / process call in this session uses them
        without re-prompting.

        Called automatically from detect_schema when automatic detection
        fails and interactive_fallback is enabled, and explicitly from the
        CLI --choose-columns flag.
        """
        if self.raw_df is None:
            self.load()
        assert self.raw_df is not None

        text_col, label_col = prompt_for_columns(
            self.raw_df,
            self.renderer,
            default_text=self.text_column_override,
            default_label=self.label_column_override,
        )
        if text_col is None or label_col is None:
            raise DatasetSchemaError(
                "Interactive column selection is unavailable in quiet / no-visuals mode."
            )

        # Persist as overrides so detect_schema / process pick them up.
        self.text_column_override = text_col
        self.label_column_override = label_col

        self.renderer.table(
            "USER COLUMN CHOICE",
            ["Role", "Column"],
            [["Text", text_col], ["Label", label_col]],
        )

        # Run detection with the explicit overrides.
        self.detection = self.detector.detect(
            self.raw_df,
            text_column=text_col,
            label_column=label_col,
        )
        self.report.text_column = self.detection.text_column
        self.report.label_column = self.detection.label_column
        return self.detection
    # -------------------------------------------------------------------------
    # LABEL SERIES
    # -------------------------------------------------------------------------

    def _label_series(self, frame: pd.DataFrame) -> pd.Series:
        assert self.detection is not None and self.detection.text_column is not None

        if self.detection.label_column == "__VAD__":
            missing = [c for c in ("V", "A", "D") if c not in frame.columns]
            if missing:
                raise DatasetSchemaError(f"EmoBank VAD target is missing: {missing}")
            return frame.apply(
                lambda row: [float(row["V"]), float(row["A"]), float(row["D"])],
                axis=1,
            )

        if self.detection.label_column:
            return frame[self.detection.label_column].map(LabelNormalizer.parse)

        assert self.detection.one_hot_label_columns
        return frame.apply(
            lambda row: LabelNormalizer.one_hot_row(row, self.detection.one_hot_label_columns),
            axis=1,
        )

    # -------------------------------------------------------------------------
    # CLEANING
    # -------------------------------------------------------------------------

    def _clean_dataframe(
    self,
    frame: pd.DataFrame,
    *,
    report: bool = True,
) -> pd.DataFrame:
        """
        Canonical cleaning pass.

        Parameters
        ----------
        frame : the raw DataFrame to clean.
        report : when True (default), mutate self.report with row-removal
                counters. The preview path passes False so a small sample
                does not corrupt the ledger that the eventual full run
                will write.
        """
        if self.detection is None:
            self.detect_schema()
        assert self.detection is not None and self.detection.text_column is not None

        text_source = frame[self.detection.text_column]
        labels = self._label_series(frame)
        cleaned_text = text_source.map(self.cleaner.clean)
        result = pd.DataFrame({"clean_text": cleaned_text, "label": labels}, index=frame.index)

        def _bump(field: str, count: int) -> None:
            if report:
                setattr(self.report, field, getattr(self.report, field) + int(count))

        if self.drop_missing_text:
            mask = text_source.map(try_is_missing)
            _bump("missing_text_rows_removed", int(mask.sum()))
            result = result.loc[~mask]

        if self.drop_empty_text:
            mask = result["clean_text"].astype(str).str.strip().eq("")
            _bump("empty_text_rows_removed", int(mask.sum()))
            result = result.loc[~mask]

        if self.drop_short_text:
            mask = result["clean_text"].astype(str).str.len() < self.min_text_chars
            result = result.loc[~mask]

        if self.drop_missing_label:
            mask = result["label"].map(lambda value: value is None or len(value) == 0)
            _bump("missing_label_rows_removed", int(mask.sum()))
            result = result.loc[~mask]

        if self.drop_duplicates:
            before = len(result)
            # -----------------------------------------------------------------
            # NOTE: vectorised dedupe.
            #
            # The previous implementation called
            #     result.apply(lambda row: stable_hash({...}), axis=1)
            # which runs a Python function once per row and does
            # json.dumps + sha256 for every one of 4,000,000 rows.
            #
            # pd.util.hash_pandas_object is C-implemented and hashes the whole
            # column in a single pass. We build one combined string column
            # (clean_text + separator + canonical-label-JSON) and hash that.
            #
            # The separator "\x1f" is ASCII Unit Separator, chosen because it
            # cannot occur in normal text or in canonical_label_json output,
            # so ("a", "b|c") and ("a|b", "c") cannot collide.
            # -----------------------------------------------------------------
            label_strings = result["label"].map(
                lambda v: canonical_label_json(v) if isinstance(v, list) else str(v)
            )
            combined = result["clean_text"].astype(str) + "\x1f" + label_strings
            dedupe_key = pd.util.hash_pandas_object(combined, index=False)
            result = result.loc[~dedupe_key.duplicated()]
            _bump("duplicate_rows_removed", before - len(result))

        result["label"] = result["label"].map(LabelNormalizer.parse)
        result = result[result["label"].map(lambda x: isinstance(x, list) and len(x) > 0)]
        return result.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # SENTIMENT
    # -------------------------------------------------------------------------

    def _score_texts(self, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        batch_size = self.sentiment_scorer.batch_size
        total = len(texts)
        progress = self.renderer.progress("Sentiment scoring", total)

        if progress:
            with progress:
                task = progress.add_task("Scoring", total=total)
                for start in range(0, total, batch_size):
                    batch = list(texts[start:start + batch_size])
                    batch_scores = self.sentiment_scorer.score_batch(batch)
                    scores.extend(batch_scores)
                    progress.update(task, advance=len(batch))
        else:
            for start in range(0, total, batch_size):
                batch = list(texts[start:start + batch_size])
                scores.extend(self.sentiment_scorer.score_batch(batch))
        return scores

    def score(self, *, force: bool = False) -> pd.DataFrame:
        if self.df is None or "sentiment_score" not in self.df.columns:
            cleaned = (
                self._clean_dataframe(self.load())
                if self.df is None
                else self.df[["clean_text", "label"]].copy()
            )
        else:
            if force:
                cleaned = self.df[["clean_text", "label"]].copy()
                cleaned = cleaned.drop(columns=[c for c in ["sentiment_score"] if c in cleaned.columns])
            else:
                return self.df.copy()

        if cleaned.empty:
            raise DatasetValidationError("No rows remain after cleaning; sentiment cannot be scored.")

        self.renderer.table(
            "SENTIMENT PLAN",
            ["Metric", "Value"],
            [
                ["Rows to score", f"{len(cleaned):,}"],
                ["Backend requested", self.sentiment_scorer.backend_request],
                ["Model", self.sentiment_scorer.model_name],
                ["Batch size", self.sentiment_scorer.batch_size],
                ["Max length", self.sentiment_scorer.max_length],
                ["Device requested", self.sentiment_scorer.device_request],
            ],
        )
        # Adapt the batch size to the device and the workload size. This
        # is a no-op if the user already asked for a large batch.
        adapted = _adaptive_sentiment_batch_size(
            self.sentiment_scorer.batch_size,
            self.sentiment_scorer._device
                if self.sentiment_scorer._device is not None
                else torch.device("cpu") if torch else None,
            n_rows=len(cleaned),
        )
        # master_dataset.py :: MasterDatasetProcessor.score()

        # ------------------------------------------------------------------
        # Skip adaptive length when sentiment scoring is disabled.
        # There is no tokenizer to probe and no cost to amortise.
        # ------------------------------------------------------------------
        backend_request = self.sentiment_scorer.backend_request.lower()
        scoring_disabled = backend_request == "none"

        if not scoring_disabled:
            texts = cleaned["clean_text"].astype(str).tolist()
            tokenizer_for_len = getattr(self.sentiment_scorer, "_tokenizer", None)
            adapted_len = _adaptive_max_length(
                self.sentiment_scorer.max_length,
                texts,
                tokenizer=tokenizer_for_len,
            )
            if adapted_len != self.sentiment_scorer.max_length:
                self.renderer.info(
                    f"Max length adjusted from {self.sentiment_scorer.max_length} "
                    f"to {adapted_len} based on the text distribution (p99)."
                )
                self.sentiment_scorer.max_length = adapted_len
        config_key = stable_hash({
            "backend": self.sentiment_scorer.backend_request,
            "model": self.sentiment_scorer.model_name,
            "max_length": self.sentiment_scorer.max_length,
        })
                # Merge the on-disk cache into the in-memory cache. This is what
        # makes re-runs nearly instant on the second invocation.
        disk_cache = _load_sentiment_cache(config_key)
        if disk_cache:
            self._score_cache.update(disk_cache)
            self.renderer.info(
                f"Loaded {len(disk_cache):,} cached sentiment scores from disk."
            )
        
        output_scores: list[float] = []
        uncached_positions: list[int] = []
        uncached_texts: list[str] = []

        for index, text in enumerate(texts):
            key = f"{config_key}:{stable_hash(text, 32)}"
            if not force and key in self._score_cache:
                output_scores.append(self._score_cache[key])
            else:
                output_scores.append(float("nan"))
                uncached_positions.append(index)
                uncached_texts.append(text)

        if uncached_texts:
            new_scores = self._score_texts(uncached_texts)
            if len(new_scores) != len(uncached_positions):
                raise SentimentBackendError(
                    f"Sentiment backend returned {len(new_scores)} scores for "
                    f"{len(uncached_positions)} texts."
                )
            for pos, text, score in zip(uncached_positions, uncached_texts, new_scores):
                value = float(score)
                output_scores[pos] = value
                self._score_cache[f"{config_key}:{stable_hash(text, 32)}"] = value
            # Persist the updated cache so the next run of the same config
            # is a no-op.
            _save_sentiment_cache(config_key, self._score_cache)
        score_series = pd.to_numeric(pd.Series(output_scores, index=cleaned.index), errors="coerce")
        finite = np.isfinite(score_series.to_numpy(dtype=float))
        self.report.invalid_sentiment_scores = int((~finite).sum())
        if not finite.all():
            raise DatasetValidationError(
                f"{self.report.invalid_sentiment_scores} invalid sentiment scores were produced."
            )

        cleaned["sentiment_score"] = score_series.clip(-1.0, 1.0).astype(float)
        cleaned["label"] = cleaned["label"].map(LabelNormalizer.parse)
        self.df = cleaned[OUTPUT_COLUMNS].reset_index(drop=True)

        self.report.sentiment_backend_resolved = self.sentiment_scorer.resolved_backend
        self.report.sentiment_model = self.sentiment_scorer.resolved_model
        self.report.device = self.sentiment_scorer.resolved_device
        self.report.dtype = self.sentiment_scorer.resolved_dtype
        return self.df.copy()

    # -------------------------------------------------------------------------
    # PROCESS
    # -------------------------------------------------------------------------

    def process(self, *, force: bool = False) -> pd.DataFrame:
        if self._processed and self.df is not None and not force:
            return self.df.copy()

        self.renderer.panel(
            "MASTER DATASET PROCESSOR",
            "Canonicalize  •  clean  •  score  •  validate  •  audit",
        )
        self.load()
        self.detect_schema()
        assert self.raw_df is not None

        cleaned = self._clean_dataframe(self.raw_df)
        self.renderer.table(
            "PREPROCESSING LEDGER",
            ["Metric", "Value"],
            [
                ["Input rows", f"{self.report.rows_input:,}"],
                ["Rows after cleaning", f"{len(cleaned):,}"],
                ["Missing text removed", f"{self.report.missing_text_rows_removed:,}"],
                ["Empty text removed", f"{self.report.empty_text_rows_removed:,}"],
                ["Missing labels removed", f"{self.report.missing_label_rows_removed:,}"],
                ["Duplicates removed", f"{self.report.duplicate_rows_removed:,}"],
                ["Canonical columns", ", ".join(OUTPUT_COLUMNS)],
            ],
        )

        self.df = cleaned
        self.score(force=force)
        self.validate(raise_on_error=True)

        self.report.rows_output = len(self.df)
        self.report.finished_at = time.time()
        self._processed = True

        self.renderer.table(
            "PROCESS COMPLETE",
            ["Metric", "Value"],
            [
                ["Rows", f"{self.report.rows_output:,}"],
                ["Columns", f"{len(self.df.columns)}"],
                ["Sentiment backend", self.report.sentiment_backend_resolved or "—"],
                ["Sentiment model", self.report.sentiment_model or "—"],
                ["Duration", f"{self.report.duration_seconds:.2f}s" if self.report.duration_seconds is not None else "—"],
            ],
        )
        self.renderer.success("Dataset is canonical, validated, and ready for downstream analysis.")
        return self.df.copy()

    # -------------------------------------------------------------------------
    # PROFILE
    # -------------------------------------------------------------------------

    def profile(
        self,
        *,
        sample: Optional[int] = DEFAULT_SAMPLE,
        top_labels: int = 15,
        top_words: int = 20,
    ) -> dict[str, Any]:
        frame = self.load()
        detection = self.detect_schema()
        used = frame if sample is None or sample >= len(frame) else frame.sample(sample, random_state=42)

        report: dict[str, Any] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "sample": len(used),
            "detection": asdict(detection),
            "columns_detail": [],
        }

        duplicate_rows = int(frame.duplicated().sum())
        memory_mb = float(frame.memory_usage(deep=True).sum() / (1024 ** 2))
        constant_columns = [str(c) for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
        null_cells = int(frame.isna().sum().sum())
        total_cells = max(1, frame.shape[0] * frame.shape[1])

        self.renderer.panel(
            "DATASET PROFILE",
            "Structure  •  schema inference  •  text diagnostics  •  label diagnostics  •  quality",
        )
        self.renderer.table(
            "STRUCTURE",
            ["Metric", "Value"],
            [
                ["Rows", f"{len(frame):,}"],
                ["Columns", f"{len(frame.columns):,}"],
                ["Profile sample", f"{len(used):,}"],
                ["Duplicate rows", f"{duplicate_rows:,}"],
                ["Duplicate rate", f"{duplicate_rows / max(len(frame), 1) * 100:.2f}%"],
                ["Memory", f"{memory_mb:.2f} MB"],
                ["Missing cells", f"{null_cells:,}"],
                ["Missing-cell rate", f"{null_cells / total_cells * 100:.2f}%"],
                ["Constant columns", f"{len(constant_columns)}"],
            ],
        )

        column_rows = []
        for column in frame.columns:
            series = frame[column]
            non_null = series.dropna()
            unique = int(series.nunique(dropna=True))
            missing_pct = float(series.isna().mean() * 100)
            top = series.mode(dropna=True)
            top_value = compact(top.iloc[0], 50) if not top.empty else "—"
            avg_len = float(non_null.astype(str).str.len().mean()) if not non_null.empty else 0.0
            role = (
                "TEXT" if str(column) == detection.text_column
                else "LABEL" if str(column) == detection.label_column
                else "OTHER"
            )
            if str(column) in detection.one_hot_label_columns:
                role = "ONE-HOT LABEL"
            column_rows.append([
                str(column), str(series.dtype), f"{series.notna().mean() * 100:.1f}%", f"{unique:,}",
                f"{missing_pct:.1f}%", f"{avg_len:.1f}", top_value, role,
            ])
            report["columns_detail"].append({
                "column": str(column),
                "dtype": str(series.dtype),
                "non_null_pct": float(series.notna().mean() * 100),
                "unique": unique,
                "missing_pct": missing_pct,
                "avg_string_length": avg_len,
                "role": role,
            })
        self.renderer.table(
            "COLUMN INTELLIGENCE",
            ["Column", "Dtype", "Non-null", "Unique", "Missing", "Mean chars", "Mode", "Role"],
            column_rows,
        )

        text = used[detection.text_column].map(self.cleaner.clean).astype(str)
        chars = text.str.len()
        words = text.str.findall(r"\b[\w'’-]+\b", flags=re.UNICODE)
        word_counts = words.map(len)
        token_counter = Counter(token.lower() for row in words for token in row if token)
        total_tokens = sum(token_counter.values())
        unique_tokens = len(token_counter)
        lexical_diversity = unique_tokens / max(total_tokens, 1)

        self.renderer.table(
            "TEXT DIAGNOSTICS",
            ["Metric", "Value"],
            [
                ["Characters mean", f"{chars.mean():.2f}"],
                ["Characters median", f"{chars.median():.2f}"],
                ["Characters p95", f"{chars.quantile(.95):.2f}"],
                ["Characters max", f"{chars.max():.0f}"],
                ["Words mean", f"{word_counts.mean():.2f}"],
                ["Words median", f"{word_counts.median():.2f}"],
                ["Words p95", f"{word_counts.quantile(.95):.2f}"],
                ["Empty strings", f"{int(text.str.strip().eq('').sum()):,}"],
                ["Very short (<3 chars)", f"{int((chars < 3).sum()):,}"],
                ["URLs", f"{int(text.str.count(TextCleaner.URL_RE).sum()):,}"],
                ["User mentions", f"{int(text.str.count(TextCleaner.USER_RE).sum()):,}"],
                ["Hashtags", f"{int(text.str.count(TextCleaner.HASHTAG_RE).sum()):,}"],
                ["Non-ASCII rows", f"{int(text.map(lambda x: any(ord(ch) > 127 for ch in x)).sum()):,}"],
                ["Emoji markers", f"{int(text.str.count(r':').sum()):,}"],
                ["Lexical diversity", f"{lexical_diversity:.4f}"],
                ["Mean punctuation density", f"{text.map(lambda x: sum(ch in '.,!?;:' for ch in x) / max(len(x),1)).mean():.4f}"],
                ["Mean digit density", f"{text.map(lambda x: sum(ch.isdigit() for ch in x) / max(len(x),1)).mean():.4f}"],
            ],
        )

        label_series = self._label_series_for_profile(frame, detection)

        is_continuous_target = (
            detection.label_column == "__VAD__"
            and all(col in frame.columns for col in ("V", "A", "D"))
        )

        label_lengths = label_series.map(lambda x: len(x) if isinstance(x, list) else 0)
        flattened = Counter(
            str(item)
            for labels in label_series.dropna()
            for item in (labels if isinstance(labels, list) else [])
        )
        single_ratio = float((label_lengths == 1).mean()) if len(label_lengths) else 0.0
        multi_ratio = float((label_lengths > 1).mean()) if len(label_lengths) else 0.0
        label_entropy = entropy_from_counts(flattened)

        if is_continuous_target:
            self.renderer.table(
                "CONTINUOUS TARGET DIAGNOSTICS",
                ["Dimension", "Min", "Mean", "Median", "Max", "Std"],
                [
                    [
                        col,
                        f"{pd.to_numeric(frame[col], errors='coerce').min():.4f}",
                        f"{pd.to_numeric(frame[col], errors='coerce').mean():.4f}",
                        f"{pd.to_numeric(frame[col], errors='coerce').median():.4f}",
                        f"{pd.to_numeric(frame[col], errors='coerce').max():.4f}",
                        f"{pd.to_numeric(frame[col], errors='coerce').std():.4f}",
                    ]
                    for col in ("V", "A", "D")
                ],
            )

        self.renderer.table(
            "LABEL DIAGNOSTICS",
            ["Metric", "Value"],
            [
                ["Rows with labels", f"{int(label_series.notna().sum()):,}"],
                ["Distinct atomic labels", f"{len(flattened):,}"],
                ["Distinct label combinations", f"{label_series.map(label_key).nunique():,}"],
                ["Single-label rows", f"{single_ratio * 100:.2f}%"],
                ["Multi-label rows", f"{multi_ratio * 100:.2f}%"],
                ["Mean labels / row", f"{label_lengths.mean():.3f}"],
                ["Median labels / row", f"{label_lengths.median():.3f}"],
                ["Label entropy (bits)", f"{label_entropy:.4f}"],
                ["Most common label", compact(flattened.most_common(1)[0][0], 60) if flattened else "—"],
                ["Most common label count", f"{flattened.most_common(1)[0][1]:,}" if flattened else "0"],
            ],
        )

        self.renderer.table(
            "TOP LABELS",
            ["Rank", "Label", "Count", "Share"],
            [
                [index, label, count, f"{count / max(len(label_series), 1) * 100:.2f}%"]
                for index, (label, count) in enumerate(flattened.most_common(top_labels), 1)
            ],
        )

        self.renderer.table(
            "TOP WORDS",
            ["Rank", "Token", "Count", "Share"],
            [
                [index, word, count, f"{count / max(total_tokens, 1) * 100:.3f}%"]
                for index, (word, count) in enumerate(token_counter.most_common(top_words), 1)
            ],
        )

        quality_rows = [
            ["Missing rows", int(frame[detection.text_column].isna().sum())],
            ["Blank text rows", int(text.str.strip().eq("").sum())],
            ["Duplicate raw rows", duplicate_rows],
            ["Constant columns", len(constant_columns)],
            ["Missing-cell rate", f"{null_cells / total_cells * 100:.2f}%"],
        ]
        self.renderer.table("QUALITY MATRIX", ["Check", "Value"], quality_rows)

        report["text"] = {
            "char_quantiles": quantiles(chars),
            "word_quantiles": quantiles(word_counts),
            "lexical_diversity": lexical_diversity,
        }
        report["labels"] = {
            "single_ratio": single_ratio,
            "multi_ratio": multi_ratio,
            "label_entropy": label_entropy,
            "distinct_atomic_labels": len(flattened),
        }
        report["quality"] = quality_rows
        self._profile_cache = report
        return report

    def _label_series_for_profile(
        self,
        frame: pd.DataFrame,
        detection: DetectionResult,
    ) -> pd.Series:
        if detection.label_column == "__VAD__":
            return frame.apply(
                lambda row: [float(row["V"]), float(row["A"]), float(row["D"])],
                axis=1,
            )
        if detection.label_column:
            return frame[detection.label_column].map(LabelNormalizer.parse)

        assert detection.one_hot_label_columns
        return frame.apply(
            lambda row: LabelNormalizer.one_hot_row(row, detection.one_hot_label_columns),
            axis=1,
        )

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    def validate(self, *, raise_on_error: bool = True) -> dict[str, Any]:
        if self.df is None:
            raise DatasetValidationError("Nothing has been processed. Run process() first.")

        df = self.df
        errors: list[str] = []
        warnings: list[str] = []

        schema_ok = list(df.columns) == OUTPUT_COLUMNS
        if not schema_ok:
            errors.append(f"Canonical schema mismatch: {list(df.columns)}")

        if df.empty:
            errors.append("Processed dataset is empty.")

        if "clean_text" in df.columns:
            empty = int(df["clean_text"].astype(str).str.strip().eq("").sum())
            if empty:
                errors.append(f"{empty} empty clean_text rows remain.")

        if "label" in df.columns:
            not_lists = int((~df["label"].map(lambda x: isinstance(x, list))).sum())
            empty_lists = int(df["label"].map(lambda x: isinstance(x, list) and len(x) == 0).sum())
            if not_lists:
                errors.append(f"{not_lists} label rows are not lists.")
            if empty_lists:
                errors.append(f"{empty_lists} label rows are empty lists.")

        if "sentiment_score" in df.columns:
            numeric = pd.to_numeric(df["sentiment_score"], errors="coerce")
            arr = numeric.to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                errors.append("sentiment_score contains NaN, Inf, or non-numeric values.")
            outside = int(((numeric < -1.0) | (numeric > 1.0)).sum())
            if outside:
                errors.append(f"{outside} sentiment scores lie outside [-1, 1].")
            if len(arr) and float(np.std(arr)) < 1e-8:
                warnings.append("Sentiment score variance is effectively zero.")

        if self.report.rows_input > 20 and len(df) < self.report.rows_input * 0.5:
            warnings.append("More than 50% of input rows were removed.")

        label_type_ok = "PASS" if "label" in df.columns and not any("label rows" in e for e in errors) else "FAIL"
        sentiment_ok = "PASS" if "sentiment_score" in df.columns and not any("sentiment" in e for e in errors) else "FAIL"
        text_ok = "PASS" if "clean_text" in df.columns and not any("empty clean_text" in e for e in errors) else "FAIL"
        schema_status = "PASS" if schema_ok else "FAIL"
        overall = "FAIL" if errors else "WARN" if warnings else "PASS"

        self.renderer.table(
            "VALIDATION MATRIX",
            ["Check", "Status", "Observed"],
            [
                ["Canonical schema", schema_status, str(list(df.columns))],
                ["Text non-empty", text_ok, f"{len(df):,} rows"],
                ["Labels are lists", label_type_ok, f"{len(df):,} rows"],
                ["Sentiment finite/in-range", sentiment_ok, "[-1, 1]"],
                ["Overall", overall, f"errors={len(errors)}, warnings={len(warnings)}"],
            ],
        )

        if "sentiment_score" in df.columns and not errors:
            stats = df["sentiment_score"].astype(float)
            self.renderer.table(
                "SENTIMENT DISTRIBUTION",
                ["Statistic", "Value"],
                [
                    ["Min", f"{stats.min():+.6f}"],
                    ["Q01", f"{stats.quantile(.01):+.6f}"],
                    ["Q25", f"{stats.quantile(.25):+.6f}"],
                    ["Median", f"{stats.median():+.6f}"],
                    ["Mean", f"{stats.mean():+.6f}"],
                    ["Q75", f"{stats.quantile(.75):+.6f}"],
                    ["Q99", f"{stats.quantile(.99):+.6f}"],
                    ["Max", f"{stats.max():+.6f}"],
                    ["Std", f"{stats.std(ddof=0):.6f}"],
                ],
            )

        if warnings:
            self.renderer.table("VALIDATION WARNINGS", ["#", "Message"], [[i, w] for i, w in enumerate(warnings, 1)])
        if errors:
            self.renderer.table("VALIDATION ERRORS", ["#", "Message"], [[i, e] for i, e in enumerate(errors, 1)])
            if raise_on_error:
                raise DatasetValidationError("; ".join(errors))
        else:
            self.renderer.success(f"Validation {overall}.")
        return {"status": overall, "errors": errors, "warnings": warnings}

    # -------------------------------------------------------------------------
    # PREVIEW / SUMMARY
    # -------------------------------------------------------------------------

    def preview(
    self,
    rows: int = 10,
    *,
    tail: bool = False,
    random_sample: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
        """
        Render a small canonical sample without paying for the full pipeline
        or for a full DataFrame load.

        The path is chosen to be the cheapest one that answers the request:

        1. Full pipeline already ran        -> sample self.df in memory.
        2. Raw CSV available on disk        -> read only `rows` rows from
                                                disk, then clean the sample.
        3. Raw source must be acquired      -> fall back to load() (rare).

        In cases 1 and 2, nothing larger than `rows` is ever materialised.
        """
        n = max(1, int(rows))
        mode = "random" if random_sample else ("tail" if tail else "head")

        # --- Path 1 — the full pipeline already ran --------------------------
        if self.df is not None and not self.df.empty:
            sample = self._sample(
                self.df, n, tail=tail, random_sample=random_sample, seed=seed,
            )
            self._render_preview(sample, title="PROCESSED PREVIEW")
            return sample.copy()

        # --- Path 2 — sample directly from the raw CSV -----------------------
        source_path = self._resolve_source_path_for_sampling()
        if source_path is not None:
            sample_raw = DatasetLoader.read_rows_only(
                source_path, n, mode=mode, seed=seed,
            )
            if sample_raw.empty:
                raise DatasetValidationError(
                    f"No rows available for preview at {source_path}"
                )

            # Detection needs to know the schema, but we must not pay for a
            # full load. If it is not already cached, run it on the sample.
            if self.detection is None:
                self.detection = self.detector.detect(
                    sample_raw,
                    text_column=self.text_column_override,
                    label_column=self.label_column_override,
                )
                self.report.text_column = self.detection.text_column
                self.report.label_column = self.detection.label_column

            clean = self._clean_dataframe(sample_raw, report=False)
            clean["sentiment_score"] = float("nan")
            clean = clean[OUTPUT_COLUMNS].reset_index(drop=True)

            self._render_preview(
                clean,
                title=f"RAW {mode.upper()} PREVIEW ({n} rows, sentiment not computed)",
            )
            return clean

        # --- Path 3 — nothing on disk yet; must acquire and load -------------
        raw = self.load()
        if raw.empty:
            raise DatasetValidationError("No rows available for preview.")
        if self.detection is None:
            self.detect_schema()
        sample_raw = self._sample(
            raw, n, tail=tail, random_sample=random_sample, seed=seed,
        )
        clean = self._clean_dataframe(sample_raw, report=False)
        clean["sentiment_score"] = float("nan")
        clean = clean[OUTPUT_COLUMNS].reset_index(drop=True)
        self._render_preview(
            clean, title="RAW PREVIEW (sentiment not yet computed)",
        )
        return clean

    @staticmethod
    def _sample(
        frame: pd.DataFrame,
        n: int,
        *,
        tail: bool,
        random_sample: bool,
        seed: int,
    ) -> pd.DataFrame:
        n = min(n, len(frame))
        if random_sample:
            return frame.sample(n=n, random_state=seed)
        if tail:
            return frame.tail(n).copy()
        return frame.head(n).copy()

    def _render_preview(self, sample: pd.DataFrame, *, title: str) -> None:
        def _score_cell(value):
            try:
                f = float(value)
            except (TypeError, ValueError):
                return "—"
            if math.isnan(f):
                return "—"
            return f"{f:+.6f}"

        self.renderer.table(
            title,
            OUTPUT_COLUMNS,
            [
                [
                    compact(row.clean_text, 120),
                    canonical_label_json(row.label),
                    _score_cell(row.sentiment_score),
                ]
                for row in sample.itertuples(index=False)
            ],
            show_lines=True,
        )

    def summary(self) -> dict[str, Any]:
        if self.df is None:
            self.process()
        assert self.df is not None
        scores = self.df["sentiment_score"].astype(float)
        label_lengths = self.df["label"].map(len)
        flattened = Counter(str(item) for labels in self.df["label"] for item in labels)
        summary = {
            "rows": len(self.df),
            "single_label_ratio": float((label_lengths == 1).mean()),
            "multi_label_ratio": float((label_lengths > 1).mean()),
            "mean_labels": float(label_lengths.mean()),
            "sentiment": asdict(self.report),
        }
        self.renderer.panel("DATASET SUMMARY", "Canonical output state and target-independent sentiment statistics")
        self.renderer.table(
            "CORE SUMMARY",
            ["Metric", "Value"],
            [
                ["Rows", f"{len(self.df):,}"],
                ["Columns", ", ".join(self.df.columns)],
                ["Single-label rows", f"{summary['single_label_ratio'] * 100:.2f}%"],
                ["Multi-label rows", f"{summary['multi_label_ratio'] * 100:.2f}%"],
                ["Mean labels / row", f"{summary['mean_labels']:.3f}"],
                ["Sentiment backend", self.report.sentiment_backend_resolved or "—"],
                ["Sentiment model", self.report.sentiment_model or "—"],
            ],
        )
        self.renderer.table(
            "SENTIMENT SUMMARY",
            ["Statistic", "Value"],
            [
                ["Minimum", f"{scores.min():+.6f}"],
                ["Q01", f"{scores.quantile(.01):+.6f}"],
                ["Q25", f"{scores.quantile(.25):+.6f}"],
                ["Median", f"{scores.median():+.6f}"],
                ["Mean", f"{scores.mean():+.6f}"],
                ["Q75", f"{scores.quantile(.75):+.6f}"],
                ["Q99", f"{scores.quantile(.99):+.6f}"],
                ["Maximum", f"{scores.max():+.6f}"],
                ["Std", f"{scores.std(ddof=0):.6f}"],
            ],
        )
        self.renderer.table(
            "LABEL SUMMARY",
            ["Rank", "Label", "Count", "Share"],
            [
                [index, label, count, f"{count / max(len(self.df), 1) * 100:.2f}%"]
                for index, (label, count) in enumerate(flattened.most_common(15), 1)
            ],
        )
        return summary

    # -------------------------------------------------------------------------
    # SAVE / MANIFEST
    # -------------------------------------------------------------------------

    def _serializable_frame(self, fmt: str) -> pd.DataFrame:
        assert self.df is not None
        frame = self.df.copy()
        if fmt == "csv":
            frame["label"] = frame["label"].map(canonical_label_json)
        elif fmt == "jsonl":
            frame["label"] = frame["label"].map(lambda x: safe_json_value(x))
        return frame

    def _manifest(self, output_path: Path, fmt: str) -> dict[str, Any]:
        source_hash = (
            file_hash(self.source_path)
            if self.source_path and self.source_path.exists()
            else stable_hash(self.dataset_link)
        )
        config = {
            "version": VERSION,
            "source": self.dataset_link,
            "source_hash": source_hash,
            "text_column": self.report.text_column,
            "label_column": self.report.label_column,
            "sentiment_backend_requested": self.report.sentiment_backend_requested,
            "sentiment_backend_resolved": self.report.sentiment_backend_resolved,
            "sentiment_model": self.report.sentiment_model,
            "device": self.report.device,
            "dtype": self.report.dtype,
            "batch_size": self.report.batch_size,
            "max_length": self.report.max_length,
            "cleaning": {
                "lowercase": self.cleaner.lowercase,
                "demojize": self.cleaner.demojize,
                "normalize_urls": self.cleaner.normalize_urls,
                "normalize_usernames": self.cleaner.normalize_usernames,
                "normalize_hashtags": self.cleaner.normalize_hashtags,
                "strip_html": self.cleaner.strip_html,
                "normalize_repeated_punctuation": self.cleaner.normalize_repeated_punctuation,
            },
            "canonical_output_columns": OUTPUT_COLUMNS,
            "canonical_label_contract": "Python list in memory; JSON array string in CSV",
        }
        return {
            "processor": config,
            "output": {"path": str(output_path), "format": fmt},
            "report": asdict(self.report),
        }

    def save(
        self,
        output: str | Path,
        *,
        fmt: Optional[str] = None,
        overwrite: bool = False,
        manifest: bool = True,
    ) -> Path:
        if self.df is None:
            self.process()
        assert self.df is not None
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise DatasetProcessorError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )

        inferred = output_path.suffix.lower().lstrip(".")
        resolved_fmt = (fmt or inferred or "csv").lower()
        if resolved_fmt == "pq":
            resolved_fmt = "parquet"
        if resolved_fmt not in {"csv", "jsonl", "json", "parquet", "xlsx"}:
            raise DatasetProcessorError(
                "Output format must be csv, jsonl, json, parquet, or xlsx."
            )

        frame = self._serializable_frame("jsonl" if resolved_fmt == "jsonl" else resolved_fmt)
        try:
            if resolved_fmt == "csv":
                frame.to_csv(output_path, index=False)
            elif resolved_fmt == "jsonl":
                frame.to_json(output_path, orient="records", lines=True, force_ascii=False)
            elif resolved_fmt == "json":
                frame.to_json(output_path, orient="records", force_ascii=False, indent=2)
            elif resolved_fmt == "parquet":
                frame.to_parquet(output_path, index=False)
            elif resolved_fmt == "xlsx":
                frame.to_excel(output_path, index=False)
        except Exception as exc:
            raise DatasetProcessorError(
                f"Failed writing '{output_path}': {type(exc).__name__}: {exc}"
            ) from exc

        if manifest:
            manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
            manifest_path.write_text(
                json.dumps(self._manifest(output_path, resolved_fmt), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        self.renderer.table(
            "DATASET SAVED",
            ["Property", "Value"],
            [
                ["Output", str(output_path)],
                ["Format", resolved_fmt],
                ["Rows", f"{len(self.df):,}"],
                ["Columns", ", ".join(OUTPUT_COLUMNS)],
                ["Manifest", "written" if manifest else "disabled"],
            ],
        )
        return output_path

    # -------------------------------------------------------------------------
    # CONFIG / INTROSPECTION
    # -------------------------------------------------------------------------

    def configuration(self) -> dict[str, Any]:
        return {
            "source": self.dataset_link,
            "text_column": self.text_column_override or "auto",
            "label_column": self.label_column_override or "auto",
            "backend": self.sentiment_scorer.backend_request,
            "model": self.sentiment_scorer.model_name,
            "batch_size": self.sentiment_scorer.batch_size,
            "max_length": self.sentiment_scorer.max_length,
            "device": self.sentiment_scorer.device_request,
            "dtype": self.sentiment_scorer.dtype_request,
            "offline": self.sentiment_scorer.offline,
            "drop_duplicates": self.drop_duplicates,
            "drop_missing_text": self.drop_missing_text,
            "drop_missing_label": self.drop_missing_label,
            "drop_empty_text": self.drop_empty_text,
            "drop_short_text": self.drop_short_text,
            "min_text_chars": self.min_text_chars,
        }

    def show_configuration(self) -> None:
        config = self.configuration()
        self.renderer.table(
            "ACTIVE CONFIGURATION",
            ["Key", "Value"],
            [[k, compact(v, 120)] for k, v in config.items()],
        )


# =============================================================================
# HELP SYSTEM
# =============================================================================

COMMAND_INFO: dict[str, dict[str, Any]] = {
    "inspect": {
        "purpose": "Load a source and inspect schema detection without modifying the dataset.",
        "usage": "python master_dataset.py inspect DATASET [options]",
        "options": [
            ("DATASET", "Source path, URL, hf://dataset, or known://key"),
            ("-t, --text-column", "Explicit text field"),
            ("-l, --label-column", "Explicit label field"),
            ("--candidates", "Show extended detection candidates"),
        ],
    },
    "profile": {
        "purpose": "Produce a deep structural, text, label, and quality profile.",
        "usage": "python master_dataset.py profile DATASET [options]",
        "options": [
            ("--sample N", "Rows used for text diagnostics; omit for full dataset"),
            ("--top-labels N", "Number of label rows to show"),
            ("--top-words N", "Number of lexical rows to show"),
            ("-t, --text-column", "Explicit text field"),
            ("-l, --label-column", "Explicit label field"),
        ],
    },
    "process": {
        "purpose": "Canonicalise labels, clean text, calculate sentiment, validate, optionally save.",
        "usage": "python master_dataset.py process DATASET [options]",
        "options": [
            ("-o, --output FILE", "Save the canonical dataset"),
            ("--format FORMAT", "csv | jsonl | json | parquet | xlsx"),
            ("--overwrite", "Permit replacement of an existing output"),
            ("-B, --sentiment-backend", "auto | transformer | vader | lexicon"),
            ("-m, --model", "Sentiment model identifier"),
            ("-b, --batch-size", "PyTorch inference batch size"),
            ("-L, --max-length", "Tokenizer maximum sequence length"),
            ("-d, --device", "auto | cpu | mps | cuda | cuda:N"),
            ("--dtype", "auto | float32 | float16 | bfloat16"),
            ("--offline", "Only use locally cached model assets"),
            ("--positive-label", "Override positive model class name"),
            ("--negative-label", "Override negative model class name"),
            ("--neutral-label", "Override neutral model class name"),
            ("--no-deduplicate", "Keep duplicate text/label rows"),
            ("--keep-missing-text", "Do not remove missing source text"),
            ("--keep-missing-label", "Do not remove missing labels"),
            ("--keep-empty-text", "Do not remove empty cleaned text"),
            ("--drop-short N", "Drop cleaned text shorter than N characters"),
            ("--no-lowercase", "Preserve text case"),
            ("--no-demojize", "Preserve Unicode emoji instead of textual names"),
            ("--no-url-normalization", "Preserve URL strings"),
            ("--no-user-normalization", "Preserve @mentions"),
            ("--normalize-hashtags", "Convert #tag to tag"),
            ("--no-html-strip", "Preserve HTML markup"),
            ("--no-manifest", "Do not write provenance manifest"),
        ],
    },
    "prepare": {
        "purpose": "Download, preprocess, and save a known dataset in one command.",
        "usage": (
            "python master_dataset.py prepare "
            "--dataset {goemo,isear,empathetic,emobank} "
            "[-o OUTPUT] [--offline]"
        ),
        "options": [
            ("--dataset", "Dataset key or display name"),
            ("-o, --output", "Optional output path; defaults to <root>/<key>/processed/<key>_clean.csv"),
            ("--source", "Optional source override for the known schema"),
            ("--offline", "Use only the project-local known-dataset copy"),
            ("-B, --sentiment-backend", "auto | transformer | vader | lexicon"),
            ("-m, --model", "Transformer sentiment model"),
            ("-b, --batch-size", "PyTorch sentiment batch size"),
            ("-L, --max-length", "Tokenizer maximum sequence length"),
            ("-d, --device", "auto | cpu | mps | cuda | cuda:N"),
            ("--text-column", "Override text column"),
            ("--label-column", "Override label column"),
            ("--overwrite", "Overwrite existing output"),
            ("--no-manifest", "Skip writing manifest"),
        ],
    },
    "score": {
        "purpose": "Process and score sentiment, optionally persisting the result.",
        "usage": "python master_dataset.py score DATASET [options]",
        "options": [
            ("-B, --sentiment-backend", "auto | transformer | vader | lexicon"),
            ("-m, --model", "Sentiment model identifier"),
            ("-b, --batch-size", "PyTorch inference batch size"),
            ("-L, --max-length", "Tokenizer maximum sequence length"),
            ("-d, --device", "auto | cpu | mps | cuda | cuda:N"),
            ("-o, --output FILE", "Optional persistence target"),
        ],
    },
    "preview": {
        "purpose": "Process the dataset and display canonical rows without writing them.",
        "usage": "python master_dataset.py preview DATASET [options]",
        "options": [
            ("-n, --rows N", "Number of rows"),
            ("--tail", "Show final rows"),
            ("--random", "Show a deterministic random sample"),
            ("--seed N", "Sampling seed"),
        ],
    },
    "validate": {
        "purpose": "Run strict canonical-schema, label-type, text, and sentiment checks.",
        "usage": "python master_dataset.py validate DATASET [options]",
        "options": [
            ("--json", "Emit the structured result as JSON after validation"),
            ("--no-raise", "Display failures without returning an error code"),
        ],
    },
    "save": {
        "purpose": "Process, validate, and persist a canonical dataset.",
        "usage": "python master_dataset.py save DATASET -o OUTPUT [options]",
        "options": [
            ("-o, --output FILE", "Required output destination"),
            ("--format FORMAT", "csv | jsonl | json | parquet | xlsx"),
            ("--overwrite", "Replace existing output"),
            ("--no-manifest", "Do not write provenance manifest"),
        ],
    },
    "summary": {
        "purpose": "Display canonical output, label balance, and sentiment distribution.",
        "usage": "python master_dataset.py summary DATASET [options]",
        "options": [],
    },
    "interactive": {
        "purpose": "Launch a persistent visual dataset laboratory with repeated commands.",
        "usage": "python master_dataset.py interactive [DATASET]",
        "options": [
            ("DATASET", "Optional initial dataset path/URL/known://key"),
            ("-q, --quiet", "Reduce rendering"),
        ],
    },
    "list-datasets": {
        "purpose": "List all known + discovered datasets.",
        "usage": "python master_dataset.py --list-datasets",
        "options": [],
    },
}


def render_root_help(renderer: Renderer) -> None:
    renderer.panel(
        "MASTER DATASET PROCESSOR",
        f"v{VERSION}  •  PyTorch-native sentiment  •  schema intelligence  •  canonical labels  •  research provenance",
    )
    renderer.table(
        "COMMAND ARSENAL",
        ["Command", "Purpose", "Primary output"],
        [
            ["inspect", "Inspect source and schema intelligence", "detection report"],
            ["profile", "Deep structural/text/label profiling", "profile dashboard"],
            ["process", "Clean + canonicalize + score + validate", "dataset + optional manifest"],
            ["prepare", "Quick prepare known datasets", "cleaned dataset"],
            ["score", "Process + target-independent sentiment", "scored dataset"],
            ["preview", "Inspect canonical rows", "terminal preview"],
            ["validate", "Strict integrity checks", "validation matrix"],
            ["save", "Process + validate + persist", "dataset + manifest"],
            ["summary", "Summarise labels and sentiment", "summary dashboard"],
            ["interactive", "Persistent guided laboratory", "interactive workspace"],
            ["list-datasets", "List all known + discovered datasets", "table"],
        ],
    )
    renderer.table(
        "GLOBAL OPTIONS",
        ["Long", "Short", "Purpose"],
        [
            ["--help", "-h", "Open visual command reference"],
            ["-help", "—", "Convenience alias for --help"],
            ["--quiet / --quite", "-q", "Suppress presentation output"],
            ["--no-visuals", "—", "Disable terminal rendering"],
            ["--version", "-V", "Display version"],
            ["--list-datasets", "—", "Show known + discovered datasets"],
        ],
    )
    renderer.table(
        "COMMON DATA / MODEL OPTIONS",
        ["Long", "Short", "Purpose"],
        [
            ["--text-column", "-t", "Explicit text field"],
            ["--label-column", "-l", "Explicit target field"],
            ["--sentiment-backend", "-B", "auto | transformer | vader | lexicon"],
            ["--model", "-m", "Transformer sentiment model"],
            ["--batch-size", "-b", "PyTorch inference batch"],
            ["--max-length", "-L", "Tokenizer truncation length"],
            ["--device", "-d", "auto | cpu | mps | cuda | cuda:N"],
            ["--dtype", "—", "auto | float32 | float16 | bfloat16"],
        ],
    )
    renderer.table(
        "EXAMPLES",
        ["Intent", "Command"],
        [
            ["Inspect", "python master_dataset.py inspect ./data.csv"],
            ["Profile", "python master_dataset.py profile ./data.csv --sample 10000"],
            ["Process", "python master_dataset.py process ./data.csv -o clean.csv --overwrite"],
            ["Prepare GoEmotions", "python master_dataset.py prepare --dataset goemo"],
            ["Prepare ISEAR (offline)", "python master_dataset.py prepare --dataset isear --offline"],
            ["Process discovered", "python master_dataset.py process known://emotion -t text -l label -o out.csv"],
            ["Process HF", "python master_dataset.py process hf://dair-ai/emotion -t text -l label -o out.csv --overwrite"],
            ["List datasets", "python master_dataset.py --list-datasets"],
            ["Interactive", "python master_dataset.py interactive"],
        ],
    )


def render_command_help(renderer: Renderer, command: str) -> None:
    info = COMMAND_INFO.get(command)
    if not info:
        renderer.error(f"Unknown command: {command}")
        render_root_help(renderer)
        return
    renderer.panel(command.upper(), info["purpose"])
    renderer.table("COMMAND USAGE", ["Field", "Value"], [["Usage", info["usage"]]])
    if info["options"]:
        renderer.table("COMMAND OPTIONS", ["Option", "Purpose"], info["options"])
    else:
        renderer.table("COMMAND OPTIONS", ["Option", "Purpose"], [["—", "No command-specific options"]])
        
        
# NOTE: append a marker for discovered datasets so the user can tell a
# cached detection ("text ✓") from a header-only guess ("text ?") from
# nothing at all ("?"). Managed datasets show no marker.
def _mark(col_value: Optional[str], spec: dict[str, Any]) -> str:
    """
    Annotate a schema column for the dataset listing.

    Conventions:
        'text'          -> managed/prescribed schema (authoritative)
        'content ?'     -> discovered, hint came from a header peek (a guess)
        'content ✓'     -> discovered, hint came from a cached sidecar
        '?'             -> no hint available at all
    """
    if not col_value:
        return "?"

    # Managed datasets carry no schema_is_hint flag; their columns are
    # prescribed by KNOWN_DATASETS and shown unadorned.
    if not spec.get("schema_is_hint"):
        return col_value

    source = spec.get("schema_hint_source", "unknown")
    if source == "cached":
        return f"{col_value} ✓"
    if source == "header":
        return f"{col_value} ?"
    return f"{col_value} ?"


def _relative_dataset_path(key: str) -> str:
    """Show '<key>/raw/<key>.csv' instead of the absolute path."""
    raw = known_dataset_local_path(key)
    if not raw.is_file():
        processed = known_dataset_processed_dir(key) / f"{key}_clean.csv"
        if processed.is_file():
            return f"{key}/processed/{processed.name}"
        return f"{key}/"
    return f"{key}/raw/{raw.name}"


def render_known_datasets(renderer: Renderer, *, verbose: bool = False) -> None:
    """
    Compact mode (default) shows the six fields you actually scan:
    Key, Local, Text, Target, Task, Classes.

    Verbose mode adds Name, Source, Path, and Notes for when you need
    provenance detail.
    """
    materialize_schema_sidecars()

    if verbose:
        columns = [
            "Key", "Name", "Local", "Source", "Path",
            "Text", "Target", "Task", "#", "Notes",
        ]
        rows = []
        for key, spec in all_datasets().items():
            rows.append([
                key,
                spec.get("name") or key,
                "READY" if known_dataset_is_local(key) else "MISSING",
                compact(spec.get("url") or "—", 60),
                str(known_dataset_local_path(key)),
                _mark(spec.get("text_column"), spec),
                _mark(spec.get("label_column"), spec),
                spec.get("task_type") or "unknown",
                str(spec.get("class_count", "—")),
                compact(spec.get("notes") or "", 100),
            ])
    else:
        columns = ["Key", "Local", "Text", "Target", "Task", "# classes"]
        rows = []
        for key, spec in all_datasets().items():
            rows.append([
                key,
                "READY" if known_dataset_is_local(key) else "MISSING",
                _mark(spec.get("text_column"), spec),
                _mark(spec.get("label_column"), spec),
                spec.get("task_type") or "unknown",
                str(spec.get("class_count", "—")),
            ])

    renderer.table(
        "KNOWN DATASETS" + (" (verbose)" if verbose else ""),
        columns,
        rows,
        caption=(
            f"Datasets root: {PROJECT_DATASETS_DIR}  •  "
            "✓ = confirmed by schema.json; ? = header-only hint; "
            "run --list-datasets --verbose for source, path, and notes."
        ),
    )


# =============================================================================
# INTERACTIVE LABORATORY
# =============================================================================

class InteractiveApp:
    def __init__(self, *, quiet: bool = False, no_visuals: bool = False, start_dataset: Optional[str] = None):
        self.renderer = Renderer(quiet=quiet, no_visuals=no_visuals)
        self.processor: Optional[MasterDatasetProcessor] = None
        self.start_dataset = start_dataset
        self.history: list[str] = []

    def _prompt(self, message: str, default: str = "") -> str:
        if not self.renderer.enabled:
            value = input(f"{message}{f' [{default}]' if default else ''}: ").strip()
            return value or default
        return Prompt.ask(message, default=default, console=self.renderer.console).strip()

    def _ensure_processor(self) -> bool:
        if self.processor is None:
            self.renderer.warning("No dataset is loaded. Choose Load or use the load command first.")
            return False
        return True

    def _load_dialog(self) -> None:
        render_known_datasets(self.renderer)
        dataset_key = self._prompt("Dataset key (or path/URL)", "").strip()

        resolved_key = resolve_known_dataset_key(dataset_key)

        if resolved_key is not None:
            spec = all_datasets()[resolved_key]
            source = spec["source"]

            if spec.get("schema_is_hint"):
                # NOTE: discovered dataset. spec["text_column"] / ["label_column"]
                # are *hints* from the header peek or a cached sidecar, not
                # prescriptions. Forcing them as overrides would silently disable
                # auto-detection and cache a possibly-wrong column. Instead,
                # surface the hint and let the user accept it or defer to auto.
                hint_text  = spec.get("text_column") or "?"
                hint_label = spec.get("label_column") or "?"
                hint_src   = spec.get("schema_hint_source", "unknown")
                self.renderer.info(
                    f"Discovered dataset '{resolved_key}' "
                    f"(schema hint from {hint_src}: text={hint_text}, "
                    f"target={hint_label})"
                )
                text_col = self._prompt(
                    f"Text column (blank = auto-detect, hint = {hint_text})",
                    "",
                ) or None
                label_col = self._prompt(
                    f"Label column (blank = auto-detect, hint = {hint_label})",
                    "",
                ) or None
            else:
                # Managed dataset: prescribed schema is authoritative.
                text_col  = spec.get("text_column")
                label_col = spec.get("label_column")
                self.renderer.info(
                    f"Using managed dataset: {spec.get('name') or resolved_key}"
                )
        else:
            source = dataset_key
            text_col = self._prompt("Text column (blank = auto)", "") or None
            label_col = self._prompt("Label column (blank = auto)", "") or None

        backend = self._prompt("Sentiment backend", "auto") or "auto"
        model = self._prompt("Sentiment model (blank = default)", "") or None
        batch_size = int(self._prompt("PyTorch batch size", str(DEFAULT_BATCH_SIZE)))
        max_length = int(self._prompt("Maximum token length", str(DEFAULT_MAX_LENGTH)))
        device = self._prompt("Device", "auto") or "auto"

        self.processor = MasterDatasetProcessor(
            source,
            text_column=text_col,
            label_column=label_col,
            sentiment_backend=backend,
            sentiment_model=model,
            sentiment_batch_size=batch_size,
            sentiment_max_length=max_length,
            device=device,
            interactive_fallback=True,
            quiet=self.renderer.quiet,
            no_visuals=self.renderer.no_visuals,
        )
        self.processor.load()
        try:
            self.processor.detect_schema()
        except DatasetSchemaError as exc:
            self.renderer.warning(
                f"Automatic detection failed: {exc}. "
                "Choose columns manually."
            )
            self.processor.interactive_select_columns()

    def _dashboard(self) -> None:
        source = self.processor.dataset_link if self.processor else "No dataset loaded"
        state = (
            "LOADED / PROCESSED" if self.processor and self.processor.df is not None
            else "LOADED / RAW" if self.processor
            else "EMPTY"
        )
        rows = (
            len(self.processor.df)
            if self.processor and self.processor.df is not None
            else len(self.processor.raw_df)
            if self.processor and self.processor.raw_df is not None
            else 0
        )
        self.renderer.panel(
            "MASTER DATASET LABORATORY",
            f"State: {state}\nRows: {rows:,}\nSource: {compact(source, 110)}\n\nType a number, command word, or 'help'.",
        )
        self.renderer.menu(
            "LAB CONTROLS",
            [
                ("1", "Load / replace", "load"),
                ("2", "Inspect schema", "inspect"),
                ("3", "Deep profile", "profile"),
                ("4", "Process + score", "process"),
                ("5", "Preview", "preview"),
                ("6", "Validate", "validate"),
                ("7", "Save", "save"),
                ("8", "Summary", "summary"),
                ("9", "Configuration", "config"),
                ("d", "List known datasets", "datasets"),
                ("h", "Help", "help"),
                ("0", "Exit", "exit"),
            ],
            footer="Aliases: p=profile, x=process, v=validate, s=save, r=summary, q=exit, d=datasets",
        )

    def run(self) -> None:
        self.renderer.panel("MASTER DATASET PROCESSOR", "Interactive laboratory — persistent session")
        if self.start_dataset:
            try:
                self.processor = MasterDatasetProcessor(
                    self.start_dataset,
                    quiet=self.renderer.quiet,
                    no_visuals=self.renderer.no_visuals,
                )
                self.processor.load()
                self.processor.detect_schema()
            except DatasetProcessorError as exc:
                self.renderer.error(f"Startup dataset failed: {type(exc).__name__}: {exc}")

        while True:
            self._dashboard()
            try:
                choice = self._prompt("Select action", "0").lower().strip()
            except (KeyboardInterrupt, EOFError):
                # NOTE: Ctrl+C / Ctrl+D at the prompt is a request to leave the lab,
                # not a crash. We exit cleanly instead of letting the KeyboardInterrupt
                # bubble out of run() and produce a raw traceback.
                self.renderer.warning("Session closed.")
                return
            self.history.append(choice)
            try:
                if choice in {"0", "exit", "quit", "q"}:
                    self.renderer.success("Interactive session closed.")
                    return
                if choice in {"1", "load", "reload"}:
                    self._load_dialog()
                elif choice in {"2", "inspect"}:
                    if self._ensure_processor():
                        # NOTE: inspect only needs a schema, not the full frame. Route
                        # through the same fast path the CLI uses.
                        source_path = self.processor._resolve_source_path_for_sampling()
                        if source_path is not None and self.processor.raw_df is None:
                            sample_df = DatasetLoader.read_rows_only(
                                source_path, 50_000, mode="random", seed=42,
                            )
                            self.processor.raw_df = sample_df
                            self.processor.source_path = source_path
                            self.processor.report.rows_input = len(sample_df)
                            self.processor.report.columns_input = len(sample_df.columns)
                            self.processor.renderer.info(
                                f"Inspecting on a {len(sample_df):,}-row sample."
                            )
                        self.processor.detect_schema()
                elif choice in {"3", "profile", "p"}:
                    if self._ensure_processor():
                        raw = self._prompt("Sample size (blank = full)", str(DEFAULT_SAMPLE))
                        sample = None if raw.lower() in {"", "full", "all"} else int(raw)
                        self.processor.profile(sample=sample)
                elif choice in {"4", "process", "score", "x"}:
                    if self._ensure_processor():
                        self.processor.process(force=True)
                elif choice in {"5", "preview"}:
                    if self._ensure_processor():
                        rows = int(self._prompt("Rows", "10"))
                        mode = self._prompt("Mode: head / tail / random", "head").lower()
                        self.processor.preview(rows, tail=mode == "tail", random_sample=mode == "random")
                elif choice in {"6", "validate", "v"}:
                    if self._ensure_processor():
                        if self.processor.df is None:
                            self.processor.process()
                        self.processor.validate(raise_on_error=False)
                elif choice in {"7", "save", "s"}:
                    if self._ensure_processor():
                        output = self._prompt("Output path", "processed_dataset.csv")
                        overwrite = self._prompt("Overwrite? y/n", "n").lower().startswith("y")
                        self.processor.save(output, overwrite=overwrite)
                elif choice in {"8", "summary", "r"}:
                    if self._ensure_processor():
                        if self.processor.df is None:
                            self.processor.process()
                        self.processor.summary()
                elif choice in {"9", "config", "configuration"}:
                    if self._ensure_processor():
                        self.processor.show_configuration()
                elif choice in {"d", "datasets"}:
                    render_known_datasets(self.renderer)
                elif choice in {"h", "help"}:
                    render_root_help(self.renderer)
                else:
                    self.renderer.warning("Unknown action. Use h/help to inspect available controls.")
            except DatasetProcessorError as exc:
                self.renderer.error(f"{type(exc).__name__}: {exc}")
            except (ValueError, TypeError) as exc:
                self.renderer.error(f"Invalid input: {exc}")
            except KeyboardInterrupt:
                self.renderer.warning("Operation cancelled; laboratory remains open.")
            except Exception as exc:
                self.renderer.error(f"Unexpected {type(exc).__name__}: {exc}")


# =============================================================================
# CLI
# =============================================================================

def preprocess_argv(argv: Sequence[str]) -> list[str]:
    return ["--help" if token == "-help" else token for token in argv]


def requested_help(argv: Sequence[str]) -> tuple[Optional[str], bool]:
    args = preprocess_argv(argv)
    help_requested = any(token in {"-h", "--help"} for token in args)
    command = next((token for token in args if token in COMMAND_INFO), None)
    return command, help_requested


def common_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-t", "--text-column", dest="text_column")
    parser.add_argument("-l", "--label-column", dest="label_column")
    parser.add_argument(
        "-B", "--sentiment-backend", dest="sentiment_backend",
        choices=["auto", "transformer", "vader", "lexicon", "none"], default="auto",
    )
    parser.add_argument(
        "-c", "--choose-columns", dest="choose_columns", action="store_true",
        help="Prompt interactively for the text and label columns after a preview.",
    )
    parser.add_argument("-m", "--model", "--sentiment-model", dest="sentiment_model", default=None)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-L", "--max-length", dest="max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("-d", "--device", dest="device", default="auto")
    parser.add_argument("--dtype", dest="dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--positive-label", default=None)
    parser.add_argument("--negative-label", default=None)
    parser.add_argument("--neutral-label", default=None)


def cleaning_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--keep-missing-text", action="store_true")
    parser.add_argument("--keep-missing-label", action="store_true")
    parser.add_argument("--keep-empty-text", action="store_true")
    parser.add_argument("--drop-short", type=int, default=None)
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--no-demojize", action="store_true")
    parser.add_argument("--no-url-normalization", action="store_true")
    parser.add_argument("--no-user-normalization", action="store_true")
    parser.add_argument("--normalize-hashtags", action="store_true")
    parser.add_argument("--no-html-strip", action="store_true")


def output_options(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("-o", "--output", required=required)
    parser.add_argument("--format", choices=["csv", "jsonl", "json", "parquet", "xlsx"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="master_dataset.py",
        description=(
            "Generalized dataset cleaning, schema detection, validation and "
            "target-independent sentiment scoring."
        ),
        add_help=False,
    )
    parser.add_argument("--quiet", "--quite", "-q", dest="quiet", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--version", "-V", action="store_true")
    parser.add_argument("--list-datasets", action="store_true", help="List known + discovered datasets")
    sub = parser.add_subparsers(dest="command")

    inspect = sub.add_parser("inspect", add_help=False)
    inspect.add_argument("dataset")
    common_source_options(inspect)
    inspect.add_argument("--candidates", action="store_true")

    profile = sub.add_parser("profile", add_help=False)
    profile.add_argument("dataset")
    common_source_options(profile)
    profile.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    profile.add_argument("--top-labels", type=int, default=15)
    profile.add_argument("--top-words", type=int, default=20)

    process = sub.add_parser("process", add_help=False)
    process.add_argument("dataset")
    common_source_options(process)
    cleaning_options(process)
    output_options(process)

    prepare = sub.add_parser("prepare", add_help=False)
    prepare.add_argument(
        "--dataset", required=True,
        help="Managed dataset key or display name",
    )
    prepare.add_argument("--source", help="Override source (local path, hf://, known://)")
    prepare.add_argument(
        "-o", "--output", default=None,
        help="Output path; defaults to <root>/<key>/processed/<key>_clean.csv",
    )
    prepare.add_argument("--offline", action="store_true", help="Use only the project-local copy")
    prepare.add_argument("--text-column", help="Override text column")
    prepare.add_argument("--label-column", help="Override label column")
    prepare.add_argument("-c", "--choose-columns", dest="choose_columns",
                         action="store_true",
                         help="Prompt interactively for text and label columns.")
    prepare.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    prepare.add_argument("--no-manifest", action="store_true", help="Skip manifest")
    prepare.add_argument(
        "-B", "--sentiment-backend", dest="sentiment_backend",
        choices=["auto", "transformer", "vader", "lexicon", "none"], default="auto",
    )
    prepare.add_argument("-m", "--model", "--sentiment-model", dest="sentiment_model", default=None)
    prepare.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    prepare.add_argument("-L", "--max-length", dest="max_length", type=int, default=DEFAULT_MAX_LENGTH)
    prepare.add_argument("-d", "--device", dest="device", default="auto")
    prepare.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    prepare.add_argument("--cache-dir", default=None)
    prepare.add_argument("--hf-config", default=None)
    prepare.add_argument("--positive-label", default=None)
    prepare.add_argument("--negative-label", default=None)
    prepare.add_argument("--neutral-label", default=None)
    cleaning_options(prepare)
    
    choose = sub.add_parser("choose", add_help=False)
    choose.add_argument("dataset")
    common_source_options(choose)
    choose.add_argument("--sample", type=int, default=7)

    score = sub.add_parser("score", add_help=False)
    score.add_argument("dataset")
    common_source_options(score)
    cleaning_options(score)
    output_options(score)

    preview = sub.add_parser("preview", add_help=False)
    preview.add_argument("dataset")
    common_source_options(preview)
    cleaning_options(preview)
    preview.add_argument("-n", "--rows", type=int, default=10)
    preview.add_argument("--tail", action="store_true")
    preview.add_argument("--random", dest="random_sample", action="store_true")
    preview.add_argument("--seed", type=int, default=42)

    validate = sub.add_parser("validate", add_help=False)
    validate.add_argument("dataset")
    common_source_options(validate)
    cleaning_options(validate)
    validate.add_argument("--json", action="store_true")
    validate.add_argument("--no-raise", action="store_true")

    save = sub.add_parser("save", add_help=False)
    save.add_argument("dataset")
    common_source_options(save)
    cleaning_options(save)
    output_options(save, required=True)

    summary = sub.add_parser("summary", add_help=False)
    summary.add_argument("dataset")
    common_source_options(summary)
    cleaning_options(summary)

    interactive = sub.add_parser("interactive", add_help=False)
    interactive.add_argument("dataset", nargs="?")

    return parser


def build_processor(args: argparse.Namespace) -> MasterDatasetProcessor:
    return MasterDatasetProcessor(
        args.dataset,
        text_column=getattr(args, "text_column", None),
        label_column=getattr(args, "label_column", None),
        interactive_fallback=getattr(args, "choose_columns", False),
        sentiment_backend=getattr(args, "sentiment_backend", "auto"),
        sentiment_model=getattr(args, "sentiment_model", None),
        sentiment_batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
        sentiment_max_length=getattr(args, "max_length", DEFAULT_MAX_LENGTH),
        device=getattr(args, "device", "auto"),
        dtype=getattr(args, "dtype", "auto"),
        cache_dir=getattr(args, "cache_dir", None),
        hf_config=getattr(args, "hf_config", None),
        positive_label=getattr(args, "positive_label", None),
        negative_label=getattr(args, "negative_label", None),
        neutral_label=getattr(args, "neutral_label", None),
        offline=getattr(args, "offline", False),
        lowercase=not getattr(args, "no_lowercase", False),
        demojize=not getattr(args, "no_demojize", False),
        normalize_urls=not getattr(args, "no_url_normalization", False),
        normalize_usernames=not getattr(args, "no_user_normalization", False),
        normalize_hashtags=getattr(args, "normalize_hashtags", False),
        strip_html=not getattr(args, "no_html_strip", False),
        drop_missing_text=not getattr(args, "keep_missing_text", False),
        drop_missing_label=not getattr(args, "keep_missing_label", False),
        drop_empty_text=not getattr(args, "keep_empty_text", False),
        drop_short_text=getattr(args, "drop_short", None) is not None,
        min_text_chars=getattr(args, "drop_short", None) or 1,
        drop_duplicates=not getattr(args, "no_deduplicate", False),
        quiet=getattr(args, "quiet", False),
        no_visuals=getattr(args, "no_visuals", False),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    command, help_requested = requested_help(raw)
    no_visuals = "--no-visuals" in raw
    quiet = any(flag in raw for flag in ("--quiet", "--quite", "-q"))
    if no_visuals:
        set_verbose(False)

    renderer = Renderer(quiet=quiet, no_visuals=no_visuals)

    if "--list-datasets" in raw:
        verbose_flag = any(f in raw for f in ("--verbose", "-v"))
        render_known_datasets(renderer, verbose=verbose_flag)
        return 0

    if not raw or (help_requested and command is None):
        render_root_help(renderer)
        return 0
    if help_requested and command is not None:
        render_command_help(renderer, command)
        return 0

    args = build_parser().parse_args(preprocess_argv(raw))
    if args.version:
        renderer.table("VERSION", ["Component", "Version"], [["Master Dataset Processor", VERSION]])
        return 0

    if args.command == "interactive":
        InteractiveApp(
            quiet=args.quiet,
            no_visuals=args.no_visuals,
            start_dataset=getattr(args, "dataset", None),
        ).run()
        return 0

    if not args.command:
        render_root_help(renderer)
        return 1

    # -------------------------------------------------------------------------
    # PREPARE
    # -------------------------------------------------------------------------
    if args.command == "prepare":
        key = resolve_known_dataset_key(args.dataset)
        if key is None:
            renderer.error(f"Unknown managed dataset: {args.dataset}")
            render_known_datasets(renderer)
            return 1

        spec = all_datasets()[key]

        source = args.source if args.source else spec["source"]
        text_col = args.text_column if args.text_column else spec.get("text_column")
        label_col = args.label_column if args.label_column else spec.get("label_column")

        renderer.info(f"Preparing dataset: {spec.get('name') or key}")
        renderer.info(f"  Source: {source}")
        renderer.info(f"  Text column: {text_col or 'auto'}")
        renderer.info(f"  Label column: {label_col or 'auto'}")

        processor = MasterDatasetProcessor(
            source,
            text_column=text_col,
            label_column=label_col,
            interactive_fallback=getattr(args, "choose_columns", False),
            lowercase=not getattr(args, "no_lowercase", False),
            demojize=not getattr(args, "no_demojize", False),
            normalize_urls=not getattr(args, "no_url_normalization", False),
            normalize_usernames=not getattr(args, "no_user_normalization", False),
            normalize_hashtags=getattr(args, "normalize_hashtags", False),
            strip_html=not getattr(args, "no_html_strip", False),
            drop_missing_text=not getattr(args, "keep_missing_text", False),
            drop_missing_label=not getattr(args, "keep_missing_label", False),
            drop_empty_text=not getattr(args, "keep_empty_text", False),
            drop_short_text=getattr(args, "drop_short", None) is not None,
            min_text_chars=getattr(args, "drop_short", None) or 1,
            drop_duplicates=not getattr(args, "no_deduplicate", False),
            offline=getattr(args, "offline", False),
            quiet=args.quiet,
            no_visuals=args.no_visuals,
            sentiment_backend=getattr(args, "sentiment_backend", "auto"),
            sentiment_model=getattr(args, "sentiment_model", None),
            sentiment_batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
            sentiment_max_length=getattr(args, "max_length", DEFAULT_MAX_LENGTH),
            device=getattr(args, "device", "auto"),
            dtype=getattr(args, "dtype", "auto"),
            cache_dir=getattr(args, "cache_dir", None),
            hf_config=getattr(args, "hf_config", None),
            positive_label=getattr(args, "positive_label", None),
            negative_label=getattr(args, "negative_label", None),
            neutral_label=getattr(args, "neutral_label", None),
        )
        processor.process(force=True)

        fmt = getattr(args, "format", None) or "csv"

        # NOTE: default output uses the *resolved key*, not the user-supplied
        # string, so passing "GoEmotions" still lands in datasets/goemo/processed/.
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else known_dataset_processed_dir(key) / f"{key}_clean.csv"
        )

        processor.save(
            output_path,
            fmt=fmt,
            overwrite=args.overwrite,
            manifest=not args.no_manifest,
        )
        return 0

    # -------------------------------------------------------------------------
    # EVERY OTHER COMMAND
    # -------------------------------------------------------------------------
    try:
        processor = build_processor(args)

        if args.command == "inspect":
            # NOTE: for inspect we only need the header plus a bounded
            # sample. Reading the full CSV defeats the purpose.
            source_path = processor._resolve_source_path_for_sampling()

            if source_path is not None:
                sample_df = DatasetLoader.read_rows_only(
                    source_path, 50_000, mode="random", seed=42,
                )
                processor.raw_df = sample_df
                processor.source_path = source_path
                processor.report.rows_input = len(sample_df)
                processor.report.columns_input = len(sample_df.columns)
                processor.renderer.table(
                    "SCHEMA SAMPLE",
                    ["Property", "Value"],
                    [
                        ["Source", compact(processor.dataset_link, 120)],
                        ["Sampled rows", f"{len(sample_df):,}"],
                        ["Columns", f"{len(sample_df.columns):,}"],
                        ["Raw snapshot", str(source_path)],
                    ],
                )
                detection = processor.detect_schema()
            else:
                processor.load()
                detection = processor.detect_schema()

            if getattr(args, "candidates", False):
                processor.renderer.table(
                    "TEXT CANDIDATES",
                    ["Rank", "Column", "Score"],
                    [[i, c, s] for i, (c, s) in enumerate(detection.text_candidates[:10], 1)],
                )
                processor.renderer.table(
                    "LABEL CANDIDATES",
                    ["Rank", "Column", "Score"],
                    [[i, c, s] for i, (c, s) in enumerate(detection.label_candidates[:10], 1)],
                )

        elif args.command == "profile":
            processor.profile(
                sample=args.sample,
                top_labels=args.top_labels,
                top_words=args.top_words,
            )
        
        elif args.command == "choose":
            processor.load()
            if processor.raw_df is None:
                raise DatasetSchemaError("No raw frame available for column choice.")
            processor.renderer.table(
                "PREVIEW",
                ["Property", "Value"],
                [
                    ["Source", compact(processor.dataset_link, 120)],
                    ["Rows", f"{len(processor.raw_df):,}"],
                    ["Columns", f"{len(processor.raw_df.columns):,}"],
                ],
            )
            detection = processor.interactive_select_columns()
            processor.renderer.table(
                "DETECTION RESULT",
                ["Role", "Column", "Confidence"],
                [
                    ["Text", detection.text_column or "—", detection.confidence],
                    ["Label", detection.label_column or "—", detection.confidence],
                ],
            )

        elif args.command == "process":
            processor.process()
            # NOTE: default output always routes through the universal helper,
            # which works for managed, discovered, and foreign sources alike.
            # This replaces the previous `known_dataset_processed_dir(key)` call
            # that silently produced datasets/none/processed/none.csv when the
            # source was not a known key.
            output_path = (
                Path(args.output).expanduser().resolve()
                if args.output
                else default_processed_output_for_source(args.dataset)
            )
            processor.save(
                output_path,
                fmt=args.format,
                overwrite=args.overwrite,
                manifest=not args.no_manifest,
            )

        elif args.command == "score":
            processor.process()
            if args.output:
                processor.save(
                    args.output,
                    fmt=args.format,
                    overwrite=args.overwrite,
                    manifest=not args.no_manifest,
                )
            else:
                processor.summary()

        elif args.command == "preview":
            # processor.process()
            processor.preview(
                args.rows,
                tail=args.tail,
                random_sample=args.random_sample,
                seed=args.seed,
            )

        elif args.command == "validate":
            processor.process()
            result = processor.validate(raise_on_error=not args.no_raise)
            if args.json:
                processor.renderer.table(
                    "VALIDATION JSON",
                    ["Payload"],
                    [[json.dumps(result, ensure_ascii=False, default=str)]],
                )

        elif args.command == "save":
            processor.process()
            processor.save(
                args.output,
                fmt=args.format,
                overwrite=args.overwrite,
                manifest=not args.no_manifest,
            )

        elif args.command == "summary":
            processor.process()
            processor.summary()

        return 0

    except DatasetProcessorError as exc:
        renderer.error(f"{type(exc).__name__}: {exc}")
        return 2
    except KeyboardInterrupt:
        renderer.warning("Cancelled by user.")
        return 130
    except Exception as exc:
        renderer.error(f"Unexpected {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())