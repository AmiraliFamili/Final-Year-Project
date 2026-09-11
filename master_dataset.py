"""
MASTER DATASET PROCESSOR v5
===========================

A research-grade, dataset-agnostic preprocessing and target-independent
sentiment scoring laboratory.

DESIGN GOALS
------------
- Accept local files, HTTP(S) dataset links, and ``hf://`` Hugging Face datasets.
- Detect text and target columns conservatively, with explicit overrides.
- Canonicalise EVERY label into a list representation (single → [27], multi → [6,22]).
- Preserve Unicode and semantic text (no destructive ASCII filtering).
- Output exactly three canonical fields: clean_text, label, sentiment_score.
- Compute sentiment independently from target labels.
- Use native PyTorch inference for transformer sentiment scoring.
- Provide deeply diagnostic profiling and strict validation.
- Offer a unified CLI with rich visual output (using ``rich`` if installed).
- Provide a persistent interactive laboratory.

CANONICAL LABEL CONTRACT
------------------------
In memory, ``label`` is ALWAYS a Python ``list``. Single labels are represented
as ``[27]``; multi-label rows as ``[6, 22]``. For text labels, analogous
representations are ``["joy"]`` and ``["joy", "excitement"]``.

CSV output serialises these lists as JSON strings, e.g. ``[27]`` or ``[6, 22]``.

SUPPORTED DATASETS (with official sources)
------------------------------------------
1. GoEmotions       : https://github.com/google-research/google-research/tree/master/goemotions
2. ISEAR            : https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
3. EmpatheticDialogues : https://github.com/facebookresearch/EmpatheticDialogues
4. EmoBank          : https://github.com/JULIELab/EmoBank

Acquisition modes:
- GoEmotions       : official raw TSV split files
- ISEAR            : verified pipe-delimited CSV acquisition file
- EmpatheticDialogues : official Meta archive
- EmoBank          : official raw CSV

Known datasets are cached under ``./datasets/<key>/raw/`` and can then be
processed completely offline. Foreign datasets may still use local paths,
HTTP(S) file URLs, or ``hf://`` sources.

USAGE EXAMPLES
--------------
# Process GoEmotions from local CSV:
python master_dataset.py process go_emotions_train.csv -t text -l labels -o goemo_clean.csv

# Process ISEAR from local CSV:
python master_dataset.py process isear.csv -t SIT -l EMOT -o isear_clean.csv

# Process EmpatheticDialogues from a direct foreign/source path if desired:
python master_dataset.py process ./datasets/empathetic/raw/empathetic_dialogues.csv -t utterance -l context -o emp_clean.csv

# Process EmoBank from the official CSV as a dimensional VAD target:
python master_dataset.py process ./datasets/emobank/raw/emobank.csv -t text -l __VAD__ -o emobank_clean.csv

# Use a unified preparation command (new):
python master_dataset.py prepare --dataset goemo --output clean_goemo.csv
python master_dataset.py prepare --dataset isear --output clean_isear.csv
python master_dataset.py prepare --dataset empathetic --output clean_emp.csv
python master_dataset.py prepare --dataset emobank --output clean_emobank.csv

# List all available datasets:
python master_dataset.py --list-datasets

# Launch interactive laboratory:
python master_dataset.py interactive
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import sys
import tarfile
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import urlparse

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

VERSION = "5.0.0"
OUTPUT_COLUMNS = ["clean_text", "label", "sentiment_score"]

DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_MULTILINGUAL_SENTIMENT_MODEL = "clapAI/modernBERT-base-multilingual-sentiment"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256
DEFAULT_SAMPLE = 10_000
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "master_dataset_cache"


# ---------------------------------------------------------------------------
# PROJECT DATASET STORE
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DATASETS_DIR = PROJECT_ROOT / "datasets"


def known_dataset_root(key: str) -> Path:
    return PROJECT_DATASETS_DIR / str(key)


def known_dataset_raw_dir(key: str) -> Path:
    path = known_dataset_root(key) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def known_dataset_processed_dir(key: str) -> Path:
    path = known_dataset_root(key) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def known_dataset_local_path(key: str) -> Path:
    spec = KNOWN_DATASETS[key]
    return known_dataset_raw_dir(key) / spec["local_raw"]


def known_dataset_is_local(key: str) -> bool:
    path = known_dataset_local_path(key)
    return path.is_file() and path.stat().st_size > 0


TEXT_NAME_HINTS = {
    "text", "sentence", "content", "utterance", "tweet", "review", "comment",
    "message", "post", "document", "description", "prompt", "response", "body",
    "statement", "caption", "query", "question", "answer", "context", "input",
    "source_text", "raw_text", "clean_text", "transcript", "title", "headline",
}
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

# Hand-selected light foreground colours. A fresh permutation is generated per
# table call, so each table invocation receives a new visual identity.
LIGHT_COLORS = [
    "#B8E7FF", "#C8F7DC", "#FFD6A5", "#E0C3FF", "#FFB7CE", "#BDE0FE",
    "#CDEAC0", "#FFE5B4", "#D8D6FF", "#F6C6EA", "#C7F9E9", "#FDE2A7",
    "#C9D9FF", "#D7F9F1", "#FFD1DC", "#E7D8FF", "#D4F1F4", "#F7D6E0",
]

BACKGROUND_COLORS = [
    "#18202A", "#1B2430", "#202735", "#20242E", "#192329", "#22212D",
]

# Known datasets for quick configuration
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
        "local_raw": "goemotions.csv",
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


def atomic_label_values(value: Any) -> list[Any]:
    if try_is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        return atomic_label_values(value.tolist())
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            out.extend(atomic_label_values(item))
        return out
    return [value]


def label_key(value: Any) -> str:
    return canonical_label_json(safe_json_value(value))


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


def save_raw_dataset(
    frame: pd.DataFrame,
    dataset_name: str,
) -> Path:
    """
    Save the unprocessed dataset exactly as acquired.

    Output:
        ./datasets/<dataset_name>/raw/<dataset_name>.csv
    """
    raw_path = dataset_raw_path(dataset_name)

    atomic_write_dataframe_csv(
        frame,
        raw_path,
    )

    return raw_path

# =============================================================================
# DATASET STORAGE / SOURCE IDENTITY
# =============================================================================

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

    # Remove URL / HF protocol prefixes.
    text = re.sub(r"^(?:https?|hf)://", "", text, flags=re.IGNORECASE)

    # Normalize obvious separators.
    text = text.replace("\\", "/")
    text = text.replace(":", "_")
    text = re.sub(r"[?#].*$", "", text)

    # Keep only the final meaningful path identity unless this is a
    # repo:config-style HF source, which has already become underscores.
    if "/" in text:
        parts = [part for part in text.split("/") if part]
        text = parts[-1] if parts else text

    # Strip common file extensions when the source is a file.
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

    Managed:
        known://goemo          -> goemo

    Hugging Face:
        hf://dair-ai/emotion         -> emotion
        hf://tweet_eval:emotion      -> tweet_eval_emotion
        hf://fancyzhx/amazon_polarity -> amazon_polarity

    URL:
        https://example.org/foo.csv  -> foo

    Local:
        ./data/my_dataset.csv        -> my_dataset
    """
    source = str(source).strip()

    if source.startswith("known://"):
        return sanitize_dataset_name(source[len("known://"):])

    if source.startswith("hf://"):
        repo_spec = source[len("hf://"):].strip()

        if ":" in repo_spec:
            repo, config = repo_spec.split(":", 1)
            base_name = sanitize_dataset_name(repo)
            config_name = sanitize_dataset_name(config)
            return sanitize_dataset_name(f"{base_name}_{config_name}")

        return sanitize_dataset_name(repo_spec)

    if is_url(source):
        parsed = urlparse(source)

        path_name = Path(parsed.path).name
        if path_name:
            name = sanitize_dataset_name(path_name)
            if name:
                return name

        # Fall back to domain identity.
        domain = parsed.netloc.split(":")[0]
        return sanitize_dataset_name(domain)

    return sanitize_dataset_name(Path(source).expanduser().name)


def dataset_root_dir(dataset_name: str) -> Path:
    return PROJECT_DATASETS_DIR / sanitize_dataset_name(dataset_name)


def dataset_raw_dir(dataset_name: str) -> Path:
    path = dataset_root_dir(dataset_name) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_processed_dir(dataset_name: str) -> Path:
    path = dataset_root_dir(dataset_name) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_raw_path(dataset_name: str) -> Path:
    """
    Canonical raw snapshot location.

    Always:
        ./datasets/<dataset_name>/raw/<dataset_name>.csv
    """
    name = sanitize_dataset_name(dataset_name)
    return dataset_raw_dir(name) / f"{name}.csv"


def dataset_processed_path(dataset_name: str) -> Path:
    """
    Canonical processed output location.

    Always:
        ./datasets/<dataset_name>/processed/<dataset_name>.csv
    """
    name = sanitize_dataset_name(dataset_name)
    return dataset_processed_dir(name) / f"{name}.csv"


def known_dataset_raw_dir(key: str) -> Path:
    return dataset_raw_dir(key)


def known_dataset_processed_dir(key: str) -> Path:
    return dataset_processed_dir(key)


def known_dataset_local_path(key: str) -> Path:
    """
    Canonical raw CSV for a managed dataset.

    The filename is deliberately independent of the remote filename so that
    every managed dataset obeys exactly the same project storage contract.
    """
    name = sanitize_dataset_name(key)
    return dataset_raw_path(name)


def known_dataset_is_local(key: str) -> bool:
    path = known_dataset_local_path(key)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def resolve_known_dataset_key(value: str) -> Optional[str]:
    """
    Resolve managed datasets case-insensitively by key or display name.

    Examples:
        goemo
        GoEmotions
        GOEMOTIONS
        emobank
        EmoBank
    """
    candidate = str(value).strip()

    if not candidate:
        return None

    normalized_candidate = normalize_column_name(candidate)

    for key, spec in KNOWN_DATASETS.items():
        key_normalized = normalize_column_name(key)
        name_normalized = normalize_column_name(spec.get("name", ""))

        if normalized_candidate in {key_normalized, name_normalized}:
            return key

    return None


def atomic_write_dataframe_csv(frame: pd.DataFrame, destination: Path) -> Path:
    """
    Atomically persist a raw DataFrame as the project's canonical raw CSV.

    The temporary file is written beside the final file and only replaced after
    a successful write. This avoids leaving a partially written raw snapshot
    after an interruption.
    """
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        frame.to_csv(
            temporary,
            index=False,
        )

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
    Universal default processed output.

    Always:

        ./datasets/<dataset_name>/processed/<dataset_name>.csv
    """
    dataset_name = dataset_name_from_source(source)
    return dataset_processed_path(dataset_name)

# =============================================================================
# RICH PRESENTATION LAYER
# =============================================================================

# =============================================================================
# ROBUST DELIMITER-SEPARATED INGESTION
# =============================================================================

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
    merge_delta_limit: int = 64          # safety cap on how much merge we allow


def sniff_delimited_policy(
    path: Path,
    sample_bytes: int = 65536,
) -> DelimitedIngestPolicy:
    """
    Sniff the delimiter and header presence from a small sample.

    Falls back to comma-delimited with a header on any sniffing failure, which
    is the safest default for the kinds of files this loader accepts.
    """
    try:
        raw = path.read_bytes()[:sample_bytes]
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
    """
    Quote-aware, row-shape-validating reader for delimiter-separated files.
    """
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

class Renderer:
    """Centralized presentation layer.

    Every table call generates a fresh palette and assigns a distinct colour to
    each column. Status output itself is represented by one-column/one-row
    tables rather than scattered prints.
    """

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
        """Create a fresh visual identity for every table invocation."""
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
                cells = []
                for i, value in enumerate(row):
                    cells.append(Text(compact(value, 240), style=palette[i]))
                table.add_row(*cells)
            self.console.print(table)
            if caption:
                self.console.print(
                    Text(
                        caption,
                        style=f"italic {title_colour}",
                    )
                )
            self.console.print()

            
        else:
            print(f"\n{title}")
            print(" | ".join(str(c) for c in columns))
            print("-" * 120)
            for row in row_list:
                print(
                    " | ".join(
                        compact(v, 120)
                        for v in row
                    )
                )

            print()

    def status(self, kind: str, message: str) -> None:
        if not self.enabled:
            return
        symbols = {"ok": "✓", "warn": "⚠", "error": "✗", "info": "◆"}
        colours = {"ok": "#B7F7C7", "warn": "#FFE6A7", "error": "#FFB3C1", "info": "#B8E7FF"}
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
            colours, background, title_colour = self._palette(max(3, min(3, len(options))))
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

    def tree(self, title: str, branches: dict[str, Sequence[str]]) -> None:
        if not self.enabled:
            return
        if self.console:
            tree = Tree(Text(title, style="bold #EAF4FF"))
            rng = random.SystemRandom()
            colours = rng.sample(LIGHT_COLORS, min(len(branches), len(LIGHT_COLORS)))
            for i, (branch, leaves) in enumerate(branches.items()):
                node = tree.add(Text(branch, style=f"bold {colours[i % len(colours)]}"))
                for leaf in leaves:
                    node.add(Text(str(leaf), style="#EAF4FF"))
            self.console.print(tree)
        else:
            print(title)
            for branch, leaves in branches.items():
                print(f"├─ {branch}")
                for leaf in leaves:
                    print(f"│  └─ {leaf}")


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
    def __init__(self, renderer: Renderer, cache_dir: Optional[str | Path] = None, timeout: tuple[int, int] = (15, 180)):
        self.renderer = renderer
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _download(
    self,
    url: str,
    destination: Optional[str | Path] = None,) -> Path:
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

        tmp = destination_path.with_suffix(
            destination_path.suffix + ".part"
        )

        try:
            with requests.get(
                url,
                stream=True,
                timeout=self.timeout,
                headers={
                    "User-Agent": f"MasterDatasetProcessor/{VERSION}"
                },
            ) as response:
                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if "text/html" in content_type:
                    raise DatasetSourceError(
                        f"Remote source returned HTML instead of a dataset "
                        f"file: {url}. Use a direct/raw dataset URL."
                    )

                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            handle.write(chunk)

            head = tmp.read_bytes()[:512].lower()

            if b"<!doctype html" in head or b"<html" in head:
                raise DatasetSourceError(
                    f"Remote source returned an HTML page instead of a "
                    f"dataset file: {url}"
                )

            tmp.replace(destination_path)

        except DatasetSourceError:
            tmp.unlink(missing_ok=True)
            raise

        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise DatasetSourceError(
                f"Could not download dataset: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return destination_path

    @staticmethod
    def _read_path(
        path: Path,
        policy: Optional["DelimitedIngestPolicy"] = None,
    ) -> pd.DataFrame:
        """
        Parse any supported dataset file into a DataFrame.

        Delimiter-separated files (.csv/.tsv/.txt) are read through
        ``read_delimited_robust``, which is quote-aware and tolerant of rows whose
        field count diverges from the header. For managed datasets the caller
        supplies an explicit ``DelimitedIngestPolicy``; otherwise the delimiter and
        header presence are sniffed from the first few kilobytes.

        All other formats (JSON, JSONL, Parquet, Excel) go through their native
        pandas readers. Every failure is re-raised as a ``DatasetSourceError``
        with a precise message, so no raw pandas traceback ever escapes the
        loader.
        """
        path = Path(path).expanduser().resolve()

        if not path.exists():
            raise DatasetSourceError(f"Dataset file does not exist: {path}")

        if not path.is_file():
            raise DatasetSourceError(f"Dataset source is not a file: {path}")

        if path.stat().st_size == 0:
            raise DatasetSourceError(f"Dataset file is empty: {path}")

        suffix = path.suffix.lower()

        try:
            # ------------------------------------------------------------------
            # DELIMITER-SEPARATED (robust path)
            # ------------------------------------------------------------------
            if suffix in {".csv", ".tsv", ".txt"}:
                effective_policy = policy or sniff_delimited_policy(path)
                return read_delimited_robust(path, effective_policy)
            # ------------------------------------------------------------------
            # JSONL / NDJSON
            # ------------------------------------------------------------------
            if suffix in {".jsonl", ".ndjson"}:
                return pd.read_json(path, lines=True)

            # ------------------------------------------------------------------
            # JSON (array-of-records or line-delimited fallback)
            # ------------------------------------------------------------------
            if suffix == ".json":
                try:
                    return pd.read_json(path)
                except ValueError:
                    return pd.read_json(path, lines=True)

            # ------------------------------------------------------------------
            # PARQUET
            # ------------------------------------------------------------------
            if suffix in {".parquet", ".pq"}:
                return pd.read_parquet(path)

            # ------------------------------------------------------------------
            # EXCEL
            # ------------------------------------------------------------------
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
        Read a managed dataset's canonical raw snapshot.

        The snapshot is always written by ``atomic_write_dataframe_csv`` using
        pandas' default comma delimiter, so the reader must use a comma policy
        regardless of the original source delimiter (which was already consumed
        during acquisition). An explicit ``ingest_policy`` on the spec, if present,
        still wins — that is the extension point for future non-comma snapshots.
        """
        spec = KNOWN_DATASETS.get(key, {})

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
    def _read_pipe_delimited_with_text_pipes(path: Path, text_column: str) -> pd.DataFrame:
        import csv
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="|", quotechar='"')
            header = next(reader)
            if text_column not in header:
                raise DatasetSourceError(
                    f"Text column '{text_column}' not found in pipe-delimited file."
                )
            text_idx = header.index(text_column)
            for parts in reader:
                extra = len(parts) - len(header)
                if extra > 0:
                    parts = (
                        parts[:text_idx]
                        + ["|".join(parts[text_idx : text_idx + extra + 1])]
                        + parts[text_idx + extra + 1 :]
                    )
                elif extra < 0:
                    parts = parts + [""] * (-extra)
                rows.append(parts)
        return pd.DataFrame(rows, columns=header)
    @staticmethod
    def _read_known_local(
        key: str,
        path: Path,
    ) -> pd.DataFrame:
        if key not in KNOWN_DATASETS:
            raise DatasetSourceError(
                f"Unknown managed dataset key: {key}"
            )

        spec = KNOWN_DATASETS[key]

        try:
            source_type = spec.get("source_type")

            if source_type == "delimited":
                frame = DatasetLoader._ingest_delimited(key, path)

            elif source_type == "goemotions_tsv":
                frame = pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                )

            else:
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
    offline: bool = False,) -> tuple[pd.DataFrame, Path]:

        resolved_key = resolve_known_dataset_key(key)

        if resolved_key is None:
            raise DatasetSourceError(
                f"Unknown known dataset key: {key}"
            )

        key = resolved_key
        spec = KNOWN_DATASETS[key]

        raw_root = known_dataset_raw_dir(key)
        local_path = known_dataset_local_path(key)

        # -------------------------------------------------------------------------
        # LOCAL-FIRST
        # -------------------------------------------------------------------------

        if known_dataset_is_local(key):
            frame = self._read_known_local(
                key,
                local_path,
            )

            self.renderer.table(
                "DATASET CACHE",
                ["Dataset", "Source mode", "Local path", "Status"],
                [[
                    spec["name"],
                    "LOCAL",
                    str(local_path),
                    "READY",
                ]],
            )

            self.renderer.panel(
                "CACHE LOCATION",
                f"{spec['name']}\n{local_path}",
            )

            return frame, local_path

        # -------------------------------------------------------------------------
        # OFFLINE WITHOUT LOCAL RAW COPY
        # -------------------------------------------------------------------------

        if offline:
            raise DatasetSourceError(
                f"Known dataset '{key}' is not available locally.\n"
                f"Expected raw snapshot:\n{local_path}\n"
                f"Run once online to acquire it."
            )

        kind = spec["source_type"]

        # -------------------------------------------------------------------------
        # GOEMOTIONS
        # -------------------------------------------------------------------------

        if kind == "goemotions_tsv":

            frames: list[pd.DataFrame] = []

            for split, url in spec["online_urls"].items():

                split_path = self._download(
                    url,
                    raw_root / f"goemotions_{split}.tsv",
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
                raise DatasetSourceError(
                    "GoEmotions acquisition produced no split files."
                )

            acquired = pd.concat(
                frames,
                ignore_index=True,
            )

            atomic_write_dataframe_csv(
                acquired,
                local_path,
            )

        # -------------------------------------------------------------------------
        # ISEAR
        # -------------------------------------------------------------------------

        elif kind == "delimited":
            downloaded_path = self._download(
                spec["online_urls"]["raw"],
                raw_root / "_source_isear.csv",
            )
            policy = DelimitedIngestPolicy(
                delimiter=spec.get("delimiter", ","),
                text_column=spec["text_column"],       # "SIT"
                on_field_mismatch="merge_into_text",
                header=True,
            )
            acquired = self._read_path(downloaded_path, policy=policy)
            atomic_write_dataframe_csv(acquired, local_path)
        # -------------------------------------------------------------------------
        # EMOBANK
        # -------------------------------------------------------------------------

        elif kind == "emobank_csv":

            downloaded_path = self._download(
                spec["online_urls"]["raw"],
                raw_root / "_source_emobank.csv",
            )

            acquired = pd.read_csv(
                downloaded_path,
            )

            atomic_write_dataframe_csv(
                acquired,
                local_path,
            )

        # -------------------------------------------------------------------------
        # EMPATHETIC DIALOGUES
        # -------------------------------------------------------------------------

        elif kind == "empathetic_archive":

            archive_path = self._download(
                spec["online_url"],
                raw_root / "empatheticdialogues.tar.gz",
            )

            archive_members = {
                "train": "empatheticdialogues/train.csv",
                "validation": "empatheticdialogues/valid.csv",
                "test": "empatheticdialogues/test.csv",
            }

            rows: list[dict[str, Any]] = []

            try:
                with tarfile.open(
                    archive_path,
                    "r:gz",
                ) as archive:

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
                            io.TextIOWrapper(
                                extracted,
                                encoding="utf-8",
                            )
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

            atomic_write_dataframe_csv(
                acquired,
                local_path,
            )

        else:
            raise DatasetSourceError(
                f"Known dataset '{key}' has unsupported "
                f"source_type '{kind}'."
            )

        # -------------------------------------------------------------------------
        # CRITICAL: REREAD THE CANONICAL RAW SNAPSHOT
        # -------------------------------------------------------------------------
        #
        # Processing must never continue from the transient acquired DataFrame.
        # The canonical raw CSV is now the authoritative local snapshot.

        if not known_dataset_is_local(key):
            raise DatasetSourceError(
                f"Known dataset '{key}' was acquired but its canonical raw "
                f"snapshot was not created successfully:\n{local_path}"
            )

        frame = self._read_known_local(
            key,
            local_path,
        )

        self.renderer.table(
            "DATASET CACHE",
            ["Dataset", "Source mode", "Local path", "Status"],
            [[
                spec["name"],
                "DOWNLOADED → CACHED → RELOADED",
                str(local_path),
                "READY",
            ]],
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
        Resolve any non-managed source into the universal project-local raw cache.

        Supported:
            - local files
            - HTTP(S)
            - hf://owner/dataset[:config]

        All paths ultimately become:

            ./datasets/<name>/raw/<name>.csv

        After acquisition, the CSV is reread and returned as the authoritative
        source for downstream processing.
        """

        source = str(source).strip()

        dataset_name = dataset_name_from_source(source)
        raw_path = dataset_raw_path(dataset_name)

        # -------------------------------------------------------------------------
        # UNIVERSAL LOCAL-FIRST RULE
        # -------------------------------------------------------------------------

        if raw_path.exists() and raw_path.is_file() and raw_path.stat().st_size > 0:

            frame = self._read_path(raw_path)

            if frame.empty:
                raise DatasetSourceError(
                    f"Cached foreign dataset is empty:\n{raw_path}"
                )

            self.renderer.table(
                "FOREIGN DATASET CACHE",
                ["Dataset", "Source mode", "Local path", "Status"],
                [[
                    dataset_name,
                    "LOCAL",
                    str(raw_path),
                    "READY",
                ]],
            )

            return frame, raw_path

        # -------------------------------------------------------------------------
        # OFFLINE WITHOUT LOCAL RAW COPY
        # -------------------------------------------------------------------------

        if offline:
            raise DatasetSourceError(
                f"Foreign dataset '{dataset_name}' is not available locally.\n"
                f"Expected raw snapshot:\n{raw_path}\n"
                f"Run once online to acquire it."
            )

        acquired: Optional[pd.DataFrame] = None

        # -------------------------------------------------------------------------
        # HUGGING FACE
        # -------------------------------------------------------------------------

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
                loaded = hf_load_dataset(
                    repo,
                    name=config,
                    trust_remote_code=True,
                )
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

                acquired = pd.concat(
                    frames,
                    ignore_index=True,
                )

            else:
                acquired = loaded.to_pandas()

        # -------------------------------------------------------------------------
        # HTTP(S)
        # -------------------------------------------------------------------------

        elif is_url(source):

            downloaded_path = self._download(
                source,
            )

            acquired = self._read_path(
                downloaded_path,
            )

        # -------------------------------------------------------------------------
        # LOCAL FILE
        # -------------------------------------------------------------------------

        else:

            local_source = Path(source).expanduser().resolve()

            if not local_source.exists():
                raise DatasetSourceError(
                    f"Dataset path does not exist: {local_source}"
                )

            if not local_source.is_file():
                raise DatasetSourceError(
                    f"Dataset source is not a file: {local_source}"
                )

            acquired = self._read_path(
                local_source,
            )

        if acquired is None:
            raise DatasetSourceError(
                f"Unable to acquire dataset from source: {source}"
            )

        if not isinstance(acquired, pd.DataFrame):
            raise DatasetSourceError(
                f"Dataset source did not produce a DataFrame: {source}"
            )

        if acquired.empty:
            raise DatasetSourceError(
                f"Dataset source contains zero rows: {source}"
            )

        # -------------------------------------------------------------------------
        # PERSIST RAW SNAPSHOT
        # -------------------------------------------------------------------------

        atomic_write_dataframe_csv(
            acquired,
            raw_path,
        )

        self.renderer.table(
            "FOREIGN DATASET CACHE",
            ["Dataset", "Source mode", "Local path", "Status"],
            [[
                dataset_name,
                "ACQUIRED → CACHED",
                str(raw_path),
                "WRITTEN",
            ]],
        )

        # -------------------------------------------------------------------------
        # CRITICAL: REREAD THE SAVED RAW DATA
        # -------------------------------------------------------------------------

        frame = self._read_path(
            raw_path,
        )

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
            raise DatasetSourceError(
                "Dataset source is empty."
            )

        # -------------------------------------------------------------------------
        # MANAGED DATASETS
        # -------------------------------------------------------------------------

        if source.startswith("known://"):

            raw_key = source[len("known://"):].strip()

            resolved_key = resolve_known_dataset_key(
                raw_key
            )

            if resolved_key is None:
                raise DatasetSourceError(
                    f"Unknown managed dataset: {raw_key}"
                )

            return self.load_known(
                resolved_key,
                offline=offline,
            )

        # -------------------------------------------------------------------------
        # ALL FOREIGN SOURCES
        #
        # local path
        # HTTP(S)
        # Hugging Face
        #
        # are normalized through exactly the same raw-cache lifecycle.
        # -------------------------------------------------------------------------

        return self._load_foreign_source_to_raw(
            source,
            hf_config=hf_config,
            offline=offline,
        )


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

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

        # Canonical JSON list.
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple, set)):
                        return cls._clean_list([cls._parse_nested_item(v) for v in parsed])
                except Exception:
                    pass

            # Handles GoEmotions-style bracketed whitespace: [6 22 27]
            inside = text[1:-1].strip()
            if inside and re.fullmatch(r"[-+]?\d+(?:\s+[-+]?\d+)+", inside):
                return [int(v) for v in inside.split()]

        # Handles bare whitespace-separated numeric multi-label strings.
        if re.fullmatch(r"[-+]?\d+(?:\s+[-+]?\d+)+", text):
            return [int(v) for v in text.split()]

        # Common delimiters for multi-label text datasets.
        for delimiter in ("|||", ";", "|", "\n"):
            if delimiter in text:
                parts = [p.strip() for p in text.split(delimiter) if p.strip()]
                if len(parts) > 1:
                    return cls._clean_list([cls._parse_nested_item(p) for p in parts])

        # Commas are only treated as multi-label separators when the whole cell
        # is clearly a label sequence rather than free text.
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
            # Preserve nested values only when explicitly present; ordinary label
            # rows remain flat.
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
    ) -> DetectionResult:
        if df.empty:
            raise DatasetSchemaError("Cannot detect schema in an empty dataset.")

        columns = [str(c) for c in df.columns]
        result = DetectionResult()

        if text_column is not None:
            if text_column not in df.columns:
                raise ColumnDetectionError(f"Requested text column '{text_column}' not found. Available: {columns}")
            result.text_column = text_column
        if label_column is not None:

            if label_column == "__VAD__":

                required_vad = {"V", "A", "D"}

                if not required_vad.issubset(
                    set(df.columns)
                ):
                    raise ColumnDetectionError(
                        "VAD target requires columns V, A, D. "
                        f"Available: {columns}"
                    )

                result.label_column = "__VAD__"

            else:

                if label_column not in df.columns:
                    raise ColumnDetectionError(
                        f"Requested label column '{label_column}' "
                        f"not found. Available: {columns}"
                    )

                result.label_column = label_column

        n = max(len(df), 1)
        candidates_text: list[tuple[str, float]] = []
        candidates_label: list[tuple[str, float]] = []

        for col in df.columns:
            name = str(col)
            normalized = normalize_column_name(name)
            series = df[col]
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

        # Detect one-hot columns conservatively: boolean-like OR numeric 0/1
        # columns, excluding obvious metadata fields and the text column.
        one_hot: list[str] = []
        for col in df.columns:
            name = str(col)
            if name == result.text_column or normalize_column_name(name) in METADATA_NAME_HINTS:
                continue
            series = df[col].dropna()
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

        if result.text_column is None and candidates_text:
            best_name, best_score = candidates_text[0]
            second_score = candidates_text[1][1] if len(candidates_text) > 1 else -math.inf
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
                raise SentimentBackendError(f"CUDA device {index} is unavailable; {torch.cuda.device_count()} device(s) detected.")
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
        """Import transformers without leaking compatibility diagnostics to the CLI."""
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
        if torch is None:
            raise SentimentBackendError("Transformer scoring requires PyTorch, but torch is not importable.")

        # Current Transformers documents PyTorch 2.5+ as its tested baseline.
        # This preflight prevents the previous misleading NameError path.
        if self._torch_version() < (2, 0, 0):
            raise SentimentBackendError(
                f"Installed PyTorch is {torch.__version__}. "
                "Transformer sentiment scoring requires PyTorch 2.0+. "
                "Upgrade with: pip install -U torch"
            )

        AutoModelForSequenceClassification, AutoTokenizer = self._import_transformers_quietly()
        self._device = self._choose_device()
        self.resolved_device = str(self._device)

        kwargs: dict[str, Any] = {"local_files_only": self.offline}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)

        try:
            self.renderer.table(
                "PyTorch SENTIMENT RUNTIME",
                ["Component", "Value"],
                [
                    ["Framework", "PyTorch"],
                    ["Model", self.model_name],
                    ["Device", str(self._device)],
                    ["Torch", str(torch.__version__)],
                    ["Offline", str(self.offline)],
                    ["Batch", str(self.batch_size)],
                    ["Max length", str(self.max_length)],
                ],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **kwargs)
            self._model.eval()
            self._model.to(self._device)
        except Exception as exc:
            raise SentimentBackendError(
                f"Could not initialize PyTorch transformer '{self.model_name}': {type(exc).__name__}: {exc}"
            ) from exc

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
            raise SentimentBackendError("VADER is unavailable. Install vaderSentiment or use transformer scoring.")
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
        if self.backend_request in {"transformer", "vader", "lexicon"}:
            return self.backend_request
        if self.backend_request != "auto":
            raise SentimentBackendError(
                "Backend must be auto, transformer, vader, or lexicon."
            )

        # AUTO: prefer transformer whenever PyTorch is importable at a
        # realistic baseline. transformers itself works on torch >= 2.0.
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
        # Once a fallback has been committed for this scorer instance, never
        # re-attempt the failed path — otherwise "auto" would try and fail the
        # transformer once per batch.
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
                    # Explicit --sentiment-backend transformer is a contract.
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

    def load(
    self,
    *,
    force: bool = False,
) -> pd.DataFrame:

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
                [
                    "Raw snapshot",
                    str(source_path) if source_path else "—",
                ],
            ],
        )

        return frame.copy()

    def detect_schema(
    self,
) -> DetectionResult:

        if self.raw_df is None:
            self.load()

        assert self.raw_df is not None

        if self.dataset_link.startswith("known://"):

            raw_key = self.dataset_link[len("known://"):].strip()
            key = resolve_known_dataset_key(raw_key)

            if key is None:
                raise DatasetSchemaError(
                    f"Unknown managed dataset: {raw_key}"
                )

            spec = KNOWN_DATASETS[key]

            expected_text = spec.get("text_column")
            expected_label = spec.get("label_column")

            if expected_text not in self.raw_df.columns:
                raise DatasetSchemaError(
                    f"Managed dataset '{key}' is missing its expected text "
                    f"column '{expected_text}'. "
                    f"Available: {list(self.raw_df.columns)}"
                )

            if expected_label == "__VAD__":

                required_vad = set(
                    spec.get(
                        "target_columns",
                        ["V", "A", "D"],
                    )
                )

                missing_vad = sorted(
                    required_vad - set(self.raw_df.columns)
                )

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

            return self.detection

        # -------------------------------------------------------------------------
        # FOREIGN DATASET
        # -------------------------------------------------------------------------

        self.detection = self.detector.detect(
            self.raw_df,
            text_column=self.text_column_override,
            label_column=self.label_column_override,
        )

        self.report.text_column = self.detection.text_column
        self.report.label_column = self.detection.label_column

        return self.detection

    # -------------------------------------------------------------------------
    # LABEL SERIES
    # -------------------------------------------------------------------------

    def _is_canonical_input(self) -> bool:
        if self.raw_df is None:
            return False
        return set(OUTPUT_COLUMNS).issubset(set(self.raw_df.columns))

    def _label_series(
    self,
    frame: pd.DataFrame,
) -> pd.Series:

        assert (
            self.detection is not None
            and self.detection.text_column is not None
        )

        if self.detection.label_column == "__VAD__":

            missing = [
                column
                for column in ("V", "A", "D")
                if column not in frame.columns
            ]

            if missing:
                raise DatasetSchemaError(
                    "EmoBank VAD target is missing: "
                    f"{missing}"
                )

            return frame.apply(
                lambda row: [
                    float(row["V"]),
                    float(row["A"]),
                    float(row["D"]),
                ],
                axis=1,
            )

        if self.detection.label_column:

            return frame[
                self.detection.label_column
            ].map(LabelNormalizer.parse)

        assert self.detection.one_hot_label_columns

        return frame.apply(
            lambda row:
                LabelNormalizer.one_hot_row(
                    row,
                    self.detection.one_hot_label_columns,
                ),
            axis=1,
        )

    # -------------------------------------------------------------------------
    # CLEANING
    # -------------------------------------------------------------------------

    def _clean_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.detection is None:
            self.detect_schema()
        assert self.detection is not None and self.detection.text_column is not None

        text_source = frame[self.detection.text_column]
        labels = self._label_series(frame)
        cleaned_text = text_source.map(self.cleaner.clean)
        result = pd.DataFrame({"clean_text": cleaned_text, "label": labels}, index=frame.index)

        if self.drop_missing_text:
            mask = text_source.map(try_is_missing)
            self.report.missing_text_rows_removed += int(mask.sum())
            result = result.loc[~mask]

        if self.drop_empty_text:
            mask = result["clean_text"].astype(str).str.strip().eq("")
            self.report.empty_text_rows_removed += int(mask.sum())
            result = result.loc[~mask]

        if self.drop_short_text:
            mask = result["clean_text"].astype(str).str.len() < self.min_text_chars
            result = result.loc[~mask]

        if self.drop_missing_label:
            mask = result["label"].map(lambda value: value is None or len(value) == 0)
            self.report.missing_label_rows_removed += int(mask.sum())
            result = result.loc[~mask]

        if self.drop_duplicates:
            before = len(result)
            dedupe_key = result.apply(
                lambda row: stable_hash({"text": str(row["clean_text"]), "label": safe_json_value(row["label"])}),
                axis=1,
            )
            result = result.loc[~dedupe_key.duplicated()]
            self.report.duplicate_rows_removed += before - len(result)

        # Guarantee canonical list type row-by-row, including values that were
        # already JSON strings in an earlier processed CSV.
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
            cleaned = self._clean_dataframe(self.load()) if self.df is None else self.df[["clean_text", "label"]].copy()
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

        # Hash-based score cache protects repeated interactive execution while
        # remaining target-independent. It is scoped to the active model/config.
        config_key = stable_hash({
            "backend": self.sentiment_scorer.backend_request,
            "model": self.sentiment_scorer.model_name,
            "max_length": self.sentiment_scorer.max_length,
        })
        texts = cleaned["clean_text"].astype(str).tolist()
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
                    f"Sentiment backend returned {len(new_scores)} scores for {len(uncached_positions)} texts."
                )
            for pos, text, score in zip(uncached_positions, uncached_texts, new_scores):
                value = float(score)
                output_scores[pos] = value
                self._score_cache[f"{config_key}:{stable_hash(text, 32)}"] = value

        score_series = pd.to_numeric(pd.Series(output_scores, index=cleaned.index), errors="coerce")
        finite = np.isfinite(score_series.to_numpy(dtype=float))
        self.report.invalid_sentiment_scores = int((~finite).sum())
        if not finite.all():
            raise DatasetValidationError(f"{self.report.invalid_sentiment_scores} invalid sentiment scores were produced.")

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
        
        # inside MasterDatasetProcessor.process (after cleaning and scoring)

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

    def profile(self, *, sample: Optional[int] = DEFAULT_SAMPLE, top_labels: int = 15, top_words: int = 20) -> dict[str, Any]:
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
            role = "TEXT" if str(column) == detection.text_column else "LABEL" if str(column) == detection.label_column else "OTHER"
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
                ["Emoji markers", f"{int(text.str.count(r":").sum()):,}"],
                ["Lexical diversity", f"{lexical_diversity:.4f}"],
                ["Mean punctuation density", f"{text.map(lambda x: sum(ch in '.,!?;:' for ch in x) / max(len(x),1)).mean():.4f}"],
                ["Mean digit density", f"{text.map(lambda x: sum(ch.isdigit() for ch in x) / max(len(x),1)).mean():.4f}"],
            ],
        )

        label_series = self._label_series_for_profile(frame, detection)

        # Continuous vector targets (e.g. EmoBank V/A/D) are not categorical labels.
        is_continuous_target = (
            detection.label_column == "__VAD__"
            and all(col in frame.columns for col in ("V", "A", "D"))
        )

        label_lengths = label_series.map(lambda x: len(x) if isinstance(x, list) else 0)
        flattened = Counter(str(item) for labels in label_series.dropna() for item in (labels if isinstance(labels, list) else []))
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
                lambda row: [
                    float(row["V"]),
                    float(row["A"]),
                    float(row["D"]),
                ],
                axis=1,
            )

        if detection.label_column:
            return frame[
                detection.label_column
            ].map(LabelNormalizer.parse)

        assert detection.one_hot_label_columns

        return frame.apply(
            lambda row:
                LabelNormalizer.one_hot_row(
                    row,
                    detection.one_hot_label_columns,
                ),
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

    def preview(self, rows: int = 10, *, tail: bool = False, random_sample: bool = False, seed: int = 42) -> pd.DataFrame:
        if self.df is None:
            self.process()
        assert self.df is not None
        n = max(1, min(int(rows), len(self.df)))
        if random_sample:
            sample = self.df.sample(n=n, random_state=seed)
        elif tail:
            sample = self.df.tail(n)
        else:
            sample = self.df.head(n)
        self.renderer.table(
            "PROCESSED PREVIEW",
            OUTPUT_COLUMNS,
            [
                [compact(row.clean_text, 120), canonical_label_json(row.label), f"{float(row.sentiment_score):+.6f}"]
                for row in sample.itertuples(index=False)
            ],
            show_lines=True,
        )
        return sample.copy()

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
        source_hash = file_hash(self.source_path) if self.source_path and self.source_path.exists() else stable_hash(self.dataset_link)
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
            raise DatasetProcessorError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

        inferred = output_path.suffix.lower().lstrip(".")
        resolved_fmt = (fmt or inferred or "csv").lower()
        if resolved_fmt == "pq":
            resolved_fmt = "parquet"
        if resolved_fmt not in {"csv", "jsonl", "json", "parquet", "xlsx"}:
            raise DatasetProcessorError("Output format must be csv, jsonl, json, parquet, or xlsx.")

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
            raise DatasetProcessorError(f"Failed writing '{output_path}': {type(exc).__name__}: {exc}") from exc

        if manifest:
            manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
            manifest_path.write_text(json.dumps(self._manifest(output_path, resolved_fmt), indent=2, ensure_ascii=False, default=str), encoding="utf-8")

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
        self.renderer.table("ACTIVE CONFIGURATION", ["Key", "Value"], [[k, compact(v, 120)] for k, v in config.items()])


# =============================================================================
# HELP SYSTEM & NEW COMMANDS
# =============================================================================

COMMAND_INFO: dict[str, dict[str, Any]] = {
    "inspect": {
        "purpose": "Load a source and inspect schema detection without modifying the dataset.",
        "usage": "python master_dataset.py inspect DATASET [options]",
        "options": [
            ("DATASET", "Source path, URL, or hf://dataset identifier"),
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
            ("--dataset", "Dataset name: goemo, isear, empathetic, emobank"),
            ("-o, --output", "Optional output path; defaults to ./datasets/<key>/processed/<key>_clean.csv"),
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
            ("DATASET", "Optional initial dataset path/URL"),
            ("-q, --quiet", "Reduce rendering"),
        ],
    },
    "list-datasets": {
        "purpose": "List all known datasets with their sources and column hints.",
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
            ["list-datasets", "List known datasets with sources", "table"],
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
            ["--list-datasets", "—", "Show known dataset configurations"],
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
            ["Prepare GoEmotions", "python master_dataset.py prepare --dataset goemo -o goemo_clean.csv"],
            ["Prepare ISEAR", "python master_dataset.py prepare --dataset isear --source isear.csv -o isear_clean.csv"],
            ["Prepare Empathetic", "python master_dataset.py prepare --dataset empathetic -o emp_clean.csv"],
            ["Prepare EmoBank", "python master_dataset.py prepare --dataset emobank -o emobank_clean.csv"],
            ["Preview", "python master_dataset.py preview ./data.csv --rows 15 --tail"],
            ["Validate", "python master_dataset.py validate ./data.csv"],
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


def render_known_datasets(
    renderer: Renderer,
) -> None:
    """Display managed datasets and local availability."""

    rows = []

    for key, spec in KNOWN_DATASETS.items():

        local_path = known_dataset_local_path(key)

        local_status = (
            "READY"
            if known_dataset_is_local(key)
            else "MISSING"
        )

        rows.append([
            key,
            spec["name"],
            local_status,
            compact(
                spec.get("url", "—"),
                52,
            ),
            str(local_path),
            spec.get("text_column", "?"),
            spec.get("label_column", "?"),
            spec.get("task_type", "?"),
            spec.get("class_count", "—"),
            compact(
                spec.get("notes", ""),
                72,
            ),
        ])

    renderer.table(
        "KNOWN DATASETS",
        [
            "Key",
            "Name",
            "Local",
            "Source",
            "Local path",
            "Text",
            "Target",
            "Task",
            "# classes",
            "Notes",
        ],
        rows,
        caption=(
            "All datasets use the project-local lifecycle: "
            "acquire → raw/<dataset>.csv → process → "
            "processed/<dataset>.csv. "
            "--offline requires the raw snapshot."
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
        # Show known datasets first
        render_known_datasets(self.renderer)
        dataset_key = self._prompt(
            "Dataset key (or path/URL)",
            "",
        ).strip()

        resolved_key = resolve_known_dataset_key(
            dataset_key
        )

        if resolved_key is not None:

            spec = KNOWN_DATASETS[resolved_key]

            source = spec["source"]
            text_col = spec["text_column"]
            label_col = spec["label_column"]

            self.renderer.info(
                f"Using known dataset: {spec['name']}"
            )

        else:

            source = dataset_key

            text_col = self._prompt(
                "Text column (blank = auto)",
                "",
            ) or None

            label_col = self._prompt(
                "Label column (blank = auto)",
                "",
            ) or None
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
            quiet=self.renderer.quiet,
            no_visuals=self.renderer.no_visuals,
        )
        self.processor.load()
        self.processor.detect_schema()

    def _dashboard(self) -> None:
        source = self.processor.dataset_link if self.processor else "No dataset loaded"
        state = "LOADED / PROCESSED" if self.processor and self.processor.df is not None else "LOADED / RAW" if self.processor else "EMPTY"
        rows = len(self.processor.df) if self.processor and self.processor.df is not None else len(self.processor.raw_df) if self.processor and self.processor.raw_df is not None else 0
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
                self.processor = MasterDatasetProcessor(self.start_dataset, quiet=self.renderer.quiet, no_visuals=self.renderer.no_visuals)
                self.processor.load()
                self.processor.detect_schema()
            except DatasetProcessorError as exc:
                self.renderer.error(f"Startup dataset failed: {type(exc).__name__}: {exc}")

        while True:
            self._dashboard()
            choice = self._prompt("Select action", "0").lower().strip()
            self.history.append(choice)
            try:
                if choice in {"0", "exit", "quit", "q"}:
                    self.renderer.success("Interactive session closed.")
                    return
                if choice in {"1", "load", "reload"}:
                    self._load_dialog()
                elif choice in {"2", "inspect"}:
                    if self._ensure_processor():
                        self.processor.load()
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
# CLI HELP / PARSER
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
    parser.add_argument("-B", "--sentiment-backend", dest="sentiment_backend", choices=["auto", "transformer", "vader", "lexicon"], default="auto")
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
        description="Generalized dataset cleaning, schema detection, validation and target-independent sentiment scoring.",
        add_help=False,
    )
    parser.add_argument("--quiet", "--quite", "-q", dest="quiet", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--version", "-V", action="store_true")
    parser.add_argument("--list-datasets", action="store_true", help="List known datasets and their configurations")
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
    prepare.add_argument("--dataset", required=True, help=("Managed dataset key or name "
        "(goemo, GoEmotions, isear, ISEAR, empathetic,"
        "EmpatheticDialogues, emobank, EmoBank)"),)
    prepare.add_argument("--source", help="Override source (local path or hf://...)")
    prepare.add_argument("-o", "--output", default=None, help="Output path; defaults to datasets/<key>/processed/<key>_clean.csv")
    prepare.add_argument("--offline", action="store_true", help="Use only the project-local known dataset copy")
    prepare.add_argument("--text-column", help="Override text column")
    prepare.add_argument("--label-column", help="Override label column")
    prepare.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    prepare.add_argument("--no-manifest", action="store_true", help="Skip manifest")
    prepare.add_argument("-B", "--sentiment-backend", dest="sentiment_backend", choices=["auto", "transformer", "vader", "lexicon"], default="auto")
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
    # Reuse common cleaning options (we'll add them manually)
    cleaning_options(prepare)

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

    # Handle --list-datasets
    if "--list-datasets" in raw:
        render_known_datasets(renderer)
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
        InteractiveApp(quiet=args.quiet, no_visuals=args.no_visuals, start_dataset=getattr(args, "dataset", None)).run()
        return 0

    if not args.command:
        render_root_help(renderer)
        return 1

    # Special handling for 'prepare'
    if args.command == "prepare":

        key = resolve_known_dataset_key(
            args.dataset
        )

        if key is None:
            renderer.error(
                f"Unknown managed dataset: {args.dataset}"
            )
            render_known_datasets(renderer)
            return 1

        spec = KNOWN_DATASETS[key]

        source = (
            args.source
            if args.source
            else spec["source"]
        )

        text_col = (
            args.text_column
            if args.text_column
            else spec["text_column"]
        )

        label_col = (
            args.label_column
            if args.label_column
            else spec["label_column"]
        )

        renderer.info(
            f"Preparing dataset: {spec['name']}"
        )
        renderer.info(
            f"  Source: {source}"
        )
        renderer.info(
            f"  Text column: {text_col}"
        )
        renderer.info(
            f"  Label column: {label_col}"
        )

        # Use the processor with the same cleaning settings
        processor = MasterDatasetProcessor(
            source,
            text_column=text_col,
            label_column=label_col,
            # Use the default cleaning (matching other datasets)
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
        # Process and save
        df = processor.process(force=True)
        # Override output format if not specified
        fmt = getattr(args, "format", None) or "csv"

        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else (
                known_dataset_processed_dir(key)
                / f"{args.dataset}_clean.csv"
            )
        )

        processor.save(
            output_path,
            fmt=fmt,
            overwrite=args.overwrite,
            manifest=not args.no_manifest,
        )
        return 0

    try:
        key = resolve_known_dataset_key(
            args.dataset
        )
        processor = build_processor(args)

        if args.command == "inspect":
            processor.load()
            detection = processor.detect_schema()
            if getattr(args, "candidates", False):
                processor.renderer.table("TEXT CANDIDATES", ["Rank", "Column", "Score"], [[i, c, s] for i, (c, s) in enumerate(detection.text_candidates[:10], 1)])
                processor.renderer.table("LABEL CANDIDATES", ["Rank", "Column", "Score"], [[i, c, s] for i, (c, s) in enumerate(detection.label_candidates[:10], 1)])

        elif args.command == "profile":
            processor.profile(sample=args.sample, top_labels=args.top_labels, top_words=args.top_words)

        elif args.command == "process":

            processor.process()

            output_path = (
                Path(args.output).expanduser().resolve()
                if args.output
                else known_dataset_processed_dir(key)
                / f"{key}.csv"
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
                processor.save(args.output, fmt=args.format, overwrite=args.overwrite, manifest=not args.no_manifest)
            else:
                processor.summary()

        elif args.command == "preview":
            processor.process()
            processor.preview(args.rows, tail=args.tail, random_sample=args.random_sample, seed=args.seed)

        elif args.command == "validate":
            processor.process()
            result = processor.validate(raise_on_error=not args.no_raise)
            if args.json:
                processor.renderer.table("VALIDATION JSON", ["Payload"], [[json.dumps(result, ensure_ascii=False, default=str)]])

        elif args.command == "save":
            processor.process()
            processor.save(args.output, fmt=args.format, overwrite=args.overwrite, manifest=not args.no_manifest)

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



"""
# ---------------------------------------------------------------------------
# MANAGED DATASETS
# ---------------------------------------------------------------------------

# First run: acquire online, cache locally, preprocess, validate, and save.
python master_dataset.py prepare --dataset goemo
python master_dataset.py prepare --dataset isear
python master_dataset.py prepare --dataset empathetic
python master_dataset.py prepare --dataset emobank

# Later runs can be completely offline.
python master_dataset.py prepare --dataset goemo --offline
python master_dataset.py prepare --dataset isear --offline
python master_dataset.py prepare --dataset empathetic --offline
python master_dataset.py prepare --dataset emobank --offline

# Managed datasets live under:
# ./datasets/<dataset>/raw/
# ./datasets/<dataset>/processed/




# ---------------------------------------------------------------------------
# DATASET SOURCES
# ---------------------------------------------------------------------------

# GoEmotions:
# https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv
# https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv
# https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv

# ISEAR:
# Official documentation:
# https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
# Downloadable dataset mirror:
# https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv

# EmpatheticDialogues:
# https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz

# EmoBank:
# https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv
#
# Project-local storage:
# ./datasets/<dataset>/raw/
# ./datasets/<dataset>/processed/
"""



"""

# 1. Emotion (dair-ai/emotion) – 6 emotions
python master_dataset.py process hf://dair-ai/emotion -t text -l label -o datasets/emotion/processed/emotion_clean.csv --overwrite

# 2. TweetEval Emotion – 4 emotions
python master_dataset.py process hf://tweet_eval:emotion -t text -l label -o datasets/tweet_eval_emotion/processed/tweet_eval_emotion_clean.csv --overwrite

# 3. SST-2 – binary sentiment
python master_dataset.py process hf://sst2 -t sentence -l label -o datasets/sst2/processed/sst2_clean.csv --overwrite

# 4. Amazon Polarity – binary sentiment (large dataset)
python master_dataset.py process hf://amazon_polarity -t content -l label -o datasets/amazon_polarity/processed/amazon_polarity_clean.csv --overwrite

# 5. Financial Phrasebank – 3‑class sentiment
python master_dataset.py process hf://financial_phrasebank -t sentence -l label -o datasets/financial_phrasebank/processed/financial_phrasebank_clean.csv --overwrite

"""



if __name__ == "__main__":
    raise SystemExit(main())


