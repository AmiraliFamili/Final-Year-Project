"""
dataset_loader.py
=================
Schema-free, research-safe dataset ingestion for hidden-state probing.

The loader intentionally reduces every supported tabular dataset to two semantic
columns:

    text  : the input presented to the language model
    label : the target to be recovered from the representation

It does not assume ISEAR, GoEmotions, sentiment, emotion IDs, or a particular
dataset contract. It infers the two columns from evidence and records the
decision in a machine-readable report.

Design principles
-----------------
1. Preserve information that a transformer may legitimately use.
   We normalise Unicode/control characters/whitespace, but do not aggressively
   delete punctuation, casing, URLs, emojis, or symbols by default.
2. Treat schema inference as a scored decision, not a hard-coded lookup.
3. Support scalar and multi-label targets.
4. Detect likely leakage/ambiguity and surface it in the report.
5. Save deterministic artifacts: cleaned CSV, label mapping, validation report.
6. Keep the output contract intentionally tiny: text + label.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
}

TEXT_NAME_PRIORS = (
    "text",
    "clean_text",
    "sentence",
    "utterance",
    "content",
    "document",
    "comment",
    "review",
    "prompt",
    "response",
    "statement",
    "description",
    "input",
    "body",
    "message",
    "caption",
)

LABEL_NAME_PRIORS = (
    "label",
    "labels",
    "target",
    "class",
    "category",
    "emotion",
    "emotion_label",
    "dominant_emotion",
    "sentiment",
    "intent",
    "y",
)

ID_NAME_HINTS = {
    "id",
    "idx",
    "index",
    "row_id",
    "user_id",
    "item_id",
    "sample_id",
    "uuid",
    "guid",
    "timestamp",
    "date",
}

# These are deliberately permissive. The loader is a representation-probing
# boundary, not a task-specific NLP normaliser.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class ColumnDecision:
    column: str
    role: str
    confidence: float
    score: float
    evidence: dict[str, Any]


@dataclass(frozen=True)
class DatasetReport:
    source: str
    format: str
    rows_loaded: int
    rows_saved: int
    rows_dropped: int
    text_column: str
    label_column: str
    task_type: str
    label_count: int
    minimum_text_length: int
    duplicate_rows_removed: int
    empty_text_removed: int
    ambiguous_text_candidates: list[dict[str, Any]]
    ambiguous_label_candidates: list[dict[str, Any]]
    likely_label_leakage_rate: float
    warnings: list[str]
    text_decision: dict[str, Any]
    label_decision: dict[str, Any]


class DatasetLoader:
    """
    Load, infer, clean, canonicalise, and persist a generic text/target dataset.

    Parameters
    ----------
    source:
        Local path, remote file URL, Hugging Face dataset ID, Hugging Face
        dataset URL, or a sequence of such sources.
    output_dir:
        Directory in which cleaned data and audit artifacts are written.
    text_column / label_column:
        Optional explicit semantic columns. When omitted, scored inference is
        used.
    min_text_length:
        Minimum number of Unicode characters after conservative normalisation.
    deduplicate:
        Remove exact duplicate (text, label) rows.
    allow_multilabel:
        Whether list-like targets are accepted and stored as integer-id lists.
    strict_inference:
        If True, low-confidence/ambiguous inference raises instead of warning.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        output_dir: str | Path = "probe_run/dataset",
        *,
        text_column: str | None = None,
        label_column: str | None = None,
        min_text_length: int = 2,
        deduplicate: bool = True,
        allow_multilabel: bool = True,
        strict_inference: bool = False,
        sample_size: int = 400,
        random_state: int = 42,
    ) -> None:
        self.sources = self._normalise_sources(source)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if min_text_length < 0:
            raise ValueError("min_text_length must be >= 0")
        if sample_size < 20:
            raise ValueError("sample_size must be >= 20")
        if not self.sources:
            raise ValueError("At least one dataset source is required")

        self.text_column = text_column
        self.label_column = label_column
        self.min_text_length = int(min_text_length)
        self.deduplicate = bool(deduplicate)
        self.allow_multilabel = bool(allow_multilabel)
        self.strict_inference = bool(strict_inference)
        self.sample_size = int(sample_size)
        self.random_state = int(random_state)

        self.report: DatasetReport | None = None
        self.label_mapping: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load and clean the dataset, then return only ``text`` and ``label``."""
        LOGGER.info("Loading dataset source(s): %s", self.sources)

        frames = [self._load_one(source) for source in self.sources]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            raise ValueError("All dataset sources loaded successfully but contained no rows.")

        df = pd.concat(frames, ignore_index=True, sort=False)
        rows_loaded = len(df)

        text_decision = self._resolve_text_column(df)
        label_decision = self._resolve_label_column(df, exclude={text_decision.column})

        text_col = text_decision.column
        label_col = label_decision.column

        working = df[[text_col, label_col]].copy()
        working.columns = ["text", "label"]

        before_dupes = len(working)
        if self.deduplicate:
            working = working.drop_duplicates(subset=["text", "label"], keep="first")
        duplicate_rows_removed = before_dupes - len(working)

        missing_mask = working["text"].isna() | working["label"].isna()
        missing_removed = int(missing_mask.sum())
        working = working.loc[~missing_mask].copy()

        working["text"] = working["text"].map(self._normalise_text)
        empty_mask = working["text"].eq("")
        empty_text_removed = int(empty_mask.sum())
        working = working.loc[~empty_mask].copy()

        length_mask = working["text"].str.len().ge(self.min_text_length)
        short_removed = int((~length_mask).sum())
        working = working.loc[length_mask].copy()

        if working.empty:
            raise ValueError(
                "No usable rows remain after missing/empty/minimum-length filtering."
            )

        labels, label_mapping, task_type, warnings = self._encode_labels(
            working["label"].tolist()
        )
        working["label"] = labels

        working = working[["text", "label"]].reset_index(drop=True)

        leakage_rate = self._estimate_label_leakage(working, task_type)
        if leakage_rate > 0.05:
            warnings.append(
                f"Potential label leakage detected in ~{leakage_rate:.1%} of sampled rows. "
                "This is a warning, not an automatic rejection."
            )

        if label_decision.confidence < 0.75:
            warnings.append(
                f"Label column inference confidence is only {label_decision.confidence:.2f}."
            )
        if text_decision.confidence < 0.75:
            warnings.append(
                f"Text column inference confidence is only {text_decision.confidence:.2f}."
            )

        if self.strict_inference and (
            text_decision.confidence < 0.70 or label_decision.confidence < 0.70
        ):
            raise RuntimeError(
                "Automatic schema inference is too ambiguous under strict_inference=True. "
                f"text={text_decision.confidence:.3f}, label={label_decision.confidence:.3f}"
            )

        self.label_mapping = label_mapping
        self._save_mapping(label_mapping)

        source_name = (
            str(self.sources[0])
            if len(self.sources) == 1
            else f"{len(self.sources)} sources"
        )
        report = DatasetReport(
            source=source_name,
            format=self._describe_source_format(self.sources),
            rows_loaded=rows_loaded,
            rows_saved=len(working),
            rows_dropped=rows_loaded - len(working),
            text_column=text_col,
            label_column=label_col,
            task_type=task_type,
            label_count=int(
                len(label_mapping.get("classes", []))
            ),
            minimum_text_length=self.min_text_length,
            duplicate_rows_removed=duplicate_rows_removed,
            empty_text_removed=empty_text_removed,
            ambiguous_text_candidates=text_decision.evidence.get("alternatives", []),
            ambiguous_label_candidates=label_decision.evidence.get("alternatives", []),
            likely_label_leakage_rate=leakage_rate,
            warnings=warnings,
            text_decision=asdict(text_decision),
            label_decision=asdict(label_decision),
        )
        self.report = report

        cleaned_path = self.output_dir / "cleaned_dataset.csv"
        working.to_csv(cleaned_path, index=False)

        with (self.output_dir / "dataset_report.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2, ensure_ascii=False)

        LOGGER.info(
            "Prepared dataset: %d/%d rows retained; task=%s; classes=%d",
            len(working),
            rows_loaded,
            task_type,
            report.label_count,
        )
        LOGGER.info("Clean dataset saved to %s", cleaned_path)

        return working

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_sources(
        source: str | Path | Sequence[str | Path],
    ) -> list[str]:
        if isinstance(source, (str, Path)):
            return [str(source)]
        if not isinstance(source, Sequence):
            raise TypeError("source must be a path/URL string or a sequence of them")
        return [str(item) for item in source]

    def _load_one(self, source: str) -> pd.DataFrame:
        source = source.strip()
        if not source:
            raise ValueError("Empty dataset source.")

        parsed = urlparse(source)

        # Hugging Face dataset IDs, e.g. "go_emotions" or
        # "stanfordnlp/imdb".
        if self._looks_like_hf_dataset_id(source):
            return self._load_huggingface_dataset(source)

        if parsed.scheme in {"http", "https"}:
            if "huggingface.co/datasets/" in parsed.netloc + parsed.path:
                return self._load_hf_url(source)
            local_file = self._download_remote_file(source)
            return self._load_tabular(local_file)

        return self._load_tabular(Path(source).expanduser())

    def _load_huggingface_dataset(self, dataset_id: str) -> pd.DataFrame:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load a Hugging Face dataset ID. "
                "Install it with: pip install datasets"
            ) from exc

        LOGGER.info("Resolving Hugging Face dataset: %s", dataset_id)
        loaded = load_dataset(dataset_id)

        if hasattr(loaded, "to_pandas"):
            return loaded.to_pandas()

        # DatasetDict: concatenate public splits. This mirrors the useful
        # behaviour you used for GoEmotions while remaining generic.
        frames: list[pd.DataFrame] = []
        for split_name, split in loaded.items():
            LOGGER.info("Loading Hugging Face split: %s", split_name)
            frames.append(split.to_pandas())
        return pd.concat(frames, ignore_index=True, sort=False)

    def _load_hf_url(self, url: str) -> pd.DataFrame:
        parsed = urlparse(url)
        match = re.search(r"/datasets/([^/]+/[^/]+)", parsed.path)
        if not match:
            local_file = self._download_remote_file(url)
            return self._load_tabular(local_file)
        return self._load_huggingface_dataset(match.group(1))

    def _download_remote_file(self, url: str) -> Path:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' package is required for remote dataset URLs."
            ) from exc

        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix.lower()

        response = requests.get(url, timeout=60, allow_redirects=True)
        response.raise_for_status()

        if suffix not in SUPPORTED_EXTENSIONS:
            content_type = response.headers.get("content-type", "").lower()
            suffix = self._suffix_from_content_type(content_type) or ".csv"

        handle = tempfile.NamedTemporaryFile(
            prefix="probe_dataset_",
            suffix=suffix,
            delete=False,
        )
        try:
            handle.write(response.content)
            return Path(handle.name)
        finally:
            handle.close()

    @staticmethod
    def _suffix_from_content_type(content_type: str) -> str | None:
        if "json" in content_type:
            return ".json"
        if "spreadsheet" in content_type or "excel" in content_type:
            return ".xlsx"
        if "tab-separated" in content_type:
            return ".tsv"
        if "csv" in content_type:
            return ".csv"
        return None

    def _load_tabular(self, path: str | Path) -> pd.DataFrame:
        path = Path(path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        LOGGER.info("Reading %s", path)

        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)

        if suffix == ".csv":
            # First try normal CSV parsing.
            try:
                df = pd.read_csv(path, low_memory=False)
            except pd.errors.ParserError:
                LOGGER.warning(
                    "Standard comma-separated parsing failed for %s; "
                    "attempting delimiter inference.",
                    path,
                )
                df = pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    low_memory=False,
                )

            # A file can technically be called .csv while using |, ;, or tab.
            # Detect the characteristic case where pandas created one giant column.
            if len(df.columns) <= 2:
                try:
                    inferred = pd.read_csv(
                        path,
                        sep=None,
                        engine="python",
                        low_memory=False,
                    )
                    if len(inferred.columns) > len(df.columns):
                        LOGGER.info(
                            "Delimiter inference selected %d-column interpretation.",
                            len(inferred.columns),
                        )
                        return inferred
                except (pd.errors.ParserError, UnicodeDecodeError):
                    pass

            return df

        if suffix == ".json":
            return self._read_json_flexibly(path)

        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        raise ValueError(
            f"Unsupported dataset format {suffix!r}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    @staticmethod
    def _read_json_flexibly(path: Path) -> pd.DataFrame:
        try:
            return pd.read_json(path)
        except ValueError:
            try:
                return pd.read_json(path, lines=True)
            except ValueError as exc:
                raise ValueError(f"Unable to parse JSON dataset: {path}") from exc

    # ------------------------------------------------------------------
    # Schema inference
    # ------------------------------------------------------------------

    def _resolve_text_column(self, df: pd.DataFrame) -> ColumnDecision:
        if self.text_column is not None:
            if self.text_column not in df.columns:
                raise KeyError(
                    f"Configured text column {self.text_column!r} not found. "
                    f"Available={list(df.columns)}"
                )
            return ColumnDecision(
                self.text_column,
                "text",
                1.0,
                float("inf"),
                {"mode": "explicit"},
            )

        scored = []
        for column in df.columns:
            s = df[column]
            non_null = s.dropna()
            if non_null.empty:
                continue

            sample = non_null.astype(str).head(self.sample_size)
            string_ratio = float(np.mean(non_null.head(self.sample_size).map(self._is_textlike)))
            lengths = sample.str.len()
            mean_len = float(lengths.mean())
            median_len = float(lengths.median())
            unique_ratio = float(sample.nunique(dropna=True) / max(len(sample), 1))

            name_bonus = self._name_prior(str(column), TEXT_NAME_PRIORS)
            id_penalty = self._name_prior(str(column), ID_NAME_HINTS) * 30.0

            score = (
                70.0 * string_ratio
                + 0.10 * min(mean_len, 1000.0)
                + 10.0 * min(unique_ratio, 1.0)
                + 25.0 * name_bonus
                - id_penalty
            )

            if string_ratio >= 0.60 and mean_len >= 3:
                scored.append(
                    {
                        "column": str(column),
                        "score": score,
                        "string_ratio": string_ratio,
                        "mean_length": mean_len,
                        "median_length": median_len,
                        "unique_ratio": unique_ratio,
                        "name_bonus": name_bonus,
                    }
                )

        if not scored:
            raise KeyError(
                f"Could not infer a text column. Available columns={list(df.columns)}"
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        margin = best["score"] - second["score"] if second else best["score"]
        confidence = self._confidence_from_margin(best["score"], margin)

        evidence = {
            "mode": "scored_auto",
            "alternatives": scored[1:5],
            "string_ratio": best["string_ratio"],
            "mean_length": best["mean_length"],
        }
        return ColumnDecision(
            column=best["column"],
            role="text",
            confidence=confidence,
            score=float(best["score"]),
            evidence=evidence,
        )

    def _resolve_label_column(
        self,
        df: pd.DataFrame,
        *,
        exclude: set[str],
    ) -> ColumnDecision:
        if self.label_column is not None:
            if self.label_column not in df.columns:
                raise KeyError(
                    f"Configured label column {self.label_column!r} not found. "
                    f"Available={list(df.columns)}"
                )
            return ColumnDecision(
                self.label_column,
                "label",
                1.0,
                float("inf"),
                {"mode": "explicit"},
            )

        scored = []
        n = max(len(df), 1)

        for column in df.columns:
            if str(column) in exclude:
                continue

            s = df[column].dropna()
            if s.empty:
                continue

            sample = s.head(self.sample_size)
            cardinality = int(s.nunique(dropna=True))
            cardinality_ratio = cardinality / n

            numeric = pd.api.types.is_numeric_dtype(s)
            parsed_lists = sample.map(self._looks_like_label_list)
            multilabel_ratio = float(parsed_lists.mean())

            name_bonus = self._name_prior(str(column), LABEL_NAME_PRIORS)
            id_penalty = self._name_prior(str(column), ID_NAME_HINTS)

            # We want categorical-like columns: enough repetition to form a
            # target, but not a constant and not an identifier.
            if cardinality < 2:
                continue

            repeatability = 1.0 - min(cardinality_ratio, 1.0)
            cardinality_shape = 1.0 if cardinality <= 1000 else max(
                0.0, 1.0 - math.log10(cardinality) / 10
            )

            score = (
                50.0 * repeatability
                + 25.0 * cardinality_shape
                + 45.0 * name_bonus
                + 35.0 * multilabel_ratio
                - 70.0 * min(id_penalty, 1.0)
            )

            # Continuous numeric columns are generally undesirable targets.
            if numeric and cardinality_ratio > 0.10:
                score -= 35.0

            scored.append(
                {
                    "column": str(column),
                    "score": score,
                    "cardinality": cardinality,
                    "cardinality_ratio": cardinality_ratio,
                    "is_numeric": numeric,
                    "multilabel_ratio": multilabel_ratio,
                    "name_bonus": name_bonus,
                }
            )

        if not scored:
            raise KeyError(
                f"Could not infer a label/target column. Available={list(df.columns)}"
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        margin = best["score"] - second["score"] if second else best["score"]
        confidence = self._confidence_from_margin(best["score"], margin)

        evidence = {
            "mode": "scored_auto",
            "alternatives": scored[1:5],
            "cardinality": best["cardinality"],
            "multilabel_ratio": best["multilabel_ratio"],
        }

        return ColumnDecision(
            column=best["column"],
            role="label",
            confidence=confidence,
            score=float(best["score"]),
            evidence=evidence,
        )

    @staticmethod
    def _is_textlike(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        return bool(stripped) and len(stripped) >= 3

    @staticmethod
    def _name_prior(name: str, priors: Iterable[str]) -> float:
        normalised = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if normalised in priors:
            return 1.0
        tokens = set(normalised.split("_"))
        prior_tokens = {
            re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_")
            for p in priors
        }
        return 0.50 if tokens & prior_tokens else 0.0

    @staticmethod
    def _confidence_from_margin(best: float, margin: float) -> float:
        if not math.isfinite(best):
            return 1.0
        scale = max(abs(best), 1.0)
        return float(np.clip(0.55 + 0.45 * np.tanh(margin / scale), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Cleaning and labels
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_text(value: Any) -> str:
        if value is None:
            return ""
        text = unicodedata.normalize("NFKC", str(value))
        text = _ZERO_WIDTH_RE.sub("", text)
        text = _CONTROL_RE.sub(" ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _encode_labels(
        self,
        raw_labels: Sequence[Any],
    ) -> tuple[list[Any], dict[str, Any], str, list[str]]:
        parsed = [self._parse_label_value(x) for x in raw_labels]
        is_multilabel = any(isinstance(x, list) for x in parsed)

        if is_multilabel:
            if not self.allow_multilabel:
                raise ValueError(
                    "Multi-label targets were detected but allow_multilabel=False."
                )
            label_lists = [
                x if isinstance(x, list) else [x]
                for x in parsed
            ]

            canonical = []
            for row in label_lists:
                cleaned = [str(item).strip() for item in row if str(item).strip()]
                deduped = list(dict.fromkeys(cleaned))
                if not deduped:
                    raise ValueError("Encountered a multi-label row with no valid labels.")
                canonical.append(deduped)

            classes = sorted(
                {label for row in canonical for label in row},
                key=lambda x: (x.lower(), x),
            )
            mapping = {name: i for i, name in enumerate(classes)}
            encoded = [
                json.dumps([mapping[label] for label in row], separators=(",", ":"))
                for row in canonical
            ]
            return (
                encoded,
                {
                    "task_type": "multi_label",
                    "classes": classes,
                    "class_to_id": mapping,
                    "id_to_class": {str(v): k for k, v in mapping.items()},
                },
                "multi_label",
                [],
            )

        scalar = [self._canonical_scalar_label(x) for x in parsed]
        encoder = LabelEncoder()
        encoded_array = encoder.fit_transform(np.asarray(scalar, dtype=object))
        classes = [str(x) for x in encoder.classes_]

        if len(classes) < 2:
            raise ValueError("The inferred target contains fewer than two classes.")

        mapping = {name: int(i) for i, name in enumerate(classes)}
        return (
            [int(x) for x in encoded_array.tolist()],
            {
                "task_type": "single_label",
                "classes": classes,
                "class_to_id": mapping,
                "id_to_class": {str(v): k for k, v in mapping.items()},
            },
            "single_label",
            [],
        )

    @staticmethod
    def _parse_label_value(value: Any) -> Any:
        if isinstance(value, (list, tuple, set, np.ndarray)):
            return [str(x).strip() for x in list(value)]

        if pd.isna(value):
            return ""

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("[", "(", "{")) and stripped.endswith(
                ("]", ")", "}")
            ):
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, (list, tuple, set)):
                        return [str(x).strip() for x in parsed]
                except (ValueError, SyntaxError):
                    pass

            # Common space/comma separated numeric-label forms used by
            # emotion corpora, e.g. "[6 7]" or "6,7".
            if re.fullmatch(r"\[?\s*\d+(?:[\s,]+\d+)*\s*\]?", stripped):
                nums = re.findall(r"\d+", stripped)
                if len(nums) > 1:
                    return nums

            return stripped

        return value

    @staticmethod
    def _canonical_scalar_label(value: Any) -> str:
        text = str(value).strip()
        return text

    @staticmethod
    def _looks_like_label_list(value: Any) -> bool:
        if isinstance(value, (list, tuple, set, np.ndarray)):
            return True
        if isinstance(value, str):
            s = value.strip()
            return (
                (s.startswith("[") and s.endswith("]"))
                or bool(re.fullmatch(r"\d+([,\s]+\d+)+", s))
            )
        return False

    # ------------------------------------------------------------------
    # Quality diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_label_leakage(
        df: pd.DataFrame,
        task_type: str,
        sample_size: int = 2000,
    ) -> float:
        sample = df.head(sample_size)
        if sample.empty:
            return 0.0

        hits = 0
        total = 0
        for text, label in zip(sample["text"], sample["label"]):
            target_strings = [str(label)]
            if task_type == "multi_label":
                try:
                    target_strings = [str(x) for x in json.loads(str(label))]
                except (ValueError, TypeError, json.JSONDecodeError):
                    target_strings = [str(label)]

            normalised_text = str(text).lower()
            if any(
                re.search(rf"(?<!\w){re.escape(target.lower())}(?!\w)", normalised_text)
                for target in target_strings
            ):
                hits += 1
            total += 1

        return float(hits / max(total, 1))

    def _save_mapping(self, mapping: dict[str, Any]) -> None:
        with (self.output_dir / "label_mapping.json").open("w", encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2, ensure_ascii=False)

    @staticmethod
    def _looks_like_hf_dataset_id(source: str) -> bool:
        if "://" in source or Path(source).exists():
            return False
        return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", source))

    @staticmethod
    def _describe_source_format(sources: Sequence[str]) -> str:
        if len(sources) > 1:
            return "multiple"
        source = sources[0]
        suffix = Path(urlparse(source).path).suffix.lower()
        return suffix.lstrip(".") if suffix else "auto"


def build_loader_from_cli(args: argparse.Namespace) -> DatasetLoader:
    return DatasetLoader(
        source=args.dataset,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        min_text_length=args.min_text_length,
        deduplicate=not args.keep_duplicates,
        allow_multilabel=not args.no_multilabel,
        strict_inference=args.strict_inference,
        random_state=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schema-free dataset preparation for representation probing."
    )
    parser.add_argument("dataset", nargs="+", help="Path, URL, or Hugging Face dataset ID.")
    parser.add_argument("--output-dir", default="probe_run/dataset")
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--min-text-length", type=int, default=2)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--no-multilabel", action="store_true")
    parser.add_argument("--strict-inference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    loader = build_loader_from_cli(args)
    loader.sources = args.dataset
    df = loader.load()
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
