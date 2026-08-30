from __future__ import annotations

"""Deterministic hidden-state extraction pipeline (corrected v7, unified v2 format).

Core rules
----------
* Existing experiment manifests are authoritative on resume.
* The model matrix and experiment configuration are resolved exactly once.
* Model revisions are pinned to immutable Hugging Face commit SHAs.
* Runtime extraction never changes batch size, dtype, pooling, or max_length.
* Model loading, probing, and extraction have separate diagnostics.
* Unexpected/non-core checkpoint keys are recorded rather than blindly fatal.
* Core missing/mismatched/meta parameters remain fatal.
* Optional poolers containing meta tensors may be disabled because they are
  irrelevant to hidden-state extraction; this intervention is recorded.
* Resume state is validated against dataset/model/experiment provenance.
* INFO focuses on speed/state; DEBUG exposes per-batch forensic records.
* Per‑sample data integrity is ensured through atomic writes, hashing, and fsync.
* Unified v2 format: sample_ids (object), text_hashes (optional), global checksum.
"""

from dataclasses import asdict, dataclass, field
from collections import deque
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from statistics import median
from typing import Any, Mapping, Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from tqdm.auto import tqdm
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers

try:
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        IncompleteSnapshotError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
except ImportError:  # pragma: no cover
    GatedRepoError = HfHubHTTPError = IncompleteSnapshotError = Exception
    LocalEntryNotFoundError = RepositoryNotFoundError = RevisionNotFoundError = Exception

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover
    Version = None

# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

EXTERNAL_MOUNT = Path("/Volumes/Amirali").resolve()
EXTERNAL_ROOT = EXTERNAL_MOUNT / "hidden_states"
HF_ROOT = EXTERNAL_ROOT / "huggingface_cache"
HF_HUB_CACHE = HF_ROOT / "hub"
HF_XET_CACHE = HF_ROOT / "xet"
HF_ASSETS_CACHE = HF_ROOT / "assets"
EXPERIMENTS_ROOT = EXTERNAL_ROOT / "experiments"
HYPERPARAMETER_CONFIG_PATH = EXPERIMENTS_ROOT / "hyperparameters.json"
MODEL_REVISION_MANIFEST = EXPERIMENTS_ROOT / "model_revisions.json"
MEASUREMENT_LEDGER = EXPERIMENTS_ROOT / "measurement_ledger.jsonl"

DEFAULT_POOLING = "mean"
DEFAULT_MAX_LENGTH = 512
DEFAULT_STORAGE_DTYPE = "float32"
DEFAULT_FLUSH_EVERY_BATCHES = 8
DEFAULT_FLUSH_SECONDS = 30.0
DEFAULT_EXPERIMENT_ID = "baseline_v5_001"

MEMORY_RESERVE_FRACTION = 0.20
MIN_MEMORY_RESERVE_GB = 0.50
MAX_MEMORY_RESERVE_GB = 2.50
MAX_MODEL_MEMORY_FRACTION = 0.80
MODEL_LOAD_SAFETY_FACTOR = 1.20

PROGRESS_REFRESH = 0.5
DIAGNOSTIC_WINDOW = 20
RUNTIME_REPORT_INTERVAL_SECONDS = 15.0
RUNTIME_REPORT_INTERVAL_SAMPLES = 5_000
DEBUG_RECENT_BATCHES = 12
SLOW_BATCH_MULTIPLIER = 2.0
MAX_FORENSIC_TRACEBACK_CHARS = 20_000

DOWNLOAD_RETRIES = 4
DOWNLOAD_MAX_WORKERS = 4
DOWNLOAD_BACKOFF_SECONDS = 2.0
DOWNLOAD_ETAG_TIMEOUT_SECONDS = 20.0

USE_LOW_CPU_MEM_USAGE = True
USE_SDPA = False
STRICT_META_VALIDATION = True
MODEL_PROBE_MAX_LENGTH = 8

TEXT_COLUMN_CANDIDATES = (
    "text", "sentence", "utterance", "content", "statement", "comment",
    "prompt", "clean_text", "raw_text", "input", "document", "review", "description",
)
TEXT_COLUMN_EXCLUDE = {
    "label", "labels", "emotion", "emotions", "target", "targets", "category",
    "categories", "class", "classes", "id", "idx", "index", "split", "fold",
}

COMMON_MODEL_FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "spiece.model", "spm.model", "sentencepiece.bpe.model", "tokenizer.model",
    "vocab.txt", "LICENSE*", "README.md", "*.json",
]
UNNECESSARY_FRAMEWORK_PATTERNS = [
    "*.h5", "*.msgpack", "*.ot", "tf_model.*", "flax_model.*", "rust_model.ot",
    "*.onnx", "*.gguf", "*.ggml",
]
NON_CORE_MODEL_PARAMETER_PREFIXES = (
    "pooler.", "classifier.", "score.", "lm_head.", "qa_outputs.",
)

# =============================================================================
# MODEL REGISTRY
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    generation: str
    parameter_billions: float
    architecture: str
    training_status: str
    group: str
    batch_hint: int
    pooling: str = DEFAULT_POOLING
    max_length: int = DEFAULT_MAX_LENGTH
    min_transformers: str | None = None
    trust_remote_code: bool = False
    gated: bool = False
    role: str = "primary"

MODEL_REGISTRY = (
    ModelSpec("google-bert/bert-base-uncased", "BERT", "2018", .110, "encoder", "pretrained", "01_encoders", 64),
    ModelSpec("distilbert/distilbert-base-uncased", "DistilBERT", "2019", .066, "encoder", "pretrained_distilled", "01_encoders", 64),
    ModelSpec("FacebookAI/roberta-base", "RoBERTa", "2019", .125, "encoder", "pretrained", "01_encoders", 64),
    ModelSpec("google/electra-small-discriminator", "ELECTRA", "2020", .014, "encoder", "pretrained_discriminator", "01_encoders", 64),
    ModelSpec("microsoft/deberta-v3-small", "DeBERTa", "2021", .140, "encoder", "pretrained", "01_encoders", 64),
    ModelSpec("gpt2", "GPT", "2019", .124, "decoder", "pretrained", "02_early_decoders", 64),
    ModelSpec("EleutherAI/gpt-neo-125m", "GPT-Neo", "2021", .125, "decoder", "pretrained", "02_early_decoders", 64),
    ModelSpec("facebook/opt-125m", "OPT", "2022", .125, "decoder", "pretrained", "02_early_decoders", 64),
    ModelSpec("HuggingFaceTB/SmolLM2-135M", "SmolLM2", "2024", .135, "decoder", "pretrained", "03_tiny_modern", 64),
    ModelSpec("HuggingFaceTB/SmolLM2-360M", "SmolLM2", "2024", .360, "decoder", "pretrained", "03_tiny_modern", 32),
    ModelSpec("google/gemma-3-270m", "Gemma", "2025", .270, "decoder", "pretrained", "03_tiny_modern", 32, min_transformers="4.50.0", gated=True),
    ModelSpec("Qwen/Qwen2-0.5B", "Qwen", "Qwen2", .500, "decoder", "pretrained_base", "03_tiny_modern", 32),
    ModelSpec("Qwen/Qwen2.5-0.5B", "Qwen", "Qwen2.5", .500, "decoder", "pretrained_base", "03_tiny_modern", 32),
    ModelSpec("Qwen/Qwen3-0.6B-Base", "Qwen", "Qwen3", .600, "decoder", "pretraining_base", "03_tiny_modern", 16, min_transformers="4.51.0"),
    ModelSpec("Qwen/Qwen2-1.5B", "Qwen", "Qwen2", 1.500, "decoder", "pretrained_base", "04_qwen_scaling", 1),
    ModelSpec("Qwen/Qwen2.5-1.5B", "Qwen", "Qwen2.5", 1.540, "decoder", "pretrained_base", "04_qwen_scaling", 1),
    ModelSpec("Qwen/Qwen2.5-3B", "Qwen", "Qwen2.5", 3.090, "decoder", "pretrained_base", "04_qwen_scaling", 1),
    ModelSpec("Qwen/Qwen3-1.7B-Base", "Qwen", "Qwen3", 1.700, "decoder", "pretraining_base", "04_qwen_scaling", 1, min_transformers="4.51.0"),
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B", "SmolLM2", "2024", 1.700, "decoder", "pretrained", "05_independent_small", 8),
    ModelSpec("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "TinyLlama", "2024", 1.100, "decoder", "pretrained_intermediate", "05_independent_small", 8),
    ModelSpec("google/gemma-3-1b-pt", "Gemma", "2025", 1.000, "decoder", "pretrained", "05_independent_small", 8, min_transformers="4.50.0", gated=True),
    ModelSpec("meta-llama/Llama-3.2-1B", "Llama", "2024", 1.000, "decoder", "pretrained", "05_independent_small", 8, min_transformers="4.43.0", gated=True),
    ModelSpec("Qwen/Qwen3-4B-Base", "Qwen", "Qwen3", 4.000, "decoder", "pretraining_base", "06_stretch", 1, min_transformers="4.51.0"),
    ModelSpec("meta-llama/Llama-3.2-3B", "Llama", "2024", 3.000, "decoder", "pretrained", "06_stretch", 1, min_transformers="4.43.0", gated=True),
    ModelSpec("google/gemma-3-4b-pt", "Gemma", "2025", 4.000, "decoder", "pretrained", "06_stretch", 1, min_transformers="4.50.0", gated=True),
)
assert len(MODEL_REGISTRY) == 25
MODEL_BY_NAME = {m.name: m for m in MODEL_REGISTRY}
GROUP_ORDER = (
    "01_encoders", "02_early_decoders", "03_tiny_modern",
    "04_qwen_scaling", "05_independent_small", "06_stretch",
)


def get_model_specs(groups: Sequence[str] | None = None) -> list[ModelSpec]:
    wanted = set(GROUP_ORDER if groups is None else groups)
    return [m for m in MODEL_REGISTRY if m.group in wanted]


def get_model_spec(model_name: str) -> ModelSpec:
    return MODEL_BY_NAME.get(
        model_name,
        ModelSpec(model_name, "unknown", "unknown", 1.0, "unknown", "unknown", "custom", 4),
    )

# =============================================================================
# BASIC HELPERS
# =============================================================================

def _separator(character: str = "═", width: int = 88) -> str:
    return character * width


def _header(title: str, width: int = 88, show: bool = True) -> None:
    if not show:
        return
    inner = width - 4
    print("\n╔" + "═" * (width - 2) + "╗")
    print(f"║ {title:<{inner}} ║")
    print("╚" + "═" * (width - 2) + "╝")


def _print_verbose(message: str, enabled: bool) -> None:
    if enabled:
        print(message)


def _print_info(message: str, enabled: bool) -> None:
    if enabled:
        print(message)


def _print_critical(message: str, enabled: bool) -> None:
    if enabled:
        print(message)


def _verbosity_name(show_verbose: bool, show_info: bool, show_critical: bool, show_debug: bool = False) -> str:
    if show_debug: return "DEBUG"
    if show_verbose: return "VERBOSE"
    if show_info: return "INFO"
    if show_critical: return "CRITICAL"
    return "SILENT"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60: return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60: return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:04.1f}s"


def _truncate_traceback(text: str, limit: int = MAX_FORENSIC_TRACEBACK_CHARS) -> str:
    return text if len(text) <= limit else text[-limit:]


def save_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str, sort_keys=True)
    tmp.replace(path)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False, default=str, sort_keys=True) + "\n")


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]

# =============================================================================
# SYSTEM / DEVICE
# =============================================================================

def _bytes_to_gb(x: int | float) -> float:
    return float(x) / (1024 ** 3)


def _sysctl_int(name: str) -> int | None:
    try:
        return int(subprocess.check_output(["sysctl", "-n", name], stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        return None


def get_memory_info() -> dict[str, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total_gb": _bytes_to_gb(vm.total), "available_gb": _bytes_to_gb(vm.available), "used_gb": _bytes_to_gb(vm.used)}
    except ImportError:
        pass
    total = _sysctl_int("hw.memsize")
    if total is None:
        return {"total_gb": 0.0, "available_gb": 0.0, "used_gb": 0.0}
    total_gb = _bytes_to_gb(total)
    return {"total_gb": total_gb, "available_gb": total_gb * 0.5, "used_gb": total_gb * 0.5}


def get_external_storage_usage() -> dict[str, float]:
    st = os.statvfs(EXTERNAL_MOUNT)
    total = st.f_blocks * st.f_frsize
    avail = st.f_bavail * st.f_frsize
    return {"total_gb": _bytes_to_gb(total), "used_gb": _bytes_to_gb(total - avail), "available_gb": _bytes_to_gb(avail)}


def get_cpu_info() -> dict[str, Any]:
    return {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def get_process_rss_gb() -> float | None:
    try:
        import psutil
        return _bytes_to_gb(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def adaptive_memory_reserve_gb(available_gb: float) -> float:
    if available_gb <= 0: return 0.0
    return min(MAX_MEMORY_RESERVE_GB, max(MIN_MEMORY_RESERVE_GB, available_gb * MEMORY_RESERVE_FRACTION))


def system_fingerprint() -> str:
    m, c = get_memory_info(), get_cpu_info()
    return stable_hash({
        "platform": c["platform"], "architecture": c["architecture"], "logical_cpus": c["logical_cpus"],
        "python": c["python"], "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "total_ram_gb": round(m["total_gb"], 4),
        "torch_threads": c["torch_num_threads"], "torch_interop_threads": c["torch_num_interop_threads"],
    }, 16)


def get_best_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def model_device(model: torch.nn.Module) -> torch.device:
    """Diagnostic-only model device resolver; never moves the model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def synchronize_device(device: torch.device | None) -> None:
    if device is None: return
    try:
        if device.type == "cuda": torch.cuda.synchronize()
        elif device.type == "mps":
            sync = getattr(torch.mps, "synchronize", None)
            if callable(sync): sync()
    except Exception:
        pass


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()


def dtype_from_name(name: str) -> torch.dtype:
    table = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    key = str(name).lower().replace("torch.", "")
    if key not in table: raise ValueError(f"Unsupported dtype: {name}")
    return table[key]


def print_system_report(show_verbose: bool = True) -> None:
    m, c, s = get_memory_info(), get_cpu_info(), get_external_storage_usage()
    _print_verbose(
        "\n" + _separator("─") + "\nSYSTEM / STORAGE REPORT\n"
        f"  Architecture        : {c['architecture']}\n"
        f"  Logical CPUs        : {c['logical_cpus']}\n"
        f"  PyTorch threads     : {c['torch_num_threads']}\n"
        f"  PyTorch interop     : {c['torch_num_interop_threads']}\n"
        f"  Python              : {c['python']}\n"
        f"  RAM total           : {m['total_gb']:.2f} GiB\n"
        f"  RAM available       : {m['available_gb']:.2f} GiB\n"
        f"  Adaptive reserve    : {adaptive_memory_reserve_gb(m['available_gb']):.2f} GiB\n"
        f"  External free       : {s['available_gb']:.2f} GiB\n"
        f"  System fingerprint  : {system_fingerprint()}",
        show_verbose,
    )


def configure_cpu_threads(thread_count: int | None) -> None:
    if thread_count is not None: torch.set_num_threads(int(thread_count))


def configure_cpu_interop_threads(thread_count: int | None) -> None:
    if thread_count is None: return
    try: torch.set_num_interop_threads(int(thread_count))
    except RuntimeError: pass

# =============================================================================
# RUNTIME DIAGNOSTICS
# =============================================================================

def runtime_diagnostic_snapshot(*, stage: str, model_name: str | None = None, dataset_name: str | None = None, experiment_id: str | None = None, hyperparameter_hash: str | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "time": time.time(), "stage": stage, "model": model_name, "dataset": dataset_name,
        "experiment_id": experiment_id, "hyperparameter_hash": hyperparameter_hash,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "device": str(get_best_device()), "system_fingerprint": system_fingerprint(),
        "memory": get_memory_info(), "storage": get_external_storage_usage(), "cpu": get_cpu_info(),
        "process_rss_gb": get_process_rss_gb(),
    }
    if extra: record.update(dict(extra))
    return record


def write_runtime_diagnostic(path: str | Path, record: Mapping[str, Any]) -> None:
    save_json(path, dict(record))

# =============================================================================
# HYPERPARAMETERS / EXPERIMENTS
# =============================================================================

@dataclass(frozen=True)
class HyperParameters:
    experiment_id: str
    pooling: str = DEFAULT_POOLING
    max_length: int = DEFAULT_MAX_LENGTH
    storage_dtype: str = DEFAULT_STORAGE_DTYPE
    cpu_dtype: str = "float16"
    accelerator_dtype: str = "float16"
    batch_sizes: dict[str, int] = field(default_factory=dict)
    cpu_threads: int | None = None
    cpu_interop_threads: int | None = None
    use_sdpa: bool = USE_SDPA
    flush_every_batches: int = DEFAULT_FLUSH_EVERY_BATCHES
    flush_every_seconds: float = DEFAULT_FLUSH_SECONDS
    download_retries: int = DOWNLOAD_RETRIES
    download_max_workers: int = DOWNLOAD_MAX_WORKERS
    download_backoff_seconds: float = DOWNLOAD_BACKOFF_SECONDS
    strict_meta_validation: bool = STRICT_META_VALIDATION
    enforce_environment_fingerprint: bool = True

    def validate(self) -> None:
        if not self.experiment_id.strip(): raise ValueError("experiment_id cannot be empty")
        if self.pooling not in {"first_token", "mean", "last_token"}: raise ValueError("invalid pooling")
        if self.max_length < 1: raise ValueError("max_length must be >= 1")
        if self.storage_dtype.lower() not in {"float32", "float16"}: raise ValueError("invalid storage dtype")
        if self.cpu_dtype.lower() not in {"float32", "bfloat16"}: raise ValueError("invalid CPU dtype")
        if self.accelerator_dtype.lower() not in {"float16", "bfloat16", "float32"}: raise ValueError("invalid accelerator dtype")
        if self.flush_every_batches < 1 or self.flush_every_seconds <= 0: raise ValueError("invalid flush settings")
        if self.download_retries < 1 or self.download_max_workers < 1 or self.download_backoff_seconds < 0: raise ValueError("invalid download settings")
        for name, batch in self.batch_sizes.items():
            if int(batch) < 1: raise ValueError(f"invalid batch size for {name}: {batch}")

    def as_dict(self) -> dict[str, Any]: return asdict(self)
    @property
    def config_hash(self) -> str: return stable_hash(self.as_dict(), 16)
    def batch_for(self, model_name: str, spec: ModelSpec) -> int: return int(self.batch_sizes.get(model_name, spec.batch_hint))
    def dtype_for(self, device: torch.device) -> torch.dtype: return dtype_from_name(self.cpu_dtype if device.type == "cpu" else self.accelerator_dtype)


def default_hyperparameters() -> HyperParameters:
    return HyperParameters(DEFAULT_EXPERIMENT_ID, batch_sizes={m.name: m.batch_hint for m in MODEL_REGISTRY})


def ensure_hyperparameter_file(path: str | Path = HYPERPARAMETER_CONFIG_PATH) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): save_json(path, default_hyperparameters().as_dict())
    return path


def load_hyperparameters(path: str | Path = HYPERPARAMETER_CONFIG_PATH) -> HyperParameters:
    path = ensure_hyperparameter_file(path)
    with path.open("r", encoding="utf-8") as f: data = json.load(f)
    params = HyperParameters(**data); params.validate(); return params


def experiment_root(base_output: str | Path, experiment_id: str) -> Path:
    root = Path(base_output).resolve() / "experiments" / experiment_id
    root.mkdir(parents=True, exist_ok=True); return root


def load_existing_experiment_parameters(exp_root: str | Path) -> HyperParameters:
    path = Path(exp_root) / "run_manifest.json"
    if not path.exists(): raise FileNotFoundError(f"Missing experiment manifest: {path}")
    with path.open("r", encoding="utf-8") as f: manifest = json.load(f)
    saved = manifest.get("hyperparameters")
    if not isinstance(saved, Mapping): raise RuntimeError(f"Manifest missing hyperparameters: {path}")
    params = HyperParameters(**saved); params.validate()
    if manifest.get("experiment_id") and manifest["experiment_id"] != params.experiment_id: raise RuntimeError("Experiment manifest is internally inconsistent.")
    if params.experiment_id != Path(exp_root).name: raise RuntimeError("Experiment directory and experiment_id disagree.")
    return params


def resolve_experiment_parameters(base_output: str | Path, initial_params: HyperParameters, *, batch_size: int | None = None, pooling: str | None = None, max_length: int | None = None, use_half_precision: bool | None = None, flush_every_batches: int | None = None, experiment_id: str | None = None, show_info: bool = True) -> tuple[HyperParameters, Path, bool]:
    base_output = Path(base_output).resolve(); requested_id = experiment_id or initial_params.experiment_id
    root = experiment_root(base_output, requested_id); manifest = root / "run_manifest.json"
    if manifest.exists():
        existing = load_existing_experiment_parameters(root)
        requested = {}
        if batch_size is not None: requested["batch_size"] = batch_size
        if pooling is not None: requested["pooling"] = pooling
        if max_length is not None: requested["max_length"] = max_length
        if use_half_precision is not None: requested["use_half_precision"] = use_half_precision
        if flush_every_batches is not None: requested["flush_every_batches"] = flush_every_batches
        _print_info("\n↻ EXISTING EXPERIMENT DETECTED\n" f"  Experiment ID        : {existing.experiment_id}\n" f"  Saved hyperparameter : {existing.config_hash}\n" f"  Manifest             : {manifest}\n" "  The saved experiment configuration is authoritative.\n" "  Current-call overrides are ignored for this resume.\n" f"  Requested overrides  : {requested}\n", show_info)
        return existing, root, True
    overrides: dict[str, Any] = {}
    if batch_size is not None:
        overrides["batch_sizes"] = {**initial_params.batch_sizes, **{name: int(batch_size) for name in initial_params.batch_sizes}}
    if pooling is not None: overrides["pooling"] = pooling
    if max_length is not None: overrides["max_length"] = int(max_length)
    if flush_every_batches is not None: overrides["flush_every_batches"] = int(flush_every_batches)
    if use_half_precision is False:
        overrides["cpu_dtype"] = "float32"; overrides["accelerator_dtype"] = "float32"
    elif use_half_precision is True:
        overrides["accelerator_dtype"] = "float16"
    if experiment_id is not None: overrides["experiment_id"] = experiment_id
    params = HyperParameters(**{**initial_params.as_dict(), **overrides}); params.validate()
    root = experiment_root(base_output, params.experiment_id)
    _print_info(f"\n✦ NEW EXPERIMENT\n  Experiment ID        : {params.experiment_id}\n  Hyperparameter hash  : {params.config_hash}\n", show_info)
    return params, root, False


def write_experiment_manifest(exp_root: Path, params: HyperParameters, model_names: Sequence[str], device: torch.device, environment: Mapping[str, Any]) -> None:
    path = exp_root / "run_manifest.json"
    current = {"experiment_id": params.experiment_id, "hyperparameter_hash": params.config_hash, "hyperparameters": params.as_dict(), "models": list(model_names), "device": str(device), "environment": dict(environment), "created_at": time.time()}
    if path.exists():
        with path.open("r", encoding="utf-8") as f: old = json.load(f)
        if old.get("hyperparameter_hash") != params.config_hash: raise RuntimeError("Existing experiment hyperparameter hash mismatch.")
        old_fp = old.get("environment", {}).get("system_fingerprint"); new_fp = environment.get("system_fingerprint")
        if params.enforce_environment_fingerprint and old_fp and new_fp and old_fp != new_fp: raise RuntimeError("Environment fingerprint changed for an existing experiment.")
        old_models = list(old.get("models", []))
        if old_models and old_models != list(model_names): raise RuntimeError("Existing experiment model matrix differs from current matrix.")
        return
    save_json(path, current); save_json(exp_root / "hyperparameters.json", params.as_dict())

# =============================================================================
# EXTERNAL STORAGE / HF CACHE
# =============================================================================

def verify_external_drive() -> None:
    if not EXTERNAL_MOUNT.is_dir(): raise RuntimeError(f"EXTERNAL DRIVE NOT MOUNTED\nExpected: {EXTERNAL_MOUNT}")


def configure_external_storage() -> None:
    verify_external_drive()
    for p in (EXTERNAL_ROOT, HF_ROOT, HF_HUB_CACHE, HF_XET_CACHE, HF_ASSETS_CACHE, EXPERIMENTS_ROOT): p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_ROOT); os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE); os.environ["HF_XET_CACHE"] = str(HF_XET_CACHE); os.environ["HF_ASSETS_CACHE"] = str(HF_ASSETS_CACHE)


configure_external_storage()


def verify_huggingface_storage(show_verbose: bool = True) -> None:
    verify_external_drive()
    for name, path in (("HF root", HF_ROOT), ("Hub cache", HF_HUB_CACHE), ("Xet cache", HF_XET_CACHE), ("Assets cache", HF_ASSETS_CACHE)):
        if not path.is_dir(): raise RuntimeError(f"Required external path missing: {path}")
        _print_verbose(f"  {name:<16}: {path} ✓", show_verbose)


def load_revision_manifest() -> dict[str, str]:
    if not MODEL_REVISION_MANIFEST.exists(): return {}
    with MODEL_REVISION_MANIFEST.open("r", encoding="utf-8") as f: data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def save_revision_manifest(data: Mapping[str, str]) -> None:
    save_json(MODEL_REVISION_MANIFEST, dict(sorted(data.items())))


def get_or_pin_model_revision(model_name: str, show_verbose: bool = True) -> str:
    revisions = load_revision_manifest()
    if model_name in revisions:
        _print_verbose(f"  Pinned revision      : {revisions[model_name]}", show_verbose); return revisions[model_name]
    try:
        info = HfApi().model_info(repo_id=model_name); revision = getattr(info, "sha", None)
    except Exception as exc: raise RuntimeError(f"MODEL REVISION LOOKUP FAILED\nModel: {model_name}\n{type(exc).__name__}: {exc}") from exc
    if not revision: raise RuntimeError(f"Hugging Face returned no commit SHA for {model_name}")
    revisions[model_name] = revision; save_revision_manifest(revisions)
    _print_verbose(f"  New pinned revision : {revision}", show_verbose); return revision


def _repository_files(model_name: str, revision: str | None = None) -> list[str]:
    try:
        api = HfApi(); kwargs = {"repo_id": model_name};
        if revision is not None: kwargs["revision"] = revision
        try: files = api.list_repo_files(repo_type="model", **kwargs)
        except TypeError as exc:
            if "repo_type" not in str(exc): raise
            files = api.list_repo_files(**kwargs)
        files = [str(x) for x in files if x]
        if not files: raise RuntimeError(f"Empty repository file list for {model_name} at {revision}")
        return files
    except Exception as exc:
        raise RuntimeError(f"HUGGING FACE FILE DISCOVERY FAILED\nModel: {model_name}\nRevision: {revision}\n{type(exc).__name__}: {exc}") from exc


def _download_patterns(model_name: str, revision: str | None = None) -> tuple[list[str], list[str]]:
    files = _repository_files(model_name, revision)
    has_safe = any(x.endswith(".safetensors") for x in files); has_bin = any(x.endswith(".bin") for x in files)
    if not has_safe and not has_bin: raise RuntimeError(f"No supported PyTorch checkpoint found for {model_name} at {revision}")
    allow = list(COMMON_MODEL_FILES) + (["*.safetensors"] if has_safe else ["*.bin"])
    if get_model_spec(model_name).trust_remote_code: allow.append("*.py")
    ignore = list(UNNECESSARY_FRAMEWORK_PATTERNS)
    if has_safe: ignore.append("*.bin")
    return allow, ignore


def _is_transient_download_error(exc: BaseException) -> bool:
    # Traverse the cause chain to find original network errors
    current = exc
    while current is not None:
        if isinstance(current, (ConnectionError, TimeoutError, HfHubHTTPError,
                                IncompleteSnapshotError, LocalEntryNotFoundError)):
            return True
        text = str(current).lower()
        markers = (
            "remote end closed connection", "connection aborted", "connection reset",
            "connection refused", "timed out", "timeout", "temporarily unavailable",
            "502", "503", "504", "connection error", "protocolerror",
            "server disconnected"
        )
        if any(m in text for m in markers):
            return True
        current = current.__cause__ or current.__context__
    return False

def _snapshot_download_compat(model_name: str, revision: str, allow_patterns: list[str], ignore_patterns: list[str], workers: int) -> Path:
    kwargs = {"repo_id": model_name, "revision": revision, "cache_dir": str(HF_HUB_CACHE), "allow_patterns": allow_patterns, "ignore_patterns": ignore_patterns, "max_workers": workers, "etag_timeout": DOWNLOAD_ETAG_TIMEOUT_SECONDS}
    try: snapshot = snapshot_download(repo_type="model", **kwargs)
    except TypeError as exc:
        if "repo_type" not in str(exc): raise
        snapshot = snapshot_download(**kwargs)
    return Path(snapshot).resolve()

# =============================================================================
# CHECKPOINT / PREFLIGHT
# =============================================================================

def checkpoint_inventory(snapshot_path: Path) -> dict[str, Any]:
    safe = sorted(p for p in snapshot_path.rglob("*.safetensors") if p.is_file())
    bins = sorted(p for p in snapshot_path.rglob("*.bin") if p.is_file())
    indexes = sorted(list(snapshot_path.rglob("*.safetensors.index.json")) + list(snapshot_path.rglob("pytorch_model.bin.index.json")))
    return {
        "safetensors": [{"path": str(p), "filename": p.name, "size_bytes": int(p.stat().st_size), "size_gb": _bytes_to_gb(p.stat().st_size)} for p in safe],
        "pytorch_bin": [{"path": str(p), "filename": p.name, "size_bytes": int(p.stat().st_size), "size_gb": _bytes_to_gb(p.stat().st_size)} for p in bins],
        "index_files": [str(p) for p in indexes],
        "safetensors_total_bytes": sum(int(p.stat().st_size) for p in safe),
        "pytorch_bin_total_bytes": sum(int(p.stat().st_size) for p in bins),
    }


def model_weight_bytes(snapshot_path: Path) -> int:
    inv = checkpoint_inventory(snapshot_path)
    if inv["safetensors"]: return int(inv["safetensors_total_bytes"])
    if inv["pytorch_bin"]: return int(inv["pytorch_bin_total_bytes"])
    return 0


def _validate_snapshot(snapshot: Path) -> None:
    if not snapshot.exists(): raise RuntimeError(f"Snapshot does not exist: {snapshot}")
    try: snapshot.relative_to(EXTERNAL_ROOT)
    except ValueError as exc: raise RuntimeError(f"Snapshot escaped external root: {snapshot}") from exc
    if not (snapshot / "config.json").exists(): raise RuntimeError(f"Snapshot incomplete: missing {snapshot / 'config.json'}")
    inv = checkpoint_inventory(snapshot)
    if not inv["safetensors"] and not inv["pytorch_bin"] and not inv["index_files"]: raise RuntimeError(f"Snapshot contains no supported PyTorch checkpoint files: {snapshot}")
    if inv["index_files"] and not (inv["safetensors"] or inv["pytorch_bin"]): raise RuntimeError(f"Checkpoint index exists but no shards are present: {snapshot}")


def prepare_model(model_name: str, hyperparameters: HyperParameters, show_verbose: bool = True, show_info: bool = True, show_critical: bool = True) -> tuple[Path, str, float]:
    verify_huggingface_storage(show_verbose)
    _print_info(f"\n→ Preparing Hugging Face model: {model_name}", show_info)
    started = time.perf_counter(); revision = get_or_pin_model_revision(model_name, show_verbose); last_exc: BaseException | None = None
    for attempt in range(1, hyperparameters.download_retries + 1):
        attempt_started = time.perf_counter()
        try:
            _print_info(f"  Preparation attempt {attempt}/{hyperparameters.download_retries}", show_info)
            allow, ignore = _download_patterns(model_name, revision)
            _print_verbose("  DOWNLOAD DIAGNOSTICS\n" f"    Revision             : {revision}\n" f"    Cache                : {HF_HUB_CACHE}\n" f"    Workers              : {hyperparameters.download_max_workers}\n" f"    ETag timeout         : {DOWNLOAD_ETAG_TIMEOUT_SECONDS}s\n" f"    Allow patterns       : {allow}\n" f"    Ignore patterns      : {ignore}", show_verbose)
            snapshot = _snapshot_download_compat(model_name, revision, allow, ignore, hyperparameters.download_max_workers)
            _validate_snapshot(snapshot)
            total_elapsed = time.perf_counter() - started; attempt_elapsed = time.perf_counter() - attempt_started
            _print_info("  ✓ Snapshot ready\n" f"    Attempt time        : {_format_duration(attempt_elapsed)}\n" f"    Total preparation   : {_format_duration(total_elapsed)}\n" f"    Checkpoint size     : {_bytes_to_gb(model_weight_bytes(snapshot)):.3f} GiB\n" f"    Snapshot            : {snapshot}", show_info)
            return snapshot, revision, total_elapsed
        except Exception as exc:
            last_exc = exc; transient = _is_transient_download_error(exc)
            _print_critical("\n" + _separator("!") + "\n" f"MODEL PREPARATION FAILURE — attempt {attempt}/{hyperparameters.download_retries}\n" f"  Model                : {model_name}\n" f"  Revision             : {revision}\n" f"  Error type           : {type(exc).__name__}\n" f"  Transient            : {transient}\n" f"  Error                : {exc}\n" + _separator("!"), show_critical)
            _print_verbose(f"  Traceback:\n{_truncate_traceback(traceback.format_exc())}", show_verbose)
            if not transient or attempt >= hyperparameters.download_retries: break
            sleep_for = hyperparameters.download_backoff_seconds * (2 ** (attempt - 1)); _print_info(f"  ↻ Retrying in {sleep_for:.1f}s...", show_info); time.sleep(sleep_for)
    raise RuntimeError(f"MODEL PREPARATION FAILED\nModel: {model_name}\nRevision: {revision}\nCache: {HF_HUB_CACHE}\n{type(last_exc).__name__ if last_exc else 'UnknownError'}: {last_exc}") from last_exc


def get_model_num_layers(config: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attr, None)
        if value is not None: return int(value)
    raise AttributeError(f"Cannot determine layer count for {type(config).__name__}")


def get_model_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, attr, None)
        if value is not None: return int(value)
    raise AttributeError(f"Cannot determine hidden size for {type(config).__name__}")


def checkpoint_dtype_from_config(config: Any) -> torch.dtype | None:
    value = getattr(config, "torch_dtype", None)
    if isinstance(value, torch.dtype): return value
    if isinstance(value, str): return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}.get(value.replace("torch.", "").lower())
    return None


def estimate_loaded_weight_gb(snapshot_path: Path, config: Any, dtype: torch.dtype, spec: ModelSpec) -> float:
    on_disk = model_weight_bytes(snapshot_path)
    if on_disk:
        source = checkpoint_dtype_from_config(config)
        if source in {torch.float16, torch.bfloat16} and dtype == torch.float32: return _bytes_to_gb(on_disk * 2)
        if source == torch.float32 and dtype in {torch.float16, torch.bfloat16}: return _bytes_to_gb(on_disk * 0.5)
        return _bytes_to_gb(on_disk)
    bpp = 2 if dtype in {torch.float16, torch.bfloat16} else 4
    return _bytes_to_gb(spec.parameter_billions * 1e9 * bpp)


def preflight_model(spec: ModelSpec, snapshot_path: Path, config: Any, device: torch.device, dtype: torch.dtype, max_length: int, show_verbose: bool = True) -> dict[str, Any]:
    memory = get_memory_info(); inv = checkpoint_inventory(snapshot_path); layers = get_model_num_layers(config); hidden = get_model_hidden_size(config); active = model_weight_bytes(snapshot_path); loaded = estimate_loaded_weight_gb(snapshot_path, config, dtype, spec); safe_total = memory["total_gb"] * MAX_MODEL_MEMORY_FRACTION
    report = {
        "model": spec.name, "model_type": getattr(config, "model_type", None), "architecture": spec.architecture,
        "dtype": str(dtype), "device": str(device), "layers": layers, "hidden_size": hidden, "hidden_states": layers + 1,
        "max_length": max_length, "checkpoint": {"active_bytes": active, "active_gb": _bytes_to_gb(active), "inventory": inv},
        "estimated_loaded_weight_gb": loaded, "estimated_model_load_peak_gb": loaded * MODEL_LOAD_SAFETY_FACTOR,
        "memory": {"total_gb": memory["total_gb"], "available_gb": memory["available_gb"], "used_gb": memory["used_gb"], "reserve_gb": adaptive_memory_reserve_gb(memory["available_gb"]), "safe_total_memory_gb": safe_total},
        "status": "ok" if loaded <= safe_total else "skip_total_memory",
    }
    _print_verbose("\n" + _separator("─") + "\nMODEL PREFLIGHT\n" + _separator("─") + "\n" + "\n".join(f"  {k:<34}: {v}" for k, v in report.items()), show_verbose)
    return report

# =============================================================================
# LOADING / PROBING
# =============================================================================

def has_meta_tensors(model: torch.nn.Module) -> tuple[bool, list[str]]:
    names = []
    for n, p in model.named_parameters(recurse=True):
        if getattr(p, "is_meta", False) or p.device.type == "meta": names.append(f"parameter:{n}")
    for n, b in model.named_buffers(recurse=True):
        if getattr(b, "is_meta", False) or b.device.type == "meta": names.append(f"buffer:{n}")
    return bool(names), names


def _accelerate_available() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


def get_forward_input_keys(model: torch.nn.Module) -> tuple[set[str], bool]:
    try: signature = inspect.signature(model.forward)
    except (TypeError, ValueError): return set(), True
    params = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepted = {n for n, p in params.items() if p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}}
    return accepted, accepts_kwargs


def prepare_model_inputs(tokenized: Mapping[str, torch.Tensor], device: torch.device, accepted_keys: set[str], accepts_kwargs: bool) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=(device.type != "cpu")) for k, v in tokenized.items() if accepts_kwargs or k in accepted_keys}


def _normalise_key_list(value: Any) -> list[str]:
    if value is None: return []
    if isinstance(value, str): return [value]
    try: return [str(x) for x in value]
    except TypeError: return [str(value)]


def classify_model_loading_issues(model: torch.nn.Module, loading_info: Mapping[str, Any] | None) -> dict[str, Any]:
    info = dict(loading_info or {})
    missing = _normalise_key_list(info.get("missing_keys")); unexpected = _normalise_key_list(info.get("unexpected_keys")); mismatched = _normalise_key_list(info.get("mismatched_keys"))
    non_core_missing, core_missing = [], []
    for name in missing:
        normalized = name[len("parameter:"):] if name.startswith("parameter:") else name
        (non_core_missing if any(normalized == p.rstrip(".") or normalized.startswith(p) for p in NON_CORE_MODEL_PARAMETER_PREFIXES) else core_missing).append(name)
    _, meta_names = has_meta_tensors(model)
    non_core_meta, core_meta = [], []
    for name in meta_names:
        normalized = name.split(":", 1)[-1]
        (non_core_meta if any(normalized == p.rstrip(".") or normalized.startswith(p) for p in NON_CORE_MODEL_PARAMETER_PREFIXES) else core_meta).append(name)
    return {"missing_keys": missing, "unexpected_keys": unexpected, "mismatched_keys": mismatched, "non_core_missing_keys": non_core_missing, "core_missing_keys": core_missing, "all_meta_tensors": meta_names, "non_core_meta_tensors": non_core_meta, "core_meta_tensors": core_meta, "has_meta_tensors": bool(meta_names), "fatal": bool(core_missing or mismatched or core_meta)}


def classify_loading_info(model: torch.nn.Module, loading_info: Mapping[str, Any] | None) -> dict[str, Any]:
    return classify_model_loading_issues(model, loading_info)


def disable_optional_meta_modules(model: torch.nn.Module, loading_report: Mapping[str, Any]) -> dict[str, Any]:
    interventions = []
    meta_names = list(loading_report.get("all_meta_tensors", []))
    if any("pooler." in n for n in meta_names) and hasattr(model, "pooler"):
        pooler = getattr(model, "pooler"); setattr(model, "pooler", None)
        interventions.append({"type": "disable_optional_module", "module": "pooler", "reason": "Pooler had meta tensors and is not required for hidden-state extraction.", "original_type": type(pooler).__name__})
    return {"interventions": interventions, "remaining_meta_tensors": has_meta_tensors(model)[1]}


def _load_model_from_snapshot(snapshot_path: Path, dtype: torch.dtype, trust_remote_code: bool, use_sdpa: bool, low_cpu_mem_usage: bool) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    kwargs: dict[str, Any] = {"local_files_only": True, "output_hidden_states": True, "torch_dtype": dtype, "trust_remote_code": trust_remote_code, "output_loading_info": True}
    if low_cpu_mem_usage and _accelerate_available(): kwargs["low_cpu_mem_usage"] = True
    if use_sdpa: kwargs["attn_implementation"] = "sdpa"
    backend = "sdpa" if use_sdpa else "model_default"
    try:
        result = AutoModel.from_pretrained(str(snapshot_path), **kwargs)
    except TypeError as exc:
        msg = str(exc).lower()
        if "attn_implementation" in msg:
            kwargs.pop("attn_implementation", None); result = AutoModel.from_pretrained(str(snapshot_path), **kwargs); backend = "model_default"
        elif "output_loading_info" in msg:
            kwargs.pop("output_loading_info", None); result = AutoModel.from_pretrained(str(snapshot_path), **kwargs); return result, backend, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "diagnostic_warning": "output_loading_info unsupported by installed Transformers"}
        else: raise
    if isinstance(result, tuple) and len(result) == 2: model, loading_info = result
    else: model, loading_info = result, {}
    return model, backend, dict(loading_info or {})


def run_model_probe(candidate: torch.nn.Module, tokenizer: Any, device: torch.device, max_length: int) -> dict[str, Any]:
    if has_meta_tensors(candidate)[0]: raise RuntimeError(f"MODEL STILL CONTAINS META TENSORS BEFORE PROBE: {has_meta_tensors(candidate)[1]}")
    accepted, accepts_kwargs = get_forward_input_keys(candidate)
    tokenized = tokenizer(["hidden state probe"], padding=True, truncation=True, max_length=min(int(max_length), MODEL_PROBE_MAX_LENGTH), return_tensors="pt", return_attention_mask=True)
    if "input_ids" not in tokenized: raise RuntimeError("Tokenizer probe produced no input_ids.")
    inputs = prepare_model_inputs(tokenized, device, accepted, accepts_kwargs)
    if "attention_mask" in tokenized: inputs["attention_mask"] = tokenized["attention_mask"].to(device, non_blocking=(device.type != "cpu"))
    if "use_cache" in accepted or accepts_kwargs: inputs["use_cache"] = False
    candidate.eval(); synchronize_device(device)
    with torch.inference_mode(): outputs = candidate(**inputs)
    synchronize_device(device)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) == 0: raise RuntimeError("Model probe returned no hidden states.")
    seq = int(tokenized["input_ids"].shape[1]); batch = int(tokenized["input_ids"].shape[0]); shapes = [tuple(int(d) for d in x.shape) for x in hidden_states]
    for i, x in enumerate(hidden_states):
        if x.ndim != 3: raise RuntimeError(f"Probe hidden state {i} is not rank-3: {x.shape}")
        if x.shape[0] != batch or x.shape[1] != seq: raise RuntimeError(f"Probe hidden state {i} has wrong shape: {tuple(x.shape)} expected batch={batch}, seq={seq}")
        if not bool(torch.isfinite(x).all()): raise RuntimeError(f"Probe hidden state {i} contains NaN/Inf")
    return {"probe_hidden_state_count": len(hidden_states), "probe_hidden_size": int(hidden_states[-1].shape[-1]), "probe_sequence_length": seq, "probe_hidden_state_shapes": shapes, "probe_dtypes": sorted({str(x.dtype) for x in hidden_states}), "device": str(model_device(candidate)), "requested_device": str(device), "meta_tensors": has_meta_tensors(candidate)[1], "finite_values": True}


def load_model(snapshot_path: str | Path, model_name: str, tokenizer: Any, dtype: torch.dtype, device: torch.device, trust_remote_code: bool = False, use_sdpa: bool = True, strict_meta_validation: bool = True, probe_max_length: int = DEFAULT_MAX_LENGTH, show_verbose: bool = True, show_info: bool = True, show_critical: bool = True) -> tuple[torch.nn.Module, str, dict[str, Any], float]:
    snapshot = Path(snapshot_path).resolve(); started = time.perf_counter(); attempts: list[dict[str, Any]] = []; interventions: list[dict[str, Any]] = []

    def attempt(name: str, sdpa: bool, low_cpu: bool) -> tuple[torch.nn.Module, str, dict[str, Any]]:
        t0 = time.perf_counter()
        _print_info("\nMODEL LOAD ATTEMPT\n" f"  Attempt              : {len(attempts)+1}/2\n" f"  Path                 : {name}\n" f"  Model                : {model_name}\n" f"  Snapshot             : {snapshot}\n" f"  dtype                : {dtype}\n" f"  device               : {device}\n" f"  SDPA                 : {sdpa}\n" f"  low_cpu_mem_usage    : {low_cpu and _accelerate_available()}", show_info)
        candidate, backend, info = _load_model_from_snapshot(snapshot, dtype, trust_remote_code, sdpa, low_cpu)
        report = classify_model_loading_issues(candidate, info)
        _print_info("  LOADING RESULT\n" f"    Missing keys        : {len(report['missing_keys'])}\n" f"    Non-core missing    : {len(report['non_core_missing_keys'])}\n" f"    Core missing        : {len(report['core_missing_keys'])}\n" f"    Unexpected keys     : {len(report['unexpected_keys'])}\n" f"    Mismatched keys     : {len(report['mismatched_keys'])}\n" f"    Meta tensors        : {len(report['all_meta_tensors'])}\n" f"    Core meta           : {len(report['core_meta_tensors'])}\n" f"    Fatal discrepancy   : {report['fatal']}", show_info)
        _print_verbose("  FULL LOADING DIAGNOSTICS\n" + json.dumps(report, indent=2, default=str), show_verbose)
        intervention_report = disable_optional_meta_modules(candidate, report); interventions.extend(intervention_report["interventions"])
        remaining = intervention_report["remaining_meta_tensors"]
        if strict_meta_validation and remaining: raise RuntimeError(f"Core meta tensors remain after remediation: {remaining}")
        if report["fatal"]: raise RuntimeError("Fatal checkpoint discrepancy:\n" + json.dumps(report, indent=2, default=str))
        if device.type != "cpu": candidate = candidate.to(device); synchronize_device(device)
        candidate.eval(); attempts.append({"attempt": len(attempts)+1, "name": name, "backend": backend, "elapsed_seconds": time.perf_counter()-t0, "loading": report, "interventions": intervention_report["interventions"]})
        return candidate, backend, report

    primary_error: BaseException | None = None
    model = None
    try:
        model, backend, _ = attempt("primary", use_sdpa, USE_LOW_CPU_MEM_USAGE)
        try:
            probe = run_model_probe(model, tokenizer, device, probe_max_length)
        except Exception as exc:
            primary_error = exc; _print_critical("\nMODEL PROBE FAILED\n" f"  Type       : {type(exc).__name__}\n" f"  Error      : {exc}\n" f"  Traceback:\n{_truncate_traceback(traceback.format_exc())}", show_critical)
            del model; model = None; cleanup(device)
        else:
            elapsed = time.perf_counter()-started
            _print_info("\n✓ MODEL READY\n" f"  Load/probe time       : {elapsed:.2f}s\n" f"  Backend               : {backend}\n" f"  Model device          : {probe['device']}\n" f"  Hidden states         : {probe['probe_hidden_state_count']}\n" f"  Hidden size           : {probe['probe_hidden_size']}", show_info)
            return model, backend, {"probe": probe, "loading_attempts": attempts, "interventions": interventions, "fallback_used": False, "primary_error": None}, elapsed
    except Exception as exc:
        primary_error = exc; _print_critical("\nPRIMARY MODEL LOAD FAILED\n" f"  Type       : {type(exc).__name__}\n" f"  Error      : {exc}\n" f"  Traceback:\n{_truncate_traceback(traceback.format_exc())}", show_critical); cleanup(device)

    try:
        model, backend, _ = attempt("materialized_fallback", False, False)
        probe = run_model_probe(model, tokenizer, device, probe_max_length)
        elapsed = time.perf_counter()-started
        _print_info("\n✓ MODEL READY AFTER FALLBACK\n" f"  Total load/probe time : {elapsed:.2f}s\n" f"  Backend               : {backend}", show_info)
        primary_record = None if primary_error is None else {"type": type(primary_error).__name__, "error": str(primary_error), "traceback": _truncate_traceback("".join(traceback.format_exception(type(primary_error), primary_error, primary_error.__traceback__)))}
        return model, backend, {"probe": probe, "loading_attempts": attempts, "interventions": interventions, "fallback_used": True, "primary_error": primary_record}, elapsed
    except Exception as fallback_error:
        if model is not None: del model
        cleanup(device)
        raise RuntimeError("MODEL LOAD/PROBE FAILED AFTER FALLBACK\n" f"Model: {model_name}\n" f"Primary error: {type(primary_error).__name__ if primary_error else 'none'}: {primary_error}\n" f"Fallback error: {type(fallback_error).__name__}: {fallback_error}\n" f"Load attempts:\n{json.dumps(attempts, indent=2, default=str)}") from fallback_error


def load_model_config(snapshot_path: str | Path) -> Any:
    return AutoConfig.from_pretrained(str(Path(snapshot_path).resolve()), local_files_only=True)


def load_tokenizer(snapshot_path: str | Path, model_name: str, trust_remote_code: bool = False, show_verbose: bool = True, show_info: bool = True) -> Any:
    snapshot = Path(snapshot_path).resolve(); started = time.perf_counter(); attempts = []; tokenizer = None; mode = None

    def record(name: str, success: bool, error: BaseException | None = None):
        attempts.append({"mode": name, "success": success, "error_type": type(error).__name__ if error else None, "error": str(error) if error else None})

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True, trust_remote_code=trust_remote_code); record("auto_fast", True); mode = "fast"
    except Exception as exc: record("auto_fast", False, exc); _print_verbose(f"  Fast tokenizer failed: {type(exc).__name__}: {exc}", show_verbose)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=False, local_files_only=True, trust_remote_code=trust_remote_code); record("auto_slow", True); mode = "slow"
        except Exception as exc: record("auto_slow", False, exc); _print_verbose(f"  Slow tokenizer failed: {type(exc).__name__}: {exc}", show_verbose)
    if tokenizer is None:
        try: model_type = getattr(load_model_config(snapshot), "model_type", None)
        except Exception: model_type = None
        if model_type in {"deberta-v2", "deberta-v3"}:
            try:
                from transformers import DebertaV2Tokenizer
                tokenizer = DebertaV2Tokenizer.from_pretrained(str(snapshot), local_files_only=True); record("explicit_deberta_sentencepiece", True); mode = "explicit_deberta_sentencepiece"
            except Exception as exc: record("explicit_deberta_sentencepiece", False, exc)
    if tokenizer is None: raise RuntimeError("TOKENIZER LOAD FAILED\n" f"Model: {model_name}\n" f"Snapshot: {snapshot}\n" f"Attempts:\n{json.dumps(attempts, indent=2, default=str)}")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token; pad_source = "eos"
        elif tokenizer.unk_token is not None: tokenizer.pad_token = tokenizer.unk_token; pad_source = "unk"
        else: raise RuntimeError(f"{model_name}: tokenizer has no usable pad/eos/unk token")
    else: pad_source = "checkpoint-defined"
    _print_info(f"  ✓ Tokenizer loaded in {time.perf_counter()-started:.2f}s Mode={mode}", show_info)
    _print_verbose("  TOKENIZER DIAGNOSTICS\n" f"    class              : {type(tokenizer).__name__}\n" f"    vocab size         : {len(tokenizer)}\n" f"    pad token          : {tokenizer.pad_token!r}\n" f"    pad source         : {pad_source}\n" f"    attempts           : {json.dumps(attempts, default=str)}", show_verbose)
    return tokenizer

# =============================================================================
# DATASET / STORAGE (v2 unified format)
# =============================================================================
def get_dataset_columns(dataset: Any) -> list[str]:
    """Return a list of column names from a Hugging Face Dataset or pandas DataFrame."""
    if hasattr(dataset, "columns"):
        # pandas DataFrame or similar
        cols = dataset.columns
        if isinstance(cols, (list, tuple)):
            return list(cols)
        # If it's a pandas Index, convert to list
        return list(cols)
    if hasattr(dataset, "column_names"):
        return list(dataset.column_names)
    if hasattr(dataset, "features"):
        return list(dataset.features.keys())
    return []

def _column_diagnostic(dataset: Any, column: str) -> dict[str, Any]:
    info = {"column": column}
    try:
        values = dataset[column]; sample = list(values[:25]) if hasattr(values, "__getitem__") else list(values)[:25]; non_null = [v for v in sample if v is not None]
        info["sample_count"] = len(sample); info["string_count"] = sum(isinstance(v, str) for v in sample); info["avg_chars"] = sum(len(str(v)) for v in non_null) / max(1, len(non_null)); info["examples"] = [str(v)[:120] for v in sample[:3]]
    except Exception as exc: info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def detect_text_column(dataset: Any, requested: str | None = None, *, show_verbose: bool = False) -> str:
    columns = get_dataset_columns(dataset)
    if not columns: raise KeyError("Dataset exposes no columns")
    if requested is not None:
        if requested not in columns: raise KeyError(f"Requested text column {requested!r} not found. Available: {columns}")
        return requested
    lowered = {c.lower(): c for c in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered and lowered[candidate].lower() not in TEXT_COLUMN_EXCLUDE:
            chosen = lowered[candidate]; _print_verbose(f"  Text-column resolver: exact candidate {chosen!r}", show_verbose); return chosen
    scored = []
    for column in columns:
        if column.lower() in TEXT_COLUMN_EXCLUDE: continue
        diag = _column_diagnostic(dataset, column); count = int(diag.get("sample_count", 0)); strings = int(diag.get("string_count", 0)); avg = float(diag.get("avg_chars", 0.0)); ratio = strings / max(1, count)
        if ratio >= 0.90 and avg >= 5.0: scored.append((ratio * 100 + min(avg, 500) / 10, column, diag))
    if not scored: raise KeyError("No safe text column found; refusing to guess from labels/IDs.\n" + json.dumps([_column_diagnostic(dataset, c) for c in columns], indent=2, default=str))
    scored.sort(reverse=True); chosen = scored[0][1]
    _print_verbose("  Text-column candidates:\n" + "\n".join(f"    score={s:6.1f} column={c!r} avg_chars={d.get('avg_chars',0):.1f}" for s,c,d in scored) + f"\n  Selected text column: {chosen!r}", show_verbose)
    return chosen


def dataset_texts(dataset: Any, text_column: str) -> list[str]:
    values = dataset[text_column]
    if hasattr(values, "fillna"): values = values.fillna("")
    return ["" if v is None else str(v) for v in values]


def dataset_fingerprint(dataset: Any, texts: Sequence[str]) -> str:
    native = getattr(dataset, "_fingerprint", None)
    if native: return str(native)
    return stable_hash({"n": len(texts), "head": list(texts[:16]), "tail": list(texts[-16:]) if texts else []}, 20)


def dataset_signature(dataset: Any, texts: Sequence[str], text_column: str) -> dict[str, Any]:
    return {"native_fingerprint": str(getattr(dataset, "_fingerprint", None)) if getattr(dataset, "_fingerprint", None) else None, "derived_fingerprint": dataset_fingerprint(dataset, texts), "sample_count": len(texts), "text_column": text_column, "head_hash": stable_hash(list(texts[:100]), 20), "tail_hash": stable_hash(list(texts[-100:]), 20)}

def build_model_directory(base_output: str | Path, model_name: str) -> Path:
    return Path(base_output) / "models" / Path(*[p for p in model_name.split("/") if p])

def build_dataset_directory(base_output: str | Path, model_name: str, dataset_name: str) -> Path:
    return build_model_directory(base_output, model_name) / "datasets" / dataset_name

def build_dataset_storage_paths(dataset_dir: str | Path) -> dict[str, Path]:
    d = Path(dataset_dir)
    data = d / "data"
    meta = d / "metadata"
    data.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_dir": d, "data_dir": data, "metadata_dir": meta,
        "states": data / "hidden_states.npy",
        "completed": data / "completed.npy",
        "labels": data / "labels.npy",
        "label_codes": meta / "label_codes.npy",
        "sample_ids": meta / "sample_ids.npy",          # v2: object dtype (strings)
        "text_hashes": meta / "text_hashes.npy",        # v2: uint8 (n_samples, 32) optional
        "checksum": meta / "checksum.sha256",           # v2: global SHA256 of hidden_states.npy
        "integrity": meta / "integrity_hashes.jsonl",
        "metadata": meta / "extraction.json",
        "events": meta / "runtime_events.jsonl",
        "runtime": meta / "runtime_state.json",
        # No per‑sample manifest – we use sample_ids + text_hashes
    }


def flush_array(array: np.ndarray) -> None:
    """Flush memmap data and force it to physical storage using fsync on the file only."""
    if not isinstance(array, np.memmap):
        return
    array.flush()
    path = Path(array.filename)
    if path.exists():
        fd = os.open(str(path), os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    # No directory fsync – it's too expensive and not required for integrity
            


def _metadata_matches_completion_signature(metadata: Mapping[str, Any], expected: Mapping[str, Any] | None) -> bool:
    if expected is None: return True
    d = metadata.get("dataset", {}); e = metadata.get("extraction", {})
    actual = {"experiment_id": metadata.get("experiment_id"), "hyperparameter_hash": metadata.get("hyperparameter_hash"), "dataset_fingerprint": d.get("fingerprint"), "text_column": d.get("text_column"), "sample_count": d.get("samples"), "pooling": e.get("pooling"), "max_length": e.get("max_length"), "batch_size": e.get("batch_size")}
    return all(actual.get(k) == v for k, v in expected.items())


def dataset_output_is_complete(output_dir: str | Path, *, expected_signature: Mapping[str, Any] | None = None) -> bool:
    p = build_dataset_storage_paths(output_dir)
    if not p["completed"].exists() or not p["states"].exists() or not p["metadata"].exists(): return False
    try:
        c = np.load(p["completed"], mmap_mode="r")
        if not isinstance(c, np.memmap) or c.dtype != np.bool_ or c.ndim != 1 or not bool(c.all()): return False
        with p["metadata"].open("r", encoding="utf-8") as f: meta = json.load(f)
        if not _metadata_matches_completion_signature(meta, expected_signature):
            return False
        if "labels" in meta.get("dataset", {}) and not p["labels"].exists():
            return False
        return True
    except Exception:
        return False


def all_model_datasets_complete(datasets: Mapping[str, Any], base_output: str | Path, model_name: str, *, hyperparameters: HyperParameters) -> bool:
    batch = hyperparameters.batch_for(model_name, get_model_spec(model_name))
    for dataset_name, dataset in datasets.items():
        col = detect_text_column(dataset, show_verbose=False); texts = dataset_texts(dataset, col); fp = dataset_fingerprint(dataset, texts)
        expected = {"experiment_id": hyperparameters.experiment_id, "hyperparameter_hash": hyperparameters.config_hash, "dataset_fingerprint": fp, "text_column": col, "sample_count": len(texts), "pooling": hyperparameters.pooling, "max_length": hyperparameters.max_length, "batch_size": batch}
        if not dataset_output_is_complete(build_dataset_directory(base_output, model_name, dataset_name), expected_signature=expected): return False
    return True

# =============================================================================
# POOLING / EXTRACTION
# =============================================================================

def pool_hidden_states(hidden_states: Sequence[torch.Tensor], attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if not hidden_states: raise ValueError("Model returned no hidden states")
    if pooling == "first_token": return torch.stack([x[:, 0, :] for x in hidden_states], dim=1)
    if pooling == "last_token":
        last = attention_mask.sum(dim=1) - 1
        if torch.any(last < 0): raise ValueError("Sample has no valid tokens")
        rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        return torch.stack([x[rows, last, :] for x in hidden_states], dim=1)
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states[0].dtype); denom = mask.sum(dim=1).clamp_min(1)
        return torch.stack([(x * mask).sum(dim=1) / denom for x in hidden_states], dim=1)
    raise ValueError("pooling must be first_token, mean, or last_token")


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "not enough memory" in text or (("mps" in text or "metal" in text) and "memory" in text)


def _find_first_incomplete(completed: np.ndarray) -> int:
    n = len(completed); pos = 0
    while pos < n:
        stop = min(n, pos + 1_000_000); local = np.flatnonzero(~completed[pos:stop])
        if local.size: return pos + int(local[0])
        pos = stop
    return n


# ---- v2 helpers for sample IDs and text hashes ----

def get_sample_ids(dataset: Any, text_column: str, texts: Sequence[str]) -> np.ndarray:
    """Return an object array of unique sample identifiers (strings)."""
    columns = set(get_dataset_columns(dataset))
    # Prefer a dedicated ID column
    for cand in ("id", "index", "sample_id", "row_id", "idx"):
        if cand in columns:
            values = dataset[cand]
            # Convert to strings to preserve non‑numeric IDs
            ids = np.array([str(v) for v in values], dtype=object)
            return ids
    # Otherwise, create a hash of the text content
    ids = np.array([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts], dtype=object)
    return ids


def get_text_hashes(texts: Sequence[str]) -> np.ndarray:
    """Return a uint8 array of shape (n, 32) with SHA256 digests of each text."""
    n = len(texts)
    arr = np.empty((n, 32), dtype=np.uint8)
    for i, t in enumerate(texts):
        arr[i] = np.frombuffer(hashlib.sha256(t.encode("utf-8")).digest(), dtype=np.uint8)
    return arr


def compute_global_checksum(states: np.memmap, batch_size: int) -> str:
    """Compute SHA256 of the entire memmap by reading it in chunks."""
    hasher = hashlib.sha256()
    n_samples, n_layers, n_hidden = states.shape
    total_elements = n_samples * n_layers * n_hidden
    # Use the same batch size as extraction to chunk efficiently
    chunk = batch_size * n_layers * n_hidden
    for start in range(0, total_elements, chunk):
        end = min(start + chunk, total_elements)
        hasher.update(states.flat[start:end].tobytes())
    return hasher.hexdigest()


def is_v2_sample_ids(path: Path) -> bool:
    """Check if sample_ids.npy is v2 (object dtype with strings)."""
    if not path.exists():
        return False
    try:
        arr = np.load(path, allow_pickle=True)  # no mmap
        return arr.dtype == object
    except Exception:
        return False



# =============================================================================
# RUNTIME REPORTER (for progress and telemetry)
# =============================================================================

class RuntimeReporter:
    """INFO=speed/state, VERBOSE=forensics, DEBUG=batch records, CRITICAL=failures/completion."""
    def __init__(self, dataset_name: str, total: int, initial_completed: int,
                 show_verbose: bool, show_info: bool, show_critical: bool, show_debug: bool,
                 events_path: Path, runtime_path: Path, context: Mapping[str, Any]):
        self.dataset_name = dataset_name
        self.total = int(total)
        self.initial_completed = int(initial_completed)
        self.completed = int(initial_completed)
        self.show_verbose = show_verbose
        self.show_info = show_info
        self.show_critical = show_critical
        self.show_debug = show_debug
        self.events_path = events_path
        self.runtime_path = runtime_path
        self.context = dict(context)
        self.start = time.perf_counter()
        self.last_report = self.start
        self.last_report_completed = self.completed
        self.last_stage = "initialising"
        self.last_stage_started = self.start
        self.successful_batches = 0
        self.last_batch = {}
        self.recent_batches = deque(maxlen=DEBUG_RECENT_BATCHES)
        self.batch_durations = deque(maxlen=DIAGNOSTIC_WINDOW)
        self.stage_totals = {}
        self.peak_rss_gb = 0.0
        self.minimum_available_ram_gb = math.inf
        self.anomaly_count = 0
        self.slow_batch_count = 0
        self._events_file = None
        self.progress = tqdm(
            total=self.total, initial=self.completed,
            desc=dataset_name, unit="sample",
            dynamic_ncols=True, mininterval=PROGRESS_REFRESH,
            disable=not show_info
        )

    @property
    def elapsed(self):
        return max(1e-9, time.perf_counter() - self.start)

    @property
    def new_samples(self):
        return max(0, self.completed - self.initial_completed)

    @property
    def samples_per_sec(self):
        return self.new_samples / self.elapsed

    @property
    def rolling_mean(self):
        return sum(self.batch_durations) / len(self.batch_durations) if self.batch_durations else 0.0

    @property
    def rolling_median(self):
        return float(median(self.batch_durations)) if self.batch_durations else 0.0

    @property
    def rolling_max(self):
        return max(self.batch_durations) if self.batch_durations else 0.0

    @property
    def rolling_p95(self):
        if not self.batch_durations:
            return 0.0
        v = sorted(self.batch_durations)
        return float(v[min(len(v)-1, max(0, math.ceil(.95*len(v))-1))])

    def __enter__(self):
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._events_file = self.events_path.open("a", encoding="utf-8")
        self.write_runtime_snapshot("running")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_current_stage()
        self.write_runtime_snapshot("error" if exc_type else "finished")
        self._flush()
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None
        self.progress.close()

    def _flush(self):
        if self._events_file is not None:
            self._events_file.flush()

    def close_current_stage(self):
        now = time.perf_counter()
        self.stage_totals[self.last_stage] = self.stage_totals.get(self.last_stage, 0.0) + now - self.last_stage_started
        self.last_stage_started = now

    def set_stage(self, stage: str):
        now = time.perf_counter()
        self.stage_totals[self.last_stage] = self.stage_totals.get(self.last_stage, 0.0) + now - self.last_stage_started
        self.last_stage = stage
        self.last_stage_started = now

    def event(self, kind: str, message: str, *, force: bool = False, debug: bool = False, **fields: Any):
        record = {
            **self.context,
            "time": time.time(),
            "elapsed_seconds": self.elapsed,
            "dataset": self.dataset_name,
            "kind": kind,
            "message": message,
            "completed": self.completed,
            "stage": self.last_stage,
            **fields
        }
        if self._events_file is not None:
            self._events_file.write(json.dumps(record, ensure_ascii=False, default=str, sort_keys=True) + "\n")
            self._flush()
        if debug and not self.show_debug:
            return
        if kind == "batch" and not (self.show_debug or force):
            return
        if self.show_info or force:
            self.progress.write(f"[{kind}] {message}") if self.show_info else print(f"[{kind}] {message}")

    def write_runtime_snapshot(self, status: str = "running"):
        write_runtime_diagnostic(
            self.runtime_path,
            runtime_diagnostic_snapshot(
                stage=self.last_stage,
                model_name=self.context.get("model"),
                dataset_name=self.dataset_name,
                experiment_id=self.context.get("experiment_id"),
                hyperparameter_hash=self.context.get("hyperparameter_hash"),
                extra={
                    "status": status,
                    "completed": self.completed,
                    "total": self.total,
                    "new_samples": self.new_samples,
                    "samples_per_second": self.samples_per_sec,
                    "last_batch": self.last_batch,
                    "recent_batches": list(self.recent_batches),
                    "stage_totals_seconds": self.stage_totals,
                    "rolling_mean_batch_seconds": self.rolling_mean,
                    "rolling_median_batch_seconds": self.rolling_median,
                    "rolling_p95_batch_seconds": self.rolling_p95,
                    "rolling_max_batch_seconds": self.rolling_max,
                    "anomaly_count": self.anomaly_count,
                    "slow_batch_count": self.slow_batch_count,
                }
            )
        )

    def batch_finished(self, *, start: int, end: int, newly_completed: int,
                       batch_size: int, metrics: Mapping[str, Any]):
        self.completed += int(newly_completed)
        self.successful_batches += 1
        rec = {
            "start": start,
            "end": end,
            "batch_size": batch_size,
            "newly_completed": newly_completed,
            **dict(metrics)
        }
        self.last_batch = rec
        self.recent_batches.append(rec)
        self.batch_durations.append(float(metrics["total_seconds"]))
        if self.show_info:
            self.progress.update(int(newly_completed))
            self.progress.set_postfix(
                sp=f"{self.samples_per_sec:.1f}",
                fw=f"{metrics['forward_seconds']:.2f}s",
                last=f"{metrics['total_seconds']:.2f}s",
                ram=f"{metrics.get('available_gb',0):.2f}G"
            )
        self.event("batch", f"{start}:{end} processed", debug=True, batch=rec)
        self.maybe_runtime_report()

    def maybe_runtime_report(self, force: bool = False):
        now = time.perf_counter()
        sample_delta = self.completed - self.last_report_completed
        if not force and now - self.last_report < RUNTIME_REPORT_INTERVAL_SECONDS and sample_delta < RUNTIME_REPORT_INTERVAL_SAMPLES:
            return
        memory, storage, rss = get_memory_info(), get_external_storage_usage(), get_process_rss_gb()
        if rss is not None:
            self.peak_rss_gb = max(self.peak_rss_gb, rss)
        self.minimum_available_ram_gb = min(self.minimum_available_ram_gb, memory["available_gb"])
        last = self.last_batch
        rate = self.samples_per_sec
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else math.inf
        anomaly = False
        ratio = 0.0
        if len(self.batch_durations) >= 5 and self.rolling_median > 0:
            ratio = float(last.get("total_seconds", 0.0)) / self.rolling_median
            anomaly = ratio >= SLOW_BATCH_MULTIPLIER
            if anomaly:
                self.anomaly_count += 1
                self.slow_batch_count += 1
        panel = (
            "\n" + _separator("─") +
            f"\nRUNTIME / SPEED TELEMETRY :: {self.dataset_name}\n" +
            _separator("─") + "\n" +
            f"  Progress             : {self.completed:,}/{self.total:,} ({100*self.completed/max(1,self.total):.2f}%)\n"
            f"  Newly computed       : {self.new_samples:,}\n"
            f"  Wall time            : {_format_duration(self.elapsed)}\n"
            f"  Throughput           : {rate:.2f} samples/s\n"
            f"  ETA                  : {_format_duration(eta) if math.isfinite(eta) else 'unknown'}\n"
            f"  Current stage        : {self.last_stage}\n\n"
            "  LAST BATCH\n"
            f"    Range              : {last.get('start','?')}:{last.get('end','?')}\n"
            f"    New samples        : {last.get('newly_completed','?')}\n"
            f"    Total              : {_format_duration(last.get('total_seconds',0.0))}\n"
            f"    Forward            : {_format_duration(last.get('forward_seconds',0.0))}\n"
            f"    Tokenization       : {_format_duration(last.get('tokenize_seconds',0.0))}\n"
            f"    Input transfer     : {_format_duration(last.get('transfer_seconds',0.0))}\n"
            f"    Pooling            : {_format_duration(last.get('pooling_seconds',0.0))}\n"
            f"    Conversion         : {_format_duration(last.get('convert_seconds',0.0))}\n"
            f"    Memmap write       : {_format_duration(last.get('write_seconds',0.0))}\n"
            f"    Flush              : {_format_duration(last.get('flush_seconds',0.0))}\n"
            f"    Sequence length    : {last.get('seq_len','?')}\n"
            f"    Tokens             : {last.get('actual_tokens','?')}\n"
            f"    Token throughput   : {last.get('tokens_per_second',0.0):.2f} tokens/s\n\n"
            "  ROLLING PERFORMANCE\n"
            f"    Mean batch         : {self.rolling_mean:.3f}s\n"
            f"    Median batch       : {self.rolling_median:.3f}s\n"
            f"    P95 batch          : {self.rolling_p95:.3f}s\n"
            f"    Max batch          : {self.rolling_max:.3f}s\n"
            f"    Slow batches       : {self.slow_batch_count:,}\n"
            f"    Current anomaly    : {'YES (x%.2f)' % ratio if anomaly else 'NO'}\n\n"
            "  MEMORY / STORAGE\n"
            f"    RAM available      : {memory['available_gb']:.2f} GiB\n"
            f"    Process RSS        : {rss if rss is not None else 'unavailable'}\n"
            f"    Peak RSS           : {self.peak_rss_gb:.2f} GiB\n"
            f"    Minimum RAM        : {self.minimum_available_ram_gb:.2f} GiB\n"
            f"    External free      : {storage['available_gb']:.2f} GiB\n"
            f"    Successful batches : {self.successful_batches:,}\n"
            f"    Debug batch output : {'ON' if self.show_debug else 'OFF'}\n" +
            _separator("─")
        )
        if self.show_info:
            self.progress.write(panel)
        elif self.show_verbose:
            print(panel)
        self.event(
            "runtime", "periodic runtime telemetry",
            throughput_samples_per_sec=rate,
            eta_seconds=None if not math.isfinite(eta) else eta,
            available_ram_gb=memory["available_gb"],
            process_rss_gb=rss,
            external_free_gb=storage["available_gb"],
            anomaly=anomaly,
            last_batch=dict(last)
        )
        self.write_runtime_snapshot()
        self.last_report = now
        self.last_report_completed = self.completed


def ensure_auxiliary_files(
    dataset: Any,
    paths: dict[str, Path],
    column: str,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    storage_dtype: np.dtype,
    show_info: bool = True,
) -> Dict[str, Any]:
    actions = []
    if not paths["states"].exists():
        return {"actions": actions, "skipped": True, "reason": "hidden_states.npy missing"}

    texts = dataset_texts(dataset, column)

    # --- sample_ids (v2) ---
    if not paths["sample_ids"].exists() or not is_v2_sample_ids(paths["sample_ids"]):
        ids = get_sample_ids(dataset, column, texts)
        np.save(paths["sample_ids"], ids)
        actions.append("created/upgraded sample_ids.npy (v2)")

    # --- text_hashes (optional) ---
    if not paths["text_hashes"].exists():
        hashes = get_text_hashes(texts)
        np.save(paths["text_hashes"], hashes)
        actions.append("created text_hashes.npy")

    # --- labels (with compatibility check) ---
    label_column = None
    dataset_cols = get_dataset_columns(dataset)
    for cand in ("labels", "label", "target", "emotion"):
        if cand in dataset_cols:
            label_column = cand
            break

    if label_column:
        labels_raw = np.asarray(dataset[label_column])
        if labels_raw.dtype.kind in ("U", "S", "O"):
            unique_labels, encoded = np.unique(labels_raw, return_inverse=True)
            np.save(paths["label_codes"], unique_labels)
            labels = encoded.astype(np.int32)
        else:
            labels = labels_raw.astype(np.int64)
            if not paths["label_codes"].exists():
                unique_labels = np.unique(labels)
                np.save(paths["label_codes"], unique_labels)

        if paths["labels"].exists():
            existing = np.load(paths["labels"], allow_pickle=False)
            if existing.shape != (n_samples,) or existing.dtype != labels.dtype:
                os.remove(paths["labels"])
                actions.append("removed incompatible labels.npy")
                labels_mmap = np.lib.format.open_memmap(
                    paths["labels"], mode="w+", dtype=labels.dtype, shape=(n_samples,)
                )
                labels_mmap[:] = labels
                flush_array(labels_mmap)
                actions.append("recreated labels.npy with consistent dtype")
            else:
                actions.append("labels.npy already compatible")
        else:
            labels_mmap = np.lib.format.open_memmap(
                paths["labels"], mode="w+", dtype=labels.dtype, shape=(n_samples,)
            )
            labels_mmap[:] = labels
            flush_array(labels_mmap)
            actions.append("created labels.npy")
    else:
        actions.append("no label column found")

    # --- integrity_hashes (batch hashes) ---
    if not paths["integrity"].exists():
        states = np.load(paths["states"], mmap_mode='r')
        with open(paths["integrity"], 'w') as f:
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                data = np.asarray(states[start:end])
                h = hashlib.sha256(data.tobytes()).hexdigest()
                f.write(json.dumps({"batch_start": start, "batch_end": end, "hash": h, "timestamp": time.time()}) + "\n")
        actions.append("created integrity_hashes.jsonl")

    # --- global checksum is computed at the end of extraction, not here ---
    # if not paths["checksum"].exists():
    #     states = np.load(paths["states"], mmap_mode='r')
    #     checksum = compute_global_checksum(states, batch_size)
    #     with open(paths["checksum"], 'w') as f:
    #         f.write(checksum)
    #     actions.append("created checksum.sha256")

    return {"actions": actions, "skipped": False}


def repair_dataset_auxiliaries(
    dataset: Any,
    output_dir: str | Path,
    batch_size: int,
    pooling: str,
    max_length: int,
    storage_dtype: np.dtype,
    experiment_id: str,
    hyperparameter_hash: str,
    show_info: bool = True,
) -> Dict[str, Any]:
    paths = build_dataset_storage_paths(output_dir)
    column = detect_text_column(dataset, show_verbose=False)
    texts = dataset_texts(dataset, column)
    n_samples = len(texts)
    fp = dataset_fingerprint(dataset, texts)

    result = ensure_auxiliary_files(
        dataset=dataset,
        paths=paths,
        column=column,
        n_samples=n_samples,
        batch_size=batch_size,
        device=get_best_device(),
        storage_dtype=storage_dtype,
        show_info=show_info,
    )
    actions = result.get("actions", [])
    skipped = result.get("skipped", False)

    metadata_updated = False
    if not skipped and paths["completed"].exists():
        completed = np.load(paths["completed"], mmap_mode='r')
        if int(completed.sum()) == n_samples:
            meta_path = paths["metadata"]
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

                if meta.get("format_version", "1.0") < "2.0":
                    meta["format_version"] = "2.0"
                    actions.append("upgraded format_version to 2.0")

                prov = meta.get("dataset", {}).get("provenance", {})
                current_prov = dataset_signature(dataset, texts, column)
                if paths["sample_ids"].exists():
                    sample_id_arr = np.load(paths["sample_ids"], allow_pickle=True)
                    if len(sample_id_arr) == n_samples:
                        meta["dataset"]["sample_id_column"] = "inferred"
                        meta["dataset"]["sample_id_type"] = "string"
                if paths["text_hashes"].exists():
                    meta["dataset"]["text_hash_type"] = "sha256"

                meta["experiment_id"] = experiment_id
                meta["hyperparameter_hash"] = hyperparameter_hash
                meta["dataset"]["fingerprint"] = fp
                meta["dataset"]["provenance"] = current_prov
                # === USE get_dataset_columns ===
                meta["dataset"]["columns"] = get_dataset_columns(dataset)
                meta["extraction"] = {
                    "pooling": pooling,
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "device": "cpu",
                    "storage_dtype": str(storage_dtype),
                    "runtime_batch_change": False,
                }
                save_json(meta_path, meta)
                actions.append("updated extraction.json with v2 fields")
                metadata_updated = True
            else:
                actions.append("metadata file missing; not updated")

    return {"actions": actions, "skipped": skipped, "metadata_updated": metadata_updated}


# ---- main extraction function (modified to update global checksum) ----
def extract_dataset(model: torch.nn.Module, tokenizer: Any, dataset: Any, text_column: str | None, output_dir: str | Path, batch_size: int, pooling: str = DEFAULT_POOLING, max_length: int | None = DEFAULT_MAX_LENGTH, storage_dtype: np.dtype = np.float32, device: torch.device | None = None, model_name: str | None = None, dataset_name: str | None = None, model_snapshot: str | Path | None = None, auto_batch_size: bool = False, max_batch_size: int | None = None, flush_every_batches: int = DEFAULT_FLUSH_EVERY_BATCHES, flush_every_seconds: float = DEFAULT_FLUSH_SECONDS, show_verbose: bool = True, show_info: bool = True, show_critical: bool = True, show_debug: bool = False, experiment_id: str = DEFAULT_EXPERIMENT_ID, hyperparameter_hash: str | None = None) -> dict[str, Any]:
    if auto_batch_size: raise ValueError("auto_batch_size=True is disabled in deterministic extraction")
    if max_batch_size is not None and int(max_batch_size) != int(batch_size): raise ValueError("max_batch_size must equal frozen batch_size")
    if batch_size < 1: raise ValueError("batch_size must be >= 1")
    max_length = int(max_length or DEFAULT_MAX_LENGTH)
    if max_length < 1: raise ValueError("max_length must be >= 1")
    if pooling not in {"first_token", "mean", "last_token"}: raise ValueError("Invalid pooling mode")
    device = device or get_best_device(); dataset_name = dataset_name or "dataset"; model_name = model_name or "unknown"
    column = detect_text_column(dataset, text_column, show_verbose=show_verbose)
    texts = dataset_texts(dataset, column); n_samples = len(texts)
    if not n_samples: raise ValueError(f"{dataset_name} contains zero samples")
    avg_chars = sum(len(t) for t in texts[:100] if t.strip()) / max(1, sum(bool(t.strip()) for t in texts[:100]))
    if avg_chars < 5: raise RuntimeError(f"Selected text column {column!r} is suspicious: average length={avg_chars:.2f}")
    config = model.config; num_layers = get_model_num_layers(config) + 1; hidden_size = get_model_hidden_size(config); expected_shape = (n_samples, num_layers, hidden_size)
    paths = build_dataset_storage_paths(output_dir); paths["dataset_dir"].resolve().relative_to(EXTERNAL_ROOT); fp = dataset_fingerprint(dataset, texts); provenance = dataset_signature(dataset, texts, column)
    _print_info("\n" + _separator("─") + f"\nDATASET {dataset_name}\n" + _separator("─") + "\n" + f"  Text column         : {column}\n  Samples             : {n_samples:,}\n  Dataset fingerprint : {fp}\n  Hidden layers       : {num_layers}\n  Hidden size         : {hidden_size}\n  Output size est.    : {(n_samples*num_layers*hidden_size*np.dtype(storage_dtype).itemsize)/(1024**3):.2f} GiB\n  Pooling             : {pooling}\n  Max length          : {max_length}\n  FROZEN batch        : {batch_size}\n  Device              : {device}\n  Runtime diagnostics : {paths['runtime']}", show_info)
    if paths["states"].exists():
        states = np.load(paths["states"], mmap_mode="r+")
        if not isinstance(states, np.memmap) or states.shape != expected_shape or states.dtype != storage_dtype: raise ValueError(f"Existing hidden_states.npy incompatible: {getattr(states,'shape',None)}/{getattr(states,'dtype',None)} vs {expected_shape}/{storage_dtype}")
    else:
        states = np.lib.format.open_memmap(paths["states"], mode="w+", dtype=storage_dtype, shape=expected_shape)
    if paths["completed"].exists():
        completed = np.load(paths["completed"], mmap_mode="r+")
        if not isinstance(completed, np.memmap) or completed.shape != (n_samples,) or completed.dtype != np.bool_: raise ValueError("Existing completed.npy incompatible")
    else:
        completed = np.lib.format.open_memmap(paths["completed"], mode="w+", dtype=np.bool_, shape=(n_samples,)); completed[:] = False; flush_array(completed)
    # Labels and sample IDs
    label_column = None
    dataset_cols = get_dataset_columns(dataset)   # use helper
    for cand in ("labels", "label", "target", "emotion"):
        if cand in dataset_cols:
            label_column = cand
            break
    # Convert labels to a consistent integer encoding.
    raw_label_fingerprint = stable_hash({"values": [str(v) for v in dataset[label_column]]}, 24)
    labels_raw = np.asarray(dataset[label_column], dtype=object)
    # Determine if multi-label (any element is a list/tuple)
    is_multi = any(isinstance(x, (list, tuple, set, np.ndarray)) for x in labels_raw[:100])

    if is_multi:
        # For multi-label, flatten all labels to get class set
        all_labels = []
        for x in labels_raw:
            if isinstance(x, (list, tuple, set, np.ndarray)):
                all_labels.extend(x)
            else:
                all_labels.append(x)
        unique_labels = sorted(set(all_labels))
        label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        # Save as object array of lists of ints (cannot be memory-mapped, use np.save)
        labels = np.empty(len(labels_raw), dtype=object)
        for i, x in enumerate(labels_raw):
            if isinstance(x, (list, tuple, set, np.ndarray)):
                labels[i] = [label_to_id[y] for y in x]
            else:
                labels[i] = [label_to_id[x]]
        np.save(paths["label_codes"], np.array(unique_labels, dtype=object))
        np.save(paths["labels"], labels)   # object array, not memmap
        labels_info = {"label_column": label_column, "labels_dtype": "object", "label_codes_path": str(paths["label_codes"])}
    else:
        # Single-label: existing logic
        unique_labels, encoded = np.unique(labels_raw, return_inverse=True)
        labels = encoded.astype(np.int64)
        np.save(paths["label_codes"], unique_labels)
        # Save as memmap
        labels_mmap = np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=labels.dtype, shape=(len(labels),))
        labels_mmap[:] = labels
        flush_array(labels_mmap)
        labels_info = {"label_column": label_column, "labels_dtype": str(labels.dtype), "label_codes_path": str(paths["label_codes"])}
        if paths["labels"].exists():
            labels_mmap = np.load(paths["labels"], mmap_mode="r+")
            if labels_mmap.shape != (n_samples,) or labels_mmap.dtype != labels.dtype:
                # Fallback: recreate (should not happen if repair ran)
                _print_critical(f"  Warning: existing labels.npy incompatible – recreating.", show_critical)
                # Close and delete
                del labels_mmap
                os.remove(paths["labels"])
                labels_mmap = np.lib.format.open_memmap(
                    paths["labels"], mode="w+", dtype=labels.dtype, shape=(n_samples,)
                )
                labels_mmap[:] = labels
                flush_array(labels_mmap)
            else:
                labels_mmap = np.lib.format.open_memmap(
                    paths["labels"], mode="w+", dtype=labels.dtype, shape=(n_samples,)
                )
                labels_mmap[:] = labels
                flush_array(labels_mmap)
            labels_info = {"label_column": label_column, "labels_dtype": str(labels.dtype), "label_codes_path": str(paths["label_codes"]) if paths["label_codes"].exists() else None}
        else:
            labels_mmap = None
            labels_info = {"label_column": None, "labels_dtype": None, "label_codes_path": None}

    # Sample IDs (v2) – ensure they exist
    if not paths["sample_ids"].exists() or not is_v2_sample_ids(paths["sample_ids"]):
        ids = get_sample_ids(dataset, column, texts)
        np.save(paths["sample_ids"], ids)
    # Load without mmap (object arrays cannot be memory-mapped)
    sample_ids = np.load(paths["sample_ids"], allow_pickle=True)
    if sample_ids.dtype == object:
        sample_ids = np.array([str(x) for x in sample_ids], dtype=object)
    if len(sample_ids) != n_samples:
        raise ValueError(f"sample_ids length {len(sample_ids)} != n_samples {n_samples}")
    # Text hashes (optional)
    if not paths["text_hashes"].exists():
        hashes = get_text_hashes(texts)
        np.save(paths["text_hashes"], hashes)
    # We don't load text_hashes unless needed for validation.

    completed_count = int(completed.sum())
    signature = {"experiment_id": experiment_id, "hyperparameter_hash": hyperparameter_hash, "dataset_fingerprint": fp, "text_column": column, "sample_count": n_samples, "pooling": pooling, "max_length": max_length, "batch_size": batch_size}
    if completed_count == n_samples:
        # Attempt metadata repair before raising
        if paths["metadata"].exists():
            try:
                with paths["metadata"].open("r", encoding="utf-8") as f: meta = json.load(f)
                core_match = (
                    meta.get("dataset", {}).get("fingerprint") == fp and
                    meta.get("dataset", {}).get("samples") == n_samples and
                    meta.get("extraction", {}).get("pooling") == pooling and
                    meta.get("extraction", {}).get("max_length") == max_length and
                    meta.get("extraction", {}).get("batch_size") == batch_size
                )
                if core_match:
                    meta["format_version"] = "2.0"
                    meta["status"] = "complete"
                    meta["experiment_id"] = experiment_id
                    meta["hyperparameter_hash"] = hyperparameter_hash
                    meta["dataset"]["text_column"] = column
                    meta["dataset"]["samples"] = n_samples
                    meta["dataset"]["fingerprint"] = fp
                    meta["dataset"]["provenance"] = provenance
                    meta["dataset"]["columns"] = list(getattr(dataset, "columns", []))
                    meta["dataset"]["labels"] = labels_info
                    meta["dataset"]["sample_id_column"] = "inferred"  # or actual
                    meta["dataset"]["sample_id_type"] = "string"
                    meta["extraction"] = {
                        "pooling": pooling,
                        "max_length": max_length,
                        "batch_size": batch_size,
                        "device": str(device),
                        "storage_dtype": str(storage_dtype),
                        "runtime_batch_change": False,
                    }
                    # Compute global checksum if missing
                    if not paths["checksum"].exists():
                        chk = compute_global_checksum(states, batch_size)
                        with open(paths["checksum"], 'w') as cf:
                            cf.write(chk)
                        meta["integrity"]["global_checksum"] = chk
                    save_json(paths["metadata"], meta)
                    _print_critical(f"✓ DATASET ALREADY COMPLETE (metadata updated): {dataset_name}", show_critical)
                    return {
                        "status": "already_complete",
                        "experiment_id": experiment_id,
                        "hyperparameter_hash": hyperparameter_hash,
                        "model": {"name": model_name, "snapshot": str(model_snapshot) if model_snapshot else None},
                        "dataset": {"name": dataset_name, "text_column": column, "samples": n_samples,
                                    "fingerprint": fp, "provenance": provenance, "labels": labels_info},
                        "extraction": {"pooling": pooling, "max_length": max_length,
                                    "batch_size": batch_size, "device": str(device),
                                    "storage_dtype": str(storage_dtype)},
                        "performance": {"completed_samples": completed_count, "new_completed_samples": 0},
                        "storage": {k: str(v) for k, v in paths.items()}
                    }
            except Exception:
                pass
        raise RuntimeError("COMPLETION MAP IS FULL BUT DATASET PROVENANCE DOES NOT MATCH THE CURRENT EXPERIMENT")
    _print_info(f"  Existing progress   : {completed_count:,}/{n_samples:,} ({100*completed_count/max(1,n_samples):.2f}%)", show_info)
    position = _find_first_incomplete(completed)
    accepted, accepts_kwargs = get_forward_input_keys(model)
    context = {"experiment_id": experiment_id, "hyperparameter_hash": hyperparameter_hash, "model": model_name, "dataset": dataset_name, "dataset_fingerprint": fp, "batch_size": batch_size, "pooling": pooling, "max_length": max_length, "device": str(device), "storage_dtype": str(storage_dtype), "model_snapshot": str(model_snapshot) if model_snapshot else None, "dataset_provenance": provenance, "labels": labels_info}
    totals = {k: 0.0 for k in ("tokenize", "transfer", "forward", "pooling", "convert", "write", "flush")}; total_tokens = 0; successful_batches = 0; started = time.perf_counter(); last_flush = started

    # Global checksum calculator (incremental)
    checksum_hasher = hashlib.sha256()
    def update_checksum(batch_data: np.ndarray):
        checksum_hasher.update(batch_data.tobytes())

    reporter = RuntimeReporter(dataset_name, n_samples, completed_count, show_verbose, show_info, show_critical, show_debug, paths["events"], paths["runtime"], context)
    try:
        with reporter, torch.inference_mode():
            reporter.event("start", f"starting at absolute sample {position}", force=True, start_position=position, existing_completed=completed_count)
            while position < n_samples:
                while position < n_samples and bool(completed[position]): position += 1
                if position >= n_samples: break
                batch_start = position; end = min(n_samples, position + batch_size); batch_started = time.perf_counter(); prior = np.asarray(completed[batch_start:end], dtype=np.bool_); newly = int((~prior).sum()); rss_before = get_process_rss_gb()
                try:
                    reporter.set_stage("tokenization"); t = time.perf_counter(); tok = tokenizer(texts[batch_start:end], padding=True, truncation=True, max_length=max_length, return_tensors="pt", return_attention_mask=True); tokenize_s = time.perf_counter()-t; seq_len = int(tok["input_ids"].shape[1]); actual_tokens = int(tok["attention_mask"].sum().item())
                    reporter.set_stage("input_transfer"); t = time.perf_counter(); inputs = prepare_model_inputs(tok, device, accepted, accepts_kwargs); mask = tok["attention_mask"].to(device, non_blocking=(device.type != "cpu")); inputs["attention_mask"] = mask; inputs["use_cache"] = False if ("use_cache" in accepted or accepts_kwargs) else inputs.get("use_cache"); inputs = {k:v for k,v in inputs.items() if v is not None}; transfer_s = time.perf_counter()-t
                    reporter.set_stage("forward"); synchronize_device(device); t = time.perf_counter(); outputs = model(**inputs); synchronize_device(device); forward_s = time.perf_counter()-t; hidden = getattr(outputs, "hidden_states", None)
                    if hidden is None: raise RuntimeError("Model returned no hidden_states")
                    if len(hidden) != num_layers: raise RuntimeError(f"Unexpected hidden-state count: returned={len(hidden)} expected={num_layers}")
                    for i, x in enumerate(hidden):
                        if x.ndim != 3: raise RuntimeError(f"Hidden state {i} is not rank-3: {x.shape}")
                        if not bool(torch.isfinite(x).all()): raise RuntimeError(f"Hidden state {i} contains NaN/Inf")
                    reporter.set_stage("pooling"); t = time.perf_counter(); pooled = pool_hidden_states(hidden, mask, pooling); pooling_s = time.perf_counter()-t
                    reporter.set_stage("cpu_conversion"); t = time.perf_counter(); pooled_np = pooled.detach().to(torch.float32).cpu().numpy(); convert_s = time.perf_counter()-t
                    expected_batch = (end-batch_start, num_layers, hidden_size)
                    if pooled_np.shape != expected_batch: raise RuntimeError(f"Unexpected pooled shape: actual={pooled_np.shape} expected={expected_batch}")
                    if not np.isfinite(pooled_np).all(): raise RuntimeError("Pooled hidden states contain NaN/Inf")

                    reporter.set_stage("memmap_write")
                    t = time.perf_counter()
                    pooled_final = pooled_np.astype(storage_dtype, copy=False)
                    states[batch_start:end] = pooled_final
                    completed[batch_start:end] = True
                    write_s = time.perf_counter() - t

                    # Update global checksum incrementally
                    update_checksum(pooled_final)

                    # Write batch integrity hash
                    batch_hash = hashlib.sha256(pooled_final.tobytes()).hexdigest()
                    with paths["integrity"].open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"batch_start": batch_start, "batch_end": end, "hash": batch_hash, "timestamp": time.time()}) + "\n")

                    successful_batches += 1; total_tokens += actual_tokens; totals["tokenize"] += tokenize_s; totals["transfer"] += transfer_s; totals["forward"] += forward_s; totals["pooling"] += pooling_s; totals["convert"] += convert_s; totals["write"] += write_s; position = end

                    flush_s = 0.0; now = time.perf_counter()
                    if successful_batches % flush_every_batches == 0 or now-last_flush >= flush_every_seconds or end >= n_samples:
                        reporter.set_stage("durable_flush"); ft = time.perf_counter()
                        flush_array(states)
                        flush_array(completed)
                        if labels_mmap is not None:
                            flush_array(labels_mmap)
                        # No directory fsync – only file fsync inside flush_array
                        flush_s = time.perf_counter()-ft; totals["flush"] += flush_s; last_flush = time.perf_counter()
                        save_json(paths["runtime"], {**context, "status": "running", "position": end, "completed": int(completed.sum()), "updated_at": time.time()})
                    mem = get_memory_info(); disk = get_external_storage_usage(); rss = get_process_rss_gb(); batch_total = time.perf_counter()-batch_started
                    reporter.set_stage("measurement"); reporter.batch_finished(start=batch_start, end=end, newly_completed=newly, batch_size=batch_size, metrics={"seq_len": seq_len, "actual_tokens": actual_tokens, "tokenize_seconds": tokenize_s, "transfer_seconds": transfer_s, "forward_seconds": forward_s, "pooling_seconds": pooling_s, "convert_seconds": convert_s, "write_seconds": write_s, "flush_seconds": flush_s, "total_seconds": batch_total, "samples_per_sec": (end-batch_start)/max(batch_total,1e-9), "new_samples_per_sec": newly/max(batch_total,1e-9), "tokens_per_second": actual_tokens/max(batch_total,1e-9), "available_gb": mem["available_gb"], "used_gb": mem["used_gb"], "process_rss_gb": rss, "external_free_gb": disk["available_gb"], "batch_memory_delta_gb": max(0.0, rss-rss_before) if rss is not None and rss_before is not None else None, "runtime_batch_change": False, "pre_batch_completed": int(prior.sum()), "newly_completed": newly})
                except Exception as exc:
                    diag = runtime_diagnostic_snapshot(stage=reporter.last_stage, model_name=model_name, dataset_name=dataset_name, experiment_id=experiment_id, hyperparameter_hash=hyperparameter_hash, extra={"status":"error", "error_type":type(exc).__name__, "error":str(exc), "traceback":_truncate_traceback(traceback.format_exc()), "batch_start":batch_start, "batch_end":end, "batch_size":batch_size, "last_successful_batch":reporter.last_batch, "recent_batches":list(reporter.recent_batches)})
                    write_runtime_diagnostic(paths["runtime"], diag); reporter.event("error", f"{type(exc).__name__}: {exc}", force=True, traceback=_truncate_traceback(traceback.format_exc()), batch_start=batch_start, batch_end=end, batch_size=batch_size); raise
    except RuntimeError as exc:
        if is_oom_error(exc): raise RuntimeError(f"FROZEN HYPERPARAMETER OOM\nModel: {model_name}\nDataset: {dataset_name}\nBatch: {batch_size}\nBatch size was NOT changed") from exc
        raise
    flush_array(states); flush_array(completed)
    if labels_mmap is not None: flush_array(labels_mmap)

    # Write global checksum (already updated incrementally, but final digest is stored here)
    global_checksum = checksum_hasher.hexdigest()
    with open(paths["checksum"], 'w') as cf:
        cf.write(global_checksum)

    elapsed = time.perf_counter()-started; completed_total = int(completed.sum()); new_completed = max(0, completed_total-completed_count); sps = new_completed/max(elapsed,1e-9); tps = total_tokens/max(elapsed,1e-9); work = sum(totals.values())
    metadata = {
        "format_version": "2.0",
        "status":"complete" if completed_total == n_samples else "partial",
        "experiment_id":experiment_id,
        "hyperparameter_hash":hyperparameter_hash,
        "model":{"name":model_name,"snapshot":str(model_snapshot) if model_snapshot else None},
        "dataset":{
            "name":dataset_name,
            "text_column":column,
            "samples":n_samples,
            "fingerprint":fp,
            "provenance":provenance,
            "columns": get_dataset_columns(dataset),
            "labels": labels_info,
            "sample_id_column": "inferred",
            "sample_id_type": "string"
        },
        "extraction":{
            "pooling":pooling,
            "max_length":max_length,
            "batch_size":batch_size,
            "device":str(device),
            "storage_dtype":str(storage_dtype),
            "runtime_batch_change":False
        },
        "integrity":{
            "global_checksum": global_checksum,
            "checksum_algorithm": "sha256"
        },
        "performance":{
            "elapsed_seconds":elapsed,
            "new_completed_samples":new_completed,
            "completed_samples":completed_total,
            "samples_per_second":sps,
            "tokens_per_second":tps,
            "successful_batches":successful_batches,
            "timing_seconds":{**totals,"work_time":work,"overhead":max(0.0,elapsed-work)},
            "final_memory":get_memory_info(),
            "final_storage":get_external_storage_usage(),
            "final_process_rss_gb":get_process_rss_gb()
        },
        "diagnostics":{
            "stage_totals_seconds":reporter.stage_totals,
            "last_batch":reporter.last_batch,
            "recent_batches":list(reporter.recent_batches),
            "rolling_mean_batch_seconds":reporter.rolling_mean,
            "rolling_median_batch_seconds":reporter.rolling_median,
            "rolling_p95_batch_seconds":reporter.rolling_p95,
            "rolling_max_batch_seconds":reporter.rolling_max,
            "anomaly_count":reporter.anomaly_count,
            "slow_batch_count":reporter.slow_batch_count,
            "peak_rss_gb":reporter.peak_rss_gb,
            "minimum_available_ram_gb":reporter.minimum_available_ram_gb,
            "verbosity":{"verbose":show_verbose,"info":show_info,"critical":show_critical,"debug":show_debug}
        },
        "storage":{k:str(v) for k,v in paths.items()}
    }
    save_json(paths["metadata"], metadata); save_json(paths["runtime"], metadata); append_jsonl(MEASUREMENT_LEDGER, metadata)
    _print_critical(f"✓ DATASET COMPLETE: {dataset_name}\n  Newly computed     : {new_completed:,}\n  Total completed    : {completed_total:,}/{n_samples:,}\n  Elapsed            : {_format_duration(elapsed)}\n  Throughput         : {sps:.2f} samples/s\n  Token throughput   : {tps:.2f} tokens/s\n  Work time          : {_format_duration(work)}\n  Pipeline overhead  : {_format_duration(max(0.0,elapsed-work))}", show_critical)
    return metadata

# =============================================================================
# EXPERIMENT AUDITING & VALIDATION
# =============================================================================

def validate_dataset_output(
    extraction_dir: str | Path,
    dataset: Any = None,
    fix_sample_ids: bool = True,
    fix_checksum: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    """
    Deep validation of a single dataset directory.
    Returns a dict with status, checksum_match, sample_ids_match, etc.
    Optionally fixes sample_ids or checksum if missing/invalid.
    """
    extraction_dir = Path(extraction_dir)
    meta_path = extraction_dir / "metadata" / "extraction.json"
    if not meta_path.exists():
        return {"status": "missing_metadata", "error": f"metadata/extraction.json not found in {extraction_dir}"}

    with open(meta_path, "r") as f:
        meta = json.load(f)

    paths = build_dataset_storage_paths(extraction_dir)
    result = {
        "extraction_dir": str(extraction_dir),
        "model": meta.get("model", {}).get("name"),
        "dataset": meta.get("dataset", {}).get("name"),
        "status": "incomplete",
        "checksum_match": None,
        "sample_ids_match": None,
        "n_samples": meta.get("dataset", {}).get("samples"),
        "completed_count": None,
        "actions_taken": [],
        "errors": [],
        "warnings": [],
        "metadata": meta,
    }

    # 1. Check required files
    required = [paths["states"], paths["completed"], paths["sample_ids"]]
    missing = [str(p.relative_to(extraction_dir)) for p in required if not p.exists()]
    if missing:
        result["status"] = "missing_files"
        result["errors"].append(f"Missing files: {missing}")
        return result

    # 2. Load arrays
    states = np.load(paths["states"], mmap_mode='r')
    completed = np.load(paths["completed"], mmap_mode='r')
    try:
        sample_ids = np.load(paths["sample_ids"], allow_pickle=True)
        if sample_ids.dtype == object:
            sample_ids = np.array([str(x) for x in sample_ids], dtype=object)
    except Exception as e:
        result["errors"].append(f"sample_ids.npy loading error: {e}")
        return result

    n_samples = len(sample_ids)
    completed_count = int(completed.sum())
    result["completed_count"] = completed_count
    if completed_count == n_samples:
        result["status"] = "complete"
    else:
        result["status"] = "partial"

    # 3. Verify checksum
    checksum_ok = None
    if paths["checksum"].exists():
        with open(paths["checksum"], "r") as f:
            stored_checksum = f.read().strip()
        # Compute checksum (chunked)
        computed = compute_global_checksum(states, 1024)  # batch_size not critical for validation
        checksum_ok = (stored_checksum == computed)
        result["checksum_match"] = checksum_ok
        if not checksum_ok:
            result["warnings"].append("Checksum mismatch! Data may be corrupted.")
            if fix_checksum:
                with open(paths["checksum"], "w") as f:
                    f.write(computed)
                result["actions_taken"].append("Updated checksum.sha256 to match current data.")
    else:
        if fix_checksum:
            computed = compute_global_checksum(states, 1024)
            with open(paths["checksum"], "w") as f:
                f.write(computed)
            result["actions_taken"].append("Created checksum.sha256.")
        else:
            result["warnings"].append("No checksum file found (can be fixed with fix_checksum=True).")

    # 4. Validate sample IDs (if dataset provided)
    if dataset is not None:
        column = meta.get("dataset", {}).get("text_column")
        if column is None:
            result["warnings"].append("No text_column in metadata, cannot verify sample IDs.")
        else:
            texts = dataset_texts(dataset, column)
            if len(texts) != n_samples:
                result["errors"].append(f"Dataset sample count {len(texts)} != extraction samples {n_samples}")
            else:
                # Recompute expected IDs (using same logic as get_sample_ids)
                expected_ids = get_sample_ids(dataset, column, texts)
                # Compare (first few or full if small)
                if len(expected_ids) > 100:
                    # Sample first and last 50
                    sample_indices = list(range(50)) + list(range(len(expected_ids)-50, len(expected_ids)))
                    match = all(sample_ids[i] == expected_ids[i] for i in sample_indices)
                else:
                    match = np.array_equal(sample_ids, expected_ids)
                result["sample_ids_match"] = match
                if not match:
                    result["warnings"].append("Sample IDs do not match dataset (possibly due to dataset ordering change).")
                    if fix_sample_ids:
                        np.save(paths["sample_ids"], expected_ids)
                        result["actions_taken"].append("Overwrote sample_ids.npy with correct IDs.")
                else:
                    result["sample_ids_match"] = True
    else:
        result["warnings"].append("Dataset not provided; sample ID validation skipped.")

    # 5. Check labels if present
    if paths["labels"].exists() and "labels" in meta.get("dataset", {}):
        labels = np.load(paths["labels"], mmap_mode='r')
        if labels.shape[0] != n_samples:
            result["warnings"].append(f"labels.npy shape {labels.shape} != samples {n_samples}")

    return result


def audit_experiment(
    experiment_root: str | Path,
    datasets: Mapping[str, Any] | None = None,
    show_details: bool = True,
    fix_issues: bool = False,
) -> list[dict[str, Any]]:
    """
    Scan an entire experiment directory, validate every model-dataset pair,
    and produce a detailed audit report.

    Args:
        experiment_root: Path to experiment directory (e.g., .../experiments/baseline_v5_001)
        datasets: Optional dict of dataset_name -> dataset object for sample ID validation.
        show_details: If True, print per-dataset details.
        fix_issues: If True, attempt to fix detected issues (checksum, sample IDs).

    Returns:ds
        List of validation results per dataset.
    """
    experiment_root = Path(experiment_root)
    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    # Discover all model-dataset pairs
    extraction_jsons = sorted(experiment_root.rglob("metadata/extraction.json"))

    if not extraction_jsons:
        print(f"⚠️ No extraction metadata found in {experiment_root}")
        return []

    results = []

    # Print header
    print("\n" + "═" * 100)
    print(f"🔍 EXPERIMENT AUDIT: {experiment_root.name}")
    print("═" * 100)
    print(f"  Found {len(extraction_jsons)} dataset extractions.")

    for meta_path in extraction_jsons:
        dataset_dir = meta_path.parent.parent  # .../datasets/{dataset_name}
        dataset_name = dataset_dir.name
        model_dir = dataset_dir.parent.parent  # .../models/{model_name}
        model_name = model_dir.name
        extraction_dir = dataset_dir

        # Get dataset object if provided
        dataset_obj = datasets.get(dataset_name) if datasets else None

        # Validate
        result = validate_dataset_output(
            extraction_dir,
            dataset=dataset_obj,
            fix_sample_ids=fix_issues,
            fix_checksum=fix_issues,
            show_progress=False,
        )
        result["model_name"] = model_name
        result["dataset_name"] = dataset_name
        results.append(result)

        if show_details:
            # Use the existing pretty printer
            print_dataset_format_report(extraction_dir, show_samples=3)

    # Summary table
    print("\n" + "═" * 100)
    print("📊 AUDIT SUMMARY")
    print("═" * 100)

    # Build summary rows
    rows = []
    for r in results:
        status = r.get("status", "unknown")
        chk = "✅" if r.get("checksum_match") is True else ("❌" if r.get("checksum_match") is False else "—")
        sid = "✅" if r.get("sample_ids_match") is True else ("❌" if r.get("sample_ids_match") is False else "—")
        rows.append([
            r["model_name"],
            r["dataset_name"],
            r.get("n_samples", "?"),
            r.get("completed_count", "?"),
            status,
            chk,
            sid,
            ", ".join(r.get("actions_taken", [])) or "—",
        ])

    # Print table
    col_widths = [22, 22, 10, 10, 10, 6, 6, 30]
    headers = ["Model", "Dataset", "Samples", "Done", "Status", "CHK", "ID", "Actions"]
    print("  " + " ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("  " + "-" * (sum(col_widths) + len(col_widths) * 2))
    for row in rows:
        # Trim long actions
        row[-1] = row[-1][:28] + "…" if len(row[-1]) > 30 else row[-1]
        print("  " + " ".join(str(col).ljust(w) for col, w in zip(row, col_widths)))

    # Count issues
    total = len(results)
    complete = sum(1 for r in results if r.get("status") == "complete")
    checksum_ok = sum(1 for r in results if r.get("checksum_match") is True)
    ids_ok = sum(1 for r in results if r.get("sample_ids_match") is True)
    actions_taken = sum(1 for r in results if r.get("actions_taken"))
    print(f"\n  ✅ Complete datasets : {complete}/{total}")
    print(f"  ✅ Checksum OK      : {checksum_ok}/{total}")
    print(f"  ✅ Sample IDs match : {ids_ok}/{total}")
    if fix_issues:
        print(f"  🔧 Fixed issues    : {actions_taken} datasets")

    return results

def print_dataset_format_report(
    extraction_dir: str | Path,
    show_samples: int = 5,
) -> None:
    """
    Print a detailed, visually rich report of the extraction format.
    Shows all files, shapes, dtypes, sample mapping, and metadata.
    """
    from pathlib import Path
    import json
    import numpy as np

    p = Path(extraction_dir)
    meta_path = p / "metadata" / "extraction.json"
    if not meta_path.exists():
        print(f"❌ No extraction metadata found in {extraction_dir}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Build paths
    paths = build_dataset_storage_paths(extraction_dir)

    print("\n" + "═" * 88)
    print(f"📦 EXTRACTION FORMAT REPORT: {meta.get('dataset', {}).get('name', 'unknown')}")
    print("═" * 88)

    # 1. Dataset & Model Overview
    ds = meta.get("dataset", {})
    ex = meta.get("extraction", {})
    print("\n📋 **DATASET**")
    print(f"  Name          : {ds.get('name')}")
    print(f"  Samples       : {ds.get('samples'):,}")
    print(f"  Text column   : {ds.get('text_column')}")
    print(f"  Fingerprint   : {ds.get('fingerprint')}")
    print(f"  Sample ID col : {ds.get('sample_id_column', 'inferred')}")
    print(f"  Labels        : {ds.get('labels', {}).get('label_column', 'None')}")

    print("\n🤖 **MODEL**")
    mod = meta.get("model", {})
    print(f"  Name          : {mod.get('name')}")
    print(f"  Snapshot      : {mod.get('snapshot')}")
    spec = meta.get("model_spec", {})
    if spec:
        print(f"  Family        : {spec.get('family')}")
        print(f"  Architecture  : {spec.get('architecture')}")
        print(f"  Parameters    : {spec.get('parameter_billions', 0):.2f}B")

    print("\n⚙️ **EXTRACTION PARAMETERS**")
    print(f"  Pooling       : {ex.get('pooling')}")
    print(f"  Max length    : {ex.get('max_length')}")
    print(f"  Batch size    : {ex.get('batch_size')}")
    print(f"  Device        : {ex.get('device')}")
    print(f"  Storage dtype : {ex.get('storage_dtype')}")
    print(f"  Format version: {meta.get('format_version', '1.0')}")

    # 2. Files & Shapes
    print("\n📁 **STORAGE FILES**")
    states_path = paths["states"]
    if states_path.exists():
        states = np.load(states_path, mmap_mode='r')
        print(f"  hidden_states.npy     : shape {states.shape}, dtype {states.dtype}, memmap")
        print(f"    → samples × layers × hidden_size")
    else:
        print("  ❌ hidden_states.npy missing")

    completed_path = paths["completed"]
    if completed_path.exists():
        comp = np.load(completed_path, mmap_mode='r')
        print(f"  completed.npy         : shape {comp.shape}, dtype {comp.dtype}, memmap")
        print(f"    → boolean mask of completed samples")
    else:
        print("  ❌ completed.npy missing")

    labels_path = paths["labels"]
    if labels_path.exists():
        lbl = np.load(labels_path, mmap_mode='r')
        print(f"  labels.npy            : shape {lbl.shape}, dtype {lbl.dtype}, memmap")
    else:
        print("  labels.npy            : not present")

    sample_ids_path = paths["sample_ids"]
    if sample_ids_path.exists():
        sid = np.load(sample_ids_path, allow_pickle=True)
        print(f"  sample_ids.npy        : shape {sid.shape}, dtype object, loaded in memory (small)")
        print(f"    → unique per‑sample ID (string)")
    else:
        print("  ❌ sample_ids.npy missing")

    checksum_path = paths["checksum"]
    if checksum_path.exists():
        with open(checksum_path, "r") as f:
            chk = f.read().strip()
        print(f"  checksum.sha256       : {chk[:16]}... (global integrity hash)")

    integrity_path = paths["integrity"]
    if integrity_path.exists():
        with open(integrity_path, "r") as f:
            lines = f.readlines()
        print(f"  integrity_hashes.jsonl: {len(lines)} batch hashes")

    # 3. Sample Mapping (first few)
    print("\n🔗 **SAMPLE MAPPING (first {} samples)**".format(show_samples))
    if sample_ids_path.exists() and states_path.exists():
        sid = np.load(sample_ids_path, allow_pickle=True)
        states = np.load(states_path, mmap_mode='r')
        n = min(show_samples, len(sid))
        print(f"  {'Index':<6} {'Sample ID':<36} {'hidden_states row shape'}")
        print("  " + "-" * 60)
        for i in range(n):
            sample_id = str(sid[i])
            if len(sample_id) > 30:
                sample_id = sample_id[:27] + "..."
            print(f"  {i:<6} {sample_id:<36} {states[i].shape}")
        if len(sid) > show_samples:
            print(f"  ... and {len(sid) - show_samples} more")

    # 4. Integrity & Performance
    print("\n📈 **PERFORMANCE**")
    perf = meta.get("performance", {})
    if perf:
        print(f"  Elapsed          : {_format_duration(perf.get('elapsed_seconds', 0))}")
        print(f"  Samples/sec      : {perf.get('samples_per_second', 0):.2f}")
        print(f"  Completed samples: {perf.get('completed_samples', 0):,}")
        print(f"  Successful batches: {perf.get('successful_batches', 0):,}")
    else:
        print("  No performance data (incomplete extraction)")

    print("\n✅ **READY FOR PROBING**")
    print("  Use `data = load_extraction_for_probing(extraction_dir)` to load all arrays as memmaps.")
    print("  The returned dict contains: hidden_states, labels, sample_ids, completed, metadata.")
    print("═" * 88 + "\n")


def load_extraction_for_probing(extraction_dir: str | Path) -> dict[str, Any]:
    """
    Load all extraction artifacts as lightweight memmaps, ready for probing.
    Returns a dictionary with:
      - hidden_states: memmap (n_samples, n_layers, hidden_size)
      - completed:     bool memmap (n_samples,)
      - labels:        int64 memmap (n_samples,) or None
      - sample_ids:    object array (n_samples,) loaded in memory (small)
      - metadata:      dict from extraction.json
      - paths:         dict of all file paths
    """
    p = Path(extraction_dir)
    meta_path = p / "metadata" / "extraction.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No extraction metadata found in {extraction_dir}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    paths = build_dataset_storage_paths(extraction_dir)

    hidden_states = np.load(paths["states"], mmap_mode='r')
    completed = np.load(paths["completed"], mmap_mode='r')

    # labels (optional)
    labels = None
    if paths["labels"].exists():
        labels = np.load(paths["labels"], mmap_mode='r')

    # sample_ids (small, load in memory)
    sample_ids = np.load(paths["sample_ids"], allow_pickle=True)
    if sample_ids.dtype == object:
        sample_ids = np.array([str(x) for x in sample_ids], dtype=object)

    return {
        "hidden_states": hidden_states,
        "completed": completed,
        "labels": labels,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "paths": paths,
        "n_samples": len(sample_ids),
        "n_layers": hidden_states.shape[1],
        "hidden_size": hidden_states.shape[2],
    }
# =============================================================================
# MODEL METADATA / EXECUTION (unchanged except calling repair)
# =============================================================================

def write_model_metadata(model_dir: str | Path, spec: ModelSpec, device: torch.device, dtype: torch.dtype, model_snapshot: str | Path, model_revision: str, runtime_report: Mapping[str, Any], hyperparameters: HyperParameters) -> None:
    save_json(Path(model_dir) / "model_metadata.json", {"model":asdict(spec),"device":str(device),"inference_dtype":str(dtype),"model_snapshot":str(model_snapshot),"model_revision":model_revision,"experiment_id":hyperparameters.experiment_id,"hyperparameter_hash":hyperparameters.config_hash,"external_root":str(EXTERNAL_ROOT),"huggingface_cache":str(HF_HUB_CACHE),"cache_mode":"external","transformers_version":transformers.__version__,"torch_version":torch.__version__,"runtime":dict(runtime_report)})


def check_transformers_version(spec: ModelSpec) -> tuple[bool, str | None]:
    if not spec.min_transformers: return True, None
    ok = Version(transformers.__version__) >= Version(spec.min_transformers) if Version is not None else tuple(map(int, transformers.__version__.split(".")[:3])) >= tuple(map(int, spec.min_transformers.split(".")[:3]))
    return (True, None) if ok else (False, f"requires transformers>={spec.min_transformers}; installed={transformers.__version__}")

def _prepare_environment(params: HyperParameters) -> tuple[torch.device, dict[str, Any]]:
    device = get_best_device()
    if device.type == "cpu" and params.cpu_dtype == "bfloat16":
        # Check if bfloat16 is supported
        try:
            torch.tensor([1.0], dtype=torch.bfloat16)
        except Exception:
            # Fallback to float32
            params = HyperParameters(**{**params.as_dict(), "cpu_dtype": "float32"})
    configure_cpu_threads(params.cpu_threads)
    configure_cpu_interop_threads(params.cpu_interop_threads)
    environment = {"system_fingerprint":system_fingerprint(),"system":{"cpu":get_cpu_info(),"memory":get_memory_info(),"platform":platform.platform()},"torch_version":torch.__version__,"transformers_version":transformers.__version__,"numpy_version":np.__version__,"device":str(device)}
    
    return device, environment

def _prepare_environment(params: HyperParameters) -> tuple[torch.device, dict[str, Any]]:
    device = get_best_device(); configure_cpu_threads(params.cpu_threads); configure_cpu_interop_threads(params.cpu_interop_threads)
    environment = {"system_fingerprint":system_fingerprint(),"system":{"cpu":get_cpu_info(),"memory":get_memory_info(),"platform":platform.platform()},"torch_version":torch.__version__,"transformers_version":transformers.__version__,"numpy_version":np.__version__,"device":str(device)}
    return device, environment


def _run_resolved_experiment(*, datasets: Mapping[str, Any], model_names: Sequence[str], params: HyperParameters, exp_root: Path, resumed: bool, continue_on_model_error: bool, show_verbose: bool, show_info: bool, show_critical: bool, show_debug: bool) -> list[dict[str, Any]]:
    device, environment = _prepare_environment(params); write_experiment_manifest(exp_root, params, model_names, device, environment); save_json(exp_root / "environment.json", environment)
    _header("HIDDEN STATE EXTRACTION EXPERIMENT", show=show_info)
    if show_info:
        print(f"  Experiment ID       : {params.experiment_id}\n  Hyperparameter hash : {params.config_hash}\n  Device              : {device}\n  Models requested    : {len(model_names)}\n  Datasets            : {len(datasets)}\n  Pooling             : {params.pooling}\n  Max sequence length : {params.max_length}\n  Runtime batch tuning: DISABLED\n  Existing experiment : {'YES' if resumed else 'NO'}\n  Output root         : {exp_root}\n  Transformers        : {transformers.__version__}\n  PyTorch             : {torch.__version__}")
    print_system_report(show_verbose)
    results: list[dict[str, Any]] = []
    for i, model_name in enumerate(model_names, 1):
        spec = get_model_spec(model_name); frozen_batch = params.batch_for(model_name, spec); model_started = time.perf_counter(); _header(f"MODEL {i}/{len(model_names)} :: {model_name}", show=show_info)
        if show_info:
            print(f"  Family              : {spec.family}\n  Generation          : {spec.generation}\n  Parameters          : {spec.parameter_billions:g}B\n  Architecture        : {spec.architecture}\n  Training status     : {spec.training_status}\n  FROZEN batch        : {frozen_batch}\n  Gated               : {spec.gated}")
        ok, reason = check_transformers_version(spec)
        if not ok:
            _print_critical(f"⚠ SKIPPED: {reason}", show_critical); results.append({"model":model_name,"status":"skipped_version","reason":reason}); continue
        # Repair missing auxiliary files for all datasets (even if complete)
        for dataset_name, dataset in datasets.items():
            out_dir = build_dataset_directory(exp_root, model_name, dataset_name)
            try:
                repair_result = repair_dataset_auxiliaries(
                    dataset=dataset,
                    output_dir=out_dir,
                    batch_size=frozen_batch,
                    pooling=params.pooling,
                    max_length=params.max_length,
                    storage_dtype=np.float32 if params.storage_dtype == "float32" else np.float16,
                    experiment_id=params.experiment_id,
                    hyperparameter_hash=params.config_hash,
                    show_info=show_info,
                )
                if repair_result.get("actions"):
                    _print_info(f"  Repaired {dataset_name}: {', '.join(repair_result['actions'])}", show_info)
                if repair_result.get("metadata_updated"):
                    _print_info(f"  Updated metadata for {dataset_name}", show_info)
            except Exception as exc:
                _print_critical(f"⚠ Repair failed for {dataset_name}: {exc}", show_critical)
        if all_model_datasets_complete(datasets, exp_root, model_name, hyperparameters=params):
            _print_critical("✓ SKIPPED — all datasets already complete.", show_critical); results.append({"model":model_name,"status":"already_complete"}); continue
        model = None; tokenizer = None
        try:
            snapshot, revision, prep = prepare_model(model_name, params, show_verbose, show_info, show_critical); config = load_model_config(snapshot); dtype = params.dtype_for(device)
            preflight = preflight_model(spec, snapshot, config, device, dtype, params.max_length, show_verbose)
            if preflight["status"] == "skip_total_memory": raise RuntimeError(f"Preflight rejected model: estimated peak {preflight['estimated_model_load_peak_gb']:.2f} GiB > safe total {preflight['memory']['safe_total_memory_gb']:.2f} GiB")
            tokenizer = load_tokenizer(snapshot, model_name, spec.trust_remote_code, show_verbose, show_info); model, backend, diagnostics, load_time = load_model(snapshot, model_name, tokenizer, dtype, device, spec.trust_remote_code, params.use_sdpa, params.strict_meta_validation, params.max_length, show_verbose, show_info, show_critical)
            runtime = {**asdict(spec),"config_model_type":getattr(config,"model_type",None),"hidden_layers":get_model_num_layers(config),"hidden_size":get_model_hidden_size(config),"inference_dtype":str(dtype),"frozen_batch_size":frozen_batch,"attention_backend":backend,"load_diagnostics":diagnostics,"preflight":preflight,"model_revision":revision,"preparation_seconds":prep,"load_seconds":load_time,"hyperparameter_hash":params.config_hash,"system_fingerprint":system_fingerprint(),"experiment_policy":{"batch_size_mutation":False,"dtype_mutation":False,"pooling_mutation":False,"max_length_mutation":False}}; write_model_metadata(build_model_directory(exp_root, model_name), spec, device, dtype, snapshot, revision, runtime, params)
            for dataset_name, dataset in datasets.items():
                result = extract_dataset(model, tokenizer, dataset, None, build_dataset_directory(exp_root, model_name, dataset_name), frozen_batch, params.pooling, params.max_length, np.float32 if params.storage_dtype == "float32" else np.float16, device, model_name, dataset_name, snapshot, False, frozen_batch, params.flush_every_batches, params.flush_every_seconds, show_verbose, show_info, show_critical, show_debug, params.experiment_id, params.config_hash)
                result.update({"model_spec":asdict(spec),"model_revision":revision,"preparation_seconds":prep,"model_load_seconds":load_time}); results.append(result)
            _print_critical(f"\n✓ MODEL COMPLETE: {model_name}", show_critical); _print_info(f"  Model elapsed       : {(time.perf_counter()-model_started)/60:.2f} min", show_info)
        except KeyboardInterrupt: raise
        except Exception as exc:
            write_runtime_diagnostic(exp_root / "runtime_state.json", runtime_diagnostic_snapshot(stage="model_failure", model_name=model_name, experiment_id=params.experiment_id, hyperparameter_hash=params.config_hash, extra={"status":"failed","error_type":type(exc).__name__,"error":str(exc),"traceback":_truncate_traceback(traceback.format_exc()),"frozen_batch_size":frozen_batch}))
            _print_critical("\n" + _separator("!") + "\n" f"MODEL FAILED: {model_name}\n" f"Error: {type(exc).__name__}: {exc}\n" f"Frozen batch: {frozen_batch}\n" f"Hyperparameter hash: {params.config_hash}\n" f"Traceback:\n{_truncate_traceback(traceback.format_exc())}\n" + _separator("!"), show_critical)
            result = {"model":model_name,"status":"failed","error_type":type(exc).__name__,"error":str(exc),"experiment_id":params.experiment_id,"hyperparameter_hash":params.config_hash,"frozen_batch_size":frozen_batch}; results.append(result); append_jsonl(MEASUREMENT_LEDGER, result)
            if not continue_on_model_error: raise
        finally:
            if model is not None: del model
            if tokenizer is not None: del tokenizer
            cleanup(device)
    save_json(exp_root / "results.json", {"experiment_id":params.experiment_id,"results":results}); _header("ALL REQUESTED MODEL JOBS PROCESSED", show=show_info)
    if show_info:
        print(f"  Experiment ID       : {params.experiment_id}\n  Hyperparameter hash : {params.config_hash}\n  Result records      : {len(results)}\n  Successful          : {sum(r.get('status') in {'complete','already_complete'} for r in results)}\n  Skipped             : {sum(str(r.get('status','')).startswith('skipped') for r in results)}\n  Failed              : {sum(r.get('status') == 'failed' for r in results)}")
    return results


def run_experiments(datasets: Mapping[str, Any], model_names: Sequence[str], base_output: str | Path = EXTERNAL_ROOT, batch_size: int | None = None, pooling: str | None = None, max_length: int | None = None, use_half_precision: bool | None = None, auto_batch_size: bool = False, flush_every_batches: int | None = None, continue_on_model_error: bool = True, show_verbose: bool = True, show_info: bool = True, show_critical: bool = True, show_debug: bool = False, hyperparameters_path: str | Path = HYPERPARAMETER_CONFIG_PATH, experiment_id: str | None = None) -> list[dict[str, Any]]:
    if auto_batch_size: raise ValueError("auto_batch_size=True is disabled in deterministic extraction")
    if not datasets or not model_names: raise ValueError("datasets and model_names must be non-empty")
    base_output = Path(base_output).resolve(); base_output.mkdir(parents=True, exist_ok=True); base_output.relative_to(EXTERNAL_ROOT)
    base_params = load_hyperparameters(hyperparameters_path); params, exp_root, resumed = resolve_experiment_parameters(base_output, base_params, batch_size=batch_size, pooling=pooling, max_length=max_length, use_half_precision=use_half_precision, flush_every_batches=flush_every_batches, experiment_id=experiment_id, show_info=show_info)
    return _run_resolved_experiment(datasets=datasets, model_names=list(model_names), params=params, exp_root=exp_root, resumed=resumed, continue_on_model_error=continue_on_model_error, show_verbose=show_verbose, show_info=show_info, show_critical=show_critical, show_debug=show_debug)


def run_model_matrix(datasets: Mapping[str, Any], groups: Sequence[str] | None = None, base_output: str | Path = EXTERNAL_ROOT, pooling: str | None = None, max_length: int | None = None, use_half_precision: bool | None = None, auto_batch_size: bool = False, flush_every_batches: int | None = None, continue_on_model_error: bool = True, show_verbose: bool = True, show_info: bool = True, show_critical: bool = True, show_debug: bool = False, hyperparameters_path: str | Path = HYPERPARAMETER_CONFIG_PATH, experiment_id: str | None = None) -> list[dict[str, Any]]:
    if auto_batch_size: raise ValueError("Automatic batch growth/shrink is disabled in deterministic extraction")
    if not datasets: raise ValueError("datasets must be non-empty")
    specs = get_model_specs(groups); names = [s.name for s in specs]
    if not names: raise ValueError("No models selected by groups")
    _header("MODEL MATRIX", show=show_info)
    if show_info:
        print(f"  Requested experiment : {experiment_id or DEFAULT_EXPERIMENT_ID}\n  Transformers         : {transformers.__version__}\n  PyTorch              : {torch.__version__}\n  Models               : {len(specs)}\n  Datasets             : {len(datasets)}\n  Runtime batch tuning : DISABLED\n  Output root          : {base_output}")
        for i, s in enumerate(specs, 1): print(f"{i:02d}. [{s.group}] {s.family:<12} {s.parameter_billions:g}B  {s.name}")
    base_output = Path(base_output).resolve(); base_output.mkdir(parents=True, exist_ok=True); base_output.relative_to(EXTERNAL_ROOT)
    base_params = load_hyperparameters(hyperparameters_path); params, exp_root, resumed = resolve_experiment_parameters(base_output, base_params, pooling=pooling, max_length=max_length, use_half_precision=use_half_precision, flush_every_batches=flush_every_batches, experiment_id=experiment_id, show_info=show_info)
    return _run_resolved_experiment(datasets=datasets, model_names=names, params=params, exp_root=exp_root, resumed=resumed, continue_on_model_error=continue_on_model_error, show_verbose=show_verbose, show_info=show_info, show_critical=show_critical, show_debug=show_debug)


def summarize_measurement(metadata: Mapping[str, Any]) -> dict[str, Any]:
    p = metadata.get("performance", {}); d = metadata.get("dataset", {}); e = metadata.get("extraction", {}); m = metadata.get("model", {}); t = p.get("timing_seconds", {})
    return {"experiment_id":metadata.get("experiment_id"),"hyperparameter_hash":metadata.get("hyperparameter_hash"),"model":m.get("name"),"dataset":d.get("name"),"dataset_fingerprint":d.get("fingerprint"),"batch_size":e.get("batch_size"),"max_length":e.get("max_length"),"pooling":e.get("pooling"),"device":e.get("device"),"elapsed_seconds":p.get("elapsed_seconds"),"samples_per_second":p.get("samples_per_second"),"tokens_per_second":p.get("tokens_per_second"),"tokenize_seconds":t.get("tokenize"),"transfer_seconds":t.get("transfer"),"forward_seconds":t.get("forward"),"pooling_seconds":t.get("pooling"),"convert_seconds":t.get("convert"),"write_seconds":t.get("write"),"completed_samples":p.get("completed_samples"),"successful_batches":p.get("successful_batches")}


def record_hyperparameter_measurement(experiment_id: str, hyperparameter_hash: str, hyperparameters: Mapping[str, Any], measurement: Mapping[str, Any]) -> None:
    append_jsonl(MEASUREMENT_LEDGER, {"record_type":"hyperparameter_measurement","experiment_id":experiment_id,"hyperparameter_hash":hyperparameter_hash,"hyperparameters":dict(hyperparameters),"measurement":dict(measurement),"recorded_at":time.time()})

ALL_PRIMARY_MODEL_NAMES = tuple(m.name for m in MODEL_REGISTRY)

__all__ = [
    "HyperParameters", "ModelSpec", "MODEL_REGISTRY", "MODEL_BY_NAME", "ALL_PRIMARY_MODEL_NAMES", "GROUP_ORDER",
    "run_experiments", "run_model_matrix", "extract_dataset", "load_hyperparameters", "save_json", "summarize_measurement",
    "record_hyperparameter_measurement", "get_model_specs", "detect_text_column", "load_model", "load_tokenizer", "model_device",
    "classify_model_loading_issues", "classify_loading_info",
]