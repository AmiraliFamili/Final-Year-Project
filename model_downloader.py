#!/usr/bin/env python3
"""
model_downloader.py — adaptive, self-healing, Iran-friendly HF downloader.

Strategies tried in order, per endpoint:
    1. local_dir            snapshot_download(local_dir=dest)
    2. cache_then_link      snapshot_download(cache_dir=…) + hardlink
    3. materialize_only     whatever is already in any local HF cache
    4. per_file             one file at a time (huggingface_hub → curl)
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── absolute paths ────────────────────────────────────────────────────────
EXTERNAL_MOUNT   = Path("/Volumes/Amirali")
MODELS_ROOT      = EXTERNAL_MOUNT / "models"
HF_CACHE_ROOT    = EXTERNAL_MOUNT / "Probing-Emotions" / ".hf_cache"
HF_HUB_CACHE     = HF_CACHE_ROOT / "hub"
ENDPOINT_CACHE   = EXTERNAL_MOUNT / "Probing-Emotions" / ".endpoint_cache.json"
USER_HF_CACHE    = Path.home() / ".cache" / "huggingface" / "hub"

MODELS_ROOT.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME",           str(HF_CACHE_ROOT))
os.environ.setdefault("HF_HUB_CACHE",      str(HF_HUB_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "900")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT",     "60")
os.environ.setdefault(
    "HF_FALLBACK_ENDPOINTS",
    "https://hf-mirror.com,https://huggingface.co",
)

import requests
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from huggingface_hub.utils import (
    HfHubHTTPError, GatedRepoError, RepositoryNotFoundError,
)
try:
    from huggingface_hub.errors import FileMetadataError
except ImportError:
    FileMetadataError = Exception
try:
    from huggingface_hub.utils import LocalEntryNotFoundError
except ImportError:
    LocalEntryNotFoundError = Exception


# ── small metadata files ──────────────────────────────────────────────────
SMALL_FILES = [
    "config.json", "generation_config.json",
    "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "vocab.txt",
    "spiece.model", "spm.model",
    "sentencepiece.bpe.model", "tokenizer.model",
]


# ── endpoint resolution ───────────────────────────────────────────────────
def _candidate_endpoints() -> list[str]:
    out, seen = [], set()

    def add(url: str | None) -> None:
        if not url:
            return
        url = url.rstrip("/")
        if url and url not in seen:
            seen.add(url)
            out.append(url)

    add(os.environ.get("HF_ENDPOINT"))
    for u in (os.environ.get("HF_FALLBACK_ENDPOINTS") or "").split(","):
        add(u.strip())
    add("https://huggingface.co")
    return out


def _health_check(endpoint: str, timeout: float = 5.0) -> bool:
    try:
        r = requests.head(
            f"{endpoint.rstrip('/')}/api/models/gpt2",
            timeout=timeout, allow_redirects=True,
        )
        return r.status_code in (200, 401, 403)
    except Exception:
        return False


def probe_endpoints(force: bool = False) -> list[str]:
    """Return only endpoints that actually answer a probe.

    huggingface.co is always kept as a final fallback, because it is the
    canonical origin and a failed health check there is almost always
    transient. Other mirrors are dropped from the plan if they do not
    answer within the probe timeout — trying them produces multi-minute
    hangs with no recovery.
    """
    cached: str | None = None
    if ENDPOINT_CACHE.is_file():
        try:
            cached = json.loads(ENDPOINT_CACHE.read_text()).get("primary")
        except Exception:
            cached = None

    healthy: list[str] = []
    for e in _candidate_endpoints():
        if e == "https://huggingface.co":
            healthy.append(e)                      # always keep the origin
        elif _health_check(e):
            healthy.append(e)

    if cached and cached in healthy and healthy[0] != cached:
        # Re-order so the last-known-good endpoint is tried first.
        healthy.remove(cached)
        healthy.insert(0, cached)

    # Record the chosen primary, if it isn't the origin.
    if healthy and healthy[0] != "https://huggingface.co":
        try:
            ENDPOINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            ENDPOINT_CACHE.write_text(json.dumps(
                {"primary": healthy[0], "probed_at": time.time()}, indent=2))
        except Exception:
            pass

    return healthy or ["https://huggingface.co"]


# ── per-repo allow / ignore (THE MISSING FUNCTION) ────────────────────────
def _dynamic_patterns(model_name: str, endpoint: str | None) -> tuple[list[str], list[str]]:
    """Compute allow/ignore for a specific repo, with a bounded API call."""
    files: list[str] = []
    if endpoint != "https://hf-mirror.com":   # known-flaky; go straight to defaults
        try:
            api = HfApi(endpoint=endpoint) if endpoint else HfApi()
            # Bound the listing call. huggingface_hub respects HF_HUB_ETAG_TIMEOUT
            # for HTTP requests; we set it defensively here so a slow endpoint
            # cannot block the whole pipeline.
            _prev = os.environ.get("HF_HUB_ETAG_TIMEOUT")
            os.environ["HF_HUB_ETAG_TIMEOUT"] = "15"
            try:
                try:
                    files = api.list_repo_files(repo_id=model_name, repo_type="model")
                except TypeError:
                    files = api.list_repo_files(model_name, repo_type="model")
            finally:
                if _prev is not None:
                    os.environ["HF_HUB_ETAG_TIMEOUT"] = _prev
                else:
                    os.environ.pop("HF_HUB_ETAG_TIMEOUT", None)
        except Exception:
            files = []

    if files:
        has_safe = any(f.endswith(".safetensors") for f in files)
        has_bin  = any(f.endswith(".bin")         for f in files)
    else:
        has_safe = has_bin = True      # can't tell → allow both

    allow = list(SMALL_FILES) + ["*.safetensors", "*.index.json", "*.py"]
    if has_bin and not has_safe:
        allow.append("*.bin")

    ignore = [
        "*.h5", "*.msgpack", "*.ot", "rust_model.ot",
        "tf_model.*", "flax_model.*",
        "*.onnx", "*.gguf", "*.ggml",
        "*.pt", "*.pth", "*.ckpt",
        "*.tflite", "*.pb",
    ]
    if has_safe:
        ignore.append("*.bin")
    return allow, ignore


# ── completeness ──────────────────────────────────────────────────────────
_REQUIRED_METADATA_ANY_OF = (
    # tokenizer.json covers most modern models; vocab.txt covers BERT; the
    # sentencepiece files cover T5-family. A model only needs one of these
    # to be loadable, but it needs at least one.
    ("tokenizer.json", "vocab.txt", "tokenizer.model", "spiece.model", "spm.model"),
)

_TOKENIZER_ALTERNATIVES = (
    "tokenizer.json",           # modern HF models
    "tokenizer.model",          # SentencePiece (T5, Llama-2 family)
    "spiece.model",             # SentencePiece v1.1
    "spm.model",                # mBART, NLLB
    "sentencepiece.bpe.model",  # mBART, NLLB
    "vocab.txt",                # BERT family
    "vocab.json",               # GPT-2 / OPT / RoBERTa family
)

def is_complete(dest: Path) -> bool:
    if not (dest / "config.json").is_file():
        return False
    if not (list(dest.glob("*.safetensors")) or list(dest.glob("*.bin"))):
        return False
    return any((dest / name).is_file() for name in _TOKENIZER_ALTERNATIVES)


def audit_all(model_names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in model_names:
        d = model_dir_for(name)
        if not d.is_dir():
            out[name] = "absent"
        elif not (d / "config.json").is_file():
            out[name] = "missing-config"
        elif not (list(d.glob("*.safetensors")) or list(d.glob("*.bin"))):
            out[name] = "missing-weights"
        elif not any((d / n).is_file() for n in _TOKENIZER_ALTERNATIVES):
            out[name] = "missing-tokenizer"
        else:
            out[name] = "complete"
    return out

def _missing_weight(dest: Path) -> bool:
    return not (any(dest.glob("*.safetensors")) or any(dest.glob("*.bin")))


def _weight_bytes(dest: Path) -> int:
    return sum(p.stat().st_size for p in dest.iterdir()
               if p.suffix in {".safetensors", ".bin"})


# ── curl for small files ──────────────────────────────────────────────────
def _curl_small_files(model_name: str, dest: Path, endpoint: str,
                      quiet: bool = False) -> list[str]:
    base = endpoint.rstrip("/")
    fetched: list[str] = []
    for name in SMALL_FILES:
        target = dest / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        url = f"{base}/{model_name}/resolve/main/{name}"
        r = subprocess.run(
            ["curl", "-fsSL", "--retry", "2", "--retry-delay", "2",
             "--connect-timeout", "30", "--max-time", "180",
             "-o", str(target), url],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            fetched.append(name)
            if not quiet:
                print(f"        + curl {name}  ({target.stat().st_size} B)")
        else:
            target.unlink(missing_ok=True)
    return fetched


# ── strategies ────────────────────────────────────────────────────────────
def _strategy_local_dir(model_name: str, dest: Path, endpoint: str | None) -> None:
    allow, ignore = _dynamic_patterns(model_name, endpoint)
    try:
        snapshot_download(
            repo_id=model_name, local_dir=str(dest),
            allow_patterns=allow, ignore_patterns=ignore,
            max_workers=4, endpoint=endpoint,
        )
    except LocalEntryNotFoundError:
        if endpoint and not _missing_weight(dest):
            _curl_small_files(model_name, dest, endpoint)
        if not is_complete(dest):
            raise


def _strategy_cache_then_link(model_name: str, dest: Path,
                              endpoint: str | None) -> None:
    allow, ignore = _dynamic_patterns(model_name, endpoint)
    try:
        snapshot_download(
            repo_id=model_name, cache_dir=str(HF_HUB_CACHE),
            allow_patterns=allow, ignore_patterns=ignore,
            max_workers=4, endpoint=endpoint,
        )
    except LocalEntryNotFoundError:
        if endpoint and not _missing_weight(dest):
            _curl_small_files(model_name, dest, endpoint)

    try:
        _materialize_from_any_cache(model_name, dest)
    except Exception as exc:
        if endpoint and not _missing_weight(dest):
            _curl_small_files(model_name, dest, endpoint)
        if not is_complete(dest):
            raise RuntimeError(
                f"cache_then_link could not populate {dest}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if not is_complete(dest):
        raise RuntimeError(
            f"cache_then_link finished without a complete snapshot at {dest}"
        )


def _strategy_materialize_only(model_name: str, dest: Path) -> None:
    _materialize_from_any_cache(model_name, dest)


def _strategy_per_file(model_name: str, dest: Path, endpoint: str | None) -> None:
    allow, ignore = _dynamic_patterns(model_name, endpoint)

    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    try:
        files = api.list_repo_files(repo_id=model_name, repo_type="model")
    except TypeError:
        files = api.list_repo_files(model_name, repo_type="model")

    wanted = [f for f in files
              if any(fnmatch.fnmatch(f, p) for p in allow)
              and not any(fnmatch.fnmatch(f, p) for p in ignore)]
    if not wanted:
        raise RuntimeError(f"No matching files for {model_name}")

    dest.mkdir(parents=True, exist_ok=True)
    base = (endpoint or "https://huggingface.co").rstrip("/")

    for fname in wanted:
        target = dest / fname
        if target.is_file() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            path = hf_hub_download(
                repo_id=model_name, filename=fname,
                cache_dir=str(HF_HUB_CACHE), endpoint=endpoint,
            )
            shutil.copy2(path, target)
            continue
        except FileMetadataError:
            pass
        except Exception:
            pass

        url = f"{base}/{model_name}/resolve/main/{fname}"
        r = subprocess.run(
            ["curl", "-fL", "--retry", "3", "--retry-delay", "2",
             "--connect-timeout", "60", "--max-time", "1800",
             "-o", str(target), url],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            if target.is_file():
                target.unlink(missing_ok=True)
            if "404" not in r.stderr:
                raise RuntimeError(
                    f"curl failed for {model_name}:{fname}: {r.stderr.strip()[:200]}")


# ── multi-cache materializer ──────────────────────────────────────────────
def _candidate_hf_dirs(model_name: str) -> list[Path]:
    slug = f"models--{model_name.replace('/', '--')}"
    out = [HF_HUB_CACHE / slug, USER_HF_CACHE / slug]
    alt = os.environ.get("HF_HOME")
    if alt:
        out.append(Path(alt) / "hub" / slug)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _materialize_from_any_cache(model_name: str, dest: Path) -> None:
    snap_root = blobs_src = None
    for hf_dir in _candidate_hf_dirs(model_name):
        sr = hf_dir / "snapshots"
        bs = hf_dir / "blobs"
        if not (sr.is_dir() and bs.is_dir()):
            continue
        for snap in sr.iterdir():
            if snap.is_dir() and is_complete(snap):
                snap_root, blobs_src = snap.parent, bs
                break
        if snap_root:
            break

    if not snap_root:
        raise RuntimeError(f"No complete snapshot in any local cache for {model_name}")

    snaps = [p for p in snap_root.iterdir() if p.is_dir() and is_complete(p)]
    snap = max(snaps, key=lambda p: p.stat().st_mtime)

    dest.mkdir(parents=True, exist_ok=True)

    for blob in blobs_src.iterdir():
        if not blob.is_file():
            continue
        dst = dest / blob.name
        if dst.exists():
            continue
        try:
            os.link(blob, dst)
        except OSError:
            shutil.copy2(blob, dst)

    for entry in snap.iterdir():
        dst = dest / entry.name
        if entry.is_symlink():
            blob = (entry.parent / os.readlink(entry)).resolve()
            if not blob.is_file() or dst.exists():
                continue
            try:
                os.link(blob, dst)
            except OSError:
                shutil.copy2(blob, dst)
        elif entry.is_file():
            if dst.exists():
                continue
            try:
                os.link(entry, dst)
            except OSError:
                shutil.copy2(entry, dst)
        elif entry.is_dir():
            shutil.copytree(entry, dst, dirs_exist_ok=True)

    try:
        (dest / ".revision").write_text(snap.name + "\n")
    except Exception:
        pass


# ── orchestrator ──────────────────────────────────────────────────────────
STRATEGIES = (
    ("local_dir",         _strategy_local_dir),
    ("cache_then_link",   _strategy_cache_then_link),
    ("materialize_only",  lambda m, d, e: _strategy_materialize_only(m, d)),
    ("per_file",          _strategy_per_file),
)


def download_model(model_name: str, dest: Path, *,
                   force: bool = False,
                   token: str | None = None,
                   quiet: bool = False) -> Path:
    dest = Path(dest)
    if not force and is_complete(dest):
        return dest

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    endpoints = probe_endpoints()
    plan: list[str | None] = []
    for e in endpoints:
        if e not in plan:
            plan.append(e)
    if None not in plan:
        plan.append(None)

    errors: list[str] = []
    for endpoint in plan:
        label_ep = endpoint or "local-cache-only"
        for name, fn in STRATEGIES:
            if name == "materialize_only" and endpoint is not None:
                continue
            t0 = time.time()
            try:
                if not quiet:
                    print(f"    · try [{name}] via {label_ep} …", flush=True)
                fn(model_name, dest, endpoint)
            except GatedRepoError:
                if not quiet:
                    print(f"      ✗ gated repo")
                raise RuntimeError(f"{model_name} is gated; set HF_TOKEN") from None
            except RepositoryNotFoundError:
                errors.append(f"[{name}/{label_ep}] 404")
                if not quiet:
                    print(f"      ✗ 404")
                continue
            except Exception as exc:
                errors.append(f"[{name}/{label_ep}] {type(exc).__name__}: {exc}")
                if not quiet:
                    print(f"      ✗ {type(exc).__name__}: {str(exc)[:140]}")
                continue

            if is_complete(dest):
                if not quiet:
                    print(f"      ✓ {dest}  "
                          f"({_weight_bytes(dest)/1024**3:.3f} GiB, "
                          f"{time.time()-t0:.1f}s)")
                return dest
            errors.append(f"[{name}/{label_ep}] produced no weights")

    try:
        _materialize_from_any_cache(model_name, dest)
        if is_complete(dest):
            return dest
    except Exception as exc:
        errors.append(f"[offline] {type(exc).__name__}: {exc}")

    if not (dest / "config.json").is_file() and dest.is_dir():
        shutil.rmtree(dest, ignore_errors=True)

    raise RuntimeError(
        f"All download strategies failed for {model_name}.\n"
        + "\n".join(f"  {e}" for e in errors[-12:])
    )


# ── helpers ───────────────────────────────────────────────────────────────
def model_dir_for(model_name: str) -> Path:
    return MODELS_ROOT / model_name.split("/")[-1]
