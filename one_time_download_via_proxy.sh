#!/bin/bash
# Robust Hugging Face download through Cloudflare Worker proxy.
# Handles Cloudflare's Content-Length stripping by:
#   1. Using --include "*.safetensors" for weights (LFS HEAD redirect is immune)
#   2. Using plain curl GETs for metadata files (GET is never stripped)
set -u

export HF_ENDPOINT="https://wandering-tooth-2acd.amirali-fmli3.workers.dev"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

download_model() {
  local repo="$1"
  local slug
  slug="models--${repo//\//--}"

  echo "=================================================="
  echo "Model: $repo"
  echo "=================================================="

  # Step 1: weights only (immune to Content-Length stripping)
  hf download "$repo" --include "*.safetensors" "*.index.json" || {
    echo "!! Weight download failed for $repo"
    return 1
  }

  # Step 2: locate snapshot dir
  local model_dir="$HF_CACHE/$slug"
  local snapshot
  snapshot=$(find "$model_dir/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
  if [ -z "$snapshot" ]; then
    echo "!! No snapshot dir for $repo"
    return 1
  fi

  # Step 3: metadata via curl GET
  local base="$HF_ENDPOINT/$repo/resolve/main"
  for FILE in config.json tokenizer.json tokenizer_config.json \
              special_tokens_map.json vocab.txt vocab.json merges.txt \
              added_tokens.json generation_config.json \
              spiece.model spm.model tokenizer.model; do
    # Skip if already present and non-empty
    [ -s "$snapshot/$FILE" ] && continue
    curl -fsSL --retry 3 --retry-delay 2 \
      --connect-timeout 60 --max-time 300 \
      -o "$snapshot/$FILE" \
      "$base/$FILE" 2>/dev/null || true
  done

  echo "  ✓ $repo done ($(du -sh "$model_dir" | cut -f1))"
}

# Models list — non-gated first
MODELS=(
  "google-bert/bert-base-uncased"
  "distilbert/distilbert-base-uncased"
  "FacebookAI/roberta-base"
  "google/electra-small-discriminator"
  "microsoft/deberta-v3-small"
  "gpt2"
  "EleutherAI/gpt-neo-125m"
  "facebook/opt-125m"
  "HuggingFaceTB/SmolLM2-135M"
  "HuggingFaceTB/SmolLM2-360M"
  "HuggingFaceTB/SmolLM2-1.7B"
  "Qwen/Qwen2-0.5B"
  "Qwen/Qwen2-1.5B"
  "Qwen/Qwen2.5-0.5B"
  "Qwen/Qwen2.5-1.5B"
  "Qwen/Qwen2.5-3B"
  "Qwen/Qwen3-0.6B-Base"
  "Qwen/Qwen3-1.7B-Base"
  "Qwen/Qwen3-4B-Base"
  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

for m in "${MODELS[@]}"; do
  download_model "$m" || echo "!! SKIPPED: $m"
done

echo "All downloads complete."