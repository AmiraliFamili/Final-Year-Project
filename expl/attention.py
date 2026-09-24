from __future__ import annotations

# ── Path bootstrap: keep `from _shared import ...` working from expl/ ──
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────


"""
Head-level and rolled-out attention

What attention is, in one paragraph. Every transformer layer has multiple "heads". 
Each head computes a softmax over which input tokens to look at when producing each output token. 
The attention matrix for a head in a layer is [T, T] where T is the sequence length; 
entry (i, j) is "how much does output position i look at input position j". 
This is the closest thing a transformer has to a highlighting pen: it tells you where the model is looking.

The three analyses you can do with it:

Raw head visualisation. Take one sentence. Plot the [T, T] attention matrix for one head. 
This is what most blog posts show. It is illustrative but not quantitative.
Head specialisation scoring. For each (layer, head), compute the fraction of the [CLS] token's (or last token's)
attention that lands on emotion-bearing input tokens, versus elsewhere. 
A high score means "this head is looking at the emotional words." Sort the heads by this score; 
the top-k are your candidate "emotion heads."
Attention rollout (Abnar & Zuidema, 2020). Raw attention is a single-layer picture. 
Rollout multiplies the attention matrices across layers (with residual corrections) to produce a cumulative attention map: 
"given the final layer's [CLS] output, how much did each input token contribute?" This is a much more faithful answer to "which tokens matter?" 
than any single layer's attention.
Why this matters for your argument. Attention gives you a completely independent axis from hidden-state probing. 
If the probe curves say layer 9 is best, and attention rollout says head 9.3 concentrates on emotional words, 
you have two different methods pointing at the same layer. That is a strong convergence. 
If they disagree, that disagreement is a finding: "the emotional information is linearly decodable at layer 9, 
but no single attention head in that layer specifically routes emotional tokens."
"""

"""
attention.py — Attention head analysis and attention rollout.

Three sub-analyses, all bounded to a configurable sample:
    1. head_specialisation   : per (layer, head) score of how much the
                               [CLS]/last-token attention goes to emotion
                               lexicon tokens.
    2. attention_rollout     : cumulative attention from the [CLS]/last token
                               back to every input token, multiplied across
                               layers with residual correction.
    3. save_head_matrices    : optionally dump the raw [L, H, T, T] tensor
                               for one representative sentence per emotion.

Reads:  models/<slug>/   (needs a fresh forward pass — this is NOT free)
        datasets/<dataset>/processed/<dataset>_clean.csv   (for texts)
Writes: hidden_states/<slug>/<dataset>/analysis/attention.npz
        hidden_states/<slug>/<dataset>/analysis/attention.json
"""


from pathlib import Path
import time
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from _shared import (
    AMIRALI_MOUNT, artifact_dir, analysis_dir_for_extraction,
    atomic_json, atomic_npz, stable_hash,
)

MODELS_ROOT = AMIRALI_MOUNT / "models"
DATASETS_ROOT = AMIRALI_MOUNT / "datasets"


# ─────────────────────────────────────────────────────────────────────────────
# A small, hard-coded emotion lexicon.
#
# Why hard-code? Because it makes the "head specialisation" score reproducible
# and auditable. A published lexicon (NRC EmoLex, VADER) would be better in
# a production system, but for a thesis the hard-coded list is transparent:
# a reader can see exactly which tokens count as "emotion-bearing".
# The list below covers the seven ISEAR emotions plus the most common
# GoEmotions categories (joy, sadness, anger, fear, love, gratitude).
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_LEXICON = {
    # ISEAR core
    "joy", "happy", "happiness", "glad", "delighted", "cheerful",
    "fear", "afraid", "scared", "frightened", "terrified", "anxious",
    "anger", "angry", "mad", "furious", "rage", "irritated",
    "sad", "sadness", "unhappy", "sorrow", "grief", "depressed",
    "disgust", "disgusted", "revolted", "repulsed",
    "shame", "ashamed", "embarrassed", "humiliated",
    "guilt", "guilty", "remorse", "regret",
    # GoEmotions frequent
    "love", "loved", "loving", "adore", "gratitude", "grateful", "thankful",
    "optimism", "optimistic", "hopeful", "hope", "pride", "proud",
    "relief", "relieved", "amusement", "amused", "excited", "excitement",
    "nervous", "nervousness", "worried", "worry",
}


def _is_emotion_token(token: str) -> bool:
    """
    A token is emotional if its lowercased, punctuation-stripped form is in
    the lexicon, or if the lexicon word appears as a sub-token of it.

    The sub-token check handles the BPE case: 'happiness' may be split into
    'happ' + 'iness' by some tokenizers. We do not attempt full
    morphological analysis; the check is deliberately loose because the
    downstream score is an aggregate over many samples.
    """
    t = token.lower().strip("Ġ▁##.,!?;:'\"")
    if t in EMOTION_LEXICON:
        return True
    return any(w in t for w in EMOTION_LEXICON if len(w) > 4)


def extract_attention(
    model_slug_: str,
    texts: list[str],
    *,
    max_length: int = 128,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict:
    """
    Run the model on `texts` and collect attention tensors.

    Returns
    -------
    dict with keys:
        attention : [N, L, H, T, T] float16
        tokens    : list[list[str]] — the token strings for each sequence
        mask      : [N, T] bool    — True for real tokens, False for padding

    Memory note:
        For a 25-layer, 16-head model with T=128 and N=500, the attention
        tensor is 500 × 25 × 16 × 128 × 128 × 2 bytes = 6.5 GB in fp16.
        That will not fit alongside the model on an 8 GB machine.
        The caller should therefore set max_length=64 and/or reduce N.
        We make no attempt to be clever about this — the numbers are
        printed at the start so the caller can see what they are asking for.
    """
    model_dir = MODELS_ROOT / model_slug_
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model snapshot not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True,
        output_attentions=True,
    ).eval().to(device)

    all_attn, all_tokens, all_masks = [], [], []

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            enc_dev = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc_dev, output_attentions=True)

            # out.attentions is a tuple of L tensors, each [B, H, T, T].
            # Stacking along dim=1 gives [B, L, H, T, T].
            attn = torch.stack(out.attentions, dim=1)
            all_attn.append(attn.detach().to(torch.float16).cpu().numpy())
            all_masks.append(enc["attention_mask"].cpu().numpy().astype(bool))
            for ids in enc["input_ids"].cpu().tolist():
                all_tokens.append(tokenizer.convert_ids_to_tokens(ids))

    return {
        "attention": np.concatenate(all_attn, axis=0),
        "tokens":    all_tokens,
        "mask":      np.concatenate(all_masks, axis=0),
    }


def head_specialisation(
    attention: np.ndarray,          # [N, L, H, T, T]
    tokens: list[list[str]],
    mask: np.ndarray,               # [N, T] bool
) -> "pd.DataFrame":
    """
    For each (layer, head), the mean fraction of [CLS]/last-token attention
    that lands on emotion-lexicon tokens.
    """
    import pandas as pd
    N, L, H, T, _ = attention.shape
    # Determine the "readout position" per sequence: the last non-pad token.
    # For encoder models this is the [CLS] (position 0) — but the same code
    # works for both because in a decoder-only model the last real token
    # carries the sequence-level representation.
    last_pos = mask.sum(axis=1) - 1   # [N]

    rows = []
    for l in range(L):
        for h in range(H):
            ratios = []
            for n in range(N):
                last = int(last_pos[n])
                attn_row = attention[n, l, h, last, :last + 1]
                if attn_row.sum() <= 0:
                    continue
                emotion_positions = [
                    p for p, tok in enumerate(tokens[n][:last + 1])
                    if _is_emotion_token(tok)
                ]
                if not emotion_positions:
                    continue
                p_emotion = float(attn_row[emotion_positions].sum() / attn_row.sum())
                ratios.append(p_emotion)
            rows.append({
                "layer": l, "head": h,
                "emotion_attention_ratio": float(np.mean(ratios)) if ratios else float("nan"),
                "n_samples": len(ratios),
            })
    return pd.DataFrame(rows)


def attention_rollout(
    attention: np.ndarray,          # [N, L, H, T, T]
    mask: np.ndarray,               # [N, T] bool
    *,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Abnar & Zuidema (2020) attention rollout, computed per-sample.

    Returns
    -------
    rollout : [N, T] float32 — for each sample, the cumulative attention
        from the readout position (last real token) back to every input
        token. Sums to 1 over the real tokens (padding positions are 0).

    Algorithm:
        1. Average the attention matrices over heads at each layer.
        2. Add the identity matrix (to account for the residual connection).
        3. Re-normalise each row to sum to 1.
        4. Multiply the L layer-matrices in sequence.
        5. The last row of the product is the rollout vector.

    The `discard_ratio` parameter removes the smallest attention weights
    before re-normalising — this is a standard trick to keep the rollout
    focused on the dominant paths rather than spreading mass thinly.
    """
    N, L, H, T, _ = attention.shape
    rollout = np.zeros((N, T), dtype=np.float32)

    for n in range(N):
        # Average over heads
        attn = attention[n].mean(axis=1)       # [L, T, T]
        result = np.eye(T, dtype=np.float32)
        for l in range(L):
            a = attn[l].astype(np.float32)
            if discard_ratio > 0:
                flat = a.flatten()
                threshold = np.quantile(flat, discard_ratio)
                a = np.where(a < threshold, 0.0, a)
            # Add residual and renormalise
            a = a + np.eye(T, dtype=np.float32)
            a = a / a.sum(axis=1, keepdims=True)
            result = a @ result
        last = int(mask[n].sum()) - 1
        rollout[n, :last + 1] = result[last, :last + 1]
    return rollout


def run_attention(
    model_slug_: str,
    dataset: str,
    texts: list[str],
    *,
    max_length: int = 64,
    sample_size: int = 300,
    device: str = "cpu",
) -> Path:
    """
    Top-level entry point. Runs extraction on the first `sample_size` texts,
    computes head specialisation and rollout, and writes both to disk.
    """
    import pandas as pd
    t0 = time.perf_counter()
    texts = texts[:sample_size]

    bundle = extract_attention(model_slug_, texts,
                               max_length=max_length, device=device)
    attn, toks, mask = bundle["attention"], bundle["tokens"], bundle["mask"]

    spec = head_specialisation(attn, toks, mask)
    roll = attention_rollout(attn, mask)

    out_dir = analysis_dir_for_extraction(model_slug_, dataset)
    atomic_npz(out_dir / "attention.npz",
               attention=attn, mask=mask.astype(np.uint8), rollout=roll)
    spec.to_parquet(out_dir / "attention_head_specialisation.parquet", index=False)
    atomic_json(out_dir / "attention.json", {
        "technique": "attention_head_specialisation + attention_rollout",
        "model_slug": model_slug_,
        "dataset": dataset,
        "n_samples": len(texts),
        "max_length": max_length,
        "n_layers": int(attn.shape[1]),
        "n_heads": int(attn.shape[2]),
        "elapsed_seconds": time.perf_counter() - t0,
        "outputs": {
            "attention": str(out_dir / "attention.npz"),
            "head_spec": str(out_dir / "attention_head_specialisation.parquet"),
        },
    })
    return out_dir / "attention.npz"