"""
D.2 — Jacobian Lens / J-Space (Anthropic, 2026)

What it is. Anthropic's 2026 technique uncovers a hidden space — the "J-space" — inside Claude Opus 4.6. 
It contains individual words related to what the model is most likely to say next, effectively exposing the model's unspoken thoughts before it commits to an output. 
The technique is called the Jacobian lens, and it works by using the Jacobian of the model's output with respect to intermediate-layer activations to read out what 
the model is "planning to say."

Why it matters for your project. The J-lens is exactly the kind of tool that would let you ask: 
does the model internally anticipate an emotional word before it produces one? If you can extract the J-space at layer L for an emotional input, 
you can see whether the model's "thought" at that layer is already gravitating toward an emotional token — even if the probe would not classify it as emotion. 
This is a completely new axis that no one in the emotion-probing literature has used yet, and it would be a novel contribution.

Caveat. The J-lens is currently demonstrated on Claude Opus 4.6 and Qwen 3.6 27B. Adapting it to your small models (Qwen2-0.5B, etc.) is a research task, 
not a drop-in. If your time budget allows, this is the highest-risk, highest-reward technique on this list.
"""

# ── Path bootstrap: keep `from _shared import ...` working from expl/ ──
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────




# this is an ongoing area of research with only test on Claude Opus 4.6 and Qwen 3.6 27B 
# it's not efficient to implement this unless we change the question to how to detect emotions using j lens