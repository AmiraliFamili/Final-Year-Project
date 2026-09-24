"""
Steering vectors / Representation Engineering (steering.py)

What it is. Given a class direction (e.g. the CAV from TCAV, 
or the difference between class centroids from geometry), add that direction to the hidden state 
at inference time and observe how the model's output changes. This is causal intervention, 
not correlation. If adding the joy direction to a neutral sentence makes the model produce joyful text, 
you have proof that the direction is causally involved in the emotion behaviour, not just correlated with it.

Why it matters for your argument. Correlation does not imply causation. Probing, geometry, CKA, and TCAV are all correlational. 
Steering is the only technique on this list that provides causal evidence. A 2026 paper reports that steering vectors face a control-quality trade-off in free-form generation, 
and that hybrid approaches are needed in practice. Another 2026 paper introduces AlphaSteer, which uses null-space constraints to preserve utility while steering safety. 
These are directly relevant: they tell you that steering is powerful but fragile, and you must report both the steering effect and its side effects.
"""

# ── Path bootstrap: keep `from _shared import ...` working from expl/ ──
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────



def steering_effect(
    model, tokenizer, probe, layer_index, direction, text, device="cpu",
    scale=2.0,
):
    """
    Run the model on `text`, but add `scale * direction` to the hidden state
    at `layer_index` before it continues through the remaining layers.
    Returns the change in the probe's score for the target class.
    """
    # Register a hook on the layer whose output we want to steer.
    steer_vec = torch.tensor(direction, dtype=torch.float32, device=device) * scale
    def hook(module, input, output):
        # output is a tuple; the first element is the hidden state.
        h = output[0]
        h = h + steer_vec
        return (h,) + output[1:]
    handle = model.model.layers[layer_index].register_forward_hook(hook)
    try:
        out = model(**tokenizer(text, return_tensors="pt").to(device),
                    output_hidden_states=True)
    finally:
        handle.remove()
    h = out.hidden_states[layer_index]
    pooled = h.mean(dim=1)
    return probe.predict_proba(pooled.cpu().numpy())




def get_layer_module(model, layer_index):
    """
    Return the nn.Module whose forward output is the hidden state we
    want to steer. Handles BERT/DistilBERT/RoBERTa/ELECTRA/DeBERTa/GPT2/
    GPT-Neo/OPT/Qwen/Llama/TinyLlama.
    """
    for path in [
        ("encoder", "layer"),                     # BERT, RoBERTa, DeBERTa
        ("model", "layers"),                      # Qwen, Llama, TinyLlama
        ("transformer", "h"),                     # GPT-2, GPT-Neo
        ("model", "decoder", "layers"),           # OPT, some Qwen variants
        ("bert", "encoder", "layer"),             # alternative paths
        ("distilbert", "transformer", "layer"),   # DistilBERT
    ]:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False; break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "__getitem__"):
            return obj[layer_index]
    raise RuntimeError("Could not locate layer module for steering.")