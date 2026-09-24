"""
Part 7 — Two 2026 methods that are directly relevant to your specific question

Two recent papers deserve a closer look because they address the exact confound your README flags.

"Linear Probes Detect Task Format, Not Reasoning Mode" (May 2026). This paper uses probes on Qwen3-14B and demonstrates that probes which appear to distinguish reasoning modes (deductive, inductive, abductive) with 100% accuracy are actually detecting the data source and task format, not the reasoning mode itself. The implication for your project is direct: your probes may be detecting the dataset (ISEAR vs GoEmotions) or the label distribution rather than emotion. The paper's recommended controls — residual de-confounding, trace-anchor, causal steering — are exactly the techniques in Tier C and D above. Citing this paper and running its controls would put your work at the methodological frontier.

"Convergence Without Understanding" (May 2026). This paper finds that models converge more on problems they collectively fail (CKA = 0.897) than on those they solve (CKA = 0.830), and that pre-decision representations align (CKA = 0.875) while post-decision representations diverge (CKA = 0.274). For your cross-model CKA analysis, this is the key insight: compute CKA at every layer, not just the best probe layer. The interesting convergence is in the middle layers; the interesting divergence is at the end.

"""

# ── Path bootstrap: keep `from _shared import ...` working from expl/ ──
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────





