from __future__ import annotations

# ── Path bootstrap: keep `from _shared import ...` working from expl/ ──
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────

"""
Testing with Concept Activation Vectors

What TCAV is, in plain language. A "concept" is a human-meaningful idea — for your project, 
examples include first-person self-report (sentences starting with "I feel"), sarcasm markers, negation, or intensifiers. 
TCAV answers: does the model's internal representation at layer L contain a direction that corresponds to concept X? And if so, 
does that direction matter for the model's emotional judgement?

Why this matters for your argument — this is the key technique for the "emotion vs linguistics" question. 
Your README explicitly identifies this as the biggest threat to your findings. TCAV is the direct answer. 
You define a linguistic concept (say, "the sentence contains an intensifier word like very, extremely, really") and measure how much the probe's score depends on that concept's direction in the representation. 
If the dependence is high, the probe is partly reading a linguistic artifact. If it is low, the probe is reading something more abstract.

The 2026 context. A June 2026 paper introduces TCAVq, a pipeline that automatically discovers concepts by clustering the model's activations 
rather than requiring you to pre-define them. This is a significant advance: instead of guessing which concepts matter, you let the clustering tell you. 
If you have time, implement both the manual TCAV (where you define the concepts) and the automatic version (where clustering discovers them).
"""


"""
tcav.py — Testing with Concept Activation Vectors.

Thesis question answered:
    "Does the probe's emotional judgement depend on a direction in the
     representation that corresponds to a specific linguistic concept
     (e.g. first-person self-report, intensifiers, negation)?"

Method (Kim et al., 2018):
    1. Define a concept C by selecting two sets of samples: C+ (concept
       present) and C- (concept absent).
    2. For each layer L, train a linear SVM to separate the hidden states
       of C+ from C-. The SVM's weight vector is the Concept Activation
       Vector (CAV) for that layer.
    3. For each test sample with the target class c, compute the directional
       derivative of the probe's score for c along the CAV direction.
    4. TCAV score = fraction of samples with positive derivative.

Reads:  hidden_states/<slug>/<dataset>/hidden_states.npy
        probe/<slug>/<dataset>/<run_key>/models/<probe>/<layer>/repeat_0/probe.joblib
Writes: probe/<slug>/<dataset>/<run_key>/analysis/tcav_<probe>.parquet
        probe/<slug>/<dataset>/<run_key>/analysis/tcav_<probe>.json
"""


from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from _shared import analysis_dir_for_probe, atomic_json


# ─────────────────────────────────────────────────────────────────────────────
# Concept definitions.
#
# Each concept is a predicate over a string. You edit this dictionary to
# define the concepts relevant to your thesis. The keys are the concept
# names that appear in the output; the values are functions that take a
# string and return True if the concept is present.
# ─────────────────────────────────────────────────────────────────────────────

FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself"}
INTENSIFIERS = {"very", "extremely", "really", "so", "incredibly",
                "absolutely", "totally", "completely"}
NEGATION_WORDS = {"not", "no", "never", "none", "nobody", "nothing",
                  "nowhere", "neither", "nor"}


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


CONCEPTS = {
    "first_person_self_report": lambda t: bool(_tokens(t) & FIRST_PERSON_PRONOUNS),
    "intensifier":              lambda t: bool(_tokens(t) & INTENSIFIERS),
    "negation":                 lambda t: bool(_tokens(t) & NEGATION_WORDS),
    # Add more as needed. For example:
    # "sarcasm_marker":         lambda t: "🙄" in t or "lol" in t.lower(),
}


def tcav_score_per_layer(
    states: np.ndarray,                  # [N, L, D]
    concept_present_idx: np.ndarray,     # indices of samples where concept is present
    concept_absent_idx: np.ndarray,      # indices of samples where concept is absent
    target_class: int,
    y: np.ndarray,                       # [N] int labels
    probe,                               # fitted probe for ONE layer
    layer_index: int,
    *,
    seed: int = 42,
) -> dict:
    """
    Compute the TCAV score for ONE layer and ONE concept.

    Parameters
    ----------
    states : the full memory-mapped [N, L, D] array.
    concept_present_idx, concept_absent_idx : row indices into the dataset.
    target_class : the class whose probe score we are attributing.
    y : labels.
    probe : the fitted probe for this layer.
    layer_index : which layer's hidden states the probe reads.

    Returns
    -------
    dict with keys:
        tcav_score  : float in [0, 1] — fraction of class-c samples whose
                      probe score increases when we move along the CAV.
        mean_dd     : mean directional derivative.
        cav_norm    : L2 norm of the CAV (a sanity check — a near-zero
                      norm means the SVM failed to find a direction).
        n_present   : number of C+ samples.
        n_absent    : number of C- samples.
        n_target    : number of class-c samples used in the derivative.
    """
    # ── Step 1: gather the two concept sets. ──
    Xp = np.asarray(states[concept_present_idx, layer_index, :], dtype=np.float32)
    Xn = np.asarray(states[concept_absent_idx,  layer_index, :], dtype=np.float32)

    if len(Xp) < 5 or len(Xn) < 5:
        return {"tcav_score": float("nan"), "mean_dd": float("nan"),
                "cav_norm": 0.0, "n_present": int(len(Xp)),
                "n_absent": int(len(Xn)), "n_target": 0}

    # ── Step 2: fit the CAV. ──
    svm = LinearSVC(random_state=seed, max_iter=5000).fit(
        np.concatenate([Xp, Xn]),
        np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]),
    )
    cav = svm.coef_[0]                   # [D]
    cav_norm = float(np.linalg.norm(cav))

    # ── Step 3: directional derivative on class-c samples. ──
    target_idx = np.where(y == target_class)[0]
    Xt = np.asarray(states[target_idx, layer_index, :], dtype=np.float32)
    if len(Xt) == 0:
        return {"tcav_score": float("nan"), "mean_dd": float("nan"),
                "cav_norm": cav_norm, "n_present": int(len(Xp)),
                "n_absent": int(len(Xn)), "n_target": 0}

    eps = 1e-3
    p0 = probe.predict_proba(Xt)[:, target_class]
    p1 = probe.predict_proba(Xt + eps * cav)[:, target_class]
    dd = (p1 - p0) / eps

    return {
        "tcav_score": float((dd > 0).mean()),
        "mean_dd":    float(dd.mean()),
        "cav_norm":   cav_norm,
        "n_present":  int(len(Xp)),
        "n_absent":   int(len(Xn)),
        "n_target":   int(len(Xt)),
    }


def run_tcav(
    model_slug_: str,
    run_key_dir: Path,
    dataset_texts: list[str],
    labels: np.ndarray,
    *,
    target_classes: list[int] | None = None,
) -> Path:
    """
    Top-level entry point. Runs TCAV for every concept × every layer.

    target_classes : which emotion classes to run the derivative for.
        None means all classes. For a 7-class ISEAR run this is fine;
        for a 28-class GoEmotions run you may want to pick a subset.
    """
    import joblib
    import pandas as pd

    meta = json.loads((run_key_dir / "complete_run_metadata.json").read_text())
    probe_name = meta["configuration"]["probes"][0]["name"]
    best = pd.read_csv(run_key_dir / "final_probe_score_matrix.csv")
    best_row = best[best["probe"] == probe_name].iloc[0]

    if target_classes is None:
        target_classes = list(range(int(best_row["class_count"])))

    # Load the full states once.
    from _shared import load_hidden_states
    states = load_hidden_states(model_slug_, meta["artifact"]["dataset_name"])

    # For each concept, split the dataset.
    concept_masks = {
        name: np.array([pred(t) for t in dataset_texts], dtype=bool)
        for name, pred in CONCEPTS.items()
    }

    rows = []
    for c_name, mask in concept_masks.items():
        present_idx = np.where(mask)[0]
        absent_idx  = np.where(~mask)[0]
        if len(present_idx) < 5 or len(absent_idx) < 5:
            continue
        for layer in range(states.shape[1]):
            probe_path = (run_key_dir / "models" / probe_name
                          / f"layer_{layer}" / "repeat_0" / "probe.joblib")
            if not probe_path.exists():
                continue
            probe = joblib.load(probe_path)
            for c in target_classes:
                res = tcav_score_per_layer(
                    states, present_idx, absent_idx, c,
                    labels, probe, layer, seed=42,
                )
                res.update({"concept": c_name, "layer": layer, "target_class": c})
                rows.append(res)

    df = pd.DataFrame(rows)
    out_dir = analysis_dir_for_probe(run_key_dir)
    out_path = out_dir / f"tcav_{probe_name}.parquet"
    df.to_parquet(out_path, index=False)
    atomic_json(out_dir / f"tcav_{probe_name}.json", {
        "technique": "TCAV",
        "probe": probe_name,
        "concepts": list(CONCEPTS.keys()),
        "n_layers": int(states.shape[1]),
        "n_rows": len(df),
        "output": str(out_path),
    })
    return out_path