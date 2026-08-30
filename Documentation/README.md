# Hidden-State Probing for Emotion Recognition in Language Models

This repository houses a complete framework for extracting frozen transformer representations and systematically probing them for emotion-related information. It is built for scientific reproducibility and long-running experiment management.

---

## Project Map

| Path | Purpose |
|------|---------|
| `Analyser_.py` | Early hidden-state analyser, superseded by unified probe. |
| `Analyser_new.ipynb` | Notebook for advanced hidden-state analysis. |
| `Analyzer_Visualizer_.ipynb` | Visual dashboard for inspecting extracted states. |
| `Extraction6_new.py` | Deterministic extraction pipeline with resume and v2 unified format. |
| `Extraction6_display.ipynb` | Notebook front-end for running and monitoring extraction. |
| `Get_Go_Emo.py` | Loader for the GoEmotions dataset. |
| `Get_Isear.py` | Loader for the ISEAR dataset. |
| `GoEmotion.ipynb` | Exploration and preprocessing of GoEmotions. |
| `ISEAR.ipynb` | Exploration and preprocessing of ISEAR. |
| `Working_Analyser.ipynb` | Main analysis notebook for probe results. |
| `unified_hidden_state_probe_v4_2.py` | Core probe training, evaluation, controls, and checkpointing. |
| `unified_hidden_state_probe_v4_2_master.ipynb` | Master notebook driving the full probe matrix. |
| `good_broken_analyser.ipynb` | Debug notebook containing working and broken attempts. |
| `Documentation/` | Logbook, poster, presentation, specification, requirements. |
| `Go_Emotion_Google/` | Raw GoEmotions CSVs (train/validation/test). |
| `isear_dataset-master/isear.csv` | Raw ISEAR dataset. |
| `__pycache__/` | Compiled Python bytecode, safe to ignore. |

---

## TODO

### Phase 0 — Dataset

- [x] Confirm GoEmotions loader returns expected columns and label format.
- [x] Confirm ISEAR loader maps numeric labels to emotion names.
- [x] Inspect class balance and sample counts.
- [ ] Add dataset statistics to documentation.
- [ ] Cache cleaned datasets as Parquet for faster reload.

### Phase 1 — Extraction

- [x] Finalise deterministic extraction pipeline.
- [x] Implement memmap storage with checksums and flush control.
- [x] Add resume capability via experiment manifest validation.
- [x] Repair auxiliary files (`sample_ids`, `labels`, integrity hashes).
- [x] Run extraction across all models and datasets.
- [ ] Audit every model‑dataset pair for completeness.
- [ ] Generate a global extraction audit report.

### Phase 2 — Probing

- [x] Implement unified probe training (logistic + MLP).
- [x] Add shuffled‑label controls for chance estimation.
- [x] Add checkpoint/resume for matrix runs.
- [x] Fix per‑class metric calculation for multiclass targets.
- [x] Save complete run metadata (config, environment, results).
- [x] Execute full probe matrix over available pairs.
- [ ] Compare results against baselines and controls.
- [ ] Run permutation tests for statistical significance.

### Phase 3 — Analysis & Visualisation

- [x] Load checkpoint results with encoding fallback.
- [x] Generate best‑layer summary table.
- [x] Plot layer‑wise Macro‑F1 curves.
- [x] Plot heatmaps (probe × layer).
- [x] Plot confusion matrices for single‑label tasks.
- [x] Plot per‑class metrics heatmap.
- [x] Compare true vs shuffled‑label performance.
- [ ] Add model performance ranking bar chart.
- [ ] Add PCA/t‑SNE visualisation of hidden states.
- [ ] Compile final publication‑quality dashboard.

### Phase 4 — Documentation & Cleanup

- [x] Maintain logbook with key decisions.
- [x] Draft poster and presentation.
- [x] Write specification document.
- [ ] Update README with final results and usage.
- [ ] Remove legacy files and `__pycache__` from repository.
- [ ] Ensure `requirements.txt` is current.

---

## Notes

- Extraction is deterministic; batch size, dtype, pooling, and max length are fixed after initialisation.
- The unified v2 format stores sample IDs as strings, text hashes, and a global checksum.
- Probe runs are checkpointed per model‑dataset pair, allowing interruption and seamless resume.
- Shuffled‑label controls use identical data splits, providing a robust chance baseline.
- Some early extractions lack explicit model/dataset names; directory‑based fallbacks are used.
- Multi‑label confusion matrices are not saved; per‑class metrics serve as an alternative.
- ISEAR labels are numeric and must be mapped to emotion names before probing.
- Checkpoint CSVs may contain non‑UTF‑8 characters; a robust reader is used in analysis.

---

## Quick Start

```bash
# Install dependencies
pip install -r Documentation/requirements.txt

# Load datasets
python -c "from Get_Go_Emo import get_go; df = get_go(); print(df.shape)"
python -c "from Get_Isear import get_isr; df = get_isr(); print(df.shape)"

# Extract hidden states (if not already done)
jupyter notebook Extraction6_display.ipynb

# Run probe matrix
jupyter notebook unified_hidden_state_probe_v4_2_master.ipynb

# Analyse results
jupyter notebook Working_Analyser.ipynb

-------====-----=-=-=-----=-=-=======-----==--==--==-=-==-=--==---------=-=-=-=-=========


Ideal Logbook Logs and Books :- Logbook on hackmd -> date, cleaner, short descriptions of papers, errors, good githubs, explaining to yourself in the logbook... 

Github Culture :- regular commits (not necessarly pushing)

capture package requirement for project :- pip freeze > requirements.txt

-=-===--=-=-=-=-=-----=-=-=-----=-=-=======-----==--==--==-=-==-=--==---------=-=-=-=-=========