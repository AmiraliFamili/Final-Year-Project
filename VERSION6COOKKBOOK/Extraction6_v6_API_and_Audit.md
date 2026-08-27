# Extraction6 v6 — API, Architecture and Specification Audit

This document accompanies the detailed Word report. It records the implementation boundary, the function-level API, and the specification gaps that remain before the project can claim a completed classifier-probing study.

## Core conclusion

Extraction6 is the **representation extraction and experiment-control layer**. It is not the completed probing/evaluation layer.

## Main call graph

`run_model_matrix` → `resolve_experiment_parameters` → `_run_resolved_experiment` → `prepare_model` → `preflight_model` → `load_tokenizer` → `load_model` → `run_model_probe` → `extract_dataset` → `RuntimeReporter` + memmap outputs.

## Scientifically important outputs

- `run_manifest.json`: immutable experiment contract.
- `model_metadata.json`: model revision, architecture, loading and probe evidence.
- `hidden_states.npy`: pooled representation tensor `[sample, hidden_state, hidden_size]`.
- `completed.npy`: resumable completion mask.
- `extraction.json`: dataset provenance and performance summary.
- `runtime_events.jsonl`: forensic runtime timeline.
- `measurement_ledger.jsonl`: cross-run evidence ledger.

## Critical specification gaps

1. No classifier probes.
2. No label/split persistence.
3. No token-level representation store.
4. No accuracy/F1/confusion-matrix evaluation.
5. No ablation engine.
6. No statistical comparison across layers/models.

## Important engineering fixes still recommended

- Make first/last-token pooling attention-mask aware.
- Add activation-memory estimation.
- Prefer validated local snapshot reuse before network calls.
- Strengthen full-dataset provenance hashing.
- Avoid duplicate anomaly counts.
- Preserve runtime-state schema separately from final metadata.
- Record probe seeds and split manifests once probing is added.

## Research interpretation rule

A hidden state existing at layer `l` does **not** demonstrate that the model has learned emotion. The defensible evidence is layer-wise held-out probe performance under a frozen protocol, compared with controls and appropriate ablations.
