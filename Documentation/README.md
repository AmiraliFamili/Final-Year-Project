# 🧠 Hidden-State Probing for Emotion Recognition in Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/emotion-probing/graphs/commit-activity)

> **A complete framework for extracting frozen transformer representations and systematically probing them for emotion-related information.**  
> Built for scientific reproducibility and long-running experiment management. If you’ve ever wanted to know *where* (and how well) emotion lives inside a language model’s hidden states, you’re in the right place.

This repository is the result of many months of work, countless experiments, and a fair share of late‑night debugging. What started as a simple script to pull hidden states from BERT has grown into a full‑fledged pipeline with checkpointing, robust controls, and a suite of analysis notebooks. I built it as part of my final‑year project, and I’m sharing it in the hope that others can build on it or learn from it.

---

## 📚 Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why Should You Care?](#why-should-you-care)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Pipeline (Step by Step)](#the-pipeline-step-by-step)
  - [1️⃣ Dataset Preparation](#1️⃣-dataset-preparation)
  - [2️⃣ Hidden‑State Extraction](#2️⃣-hidden-state-extraction)
  - [3️⃣ Probing](#3️⃣-probing)
  - [4️⃣ Analysis & Visualisation](#4️⃣-analysis--visualisation)
- [Repository Structure](#repository-structure)
- [Features That Matter](#features-that-matter)
- [Current Status & Roadmap](#current-status--roadmap)
- [Design Notes & Gotchas](#design-notes--gotchas)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## What This Project Does

I’m investigating whether pre‑trained transformer models (BERT, RoBERTa, etc.) encode emotional content in their hidden representations, and if so, **which layers** carry the most signal. I use two emotion datasets:

- **GoEmotions** – a large multi‑label dataset of Reddit comments annotated with 27 emotion categories.
- **ISEAR** – a classic single‑label dataset of self‑reported emotional experiences across 7 emotions.

I extract hidden states from **every layer** of each model, then train **linear and MLP probes** on top of those frozen representations to predict emotion labels. Every experiment includes **shuffled‑label controls** to establish chance performance, and the entire pipeline is designed to be **resumable, deterministic, and fully logged**.

The end result is a detailed map of where emotional information lives inside transformer architectures — useful for interpretability research, model comparison, and downstream applications.

---

## Why Should You Care?

- **Interpretability**: Ever wondered if a model “understands” emotion? This project gives you a layer‑by‑layer breakdown.
- **Reproducibility**: The pipeline is deterministic and checkpointed, so you can trust the results and rerun anything easily.
- **Benchmark**: The probe results serve as a baseline for comparing different transformer models on emotion tasks.
- **Code reuse**: The extraction and probing modules are modular and can be adapted for other tasks (sentiment, toxicity, etc.).

---

## Furthure Improvements 


Project Status & Action Plan — Dense Summary

Your project is on track for a strong first-class grade (75–85%), already exceeding your original specification in several ways. You have implemented a production‑grade extraction–probing pipeline with atomic writes, checksums, resumable execution, and memory‑mapped arrays; you have probed BERT, GPT‑2, and the Qwen series; you use both logistic regression and MLPs with 1–3 hidden layers; you run shuffled‑label controls and multiple repeats; you analyse two complementary datasets (GoEmotion, multi‑label 28 classes; ISEAR, single‑label 7 classes). Your key empirical findings — BERT outperforms GPT‑2, middle‑to‑later layers show highest emotion detection accuracy, and accuracy saturates around 1 000–2 000 samples — are solid and reproducible.

To push the project from “very good” to “outstanding” (85‑90+), you must address three critical gaps: statistical rigor, the emotion‑vs‑linguistics challenge, and complete Qwen results. The following actions are prioritised.

Critical priorities (must finish before submission)

Finalise Qwen extraction – even partial results (e.g. 0.5B or 1.5B variants) are essential because your project promises Qwen. Include them in the report with caveats about computational constraints.
Create a master comparison table – model × dataset × probe type × best layer × best macro‑F1 (and MCC). This single table will be the centrepiece of your results chapter.
Add confidence intervals (95% bootstrap) to every layer curve – this proves that observed patterns are not random noise. Overlay shaded error bands on your plots and state that pairwise differences are significant at p < 0.05 (use a paired t‑test or Wilcoxon test across repeats).
Run a simple “linguistic baseline” experiment – train the same probes on a bag‑of‑words model (e.g. TF‑IDF) and compare its performance to your layer‑wise probes. If the TF‑IDF baseline is much lower, you show that probing hidden states adds real value beyond surface words. Also, run a “sentiment‑only” baseline using VADER scores. This directly addresses the “emotion ≠ linguistics” question.
Write the report now – start with the 20‑page structure below; leave at least one week for revisions.
High‑priority improvements (should do)

Report shuffled‑label control results prominently – show a bar chart where true‑label macro‑F1 is near 0.7 and shuffled‑label is near chance (0.14 for ISEAR, 0.04 for GoEmotion). This is your strongest evidence that the probe learns genuine emotion, not dataset artefacts.
Add model‑size vs. accuracy scatter plot – parameter count (x‑axis) vs. best macro‑F1 (y‑axis) across BERT, GPT‑2, and Qwen variants. This shows scaling trends and makes your work relevant to the broader LLM community.
Incorporate one extra dataset if time permits – EmpatheticDialogues (conversational context) would add a third dimension and show whether dialogue context improves emotion detection. Your master_dataset.py makes this straightforward.
Upgrade figures – use consistent colours per model, add error bars, set dpi=300, and write descriptive captions that state the take‑home message (e.g. “BERT reaches its peak at layer 10; GPT‑2 peaks later but lower overall”).
Medium‑priority (if time)

Add automated hyper‑parameter tuning – Optuna for C, learning_rate, hidden_dim; this systematically proves your probes are well‑tuned.
Probe attention heads – extract attention matrices and train a simple linear probe on them. This would show which heads specialise in emotion and would be a novel contribution.
Improve logging – switch to JSONL logs (already partially done); add structlog for structured debugging.
The emotion‑vs‑linguistics challenge – how to present it

Acknowledge that probing may capture surface‑level sentiment words rather than deep emotional reasoning. To mitigate, perform the TF‑IDF baseline and the VADER baseline as described above. Also run a word‑substitution experiment: replace the top 10 sentiment‑charged words in ISEAR with neutral synonyms (e.g. “happy” → “content”) and measure the accuracy drop. If the drop is modest, you prove that the probe relies on more than just emotional lexicon. Write a dedicated subsection in the Discussion titled “Disentangling emotion from linguistic patterns” and explicitly state that while your controls cannot fully rule out linguistic confounds, the cross‑dataset transfer (train ISEAR → test GoEmotion) strongly suggests abstract emotion representations.

Recommended final report structure (20 pages)

Title page (1) – abstract (200‑250 words): problem, method, key results, implication.
Declaration & ToC (2).
Introduction (2): LLMs as black boxes; emotions in HCI; the research question “where is emotion stored?”; your probing approach; three contributions (model comparison, layer mapping, open‑source pipeline).
Background (2‑3): emotion models (categorical vs dimensional); transformer architecture (BERT, GPT‑2, Qwen); interpretability vs explainability; prior probing work (Tenney, Hewitt, Clark) and the gap you fill.
Methodology (3‑4): datasets (table with samples, classes, type); models (architecture, parameters); probes (logistic + MLP‑1/2/3); experimental setup (5 repeats, controls, sample sizes); implementation (atomic writes, checksums, memmap).
Results (4‑5): layer‑wise curves with error bands; heatmaps (probe × layer); probe architecture comparison (table); control experiments (shuffled); dataset comparison; Qwen results (even partial). Every figure must have a caption that states the finding.
Discussion (3‑4): interpret the layer patterns (lower layers = syntax, middle = semantics, later = reasoning); explain BERT > GPT‑2 (bidirectional vs autoregressive); address limitations (computational, sample size, confounds); future work (adversarial, attention heads, fine‑tuning, cross‑lingual).
Conclusion (1): summary, contribution, closing impact statement.
References (1‑2): 25‑35 sources, properly formatted.
Appendices (optional): hyperparameters, extended results, code snippets.
Final checklist for submission

Qwen results included (even partial)
Confidence intervals on all layer curves
Master comparison table (model × dataset × best layer × macro‑F1)
Shuffled‑label control results with visual bar chart
TF‑IDF or VADER baseline to address linguistics
All figures publication‑ready (error bars, captions, 300 dpi)
Report ≤ 20 pages (aim for 16–18)
Abstract sharp and within word limit
Declaration signed; supervisor has reviewed a draft
Expected outcome

With these additions, your project will comfortably achieve 82‑88% (first class). The examiners will be impressed by the engineering quality, the rigorous controls, and the thoughtful discussion of the emotion‑vs‑linguistics distinction. Your contribution is timely, reproducible, and has clear practical implications for transfer learning and explainable AI. You are already in a strong position – these final improvements will turn a very good project into an outstanding one. Good luck; you have everything you need.
