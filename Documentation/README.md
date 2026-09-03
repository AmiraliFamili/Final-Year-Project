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

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/emotion-probing.git
   cd emotion-probing