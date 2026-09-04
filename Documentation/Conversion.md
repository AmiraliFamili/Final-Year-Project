# Three-Machine Offline Research Deployment Architecture

## Python 3.8 • Windows 7 Legacy Target • Kali CPU Workstation • macOS Master

> **Purpose:** establish one scientifically controlled research project across three physically different computers, with explicit separation between **scientific authority**, **computational execution**, and **legacy compatibility**.

**Project context:** layer-wise emotion probing in transformer models, using local models/datasets and reproducible hidden-state extraction.

---

## 0. The Core Idea

This is **not three independent installations**.

It is one research system with three machine roles:

```text
                         ┌──────────────────────────────┐
                         │ MACHINE 1 — macOS MASTER     │
                         │ Development / authority      │
                         │ Internet-enabled             │
                         │                              │
                         │ • source of truth            │
                         │ • configs                    │
                         │ • dependency planning       │
                         │ • model acquisition         │
                         │ • release construction      │
                         └──────────────┬───────────────┘
                                        │
                              controlled release
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ USB RELEASE PACK  │
                              │ transport layer   │
                              │                   │
                              │ source + wheels   │
                              │ models + datasets │
                              │ manifests         │
                              │ checksums         │
                              └───────┬───────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                     │
                   ▼                                     ▼
       ┌─────────────────────────┐          ┌─────────────────────────┐
       │ MACHINE 2 — ASUS / KALI │          │ MACHINE 3 — SONY VAIO  │
       │ Modern CPU workstation  │          │ Windows 7 legacy target│
       │ Intel Core i5           │          │ Core 2 Duo / 4 GB RAM   │
       │ No NVIDIA GPU           │          │ ATI Mobility Radeon     │
       │                         │          │ CPU-only execution      │
       │ • main CPU execution    │          │ • compatibility target  │
       │ • full research trials  │          │ • micro-experiments     │
       │ • cross-checking        │          │ • offline validation    │
       └─────────────────────────┘          └─────────────────────────┘
```

### Non-negotiable principle

> **The operating system may change. The scientific experiment must not silently change.**

The macOS environment is the **master research environment**. The Kali machine is the **modern CPU execution/validation environment**. The Windows 7 machine is the **legacy execution target**. The USB is a **controlled release artifact**, not another development environment.

---

# 1. Machine Responsibility Matrix

| Concern | macOS Master | Asus / Kali Linux | Sony VAIO / Windows 7 |
|---|---|---|---|
| Internet access | **Yes** | Optional / controlled | **No** |
| Scientific authority | **YES — canonical** | Validation/replication | Legacy deployment only |
| Source code editing | **Primary** | Permitted for controlled fixes | Minimal |
| Model downloading | **Primary** | Allowed when required | **Never required** |
| Dataset downloading | **Primary** | Local copy only | **Never required** |
| Dependency resolution | **Primary** | Linux-specific | Prebuilt Windows wheels only |
| Windows wheel construction | **YES** | No | No |
| Full modern model suite | **YES** | **YES, subject to CPU/RAM** | **NO** |
| GPU | Mac hardware-dependent | **No NVIDIA GPU** | No CUDA |
| Large-model experimentation | Preferred where feasible | Preferred over VAIO | Infeasible/unsupported by design |
| Hidden-state extraction | Master/reference | Main CPU execution | Small-model legacy execution |
| Probe analysis | Master/reference | Main execution | Lightweight/subset only |
| Final scientific configuration | **Authority** | Must match | Must match |
| Runtime-specific configuration | Reference | Linux-specific | Windows-specific |
| Internet-free operation | Optional | Optional | **Required** |
| Release manifest | Creates | Verifies | Verifies |
| SHA-256 verification | Creates | Verifies | Verifies where tooling permits |
| VS Code | Modern version as appropriate | Modern version as appropriate | **1.70.3 portable ZIP** |
| Python | Research/master version | Linux project environment | **3.8.10** |
| Package source | Internet/cache | Local/online as controlled | **USB wheel repository only** |

---

# 2. Scientific Authority vs Deployment Configuration

The project should be thought of as two layers.

## Layer A — Scientific configuration

These values describe the experiment itself and should remain synchronized on all machines:

```text
Project / experiment ID
Model ID + exact revision
Tokenizer + tokenizer revision
Dataset identity + version/split
Dataset preprocessing
Text-column definition
Label mapping / class order
Train / validation / test split
Random seed
Maximum sequence length
Pooling method
Layer selection
Probe family
Probe architecture
Probe hyperparameters
Evaluation metrics
Shuffle-control policy
Analysis configuration
Output naming convention
```

## Layer B — Deployment configuration

These values are allowed to differ by machine:

```text
Operating system
Python installation path
Virtual-environment path
Device = CPU / MPS / CUDA
CPU thread count
Batch size, where the scientific protocol explicitly permits a deployment setting
Storage root
HF cache location
VS Code version
Executable paths
Path separators
Windows/Linux/macOS shell commands
```

### Important distinction

A deployment setting must **never silently alter a scientific variable**.

For example:

```text
Allowed:
Windows 7 → batch_size=1 because the machine cannot safely sustain a larger frozen setting

Not allowed:
Windows 7 → change pooling from first_token to mean because it seems faster
```

The second case is a different experiment and therefore requires a new experiment identity/configuration.

---

# 3. Machine 1 — macOS MASTER

## Mission

The Mac is the **source of truth and release factory**.

It owns the canonical project state and performs all internet-dependent preparation:

```text
Research design
    ↓
source code
    ↓
scientific configuration
    ↓
model acquisition
    ↓
dataset acquisition
    ↓
dependency resolution
    ↓
Windows wheel construction
    ↓
release validation
    ↓
USB package
```

Your existing Mac workflow has used multiple Python installations (including pyenv and system/framework Python). Therefore the first rule is to **make the master interpreter explicit for each task** rather than relying on whichever `python` happens to be first on PATH.

### Master environment philosophy

- Keep the master research environment separate from the Windows compatibility environment.
- Do **not** export a macOS `.venv` and copy it to Windows.
- Do **not** treat the host Python version used for Mac development as the Windows target Python.
- Build Windows dependencies against the Windows ABI/platform explicitly.
- Keep the canonical project source and scientific configuration here.

---

## 3.1 Master directory

A recommended research layout is:

```text
Final-Year-Project/
├── src/
│   ├── Extraction*.py
│   ├── unified_hidden_state_probe*.py
│   ├── hidden_state_bundle.py
│   └── ...
├── notebooks/
├── configs/
├── datasets/
├── models/
├── reports/
├── tests/
├── docs/
└── releases/
```

The deployment package is built separately:

```text
releases/
└── PROJECT_USB/
    ├── 00_INSTALLERS/
    ├── 01_WHEELS/
    ├── 02_PROJECT/
    ├── 03_DATASETS/
    ├── 04_MODELS/
    ├── 05_CONFIG/
    ├── 06_TESTS/
    └── 07_MANIFEST/
```

---

## 3.2 Mac — dependency engineering

The Windows target is Python **3.8.10**, so the Windows dependency set must be solved independently of the Mac environment.

### Candidate Windows baseline

```text
Python           3.8.10
NumPy            1.24.4
pandas           2.0.3
SciPy            1.10.1
scikit-learn     1.3.2
Matplotlib       3.7.5
Pillow           10.4.0
PyTorch          2.2.2       [CPU target; verify on Windows 7]
Transformers     4.45.2
Accelerate       0.34.0
datasets         2.20.0
PyArrow          15.0.2
joblib           1.4.2
psutil           6.1.1
tqdm             4.67.1
safetensors      0.4.5
tokenizers       0.19.1
sentencepiece    0.2.0
huggingface-hub  0.24.6
filelock         3.16.1
packaging        24.2
PyYAML           6.0.2
requests         2.32.3
regex            2024.11.6
typing_extensions 4.12.2
sympy            1.13.3
fsspec           2024.9.0
dill             0.3.8
multiprocess     0.70.16
xxhash           3.5.0
networkx         3.1        [only if actually required]
```

### Compatibility corrections to preserve

Do **not** blindly carry modern package pins into Python 3.8:

```text
numpy 1.26.x       → NOT the Python 3.8 baseline
networkx 3.4.x     → NOT the Python 3.8 baseline
```

The candidate replacements are:

```text
numpy     → 1.24.4
networkx  → 3.1, only if required
```

The final set must still be tested against the project's actual imports and dependency closure.

---

## 3.3 Mac — construct the Windows wheel repository

Create:

```bash
mkdir -p PROJECT_USB/01_WHEELS/WINDOWS7_X64
cd PROJECT_USB/01_WHEELS/WINDOWS7_X64
```

Use an explicit Python 3.8 interpreter when possible:

```bash
python3.8 -m pip download \
  --only-binary=:all: \
  --platform win_amd64 \
  --python-version 38 \
  --implementation cp \
  --abi cp38 \
  numpy==1.24.4 \
  pandas==2.0.3 \
  scipy==1.10.1 \
  scikit-learn==1.3.2 \
  matplotlib==3.7.5 \
  pillow==10.4.0 \
  torch==2.2.2 \
  transformers==4.45.2 \
  accelerate==0.34.0 \
  datasets==2.20.0 \
  pyarrow==15.0.2 \
  joblib==1.4.2 \
  psutil==6.1.1 \
  tqdm==4.67.1 \
  safetensors==0.4.5 \
  tokenizers==0.19.1 \
  sentencepiece==0.2.0 \
  huggingface-hub==0.24.6 \
  filelock==3.16.1 \
  packaging==24.2 \
  PyYAML==6.0.2 \
  requests==2.32.3 \
  regex==2024.11.6 \
  typing_extensions==4.12.2 \
  sympy==1.13.3 \
  fsspec==2024.9.0 \
  dill==0.3.8 \
  multiprocess==0.70.16 \
  xxhash==3.5.0 \
  networkx==3.1 \
  -d .
```

### What the flags enforce

| Flag | Purpose |
|---|---|
| `--only-binary=:all:` | Avoid source builds on Windows 7 |
| `--platform win_amd64` | Target 64-bit Windows |
| `--python-version 38` | Target CPython 3.8 |
| `--implementation cp` | Target CPython |
| `--abi cp38` | Target Python 3.8 ABI |
| `-d .` | Place wheels in the controlled repository |

### Important

A successful `pip download` means **pip found compatible artifacts according to package metadata**. It does not prove that the complete software stack will execute successfully on Windows 7.

The Windows machine therefore remains the final compatibility judge.

---

## 3.4 Mac — acquire the CPU PyTorch wheel carefully

The target has no CUDA-capable NVIDIA GPU, so the Windows release should contain a CPU-compatible PyTorch distribution.

A separately acquired CPU wheel may be stored in:

```text
01_WHEELS/WINDOWS7_X64/
```

Example filename used by the deployment plan:

```text
torch-2.2.2+cpu-cp38-cp38-win_amd64.whl
```

**Do not assume the filename or OS compatibility merely because a file exists.** Validate the exact wheel on the Windows target before treating PyTorch as supported.

---

## 3.5 Mac — acquire local models and datasets

The Mac is the best place to assemble all immutable research inputs:

```text
PROJECT_USB/
├── 02_PROJECT/
├── 03_DATASETS/
└── 04_MODELS/
```

### Dataset policy

The project currently centers on:

```text
GoEmotions
ISEAR
```

The deployment package must include the exact dataset snapshot used by the scientific experiment.

### Model policy

The current project registry contains **25 primary model entries**, organized conceptually into:

```text
01_encoders
02_early_decoders
03_tiny_modern
04_qwen_scaling
05_independent_small
06_stretch
```

The registry includes models such as BERT-family encoders, GPT-2/GPT-Neo/OPT, SmolLM2, Qwen, and several ~1B–4B models.

For Windows 7, the registry is **not the same thing as the runnable set**. The VAIO must use a conservative subset based on actual RAM feasibility.

---

## 3.6 Mac — validate source before release

Before copying the project to USB, verify that the package contains the authoritative code used by the current experiment.

Particularly important files include the project's extraction, probing, dataset, and analysis modules, for example:

```text
Extraction*.py
unified_hidden_state_probe*.py
hidden_state_bundle.py
hidden_state_report.py
Get_Go_Emo.py
Get_Isear.py
probing_pipeline.py
```

Use the **current validated source**, not an obsolete notebook cell or an older experimental script.

---

# 4. Machine 2 — Asus / Kali Linux

## Mission

The Asus is the **modern CPU research workstation**.

It is the bridge between the highly flexible Mac environment and the extremely constrained Windows 7 target.

Known machine constraints:

```text
OS:      Kali Linux
CPU:     Intel Core i5
GPU:     No NVIDIA GPU
Mode:    CPU execution
```

Unlike the VAIO, this machine should be treated as a serious execution platform rather than merely a compatibility demonstration.

---

## 4.1 Kali responsibilities

### Primary

- Run the research pipeline on CPU.
- Execute hidden-state extraction for feasible models.
- Run the unified probing system.
- Verify dataset loading and label semantics.
- Cross-check Mac results.
- Benchmark the practical cost of CPU-only inference.
- Validate that the scientific configuration is not OS-dependent.

### Secondary

- Act as a recovery machine if the Mac environment is unavailable.
- Validate source releases before they are sent to Windows 7.
- Run larger experiments that are unrealistic on the VAIO.

### Not its responsibility

The Kali machine should **not silently redefine the experiment** merely because it is faster or slower.

---

## 4.2 Kali environment

Create an isolated environment for the project rather than mixing dependencies with the entire operating system:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
which python
```

Use a project-specific dependency specification and record:

```bash
python -m pip freeze > 07_MANIFEST/kali-pip-freeze.txt
```

The exact Python version can differ from Windows because Linux is not the Windows deployment target, but the **scientific configuration must remain identical**.

---

## 4.3 Kali — device policy

There is no NVIDIA GPU, so the intended baseline is:

```python
device = "cpu"
```

Do not insert CUDA-specific assumptions into the project.

A hardware-neutral implementation should use the project's device-selection layer rather than hard-coded machine-specific logic.

---

## 4.4 Kali — model execution strategy

The modern model registry is broader than the Windows 7 runtime envelope.

Recommended grouping:

| Registry group | Kali role |
|---|---|
| `01_encoders` | Main experiments where resources permit |
| `02_early_decoders` | Main experiments |
| `03_tiny_modern` | Main experiments |
| `04_qwen_scaling` | Primary scaling study |
| `05_independent_small` | Comparative experiments |
| `06_stretch` | Attempt selectively after preflight |

The extraction code already contains model metadata, parameter counts, architecture information and batch hints. Those should drive **preflight decisions**, not ad-hoc changes during a run.

---

# 5. Machine 3 — Sony VAIO / Windows 7

## Mission

The Sony VAIO is a **legacy execution target**, not the master research computer.

Known constraints:

```text
OS:        Windows 7 SP1, 64-bit
CPU:       Intel Core 2 Duo
Cores:     2
RAM:       4 GB
GPU:       ATI Mobility Radeon HD 4650
CUDA:      No
Execution: CPU-only
```

This machine should be treated as a constrained compatibility environment.

### The correct objective

Not:

> "Make the whole modern AI stack run on the VAIO."

Instead:

> "Demonstrate that a controlled, reproducible subset of the scientific pipeline can execute offline under a fixed Windows 7 runtime."

---

# 6. Windows 7 — What Must Be Frozen

The following deployment choices should be treated as fixed unless a new compatibility release is deliberately created:

```text
Python       3.8.10
Architecture x86-64
VS Code      1.70.3
Runtime      CPU-only
Precision    float32 baseline
Network      offline
Package      local wheel repository
Model        local files only
Dataset      local files only
```

### Why float32?

On the Core 2 Duo, do **not** automatically assume that half precision is an advantage. Keep the baseline deterministic and conservative.

Use:

```text
use_half_precision = False
```

unless controlled testing proves otherwise and the change is recorded as an explicit experimental/deployment change.

---

# 7. Build the USB Release Package on the Mac

Use this exact conceptual release:

```text
PROJECT_USB/
├── 00_INSTALLERS/
│   ├── python/
│   │   └── python-3.8.10-amd64.exe
│   ├── vscode/
│   │   └── VSCode-win32-x64-1.70.3.zip
│   └── vc_redist/
│       └── vc_redist.x64.exe
│
├── 01_WHEELS/
│   └── WINDOWS7_X64/
│       └── *.whl
│
├── 02_PROJECT/
│   ├── source files
│   ├── notebooks
│   ├── configs
│   └── documentation
│
├── 03_DATASETS/
│   ├── GoEmotions/
│   └── ISEAR/
│
├── 04_MODELS/
│   ├── bert-base-uncased/
│   ├── distilbert/
│   ├── roberta-base/
│   ├── gpt2/
│   ├── gpt-neo-125m/
│   ├── SmolLM2-135M/
│   └── ...
│
├── 05_CONFIG/
│   ├── scientific_config.json
│   ├── model_registry.json
│   └── deployment_windows7.json
│
├── 06_TESTS/
│   ├── test_environment.py
│   ├── test_torch.py
│   ├── test_local_model.py
│   └── test_offline.py
│
└── 07_MANIFEST/
    ├── WINDOWS7_MANIFEST.txt
    ├── KALI_MANIFEST.txt
    ├── MAC_MASTER_MANIFEST.txt
    ├── pip-freeze.txt
    └── CHECKSUMS.sha256
```

---

# 8. Windows 7 — Install in a Controlled Sequence

## Stage 1 — Install Python

Run:

```text
00_INSTALLERS\python\python-3.8.10-amd64.exe
```

Prefer a dedicated installation such as:

```text
C:\Research\Python38
```

Verify:

```cmd
python --version
python -m pip --version
```

Expected:

```text
Python 3.8.10
```

---

## Stage 2 — Create the virtual environment

Copy the project from USB to local disk, for example:

```text
C:\Research\project
```

Then:

```cmd
cd C:\Research\project
python -m venv .venv
.venv\Scripts\activate
```

Verify:

```cmd
python --version
where python
```

The Python path should point to:

```text
.venv\Scripts\python.exe
```

---

## Stage 3 — Do not upgrade pip automatically

Do **not** run:

```cmd
pip install --upgrade pip
```

The Windows environment should remain controlled by the release package.

---

## Stage 4 — Install from wheels only

Use the USB repository:

```cmd
python -m pip install ^
--no-index ^
--find-links=..\01_WHEELS\WINDOWS7_X64 ^
numpy==1.24.4 ^
pandas==2.0.3 ^
scipy==1.10.1 ^
scikit-learn==1.3.2 ^
matplotlib==3.7.5 ^
pillow==10.4.0 ^
transformers==4.45.2 ^
accelerate==0.34.0 ^
datasets==2.20.0 ^
pyarrow==15.0.2 ^
joblib==1.4.2 ^
psutil==6.1.1 ^
tqdm==4.67.1 ^
safetensors==0.4.5 ^
tokenizers==0.19.1 ^
sentencepiece==0.2.0 ^
huggingface-hub==0.24.6 ^
filelock==3.16.1 ^
packaging==24.2 ^
PyYAML==6.0.2 ^
requests==2.32.3 ^
regex==2024.11.6 ^
typing_extensions==4.12.2 ^
sympy==1.13.3 ^
fsspec==2024.9.0 ^
dill==0.3.8 ^
multiprocess==0.70.16 ^
xxhash==3.5.0
```

Install the exact PyTorch wheel separately when required:

```cmd
python -m pip install --no-index C:\Research\01_WHEELS\WINDOWS7_X64\torch-2.2.2+cpu-cp38-cp38-win_amd64.whl
```

Adjust the path and filename to the actual release artifact.

Remove `networkx==3.1` unless the project actually imports NetworkX.

---

# 9. Windows 7 — Make the Project Truly Offline

Before executing the scientific code:

```cmd
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

Then use **explicit local model paths**:

```python
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "04_MODELS"

model_path = MODEL_ROOT / "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    model_path,
    local_files_only=True,
)
```

### Offline rule

Do not rely only on environment variables. The code should be designed so that a missing local resource causes an explicit failure rather than an attempted network lookup.

---

# 10. Windows 7 — Portable Paths

Never hard-code the USB drive letter.

Use:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "03_DATASETS"
MODEL_ROOT = PROJECT_ROOT / "04_MODELS"
CONFIG_ROOT = PROJECT_ROOT / "05_CONFIG"
```

This allows the same project structure to run from:

```text
C:\Research\project
D:\Research\project
E:\Research\project
USB:\PROJECT_USB\02_PROJECT
```

without embedding one drive letter into scientific code.

---

# 11. Windows 7 — Install Visual Studio Code

The deployment target uses the final VS Code release intended for Windows 7:

```text
VS Code 1.70.3
```

Copy:

```text
VSCode-win32-x64-1.70.3.zip
```

to:

```text
C:\Research\VSCode
```

Extract and run:

```text
Code.exe
```

Open the project directory rather than editing files directly on the USB.

---

# 12. Windows 7 — Python Interpreter in VS Code

Select:

```text
C:\Research\project\.venv\Scripts\python.exe
```

Then verify in the integrated terminal:

```cmd
python --version
where python
```

The terminal should show the `.venv` environment as active.

---

# 13. Windows 7 — Extensions

Extensions are optional deployment components.

If required, acquire compatible `.vsix` files on the Mac and copy them into the USB release.

The minimum useful Python tooling is typically:

```text
ms-python.python
```

Pylance may be added when a compatible version for VS Code 1.70.3 is verified.

Offline installation:

```cmd
code --install-extension ms-python.python-XXXX.X.X.vsix
```

Do not assume that the latest extension is compatible with the old editor.

---

# 14. The Scientific Pipeline Across the Three Machines

```text
                     SCIENTIFIC MASTER
                       macOS / Master
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
             SOURCE        DATASETS       MODELS
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                     SCIENTIFIC CONFIG
                              │
                              ▼
                       RELEASE PACKAGE
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              KALI / CPU         WINDOWS 7 / CPU
                    │                   │
           full/modern runs       constrained runs
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     comparable outputs
                              │
                              ▼
                     analysis / thesis
```

The scientific flow should therefore be:

```text
Dataset
   ↓
preprocessing
   ↓
tokenization with the model's tokenizer
   ↓
model forward passes
   ↓
layer-wise hidden-state extraction
   ↓
verified hidden-state artifact
   ↓
probe validation
   ↓
layer-wise probing
   ↓
metrics + controls
   ↓
scientific interpretation
```

The deployment machine must not change the meaning of any of these stages.

---

# 15. Hidden-State Extraction — Reproducibility Contract

The current project architecture has already moved toward deterministic extraction, resumable storage, explicit provenance, and forensic validation.

The extraction layer should preserve:

```text
experiment_id
hyperparameter_hash
model ID
model revision
model metadata
dataset identity
dataset fingerprint
text provenance
label provenance
sample count
layer count
hidden size
pooling
max sequence length
batch configuration
device
precision
torch version
transformers version
system fingerprint
```

The output should remain compatible with the existing artifact layout, including structures such as:

```text
data/
hidden_states.npy
completed.npy
metadata/
bundle_manifest.json
```

where applicable.

---

# 16. Why the Hidden-State Artifact Matters

The extracted representation is an **intermediate scientific artifact**, not the final conclusion.

A representation containing information about emotion is only a hypothesis until a properly controlled probe demonstrates recoverability.

Therefore:

```text
Extraction ≠ proof of emotion learning

Extraction → supplies controlled representations
Probe      → tests recoverability
Controls   → tests selectivity / robustness
Metrics    → quantify evidence
```

This distinction should remain visible throughout all three environments.

---

# 17. Provenance and Label Integrity

The project has previously exposed several important failure modes that must be prevented in the three-machine deployment.

## 17.1 Text fingerprint contract

The extraction and probing stages must use the **same text provenance contract**.

Do not allow one component to use a native dataset fingerprint while another silently computes a different fingerprint definition.

## 17.2 ISEAR labels

The project has encountered a semantic mismatch where ISEAR labels were represented as numeric IDs while a probe canonicalizer expected emotion strings.

The release must therefore define:

```text
raw label column
label resolution method
canonical class names
class order
numeric mapping
label fingerprint
```

## 17.3 Sample identity

Text-only IDs are unsafe when duplicate texts exist.

Use unique sample identities with duplicate-occurrence suffixes or an equivalent deterministic row identity.

## 17.4 Train/test leakage

Best-layer selection must not use the test set.

Correct structure:

```text
TRAIN      → fit probe
VALIDATION → select model/layer/hyperparameters
TEST       → final reporting
```

The test set is not a tuning set.

---

# 18. Probe Stage — Machine-Neutral Scientific Rules

The unified probe system should verify before training:

```text
hidden-state integrity
representation shape
finite values
sample count
text alignment
label alignment
class range
class frequencies
provenance hashes
```

The intended representation contract includes support for forms such as:

```text
[N, D]       → single representation / one-layer interpretation
[N, L, D]    → layer-wise representation tensor
```

The project has also evolved toward:

```text
linear probes
MLP probes
shuffled-label controls
silhouette / geometry diagnostics
PCA diagnostics
multiple evaluation metrics
layer-wise scorecards
```

These are scientific analysis components and should remain consistent across Mac, Kali and Windows.

---

# 19. Experiment Identity — The Firewall Against Silent Drift

Every serious extraction should have a unique experimental identity.

Example:

```text
experiment_id:
baseline_v5_001
```

and an associated hash such as:

```text
hyperparameter_hash:
e01b20ed0608f735
```

The exact current values should come from the authoritative experiment manifest rather than being typed manually into deployment notes.

### Deterministic rule

Once an experiment already exists:

```text
experiment ID
        ↓
existing manifest
        ↓
hyperparameter hash
        ↓
configuration becomes authoritative
```

A request to change batch size, pooling, max length, precision, or another frozen hyperparameter should create a deliberate new configuration/experiment identity rather than mutating the old run.

---

# 20. Resume Policy

The extraction system is intentionally resumable.

A hidden-state artifact may contain:

```text
hidden_states.npy
completed.npy
metadata/
```

The completion map identifies which samples/batches have been successfully written.

The resume system should validate before continuing:

```text
expected shape
expected dtype
sample count
experiment identity
hyperparameter hash
model identity
dataset fingerprint
```

Never resume an artifact simply because its filename looks correct.

---

# 21. Storage Architecture

### Mac

Large hidden-state storage may live on the external research volume, e.g. a path conceptually equivalent to:

```text
/Volumes/Amirali/hidden_states
```

### Kali

Use a Linux-native high-capacity storage root.

### Windows 7

Use a local disk location rather than permanently executing large extraction jobs directly from USB.

A portable release may contain the software and inputs, but active outputs should normally be written to:

```text
C:\Research\outputs
```

This reduces USB wear and avoids performance bottlenecks.

---

# 22. Windows 7 — Memory Strategy

For the Core 2 Duo / 4 GB RAM target:

```text
Start with:

batch_size = 1
```

Benchmark before considering an increase.

Do not automatically enable dynamic batch-size tuning if deterministic reproducibility is the goal.

A deployment performance setting belongs in the runtime manifest.

A scientific configuration change belongs in a new experiment.

---

# 23. Model Feasibility on the VAIO

Use the following operational categories:

| Category | Meaning on Windows 7 |
|---|---|
| **SAFE** | Small models with realistic RAM requirements |
| **POSSIBLE** | May execute with very low batch size and short sequences |
| **MEMORY-CONSTRAINED** | May load but leave insufficient memory for stable execution |
| **INFEASIBLE** | Not a reasonable target for 4 GB RAM |

### Practical first targets

The safest initial targets are the smallest encoders/decoders, for example:

```text
BERT-base-scale experiments only if memory permits
DistilBERT
GPT-2
GPT-Neo-125M
SmolLM2-135M
other genuinely small models from the registry
```

Avoid making 7B-class models a Windows goal. Their presence in the scientific model registry does not imply that they belong in the VAIO deployment.

---

# 24. Preflight Before Every Large Model

The current extractor architecture uses model metadata, parameter counts, batch hints and environment reporting.

On every target machine, perform a preflight that answers:

```text
How much RAM exists?
How much RAM is currently free?
How large is the model?
How large will all-layer hidden-state activations be?
What batch size is frozen?
What sequence length is frozen?
Can the model be loaded safely?
```

A model should be skipped rather than crash the whole run when preflight indicates that the target cannot safely execute it.

---

# 25. The 25-Model Registry vs Deployment Subsets

The scientific registry is a **research catalogue**.

The runtime subset is a **machine-specific feasibility decision**.

```text
                 25-model scientific registry
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
             macOS           Kali          Windows 7
             broad           broad          conservative
             where           CPU           small only
             feasible        feasible
```

This prevents the common mistake of deleting models from the scientific definition simply because one old machine cannot execute them.

---

# 26. Verification Ladder — All Machines

Never jump directly from installation to a full experiment.

Use this ladder:

```text
01. OS / architecture identified
02. Python identified
03. virtual environment active
04. dependency repository verified
05. core imports succeed
06. PyTorch tensor arithmetic succeeds
07. CPU matrix multiplication succeeds
08. Transformers imports
09. local tokenizer loads
10. local model loads
11. local dataset loads
12. labels verified
13. one forward pass succeeds
14. hidden-state shape verified
15. hidden-state file reopens
16. metadata validates
17. probe artifact loads
18. micro-experiment succeeds
19. controlled small run succeeds
20. larger experiment begins
```

A failure at a lower stage invalidates confidence in every higher stage.

---

# 27. Test 1 — Core Environment

Create:

```text
06_TESTS/test_environment.py
```

Example:

```python
import numpy
import pandas
import scipy
import sklearn
import matplotlib
import torch
import transformers
import datasets
import safetensors

print("Environment OK")
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
```

Run:

```cmd
python 06_TESTS\test_environment.py
```

---

# 28. Test 2 — PyTorch Without a Model

```cmd
python -c "import torch; x=torch.tensor([1.0,2.0,3.0]); print(x); print(x*x)"
```

Then:

```cmd
python -c "import torch; x=torch.randn(100,100); print(torch.mm(x,x).shape)"
```

Then:

```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.device('cpu'))"
```

Expected CPU target state:

```text
CUDA available: False
Device: cpu
```

---

# 29. Test 3 — Local Transformer Model

Do not use an online model ID for the Windows validation test.

Example:

```cmd
python -c "from transformers import AutoTokenizer, AutoModel; import torch; model_path='04_MODELS/gpt2'; tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True); model=AutoModel.from_pretrained(model_path, local_files_only=True); inputs=tokenizer('Hello world', return_tensors='pt'); outputs=model(**inputs); print(outputs.last_hidden_state.shape)"
```

This validates:

```text
Python
→ PyTorch
→ Transformers
→ local tokenizer
→ local model files
→ tokenization
→ CPU forward pass
→ hidden state output
```

---

# 30. Test 4 — Offline Failure Test

Disconnect the Windows machine from the network.

Then run the exact local model test again.

A successful result demonstrates that the software does not depend on a hidden network lookup.

If a network attempt occurs, investigate:

```text
missing model file
missing tokenizer file
online from_pretrained() call
missing dataset asset
unexpected Hub/cache lookup
extension/update process
```

Do not reconnect and simply retry. Fix the dependency gap.

---

# 31. Test 5 — Micro-Experiment

Do not start with the complete GoEmotions/ISEAR dataset.

First run:

```text
1 small model
1 dataset
very small sample count
1–few batches
```

Verify:

```text
model loads
↓
tokenizer loads
↓
dataset loads
↓
labels are correct
↓
forward pass succeeds
↓
hidden states are produced
↓
completion map is updated
↓
metadata is written
↓
artifact can be reopened
```

Only then begin the full extraction.

---

# 32. Mac ↔ Kali ↔ Windows Scientific Cross-Check

Before accepting Windows results, compare a small identical experiment across platforms.

Use the same:

```text
model ID
model revision
tokenizer
sample subset
text order
label mapping
random seed
max length
pooling
layer definition
output dtype
probe configuration
```

Expected differences:

```text
runtime speed
CPU thread behavior
filesystem path
Python executable path
operating-system metadata
```

Unexpected differences:

```text
sample order
label IDs
hidden-state shape
model revision
pooling
sequence length
probe configuration
```

Unexpected scientific differences should stop the validation process.

---

# 33. Manifest Architecture

Every machine should write a manifest.

## macOS master

```text
MAC_MASTER_MANIFEST.txt
```

Capture:

```text
OS
Python
main project environment
project release ID
Git/source revision
model inventory
dataset inventory
scientific configuration hash
```

## Kali

```text
KALI_MANIFEST.txt
```

Capture:

```text
OS
kernel
CPU
RAM
Python
package versions
project release ID
scientific configuration hash
execution device
thread configuration
```

## Windows 7

```text
WINDOWS7_MANIFEST.txt
```

Capture:

```text
Operating System: Windows 7 SP1
Architecture: x86-64
Python: 3.8.10
PyTorch: [actual installed version]
Transformers: [actual installed version]
NumPy: [actual installed version]
Project release: [release ID]
Model files: [list]
Dataset files: [list]
CPU: Intel Core 2 Duo
RAM: 4 GB
Execution: CPU-only
Offline: YES
```

Also record:

```cmd
python -m pip freeze
```

---

# 34. SHA-256 Release Integrity

On the Mac, generate checksums for the complete controlled release:

```bash
find PROJECT_USB -type f -exec shasum -a 256 {} \; > PROJECT_USB/07_MANIFEST/CHECKSUMS.sha256
```

The checksum file becomes the release's integrity ledger.

Recommended release identity:

```text
RELEASE_ID = windows7_offline_r01
```

The exact release ID should also appear in the manifests and scientific deployment records.

---

# 35. Do Not Checksum a Mutable Runtime and Call It a Release

The release package should be the immutable input set.

Once installed:

```text
USB RELEASE
     ↓
LOCAL COPY
     ↓
VIRTUAL ENVIRONMENT
     ↓
RUNTIME OUTPUTS
```

Do not confuse:

```text
release checksum
```

with:

```text
experiment output checksum
```

They serve different purposes.

---

# 36. What the Mac Must Never Do

The Mac should not:

- treat a Windows virtual environment as portable across operating systems;
- overwrite the scientific configuration merely to satisfy VAIO limitations;
- silently update dependency versions after the release is frozen;
- let a convenience package replace a required exact model revision;
- make the Windows machine dependent on a network service at runtime;
- accept a successful installation as proof of experimental correctness.

---

# 37. What the Kali Machine Must Never Do

The Kali machine should not:

- become an uncontrolled second source of truth;
- use different dataset preprocessing without recording it;
- change labels to compensate for a loading issue;
- tune on the test set;
- change frozen extraction parameters during resume;
- interpret faster execution as scientific equivalence without a cross-check.

---

# 38. What the Windows 7 Machine Must Never Do

The VAIO should not:

- download packages from the internet;
- download models at runtime;
- download datasets at runtime;
- auto-upgrade pip;
- auto-upgrade Python packages;
- modify the canonical model registry;
- silently change pooling or sequence length;
- run a 7B-class model merely because it exists in the research catalogue;
- execute directly from an unstable USB working directory for large outputs when local storage is available.

---

# 39. Troubleshooting Decision Tree

```text
INSTALLATION FAILS
        │
        ▼
Can Python start?
   │            │
  NO           YES
   │            ▼
fix Python     Can pip locate local wheel?
                  │          │
                 NO         YES
                  │          ▼
        inspect wheel set   Can core imports work?
                               │       │
                              NO      YES
                               │       ▼
                       dependency issue   Can torch run?
                                              │      │
                                             NO     YES
                                              │      ▼
                                      runtime/OS issue  Can local model load?
                                                          │       │
                                                         NO      YES
                                                          │       ▼
                                              missing/incompatible files
                                                                  │
                                                                  ▼
                                                       Can micro-experiment run?
                                                                  │      │
                                                                 NO     YES
                                                                  │      ▼
                                                          inspect provenance
                                                                         │
                                                                         ▼
                                                               full experiment
```

---

# 40. Failure Classification

A failure should be classified before being repaired.

| Failure type | Example | Correct response |
|---|---|---|
| **Deployment failure** | Missing DLL / incompatible wheel | Fix release/runtime layer |
| **Resource failure** | OOM on VAIO | Reduce deployment scope or skip model |
| **Path failure** | Hard-coded `/Volumes/...` | Make path portable |
| **Data failure** | Sample count mismatch | Stop; repair provenance/alignment |
| **Label failure** | ISEAR numeric/string mismatch | Repair canonical mapping |
| **Scientific drift** | Different pooling | New experiment/configuration |
| **Integrity failure** | Checksum mismatch | Stop; do not overwrite evidence |
| **Resume failure** | Hash mismatch | Treat existing artifact as incompatible |
| **Network failure** | Offline run attempts Hub access | Make resource truly local |

---

# 41. Important Lessons Already Established by the Project

The deployment design should preserve the engineering corrections already made in the extraction/probing architecture.

## 41.1 Fingerprints must have one definition

Text provenance should be computed consistently between extraction and probing.

## 41.2 Labels are a scientific contract

A numeric ID and the semantic class it represents are not interchangeable unless the mapping is explicit and verified.

## 41.3 Memmap is for numeric dense arrays, not arbitrary object labels

Large hidden-state arrays can be memory-mapped. Object-heavy label structures require different loading semantics.

## 41.4 Resumed checksums must represent the complete artifact

An incremental checksum must not accidentally describe only newly processed batches.

## 41.5 Corruption signals must not be silently repaired

A validator should report a mismatch, not overwrite the evidence until the artifact looks clean.

## 41.6 Duplicate texts need deterministic identities

A repeated text is not necessarily the same sample.

## 41.7 Test selection must remain independent

Use validation data for selection; reserve the test set for final reporting.

## 41.8 Probe complexity should remain interpretable

The current project explores linear and MLP probes. Complexity should be selected deliberately because highly expressive probes can detect nonlinear information while simultaneously making interpretation more difficult.

---

# 42. Recommended Operating Policy for the Three Machines

## macOS — AUTHOR

```text
Design
Develop
Acquire
Resolve
Freeze
Package
Manifest
Checksum
```

## Kali — EXECUTE / CROSS-CHECK

```text
Install
Benchmark
Extract
Probe
Validate
Cross-check
Report
```

## Windows 7 — DEPLOY / PROVE COMPATIBILITY

```text
Install
Verify
Run offline
Execute small models
Validate artifacts
Run controlled subset
Record limitations
```

---

# 43. Recommended Experiment Flow

```text
PHASE A — RESEARCH

macOS
  ↓
scientific design
  ↓
model/dataset selection
  ↓
master configuration

PHASE B — MODERN CPU VALIDATION

Kali
  ↓
run configuration
  ↓
validate code
  ↓
validate data/labels
  ↓
validate extraction
  ↓
validate probes

PHASE C — LEGACY RELEASE

macOS
  ↓
build Windows wheels
  ↓
assemble local models/datasets
  ↓
create manifest
  ↓
checksum release

PHASE D — WINDOWS DEPLOYMENT

VAIO
  ↓
install Python
  ↓
create .venv
  ↓
install local wheels
  ↓
install VS Code
  ↓
set offline mode
  ↓
local model test
  ↓
micro-experiment
  ↓
controlled research subset

PHASE E — SCIENTIFIC ACCEPTANCE

Mac + Kali + VAIO records
  ↓
compare manifests
  ↓
compare configuration hashes
  ↓
compare sample/label alignment
  ↓
compare output dimensions
  ↓
accept / reject release
```

---

# 44. Acceptance Criteria

The Windows 7 deployment should be declared **PASS** only if all of the following are true:

```text
[ ] Windows 7 SP1 x64 confirmed
[ ] Python 3.8.10 confirmed
[ ] isolated .venv confirmed
[ ] local wheels installed with --no-index
[ ] no automatic package upgrade performed
[ ] core scientific imports succeed
[ ] PyTorch CPU arithmetic succeeds
[ ] CUDA correctly reports unavailable
[ ] Transformers imports
[ ] local tokenizer loads
[ ] local model loads
[ ] local dataset loads
[ ] labels pass semantic validation
[ ] text ordering/provenance passes
[ ] hidden-state shape passes
[ ] hidden-state artifact reopens
[ ] metadata passes validation
[ ] probe artifact loads
[ ] micro-experiment completes
[ ] offline failure test produces no dependency on internet
[ ] Windows manifest exists
[ ] pip freeze recorded
[ ] release checksum exists
[ ] scientific configuration matches master
```

A machine can be **runtime PASS** while still being **scientifically NOT ACCEPTED** if provenance, labels or configuration differ.

---

# 45. The Three-Machine Definition of Done

## macOS Master

```text
✓ Canonical source identified
✓ Scientific configuration frozen
✓ Models acquired
✓ Datasets acquired
✓ Windows wheel repository built
✓ Complete dependency closure verified
✓ Release manifest generated
✓ SHA-256 checksums generated
```

## Asus / Kali

```text
✓ Isolated environment
✓ CPU execution validated
✓ Modern model subset validated
✓ Extraction validated
✓ Provenance validated
✓ Probing validated
✓ Results cross-checkable against master
```

## Sony VAIO / Windows 7

```text
✓ Python 3.8.10
✓ VS Code 1.70.3
✓ Local wheel installation
✓ Local model loading
✓ Local dataset loading
✓ Offline operation
✓ Small hidden-state extraction
✓ Probe execution on feasible subset
✓ Manifest + pip freeze
✓ Scientific configuration preserved
```

---

# 46. Final Architecture

```text
                         ┌─────────────────────────┐
                         │       MACOS MASTER      │
                         │                         │
                         │ Scientific authority   │
                         │ Source of truth        │
                         │ Model + dataset source  │
                         │ Wheel release factory   │
                         │ Experiment definitions │
                         └────────────┬────────────┘
                                      │
                       controlled immutable release
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      USB RELEASE        │
                         │                         │
                         │ installables            │
                         │ wheels                  │
                         │ source                  │
                         │ datasets                │
                         │ models                  │
                         │ configs                 │
                         │ tests                   │
                         │ manifests               │
                         │ checksums               │
                         └───────┬─────────┬───────┘
                                 │         │
                                 │         │
                                 ▼         ▼
                 ┌──────────────────┐  ┌──────────────────┐
                 │   ASUS / KALI    │  │    SONY VAIO     │
                 │                  │  │                  │
                 │ Modern CPU       │  │ Windows 7 x64    │
                 │ No NVIDIA GPU    │  │ Core 2 Duo / 4GB │
                 │ Main CPU runner  │  │ Legacy target    │
                 │ Full validation  │  │ Offline subset   │
                 └────────┬─────────┘  └─────────┬────────┘
                          │                      │
                          └──────────┬───────────┘
                                     ▼
                           Comparable evidence
                                     │
                                     ▼
                           Scientific interpretation
```

---

# 47. Final Principle

The project should ultimately exist as:

```text
ONE SCIENTIFIC PROJECT
        +
ONE CANONICAL CONFIGURATION
        +
ONE CONTROLLED SOURCE TREE
        +
EXACT MODEL/DATASET IDENTITIES
        +
PROVENANCE-AWARE HIDDEN-STATE ARTIFACTS
        +
REPRODUCIBLE PROBE PROTOCOL
        +
MAC MASTER ENVIRONMENT
        +
KALI CPU EXECUTION ENVIRONMENT
        +
WINDOWS 7 LEGACY DEPLOYMENT ENVIRONMENT
        +
CONTROLLED USB RELEASE
        +
CHECKSUMS + MANIFESTS
        +
OFFLINE OPERATION WHERE REQUIRED
```

The Mac remains the **scientific authority**.

The Kali machine provides the **modern CPU execution and independent cross-check**.

The Sony VAIO provides the **legacy Windows 7 compatibility target**.

The USB is the **controlled bridge between them**.

The operating systems, filesystems, Python installations and hardware may differ. The scientific identity of the experiment must not.

---

# Appendix A — Quick Start Checklist

## Mac

```text
[ ] Freeze experiment configuration
[ ] Validate current source
[ ] Acquire exact datasets
[ ] Acquire exact model snapshots
[ ] Build Windows 3.8 / win_amd64 wheels
[ ] Add Python 3.8.10 installer
[ ] Add VS Code 1.70.3 ZIP
[ ] Add VC++ runtime
[ ] Add tests
[ ] Create manifests
[ ] Generate SHA-256 checksums
```

## Kali

```text
[ ] Create project venv
[ ] Install controlled dependencies
[ ] Verify CPU mode
[ ] Run smoke tests
[ ] Validate labels
[ ] Validate provenance
[ ] Run extraction micro-test
[ ] Run probe micro-test
[ ] Record environment manifest
```

## Windows 7

```text
[ ] Confirm Windows 7 SP1 x64
[ ] Install Python 3.8.10
[ ] Create .venv
[ ] Install from local wheels only
[ ] Install VS Code 1.70.3 portable
[ ] Select .venv interpreter
[ ] Set HF_HUB_OFFLINE=1
[ ] Set TRANSFORMERS_OFFLINE=1
[ ] Test PyTorch
[ ] Test local transformer
[ ] Test local dataset
[ ] Verify labels
[ ] Run micro-experiment
[ ] Run offline failure test
[ ] Generate manifest + pip freeze
[ ] Start only feasible models
```

---

# Appendix B — Research Lineage

This deployment architecture incorporates lessons from the project's existing hidden-state extraction/probing work, including:

```text
• deterministic/frozen extraction hyperparameters
• experiment IDs + hyperparameter hashes
• resumable memmap-based hidden-state storage
• model registry + resource-aware batch hints
• external storage support
• environment/system fingerprinting
• explicit model revisions
• text/label provenance contracts
• dataset-to-hidden-state alignment checks
• [N,D] and [N,L,D] representation handling
• train/validation/test separation
• validation-based model/layer selection
• shuffled-label controls
• probe scorecards and multiple metrics
• forensic artifact validation
```

These are not optional decorations. They are what turns a multi-machine deployment into a reproducible research system.

---

**Document role:** stand-alone deployment and reproducibility specification for the three-machine research environment.

**Primary target:** offline Python + VS Code research execution on Windows 7, with macOS as master and Kali Linux as modern CPU execution/validation platform.
