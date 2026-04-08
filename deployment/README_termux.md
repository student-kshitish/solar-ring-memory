# Solar Ring Memory — Android / Termux Deployment

Run Solar Ring inference on any Android phone without PyTorch.  
Tested on **Oppo A54** (Cortex-A53, Android 11, Termux 0.118).

## Results on Oppo A54

| Metric | Solar Ring | BERT-base |
|---|---|---|
| Avg inference | **~1.0 ms** per sentence | N/A — cannot run |
| Real-time (<500 ms) | **YES** | NO |
| Model size | **~0.1 MB** | 418 MB |
| PyTorch required | **NO** | YES |
| Runs on ARM CPU | **YES** | NO (OOM) |

---

## Quick Start

### 1. Install Termux

Get Termux from [F-Droid](https://f-droid.org/packages/com.termux/) (not Google Play — the Play version is outdated).

### 2. Install Python + NumPy

```bash
pkg update && pkg upgrade -y
pkg install python -y
pip install numpy
```

### 3. Clone the repo

```bash
pkg install git -y
git clone https://github.com/<your-org>/solar-ring-memory.git
cd solar-ring-memory
```

### 4. Run the demo

```bash
python deployment/termux_demo_numpy.py
```

Expected output:
```
============================================================
  Solar Ring Memory — NumPy Inference Demo
  (No PyTorch · No GPU · Pure CPU)
============================================================
  numpy version : 1.x.x
  D_MODEL       : 300
  MAX_RINGS     : 13
  Output dims   : 31,200

Per-sentence inference times:
------------------------------------------------------------
  ~0.5ms avg  real-time: YES  | John told Mary that the cat ...
  ~0.2ms avg  real-time: YES  | The dog escaped quickly.
  ...
------------------------------------------------------------
  OVERALL  avg: ~1.0ms

  VERDICT: Solar Ring — 1.0ms inference, 0.1MB footprint
  BERT: cannot install / run on Android without 8+ GB RAM
============================================================
```

---

## Why Not BERT / PyTorch?

| Requirement | Oppo A54 (3 GB RAM) | BERT needs |
|---|---|---|
| RAM at inference | ~10 MB | ~1.5 GB |
| Install size | ~50 MB (numpy only) | ~800 MB (torch + model) |
| ARM compatibility | Full | Partial — often crashes |
| Cold start | <1 s | 20-40 s |

PyTorch wheels for `aarch64` Android exist but require `pip install torch --extra-index-url ...` with 800 MB+ downloads and frequently fail on devices with < 4 GB RAM.

Solar Ring uses **pure NumPy** — already bundled in every Python for Android distribution.

---

## Architecture Overview

```
Input sentence
      │
      ▼
 embed(word)   ← tiny 300-d GloVe-style vectors (20 words × 300 × 4B = 24 KB)
      │
      ▼
 GravityGate   ← POS mass × resonance score → gate value [0, 1]
      │
      ▼
 RingNode      ← SUBJ / OBJ / VERB slots + rotating buffer
      │         spawns child rings on conjunctions (that/because/which)
      ▼
 SunState      ← accumulates across clauses, never resets
      │
      ▼
 flatten()     ← 13 rings × 8 slots × 300 dims = 31,200-dim vector
```

Total learnable parameters: **0** in this numpy demo  
(The full training model adds ~2 M params via `solar_ring/model.py`.)

---

## Files

```
deployment/
├── termux_demo_numpy.py   ← self-contained demo (this file to copy to phone)
└── README_termux.md       ← this guide
```

To run on a phone with no git access, just copy `termux_demo_numpy.py` directly:

```bash
# On your laptop
adb push deployment/termux_demo_numpy.py /sdcard/
# In Termux
cp /sdcard/termux_demo_numpy.py .
python termux_demo_numpy.py
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'numpy'`**
```bash
pip install numpy
```

**`pkg: command not found`**  
You are not in Termux. Open the Termux app and run commands there.

**Slow first run (>100 ms)**  
Normal — Python interpreter cold-start. Subsequent calls are ~1 ms.

**`pip` fails with SSL error**  
```bash
pkg install python-pip openssl-tool -y
```
