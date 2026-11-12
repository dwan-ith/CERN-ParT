# CERN Particle Transformer (ParT) Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CERN OpenData](https://img.shields.io/badge/CERN-OpenData%20CMS-blueviolet.svg)](https://opendata.cern.ch/)
[![Architecture](https://img.shields.io/badge/Architecture-ParT%20(Qu%20et%20al.)-brightgreen.svg)](https://arxiv.org/abs/2202.03772)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A self-contained PyTorch foundation model for High-Energy Physics (HEP) implementing the **Particle Transformer (ParT)** architecture with **Masked Particle Modeling (MPM)**. It streams collision events directly from **CERN CMS NanoAOD Run 1 Open Data over HTTPS** using `uproot`, requiring **no local storage of terabyte ROOT files** and **no complex XRootD/EOS installation**.

The entire ~404M parameter model is engineered to train and evaluate on low end GPUs using activation gradient checkpointing and Automatic Mixed Precision (AMP).

---

## Key Highlights

* **XRootD-Free Remote Streaming**: Directly streams CMS NanoAOD ROOT trees over HTTPS into Awkward arrays and PyTorch tensors. Zero terabyte downloads, zero EOS setup, 100% native on Windows, Linux, and macOS.
* **Kinematic Interaction Bias**: Implements pairwise particle attention biases based on Minkowski spacetime and jet clustering metrics ($\Delta R$, $k_T$, $z$, $\Delta m$), reflecting the physical geometry of particle showers.
* **Self-Supervised Masked Particle Modeling (MPM)**: Pre-trains on unlabeled collision events by masking 15% of particle constituents and predicting 4-momenta and particle identification observables.
* **Consumer-Grade Hardware Optimization**: Scales up to **404 Million parameters** while maintaining a peak VRAM footprint of **~3.49 GB** via PyTorch 2.0 `torch.utils.checkpoint` and `torch.amp.autocast`.

---

## Architecture Overview

```
                               CMS Collision Event
                       (Muons, Electrons, Photons, Jets, Taus)
                                        │
                                        ▼
             [p_T, η, φ, m, charge, PDG ID, type]  (N constituents)
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       Particle Feature Embedding                     Pairwise Kinematics
            Linear + GELU + LN                          (ΔR, kT, z, Δm)
                 │                                             │
                 │                                             ▼
                 │                                      Interaction MLP
                 │                                             │
                 └──────────────┬──────────────────────────────┘
                                ▼
                 ┌─────────────────────────────┐
                 │    InteractionBlock x 24    │
                 │ ─────────────────────────── │
                 │  • Multi-Head Self-Attn     │ ◄── Attention Bias
                 │  • Residual + LayerNorm     │
                 │  • Feed-Forward MLP (6144)  │
                 │  • Gradient Checkpointed    │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       Event Representation          Constituent Embeddings
       [CLS] Token (1024-d)          (N x 1024-d)
         ├─ Anomaly Detection          └─ Masked Particle Modeling Head
         └─ Jet / Event Tagging            (Pre-training Reconstruction)
```

### 1. Pairwise Kinematic Interaction Bias
Standard self-attention computes $\text{Attn}(Q, K) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)$.  
In this model, $B$ is a continuous interaction matrix computed per particle pair $(i, j)$:
* **Angular distance**: $\Delta R = \sqrt{(\eta_i - \eta_j)^2 + (\Delta\phi_{ij})^2}$
* **Relative transverse momentum**: $k_T = \min(p_{T, i}, p_{T, j}) \cdot \Delta R$
* **Energy sharing**: $z = \frac{p_{T, i}}{p_{T, i} + p_{T, j}}$
* **Mass difference**: $\Delta m = |m_i - m_j|$

### 2. Feature Extractor
Processes reconstructed particles into 7 canonical features per constituent:
$$[p_T, \, \eta, \, \phi, \, m, \, \text{charge}, \, \text{PDG ID}, \, \text{type ID}]$$
Covering Muons, Electrons, Photons, Jets, and Taus from CMS NanoAOD trees.

---

## Hardware & Memory Budget

Targeted for low end hardware while maintaining full foundation model capacity:

| Component | Dimensions / Specs | Parameter Count |
| :--- | :--- | :--- |
| **Embedding Dimension ($d_{\text{model}}$)** | 1024 | ~0.01M |
| **Attention Heads** | 16 ($d_{\text{head}} = 64$) | Included in blocks |
| **Transformer Layers** | 24 | ~403M |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 6144 | Included in blocks |
| **Interaction MLP** | $4 \to 128 \to 1$ | ~0.02M |
| **Pre-training Head** | $1024 \to 2048 \to 2048 \to 7$ | ~6.3M |
| **Total Model Parameters** | — | **~404.11 Million** |
| **Measured Peak VRAM (Training)** | Batch 16, 64 particles, AMP | **~3.49 GB / 6.0 GB** |

---

## Repository Structure

```
.
├── data/
│   ├── all_urls.txt          # 1,382 CERN OpenData CMS NanoAOD ROOT URLs
│   ├── feature_extractor.py  # Branch parsing & kinematic feature engineering
│   └── https_dataset.py      # PyTorch IterableDataset streaming via uproot
├── model/
│   ├── config.py             # ModelConfig, TrainingConfig, DataConfig
│   ├── particle_transformer.py # ParT with interaction attention & checkpointing
│   ├── pretraining_head.py   # Masked Particle Modeling (MPM) objective
│   └── task_head.py          # Downstream fine-tuning & evaluation heads
├── tests/
│   ├── test_pipeline.py      # End-to-end integration & VRAM verification test
│   ├── test_features.py      # Fast unit tests for branch parsing
│   └── test_model.py         # Unit tests for ParT kinematics & attention
├── checkpoint.py             # Atomic checkpoint saving and resumption
├── generate_urls.py          # Scraper for CERN OpenData record indices
├── infer.py                  # Inference, embedding extraction & reconstruction demo
├── train.py                  # Pre-training loop with cosine LR schedule & checkpoints
├── pyproject.toml            # Package configuration
├── requirements.txt          # Production dependencies
├── LICENSE                   # MIT License
└── README.md
```

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dwan-ith/CERN-ParT.git
cd CERN-ParT
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Pipeline

Run the comprehensive end-to-end test suite (fetches a remote CMS NanoAOD file, extracts particle features, performs forward pass, tests MPM loss, and checks GPU VRAM):

```bash
python tests/test_pipeline.py
```

Expected output:
```text
============================================================
CERN Foundation Model — Pipeline Test
============================================================
--- Feature extractor ---
[test] Required branches: 26 -> PASS
--- HTTPS streaming ---
[test] Opening http://opendata.cern.ch/.../Run2010A_Mu/...root -> PASS
--- Model forward ---
[test] Model: 404.11M parameters -> PASS
--- Pretraining head ---
[test] MPM loss: 1.0263 -> PASS
--- GPU memory ---
[test] GPU peak memory: 3.49 GB -> PASS

ALL TESTS PASSED
```

You can also run unit tests with `pytest`:
```bash
pytest tests/
```

### 3. Run Inference & Embedding Extraction

Extract high-dimensional latent representations for downstream classification or inspect masked particle predictions:

```bash
# Uses pre-trained checkpoint if available in checkpoints/final.pt, or architecture mode
python infer.py
```

### 4. Self-Supervised Pre-Training

To launch the full training loop streaming across the 1,382 CMS NanoAOD collision files:

```bash
python train.py --preset base --n-steps 5000
```

## Dataset & Scientific References

* **Dataset**: CMS Collaboration. Open Data from the CMS experiment at the LHC. Derived dataset: NanoAOD Run 1 (2010–2012 collision datasets). Available via the [CERN Open Data Portal](https://opendata.cern.ch/).
* **Particle Transformer Architecture**:
  * Qu, H., Li, C., & Qian, S. (2022). *Particle Transformer for Jet Tagging*. [arXiv:2202.03772](https://arxiv.org/abs/2202.03772).
* **Uproot / Scikit-HEP**:
  * Pivarski, J., et al. *Uproot: ROOT I/O in pure Python and NumPy*. [JOSS](https://doi.org/10.21105/joss.00651).

---

## License

This project is licensed under the [MIT License](LICENSE).

