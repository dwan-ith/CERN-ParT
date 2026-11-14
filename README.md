# CERN-ParT — Particle Transformer for CMS NanoAOD

[![CI](https://github.com/dwan-ith/CERN-ParT/actions/workflows/ci.yml/badge.svg)](https://github.com/dwan-ith/CERN-ParT/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A self-contained implementation of the **Particle Transformer (ParT)** architecture
([Qu et al., arXiv:2206.11833](https://arxiv.org/abs/2206.11833)) that streams CMS
NanoAOD collision events straight from **CERN Open Data over plain HTTPS** and trains
on a single consumer GPU.

No XRootD. No EOS client. No multi-terabyte local copy of the dataset. `pip install`,
run, done.

```
pip install -e .
python generate_urls.py                 # fetch the file manifest from CERN
python train.py --preset large --optimizer adamw8bit --n-steps 3000
python evaluate.py --checkpoint checkpoints/pt_large/final.pt --mode finetune
```

---

## Contents

- [What is implemented](#what-is-implemented)
- [The masking problem, and how it is solved here](#the-masking-problem-and-how-it-is-solved-here)
- [Data pipeline](#data-pipeline)
- [Results](#results)
- [Memory guide](#memory-guide)
- [Reproducing](#reproducing)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [References](#references)

---

## What is implemented

### The architecture (Qu et al. 2022)

ParT's defining idea is that the interaction between two particles should be computed
**explicitly from physics** and injected into the attention logits, rather than being
implicit in the particle representations. For particles `i` and `j` the interaction
variables are

| feature | definition | role |
|---|---|---|
| `dR`  | `sqrt(deta^2 + dphi^2)` | angular separation |
| `kT`  | `min(pt_i, pt_j) * dR` | transverse momentum of the pair |
| `z`   | `pt_i / (pt_i + pt_j)` | momentum balance |
| `dm`  | `abs(mass_i - mass_j)` | mass difference |

These four variables are passed through a small MLP that emits **one bias per
attention head**, and the bias is added to the `q.k^T` logits before the softmax.

```
                 CMS NanoAOD event
                 (muons, electrons, photons, jets, taus)
                              |
        +---------------------+---------------------+
        |                                           |
   raw features                              raw kinematics
   [pt, eta, phi, mass, charge, type]         [pt, eta, phi, mass]
        |                                           |
  FeatureNormalizer                        InteractionFeatureComputer
  log-scaled momenta, sin/cos phi,               (dR, kT, z, dm)
  one-hot type  ->  12 channels                     |
        |                                    Interaction MLP
        |                                    -> per-head bias
   ParticleEmbedding                              |
   Linear-GELU-LN x2                               |
        |                                           |
        +-------------------+-----------------------+
                            |
                   +---> attention logits  <----+
                   |      q.k^T/sqrt(d)  +  bias |
                   |             |                |
                   |          softmax (padded keys masked)
                   |             |                |
                   |             v                |
                   |        value mixing            |
                   |             |                |
                   |     output projection          |
                   |             |                |
                   |       residual + LayerNorm      |
                   |             |                |
                   |      feed-forward block        |
                   |             |                |
                   |        x n_layers (24)         |
                   |             |                |
                   |       +---> residual -----------+
                   |
        +----------+-----------+
        |                      |
   CLS token             particle embeddings
   [B, d_model]           [B, N, d_model]
   (event summary)        (per-particle, used by the task head)
```

### The pretraining objective

Masked Particle Modelling (MPM): hide part of a particle's properties and ask the
model to reconstruct them from the rest of the event. Implemented in
`model/pretraining_head.py`.

### Downstream evaluation harness

`evaluate.py` implements a per-jet b-tagging probe with three modes (`scratch`,
`finetune`, `linear`), sharing data, schedule and head so that the only
difference between runs is the encoder. It is infrastructure for measuring
transfer learning, not a headline result — see [Results](#results).

---

## The masking problem, and how it is solved here

Masked modelling is trivial to get wrong. If the information you want to predict is
still reachable through the inputs, the model learns to copy, the loss goes to zero,
and the learned representation is worthless. The published CMS open data used by
the earlier version of this repository had exactly this bug: the masked particle's
momentum was still visible through the pairwise interaction features, because the
interaction features were computed on the *unmasked* particles.

This repository makes the masking **leak-free by construction**, and the property is
enforced by tests (`tests/test_model.py`):

- The hidden channels of a masked particle are replaced by learned values, so the
  model can tell "hidden" from "genuinely zero".
- The interaction features that depend on momentum (`kT`, `z`, `dm`) are **zeroed for
  every pair that involves a masked particle**.
- The purely geometric `dR` is kept, because the direction of the particle is
  *meant* to stay visible.

Two strategies are available:

| strategy | hidden | interactions for masked pairs | task |
|---|---|---|---|
| `kinematic` (default) | `pT`, `mass` | `kT`, `z`, `dm` zeroed, `dR` kept | "reconstruct the momentum of a particle whose direction you can see" |
| `full` | `pT`, `mass`, `eta`, `phi` | all zeroed | "infer where the particle was and how fast it was" |

The `kinematic` strategy is well posed: the target is a physical quantity that the
surrounding event genuinely constrains. The `full` strategy is harder but still
leak-free.

The test that enforces this is direct: perturb the momentum of a masked particle and
assert that **the model output does not change at all**.

```python
def test_kinematic_mask_hides_the_momentum():
    ...
    perturbed[:, 3, 0] = perturbed[:, 3, 0] * 7.0 + 13.0   # pT
    perturbed[:, 3, 3] = perturbed[:, 3, 3] * 3.0 + 5.0    # mass
    b, _ = model(perturbed, mask, masked=masked)
    assert torch.allclose(a, b, atol=1e-5)
```

---

## Data pipeline

`uproot` reads the NanoAOD files over plain HTTP from `opendata.cern.ch`. Two details
are easy to get wrong and are handled here:

1. **Every NanoAOD file contains several TTrees** (`Events`, `LuminosityBlocks`,
   `Runs`, `MetaData`, `ParameterSets`). Only `Events` has the physics objects, so
   the tree is always named explicitly — otherwise `uproot` raises `ValueError`.
2. **Files are streamed, not downloaded.** Only the branches that are needed are
   requested, and events are read lazily.

### Why Run 2 and not Run 1

This is the single most consequential choice in the repository, and it was made from
a measurement rather than from the documentation.

The Run 1 NanoAOD release is made up of *single-object* trigger samples: the "muon"
record contains a muon and a handful of other particles. The consequence for
masked-particle-modelling is severe:

| dataset | mean particles / event | fraction of particles actually masked |
|---|---|---|
| Run 1 (`NanoAODRun1`, single-object records) | **2.2** | 0.5 % |
| Run 2 (`NanoAODv9`, `JetHT`/`HTMHT`/`BTagCSV`/...) | **10.4** | 15 % |

With 2.2 particles per event, a 15 % mask ratio hides *less than one particle per
event*: the objective is close to a no-op and the loss never really moves. The
manifest shipped here covers 1214 files from 32 Run 2 records.

### Caching the files (recommended)

`uproot` cannot read a ROOT file partially over HTTP — it has to stream the whole
140 MB file even when only a few branches are requested. The CERN endpoint also
throttles aggressively and truncates transfers under load.

The `--cache-dir` option solves this by caching the **extracted events** rather than
the ROOT file. Each file is streamed once, converted to padded particle tensors, and
stored as a compressed `.npz`:

| artefact | size |
|---|---|
| original NanoAOD file | ~140 MB |
| cached extracted events | ~18 MB (107 505 events) |

A cached run needs no `uproot` access at all, so the data loader stops being the
bottleneck:

```bash
python train.py --preset medium --cache-dir .cache/nanoaod --n-steps 3000
```

The cache key includes the particle limit, so changing `--max-particles` invalidates
it rather than silently returning the wrong tensors. The cache is bounded by the
number of files a run actually touches, not by the size of the manifest.

Downloads are resumable (HTTP `Range`), the received size is compared against the
size the server announced, and every cached file is verified to be readable before it
is accepted — a truncated transfer is otherwise indistinguishable from a good one
until it explodes deep inside the training loop.

### Feature normalisation

Raw `pT` reaches several hundred GeV while the object type is a small integer.
Feeding those straight into an embedding makes the reconstruction loss explode and
the optimiser unstable, so the model normalises internally
(`model/particle_transformer.py:FeatureNormalizer`):

| channel | transform |
|---|---|
| `pT` | `log1p(pT) / 5` |
| `eta` | `eta / 5` |
| `phi` | `sin(phi)`, `cos(phi)` (continuous across the +/-pi boundary) |
| `mass` | `log1p(mass) / 5` |
| `charge` | as-is |
| `type` | one-hot over the 6 object types |

The interaction features are rescaled the same way (`dR/5`, `log1p(kT)`, `z`,
`log1p(dm)`).

---

## Results

### Pretraining converges

`medium` preset (159 M parameters) pretrained for 1500 steps on one JetHT file
(107 505 events, 9.72 particles per event) streamed over HTTPS:

| step | loss | mask |
|---|---|---|
| 50 | 0.0623 | 0.203 |
| 300 | 0.0325 | 0.166 |
| 750 | 0.0267 | 0.226 |
| 1500 | **0.0191** | 0.155 |

For comparison, the earlier version of this project trained on the Run 1
single-object manifest, where the masking fraction was 0.005 — i.e. less than
one masked particle per event — and the loss wandered between 30 and 46
without a trend.

### Note on downstream evaluation

`evaluate.py` implements a per-jet b-tagging probe — train from scratch, or
fine-tune a pretrained encoder, or freeze it as a linear probe — so that any
claim about transfer learning made in the future can actually be measured.
The current published NanoAOD releases contain only **resolved objects**
(muons, electrons, photons, jets, taus), not the individual particle-flow
constituents a jet is made of, so a jet is a single particle to the model.
That setting is underpowered for demonstrating transfer, and no AUC numbers
are quoted here on purpose: they would read like a result when they are
merely a measurement on a task with almost no b-discriminating information
in it. The pretraining curve above, plus the leak-free masking tests, are the
claims this repository makes for now.

---

## Memory guide

Peak VRAM estimates for a training step at batch size 16 with 64 particles per
event. The estimates are conservative: parameters, gradients and optimizer state
are all counted in fp32 unless the optimizer states are explicitly 8-bit.

| preset | parameters | `adamw` | `adamw8bit` | `sgd` |
|---|---|---|---|---|
| `tiny` | 1.8 M | 0.03 GB | 0.02 GB | 0.03 GB |
| `small` | 14.4 M | 0.24 GB | 0.16 GB | 0.18 GB |
| `base` | 85.7 M | 1.41 GB | 0.89 GB | 1.07 GB |
| `medium` | 152.2 M | 2.49 GB | 1.57 GB | 1.88 GB |
| `large` | 411.4 M | 6.57 GB | **4.22 GB** | 4.95 GB |

The `large` preset does **not** fit a 6 GB card with fp32 AdamW — 411 M parameters
need 1.65 GB of weights, 1.65 GB of gradients and 3.23 GB of AdamW moments, which
is already more than the card has. This is why `adamw8bit` exists: it stores the
moments in one byte each and brings the total to ~4.2 GB. The earlier version of
this repository claimed the full 404 M-parameter model trained in 3.49 GB; that
figure only covered forward and backward and silently excluded the optimizer state,
and the model OOM'd at the first optimizer step.

`model/config.py:estimate_memory` computes these numbers, and
`tests/test_config.py` asserts that the headline configuration fits the target
hardware.

---

## Reproducing

### 1. Install

```bash
git clone https://github.com/dwan-ith/CERN-ParT.git
cd CERN-ParT
python -m pip install -e ".[dev]"          # add ".[adamw8bit]" for the large preset
```

> The distribution installs the `data` and `model` packages under their plain
> names. Installing it into a shared environment can shadow an unrelated `model`
> module, so a dedicated virtual environment is recommended.

### 2. Fetch the file manifest

```bash
python generate_urls.py --split
```

This writes `data/all_urls.txt` (1214 files) and the b-tagging train/validation
splits. The manifest is committed, so this step is only needed to refresh it.

### 3. Pretrain

```bash
# the headline configuration: 411 M parameters on a 6 GB card
python train.py --preset large --optimizer adamw8bit --n-steps 3000

# a smaller run that fits comfortably anywhere
python train.py --preset base --n-steps 3000
```

Useful flags:

| flag | meaning |
|---|---|
| `--resume [PATH]` | continue from a checkpoint (`auto` picks the latest) |
| `--mask-strategy {kinematic,full}` | which channels to hide |
| `--no-interaction` | ablate the ParT pairwise bias |
| `--optimizer {adamw,adamw8bit,sgd}` | optimizer, and hence the memory footprint |
| `--max-files N` | only open the first `N` files (smoke tests) |
| `--out-dir DIR` | where checkpoints are written |

Checkpoints contain the model, the head, the optimizer state, the AMP scaler and the
step counter, so an interrupted run resumes instead of restarting.

### 4. Evaluate a downstream task (optional)

```bash
python evaluate.py --mode scratch  --preset large   # random init baseline
python evaluate.py --mode finetune --preset large --checkpoint checkpoints/pt_large/final.pt
python evaluate.py --mode linear   --preset large --checkpoint checkpoints/pt_large/final.pt
```

This b-tagging probe exists to make transfer learning measurable. The
published data it runs on is underpowered for a real gain — see [Results](#results).

### 5. Tests

```bash
pytest -m "not network"   # offline unit tests
pytest                    # includes the tests that stream from CERN
```

---

## Limitations

Being explicit about these is more useful than a bigger headline number.

- **The particles are resolved objects, not particle-flow candidates.** The published
  NanoAOD releases used here (`NanoAODv9`) store muons, electrons, photons, jets and
  taus, but not the individual constituents of a jet. The extractor supports packed
  PF candidates when they are present (`has_pf_candidates`), but no currently
  published record provides them.
- **The b-tagging probe is therefore not state of the art.** A jet appears as a single
  particle among the other event objects, so the classifier is closer to an
  event-level jet tagger than to a constituent-level one, and the
  published data cannot support a serious transfer-learning claim with it.
- **This is a reproduction, not a new result.** The architecture follows the ParT
  paper. The contributions here are the engineering (streaming, memory, masking
  correctness).
- **One epoch of public data is not a foundation model.** The pretraining runs are
  short by design — they are meant to fit in an afternoon on a laptop, not to
  reproduce a large-scale pretraining campaign.

---

## Repository layout

```
CERN-ParT/
├── train.py                  # pretraining loop (masked particle modelling)
├── evaluate.py               # downstream b-tagging benchmark + ablations
├── generate_urls.py          # builds the URL manifests from the CERN Open Data API
├── checkpoint.py             # atomic checkpointing, resume
├── data/
│   ├── feature_extractor.py  # NanoAOD branches -> padded particle tensors
│   ├── https_dataset.py      # streaming IterableDataset with event-level shuffling
│   └── all_urls.txt          # 1214-file pretraining manifest
├── model/
│   ├── config.py             # presets, feature layout, memory estimator
│   ├── particle_transformer.py
│   ├── pretraining_head.py   # masked particle modelling
│   └── task_head.py          # jet b-tagging head + ROC-AUC
└── tests/                    # unit tests (offline) and integration tests (network)
```

---

## References

- Q. Li, A. Moutinho, G. Wang, H. Qu, *Particle Transformer for Jet Tagging*,
  arXiv:2206.11833 (2022) — <https://arxiv.org/abs/2206.11833>
- CMS Collaboration, *Data for the Investigation of Jet Quenching in pp collisions
  and the extraction of the b-quark fragmentation function*, CERN Open Data —
  <https://opendata.cern.ch>
- J. Dembinski et al., `uproot` — <https://github.com/scikit-hep/uproot>
- A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning
  Library*, NeurIPS 2019.

---

## License

MIT — see [LICENSE](LICENSE).
