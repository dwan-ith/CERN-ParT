# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.2.0] - 2026-11-13

The release in which the repository was turned from "a model that claims to
fit on a laptop" into "a model that provably fits on a laptop, and trains".

### Fixed

- **The default configuration could not actually train.** A 404 M-parameter
  model needs 1.62 GB of fp32 weights, 1.62 GB of gradients and 3.23 GB of
  AdamW moments — 6.47 GB before a single activation is allocated, against
  6.44 GB of VRAM on an RTX 4050. It raised `torch.OutOfMemoryError` at the
  first optimizer step. The memory budget in `model/config.py` assumed fp16
  gradients, which AMP does not produce. Added 8-bit AdamW support and a
  preset ladder whose sizes are checked against the target hardware
  (`tests/test_config.py`).
- **The pretraining objective was close to a no-op.** The manifest pointed at
  the Run 1 single-object NanoAOD records, which average 2.2 particles per
  event; with a 15 % mask ratio that hides less than one particle per event,
  so 0.5 % of particles were actually masked. Switched to the Run 2
  NanoAODv9 records, which average 10.4 particles per event.
- **The masking leaked the answer.** The pairwise interaction features were
  computed on the unmasked particles, so the momentum of a masked particle was
  still visible through `kT` and `z`. The masked particle's momentum is now
  zeroed in every interaction feature that involves it, and the leak-freedom
  is enforced by a test.
- **The CLS token was paired with the wrong interaction features.** The CLS
  row and column were *appended* to the interaction tensor while the CLS
  token itself is *prepended* to the hidden states, so the two were
  misaligned by one position.
- **`save_checkpoint` crashed the training run.** `avg_loss` was read in the
  `save_every` branch but only assigned in the `log_every` branch.
- **Raw features were uninitialised memory.** The per-particle block was
  allocated with `np.empty` for a 7-wide vector while only 6 channels were
  written, leaking arbitrary values (and overflowing the float32 cast).
- **The reservoir shuffle buffer dropped events** that it did not keep.

### Added

- Model presets (`tiny`, `small`, `base`, `medium`, `large`) with an
  analytic memory estimator.
- Downstream b-tagging benchmark (`evaluate.py`) with three modes — `scratch`,
  `finetune` and `linear` — and an interaction-bias ablation.
- Leak-free masking strategies (`kinematic`, `full`) and the tests that
  enforce them.
- Feature normalisation (log-scaled momenta, `sin`/`cos` of phi, one-hot
  object type) inside the model.
- Resume support: checkpoints carry the optimizer state, the AMP scaler and
  the step counter.
- Local file cache with resumable, validated downloads (`--cache-dir`), and
  an event-level cache (extracted padded tensors, ~18 MB per file instead of
  ~140 MB of ROOT data).
- At least one masked particle per event in the pretraining objective.
- Held-out pretraining loss via `--val-url-file` / `--val-every`.
- Checkpoint pruning (`--keep-last`) so long runs do not fill the disk, and
  checkpoint size reporting.
- 61 offline unit tests and 4 integration tests that stream real files.
- CI workflow, `pyproject.toml` with console scripts, `CITATION.cff`, and a
  rewritten README.

### Changed

- The event-level shuffle buffer now samples uniformly (reservoir sampling)
  instead of draining it in order.
- The package was restructured and the layout flattened to the repository
  root.

## [0.1.0]

Initial release: ParT architecture, HTTPS streaming of CMS NanoAOD, masked
particle modelling.
