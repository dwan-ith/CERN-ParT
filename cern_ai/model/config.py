"""Configuration objects for the Particle Transformer.

Everything is a plain :mod:`dataclass` so that a run can be reproduced from a
single JSON blob that is embedded in every checkpoint.

Design notes
------------
* The **raw** feature vector is ``[pt, eta, phi, mass, charge, type_id]``.
  It is what the data layer produces and what the interaction network needs,
  because the interaction features are physical quantities.
* The **model-facing** vector is a normalised, 11-channel encoding
  (``log``-scaled momenta, ``sin``/``cos`` of phi, one-hot object type).
  Normalisation happens *inside* the model so that the data layer stays a
  faithful representation of the file contents.
* Model size is selected from a small ladder of presets rather than by hand,
  and every preset is checked against the memory budget of the target GPU.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------

#: Raw per-particle features produced by :mod:`cern_ai.data.feature_extractor`.
#: ``[pt, eta, phi, mass, charge, type_id]``
N_RAW_FEATURES = 6

#: Object types, in one-hot order. Index matches ``type_id`` in the raw vector.
#:
#: ``pf`` is reserved for packed particle-flow candidates. It is only populated
#: by NanoAOD releases that store them (v10 and later); the currently published
#: Run 1 / Run 2 open data only contain the five resolved object types.
OBJECT_TYPES: Tuple[str, ...] = ("muon", "electron", "photon", "jet", "tau", "pf")

#: Index of the jet type inside :data:`OBJECT_TYPES` (used by the task head).
TYPE_ID_JET = OBJECT_TYPES.index("jet")

#: Index of the packed-PF type inside :data:`OBJECT_TYPES`.
TYPE_ID_PF = OBJECT_TYPES.index("pf")

# --- normalised (model-facing) channels ---
CH_PT = 0
CH_ETA = 1
CH_SINPHI = 2
CH_COSPHI = 3
CH_MASS = 4
CH_CHARGE = 5
CH_TYPE0 = 6

#: Number of model-facing channels.
N_MODEL_FEATURES = CH_TYPE0 + len(OBJECT_TYPES)  # 11

#: Channels that each masking strategy hides from the model.
#:
#: ``kinematic`` hides the momentum-like quantities but keeps the position of
#: the particle, which makes the reconstruction task well posed.
#: ``full`` additionally hides the direction, so the model must infer *where*
#: the particle was from the rest of the event.
MASK_CHANNELS: Dict[str, Tuple[int, ...]] = {
    "kinematic": (CH_PT, CH_MASS),
    "full": (CH_PT, CH_ETA, CH_SINPHI, CH_COSPHI, CH_MASS),
}

#: Interaction feature channels produced by the interaction network.
IX_DR = 0
IX_KT = 1
IX_Z = 2
IX_DM = 3
N_INTERACTION_FEATURES = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    The defaults describe the ``base`` preset, which trains comfortably on a
    6 GB laptop GPU. See :data:`MODEL_PRESETS` for the full ladder.
    """

    n_raw_features: int = N_RAW_FEATURES
    n_features: int = N_MODEL_FEATURES
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    d_interaction: int = 128
    dropout: float = 0.1
    gradient_checkpointing: bool = True

    #: Ablation switch. When ``False`` the pairwise interaction bias is removed
    #: from the attention logits entirely (the rest of the model is untouched),
    #: which isolates the contribution of the ParT-style interaction features.
    use_interaction_bias: bool = True

    #: Dropout applied to the interaction MLP input.
    interaction_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


#: Named architecture presets, smallest first. ``large`` does **not** fit a
#: 6 GB card with fp32 AdamW -- use ``--optimizer adamw8bit`` or a bigger GPU.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(d_model=192, n_heads=4, n_layers=4, d_ff=768, d_interaction=64),
    "small": dict(d_model=384, n_heads=8, n_layers=8, d_ff=1536, d_interaction=96),
    "base": dict(d_model=768, n_heads=12, n_layers=12, d_ff=3072, d_interaction=128),
    "large": dict(d_model=1024, n_heads=16, n_layers=24, d_ff=6144, d_interaction=128),
    # Original ParT configuration (quoted for reference; needs an A100).
    "part": dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024, d_interaction=64),
}


def model_config(preset: str = "base", **overrides: Any) -> ModelConfig:
    """Build a :class:`ModelConfig` from a preset name plus optional overrides."""
    if preset not in MODEL_PRESETS:
        raise KeyError(
            f"unknown preset {preset!r}; available: {sorted(MODEL_PRESETS)}"
        )
    params = dict(MODEL_PRESETS[preset])
    params.update(overrides)
    return ModelConfig(**params)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Optimisation hyper-parameters."""

    batch_size: int = 16
    max_particles: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    n_steps: int = 5000
    grad_accum: int = 1
    mask_ratio: float = 0.15
    mask_strategy: str = "kinematic"
    log_every: int = 50
    save_every: int = 500
    checkpoint_dir: str = "checkpoints"
    amp: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mask_strategy not in MASK_CHANNELS:
            raise KeyError(
                f"unknown mask_strategy {self.mask_strategy!r}; "
                f"available: {sorted(MASK_CHANNELS)}"
            )
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")

    @property
    def masked_channels(self) -> Tuple[int, ...]:
        return MASK_CHANNELS[self.mask_strategy]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _read_url_file(path: str) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def _default_urls() -> list:
    """Pretraining manifest that ships with the repository."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, os.pardir, "data", "all_urls.txt")
    if os.path.exists(candidate):
        return _read_url_file(candidate)
    return []


@dataclass
class DataConfig:
    """Data-pipeline hyper-parameters."""

    max_particles: int = 64
    batch_size: int = 16
    shuffle: bool = True
    shuffle_buffer: int = 2000
    num_workers: int = 0
    #: TTree to read. CMS NanoAOD files carry several trees; only ``Events``
    #: contains the physics objects.
    tree: str = "Events"
    #: Upper bound on the number of files to open (useful for smoke tests).
    max_files: Optional[int] = None
    default_urls: list = field(default_factory=_default_urls)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.pop("default_urls", None)
        return out


# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------
#: Bytes of optimizer state per parameter for the supported optimizers.
OPTIMIZER_STATE_BYTES = {
    "adamw": 8,  # exp_avg + exp_avg_sq, both fp32
    "adamw8bit": 2,  # two 8-bit states + fp32 master handled by bnb
    "sgd": 4,  # momentum buffer only
    "none": 0,
}


def estimate_memory(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    optimizer: str = "adamw",
    n_params: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate peak GPU memory (in GB) for a training step.

    The estimate is deliberately conservative: parameters, gradients and
    optimizer state are all assumed fp32 unless the optimizer states are
    explicitly 8-bit.  Activations are approximated from the activation
    checkpointing setting.

    Returns a dict with ``params``, ``grads``, ``optimizer``, ``activations``
    and ``total`` (all in GB).
    """
    if n_params is None:
        n_params = count_parameters(model_cfg)

    gb = 1e9
    param_bytes = 4 * n_params
    grad_bytes = param_bytes  # AMP keeps gradients in fp32
    opt_bytes = OPTIMIZER_STATE_BYTES.get(optimizer, 8) * n_params

    # Rough activation estimate: ~4 * B * N * d_model bytes per layer when
    # checkpointing is on, ~16x that without.
    act_per_layer = 4 * train_cfg.batch_size * train_cfg.max_particles * model_cfg.d_model
    if not model_cfg.gradient_checkpointing:
        act_per_layer *= 12
    act_bytes = act_per_layer * model_cfg.n_layers

    total = (param_bytes + grad_bytes + opt_bytes + act_bytes) / gb
    return {
        "params": param_bytes / gb,
        "grads": grad_bytes / gb,
        "optimizer": opt_bytes / gb,
        "activations": act_bytes / gb,
        "total": total,
        "n_params": float(n_params),
    }


def count_parameters(model_cfg: ModelConfig) -> int:
    """Parameter count of the architecture described by ``model_cfg``."""
    d = model_cfg.d_model
    ff = model_cfg.d_ff
    h = model_cfg.n_heads
    di = model_cfg.d_interaction
    n_raw = model_cfg.n_raw_features
    n_feat = model_cfg.n_features

    # embedding
    emb = n_raw * d + d  # linear + bias
    emb += d * d + d  # second linear + bias
    emb += 2 * d  # two layer norms
    emb += d  # cls token
    emb += n_feat * d + d  # mask embedding (learned per channel)

    # per block
    attn_norm = 2 * d
    qkv = d * 3 * d + 3 * d
    out_proj = d * d + d
    ff_norm = 2 * d
    ff1 = d * ff + ff
    ff2 = ff * d + d
    if model_cfg.use_interaction_bias:
        ix_mlp = 4 * di + di + di * 1 + 1
    else:
        ix_mlp = 0
    per_block = attn_norm + qkv + out_proj + ff_norm + ff1 + ff2 + ix_mlp

    final_norm = 2 * d
    total = emb + model_cfg.n_layers * per_block + final_norm
    return int(total)


def cosine_lr(
    step: int,
    warmup: int,
    total: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup followed by cosine decay down to ``min_lr``."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
