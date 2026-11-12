"""Model components: encoder, pretraining head and downstream task head."""

from .config import (
    MASK_CHANNELS,
    N_MODEL_FEATURES,
    N_RAW_FEATURES,
    OBJECT_TYPES,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    count_parameters,
    estimate_memory,
    model_config,
)
from .particle_transformer import (
    FeatureNormalizer,
    InteractionFeatureComputer,
    ParticleTransformer,
)
from .pretraining_head import MaskedParticleModeling
from .task_head import JetTaggingHead, roc_auc

__all__ = [
    "MASK_CHANNELS",
    "N_MODEL_FEATURES",
    "N_RAW_FEATURES",
    "OBJECT_TYPES",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "FeatureNormalizer",
    "InteractionFeatureComputer",
    "ParticleTransformer",
    "MaskedParticleModeling",
    "JetTaggingHead",
    "count_parameters",
    "estimate_memory",
    "model_config",
    "roc_auc",
]
