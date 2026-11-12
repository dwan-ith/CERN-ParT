"""
Model and training configuration.
RTX 4050 Laptop (6.4 GB VRAM) + 16 GB RAM — max params with gradient checkpointing.

Memory budget (410M params):
  - Model params (fp32):  ~1.6 GB
  - Optimizer (AdamW):    ~3.2 GB
  - Gradients (fp16):     ~0.8 GB
  - Activations (ckpt):   ~0.2 GB
  - Total:                ~5.8 GB (of 6.4 GB available)

Data: 1,382 ROOT files from 16 CMS NanoAOD Run1 datasets (~20M events)
"""
import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    n_features: int = 7
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 24
    d_ff: int = 6144
    d_interaction: int = 128
    max_particles: int = 64
    dropout: float = 0.1
    gradient_checkpointing: bool = True

    @property
    def d_head(self):
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    batch_size: int = 8
    max_particles: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    n_steps: int = 5000
    grad_accum: int = 4
    mask_ratio: float = 0.15
    mask_loss_weight: float = 1.0
    log_every: int = 50
    save_every: int = 500
    checkpoint_dir: str = "checkpoints"
    amp: bool = True


def _load_urls():
    url_file = os.path.join(os.path.dirname(__file__), "..", "data", "all_urls.txt")
    if os.path.exists(url_file):
        with open(url_file) as f:
            return [line.strip() for line in f if line.strip()]
    return [
        "http://opendata.cern.ch/eos/opendata/cms/derived-data/NanoAODRun1/01-Jul-22/Run2010A_Mu/02A55927-7112-4A20-B06E-86E24B644B59.root",
    ]


@dataclass
class DataConfig:
    max_particles: int = 64
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0
    default_urls: list = field(default_factory=_load_urls)
