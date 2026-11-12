"""Checkpointing utilities.

A checkpoint contains everything needed to resume a run bit-for-bit: the
model and head weights, the optimizer state, the AMP gradient scaler, the
step counter and the configuration that produced the run.

The optimizer state is by far the largest part of a checkpoint (two fp32
moments per parameter, i.e. 8 bytes/parameter).  It can therefore be turned
off with ``save_optimizer=False`` for runs where resumption is not needed.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    step: int,
    model: torch.nn.Module,
    head: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    loss: float = float("nan"),
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    save_optimizer: bool = True,
) -> str:
    """Write a checkpoint to ``path`` and return the path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload: Dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "loss": float(loss),
        "model_config": model_config or {},
        "training_config": training_config or {},
    }
    if head is not None:
        payload["head"] = head.state_dict()
    if optimizer is not None and save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra

    # Write to a temporary file first so that an interrupted run never
    # leaves a half-written checkpoint behind.
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    head: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and restore the requested components."""
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"], strict=strict)
    if head is not None and "head" in payload:
        head.load_state_dict(payload["head"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


def latest_checkpoint(directory: str) -> Optional[str]:
    """Return the path of the most recent ``step_*.pt`` in ``directory``."""
    if not os.path.isdir(directory):
        return None
    candidates = sorted(glob.glob(os.path.join(directory, "step_*.pt")))
    if candidates:
        return candidates[-1]
    final = os.path.join(directory, "final.pt")
    return final if os.path.exists(final) else None


def write_run_config(path: str, config: Dict[str, Any]) -> None:
    """Persist the configuration of a run next to its checkpoints."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)
