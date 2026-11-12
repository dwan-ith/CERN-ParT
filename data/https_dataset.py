"""Streaming CMS NanoAOD datasets over plain HTTPS.

Why HTTP instead of XRootD?
---------------------------
XRootD is painful to install on Windows.  ``uproot`` can read the very same
NanoAOD files over plain HTTP from the CERN Open Data portal, which makes the
whole project runnable with ``pip install -r requirements.txt`` and nothing
else.  This module is the reason the project is usable on a laptop.

Every NanoAOD file contains several TTrees (``Events``, ``LuminosityBlocks``,
``Runs``, ...).  Only ``Events`` holds the physics objects, so the tree name
is always passed explicitly; ``uproot`` otherwise raises a ``ValueError``
because it cannot guess.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .feature_extractor import extract_batch, get_optional_branches, get_required_branches


class _EventBuffer:
    """Reservoir-sample shuffle buffer over individual events.

    Shuffling at *event* granularity (rather than file granularity) matters:
    the files in a single open-data record are highly correlated, so training
    on them in file order produces correlated batches.

    This is textbook reservoir sampling.  The buffer always holds a uniform
    random sample of the events seen so far, and every incoming event
    displaces a uniformly chosen buffered event with probability
    ``size / seen``.  The displaced event is handed back to the caller and
    emitted, so the buffer never stalls waiting to fill up -- important for
    a streaming dataset.
    """

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = max(size, 0)
        self._items: List[Dict[str, np.ndarray]] = []
        self._seen = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Add an event; return the event it displaced (if any)."""
        if self.size == 0:
            return None
        self._seen += 1
        if len(self._items) < self.size:
            self._items.append(item)
            return None
        idx = self._rng.randrange(self._seen)
        if idx < self.size:
            displaced = self._items[idx]
            self._items[idx] = item
            return displaced
        return None

    def drain(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield and remove all buffered events, in random order."""
        while self._items:
            idx = self._rng.randrange(len(self._items))
            yield self._items.pop(idx)


def _collate(events: Sequence[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    return {
        "particles": torch.from_numpy(np.stack([e["particles"] for e in events])),
        "mask": torch.from_numpy(np.stack([e["mask"] for e in events])),
        "is_jet": torch.from_numpy(np.stack([e["is_jet"] for e in events])),
        "btag": torch.from_numpy(np.stack([e["btag"] for e in events])),
    }


class CMSNanoAODDataset(IterableDataset):
    """Iterate over CMS NanoAOD events, batch by batch, streamed over HTTPS.

    Args:
        urls: HTTP(S) URLs of NanoAOD ROOT files.
        max_particles: particles retained per event (highest pT first).
        batch_size: events per yielded batch.
        shuffle: shuffle the order of the input files.
        shuffle_buffer: number of events buffered for shuffling. ``0``
            disables event-level shuffling.
        tree: name of the TTree to read.
        use_pf_candidates: use packed PF candidates when the file has them.
        max_files: only open the first ``max_files`` files (smoke tests).
        seed: RNG seed for shuffling.
    """

    def __init__(
        self,
        urls: Sequence[str],
        max_particles: int = 64,
        batch_size: int = 16,
        shuffle: bool = True,
        shuffle_buffer: int = 2000,
        tree: str = "Events",
        use_pf_candidates: bool = True,
        max_files: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.urls = list(urls)
        if max_files is not None:
            self.urls = self.urls[:max_files]
        self.max_particles = max_particles
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.tree = tree
        self.use_pf_candidates = use_pf_candidates
        self.seed = seed

        #: Running statistics, useful for sanity checks and for the README.
        self.stats: Dict[str, float] = {
            "events": 0.0,
            "chunks": 0.0,
            "particles": 0.0,
        }

    # -- helpers ---------------------------------------------------------
    def _sharded_urls(self) -> List[str]:
        urls = list(self.urls)
        if self.shuffle:
            random.Random(self.seed).shuffle(urls)
        info = get_worker_info()
        if info is None or info.num_workers <= 1:
            return urls
        # Round-robin so that every worker sees a mix of datasets.
        return urls[info.id :: info.num_workers]

    def _iterate_files(self, urls: List[str]) -> Iterator:
        """Yield awkward arrays, tolerating single-tree files."""
        wanted = get_required_branches(self.use_pf_candidates)
        wanted += get_optional_branches()
        try:
            for arrays in uproot_iterate(urls, self.tree, wanted):
                yield arrays
        except ValueError:
            # File has a single TTree with a different name: retry without
            # pinning the tree.
            for arrays in uproot_iterate(urls, None, wanted):
                yield arrays

    # -- iteration -------------------------------------------------------
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        urls = self._sharded_urls()
        buffer = _EventBuffer(self.shuffle_buffer, seed=self.seed)
        pending: List[Dict[str, np.ndarray]] = []

        for arrays in self._iterate_files(urls):
            self.stats["chunks"] += 1
            try:
                batch = extract_batch(
                    arrays,
                    max_particles=self.max_particles,
                    use_pf_candidates=self.use_pf_candidates,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[CMSNanoAODDataset] skipping a batch: {exc}")
                continue

            n_events = batch["particles"].shape[0]
            self.stats["events"] += n_events
            self.stats["particles"] += float(batch["n_particles"].sum())

            for i in range(n_events):
                event = {key: value[i] for key, value in batch.items()}
                if self.shuffle_buffer > 0:
                    displaced = buffer.add(event)
                    if displaced is None:
                        continue
                    pending.append(displaced)
                else:
                    pending.append(event)
                if len(pending) == self.batch_size:
                    yield _collate(pending)
                    pending = []

        # Flush whatever is left.
        if self.shuffle_buffer > 0:
            for item in buffer.drain():
                pending.append(item)
                if len(pending) == self.batch_size:
                    yield _collate(pending)
                    pending = []
        if pending:
            yield _collate(pending)

    def __len__(self) -> int:
        """Number of batches, estimated from the file count.

        The exact count is not knowable up front because it depends on the
        file contents, so this is an estimate used only for progress bars.
        """
        return max(1, len(self.urls) * 1000 // max(self.batch_size, 1))


def uproot_iterate(urls: Sequence[str], tree: Optional[str], branches: Sequence[str]):
    """``uproot.iterate`` wrapper that pins the TTree when possible."""
    import uproot

    if tree:
        spec = {url: tree for url in urls}
    else:
        spec = list(urls)
    return uproot.iterate(spec, filter_name=list(branches), step_size=1000)


def create_dataloader(
    urls: Sequence[str],
    max_particles: int = 64,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_buffer: int = 2000,
    tree: str = "Events",
    seed: int = 0,
) -> IterableDataset:
    """Return a dataset that yields *pre-batched* tensors.

    The dataset batches internally, so it must **not** be wrapped in a
    ``DataLoader`` with a batch size; use ``num_workers`` on the dataset
    itself via :func:`torch.utils.data.DataLoader` with ``batch_size=None``.
    """
    return CMSNanoAODDataset(
        urls,
        max_particles=max_particles,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        tree=tree,
        seed=seed,
    )
