"""
Streaming CMS NanoAOD Dataset via HTTPS (Windows-native, no XRootD needed).

Usage:
    urls = ["https://opendata.cern.ch/record/12350/files/SMHiggsToZZTo4L.root"]
    dataset = CMSNanoAODDataset(urls, max_particles=64, batch_size=16)
    for batch in dataset:
        particles = batch["particles"]  # [B, 64, 7]
        mask = batch["mask"]            # [B, 64]
"""
import torch
from torch.utils.data import IterableDataset
import uproot
import numpy as np
from .feature_extractor import extract_batch, get_required_branches


class CMSNanoAODDataset(IterableDataset):
    """
    Streams CMS NanoAOD ROOT files via HTTPS using uproot.
    Only loads the branches needed for particle feature extraction.
    Yields batches of padded particle tensors ready for the model.
    """

    def __init__(self, urls, max_particles=64, batch_size=16, shuffle=True):
        self.urls = urls
        self.max_particles = max_particles
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.branches = get_required_branches()

    def __iter__(self):
        urls = self.urls.copy()
        if self.shuffle:
            import random
            random.shuffle(urls)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Shard URLs across workers
            per_worker = len(urls) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(urls)
            urls = urls[start:end]

        for arrays in uproot.iterate(
            urls,
            filter_name=self.branches,
            step_size=self.batch_size,
        ):
            try:
                batch = extract_batch(arrays, self.max_particles)
                yield {
                    "particles": torch.from_numpy(batch["particles"]),
                    "mask": torch.from_numpy(batch["mask"]),
                }
            except Exception as e:
                print(f"[CMSNanoAODDataset] skipping batch: {e}")
                continue

    def __len__(self):
        return max(1, len(self.urls) * 1000 // self.batch_size)


def create_dataloader(urls, max_particles=64, batch_size=16, num_workers=0):
    """
    Convenience: returns an IterableDataset that yields pre-batched tensors.
    Use directly in a for-loop (not DataLoader, which re-batches).
    """
    return CMSNanoAODDataset(urls, max_particles=max_particles, batch_size=batch_size)
