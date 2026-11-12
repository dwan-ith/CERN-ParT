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

import hashlib
import os
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .feature_extractor import extract_batch, get_optional_branches, get_required_branches


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str, url: str) -> str:
    """Deterministic local path for a remote file."""
    name = url.rsplit("/", 1)[-1]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"{digest}_{name}")


def cached_events_path(
    url: str,
    cache_dir: str,
    max_particles: int = 64,
    use_pf_candidates: bool = True,
) -> str:
    """Deterministic path of the cached, extracted events for a URL."""
    key = hashlib.sha1(
        f"{url}|{max_particles}|{int(use_pf_candidates)}".encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(cache_dir, f"{key}.npz")


def extract_and_cache(
    url: str,
    cache_dir: str,
    max_particles: int = 64,
    use_pf_candidates: bool = True,
    tree: str = "Events",
    step_size: int = 2000,
    timeout: int = 600,
) -> str:
    """Stream a remote NanoAOD file and store the *extracted* events locally.

    Caching the extracted tensors rather than the ROOT file is a much better
    trade:

    * the ROOT file is ~140 MB, the extracted events are a few MB;
    * ``uproot`` is not needed at all on a cached re-run, so the data loader
      stops being the bottleneck entirely;
    * the cached artefact is independent of ROOT's tree layout.

    The cache key includes the particle limit, so changing ``max_particles``
    invalidates the cache instead of silently returning the wrong tensors.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = cached_events_path(url, cache_dir, max_particles, use_pf_candidates)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    wanted = _branch_list(use_pf_candidates)
    tmp = f"{path}.part.npz"

    particles: List[np.ndarray] = []
    mask: List[np.ndarray] = []
    is_jet: List[np.ndarray] = []
    btag: List[np.ndarray] = []

    for arrays in uproot_iterate([url], tree, wanted, timeout=timeout):
        try:
            batch = extract_batch(
                arrays,
                max_particles=max_particles,
                use_pf_candidates=use_pf_candidates,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[cache] skipping a chunk of {url}: {exc}", flush=True)
            continue
        particles.append(batch["particles"])
        mask.append(batch["mask"])
        is_jet.append(batch["is_jet"])
        btag.append(batch["btag"])

    if not particles:
        raise IOError(f"no events could be extracted from {url}")

    np.savez_compressed(
        tmp,
        particles=np.concatenate(particles, axis=0),
        mask=np.concatenate(mask, axis=0),
        is_jet=np.concatenate(is_jet, axis=0),
        btag=np.concatenate(btag, axis=0),
    )
    os.replace(tmp, path)
    return path


def _branch_list(use_pf_candidates: bool = True) -> List[str]:
    """The branches the extractor needs, plus the optional b-tagging ones."""
    branches = get_required_branches(use_pf_candidates)
    branches += get_optional_branches()
    return branches


def content_length(url: str, timeout: int = 30) -> Optional[int]:
    """Expected size of a remote file, or ``None`` if it cannot be determined."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def _looks_like_valid_root(path: str) -> bool:
    """Check that a downloaded file really is a readable NanoAOD file.

    The CERN endpoint occasionally truncates a transfer.  A truncated ROOT
    file does not fail at download time -- it fails much later, deep inside
    the training loop, with a confusing error.  Validating the file here
    turns that into an immediate, actionable retry.
    """
    try:
        import uproot

        with uproot.open(path) as handle:
            keys = list(handle.keys())
        return "Events" in keys
    except Exception:
        return False


def download_file(
    url: str,
    cache_dir: str,
    timeout: int = 300,
    retries: int = 4,
    validate: bool = True,
) -> str:
    """Download ``url`` into ``cache_dir`` and return the local path.

    ROOT files are not random-access readable over HTTP: ``uproot`` has to
    stream the whole file even when only a handful of branches are
    requested.  Caching the file locally therefore turns a stream-per-batch
    workload into a stream-per-file one, and it turns a network-bound
    pipeline into a GPU-bound one.

    The CERN endpoint throttles aggressively and truncates transfers, so the
    download is resumable (HTTP ``Range``) and the received size is compared
    against the size the server announced.  A truncated file is otherwise
    indistinguishable from a good one until it explodes deep inside the
    training loop.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, url)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        if not validate or _looks_like_valid_root(path):
            return path
        os.remove(path)

    tmp = f"{path}.part"
    total = 0

    for attempt in range(retries):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if have and getattr(response, "status", 0) == 206:
                    content_range = response.headers.get("Content-Range", "")
                    try:
                        total = int(content_range.rsplit("/", 1)[-1])
                    except (ValueError, IndexError):
                        total = 0
                    mode = "ab"
                else:
                    total = int(response.headers.get("Content-Length") or 0)
                    mode = "wb"
                    have = 0
                with open(tmp, mode) as handle:
                    shutil.copyfileobj(response, handle, length=1 << 20)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[cache] transfer error ({exc}); retrying", flush=True)
            continue

        size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if total and size >= total:
            break
        if not total and size > 0 and not validate:
            break
        print(
            f"[cache] incomplete transfer ({size}/{total or '?'} bytes); resuming",
            flush=True,
        )

    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise IOError(f"could not download {url}")

    if validate and not _looks_like_valid_root(tmp):
        # Keep the partial file: the next attempt resumes from here.
        raise IOError(f"downloaded file is not a valid NanoAOD file yet: {url}")

    os.replace(tmp, path)
    return path


class _EventCachePrefetcher:
    """Download and extract files in the background, a bounded window ahead.

    The extraction is the expensive part (it reads the remote file over
    HTTP), so it is overlapped with the training loop: while the model is
    working on the events of the current file, the next few files are being
    fetched and converted.

    The window is deliberately bounded.  The full manifest is 1214 files, and
    a training run only ever touches a small fraction of them.
    """

    def __init__(
        self,
        urls: Sequence[str],
        cache_dir: str,
        max_particles: int = 64,
        use_pf_candidates: bool = True,
        tree: str = "Events",
        workers: int = 2,
        window: int = 4,
    ) -> None:
        self.urls = list(urls)
        self.cache_dir = cache_dir
        self.max_particles = max_particles
        self.use_pf_candidates = use_pf_candidates
        self.tree = tree
        self.workers = max(workers, 1)
        self.window = max(window, 1)

    def _fetch(self, url: str) -> str:
        return extract_and_cache(
            url,
            self.cache_dir,
            max_particles=self.max_particles,
            use_pf_candidates=self.use_pf_candidates,
            tree=self.tree,
        )

    def __iter__(self) -> Iterator[str]:
        pool = ThreadPoolExecutor(max_workers=self.workers)
        futures: List = []
        index = 0
        try:
            while index < len(self.urls) and len(futures) < self.window:
                futures.append(pool.submit(self._fetch, self.urls[index]))
                index += 1
            while futures:
                future = futures.pop(0)
                try:
                    local = future.result()
                except Exception as exc:  # pragma: no cover - network dependent
                    print(f"\n[cache] skipping file: {exc}", flush=True)
                    local = ""
                if index < len(self.urls):
                    futures.append(pool.submit(self._fetch, self.urls[index]))
                    index += 1
                if local:
                    yield local
        finally:
            pool.shutdown(wait=False)


def materialize(
    urls: Sequence[str],
    cache_dir: str,
    workers: int = 8,
    desc: str = "downloading",
    verbose: bool = True,
    validate: bool = True,
) -> List[str]:
    """Download every URL in parallel and return the local paths."""
    os.makedirs(cache_dir, exist_ok=True)
    paths: List[str] = []
    pending: List[str] = []
    for url in urls:
        local = cache_path(cache_dir, url)
        if os.path.exists(local) and os.path.getsize(local) > 0:
            paths.append(local)
        else:
            pending.append(url)

    if not pending:
        return paths

    done = [0]

    def fetch(url: str) -> str:
        try:
            local = download_file(url, cache_dir, validate=validate)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"\n[cache] FAILED {url}: {exc}", flush=True)
            return ""
        done[0] += 1
        if verbose:
            print(
                f"\r[{desc}] {done[0]}/{len(pending)} files",
                end="",
                flush=True,
            )
        return local

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        for local in pool.map(fetch, pending):
            if local:
                paths.append(local)
    if verbose:
        print()
    return paths


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
        """Add an event; return the event that should be emitted next.

        The buffer holds a uniform random sample of the events seen so far.
        When the new event displaces a buffered one, the *displaced* event is
        returned; when it does not, the *new* event is returned instead.
        Either way every event is emitted exactly once -- dropping it would
        silently shrink the training set.
        """
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
        return item

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
        cache_dir: Optional[str] = None,
        download_workers: int = 4,
        prefetch_window: int = 8,
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
        self.cache_dir = cache_dir
        self.download_workers = download_workers
        self.prefetch_window = prefetch_window

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

    def _file_source(self) -> Iterator[str]:
        """Yield the remote files to stream, in a shuffled order."""
        return iter(self._sharded_urls())

    # -- event sources --------------------------------------------------
    def _stream_events(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield single events straight from the remote ROOT files."""
        for path in self._file_source():
            for arrays in self._iterate_files([path]):
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
                    yield {key: value[i] for key, value in batch.items()}

    def _cached_events(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield single events from the locally cached extractions."""
        urls = list(self.urls)
        if self.shuffle:
            random.Random(self.seed).shuffle(urls)
        source = _EventCachePrefetcher(
            urls,
            self.cache_dir,
            max_particles=self.max_particles,
            use_pf_candidates=self.use_pf_candidates,
            tree=self.tree,
            workers=self.download_workers,
            window=self.prefetch_window,
        )
        for path in source:
            with np.load(path) as data:
                # ``NpzFile`` decompresses lazily, so every key has to be
                # materialised exactly once -- indexing it inside the event
                # loop would re-inflate the whole array for every event.
                arrays = {key: data[key] for key in data.files}
                n_events = arrays["particles"].shape[0]
                self.stats["events"] += n_events
                self.stats["particles"] += float(arrays["mask"].sum())
                for i in range(n_events):
                    yield {key: value[i] for key, value in arrays.items()}

    def _batched(
        self, events: Iterator[Dict[str, np.ndarray]]
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Group events into batches, shuffling at the event level."""
        buffer = _EventBuffer(self.shuffle_buffer, seed=self.seed)
        pending: List[Dict[str, np.ndarray]] = []
        for event in events:
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
        if self.shuffle_buffer > 0:
            for item in buffer.drain():
                pending.append(item)
                if len(pending) == self.batch_size:
                    yield _collate(pending)
                    pending = []
        if pending:
            yield _collate(pending)

    # -- iteration -------------------------------------------------------
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.cache_dir is not None:
            yield from self._batched(self._cached_events())
        else:
            yield from self._batched(self._stream_events())

    def __len__(self) -> int:
        """Number of batches, estimated from the file count.

        The exact count is not knowable up front because it depends on the
        file contents, so this is an estimate used only for progress bars.
        """
        return max(1, len(self.urls) * 1000 // max(self.batch_size, 1))


def uproot_iterate(
    urls: Sequence[str],
    tree: Optional[str],
    branches: Sequence[str],
    timeout: int = 600,
):
    """``uproot.iterate`` wrapper that pins the TTree when possible."""
    import uproot

    if tree:
        spec = {url: tree for url in urls}
    else:
        spec = list(urls)
    return uproot.iterate(
        spec, filter_name=list(branches), step_size=1000, timeout=timeout
    )


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
