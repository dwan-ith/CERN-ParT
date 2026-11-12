"""Tests for the local cache and the event-level shuffle buffer."""
from __future__ import annotations

import os

import pytest

from data.https_dataset import (
    _EventBuffer,
    _looks_like_valid_root,
    cache_path,
    download_file,
    materialize,
)


def test_cache_path_is_deterministic_and_unique():
    a = cache_path("/tmp/cache", "http://x/y.root")
    b = cache_path("/tmp/cache", "http://x/y.root")
    c = cache_path("/tmp/cache", "http://x/z.root")
    assert a == b
    assert a != c
    assert a.endswith("y.root")


def test_download_file_with_a_local_url(tmp_path):
    """``file://`` URLs exercise the download path without the network."""
    source = tmp_path / "source.root"
    source.write_bytes(b"not really a root file")
    cache = tmp_path / "cache"

    target = download_file(source.as_uri(), str(cache), validate=False)
    assert os.path.exists(target)
    assert os.path.getsize(target) == source.stat().st_size

    # A second call must reuse the cached copy rather than download again.
    stamp = os.path.getmtime(target)
    again = download_file(source.as_uri(), str(cache), validate=False)
    assert again == target
    assert os.path.getmtime(again) == stamp


def test_download_file_rejects_an_invalid_root_file(tmp_path):
    source = tmp_path / "bad.root"
    source.write_bytes(b"garbage")
    cache = tmp_path / "cache"
    with pytest.raises(IOError):
        download_file(source.as_uri(), str(cache), validate=True)


def test_looks_like_valid_root_rejects_garbage(tmp_path):
    path = tmp_path / "x.root"
    path.write_bytes(b"garbage")
    assert _looks_like_valid_root(str(path)) is False


def test_materialize_collects_local_files(tmp_path):
    files = []
    for i in range(3):
        source = tmp_path / f"f{i}.root"
        source.write_bytes(b"x")
        files.append(source.as_uri())
    cache = tmp_path / "cache"
    paths = materialize(files, str(cache), workers=2, verbose=False, validate=False)
    assert len(paths) == 3
    assert all(os.path.exists(p) for p in paths)


def test_event_buffer_is_a_uniform_reservoir():
    """The buffer must hold a uniform sample and lose no events."""
    buffer = _EventBuffer(size=10, seed=0)
    seen = []
    for i in range(1000):
        displaced = buffer.add({"i": i})
        if displaced is not None:
            seen.append(displaced["i"])
    # Everything that was evicted plus the residual buffer accounts for
    # every event exactly once.
    drained = [item["i"] for item in buffer.drain()]
    assert sorted(seen + drained) == list(range(1000))


def test_event_buffer_disables_itself_at_size_zero():
    buffer = _EventBuffer(size=0)
    assert buffer.add({"i": 1}) is None
    assert len(buffer) == 0


def test_event_buffer_fills_before_emitting():
    buffer = _EventBuffer(size=4, seed=0)
    assert buffer.add({"i": 0}) is None
    assert buffer.add({"i": 1}) is None
    assert buffer.add({"i": 2}) is None
    assert buffer.add({"i": 3}) is None
    assert len(buffer) == 4
    # The fifth event displaces one of the buffered ones.
    assert buffer.add({"i": 4}) is not None
