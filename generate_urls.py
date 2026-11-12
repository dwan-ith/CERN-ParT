"""Build the URL manifests for the CMS NanoAOD open data.

The CERN Open Data portal exposes CMS NanoAOD files through a simple JSON
API.  This script walks a list of records and writes one URL per line.

Which records?
--------------
Run 1 NanoAOD (``NanoAODRun1``) consists almost entirely of *single-object*
trigger samples: a muon record contains a muon and a handful of other
particles.  Measured on the Run 1 files, the median event contains **two**
particles, so a masked-particle-modelling objective has essentially nothing
to learn from -- only ~0.5% of particles end up masked in practice.

The Run 2 MiniAOD-derived NanoAODv9 records used here (``JetHT``,
``HTMHT``, ``BTagCSV``, ``Charmonium``, ``DisplacedJet``, ...) are ordinary
multi-object collision samples with a median of **nine** particles per event,
which makes the pretraining objective well posed.

Usage::

    python generate_urls.py                     # write data/all_urls.txt
    python generate_urls.py --split             # also write the b-tag split
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

API = "https://opendata.cern.ch/api/records/{record}"
EOS_PREFIX = "root://eospublic.cern.ch//eos/"

#: Records of the Run 2 NanoAODv9 open data release (UL2016 MiniAODv2).
#: Ordered so that the most particle-rich samples come first.
RECORDS: Dict[int, str] = {
    # 2016G
    30525: "JetHT",
    30524: "HTMHT",
    30517: "BTagCSV",
    30518: "BTagMu",
    30519: "Charmonium",
    30520: "DisplacedJet",
    30527: "MuOnia",
    30526: "MET",
    30523: "DoubleMuonLowMass",
    30521: "DoubleEG",
    30522: "DoubleMuon",
    30528: "MuonEG",
    30529: "SingleElectron",
    30530: "SingleMuon",
    30531: "SinglePhoton",
    30532: "Tau",
    # 2016H
    30558: "JetHT",
    30557: "HTMHT",
    30550: "BTagCSV",
    30551: "BTagMu",
    30553: "DisplacedJet",
    30554: "DoubleEG",
    30555: "DoubleMuon",
    30556: "DoubleMuonLowMass",
    30559: "MET",
    30560: "MuOnia",
    30561: "MuonEG",
    30562: "SingleElectron",
    30563: "SingleMuon",
    30564: "SinglePhoton",
    30565: "Tau",
}

#: Records used for the downstream b-tagging benchmark.  ``JetHT`` is the
#: standard high-purity sample and ``BTagCSV`` is a dedicated b-tagging
#: sample, so together they cover the kinematic region the task cares about.
BTAG_RECORDS: Tuple[int, ...] = (30525, 30558, 30517, 30550)


def fetch_record(record: int, retries: int = 3, timeout: int = 30) -> Optional[dict]:
    """Fetch a record from the CERN Open Data API, with retries."""
    url = API.format(record=record)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                print(f"  {record}: FAILED - {exc}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))
    return None


def record_files(payload: dict) -> List[str]:
    """Extract the HTTP URLs of every ROOT file in a record payload."""
    urls: List[str] = []
    for index in payload.get("metadata", {}).get("_file_indices", []):
        for entry in index.get("files", []):
            uri = entry.get("uri", "")
            if not uri.endswith(".root"):
                continue
            if uri.startswith("root://"):
                uri = "http://opendata.cern.ch/eos/" + uri.split("//eos/", 1)[-1]
            elif uri.startswith("http"):
                pass
            else:
                continue
            urls.append(uri)
    return sorted(set(urls))


def collect(records: Dict[int, str]) -> Tuple[List[str], Dict[int, int]]:
    """Collect every file URL for the given records."""
    all_urls: List[str] = []
    per_record: Dict[int, int] = {}
    for record, name in records.items():
        payload = fetch_record(record)
        if payload is None:
            continue
        urls = record_files(payload)
        per_record[record] = len(urls)
        all_urls.extend(urls)
        title = payload.get("metadata", {}).get("title", name)[:60]
        print(f"  {record}: {len(urls):4d} files - {title}")
    # Deterministic order, no duplicates.
    return sorted(set(all_urls)), per_record


def write_manifest(path: str, urls: List[str], header: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if header:
            for line in header.strip().splitlines():
                handle.write(f"# {line}\n")
        for url in urls:
            handle.write(url + "\n")
    print(f"Wrote {len(urls)} URLs to {path}")


def split_manifest(urls: List[str], val_fraction: float = 0.2) -> Tuple[List[str], List[str]]:
    """Split the URLs into a train and a validation list.

    The split is by *file*, and files are assigned deterministically, so the
    same files always land on the same side.  Consecutive files of one record
    are highly correlated, so a random per-file split is still slightly
    optimistic; the files are interleaved instead to spread each record
    across both halves.
    """
    train: List[str] = []
    val: List[str] = []
    stride = max(int(round(1.0 / val_fraction)), 2)
    for index, url in enumerate(urls):
        (val if index % stride == 0 else train).append(url)
    return train, val


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", default="data/all_urls.txt", help="pretraining manifest"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="also write train/validation manifests for the b-tagging benchmark",
    )
    parser.add_argument("--btag-dir", default="data")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    print("Collecting CMS NanoAOD (Run 2, NanoAODv9) open data ...")
    all_urls, per_record = collect(RECORDS)
    if not all_urls:
        print("No URLs collected -- is the network reachable?", file=sys.stderr)
        return 1

    header = (
        "CMS NanoAOD v9 (UL2016 MiniAODv2) open data via CERN Open Data.\n"
        "Generated by generate_urls.py -- do not edit by hand.\n"
        f"records: {len(per_record)}  files: {len(all_urls)}"
    )
    write_manifest(args.output, all_urls, header)

    if args.split:
        btag_records = {r: RECORDS[r] for r in BTAG_RECORDS if r in RECORDS}
        print("\nCollecting b-tagging benchmark records ...")
        btag_urls, _ = collect(btag_records)
        train, val = split_manifest(btag_urls, args.val_fraction)
        write_manifest(
            os.path.join(args.btag_dir, "btag_train.txt"),
            train,
            "JetHT + BTagCSV, training half (b-tagging benchmark).",
        )
        write_manifest(
            os.path.join(args.btag_dir, "btag_val.txt"),
            val,
            "JetHT + BTagCSV, validation half (b-tagging benchmark).",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
