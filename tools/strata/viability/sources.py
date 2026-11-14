"""Strata viability: the extended decision set of gate V7 (`05-gate-amendments.md`, amendment 4).

    python -I tools/strata/viability/sources.py            # inventory, role check, overlap with the frozen test set
    python -I tools/strata/viability/sources.py --fetch    # download what is missing (public files, generic agent)

Seven slices of 1 MiB: two text sources, three log systems, two structured sources. They are DECISION files: no
Strata model may be evaluated on them before the ladder configuration is registered (`08-capacity-ladder.md`);
classical codecs may. No slice shares a source with a training or selection source of `bakeoff/common.py`:
`check_roles()` verifies the paths and the Loghub system names and is run by every script that loads this set.

Downloads follow `bakeoff/fetch_data.py`: plain GET, generic User-Agent, nothing else is sent.
  bible.txt                 Canterbury "large" corpus (the archive also holds world192.txt, a training file: a
                            different document)
  Apache / Linux / OpenSSH  first 2,000,000 bytes of the full logs of the public Loghub record that the training
                            logs come from (the OpenSSH log is the member "SSH" of that record)
"""
import hashlib
import io
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bakeoff"))
from common import ROOT, SOURCES  # noqa: E402

EXT_BYTES = 1 << 20
LOG_CAP = 2_000_000
# name: (class, path relative to ROOT, offset, length)
EXTENDED = {
    "ext-dickens": ("text", "data/benchmark/silesia/dickens", 500_000, EXT_BYTES),    # frozen test = [0, 500000)
    "ext-bible": ("text", "data/strata/raw/canterbury/bible.txt", 0, EXT_BYTES),
    "ext-apache": ("logs", "data/strata/raw/loghub/Apache.log", 0, EXT_BYTES),
    "ext-linux": ("logs", "data/strata/raw/loghub/Linux.log", LOG_CAP - EXT_BYTES, EXT_BYTES),
    "ext-openssh": ("logs", "data/strata/raw/loghub/OpenSSH.log", LOG_CAP - EXT_BYTES, EXT_BYTES),
    "ext-gharchive": ("structured", "data/strata/raw/gharchive/2015-01-01-15.json", 0, EXT_BYTES),
    "ext-nci": ("structured", "data/benchmark/silesia/nci", 0, EXT_BYTES),
}
# Amendment 5 (2026-11-14): the first MiB of the full Linux and OpenSSH logs holds every line of the frozen-test
# samples, so both slices are the last MiB of the 2,000,000 bytes on disk (no shared line). No alternative is open.
ALTERNATIVES = {}
LOGHUB_MEMBER = {"Apache": "Apache", "Linux": "Linux", "OpenSSH": "SSH"}
CANTERBURY_LARGE = "https://corpus.canterbury.ac.nz/resources/large.zip"
# frozen-test files of the same source as an extended slice (same role; the overlap is reported, not hidden)
SAME_SOURCE_TEST = {"ext-dickens": "test-dickens", "ext-apache": "test-apache", "ext-linux": "test-linux",
                    "ext-openssh": "test-openssh"}


def is_decision(name):
    """True for every name a Strata model must not be evaluated on without an explicit override."""
    return name in EXTENDED or name in ALTERNATIVES or (name in SOURCES and SOURCES[name][0] != "select")


def load_slice(spec):
    _, rel, off, length = spec
    with open(os.path.join(ROOT, rel), "rb") as f:
        f.seek(off)
        data = f.read(length)
    if len(data) != length:
        raise SystemExit(f"{rel}: {len(data):,d} bytes available at offset {off}, {length:,d} needed")
    return np.frombuffer(data, dtype=np.uint8).copy()


def load_extended(name):
    return load_slice(EXTENDED[name])


def check_roles():
    """No extended slice may use the file of a training or selection source; returns the list of checked pairs."""
    used = {os.path.normcase(os.path.normpath(v[2])): n for n, v in SOURCES.items() if v[0] in ("train", "select")}
    checked = []
    for name, (_, rel, _, _) in {**EXTENDED, **ALTERNATIVES}.items():
        key = os.path.normcase(os.path.normpath(rel))
        if key in used:
            raise SystemExit(f"{name} uses the file of the {SOURCES[used[key]][0]} source {used[key]}")
        checked.append((name, rel))
    return checked


def fetch():
    from fetch_data import ZENODO, log, open_url, stream_tar_logs
    for system, member in LOGHUB_MEMBER.items():
        dst = os.path.join(ROOT, "data", "strata", "raw", "loghub", system + ".log")
        if os.path.isfile(dst) and os.path.getsize(dst) >= EXT_BYTES:
            log(f"  loghub {system:8s} present ({os.path.getsize(dst):,d} B)")
            continue
        got = stream_tar_logs(ZENODO.format(member), LOG_CAP, dst)
        log(f"  loghub {system:8s} {got:>10,d} B (record member {member})")
    dst = os.path.join(ROOT, "data", "strata", "raw", "canterbury", "bible.txt")
    if os.path.isfile(dst):
        log(f"  canterbury bible.txt present ({os.path.getsize(dst):,d} B)")
        return
    with open_url(CANTERBURY_LARGE, 60) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        data = z.read("bible.txt")
    with open(dst, "wb") as f:
        f.write(data)
    log(f"  canterbury bible.txt {len(data):,d} B")


def line_overlap(name):
    """Share of the lines of the same-source frozen-test file that also occur in the extended slice."""
    from common import load_source
    ext = set(bytes(load_extended(name)).split(b"\n"))
    lines = [ln for ln in bytes(load_source(SAME_SOURCE_TEST[name])).split(b"\n") if ln]
    return sum(ln in ext for ln in lines), len(lines)


def main():
    if "--fetch" in sys.argv:
        fetch()
    for name, rel in check_roles():
        spec = {**EXTENDED, **ALTERNATIVES}[name]
        cls, _, off, length = spec
        data = load_slice(spec)
        print(f"{name:14s} {cls:10s} {rel} [{off:,d}, {off + length:,d})  sha256[:16]="
              f"{hashlib.sha256(data.tobytes()).hexdigest()[:16]}", flush=True)
    print("role check: no extended slice uses the file of a training or selection source")
    for name in SAME_SOURCE_TEST:
        hit, total = line_overlap(name)
        print(f"{name}: {hit} of the {total} lines of {SAME_SOURCE_TEST[name]} occur in the slice (same source)")


if __name__ == "__main__":
    main()
