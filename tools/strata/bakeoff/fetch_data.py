"""Strata bake-off: download the public training / selection corpora into data/strata/raw/.

Resumable: every output that already exists with the expected size is skipped.
Privacy: plain GET requests to public URLs with a generic User-Agent ("Wget/1.21"); nothing else is sent.

Usage (from the repo root):   python -I tools/strata/bakeoff/fetch_data.py [--only wiki|loghub|canterbury]

Sources (see split.py for the role of each one):
  wiki        bytes [100,000,000, 300,000,000) of enwik9 (the first 10^8 bytes ARE enwik8 and are verified and dropped)
  loghub      first 16 MB of the full log of 11 systems (train) and first 2 MB of Zookeeper / Proxifier (selection)
  canterbury  non-literary text files of the Canterbury "large" and "cantrbry" sets (train, breadth only)
  gharchive   one hour of public GitHub events, JSON (evaluation only: short-message protocol of R-113)
"""
import hashlib
import io
import os
import sys
import tarfile
import time
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "strata", "raw")
ENWIK8 = os.path.join(ROOT, "data", "benchmark", "enwik8")

ENWIK9_URL = "https://mattmahoney.net/dc/enwik9.zip"
WIKI_LO, WIKI_HI = 100_000_000, 300_000_000
ZENODO = "https://zenodo.org/api/records/3227177/files/{}.tar.gz/content"
LOG_TRAIN = ["BGL", "HDFS_1", "Hadoop", "Spark", "HPC", "Mac", "Android", "Windows", "Thunderbird",
             "OpenStack", "HealthApp"]
LOG_SELECT = ["Zookeeper", "Proxifier"]
LOG_TRAIN_CAP = 16_000_000
LOG_SELECT_CAP = 2_000_000
CANTERBURY = {
    "https://corpus.canterbury.ac.nz/resources/large.zip": ["world192.txt"],
    "https://corpus.canterbury.ac.nz/resources/cantrbry.zip": ["lcet10.txt", "cp.html", "fields.c", "grammar.lsp",
                                                                "xargs.1"],
}


UA = {"User-Agent": "Wget/1.21"}   # generic; one of the hosts rejects other generic agents with 403


def log(msg):
    print(msg, flush=True)


def open_url(url, timeout, headers=None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        while True:
            n = 1 << 20 if left is None else min(1 << 20, left)
            if n == 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def download(url, dst):
    """Plain resumable download (HTTP Range when a partial file exists)."""
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    t0 = time.time()
    hdr = {"Range": f"bytes={have}-"} if have else None
    with open_url(url, 60, hdr) as r, open(part, "ab" if have else "wb") as f:
        if have and r.status != 206:
            f.truncate(0)
            have = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(part, dst)
    log(f"    downloaded {os.path.getsize(dst):,d} B in {time.time() - t0:.0f} s")


def fetch_wiki():
    out_dir = os.path.join(RAW, "wiki")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "enwik9-100M-300M.bin")
    if os.path.isfile(dst) and os.path.getsize(dst) == WIKI_HI - WIKI_LO:
        log(f"  wiki: present ({os.path.getsize(dst):,d} B)")
        return
    zpath = os.path.join(out_dir, "enwik9.zip")
    if not os.path.isfile(zpath):
        log("  wiki: downloading enwik9.zip")
        download(ENWIK9_URL, zpath)
    log("  wiki: extracting, verifying that the first 10^8 bytes equal the local enwik8")
    h = hashlib.sha256()
    with zipfile.ZipFile(zpath) as z, z.open("enwik9") as src, open(dst + ".part", "wb") as out:
        pos = 0
        while pos < WIKI_HI:
            b = src.read(min(1 << 22, WIKI_HI - pos))
            if not b:
                break
            lo, hi = pos, pos + len(b)
            if lo < WIKI_LO:
                h.update(b[:max(0, min(hi, WIKI_LO) - lo)])
            if hi > WIKI_LO:
                out.write(b[max(0, WIKI_LO - lo):])
            pos = hi
    head = h.hexdigest()
    local = sha256_file(ENWIK8) if os.path.isfile(ENWIK8) else "(local enwik8 missing)"
    log(f"    sha256 enwik9[0:1e8] = {head}")
    log(f"    sha256 local enwik8  = {local}")
    if head != local:
        os.remove(dst + ".part")
        raise SystemExit("enwik9 head does not match the local enwik8: refusing to build the wiki training slice")
    os.replace(dst + ".part", dst)
    os.remove(zpath)
    log(f"    wrote {dst} ({os.path.getsize(dst):,d} B); the first 10^8 bytes (= enwik8) were dropped; zip removed")


def stream_tar_logs(url, cap, dst):
    """First `cap` bytes of the concatenated regular-file members of a .tar.gz, streamed (never stored whole)."""
    got = 0
    with open_url(url, 120) as r, open(dst + ".part", "wb") as out:
        with tarfile.open(fileobj=r, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile() or m.size == 0:
                    continue
                name = m.name.lower()
                if not (name.endswith(".log") or name.endswith(".txt") or "." not in os.path.basename(name)):
                    continue
                f = tar.extractfile(m)
                while got < cap:
                    b = f.read(min(1 << 20, cap - got))
                    if not b:
                        break
                    out.write(b)
                    got += len(b)
                if got >= cap:
                    break
    os.replace(dst + ".part", dst)
    return got


def fetch_loghub():
    out_dir = os.path.join(RAW, "loghub")
    os.makedirs(out_dir, exist_ok=True)
    for system, cap in [(s, LOG_TRAIN_CAP) for s in LOG_TRAIN] + [(s, LOG_SELECT_CAP) for s in LOG_SELECT]:
        dst = os.path.join(out_dir, system + ".log")
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            log(f"  loghub {system:12s} present ({os.path.getsize(dst):,d} B)")
            continue
        t0 = time.time()
        try:
            got = stream_tar_logs(ZENODO.format(system), cap, dst)
            log(f"  loghub {system:12s} {got:>11,d} B in {time.time() - t0:.0f} s")
        except Exception as e:  # report and continue: a missing system is recorded, not hidden
            log(f"  loghub {system:12s} FAILED: {type(e).__name__}: {e}")


def fetch_canterbury():
    out_dir = os.path.join(RAW, "canterbury")
    os.makedirs(out_dir, exist_ok=True)
    for url, names in CANTERBURY.items():
        if all(os.path.isfile(os.path.join(out_dir, n)) for n in names):
            log(f"  canterbury: {', '.join(names)} present")
            continue
        with open_url(url, 60) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for n in names:
                data = z.read(n)
                with open(os.path.join(out_dir, n), "wb") as f:
                    f.write(data)
                log(f"  canterbury {n:14s} {len(data):>9,d} B")


def fetch_gharchive():
    """One hour of public GitHub events (JSON lines): the JSON source of the R-113 short-message protocol."""
    import gzip
    out_dir = os.path.join(RAW, "gharchive")
    os.makedirs(out_dir, exist_ok=True)
    for hour in ("2015-01-01-15", "2015-01-02-15"):     # the second hour is the foreign primer of G0
        dst = os.path.join(out_dir, hour + ".json")
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            log(f"  gharchive {hour}: present ({os.path.getsize(dst):,d} B)")
            continue
        with open_url(f"https://data.gharchive.org/{hour}.json.gz", 120) as r:
            blob = gzip.decompress(r.read())[:4_000_000]
        with open(dst, "wb") as f:
            f.write(blob)
        log(f"  gharchive {hour}: {len(blob):,d} B (first 4 MB of the decompressed hour)")


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(RAW, exist_ok=True)
    if only in (None, "gharchive"):
        fetch_gharchive()
    if only in (None, "canterbury"):
        fetch_canterbury()
    if only in (None, "loghub"):
        fetch_loghub()
    if only in (None, "wiki"):
        fetch_wiki()
    log("inventory:")
    for d, _, files in sorted(os.walk(RAW)):
        for n in sorted(files):
            p = os.path.join(d, n)
            log(f"  {os.path.relpath(p, RAW):40s} {os.path.getsize(p):>12,d} B  sha256[:16]={sha256_file(p)[:16]}")


if __name__ == "__main__":
    main()
