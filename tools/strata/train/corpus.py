"""Strata ladder trainer: the training corpus of gate V7 arm W (public files, by-source rule) and its sampler.

    python -I tools/strata/train/corpus.py --root DIR                  # download what is missing, print size + sha256
    python -I tools/strata/train/corpus.py --root DIR --mirror RAW     # first hard-link the files of a bake-off raw dir

Downloads are plain GET requests to the original public hosts with a generic User-Agent; nothing else is sent.

| Group | Source | Bytes |
|---|---|---|
| text | enwik9 bytes [1e8, 1e9): the first 1e8 bytes are enwik8, checked by sha256 and never written | 900 MB |
| logs | Loghub (Zenodo record 3227177), the 11 training systems of the bake-off, first 64 MB of each | ~620 MB |
| misc | first 64 MB of the DBLP XML dump (monthly snapshot 2026-01-01, CC0) + 4 small Canterbury files | 64 MB |

Never training bytes (selection and decision sources, `check_training_list`): Silesia webster / xml / dickens / nci,
Loghub Zookeeper / Proxifier / Apache / Linux / OpenSSH, Canterbury bible.txt, the GitHub event archive, enwik8 and
the invoice files. The two Project Gutenberg etexts of the bake-off's training set (world192.txt, lcet10.txt) are
left out too: specification 8.6 excludes Gutenberg by source because its boilerplate occurs in webster and dickens.
"""
import argparse
import gzip
import hashlib
import io
import os
import re
import sys
import tarfile
import time
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bakeoff"))
from common import boot_calibration  # noqa: E402

UA = {"User-Agent": "Wget/1.21"}
ENWIK9_URL = "https://mattmahoney.net/dc/enwik9.zip"
ENWIK8_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
ENWIK8_BYTES = 100_000_000
ZENODO = "https://zenodo.org/api/records/3227177/files/{}.tar.gz/content"
CANTERBURY_URL = "https://corpus.canterbury.ac.nz/resources/cantrbry.zip"
DBLP_URL = "https://dblp.org/xml/release/dblp-2026-01-01.xml.gz"
LOG_SYSTEMS = ("BGL", "HDFS_1", "Hadoop", "Spark", "HPC", "Mac", "Android", "Windows", "Thunderbird", "OpenStack",
               "HealthApp")
CANTERBURY = ("cp.html", "fields.c", "grammar.lsp", "xargs.1")
FULL = dict(wiki=900_000_000, log=64_000_000, dblp=64_000_000)     # bytes kept per source
# name, group, file under the corpus root
TRAIN = ([("wiki", "text", "wiki/enwik9-train.bin")]
         + [(f"log-{s}", "logs", f"loghub/{s}.log") for s in LOG_SYSTEMS]
         + [("dblp", "misc", "markup/dblp-head.bin")]
         + [(n, "misc", f"canterbury/{n}") for n in CANTERBURY])
GROUP_WEIGHT = {"text": 0.45, "logs": 0.45, "misc": 0.10}
FORBIDDEN = ("webster", "xml", "dickens", "nci", "zookeeper", "proxifier", "apache", "linux", "openssh", "ssh",
             "bible", "gharchive", "enwik8", "facturae", "ubl", "invoice", "world192", "lcet10")
Q0_INDEX = 10_000_019          # sampler index of the boot-calibration chunks, outside the training range


def check_training_list():
    """Fails if a selection or decision source is named in the training list (names and paths, word by word)."""
    for name, _, rel in TRAIN:
        words = set(re.split(r"[^a-z0-9]+", f"{name} {rel}".lower()))
        bad = words.intersection(FORBIDDEN)
        assert not bad, f"forbidden source in the training list: {name} ({sorted(bad)})"


def log(msg):
    print(msg, flush=True)


def open_url(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def write_atomic(dst, blob):
    with open(dst + ".part", "wb") as f:
        f.write(blob)
    os.replace(dst + ".part", dst)


def fetch_wiki(dst, nbytes):
    """enwik9 bytes [1e8, 1e8 + nbytes); the zip is streamed to disk, its head is checked against enwik8."""
    zpath = dst + ".zip"
    with open_url(ENWIK9_URL) as r, open(zpath, "wb") as f:
        while True:
            blob = r.read(1 << 22)
            if not blob:
                break
            f.write(blob)
    head = hashlib.sha256()
    with zipfile.ZipFile(zpath) as z, z.open("enwik9") as src, open(dst + ".part", "wb") as out:
        left = ENWIK8_BYTES
        while left:
            blob = src.read(min(1 << 22, left))
            head.update(blob)
            left -= len(blob)
        if head.hexdigest() != ENWIK8_SHA256:
            raise SystemExit("the first 1e8 bytes of enwik9 are not enwik8: refusing to build the text slice")
        left = nbytes
        while left:
            blob = src.read(min(1 << 22, left))
            if not blob:
                raise SystemExit(f"enwik9 ended {left:,d} bytes early")
            out.write(blob)
            left -= len(blob)
    os.replace(dst + ".part", dst)
    os.remove(zpath)


def fetch_log(system, dst, cap):
    """First `cap` bytes of the concatenated log members of a Loghub .tar.gz, streamed (never stored whole)."""
    got = 0
    with open_url(ZENODO.format(system)) as r, open(dst + ".part", "wb") as out:
        with tarfile.open(fileobj=r, mode="r|gz") as tar:
            for m in tar:
                base = os.path.basename(m.name.lower())
                if not m.isfile() or m.size == 0 or not (base.endswith((".log", ".txt")) or "." not in base):
                    continue
                f = tar.extractfile(m)
                while got < cap:
                    blob = f.read(min(1 << 20, cap - got))
                    if not blob:
                        break
                    out.write(blob)
                    got += len(blob)
                if got >= cap:
                    break
    os.replace(dst + ".part", dst)


def fetch_dblp(dst, cap):
    with open_url(DBLP_URL) as r, gzip.GzipFile(fileobj=r) as gz:
        blob = gz.read(cap)
    if len(blob) != cap:
        raise SystemExit(f"DBLP dump: {len(blob):,d} bytes read, {cap:,d} expected")
    write_atomic(dst, blob)


def fetch_canterbury(root):
    with open_url(CANTERBURY_URL) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in CANTERBURY:
            write_atomic(os.path.join(root, "canterbury", n), z.read(n))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blob in iter(lambda: f.read(1 << 22), b""):
            h.update(blob)
    return h.hexdigest()


def fetch(root, sizes):
    """Download every training file that is missing under `root`; print the inventory (bytes, sha256)."""
    check_training_list()
    for name, _, rel in TRAIN:
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(dst):
            continue
        t0 = time.time()
        if name == "wiki":
            fetch_wiki(dst, sizes["wiki"])
        elif name.startswith("log-"):
            fetch_log(name[4:], dst, sizes["log"])
        elif name == "dblp":
            fetch_dblp(dst, sizes["dblp"])
        else:
            fetch_canterbury(root)
        log(f"  fetched {name} in {time.time() - t0:.0f} s")
    total = 0
    for name, group, rel in TRAIN:
        path = os.path.join(root, rel)
        total += os.path.getsize(path)
        log(f"  {name:16s} {group:5s} {os.path.getsize(path):>12,d} B  sha256={sha256_file(path)}")
    log(f"  training bytes: {total:,d}")
    return total


def mirror(raw, root):
    """Hard-link the training files a bake-off raw directory already holds (local runs: no download of them)."""
    pairs = [("wiki/enwik9-100M-300M.bin", "wiki/enwik9-train.bin")]
    pairs += [(f"loghub/{s}.log", f"loghub/{s}.log") for s in LOG_SYSTEMS]
    pairs += [(f"canterbury/{n}", f"canterbury/{n}") for n in CANTERBURY]
    for src, rel in pairs:
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.isfile(dst):
            os.link(os.path.join(raw, src), dst)


class Corpus:
    """Training sources in memory and a deterministic chunk sampler (same rules as the bake-off's)."""

    def __init__(self, root):
        check_training_list()
        self.groups = {}
        for name, group, rel in TRAIN:
            self.groups.setdefault(group, []).append(np.fromfile(os.path.join(root, rel), dtype=np.uint8))
        self.names = sorted(self.groups)
        w = np.array([GROUP_WEIGHT[g] for g in self.names], dtype=np.float64)
        self.group_p = w / w.sum()
        self.nbytes = sum(len(d) for members in self.groups.values() for d in members)

    def sample(self, seed, index, count, chunk):
        """`count` chunks of `chunk` bytes, a pure function of (seed, index): group by weight, then a log system
        uniformly or a text / misc file by size; a chunk start moves to the next line start within 512 bytes."""
        rng = np.random.default_rng([seed, index])
        out = np.zeros((count, chunk), dtype=np.uint8)
        for i in range(count):
            g = self.names[rng.choice(len(self.names), p=self.group_p)]
            members = self.groups[g]
            if g == "logs":
                data = members[rng.integers(len(members))]
            else:
                sizes = np.array([len(d) for d in members], dtype=np.float64)
                data = members[rng.choice(len(members), p=sizes / sizes.sum())]
            off = 0
            if len(data) > chunk + 600:
                off = int(rng.integers(0, len(data) - chunk - 600))
                nl = np.flatnonzero(data[off:off + 512] == 10)
                if len(nl):
                    off += int(nl[0]) + 1
            piece = data[off:off + chunk]
            out[i, :len(piece)] = piece
        return out

    def boot_q0(self, seed):
        """Boot calibration of the successor evidence, counted on 256 training chunks (specification 8.4)."""
        return boot_calibration(self.sample(seed, Q0_INDEX, 256, 8192))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mirror", default="", help="bake-off raw directory whose training files are hard-linked first")
    ap.add_argument("--wiki-mb", type=int, default=FULL["wiki"] // 1_000_000)
    ap.add_argument("--log-mb", type=int, default=FULL["log"] // 1_000_000)
    ap.add_argument("--dblp-mb", type=int, default=FULL["dblp"] // 1_000_000)
    args = ap.parse_args()
    if args.mirror:
        mirror(args.mirror, args.root)
    fetch(args.root, dict(wiki=args.wiki_mb * 1_000_000, log=args.log_mb * 1_000_000, dblp=args.dblp_mb * 1_000_000))


if __name__ == "__main__":
    main()
