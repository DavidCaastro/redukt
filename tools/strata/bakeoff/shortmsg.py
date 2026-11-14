"""Strata bake-off: short messages (path 3), real coded bytes from the integer reference.

    python -I tools/strata/bakeoff/shortmsg.py --run strata_f --tier 1 --sources log-apache,log-linux [--primer 65536]
    python -I tools/strata/bakeoff/shortmsg.py --summary strata_f [--mamba data/benchmark/results/short-message-bench/run-1/short-message-results.csv]

Protocol of R-113 (`notebooks/short-message-bench.py`): 16 messages of 128 / 256 / 512 / 1,024 / 2,048 bytes per
source, cut from the second half of the source at line starts; classical codecs with minimal framing; the zstd
dictionary (110 KB) is trained on the first half of the SAME source, which is the best case for zstd.
Strata bytes are real range-coder output of `ref_int.py` (tight flush), each message coded from the boot state.
`--primer N` (hook H2 of the specification, not part of v1): the stream machine first reads the last N bytes of
the first half of the source, and every message starts from that snapshot — the counterpart of the dictionary.
`--primer-src foreign` takes the primer from a different source of the same type (FOREIGN, fixed before G0);
the classical rows then include `zstd-19-fdict`, a dictionary trained on that foreign source.
`--trunk off` zeroes the node and gain weights of the exported tier (ablation: biases + tables + mixer only).
Rows are appended to data/strata/runs/<run>/<--out> (default shortmsg.csv; resumable: finished (source, label)
pairs are skipped); --summary reads every shortmsg*.csv of the run and drops duplicate rows.
`--mamba` aligns the messages with the R-113 per-message file (same cut function, same sources): the classical
codec sizes must match message by message, and Mamba-130M's payload is its archive size minus the 20-byte
framing (16-byte header + CRC-32).
Sources: Wikipedia = enwik8 (SEEN source for these models), prose = dickens, three Loghub 2k samples, GitHub JSON.
"""
import argparse
import copy
import csv
import lzma
import os
import re
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BENCH, CACHE, RAW, RUNS  # noqa: E402
from ref_int import RangeEncoder, StrataRef, code_stream  # noqa: E402

SIZES = [128, 256, 512, 1024, 2048]
PER_CELL = 16
PATHS = {
    "wikipedia": (os.path.join(BENCH, "enwik8"), 4_000_000),
    "prose": (os.path.join(BENCH, "silesia", "dickens"), 4_000_000),
    "log-apache": (os.path.join(BENCH, "industry", "apache-access.log"), None),
    "log-linux": (os.path.join(BENCH, "industry", "linux-syslog.log"), None),
    "log-openssh": (os.path.join(BENCH, "industry", "openssh-auth.log"), None),
    "json-events": (os.path.join(RAW, "gharchive", "2015-01-01-15.json"), 4_000_000),
    # selection-set protocol of gate G0 (sources never trained on, distinct from the frozen test set)
    "sel-webster": (os.path.join(BENCH, "silesia", "webster"), 4_000_000),
    "sel-xml": (os.path.join(BENCH, "silesia", "xml"), 4_000_000),
    "sel-zookeeper": (os.path.join(RAW, "loghub", "Zookeeper.log"), None),
    "sel-proxifier": (os.path.join(RAW, "loghub", "Proxifier.log"), None),
}
R113 = ["wikipedia", "prose", "log-apache", "log-linux", "log-openssh", "json-events"]
SELECTION = ["sel-webster", "sel-xml", "sel-zookeeper", "sel-proxifier"]
# foreign primer (and foreign dictionary), fixed in 05-gate-amendments.md (G0 amendment 1): never a training
# source; for the selection protocol the other selection source of the pair
_WEB = (os.path.join(BENCH, "silesia", "webster"), 4_000_000)
_XML = (os.path.join(BENCH, "silesia", "xml"), 4_000_000)
_ZOO = (os.path.join(RAW, "loghub", "Zookeeper.log"), None)
_PROX = (os.path.join(RAW, "loghub", "Proxifier.log"), None)
_GH2 = (os.path.join(RAW, "gharchive", "2015-01-02-15.json"), 4_000_000)   # same archive: SAME source, labelled so
FOREIGN = {"sel-webster": _XML, "sel-xml": _WEB, "sel-zookeeper": _PROX, "sel-proxifier": _ZOO,
           "wikipedia": _WEB, "prose": _WEB, "log-apache": _ZOO, "log-linux": _ZOO, "log-openssh": _ZOO,
           "json-events": _GH2}


def read_source(spec):
    path, cap = spec
    return open(path, "rb").read() if cap is None else open(path, "rb").read(cap)


def cut_messages(blob, size, count):
    half = len(blob) // 2
    starts = [m.end() for m in re.finditer(b"\n", blob[half:])]
    usable = [half + s for s in starts if half + s + size <= len(blob)]
    if len(usable) < count:
        usable = list(range(half, len(blob) - size, max(1, (len(blob) - half - size) // count)))
    step = max(1, len(usable) // count)
    return [blob[o:o + size] for o in usable[::step][:count]]


def _dictionary(zstandard, train_half):
    step = max(256, len(train_half) // 4000)
    samples = [train_half[i:i + 1024] for i in range(0, len(train_half) - 1024, step)]
    try:
        return zstandard.train_dictionary(110_000, samples)
    except Exception:   # tiny sources cannot train a 110 KB dictionary: raw-content prefix dictionary instead
        return zstandard.ZstdCompressionDict(train_half[-110_000:], dict_type=zstandard.DICT_TYPE_RAWCONTENT)


def classical_tools(train_half, foreign_half=None):
    import brotli
    import zstandard
    params = zstandard.ZstdCompressionParameters.from_level(
        19, format=zstandard.FORMAT_ZSTD1_MAGICLESS, write_checksum=False, write_content_size=False,
        write_dict_id=False)
    tools = {
        "xz-9e-raw": lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=[
            {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]),
        "gzip-9-raw": lambda b: zlib.compress(b, 9, wbits=-15),
        "brotli-11": lambda b: brotli.compress(b, quality=11),
        "zstd-19": zstandard.ZstdCompressor(compression_params=params).compress,
    }
    dict_data = _dictionary(zstandard, train_half)
    tools["zstd-19-dict"] = zstandard.ZstdCompressor(compression_params=params, dict_data=dict_data).compress
    if foreign_half is not None:     # dictionary trained on the foreign source (the symmetric unfavourable case)
        fdict = _dictionary(zstandard, foreign_half)
        tools["zstd-19-fdict"] = zstandard.ZstdCompressor(compression_params=params, dict_data=fdict).compress
    return tools


def summary(run, mamba=None):
    import glob
    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(RUNS, run, "shortmsg*.csv"))):
        for r in csv.DictReader(open(path)):
            key = (r["source"], r["size"], r["index"], r["tool"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    if mamba:
        ref = {}
        for r in csv.DictReader(open(mamba)):
            ref[(r["source"], r["size"], r["index"], r["tool"])] = int(r["out_bytes"])
        same = diff = 0
        for r in rows:
            k = (r["source"], r["size"], r["index"], r["tool"])
            if r["tool"] in ("gzip-9-raw", "xz-9e-raw", "brotli-11", "zstd-19", "zstd-19-dict") and k in ref:
                same += ref[k] == int(r["bytes"])
                diff += ref[k] != int(r["bytes"])
        print(f"alignment with the R-113 file: classical sizes identical for {same} messages, different for {diff}")
        for k, v in ref.items():
            if k[3] == "redukt-azth":
                rows.append({"source": k[0], "size": k[1], "index": k[2], "tool": "mamba130m-payload",
                             "bytes": str(v - 20)})
                rows.append({"source": k[0], "size": k[1], "index": k[2], "tool": "mamba130m-azth",
                             "bytes": str(v)})
    tools = sorted({r["tool"] for r in rows})
    cell = {}
    for r in rows:
        cell.setdefault((r["source"], int(r["size"]), r["tool"]), []).append(int(r["bytes"]))
    sources = sorted({r["source"] for r in rows})
    print("mean bytes per message; cells = source x size")
    print(f"{'source':12s} {'size':>5s} " + " ".join(f"{t:>22s}" for t in tools))
    for src in sources:
        for size in SIZES:
            vals = [np.mean(cell[(src, size, t)]) if (src, size, t) in cell else float("nan") for t in tools]
            print(f"{src:12s} {size:5d} " + " ".join(f"{v:22.1f}" for v in vals))
    totals = {}
    for r in rows:
        totals[r["tool"]] = totals.get(r["tool"], 0) + int(r["bytes"])
    print("total bytes over all messages of the file(s), per tool:")
    for t in tools:
        print(f"  {t:44s} {totals[t]:>10,d}")
    for ref in ("zstd-19-dict", "zstd-19-fdict", "mamba130m-payload"):
        for t in tools:
            if t == ref or not (t.startswith("strata") or t.startswith("mamba")) or t.endswith("azth"):
                continue
            for extra in (0, 4, 20):
                wins = total = 0
                for src in sources:
                    for size in SIZES:
                        if (src, size, t) in cell and (src, size, ref) in cell:
                            total += 1
                            wins += np.mean(cell[(src, size, t)]) + extra < np.mean(cell[(src, size, ref)])
                if total:
                    print(f"{t} + {extra:2d} B framing: smaller than {ref} in {wins} of {total} cells")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--sources", default=",".join(R113))
    ap.add_argument("--primer", type=int, default=0)
    ap.add_argument("--primer-src", default="same", choices=("same", "foreign"))
    ap.add_argument("--trunk", default="on", choices=("on", "off"))
    ap.add_argument("--out", default="shortmsg.csv")
    ap.add_argument("--summary")
    ap.add_argument("--mamba")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary, args.mamba)
        return
    import torch
    from models import build, export_int
    ck = torch.load(os.path.join(RUNS, args.run, "ckpt.pt"), weights_only=False)
    model = build(ck["args"]["model"])
    model.load_state_dict(ck["model"])
    if not model.quant:
        raise SystemExit("integer Strata models only")
    params = export_int(model, args.tier)
    if args.trunk == "off":
        params["hw"] = np.zeros_like(params["hw"])
        params["gw"] = np.zeros_like(params["gw"])
    q0 = np.load(os.path.join(CACHE, "q0.npy"))
    out_path = os.path.join(RUNS, args.run, args.out)
    done = set()
    if os.path.isfile(out_path):
        done = {(r["source"], r["tool"]) for r in csv.DictReader(open(out_path))}
    else:
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerow(["source", "size", "index", "tool", "bytes"])
    label = (f"strata-t{args.tier}-payload" + ("-notrunk" if args.trunk == "off" else "")
             + ((f"-primer{args.primer}" if args.primer_src == "same" else f"-fprimer{args.primer}")
                if args.primer else ""))
    for src in args.sources.split(","):
        blob = read_source(PATHS[src])
        half = len(blob) // 2
        fblob = read_source(FOREIGN[src])
        fhalf = len(fblob) // 2
        rows = []
        t0 = time.time()
        if (src, "zstd-19-dict") not in done:
            tools = classical_tools(blob[:half], fblob[:fhalf])
            for size in SIZES:
                for i, msg in enumerate(cut_messages(blob, size, PER_CELL)):
                    rows += [[src, size, i, name, len(fnc(msg))] for name, fnc in tools.items()]
        if (src, label) not in done:
            base = StrataRef(params, q0)
            if args.primer:
                # the snapshot keeps the tables, the calibration cells and the mixer; the trunk, the delay taps
                # and the context history go back to boot, so a message still starts as a stream start
                primer = (blob[max(0, half - args.primer):half] if args.primer_src == "same"
                          else fblob[max(0, fhalf - args.primer):fhalf])
                code_stream(base, np.frombuffer(primer, dtype=np.uint8))
                base.hist = [256, 256, 256, 256]
                base.s_in[:] = 0
                for acc in base.s:
                    acc[:] = 0
                base.ctx, base.n, base.trun = [], 0, [0] * len(base.trun)
            for size in SIZES:
                for i, msg in enumerate(cut_messages(blob, size, PER_CELL)):
                    enc = RangeEncoder()
                    code_stream(copy.deepcopy(base), np.frombuffer(msg, dtype=np.uint8), enc)
                    rows.append([src, size, i, label, len(enc.finish_tight())])
        with open(out_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"{src}: {len(rows)} rows in {time.time() - t0:.0f} s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
