"""Strata ladder: build the self-contained Kaggle script kernel (code only; data is downloaded inside the kernel).

    python tools/strata/train/build_kernel.py --mode smoke      # <= 30 min: exactness, throughput, resume, export
    python tools/strata/train/build_kernel.py --mode long       # the training run of gate V7 arm W
    python tools/strata/train/build_kernel.py --mode dry --out DIR   # CPU rehearsal of the driver, tiny model

Writes notebooks/strata-ladder/strata-ladder.py (one file: configuration at the top, then the embedded modules of
tools/strata/train/ and the two frozen bake-off modules they import) and kernel-metadata.json (script kernel,
private, GPU, internet). Nothing is pushed: `kaggle kernels push -p notebooks/strata-ladder` is the orchestrator's.
The build refuses to embed anything that is not ASCII or that looks like a local path, a user name or an e-mail.
"""
import argparse
import getpass
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
KERNEL_DIR = os.path.join(ROOT, "notebooks", "strata-ladder")
MODULES = ["tools/strata/train/" + n for n in ("features.py", "model.py", "corpus.py", "run_train.py", "export.py",
                                               "run_eval.py", "check_exact.py", "kernel_main.py")]
MODULES += ["tools/strata/bakeoff/common.py", "tools/strata/bakeoff/ref_int.py"]

SMOKE = {
    "mode": "smoke", "device": "cuda", "work": "/kaggle/working", "data_root": "", "deadline_minutes": 27,
    "data_mb": {"wiki": 64, "log": 8, "dblp": 8}, "chunk": 8192, "lr": 2e-3, "check_bytes": 12000,
    "eval_bytes": 100000,
    "smoke": {"exact_rungs": ["ladder3", "ladder4"],
              "probes": [["ladder3", 256], ["ladder4", 256], ["ladder3", 512], ["ladder4", 512], ["ladder3", 1024],
                         ["r4", 256]],
              "probe_updates": 64, "rung": "ladder3", "batch": 512, "updates": 256, "lrs": [1e-3, 4e-3]},
}
LONG = {
    "mode": "long", "device": "cuda", "work": "/kaggle/working", "data_root": "", "deadline_minutes": 300,
    "data_mb": {"wiki": 900, "log": 64, "dblp": 64}, "chunk": 8192, "lr": 2e-3, "check_bytes": 12000,
    "eval_bytes": 300000,
    "long": {"rung": "ladder3", "batch": 256, "updates": 22000, "minutes": 260, "seeds": [1204, 1205],
             "tail_minutes": 25, "resume_glob": ""},
}
DRY = {
    "mode": "smoke", "device": "cpu", "work": "", "data_root": "", "deadline_minutes": 12,
    "data_mb": {"wiki": 200, "log": 16, "dblp": 8}, "chunk": 8192, "lr": 2e-3, "check_bytes": 2000,
    "eval_bytes": 20000,
    "smoke": {"exact_rungs": ["tiny"], "probes": [["tiny", 8], ["tiny", 16]], "probe_updates": 32, "rung": "tiny",
              "batch": 8, "updates": 64, "lrs": [4e-3]},
}
HEADER = '''#!/usr/bin/env python3
"""Strata capacity ladder (gate V7, arm W): integer-exact trainer on a Kaggle GPU. GENERATED FILE, do not edit:
rebuild with tools/strata/train/build_kernel.py. Code and public corpora only; outputs go to the work directory.

Modes: "smoke" (exactness on the GPU, bytes/s per batch size, resume, export + reference check; <= 30 min) and
"long" (one training run per GPU, then export). The mode and every number of the run are in CONFIG below.
"""
import json
import os
import runpy
import sys

CONFIG = json.loads(r\'\'\'
@CONFIG@
\'\'\')
'''
FOOTER = '''

def bootstrap():
    """Write the embedded modules to a source tree outside the output directory and run the driver."""
    on_kaggle = os.path.isdir("/kaggle/working")
    src = "/tmp/strata-src" if on_kaggle else os.path.abspath("strata-src")
    if not CONFIG["work"]:
        CONFIG["work"] = os.path.abspath("strata-out")
    if len(sys.argv) > 1:                        # rehearsal outside Kaggle: an existing corpus root
        CONFIG["data_root"] = os.path.abspath(sys.argv[1])
    for rel, text in FILES.items():
        path = os.path.join(src, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="\\n") as f:
            f.write(text)
    driver = os.path.join(src, "tools", "strata", "train", "kernel_main.py")
    with open(os.path.join(os.path.dirname(driver), "kernel-config.json"), "w") as f:
        json.dump(CONFIG, f)
    runpy.run_path(driver, run_name="__main__")


bootstrap()
'''


def privacy_check(name, text):
    """Refuse non-ASCII text, drive or home paths, e-mail addresses and the local account name."""
    if not text.isascii():
        raise SystemExit(f"{name}: non-ASCII characters")
    words = {w for w in re.split(r"[^a-z0-9]+", getpass.getuser().lower()) if len(w) > 3}
    for pattern in [r"[A-Za-z]:\\\\", r"/Users/", r"/home/", r"[\w.]+@[\w-]+\.[a-z]{2,}"] + sorted(words):
        hit = re.search(pattern, text, flags=re.IGNORECASE)
        if hit:
            raise SystemExit(f"{name}: refused, matches the privacy pattern {pattern!r}: {hit.group(0)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("smoke", "long", "dry"))
    ap.add_argument("--out", default=KERNEL_DIR)
    args = ap.parse_args()
    config = {"smoke": SMOKE, "long": LONG, "dry": DRY}[args.mode]
    files = {}
    for rel in MODULES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            files[rel] = f.read()
    body = HEADER.replace("@CONFIG@", json.dumps(config, indent=1))
    body += "\nFILES = {\n" + "".join(f"    {json.dumps(rel)}:\n    {json.dumps(text)},\n"
                                      for rel, text in files.items()) + "}\n" + FOOTER
    privacy_check("kernel", body)
    os.makedirs(args.out, exist_ok=True)
    script = os.path.join(args.out, "strata-ladder.py")
    with open(script, "w", newline="\n") as f:
        f.write(body)
    print(f"wrote {script}: mode {args.mode}, {len(body):,d} bytes, {len(files)} embedded modules")
    if args.mode == "dry":
        return
    with open(os.path.join(ROOT, "notebooks", "log-specialist-train", "kernel-metadata.json")) as f:
        owner = json.load(f)["id"].split("/")[0]           # the account of the repo's existing kernels
    meta = {"id": f"{owner}/strata-ladder", "title": "Strata Ladder", "code_file": "strata-ladder.py",
            "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "docker_image_pinning_type": "latest"}
    with open(os.path.join(args.out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote kernel-metadata.json (script kernel, private, GPU {meta['machine_shape']}, internet on)")


if __name__ == "__main__":
    main()
