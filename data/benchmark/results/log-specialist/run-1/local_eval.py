"""Local CPU evaluation of the LSTM specialists trained in Kaggle run-1 (the in-kernel evaluation died on a cuDNN
error). Same functions as the notebook; adds a random-init control for the dynamic mode."""
import csv, importlib.util, lzma, os, sys, time
import torch
nb, spec_dir, out_csv, name, modes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")
spec = importlib.util.spec_from_file_location("nb", nb); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.set_num_threads(4)
rd = lambda p: open(p, "rb").read()
ind = "data/benchmark/industry/"
evals = [("apache", "heldout", rd(ind + "apache-access.log")), ("linux", "heldout", rd(ind + "linux-syslog.log")),
         ("openssh", "heldout", rd(ind + "openssh-auth.log")),
         ("zookeeper", "heldout", rd(spec_dir + "/data/Zookeeper.log")[:300_000]),
         ("proxifier", "heldout", rd(spec_dir + "/data/Proxifier.log")[:300_000]),
         ("HPC", "seen", rd(spec_dir + "/data/HPC.log")[-300_000:]), ("Mac", "seen", rd(spec_dir + "/data/Mac.log")[-300_000:]),
         ("HealthApp", "seen", rd(spec_dir + "/data/HealthApp.log")[-300_000:])]
if name == "check":
    for n, k, b in evals:
        print(n, k, len(b), "xz-9e bpb", round(len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)) * 8 / len(b), 4))
    sys.exit()
ck = torch.load(f"{spec_dir}/models/{name}.pt", map_location="cpu", weights_only=True)
model = m.build_model(ck["kind"], ck["cfg"]); model.load_state_dict(ck["state"]); model.eval()
rows = []
for n, k, b in evals:
    cells = []
    for mode in modes:
        t0 = time.time()
        if mode.startswith("static-"):
            c = mode.split("-")[1]
            bits = m.static_bits(model, "lstm", b, 0 if c == "whole" else int(c), "cpu")
        elif mode == "dynamic-256":
            bits = m.dynamic_bits(model, b, "cpu")
        elif mode == "dynamic-256-random-init":
            torch.manual_seed(7)
            bits = m.dynamic_bits(m.build_model(ck["kind"], ck["cfg"]), b, "cpu")
        rows.append((name, n, k, mode, len(b), f"{bits:.1f}", f"{bits / len(b):.4f}"))
        cells.append(f"{mode} {bits / len(b):.4f} ({time.time() - t0:.0f}s)")
    print(n, k, "  ".join(cells), flush=True)
    new = not os.path.isfile(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["model", "eval_set", "kind", "mode", "bytes", "bits", "bpb"])
        w.writerows(rows)
    rows = []
