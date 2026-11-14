# Strata backbone — 09b. Trainer port for the capacity ladder (stage 4c, gate V7 arm W)

> Written 2026-11-14. Code: `tools/strata/train/`; kernel: `notebooks/strata-ladder/`. Nothing was pushed, no
> decision file was evaluated, no GPU hour was used. Labels: MEASURED (with its command, run on the i5-1235U laptop,
> torch 2.14 CPU, from the repo root), INFERENCE, UNTESTED. Run directories are under `data/strata/runs-ladder/`
> (gitignored).

## 0. Weak points first

1. **Nothing has run on a GPU.** Every CUDA statement below is INFERENCE: the throughput, the memory per batch
   size, the exactness of the integer paths on the device (test B exists and passes only as cpu-vs-cpu), the two
   processes sharing 4 vCPUs. The smoke kernel exists to measure exactly these. Section 7 lists what is untested.
2. **Training is not bit-reproducible run to run, on CPU with more than one thread and (INFERENCE) on CUDA.** The
   forward pass is exact integers for given weights; the gradients are float sums whose order varies. MEASURED: two
   identical uninterrupted 96-update runs with `--threads 4` end with different weights (logged costs equal to four
   decimals); with `--threads 1` an uninterrupted run and a paused + resumed run end with identical state dicts.
   So the resume logic is exact, and "resume reproduces the loss curve" can only be checked within a tolerance on
   the GPU (the smoke run reports the largest difference of the logged costs).
3. **The training set deviates from the task text in one place, on purpose.** The two Project Gutenberg etexts of
   the bake-off's `misc` group (`world192.txt`, `lcet10.txt`) are left out: specification 8.6 excludes Gutenberg by
   source, and MEASURED `grep -aci gutenberg` gives 21 and 2 matches in them against 276 in `webster` (selection)
   and 297 in `dickens` (decision). Without them `misc` would be 43 KB, so the first 64 MB of the DBLP XML dump was
   added (reachable now, pinned monthly snapshot). Effect: about 10 % of the training bytes are bibliographic XML.
   Two lines of `corpus.py` restore the bake-off's list if the orchestrator decides otherwise.
4. **Chunks are 8,192 bytes from the boot state, as in the bake-off**, not the 32,768 bytes + 25 % short chunks of
   specification 8.3. `--chunk 32768` works (MEASURED on the tiny model) but the short-chunk mix is not implemented.
   The ladder is read on streams of 300 KB to 1 MiB without reset, so this is a real open choice (section 8).
5. **The learning rate for the larger batch is not established.** The bake-off used 64 streams per update at 2e-3;
   the recommended batch here is 512 with the same 2e-3. The smoke run compares 1e-3 / 2e-3 / 4e-3 over 256 updates
   only, which is weak evidence for a 7,648-update run.
6. **The exported file is the design-stage layout of `ref_int.py` (a `.npz`), not the weight file of specification
   5.** That file and the Rust loader do not exist yet (stages 4a / 4b).
7. Kaggle specifics taken from the installed CLI's source, not from a run: `machine_shape: NvidiaTeslaT4` in the
   metadata, T4 x2 as what that shape provides, quota counted per session hour. `kernel-metadata.json` carries the
   Kaggle account name in its `id`, as the repo's other kernels do; the script itself carries none (the build
   refuses non-ASCII text, home or drive paths, e-mail addresses and the local account name).

## 1. What was built

| File | Role |
|---|---|
| `tools/strata/train/model.py` | geared trunk + head of `bakeoff/models.py` with the three sequential parts on the device; rung table; budget; exactness bound; `export_int` |
| `features.py` | torch port of the successor evidence, calibration cells and bit cells (any device) |
| `corpus.py` | training list, forbidden-source assertion, downloads, deterministic sampler, boot calibration |
| `run_train.py` | trainer: `--updates --batch --minutes`, atomic checkpoint, resume, CSV log, `summary.json` |
| `export.py` | integer weights of every tier to `.npz`, read back and checked against `ref_int.StrataRef` (test C) |
| `run_eval.py` | ideal code length per file and tier, role-gated, resumable |
| `check_exact.py` | tests A (vs prototype), B (cpu vs device), features (vs prototype) |
| `kernel_main.py`, `build_kernel.py` | driver of the Kaggle kernel and its build step |
| `notebooks/strata-ladder/strata-ladder.py`, `kernel-metadata.json` | generated script kernel (currently mode `smoke`) |

`tools/strata/bakeoff/` is untouched. The new modules import its constants and tables (`common.py`) and the
sequential reference (`ref_int.py`); the kernel embeds those two files verbatim.

How the three loops run on the device (same integers as the prototype, checked by test A):
- accumulators (input bank and the three geared layers): int32 tensors, `S -= (S - (u << 12)) >> shift`, one step
  per byte, batched over streams and channels; the backward pass is one `addcmul` per step;
- online mixers: one loop for all tiers at once, int32 state, the dot product summed in int64; the loop also
  yields the final stretch, so the float64 tensors of the prototype's head are not needed (a custom autograd
  function gives the same gradient);
- matrix products stay float32 with integer values; `Ladder.check_exact()` computes, from the actual weights, the
  largest pre-activation any row can reach and stops the run at 2^24. Widest rung (928): worst case 127 * 127 * 928
  = 14.97 M < 16.78 M before biases. TF32 is switched off in `model.py`.

## 2. Rungs (MEASURED: `Ladder(r).check_budget()`; same arithmetic as the prototype's `budget()`)

L = 3, geared, bit cells on, widths nested in one model. Caps are the V7 rungs of amendment 4.

| Config | Tier width | MAC / byte | Cap | Parameters |
|---|---|---|---|---|
| `ladder3` | 88 | 27,752 | 32,000 | 73,392 |
| | 208 | 113,912 | 128,000 | 220,437 |
| | **456** | 499,552 | 512,000 | 731,906 |
| `ladder4` (slope probe R4, nested) | 88 / 208 / 456 as above, then **928** | 1,985,408 | 2,000,000 | 2,457,243 |
| `r4` (R4 alone) | **760** | 1,961,768 | 2,000,000 | 2,348,364 |

456, 928 and 760 are the largest multiples of 8 under their caps: 464 gives 518,904, 936 gives 2,023,552, 768 gives
2,003,152 (MEASURED with both this model and the prototype's `macs()` / `params()`, identical). `ladder3` has no
width-0 tier: R0 is the `cm` tier that G0 already trained, and a nested width-0 tier would take a share of the loss.

## 3. Exactness

| Test | Command | Result (MEASURED) |
|---|---|---|
| A | `python tools/strata/train/check_exact.py --test A` | 5 cases, 6 streams x 4 consecutive windows with carried state: checkpoints `strata_g_bc` and `strata_g_bc_s1205`, random weights for (0, 88, 208), (16, 40), (88, 208, 456). Against `bakeoff/models.py`: 0 mismatches in `p1` (147,456 values per 3-tier case), in the float64 costs (bit-identical), in the trunk state and in the mixer state; exported integers equal `models.export_int`; gradients differ by at most 6.3e-6 of the largest entry of the same tensor (tolerance 1e-3) |
| features | `python tools/strata/train/check_exact.py --test features --data data/strata/ladder-local` | 0 mismatches against `common.py` on 5 inputs (corpus chunks, random bytes, two symbols, 11-byte streams, 72 x 8,192 bytes), 66 M values |
| C, tiny | `python tools/strata/train/export.py --ckpt data/strata/runs-ladder/tiny-d/ckpt.pt --out .../weights-int.npz --check sel-zookeeper --bytes 12000` | widths 16 and 40: 0 mismatches of 96,000 probabilities per tier against `ref_int.StrataRef` reading the saved file; range-coder round trip of 3,000 bytes OK |
| C, ladder3 | same with `l3-cpu/ckpt.pt` (32 updates) | widths 88, 208, 456: 0 mismatches of 96,000 per tier; round trips OK; 56 s |
| evaluator | `run_eval.py` on the G0 checkpoint converted to the new format, 4 selection files x 300,000 bytes, window 2,048 | all 12 bpb equal `data/strata/runs/strata_g_bc/eval-select-on.csv` to the 4 decimals recorded (e.g. 1.7387 / 1.7279 / 1.7171 on `webster`) |
| B | `check_exact.py --test B --data DIR --rung R --device cuda` | UNTESTED on a GPU. cpu vs cpu: 0 mismatches, gradient difference 4.7e-7 |

Why gradients differ from the prototype: the backward filters use `addcmul` (possibly fused) instead of numpy's
multiply then add, and the eight per-channel gate gradients are summed in a different order. Both are float32
rounding. Test A's integers do not depend on it.

What test A does not cover: the prototype's ablation switches (`online=False`, `trunk_on=False`) and the non-geared
and float variants are not ported (the ladder does not use them).

## 4. Feature pipeline

MEASURED (128 chunks x 8,192 bytes of the bake-off corpus, `torch.set_num_threads(1)`; timing of
`common.successor_features` + `evidence` + `bitcell_evidence`, then `features.features(chunks, q0, "cpu")`):

| Implementation | Seconds for 1,048,576 bytes | KB/s on one core |
|---|---|---|
| prototype, numpy (`superbatch` without its disk cache) | 7.57 (0.95 + 2.82 + 3.80) | **138.5** |
| torch port on CPU | 13.91 | 75.4 |

One core feeds 138 KB/s, and a Kaggle session has 4 vCPUs for two trainer processes, so the features run on the
GPU: the port keeps the per-chunk sort for the successor tables and merges the calibration-cell loop and the
bit-cell loop into one loop over time (about 14 small operations per byte position, independent of the number of
streams). Features are computed per group of chunks, moved to the CPU as int16 / uint8 (100 bytes per input byte)
and uploaded per window; nothing is cached on disk. On the CPU the port is slower than numpy, so `run_eval.py`
uses the numpy functions when `--device cpu` (18 s against 61 s for one 300 KB stream, same bpb).

## 5. Training data (`corpus.py`)

| Group | Weight | Source | Bytes |
|---|---|---|---|
| text | 0.45 | enwik9 bytes [1e8, 1e9); the first 1e8 bytes are hashed, compared with the sha256 of enwik8 and never written | 900,000,000 |
| logs | 0.45, uniform over systems | Loghub, Zenodo record 3227177: BGL, HDFS_1, Hadoop, Spark, HPC, Mac, Android, Windows, Thunderbird, OpenStack, HealthApp; first 64 MB of each or the whole log | about 560 MB (INFERENCE: only HPC, 33,553,503 B, was downloaded whole) |
| misc | 0.10, by size | DBLP XML monthly snapshot 2026-01-01, first 64 MB; Canterbury `cp.html`, `fields.c`, `grammar.lsp`, `xargs.1` | 64,043,701 |

Total about 1.5 GB (>= 335 MB by the text group alone). `check_training_list()` runs in every loader and fails if
a word of a training name or path is one of: webster, xml, dickens, nci, zookeeper, proxifier, apache, linux,
openssh, ssh, bible, gharchive, enwik8, facturae, ubl, invoice, world192, lcet10. Boot calibration `Q0`: counted on
256 chunks of 8,192 bytes at sampler index 10,000,019, as `get_q0`; stored in the checkpoint and in the export.

MEASURED downloads (generic agent, from this laptop): `fetch_wiki` with a 3 MB slice (322 MB zip, head sha256
verified, bytes equal to the bake-off's slice, 52 s); `fetch_log("HPC")` (first 16 MB equal to the bake-off's
file); `fetch_canterbury` (4 files equal); `fetch_dblp` 8 MB (14 s); Zookeeper 2 MB for the selection check (same
sha256 as the bake-off's file). UNTESTED: the full-size downloads inside Kaggle.

## 6. Local end-to-end proof and throughput (MEASURED, CPU)

```
python -I tools/strata/train/corpus.py --root data/strata/ladder-local --mirror data/strata/raw --dblp-mb 8
T="python tools/strata/train/run_train.py --rung tiny --data data/strata/ladder-local --batch 32 --updates 96"
T="$T --threads 1"
$T --out data/strata/runs-ladder/tiny-d                          # uninterrupted
$T --out data/strata/runs-ladder/tiny-e --session-updates 32     # PAUSED at update 32/96
$T --out data/strata/runs-ladder/tiny-e                          # resumed -> state dict identical to tiny-d
python tools/strata/train/export.py --ckpt data/strata/runs-ladder/tiny-d/ckpt.pt \
    --out data/strata/runs-ladder/tiny-d/weights-int.npz --check sel-zookeeper --bytes 12000     # test C: 0 mismatches
python tools/strata/train/run_eval.py --ckpt data/strata/runs-ladder/tiny-d/ckpt.pt --sources sel-zookeeper
python tools/strata/train/run_eval.py --ckpt ... --sources test-apache      # REFUSED (exit 1); ext-bible too
```

`sel-zookeeper` (300,000 B) with the tiny model after 96 updates of 32 streams: 0.1523 / 0.1527 bpb (widths 16 /
40), 73 s. The driver of the kernel was rehearsed on CPU with the tiny model
(`build_kernel.py --mode dry --out DIR`, then `python DIR/strata-ladder.py data/strata/ladder-local`): 12 steps,
all ok, 3.3 minutes, resume difference 0.0000.

| Model, batch (4 threads, `--updates 32` unless noted) | KB/s | Updates/s | Feature share |
|---|---|---|---|
| tiny (16, 40), 32 streams, 96 updates | 36.2 | 4.42 | 39 % |
| tiny, 256 streams | 39.4 | 0.60 | 33 % |
| G0 widths (0, 88, 208), 64 streams | 16.3 | 1.00 | 17 % |
| `ladder3`, 16 streams | 7.6 | 1.86 | 14 % |
| `ladder3`, 64 streams | 8.4 | 0.51 | 8 % |

(`summary.json` of each run directory; the bake-off recorded 80 minutes for 2,048 updates of 64 streams, 7 KB/s.)
On the CPU `ladder3` is bound by the matrix products, which is the part a GPU removes.

## 7. T4 projection (INFERENCE) and what the smoke run measures

Per window of 256 bytes, `ladder3`: about 7,700 small device operations in the loops (3 per step for each of the
4 accumulator scans, 1 per step for each backward scan, about 14 per step for the mixers) and about 3,600 per
window-equivalent for the features. At 25-40 us per small operation on a Kaggle host that is 0.3-0.45 s per
window, independent of the batch. Matrix products: 0.39 TFLOP per window at 512 streams (forward + backward),
0.10-0.13 s at 3-4 TFLOP/s effective. Projection: **240 KB/s at 512 streams (range 150-350), 360 KB/s at 1,024
(range 200-500)**; memory about 5-6 GB at 512 and 10-11 GB at 1,024 for width 456 (six saved activation tensors of
B x 256 x 456 floats per layer), so 1,024 may not fit in the T4's 15 GB.

Recommendation for a (88, 208, 456) run seeing about 1 GB: **batch 512, 7,648 updates** (7,648 x 512 x 256 =
1.0024 GB, 239 groups), lr 2e-3 unless the probe says otherwise, two seeds at once on a T4 x2 session. Expected
1.2 hours of session time (range 0.8-2.0); `LONG` in `build_kernel.py` stops training at 150 minutes and the script
at 180. If the smoke run measures less than 100 KB/s, cut the bytes rather than extend the session.

The smoke kernel (deadline 27 minutes, each step a subprocess with a timeout from the time left) runs: data slice
(64 MB text, 8 MB per log system, 8 MB XML); `features` on cuda against numpy; test B for `ladder3` and `ladder4`
with random weights; bytes/s, updates/s, feature seconds and peak memory for `ladder3` at 256 / 512 / 1,024,
`ladder4` at 256 / 512 and `r4` at 256 (64 updates each, one queue per GPU); 256 updates of `ladder3` at batch 512
uninterrupted (GPU 0) and paused at 128 + resumed (GPU 1), difference of the logged costs; the same at lr 1e-3 and
4e-3; test B on the trained checkpoint; export + test C on 12,000 bytes of Zookeeper on the Kaggle CPU; 100 KB of
`sel-zookeeper`. Outputs: `/kaggle/working/smoke-report.json`, `logs/*.log`, run directories.

UNTESTED until that run: every CUDA code path (int32 shifts with tensor counts, int32 gather / scatter, stable
sort, `unique`, `cummax`), exactness of float32 `matmul` on the device at widths 456 and 928 (argued by the 2^24
bound, not measured), memory, throughput, two processes on 4 vCPUs, resuming a long run across sessions
(`resume_glob` copies a previous output attached as a dataset; never exercised).

## 8. Open problems and doubts

- Chunk length and short chunks (weak point 4): 8,192 keeps the comparison with G0; 32,768 follows the
  specification. Not decided here.
- Batch 512 gives 7,648 updates for 1 GB; 256 would give twice the updates at about 60 % of the bytes per second
  (INFERENCE). The smoke probe measures speed, not which one trains better.
- Evaluation cost on CPU: 4 files x 300 KB in 118 s with the G0 widths (MEASURED; 33 s of it features), i.e.
  about 7 minutes per MiB of stream length whatever the number of files, more for `ladder3` (INFERENCE). That is at
  the 9-minute limit: `--minutes` pauses and resumes the model pass; the features (about 2 minutes per MiB of
  stream length with numpy) are recomputed on resume.
- `sel-zookeeper` is the only selection file the kernel downloads; the other three need Silesia and are evaluated
  locally from the returned checkpoint (`ckpt.pt` carries the weights, `Q0` and the rung).
- DBLP is new to the project: it is not a selection or decision source today (`viability/sources.py` was read:
  its structured decision slices are GH Archive and `nci`), and it must stay out of any future one.
- The specification's training list (8.6) also names source code, JSON and CSV corpora; none was added.

## 9. Commands

```
python tools/strata/train/build_kernel.py --mode smoke     # writes notebooks/strata-ladder/ (not pushed)
python tools/strata/train/build_kernel.py --mode long      # after the smoke results; edit LONG in build_kernel.py
```
