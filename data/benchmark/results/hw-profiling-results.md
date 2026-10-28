# Hardware Profiling Results — Protocols 01-08

> Date: 2026-10-28
> CPU: Intel Core i5-1235U (2P+8E, 12 threads)
> RAM: 32GB DDR4-3200 @ 2667 MT/s (Crucial CT16G4SFRA32A x2)
> Build: cargo build --release (Q8 SIMD auto-dispatch)

---

## PROTO-03: Forward Pass Phase Breakdown [MEASURED]

| File | Tokens | tok/s | TimeMix % | ChannelMix % | Layers % | Head % |
|---|---|---|---|---|---|---|
| enwik8 10KB | 3005 | 36.5 | 88.0* | N/A* | 88.0 | 12.0 |
| apache-access.log 10KB | 3570 | 34.4 | 74.1 | 13.6 | 87.8 | 12.2 |
| facturae-batch.xml 10KB | 2872 | 40.9 | 72.7 | 14.6 | 87.3 | 12.7 |

*enwik8 run was the first profiling pass before TimeMix/ChannelMix split was added.

### Per-token latency [MEASURED]:

| File | TimeMix (ms/tok) | ChannelMix (ms/tok) | Head (ms/tok) | Total (ms/tok) |
|---|---|---|---|---|
| apache | 20.095 | 3.691 | 3.316 | 27.102 |
| facturae | 16.393 | 3.289 | 2.863 | 22.545 |

### Key finding:
**TimeMix dominates at 73-88% of forward time.** Head is only 12%.
The CPU-ARCHITECTURE-MAP.md hypothesis that head=70-80% was WRONG.

TimeMix contains many small operations:
- 12 layers x (~10 matmuls + lerp + element-wise + state matrix update)
- Most matmuls are 768x768 (L2-resident, fast)
- But LoRA matmuls (w1,w2,a1,a2,g1,g2) are F32 scalar (no SIMD!)
- State matrix update (N*N per head, 12 heads) is pure scalar
- Element-wise ops (exp, sigmoid, multiply) are scalar

ChannelMix is ~14% — simpler (2 matmuls + activation).
Head is ~12% — single 65536x768 matmul, DRAM-bound but only ~3ms/token.

---

## PROTO-01: Power Configuration [MEASURED]

- Power plan: "Equilibrado" (Balanced)
- PL1 (sustained): 15W [SOURCED: Intel ARK i5-1235U]
- PL2 (burst): 55W [SOURCED: Intel ARK]
- RAM speed: 2667 MT/s (rated 3200 MT/s) [MEASURED: Win32_PhysicalMemory]
- **RAM is underclocked by 17%** — XMP/DOCP not enabled

---

## PROTO-02: RAM XMP Status [MEASURED]

- Crucial CT16G4SFRA32A rated DDR4-3200
- Running at DDR4-2667 (ConfiguredClockSpeed=2667)
- **Action required**: Enable XMP in BIOS for +20% DRAM bandwidth
- Expected impact: +5-10% on DRAM-bound operations (head matmul, model load)

---

## PROTO-04: DRAM Bandwidth [MEASURED]

| Buffer Size | BW (GB/s) | Resides in |
|---|---|---|
| 1 MB | 63.35 | L2/L3 |
| 4 MB | 57.17 | L3 |
| 16 MB | 22.02 | L3 overflow -> DRAM |
| 64 MB | 20.61 | DRAM |

- Theoretical max (DDR4-2667 dual channel): 42.7 GB/s
- Measured DRAM: **20.61 GB/s** = 48% efficiency [MEASURED]
- At DDR4-3200 (if XMP enabled): theoretical 51.2 GB/s, estimated practical ~24.8 GB/s

### DRAM demand from Redukt:
- Head matmul: 48MB * 39 tok/s = ~1.87 GB/s [CALCULATED]
- DRAM utilization: 1.87 / 20.6 = **9%** [CALCULATED]
- **DRAM bandwidth is NOT the bottleneck** at current throughput

---

## PROTO-05: Thread Affinity P-core vs E-core [MEASURED]

Test: compress enwik8 2KB, pinned to each logical processor individually.

| LP | Affinity Mask | tok/s | Notes |
|---|---|---|---|
| 0 | 0x001 | 34.7 | |
| 1 | 0x002 | 36.4 | |
| 2 | 0x004 | 36.8 | |
| 3 | 0x008 | 37.1 | |
| 4 | 0x010 | 36.7 | |
| 5 | 0x020 | 36.8 | |
| 6 | 0x040 | 36.3 | |
| 7 | 0x080 | 37.6 | Fastest |
| 8 | 0x100 | 37.4 | |
| 9 | 0x200 | 36.6 | |
| 10 | 0x400 | 36.5 | |
| 11 | 0x800 | 37.4 | |

- **Range: 34.7 - 37.6 tok/s (8% spread)**
- **P-core vs E-core difference: NEGLIGIBLE (<8%)**

### Why P and E cores perform similarly:
The workload is dominated by TimeMix (74-88%), which is mostly SCALAR operations
(state updates, element-wise math, F32 LoRA matmuls without SIMD).
Only ~12% of time is in Q8 SIMD matmuls (head) where P-cores would have an advantage.
The small scalar-dominated workload profile neutralizes the P/E core SIMD difference.

### Implication for R-012 (multi-file parallelism):
**ALL 10 cores are usable** for multi-file compression at ~equal throughput.
Expected scaling: ~10x with 10 parallel files (each core does ~36 tok/s).
This is MUCH better than the 2P-only assumption.

---

## PROTO-06: SIMD vs Scalar (end-to-end) [MEASURED]

| Mode | tok/s | Delta |
|---|---|---|
| SIMD (default) | 36.8 | baseline |
| Scalar (REDUKT_NO_SIMD=1) | 32.0 | -13% |

SIMD provides only **+15% end-to-end** speedup because:
- Q8 SIMD matmuls are only ~12% of total time (head)
- TimeMix layer matmuls benefit from SIMD but are interleaved with scalar ops
- LoRA matmuls (w1/w2/a1/a2/g1/g2) are F32 scalar (SIMD disabled for determinism)

---

## PROTO-07: Ring Bus Penalty [NOT MEASURED]

Skipped — PROTO-05 showed P/E cores have negligible performance difference,
making the ring bus clock penalty (3.6 GHz when E active vs 4.7 GHz P-only)
irrelevant for this workload.

---

## PROTO-08: Thermal Sustained Test [MEASURED]

Test: enwik8 50KB, 12830 tokens, 330 seconds continuous compression.

| Token range | tok/s | Delta from start |
|---|---|---|
| 0-1000 | 38.9 | baseline |
| 1000-2000 | 39.1 | +0.5% |
| 5000-6000 | 39.0 | +0.3% |
| 10000-11000 | 39.1 | +0.5% |
| 12000-12830 | 39.2 | +0.8% |

- **Total time: 330.9s at 38.8 tok/s sustained**
- **0% thermal degradation** over 5.5 minutes
- PL1=15W is sufficient for single-thread sustained boost
- Slight improvement over time likely due to model weights warming into cache

---

## Summary: Revised Bottleneck Analysis

```
BEFORE PROFILING (hypothesis):
  Head matmul (DRAM-bound): 70-80%  <-- WRONG
  Layers (compute-bound):   15-20%  <-- WRONG

AFTER PROFILING (measured):
  TimeMix (scalar-dominated): 73-88%  <-- ACTUAL BOTTLENECK
  ChannelMix:                 13-15%
  Head matmul:                12%
  Mixer + Coder:              ~4%
```

### Optimization Priority (revised):

1. **TimeMix scalar operations** (73-88%): LoRA matmuls F32 are scalar.
   State matrix updates are N*N per head (64*64=4096 ops, 12 heads, 12 layers).
   Lerp, sigmoid, exp are all scalar per-element.
   -> KV caching (R-015) WILL help if it eliminates redundant state recalculation.
   -> F32 SIMD for LoRA matmuls would help IF determinism can be maintained.

2. **Multi-file parallelism** (R-012): ALL 10 cores are useful (~10x potential).
   This is now the HIGHEST IMPACT optimization.

3. **Head matmul** (12%): Already SIMD-optimized. Limited further improvement.

4. **XMP RAM** (free): +5-10% on DRAM operations, affects model load + head.

### Revised Throughput Projections:

| Optimization | Speedup | Cumulative B/s |
|---|---|---|
| Current (1 core) | 1x | ~150 B/s |
| R-012 (10 cores) | ~10x | ~1500 B/s |
| R-015 (KV cache) | ~1.5-2x* | ~2250-3000 B/s |
| XMP RAM (3200) | ~1.05x | ~2360-3150 B/s |

*KV cache benefit depends on how much of TimeMix is redundant recalculation.
If state is already maintained incrementally (RNN mode), KV cache may have
LESS impact than originally projected.
