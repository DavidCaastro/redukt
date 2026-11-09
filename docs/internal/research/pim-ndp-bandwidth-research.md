# PIM / NDP / Computational Memory Research
# Reducing DRAM Bandwidth in Mat-Vec Workloads from Userspace Rust

**Date:** 2026-11-02
**Context:** Mamba-130M inference on i5-1235U (DDR4-2667, ~21 GB/s theoretical, ~60% saturated)
**Bottleneck:** Sequential mat-vec multiplications (768×1536, 1536×768, 50280×768), read-only weights
**Goal:** Identify what can be exploited from userspace Rust without kernel drivers or exotic hardware

---

## Executive Summary

Hardware PIM (UPMEM, Samsung HBM-PIM, SK Hynix AiM, MVDRAM) is uniformly not available
on commodity DDR4 laptop hardware and cannot be accessed from userspace without either (a)
specialized hardware slots, (b) FPGA-based memory controllers, or (c) purchasing dedicated
UPMEM DDR4 modules. None of these apply to the i5-1235U.

However, the software-only landscape is rich. Five techniques are immediately applicable to
Redukt in pure Rust on existing hardware and together represent a credible path to 1.5–2.5×
throughput improvement without any hardware change.

---

## Section 1 — Hardware PIM / NDP Landscape

### 1.1 UPMEM

**What it is:** The only commercially available general-purpose PIM architecture. Integrates
RISC-style DPU cores directly into DDR4 DRAM chips (64 MiB DRAM + 64 KiB scratchpad per
core). Exposed via userspace library `libdpu.so` — no kernel driver needed.

**Programming model:** C (LLVM toolchain). Rust FFI to `libdpu.so` is theoretically possible
but not officially supported as of 2025. Host binary runs on CPU; device binary compiled for
DPU cores.

**Commodity hardware?** NO. Requires purchasing UPMEM DDR4 DIMMs (~$2,000–$5,000 per server).
Cannot be used in a laptop DDR4 slot — these are custom modules that replace standard DIMMs.

**Realistic speedup for mat-vec:** UPMEM targets memory-bound operations. Empirical bandwidth
achieved: 11.6–15.5% of peak due to software bottlenecks (coarse thread scheduling, no
interleaving). Net speedup for LLM GEMV is 1.5–3× depending on quantization.

**Open-source:** SimplePIM (CMU-SAFARI/SimplePIM) — 66–83% code reduction for PIM programs,
10–37% throughput gain over baseline UPMEM C programs. Requires UPMEM hardware.

**Verdict for Redukt:** NOT AVAILABLE. Would require hardware purchase and hardware replacement.

Sources:
- https://arxiv.org/pdf/2310.01893 (SimplePIM)
- https://arxiv.org/pdf/2510.15927 (UPMEM Unleashed)
- https://bu-icsg.github.io/publications/2024/pim_sigmetrics_2024.pdf (scalability limits)

---

### 1.2 Samsung HBM-PIM (Aquabolt-XL / FIMDRAM)

**What it is:** HBM2 memory stack with in-memory compute units. Each DRAM bank has an
Arithmetic Functional Unit (AFU) with 16-wide FP16 multiply-add. Marketed for ML accelerators.

**Commodity hardware?** NO. HBM is a GPU/datacenter package technology, not DDR4 DIMMs.
Requires a compatible GPU or accelerator card (NVIDIA H200, AMD MI series).

**Userspace access:** Via GPU driver extensions (CUDA/ROCm vendor API), not raw userspace.

**Speedup:** 2–8× for HBM-bound GEMV workloads on large GPU accelerators.

**Verdict for Redukt:** NOT AVAILABLE. GPU-only, incompatible with CPU-only constraint.

Sources:
- https://www.servethehome.com/samsung-hbm2-pim-and-aquabolt-xl-at-hot-chips-33/
- https://arxiv.org/html/2401.14428v1 (CNM landscape overview)

---

### 1.3 SK Hynix AiM (Accelerator-in-Memory)

**What it is:** GDDR6 PIM architecture. 16 FP16 multipliers + adder tree + accumulator + LUT
activations per bank. Targets ML inference on mobile/edge devices.

**Commodity hardware?** NO. GDDR6 is discrete GPU memory. Not in any laptop DDR4 slot.

**Userspace access:** Custom DRAM commands via vendor driver. Not publicly available API.

**Verdict for Redukt:** NOT AVAILABLE.

Sources:
- https://blocksandfiles.com/2022/02/16/sk-hynix-builds-dram-processor-in-memory-chip/
- https://arxiv.org/html/2401.14428v1

---

### 1.4 MVDRAM (Processing-Using-DRAM on Unmodified DDR4)

**What it is:** The most technically interesting finding. MVDRAM exploits analog DRAM
properties (simultaneous row activation, RowCopy, MAJX majority operations) on STANDARD
unmodified DDR4 modules to execute GeMV in-memory. Tested on SK Hynix DDR4-2400 modules with
Intel Core i7-9700K.

**Performance:** Up to 7.29× speedup and 30.5× energy efficiency for sub-4-bit GeMV
operations. 2.18× end-to-end for 2-bit models, 1.31× for 4-bit models.

**The catch — hardware requirement:** Requires a MODIFIED MEMORY CONTROLLER. The paper uses an
FPGA-based memory controller (Xilinx Alveo U200) that bypasses the CPU's integrated IMC to
issue non-standard DRAM timing commands (tRAS/tRP violations). The Intel i5-1235U's integrated
memory controller strictly enforces JEDEC timings and cannot issue these commands.

**Kernel driver?** Not just a driver — requires physically replacing the memory controller with
an FPGA. The operation is analogous to Rowhammer bit-flips but intentionally controlled via
timing commands that consumer IMCs will never issue.

**Verdict for Redukt:** NOT AVAILABLE on i5-1235U. Fundamental hardware constraint: the
integrated IMC cannot be bypassed from software. This is research-grade hardware modification.

**Future interest:** If Redukt ever targets specialized inference hardware (custom SoC), MVDRAM
principles are directly applicable for weight-only GeMV acceleration.

Sources:
- https://arxiv.org/html/2503.23817v1 (MVDRAM paper)
- https://quantumzeitgeist.com/using-unmodified-dram-mvdram-accelerates-matrix-vector-multiplication-for-efficient-llm-inference/
- https://arxiv.org/pdf/2505.05266 (PUDTune calibration)

---

### 1.5 x86 Instructions That Process Data "Closer to Memory"

**Non-temporal instructions (MOVNTQ, MOVNTPS, MASKMOVQ):** These bypass the cache hierarchy
on WRITE. They do NOT reduce bandwidth on reads — they skip the cache on stores, useful for
write-only streaming buffers. NOT applicable to weight streaming (read-only).

**PREFETCHNTA:** Brings one cache line into one L2 set with non-temporal replacement priority.
Used to pre-load weight data without polluting L3 with data that will be read once.
This IS applicable and is addressed in Section 2.

**MOVNTDQA:** Non-temporal load — loads from write-combining memory or WC buffers. On normal
DDR4 (WB policy) it behaves like a regular load. NOT a bandwidth reduction mechanism.

**Verdict:** No x86 instruction exists that places computation inside DRAM. All x86 NDP is
software-side cache management, not in-DRAM computing.

Sources:
- https://www.felixcloutier.com/x86/prefetchh
- https://instructionsets.com/x86/prefetchnta/
- https://community.intel.com/t5/Intel-ISA-Extensions/Do-Non-Temporal-Loads-Prefetch/m-p/1027104

---

### 1.6 Cache-Resident LLM Inference (GB-Scale LLC)

**What it is:** 2026 research demonstrating that server CPUs with GB-scale L3 caches (Intel
Sapphire Rapids Xeon w/ HBM, 3D-stacked V-cache AMD Threadripper) can keep entire model weight
sets cache-resident. Separates weight-centric operators from attention/KV into dedicated
resource domains.

**On i5-1235U:** The i5-1235U has 12 MB L3 cache. Mamba-130M weights at Q8 = ~130 MB, at Q4 =
~65 MB. Both are 5–11× larger than the entire L3. Full cache residency is NOT achievable on
this platform.

**Partial residency (hot layers):** The SSM recurrence state update (TimeMix equivalent in
Mamba) is ~8–12 MB of weights per token traversal. Even partial residency of the most
bandwidth-critical layers would require targeted management — see Section 2.3.

Sources:
- https://arxiv.org/abs/2606.25353 (Cache-Resident LLM Inference)

---

## Section 2 — Software Techniques Available on Commodity Hardware

These are applicable to i5-1235U from pure userspace Rust. Sorted by estimated impact.

---

### 2.1 Weight Quantization to Q4 / Q2 (HIGHEST IMPACT — 2–4× bandwidth reduction)

**Mechanism:** Each token's forward pass streams all weight bytes once. Reducing bits-per-weight
directly reduces DRAM bandwidth proportionally:
- BF16 (2 bytes): baseline
- Q8 (1 byte): 2× bandwidth reduction — ALREADY IMPLEMENTED in Redukt
- Q4 (0.5 bytes): 4× reduction vs BF16, 2× vs Q8
- Q2 / ternary (0.25 bytes): 8× reduction vs BF16

**Evidence from Mamba literature:**
- Quamba2 (W4A8): 3.89× memory reduction, 3.05× faster generation on GPU
- Fully Quantized Mamba 1.58-bit (ternary): ~90% parameter bit reduction
- BitMamba-2 (ternary, CPU): 82.5 tok/s (255M) and 29.6 tok/s (1B) on Apple M1
  — vs Redukt Mamba-130M at 34 tok/s Q8 on i5-1235U, this is in the same ballpark
- W4A8 halves traffic vs Q8: from 5.14 GB/token → 2.64 GB/token for Mamba

**BPB risk for Redukt:** Quamba2 W4A8 on Mamba2 shows degradation is task-dependent.
For compression specifically, empirical validation is mandatory. Existing Redukt Q4 attempt
showed +12.7% BPB (see MEMORY.md optimization summary) — meaning naive Q4 FAILED. A
specialized Q4 for compression (keeping SSM state in F32, quantizing only projection weights)
may be viable. This is the primary research direction.

**Implementation in Rust:** Feasible without external dependencies. Pack two INT4 values per
byte. Dequantize in the AVX2 inner loop with `_mm256_cvtepi8_epi16` and table lookups.
AVX2 VNNI (_mm256_dpbusd_epi32) provides dot products over UINT8 with INT8, usable for W4A8.

**Verdict:** HIGH PRIORITY. Even 1.5× bandwidth reduction at +3% BPB would be net positive
for throughput. Needs targeted compression quality benchmark before commit.

Sources:
- https://arxiv.org/html/2503.22879 (Quamba2)
- https://aclanthology.org/2025.coling-main.316.pdf (Fully Quantized Mamba 1.58-bit)
- https://engrxiv.org/preprint/view/6686 (BitMamba-2 cross-platform)
- https://arxiv.org/pdf/2605.30571 (Physical AI inference gap)

---

### 2.2 L2-Sized Weight Tile Layout (MEDIUM IMPACT — 1.12–1.36× bandwidth reduction)

**Mechanism:** Instead of storing weight matrices in row-major order (768×1536 F32/Q8),
partition them into tiles whose footprint fits in L2 cache (~256–512 KB on i5-1235U P-cores).
Read tiles in compute-consumption order so each cache line is used completely before eviction.

**cflow evidence (Pipeline-Native Transformers, 2026):**
- "128×256 Q4 tiles, ~18 KB each" stored in compute-consumption order
- L1-d cache misses reduced 7.29× vs row-major for GEMV
- 1.12–1.36× GEMV speedup measured across projection layer sizes
- Largest speedup (1.36×) when activation vector fits entirely in L1

**Application to Redukt:**
- 768×1536 projection: current layout = row-major F32/Q8
- Tile into 256×512 subtiles (8 KB Q8 or 32 KB F32 — fits L2)
- Inner loop over tile columns (512) → dot product with activation slice
- This is a weight file format change; `model.rs` loading order changes, no codec impact

**Rust implementation:** Straightforward. Change the buffer layout during model load (or
pre-convert the SafeTensors once on disk). No unsafe required beyond existing AVX2 paths.

**Combined with prefetch (2.3):** Tile layout + prefetch NTA together hit ~1.5–1.7× GEMV
throughput in cflow measurements.

Sources:
- https://arxiv.org/html/2608.23841v1 (Pipeline-Native Transformers / cflow)
- Cross-platform SSM CPU benchmark: CPU SSM inference with tiled weights 1.12–1.36× speedup

---

### 2.3 PREFETCHNTA for Weight Streaming (MEDIUM IMPACT — prevents L3 pollution)

**Mechanism:** When streaming read-only weight tiles that will not be reused (each token
reads each weight once), use `_mm_prefetch(ptr, _MM_HINT_NTA)` to bring the data into L2
without filling L3 ways. This keeps L3 free for the SSM recurrent state (which IS reused
every token) and activation vectors.

**Observed in production systems:**
- Nemotron-30B CPU (nemotron-30b-cpu-256k project): "non-temporal hints for cache eviction"
  for older KV tokens and weight streams, auto-tuned to occupy ≤25% of L3
- Intel Optimization Manual: PREFETCHNTA brings one cache line into one L2 set only
- Net effect: L3 stays warm with state data; weights stream through L2 only

**On i5-1235U specifics:**
- P-core L2: 1.25 MB per core. E-core L2: 2 MB shared (4 cores). L3: 12 MB shared.
- SSM recurrence state for d_model=768, d_state=16: 768×16×4 bytes = 48 KB — fits in L1!
- Linear projection weights 768×1536 Q8 = 1.15 MB — fits exactly in one P-core L2
- This is the ideal scenario for PREFETCHNTA: prefetch next tile while computing current

**Implementation in Rust:**
```rust
// In AVX2 matmul kernel, prefetch rows N+2 ahead
unsafe {
    std::arch::x86_64::_mm_prefetch(
        weight_ptr.add(row_stride * (row + 2)) as *const i8,
        std::arch::x86_64::_MM_HINT_NTA,
    );
}
```
No unsafe surface increase beyond what already exists in `tensor.rs`.

**Verdict:** Low-risk, 0 BPB impact, potentially 10–20% throughput gain by keeping state hot.
Most impactful for the 50280×768 LM head (100% DRAM-bound, never fits in any cache).

Sources:
- https://github.com/synchopate/nemotron-30b-cpu-256k (NTA L3 isolation strategy)
- https://instructionsets.com/x86/prefetchnta/
- https://www.felixcloutier.com/x86/prefetchh (Intel ISA ref)

---

### 2.4 2MB Transparent HugePages for Weight Buffers (LOW-MEDIUM IMPACT — 5–24% TLB reduction)

**Mechanism:** Model weights (130 MB at Q8) span ~32,500 standard 4 KB pages. The CPU's TLB
holds ~1,500 entries (P-core). During weight streaming, TLB misses trigger page-table walks at
~40–100 ns each. Using 2 MB pages reduces the page count 512×: 130 MB = 65 huge pages,
comfortably fitting in STLB (second-level TLB, 1,536 entries on Alder Lake).

**Measured impact:**
- Nemotron-30B CPU project: "24% page-table walk overhead eliminated" during weight streaming
  (measured on AMD Zen 5; similar benefit expected on Intel with lighter MoE overhead)
- Google TCMalloc Temeraire: 7% fleet-wide RPS improvement via THP
- TLB miss latency reduction: up to 40% in memory-intensive benchmarks

**On Linux (Rust):**
```rust
// After mmap of weight file:
libc::madvise(ptr, size, libc::MADV_HUGEPAGE);
```
Or: use `mmap` with `MAP_HUGETLB | MAP_HUGE_2MB` flags directly.

**On Windows (Redukt's current target):**
- Requires `SeLockMemoryPrivilege` in token — available via Local Security Policy or
  running as Administrator
- Call `VirtualAlloc` with `MEM_LARGE_PAGES` flag
- OR: Windows THP (enabled in Windows 11 via registry) — less reliable than Linux
- This is an operational constraint; for production Linux deployment it's straightforward

**Verdict:** Worth adding for Linux builds. Windows path requires privilege escalation which
is not acceptable for a compression CLI. Skip on Windows, enable on Linux.

Sources:
- https://github.com/synchopate/nemotron-30b-cpu-256k (2MB HugePages implementation)
- https://www.redpanda.com/blog/oxla-using-huge-pages-in-linux-applications-part-2-transparent-hugepage/
- https://forums.factorio.com/viewtopic.php?t=106301 (Windows large pages)

---

### 2.5 Aggressive In-Memory Weight Repacking (MEDIUM IMPACT — +19% GEMV throughput)

**Mechanism:** For Q4 weights, naive storage requires unpacking two INT4 values per byte before
multiplication. This unpacking (bit-shifting, masking) constitutes 30% of total GEMV compute
per the nemotron-30b-cpu-256k measurements. Repacking weights into AVX2-native layouts
(where VPDPBUSD / AVX2 VNNI instructions can consume them with single-cycle vector permutes)
eliminates this overhead.

**Observed:** "Eliminates 30% bit-unpacking overhead via single-cycle vector-permute-dotproduct
unsigned 8-bit to signed 32-bit (VPDPBUSD) instructions, yielding +19% GEMV/GEMM throughput."

**i5-1235U support:** Intel 12th gen (Alder Lake) supports AVX-VNNI (vector neural network
instructions) in addition to AVX2. VPDPBUSD (dot product of uint8 × int8 with int32
accumulation) is available. This is distinct from AVX-512 VNNI which is NOT present on
mobile Alder Lake.

**Rust implementation:** requires unsafe + x86_64 intrinsics. Ties directly into the
existing `tensor.rs` Q8 matmul path. Would need a Q4 matmul variant using `_mm256_dpbusd_epi32`
(AVX-VNNI) or fallback via `_mm256_maddubs_epi16` (AVX2).

**Verdict:** Applicable only if Q4 quantization is pursued (Section 2.1). Zero benefit at Q8.

Sources:
- https://github.com/synchopate/nemotron-30b-cpu-256k (weight tile repacking)
- https://arxiv.org/html/2503.22879 (Quamba2 AVX2/AVX-VNNI W4A8 fast path)

---

### 2.6 Pipeline-Native Parallel Weight Streaming (SPECULATIVE — up to 2× bandwidth per token)

**Mechanism:** Pipeline-Native Transformers (cflow, 2026) demonstrates that by redesigning
the inter-layer dependency graph, two successive layer's weight streams can be processed in
parallel — effectively halving the critical-path bandwidth. The paper shows 2.00× reduction
in critical-path weight bandwidth (9.00 → 4.50 MB/token).

**Applicability to Mamba/SSM:** The paper explicitly does NOT address SSMs. SSM recurrence
has a strict sequential dependency (state[t] depends on state[t-1]), and layers are connected
via residual stream with no obvious delay-insertion point. However:
- The LM head (50280×768, the largest matmul) is independent of the SSM recurrence
- It can be overlapped with the SSM state update using a two-thread scheme (one thread
  does SSM forward, another prefetches/computes LM head partial sums)
- Existing Redukt parallel matmul (R-054) already spawns threads for rows>4096;
  the LM head (50280 rows) already uses this path

**Verdict:** The LM head parallelism is already partially addressed by R-054. Full
pipeline-native redesign would require architectural changes beyond scope.

Sources:
- https://arxiv.org/html/2608.23841v1 (Pipeline-Native Transformers)
- https://www.alphaxiv.org/abs/2608.23841 (abstract)

---

### 2.7 MoE Expert Selective Loading (NOT APPLICABLE to Mamba-130M)

**Mechanism:** cflow loads only top-k MoE experts, skipping 87.5–93.75% of expert weight
traffic. On Gemma 4 geometry (128 experts, top-8), this is a 16× bandwidth reduction for
expert parameters.

**Verdict:** Mamba-130M is a dense model with no MoE. NOT APPLICABLE.

---

## Section 3 — Quantitative Summary

| Technique | Hardware Required | Userspace Rust? | Est. Speedup | BPB Risk | Priority |
|---|---|---|---|---|---|
| Q4 weight quant | None | Yes | 1.5–2.0× | HIGH (+12.7% seen) | INVESTIGATE |
| Q2/ternary weight quant | None | Yes | 2.0–4.0× | VERY HIGH | FUTURE |
| L2-tile weight layout | None | Yes | 1.12–1.36× | Zero | HIGH |
| PREFETCHNTA streaming | None | Yes | 1.10–1.20× | Zero | HIGH |
| AVX-VNNI tile repacking | None (Alder Lake has it) | Yes | +19% GEMV | Zero (orthogonal) | WITH Q4 |
| 2MB HugePages | None (Linux) | Yes (Linux) | 1.05–1.24× | Zero | MEDIUM (Linux) |
| Pipeline-native overlap | None | Partial (R-054) | 1.2–2.0× | Zero | FUTURE |
| UPMEM PIM | UPMEM DDR4 modules | Via libdpu.so | 1.5–3× | Zero | NOT AVAILABLE |
| Samsung HBM-PIM | HBM GPU | Via CUDA/ROCm | 2–8× | N/A | NOT AVAILABLE |
| SK Hynix AiM | GDDR6 GPU | Via vendor SDK | 2–4× | N/A | NOT AVAILABLE |
| MVDRAM PUD | FPGA memory controller | Not from IMC | 7.29× (sub-4-bit) | N/A | NOT AVAILABLE |
| x86 in-DRAM compute | None exists | N/A | 1.0× | N/A | IMPOSSIBLE |

---

## Section 4 — Recommended Action Plan for Redukt

### Immediate (zero BPB risk, pure throughput):

**R-059: PREFETCHNTA in matmul kernel**
- Add `_mm_prefetch(..., _MM_HINT_NTA)` in the hot path of `matmul_q8_avx2` and `matmul_f32_avx2`
  prefetching 2 rows ahead
- Cost: ~20 lines of unsafe Rust in `tensor.rs`
- Expected: +10–20% tok/s, no BPB change, verifiable with controlled A/B

**R-060: L2-tile weight layout**
- During model load, reorder weight rows into 256×512 Q8 tiles (fits one L2 way)
- Change: `model.rs` loading, one-time transformation of weight buffer
- Expected: +12–36% GEMV throughput for projection layers (768×1536, 1536×768)
- Cost: ~100 lines in model_loader.rs, zero codec change

### Investigate (BPB validation required):

**R-061: Selective Q4 for projection weights**
- Keep SSM recurrence state update in Q8/F32 (accuracy-critical)
- Quantize only the projection weights (768×1536, 1536×768) to Q4
- Use group quantization (group size 64 or 128) to minimize BPB degradation
- Target: 1.8× bandwidth reduction at ≤3% BPB cost
- Validation: full 19-file benchmark required before commit

### Future (non-trivial architecture change):

**R-062: AVX-VNNI Q4 matmul kernel** (depends on R-061 viability)
**R-063: 2MB HugePages for Linux build** (Linux deployment)
**R-064: MambaByte byte-level model** (eliminates tokenizer bandwidth, see R-041)

---

## Section 5 — Why Full Cache Residency Is Impossible on i5-1235U

The "cache-resident inference" papers (2606.25353) require GB-scale LLC. The i5-1235U has 12 MB.
At Q4, Mamba-130M = ~65 MB. The LM head alone (50280×768 Q8 = 37 MB) exceeds the entire L3.

The practical implication: for the LM head, Redukt will always be 100% DRAM-bandwidth-bound
regardless of software technique. The only levers are:
1. Reduce bytes streamed (quantization Q4/Q2)
2. Reduce the LM head computation entirely (top-K vocabulary pruning — already discussed in R-056 research)
3. Cache the output distribution (PredCache R-048/R-049 — was attempted, +17.4% BPB, abandoned)

The LM head is Redukt's permanent ceiling unless the model is retrained at byte level (R-041).

---

## Sources (Full List)

- [SimplePIM: Software Framework for PIM](https://arxiv.org/pdf/2310.01893)
- [UPMEM Unleashed: Software Secrets for Speed](https://arxiv.org/pdf/2510.15927)
- [PIM Scalability Limitations (SIGMETRICS 2024)](https://bu-icsg.github.io/publications/2024/pim_sigmetrics_2024.pdf)
- [Landscape of CNM and CIM (survey)](https://arxiv.org/html/2401.14428v1)
- [Samsung HBM2-PIM Aquabolt-XL (ServeTheHome)](https://www.servethehome.com/samsung-hbm2-pim-and-aquabolt-xl-at-hot-chips-33/)
- [SK Hynix AiM DRAM PIM Chip](https://blocksandfiles.com/2022/02/16/sk-hynix-builds-dram-processor-in-memory-chip/)
- [MVDRAM: GeMV in Unmodified DRAM](https://arxiv.org/html/2503.23817v1)
- [MVDRAM analysis (QuantumZeitgeist)](https://quantumzeitgeist.com/using-unmodified-dram-mvdram-accelerates-matrix-vector-multiplication-for-efficient-llm-inference/)
- [PUDTune: Multi-Level Charging for PUD Calibration](https://arxiv.org/pdf/2505.05266)
- [Processing Using Memory Paradigm (RowCopy, MAJX)](https://arxiv.org/pdf/1610.09603)
- [Cache-Resident LLM Inference (GB-Scale LLC)](https://arxiv.org/abs/2606.25353)
- [Memory-Bound Not Bandwidth-Limited: Physical AI Inference Gap](https://arxiv.org/pdf/2605.30571)
- [Pipeline-Native Transformers / cflow](https://arxiv.org/html/2608.23841v1)
- [Nemotron-30B CPU-256K (NTA, HugePages, tile repacking)](https://github.com/synchopate/nemotron-30b-cpu-256k)
- [Quamba2: W4A8/W8A8 PTQ for Mamba2](https://arxiv.org/html/2503.22879)
- [Fully Quantized Mamba 1.58-bit](https://aclanthology.org/2025.coling-main.316.pdf)
- [BitMamba-2 Cross-Platform CPU Benchmark](https://engrxiv.org/preprint/view/6686)
- [Breaking the Memory Wall: LLM Inference on CPU](https://arxiv.org/pdf/2410.04466)
- [prefetchnta x86 ISA reference](https://instructionsets.com/x86/prefetchnta/)
- [Intel PREFETCHh reference](https://www.felixcloutier.com/x86/prefetchh)
- [Intel Non-Temporal Load Community Thread](https://community.intel.com/t5/Intel-ISA-Extensions/Do-Non-Temporal-Loads-Prefetch/m-p/1027104)
- [Transparent Huge Pages (Linux kernel docs)](https://docs.kernel.org/admin-guide/mm/transhuge.html)
- [Huge Pages Linux Applications (Redpanda/Oxla)](https://www.redpanda.com/blog/oxla-using-huge-pages-in-linux-applications-part-2-transparent-hugepage/)
- [New Tools for PIM Architectures (survey 2025)](https://arxiv.org/pdf/2508.19868)
- [EMMLI: Efficient MatMul in On-Chip Memory](https://arxiv.org/html/2312.15896v2)
- [Compiler-First SSM O(1) Autoregressive Caching](https://arxiv.org/pdf/2603.09555)
