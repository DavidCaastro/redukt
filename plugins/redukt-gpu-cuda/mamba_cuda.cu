/* =====================================================================
 * Redukt GPU Backend Plugin — CUDA Implementation (R-090, R-096)
 *
 * Mamba-1 SSM inference on NVIDIA GPUs via the BackendVTable FFI.
 *   - FFI v1 (single stream, process-global state): redukt_backend_{init,forward,reset_state,free}
 *   - FFI v2 (additive, N independent streams per step): redukt_backend_{abi_version,batch_init,
 *     batch_reset,forward_batch}.  Contract: docs/api/PLUGIN-FFI-V2.md
 *
 * v1 is implemented as the batched pipeline with n_streams=1 on its own private state pool, so the
 * logits of a stream are bit-identical between v1 and v2 for any batch size (see block_sum / k_gemm_rows).
 *
 * Build:  plugins/redukt-gpu-cuda/build.sh   (or `make`)  -> libredukt_gpu.so
 *
 * Official NVIDIA references used (consulted 2026-11-10); PG = CUDA C++ Programming Guide:
 *   PG  = https://docs.nvidia.com/cuda/cuda-programming-guide/
 *   RT  = https://docs.nvidia.com/cuda/cuda-runtime-api/
 *   [PG-SMEM]   shared memory, dynamic shared memory (3rd launch parameter, one dynamic array):
 *               PG 02-basics/writing-cuda-kernels.html#shared-memory
 *               PG 02-basics/writing-cuda-kernels.html#dynamic-allocation-of-shared-memory
 *   [PG-SYNC]   __syncthreads():  PG 02-basics/writing-cuda-kernels.html#thread-block-synchronization
 *   [PG-SHFL]   warp shuffle (__shfl_down_sync; reading an inactive lane is undefined):
 *               PG 05-appendices/cpp-language-extensions.html#warp-shuffle-functions
 *   [PG-FP]     FP non-associativity, FMA, "Deterministic Results" (atomic order varies between runs):
 *               PG 05-appendices/mathematical-functions.html#associativity
 *               PG 05-appendices/mathematical-functions.html#fused-multiply-add-fma
 *   [PG-CC]     compute capabilities (48 KB default smem/block, opt-in above; warp size 32):
 *               PG 05-appendices/compute-capabilities.html
 *   [PG-STREAM] stream ordering, non-blocking streams, pageable cudaMemcpyAsync, cudaStreamSynchronize:
 *               PG 02-basics/asynchronous-execution.html
 *   [RT-MEM]    cudaMalloc ("The memory is not cleared"), cudaHostAlloc, cudaMemcpyAsync, cudaFree:
 *               RT group__CUDART__MEMORY.html
 *   [RT-ERR]    cudaGetLastError (returns and resets the last runtime error):
 *               RT group__CUDART__ERROR.html
 *
 * Copyright 2026 Redukt Contributors. Apache-2.0 License.
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* ---- Launch geometry ----
 * THREADS must be a multiple of the warp size (32, [PG-CC]) so every warp is full and all 32 lanes
 * take part in each __shfl_down_sync with mask 0xffffffff ([PG-SHFL]: reading an inactive lane is
 * undefined).  NWARPS is used to size the inter-warp reduction buffers. */
#define THREADS 256
#define NWARPS  (THREADS / 32)
#define FULL_MASK 0xffffffffu
/* Streams accumulated per pass of the batched GEMM (registers per thread = GEMM_TILE accumulators). */
#define GEMM_TILE 8
/* Upper bound accepted by redukt_backend_batch_init. */
#define MAX_STREAMS 4096u
/* Default dynamic shared memory limit without opt-in ([PG-CC] footnote, [PG-SMEM] 48 KB example). */
#define SMEM_DEFAULT_LIMIT (48u * 1024u)

/* ---- FFI types (must match src/domain/backend_ffi.rs) ---- */

extern "C" {

typedef enum {
    BACKEND_OK                = 0,
    BACKEND_INIT_FAILED       = 1,
    BACKEND_FORWARD_FAILED    = 2,
    BACKEND_VERSION_MISMATCH  = 3,
    BACKEND_WEIGHTS_NOT_FOUND = 4,
    BACKEND_INVALID_ARGUMENT  = 5, /* v2 only: returned as int32_t by the batch entry points */
} BackendResult;

#define BACKEND_FFI_VERSION 1   /* BackendInfo.ffi_version: base (v1) contract, unchanged */
#define BACKEND_ABI_VERSION 2   /* returned by redukt_backend_abi_version(): v2 batch available */
#define BACKEND_NAME_MAX 64

typedef struct {
    uint32_t ffi_version;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint8_t  name[BACKEND_NAME_MAX];
} BackendInfo;

} /* extern "C" */

#define CUDA_TRY(call, code) do {                                   \
    cudaError_t err_ = (call);                                      \
    if (err_ != cudaSuccess) {                                      \
        fprintf(stderr, "[redukt-gpu-cuda] CUDA error at %s:%d: "   \
                "%s\n", __FILE__, __LINE__,                         \
                cudaGetErrorString(err_));                          \
        return (code);                                              \
    }                                                               \
} while (0)

/* ---- Binary weight format (.rbin) ----
 * v1: 44-byte header, word 10 = reserved (0).
 * v2: same 44-byte header, word 10 = tokenizer kind (RBIN_TOK_*).  Body identical.
 * The reader accepts both; v1 files imply tokenizer from vocab (256 -> bytes, else GPT-NeoX BPE). */

#define RBIN_MAGIC      0x55504752u  /* "RGPU" LE */
#define RBIN_VERSION_V1 1u
#define RBIN_VERSION_V2 2u
#define RBIN_TOK_BYTE   0u
#define RBIN_TOK_BPE    1u
#define FLAG_NORM_BIAS   (1u << 0)
#define FLAG_UNTIED_HEAD (1u << 1)

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t d_model;
    uint32_t d_inner;
    uint32_t d_state;
    uint32_t d_conv;
    uint32_t dt_rank;
    uint32_t n_layer;
    uint32_t flags;
    uint32_t tokenizer;  /* v1: reserved (0) */
} RbinHeader;

/* ---- Per-layer weight pointers on GPU ---- */

typedef struct {
    float* norm_w;       /* d_model */
    float* norm_b;       /* d_model or NULL */
    float* in_proj_w;    /* 2*d_inner * d_model */
    float* conv1d_w;     /* d_inner * d_conv */
    float* conv1d_b;     /* d_inner */
    float* x_proj_w;     /* (dt_rank + 2*d_state) * d_inner */
    float* dt_proj_w;    /* d_inner * dt_rank */
    float* dt_proj_b;    /* d_inner */
    float* a_log;        /* d_inner * d_state */
    float* d_vec;        /* d_inner */
    float* out_proj_w;   /* d_model * d_inner */
} LayerWeightsGpu;

/* ---- State pool: recurrent state for `cap` streams + per-step scratch ----
 * All buffers live in device memory for the lifetime of the pool (persistent across steps).
 * Layout is stream-major: stream b (batch position) or slot s (state index) owns a contiguous row. */

typedef struct {
    uint32_t cap;
    float**  conv;       /* [n_layer] -> cap * d_inner * (d_conv-1), indexed by SLOT */
    float**  ssm;        /* [n_layer] -> cap * d_inner * d_state,    indexed by SLOT */
    /* scratch, indexed by BATCH POSITION b in [0, n_streams) */
    float* x;            /* cap * d_model */
    float* normed;       /* cap * d_model */
    float* xz;           /* cap * 2*d_inner */
    float* conv_out;     /* cap * d_inner */
    float* zp;           /* cap * (dt_rank + 2*d_state) */
    float* dt;           /* cap * d_inner */
    float* y;            /* cap * d_inner */
    float* gated;        /* cap * d_inner */
    float* res;          /* cap * d_model */
    float* logits;       /* cap * vocab_size */
    uint32_t* d_slots;   /* cap */
    uint32_t* d_tokens;  /* cap */
    uint32_t* h_slots;   /* cap, pinned host staging ([RT-MEM] cudaHostAlloc) */
    uint32_t* h_tokens;  /* cap, pinned host staging */
    uint8_t*  h_seen;    /* cap, duplicate-slot check */
} StatePool;

/* ---- GPU model ---- */

typedef struct {
    RbinHeader cfg;
    float* d_emb;
    float* d_lm_head;
    float* d_norm_f_w;
    float* d_norm_f_b;
    int    head_is_tied;
    LayerWeightsGpu* layers;
    cudaStream_t stream;
    StatePool v1;        /* cap = 1, used by the v1 entry points */
    StatePool batch;     /* cap = N, used by the v2 entry points (cap 0 until batch_init) */
    int ready;
} GpuModel;

static GpuModel G;

/* ====================================================================
 * Device helpers
 * ==================================================================== */

/* Block-wide sum with a FIXED reduction tree (bit-reproducible run to run):
 *   1. each thread already holds a partial accumulated in a fixed k order,
 *   2. butterfly-free shuffle-down tree inside each warp (offsets 16,8,4,2,1) [PG-SHFL],
 *   3. lane 0 of each warp writes its sum to shared memory [PG-SMEM], __syncthreads() [PG-SYNC],
 *   4. thread 0 adds the NWARPS warp sums sequentially, w = 0..NWARPS-1.
 * No atomics: [PG-FP] "Deterministic Results" states that the ordering of arithmetic atomic
 * operations depends on hardware scheduling and can vary between runs; FP addition is not
 * associative ([PG-FP] #associativity), so an order-varying reduction would change the logits
 * and break encoder/decoder symmetry of the arithmetic coder.
 * Result is broadcast to every thread through s_buf[0]. Requires blockDim.x == THREADS. */
__device__ float block_sum(float v, float* s_buf) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(FULL_MASK, v, off);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) s_buf[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < NWARPS; w++) t += s_buf[w];
        s_buf[NWARPS] = t;
    }
    __syncthreads();
    float total = s_buf[NWARPS];
    __syncthreads(); /* s_buf may be reused by the caller right after */
    return total;
}

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + expf(-v)); }

/* ====================================================================
 * Kernels (all batched: blockIdx.y or blockIdx.x selects the stream)
 * ==================================================================== */

/* Embedding gather: x[b, :] = emb[tokens[b], :].  grid = (ceil(d/THREADS), n). */
__global__ void k_embed(float* __restrict__ x, const float* __restrict__ emb,
                        const uint32_t* __restrict__ tokens, int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (j < d) x[(size_t)b * d + j] = emb[(size_t)tokens[b] * d + j];
}

/* RMSNorm, out-of-place: out[b] = x[b] * rsqrt(mean(x[b]^2) + eps) * w.  grid = n, block = THREADS. */
__global__ void k_rms_norm(float* __restrict__ out, const float* __restrict__ x,
                           const float* __restrict__ w, int d, float eps) {
    __shared__ float s_buf[NWARPS + 1];
    const float* xb = x + (size_t)blockIdx.x * d;
    float* ob = out + (size_t)blockIdx.x * d;
    float sq = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) sq = fmaf(xb[j], xb[j], sq);
    float scale = rsqrtf(block_sum(sq, s_buf) / d + eps);
    for (int j = threadIdx.x; j < d; j += blockDim.x) ob[j] = xb[j] * scale * w[j];
}

/* LayerNorm, out-of-place.  grid = n, block = THREADS. */
__global__ void k_layer_norm(float* __restrict__ out, const float* __restrict__ x,
                             const float* __restrict__ w, const float* __restrict__ bias,
                             int d, float eps) {
    __shared__ float s_buf[NWARPS + 1];
    const float* xb = x + (size_t)blockIdx.x * d;
    float* ob = out + (size_t)blockIdx.x * d;
    float s = 0.0f, sq = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) { s += xb[j]; sq = fmaf(xb[j], xb[j], sq); }
    float mean = block_sum(s, s_buf) / d;
    float var = block_sum(sq, s_buf) / d - mean * mean;
    float inv_std = rsqrtf(var + eps);
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        ob[j] = (xb[j] - mean) * inv_std * w[j] + bias[j];
}

/* Batched GEMM "weights read once per step":  Y[b, r] = sum_k W[r, k] * X[b*ldx + k]
 *   grid = M (one block per weight row), block = THREADS, dynamic smem = K floats.
 * The row W[r, :] is loaded ONCE from global memory into dynamic shared memory ([PG-SMEM]: size
 * given as the 3rd launch parameter; only one dynamic array per kernel) and reused for every
 * stream, so DRAM weight traffic per step is independent of n_streams.
 * Reduction order for a given (b, r) is identical for every n_streams and every tile position:
 * thread t accumulates k = t, t+THREADS, ... with explicit fmaf (contraction pinned, [PG-FP] #fma),
 * then the fixed warp-shuffle tree and a sequential sum of the NWARPS warp partials.  Hence the
 * v1 (n=1) and v2 (any n) results are bit-identical.  No atomics (see block_sum). */
__global__ void k_gemm_rows(const float* __restrict__ W, const float* __restrict__ X,
                            float* __restrict__ Y, int M, int K, int ldx, int n) {
    extern __shared__ float s_w[];
    __shared__ float s_red[GEMM_TILE][NWARPS];
    int row = blockIdx.x;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const float* wr = W + (size_t)row * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) s_w[k] = wr[k];
    __syncthreads(); /* [PG-SYNC]: s_w fully written before any thread reads it */

    for (int s0 = 0; s0 < n; s0 += GEMM_TILE) {
        int ns = min(GEMM_TILE, n - s0);
        float acc[GEMM_TILE];
#pragma unroll
        for (int t = 0; t < GEMM_TILE; t++) acc[t] = 0.0f;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            float w = s_w[k];
#pragma unroll
            for (int t = 0; t < GEMM_TILE; t++)
                if (t < ns) acc[t] = fmaf(w, X[(size_t)(s0 + t) * ldx + k], acc[t]);
        }
        /* Every lane executes every shuffle (loop bounds are warp-uniform) [PG-SHFL]. */
#pragma unroll
        for (int t = 0; t < GEMM_TILE; t++) {
            for (int off = 16; off > 0; off >>= 1)
                acc[t] += __shfl_down_sync(FULL_MASK, acc[t], off);
        }
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < GEMM_TILE; t++) s_red[t][wid] = acc[t];
        }
        __syncthreads();
        if (threadIdx.x < ns) {
            float total = 0.0f;
            for (int w2 = 0; w2 < NWARPS; w2++) total += s_red[threadIdx.x][w2];
            Y[(size_t)(s0 + threadIdx.x) * M + row] = total;
        }
        __syncthreads(); /* s_red reused by the next tile */
    }
}

/* Causal depthwise conv1d (shift register) + bias + SiLU.  grid = (ceil(di/THREADS), n).
 * conv state of SLOT s: conv_base + s * di * (dc-1). */
__global__ void k_conv1d(float* __restrict__ conv_out, float* __restrict__ conv_base,
                         const float* __restrict__ xz, const float* __restrict__ w,
                         const float* __restrict__ bias, const uint32_t* __restrict__ slots,
                         int di, int dc) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (j >= di) return;
    int dc1 = dc - 1;
    float* st = conv_base + ((size_t)slots[b] * di + j) * dc1;
    float xin = xz[(size_t)b * 2 * di + j];
    const float* wj = w + (size_t)j * dc;
    float val = 0.0f;
    for (int k = 0; k < dc1; k++) val = fmaf(st[k], wj[k], val);
    val = fmaf(xin, wj[dc1], val) + bias[j];
    for (int k = 0; k < dc1 - 1; k++) st[k] = st[k + 1];
    st[dc1 - 1] = xin;
    conv_out[(size_t)b * di + j] = silu_f(val);
}

/* dt = softplus(dt + bias) (threshold 20 like torch.nn.functional.softplus).  grid = (.., n). */
__global__ void k_softplus_bias(float* __restrict__ dt, const float* __restrict__ bias, int di) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= di) return;
    float* p = dt + (size_t)blockIdx.y * di + j;
    float v = *p + bias[j];
    *p = (v > 20.0f) ? v : logf(1.0f + expf(v));
}

/* Selective-scan single step.  One thread per (stream, channel); the d_state loop is sequential
 * so the y reduction order is fixed.  B = zp[b, dr:dr+ds], C = zp[b, dr+ds:dr+2ds].
 * ssm state of SLOT s: ssm_base + s * di * ds.  grid = (ceil(di/THREADS), n). */
__global__ void k_ssm_step(float* __restrict__ y_out, float* __restrict__ ssm_base,
                           const float* __restrict__ dt, const float* __restrict__ xc,
                           const float* __restrict__ a_log, const float* __restrict__ zp,
                           const float* __restrict__ Dv, const uint32_t* __restrict__ slots,
                           int di, int ds, int dr) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (j >= di) return;
    int xr = dr + 2 * ds;
    const float* Bv = zp + (size_t)b * xr + dr;
    const float* Cv = Bv + ds;
    float* st = ssm_base + ((size_t)slots[b] * di + j) * ds;
    float dt_j = dt[(size_t)b * di + j];
    float x_j = xc[(size_t)b * di + j];
    float y_j = 0.0f;
    for (int k = 0; k < ds; k++) {
        float dA = expf(dt_j * -expf(a_log[(size_t)j * ds + k]));
        float s = fmaf(st[k], dA, x_j * (dt_j * Bv[k]));
        st[k] = s;
        y_j = fmaf(s, Cv[k], y_j);
    }
    y_out[(size_t)b * di + j] = fmaf(Dv[j], x_j, y_j);
}

/* gated[b, j] = y[b, j] * silu(z[b, j]) with z = xz[b, di:2di].  grid = (ceil(di/THREADS), n). */
__global__ void k_gated_silu(float* __restrict__ gated, const float* __restrict__ y,
                             const float* __restrict__ xz, int di) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (j >= di) return;
    float z = xz[(size_t)b * 2 * di + di + j];
    gated[(size_t)b * di + j] = y[(size_t)b * di + j] * silu_f(z);
}

/* x += r over n*d contiguous floats. */
__global__ void k_residual_add(float* __restrict__ x, const float* __restrict__ r, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += r[i];
}

/* Zero the state rows of the listed slots: base + slots[b] * stride.  grid = (ceil(stride/T), n). */
__global__ void k_zero_slots(float* __restrict__ base, const uint32_t* __restrict__ slots,
                             size_t stride) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride) base[(size_t)slots[blockIdx.y] * stride + i] = 0.0f;
}

/* ====================================================================
 * Host helpers
 * ==================================================================== */

static unsigned blocks_for(size_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

static int dev_alloc_zero(float** p, size_t count) {
    /* cudaMalloc does not clear memory ([RT-MEM]), so state buffers are explicitly zeroed. */
    if (cudaMalloc(p, count * sizeof(float)) != cudaSuccess) { *p = NULL; return -1; }
    return cudaMemset(*p, 0, count * sizeof(float)) == cudaSuccess ? 0 : -1;
}

static int read_f32_to_gpu(FILE* f, float** d_ptr, size_t count) {
    size_t bytes = count * sizeof(float);
    float* host = (float*)malloc(bytes);
    if (!host) return -1;
    if (fread(host, sizeof(float), count, f) != count) { free(host); return -1; }
    cudaError_t err = cudaMalloc(d_ptr, bytes);
    if (err != cudaSuccess) { *d_ptr = NULL; free(host); return -1; }
    err = cudaMemcpy(*d_ptr, host, bytes, cudaMemcpyHostToDevice);
    free(host);
    return (err == cudaSuccess) ? 0 : -1;
}

/* cudaFree(NULL) is a no-op per the Runtime API ([RT-MEM]); pointers are reset after freeing. */
static void pool_free(StatePool* p) {
    uint32_t nl = G.cfg.n_layer;
    for (uint32_t i = 0; p->conv && i < nl; i++) cudaFree(p->conv[i]);
    for (uint32_t i = 0; p->ssm && i < nl; i++) cudaFree(p->ssm[i]);
    free(p->conv);
    free(p->ssm);
    float* scratch[] = { p->x, p->normed, p->xz, p->conv_out, p->zp, p->dt, p->y, p->gated,
                         p->res, p->logits };
    for (size_t i = 0; i < sizeof(scratch) / sizeof(scratch[0]); i++) cudaFree(scratch[i]);
    cudaFree(p->d_slots);
    cudaFree(p->d_tokens);
    if (p->h_slots) cudaFreeHost(p->h_slots);
    if (p->h_tokens) cudaFreeHost(p->h_tokens);
    free(p->h_seen);
    memset(p, 0, sizeof(*p));
}

static int pool_alloc_state(StatePool* p, size_t cap) {
    const RbinHeader* c = &G.cfg;
    size_t di = c->d_inner;
    p->conv = (float**)calloc(c->n_layer, sizeof(float*));
    p->ssm = (float**)calloc(c->n_layer, sizeof(float*));
    if (!p->conv || !p->ssm) return -1;
    for (uint32_t i = 0; i < c->n_layer; i++) {
        if (dev_alloc_zero(&p->conv[i], cap * di * (c->d_conv - 1))) return -1;
        if (dev_alloc_zero(&p->ssm[i], cap * di * c->d_state)) return -1;
    }
    return 0;
}

static int pool_alloc(StatePool* p, uint32_t cap) {
    const RbinHeader* c = &G.cfg;
    size_t n = cap, d = c->d_model, di = c->d_inner, xr = c->dt_rank + 2 * c->d_state;
    memset(p, 0, sizeof(*p));
    p->cap = cap;
    if (pool_alloc_state(p, n)) return -1;
    int bad = 0;
    bad |= dev_alloc_zero(&p->x, n * d);
    bad |= dev_alloc_zero(&p->normed, n * d);
    bad |= dev_alloc_zero(&p->xz, n * 2 * di);
    bad |= dev_alloc_zero(&p->conv_out, n * di);
    bad |= dev_alloc_zero(&p->zp, n * xr);
    bad |= dev_alloc_zero(&p->dt, n * di);
    bad |= dev_alloc_zero(&p->y, n * di);
    bad |= dev_alloc_zero(&p->gated, n * di);
    bad |= dev_alloc_zero(&p->res, n * d);
    bad |= dev_alloc_zero(&p->logits, n * c->vocab_size);
    if (bad) return -1;
    if (cudaMalloc(&p->d_slots, n * sizeof(uint32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&p->d_tokens, n * sizeof(uint32_t)) != cudaSuccess) return -1;
    if (cudaHostAlloc(&p->h_slots, n * sizeof(uint32_t), cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaHostAlloc(&p->h_tokens, n * sizeof(uint32_t), cudaHostAllocDefault) != cudaSuccess) return -1;
    p->h_seen = (uint8_t*)calloc(n, 1);
    return p->h_seen ? 0 : -1;
}

/* ====================================================================
 * Weight loading
 * ==================================================================== */

static int validate_header(const RbinHeader* h) {
    if (h->magic != RBIN_MAGIC) return -1;
    if (h->version != RBIN_VERSION_V1 && h->version != RBIN_VERSION_V2) return -1;
    if (!h->vocab_size || !h->d_model || !h->d_inner || !h->d_state || !h->dt_rank || !h->n_layer)
        return -1;
    if (h->d_conv < 2) return -1;
    if (h->version == RBIN_VERSION_V2) {
        if (h->tokenizer > RBIN_TOK_BPE) return -1;
        if (h->tokenizer == RBIN_TOK_BYTE && h->vocab_size != 256) return -1;
    }
    /* Largest GEMM reduction dim is K = max(d_model, d_inner, dt_rank); its row must fit in the
     * default 48 KB dynamic shared memory budget, otherwise an explicit opt-in via
     * cudaFuncSetAttribute would be required ([PG-SMEM], [PG-CC]). */
    uint32_t k = h->d_model > h->d_inner ? h->d_model : h->d_inner;
    size_t smem = (size_t)k * sizeof(float) + sizeof(float) * GEMM_TILE * NWARPS;
    return smem <= SMEM_DEFAULT_LIMIT ? 0 : -1;
}

static int load_layer(FILE* f, LayerWeightsGpu* lw, int has_nb) {
    const RbinHeader* c = &G.cfg;
    size_t d = c->d_model, di = c->d_inner, ds = c->d_state, dc = c->d_conv, dr = c->dt_rank;
    if (read_f32_to_gpu(f, &lw->norm_w, d)) return -1;
    if (has_nb && read_f32_to_gpu(f, &lw->norm_b, d)) return -1;
    if (read_f32_to_gpu(f, &lw->in_proj_w, 2 * di * d)) return -1;
    if (read_f32_to_gpu(f, &lw->conv1d_w, di * dc)) return -1;
    if (read_f32_to_gpu(f, &lw->conv1d_b, di)) return -1;
    if (read_f32_to_gpu(f, &lw->x_proj_w, (dr + 2 * ds) * di)) return -1;
    if (read_f32_to_gpu(f, &lw->dt_proj_w, di * dr)) return -1;
    if (read_f32_to_gpu(f, &lw->dt_proj_b, di)) return -1;
    if (read_f32_to_gpu(f, &lw->a_log, di * ds)) return -1;
    if (read_f32_to_gpu(f, &lw->d_vec, di)) return -1;
    return read_f32_to_gpu(f, &lw->out_proj_w, d * di);
}

static int load_body(FILE* f) {
    const RbinHeader* c = &G.cfg;
    int has_nb = (c->flags & FLAG_NORM_BIAS) != 0;
    int has_uh = (c->flags & FLAG_UNTIED_HEAD) != 0;
    size_t vs = c->vocab_size, d = c->d_model;
    if (read_f32_to_gpu(f, &G.d_emb, vs * d)) return -1;
    G.layers = (LayerWeightsGpu*)calloc(c->n_layer, sizeof(LayerWeightsGpu));
    if (!G.layers) return -1;
    for (uint32_t i = 0; i < c->n_layer; i++) {
        if (load_layer(f, &G.layers[i], has_nb)) return -1;
        if ((i + 1) % 8 == 0 || i + 1 == c->n_layer)
            fprintf(stderr, "[redukt-gpu-cuda] loaded layer %u/%u\n", i + 1, c->n_layer);
    }
    if (read_f32_to_gpu(f, &G.d_norm_f_w, d)) return -1;
    if (has_nb && read_f32_to_gpu(f, &G.d_norm_f_b, d)) return -1;
    G.head_is_tied = !has_uh;
    if (!has_uh) { G.d_lm_head = G.d_emb; return 0; }
    return read_f32_to_gpu(f, &G.d_lm_head, vs * d);
}

static int load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[redukt-gpu-cuda] cannot open: %s\n", path);
        return -1;
    }
    RbinHeader hdr;
    int rc = -1;
    if (fread(&hdr, sizeof(RbinHeader), 1, f) == 1 && validate_header(&hdr) == 0) {
        G.cfg = hdr;
        if (hdr.version == RBIN_VERSION_V1) G.cfg.tokenizer = hdr.vocab_size == 256 ? RBIN_TOK_BYTE : RBIN_TOK_BPE;
        fprintf(stderr, "[redukt-gpu-cuda] rbin v%u: d=%u di=%u ds=%u dc=%u dr=%u nl=%u vs=%u flags=0x%02x "
                "tok=%u\n", hdr.version, hdr.d_model, hdr.d_inner, hdr.d_state, hdr.d_conv, hdr.dt_rank,
                hdr.n_layer, hdr.vocab_size, hdr.flags, G.cfg.tokenizer);
        rc = load_body(f);
    } else {
        fprintf(stderr, "[redukt-gpu-cuda] bad/unsupported .rbin header in %s\n", path);
    }
    fclose(f);
    return rc;
}

static void free_model(void) {
    pool_free(&G.v1);
    pool_free(&G.batch);
    for (uint32_t i = 0; G.layers && i < G.cfg.n_layer; i++) {
        LayerWeightsGpu* lw = &G.layers[i];
        float* ws[] = { lw->norm_w, lw->norm_b, lw->in_proj_w, lw->conv1d_w, lw->conv1d_b,
                        lw->x_proj_w, lw->dt_proj_w, lw->dt_proj_b, lw->a_log, lw->d_vec,
                        lw->out_proj_w };
        for (size_t k = 0; k < sizeof(ws) / sizeof(ws[0]); k++) cudaFree(ws[k]);
    }
    free(G.layers);
    if (!G.head_is_tied) cudaFree(G.d_lm_head);
    cudaFree(G.d_emb);
    cudaFree(G.d_norm_f_w);
    cudaFree(G.d_norm_f_b);
    if (G.stream) cudaStreamDestroy(G.stream);
    memset(&G, 0, sizeof(G));
}

/* ====================================================================
 * Batched forward (shared by v1 and v2)
 * ==================================================================== */

static void launch_gemm(const float* W, const float* X, float* Y, int M, int K, int ldx, int n) {
    size_t smem = (size_t)K * sizeof(float); /* 3rd launch parameter = dynamic smem bytes [PG-SMEM] */
    k_gemm_rows<<<M, THREADS, smem, G.stream>>>(W, X, Y, M, K, ldx, n);
}

static void launch_norm(float* out, const float* x, const float* w, const float* b, int n) {
    int d = (int)G.cfg.d_model;
    if (b) k_layer_norm<<<n, THREADS, 0, G.stream>>>(out, x, w, b, d, 1e-5f);
    else   k_rms_norm<<<n, THREADS, 0, G.stream>>>(out, x, w, d, 1e-5f);
}

static void launch_layer(StatePool* p, int li, int n) {
    const RbinHeader* c = &G.cfg;
    const LayerWeightsGpu* lw = &G.layers[li];
    int d = c->d_model, di = c->d_inner, ds = c->d_state, dr = c->dt_rank, xr = dr + 2 * ds;
    dim3 g_di(blocks_for(di), n);
    int has_nb = (c->flags & FLAG_NORM_BIAS) != 0;
    launch_norm(p->normed, p->x, lw->norm_w, has_nb ? lw->norm_b : NULL, n);
    launch_gemm(lw->in_proj_w, p->normed, p->xz, 2 * di, d, d, n);
    k_conv1d<<<g_di, THREADS, 0, G.stream>>>(p->conv_out, p->conv[li], p->xz, lw->conv1d_w,
                                             lw->conv1d_b, p->d_slots, di, c->d_conv);
    launch_gemm(lw->x_proj_w, p->conv_out, p->zp, xr, di, di, n);
    launch_gemm(lw->dt_proj_w, p->zp, p->dt, di, dr, xr, n);   /* reads zp[b, 0:dr] (ldx = xr) */
    k_softplus_bias<<<g_di, THREADS, 0, G.stream>>>(p->dt, lw->dt_proj_b, di);
    k_ssm_step<<<g_di, THREADS, 0, G.stream>>>(p->y, p->ssm[li], p->dt, p->conv_out, lw->a_log,
                                               p->zp, lw->d_vec, p->d_slots, di, ds, dr);
    k_gated_silu<<<g_di, THREADS, 0, G.stream>>>(p->gated, p->y, p->xz, di);
    launch_gemm(lw->out_proj_w, p->gated, p->res, d, di, di, n);
    size_t tot = (size_t)n * d;
    k_residual_add<<<blocks_for(tot), THREADS, 0, G.stream>>>(p->x, p->res, tot);
}

/* Validates arguments, uploads slots/tokens, runs all layers + head, copies logits to `out`.
 * All kernels are enqueued on G.stream, so they execute in enqueue order ([PG-STREAM]
 * #cuda-stream-ordering); one cudaStreamSynchronize at the end makes `out` valid on return. */
static int pool_forward(StatePool* p, uint32_t n, const uint32_t* slots, const uint32_t* tokens,
                        float* out) {
    const RbinHeader* c = &G.cfg;
    memset(p->h_seen, 0, p->cap);
    for (uint32_t b = 0; b < n; b++) {
        if (slots[b] >= p->cap || p->h_seen[slots[b]] || tokens[b] >= c->vocab_size)
            return BACKEND_INVALID_ARGUMENT;
        p->h_seen[slots[b]] = 1;
        p->h_slots[b] = slots[b];
        p->h_tokens[b] = tokens[b];
    }
    size_t ib = (size_t)n * sizeof(uint32_t);
    /* Pinned staging buffers -> truly asynchronous H2D copies in stream order ([PG-STREAM]). */
    CUDA_TRY(cudaMemcpyAsync(p->d_slots, p->h_slots, ib, cudaMemcpyHostToDevice, G.stream),
             BACKEND_FORWARD_FAILED);
    CUDA_TRY(cudaMemcpyAsync(p->d_tokens, p->h_tokens, ib, cudaMemcpyHostToDevice, G.stream),
             BACKEND_FORWARD_FAILED);
    int d = c->d_model, vs = c->vocab_size;
    k_embed<<<dim3(blocks_for(d), n), THREADS, 0, G.stream>>>(p->x, G.d_emb, p->d_tokens, d);
    for (uint32_t li = 0; li < c->n_layer; li++) launch_layer(p, (int)li, (int)n);
    launch_norm(p->normed, p->x, G.d_norm_f_w, G.d_norm_f_b, (int)n);
    launch_gemm(G.d_lm_head, p->normed, p->logits, vs, d, d, (int)n);
    /* Kernel launches report configuration errors through cudaGetLastError ([RT-ERR]). */
    CUDA_TRY(cudaGetLastError(), BACKEND_FORWARD_FAILED);
    /* `out` is caller-owned pageable memory: cudaMemcpyAsync then reverts to synchronous
     * behaviour ([PG-STREAM] #launching-memory-transfers-in-cuda-streams), which is fine here;
     * the explicit stream synchronize guarantees completion before returning. */
    CUDA_TRY(cudaMemcpyAsync(out, p->logits, (size_t)n * vs * sizeof(float), cudaMemcpyDeviceToHost,
                             G.stream), BACKEND_FORWARD_FAILED);
    CUDA_TRY(cudaStreamSynchronize(G.stream), BACKEND_FORWARD_FAILED);
    return BACKEND_OK;
}

static int pool_reset(StatePool* p, uint32_t n, const uint32_t* slots) {
    const RbinHeader* c = &G.cfg;
    for (uint32_t b = 0; b < n; b++) {
        uint32_t s = slots ? slots[b] : b;
        if (s >= p->cap) return BACKEND_INVALID_ARGUMENT;
        p->h_slots[b] = s;
    }
    if (n == 0) return BACKEND_OK;
    CUDA_TRY(cudaMemcpyAsync(p->d_slots, p->h_slots, (size_t)n * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             G.stream), BACKEND_FORWARD_FAILED);
    size_t conv_stride = (size_t)c->d_inner * (c->d_conv - 1);
    size_t ssm_stride = (size_t)c->d_inner * c->d_state;
    for (uint32_t li = 0; li < c->n_layer; li++) {
        k_zero_slots<<<dim3(blocks_for(conv_stride), n), THREADS, 0, G.stream>>>(p->conv[li], p->d_slots,
                                                                                 conv_stride);
        k_zero_slots<<<dim3(blocks_for(ssm_stride), n), THREADS, 0, G.stream>>>(p->ssm[li], p->d_slots,
                                                                                ssm_stride);
    }
    CUDA_TRY(cudaGetLastError(), BACKEND_FORWARD_FAILED);
    CUDA_TRY(cudaStreamSynchronize(G.stream), BACKEND_FORWARD_FAILED);
    return BACKEND_OK;
}

/* ====================================================================
 * FFI exports — v1 (unchanged signatures) and v2 (additive)
 * ==================================================================== */

static char* build_rbin_path(const uint8_t* weights_path) {
    size_t len = strlen((const char*)weights_path);
    char* path = (char*)malloc(len + 16);
    if (!path) return NULL;
    memcpy(path, weights_path, len + 1);
    if (len < 5 || strcmp(path + len - 5, ".rbin") != 0) strcat(path, "/model.rbin");
    return path;
}

extern "C" {

BackendResult redukt_backend_init(const uint8_t* weights_path, int32_t device_id, BackendInfo* info_out) {
    if (!weights_path || !info_out) return BACKEND_INIT_FAILED;
    if (G.ready) free_model();
    char* path = build_rbin_path(weights_path);
    if (!path) return BACKEND_INIT_FAILED;
    fprintf(stderr, "[redukt-gpu-cuda] init: weights=%s device=%d\n", path, device_id);

    if (cudaSetDevice(device_id >= 0 ? device_id : 0) != cudaSuccess
        || cudaStreamCreateWithFlags(&G.stream, cudaStreamNonBlocking) != cudaSuccess) {
        /* Non-blocking stream: no implicit sync with the legacy default stream ([PG-STREAM]). */
        free(path);
        free_model();
        return BACKEND_INIT_FAILED;
    }
    int lw = load_weights(path);
    free(path);
    if (lw != 0) { free_model(); return BACKEND_WEIGHTS_NOT_FOUND; }
    if (pool_alloc(&G.v1, 1) != 0) { free_model(); return BACKEND_INIT_FAILED; }
    G.ready = 1;

    memset(info_out, 0, sizeof(*info_out));
    info_out->ffi_version = BACKEND_FFI_VERSION;
    info_out->vocab_size = G.cfg.vocab_size;
    info_out->hidden_dim = G.cfg.d_model;
    const char* name = "cuda-mamba";
    memcpy(info_out->name, name, strlen(name));

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "[redukt-gpu-cuda] ready: vocab=%u d_model=%u n_layer=%u abi=%d | GPU %.0f/%.0f MB used\n",
            G.cfg.vocab_size, G.cfg.d_model, G.cfg.n_layer, BACKEND_ABI_VERSION,
            (total_mem - free_mem) / 1048576.0, total_mem / 1048576.0);
    return BACKEND_OK;
}

BackendResult redukt_backend_forward(uint32_t token, float* out_logits, uint32_t vocab_size) {
    if (!G.ready || !out_logits || vocab_size != G.cfg.vocab_size) return BACKEND_FORWARD_FAILED;
    uint32_t slot = 0;
    int rc = pool_forward(&G.v1, 1, &slot, &token, out_logits);
    return rc == BACKEND_OK ? BACKEND_OK : BACKEND_FORWARD_FAILED;
}

void redukt_backend_reset_state(void) {
    if (!G.ready) return;
    uint32_t slot = 0;
    if (pool_reset(&G.v1, 1, &slot) != BACKEND_OK)
        fprintf(stderr, "[redukt-gpu-cuda] reset_state failed\n");
}

void redukt_backend_free(void) {
    free_model();
    fprintf(stderr, "[redukt-gpu-cuda] freed\n");
}

/* ---- v2 ---- */

uint32_t redukt_backend_abi_version(void) { return BACKEND_ABI_VERSION; }

int32_t redukt_backend_batch_init(uint32_t max_streams) {
    if (!G.ready) return BACKEND_INIT_FAILED;
    if (max_streams > MAX_STREAMS) return BACKEND_INVALID_ARGUMENT;
    CUDA_TRY(cudaStreamSynchronize(G.stream), BACKEND_INIT_FAILED);
    pool_free(&G.batch);
    if (max_streams == 0) return BACKEND_OK;
    if (pool_alloc(&G.batch, max_streams) != 0) {
        pool_free(&G.batch);
        return BACKEND_INIT_FAILED;
    }
    fprintf(stderr, "[redukt-gpu-cuda] batch pool: %u streams\n", max_streams);
    return BACKEND_OK;
}

int32_t redukt_backend_batch_reset(const uint32_t* slots, uint32_t n_slots) {
    if (!G.ready || G.batch.cap == 0) return BACKEND_INIT_FAILED;
    if (!slots) return pool_reset(&G.batch, G.batch.cap, NULL);
    if (n_slots > G.batch.cap) return BACKEND_INVALID_ARGUMENT;
    return pool_reset(&G.batch, n_slots, slots);
}

int32_t redukt_backend_forward_batch(uint32_t n_streams, const uint32_t* slots, const uint32_t* tokens,
                                     float* out_logits, uint32_t vocab_size) {
    if (!G.ready || G.batch.cap == 0) return BACKEND_INIT_FAILED;
    if (vocab_size != G.cfg.vocab_size || n_streams > G.batch.cap) return BACKEND_INVALID_ARGUMENT;
    if (n_streams == 0) return BACKEND_OK;
    if (!slots || !tokens || !out_logits) return BACKEND_INVALID_ARGUMENT;
    return pool_forward(&G.batch, n_streams, slots, tokens, out_logits);
}

} /* extern "C" */
