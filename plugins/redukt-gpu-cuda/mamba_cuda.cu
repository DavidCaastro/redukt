/* =====================================================================
 * Redukt GPU Backend Plugin — CUDA Implementation (R-090)
 *
 * Mamba SSM inference on NVIDIA GPUs via BackendVTable FFI.
 * Compiles to libredukt_gpu.so on Linux (Kaggle T4).
 *
 * Build:  nvcc -shared -O3 -arch=sm_75 -o libredukt_gpu.so mamba_cuda.cu
 * Test:   python redukt-gpu-cuda-test.py
 *
 * Copyright 2026 Redukt Contributors. Apache-2.0 License.
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* ---- Macros ---- */

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "[redukt-gpu-cuda] CUDA error at %s:%d: "   \
                "%s\n", __FILE__, __LINE__,                         \
                cudaGetErrorString(err));                            \
        return BACKEND_INIT_FAILED;                                 \
    }                                                               \
} while(0)

#define CUDA_CHECK_FWD(call) do {                                   \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "[redukt-gpu-cuda] CUDA error at %s:%d: "   \
                "%s\n", __FILE__, __LINE__,                         \
                cudaGetErrorString(err));                            \
        return BACKEND_FORWARD_FAILED;                              \
    }                                                               \
} while(0)

#define THREADS 256

/* ---- FFI types (must match src/domain/backend_ffi.rs) ---- */

extern "C" {

typedef enum {
    BACKEND_OK              = 0,
    BACKEND_INIT_FAILED     = 1,
    BACKEND_FORWARD_FAILED  = 2,
    BACKEND_VERSION_MISMATCH= 3,
    BACKEND_WEIGHTS_NOT_FOUND=4,
} BackendResult;

#define BACKEND_FFI_VERSION 1
#define BACKEND_NAME_MAX 64

typedef struct {
    uint32_t ffi_version;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint8_t  name[BACKEND_NAME_MAX];
} BackendInfo;

} /* extern "C" */

/* ---- Binary weight format (.rbin) ---- */

#define RBIN_MAGIC   0x55504752u  /* "RGPU" LE */
#define RBIN_VERSION 1u
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
    uint32_t reserved;
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

/* ---- GPU model state ---- */

typedef struct {
    RbinHeader cfg;

    /* Global weights */
    float* d_emb;         /* vocab_size * d_model */
    float* d_lm_head;     /* vocab_size * d_model (or alias to d_emb) */
    float* d_norm_f_w;    /* d_model */
    float* d_norm_f_b;    /* d_model or NULL */
    int    head_is_tied;  /* 1 if lm_head == emb */

    /* Per-layer weights */
    LayerWeightsGpu* layers;  /* [n_layer] */

    /* Recurrent state */
    float** d_conv_states; /* [n_layer] d_inner*(d_conv-1) each */
    float** d_ssm_states;  /* [n_layer] d_inner*d_state each */

    /* Scratch (GPU) */
    float* d_x;           /* d_model */
    float* d_normed;      /* d_model */
    float* d_xz;          /* 2*d_inner */
    float* d_conv_out;    /* d_inner */
    float* d_z_proj;      /* dt_rank + 2*d_state */
    float* d_dt;          /* d_inner */
    float* d_dt_raw;      /* dt_rank */
    float* d_y;           /* d_inner */
    float* d_gated;       /* d_inner */
    float* d_result;      /* d_model */
    float* d_logits;      /* vocab_size */

    /* Host readback */
    float* h_logits;      /* vocab_size */

    int ready;
} GpuModel;

static GpuModel G = {0};

/* ====================================================================
 * CUDA Kernels
 * ==================================================================== */

/* ---- GEMV: y[row] = dot(A[row,:], x)  for row in [0, M) ---- */
/* One block per row, THREADS threads cooperate on the dot product. */
/* Uses float4 vectorized loads + warp shuffle reduction.           */

__global__ void k_gemv(const float* __restrict__ A,
                       const float* __restrict__ x,
                       float* __restrict__ y,
                       int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* rp = A + (size_t)row * N;
    float sum = 0.0f;

    /* Vectorized path (float4 = 16 bytes) */
    int n4 = N >> 2;
    const float4* rp4 = (const float4*)rp;
    const float4* x4  = (const float4*)x;
    for (int j = threadIdx.x; j < n4; j += blockDim.x) {
        float4 a = rp4[j];
        float4 b = x4[j];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    /* Remainder */
    for (int j = (n4 << 2) + threadIdx.x; j < N; j += blockDim.x) {
        sum += rp[j] * x[j];
    }

    /* Warp reduction */
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);

    /* Inter-warp reduction via shared memory */
    __shared__ float ws[8]; /* max 256/32 = 8 warps */
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float t = 0.0f;
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; w++) t += ws[w];
        y[row] = t;
    }
}

/* ---- GEMV + bias: y[row] = dot(A[row,:], x) + b[row] ---- */

__global__ void k_gemv_bias(const float* __restrict__ A,
                            const float* __restrict__ x,
                            const float* __restrict__ b,
                            float* __restrict__ y,
                            int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* rp = A + (size_t)row * N;
    float sum = 0.0f;
    int n4 = N >> 2;
    const float4* rp4 = (const float4*)rp;
    const float4* x4  = (const float4*)x;
    for (int j = threadIdx.x; j < n4; j += blockDim.x) {
        float4 a = rp4[j]; float4 bv = x4[j];
        sum += a.x*bv.x + a.y*bv.y + a.z*bv.z + a.w*bv.w;
    }
    for (int j = (n4 << 2) + threadIdx.x; j < N; j += blockDim.x)
        sum += rp[j] * x[j];

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);

    __shared__ float ws[8];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; w++) t += ws[w];
        y[row] = t + b[row];
    }
}

/* ---- RMSNorm (in-place): x = x / rms(x) * w ---- */

__global__ void k_rms_norm(float* __restrict__ x,
                           const float* __restrict__ w,
                           int d, float eps) {
    __shared__ float s_sum[8];
    float local_sq = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = x[j]; local_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sq += __shfl_down_sync(0xffffffff, local_sq, off);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) s_sum[wid] = local_sq;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) >> 5;
        for (int w2 = 0; w2 < nw; w2++) total += s_sum[w2];
        s_sum[0] = rsqrtf(total / d + eps);
    }
    __syncthreads();
    float scale = s_sum[0];
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        x[j] = x[j] * scale * w[j];
}

/* ---- LayerNorm: out = (x - mean) / sqrt(var + eps) * w + b ---- */

__global__ void k_layer_norm(float* __restrict__ out,
                             const float* __restrict__ x,
                             const float* __restrict__ w,
                             const float* __restrict__ b,
                             int d, float eps) {
    __shared__ float s_buf[16]; /* [0..7]=sum, [8..15]=sq_sum */
    float local_sum = 0.0f, local_sq = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = x[j]; local_sum += v; local_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
        local_sq  += __shfl_down_sync(0xffffffff, local_sq, off);
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { s_buf[wid] = local_sum; s_buf[wid+8] = local_sq; }
    __syncthreads();
    float mean = 0.0f, var = 0.0f;
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) >> 5;
        float ts = 0.0f, tsq = 0.0f;
        for (int w2 = 0; w2 < nw; w2++) { ts += s_buf[w2]; tsq += s_buf[w2+8]; }
        mean = ts / d;
        var  = tsq / d - mean * mean;
        s_buf[0] = mean;
        s_buf[1] = rsqrtf(var + eps);
    }
    __syncthreads();
    mean = s_buf[0]; float inv_std = s_buf[1];
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        out[j] = (x[j] - mean) * inv_std * w[j] + b[j];
}

/* ---- SiLU in-place: x[i] = x[i] * sigmoid(x[i]) ---- */

__global__ void k_silu(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ---- Conv1d shift register + bias + activation ---- */
/* Each thread handles one channel j in [0, d_inner).  */
/* conv_state: d_inner * (d_conv-1) flat               */

__global__ void k_conv1d_shift(float* __restrict__ conv_out,
                               float* __restrict__ conv_state,
                               const float* __restrict__ x_branch,
                               const float* __restrict__ w,
                               const float* __restrict__ bias,
                               int d_inner, int d_conv) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    int dc1 = d_conv - 1;
    float val = 0.0f;

    /* Convolve: state columns 0..dc-2, plus new input */
    for (int k = 0; k < dc1; k++)
        val += conv_state[j * dc1 + k] * w[j * d_conv + k];
    val += x_branch[j] * w[j * d_conv + dc1];
    val += bias[j];

    /* Shift state left, insert new input */
    for (int k = 0; k < dc1 - 1; k++)
        conv_state[j * dc1 + k] = conv_state[j * dc1 + k + 1];
    conv_state[j * dc1 + dc1 - 1] = x_branch[j];

    /* SiLU activation (fused) */
    conv_out[j] = val / (1.0f + expf(-val));
}

/* ---- Softplus: dt[i] = log(1 + exp(dt_proj[i] + bias[i])) ---- */

__global__ void k_softplus_bias(float* __restrict__ dt,
                                const float* __restrict__ bias,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = dt[i] + bias[i];
        dt[i] = (v > 20.0f) ? v : logf(1.0f + expf(v));
    }
}

/* ---- SSM recurrence (d_state=16 specialized) ---- */
/* Each thread handles one d_inner channel.           */

__global__ void k_ssm_recurrence(float* __restrict__ y_out,
                                 float* __restrict__ ssm_state,
                                 const float* __restrict__ dt,
                                 const float* __restrict__ conv_out,
                                 const float* __restrict__ a_log,
                                 const float* __restrict__ B,
                                 const float* __restrict__ C,
                                 const float* __restrict__ D,
                                 int d_inner, int d_state) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_inner) return;

    float dt_j = dt[j];
    float x_j  = conv_out[j];
    float y_j  = 0.0f;

    for (int k = 0; k < d_state; k++) {
        float a_val = -expf(a_log[j * d_state + k]);
        float dA = expf(dt_j * a_val);
        float dB = dt_j * B[k];
        int idx = j * d_state + k;
        float s = ssm_state[idx] * dA + x_j * dB;
        ssm_state[idx] = s;
        y_j += s * C[k];
    }

    y_out[j] = y_j + D[j] * x_j;
}

/* ---- Gated SiLU multiply: out[i] = y[i] * silu(z[i]) ---- */

__global__ void k_gated_silu(float* __restrict__ out,
                             const float* __restrict__ y,
                             const float* __restrict__ z,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float zv = z[i];
        out[i] = y[i] * zv / (1.0f + expf(-zv));
    }
}

/* ---- Embedding lookup: copy one row ---- */

__global__ void k_emb_lookup(float* __restrict__ out,
                             const float* __restrict__ emb,
                             int token, int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) out[j] = emb[(size_t)token * d + j];
}

/* ---- Residual add: x[i] += r[i] ---- */

__global__ void k_residual_add(float* __restrict__ x,
                               const float* __restrict__ r, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) x[i] += r[i];
}

/* ====================================================================
 * Weight Loading
 * ==================================================================== */

static int read_f32_to_gpu(FILE* f, float** d_ptr, size_t count) {
    size_t bytes = count * sizeof(float);
    float* host = (float*)malloc(bytes);
    if (!host) return -1;
    if (fread(host, sizeof(float), count, f) != count) {
        free(host); return -1;
    }
    cudaError_t err = cudaMalloc(d_ptr, bytes);
    if (err != cudaSuccess) { free(host); return -1; }
    err = cudaMemcpy(*d_ptr, host, bytes, cudaMemcpyHostToDevice);
    free(host);
    return (err == cudaSuccess) ? 0 : -1;
}

static int load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[redukt-gpu-cuda] cannot open: %s\n", path);
        return -1;
    }

    RbinHeader hdr;
    int has_nb, has_uh;
    uint32_t d, di, ds, dc, dr, nl, vs;

    if (fread(&hdr, sizeof(RbinHeader), 1, f) != 1) goto fail;
    if (hdr.magic != RBIN_MAGIC || hdr.version != RBIN_VERSION) {
        fprintf(stderr, "[redukt-gpu-cuda] bad magic/version in %s\n", path);
        goto fail;
    }

    G.cfg = hdr;
    has_nb = (hdr.flags & FLAG_NORM_BIAS) != 0;
    has_uh = (hdr.flags & FLAG_UNTIED_HEAD) != 0;
    d  = hdr.d_model;
    di = hdr.d_inner;
    ds = hdr.d_state;
    dc = hdr.d_conv;
    dr = hdr.dt_rank;
    nl = hdr.n_layer;
    vs = hdr.vocab_size;

    fprintf(stderr, "[redukt-gpu-cuda] loading: d=%u di=%u ds=%u dc=%u "
            "dr=%u nl=%u vs=%u flags=0x%02x\n",
            d, di, ds, dc, dr, nl, vs, hdr.flags);

    /* Embeddings */
    if (read_f32_to_gpu(f, &G.d_emb, (size_t)vs * d)) goto fail;

    /* Per-layer weights */
    G.layers = (LayerWeightsGpu*)calloc(nl, sizeof(LayerWeightsGpu));
    G.d_conv_states = (float**)calloc(nl, sizeof(float*));
    G.d_ssm_states  = (float**)calloc(nl, sizeof(float*));
    if (!G.layers || !G.d_conv_states || !G.d_ssm_states) goto fail;

    for (uint32_t i = 0; i < nl; i++) {
        LayerWeightsGpu* lw = &G.layers[i];
        if (read_f32_to_gpu(f, &lw->norm_w, d)) goto fail;
        if (has_nb) {
            if (read_f32_to_gpu(f, &lw->norm_b, d)) goto fail;
        }
        if (read_f32_to_gpu(f, &lw->in_proj_w, (size_t)2*di*d)) goto fail;
        if (read_f32_to_gpu(f, &lw->conv1d_w, (size_t)di*dc)) goto fail;
        if (read_f32_to_gpu(f, &lw->conv1d_b, di)) goto fail;
        if (read_f32_to_gpu(f, &lw->x_proj_w, (size_t)(dr+2*ds)*di)) goto fail;
        if (read_f32_to_gpu(f, &lw->dt_proj_w, (size_t)di*dr)) goto fail;
        if (read_f32_to_gpu(f, &lw->dt_proj_b, di)) goto fail;
        if (read_f32_to_gpu(f, &lw->a_log, (size_t)di*ds)) goto fail;
        if (read_f32_to_gpu(f, &lw->d_vec, di)) goto fail;
        if (read_f32_to_gpu(f, &lw->out_proj_w, (size_t)d*di)) goto fail;

        /* Allocate state buffers (zeros) */
        size_t conv_bytes = (size_t)di * (dc - 1) * sizeof(float);
        size_t ssm_bytes  = (size_t)di * ds * sizeof(float);
        cudaMalloc(&G.d_conv_states[i], conv_bytes);
        cudaMemset(G.d_conv_states[i], 0, conv_bytes);
        cudaMalloc(&G.d_ssm_states[i], ssm_bytes);
        cudaMemset(G.d_ssm_states[i], 0, ssm_bytes);

        if ((i + 1) % 8 == 0 || i == nl - 1)
            fprintf(stderr, "[redukt-gpu-cuda] loaded layer %u/%u\n", i+1, nl);
    }

    /* Final norm */
    if (read_f32_to_gpu(f, &G.d_norm_f_w, d)) goto fail;
    if (has_nb) {
        if (read_f32_to_gpu(f, &G.d_norm_f_b, d)) goto fail;
    }

    /* LM head */
    if (has_uh) {
        if (read_f32_to_gpu(f, &G.d_lm_head, (size_t)vs * d)) goto fail;
        G.head_is_tied = 0;
    } else {
        G.d_lm_head = G.d_emb; /* tied */
        G.head_is_tied = 1;
    }

    /* Scratch buffers */
    cudaMalloc(&G.d_x,       d  * sizeof(float));
    cudaMalloc(&G.d_normed,  d  * sizeof(float));
    cudaMalloc(&G.d_xz,      2*di * sizeof(float));
    cudaMalloc(&G.d_conv_out,di * sizeof(float));
    cudaMalloc(&G.d_z_proj,  (dr + 2*ds) * sizeof(float));
    cudaMalloc(&G.d_dt,      di * sizeof(float));
    cudaMalloc(&G.d_dt_raw,  dr * sizeof(float));
    cudaMalloc(&G.d_y,       di * sizeof(float));
    cudaMalloc(&G.d_gated,   di * sizeof(float));
    cudaMalloc(&G.d_result,  d  * sizeof(float));
    cudaMalloc(&G.d_logits,  vs * sizeof(float));

    G.h_logits = (float*)malloc(vs * sizeof(float));
    if (!G.h_logits) goto fail;

    fclose(f);

    /* Report GPU memory usage */
    {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "[redukt-gpu-cuda] GPU memory: %.0f MB used, "
            "%.0f MB free / %.0f MB total\n",
            (total_mem - free_mem) / 1048576.0,
            free_mem / 1048576.0, total_mem / 1048576.0);
    }

    return 0;

fail:
    fclose(f);
    return -1;
}

/* ====================================================================
 * Forward Pass Orchestration
 * ==================================================================== */

static BackendResult mamba_forward(uint32_t token) {
    RbinHeader* c = &G.cfg;
    int d  = c->d_model;
    int di = c->d_inner;
    int ds = c->d_state;
    int dc = c->d_conv;
    int dr = c->dt_rank;
    int nl = c->n_layer;
    int vs = c->vocab_size;
    int has_nb = (c->flags & FLAG_NORM_BIAS) != 0;

    int blk_d  = (d  + THREADS - 1) / THREADS;
    int blk_di = (di + THREADS - 1) / THREADS;

    /* 1. Embedding lookup → d_x */
    k_emb_lookup<<<blk_d, THREADS>>>(G.d_x, G.d_emb, token, d);

    /* 2. Layer loop */
    for (int li = 0; li < nl; li++) {
        LayerWeightsGpu* lw = &G.layers[li];

        /* 2a. Norm */
        if (has_nb && lw->norm_b) {
            k_layer_norm<<<1, THREADS>>>(
                G.d_normed, G.d_x, lw->norm_w, lw->norm_b, d, 1e-5f);
        } else {
            cudaMemcpy(G.d_normed, G.d_x, d * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            k_rms_norm<<<1, THREADS>>>(G.d_normed, lw->norm_w, d, 1e-5f);
        }

        /* 2b. in_proj: (2*di, d) @ normed → xz */
        k_gemv<<<2*di, THREADS>>>(lw->in_proj_w, G.d_normed, G.d_xz,
                                  2*di, d);

        /* 2c. Conv1d + SiLU (fused) */
        /* x_branch = xz[0..di], z_branch = xz[di..2*di] */
        k_conv1d_shift<<<blk_di, THREADS>>>(
            G.d_conv_out, G.d_conv_states[li], G.d_xz,
            lw->conv1d_w, lw->conv1d_b, di, dc);

        /* 2d. x_proj: (dr+2*ds, di) @ conv_out → z_proj */
        int xr = dr + 2 * ds;
        k_gemv<<<xr, THREADS>>>(lw->x_proj_w, G.d_conv_out, G.d_z_proj,
                                xr, di);

        /* 2e. dt_proj: (di, dr) @ z_proj[0..dr] → dt, then softplus+bias */
        k_gemv<<<di, THREADS>>>(lw->dt_proj_w, G.d_z_proj, G.d_dt, di, dr);
        k_softplus_bias<<<blk_di, THREADS>>>(G.d_dt, lw->dt_proj_b, di);

        /* 2f. SSM recurrence */
        /* B = z_proj[dr..dr+ds], C = z_proj[dr+ds..dr+2*ds] */
        float* d_B = G.d_z_proj + dr;
        float* d_C = G.d_z_proj + dr + ds;
        k_ssm_recurrence<<<blk_di, THREADS>>>(
            G.d_y, G.d_ssm_states[li], G.d_dt, G.d_conv_out,
            lw->a_log, d_B, d_C, lw->d_vec, di, ds);

        /* 2g. Gated SiLU: gated[i] = y[i] * silu(z_branch[i]) */
        float* d_z_branch = G.d_xz + di;
        k_gated_silu<<<blk_di, THREADS>>>(G.d_gated, G.d_y, d_z_branch, di);

        /* 2h. out_proj: (d, di) @ gated → result */
        k_gemv<<<d, THREADS>>>(lw->out_proj_w, G.d_gated, G.d_result, d, di);

        /* 2i. Residual add */
        k_residual_add<<<blk_d, THREADS>>>(G.d_x, G.d_result, d);
    }

    /* 3. Final norm */
    if (has_nb && G.d_norm_f_b) {
        float* d_tmp;
        cudaMalloc(&d_tmp, d * sizeof(float));
        cudaMemcpy(d_tmp, G.d_x, d * sizeof(float), cudaMemcpyDeviceToDevice);
        k_layer_norm<<<1, THREADS>>>(G.d_x, d_tmp, G.d_norm_f_w,
                                     G.d_norm_f_b, d, 1e-5f);
        cudaFree(d_tmp);
    } else {
        k_rms_norm<<<1, THREADS>>>(G.d_x, G.d_norm_f_w, d, 1e-5f);
    }

    /* 4. LM head: (vs, d) @ x → logits */
    k_gemv<<<vs, THREADS>>>(G.d_lm_head, G.d_x, G.d_logits, vs, d);

    /* 5. Copy logits to host */
    cudaError_t err = cudaMemcpy(G.h_logits, G.d_logits,
                                 vs * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return BACKEND_FORWARD_FAILED;

    return BACKEND_OK;
}

/* ====================================================================
 * Cleanup
 * ==================================================================== */

static void free_model(void) {
    if (!G.ready) return;
    uint32_t nl = G.cfg.n_layer;

    for (uint32_t i = 0; i < nl; i++) {
        LayerWeightsGpu* lw = &G.layers[i];
        cudaFree(lw->norm_w);
        if (lw->norm_b) cudaFree(lw->norm_b);
        cudaFree(lw->in_proj_w);
        cudaFree(lw->conv1d_w);
        cudaFree(lw->conv1d_b);
        cudaFree(lw->x_proj_w);
        cudaFree(lw->dt_proj_w);
        cudaFree(lw->dt_proj_b);
        cudaFree(lw->a_log);
        cudaFree(lw->d_vec);
        cudaFree(lw->out_proj_w);
        cudaFree(G.d_conv_states[i]);
        cudaFree(G.d_ssm_states[i]);
    }
    free(G.layers);
    free(G.d_conv_states);
    free(G.d_ssm_states);

    cudaFree(G.d_emb);
    if (!G.head_is_tied) cudaFree(G.d_lm_head);
    cudaFree(G.d_norm_f_w);
    if (G.d_norm_f_b) cudaFree(G.d_norm_f_b);

    cudaFree(G.d_x);
    cudaFree(G.d_normed);
    cudaFree(G.d_xz);
    cudaFree(G.d_conv_out);
    cudaFree(G.d_z_proj);
    cudaFree(G.d_dt);
    cudaFree(G.d_dt_raw);
    cudaFree(G.d_y);
    cudaFree(G.d_gated);
    cudaFree(G.d_result);
    cudaFree(G.d_logits);

    free(G.h_logits);
    memset(&G, 0, sizeof(G));
}

/* ====================================================================
 * BackendVTable FFI Exports
 * ==================================================================== */

extern "C" {

BackendResult redukt_backend_init(
    const uint8_t* weights_path,
    int32_t device_id,
    BackendInfo* info_out
) {
    if (!weights_path || !info_out)
        return BACKEND_INIT_FAILED;

    /* Read null-terminated path */
    int len = 0;
    while (weights_path[len]) len++;
    char* path = (char*)malloc(len + 32);
    if (!path) return BACKEND_INIT_FAILED;
    memcpy(path, weights_path, len);
    path[len] = '\0';

    /* Append /model.rbin if path is a directory */
    /* Simple heuristic: if doesn't end in .rbin, append */
    if (len < 5 || strcmp(path + len - 5, ".rbin") != 0) {
        strcat(path, "/model.rbin");
    }

    fprintf(stderr, "[redukt-gpu-cuda] init: weights=%s device=%d\n",
            path, device_id);

    /* Set CUDA device */
    CUDA_CHECK(cudaSetDevice(device_id >= 0 ? device_id : 0));

    /* Load weights */
    if (load_weights(path) != 0) {
        free(path);
        return BACKEND_WEIGHTS_NOT_FOUND;
    }
    free(path);

    G.ready = 1;

    /* Fill info */
    memset(info_out, 0, sizeof(*info_out));
    info_out->ffi_version = BACKEND_FFI_VERSION;
    info_out->vocab_size  = G.cfg.vocab_size;
    info_out->hidden_dim  = G.cfg.d_model;
    const char* name = "cuda-mamba";
    memcpy(info_out->name, name, strlen(name));

    fprintf(stderr, "[redukt-gpu-cuda] ready: vocab=%u d_model=%u "
            "n_layer=%u\n",
            G.cfg.vocab_size, G.cfg.d_model, G.cfg.n_layer);

    return BACKEND_OK;
}

BackendResult redukt_backend_forward(
    uint32_t token,
    float* out_logits,
    uint32_t vocab_size
) {
    if (!G.ready || !out_logits) return BACKEND_FORWARD_FAILED;
    if (vocab_size != G.cfg.vocab_size) return BACKEND_FORWARD_FAILED;
    if (token >= G.cfg.vocab_size) return BACKEND_FORWARD_FAILED;

    BackendResult res = mamba_forward(token);
    if (res != BACKEND_OK) return res;

    memcpy(out_logits, G.h_logits, vocab_size * sizeof(float));
    return BACKEND_OK;
}

void redukt_backend_reset_state(void) {
    if (!G.ready) return;
    uint32_t nl = G.cfg.n_layer;
    uint32_t di = G.cfg.d_inner;
    uint32_t ds = G.cfg.d_state;
    uint32_t dc = G.cfg.d_conv;
    for (uint32_t i = 0; i < nl; i++) {
        cudaMemset(G.d_conv_states[i], 0,
                   (size_t)di * (dc - 1) * sizeof(float));
        cudaMemset(G.d_ssm_states[i], 0,
                   (size_t)di * ds * sizeof(float));
    }
}

void redukt_backend_free(void) {
    free_model();
    fprintf(stderr, "[redukt-gpu-cuda] freed\n");
}

} /* extern "C" */
