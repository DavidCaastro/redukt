# Strata backbone — 01 Integration map (code)

> Stage 1a of `00-charter.md`. Discovery only: how a backbone plugs into Redukt today.
> No design and no architecture proposal — only what exists, where, and what would have to be touched.
> Snapshot: `master` @ `692d985` (clean tree), read on 2026-11-12.
> **Static reading only: nothing was built, run or measured for this file.** "Asserted by test X" means the
> test source says so, not that the test was executed.

## 0. How to read this file

All references are `path:line` relative to the repo root. Evidence tags:

| Tag | Meaning |
|---|---|
| (none) | **Verified**: the cited lines were read by the author of this map in this session. |
| **[S]** | Read in full by a delegated reader (kernels / RWKV / plugin files); the author re-read only the anchor lines. |
| **[I]** | **Inference** from verified code. Not asserted by any test, not executed. |
| **[D]** | Statement taken from a repo document (CLAUDE.md, FORMAT-SPEC.md, ...). Not checkable from the code. |

Short path names: `backbone.rs`, `coder.rs`, `tensor.rs`, `gemv.rs`, `gemv_fma.rs`, `fmath.rs`, `quant.rs`,
`par_pool.rs`, `cm_ensemble.rs`, `mixer.rs`, `bias_head.rs`, `ngram.rs`, `error.rs`, `backend_ffi.rs` live in
`src/domain/`; `codec.rs`, `lockstep.rs`, `api.rs`, `model_loader.rs`, `benchmark.rs` in `src/application/`;
`mamba/…`, `rwkv7/…`, `plugin_backend/…`, `governor.rs` in `src/infrastructure/`.

### 0.1 Twelve facts that shape everything else

1. A backbone is four required methods: `vocab_size`, `create_state`, `create_scratch`,
   `forward_into(token: usize, &mut State, &mut Scratch, out_logits: &mut [f32])` (`backbone.rs:141-158`), plus
   `State::reset` (`:126-128`). RWKV implements exactly that and nothing else (`rwkv7/model.rs:1399-1423`).
2. The output contract is **f32 logits in natural-log units, one per vocabulary entry, in a caller-owned
   buffer** that is never cleared between tokens (`codec.rs:1100`, `lockstep.rs:174`). There is no integer or
   probability interface and no error return.
3. Dispatch is **closed enums** (`DynModel`/`DynState`/`DynScratch`/`DynTokenizer`, `model_loader.rs:81-110`)
   plus static generics (`lockstep.rs:163`, `api.rs:477`). No `dyn Backbone` exists in `src/`.
4. Everything after the backbone is floating point until the CDF: f32 mixing, **f64 softmax through `fmath`**,
   then a 24-bit integer CDF and an integer range coder (`codec.rs:737-750`, `coder.rs:43-81`). That path is
   scalar IEEE code with no SIMD intrinsics, so the backbone is the only place where ISA variance enters
   today [I] (section 4.4).
5. The host path costs **O(V) per token several times over, with four heap allocations per token**
   (section 4.2). At V = 50,280 that is the measured R-110 bottleneck [D]; at V = 256 it is ~200x smaller [I].
6. BACKBONE ids 0-3 are taken (RWKV, Mamba, MambaByte, Plugin: `backbone.rs:20-38`); **4 is the next free
   id**. All 8 FLAGS bits are used (`codec.rs:269-271`). Free header values: PROFILE bit 7, QUANT codes 2-15,
   contract code 3 (`codec.rs:209-230`). Every unknown value fails in `parse_header`, before any model load.
7. PROFILE semantics (quant, FMA contract, reset period) are **hard-wired to "Mamba family"** in a handful of
   predicates (`codec.rs:139-141, 431-445, 662-694`; `model_loader.rs:340-369, 463-468`; `api.rs:321-330`).
   A new `BackboneChoice` variant silently gets: QUANT = F32, contract 1, no reset, not auto-routed (section 3.5).
8. The header identifies neither the weights file nor any model dimension. A wrong weights file surfaces only
   as a CRC-32 mismatch after decoding everything (`codec.rs:1701-1709`).
9. SIMD exists only for x86-64 (AVX2, FMA, AVX-VNNI) with scalar fallbacks; **there is no NEON code**
   (section 6.2). Existing F32 kernels are *not* bit-identical between AVX2 and scalar, and the tier is not in
   the header (section 4.4).
10. The only integer matrix kernel is RWKV's W8A8 (int8 x int8 -> i32 -> f32, `tensor.rs:1699-1747`), private
    to `tensor.rs` apart from its f32-in/f32-out entry point. There is **no int8 x int16 kernel and no
    fixed-point pipeline** (section 6.1).
11. The crate is **binary-only** (no `src/lib.rs`): integration tests can only drive the executable; anything
    that needs internal types must be an in-crate `#[cfg(test)]` unit test (section 9).
12. Warnings are errors (CI `RUSTFLAGS=-D warnings`, `.github/workflows/ci.yml:11`), dead code included, and
    `tests/libm_lint.rs` textually bans `.exp(`, `.ln(`, `.log(`, `.tanh(`, `.powi(`… in every non-test line
    under `src/` (section 4.5).

---

## 1. Trait contract and dispatch

### 1.1 Signatures (complete, `src/domain/backbone.rs`)

```rust
pub trait BackboneState: Send {                                   // :126
    fn reset(&mut self);                                          // :128
    fn set_reset_every(&mut self, _every: u64) {}                 // :131  default no-op (R-108)
}
pub trait BackboneScratch {                                       // :135  no Send / Sync bound
    fn reset_perf(&mut self) {}                                   // :137
}
pub trait Backbone: Sync {                                        // :141
    type State: BackboneState;                                    // :142
    type Scratch: BackboneScratch;                                // :143
    fn vocab_size(&self) -> usize;                                // :146
    fn create_state(&self) -> Self::State;                        // :149
    fn create_scratch(&self) -> Self::Scratch;                    // :152
    fn forward_into(&self, token: usize, state: &mut Self::State,
                    scratch: &mut Self::Scratch, out_logits: &mut [f32]);            // :155-158
    fn forward_multi(&self, tokens: &[usize], states: &mut [&mut Self::State],
                     scratches: &mut [&mut Self::Scratch],
                     out_logits: &mut [&mut [f32]]) { /* N x forward_into */ }       // :164-171
    fn stream_bytes(&self) -> usize { 0 }                         // :175-177  memory report only
    fn prepare_streams(&self, _n: usize) -> Result<(), String> { Ok(()) }            // :182-184  R-109
}
pub trait BackendTokenizer: Sync {                                // :188
    fn encode(&self, data: &[u8]) -> Vec<u32>;                    // :189
    fn decode_token(&self, id: u32) -> &[u8];                     // :190
}
```

### 1.2 Semantics the callers rely on

| Aspect | Fact | Evidence |
|---|---|---|
| Output type | `f32` logits, length exactly `vocab_size()`. | `backbone.rs:155-158` |
| Buffer ownership | Caller allocates once per stream (`vec![0.0f32; v]`) and reuses it; it is never zeroed between tokens, so the backbone must overwrite every element. | `codec.rs:1100, 1518`; `lockstep.rs:174`; `api.rs:502` |
| Units | Additive log domain, base e: CM biases are added element-wise, then `exp` is applied. | `mixer.rs:66-73`; `codec.rs:743`; `tensor.rs:1807` |
| Step meaning | One call consumes token *t* and produces the logits used to code token *t+1*. Token 0 is coded with a uniform CDF before any forward. | `codec.rs:1197-1201, 1239-1241`; `lockstep.rs:310-322` |
| Errors | None. `forward_into` returns `()`; the only failure channel is a panic (the plugin does this). | `plugin_backend/model.rs:385, 398, 429` |
| Inputs available | Only the token id. No byte offset, no file length, no access to the raw bytes or to CM state. | `backbone.rs:155-158` |
| Per-stream configuration | Exactly one hook: `set_reset_every(u64)`, called once per file. Anything else must be fixed at model load. | `codec.rs:1098, 1516`; `lockstep.rs:194`; `api.rs:500` |
| State reuse | In lockstep a slot's `State` is reused for the next file after `reset()` + `set_reset_every()` + `scratch.reset_perf()`. `reset()` must restore the exact post-`create_state` condition, counters included. | `lockstep.rs:193-195`; `mamba/model.rs:298-302` |
| `forward_multi` | `out_logits[i]` must be bit-identical to `forward_into` on stream *i* alone, for any N and any batch composition. | `backbone.rs:160-163`; test `mamba/model.rs:1086-1128` |
| N = 1 | `/v1/compress` always goes through `forward_multi` with one stream; CLI single-file goes through `forward_into` (or the Mamba split-head pipeline). API == CLI (gate 2c) therefore tests `forward_multi(1) == forward_into`. | `api.rs:305-314`; `lockstep.rs:244`; `codec.rs:1129-1190, 1239` |
| Token range | Token ids must be `< vocab_size()`. Nothing validates it; a larger id panics on an index (`cdf[symbol + 1]`, `probs[correct_token]`). | `coder.rs:96-98`; `bias_head.rs:90`; `mixer.rs:83` |

Thread-safety as used: the model is shared by `&M` across pool threads (`Backbone: Sync`); states and scratches
are created and kept on the engine thread and lent as `&mut` to `forward_multi` (`lockstep.rs:171-172, 231-244`);
the tokenizer is called from pool threads in Phase A (`lockstep.rs:210-213`), hence `BackendTokenizer: Sync`.
`State: Send` is required by the trait, but no current engine moves a state to another thread [I].

### 1.3 Dispatch

- **Static generics** in the engines: `run_lockstep<M: Backbone, T: BackendTokenizer>` (`lockstep.rs:163-167`),
  `decompress_bytes<M, T>` (`api.rs:477-481`), `compress_and_measure` / `verify_roundtrip`
  (`benchmark.rs:302, 401`).
- **Closed enums** in the application layer: `DynModel`, `DynState`, `DynScratch`, `DynTokenizer`
  (`model_loader.rs:81-110`) implement the four traits by `match` (`:112-275`). Mismatched variants panic
  (`:190, 212, 234`). A grep for `dyn Backbone` / `dyn BackendTokenizer` in `src/` returns nothing.
- The CLI single-file paths use `DynModel` concretely because they call inherent helpers that are **not** in
  the trait (`model_loader.rs:277-380`):

| Helper | Lines | Who answers today |
|---|---|---|
| `supports_pipeline()` | 279-281 | Mamba / MambaByte only (`matches!`, so a new variant is `false`) |
| `hidden_dim()` | 284-291 | exhaustive `match` (RWKV returns `vocab_size()`, unused there) |
| `forward_layers_normed()`, `compute_head_into()` | 295-320 | Mamba family; everything else panics |
| `embedding_surgery()` | 323-327 | RWKV only (no-op otherwise) |
| `weight_profile()` | 340-345 | exhaustive `match`; non-Mamba report `F32` |
| `set_numeric_contract()` | 350-361 | exhaustive; contract 2 is an error for non-Mamba |
| `write_contract()` | 364-369 | exhaustive; non-Mamba return `NumericContract::CURRENT` |
| `rwkv_perf()`, `mamba_sparsity()` | 330-336, 372-379 | diagnostics, `if let` / `_ => None` |

- `DynModel::forward_multi` has one explicit arm per variant; RWKV loops over `forward_into`
  (`model_loader.rs:197-238`).

### 1.4 Reference implementations to copy from

| Implementation | Where | What it shows |
|---|---|---|
| RWKV-7 | `rwkv7/model.rs:1399-1423` | The minimal surface: no `forward_multi`, no `stream_bytes`, no reset period. |
| Mamba | `mamba/model.rs:983-1023` (+ inherent fns `:635-976`) | Full surface: lockstep GEMM, periodic reset inside the state (`:315-320`), `stream_bytes`. |
| GPU plugin | `plugin_backend/model.rs:362-452` | State that lives outside the host (`prepare_streams`, slots). |
| Test mock | `lockstep.rs:805-829` | 25-line deterministic backbone with vocab 256 + `ByteTokenizer`; drives the whole lockstep engine without weights. |
| Synthetic model | `mamba/model.rs:1045-1076` | How unit tests build a real-structure model from a PRNG, no weights file. |

---

## 2. Selection and loading

### 2.1 CLI flags (compress: `codec.rs:950-984`; decompress: `codec.rs:1374-1391`)

| Flag | Parsed at | Resolution | Notes |
|---|---|---|---|
| `--backbone <B>` | `codec.rs:979` | `BackboneChoice::from_str_opt` (`backbone.rs:42-50`): `rwkv`, `mamba`, `mambabyte`, `plugin`; anything else -> `None` -> `route_backbone` (`backbone.rs:121-123`), which always returns Mamba. | **An unknown or misspelled name silently means "auto" = Mamba** (`codec.rs:1055-1059`). Compress only; decompress takes the backbone from the header (`codec.rs:1493-1495`). |
| `--quant <L>` | `codec.rs:964, 1383` | Two independent parsers: `WeightProfile::from_cli` (`backbone.rs:96-105`) via `resolve_compress_profile` (`codec.rs:662-669`) for the Mamba family; `parse_quant_flag` (`model_loader.rs:444-451`) -> `QuantPolicy` for RWKV (`codec.rs:1074`). | Non-Mamba backbones always get profile `F32` (`codec.rs:663-665`). On decompress a conflicting value is an error before load (`codec.rs:673-694`). |
| `--numeric <C>` | `codec.rs:980` | `parse_numeric_flag` (`codec.rs:449-455`) -> `model_loader::set_process_contract` (`codec.rs:988-989`), a process-global `AtomicU8` (`model_loader.rs:387-397`). | Applied in `load_dyn_model` to the Mamba family only (`model_loader.rs:420, 431`). Not listed in `--help` (`main.rs:37-103`). |
| `--reset-every <N>` | `codec.rs:959` | `resolve_reset_log2` (`codec.rs:646-656`) -> `parse_reset_every` (`:144-158`): 0 or a power of two in 256..=1,048,576. | Default per backbone: `default_reset_log2` (`:134-136`) = 12 for Mamba / MambaByte / Plugin, 0 otherwise. For other backbones a non-zero value is ignored with a warning (`:651-654`). |
| `--jobs <N>` | `codec.rs:957, 1381` | Lockstep streams; default `DEFAULT_SLOTS = 8`, clamp 1..=`MAX_SLOTS` = 256 (`lockstep.rs:46-48, 647`). | Only with `--input-dir` (`codec.rs:1014-1031`). Worker threads come from `REDUKT_THREADS`, not from `--jobs`. |
| `--weights <DIR>` | `codec.rs:958, 1382` | Default `weights/rwkv7-0.1b` (`codec.rs:930, 1367`; `api.rs:58`). | Overloaded: see 2.3. |
| `--lr`, `--ngram-scale`, `--mix-eta` | `codec.rs:961-963` | CM hyperparameters; stored in the header only when different from the defaults 0.30 / 0.50 / 0.01 (`codec.rs:52-54, 539, 569-573`). | One global default set; there is no per-backbone default table. |
| `--adaptive-skip`, `--skip-threshold` | `codec.rs:971-975` | Entropy gate (R-077), single-file only (`codec.rs:1020-1024`). | Not listed in `--help`. |
| `--gpu`, `--tokenizer bpe`, `--model-size tiny` | `codec.rs:976-978, 992-1006` | Stubs that return an error. | |
| anything else | `codec.rs:981, 1388` | `_ => {}`: **unknown flags are silently ignored** by compress/decompress. | The API rejects unknown flags (`api.rs:75-77`). |

### 2.2 `load_dyn_model`

`pub fn load_dyn_model(backbone: BackboneChoice, rwkv_weights: &str, quant_override: Option<QuantPolicy>,
mamba_profile: WeightProfile) -> (DynModel, DynTokenizer)` (`model_loader.rs:402-441`).

- Returns a tuple, **not a `Result`**: every loader failure is a panic or `process::exit`
  (`mamba/model.rs:387`; `mamba/tokenizer.rs:33`; `rwkv7/safetensors.rs:99`; `model_loader.rs:437`;
  `ram_monitor.rs:90`).
- One `match` arm per `BackboneChoice` (exhaustive): the place where a new backbone is constructed.
- Call sites: `codec.rs:1075` (compress), `codec.rs:1493` (decompress), `lockstep.rs:567` (batch, once per
  model group), `api.rs:190` (server start-up).
- MambaByte is the precedent for "same model struct, different `BackboneChoice` and tokenizer":
  `MambaModel` + `ByteTokenizer` under header id 2 (`model_loader.rs:423-434`).

### 2.3 Where weights live and how they are found

- Paths are **relative to the process working directory**; there is no env var and no executable-relative
  lookup for model weights (`model_loader.rs:42, 45`; `deploy/gcp/entrypoint.sh:24-25` works around it with a
  symlink).
- Mamba: `weights/mamba-130m` if that directory exists in the CWD, **otherwise** the `--weights` value
  (`model_loader.rs:414-418`). So `--weights` cannot override Mamba while `weights/mamba-130m` exists.
- MambaByte: `weights/mambabyte-96m` if it exists, else `--weights` (`model_loader.rs:424-428`). That directory
  does not exist on disk (only `weights/mambabyte-353m/` does).
- RWKV: `--weights` is a directory (-> `model.safetensors`), a `.safetensors` file or a `.gguf` file
  (`rwkv7/model.rs:472-486`); vocab file resolved by `resolve_vocab_path` (`model_loader.rs:27-39`).
- Plugin: `--weights` is passed verbatim to the plugin, device 0 (`model_loader.rs:436`).
- File names inside a Mamba directory: `model.safetensors` (`mamba/model.rs:378-382`) and `tokenizer.json`
  (`mamba/tokenizer.rs:25-29`).
- Policy: `weights/` is gitignored by extension (`.gitignore:3-8`: `.safetensors`, `.gguf`, `.bin`, `.json`,
  `.txt` only — a new extension would **not** be ignored). Every weights directory needs an entry in
  `weights/MANIFEST.md` before use (`weights/MANIFEST.md:86`).

### 2.4 Weight formats and loaders

| Format | Loader | API | Limits |
|---|---|---|---|
| SafeTensors | `rwkv7/safetensors.rs` (generic despite its location; Mamba imports it: `mamba/model.rs:20`) | `SafeTensorsFile::open(&Path) -> AzResult<Self>` (`:32-55`), `load_tensor(name) -> AzResult<Tensor>` (`:62-93`), `get_tensor(name) -> Tensor` (panics, `:97-100`), `has_tensor` (`:102`), `tensor_names` (`:57`) | Whole file read into memory, capped at 2 GiB (`error.rs:88-101`). Dtypes **`F32`, `F16`, `BF16` only, always widened to an f32 `Tensor`** (`:70-90`). No integer dtypes, no raw-bytes accessor. **`__metadata__` is skipped and not exposed** (`:152-154`). Tensor offsets are sliced unchecked (`:66-68`). |
| GGUF | `rwkv7/gguf.rs` (RWKV only) **[S]** | `GgufFile::open`, `load_tensor`, `load_tensor_raw`, metadata access | v2/v3; F32/F16/BF16 and llama.cpp quant types dequantized to f32 or kept as native blocks. |
| `.rbin` | Parsed by the CUDA plugin; the host only peeks at the 44-byte header (`plugin_backend/model.rs:114-168`) | n/a | Mamba-1 specific layout, f32 only **[S]**. |

- Model dimensions: Mamba derives them from tensor shapes and tensor-name probing (`mamba/model.rs:390-404`,
  `d_state = 16`, `d_conv = 4` hard-coded at `:39-40`). RWKV picks a constant config by a substring of the
  weights path (`rwkv7/model.rs:41-51`). Nothing in `src/` reads a `config.json` **[S]**.
- Quantized profiles are produced **at load time from f32 weights** (`MambaMatrix::new`, `mamba/model.rs:66-71`
  -> `Q8wMatrix::quantize`, `gemv.rs:254-271`). The quantization arithmetic is therefore part of the format
  contract (`mamba/model.rs:50-56`). There is no loader for weights that are stored quantized, except GGUF.
- RAM guard before loading: `check_ram_or_abort(90.0, …)` (`mamba/model.rs:384`), Windows-only (section 7).

### 2.5 Automatic profiles

- Mamba family: `--quant auto` / unset -> `AUTO_MAMBA_PROFILE = Q8` (`model_loader.rs:457`). Deterministic and
  written to the header.
- RWKV: `--quant auto` / unset -> **a wall-clock benchmark at load time** (`rwkv7/model.rs:296-308` ->
  `quant.rs:477-531`: `score = speed_factor * sqrt(memory_factor)` among levels that pass a quality gate), and
  the chosen level is **not recorded in the archive** (`model_loader.rs:338-345`; FORMAT-SPEC.md:213-214).
  The decoder repeats the benchmark or trusts its own `--quant` (`codec.rs:1492-1495`).

---

## 3. AZTH header

Authoritative text: `docs/api/FORMAT-SPEC.md`. This section maps it to code.

### 3.1 Layout as implemented (v5, the only version written)

Parse order (`parse_header`, `codec.rs:264-423`) = write order (`build_v4_header` `:534-591` then
`set_v5_profile` `:617-625`):

| # | Field | Size | Present when | Written | Read |
|---|---|---|---|---|---|
| 1 | MAGIC `AZTH\x05\0\0\0` | 8 | always | `:566, 618` | `validate_magic` `:171-197` |
| 2 | TOTAL_BYTES (u32 LE) | 4 | always | `:567` | `:265-267` |
| 3 | FLAGS | 1 | always | `:547-555, 568` | `:268-277` |
| 4 | PROFILE | 1 | v5 | `profile_byte` `:233-235`, `:619-624` | `parse_profile_byte` `:209-230` |
| 5 | RESET_LOG2 | 1 | PROFILE bit 6 | `:620-623` | `:287-298`, range check `:161-168` |
| 6 | BIAS_LR, NGRAM_SCALE, MIX_ETA (f32 LE) | 12 | FLAGS 0x01 | `:569-573` | `:301-331` |
| 7 | EMB_METHOD + EMB_ALPHA | 5 | FLAGS 0x02 | `:574-577` | `:334-350` |
| 8 | SHUFFLE stride | 1 | FLAGS 0x08 | `:578-580` | `:355-366` |
| 9 | DELTA stride | 1 | FLAGS 0x10 | `:581-583` | `:369-380` |
| 10 | SKIP threshold | 1 | FLAGS 0x20 | `:584-586` | `:383-394` |
| 11 | **BACKBONE id** | 1 | FLAGS 0x40 (every backbone except RWKV: `:545`) | `:587-589` | `:397-410` |
| — | STREAM | var | | `codec.rs:1324` | `extract_stream_and_crc_v` `:493-511` |
| — | CRC32 (LE) of the bytes that entered the coder | 4 | v3+ | `codec.rs:1314, 1325` | `:502-510`, verified `:1701-1709` |

FLAGS bits: 0x01 HP, 0x02 EMB, 0x04 E8E9, 0x08 SHUFFLE, 0x10 DELTA, 0x20 SKIP, 0x40 BACKBONE,
0x80 ENHANCED_CM (`codec.rs:44-51`). **All eight are assigned** (`codec.rs:269-271`; FORMAT-SPEC.md:52).
Current encoders always set 0x80 (`codec.rs:1320`; `lockstep.rs:406`).

PROFILE (`codec.rs:25-42`): bits 0-3 QUANT (0 = F32, 1 = Q8, 2 = Q6 reserved), bits 4-5 NUMERIC_CONTRACT
(0 legacy libm, 1 fmath-v1, 2 fma-v1), bit 6 RESET_PERIOD, bit 7 reserved.

Two builders exist. `build_header_with(…, profile, contract, reset_log2)` (`codec.rs:629-642`) is used by
single-file compress with `model.write_contract()` (`codec.rs:1079, 1317-1321`). `build_header(…, profile,
reset_log2)` (`:600-613`) is used by lockstep and therefore by the API (`lockstep.rs:404-407`); it takes the
contract from the process-global for the Mamba family and `NumericContract::CURRENT` for everything else
(`codec.rs:606`).

### 3.2 What the current decoder does with unknown values

All of these are raised by `validate_magic` / `parse_header` / `resolve_decompress_profile`, i.e. **before**
`load_dyn_model` (`codec.rs:1427-1428, 1486` precede `:1493`); none of them reaches the CRC check.

| Value | Error text | Where |
|---|---|---|
| version byte not in 2..=5 | `unsupported AZTH version {} (expected 2, 3, 4 or 5)` | `codec.rs:182-190` |
| unknown FLAGS bit | cannot happen: the mask covers all 8 bits | `codec.rs:269-277` |
| PROFILE bit 7 | `AZTH v5 PROFILE has reserved bits set: 0x..` | `codec.rs:210-214` |
| NUMERIC_CONTRACT = 3 | `AZTH v5 numeric contract 3 not supported by this build …` | `codec.rs:215-221` |
| QUANT = 2..15 | `AZTH v5 weight profile code N = Q6 (reserved) / unknown — not supported by this build` | `codec.rs:222-228` |
| RESET_LOG2 outside 8..=20 | `AZTH v5 RESET_LOG2 out of range` | `codec.rs:161-168, 293` |
| **BACKBONE id >= 4** | `AZTH v{} unknown backbone: {}` | `backbone.rs:30-38`; `codec.rs:403-407` |
| contract 2 with a non-Mamba backbone | `AZTH numeric contract fma-v1 is only defined for the Mamba family…` | `codec.rs:413, 437-445` |
| QUANT != F32 with a non-Mamba backbone | `… declares weight profile … (only Mamba supports profiles)` | `codec.rs:674-683`; `api.rs:321-330` |

Behaviour of decoders older than this tree is documented, not verifiable here [D]: a v4-only decoder says
`unsupported AZTH version 5`; an R-099 decoder says `PROFILE has reserved bits set` (FORMAT-SPEC.md:169-171).

### 3.3 Where the parsed header is consumed

`HeaderInfo` (`codec.rs:99-117`) is the single struct that carries the header to all three decode paths:
`cmd_decompress` (`codec.rs:1428`), lockstep `decompress_key` (`lockstep.rs:704-721`) and the API
`check_archive` (`api.rs:333-349`). It is constructed in exactly two places (`codec.rs:249-262, 415-422`).
`header_size` is simply the running parse offset (`codec.rs:421`).

### 3.4 Extension points for a backbone with its own configuration (facts, not a recommendation)

| Option | What it is today | What current decoders do with it | What would have to change |
|---|---|---|---|
| New BACKBONE id (4..=255) | `u8`, ids 0-3 known (`backbone.rs:20-38`; FORMAT-SPEC.md:121-127). | Fail early: `unknown backbone: 4`. | `BackboneChoice` + the unit test that asserts `from_header_byte(4) == None` (`codec.rs:2727`). |
| Extra bytes right after the BACKBONE byte | BACKBONE is the **last** header field (`codec.rs:397-410, 421`); whatever follows is STREAM for a current decoder. | Never reached: the decoder stops at the unknown id first. | Read them in `parse_header` after `:407` (advance `offset`), write them in `build_v4_header` after `:588`, add a `HeaderInfo` field. Archives with ids 0-3 are byte-for-byte unaffected. |
| PROFILE bit 7 | The only unassigned PROFILE bit (`codec.rs:25-28, 210`). Bit 6 + RESET_LOG2 is the precedent (R-108: `:286-298, 617-625`). | Fail early: `reserved bits set`. | `parse_profile_byte` mask, `set_v5_profile`, `parse_header`. |
| QUANT codes 2-15 | Reserved; `WeightProfile` is a 2-variant enum (`backbone.rs:68-92`). | Fail early: `weight profile code N …`. | Also the two "only Mamba supports profiles" checks (`codec.rs:674-683`; `api.rs:321-330`) and `resolve_compress_profile` (`:662-669`). |
| NUMERIC_CONTRACT code 3 | Last free code (`fmath.rs:60-67`). Today a contract means "which exp/ln feed the CDF" plus "FMA GEMV for Mamba" (`fmath.rs:30-44`). | Fail early: `numeric contract 3 not supported`. | `NumericContract`, `check_contract_backbone` (`codec.rs:437-445`). |
| New version byte (6) | `validate_magic` accepts 2..=5 (`codec.rs:183-190`). | Fail early: `unsupported AZTH version 6`. | New MAGIC const (`:17-21`), `validate_magic`, a `parse_header` arm (`:264`), `set_v5_profile` (`:618`), `formats_read` string (`api.rs:171`). The v2-v5 arms stay untouched. |
| A FLAGS bit | None free. | — | — |

Related constraints:

- `read_header_prefix` reads only the first **64 bytes** of an archive (`codec.rs:458-464`), used by
  `batch_numeric_contract` (`:470-490`). The largest header today is 36 bytes.
- TOTAL_BYTES is `u32` (`codec.rs:1315`; `error.rs:105-113`).
- Fixed overhead for a non-RWKV archive without RESET_LOG2: 8 + 4 + 1 + 1 + 1 = **15 bytes of header + 4 bytes
  of CRC**; the first token costs `log2(V)` bits under the uniform CDF (`coder.rs:84-92`); the coder flush
  emits its pending bytes in `finish` (`coder.rs:184-197`).
- The header carries **no weights fingerprint and no model dimensions** (`codec.rs:99-117`).

### 3.5 What a new `BackboneChoice` variant gets if `codec.rs` is not touched

`codec.rs` has no exhaustive `match` on `BackboneChoice` (it uses `matches!` / `==`), so it compiles unchanged
and applies the non-Mamba defaults:

- QUANT = F32 regardless of `--quant` (`codec.rs:662-665`); the decoder **requires** F32 (`:674-683`).
- NUMERIC_CONTRACT = whatever the new `DynModel::write_contract` arm returns in single-file mode
  (`codec.rs:1079`), and `CURRENT` (1) in lockstep / API (`codec.rs:606`). `--numeric fma` only prints a
  warning (`codec.rs:1068-1071`).
- No RESET_LOG2; `--reset-every N` is ignored with a warning (`codec.rs:134-141, 651-654`).
- BACKBONE byte written (`codec.rs:545`), ENHANCED_CM set.
- Never auto-routed (`backbone.rs:121-123`): reachable only through `--backbone <name>`; in the API only if
  listed at start-up (`api.rs:86-103, 286-287`).
- The non-pipeline encode loop (`codec.rs:1123, 1193-1257`).
- Lockstep groups by `(backbone, profile, emb_surgery)` only (`lockstep.rs:499-503`).

---

## 4. Numeric flow: backbone -> CM / mixer -> CDF -> coder

### 4.1 Per-token sequence

Encoder, lockstep path (`lockstep.rs:304-322`); the single-file loops are the same sequence
(`codec.rs:1193-1257`), with the differences listed in 4.3:

| Step | Call | Types in -> out | Lines |
|---|---|---|---|
| 0 | token 0: `enc.encode_symbol(tok, &Cdf::uniform(v))` | `usize` -> integer coder state | `lockstep.rs:310-311`; `coder.rs:84-92` |
| 1 | `cm.compute_biases(None)` | CM state -> 5 x `Vec<f32>` of length V (each zero-filled, then sparse writes) | `cm_ensemble.rs:68-74` |
| 2 | `cm.combine_into(ensemble, logits)` | `&[f32]` backbone logits + biases -> `&mut [f32]` | `cm_ensemble.rs:77-91`; `mixer.rs:59-74` |
| 3 | `cdf_from_logits(ensemble)` | `&[f32]` -> **f64** softmax (`contract.exp`) -> `Vec<f32>` probs -> `Cdf` (`Vec<u32>`, total 2^24, min freq 1) | `codec.rs:729-750`; `coder.rs:43-81` |
| 4 | `enc.encode_symbol(tok, &cdf)` | integer only (`u64` low, `u32` range) | `coder.rs:162-181` |
| 5 | `softmax_into(ensemble, probs)` | `&[f32]` -> `&mut [f32]` with f32 `portable_expf` | `tensor.rs:1797-1813` |
| 6 | `cm.update(probs, tok)` | f32 SGD on 5 mixer weights + bias head | `cm_ensemble.rs:94-105`; `mixer.rs:77-108`; `bias_head.rs:88-109` |
| 7 | `cm.observe(tok as u32)` | integer / hash-table updates, plus one f32 surprise EMA in `SegmentModel` (`segment_model.rs:61-71`) | `cm_ensemble.rs:157-163` |
| 8 | Phase B: `model.forward_multi(tokens, states, scratches, logits)` | `usize` tokens -> `&mut [f32]` logits per stream | `lockstep.rs:228-246` |

The decoder mirrors it (`lockstep.rs:324-368`; `codec.rs:1595-1665`; `api.rs:535-573`): build the same CDF,
`decode_symbol_checked` (`coder.rs:286-294`), then steps 5-8 with the decoded symbol.

### 4.2 Where floating point is, and what it costs

- Floating point: steps 1, 2, 3 (f32 then f64), 5, 6 and a small part of 7. Integer: steps 0, 4 and the coder.
  The float -> integer boundary is `Cdf::from_probs` (`coder.rs:57`: `(p as f64 * inv * remaining as f64 + 0.5)
  as u64`).
- Two separate softmaxes per token: f64 through `fmath` for the CDF (step 3) and f32 through `portable_expf`
  for the CM update (step 5).
- CM math uses f32 `portable_lnf` (`mixer.rs:104`; `bias_head.rs:90`; `ngram.rs:127-128`;
  `segment_model.rs:44-50`; `match_model.rs:54`; `run_model.rs:29`; `word_model.rs:87`).
- **O(V) work per token**: 5 buffer fills (step 1), 6 x V mul-adds (step 2), V f64 `exp` + V f32 `exp`
  (steps 3, 5), ~6 x V in step 6, plus the CDF construction.
- **Four O(V) heap allocations per token**: `probs` and `probs_f32` (`codec.rs:740, 748`), `freqs` and `cdf`
  (`coder.rs:50, 71`).
- The CM is token-level over the backbone vocabulary: `CmEnsemble::new(v, …)` (`cm_ensemble.rs:35-58`) holds
  six predictors (n-gram order 4, bias head, match, run, word, segment); the mixer weighs five and the segment
  bias is added unweighted (`mixer.rs:19-27`; `cm_ensemble.rs:87-90`).
- The backbone cannot switch the CM off or parametrize it other than through the three header hyperparameters;
  `compute_biases` is always called with `entropy_hint = None` in codec paths (`codec.rs:1166, 1203, 1563,
  1601`; `lockstep.rs:313, 328`; `api.rs:515, 540`).
- [D] CLAUDE.md (R-110) attributes ~0.85 ms per token-stream at V = 50,280 to this host path plus the logits
  copy, and states that the split has not been measured.

### 4.3 Differences between the three encode / decode loops

| Loop | Extra work | Evidence |
|---|---|---|
| Single-file compress, non-pipeline (every backbone that is not Mamba, or any with the gate on) | Calls `cm.cm_only_entropy(&mut ensemble_buf)` **on every token, gate enabled or not**: V CM-only logits + 2 x V f64 `exp`. Also runs one forward for the last token that nobody uses. | `codec.rs:1207, 1239-1241`; `cm_ensemble.rs:133-154` |
| Single-file compress, Mamba pipeline | `forward_layers_normed` + `compute_head_into`; overlaps head(t-1) with layers(t) on a scoped thread only when the pool has 1 thread. | `codec.rs:1123-1190` |
| Single-file decompress | Calls `cm_only_entropy_with(…, hdr.numeric_contract)` on every token. | `codec.rs:1604` |
| Lockstep (batch, and API compress) | No entropy-gate code at all; archives with SKIP are rejected. | `lockstep.rs:304-368, 713-717` |
| API decompress | Own loop `decompress_bytes`; calls `cm_only_entropy` (contract `CURRENT`, not the header's). | `api.rs:535-573` |

When the gate skips a token the backbone is **not** advanced for it (`codec.rs:1235-1243`).

### 4.4 Determinism mechanisms that exist

| Mechanism | What it guarantees | Evidence |
|---|---|---|
| `fmath::exp_f64` / `log_f64` | Pure-Rust f64 ports of musl; only `+ - * /`, comparisons and bit ops; no `mul_add`, no libm. Golden bit patterns and a sweep hash in unit tests **[S]**. | `fmath.rs:1-24, 383, 480, 525-669` |
| `NumericContract` in PROFILE bits 4-5 | Selects `exp`/`ln` for the CDF and the entropy gate; contracts 1 and 2 share the same transcendentals. The decoder passes the header contract to `cdf_from_logits_with`. | `fmath.rs:33-103`; `codec.rs:1614`; `lockstep.rs:330` |
| `tensor::portable_*` (f32) | `portable_expf` and `portable_sigmoidf` are asserted bit-identical to their AVX2 twins; `portable_lnf` is scalar only. | `tensor.rs:765-861`; tests `:2641-2698` |
| `Kernel::Plain` Q8 GEMV | AVX2 and scalar implement the same order: 8 lanes (lane *l* = columns ≡ *l* mod 8), `hsum ((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7))`, sequential tail, `sum += dot * scale` per group. Asserted bit-identical **[S]**. | `gemv.rs:13-20, 460-500, 515-602`; test `:719` |
| `Kernel::Fma` (contract 2) | Same order with one `fma` per multiply-add; scalar fallback uses `f32::mul_add`. Asserted bit-identical **[S]**. | `gemv_fma.rs:1-24, 64-102`; test `gemv.rs:853` |
| Row partitioning on the pool | The pool decides which thread computes a block, never how; outputs are independent of the thread count. Asserted for 1/3/8 and 1/2/5/12 threads **[S]**. | `par_pool.rs:10-12`; `gemv.rs:107-127`; tests `:640, 761` |
| Lockstep identity | Each stream executes the single-file sequence on private state; multi-stream kernels keep the per-(stream, row) order. | `lockstep.rs:15-21`; tests `gemv.rs:791, 821`; `mamba/model.rs:1086`; `lockstep.rs:871` |
| Header-declared profile / contract / reset period | Encoder and decoder cannot disagree silently on these three. | section 3 |
| Rust semantics | No fast-math: no reassociation and no implicit FMA contraction, so scalar loops keep source order even when auto-vectorized [I]. | stated in `fmath.rs:15-21` |

**What is not covered:**

- **SIMD tier of the F32 paths.** The Plain F32 GEMV accumulates in 8 lanes under AVX2
  (`tensor.rs:1590-1663`) and strictly sequentially in scalar (`tensor.rs:1562-1585`); the SSM scan differs the
  same way and its test only asserts a 1e-4 tolerance (`tensor.rs:1194-1281`, test `:2732-2773`). The tier is
  not in the header; `REDUKT_NO_SIMD` "must match between compress and decompress" (`main.rs:102`;
  FORMAT-SPEC.md:237-238, 478-481). `tests/roundtrip.rs:374-458` documents the divergence without asserting.
- **RWKV quantization level** (section 2.5).
- **Weights identity** (section 3.4).
- **Non-x86 targets.** The post-backbone path (CM, softmax, CDF, coder) contains no SIMD intrinsics and no libm
  call, so it should be ISA-invariant [I]; the only cross-machine evidence quoted in the repo is
  Windows <-> Linux on x86-64-v3 [D] (CLAUDE.md gate 2b). ARM is unverified (R-115 open [D]).

### 4.5 What `tests/libm_lint.rs` forbids, exactly

- Scope: every `*.rs` under `src/` (`:89-91`), except four exempt files (`domain/fmath.rs`,
  `domain/spectral.rs`, `application/telemetry.rs`, `application/benchmark.rs`; `:29-34`). A new module is
  scanned automatically.
- Only the lines **before the first** `#[cfg(test)]` that is immediately followed by `mod …` are scanned
  (`:49-57`); test helpers placed above that point are linted.
- Banned names (`:23-26`): `exp exp2 exp_m1 ln ln_1p log log2 log10 powf powi cbrt hypot sin cos tan asin acos
  atan atan2 sinh cosh tanh asinh acosh atanh`.
- Matching is **textual** (`:64-85`): any `.name(` method call on any receiver, and any `f32::name(` /
  `f64::name(`. A method called `.log(…)` or `.exp(…)` on a non-float type trips it too.
- Allowed: receiver literally named `contract` for `.exp(` / `.ln(` (`:72`); `.log2(` on a line containing
  `ce_bits` (`:73`); any line carrying a `libm-ok:` comment (`:104`); comments (`:65`).
- **Not banned**: `sqrt` (`:11`), `mul_add`, `round`, `floor`, `abs`, `clamp`, `max`.

---

## 5. Tokenization

- Trait: `encode(&[u8]) -> Vec<u32>`, `decode_token(u32) -> &[u8]`, `Sync` (`backbone.rs:188-191`).
- The **whole input is tokenized up front** into a `Vec<u32>` (`codec.rs:1087`; `lockstep.rs:542`;
  `api.rs:289`), after preprocessing (E8E9 / shuffle / delta, `codec.rs:812-922`), which is backbone-agnostic
  and recorded in FLAGS.
- Existing tokenizers:

| Tokenizer | File | Vocab | Notes |
|---|---|---|---|
| `ByteTokenizer` | `mamba/byte_tokenizer.rs:7-37` | 256, identity | `encode` = `b as u32`; `decode_token(id)` returns a 1-byte slice for `id < 256`, `&[]` otherwise. Needs no file. |
| `BpeTokenizer` | `mamba/tokenizer.rs:11-195` | from `tokenizer.json` | Adds synthetic ids for single bytes missing from the vocabulary (`:66-87`). |
| `WorldTokenizer` | `rwkv7/tokenizer.rs:10-94` | fixed 65,536 slots | Greedy longest match (`:50-75`); `decode_token` indexes unchecked (`:77-79`). |
| `PluginTokenizer` | `plugin_backend/model.rs:460-501` | 256 or BPE | Chosen from the `.rbin` header or the plugin's vocab (`:188-206`). |

- **Termination when token != byte** (v4/v5): the decoder appends `decode_token(tok)` after every symbol and
  stops when `bytes_written >= total_bytes` (`codec.rs:1621-1637`; `lockstep.rs:346-354`; `api.rs:559-562`),
  then requires equality (`codec.rs:1694-1699`; `lockstep.rs:420-424`; `api.rs:576-581`). After the final token
  the decoder stops before `cm.observe` and before the forward (`codec.rs:1634-1650`).
- Guard against zero-length tokens: `check_token_budget` (`codec.rs:515-523, 1651`) and its lockstep twin
  (`lockstep.rs:358-365`): at most `total_bytes` tokens. The API loop has no such guard (`api.rs:535-573`).
- TOTAL_BYTES is the original length, while the tokenizer sees the preprocessed bytes; the decoder compares
  decoded length against TOTAL_BYTES, so the preprocessing transforms are assumed length-preserving
  (`codec.rs:1044-1052, 1314-1315`) [I].
- `WordModel` hard-codes token ids 9, 10, 13, 32 as separators (`word_model.rs:47-51`): meaningful only when
  token id == byte value, i.e. with an identity byte tokenizer.

---

## 6. Reusable compute kernels

### 6.1 Inventory

| Kernel | Weights x activations -> accumulator | Variants | Scalar == SIMD bits | Used by |
|---|---|---|---|---|
| `gemv::f32_rows` (`gemv.rs:179`), `gemv_f32_into` (`:195`), `gemv_f32_multi` (`:211`) | f32 x f32 -> f32 | Plain: AVX2 no-FMA (`tensor.rs:1590`) / scalar (`tensor.rs:1562`). Fma: AVX2+FMA (`gemv_fma.rs:205, 245`) / scalar `mul_add` (`gemv_fma.rs:83`) | Plain: **no**. Fma: yes | Mamba x_proj, dt_proj, and in/out_proj + head under F32 (`mamba/model.rs:77, 181, 679`) |
| `gemv_q8w_into` (`gemv.rs:297`), `gemv_q8w_multi` (`:317`) | i8 (one f32 scale per 64 columns) x **f32** -> f32 | Plain: AVX2 (`gemv.rs:574`, fused 4..8 streams `:400`) / scalar (`:488`). Fma: `gemv_fma.rs:327, 374` / `:90` | yes (both) | Mamba Q8 in/out_proj, embedding / head (`mamba/model.rs:78, 88`) |
| `tensor::mat_vec_mul_into` (`tensor.rs:1500`) | f32 x f32 -> f32 | AVX2 no-FMA / scalar; spawns up to 4 scoped OS threads per call for >= 4096 rows (`:1477-1559`) | no | RWKV (LoRA, F32 matrices) |
| **`tensor::q8_mat_vec_mul_into`** (`tensor.rs:1699`) | **i8 (per-row f32 scale) x i8 (one f32 scale per call) -> i32 -> f32** | AVX-VNNI `dpbusd` (`:236-325`, needs `cols % 32 == 0`), AVX2 `maddubs` + `madd` (`:330-432`), scalar (`:1719-1747`) | yes, exact integer sums; asserted on 768x768 **[S]** (`:2058, 2091`) | **RWKV only** (`quant.rs:328`) |
| `tensor::q4_mat_vec_mul_into` (`tensor.rs:552`) | 4-bit x i8 -> i32 per group, f32 across groups **[S]** | AVX2 only for group 32 / scalar | tolerance tests only **[S]** | RWKV Q4 |
| `GgufQuantTensor::mat_vec_mul` (`quant.rs:83`) | GGUF blocks x f32 **[S]** | scalar | — | RWKV GGUF |
| `tensor::ssm_recurrence_d16` (`tensor.rs:1174`) | f32, `d_state` literal 16 | AVX2 / scalar | **no** (1e-4 tolerance test) | Mamba |
| `silu_inplace`, `gated_silu_mul` (`tensor.rs:1120, 1145`), `tanh_inplace`, `sigmoid_inplace` (`:971, 980`) | f32 element-wise | AVX2 body + portable scalar tail | tolerance tests only | Mamba / RWKV |
| `rms_norm_into`, `layer_norm_into`, `group_norm_into` (`tensor.rs:864, 1287, 1297`) | f32, sequential, `f32::sqrt` | scalar | n/a | Mamba / RWKV |

Answers to the explicit questions:

- **int8 x int8**: yes, one family (the W8A8 row above). Activations are quantized per call by
  `quantize_vec_i8_into` (`tensor.rs:1671-1680`): `scale = max|v| / 127`, `q = round(v / scale)` clamped to
  ±127 (`f32::round`, half away from zero). The result returns to f32 as `sum as f32 * (row_scale * v_scale)`
  (`tensor.rs:307, 408, 1736`). The inner kernels are private `unsafe fn`s bound to `Q8Tensor`.
- **int8 x int16**: none. A grep for 16-bit integer kernels in `src/` finds only the `madd_epi16` step above
  **[S]**.
- **Any end-to-end fixed-point path**: none. `Tensor` is `Vec<f32>` (`tensor.rs:8-11`); the Mamba Q8 profile
  converts int8 weights to f32 before multiplying (`gemv.rs:506-509`) and never quantizes activations
  (`gemv.rs:8-10`).

### 6.2 Runtime SIMD detection

| Function | Check | Env override |
|---|---|---|
| `gemv::simd_avx2()` (`gemv.rs:48-62`) | `is_x86_feature_detected!("avx2")`, cached in a `OnceLock`; `false` off x86-64 | `REDUKT_NO_SIMD=1` |
| `gemv_fma::simd_fma()` (`gemv_fma.rs:44-58`) | `simd_avx2() && is_x86_feature_detected!("fma")` | `REDUKT_NO_FMA=1` |
| `tensor::no_simd_override()` (`tensor.rs:1684-1687`) | env only; each call site adds its own `is_x86_feature_detected!` (`"avx2"`, or `"avxvnni"` first at `:1705`) | `REDUKT_NO_SIMD=1` |
| `tensor::no_prefetch_override()` (`tensor.rs:1691-1694`) | env only | `REDUKT_NO_PREFETCH=1` |

- Flags are compared with `== "1"` and read once per process. `REDUKT_NO_SIMD` has two independent caches
  (`gemv.rs:51`; `tensor.rs:1686`).
- All SIMD is behind `#[cfg(target_arch = "x86_64")]`. There is no `aarch64` / NEON / AVX-512 code: a grep for
  `target_arch` finds only `x86_64`. Any other architecture runs the scalar tier.
- Local builds use `-C target-cpu=native` (`.cargo/config.toml:2`); the Docker images override it with
  `x86-64-v3` (`deploy/gcp/Dockerfile:36-44`); CI sets `RUSTFLAGS=-D warnings`
  (`.github/workflows/ci.yml:11`), and an environment `RUSTFLAGS` takes the place of the config-file flags, so
  CI builds are not `native` [I].
- Alignment: every load is unaligned; no kernel requires aligned buffers **[S]**.

### 6.3 `par_pool` (`src/domain/par_pool.rs`)

```rust
pub const MAX_BLOCKS: usize = 0xFFFF;                                            // :29
pub fn configured_threads() -> usize                                             // :67
pub fn global() -> &'static Pool                                                 // :83
impl Pool {
    pub fn new(n_threads: usize, spin: Duration) -> Pool                         // :91
    pub fn threads(&self) -> usize                                               // :115
    pub fn run(&self, n_blocks: usize, f: &(dyn Fn(usize) + Sync))               // :123
}
pub struct SyncPtr<T>(*mut T);                                                   // :237  Send + Sync if T: Send
impl<T> SyncPtr<T> {
    pub fn new(p: *mut T) -> Self                                                // :246
    pub unsafe fn slice_mut<'a>(&self, start: usize, len: usize) -> &'a mut [T]  // :256
}
```

- `run` is the only execution primitive (no scope, no spawn, no thread index passed to the closure).
- Threads: `REDUKT_THREADS=N` (1..=256), else `available_parallelism()` (`:67-73`); the count includes the
  caller. Workers are spawned lazily once, never joined (`:83-112`).
- Workers spin for `REDUKT_SPIN_US` (default 100 µs) and then park (`:30, 75-80, 212-232`).
- Blocks are claimed one at a time by CAS on a packed word (`:176-183`), which self-balances P / E cores; the
  submitter works too and then **spin-waits** for completion (`:150-153`).
- **One job in flight.** A concurrent or nested `run` executes inline on the calling thread (`:128-136`,
  doc `:14-16`). Also inline when there are no workers, `n_blocks <= 1` or `n_blocks > 65,535` (`:124-127`).
- Panics in a block are caught and re-raised on the submitter (`:155-160, 189-192`).
- Generic over the work: nothing in it is f32-specific. In contrast `gemv::par_rows` / `par_rows_multi` are
  private and typed for `&mut [f32]` outputs (`gemv.rs:107, 141`).
- Granularity constants in `gemv.rs`: matrices under `PAR_MIN_BYTES = 128 KiB`, or with fewer than 8 rows, run
  on the calling thread (`:41, 113, 150`); blocks hold at most 64 KiB of weights (`:43`) and aim at 8 blocks
  per thread (`:45`). Mamba's per-channel phases use `CONV_BLOCK = 256` and `SSM_BLOCK = 64` channels
  (`mamba/model.rs:124-126`).

### 6.4 How lockstep shares weight reads between streams

- Engine: per step, Phase A runs CM + coder for every active stream as one pool unit per stream
  (`lockstep.rs:205-216`); Phase B makes **one** `forward_multi` call for all active streams (`:227-246`).
  Finished streams are replaced from the source (continuous batching, `:182-199`).
- Mamba's `forward_multi` turns every large GEMV into an N-stream GEMM (`mamba/model.rs:826-860, 920-924`):
  `par_rows_multi` partitions by rows and each row block is applied to all N activation vectors while it is in
  cache (`gemv.rs:134-165`). F32 walks 4-row sub-blocks per stream (`gemv.rs:132, 222-232`); Q8 uses a fused
  kernel for 4..8 streams that converts each int8 chunk once (`gemv.rs:338-437`).
- Per-channel phases become `(stream, channel block)` pool units (`mamba/model.rs:862-918`).
- `forward_multi` falls back to N sequential forwards for `n == 1` (`mamba/model.rs:778-783`).
- Memory report: `stream_bytes()` + `stream_bytes_estimate` (`lockstep.rs:152-156, 578-587`), which assumes ten
  vocab-sized f32 buffers per stream plus a 256 KiB match table.

---

## 7. Hardware probes that already exist (zero dependencies)

| Information | How | Where | Platforms |
|---|---|---|---|
| Logical CPU count | `std::thread::available_parallelism()` | `par_pool.rs:71`; `tensor.rs:1486`; `benchmark.rs:651` | all |
| x86 features | `is_x86_feature_detected!` for `avx2`, `fma`, `avxvnni` | section 6.2 | x86-64 |
| Physical RAM (total / available / % used) | `GlobalMemoryStatusEx` via `extern "system"` | `ram_monitor.rs:26-76` | **Windows only** (`None` elsewhere, `:72-75`) |
| AC / battery | `GetSystemPowerStatus` | `governor.rs:50-86` | **Windows only** |
| CPU model string | `wmic cpu get Name` subprocess, else `/proc/cpuinfo` `model name` | `benchmark.rs:567-601` | `benchmark` command only |
| OS / arch | `std::env::consts::{OS, ARCH}` | `benchmark.rs:676, 680` | all |

- **Not discovered anywhere**: cache sizes, physical cores, P / E core topology, frequency, NUMA, memory
  bandwidth, thread affinity (a grep for `cpuid`, `GetLogicalProcessor`, `/sys/devices`, `affinity` in `src/`
  returns nothing).
- `HardwareGovernor` (`governor.rs:89-209`): `preflight()` rejects RAM > 95 % or battery < 5 %; a watchdog
  thread polls every 2 s; `apply()` is called every 100 tokens / steps (`codec.rs:1178-1180, 1245-1247`;
  `lockstep.rs:250-254`) and either sleeps 2 ms (Eco) or returns an error (Abort). It changes timing only,
  never the bitstream. On non-Windows hosts both probes return `None`, so it is inert.
- `check_ram_or_abort` terminates the process with `exit(1)` above the threshold (`ram_monitor.rs:80-93`).
- `telemetry.rs` contains no hardware probe (progress tracker, JSONL loggers, `PhaseTimer`).

---

## 8. `redukt api` versus the CLI

| Aspect | CLI | API | Evidence |
|---|---|---|---|
| Models | Loaded per invocation (per model group in batch mode) | Loaded once at start-up; one immutable `(DynModel, DynTokenizer)` per backbone | `api.rs:178-200` |
| Backbone choice | `--backbone` or auto route per file | `--backbone` may be a comma list; one explicit name forces it, otherwise routed per request | `api.rs:86-103, 286` |
| Compress engine | `forward_into` loop (or Mamba pipeline) | `run_lockstep` with one slot -> `forward_multi(N = 1)` | `api.rs:305-318` |
| Decompress engine | `cmd_decompress` loop, or lockstep for directories | Own generic loop `decompress_bytes<M, T>` (also used by `benchmark --verify`) | `api.rs:477-592`; `benchmark.rs:407` |
| Header builder | `build_header_with(model.write_contract())` | `build_header` (process contract, never set by the API -> contract 1) | `codec.rs:1317`; `lockstep.rs:404`; FORMAT-SPEC.md:268-269 |
| Flags | all of section 2.1 | only `--bind/--host`, `--port`, `--weights`, `--backbone`, `--quant`; unknown flags are errors | `api.rs:56-82` |
| Preprocessing, CM hyperparameters, reset | flags | fixed: auto preprocessing, default hyperparameters, `default_reset_log2(choice)` | `api.rs:285-304` |
| Numeric contract on decode | `model.set_numeric_contract(hdr.numeric_contract)` | **not applied** (model is `&DynModel`); a contract-2 archive is decoded with Plain kernels and ends in a decode / CRC error [I] | `codec.rs:1502`; `api.rs:333-358` |
| Archive / model match | header decides which model to load | 422 if the backbone or weight profile is not loaded; key is `(choice, profile)` only | `api.rs:140-155` |
| Embedding surgery | applied from the header | rejected (422) | `api.rs:342-346` |
| Entropy-gate archives | supported (single-file) | supported in `decompress_bytes`; never produced | `api.rs:543-552` |
| Zero-length-token guard | yes | no | section 5 |
| Concurrency | — | single-threaded accept loop, one request at a time | `api.rs:206-211` |
| Limits | 4 GiB (u32) | `MAX_BODY` = 256 MiB | `api.rs:38, 337-341` |
| Status endpoint | — | `GET /v1/status` lists backbone, profile, vocab; reports `format_write` "v4" for F32 models although the encoder writes v5 | `api.rs:161-175` |

Places in `api.rs` that enumerate backbones or the Mamba family: `parse_backbones` error text (`:95`),
`required_profile` (`:321-330`), `status_json` (`:161-175`), `LoadedInfo` / `engine_for` (`:108-155`).

---

## 9. Tests, build profiles, hook

### 9.1 Where the tests are

| Location | Count (static grep of `#[test]`) | Needs |
|---|---|---|
| In-crate unit tests (`#[cfg(test)] mod tests` at the end of 34 files under `src/`) | 310, of which 6 `#[ignore]` (benches / real weights) -> **304 run by default** | nothing |
| `tests/libm_lint.rs` | 3 | source tree only |
| `tests/roundtrip.rs` | 17: **5 run by default** (header / empty-input rejection), 12 `#[ignore]` | built binary; ignored ones need weights |
| `tests/api_parity.rs` | 1, `#[ignore]` | built binary + `weights/mamba-130m` |

This matches CLAUDE.md ("304 unit + 3 lint + 5 integration").

### 9.2 How they are structured

- **Binary-only crate.** There is no `src/lib.rs` (`src/main.rs:1-3` declares the modules), so files under
  `tests/` cannot import any crate item. They locate `target/<profile>/redukt[.exe]` from the test executable's
  path and drive it with `std::process::Command` (`tests/roundtrip.rs:9-50`; `tests/api_parity.rs:12-20`).
- Weight-dependent tests skip themselves when the files are absent (`tests/roundtrip.rs:20-22, 502-504`) and
  are `#[ignore]` on top of that.
- **Roundtrip** patterns: CLI compress -> decompress -> compare (`tests/roundtrip.rs:56-87`); thread-count
  invariance (`:558-580`); lockstep batch == single file, both directions (`:589-636`); contract 2 (`:646-703`).
- **Golden**: one test hashes the archive of a 40-byte input (SHA-256 implemented in the test file,
  `:709-788`) and compares it with a constant recorded for "Windows x86_64, AVX2, `--quant f32`"
  (`:257-313`). There are no golden `.azth` files in the tree, and `*.azth` is gitignored globally
  (`.gitignore:2`), as are `*.log` and `*.jsonl` (`.gitignore:9-10`).
- **Weight-free end-to-end unit tests** (templates): container roundtrip through a tiny Q8 "model"
  (`codec.rs:3108-3165`); lockstep engine with `MockBackbone` (`lockstep.rs:851-903`); model-level
  multi == single on a synthetic Mamba (`mamba/model.rs:1085-1128`); header build / parse tables
  (`codec.rs:2816-3058`).
- **Numeric goldens**: `fmath` bit patterns and sweep hash (`fmath.rs:525-669`) **[S]**.
- **Lint**: section 4.5.

### 9.3 How they are run

- `cargo test` — unit + lint + the 5 default integration tests (CI: `.github/workflows/ci.yml:26-27`, on
  ubuntu / windows / macos runners with `RUSTFLAGS=-D warnings`).
- `cargo test --test roundtrip -- --ignored` (weights; "run with --release for speed",
  `tests/roundtrip.rs:4, 517`), `cargo test --release --test api_parity -- --ignored`
  (`tests/api_parity.rs:3-4`), `cargo test --test libm_lint` (`tests/libm_lint.rs:17`).
- Ignored unit benches: e.g. `cargo test --release bench_forward_multi -- --ignored --nocapture`
  (`mamba/model.rs:1130-1134`).
- Quality gates before a commit are listed in CLAUDE.md [D].

### 9.4 Build profiles

- `Cargo.toml:11-20`: `release` = opt-level 3, fat LTO, 1 codegen unit, `panic = "unwind"`; `maxperf` =
  release + `panic = "abort"` + strip. No dependencies (`Cargo.toml:9`).
- `.cargo/config.toml:2`: `rustflags = ["-C", "target-cpu=native"]` for every local build, tests included.
- With `panic = "abort"` any panic on the forward path (index out of range, plugin failure, pool block)
  terminates the process, batch included.

### 9.5 `.claude/hooks/cargo-check.sh`

- Registered as a `PostToolUse` hook for `Edit|Write`, timeout 240 s (`.claude/settings.json:15-26`).
- Reads the hook JSON on stdin with `python`, acts only when the edited path ends in `.rs` (`:7-21`).
- Skips when any `cargo.exe` is already running (`tasklist`, `:22`).
- Runs `cargo check --all-targets --message-format short`, keeps lines matching
  `^(src|tests|plugins)…: (warning|error)`, first 20 (`:24-25`), and returns them as `additionalContext`
  (`:26-32`). It always exits 0: it informs, it never blocks.
- Consequence [I]: a new module that is not yet referenced from `main` produces dead-code warnings on every
  edit until it is wired in (or marked `#[allow(dead_code)]`, as several existing files do for single items,
  e.g. `src/domain/mod.rs:1`, `mamba/byte_tokenizer.rs:21`).

---

## 10. Plugin ABI (FFI v1 / v2) in brief

Contract: `docs/api/PLUGIN-FFI-V2.md`; types in `backend_ffi.rs`.

| Symbol | C signature | Lines |
|---|---|---|
| `redukt_backend_init` | `(const u8* weights_path, i32 device_id, BackendInfo* info_out) -> BackendResult` | `backend_ffi.rs:91-95, 127` |
| `redukt_backend_forward` | `(u32 token, f32* out_logits, u32 vocab_size) -> BackendResult` | `:102-106, 128` |
| `redukt_backend_reset_state` | `()` | `:110, 129` |
| `redukt_backend_free` | `()` | `:114, 130` |
| `redukt_backend_abi_version` (v2) | `() -> u32`, must be >= 2 | `:137, 174` |
| `redukt_backend_batch_init` (v2) | `(u32 max_streams) -> i32` | `:142, 175` |
| `redukt_backend_batch_reset` (v2) | `(const u32* slots, u32 n_slots) -> i32` | `:146, 176` |
| `redukt_backend_forward_batch` (v2) | `(u32 n_streams, const u32* slots, const u32* tokens, f32* out_logits, u32 vocab_size) -> i32` | `:156-162, 177` |

- `BackendInfo { ffi_version: u32 (must be 1), vocab_size: u32, hidden_dim: u32, name: [u8; 64] }`
  (`backend_ffi.rs:61-72`). Result codes 0..5 (`:25-41`).
- v2 is additive: it is enabled only if all four v2 symbols resolve and `abi_version() >= 2`
  (`backend_ffi.rs:7-10`; PLUGIN-FFI-V2.md:18-23).
- Logits are caller-owned `f32`: `vocab_size` values in v1, `n_streams * vocab_size` row-major in v2
  (`backend_ffi.rs:97-100, 151-152`). The host then copies each row into the stream's buffer
  (`plugin_backend/model.rs:431-433`).
- State is a `u32` slot index into a device-side pool; the host never holds device pointers
  (PLUGIN-FFI-V2.md:55-66). `prepare_streams(n)` -> `batch_init(n)`; each `create_state` takes a slot
  (`plugin_backend/model.rs:370-373, 438-451`).
- The periodic reset is driven by the host: `PluginState::tick()` counts forwards and calls `reset()`
  (`plugin_backend/model.rs:44-49, 60-75`).

What an out-of-core architecture has to provide today, and what the host fixes for it:

- A shared library with **one fixed file name next to the executable** (`redukt-gpu.dll`, `libredukt_gpu.so`,
  `libredukt_gpu.dylib`: `plugin_backend/loader.rs:116-121, 143-144`). One plugin per installation, one model
  per process. If it is missing, `load_dyn_model` panics; there is no CPU fallback (`model_loader.rs:436-437`).
- The four v1 symbols (single stream); the four v2 symbols for lockstep with more than one stream
  (`plugin_backend/model.rs:438-451`).
- Bit-reproducible f32 logits with v1 == v2 for any batch (`plugin_backend/model.rs:383`;
  PLUGIN-FFI-V2.md:97-100).
- Its own weight loading. The host only peeks at a 44-byte `RGPU` header at `<weights>/model.rbin` to choose
  the tokenizer (`plugin_backend/model.rs:114-177`).
- A tokenizer the host already has: **identity bytes (vocab exactly 256) or the Mamba GPT-NeoX BPE**
  (`plugin_backend/model.rs:188-206`). The ABI has no tokenizer hook.
- The host hard-codes: BACKBONE id 3 for every plugin, QUANT = F32, contract 1, and the Mamba reset period
  4096 by default (`backbone.rs:25`; `codec.rs:139-141, 606, 662-665`). **The archive does not say which
  plugin, architecture or weights produced it.**
- The ABI has no channel for a per-archive configuration (tier, contract, reset period) other than what the
  host does around the calls [I].
- Plugin archives are decodable only with the same plugin build on the same GPU architecture [D]
  (PLUGIN-FFI-V2.md:102-108).

---

## 11. Minimal change list for a third in-core backbone (ordered by dependency)

Each item is a place that enumerates backbones today. "Exhaustive" means the compiler will force the edit.

**Step 1 — identity (`src/domain/backbone.rs`)**

| Point | Lines | Note |
|---|---|---|
| `enum BackboneChoice` | 11-16 | new variant |
| `header_byte` | 20-27 | exhaustive; next free id is 4 |
| `from_header_byte` | 30-38 | not exhaustive (`_ => None`) — easy to forget |
| `from_str_opt` | 42-50 | not exhaustive — forgetting it makes `--backbone <new>` mean Mamba |
| `Display` | 53-62 | exhaustive |
| `WeightProfile` | 68-115 | only if the backbone reuses PROFILE.QUANT |
| `route_backbone` | 121-123 | unchanged unless the default route changes (it changes default output bytes) |

**Step 2 — the backbone itself (`src/infrastructure/<new>/`)**

| Point | Lines | Note |
|---|---|---|
| module registration | `src/infrastructure/mod.rs:1-4` | |
| `impl Backbone / BackboneState / BackboneScratch` | pattern `mamba/model.rs:983-1023`, minimal `rwkv7/model.rs:1399-1423` | |
| tokenizer | reuse `mamba/byte_tokenizer.rs:7-37` or a new `impl BackendTokenizer` | cross-module import is already the norm (`mamba/model.rs:20`) |
| loader | `rwkv7/safetensors.rs` as-is yields f32 tensors only (section 2.4) | |
| new domain kernels, if any | `src/domain/mod.rs:1-27` | scanned by the libm lint automatically |

**Step 3 — registration (`src/application/model_loader.rs`)**

| Point | Lines | Note |
|---|---|---|
| imports, weights-dir constant | 15-22, 41-45 | |
| `DynModel`, `DynState`, `DynScratch`, `DynTokenizer` | 81-110 | new variants |
| `impl BackboneState for DynState` | 112-129 | exhaustive (`reset`, `set_reset_every`) |
| `impl BackboneScratch for DynScratch` | 131-140 | exhaustive |
| `impl Backbone for DynModel` | 142-255 | exhaustive: `vocab_size`, `create_state`, `create_scratch`, `stream_bytes`, `prepare_streams`, `forward_multi` (`:201-224`); `forward_into` needs a tuple arm (`:177-191`, otherwise it panics at run time) |
| `impl BackendTokenizer for DynTokenizer` | 257-275 | exhaustive if a tokenizer variant is added |
| `hidden_dim`, `weight_profile`, `set_numeric_contract`, `write_contract` | 284-291, 340-345, 350-361, 364-369 | exhaustive |
| `load_dyn_model` | 402-441 | exhaustive; where the model and tokenizer are built |
| `resolve_auto_profile` | 463-468 | only if profile-aware |

**Step 4 — header and CLI (`src/application/codec.rs`)** — nothing here is compiler-enforced (section 3.5).

| Point | Lines | Note |
|---|---|---|
| `HeaderInfo` + its two constructors | 99-117; 249-262, 415-422 | new field for backbone-specific configuration |
| `parse_header`, backbone block | 397-410 | where extra bytes after the id would be read |
| `build_v4_header` | 534-591 (`:545, 563, 587-589`) | where they would be written; signature has 11 positional parameters |
| `build_header`, `build_header_with` | 600-613, 629-642 | 13-14 positional parameters; production call sites `codec.rs:1317` and `lockstep.rs:404`, plus a dozen unit-test call sites (`codec.rs:2817-3144`, `api.rs:624`) |
| reset semantics | 134-141, 646-656 | `resets_with_length` |
| profile / contract semantics | 431-445, 606, 662-669, 673-694 | "Mamba family" predicates |
| `cmd_compress` | flags 950-984; routing 1055-1062; profile 1064-1071; load 1074-1080; state 1096-1098; loop choice 1123; header 1317-1321; report 1346 | |
| `cmd_decompress` | 1486-1505; state 1515-1516 | |
| unit tests that pin the id table | 2722-2741 (`:2727` asserts id 4 is unknown) | |

**Step 5 — batch (`src/application/lockstep.rs`)**

| Point | Lines | Note |
|---|---|---|
| `GroupKey` (+ `ord_tuple`, `group_by_key` key type) | 499-519 | if two archives of the same backbone can need different loaded models |
| `compress_key`, `decompress_key` | 522-528, 704-721 | |
| `EncodeMeta` + constructors | 103-117; 543-556; `api.rs:290-304`; tests 831-837 | |
| `encode_output` | 401-413 | header call |
| `load_group_model` | 561-589 | |

**Step 6 — API (`src/application/api.rs`)**: `parse_backbones` text (`:95`), `LoadedInfo` / `engine_for`
(`:108-155`), `status_json` (`:161-175`), `compress_request` (`:278-306`), `required_profile` (`:321-330`),
`check_archive` (`:333-349`).

**Step 7 — usage text (`src/main.rs`)**: `--quant` (`:72-77`), `--backbone` (`:78-79`), `--reset-every`
(`:93-96`), environment (`:99-102`).

**Step 8 — tests**: in-crate unit tests in the new module; container and lockstep templates (section 9.2);
CLI roundtrips in `tests/roundtrip.rs` gated on weights (pattern `:502-555, 589-636`).

**Step 9 — docs and infrastructure**: `docs/api/FORMAT-SPEC.md` (id table `:121-127`, PROFILE `:193-198`,
history `:315-320`), CLAUDE.md, README.md, `weights/MANIFEST.md`, `.gitignore:2-8`, `.dockerignore`,
`deploy/gcp/entrypoint.sh:51-77` (weight sets per backbone), NOTICE if third-party code or weights are added.

**Not required, but backbone-specific today** (they will not see a new backbone): `hybrid-eval`, `baseline`,
`cm-eval`, `benchmark` are RWKV-only (`codec.rs:1817, 1955, 2056`; `benchmark.rs:148, 364-366`); `mamba-eval`
is Mamba-only (`codec.rs:2061-2197`).

---

## 12. Traps and debt that affect the integration

**Interface and types**

1. f32 logits are the only interface (`backbone.rs:155-158`); `Tensor`, the SafeTensors loader and the FFI are
   f32-only too (`tensor.rs:8-11`; `rwkv7/safetensors.rs:70-90`; `backend_ffi.rs:102-106`).
2. No error path out of `forward_into`; with `maxperf` (`panic = "abort"`) a panic kills the whole batch.
3. Model loading cannot fail gracefully: `load_dyn_model` returns a tuple (`model_loader.rs:402-407`).
4. Token ids are unchecked against `vocab_size()` (section 1.2).
5. `Cdf::from_probs` asserts `vocab < 2^24` and guarantees frequency >= 1 per symbol, so the floor probability
   is 2^-24 and V symbols always consume V / 2^24 of the range (`coder.rs:12, 43-50`).

**Vocabulary-size assumptions**

6. Host work and allocations per token are O(V) (section 4.2), tuned and measured only at V = 50,280 / 65,536.
7. RWKV hard-codes 65,536 (`rwkv7/model.rs:24-38`; `rwkv7/tokenizer.rs:20-24`); diagnostics print
   "12 layers" / "65536x768" literally (`codec.rs:1301-1304`); API tests use 50,280 (`api.rs:605`).
8. `WordModel` separators are byte values used as token ids (`word_model.rs:47-51`); n-gram smoothing depends
   on V (`ngram.rs:121-122`); `stream_bytes_estimate` assumes ten V-sized f32 buffers (`lockstep.rs:152-156`).
9. `ssm_recurrence_d16` and `MambaConfig` fix `d_state = 16`, `d_conv = 4` (`tensor.rs:1204-1215`;
   `mamba/model.rs:39-40`).

**Header and "Mamba family" coupling**

10. Profile, FMA contract and reset period are defined only for the Mamba family (fact 7, section 3.5).
11. `--backbone` typos fall back to Mamba silently; unknown CLI flags are ignored (section 2.1).
12. `build_header` (lockstep, API) and `build_header_with` (single-file) derive the contract differently
    (section 3.1): a backbone whose single-file and lockstep headers must be byte-identical has to make both
    agree.
13. The numeric contract is process-global in batch mode (`codec.rs:1411-1412`) and the API never switches it
    (section 8): any per-archive setting that changes how the model computes must either be part of `GroupKey`
    / the API engine key or travel through a per-state hook.
14. `embedding_surgery` is RWKV-only, yet `--emb-surgery` is written to the header for any backbone
    (`codec.rs:1082-1084, 1317-1321`) and changes the lockstep group key (`lockstep.rs:499-503`).

**Determinism**

15. Existing F32 kernels are SIMD-tier-bound and the tier is not recorded (section 4.4).
16. RWKV `--quant auto` is chosen by timing and not recorded (section 2.5).
17. `-C target-cpu=native` in local builds (`.cargo/config.toml:2`): binaries are not portable across CPUs, and
    `is_x86_feature_detected!` may be resolved at compile time [I].
18. `f32::mul_add` compiles to a hardware FMA only when the target has it; otherwise it calls the platform
    `fmaf` (`gemv_fma.rs:14-16`). The lint does not flag `mul_add`.
19. `quantize` uses `f32::round` and `f32::abs` at load time (`gemv.rs:262-266`): part of the Q8 contract.

**Kernels and threading**

20. Kernel entry points are typed for f32 outputs; `par_rows*` are private (`gemv.rs:107, 141`). Only
    `par_pool::global().run` and `SyncPtr` are generic and public.
21. The `gemv_*_with(pool, …)` variants are private, so library callers always use the global pool
    (`gemv.rs:199, 215, 301, 321`).
22. One pool job at a time; nested calls run inline; the submitter spin-waits (section 6.3). The threading
    thresholds (128 KiB per matrix, 64 KiB per block) sit next to Mamba-130M matrices of roughly 0.3-154 MB in
    F32 (shapes at `mamba/model.rs:24-32, 236-244`) [I]; a matrix under 128 KiB never leaves the calling thread.
23. `f32_rows` on the Plain AVX2 path forwards to a raw-pointer kernel without a length check
    (`gemv.rs:184-190`) **[S]**; the fused Q8 kernels read through raw pointers (`gemv.rs:400-437`), guarded
    only by `check_multi_shapes` (`gemv.rs:170-173`).
24. Legacy `mat_vec_mul_into` spawns OS threads per call above 4096 rows and ignores `REDUKT_THREADS`
    (`tensor.rs:1477-1559`).
25. No NEON: on ARM everything is the scalar tier.

**Loaders and files**

26. SafeTensors: f32 output only, no `__metadata__`, 2 GiB cap, whole file in RAM plus a copy per tensor,
    unchecked offsets (section 2.4).
27. Weights are found relative to the CWD; `--weights` is overloaded as a fallback for every backbone
    (section 2.3).
28. `.gitignore` ignores `*.azth`, `*.log`, `*.jsonl` everywhere and only five extensions under `weights/`:
    committed golden archives or a new weight extension need explicit rules (`.gitignore:2-10`).

**Tests and tooling**

29. Binary-only crate: no integration test can call internal functions (section 9.2).
30. Dead code is an error in CI; the edit hook reports it on every save (section 9.5).
31. The libm lint is textual and scans everything before the first `#[cfg(test)] mod` (section 4.5).
32. The golden hash in `tests/roundtrip.rs:257-313` is platform- and tier-specific by its own comment.

**Engines**

33. Lockstep does not implement the entropy gate; single-file loops pay for `cm_only_entropy` on every token
    even with the gate off (section 4.3).
34. The API decode loop has no zero-length-token budget and does not apply the header contract to the model
    (sections 5, 8).
35. `MAX_SLOTS` = 256 streams (`lockstep.rs:48`); TOTAL_BYTES u32; `MatchModel` positions are `u32`
    (`match_model.rs:15-17, 72`); n-gram tables freeze at 500,000 contexts per order (`ngram.rs:16, 63-68`).

---

## 13. Discrepancies between documents and code

| # | Document says | Code says | Evidence |
|---|---|---|---|
| D1 | CLAUDE.md: `[BACKBONE] … 0=RWKV, 1=Mamba`. | Four ids: 0 RWKV, 1 Mamba, 2 MambaByte, 3 Plugin (FORMAT-SPEC.md agrees with the code). | `backbone.rs:20-38`; FORMAT-SPEC.md:121-127 |
| D2 | CLAUDE.md FLAGS list ends at `0x40=backbone`. | `0x80` = ENHANCED_CM is defined and always written. | `codec.rs:51, 1320` |
| D3 | CLAUDE.md and FORMAT-SPEC.md:182: RESET_LOG2 is "never" written for RWKV / plugin. | The plugin gets the 4096 default, like the Mamba family; a unit test pins it. | `codec.rs:139-141, 2873` |
| D4 | CLAUDE.md Portability Principle: the CPU is "always able to decode anything produced anywhere". | F32-path archives are bound to the encoder's SIMD tier, which is not in the header; a CPU without AVX2 (or any non-x86 CPU) cannot decode AVX2-produced archives. FORMAT-SPEC.md states the limitation. | `tensor.rs:1562-1663`; `tests/roundtrip.rs:374-458`; FORMAT-SPEC.md:237-238, 478-481 |
| D5 | FORMAT-SPEC.md:331-333: "A timing-based auto-selection of the weight profile is never used". | True for the Mamba profile. RWKV `--quant auto` is a load-time timing benchmark and its result is not in the header. | `rwkv7/model.rs:296-308`; `quant.rs:477-531` |
| D6 | `tests/roundtrip.rs:539` expects PROFILE `0x11`; `:662, 672-673` expect `0x21` / `0x20` for default Mamba archives. | Since R-108 the default Mamba PROFILE carries bit 6: `0x51`, `0x61`, `0x60` (FORMAT-SPEC.md:200-201 agrees with the code). Both tests are `#[ignore]` (weights), so default `cargo test` does not see it. **Not executed here.** | `codec.rs:134-141, 617-625` |
| D7 | `tests/roundtrip.rs:557` doc comment: "F32 Mamba archives stay AZTH v4". | The same test asserts v5 (`:576`); the encoder always writes v5. | `codec.rs:617-618` |
| D8 | CLAUDE.md: "Commands: compress, decompress, hybrid-eval, baseline, cm-eval, api". | `benchmark` and `mamba-eval` are also dispatched. | `main.rs:13-21` |
| D9 | `--help` text. | Omits `--numeric`, `--adaptive-skip`, `--skip-threshold`, `plugin` as a backbone, `REDUKT_NO_FMA`, `REDUKT_NO_PREFETCH`. | `main.rs:37-103`; `codec.rs:971-980`; `gemv_fma.rs:47`; `tensor.rs:1693` |
| D10 | `api.rs:164`: `format_write` = "v4" for F32 models. | Every archive is v5. | `codec.rs:617-618` |
| D11 | `model_loader.rs:45`: default `weights/mambabyte-96m`. | Only `weights/mambabyte-353m/` exists on disk; the deploy script also defaults to `mambabyte-353m`; it has no MANIFEST entry. | `deploy/gcp/entrypoint.sh:75`; `weights/MANIFEST.md:7-78` |
| D12 | CLAUDE.md: "Hybrid CM (5 models + adaptive SGD mixer)". | Six predictors; five are weighted by the mixer, the segment model is added unweighted. | `cm_ensemble.rs:17-24, 87-90`; `mixer.rs:19` |
| D13 | `docs/ARCHITECTURE.md:169`: `backbone.rs` holds "ByteBackbone trait, RwkvBackbone, orchestrator". | No such items; the file holds `BackboneChoice`, `WeightProfile` and the four traits. | `backbone.rs:1-191` |
| D14 | PLUGIN-FFI-V2.md:147-159: "the application layer does not use it yet"; lists `PluginBackend::batch_reset` and `&mut self` receivers; status line says GPU not validated. | Lockstep calls `prepare_streams`; there is no `batch_reset` method (reset goes through `PluginState::reset`); receivers are `&self`. CLAUDE.md says the plugin was validated on a T4 (doc vs doc). | `lockstep.rs:170`; `plugin_backend/model.rs:60-75, 310-354` |
| D15 | `plugins/redukt-gpu/README.md:46-52, 72`: `--backend plugin`, "If no plugin is found, Redukt falls back to CPU automatically". | The flag is `--backbone plugin`; a missing plugin panics in `load_dyn_model`, there is no fallback. | `backbone.rs:47`; `model_loader.rs:436-437` |

---

## 14. Method and coverage

- Read in full by the author: `backbone.rs`, `main.rs`, `model_loader.rs`, `lockstep.rs`, `mamba/model.rs`,
  `mamba/byte_tokenizer.rs`, `cm_ensemble.rs`, `mixer.rs`, `bias_head.rs`, `coder.rs` (non-test), `error.rs`,
  `par_pool.rs` (non-test), `governor.rs`, `ram_monitor.rs`, `backend_ffi.rs` (non-test), `tests/libm_lint.rs`,
  `tests/api_parity.rs`, `docs/api/FORMAT-SPEC.md`, `docs/api/PLUGIN-FFI-V2.md`, the hook and CI files.
- Read in the parts cited: `codec.rs` (lines 1-1773, 2036-2197, 3105-3166), `api.rs` (1-650), `gemv.rs`
  (1-603), `gemv_fma.rs` (1-172), `tensor.rs` (kernels cited), `quant.rs` (405-535), `fmath.rs` (1-109),
  `mamba/tokenizer.rs` (1-200), `rwkv7/model.rs` and `rwkv7/safetensors.rs` (cited ranges),
  `plugin_backend/model.rs` (24-173, 244-503), `tests/roundtrip.rs` (cited ranges), `benchmark.rs` (540-714),
  `ngram.rs` (1-150), `word_model.rs` (1-125), `match_model.rs` (1-100), `segment_model.rs` and
  `run_model.rs` (non-test).
- Delegated full reads, cross-checked on anchor lines: `gemv.rs`, `gemv_fma.rs`, `quant.rs`, `tensor.rs`,
  `fmath.rs`, `par_pool.rs` (including tests); `rwkv7/*`, `plugin_backend/*`, the CUDA plugin source and the
  conversion scripts.
- Mechanical checks on this file: all 570 explicit `file:line` references are within range of their files, and
  235 anchor lines were matched against an expected substring of the source. Range checks do not prove that a
  reference says what the sentence claims; that rests on the reading above.
- Not read: `spectral.rs`, `preprocess.rs`, `byte_shuffle.rs`, `delta.rs`, `record_detect.rs`,
  `content_detect.rs` (structure only), `telemetry.rs` (structure only), `cmd_hybrid_eval` / `cmd_baseline`
  bodies.
