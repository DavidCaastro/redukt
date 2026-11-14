# Architecture Snapshot — Redukt

> Living document. Update when modifying `rwkv7/`, `quant.rs`, `backbone.rs`, or `bridge.rs`.


---

## 1. Backbone Taxonomy

| | 0.1B World | 1.5B G1k | 2.9B G1k |
|---|---|---|---|
| Variant | non-G1k (`g1k=false`) | G1k (`g1k=true`) | G1k (`g1k=true`) |
| Source format | SafeTensors | GGUF v3 | GGUF v3 |
| Key format | BlinkDL (`blocks.N.att.*`) | llama.cpp (`blk.N.attn_*`) | llama.cpp (`blk.N.attn_*`) |
| Layers | 12 | 24 | 32 |
| D (n_embd) | 768 | 2048 | 2560 |
| Heads (H) | 12 | 32 | 40 |
| Head size (N) | 64 | 64 | 64 |
| LoRA rank | 32 | 64 | 64 |
| FFN width | ~3×D (~2304) | ~3.5×D (~7168) | ~3.5×D (~8960) |
| Vocab | 65536 (World tokenizer) | 65536 | 65536 |

Source: `model.rs` (Rwkv7Config)

## 2. Storage Morphology

| | 0.1B ST | 1.5B Q8_0 | 1.5B Q4_K_M | 2.9B Q4_K_M |
|---|---|---|---|---|
| File | `model.safetensors` | `.gguf` | `.gguf` | `.gguf` |
| Disk | ~400 MB | ~1.5 GB | ~950 MB | ~1.8 GB |
| Layer quant | Auto Q8 (per-row) | Q8_0 native (32-block) | Q4_K native (256-block) | Q4_K native (256-block) |
| Head quant | Auto Q8 | Q6_K native | Q6_K native | Q6_K native |
| RAM loaded | ~130 MB | ~1.5 GB | ~753 MB | ~1.5 GB |
| RAM peak | ~1.2 GB | ~16.8 GB | ~15.7 GB | ~17.7 GB |
| Dim order | PyTorch row-major | GGUF ne[] reversed | GGUF ne[] reversed | GGUF ne[] reversed |
| LoRA transpose | Yes (in loader) | No (dims correct) | No (dims correct) | No (dims correct) |

Source: `model.rs` (load, load_gguf, load_auto)

## 3. Load Pipeline

```
weights path
  │
  ├─ .gguf? ──→ load_gguf()
  │               ├─ GgufFile::open() → parse header
  │               ├─ detect_config() from GGUF metadata
  │               ├─ Key detect: token_embd→llama, time_mix_w1→llama_v2, model.embeddings→HF, blocks→blink
  │               ├─ Embedding: get_tensor + LayerNorm pre-bake (V×D)
  │               ├─ Native quant? ggml_block_info() → bypass auto-detect
  │               ├─ Head: load_gguf_matrix() (Q6_K native or F32 fallback)
  │               ├─ Layers: llama_v2 | llama | hf | blink loader × N
  │               └─ RAM check every 4 layers (abort at 95%)
  │
  └─ dir/ST? ──→ load()
                  ├─ SafeTensorsFile::open()
                  ├─ Key detect: model.embeddings→HF, else BlinkDL
                  ├─ Embedding: get_tensor + LayerNorm pre-bake (V×D)
                  ├─ Quant policy: --quant, else fixed Q8 (Rwkv7Model::AUTO_QUANT; never timed)
                  ├─ Head: quantize_matrix(Q8)
                  └─ Layers: hf | blink loader × N (with transpose)
```

Layer loaders (6 total): `model.rs,946,1013,1085,1155,1213`

## 4. Forward Pass — 12 Steps

Per-token, zero-allocation. Source: `model.rs`

```
token → emb[token] ──→ x (D-dim)
  │
  for layer in 0..N_LAYER:
  │
  ├─ LN1(x) → ln_buf
  ├─ TIME MIXING (steps 1-12):
  │   1. Token shift: x - x_prev → 6 lerp mixes (xr,xw,xk,xv,xa,xg)
  │   2. Linear projections: R,K,V = Wr·xr, Wk·xk, Wv·xv     [QuantMatrix]
  │   3. Decay LoRA: w1·tanh(xw) → w2 + w0 → exp(-0.606531·σ(val))
  │   4. A-gate LoRA: a1·xa → a2 + a0 → sigmoid                [no tanh]
  │   5. G-gate LoRA: g1·σ(xg) → g2
  │   6. Key norm: k·k_k → L2 normalize per head
  │   7. Key mod: k_mod = k·(1 + (a-1)·k_a)
  │   8. V-first mix: v + (v_first - v)·σ(v0 + lora)           [skip layer 0]
  │   9. State update: S = S·w + S@(-kk⊗kk_a) + v_out⊗k_mod   [H × N×N matrices]
  │  10. GroupNorm: per-head normalize (eps=64e-5)
  │  11. Bonus: r·k_src·r_k → scale v_out                      [DIVERGES: see §5]
  │  12. Gate + output: (out·g) @ Wo → tm_out
  │
  ├─ x += tm_out
  ├─ LN2(x) → ln_buf
  ├─ CHANNEL MIXING (FFN):
  │   shift → lerp → key_w·x → SqReLU → value_w·act → cm_out
  └─ x += cm_out
  │
  LN_out(x) → head_w·x → logits[65536]
```

## 5. Variant Divergences

Only two behavioral differences between non-G1k and G1k in the forward pass:

| Aspect | non-G1k (0.1B) | G1k (1.5B, 2.9B) |
|---|---|---|
| **Bonus key source** (step 11) | `k` (original) | `k_mod` (after a-gate) |
| **MatVec dispatch** | Q8 per-row scale | Q4_K/Q6_K GGML native blocks |

Everything else is identical: decay formula, activation functions, state update, normalization.

Source: `model.rs` (bonus conditional), `quant.rs` (dispatch)

## 6. Quantization Dispatch

| Type | Block size | Bytes/block | Used in | Source |
|---|---|---|---|---|
| F32 | 1 | 4 | Embeddings, fallback | `quant.rs` |
| Q8 (custom) | per-row | 1/elem + 4 | 0.1B layers | `quant.rs` |
| Q8_0 (GGML) | 32 | 34 | 1.5B Q8 layers | `quant.rs` |
| Q4_K (GGML) | 256 | 144 | 1.5B/2.9B Q4 layers | `quant.rs` |
| Q5_K (GGML) | 256 | 176 | Available, unused | `quant.rs` |
| Q6_K (GGML) | 256 | 210 | Head matrix (all GGUF) | `quant.rs` |

Native GGML quant: weights stay compressed, dequant happens per-block during matmul.
Custom Q8: per-row scale+zero, full dequant at multiply time.

## 7. Recurrent State

| Component | 0.1B | 1.5B | 2.9B |
|---|---|---|---|
| x_prev_att | 12×768 | 24×2048 | 32×2560 |
| state_mat | 12×12×64² = 590K | 24×32×64² = 3.1M | 32×40×64² = 5.2M |
| x_prev_ffn | 12×768 | 24×2048 | 32×2560 |
| v_first | 768 | 2048 | 2560 |
| **Total** | **2.6 MB** | **14.2 MB** | **23.6 MB** |
| Scratch workspace | 149 KB | 315 KB | 381 KB |

Source: `model.rs` (Rwkv7State), `model.rs` (Scratch)

## 8. Use-Case Data Flows

### Compression (`hybrid-eval`)

```
bytes ──┬── CM (19 models, hash tables, ~105 MB) → 19 bit-logits ──┐
        │                                                            │
        └── RWKV 0.1B → token logits → ByteBridge trie ──┐         │
                          └─ MatchModel (hash match) ─────┤         │
                                                          │         │
             softmax → byte_probs → 8 bit preds ─────────┘         │
                                                                    │
                     LSTM mixer (H=64, L=1, 22594 params) ← SGD ───┘
                              │
                        SSE → BPB prediction
```

- Only 0.1B backbone (SafeTensors, emb-surgery center0.3)
- Online adaptation: lr=0.002, per-byte update
- Speed: ~139 B/s | Best: 1.1535 BPB (enwik8 10KB)

## 9. Key File Map

| File | Role | Lines |
|---|---|---|
| `src/infrastructure/rwkv7/model.rs` | Forward pass, 6 layer loaders, config | ~1280 |
| `src/infrastructure/rwkv7/gguf.rs` | GGUF parser, dequantization | ~500 |
| `src/infrastructure/rwkv7/safetensors.rs` | SafeTensors parser | ~200 |
| `src/infrastructure/rwkv7/tokenizer.rs` | World tokenizer (65K vocab) | ~300 |
| `src/domain/quant.rs` | QuantMatrix, auto-detect, GGML native | ~600 |
| `src/domain/backbone.rs` | ByteBackbone trait, RwkvBackbone, orchestrator | ~250 |
| `src/domain/bridge.rs` | Token→byte trie marginalization | ~200 |
| `src/domain/cm.rs` | Context Mixer (19 models, LSTM mixer) | ~1200 |
| `src/application/model_loader.rs` | Centralized load + vocab resolution | ~39 |
