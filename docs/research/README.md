# Redukt Research Documents

Technical research documents covering backbone selection, quantization, codec design, and evaluation methodology.

---

## Backbone & Quantization

| Document | Description |
|---|---|
| [backbone-evaluation.md](backbone-evaluation.md) | Due diligence on RWKV-7: credibility, training data, security, and alternatives |
| [rwkv7-forward-pass.md](rwkv7-forward-pass.md) | Complete RWKV-7 forward pass specification for Rust implementation |
| [hardware-optimization.md](hardware-optimization.md) | CPU inference bottleneck analysis: roofline model, Q8 kernels, VNNI |
| [competitive-landscape.md](competitive-landscape.md) | Sub-1.0 BPB compression ecosystem survey and architectural paradigm analysis |
| [scaling-analysis.md](scaling-analysis.md) | RWKV-7 scaling from 0.1B to 0.4B/1.5B — projections and empirical results |
| [q8-quantization.md](q8-quantization.md) | Q8 per-row quantization with integer accumulation — validated, +80% speed |
| [q4-quantization-killed.md](q4-quantization-killed.md) | Q4 group quantization experiment — killed due to BPB and speed regression |
| [mixed-precision.md](mixed-precision.md) | Literature review on per-matrix mixed-precision quantization strategies |
| [number-systems-survey.md](number-systems-survey.md) | Exhaustive survey of alternative number systems for CPU neural inference |

## Codec & Prediction Pipeline

| Document | Description |
|---|---|
| [arithmetic-coder.md](arithmetic-coder.md) | CDF-24 range coder implementation and ecosystem validation |
| [context-mixing.md](context-mixing.md) | Byte-level context mixing with 9 order models and logistic mixer |
| [token-byte-bridge.md](token-byte-bridge.md) | Token-to-byte marginalization via vocabulary trie for hybrid prediction |
| [lstm-mixer.md](lstm-mixer.md) | Temporal LSTM mixer replacing logistic mixing — validated, -0.0375 BPB |
| [match-model.md](match-model.md) | Longest-match predictor with hash-based variable-length context matching |

## Evaluation

| Document | Description |
|---|---|
| [benchmark-methodology.md](benchmark-methodology.md) | Benchmark selection: Silesia, AIT 2026, Weissman Score methodology |
