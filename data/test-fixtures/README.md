# Golden fixtures

| File | Bytes | sha256 | What |
|---|---|---|---|
| `golden-input.bin` | 39 | — | `Determinism: the quick brown fox jumps!` |
| `golden.rdkt` | 33 | `be14e088b992407280fdde8435801f50fc29595590922da961c775fc124d5cf2` | RDKT v5 archive of the input: `redukt compress --quant f32` (Mamba-130M, F32, numeric contract 1, RESET_LOG2 12, PROFILE 0x50) |
| `golden.azth` | 33 | `b8719bf8d1ac6a7ed3426981ccb859f53f195023af7d869000fbb294bdb8fc0d` | the same archive with the legacy magic `AZTH` (what the encoder wrote before 2026-11-16); keeps the legacy read path under test |

Both are read by `tests/roundtrip.rs`: `golden_fixtures_are_pinned` (no weights needed: hashes, sizes, the two
files differ only in the four magic bytes) and `golden_reference_determinism` (real weights: a fresh compress must
equal `golden.rdkt` byte for byte, and both fixtures must decode to the input).

Regenerate only when the format or the pipeline changes on purpose:

```
redukt compress --input data/test-fixtures/golden-input.bin --output data/test-fixtures/golden.rdkt --quant f32
```

then write `golden.azth` as `AZTH` + bytes 4 onward of `golden.rdkt`, and update the two hashes in the test.

## Linux reference archives (`linux-x86_64/`)

Archives written on Linux (Kaggle, 4-vCPU Xeon, x86-64-v3, 2026-11-10, run `run-a-r108` of
`redukt-gpu-fullsize-bench`) for the cross-OS gate 2b of `CLAUDE.md`. They carry the legacy magic `AZTH`.

| File | Bytes | sha256 | What |
|---|---|---|---|
| `enwik8.10k` | 10240 | — | first 10,240 bytes of enwik8 |
| `enwik8.10k.q8.azth` | 1241 | `d550671b4258c908d859b9159530a9db84e107aa525848f317cbbf3cb5886736` | `compress --quant q8` on Linux |
| `enwik8.10k.f32.azth` | 1235 | `fb07e20d4fff04b0a19559a35e3c29d1bd623cc09604f775aa41aa962fa3484f` | `compress --quant f32` on Linux |

Tests: `golden_fixtures_are_pinned` pins the two hashes; `linux_reference_archives_match` (real weights)
compresses `enwik8.10k` on the current machine and requires bytes 4 onward to equal the Linux archive, then
decodes the Linux archive. Verified on Windows x86-64 AVX2 on 2026-11-16, both profiles.

History: until 2026-11-16 `golden.azth` was an RWKV v4 archive of 2026-10-29 (35 bytes, sha256
`736a2024822ef712bf9847cffd2b1065f03f7f1de152329a02b816f71b6c3e21`) that no test read and that had stopped decoding (size mismatch;
the RWKV quantization level is not recorded in the header and the runtime changed after that date; not bisected).
It is still in the git history (commit 112728b).
