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

History: until 2026-11-16 `golden.azth` was an RWKV v4 archive of 2026-10-29 (35 bytes, sha256
`736a2024822ef712bf9847cffd2b1065f03f7f1de152329a02b816f71b6c3e21`) that no test read and that had stopped decoding (size mismatch;
the RWKV quantization level is not recorded in the header and the runtime changed after that date; not bisected).
It is still in the git history (commit 112728b).
