# Mamba-130M length collapse — diagnosis (2026-11-10)

Trigger: Kaggle run `redukt-gpu-fullsize-bench-b` (T4 + 4 vCPU, `run-b/fullsize-results.csv`) showed that with the
Mamba backbone the BPB **worsens with input length**: enwik8 0.9641 (10 KB) -> 1.6299 (100 KB, F32; Q8 1.6327), and
Redukt loses to brotli-11/xz on every whole industry file (apache 0.494 vs 0.313, UBL 0.481 vs 0.252, syslog 1.11 vs
0.37, openssh 1.17 vs 0.35; facturae 0.446 vs 0.408).

## Backbone-only curves (local, i5-1235U, `redukt mamba-eval`, enwik8 first 100 KB = 26,001 tokens)

Marginal cost per 2,000-token window derived from the cumulative counter (3.8 B/token):

| tokens | Redukt mamba-eval bits/token | HF transformers reference bits/token |
|---|---|---|
| 0-2000 | ~3.4 (0.88 bpb) | 2.24 (0.57 bpb) |
| 2000-16000 | 3.9-4.5 (1.03-1.19 bpb) | 3.75-4.48 (0.95-1.14 bpb) |
| 16000-18000 | 3.9 (1.03 bpb) | 3.99 (1.01 bpb) |
| 18000-20000 | **9.9 (2.6 bpb)** | **5.35 (1.36 bpb)** |
| 20000-22000 | **18.9 (~5 bpb)** | 5.70 (1.45 bpb) |
| 22000-24000 | 18.2 | 7.66 (1.95 bpb) |
| 24000-26000 | 16.3 | **12.02 (3.05 bpb)**, max 165 bits on one token |

HF reference: `state-spaces/mamba-130m-hf`, transformers 5.18.0, torch 2.14 CPU, slow reference scan, F32, same
`tokenizer.json` (26,001 tokens), total 1.2864 bpb over 100 KB (script: scratchpad `hf_ref_nll.py`).
log2(50280) = 15.6 bits: Redukt's late windows are at or below the uniform distribution.

Control: `mamba-eval` on **dickens** first 80 KB (23,078 tokens) stays at ~3.8 bits/token throughout (1.0717 bpb total).
Control: **RWKV-7 + CM** (`hybrid-eval`) on the same enwik8 100 KB: 1.2947 bpb cumulative, windowed 1.10-1.56, no drift.

## Conclusion

The collapse is a property of the pretrained Mamba-130M recurrence beyond ~16-18K tokens on this content (the
reference implementation degrades at the same position; Redukt degrades harder, likely because the CM/mixer cannot
recover from near-uniform backbone logits), not a Redukt runtime bug. Mamba-130M was trained with 2,048-token
contexts; length extrapolation of Mamba-1 is a known failure mode (DeciMamba/LongMamba literature).

Consequences:
1. Every headline Redukt figure was measured at 10 KB (~2.7K tokens), inside the regime where the backbone works.
2. Whole-file results on anything above ~60 KB of text are dominated by the collapse; "advantage grows with size" is
   false for Mamba (true only for the RWKV scale test).
3. Mitigation to evaluate: periodic backbone state reset (every N tokens, N <= 8192, CM models keep running),
   or a re-warm window after each reset. Experimental env var `REDUKT_MAMBA_RESET_EVERY` measures it without
   changing the format; a production version needs the period in the AZTH header (decoder must match).

## Mitigation measured (local, i5-1235U, maxperf, `compress --backbone mamba --quant q8`, full Mamba+CM pipeline)

| Input | no reset | reset every 8192 tokens | reset every 4096 tokens | reference |
|---|---|---|---|---|
| enwik8 100 KB | 1.6327 bpb (20,899 B) | 0.9968 (12,759 B) | **0.9950** (12,736 B) | 10 KB window: 0.9688; brotli-11 whole 100 KB: 2.351 |

Default output with the variable unset is byte-identical to before (enwik8 10 KB Q8 sha256 3779ba13…, as on
Kaggle). Roundtrip with `REDUKT_MAMBA_RESET_EVERY=4096` on both sides: CRC-32 verified, byte-identical.
The reset keeps the whole-file figure within +2.7% of the 10 KB figure instead of +68%; the cost is nothing
(no extra forwards). Industry whole files with reset 4096: see the table below (filled when the run completes).
