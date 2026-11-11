# Small-backbone probe (R-105, step 1) — 2026-11-11

Backbone-only cross-entropy (bits per input byte; no CM, no coder) of pretrained candidates smaller than
Mamba-130M, on the same data and protocol. Kaggle T4 + 4 vCPU, `notebooks/small-backbone-probe.py`.
Full analysis and decision: `docs/internal/research/small-backbone-probe-2026-11-11.md`.

- `run-1/` — kernel v1: Pythia 14M/31M/70M/160M, distilgpt2, gpt2, SmolLM2-135M and the Mamba-130M reference
  (0.61 h). nGPT-enwiki8 failed to build (`nGPT-pytorch` vs the Kaggle PyTorch); L3TC skipped (no tokenizer in its
  bundle).
- `run-2-l3tc/` — kernel v3: L3TC 200K/800K/3.2M/12M with the tokenizer rebuilt from enwik8 using the parameters of
  the L3TC repo (0.06 h). All four checkpoints load with every key matched and reproduce the published enwik8 figures.

Files per run: `small-backbone-results.csv` (model, dataset, chunk, bytes, tokens, bits, bpb),
`small-backbone-speed.csv` (params, MFLOP/token, PyTorch CPU 1-thread tok/s), `small-backbone-classical.csv`,
`small-backbone-report.txt`.

## bpb per dataset (best chunk per model; bold = better than the best classical codec)

| Model | Active params | Cost/byte vs Mamba | enwik8 8x10K | enwik8 500K | enwik9 tail | dickens | facturae | UBL | apache | syslog | openssh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mamba-130m | 129.1M | 1.000x | **1.028** | **0.938** | **0.998** | **1.018** | **0.231** | 0.414 | **0.175** | **0.246** | **0.214** |
| smollm2-135m | 134.5M | 1.103x | **0.919** | **0.834** | **0.915** | **0.964** | **0.183** | 0.336 | **0.182** | **0.253** | **0.231** |
| pythia-160m | 123.7M | 0.958x | **1.149** | **1.026** | **1.094** | **1.112** | **0.292** | 0.599 | **0.236** | **0.308** | **0.302** |
| gpt2 | 124.4M | 1.014x | **1.287** | **1.173** | **1.270** | **1.766** | 0.865 | 1.136 | 0.769 | 0.702 | 0.747 |
| distilgpt2 | 81.9M | 0.667x | **1.437** | **1.311** | **1.406** | **1.953** | 1.175 | 1.375 | 0.734 | 0.800 | 0.848 |
| pythia-70m | 44.7M | 0.346x | **1.313** | **1.193** | **1.269** | **1.292** | **0.369** | 0.740 | **0.263** | **0.362** | 0.402 |
| pythia-31m | 17.6M | 0.136x | **1.420** | **1.283** | **1.368** | **1.377** | 0.470 | 0.832 | **0.289** | 0.412 | 0.441 |
| pythia-14m | 7.6M | 0.059x | **1.593** | **1.448** | **1.541** | **1.551** | 0.629 | 0.985 | 0.356 | 0.502 | 0.529 |
| l3tc-12m | 13.0M | 0.104x | **0.902** | **0.850** | **1.467** | **1.909** | 5.164 | 6.413 | 4.791 | 5.107 | 5.345 |
| l3tc-3.2m | 6.2M | 0.049x | **1.102** | **1.039** | **1.455** | **1.900** | 5.146 | 6.210 | 4.314 | 4.573 | 4.912 |
| l3tc-800k | 3.4M | 0.027x | **1.233** | **1.176** | **1.587** | **2.005** | 5.396 | 6.544 | 4.676 | 4.683 | 4.700 |
| l3tc-200k | 1.7M | 0.014x | **1.378** | **1.318** | **1.628** | **2.076** | 6.562 | 7.281 | 4.396 | 4.617 | 4.608 |
| best classical | — | — | 2.735 | 2.253 | 2.361 | 2.189 | 0.408 | 0.252 | 0.313 | 0.370 | 0.346 |

## Verdict

No published smaller backbone keeps the ratio. General small models (Pythia) degrade steadily with size and lose
to brotli/xz on XML and logs; tiny specialists (L3TC, trained on enwik8) match Mamba-130M only on their own
training file, lose 46% on unseen Wikipedia and are useless out of domain (4.3-7.3 bpb on XML/logs). At equal size
SmolLM2-135M is 5-11% better on text and 19-21% better on XML, 3-8% worse on logs. Even the cheapest candidate
(L3TC-200K, 72x cheaper per byte) projects to ~$0.77/GB against $0.18/GB of saving value on unseen Wikipedia text.
These are backbone-only numbers: the CM ensemble is not included (UBL is 0.414 backbone-only vs 0.239 in the real
archive), and the projected cost assumes the current runtime efficiency per FLOP, which is not measured for
small models.
