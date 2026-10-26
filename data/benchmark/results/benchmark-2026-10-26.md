# Redukt Benchmark Report

**Generated**: 2026-10-26T10:37:18.505550+00:00
**Platform**: Windows-11-10.0.26300-SP0
**Processor**: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel
**Git commit**: 7405cb9
**Window**: 10,240 bytes per file

## Compressors Tested

| Compressor | Level | Library |
|---|---|---|
| **Redukt** | default | `target\release\redukt.exe` |
| gzip | 9 | 9 (stdlib) |
| brotli | 11 | 11 (pip 1.2.1) |
| zstd | 19 | 19 (pip 0.25.0) |

## Results

| File | Domain | Bytes | Redukt BPB | gzip-9 | brotli-11 | zstd-19 | vs brotli |
|---|---|---|---|---|---|---|---|
| enwik8 | text/encyclopedia | 10,240 | **1.3375** | 2.9797 | 2.3625 | 2.9375 | -43.4% |
| dickens | text/literature | 10,240 | **1.8188** | 3.6109 | 2.7742 | 3.5406 | -34.4% |
| samba | code/c-source | 10,240 | **1.3109** | 2.8805 | 2.2172 | 2.7906 | -40.9% |
| xml | markup/xml | 10,240 | **0.6922** | 1.0680 | 0.8906 | 1.0219 | -22.3% |
| apache-access.log | logs/apache | 10,240 | **0.3031** | 0.5453 | 0.4648 | 0.4898 | -34.8% |
| ubl-invoice-batch.xml | e-invoicing/ubl | 10,240 | **0.5453** | 1.0773 | 0.8719 | 1.0125 | -37.5% |

**Redukt mean BPB**: 1.0013 (across 6 files)

## Roundtrip Verification

| File | SHA-256 (original) | Roundtrip OK |
|---|---|---|
| enwik8 | `25ce06fbf6e86fb6...` | PASS |
| dickens | `9218d73b24934bce...` | PASS |
| samba | `7dbdbc08eb9d31e1...` | PASS |
| xml | `d006fafdff6ffde1...` | PASS |
| apache-access.log | `8040314619d079d7...` | PASS |
| ubl-invoice-batch.xml | `a1b6a4e70a9a5a90...` | PASS |

## Input File Hashes

```
25ce06fbf6e86fb6fa38080c1b19e63edb7c0599c829ca5954e57bca1dc2d836  enwik8 (10,240B window)
9218d73b24934bce91a8a357b1a977e599ba4f4588ebe42f493b3f09be45c692  dickens (10,240B window)
7dbdbc08eb9d31e14a31ac9278467483f97f1a09beba508fae364aaa6820e908  samba (10,240B window)
d006fafdff6ffde120216826cd70c76545be3e48b6739217e2070f5972146e08  xml (10,240B window)
8040314619d079d7c175ebf5050a81fa59492f8615914ee115013f3b539fd296  apache-access.log (10,240B window)
a1b6a4e70a9a5a90e0fd5d8c4189e72e38fe5e3af5450c0640367b07dc34ce23  ubl-invoice-batch.xml (10,240B window)
```

## Methodology

- Window: first 10,240 bytes of each file
- SHA-256 computed on the windowed input (not full file)
- Redukt: `compress --input FILE --output FILE.azth` (actual file size)
- Roundtrip: decompress output and compare SHA-256 with original
- Competitors: Python stdlib/pip libraries, in-memory compression
- BPB = (compressed_size * 8) / original_size
- Single run per file (deterministic compressors)
