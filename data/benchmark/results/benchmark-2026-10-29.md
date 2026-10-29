# Redukt Benchmark Report

**Generated**: 2026-10-29T14:56:48.660090+00:00
**Platform**: Windows-11-10.0.26300-SP0
**Processor**: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel
**Git commit**: 382cf08
**Window**: 10,000 bytes per file

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
| enwik8 | text/encyclopedia | 10,000 | **1.3328** | 2.9720 | 2.3520 | 2.9320 | -43.3% |
| dickens | text/literature | 10,000 | **1.7968** | 3.6208 | 2.7784 | 3.5504 | -35.3% |
| mozilla | binary/executable | 10,000 | **2.0520** | 2.3728 | 2.0208 | 2.2112 | +1.5% |
| mr | medical/mri | 10,000 | **2.3888** | 2.7800 | 2.3728 | 2.5960 | +0.7% |
| nci | scientific/chemistry | 10,000 | **0.6688** | 0.9344 | 0.7344 | 0.7816 | -8.9% |
| ooffice | binary/office | 10,000 | **3.7672** | 2.9104 | 2.7424 | 2.7664 | +37.4% |
| osdb | database/mysql | 10,000 | **5.7064** | 4.9672 | 4.6760 | 4.8840 | +22.0% |
| reymont | text/polish | 10,000 | **1.8992** | 2.3872 | 2.1360 | 2.2872 | -11.1% |
| samba | code/c-source | 10,000 | **1.2976** | 2.8840 | 2.2200 | 2.7944 | -41.5% |
| sao | scientific/astronomy | 10,000 | **9.2432** | 6.3888 | 5.9880 | 6.2240 | +54.4% |
| webster | text/dictionary | 10,000 | **1.8048** | 3.6608 | 2.8088 | 3.5832 | -35.7% |
| x-ray | medical/xray | 10,000 | **5.8416** | 5.8728 | 4.5920 | 5.6216 | +27.2% |
| xml | markup/xml | 10,000 | **0.6888** | 1.0752 | 0.9008 | 1.0288 | -23.5% |
| apache-access.log | logs/apache | 10,000 | **0.2960** | 0.5464 | 0.4664 | 0.4928 | -36.5% |
| facturae-batch.xml | e-invoicing/facturae | 10,000 | **0.4528** | 1.0704 | 0.8904 | 1.0008 | -49.1% |
| linux-syslog.log | logs/syslog | 10,000 | **0.4120** | 0.7136 | 0.5816 | 0.6352 | -29.2% |
| openssh-auth.log | logs/ssh | 10,000 | **0.5272** | 0.9904 | 0.7832 | 0.8992 | -32.7% |
| sample-invoice.pdf | unknown | 10,000 | **11.1224** | 7.9376 | 7.8968 | 7.8928 | +40.8% |
| ubl-invoice-batch.xml | e-invoicing/ubl | 10,000 | **0.5416** | 1.0928 | 0.8880 | 1.0296 | -39.0% |

**Redukt mean BPB**: 2.7284 (across 19 files)

## Roundtrip Verification

| File | SHA-256 (original) | Roundtrip OK |
|---|---|---|
| enwik8 | `321868b6f37a7bc0...` | PASS |
| dickens | `321a58f334032b67...` | PASS |
| mozilla | `80cffb93806af903...` | PASS |
| mr | `2fcad7ac25997253...` | PASS |
| nci | `da00ca979a3980ff...` | PASS |
| ooffice | `0c5f5c7bd38539d1...` | PASS |
| osdb | `5f7b8f7b27486e2d...` | PASS |
| reymont | `11b7f163287cb930...` | PASS |
| samba | `84e39dc025c628af...` | PASS |
| sao | `c0ac4057e8a9e8cb...` | PASS |
| webster | `e98c8fd45ea3de36...` | PASS |
| x-ray | `9e5cb872391fb3fd...` | PASS |
| xml | `f495b724478a5f5b...` | PASS |
| apache-access.log | `de85295e086390d9...` | PASS |
| facturae-batch.xml | `002c9afdd1fadb64...` | PASS |
| linux-syslog.log | `54954adf3798c2d2...` | PASS |
| openssh-auth.log | `30d9712a716deb6d...` | PASS |
| sample-invoice.pdf | `d3726e0748666a2b...` | PASS |
| ubl-invoice-batch.xml | `a8b283dcd393cb6c...` | PASS |

## Input File Hashes

```
321868b6f37a7bc0b1253c7fa04c3823de4a6f4a61f939cd514720d2aef38284  enwik8 (10,000B window)
321a58f334032b67304b20d57c9a5c1f5ced91fc90a63a3c14e26d76f7a7d4a6  dickens (10,000B window)
80cffb93806af903aec6ed1b0f7268cb811ee0e69144ed22ffd49575628400ce  mozilla (10,000B window)
2fcad7ac25997253ed1e1baea1992b5e8576d6d34a660cf216d0af6b4d856790  mr (10,000B window)
da00ca979a3980ffa4af018fdda95e9221cf086016c231f1550b313bd6f0fec1  nci (10,000B window)
0c5f5c7bd38539d13da191a97c0f7ff2650fa244114bea7e411fd2f6863b5156  ooffice (10,000B window)
5f7b8f7b27486e2d53a81252a5471caef8e4a3aceca98cc24f2b604d99e5aa27  osdb (10,000B window)
11b7f163287cb93035fbbd56ae1576c73334a243c7248fff160d02faec504472  reymont (10,000B window)
84e39dc025c628afb72b4dfdf4f049a4b3230d5c535162b48f8f89acff8f4ef7  samba (10,000B window)
c0ac4057e8a9e8cb1ce4b22d673f75be76b5bfa9a507d39dc6a7eac4159388b9  sao (10,000B window)
e98c8fd45ea3de36da8003ec2f90e1ca06f8fc2c8cd1fabd58599581f7c7151c  webster (10,000B window)
9e5cb872391fb3fd09ab0de7fcd06ed46492b7a608c1e75b8f49cbeb9a61ba89  x-ray (10,000B window)
f495b724478a5f5b00ae76fc41d39534abd0dfd179e2752cd6f35a7c05b3035c  xml (10,000B window)
de85295e086390d9e995f541a627864ac74df6ed007c073c944bebb451bd9240  apache-access.log (10,000B window)
002c9afdd1fadb64c5b2e61bfe9934481af0502209ee40c56c3f93afb8dcbb5c  facturae-batch.xml (10,000B window)
54954adf3798c2d25e89b2662aba7b5d3a1d4422edf49bd355cd934aa39bb33e  linux-syslog.log (10,000B window)
30d9712a716deb6dc8179f08da81a3280711279cf5f7290d5463b31ebc0f0b58  openssh-auth.log (10,000B window)
d3726e0748666a2bffb0a978bb4be9bd08dfb2ff686bbbf9b64245b4e3ba36d2  sample-invoice.pdf (10,000B window)
a8b283dcd393cb6ce0d3568054581180e87b1eedd9785beb14413a70b4c74c8a  ubl-invoice-batch.xml (10,000B window)
```

## Methodology

- Window: first 10,000 bytes of each file
- SHA-256 computed on the windowed input (not full file)
- Redukt: `compress --input FILE --output FILE.azth` (actual file size)
- Roundtrip: decompress output and compare SHA-256 with original
- Competitors: Python stdlib/pip libraries, in-memory compression
- BPB = (compressed_size * 8) / original_size
- Single run per file (deterministic compressors)
