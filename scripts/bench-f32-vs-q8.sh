#!/usr/bin/env bash
# Benchmark F32 vs Q8 quantization on Silesia corpus + enwik8 (10KB each)
# Outputs CSV to stdout, logs to stderr

set -euo pipefail

REDUKT="./target/release/redukt"
BYTES=10000
SILESIA_DIR="data/benchmark/silesia"
ENWIK8="data/benchmark/enwik8"
INDUSTRY_DIR="data/benchmark/industry"
OUT_CSV="data/benchmark/results/f32-vs-q8.csv"

echo "file,domain,bytes,quant,bpb,tok_per_sec,tokens,w_ngram,w_bias" > "$OUT_CSV"

run_eval() {
    local file="$1"
    local name="$2"
    local domain="$3"
    local quant="$4"

    # Run hybrid-eval, capture stderr
    local output
    output=$("$REDUKT" hybrid-eval --input "$file" --bytes "$BYTES" --quant "$quant" 2>&1)

    # Extract BPB
    local bpb
    bpb=$(echo "$output" | grep "BPB:" | head -1 | awk '{print $2}')

    # Extract tok/s
    local tps
    tps=$(echo "$output" | grep "time:" | head -1 | sed 's/.*(\([0-9.]*\) tok\/s).*/\1/')

    # Extract tokens
    local tokens
    tokens=$(echo "$output" | grep "tokenized:" | head -1 | awk '{print $3}')

    # Extract mixer weights
    local w_ngram w_bias
    w_ngram=$(echo "$output" | grep "mixer:" | head -1 | sed 's/.*w_ngram=\([0-9.-]*\),.*/\1/')
    w_bias=$(echo "$output" | grep "mixer:" | head -1 | sed 's/.*w_bias=\([0-9.-]*\)/\1/')

    echo "$name,$domain,$BYTES,$quant,$bpb,$tps,$tokens,$w_ngram,$w_bias" >> "$OUT_CSV"
    echo >&2 "  $name ($quant): BPB=$bpb  tok/s=$tps"
}

echo >&2 "=== F32 vs Q8 Benchmark ==="
echo >&2 "Window: ${BYTES} bytes"
echo >&2 ""

# Silesia corpus
SILESIA_FILES=(
    "dickens:text-english"
    "mozilla:executable"
    "mr:medical-image"
    "nci:chemical-db"
    "ooffice:executable"
    "osdb:database"
    "reymont:text-polish"
    "samba:source-code"
    "sao:astronomy-db"
    "webster:dictionary"
    "x-ray:medical-image"
    "xml:markup"
)

for entry in "${SILESIA_FILES[@]}"; do
    IFS=: read -r name domain <<< "$entry"
    file="$SILESIA_DIR/$name"
    if [ ! -f "$file" ]; then
        echo >&2 "SKIP: $file not found"
        continue
    fi
    echo >&2 "[$name]"
    run_eval "$file" "$name" "$domain" "f32"
    run_eval "$file" "$name" "$domain" "q8"
done

# enwik8
if [ -f "$ENWIK8" ]; then
    echo >&2 "[enwik8]"
    run_eval "$ENWIK8" "enwik8" "text-encyclopedia" "f32"
    run_eval "$ENWIK8" "enwik8" "text-encyclopedia" "q8"
fi

# Industry files
INDUSTRY_FILES=(
    "apache-access.log:server-logs"
    "linux-syslog.log:system-logs"
    "openssh-auth.log:auth-logs"
    "facturae-batch.xml:e-invoicing-xml"
    "ubl-invoice-batch.xml:e-invoicing-ubl"
)

for entry in "${INDUSTRY_FILES[@]}"; do
    IFS=: read -r name domain <<< "$entry"
    file="$INDUSTRY_DIR/$name"
    if [ ! -f "$file" ]; then
        echo >&2 "SKIP: $file not found"
        continue
    fi
    echo >&2 "[$name]"
    run_eval "$file" "$name" "$domain" "f32"
    run_eval "$file" "$name" "$domain" "q8"
done

echo >&2 ""
echo >&2 "=== Results written to $OUT_CSV ==="
