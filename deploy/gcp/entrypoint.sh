#!/usr/bin/env bash
# Redukt container entrypoint for GCP (Cloud Run / Cloud Batch / GCE).
#
# Usage (first argument = mode):
#   redukt-entrypoint.sh api              HTTP server on $PORT (default 8080), models loaded before bind
#   redukt-entrypoint.sh batch-compress   shard $IN_DIR by task and compress to $OUT_DIR (lockstep, --jobs N streams)
#   redukt-entrypoint.sh batch-decompress shard $IN_DIR/*.azth by task and restore them to $OUT_DIR
#   redukt-entrypoint.sh fetch-weights    only make sure the weights are present, then exit
#   redukt-entrypoint.sh <anything else>  weights bootstrap, then `exec redukt "$@"`
#   redukt-entrypoint.sh --help | -h | help   no bootstrap, straight to `redukt`
#
# Weights (never baked in the image):
#   1. Already mounted at $REDUKT_WEIGHTS_DIR (default /weights; GCS FUSE) -> nothing to do.
#   2. $WEIGHTS_TARBALL_URL (signed HTTPS URL to a .tar/.tar.gz of the weights dir) -> curl | tar.
#   3. $WEIGHTS_BUCKET (+ optional $WEIGHTS_PREFIX) -> `gcloud storage cp` if gcloud exists in the
#      image, otherwise curl + OAuth token (metadata server, or $WEIGHTS_ACCESS_TOKEN).
#
# Tunables (all optional):
#   REDUKT_BACKBONE  mamba (default) | rwkv | auto | comma list (api only, e.g. mamba,rwkv for routing by content)
#   REDUKT_QUANT     auto (default; Q8 for Mamba, AZTH v5) | q8 | f32 (f32 recommended for batches of >= 16 streams)
#   JOBS             lockstep streams for batch modes (default 16); worker threads come from REDUKT_THREADS (default: all CPUs)
#   WEIGHTS_SET      weight sets to make sure are present; derived from REDUKT_BACKBONE when unset
#
# Redukt has NO env var / flag-independent weights path: it resolves `weights/<dir>` relative to
# the CWD (/app).  /app/weights is therefore a symlink to $REDUKT_WEIGHTS_DIR, re-pointed here.
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

MODE="${1:-}"

case "$MODE" in
  ""|-h|--help|help) exec redukt "$@" ;;
esac

REDUKT_WEIGHTS_DIR="${REDUKT_WEIGHTS_DIR:-/weights}"
WEIGHTS_BUCKET="${WEIGHTS_BUCKET:-}"          # bucket name, no gs:// prefix
WEIGHTS_PREFIX="${WEIGHTS_PREFIX:-}"          # optional folder inside the bucket ("" or "x/")
WEIGHTS_TARBALL_URL="${WEIGHTS_TARBALL_URL:-}"
WEIGHTS_ACCESS_TOKEN="${WEIGHTS_ACCESS_TOKEN:-}"
GCS_ENDPOINT="${GCS_ENDPOINT:-https://storage.googleapis.com}"   # e.g. https://restricted.googleapis.com
WEIGHTS_CACHE_DIR="${WEIGHTS_CACHE_DIR:-/tmp/redukt-weights}"   # used if REDUKT_WEIGHTS_DIR is read-only
VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-0}"

REDUKT_BACKBONE="${REDUKT_BACKBONE:-mamba}"
REDUKT_QUANT="${REDUKT_QUANT:-auto}"
case "$REDUKT_BACKBONE" in *[!a-z,]*) die "REDUKT_BACKBONE='$REDUKT_BACKBONE': use lowercase names / comma list" ;; esac
case "$REDUKT_QUANT" in auto|q8|f32|q4) ;; *) die "REDUKT_QUANT='$REDUKT_QUANT' (expected auto|q8|f32)" ;; esac

# Which weight sets this container needs.  Mamba loads from weights/mamba-130m, RWKV from weights/rwkv7-0.1b.
# batch-decompress needs whatever backbones the archives' headers name, so it asks for both by default.
default_weights_set() {
  local b=",$REDUKT_BACKBONE," set=""
  case "$MODE" in
    api|batch-compress|fetch-weights) ;;
    *) echo "mamba,rwkv"; return ;;          # batch-decompress and passthrough (decompress reads the header)
  esac
  case "$b" in *,mamba,*|*,auto,*) set="mamba" ;; esac
  case "$b" in *,rwkv,*) set="${set:+$set,}rwkv" ;; esac
  echo "$set"
}
WEIGHTS_SET="${WEIGHTS_SET:-$(default_weights_set)}"

# ---- weight inventory (paths relative to the weights root; see weights/MANIFEST.md) ----------
required_files() {
  local -a sets
  IFS=',' read -ra sets <<<"$WEIGHTS_SET"
  for s in "${sets[@]}"; do
    case "$s" in
      rwkv)  echo "rwkv7-0.1b/model.safetensors"
             echo "rwkv7-0.1b/rwkv_vocab_v20230424.txt" ;;
      mamba) echo "mamba-130m/model.safetensors"
             echo "mamba-130m/tokenizer.json" ;;
      gpu)   echo "${GPU_WEIGHTS_SUBDIR:-mambabyte-353m}/model.rbin" ;;  # CUDA plugin (byte-level vocab=256)
      "")    ;;
      *)     die "unknown WEIGHTS_SET entry '$s' (rwkv|mamba|gpu)" ;;
    esac
  done
}

missing_files() {
  local root="$1" f
  while read -r f; do
    [ -s "$root/$f" ] || echo "$f"
  done < <(required_files)
}

# ---- download back-ends -------------------------------------------------------------------
gcs_token() {
  if [ -n "$WEIGHTS_ACCESS_TOKEN" ]; then printf '%s' "$WEIGHTS_ACCESS_TOKEN"; return; fi
  curl -sf -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | sed -n 's/.*"access_token" *: *"\([^"]*\)".*/\1/p'
}

fetch_from_bucket() {
  local dest="$1" rel
  local -a want=(SHA256SUMS)
  while read -r rel; do want+=("$rel"); done < <(missing_files "$dest")
  if command -v gcloud >/dev/null 2>&1; then
    log "downloading with gcloud storage cp from gs://$WEIGHTS_BUCKET/$WEIGHTS_PREFIX"
    for rel in "${want[@]}"; do
      mkdir -p "$dest/$(dirname "$rel")"
      gcloud storage cp "gs://$WEIGHTS_BUCKET/$WEIGHTS_PREFIX$rel" "$dest/$rel.part" \
        || { [ "$rel" = SHA256SUMS ] && continue; die "download failed: $rel"; }
      mv "$dest/$rel.part" "$dest/$rel"
    done
    return
  fi
  local token; token="$(gcs_token)" || true
  [ -n "$token" ] || die "no OAuth token: not on GCP (metadata server) and WEIGHTS_ACCESS_TOKEN unset"
  log "downloading with curl from gs://$WEIGHTS_BUCKET/$WEIGHTS_PREFIX"
  for rel in "${want[@]}"; do
    mkdir -p "$dest/$(dirname "$rel")"
    if curl -fL --retry 5 --retry-delay 2 --connect-timeout 10 -sS \
         -H "Authorization: Bearer $token" \
         -o "$dest/$rel.part" "$GCS_ENDPOINT/$WEIGHTS_BUCKET/$WEIGHTS_PREFIX$rel"; then
      mv "$dest/$rel.part" "$dest/$rel"
    else
      rm -f "$dest/$rel.part"
      [ "$rel" = SHA256SUMS ] && { log "no SHA256SUMS in bucket (integrity not verified)"; continue; }
      die "download failed: $rel"
    fi
  done
}

fetch_from_tarball() {
  local dest="$1"
  log "downloading weights tarball (signed URL)"
  local zflag=""
  case "${WEIGHTS_TARBALL_URL%%\?*}" in *.tar.gz|*.tgz) zflag="z" ;; esac   # tar cannot sniff gzip on a pipe
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 10 -sS "$WEIGHTS_TARBALL_URL" \
    | tar -x${zflag} -C "$dest" -f -
}

verify_sums() {
  local root="$1"
  [ -f "$root/SHA256SUMS" ] || return 0
  log "verifying SHA256SUMS"
  (cd "$root" && sha256sum --quiet --ignore-missing -c SHA256SUMS) || die "weights checksum mismatch"
}

point_weights_at() {
  local target="$1"
  if [ "$(readlink -f /app/weights 2>/dev/null || true)" != "$(readlink -f "$target")" ]; then
    ln -sfn "$target" /app/weights || die "cannot re-point /app/weights -> $target (read-only rootfs?)"
  fi
}

ensure_weights() {
  local root="$REDUKT_WEIGHTS_DIR" missing
  missing="$(missing_files "$root")"
  if [ -z "$missing" ]; then
    log "weights present in $root (set: $WEIGHTS_SET)"
    [ "$VERIFY_WEIGHTS" = 1 ] && verify_sums "$root"
    point_weights_at "$root"
    return
  fi
  log "missing in $root: $(echo "$missing" | tr '\n' ' ')"

  local dest="$root"
  if ! { mkdir -p "$dest" && [ -w "$dest" ]; } 2>/dev/null; then
    dest="$WEIGHTS_CACHE_DIR"
    log "$root is read-only; using $dest (on Cloud Run /tmp is RAM-backed and counts against memory)"
    mkdir -p "$dest"
  fi

  if [ -n "$WEIGHTS_TARBALL_URL" ]; then
    fetch_from_tarball "$dest"
  elif [ -n "$WEIGHTS_BUCKET" ]; then
    fetch_from_bucket "$dest"
  else
    die "weights missing and neither WEIGHTS_BUCKET nor WEIGHTS_TARBALL_URL set. Mount them at $root \
(GCS FUSE) or set WEIGHTS_BUCKET."
  fi

  missing="$(missing_files "$dest")"
  [ -z "$missing" ] || die "still missing after download: $(echo "$missing" | tr '\n' ' ')"
  verify_sums "$dest"
  point_weights_at "$dest"
  log "weights ready in $dest"
}

# ---- run a child while forwarding SIGTERM/SIGINT (Spot preemption, Cloud Run shutdown) -------
CHILD=0
forward() { [ "$CHILD" -gt 0 ] && kill -TERM "$CHILD" 2>/dev/null || true; }
run_child() {
  trap forward TERM INT
  "$@" &
  CHILD=$!
  local rc=0
  wait "$CHILD" || rc=$?
  # a trapped signal interrupts `wait` (rc > 128); if the child is still alive, wait for its real status
  if [ "$rc" -gt 128 ] && kill -0 "$CHILD" 2>/dev/null; then
    wait "$CHILD" || rc=$?
  fi
  return "$rc"
}

# ---- modes ------------------------------------------------------------------------------------
case "$MODE" in
  fetch-weights)
    ensure_weights
    ;;

  api)
    shift
    ensure_weights
    # Cloud Run injects $PORT.  Every requested model is loaded BEFORE the socket is bound (api.rs), so a
    # startup probe on /v1/status only succeeds once the models are in memory.  No --weights: Mamba and
    # RWKV resolve weights/<dir> against the CWD (/app -> $REDUKT_WEIGHTS_DIR).
    exec redukt api --bind 0.0.0.0 --port "${PORT:-8080}" \
      --backbone "$REDUKT_BACKBONE" --quant "$REDUKT_QUANT" "$@"
    ;;

  batch-compress|batch-decompress)
    shift
    ensure_weights
    case "$MODE" in
      batch-compress)   VERB=compress;   SUFFIX="" ;;
      batch-decompress) VERB=decompress; SUFFIX=".azth" ;;
    esac

    IN_DIR="${IN_DIR:-/mnt/in}"
    OUT_DIR="${OUT_DIR:-/mnt/out}"
    JOBS="${JOBS:-16}"
    SKIP_EXISTING="${SKIP_EXISTING:-1}"
    # Cloud Batch predefined variables; SHARD_* are the manual fallback.
    SHARD_COUNT="${BATCH_TASK_COUNT:-${SHARD_COUNT:-1}}"
    if [ -n "${BATCH_TASK_INDEX:-}" ]; then
      SHARD_INDEX="$BATCH_TASK_INDEX"
    elif [ "$SHARD_COUNT" -gt 1 ] && [ -z "${SHARD_INDEX:-}" ]; then
      # Without an index every task would process shard 0 and the other shards would never be done.
      die "SHARD_COUNT=$SHARD_COUNT but neither BATCH_TASK_INDEX nor SHARD_INDEX is set"
    else
      SHARD_INDEX="${SHARD_INDEX:-0}"
    fi
    [ -d "$IN_DIR" ] || die "IN_DIR $IN_DIR is not a directory"
    mkdir -p "$OUT_DIR"

    # redukt's --input-dir mode has no sharding of its own: every process would handle EVERY
    # file.  Build a per-task staging dir of symlinks: files sorted by size (desc) and dealt
    # round-robin to tasks so shards have similar total bytes.  Files whose result already exists
    # in OUT_DIR (<name>.azth when compressing, <name> when decompressing) are skipped, which makes
    # Spot-preemption retries resumable (an object only appears in GCS once the write is finalised).
    # compress skips *.azth inputs itself; decompress only reads *.azth.
    STAGE="$(mktemp -d /tmp/redukt-shard.XXXXXX)"
    trap 'rm -rf "$STAGE"' EXIT

    n=0; kept=0; skipped=0
    while IFS=$'\t' read -r -d '' size name; do
      if [ "$VERB" = compress ]; then
        case "$name" in *.azth) continue ;; esac
        done_name="$name.azth"
      else
        case "$name" in *.azth) ;; *) continue ;; esac
        done_name="${name%.azth}"
      fi
      if [ $(( n % SHARD_COUNT )) -eq "$SHARD_INDEX" ]; then
        if [ "$SKIP_EXISTING" = 1 ] && [ -e "$OUT_DIR/$done_name" ]; then
          skipped=$((skipped + 1))
        else
          ln -s "$IN_DIR/$name" "$STAGE/$name"; kept=$((kept + 1))
        fi
      fi
      n=$((n + 1))
    done < <(find "$IN_DIR" -maxdepth 1 -type f -printf '%s\t%f\0' | sort -z -t $'\t' -k1,1nr -k2,2)

    log "$VERB shard $SHARD_INDEX/$SHARD_COUNT: $n candidate files in $IN_DIR, $kept to do, $skipped already done," \
        "streams=$JOBS threads=${REDUKT_THREADS:-all} backbone=$REDUKT_BACKBONE quant=$REDUKT_QUANT"
    if [ "$kept" -eq 0 ]; then log "nothing to do"; exit 0; fi

    # compress takes --backbone/--quant (a single backbone; a comma list is api-only).  decompress takes the
    # backbone and profile from each archive header (a conflicting --quant would fail), so it gets neither.
    verb_args=()
    if [ "$VERB" = compress ]; then
      case "$REDUKT_BACKBONE" in *,*) die "batch-compress needs a single REDUKT_BACKBONE, got '$REDUKT_BACKBONE'" ;; esac
      verb_args=(--backbone "$REDUKT_BACKBONE" --quant "$REDUKT_QUANT")
    fi
    # shellcheck disable=SC2086   # REDUKT_EXTRA_ARGS is intentionally word-split
    run_child redukt "$VERB" --input-dir "$STAGE" --output-dir "$OUT_DIR" --jobs "$JOBS" \
      "${verb_args[@]+"${verb_args[@]}"}" ${REDUKT_EXTRA_ARGS:-} "$@"
    ;;

  *)
    ensure_weights
    exec redukt "$@"
    ;;
esac
