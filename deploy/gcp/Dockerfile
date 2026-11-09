# syntax=docker/dockerfile:1
#
# Redukt — CPU image (api / compress / decompress / batch).
#
# References (consulted 2026-11-09):
#   https://hub.docker.com/_/rust    tag rust:1.98-bookworm (existence verified with `docker pull`; rustc 1.98.1)
#   https://hub.docker.com/_/debian  debian:bookworm-slim
# MSRV (measured, not documented upstream): Rust >= 1.89.  `--build-arg RUST_VERSION=1.88` fails with
# "the target feature `avxvnni` is currently unstable"; 1.89 builds.  The legacy ./Dockerfile pins
# rust:1.82 and therefore does NOT compile any more.
#
# Build (from the repo root, context = repo root):
#   docker build -f deploy/gcp/Dockerfile -t redukt:cpu .
#
# Weights are NOT baked into the image (they are ~900 MB and have their own
# provenance, see weights/MANIFEST.md).  They are either
#   (a) mounted at /weights (GCS FUSE on Cloud Run / Cloud Batch, `-v` on docker), or
#   (b) downloaded at start-up by deploy/gcp/entrypoint.sh from gs://$WEIGHTS_BUCKET
#       when /weights is empty.
#
# Determinism: decompression must use the SAME model, SAME weights and the SAME
# SIMD tier as compression.  The encoder reference is x86-64 AVX2, so the binary is
# built for x86-64-v3 (AVX2 + FMA + BMI2).  NEVER use target-cpu=native here: it would
# tie the binary to whatever CPU the Cloud Build worker happens to have.  A non-AVX2
# host fails fast with SIGILL instead of silently taking another numeric path.

ARG RUST_VERSION=1.98
ARG DEBIAN_RELEASE=bookworm

# ---------------------------------------------------------------- builder
FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS builder

# maxperf = release + panic=abort + strip (see Cargo.toml).  Override with
# `--build-arg PROFILE=release` for a faster (non-LTO-identical) smoke build.
ARG PROFILE=maxperf
ARG TARGET_CPU=x86-64-v3

WORKDIR /src
# Zero external crates: only the manifest, the lockfile and the sources are needed.
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# RUSTFLAGS overrides any `.cargo/config.toml` (which is also excluded by .dockerignore).
RUN RUSTFLAGS="-C target-cpu=${TARGET_CPU}" \
    cargo build --profile "${PROFILE}" --locked \
 && install -D "target/${PROFILE}/redukt" /out/redukt \
 && strip /out/redukt

# ---------------------------------------------------------------- runtime
FROM debian:${DEBIAN_RELEASE}-slim

# ca-certificates + curl: weights download (metadata-server token / signed URL).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 redukt \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin redukt \
 && mkdir -p /app /weights \
 && chown redukt:redukt /app /weights

COPY --from=builder /out/redukt /usr/local/bin/redukt
COPY deploy/gcp/entrypoint.sh /usr/local/bin/redukt-entrypoint.sh
RUN chmod 0755 /usr/local/bin/redukt-entrypoint.sh

# Redukt resolves weights RELATIVE to the working directory
# (default `weights/rwkv7-0.1b`, hard-coded `weights/mamba-130m` for the Mamba backbone
# in src/application/model_loader.rs).  There is no env var, so /app/weights is a symlink
# to the mount point.  The entrypoint re-points it if REDUKT_WEIGHTS_DIR is set.
WORKDIR /app
RUN ln -s /weights /app/weights && chown -h redukt:redukt /app/weights

USER 10001:10001

ENV REDUKT_WEIGHTS_DIR=/weights \
    PORT=8080

EXPOSE 8080

# `docker run IMAGE --help` / `docker run IMAGE compress ...` call the binary directly.
# Cloud Run / Cloud Batch override the command with /usr/local/bin/redukt-entrypoint.sh
# (weights bootstrap + sharding), see cloudrun-service.yaml and batch-job.json.
ENTRYPOINT ["redukt"]
CMD ["--help"]
