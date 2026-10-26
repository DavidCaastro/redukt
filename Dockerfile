# Multi-stage build for Redukt API server.
# Stage 1: build the binary
# Stage 2: minimal runtime with weights

# --- Build stage ---
FROM rust:1.82-bookworm AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# Release build with fat LTO
RUN cargo build --release && strip target/release/redukt

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /src/target/release/redukt /app/redukt

# Weights must be provided at build time or mounted at runtime.
# Option A: bake weights into image (larger image, simpler deploy)
#   COPY weights/rwkv7-0.1b/ /app/weights/rwkv7-0.1b/
# Option B: mount at runtime (smaller image, requires volume)
#   docker run -v /path/to/weights:/app/weights redukt api

# Create weights dir for mount point
RUN mkdir -p /app/weights/rwkv7-0.1b

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8080/v1/status || exit 1

ENTRYPOINT ["/app/redukt"]
CMD ["api", "--host", "0.0.0.0", "--port", "8080"]
