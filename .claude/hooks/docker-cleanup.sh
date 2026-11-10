#!/usr/bin/env bash
# Claude Code SessionEnd hook (CLAUDE.md "Process Safety"): agents/sessions that build Docker
# images must clean up. Removes Redukt test containers and images (redukt:*) and the build
# cache; keeps base images (rust, nvidia/cuda) so the next rebuild does not re-download them.
# Safe to run when Docker is absent or stopped: exits 0 silently.
set -u
command -v docker >/dev/null 2>&1 || exit 0
docker info >/dev/null 2>&1 || exit 0

docker ps -a --format '{{.ID}} {{.Image}}' | awk '$2 ~ /^redukt(:|$)/ {print $1}' \
  | xargs -r docker rm -f >/dev/null 2>&1
docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^redukt:' \
  | xargs -r docker rmi -f >/dev/null 2>&1
docker builder prune -af >/dev/null 2>&1
exit 0
