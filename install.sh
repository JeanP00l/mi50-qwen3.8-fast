#!/usr/bin/env bash
# One-shot setup: check the system, build the patched llama.cpp image, download the models.
#   ./install.sh            everything
#   ./install.sh --no-models   only check + build
# MODELS_DIR=/path ./install.sh   put models elsewhere (~18 GB)
set -u
cd "$(dirname "$0")" || exit 1
IMAGE=${IMAGE:-mi50-llama:b11235}
MODELS_DIR=${MODELS_DIR:-./models}

echo "### 1/3 System check"
if ! scripts/check-system.sh; then
    echo
    echo "Fix the [FAIL] items above, then run ./install.sh again."
    [ "${FORCE:-0}" = 1 ] || exit 1
fi

DOCKER=docker
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

echo; echo "### 2/3 Image $IMAGE"
if $DOCKER image inspect "$IMAGE" >/dev/null 2>&1 && [ "${REBUILD:-0}" != 1 ]; then
    echo "already built (REBUILD=1 to rebuild)"
else
    echo "building: ~7 GB base download + 20-40 min compile, once"
    $DOCKER build -f docker/Dockerfile -t "$IMAGE" . || { echo "build failed"; exit 1; }
    $DOCKER run --rm --entrypoint cat "$IMAGE" /src/PATCHES
fi

if [ "${1:-}" != "--no-models" ]; then
    echo; echo "### 3/3 Models -> $MODELS_DIR"
    python3 scripts/download-models.py "$MODELS_DIR" || exit 1
fi

echo
echo "Done. Start the server:   ./run.sh"
echo "Then test it:             ./bench.sh"
