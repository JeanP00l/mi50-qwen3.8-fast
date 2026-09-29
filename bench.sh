#!/usr/bin/env bash
# Benchmarks.
#   ./bench.sh            measure the running server (./run.sh in another terminal)
#   ./bench.sh kernels    raw passes with llama-bench: 1 token vs 4 tokens (the MTP verify pass),
#                         needs the GPUs free (stop the server first)
set -u
cd "$(dirname "$0")" || exit 1
PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
IMAGE=${IMAGE:-mi50-llama:b11235}
MODELS_DIR=$(realpath -m "${MODELS_DIR:-./models}")

if [ "${1:-}" = kernels ]; then
    DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"
    # shellcheck source=scripts/lib.sh
    . scripts/lib.sh
    mapfile -t gpus < <(gfx906_gpus)
    sm=(-sm none); [ "${#gpus[@]}" -ge 2 ] && sm=(-sm tensor -ts 1/1)   # llama-bench: slash, not comma
    render_gid=$(getent group render | cut -d: -f3)
    $DOCKER run --rm --device=/dev/kfd --device=/dev/dri --group-add video ${render_gid:+--group-add $render_gid} \
        --security-opt seccomp=unconfined -v "$MODELS_DIR":/models:ro \
        --entrypoint /usr/local/bin/llama-bench "$IMAGE" \
        -m /models/Qwen3.8-27B-Q4_0.gguf -ngl 999 "${sm[@]}" -fa 1 -p 1,4 -n 128 -r 10
    echo "pp1 / pp4 are single passes of 1 and 4 tokens: ms per pass = 1000 * n / (t/s)."
    exit 0
fi

python3 scripts/bench-client.py --url "http://$HOST:$PORT" --key "${API_KEY:-}" "$@"
