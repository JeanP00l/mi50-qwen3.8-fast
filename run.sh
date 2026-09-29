#!/usr/bin/env bash
# Starts llama-server with Qwen3.8-27B on MI50 (gfx906). Picks the mode by the GPUs it finds:
#   2+ GPUs  -> -sm tensor across two GPUs + MTP (measured: ~48 tok/s on an agent request)
#   1x 32 GB -> single GPU + MTP (not measured by the author)
#   1x 16 GB -> refuses: the Q4_0 model (16.1 GB) does not fit, see docs/en/single-gpu.md
#
# Settings (environment variables):
#   MODELS_DIR=./models   HOST=127.0.0.1   PORT=8080   API_KEY=       CTX=131072
#   IMAGE=mi50-llama:b11235   MTP=1   DRAFT=3   REASONING_BUDGET=8192
#   TEMPLATE=agent|model  (agent = templates/qwen3.8-agent.jinja, works with Claude Code)
#   GPUS=0,1   DETACH=0   NAME=mi50-qwen   EXTRA="more llama-server flags"
#   MODEL=Qwen3.8-27B-Q4_0.gguf   MTP_MODEL=mtp-Qwen3.8-27B-Q4_0.gguf   (file names in MODELS_DIR)
#   FORCE=1   start even if the model looks too big for one GPU
#
#   ./run.sh              start in the foreground (Ctrl+C stops)
#   ./run.sh --dry-run    print the docker command, start nothing
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=scripts/lib.sh
. scripts/lib.sh

MODELS_DIR=$(realpath -m "${MODELS_DIR:-./models}")
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
API_KEY=${API_KEY:-}
CTX=${CTX:-131072}
IMAGE=${IMAGE:-mi50-llama:b11235}
MTP=${MTP:-1}
DRAFT=${DRAFT:-3}
REASONING_BUDGET=${REASONING_BUDGET:-8192}
TEMPLATE=${TEMPLATE:-agent}
DETACH=${DETACH:-0}
NAME=${NAME:-mi50-qwen}
EXTRA=${EXTRA:-}
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

MODEL=${MODEL:-Qwen3.8-27B-Q4_0.gguf}
MTP_MODEL=${MTP_MODEL:-mtp-Qwen3.8-27B-Q4_0.gguf}

mapfile -t gpus < <(gfx906_gpus)
n=${#gpus[@]}
[ -n "${FAKE_GPUS:-}" ] && n=$FAKE_GPUS   # for --dry-run testing without GPUs
if [ "$n" -eq 0 ]; then say "No gfx906 GPU found. Run scripts/check-system.sh"; [ "$DRY" = 1 ] || exit 1; n=2; fi

vram=0
for g in "${gpus[@]}"; do read -r _ v _ _ <<<"$g"; [ "$v" -gt "$vram" ] && vram=$v; done
[ -n "${FAKE_VRAM_GB:-}" ] && vram=$((FAKE_VRAM_GB * 1073741824))

if [ "$n" -ge 2 ]; then
    GPUS=${GPUS:-0,1}
    split=(--split-mode tensor --tensor-split 1,1)
    mode="2 GPUs, -sm tensor"
elif [ "$vram" -gt $((24 * 1073741824)) ]; then
    GPUS=${GPUS:-0}
    split=(--split-mode none)
    mode="1 GPU ($(gib "$vram") GiB)"
elif [ "${FORCE:-0}" = 1 ]; then
    GPUS=${GPUS:-0}
    split=(--split-mode none)
    mode="1 GPU ($(gib "$vram") GiB), FORCE=1"
else
    say "One GPU with $(gib "$vram") GiB: Qwen3.8-27B Q4_0 (16.1 GB) + MTP head do not fit."
    say "See docs/en/single-gpu.md for options."
    exit 1
fi

need=("$MODEL"); [ "$MTP" = 1 ] && need+=("$MTP_MODEL")
for f in "${need[@]}"; do
    [ -f "$MODELS_DIR/$f" ] || { say "Missing $MODELS_DIR/$f - run ./install.sh first"; [ "$DRY" = 1 ] || exit 1; }
done

tmpl=()
if [ "$TEMPLATE" = agent ]; then
    tmpl=(--chat-template-file /templates/qwen3.8-agent.jinja)
fi
mtp=()
if [ "$MTP" = 1 ]; then
    mtp=(--spec-type draft-mtp --spec-draft-model "/models/$MTP_MODEL" --spec-draft-n-max "$DRAFT")
fi
key=()
[ -n "$API_KEY" ] && key=(--api-key "$API_KEY")

threads=$(nproc); [ "$threads" -gt 16 ] && threads=16

grp=(--group-add video)
render_gid=$(getent group render | cut -d: -f3)
[ -n "$render_gid" ] && grp+=(--group-add "$render_gid")

docker_cmd=(docker)
docker info >/dev/null 2>&1 || docker_cmd=(sudo docker)

run=("${docker_cmd[@]}" run --rm --name "$NAME"
    --device=/dev/kfd --device=/dev/dri "${grp[@]}"
    --security-opt seccomp=unconfined
    -e HIP_VISIBLE_DEVICES="$GPUS"
    -v "$MODELS_DIR":/models:ro -v "$PWD/templates":/templates:ro
    -p "$HOST:$PORT:8080")
[ "$DETACH" = 1 ] && run+=(-d)
run+=("$IMAGE"
    --model "/models/$MODEL" --alias qwen3.8
    --host 0.0.0.0 --port 8080 "${key[@]}"
    --ctx-size "$CTX" --n-gpu-layers 999 "${split[@]}"
    --parallel 1 --batch-size 2048 --ubatch-size 512
    --jinja "${tmpl[@]}"
    # The three cache flags avoid "Memory access fault" on gfx906 (see docs/en/troubleshooting.md).
    --cache-reuse 0 --cache-ram 0 --ctx-checkpoints 0 --no-context-shift
    --threads "$threads" --flash-attn on
    "${mtp[@]}"
    --reasoning-format deepseek --reasoning-budget "$REASONING_BUDGET"
    --reasoning-budget-message "Thinking budget exhausted - acting now."
    --metrics)
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && run+=($EXTRA)

say "Mode: $mode, MTP=$MTP (draft $DRAFT), ctx $CTX, http://$HOST:$PORT"
if [ "$DRY" = 1 ]; then printf '%q ' "${run[@]}"; echo; exit 0; fi
exec "${run[@]}"
