#!/usr/bin/env bash
# Canonical docker run for Gentleman TTS leaderboard on Jetson (GPU required).
#
# Matcha + BigVGAN Python ORT CUDA. No sherpa, no TensorRT engines.
#
# Usage:
#   ./deploy/judgeflow_tts_run.sh <image> <container_name> <mcp_port> <ws_port>
set -euo pipefail

IMAGE="${1:?image required}"
NAME="${2:?container name required}"
MCP_PORT="${3:?MCP_PORT required}"
WS_PORT="${4:?WS_PORT required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

RUN_ARGS=(
    docker run -d
    --name "${NAME}"
    --runtime nvidia
    --network host
    --privileged
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e MCP_PORT="${MCP_PORT}"
    -e WS_PORT="${WS_PORT}"
    -e TTS_RANKING_MODE="${TTS_RANKING_MODE:-0}"
    -e TTS_REQUIRE_CUDA=1
)

# Git checkout overrides the image /work copies so ranking can git pull
# without rebuilding phanthymotus-perception-tts.
if [[ -f "$ROOT/perception/plugins/tts.py" ]]; then
  RUN_ARGS+=(
    -v "$ROOT/perception/plugins/tts.py:/work/plugins/tts.py:ro"
    -v "$ROOT/perception/utils/matcha_ort.py:/work/utils/matcha_ort.py:ro"
    -v "$ROOT/perception/utils/moss_ort.py:/work/utils/moss_ort.py:ro"
    -v "$ROOT/perception/utils/model_downloader.py:/work/utils/model_downloader.py:ro"
    -v "$ROOT/perception/utils/phonetone:/work/utils/phonetone:ro"
    -v "$ROOT/perception/config.yaml:/work/config.yaml:ro"
    -v "$ROOT/perception/main.py:/work/main.py:ro"
  )
fi
if [[ -f "$ROOT/perception/deploy/entrypoint.sh" ]]; then
  RUN_ARGS+=(-v "$ROOT/perception/deploy/entrypoint.sh:/deploy/entrypoint.sh:ro")
fi

if [ -d /models ]; then
    RUN_ARGS+=(-v /models:/models)
fi

if [ -n "${ROS_DOMAIN_ID:-}" ]; then
    RUN_ARGS+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi
if [ -n "${FASTDDS_BUILTIN_TRANSPORTS:-}" ]; then
    RUN_ARGS+=(-e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}")
fi

RUN_ARGS+=("${IMAGE}")

echo "[judgeflow_tts_run] gentleman-ort ${RUN_ARGS[*]}"
"${RUN_ARGS[@]}"
