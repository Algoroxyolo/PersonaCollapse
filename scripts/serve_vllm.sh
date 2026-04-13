#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?Usage: bash scripts/serve_vllm.sh <model> [port]}"
PORT="${2:-8000}"
echo "Starting vLLM server for ${MODEL} on port ${PORT}..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --trust-remote-code
