#!/usr/bin/env bash
#
# Unified self-introduction experiment runner.
#
# Usage
# -----
#   bash scripts/run_self_introduction.sh <provider> [model1 model2 ...]
#
# Examples
# --------
#   # OpenRouter -- run all registered models
#   bash scripts/run_self_introduction.sh openrouter
#
#   # OpenRouter -- specific models only
#   bash scripts/run_self_introduction.sh openrouter kimi-k2.5 claude-sonnet-4.6
#
#   # vLLM -- requires VLLM_BASE_URL
#   bash scripts/run_self_introduction.sh vllm Qwen/Qwen3-4B-Instruct-2507
#
# Environment variables
# ---------------------
#   OPENROUTER_API_KEY  -- required for provider=openrouter
#   VLLM_BASE_URL       -- required for provider=vllm (e.g. http://localhost:8123/v1)
#   VLLM_API_KEY        -- for provider=vllm (default: "dummy")
#   SAMPLES             -- responses per persona per model (default: 3)
#   MAX_WORKERS         -- concurrent API threads (default: 6)
#   OUTPUT_DIR          -- output directory (default: self_introduction_results)

set -euo pipefail

# -- Argument parsing -------------------------------------------------------
PROVIDER="${1:?Usage: bash scripts/run_self_introduction.sh <provider> [model1 model2 ...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Remaining arguments are model names (may be empty -- script defaults apply)
MODELS=("$@")

# -- Resolve API key and base URL per provider ------------------------------
case "${PROVIDER}" in
    openrouter)
        API_KEY="${OPENROUTER_API_KEY:-}"
        if [ -z "$API_KEY" ]; then
            echo "ERROR: OPENROUTER_API_KEY is not set."
            echo "  export OPENROUTER_API_KEY=\"<your-key>\""
            exit 1
        fi
        BASE_URL=""
        ;;
    vllm)
        API_KEY="${VLLM_API_KEY:-dummy}"
        BASE_URL="${VLLM_BASE_URL:-}"
        if [ -z "$BASE_URL" ]; then
            echo "ERROR: VLLM_BASE_URL is not set."
            echo "  export VLLM_BASE_URL=\"http://localhost:8123/v1\""
            exit 1
        fi
        ;;
    *)
        echo "ERROR: Unknown provider '${PROVIDER}'."
        echo "  Supported: openrouter | vllm"
        exit 1
        ;;
esac

# -- Configuration ----------------------------------------------------------
SAMPLES="${SAMPLES:-3}"
MAX_WORKERS="${MAX_WORKERS:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-self_introduction_results}"

# -- Build command ----------------------------------------------------------
CMD_ARGS=(
    --provider "${PROVIDER}"
    --api-key "${API_KEY}"
    --samples "${SAMPLES}"
    --max-workers "${MAX_WORKERS}"
    --output-dir "${OUTPUT_DIR}"
)

if [ -n "${BASE_URL}" ]; then
    CMD_ARGS+=(--base-url "${BASE_URL}")
fi

if [ ${#MODELS[@]} -gt 0 ]; then
    CMD_ARGS+=(--models "${MODELS[@]}")
fi

# -- Run --------------------------------------------------------------------
echo "========================================================"
echo "  Self-Introduction Collection"
echo "  Provider:    ${PROVIDER}"
if [ ${#MODELS[@]} -gt 0 ]; then
    echo "  Models:      ${MODELS[*]}"
else
    echo "  Models:      (all defaults for ${PROVIDER})"
fi
echo "  Samples:     ${SAMPLES}"
echo "  Workers:     ${MAX_WORKERS}"
echo "  Output dir:  ${OUTPUT_DIR}"
[ -n "${BASE_URL}" ] && echo "  Base URL:    ${BASE_URL}"
echo "========================================================"

python "${ROOT_DIR}/experiments/self_introduction.py" "${CMD_ARGS[@]}"

echo ""
echo "Done!  Results in: ${OUTPUT_DIR}/"
