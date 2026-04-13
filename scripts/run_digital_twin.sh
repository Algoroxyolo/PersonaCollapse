#!/usr/bin/env bash
#
# Unified digital-twin (BFI) experiment runner.
#
# Supports all providers: openai, openrouter, bedrock, vllm.
#
# Workflow
# -------
#   1. Budget estimation (always runs first):
#        bash scripts/run_digital_twin.sh <provider> <model> <limit>
#
#   2. Actually execute the experiment:
#        bash scripts/run_digital_twin.sh <provider> <model> <limit> --run
#
# Examples
# --------
#   bash scripts/run_digital_twin.sh openrouter kimi-k2.5 2000
#   bash scripts/run_digital_twin.sh openrouter kimi-k2.5 2000 --run
#   bash scripts/run_digital_twin.sh vllm deepseek-r1 2000 --run
#   bash scripts/run_digital_twin.sh openai gpt-5-mini 500 --run
#   bash scripts/run_digital_twin.sh bedrock claude-haiku-4.5 2000 --run
#
# Environment variables (all optional unless noted)
# --------------------------------------------------
#   OPENAI_API_KEY          -- required for provider=openai
#   OPENROUTER_API_KEY      -- required for provider=openrouter
#   AWS_BEARER_TOKEN_BEDROCK -- required for provider=bedrock
#   VLLM_API_KEY            -- for provider=vllm (default: "dummy")
#   VLLM_BASE_URL           -- for provider=vllm (default: http://localhost:8000/v1)
#   MAX_TOKENS              -- max completion tokens   (default: 100)
#   MAX_WORKERS             -- parallel worker threads (default: 8)
#   THINKING_MODEL          -- "true" to enable thinking mode
#   THINKING_MAX_TOKENS     -- token budget for thinking (default: 8192)
#   JUDGE_MODEL             -- model alias for LLM judge
#   JUDGE_PROVIDER          -- provider for judge (default: same as main)
#   JUDGE_API_KEY           -- API key for judge (default: same as main)
#
# Flagged-persona cost-saving
# ---------------------------
# Personas listed in data/flagged_personas.pkl are excluded from API
# calls.  Placeholder rows are injected into the DB so resume logic skips them.

set -euo pipefail

# -- Argument parsing -------------------------------------------------------
PROVIDER="${1:?Usage: bash scripts/run_digital_twin.sh <provider> <model> <limit> [--run]}"
MODEL="${2:?Usage: bash scripts/run_digital_twin.sh <provider> <model> <limit> [--run]}"
LIMIT="${3:?Usage: bash scripts/run_digital_twin.sh <provider> <model> <limit> [--run]}"
RUN_FLAG="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -- Resolve API key and base URL per provider ------------------------------
case "${PROVIDER}" in
    openai)
        API_KEY="${OPENAI_API_KEY:-}"
        if [ -z "$API_KEY" ]; then
            echo "ERROR: OPENAI_API_KEY is not set."
            echo "  export OPENAI_API_KEY=\"<your-key>\""
            exit 1
        fi
        BASE_URL=""
        ;;
    openrouter)
        API_KEY="${OPENROUTER_API_KEY:-}"
        if [ -z "$API_KEY" ]; then
            echo "ERROR: OPENROUTER_API_KEY is not set."
            echo "  export OPENROUTER_API_KEY=\"<your-key>\""
            exit 1
        fi
        BASE_URL=""
        ;;
    bedrock)
        API_KEY="${AWS_BEARER_TOKEN_BEDROCK:-}"
        if [ -z "$API_KEY" ]; then
            echo "ERROR: AWS_BEARER_TOKEN_BEDROCK is not set."
            echo "  export AWS_BEARER_TOKEN_BEDROCK=\"<your-token>\""
            exit 1
        fi
        BASE_URL=""
        ;;
    vllm)
        API_KEY="${VLLM_API_KEY:-dummy}"
        BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
        ;;
    *)
        echo "ERROR: Unknown provider '${PROVIDER}'."
        echo "  Supported: openai | openrouter | bedrock | vllm"
        exit 1
        ;;
esac

# -- Configuration ----------------------------------------------------------
PERSONAS_FILE="data/sampled_personas_2000.json"
FLAGGED_FILE="data/flagged_personas.pkl"
MAX_TOKENS="${MAX_TOKENS:-100}"
MAX_WORKERS="${MAX_WORKERS:-8}"

# -- Thinking-model configuration -------------------------------------------
THINKING_MODEL="${THINKING_MODEL:-false}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-8192}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-${PROVIDER}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${API_KEY}}"

THINKING_ARGS=()
if [ "${THINKING_MODEL}" = "true" ]; then
    THINKING_ARGS+=(--thinking-model --thinking-max-tokens "${THINKING_MAX_TOKENS}")
    if [ -n "${JUDGE_MODEL}" ]; then
        THINKING_ARGS+=(
            --judge-model "${JUDGE_MODEL}"
            --judge-provider "${JUDGE_PROVIDER}"
            --judge-api-key "${JUDGE_API_KEY}"
        )
    fi
fi

# -- Build base URL argument ------------------------------------------------
BASE_URL_ARGS=()
if [ -n "${BASE_URL}" ]; then
    BASE_URL_ARGS+=(--base-url "${BASE_URL}")
fi

# -- Step 1: Budget estimate ------------------------------------------------
echo "========================================================"
echo "  Budget Estimation -- Digital Twin (BFI)"
echo "  Provider: ${PROVIDER}"
echo "  Model:    ${MODEL}"
echo "  Limit:    ${LIMIT}"
[ -n "${BASE_URL}" ] && echo "  Base URL: ${BASE_URL}"
echo "========================================================"

python "${ROOT_DIR}/experiments/estimate_budget.py" \
    --experiment digital_twin \
    --personas-file "${PERSONAS_FILE}" \
    --limit "${LIMIT}" \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --save-report "budget_digital_twin_${MODEL}.txt" \
    --flagged-personas-file "${FLAGGED_FILE}"

if [ "$RUN_FLAG" != "--run" ]; then
    echo ""
    echo "To proceed, re-run with --run:"
    echo "  bash scripts/run_digital_twin.sh ${PROVIDER} ${MODEL} ${LIMIT} --run"
    exit 0
fi

# -- Step 2: Run experiment -------------------------------------------------
echo ""
echo "========================================================"
echo "  Running Experiment: ${MODEL} (${PROVIDER})"
echo "========================================================"

DB_FILE="results_replicate_humans_${MODEL}_bfi.db"

python "${ROOT_DIR}/experiments/digital_twin.py" \
    --provider "${PROVIDER}" \
    --model "${MODEL}" \
    --limit "${LIMIT}" \
    --max-workers "${MAX_WORKERS}" \
    --api-key "${API_KEY}" \
    "${BASE_URL_ARGS[@]}" \
    --db-path "${DB_FILE}" \
    --use-sampled-personas \
    --personas-file "${PERSONAS_FILE}" \
    --flagged-personas-file "${FLAGGED_FILE}" \
    --max-tokens "${MAX_TOKENS}" \
    "${THINKING_ARGS[@]}"

echo ""
echo "Experiment complete.  Results in: ${DB_FILE}"
echo "Done!"
