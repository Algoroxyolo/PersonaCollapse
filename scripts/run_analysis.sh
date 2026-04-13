#!/usr/bin/env bash
# run_analysis.sh -- Run the full analysis pipeline
#
# Usage:
#   bash scripts/run_analysis.sh [DATA_DIR] [SELFINTRO_DIR] [OUTPUT_DIR]
#
# Examples:
#   # Default paths
#   bash scripts/run_analysis.sh
#
#   # Custom paths
#   bash scripts/run_analysis.sh ./csv_exports ./self_introduction_results ./my_results
#
# Environment variables:
#   HUMAN_REF    Path to wave_1_numbers.csv  (default: data/human_reference/wave_1_numbers.csv)
#   FLAGGED      Path to flagged_personas.pkl (default: empty, no filtering)
#   SKIP_SELFINTRO  Set to 1 to skip self-intro analysis
#   SKIP_FIGURES    Set to 1 to skip figure generation
set -euo pipefail

DATA_DIR="${1:-./csv_exports}"
SELFINTRO_DIR="${2:-./self_introduction_results}"
OUTPUT_DIR="${3:-./results}"
HUMAN_REF="${HUMAN_REF:-./data/human_reference/wave_1_numbers.csv}"
FLAGGED="${FLAGGED:-}"

echo "=============================="
echo " Analysis Pipeline"
echo "=============================="
echo "  data_dir     : ${DATA_DIR}"
echo "  selfintro_dir: ${SELFINTRO_DIR}"
echo "  human_ref    : ${HUMAN_REF}"
echo "  output_dir   : ${OUTPUT_DIR}"
echo "  flagged      : ${FLAGGED:-<none>}"
echo "=============================="

EXTRA_ARGS=()

if [ -n "${FLAGGED}" ]; then
    EXTRA_ARGS+=(--flagged "${FLAGGED}")
fi

if [ "${SKIP_SELFINTRO:-0}" = "1" ]; then
    EXTRA_ARGS+=(--skip_selfintro)
fi

if [ "${SKIP_FIGURES:-0}" = "1" ]; then
    EXTRA_ARGS+=(--skip_figures)
fi

python -m analysis.analysis_pipeline \
    --data_dir "${DATA_DIR}" \
    --selfintro_dir "${SELFINTRO_DIR}" \
    --human_ref "${HUMAN_REF}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
