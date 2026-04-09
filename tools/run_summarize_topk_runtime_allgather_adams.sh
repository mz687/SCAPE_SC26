#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/AdamS"
SCRIPT_PATH="/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/summarize_topk_runtime.py"

ROOT_DIR="${1:-${DEFAULT_ROOT}}"
SKIP_STEPS="${SKIP_STEPS:-5}"
START_ITERATION="${START_ITERATION:-}"
END_ITERATION="${END_ITERATION:-}"

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "Error: root directory not found: ${ROOT_DIR}" >&2
    exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Error: summarize script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

mapfile -t TARGET_DIRS < <(
    find "${ROOT_DIR}" -type f \( -name "*.out" -o -name "*.log" -o -name "*.txt" \) -printf '%h\n' \
        | sort -u
)

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
    echo "No log-containing subdirectories found under: ${ROOT_DIR}" >&2
    exit 2
fi

echo "Root directory: ${ROOT_DIR}"
echo "Subdirectories with logs: ${#TARGET_DIRS[@]}"
echo "SKIP_STEPS=${SKIP_STEPS}"
if [[ -n "${START_ITERATION}" ]]; then
    echo "START_ITERATION=${START_ITERATION}"
fi
if [[ -n "${END_ITERATION}" ]]; then
    echo "END_ITERATION=${END_ITERATION}"
fi
echo

for dir in "${TARGET_DIRS[@]}"; do
    echo "===== ${dir} ====="

    cmd=(python3 "${SCRIPT_PATH}" --paths "${dir}" --skip-steps "${SKIP_STEPS}")
    if [[ -n "${START_ITERATION}" ]]; then
        cmd+=(--start-iteration "${START_ITERATION}")
    fi
    if [[ -n "${END_ITERATION}" ]]; then
        cmd+=(--end-iteration "${END_ITERATION}")
    fi

    "${cmd[@]}"
    echo

done
