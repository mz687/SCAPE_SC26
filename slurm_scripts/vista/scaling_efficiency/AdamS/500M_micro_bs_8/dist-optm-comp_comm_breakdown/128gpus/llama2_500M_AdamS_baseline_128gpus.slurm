#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=00:05:00
#SBATCH -o 128gpus-%J.out
#SBATCH -e 128gpus-%J.err
#SBATCH -N 128
#SBATCH --ntasks-per-node=1
#SBATCH -p YOUR_PARTITION

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/llama2_500M_AdamS_baseline.sh"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "Launcher script not found: ${LAUNCH_SCRIPT}" >&2
  exit 1
fi

srun -A YOUR_ACCOUNT -t 00:05:00 -p YOUR_PARTITION --jobid="${SLURM_JOB_ID}" --nodes=128 --ntasks=128 --ntasks-per-node=1 --chdir="${SCRIPT_DIR}" bash "${LAUNCH_SCRIPT}"
