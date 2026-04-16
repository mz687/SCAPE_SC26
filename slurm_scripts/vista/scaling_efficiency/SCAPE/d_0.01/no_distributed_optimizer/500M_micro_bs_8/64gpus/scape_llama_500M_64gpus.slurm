#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=00:05:00
#SBATCH -o 64gpus-%J.out
#SBATCH -e 64gpus-%J.err
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH -p YOUR_PARTITION

set -euo pipefail

cd "$(dirname "$0")"

srun -A YOUR_ACCOUNT -t 00:05:00 -p YOUR_PARTITION --jobid="${SLURM_JOB_ID}" --nodes=64 --ntasks=64 --ntasks-per-node=1 ../scape_llama_500M.sh
