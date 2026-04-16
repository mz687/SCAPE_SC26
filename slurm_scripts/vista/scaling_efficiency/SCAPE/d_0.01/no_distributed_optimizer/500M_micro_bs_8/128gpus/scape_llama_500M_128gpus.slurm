#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=00:05:00
#SBATCH -o 128gpus-%J.out
#SBATCH -e 128gpus-%J.err
#SBATCH -N 128
#SBATCH --ntasks-per-node=1
#SBATCH -p YOUR_PARTITION

set -euo pipefail

cd "$(dirname "$0")"

srun -A YOUR_ACCOUNT -t 00:05:00 -p YOUR_PARTITION --jobid="${SLURM_JOB_ID}" --nodes=128 --ntasks=128 --ntasks-per-node=1 ../scape_llama_500M.sh
