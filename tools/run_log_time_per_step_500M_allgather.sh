#!/bin/bash
set -euo pipefail

# activate virtual env for plotting
source /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate
ml cuda/12.4

run_compute() {
    local path="$1"

    if [[ ! -d "${path}" ]]; then
        echo "Skipping missing directory: ${path}" >&2
        echo
        return 0
    fi

    if ! python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
        --skip-steps 5 \
        --paths "${path}"; then
        echo "Skipping path (missing files or no matches): ${path}" >&2
    fi
    echo
}


# AdamS 
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/AdamS/500M_micro_bs_8/comp_comm_breakdown
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

# AdamS w/ dist-optm
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/AdamS/500M_micro_bs_8/dist-optm-comp_comm_breakdown
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

# SCAPE (d=0.1)
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/SCAPE/d_0.1/no_distributed_optimizer/500M_micro_bs_8
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

# SCAPE (d=0.1) w/ dist-optm
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/SCAPE/d_0.1/use_distributed_optimizer/500M_micro_bs_8
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

# SCAPE (d=0.1) w/ dist-optm & CPUOffload
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/SCAPE_cpu_offloadv2/d_0.1/use_distributed_optimizer/500M_micro_bs_8
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

# SCAPE (d=0.01)
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/SCAPE/d_0.01/no_distributed_optimizer/500M_micro_bs_8
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

# SCAPE (d=0.01) w/ dist-optm
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/SCAPE/d_0.01/use_distributed_optimizer/500M_micro_bs_8
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

# SCAPE (d=0.01) w/ dist-optm & CPUOffload
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed_all_gather/SCAPE_cpu_offloadv2/d_0.01/use_distributed_optimizer/500M_micro_bs_8
run_compute "${PARENT}/4gpus"

run_compute "${PARENT}/8gpus"

run_compute "${PARENT}/16gpus"

run_compute "${PARENT}/32gpus"

run_compute "${PARENT}/64gpus"

