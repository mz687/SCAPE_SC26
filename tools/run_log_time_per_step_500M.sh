#!/bin/bash
set -euo pipefail

# activate virtual env for plotting
source /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate
ml cuda/12.4

# AdamS 
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed/AdamS/500M/comp_comm_breakdown
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  

# AdamS w/ dist-optm
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed/AdamS/500M/dist-optm-comp_comm_breakdown
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  


# SCAPE (d=0.1)
PARENT=/home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/scaling_efficiency/SCAPE/d_0.1/500M
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  

# SCAPE (d=0.1) w/ dist-optm
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed/SCAPE/d_0.1/500M
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  

# SCAPE (d=0.1) w/ dist-optm & CPUOffload
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed/SCAPE-cpu-offload/d_0.1/500M
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  


# SCAPE (d=0.01)
PARENT=/home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/scaling_efficiency/SCAPE/d_0.01/500M
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  

# SCAPE (d=0.01) w/ dist-optm
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed/SCAPE/d_0.01/500M
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  

# SCAPE (d=0.01) w/ dist-optm & CPUOffload
PARENT=/home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/slurm_scripts/vista/scaling_efficiency_fixed/SCAPE-cpu-offload/d_0.01/500M
python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/4gpus 

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/8gpus

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/16gpus  

python3 /home1/09308/zhengmk/work/optimus-cc/SCAPE_dist_optm/SCAPE/tools/compute_avg_step_time.py \
    --skip-steps 5 \
    --paths ${PARENT}/32gpus  