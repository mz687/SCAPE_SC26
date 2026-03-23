#!/bin/bash
set -euo pipefail

# activate virtual env for plotting
source /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate

ml cuda/12.4

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/scraper.py \
    --log_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M/llama2_500M_AdamS.out \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M \
    --mode train

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/scraper.py \
    --log_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M/llama2_500M_AdamS.out \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M \
    --mode validation

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/plot.py \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M/log_train_PP1_TP1_DP4.csv \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M \
    --plot_interval 1 \
    --model GPT-345M \
    --mode train \
    --column_key loss 

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/plot.py \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M/log_val_PP1_TP1_DP4.csv \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M \
    --plot_interval 1 \
    --model GPT-345M \
    --mode validation \
    --column_key loss 
