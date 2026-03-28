#!/bin/bash
set -euo pipefail

# activate virtual env for plotting
ml gcc/13.2.0  cuda/12.8 python3/3.11.8
source ~/work/python_vir_envs/vista/fp8/bin/activate


python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/scraper.py \
    --log_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1/GPT_345M_AdamS_d_0.1_comp_start_step_0_d_warmup_steps_10000_lr_warmup_steps_5000_1774501407.out \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1 \
    --mode train

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/scraper.py \
    --log_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1/GPT_345M_AdamS_d_0.1_comp_start_step_0_d_warmup_steps_10000_lr_warmup_steps_5000_1774501407.out \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1 \
    --mode validation

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/plot.py \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1/log_train_PP1_TP1_DP4.csv \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/TBD/slurm_scripts/vista/baseline/AdamS/355M/log_train_PP1_TP1_DP4.csv \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1 \
    --plot_interval 1 \
    --model GPT-345M \
    --mode train \
    --column_key loss 

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/plot.py \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1/log_val_PP1_TP1_DP4.csv \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/TBD/slurm_scripts/vista/baseline/AdamS/355M/log_val_PP1_TP1_DP4.csv \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/gpt/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/d_0.1 \
    --plot_interval 1 \
    --model GPT-345M \
    --mode validation \
    --column_key loss 
