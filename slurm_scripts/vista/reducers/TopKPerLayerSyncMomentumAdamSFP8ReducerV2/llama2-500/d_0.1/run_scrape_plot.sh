#!/bin/bash
set -euo pipefail

# activate virtual env for plotting
source /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate

ml cuda/12.4

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/scraper.py \
    --log_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1/llama2_500M_AdamS_d_0.1_comp_start_step_2000_d_warmup_steps_3000_lr_warmup_steps_2000.out \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1 \
    --mode train

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/scraper.py \
    --log_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1/llama2_500M_AdamS_d_0.1_comp_start_step_2000_d_warmup_steps_3000_lr_warmup_steps_2000.out \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1 \
    --mode validation

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/plot.py \
    --title 'Pretrain Llama2-500M' \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1/log_train_PP1_TP1_DP4.csv \
    --legends "Sparsified Top-\$k\$ grad" \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M/log_train_PP1_TP1_DP4.csv \
    --legends "Baseline (dense grad)" \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1 \
    --plot_interval 1 \
    --model GPT-345M \
    --mode train \
    --column_key loss 

python3 /home1/09308/zhengmk/work/optimus-cc/TBD/plot_loss_curves/plot.py \
    --title 'Pretrain Llama2-500M' \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1/log_val_PP1_TP1_DP4.csv \
    --legends "Sparsified grad" \
    --csv_files /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/baseline/AdamS/llama2-H2O-500M/log_val_PP1_TP1_DP4.csv \
    --legends "Baseline (dense all-reduce grad)" \
    --output_dir /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/slurm_scripts/vista/reducers/TopKPerLayerSyncMomentumAdamSFP8ReducerV2/llama2-500/d_0.1 \
    --plot_interval 1 \
    --model GPT-345M \
    --mode validation \
    --column_key loss 