#!/bin/bash


export HF_TOKEN=$(cat ~/hf_token)
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"

GPUS_PER_NODE=1

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must run inside a Slurm allocation."
  exit 1
fi

NNODES=${SLURM_NNODES:-1}
export GPUS_PER_NODE=$GPUS_PER_NODE
export MASTER_PORT=12345
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT=${MASTER_PORT}"

ml tacc-apptainer
CONTAINER_CMD="apptainer exec --nv --bind /work --bind /scratch --bind /home1/09308/zhengmk --fakeroot /work/09308/zhengmk/optimus-cc/TBD/apptainer_images/pytorch_26.02-py3.sif"

# AWS OFI NCCL plugin on this image is not loadable (missing libs/version mismatch).
# Force NCCL socket transport for stability.
export NCCL_NET=Socket
unset NCCL_NET_PLUGIN
unset NCCL_NET_OFI_PROVIDER
unset FI_PROVIDER
unset AWS_OFI_NCCL_VERSION
unset EFA_VERSION
export NCCL_IB_DISABLE=1

echo "NCCL_NET=${NCCL_NET:-<unset>}"
echo "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-<unset>}"

# Change for multinode config
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
DATA_PARALLEL=$(($WORLD_SIZE/$PIPELINE_PARALLEL/$TENSOR_PARALLEL))

# define hp for topk reducer
density=0.1
start_step=2000 # try using dense for first 200 steps
start_density=1
density_warmup_steps=8000 # then warmup for 00 steps

lr_warmup_steps=2000

# DATA_PATH=<Specify path and file prefix>_text_document
# NOTE: This should point to data preprocessed with the Llama2 tokenizer.
DATA_PATH=/scratch/09308/zhengmk/slimpajama6b/slimpajama6b_llama2_text_document
TOKENIZER_MODEL=meta-llama/Llama-2-7b-hf
CHECKPOINT_PATH=$SCRATCH/megatron-lm_checkpoints/llama2_danube3_500M_model_bf16_gradient_fp32_AdamS_topk_sparsify_d_${density}_comp_start_step_${start_step}_density_warmup_${density_warmup_steps}_lr_warmup_${lr_warmup_steps}


global_batch_size=1024
echo "global_batch_size=${global_batch_size}"


export WANDB_API_KEY=$(cat ~/wandb_key)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_PROJECT="${WANDB_PROJECT:-SCAPE}"
  WANDB_EXP_NAME="${WANDB_EXP_NAME:-llama2_500M_AdamS_d_${density}_comp_start_step_${start_step}_d_warmup_steps_${density_warmup_steps}_lr_warmup_steps_${lr_warmup_steps}}"
  WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-${CHECKPOINT_PATH}/wandb}"
  WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
  WANDB_ARGS+=(--wandb-exp-name "$WANDB_EXP_NAME")
  WANDB_ARGS+=(--wandb-save-dir "$WANDB_SAVE_DIR")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  echo "W&B logging enabled: project=${WANDB_PROJECT}, exp=${WANDB_EXP_NAME}, dir=${WANDB_SAVE_DIR}"
else
  echo "W&B logging disabled (set WANDB_API_KEY to enable)."
fi

set +x 

TORCHRUN_ARGS="--nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${SLURM_NODEID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}" 
echo "TORCHRUN_ARGS=${TORCHRUN_ARGS}"

if [[ -z "${SLURM_PROCID:-}" ]]; then
  echo "Launching one worker task per node with srun (${NNODES} nodes)..."
  exec srun --ntasks="$NNODES" --ntasks-per-node=1 --nodes="$NNODES" bash "$0" "$@"
fi

NODE_RANK=${SLURM_PROCID:-${SLURM_NODEID:-0}}
echo "SLURM_PROCID=${SLURM_PROCID:-unset} SLURM_NODEID=${SLURM_NODEID:-unset} node_rank=${NODE_RANK}"

$CONTAINER_CMD env CC="$CC" CXX="$CXX" CUDAHOSTCXX="$CUDAHOSTCXX" \
  torchrun $TORCHRUN_ARGS /home1/09308/zhengmk/work/optimus-cc/Megatron-LM/pretrain_gpt.py \
    --tensor-model-parallel-size $TENSOR_PARALLEL \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL \
    --num-layers 16 \
    --hidden-size 1536 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 96 \
    --position-embedding-type rope \
    --rotary-base 100000 \
    --rotary-percent 1.0 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --micro-batch-size 16 \
    --global-batch-size $global_batch_size \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --train-iters 100000 \
    --lr-warmup-iters $lr_warmup_steps \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-size 32000 \
    --split 949,50,1 \
    --distributed-backend nccl \
    --optimizer adams \
    --lr 3e-4 \
    --lr-decay-style cosine \
    --min-lr 3e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --use-flash-attn \
    --bf16 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-topk-adams-reducer \
    --topk-adams-density $density \
    --topk-adams-start-iter $start_step \
    --topk-adams-density-start $start_density \
    --topk-adams-density-warmup-steps $density_warmup_steps \
    --move-clip-grad-to-reducer \
    "${WANDB_ARGS[@]}" \
    > >(tee -a llama2_500M_AdamS_d_${density}_comp_start_step_${start_step}_d_warmup_steps_${density_warmup_steps}_lr_warmup_steps_${lr_warmup_steps}.out) \
    2> >(tee -a llama2_500M_AdamS_d_${density}_comp_start_step_${start_step}_d_warmup_steps_${density_warmup_steps}_lr_warmup_steps_${lr_warmup_steps}.err >&2)
    
