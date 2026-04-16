#!/bin/bash


export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_TOKEN_FILE:-$HOME/hf_token}" 2>/dev/null || true)}"
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
CONTAINER_CMD="apptainer exec --nv --bind /path/to/repo --bind /path/to/data --bind /path/to/ckpts --fakeroot /path/to/pytorch_26.01-py3.sif"


# Change for multinode config
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
DATA_PARALLEL=$(($WORLD_SIZE/$PIPELINE_PARALLEL/$TENSOR_PARALLEL))

export HF_HOME=/tmp/$USER/hf_home_${SLURM_JOB_ID}_${SLURMD_NODENAME}
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# define hp for topk reducer
density=0.01
start_step=0 # try using dense for first 200 steps
start_density=1
density_warmup_steps=10000 # then warmup for 00 steps

lr_warmup_steps=5000
global_batch_size=512
echo "global_batch_size=${global_batch_size}"

# DATA_PATH=<Specify path and file prefix>_text_document
# NOTE: This should point to data preprocessed with the Llama2 tokenizer.
# DATA_PATH=/path/to/preprocessed/openwebtext/data
# TOKENIZER_MODEL=meta-llama/Llama-2-7b-hf
# CHECKPOINT_PATH=/path/to/ckpts
DATA_PATH="${DATA_PATH:-/path/to/preprocessed/openwebtext/data}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/ckpts}"
GPT2_VOCAB_FILE="${GPT2_VOCAB_FILE:-/path/to/gpt2-vocab.json}"
GPT2_MERGE_FILE="${GPT2_MERGE_FILE:-/path/to/gpt2-merges.txt}"


# export WANDB_API_KEY=$(cat ~/wandb_key)

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

timestamp=$(date +%s)
log_prefix="GPT_345M_AdamS_d_${density}_comp_start_step_${start_step}_d_warmup_steps_${density_warmup_steps}_lr_warmup_steps_${lr_warmup_steps}_${timestamp}"


$CONTAINER_CMD env CC="$CC" CXX="$CXX" CUDAHOSTCXX="$CUDAHOSTCXX" \
  torchrun $TORCHRUN_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size $TENSOR_PARALLEL \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 8 \
    --global-batch-size $global_batch_size \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 100000 \
    --lr-warmup-iters 5000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --use-checkpoint-lr-scheduler \
    --data-path $DATA_PATH \
    --vocab-file $GPT2_VOCAB_FILE \
    --merge-file $GPT2_MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --optimizer adams \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --activations-checkpoint-method uniform \
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --accumulate-allreduce-grads-in-fp32 \
    --bf16 \
    --use-flash-attn \
    --use-topk-adams-reducer \
    --topk-adams-density $density \
    --topk-adams-start-iter $start_step \
    --topk-adams-density-start $start_density \
    --topk-adams-density-warmup-steps $density_warmup_steps \
    --no-topk-adams-use-exclude-from-topk \
    "${WANDB_ARGS[@]}" \
     > >(tee -a "${log_prefix}.out") \
     2> >(tee -a "${log_prefix}.err" >&2)
