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


# DATA_PATH=<Specify path and file prefix>_text_document
# NOTE: This should point to data preprocessed with the Llama2 tokenizer.
DATA_PATH="${DATA_PATH:-/path/to/processed/slimpajama/data}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-2-7b-hf}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/ckpts}"

# define hp for topk reducer
density=0.1
start_step=0 # try using dense for first 200 steps
start_density=$density
density_warmup_steps=0 # then warmup for 00 steps

lr_warmup_steps=2000

global_batch_size=1024
echo "global_batch_size=${global_batch_size}"

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
$CONTAINER_CMD env CC="$CC" CXX="$CXX" CUDAHOSTCXX="$CUDAHOSTCXX" \
  torchrun $TORCHRUN_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size $TENSOR_PARALLEL \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL \
    --num-layers 24 \
    --hidden-size 2560 \
    --ffn-hidden-size 6912 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 80 \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --rotary-percent 1.0 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --micro-batch-size 8 \
    --global-batch-size $global_batch_size \
    --seq-length 2048 \
    --max-position-embeddings 16384 \
    --train-iters 110 \
    --lr-warmup-iters 0 \
    --data-path $DATA_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-size 128256 \
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
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --use-flash-attn \
    --bf16 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-distributed-optimizer \
    --log-interval 1 \
    --log-memory-interval 1 \
    --timing-log-level 2 \
    --timing-log-option minmax \
    > >(tee -a llama2_1.8B_AdamS_baseline_${timestamp}.out) \
    2> >(tee -a llama2_1.8B_AdamS_baseline_${timestamp}.err >&2)
