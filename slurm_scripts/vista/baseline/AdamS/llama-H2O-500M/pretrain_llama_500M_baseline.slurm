#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=08:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 16
#SBATCH -n 16
#SBATCH -p YOUR_PARTITION
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS
#SBATCH --mail-type=all


CONTAINER_CMD="apptainer exec --nv --bind /path/to/repo --bind /path/to/data --bind /path/to/ckpts --fakeroot /path/to/pytorch_26.01-py3.sif"

export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_TOKEN_FILE:-$HOME/hf_token}" 2>/dev/null || true)}"
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"

export GPUS_PER_NODE=4
export MASTER_PORT=12346
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

# Change for multinode config
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
DATA_PARALLEL=$(($WORLD_SIZE/$PIPELINE_PARALLEL/$TENSOR_PARALLEL))

echo "TENSOR_PARALLEL=${TENSOR_PARALLEL}"
echo "PIPELINE_PARALLEL=${PIPELINE_PARALLEL}"
echo "DATA_PARALLEL=${DATA_PARALLEL}"


# DATA_PATH=<Specify path and file prefix>_text_document
# NOTE: This should point to data preprocessed with the Llama2 tokenizer.
DATA_PATH="${DATA_PATH:-/path/to/processed/slimpajama/data}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-2-7b-hf}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/ckpts}"


global_batch_size=1024
echo "global_batch_size=${global_batch_size}"

set +x 

TORCHRUN_ARGS="--nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${SLURM_NODEID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}" 
echo "TORCHRUN_ARGS=${TORCHRUN_ARGS}"

srun --nodes="${SLURM_NNODES}" \
     --ntasks="${SLURM_NNODES}" \
     --ntasks-per-node=1 \
     --cpu_bind=cores \
     --gpus-per-node=${GPUS_PER_NODE} \
     $CONTAINER_CMD env CC="$CC" CXX="$CXX" CUDAHOSTCXX="$CUDAHOSTCXX" \
       torchrun $TORCHRUN_ARGS pretrain_gpt.py \
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
       --train-iters 10000 \
       --lr-warmup-iters 500 \
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
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 1 \
       --save-interval 5000 \
       --eval-interval 100 \
       --eval-iters 10 \
       --use-flash-attn \
       --bf16 \
       --accumulate-allreduce-grads-in-fp32
