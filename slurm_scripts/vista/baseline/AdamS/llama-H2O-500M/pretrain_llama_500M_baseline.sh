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
STRICT_RESUME="${STRICT_RESUME:-1}"

lr_warmup_steps=2000

global_batch_size=1024
echo "global_batch_size=${global_batch_size}"

RESUME_ARGS=(--load "$CHECKPOINT_PATH")
if [[ "${STRICT_RESUME}" == "1" ]]; then
  RESUME_ARGS+=(--exit-on-missing-checkpoint)
fi

echo "STRICT_RESUME=${STRICT_RESUME}"

export WANDB_API_KEY="${WANDB_API_KEY:-$(cat "${WANDB_API_KEY_FILE:-$HOME/wandb_key}" 2>/dev/null || true)}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_PROJECT="${WANDB_PROJECT:-SCAPE}"
  WANDB_EXP_NAME="${WANDB_EXP_NAME:-llama2_500M_AdamS_d_${density}_comp_start_step_${start_step}_d_warmup_steps_${density_warmup_steps}_lr_warmup_steps_${lr_warmup_steps}}"
  WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-${CHECKPOINT_PATH}/wandb}"

  # Auto-resume W&B run/curve when restarting from the same checkpoint path.
  # Override behavior with:
  #   WANDB_AUTO_RESUME=0    -> always start a fresh run
  #   WANDB_RUN_ID=<run_id>  -> force a specific run id
  # WANDB_AUTO_RESUME="${WANDB_AUTO_RESUME:-0}"
  # if [[ "${WANDB_AUTO_RESUME,,}" != "0" && "${WANDB_AUTO_RESUME,,}" != "false" ]]; then
  #   if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  #     latest_run_link="${WANDB_SAVE_DIR}/wandb/latest-run"
  #     latest_run_name=""
  #     if [[ -L "${latest_run_link}" ]]; then
  #       latest_run_name="$(basename "$(readlink -f "${latest_run_link}")")"
  #     else
  #       latest_run_name="$(ls -1dt "${WANDB_SAVE_DIR}"/wandb/run-* "${WANDB_SAVE_DIR}"/wandb/offline-run-* 2>/dev/null | head -n 1 | xargs -r basename)"
  #     fi

  #     if [[ -n "${latest_run_name}" ]]; then
  #       inferred_run_id="$(echo "${latest_run_name}" | sed -E 's/^(offline-)?run-[0-9_]+-([A-Za-z0-9]+)$/\2/')"
  #       if [[ "${inferred_run_id}" != "${latest_run_name}" && -n "${inferred_run_id}" ]]; then
  #         export WANDB_RUN_ID="${inferred_run_id}"
  #       fi
  #     fi
  #   fi

  #   if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  #     export WANDB_RESUME="${WANDB_RESUME:-allow}"
  #     echo "W&B resume enabled: run_id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"
  #   fi
  # fi

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
    --train-iters 100000 \
    --lr-warmup-iters $lr_warmup_steps \
    --save $CHECKPOINT_PATH \
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
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --use-flash-attn \
    --bf16 \
    --accumulate-allreduce-grads-in-fp32 \
    "${WANDB_ARGS[@]}" \
    > >(tee -a llama2_500M_AdamS_${timestamp}.out) \
    2> >(tee -a llama2_500M_AdamS_${timestamp}.err >&2)
