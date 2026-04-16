#!/usr/bin/env bash
# set -euo pipefail

ml tacc-apptainer

CKPT_ROOT=${CKPT_ROOT:-/path/to/torch/dist}
HF_OUT=${HF_OUT:-${CKPT_ROOT}_hf}
IMG=${IMG:-/path/to/pytorch_26.02-py3.sif}
MEGATRON=${MEGATRON:-/path/to/SCAPE/root}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/hf_token}

if [[ ! -f "$HF_TOKEN_FILE" ]]; then
  echo "ERROR: HF token file not found: $HF_TOKEN_FILE" >&2
  exit 1
fi

if [[ ! -f "$CKPT_ROOT/latest_checkpointed_iteration.txt" ]]; then
  echo "ERROR: missing $CKPT_ROOT/latest_checkpointed_iteration.txt" >&2
  exit 1
fi

mkdir -p "$HF_OUT"

apptainer exec --nv --fakeroot \
  --bind /work --bind /scratch --bind /home1/09308/zhengmk \
  "$IMG" bash -lc "
set -euo pipefail
export PYTHONPATH=$MEGATRON:\${PYTHONPATH:-}
export HF_TOKEN=\$(cat \"$HF_TOKEN_FILE\")

torchrun --nproc_per_node=1 $MEGATRON/examples/post_training/modelopt/export.py \
  --load $CKPT_ROOT \
  --ckpt-format torch_dist \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 16 \
  --hidden-size 1536 \
  --ffn-hidden-size 4096 \
  --num-attention-heads 16 \
  --group-query-attention \
  --num-query-groups 8 \
  --kv-channels 96 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --position-embedding-type rope \
  --rotary-base 100000 \
  --normalization RMSNorm \
  --swiglu \
  --untie-embeddings-and-output-weights \
  --disable-bias-linear \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model meta-llama/Llama-2-7b-hf \
  --vocab-size 32000 \
  --make-vocab-size-divisible-by 128 \
  --transformer-impl transformer_engine \
  --bf16 \
  --micro-batch-size 1 \
  --no-load-optim --no-load-rng \
  --pretrained-model-name meta-llama/Llama-2-7b-hf \
  --export-te-mcore-model \
  --export-dir $HF_OUT
"
