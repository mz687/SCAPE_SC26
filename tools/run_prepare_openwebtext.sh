#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_SCRIPT="$REPO_ROOT/tools/download_openwebtext.py"
PREPROCESS_SCRIPT="$REPO_ROOT/tools/preprocess_data.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_DIR="${RAW_DIR:-/path/to/raw/openwebtext}"
RAW_JSONL="${RAW_JSONL:-${RAW_DIR}/openwebtext_text.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/path/to/processed/openwebtext/openwebtext_gpt2}"
GPT2_VOCAB_FILE="${GPT2_VOCAB_FILE:-/path/to/gpt2-vocab.json}"
GPT2_MERGE_FILE="${GPT2_MERGE_FILE:-/path/to/gpt2-merges.txt}"
DATASET_NAME="${DATASET_NAME:-Skylion007/openwebtext}"
DATASET_CONFIG="${DATASET_CONFIG:-plain_text}"
SPLIT="${SPLIT:-train}"
WORKERS="${WORKERS:-32}"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

if [[ "$RAW_DIR" == /path/to/* ]]; then
  echo "Set RAW_DIR to a real writable directory before running this script." >&2
  exit 1
fi
if [[ "$OUTPUT_PREFIX" == /path/to/* ]]; then
  echo "Set OUTPUT_PREFIX to a real output prefix before running this script." >&2
  exit 1
fi
if [[ "$GPT2_VOCAB_FILE" == /path/to/* ]] || [[ ! -f "$GPT2_VOCAB_FILE" ]]; then
  echo "Set GPT2_VOCAB_FILE to a real GPT-2 vocab file before running this script." >&2
  exit 1
fi
if [[ "$GPT2_MERGE_FILE" == /path/to/* ]] || [[ ! -f "$GPT2_MERGE_FILE" ]]; then
  echo "Set GPT2_MERGE_FILE to a real GPT-2 merges file before running this script." >&2
  exit 1
fi
if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
  echo "Downloader not found: $DOWNLOAD_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
  echo "Preprocess script not found: $PREPROCESS_SCRIPT" >&2
  exit 1
fi

mkdir -p "$RAW_DIR"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

DOWNLOAD_ARGS=(
  --output-jsonl "$RAW_JSONL"
  --dataset-name "$DATASET_NAME"
  --split "$SPLIT"
)
if [[ -n "$DATASET_CONFIG" ]]; then
  DOWNLOAD_ARGS+=(--dataset-config "$DATASET_CONFIG")
fi

echo "Preparing OpenWebText export at $RAW_JSONL"
"$PYTHON_BIN" "$DOWNLOAD_SCRIPT" "${DOWNLOAD_ARGS[@]}"

echo "Preprocessing OpenWebText export with GPT-2 BPE tokenizer"
cd "$REPO_ROOT"
"$PYTHON_BIN" "$PREPROCESS_SCRIPT" \
  --input "$RAW_JSONL" \
  --output-prefix "$OUTPUT_PREFIX" \
  --json-keys text \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file "$GPT2_VOCAB_FILE" \
  --merge-file "$GPT2_MERGE_FILE" \
  --workers "$WORKERS" \
  --append-eod

echo "Prepared DATA_PATH=${OUTPUT_PREFIX}_text_document"
