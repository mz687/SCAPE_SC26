#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_SCRIPT="$REPO_ROOT/tools/download_slimpajama_6b.py"
PREPROCESS_SCRIPT="$REPO_ROOT/tools/preprocess_data.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_DIR="${RAW_DIR:-/path/to/raw/slimpajama6b}"
RAW_JSONL="${RAW_JSONL:-${RAW_DIR}/slimpajama6b_text.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/path/to/processed/slimpajama/slimpajama6b_llama2}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-2-7b-hf}"
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

echo "Preparing full SlimPajama export at $RAW_JSONL"
"$PYTHON_BIN" "$DOWNLOAD_SCRIPT" \
  --output-jsonl "$RAW_JSONL"

echo "Preprocessing SlimPajama export with tokenizer $TOKENIZER_MODEL"
cd "$REPO_ROOT"
"$PYTHON_BIN" "$PREPROCESS_SCRIPT" \
  --input "$RAW_JSONL" \
  --output-prefix "$OUTPUT_PREFIX" \
  --json-keys text \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --workers "$WORKERS" \
  --append-eod

echo "Prepared DATA_PATH=${OUTPUT_PREFIX}_text_document"
