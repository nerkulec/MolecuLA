#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array job or set SLURM_ARRAY_TASK_ID}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/lignin_solubility/full_shards}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-100000}"
NONAR_BATCH_SIZE="${NONAR_BATCH_SIZE:-1024}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-128}"

SHARD_ID="${SLURM_ARRAY_TASK_ID}"
START_ROWID=$((SHARD_ID * ROWS_PER_SHARD + 1))
END_ROWID=$(((SHARD_ID + 1) * ROWS_PER_SHARD))
SHARD_DIR=$(printf "%s/shard_%05d" "${OUTPUT_ROOT}" "${SHARD_ID}")

mkdir -p "${SHARD_DIR}"

"${PYTHON_BIN}" scripts/preprocess_lignin_solubility.py \
  --start-rowid "${START_ROWID}" --end-rowid "${END_ROWID}" \
  --output-dir "${SHARD_DIR}/preprocess"

"${PYTHON_BIN}" scripts/tokenize_lignin_nonar.py \
  --rows "${SHARD_DIR}/preprocess/rows.csv.gz" \
  --output-dir "${SHARD_DIR}/tokenization/nonar"

"${PYTHON_BIN}" scripts/tokenize_lignin_ar.py \
  --rows "${SHARD_DIR}/preprocess/rows.csv.gz" \
  --output-dir "${SHARD_DIR}/tokenization/ar"

"${PYTHON_BIN}" scripts/combine_lignin_eligibility.py \
  --nonar-mask "${SHARD_DIR}/tokenization/nonar/eligible_mask.npy" \
  --ar-mask "${SHARD_DIR}/tokenization/ar/eligible_mask.npy" \
  --rows "${SHARD_DIR}/preprocess/rows.csv.gz" \
  --output "${SHARD_DIR}/common_eligible.npy"

for MODEL in linear_attention simple_attention; do
  "${PYTHON_BIN}" scripts/encode_decode_lignin.py \
    --model "${MODEL}" \
    --rows "${SHARD_DIR}/preprocess/rows.csv.gz" \
    --tokens "${SHARD_DIR}/tokenization/nonar/tokens.npy" \
    --eligible-mask "${SHARD_DIR}/common_eligible.npy" \
    --tokenizer "${SHARD_DIR}/tokenization/nonar/tokenizer.json" \
    --output-dir "${SHARD_DIR}/models/${MODEL}" \
    --device cuda --batch-size "${NONAR_BATCH_SIZE}"
done

"${PYTHON_BIN}" scripts/encode_decode_lignin.py \
  --model autoregressive \
  --rows "${SHARD_DIR}/preprocess/rows.csv.gz" \
  --tokens "${SHARD_DIR}/tokenization/ar/tokens.npy" \
  --eligible-mask "${SHARD_DIR}/common_eligible.npy" \
  --tokenizer "${SHARD_DIR}/tokenization/ar/tokenizer.json" \
  --output-dir "${SHARD_DIR}/models/autoregressive" \
  --device cuda --batch-size "${AR_BATCH_SIZE}"
