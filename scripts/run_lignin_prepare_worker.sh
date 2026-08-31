#!/usr/bin/env bash
# Claim and process shared preparation shards until the global stage is complete.
set -euo pipefail

: "${STAGE:?}"
: "${WORKER_SLOT:?}"
: "${REPO_DIR:?}"
: "${PYTHON_BIN:?}"
: "${DB_PATH:?}"
: "${OUTPUT_ROOT:?}"
: "${TOTAL_ROWS:?}"
: "${ROWS_PER_SHARD:?}"
: "${SHARD_COUNT:?}"
: "${CLAIM_STALE_SECONDS:?}"
: "${HEARTBEAT_SECONDS:?}"

WORKER_KEY="$(hostname)-$$-${WORKER_SLOT}"
CURRENT_CLAIM=""
HEARTBEAT_PID=""

release_claim() {
  local claim="$1"
  [[ -d "$claim" ]] || return 0
  find "$claim" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
  rmdir "$claim" 2>/dev/null || true
}

cleanup() {
  if [[ -n "$HEARTBEAT_PID" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
  fi
  [[ -z "$CURRENT_CLAIM" ]] || release_claim "$CURRENT_CLAIM"
}
trap cleanup EXIT INT TERM

marker_valid() {
  local shard_dir="$1" marker
  if [[ "$STAGE" == "preprocess" ]]; then
    [[ -f "$shard_dir/.complete" ]]
  else
    marker="$shard_dir/.tokenizer_sha256"
    [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$TOKENIZER_SHA" ]]
  fi
}

claim_is_stale() {
  local claim="$1" heartbeat="$claim/heartbeat" modified now
  [[ -e "$heartbeat" ]] || heartbeat="$claim"
  modified="$(stat -c %Y "$heartbeat" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  (( now - modified > CLAIM_STALE_SECONDS ))
}

try_claim() {
  local claim="$1" stale
  if mkdir "$claim" 2>/dev/null; then
    printf '%s\n' "$WORKER_KEY" > "$claim/owner"
    touch "$claim/heartbeat"
    return 0
  fi
  claim_is_stale "$claim" || return 1
  stale="${claim}.stale.${WORKER_KEY}"
  if mv "$claim" "$stale" 2>/dev/null; then
    find "$stale" -depth -delete 2>/dev/null || true
    if mkdir "$claim" 2>/dev/null; then
      printf '%s\n' "$WORKER_KEY" > "$claim/owner"
      touch "$claim/heartbeat"
      return 0
    fi
  fi
  return 1
}

heartbeat() {
  local claim="$1"
  while [[ -d "$claim" ]]; do
    touch "$claim/heartbeat" 2>/dev/null || return
    sleep "$HEARTBEAT_SECONDS"
  done
}

process_shard() {
  local shard_id="$1" shard_name shard_dir claim log_path status start_rowid end_rowid marker_tmp
  shard_name="$(printf 'shard_%03d' "$shard_id")"
  if [[ "$STAGE" == "preprocess" ]]; then
    shard_dir="$OUTPUT_ROOT/preprocessed/$shard_name"
    log_path="$shard_dir/preprocess.log"
  else
    shard_dir="$OUTPUT_ROOT/encoded/$shard_name"
    log_path="$shard_dir/encode.log"
  fi
  mkdir -p "$shard_dir"
  marker_valid "$shard_dir" && return 1
  [[ ! -f "$shard_dir/.failed" ]] || return 1
  claim="$shard_dir/.claim"
  try_claim "$claim" || return 1
  CURRENT_CLAIM="$claim"
  heartbeat "$claim" &
  HEARTBEAT_PID=$!

  set +e
  if [[ "$STAGE" == "preprocess" ]]; then
    start_rowid=$((shard_id * ROWS_PER_SHARD + 1))
    end_rowid=$(((shard_id + 1) * ROWS_PER_SHARD))
    (( end_rowid <= TOTAL_ROWS )) || end_rowid="$TOTAL_ROWS"
    "$PYTHON_BIN" "$REPO_DIR/scripts/preprocess_lignin_solubility.py" \
      --db "$DB_PATH" --output-dir "$shard_dir" \
      --start-rowid "$start_rowid" --end-rowid "$end_rowid" --no-progress \
      > "$log_path" 2>&1
    status=$?
  else
    "$PYTHON_BIN" "$REPO_DIR/scripts/encode_lignin_training_shard.py" \
      --rows "$OUTPUT_ROOT/preprocessed/$shard_name/rows.csv.gz" \
      --tokenizer "$OUTPUT_ROOT/unified_tokenizer.json" \
      --output-dir "$shard_dir" --split-by "$SPLIT_BY" --split-seed "$SPLIT_SEED" \
      > "$log_path" 2>&1
    status=$?
  fi
  set -e

  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  HEARTBEAT_PID=""
  if (( status == 0 )); then
    rm -f "$shard_dir/.failed"
    if [[ "$STAGE" == "preprocess" ]]; then
      marker_tmp="$shard_dir/.complete.${WORKER_KEY}.tmp"
      touch "$marker_tmp"
      mv "$marker_tmp" "$shard_dir/.complete"
    else
      marker_tmp="$shard_dir/.tokenizer_sha256.${WORKER_KEY}.tmp"
      printf '%s\n' "$TOKENIZER_SHA" > "$marker_tmp"
      mv "$marker_tmp" "$shard_dir/.tokenizer_sha256"
    fi
  else
    marker_tmp="$shard_dir/.failed.${WORKER_KEY}.tmp"
    printf 'worker=%s\nexit_code=%s\nlog=%s\n' "$WORKER_KEY" "$status" "$log_path" > "$marker_tmp"
    mv "$marker_tmp" "$shard_dir/.failed"
  fi
  release_claim "$claim"
  CURRENT_CLAIM=""
  return 0
}

start_index=$((WORKER_SLOT % SHARD_COUNT))
while true; do
  acquired=0
  incomplete=0
  failed=0
  for ((offset = 0; offset < SHARD_COUNT; offset++)); do
    shard_id=$(((start_index + offset) % SHARD_COUNT))
    shard_name="$(printf 'shard_%03d' "$shard_id")"
    if [[ "$STAGE" == "preprocess" ]]; then
      shard_dir="$OUTPUT_ROOT/preprocessed/$shard_name"
    else
      shard_dir="$OUTPUT_ROOT/encoded/$shard_name"
    fi
    if marker_valid "$shard_dir"; then
      continue
    fi
    incomplete=1
    if [[ -f "$shard_dir/.failed" ]]; then
      failed=1
      continue
    fi
    if process_shard "$shard_id"; then
      acquired=1
      start_index=$(((shard_id + 1) % SHARD_COUNT))
      break
    fi
  done
  (( failed == 0 )) || exit 1
  (( incomplete != 0 )) || exit 0
  (( acquired != 0 )) || sleep 5
done
