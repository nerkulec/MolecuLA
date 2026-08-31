#!/usr/bin/env bash
# Cooperative, restartable lignin preparation on any number of shared-filesystem nodes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MOLECULA_PYTHON:-python}"
DB_PATH="${DB_PATH:-$REPO_DIR/data/lignin_solubility.db}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/artifacts/lignin_retraining}"
REQUESTED_ROWS_PER_SHARD="${ROWS_PER_SHARD-}"
REQUESTED_SPLIT_BY="${SPLIT_BY-}"
REQUESTED_SPLIT_SEED="${SPLIT_SEED-}"
REQUESTED_MAX_ROWS="${MAX_ROWS-}"
AVAILABLE_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
PREPROCESS_JOBS="${PREPROCESS_JOBS:-$AVAILABLE_CPUS}"
ENCODE_JOBS="${ENCODE_JOBS:-$AVAILABLE_CPUS}"
CLAIM_STALE_SECONDS="${CLAIM_STALE_SECONDS:-1800}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"

for value in "$PREPROCESS_JOBS" "$ENCODE_JOBS" "$CLAIM_STALE_SECONDS" "$HEARTBEAT_SECONDS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Worker and timeout values must be positive integers" >&2; exit 2; }
done

cd "$REPO_DIR"
command -v "$PYTHON_BIN" >/dev/null
if [[ "$DB_PATH" == "$REPO_DIR/data/lignin_solubility.db" ]] && [[ ! -f "$DB_PATH" ]]; then
  bash "$REPO_DIR/assemble_lignin_database.sh"
fi
[[ -f "$DB_PATH" ]] || { echo "Missing database: $DB_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/preprocessed" "$OUTPUT_ROOT/encoded" "$OUTPUT_ROOT/state"

lock_is_stale() {
  local lock="$1" heartbeat="$lock/heartbeat" modified now
  [[ -e "$heartbeat" ]] || heartbeat="$lock"
  modified="$(stat -c %Y "$heartbeat" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  (( now - modified > CLAIM_STALE_SECONDS ))
}

try_global_lock() {
  local lock="$1" stale
  if mkdir "$lock" 2>/dev/null; then
    touch "$lock/heartbeat"
    return 0
  fi
  lock_is_stale "$lock" || return 1
  stale="${lock}.stale.$(hostname).$$"
  if mv "$lock" "$stale" 2>/dev/null; then
    find "$stale" -depth -delete 2>/dev/null || true
    mkdir "$lock" 2>/dev/null || return 1
    touch "$lock/heartbeat"
    return 0
  fi
  return 1
}

release_global_lock() {
  local lock="$1"
  find "$lock" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
  rmdir "$lock" 2>/dev/null || true
}

global_heartbeat() {
  local lock="$1"
  while [[ -d "$lock" ]]; do
    touch "$lock/heartbeat" 2>/dev/null || return
    sleep "$HEARTBEAT_SECONDS"
  done
}

DATABASE_ROWS="$($PYTHON_BIN -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("select count(*) from functionalized_lignins").fetchone()[0])' "$DB_PATH")"
STATE_FILE="$OUTPUT_ROOT/state/prepare.env"
STATE_LOCK="$OUTPUT_ROOT/state/prepare.lock"

# The first node fixes the shard layout and split configuration. Later nodes
# adopt it automatically, preventing incompatible jobs from sharing a queue.
while [[ ! -f "$STATE_FILE" ]]; do
  if try_global_lock "$STATE_LOCK"; then
    ROWS_PER_SHARD="${REQUESTED_ROWS_PER_SHARD:-100000}"
    SPLIT_BY="${REQUESTED_SPLIT_BY:-rowid}"
    SPLIT_SEED="${REQUESTED_SPLIT_SEED:-42}"
    TOTAL_ROWS="$DATABASE_ROWS"
    if [[ -n "$REQUESTED_MAX_ROWS" ]] && (( REQUESTED_MAX_ROWS < TOTAL_ROWS )); then
      TOTAL_ROWS="$REQUESTED_MAX_ROWS"
    fi
    [[ "$ROWS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] || { echo "ROWS_PER_SHARD must be positive" >&2; exit 2; }
    [[ "$SPLIT_SEED" =~ ^[0-9]+$ ]] || { echo "SPLIT_SEED must be non-negative" >&2; exit 2; }
    [[ "$SPLIT_BY" == "rowid" || "$SPLIT_BY" == "scaffold" ]] || { echo "Invalid SPLIT_BY=$SPLIT_BY" >&2; exit 2; }
    SHARD_COUNT=$(((TOTAL_ROWS + ROWS_PER_SHARD - 1) / ROWS_PER_SHARD))
    state_tmp="$OUTPUT_ROOT/state/prepare.env.$$.tmp"
    printf 'TOTAL_ROWS=%s\nROWS_PER_SHARD=%s\nSHARD_COUNT=%s\nSPLIT_BY=%s\nSPLIT_SEED=%s\n' \
      "$TOTAL_ROWS" "$ROWS_PER_SHARD" "$SHARD_COUNT" "$SPLIT_BY" "$SPLIT_SEED" > "$state_tmp"
    mv "$state_tmp" "$STATE_FILE"
    release_global_lock "$STATE_LOCK"
  else
    sleep 1
  fi
done
# shellcheck disable=SC1090
source "$STATE_FILE"

[[ "$DATABASE_ROWS" -ge "$TOTAL_ROWS" ]] || { echo "Database now has fewer rows than the shared queue state" >&2; exit 1; }
if [[ -n "$REQUESTED_ROWS_PER_SHARD" && "$REQUESTED_ROWS_PER_SHARD" != "$ROWS_PER_SHARD" ]]; then
  echo "ROWS_PER_SHARD conflicts with established shared state ($ROWS_PER_SHARD)" >&2; exit 2
fi
if [[ -n "$REQUESTED_SPLIT_BY" && "$REQUESTED_SPLIT_BY" != "$SPLIT_BY" ]]; then
  echo "SPLIT_BY conflicts with established shared state ($SPLIT_BY)" >&2; exit 2
fi
if [[ -n "$REQUESTED_SPLIT_SEED" && "$REQUESTED_SPLIT_SEED" != "$SPLIT_SEED" ]]; then
  echo "SPLIT_SEED conflicts with established shared state ($SPLIT_SEED)" >&2; exit 2
fi
if [[ -n "$REQUESTED_MAX_ROWS" && "$REQUESTED_MAX_ROWS" != "$TOTAL_ROWS" ]]; then
  echo "MAX_ROWS conflicts with established shared state ($TOTAL_ROWS)" >&2; exit 2
fi

export REPO_DIR PYTHON_BIN DB_PATH OUTPUT_ROOT TOTAL_ROWS ROWS_PER_SHARD SHARD_COUNT SPLIT_BY SPLIT_SEED
export CLAIM_STALE_SECONDS HEARTBEAT_SECONDS

run_worker_pool() {
  local stage="$1" jobs="$2" tokenizer_sha="${3:-}" monitor_status worker_status=0 pid stage_root
  local -a worker_pids=() monitor_args=()
  export STAGE="$stage" TOKENIZER_SHA="$tokenizer_sha"
  for ((slot = 0; slot < jobs; slot++)); do
    WORKER_SLOT="$slot" bash "$REPO_DIR/scripts/run_lignin_prepare_worker.sh" &
    worker_pids+=("$!")
  done

  stage_root="$OUTPUT_ROOT/encoded"
  [[ "$stage" == "preprocess" ]] && stage_root="$OUTPUT_ROOT/preprocessed"
  monitor_args=(--stage "$stage" --root "$stage_root" --shards "$SHARD_COUNT"
    --total-rows "$TOTAL_ROWS" --rows-per-shard "$ROWS_PER_SHARD")
  [[ "$stage" == "preprocess" ]] || monitor_args+=(--tokenizer-sha "$tokenizer_sha")
  set +e
  "$PYTHON_BIN" "$REPO_DIR/scripts/monitor_lignin_prepare.py" "${monitor_args[@]}"
  monitor_status=$?
  if (( monitor_status != 0 )); then
    kill "${worker_pids[@]}" 2>/dev/null || true
  fi
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || worker_status=$?
  done
  set -e
  (( monitor_status == 0 && worker_status == 0 ))
}

echo "Shared queue: $TOTAL_ROWS molecules, $SHARD_COUNT shards of up to $ROWS_PER_SHARD rows"
echo "This node: $PREPROCESS_JOBS canonicalization workers"
if ! run_worker_pool preprocess "$PREPROCESS_JOBS"; then
  echo "Canonicalization failed; inspect preprocessed/shard_*/preprocess.log and .failed" >&2
  exit 1
fi

ROWS_FILES=("$OUTPUT_ROOT"/preprocessed/shard_*/rows.csv.gz)
[[ ${#ROWS_FILES[@]} -eq "$SHARD_COUNT" ]] || { echo "Expected $SHARD_COUNT row shards, found ${#ROWS_FILES[@]}" >&2; exit 1; }

TOKENIZER="$OUTPUT_ROOT/unified_tokenizer.json"
TOKENIZER_MARKER="$OUTPUT_ROOT/state/tokenizer.sha256"
TOKENIZER_CLAIM="$OUTPUT_ROOT/state/tokenizer.claim"
TOKENIZER_FAILED="$OUTPUT_ROOT/state/tokenizer.failed"
while [[ ! -f "$TOKENIZER_MARKER" ]]; do
  [[ ! -f "$TOKENIZER_FAILED" ]] || { echo "Tokenizer construction failed; see $TOKENIZER_FAILED" >&2; exit 1; }
  if try_global_lock "$TOKENIZER_CLAIM"; then
    echo "Building unified tokenizer on $(hostname)"
    global_heartbeat "$TOKENIZER_CLAIM" &
    tokenizer_heartbeat_pid=$!
    tokenizer_tmp="$OUTPUT_ROOT/unified_tokenizer.$$.tmp"
    set +e
    "$PYTHON_BIN" scripts/build_lignin_training_tokenizer.py --rows "${ROWS_FILES[@]}" --output "$tokenizer_tmp"
    tokenizer_status=$?
    set -e
    if (( tokenizer_status == 0 )); then
      mv "$tokenizer_tmp" "$TOKENIZER"
      tokenizer_sha="$($PYTHON_BIN -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$TOKENIZER")"
      printf '%s\n' "$tokenizer_sha" > "$TOKENIZER_MARKER.$$.tmp"
      mv "$TOKENIZER_MARKER.$$.tmp" "$TOKENIZER_MARKER"
    else
      printf 'host=%s\nexit_code=%s\n' "$(hostname)" "$tokenizer_status" > "$TOKENIZER_FAILED"
    fi
    kill "$tokenizer_heartbeat_pid" 2>/dev/null || true
    wait "$tokenizer_heartbeat_pid" 2>/dev/null || true
    release_global_lock "$TOKENIZER_CLAIM"
  else
    echo "Waiting for another node to finish the unified tokenizer"
    sleep 5
  fi
done
TOKENIZER_SHA="$(<"$TOKENIZER_MARKER")"
actual_tokenizer_sha="$($PYTHON_BIN -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$TOKENIZER")"
[[ "$actual_tokenizer_sha" == "$TOKENIZER_SHA" ]] || { echo "Tokenizer checksum mismatch" >&2; exit 1; }

echo "This node: $ENCODE_JOBS encoding workers"
if ! run_worker_pool encode "$ENCODE_JOBS" "$TOKENIZER_SHA"; then
  echo "Encoding failed; inspect encoded/shard_*/encode.log and .failed" >&2
  exit 1
fi

echo "Training data ready under $OUTPUT_ROOT"
