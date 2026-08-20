#!/usr/bin/env bash
# Canonicalize, build the unified vocabulary, and create packed training shards.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MOLECULA_PYTHON:-python}"
DB_PATH="${DB_PATH:-$REPO_DIR/data/lignin_solubility.db}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/artifacts/lignin_retraining}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-100000}"
SPLIT_BY="${SPLIT_BY:-rowid}"
SPLIT_SEED="${SPLIT_SEED:-42}"
AVAILABLE_CPUS="${SLURM_CPUS_PER_TASK:-4}"
PREPROCESS_JOBS="${PREPROCESS_JOBS:-$((AVAILABLE_CPUS / 2))}"
ENCODE_JOBS="${ENCODE_JOBS:-$AVAILABLE_CPUS}"
(( PREPROCESS_JOBS >= 1 )) || PREPROCESS_JOBS=1
(( ENCODE_JOBS >= 1 )) || ENCODE_JOBS=1

cd "$REPO_DIR"
command -v "$PYTHON_BIN" >/dev/null
if [[ "$DB_PATH" == "$REPO_DIR/data/lignin_solubility.db" ]] && [[ ! -f "$DB_PATH" ]]; then
  bash "$REPO_DIR/assemble_lignin_database.sh"
fi
[[ -f "$DB_PATH" ]] || { echo "Missing database: $DB_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/preprocessed" "$OUTPUT_ROOT/encoded"

show_progress() {
  local label="$1" root="$2" marker="$3" total="$4" worker_pid="$5"
  local done_count=-1 previous_count=-1 width=40 filled empty bar
  while kill -0 "$worker_pid" 2>/dev/null; do
    done_count="$(find "$root" -mindepth 2 -maxdepth 2 -name "$marker" -type f | wc -l)"
    if [[ -t 1 ]]; then
      filled=$((done_count * width / total))
      empty=$((width - filled))
      printf -v bar '%*s' "$filled" ''
      bar="${bar// /#}"
      printf '\r%-12s [%s%*s] %3d/%d' "$label" "$bar" "$empty" '' "$done_count" "$total"
    elif (( done_count != previous_count )); then
      printf '%s: %d/%d shards complete\n' "$label" "$done_count" "$total"
      previous_count="$done_count"
    fi
    sleep 1
  done
  wait "$worker_pid"
  local worker_status=$?
  done_count="$(find "$root" -mindepth 2 -maxdepth 2 -name "$marker" -type f | wc -l)"
  if [[ -t 1 ]]; then
    filled=$((done_count * width / total))
    empty=$((width - filled))
    printf -v bar '%*s' "$filled" ''
    bar="${bar// /#}"
    printf '\r%-12s [%s%*s] %3d/%d\n' "$label" "$bar" "$empty" '' "$done_count" "$total"
  elif (( done_count != previous_count )); then
    printf '%s: %d/%d shards complete\n' "$label" "$done_count" "$total"
  fi
  return "$worker_status"
}

TOTAL_ROWS="$($PYTHON_BIN -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("select count(*) from functionalized_lignins").fetchone()[0])' "$DB_PATH")"
if [[ -n "${MAX_ROWS:-}" ]] && (( MAX_ROWS < TOTAL_ROWS )); then
  TOTAL_ROWS="$MAX_ROWS"
fi
SHARD_COUNT=$(((TOTAL_ROWS + ROWS_PER_SHARD - 1) / ROWS_PER_SHARD))
echo "Preparing $TOTAL_ROWS rows in $SHARD_COUNT shards ($PREPROCESS_JOBS canonicalization workers)"

preprocess_one() {
  local shard_id="$1" shard_name shard_dir start_rowid end_rowid
  shard_name="$(printf 'shard_%03d' "$shard_id")"
  shard_dir="$OUTPUT_ROOT/preprocessed/$shard_name"
  start_rowid=$((shard_id * ROWS_PER_SHARD + 1))
  end_rowid=$(((shard_id + 1) * ROWS_PER_SHARD))
  (( end_rowid <= TOTAL_ROWS )) || end_rowid="$TOTAL_ROWS"
  if [[ -f "$shard_dir/.complete" ]]; then
    return
  fi
  mkdir -p "$shard_dir"
  "$PYTHON_BIN" scripts/preprocess_lignin_solubility.py \
    --db "$DB_PATH" --output-dir "$shard_dir" \
    --start-rowid "$start_rowid" --end-rowid "$end_rowid" --no-progress \
    > "$shard_dir/preprocess.log" 2>&1
  touch "$shard_dir/.complete"
}
export -f preprocess_one
export REPO_DIR PYTHON_BIN DB_PATH OUTPUT_ROOT ROWS_PER_SHARD TOTAL_ROWS
seq 0 $((SHARD_COUNT - 1)) | xargs -n 1 -P "$PREPROCESS_JOBS" bash -c 'preprocess_one "$1"' _ &
PREPROCESS_PID=$!
set +e
show_progress "canonicalize" "$OUTPUT_ROOT/preprocessed" ".complete" "$SHARD_COUNT" "$PREPROCESS_PID"
PREPROCESS_STATUS=$?
set -e
if (( PREPROCESS_STATUS != 0 )); then
  echo "Canonicalization failed; inspect $OUTPUT_ROOT/preprocessed/shard_*/preprocess.log" >&2
  exit "$PREPROCESS_STATUS"
fi

ROWS_FILES=("$OUTPUT_ROOT"/preprocessed/shard_*/rows.csv.gz)
[[ ${#ROWS_FILES[@]} -eq "$SHARD_COUNT" ]] || {
  echo "Expected $SHARD_COUNT preprocessed shards, found ${#ROWS_FILES[@]}" >&2
  exit 1
}

echo "Building unified tokenizer"
"$PYTHON_BIN" scripts/build_lignin_training_tokenizer.py \
  --rows "${ROWS_FILES[@]}" --output "$OUTPUT_ROOT/unified_tokenizer.json"
TOKENIZER_SHA="$($PYTHON_BIN -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$OUTPUT_ROOT/unified_tokenizer.json")"

# Invalidate completion markers made with an older tokenizer so the aggregate
# counter reflects only shards valid for the tokenizer built above.
for marker in "$OUTPUT_ROOT"/encoded/shard_*/.tokenizer_sha256; do
  [[ -e "$marker" ]] || continue
  [[ "$(<"$marker")" == "$TOKENIZER_SHA" ]] || rm -f "$marker"
done

encode_one() {
  local shard_id="$1" shard_name shard_dir marker
  shard_name="$(printf 'shard_%03d' "$shard_id")"
  shard_dir="$OUTPUT_ROOT/encoded/$shard_name"
  marker="$shard_dir/.tokenizer_sha256"
  if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$TOKENIZER_SHA" ]]; then
    return
  fi
  mkdir -p "$shard_dir"
  "$PYTHON_BIN" scripts/encode_lignin_training_shard.py \
    --rows "$OUTPUT_ROOT/preprocessed/$shard_name/rows.csv.gz" \
    --tokenizer "$OUTPUT_ROOT/unified_tokenizer.json" \
    --output-dir "$shard_dir" --split-by "$SPLIT_BY" --split-seed "$SPLIT_SEED" \
    > "$shard_dir/encode.log" 2>&1
  echo "$TOKENIZER_SHA" > "$marker"
}
export -f encode_one
export TOKENIZER_SHA SPLIT_BY SPLIT_SEED
echo "Encoding packed shards ($ENCODE_JOBS workers)"
seq 0 $((SHARD_COUNT - 1)) | xargs -n 1 -P "$ENCODE_JOBS" bash -c 'encode_one "$1"' _ &
ENCODE_PID=$!
set +e
show_progress "encode" "$OUTPUT_ROOT/encoded" ".tokenizer_sha256" "$SHARD_COUNT" "$ENCODE_PID"
ENCODE_STATUS=$?
set -e
if (( ENCODE_STATUS != 0 )); then
  echo "Encoding failed; inspect $OUTPUT_ROOT/encoded/shard_*/encode.log" >&2
  exit "$ENCODE_STATUS"
fi

echo "Training data ready under $OUTPUT_ROOT"
