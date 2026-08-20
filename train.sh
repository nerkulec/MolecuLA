#!/usr/bin/env bash
# Usage: bash train.sh linear_attention [additional train_lignin_vae.py options]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MOLECULA_PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/artifacts/lignin_retraining}"
MODEL="${1:-${MODEL:-}}"
[[ -n "$MODEL" ]] || { echo "Usage: bash train.sh {linear_attention|simple_attention|autoregressive}" >&2; exit 2; }
shift || true
case "$MODEL" in
  linear_attention|simple_attention|autoregressive) ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

cd "$REPO_DIR"
TOKENIZER="$OUTPUT_ROOT/unified_tokenizer.json"
[[ -f "$TOKENIZER" ]] || { echo "Missing $TOKENIZER; run bash prepare_data.sh first" >&2; exit 1; }
SHARDS=("$OUTPUT_ROOT"/encoded/shard_*)
[[ -f "${SHARDS[0]}/manifest.json" ]] || { echo "No encoded shards; run bash prepare_data.sh first" >&2; exit 1; }

GPU_COUNT="${NPROC_PER_NODE:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
(( GPU_COUNT >= 1 )) || { echo "No CUDA GPU is visible" >&2; exit 1; }
RUN_DIR="$OUTPUT_ROOT/checkpoints/$MODEL"
mkdir -p "$RUN_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RESUME_ARGS=()
if [[ "${RESUME:-auto}" == "auto" ]] && [[ -f "$RUN_DIR/last.pt" ]]; then
  RESUME_ARGS=(--resume "$RUN_DIR/last.pt")
elif [[ -n "${RESUME:-}" ]] && [[ "${RESUME}" != "none" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

echo "Training $MODEL on $GPU_COUNT GPU(s); output: $RUN_DIR"
if (( GPU_COUNT == 1 )); then
  LAUNCH=("$PYTHON_BIN")
else
  LAUNCH=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT")
fi
"${LAUNCH[@]}" \
  scripts/train_lignin_vae.py --model "$MODEL" \
  --tokenizer "$TOKENIZER" --shards "${SHARDS[@]}" --output-dir "$RUN_DIR" \
  --epochs "${EPOCHS:-50}" --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${NUM_WORKERS:-4}" --precision "${PRECISION:-bf16}" \
  --greedy-val-samples "${GREEDY_VAL_SAMPLES:-64}" \
  "${RESUME_ARGS[@]}" "$@"
