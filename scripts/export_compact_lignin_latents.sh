#!/usr/bin/env bash
set -euo pipefail

# Reconstruction-focused checkpoint for latent size 64 (run 634itik5, epoch 55).
python scripts/export_lignin_latents.py \
  --database data/lignin_solubility.db \
  --checkpoint artifacts/lignin_retraining/sweeps/autoregressive_compact/634itik5/best.pt \
  --shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/latents/compact/latent_64_634itik5 \
  --batch-size 512 \
  --device cuda \
  --precision bf16 \
  --padding-mode batch-max

# Reconstruction-focused checkpoint for latent size 128 (run 3bv5lnom, epoch 57).
python scripts/export_lignin_latents.py \
  --database data/lignin_solubility.db \
  --checkpoint artifacts/lignin_retraining/sweeps/autoregressive_compact/3bv5lnom/best.pt \
  --shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/latents/compact/latent_128_3bv5lnom \
  --batch-size 512 \
  --device cuda \
  --precision bf16 \
  --padding-mode batch-max

# Epoch 38 is the best retained reconstruction checkpoint for latent size 256.
# Its last.pt is preferable to the composite-loss-selected epoch-36 best.pt.
python scripts/export_lignin_latents.py \
  --database data/lignin_solubility.db \
  --checkpoint artifacts/lignin_retraining/sweeps/autoregressive_compact/17nxcyq1/last.pt \
  --shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/latents/compact/latent_256_17nxcyq1 \
  --batch-size 512 \
  --device cuda \
  --precision bf16 \
  --padding-mode batch-max
