#!/usr/bin/env bash
set -euo pipefail

python scripts/analyze_lignin_solubility_latents.py \
  --database data/lignin_solubility.db \
  --latents-dir artifacts/lignin_retraining/latents/compact/latent_64_634itik5 \
  --preprocessed-rows artifacts/lignin_retraining/preprocessed/shard_*/rows.csv.gz \
  --encoded-shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/solubility_analysis/compact/latent_64_634itik5 \
  --split-modes rowid scaffold \
  --pca-components 8 64 \
  --batch-size 65536 \
  --device cuda \
  --seed 42 \
  --prediction-sample-size 100000

python scripts/analyze_lignin_solubility_latents.py \
  --database data/lignin_solubility.db \
  --latents-dir artifacts/lignin_retraining/latents/compact/latent_128_3bv5lnom \
  --preprocessed-rows artifacts/lignin_retraining/preprocessed/shard_*/rows.csv.gz \
  --encoded-shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/solubility_analysis/compact/latent_128_3bv5lnom \
  --split-modes rowid scaffold \
  --pca-components 8 64 128 \
  --batch-size 65536 \
  --device cuda \
  --seed 42 \
  --prediction-sample-size 100000

python scripts/analyze_lignin_solubility_latents.py \
  --database data/lignin_solubility.db \
  --latents-dir artifacts/lignin_retraining/latents/compact/latent_256_17nxcyq1 \
  --preprocessed-rows artifacts/lignin_retraining/preprocessed/shard_*/rows.csv.gz \
  --encoded-shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/solubility_analysis/compact/latent_256_17nxcyq1 \
  --split-modes rowid scaffold \
  --pca-components 8 64 128 256 \
  --batch-size 65536 \
  --device cuda \
  --seed 42 \
  --prediction-sample-size 100000

python scripts/compare_compact_lignin_solubility.py \
  --analysis-64 artifacts/lignin_retraining/solubility_analysis/compact/latent_64_634itik5 \
  --analysis-128 artifacts/lignin_retraining/solubility_analysis/compact/latent_128_3bv5lnom \
  --analysis-256 artifacts/lignin_retraining/solubility_analysis/compact/latent_256_17nxcyq1 \
  --output-dir artifacts/lignin_retraining/solubility_analysis/compact/comparison
