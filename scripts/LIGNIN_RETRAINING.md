# Retraining the autoregressive VAE on lignin solubility molecules

This pipeline trains on the canonical `selfies_final` produced by
`preprocess_lignin_solubility.py`. It deliberately does not use solubility as a
training target: the model remains a molecular VAE. Failed preprocessing
rows are recorded in the CSV reports and excluded at encoding time.

## Changes from the released models

- Unified indexing: `<PAD>=0`, `<SOS>=1`,
  `<EOS>=2`, `MASK=3`, followed by sorted SELFIES tokens.
- The autoregressive latent dimension is doubled to **512** (from 256).
- No fixed 77-token filter. Packed variable-length shards are padded only within
  each length-bucketed batch; positional encodings grow dynamically.
- Autoregressive greedy decoding honors its configured maximum length.
- Stable 80/10/10 splits are derived from row IDs by default. Set
  `SPLIT_BY=scaffold` for a harder scaffold-disjoint split (all shards must use the
  same choice and seed).
- Checkpoints include optimizer/scaler state, architecture, metrics and tokenizer
  SHA-256.

## HPC profile and environment

The Mackup HPC configuration contains a `molecula` project pointing at
`$HOME/casus/MolecuLA` and `$HOME/scripts/profiles/molecula.sh`. The profile loads
Python 3.12.4, CUDA 12.8 and activates
`$HOME/casus/MolecuLA/molecula-venv`, matching the Wyckoff convention.

After syncing the Mackup configuration, create the environment from the copied
repository. The profile intentionally only warns if the environment does not yet
exist, so this bootstrap command works:

```bash
rung --project molecula bash setup_venv.sh
```

It may also be run directly on a login node if that is where environments are
normally created:

```bash
source /etc/profile.d/lmod.sh
module load python/3.12.4 cuda/12.8
bash setup_venv.sh
```

## Prepare the dataset

Preparation is CPU-heavy and needs no GPU. It automatically determines the table
size, canonicalizes 100,000-row shards, builds the shared vocabulary, and encodes
packed shards. Completed canonicalization shards and tokenizer-matched encoded
shards are skipped when the job is restarted.

`prepare_data.sh` is a cooperative shared-filesystem queue. The same command may
be started repeatedly on different nodes at any time. Each local process claims
an unprocessed shard with an atomic directory, refreshes a heartbeat while it is
working, and publishes its completion marker only after successful output. Other
nodes immediately take the next unclaimed shards. Claims left by terminated nodes
are reclaimed after 30 minutes by default (`CLAIM_STALE_SECONDS` controls this).
Every node shows one tqdm bar for total dataset progress, not merely its local
share. Detailed worker output remains in each shard's `preprocess.log` or
`encode.log`.

The database is stored as two Git LFS chunks because GitHub rejects individual
LFS objects larger than 2 GiB. `prepare_data.sh` automatically reconstructs the
ignored `data/lignin_solubility.db` and verifies its SHA-256 before reading it.
It can also be assembled explicitly with `bash assemble_lignin_database.sh`.

```bash
rung --project molecula bash prepare_data.sh
```

Additional nodes can join the same output queue without special coordination:

```bash
# Start these now or hours later; allocation size is detected independently.
rung --project molecula bash prepare_data.sh
rung_big --project molecula bash prepare_data.sh
runt_big --project molecula bash prepare_data.sh
```

The first invocation persists the shard size and split settings under
`artifacts/lignin_retraining/state/prepare.env`; later invocations adopt them.
For a fresh output directory intended for hundreds of CPUs, choose more granular
shards on the first invocation, for example `ROWS_PER_SHARD=10000`. Subsequent
nodes do not need to repeat this setting. `PREPROCESS_JOBS` and `ENCODE_JOBS`
default to the CPUs allocated to each node and may be lowered to cap memory use.

If a molecule shard fails, all nodes stop and report its `.failed` marker. Inspect
the adjacent log, correct the cause, remove that shard's `.failed` marker, and
restart any number of preparation jobs.

The script defaults to half as many RDKit workers as allocated CPUs and one
encoding worker per CPU. Override these if memory is limiting:

```bash
PREPROCESS_JOBS=4 ENCODE_JOBS=8 rung --project molecula bash prepare_data.sh
```

For a quick pipeline check in a separate output directory, set `MAX_ROWS`, for
example `MAX_ROWS=10000 OUTPUT_ROOT=/tmp/molecula-10k bash prepare_data.sh`.

Use `SPLIT_BY=scaffold` on the first run if a scaffold-disjoint 80/10/10 split is
preferred. Do not change the split strategy between shards.

## Train

Only the autoregressive retraining path is supported. It is a single-process
Python command with explicit input and output paths and named arguments:

```bash
python scripts/train_lignin_vae.py \
  --model autoregressive \
  --tokenizer artifacts/lignin_retraining/unified_tokenizer.json \
  --shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/smoke/autoregressive_50k \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 4 \
  --precision bf16 \
  --max-train-samples 50000 \
  --max-val-samples 5000 \
  --save-every 0
```

The script uses one visible CUDA GPU, or CPU when CUDA is unavailable. It does
not provide a bash wrapper, distributed launch, alternate architectures,
automatic path discovery, environment-variable configuration, or checkpoint
resume mode. Use a fresh `--output-dir` for each run.

By default, all packed token data is loaded into one GPU-resident int32 buffer;
only the current padded batch is expanded to int64. For the full corpus this
uses approximately 6.49 GiB (6.89 GB), including global offsets. Use
`--dataset-device cpu` to retain the memory-mapped CPU loader instead. GPU data
loading, training, and validation each display a tqdm bar.

Each model directory contains `last.pt`, validation-selected `best.pt`, periodic
epoch checkpoints, `run_config.json`, and append-only `metrics.jsonl`. Training
and validation report token accuracy and one full-sequence `exact_accuracy`,
computed in a single teacher-forced decoder pass. Padding positions are ignored.

## W&B sweep

The Bayesian sweep in `sweeps/lignin_autoregressive.yaml` varies learning rate,
KL ceiling, hidden size, latent size, slot count, encoder depth, decoder depth,
and batch size. Each trial uses the full deterministic train/validation splits
and trains for up to 36 hours. The 1,000-epoch ceiling is only a safety bound;
training stops between epochs before another epoch is likely to exceed the wall
clock budget. Model selection minimizes
`val/selection_loss = reconstruction_loss + 0.03 * kl_loss`; the fixed comparison
weight makes trials with different training KL schedules directly comparable.
Every 100 batches, W&B receives current and running total/reconstruction/KL
losses, token accuracy, gradient norm, learning rate, beta, throughput, sequence
width, and CUDA allocated/reserved/peak memory. Full epoch metrics add one
`exact_accuracy` per split and validation selection loss.

For numerical stability, latent variance aggregation and KL are evaluated in
FP32, log-variance is constrained to `[-12, 6]`, gradients are clipped at norm
1.0 with non-finite gradients treated as errors, and the learning rate warms up
over 1,000 optimizer steps. KL beta increases linearly on every training batch
and reaches its configured maximum after exactly two complete passes through the
training split; it no longer spends an entire full-data epoch at zero, jumps only
at epoch boundaries, or resets cyclically. The sweep learning-rate range is
`3e-5` to `3e-4`.

Newly trained autoregressive models mask PAD embeddings before convolution and
use the correct token/head transpose in custom encoder attention. Their encoder
means are therefore invariant to right-padding width. Legacy checkpoints retain
their original encoder behavior for strict compatibility.

## Export the best model's latents in database order

The exporter verifies that the encoded shards contain every dense SQLite rowid
from 1 through 9,766,400, verifies the tokenizer and checkpoint hashes, and
writes deterministic FP32 encoder means in exactly that order:

```bash
python scripts/export_lignin_latents.py \
  --database data/lignin_solubility.db \
  --checkpoint artifacts/lignin_retraining/sweeps/autoregressive/9yibqc0g/best.pt \
  --shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/latents/9yibqc0g \
  --batch-size 512 \
  --device cuda \
  --precision bf16 \
  --padding-mode checkpoint-max
```

The best checkpoint is legacy, so `checkpoint-max` pads every input to its
420-token configured width and makes its latent definition independent of batch
size and batch composition. The exporter is resumable at encoded-shard
boundaries. It writes `latents.partial.npy` while running and atomically renames
it to `latents.npy` only after all shards finish. The final outputs are:

- `latents.npy`: `(9_766_400, 1024)` FP32, 37.26 GiB;
- `rowids.npy`: explicit `1..9_766_400` alignment vector;
- `manifest.json`: hashes, shape, checkpoint epoch, padding policy, and shard
  ranges;
- `export_state.json`: completed-shard state used for automatic restart.

For new padding-invariant checkpoints, `--padding-mode batch-max` is also safe
and faster. The exporter rejects that mode for legacy checkpoints.

## Analyze solubility predictability

Run the analysis on the completed, row-ordered latent export. On an H100, CUDA
accelerates the only large operation: streaming sufficient-statistic matrix
products over the 37.26-GiB latent matrix. The script never copies the complete
latent matrix to either host RAM or GPU RAM.

```bash
python scripts/analyze_lignin_solubility_latents.py \
  --database data/lignin_solubility.db \
  --latents-dir artifacts/lignin_retraining/latents/9yibqc0g \
  --preprocessed-rows artifacts/lignin_retraining/preprocessed/shard_*/rows.csv.gz \
  --encoded-shards artifacts/lignin_retraining/encoded/shard_* \
  --output-dir artifacts/lignin_retraining/solubility_analysis/9yibqc0g \
  --split-modes rowid scaffold \
  --pca-components 8 64 512 \
  --batch-size 32768 \
  --device cuda \
  --seed 42 \
  --prediction-sample-size 100000
```

`rowid` reuses the stable random 80/10/10 split stored in the encoded training
data. `scaffold` independently makes a deterministic 80/10/10 split from the
Bemis-Murcko scaffold hashes, so no scaffold appears across its splits. All
preprocessing, PCA, scaling, confound residualization, and Ridge fitting use
training data only; validation selects the Ridge alpha from
`logspace(-3, 3, 13)`, and test data is evaluated once.

The script evaluates the full latent and the leading 8, 64, and 512 PCA
components. For each representation it reports direct log-solubility R²,
unconfounded residual-target R², combined confound-plus-latent R², and latent
probe R² for each of the four sequence confounds. It saves:

- compact `solubility_r2_summary.csv`, comprehensive `probe_metrics.csv`, and
  the self-contained `analysis_report.json`;
- fitted PCA transforms and explained-variance tables;
- fitted confound, raw-target, residual-target, and confound-target Ridge probes;
- both split assignment arrays;
- a reproducible 100,000-row prediction sample, PCA scores, and Pearson/Spearman
  correlation tables for later plotting and report preparation.

Copy the complete output directory back locally; the fitted artifacts are small
and the sampled tables should be far below one GiB.

Create the sweep once:

```bash
wandb sweep sweeps/lignin_autoregressive.yaml
```

Use the printed sweep path to test one run in an existing GPU allocation:

```bash
wandb agent --count 1 ENTITY/molecula-lignin-autoregressive-rosi/SWEEP_ID
```

Additional agents can join the same sweep independently. Each run stores its
local checkpoints under `artifacts/lignin_retraining/sweeps/autoregressive/RUN_ID`
and logs its best checkpoint as a W&B model artifact.

### Compact-latent/deep-encoder sweep

The follow-up sweep targets latent sizes 64, 128, and 256, decoder depths 2 and
3, and encoder depths 2, 3, 4, 6, 8, 10, and 12. It additionally searches hidden
width, slot count, batch size, learning rate, and KL ceiling. Trials retain the
same 36-hour budget and selection metric as the original sweep and are logged in
the same W&B project for direct comparison:

```bash
wandb sweep sweeps/lignin_autoregressive_compact_latent.yaml
```

Test one trial in the current allocation using the sweep path printed above:

```bash
wandb agent --count 1 ENTITY/molecula-lignin-autoregressive-rosi/SWEEP_ID
```

Start further agents with the same sweep path. Compact-run checkpoints are kept
separately under
`artifacts/lignin_retraining/sweeps/autoregressive_compact/RUN_ID`.

## Export and evaluate the compact-latent winners

The reconstruction-focused checkpoints are run `634itik5` (`best.pt`) for 64
dimensions, run `3bv5lnom` (`best.pt`) for 128 dimensions, and run `17nxcyq1`
(`last.pt`, epoch 38) for 256 dimensions. The last checkpoint is intentional for
the 256-dimensional model: its retained final epoch has lower reconstruction
loss than its composite-loss-selected `best.pt`.

Run the three row-aligned latent exports sequentially on a GPU node:

```bash
bash scripts/export_compact_lignin_latents.sh
```

The exporter uses fast `batch-max` padding because all three new checkpoints are
padding invariant. It is safe to restart: completed exports are verified and
skipped, while partial exports resume at encoded-shard boundaries. The FP32
latent matrices require 2.33 GiB, 4.66 GiB, and 9.31 GiB respectively (16.30 GiB
total), excluding the small rowid and manifest files.

After all exports complete, run the solubility probes and cross-model comparison:

```bash
bash scripts/analyze_compact_lignin_solubility.sh
```

Each model receives the full raw/confound-residualized analysis under both rowid
and scaffold splits. Common PCA-8 and PCA-64 probes are fitted for all three;
PCA-128 and PCA-256 are additionally fitted where the latent width permits. The
last command combines the outputs into:

```text
artifacts/lignin_retraining/solubility_analysis/compact/comparison/
  comparison_report.json
  model_checkpoints.csv
  full_latent_test_r2.csv
  solubility_r2_comparison.csv
  probe_metrics_comparison.csv
  pca_variance_comparison.csv
```

Both expensive stages should run on the cluster. Export requires neural-network
inference over 9.77 million molecules and benefits strongly from the H100. Probe
fitting streams billions of covariance operations; it is possible on a laptop
with enough RAM and the copied latent files, but substantially slower and would
require transferring more than 16 GiB of latents plus source metadata.

Only the compact `solubility_analysis/compact` directory needs to be copied back
locally. Combining already-generated result tables, plotting, and report writing
are lightweight laptop tasks. The comparison can be regenerated locally with
explicit paths using `scripts/compare_compact_lignin_solubility.py` if needed.

## Direct local/single-GPU use

The stages are ordinary Python commands. For an existing preprocessed sample:

```bash
python scripts/build_lignin_training_tokenizer.py \
  --rows artifacts/lignin_solubility/local_10k/preprocess/rows.csv.gz \
  --output /tmp/lignin-smoke/tokenizer.json
python scripts/encode_lignin_training_shard.py \
  --rows artifacts/lignin_solubility/local_10k/preprocess/rows.csv.gz \
  --tokenizer /tmp/lignin-smoke/tokenizer.json \
  --output-dir /tmp/lignin-smoke/shard_000
python scripts/train_lignin_vae.py --model autoregressive \
  --tokenizer /tmp/lignin-smoke/tokenizer.json --shards /tmp/lignin-smoke/shard_000 \
  --output-dir /tmp/lignin-smoke/run --epochs 1 --batch-size 4 --num-workers 0 \
  --precision fp32 --max-train-samples 16 --max-val-samples 8 \
  --save-every 0
```
