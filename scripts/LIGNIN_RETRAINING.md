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
  --greedy-val-samples 16 \
  --save-every 0
```

The script uses one visible CUDA GPU, or CPU when CUDA is unavailable. It does
not provide a bash wrapper, distributed launch, alternate architectures,
automatic path discovery, environment-variable configuration, or checkpoint
resume mode. Use a fresh `--output-dir` for each run.

Each model directory contains `last.pt`, validation-selected `best.pt`, periodic
epoch checkpoints, `run_config.json`, and append-only `metrics.jsonl`. Validation
reports teacher-forced token/exact accuracy and a bounded greedy full-sequence
accuracy; `--greedy-val-samples 0` disables the latter when fast epochs matter.

## W&B sweep

The Bayesian sweep in `sweeps/lignin_autoregressive.yaml` varies learning rate,
KL ceiling, hidden size, latent size, slot count, encoder depth, decoder depth,
and batch size. Each trial uses the full deterministic train/validation splits
and trains for up to 36 hours. The 1,000-epoch ceiling is only a safety bound;
training stops between epochs before another epoch is likely to exceed the wall
clock budget. Model selection minimizes
`val/selection_loss = reconstruction_loss + 0.03 * kl_loss`; the fixed comparison
weight makes trials with different training KL schedules directly comparable.

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
  --greedy-val-samples 2 --save-every 0
```
