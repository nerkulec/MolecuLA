# Retraining the three VAEs on lignin solubility molecules

This pipeline trains on the canonical `selfies_final` produced by
`preprocess_lignin_solubility.py`. It deliberately does not use solubility as a
training target: all three models remain molecular VAEs. Failed preprocessing
rows are recorded in the CSV reports and excluded at encoding time.

## Changes from the released models

- One vocabulary and one indexing scheme for every model: `<PAD>=0`, `<SOS>=1`,
  `<EOS>=2`, `MASK=3`, followed by sorted SELFIES tokens.
- Latent dimensions are doubled: linear attention **1024** (was 512), simple
  attention **512** (was 256), autoregressive **512** (was 256).
- No fixed 77-token filter. Packed variable-length shards are padded only within
  each length-bucketed batch; positional encodings grow dynamically.
- Non-autoregressive reconstruction is evaluated over the complete target during
  training. The prior objective omitted suffixes longer than the predicted length.
- Autoregressive greedy decoding honors its configured maximum length.
- The raw token-count MSE from the non-autoregressive length head is weighted by
  `1/max_sequence_length²` (and by an additional 0.1 for linear attention), so it
  cannot swamp token cross-entropy merely because this corpus has longer molecules.
- Stable 80/10/10 splits are derived from row IDs by default. Set
  `SPLIT_BY=scaffold` for a harder scaffold-disjoint split (all shards must use the
  same choice and seed).
- Checkpoints include optimizer/scaler state, architecture, metrics and tokenizer
  SHA-256, and resume performs strict compatibility checks.

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

The database is stored as two Git LFS chunks because GitHub rejects individual
LFS objects larger than 2 GiB. `prepare_data.sh` automatically reconstructs the
ignored `data/lignin_solubility.db` and verifies its SHA-256 before reading it.
It can also be assembled explicitly with `bash assemble_lignin_database.sh`.

```bash
rung --project molecula bash prepare_data.sh
```

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

Submit one ordinary GPU job per architecture. `train.sh` detects the number of
visible GPUs, starts that many distributed workers, and treats `BATCH_SIZE` as the
per-GPU batch size:

```bash
runh1 --project molecula bash train.sh linear_attention
runh1 --project molecula bash train.sh simple_attention
runh1 --project molecula bash train.sh autoregressive
```

For a four-GPU allocation, use `runh4` with the same command. Common overrides:

```bash
BATCH_SIZE=64 EPOCHS=75 runh1 --project molecula bash train.sh autoregressive
PRECISION=fp16 runa1 --project molecula bash train.sh simple_attention
```

Existing `last.pt` files are resumed automatically. Set `RESUME=none` to start
without resuming, or `RESUME=/path/to/checkpoint.pt` to select a checkpoint. Note
that starting without resume in an existing output directory appends to its
metrics file; use a different `OUTPUT_ROOT` for a genuinely separate experiment.

The standard attention encoder is quadratic in batch-local sequence length; lower
`BATCH_SIZE` if a long bucket exhausts memory. BF16 is the default.

Each model directory contains `last.pt`, validation-selected `best.pt`, periodic
epoch checkpoints, `run_config.json`, and append-only `metrics.jsonl`. Validation
reports teacher-forced token/exact accuracy and a bounded greedy full-sequence
accuracy; `--greedy-val-samples 0` disables the latter when fast epochs matter.

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
python scripts/train_lignin_vae.py --model simple_attention \
  --tokenizer /tmp/lignin-smoke/tokenizer.json --shards /tmp/lignin-smoke/shard_000 \
  --output-dir /tmp/lignin-smoke/run --epochs 1 --batch-size 4 --num-workers 0 \
  --precision fp32 --max-train-samples 16 --max-val-samples 8 --greedy-val-samples 2
```
