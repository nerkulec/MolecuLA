# Lignin solubility latent evaluation

The pipeline evaluates the three released MolecuLA checkpoints on the SMILES in
`data/lignin_solubility.db`. It uses the paper conversion path exactly:

```text
raw SMILES -> RDKit canonical isomeric SMILES -> SELFIES -> SMILES -> SELFIES
```

It does not truncate sequences or deduplicate canonical structures. Molecules are
eligible only when every final SELFIES token belongs to the checkpoint vocabulary
and the sequence, including SOS/EOS, is at most 77 tokens.

## Local smoke test

Create the environment, then run the seeded 10,000-row evaluation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_lignin_local_e2e.py \
  --sample-size 10000 \
  --seed 42 \
  --output-dir artifacts/lignin_solubility/local_10k
python scripts/verify_lignin_run.py \
  --run-dir artifacts/lignin_solubility/local_10k
```

Use `--resume` to keep completed stages. The orchestration script calls two
different tokenization entry points:

- `tokenize_lignin_nonar.py`: the released 110-token JSON mapping.
- `tokenize_lignin_ar.py`: the checkpoint-faithful 111-token mapping with `MASK`
  inserted at ID 3.

## Cluster run

The database contains rowids 1 through 9,766,400. With 100,000 rows per shard,
submit array indices 0 through 97. A representative Slurm invocation is:

```bash
sbatch --array=0-97 --gres=gpu:1 --cpus-per-task=8 \
  --export=ALL,PYTHON_BIN=.venv/bin/python,OUTPUT_ROOT=artifacts/lignin_solubility/full_shards \
  scripts/run_lignin_cluster_shard.sh
```

Tune the site-specific partition, memory, time, and GPU flags separately. Each
array task canonicalizes one rowid range, creates both integer encodings, intersects
eligibility, then runs all three checkpoints sequentially. Shards are immutable and
can be rerun independently.

After every shard succeeds, fit probes without concatenating the latent matrices:

```bash
python scripts/fit_lignin_probes.py \
  --shard-root artifacts/lignin_solubility/full_shards \
  --split-mode random \
  --seed 42 \
  --output-dir artifacts/lignin_solubility/full_probes_random

python scripts/fit_lignin_probes.py \
  --shard-root artifacts/lignin_solubility/full_shards \
  --split-mode scaffold \
  --seed 42 \
  --output-dir artifacts/lignin_solubility/full_probes_scaffold
```

The probe implementation streams latent shards, calculates train-only scaling,
selects Ridge alpha from `logspace(-3, 3, 13)` on validation data, and evaluates the
test split once. It reports raw `Z -> logS`, residual `Z -> residual(logS | C)`, and
combined `C + Z_residual` R-squared values. The four confounds are SELFIES token
length, branch count, ring count, and token entropy.

Aggregate full-run failure rates after the random-split probe run:

```bash
python scripts/aggregate_lignin_shards.py \
  --shard-root artifacts/lignin_solubility/full_shards \
  --probe-report artifacts/lignin_solubility/full_probes_random/probe_report.json \
  --output-dir artifacts/lignin_solubility/full_report
```

## Output layout

Each local run or cluster shard contains:

```text
preprocess/rows.csv.gz
tokenization/{nonar,ar}/{tokens.npy,eligible_mask.npy,tokenizer.json}
common_eligible.npy
probe_targets.npz
models/<model>/{rowids.npy,latents.npy,reconstruction_rows.csv.gz,reconstruction_report.json}
probes/{probe_metrics.csv,probe_report.json,*_probe.npz}
```

`latents.npy` stores deterministic encoder means in FP32. Reconstruction is greedy
and uses those means; no latent sampling or teacher forcing is used.
