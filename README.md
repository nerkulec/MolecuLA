# Molecules Meet Language: Confound-Aware Representation Learning and Chemical Property Steering in Transformer-VAE Latent Spaces

This repository is an anonymized, GitHub-ready export of a "Molecules Meet Language: Confound-Aware Representation Learning and Chemical Property Steering in Transformer-VAE Latent Spaces" paper built around Transformer VAEs trained on SELFIES strings. It preserves the runnable workflow and selected evidence without carrying local machine paths, private Git history, or large regenerated caches.

## Models

The export uses three canonical model identifiers:

| Identifier | Model file | Checkpoint | Status |
|---|---|---|---|
| `linear_attention` | `models/linear_attention_vae.py` | `checkpoints/linear_attention_h256_l512.pt` | included |
| `simple_attention` | `models/simple_attention_vae.py` | `checkpoints/simple_attention_h256_l256.pt` | included |
| `autoregressive` | `models/autoregressive_vae.py` | `checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt` | included |

`simple_attention` corresponds to the `nat_model_h256_l256` model used by the model-comparison workflow.

## Repository Layout

```text
models/           Model definitions and registry
checkpoints/      Included model weights
data/             SELFIES/SMILES dataset and tokenizer
study/common/     Reusable data, chemistry, probe, latent, traversal, and plotting helpers
study/notebooks/  Five output-stripped workflow notebooks
study/results/    Curated compact figures, JSON summaries, and CSV tables
scripts/          Export verification script
```

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

RDKit installation is easiest through Conda or Mamba if pip wheels are unavailable on your platform:

```bash
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

## Data And Checkpoints

The dataset is expected at:

```text
data/smiles_selfies_full.csv
```

The included checkpoints are:

```text
checkpoints/linear_attention_h256_l512.pt
checkpoints/simple_attention_h256_l256.pt
checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt
```


## Workflow

Run notebooks in this order:

1. `study/notebooks/00_data_tokenization_and_training.ipynb`
2. `study/notebooks/01_latent_quality_benchmarks_and_family_tests.ipynb`
3. `study/notebooks/02_linear_probes_panels_and_residuals.ipynb`
4. `study/notebooks/03_latent_traversals.ipynb`
5. `study/notebooks/04_mlp_probes_all_properties.ipynb`

The notebooks write regenerated panels, latents, direction arrays, and probe models under local output folders. These caches are intentionally not committed because the full panels and latent arrays are large and reproducible from the dataset and checkpoints.

## Experimental Setup And Reproducibility

The repository is organized so the released checkpoints are fixed pretrained inputs and every downstream study artifact can be regenerated from `data/`, `checkpoints/`, `models/`, and `study/notebooks/`. All paths in the public workflow are repository-relative. The large dataset CSV and model checkpoints are Git LFS assets, so after cloning the repository run `git lfs pull` before executing notebooks or verification.

### Randomness And Determinism

Unless a notebook explicitly overrides it, the study uses `SEED = 42`. The canonical split helper in `study/common/data.py` calls `train_test_split(..., random_state=42, shuffle=True)` for both split stages. The notebooks seed Python `random`, NumPy, `torch.manual_seed`, and `torch.cuda.manual_seed_all` when CUDA is available. Code that needs an explicit generator uses `np.random.default_rng(42)` or a documented offset from that seed.

Exact floating point values can still vary slightly across PyTorch, CUDA, cuDNN, BLAS, GPU model, and RDKit versions. The intended reproducibility contract is therefore: identical dataset rows, tokenizer, splits, checkpoint weights, model registry settings, probe grids, random seeds, and output schemas. Small numerical differences in neural training or GPU decoding are expected across machines.

### Dataset, Tokenization, And Splits

The dataset is loaded from `data/smiles_selfies_full.csv` and must contain at least `smiles` and `selfies` columns. The released tokenizer is `data/selfies_tokenizer.json`, with `max_len = 77` and `vocab_size = 110` for the non-autoregressive checkpoints. SELFIES are encoded by prepending `<SOS>`, appending `<EOS>`, and padding with `<PAD>`.

The autoregressive training notebook constructs its own vocabulary with `<PAD>`, `<SOS>`, `<EOS>`, and `MASK`; the included autoregressive checkpoint records `vocab_size = 111`. This difference is intentional and is handled by the autoregressive notebook/model path.

The standard split is 80 percent train, 10 percent validation, and 10 percent test:

1. Split all row indices into train and temporary sets with `test_size=0.2`, `random_state=42`, and `shuffle=True`.
2. Split the temporary set into validation and test with `test_size=0.5`, `random_state=42`, and `shuffle=True`.
3. For the released dataset this gives expected split sizes of `635522` train, `79440` validation, and `79441` test rows.

RDKit-derived panels keep invalid molecules in the table but mark them through `is_rdkit_valid`. Probe fitting and property summaries use finite property values and valid molecules for the relevant target.

### Transformer-VAE Model Settings

The model registry in `models/registry.py` is the source of truth for public model configuration:

| Model | Hidden size | Latent size | Max length | Heads | Slots | Encoder/decoder depth | Checkpoint vocab | Recorded checkpoint epoch |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `linear_attention` | 256 | 512 | 77 | 8 | 8 | 1 shared non-AR layer | 110 | 50 |
| `simple_attention` | 256 | 256 | 77 | 8 | 8 | 1 shared non-AR layer | 110 | 18 |
| `autoregressive` | 256 | 256 | 77 | 8 | 8 | 3 encoder layers, 2 decoder layers | 111 | 48 |

The saved checkpoints contain model weights, optimizer state, training history, checkpoint epoch, and vocabulary size. The optimizer state in the released checkpoints is AdamW with learning rate `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `1e-2`. For the non-autoregressive models, the release treats these checkpoints as the reproducible pretrained artifacts; the checkpoint payloads do not encode every original data-loader setting from the training run.

The model loss functions are defined in the model files:

| Model | Reconstruction term | KL term | Length term |
|---|---|---|---|
| `linear_attention` | token cross entropy over non-pad targets | `beta = 0.01` by default | MSE on predicted length with `alpha = 0.1` |
| `simple_attention` | token cross entropy over non-pad targets | `beta = 0.01` by default | MSE on predicted length with `alpha = 1.0` |
| `autoregressive` | token cross entropy with `ignore_index=<PAD>` | `beta = 0.01` by default, overridden by the training schedule below | none |

All three architectures use Transformer-style attention blocks with dropout `p = 0.1` inside the feed-forward blocks. The non-autoregressive blocks use a feed-forward width of `4 * hidden_size`; the autoregressive blocks use `2 * hidden_size`.

### Autoregressive Training Settings

`study/notebooks/00_data_tokenization_and_training.ipynb` contains the rerunnable autoregressive training workflow. Its training settings are:

| Setting | Value |
|---|---:|
| `epochs` | 50 |
| Included checkpoint recorded epoch | 48 |
| `batch_size` | 512 |
| `hidden_size` | 256 |
| `latent_size` | 256 |
| `attention_heads` | 8 |
| `num_slots` | 8 |
| `encoder_layers` | 3 |
| `decoder_layers` | 2 |
| optimizer | AdamW |
| optimizer learning rate | `3e-4` |
| optimizer betas | `(0.9, 0.999)` |
| optimizer epsilon | `1e-8` |
| checkpoint optimizer weight decay | `1e-2` |
| initial `beta` variable | `0.01` |
| effective beta schedule | `(epoch % cycle_lenght / cycle_lenght) * max_beta` |
| `max_beta` | `0.03` |
| `cycle_lenght` | 15 |
| `corruption` | `False` |
| `mask_prob` if corruption is enabled | `0.05` |
| random-token corruption band if enabled | `0.10` |
| training loader shuffle | `True` |
| validation loader shuffle | `False` |
| notebook `num_workers` | 1 |
| checkpoint selection | best validation sequence accuracy |

The checkpoint epoch can be lower than `epochs` because the notebook saves the best validation checkpoint during the run. The released autoregressive results can be used directly without rerunning training, but rerunning the notebook with the same settings will regenerate the model from the same tokenization and split logic.

### Latent Encoding And Model Quality Benchmarks

Latents are encoded with the model in evaluation mode and without gradient tracking. `study/common/latents.py` returns the encoder mean `mu` as `float32`; its helper default is `batch_size=256`. The main benchmark notebooks use larger fixed evaluation batches where appropriate:

| Workflow | Batch/seed settings |
|---|---|
| `01_latent_quality_benchmarks_and_family_tests.ipynb` Phase 1 reconstruction-style evaluation | `PHASE1_BATCH_SIZE = 1024`, `SEED = 42` |
| Phase 3 prior or latent sampling evaluation | `PHASE3_BATCH_SIZE = 1024`, `SEED = 42` |
| interpolation tests | `N_STEPS = 11`, generator seeded from `SEED = 42` |
| family-retention tests | `PHASE5_RANDOM_SEED = 42`, minimum family group size `20` |
| AR source latent-quality notebook | latent encode batch `256`, traversal decode batch `128` |

Quality outputs include token accuracy, sequence accuracy, reconstruction validity, prior validity, interpolation validity, novelty/uniqueness summaries where available, and family-retention summaries. The curated result folders contain compact CSV/JSON summaries and figures only; regenerated latent arrays and full decode caches are intentionally excluded.

### Linear And Ridge Probe Settings

`study/notebooks/02_linear_probes_panels_and_residuals.ipynb` is the canonical linear-probe and residual-panel workflow. It uses:

| Setting | Value |
|---|---:|
| `SEED` | 42 |
| latent encoding `BATCH_SIZE` | 1024 |
| Ridge alpha grid | `np.logspace(-3, 3, 13)` |
| split random state | 42 |
| input scaling | `StandardScaler` fit on train rows |
| target scaling | `StandardScaler` fit on train rows |
| model selection | `RidgeCV(alphas=RIDGE_ALPHAS)` |
| reported metrics | train/validation/test `R2`, plus selected `alpha` where applicable |

For each target, finite rows are selected, latent coordinates are standardized using train rows, target values are standardized using train rows, and `RidgeCV` chooses an alpha from `0.001` through `1000` on the log-spaced grid. The fitted coefficients are transformed back to the original latent coordinate scale for direction analysis.

The same notebook fits three related probe families:

- `Z -> Y`: latent means to chemical properties.
- `Z -> C`: latent means to confound or syntax-like variables.
- `Z -> Y_residual`: latent means to property residuals after removing confound-predictable signal.

Residualization fits a multivariate `C -> Y` Ridge model on train rows using the same `RIDGE_ALPHAS` grid, predicts all valid rows, and stores `resid_{property}` columns. The residual probe then reruns the same `RidgeCV` procedure on the residual targets.

The reusable helper `study/common/probes.py` also exposes a fixed-alpha API. `fit_linear_probes(..., model_kind="ridge", alpha=1.0)` uses `Ridge(alpha=1.0)` by default, while `model_kind="linear"` uses ordinary `LinearRegression` and therefore has no alpha. `residualize_properties(..., alpha=10.0)` uses fixed `Ridge(alpha=10.0)` when the helper API is called directly outside the canonical notebook.

Recovered autoregressive Step 3 source notebooks used fixed Ridge settings for some compact recovered tables: `alpha=1e-2` for `Z -> Y` and `Z -> Y_residual`, `alpha=1e1` for `C -> Y` residualization, and `alpha=1.0` for `Z -> C` confound directions.

### MLP Probe Settings

`study/notebooks/04_mlp_probes_all_properties.ipynb` is the canonical MLP-probe workflow for the public export. It compares nonlinear MLP probes against the Step 3 Ridge baselines on raw and residualized targets.

| Setting | Value |
|---|---:|
| `SEED` | 42 |
| `BATCH_SIZE` | 8192 |
| `MAX_EPOCHS` | 15 |
| `MIN_EPOCHS` | 6 |
| `PATIENCE` | 4 |
| minimum validation improvement | `1e-3` in validation `R2` |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| hidden width | 256 |
| architecture | Linear, ReLU, Linear, ReLU, Linear |
| optimizer | AdamW |
| loss | MSE on standardized target |
| input scaling | latent train mean/std; std values below `1e-8` replaced by `1.0` |
| target scaling | train target mean/std; std values below `1e-8` replaced by `1.0` |
| data-loader shuffle | `True` for training |
| AMP | enabled when CUDA is available |

The MLP notebook stores the best in-memory state by validation `R2`. Training stops after at least `MIN_EPOCHS` once `PATIENCE` consecutive epochs fail to improve validation `R2` by more than `1e-3`. Reported scores are computed after inverse-transforming predictions back to original target units.

The autoregressive source MLP notebooks used the same batch size, max epochs, patience, learning rate, weight decay, and hidden width, but their compact recovered MLP tables come from an architecture `Linear -> GELU -> LayerNorm -> Linear -> GELU -> Linear`. Those AR source notebooks early-stopped on validation loss improvement greater than `1e-5`, with patience `4`.

The lightweight helper `study/common/probes.py` also exposes `run_mlp_probes` through scikit-learn for quick smoke checks. Its defaults are `hidden_layer_sizes=(256, 128)`, `random_state=42`, `max_iter=200`, and `early_stopping=True`. The notebook MLP described above is the configuration used for the main reported MLP comparison.

### Latent Traversal And Direction Settings

The generic traversal helper `study/common/traversal.py` normalizes the supplied direction vector and uses `np.linspace(-8.0, 8.0, 17)` when no alpha grid is provided. The main traversal notebook overrides this with the broader grid used in the study:

| Setting | Value |
|---|---:|
| traversal alphas | `np.linspace(-20.0, 20.0, 21)` |
| seeds per property | 3 |
| random generator | `np.random.default_rng(42)` |
| direction fit for notebook traversals | Ridge direction with `alpha=1.0` |
| decode batch size | 128 |
| traversal metrics | valid fraction, Spearman correlation of step vs actual property, actual-property slope vs alpha, monotonicity violations |

The source global-direction notebooks use `RIDGE_ALPHA_GRID = np.logspace(-3, 3, 13)` and select alphas on validation data when Step 3 alphas are not already available. Direction stability is estimated with `B = 8` bootstrap fits, using seed offsets `1000` for raw directions and `2000` for residual directions. Permutation and random-direction controls use seed offsets `3000` and `4000`.

The single-property traversal validation notebooks use a more detailed validation protocol:

| Setting | Value |
|---|---:|
| `SMOKE_MODE` | `False` for full runs |
| `SEED_COUNT` | 12 in full mode, 4 in smoke mode |
| `PACK_SIZE` | 16 in full mode, 4 in smoke mode |
| `STEP_GRID` | `np.arange(-4, 5)` in full mode |
| decode modes | `free_length`, `fixed_length_seed_pred` |
| Ridge alpha grid | `np.logspace(-4, 4, 17)` |
| benchmark pair count | 200 in full mode, 64 in smoke mode |
| default total span | `0.50` times the benchmark latent-distance span |
| maximum total span | `0.75` times the benchmark latent-distance span |
| seed bands | `low`, `mid`, `high` |

Those notebooks choose validation seeds from low, middle, and high property bands, traverse both raw and residual directions where available, and save compact summaries plus representative strips. Full generated path panels and intermediate latent arrays are not committed.

### Output Regeneration Policy

The notebooks regenerate large panels, latent matrices, direction arrays, MLP weights, and decode caches locally. The repository commits only compact artifacts needed for inspection: summary CSV files, JSON manifests, selected figures, and checkpoint/data assets tracked by Git LFS. To verify a checkout before rerunning studies, execute:

```bash
python scripts/verify_export.py
```

This parses notebooks, checks required assets, runs anonymization scans, compiles Python files, and validates available model/checkpoint compatibility.

## Results

Curated evidence is under `study/results/`:

- benchmark and family-retention metrics
- Step 3 confound summaries and compact R2/correlation tables
- Step 4 direction summaries and compact direction-quality tables
- single-property traversal summaries and representative figures
- MLP-vs-Ridge probe summaries
- ablation comparison summaries

The autoregressive result folder includes compact outputs from executed source notebooks: reconstruction and family-retention summaries, Ridge and linear-regression probe tables, MLP probe comparisons, missing-property probes, and copied traversal figures. The AR checkpoint is included under `checkpoints/`, so the AR notebooks can be rerun from the exported assets.

Large regenerated assets such as full panels, latent arrays, `.npz` direction caches, duplicate 300-550 MB CSVs, and MLP weight files are excluded.

