## Overlap Integral Prediction — Full Process

This document explains the end-to-end workflow used to generate data, analyse it, and train a graph neural network (GNN) that predicts atomic orbital overlap integrals. It expands the short outline in `Overlap_Prediction.md` into a full reproducible guide and references the Hydra-based training entrypoint `train_overlap_hydra.py`.

## Contents
- Overview
- Prerequisites
- Generating the dataset
- Data format and metadata
- Data inspection and analysis tools
- Overlap normalisation procedure
- The ML model (architecture & training)
- How to train (examples using `train_overlap_hydra.py`)
- Checkpoints, logging and reproducibility
- Troubleshooting & tips
- Next steps and references

## Overview

The goal is to predict overlap integrals between Gaussian-type orbitals (GTOs) using a GNN that consumes molecular graphs, node features (including GTO exponents), and optional virtual nodes. The pipeline has three broad stages:

1. Generate and store dataset in a structured form.
2. Analyse and normalise overlap values (per-L when relevant) and create transforms for training.
3. Train the GNN (Hydra + PyTorch Lightning) using `train_overlap_hydra.py` which wires together the `OverlapDataModule` and `OverlapLightningModule` from `scdp.model.overlap`.

## Prerequisites

- A working Python environment with project dependencies installed - use the `environment.yaml` file to create the scdp-overlap env.
- Access to the scdp molecular dataset (the project assumes data in a shared scratch area for full generation; see notes below).


## Generating the dataset

Primary generator: `overlap_pred/gen_ovlp_data.py`.

Key behaviours and common arguments:

- `--full_dataset` : iterate over entire dataset of molecules
- `--use_vnodes` : include virtual nodes in the graph representation
- `--use_custom_gtos` : construct GTOs consistently across nodes and vnodes (e.g., up to L=5)
- `--add_diversity` : randomise initial exponent values to increase exponent coverage across molecules

Example usage:

```
python overlap_pred/gen_ovlp_data.py --full_dataset --use_vnodes --use_custom_gtos --add_diversity
```

**Note:** The scdp data is accessed from the scratch folder of either plippman or mklockow depending on whether vnodes are used or not. The longevity of this data is not guaranteed. Please check with Peter or Manuel if you are having issues accessing the scdp data.

## Data format and metadata

Each generated dataset contains:

- Per-molecule files or an aggregated dataset containing graphs, node/GTO features, and pairs for which overlaps are computed.
- For each overlap record: the pair of orbitals (node indices or vnode indices), the exponent(s) used, angular momentum L, and the scalar overlap value.
- Metadata file (JSON) produced by the `OverlapDataModule` when `setup(stage="fit")` is called; `train_overlap_hydra.py` writes `metadata.json` into the configured `core.storage_dir`.

Store these consistently: the training scripts expect a dataset layout compatible with `OverlapDataModule` (see `scdp.model.overlap`).

## Data inspection and analysis tools

Several scripts help inspect dataset structure and the relationship between overlaps and exponents. Key scripts include:

- `scripts/analyse_exponent_overlap.py` — scatter plots, interactive visualisations, and per-L views of overlap vs exponent.
  - Important args: `--data-dir` (path to generated data), `--sample-size` (number of molecules to plot), `--outdir` (where plots go), `--interactive-l` (which L value to make interactive plots for). Example:

```
python scripts/analyse_exponent_overlap.py --data-dir /export/data/hmichael/scdp/data/full_log_gen_new --sample-size 20 --outdir plots/overlap_analysis --interactive-l 0
```

- `scripts/apply_overlap_normalisation.py` — performs the binned normalisation procedure (means/stds per bin and per-L), smooths the curve, and saves normalisation curves to disk. These curves are consumed by the training data transforms.

Use these tools to:

- Explore coverage of exponent space across molecules
- Visualise how overlap values vary with exponents and L
- Generate and validate normalisation curves that will be applied as transforms during training

## Overlap normalisation procedure

Why: Overlap values vary across exponent ranges and L shells. Normalisation reduces scale differences and helps stable training.

High-level procedure (implemented in `apply_overlap_normalisation.py`):

1. Bin exponent values into a fixed set of bins across the exponent range.
2. For each bin and each L, compute mean and standard deviation of overlap values in that bin.
3. Fit smooth curves (e.g., spline or low-degree polynomial smoothing) over the binned means and stds to avoid noisy bin edges.
4. Save the per-L smoothing curves (means & stds) to disk.
5. Training transforms use these saved curves to normalise overlap targets on-the-fly.

Key args to `apply_overlap_normalisation.py` are `--sample-size` and `--bins`.

## The ML model (architecture & training)

High-level components:

- `OverlapDataModule` (in `scdp.model.overlap`) — prepares train/val/test splits, applies transforms including the overlap normalisation, and provides dataloaders.
- `OverlapLightningModule` (in `scdp.model.overlap`) — Lightning module that wraps model forward, loss computation, metrics, optimizer and scheduler.
- The model uses a GNN from the main scdp codebase but with two key changes for overlap prediction:
  - An additional per-orbital input: the exponent value embedded equivariantly.
  - The output is the overlap integral (scalar) for a given pair (or set) of orbitals rather than density reconstruction coefficients.

Reference training entrypoint: `train_overlap_hydra.py`.

Important implementation notes from `train_overlap_hydra.py`:

- Hydra-based configuration. The file is invoked with `config_path=str(PROJECT_ROOT / "scdp" / "config")` and default `config_name="overlap_full"`.
- The script instantiates the `cfg.data` datamodule and calls `datamodule.setup(stage="fit")` to populate metadata. The metadata is saved to `metadata.json` in the training storage directory.
- The model is instantiated via `hydra.utils.instantiate(cfg.model, train=cfg.train, metadata=metadata, no_val_data=not has_val_data, _recursive_=False)`.
- Custom callbacks included: `EpochTimingCallback` and `CheckpointTimingCallback` to log timing information. The script also adapts early stopping / checkpoint monitors to `train_loss` when no validation data exists.
- Lightning `Trainer` setup respects `cfg.train.trainer` and supports DDP when requested (`trainer_cfg['strategy'] = 'ddp'`). When DDP is enabled and `LOCAL_RANK` is present, the script sets `TORCH_DISTRIBUTED_TIMEOUT` to extend timeouts and prepares device assignment.

Contract (what the training run expects and produces):

- Inputs: dataset path (configured via `cfg.data`), saved normalisation curves (if used), Hydra configuration.
- Outputs: checkpoints in `core.storage_dir`, `config.yaml` and `metadata.json` saved to the same directory, tensorboard logs (if enabled), and final metrics.
- Error modes: missing dataset files, mismatched metadata, or GPU/Distributed misconfiguration produce explicit logs and exceptions.

Edge cases to watch:

- Sparse coverage in exponent bins: smoothing may be poor if a bin has few points — increase dataset sample or adjust bins.

## How to train (examples using `train_overlap_hydra.py`)

Basic run (default config `overlap_full` in `scdp/config`):

```
python train_overlap_hydra.py
```

Run with a different storage directory or hyperparameter override (Hydra syntax):

```
python train_overlap_hydra.py core.storage_dir=/path/to/output train.trainer.max_epochs=10 data.batch_size=32
```

Run on multiple GPUs with DDP:

```
python train_overlap_hydra.py trainer.devices=[0,1,2,3] trainer.strategy=ddp
```

Notes about resuming and checkpoints:

- The script looks for `last.ckpt` in `core.storage_dir`. If absent it searches for `*epoch*.ckpt` and picks the most recent.
- To resume manually, pass `trainer.fit(..., ckpt_path='/path/to/ckpt')` by editing the Hydra config or copying the checkpoint into the storage dir as `last.ckpt`.

Running tests after training:

If the datamodule provides test data (`datamodule.test_dataset is not None`), `train_overlap_hydra.py` calls `trainer.test(datamodule=datamodule, ckpt_path="best")` after training completes.

## Checkpoints, logging and reproducibility

- Hydra config and an uploaded `config.yaml` are saved to `core.storage_dir` for reproducibility.
- `metadata.json` from the datamodule is saved alongside configs.
- If a TensorBoard logger is configured in `cfg.train.logging.tensorboard`, a `TensorBoardLogger` is created and the safe subset of hyperparameters is logged.
- The script attempts to log a safe subset of hyperparameters (model name, data module name, batch size, learning rate, number of layers, hidden channels, cutoff, etc.). If logging fails, a warning is issued but training proceeds.

Reproducibility tips:

- Use `cfg.train.seed` and ensure `seed_everything(cfg.train.seed, workers=True)` is set (the script respects `cfg.train.seed`).
- Save `core.storage_dir` contents and consider versioning the directory (git doesn't store large artifacts — use digitalocean/s3 or institutional storage).

## Troubleshooting & tips

- Note there is an unresolved issue with workers after normalisation - runs sometimes fail if workers > 0.
- If training reports no validation data: verify `OverlapDataModule` splits and that `datamodule.setup(stage='fit')` populates `val_dataset`.
- If DDP errors appear, check `LOCAL_RANK` environment variable and CUDA device assignment. The script sets `TORCH_DISTRIBUTED_TIMEOUT` to 1800s when a distributed environment is detected.
- If overlap normalisation looks wrong near the edges of exponent ranges, increase sample size for binning or use wider smoothing windows.

Debugging workflow:

1. Run `gen_ovlp_data.py` with `--sample-only` or a light sample to generate a small dataset.
2. Use `scripts/analyse_exponent_overlap.py` with a small `--sample-size` to visualise behaviour.
3. Run `scripts/apply_overlap_normalisation.py` to generate normalisation curves and inspect them.
4. Run `train_overlap_hydra.py` with a reduced `data.batch_size` and `train.trainer.max_epochs=1` to ensure end-to-end wiring works.

## Next steps and references

- Add a small test dataset with known overlap values for a unit test that verifies `OverlapDataModule` and the normalisation transforms.
- Expand `scripts/analyse_exponent_overlap.py` with more interactive tooling (zooming & brushing) for per-molecule inspection.
- Add visualization callbacks to Lightning to summarise overlap prediction errors grouped by exponent bin and L during training.

References

- `overlap_pred/gen_ovlp_data.py` — dataset generation
- `scripts/analyse_exponent_overlap.py` — data visualisation
- `scripts/apply_overlap_normalisation.py` — normalisation pipeline
- `train_overlap_hydra.py` — Hydra entrypoint for training (uses `OverlapDataModule` and `OverlapLightningModule` in `scdp.model.overlap`)

If you want, I can also:

- Add a short example Hydra config for a minimal local run.
- Create a small unit test that loads the datamodule for a tiny dataset and verifies shapes.

---
File created to capture the full process for overlap integral prediction and training using the scdp codebase.
