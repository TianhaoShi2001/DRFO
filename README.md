# DRFO

**Version 1.0.0**

**Distributionally Robust Fair Optimization** for recommender systems with limited sensitive attributes.

This repository releases the **DRFO** training and evaluation pipeline, including:

- sensitive-attribute reconstruction
- MF pretraining
- DRFO training with utility-constrained early stopping
- the reconstruction-refusal extension
- released checkpoints for a fast evaluation path

Paper: *Fair Recommendations with Limited Sensitive Attributes: A Distributionally Robust Optimization Approach* (SIGIR 2024).

## Requirements

- Linux + NVIDIA GPU (the reference run used 8× 24GB GPUs; one GPU is enough)
- Conda environment (reference: Python 3.10, PyTorch 2.0.1 + CUDA 11.7)

```bash
conda env create -f environment.yml
conda activate drfo
```

## Repository layout

```text
drfo/engine/          # training engine
scripts/              # public entry points
configs/default.yaml  # hyper-parameters
artifacts/            # gammas, reconstructed labels, pretrained MF, checkpoints
docs/                 # data preparation
workspace/            # runtime workspace used by the engine
```

## Quick start (fast path)

Evaluate a released checkpoint (no training):

```bash
python scripts/eval_checkpoint.py --dataset ml-1m --known_ratio 0.3
python scripts/eval_checkpoint.py --dataset ml-1m --known_ratio 0.3 --refusal_ratio 1.0
```

## Full pipeline (slow path)

1. Prepare data: see `docs/DATA.md`
2. (Optional) reconstruct attributes / pretrain MF, or use `artifacts/`
3. Train DRFO:

```bash
python scripts/train_drfo.py --dataset ml-1m --known_ratio 0.3 --gpu 0
```

4. Train the refusal extension (`refusal_ratio=1.0` = all unknown users refuse reconstruction):

```bash
python scripts/train_drfo_refusal.py --dataset ml-1m --known_ratio 0.3 --refusal_ratio 1.0 --gpu 0
```

Default early stopping keeps the fairest validation model within a 2% RMSE budget (`performance_scale=0.98`), for at most 500 epochs.

## Default grids

| Setting | Values |
|---------|--------|
| Datasets | `ml-1m`, `tenrec` |
| `known_ratio` | `0.1, 0.3, 0.5, 0.7, 0.9` |
| Refusal (`known_ratio` ∈ {0.3, 0.5}) | `refusal_ratio` ∈ {0.25, 0.5, 0.75, 1.0} |
| Seed | `2004` |

See `configs/default.yaml` for optimizer hyper-parameters.
