# Data preparation

## Datasets

- **ML-1M**: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- **Tenrec**: follow the official Tenrec release used in the paper (QB-video subset)

## Preprocess

1. Download the raw files into a working directory.
2. Run the notebooks under `docs/`:
   - `preprocess_ml-1m.ipynb`
   - `preprocess_tenrec.ipynb`
3. Place the resulting 50-core HDF5 tables and point the engine to them
   (absolute paths are set in `drfo/engine/Dataset.py`; edit to your data root):

```text
<DATA_ROOT>/ml-1m-50core/ml-1m.h5
<DATA_ROOT>/tenrec-50core/tenrec.h5
```


## Released artifacts (no recompute needed)

For the default seed (`2004`) we ship:

| Artifact | Path |
|----------|------|
| Reconstructed labels | `artifacts/reconstructed_labels/{dataset}/known_ratio_*.csv` |
| Ambiguity radii (γ) | `artifacts/gammas/{dataset}/known_ratio_*.pt` |
| Pretrained MF | `artifacts/pretrained_mf/{dataset}/known_ratio_*.pt` |
| DRFO checkpoints | `artifacts/checkpoints/{dataset}/known_ratio_*_seed_2004/` |

These files are sufficient for the **fast** evaluation path.
