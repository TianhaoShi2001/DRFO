# DRFO
This is an implementation for our SIGIR 2024 paper "Fair Recommendations with Limited Sensitive Attributes: A Distributionally Robust Optimization Approach" based on PyTorch. This Repo contains the required code to run DRFO on Tenrec and MovieLens.
Below are the file descriptions:

## Files

- `Dataset.py`: Handles dataset preprocessing and management.
- `MF_drfo.py`: Implements DRFO algorithm.
- `MF_drfo_extension.py`: Extends DRFO to manage user rejection of sensitive attribute reconstruction.
- `mymodels.py`: Contains model definitions and implementations.
- `Reconstructor.py`: Implements sensitive attribute reconstruction using SVM.

## Notes
- Only the code is included here. 
- A cleaned and faster version of the full pipeline will be uploaded by October.
