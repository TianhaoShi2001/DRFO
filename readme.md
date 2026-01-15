# Project Overview

This is an implementation for our SIGIR 2024 paper "Fair Recommendations with Limited Sensitive Attributes: A Distributionally Robust Optimization Approach" based on PyTorch. This Repo contains the required code to run DRFO on Tenrec and MovieLens. Below are the file descriptions:

## Files

`Dataset.py`:  
Dataset loading and preprocessing logic.

`experiment_config.py`:  
Experimental configuration.

`load_models.py`:  
Model loading utilities.

`MF_base.py`:  
Standard matrix factorization (MF) backbone.

`MF_drfo.py`:  
MF with DRFO optimization.

`MF_drfo_extension.py`:  
DRFO with user rejection modeling.

`mybasemodels.py`:  
Base model definitions.

`mymodels.py`:  
Extended model definitions.

`Reconstructor.py`:  
Sensitive attribute reconstruction using SVM.

`preprocess_ml-1m.ipynb`:  
Preprocessing script for MovieLens-1M.

`preprocess_tenrec.ipynb`:  
Preprocessing script for TenRec.

`ratings.dat`, `users.dat`, `movies.dat`:  
Raw MovieLens-1M data files.

`QB-video.csv`:  
Raw TenRec data file.

## Usage

### 1. Preprocess data

1. Run the corresponding notebook first.

jupyter notebook preprocess_ml-1m.ipynb
jupyter notebook preprocess_tenrec.ipynb

2. Sensitive attribute reconstruction

python Reconstructor.py --data_name ml-1m
python Reconstructor.py --data_name tenrec

3. MF backbone
python MF_base.py --data_name ml-1m
python MF_base.py --data_name tenrec


4. MF with DRFO
python MF_drfo.py --data_name ml-1m --know_size 0.3
python MF_drfo.py --data_name tenrec --know_size 0.3
know_size ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

5. MF with DRFO extension
python MF_drfo_extension.py --data_name ml-1m --know_size 0.3 --lack_profile_prob 0.5 
python MF_drfo_extension.py --data_name tenrec --know_size 0.3 --lack_profile_prob 0.5
know_size ∈ {0.3, 0.5}
lack_profile_prob ∈ {0, 0.25, 0.5, 0.75, 1}