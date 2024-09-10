#!/usr/bin/env bash
nvidia-smi
conda env create -f ~/concept-driven-continual-learning/setup/denv.yml -p ~/concept-driven-continual-learning/conda_env/merge_env
#conda env list
source activate ~/concept-driven-continual-learning/conda_env/merge_env
pip install Ninja
pip install avalanche-lib==0.2.1
conda deactivate
