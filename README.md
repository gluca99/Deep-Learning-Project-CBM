# Codebase for Deep Learning Project at ETH Zurich HS2024
By Marcus Roberto Nielsen, Luca Ghafourpour and Tijn De Wringer

## Clone GitHub Repository
```
git clone https://github.com/gluca99/Deep-Learning-Project-CBM.git
cd Deep-Learning-Project-CBM
```

## Setup conda environment
Execute the following code to set up the code environment. (This may take a couple of minutes to finish)
```
conda env create -f setup/denv.yml
conda activate denv
pip uninstall torch, torchvision
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install Ninja
pip install avalanche-lib==0.2.1
pip install ftfy regex
conda deactivate
```
## Running IN2 without pretrained backbone
```
python sandbox-lf-cbm/train_cbm.py --nonlinear True --seed 3456 --gamma 0.2 --solver grad --lam 0 --concept_method c1 --dataset cifar100_task_0_5_3456 --concept_set sandbox-lf-cbm/data/concept_sets/separated_concepts/cifar100_task_0_3456_filtered.txt --save_dir ~/result/test --freeze_wc True --normalize_wf False --proj_batch_size 128 --proj_steps 100 --interpretable --extend_concept_set
```

## Download pretrained backbone 
Download pretrained backbone from this repo https://github.com/Trustworthy-ML-Lab/Label-free-CBM and put into sandbox-lf-cbm/data folder

## Run experiments
The experiments with and without pretrained backbone can be run using the following bash commands. Look into each batch script for parameter specific details.
```
bash script_dir/CBM/nopretrain_in2.sh
bash script_dir/CBM/pretrain_in2.sh
```

### Get experiment results
Execute the following code to get the experiment results. 
```
python evaluate/metric.py --file_dir results/cc_cbm --strategy cc_cbm --task_num 5 
```

