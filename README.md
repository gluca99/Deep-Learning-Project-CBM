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
pip install Ninja
pip install avalanche-lib==0.2.1
pip install ftfy regex
conda deactivate
```
## Running IN2 without pretrained backbone
python sandbox-lf-cbm/train_cbm.py --nonlinear True --seed 3456 --gamma 0.2 --solver grad --lam 0 --concept_method c1 --dataset cifar100_task_0_5_3456 --concept_set sandbox-lf-cbm/data/concept_sets/separated_concepts/cifar100_task_0_3456_filtered.txt --save_dir ~/result/pretrain_correct --freeze_wc True --normalize_wf False --pretrain

### 4. Get experiment results
Execute the following code to get the experiment results. 
```
python evaluate/metric.py --file_dir results/cc_cbm --strategy cc_cbm --task_num 5 
```

