#!/usr/bin/env bash
#nvidia-smi
ROOT=Deep-Learning-Project-CBM
#conda init
source ~/miniconda3/etc/profile.d/conda.sh
conda activate denv
echo "python training"
con_method=c1
MODEL=resnet18
DATASET=cifar100
dataset=cifar100
BATCH_SIZE=256
SEED=3456
ewc_lambda=0.2
spar_lam=0
mem=150
nonlinear=True
#pretrain=False
dif=False
download_data=~/${ROOT}/download_data

task_num=5

freeze=True # adjust save_dir as well!
nor_wf=False
nor_method=all
solver=grad
gamma=0.2

SAVE_DIR=~/result/cc_cbm/

for ((i=0; i<$task_num; i++));
do
DATASET=${dataset}_task_${i}_${task_num}_${SEED}
concept_path=~/${ROOT}/sandbox-lf-cbm/data/concept_sets/separated_concepts/${dataset}_task_${i}_${SEED}_filtered.txt
python ~/${ROOT}/sandbox-lf-cbm/train_cbm.py --nonlinear ${nonlinear} --seed ${SEED} --gamma ${gamma} --solver ${solver} --lam ${spar_lam} --concept_method ${con_method} --dataset ${DATASET} --concept_set ${concept_path} --save_dir ${SAVE_DIR} --freeze_wc ${freeze} --normalize_wf ${nor_wf} 
done

python ~/${ROOT}/sandbox-lf-cbm/evaluate_cbm.py --task_num ${task_num} --dataset ${dataset} --seed ${SEED} --save_dir ${SAVE_DIR}  --nonlinear ${nonlinear}
