#!/bin/bash
#SBATCH -A dl_jobs
#SBATCH -n 2
#SBATCH --time=240
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=24G
#SBATCH --tmp=8G
#SBATCH --job-name=trainjob
#SBATCH --output=out/%j.out
#SBATCH --error=err/%j.err
ROOT=Deep-Learning-Project-CBM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate denv
echo "Training CBMs without pretrained backbone"

con_method=c1
MODEL=resnet18_places # (same as resnet18)
dataset=cifar100
PROJ_BATCH_SIZE=128
PROJ_STEPS=1000
VAL_LOG_INTERVAL=50 # number of epochs to check whether val sim has increased
SEED=3456

spar_lam=0
nonlinear=True

task_num=5 # number of tasks

freeze=True # adjust save_dir as well!
nor_wf=False
solver=grad
gamma=0.2

SAVE_DIR=~/result/seed${SEED}/nopretrain
#SAVE_DIR=~/result/${SEED}/nopretrain

for ((i=0; i<$task_num; i++));
do
DATASET=${dataset}_task_${i}_${task_num}_${SEED}
concept_path=~/${ROOT}/sandbox-lf-cbm/data/concept_sets/separated_concepts/${dataset}_task_${i}_${SEED}_filtered.txt
#concept_path=~/${ROOT}/sandbox-lf-cbm/data/concept_sets/separated_concepts/our_separated_concepts/${dataset}_task_${i}_${SEED}.txt
python ~/${ROOT}/sandbox-lf-cbm/train_cbm.py --nonlinear ${nonlinear} --seed ${SEED} --gamma ${gamma} --solver ${solver} --lam ${spar_lam} --concept_method ${con_method} --dataset ${DATASET} --concept_set ${concept_path} --save_dir ${SAVE_DIR} --freeze_wc ${freeze} --normalize_wf ${nor_wf} --proj_steps ${PROJ_STEPS} --proj_batch_size ${PROJ_BATCH_SIZE} --val_log_interval ${VAL_LOG_INTERVAL} --interpretable
done

python ~/${ROOT}/sandbox-lf-cbm/evaluate_cbm.py --task_num ${task_num} --dataset ${dataset} --seed ${SEED} --save_dir ${SAVE_DIR} --nonlinear ${nonlinear}
#python ~/${ROOT}/sandbox-lf-cbm/evaluate_cbm.py --task_num ${task_num} --dataset ${dataset} --seed fix --task_name ${SEED} --save_dir ${SAVE_DIR} --nonlinear True
