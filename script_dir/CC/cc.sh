#!/usr/bin/env bash
nvidia-smi
source activate ~/concept-driven-continual-learning/conda_env/merge_env

echo "python training"
MODEL=resnet18
strategy=SRT
task_num=10
pretrain=none

sol=sol0
DATASET=cifar100
download_data=~/concept-driven-continual-learning/download_data

BATCH_SIZE=128
ewc_lambda=0.4
zero_threshold=0.1

SEED=3456
result=~/concept-driven-continual-learning/results/rebuttal/tab1/SEED_${SEED}/${MODEL}_${DATASET}_${task_num}task_20epoch/${strategy}_${sol}
python ~/concept-driven-continual-learning/continual_learning/train_all.py --task_num ${task_num} --pretrain ${pretrain} --ewc_lambda ${ewc_lambda} --sol ${sol} --probing_method CLIP-Dissect --seed ${SEED} --ck_dir ~/concept-driven-continual-learning/ck_dir/${MODEL}/${strategy}/SEED_${SEED}/ --l1_coeff 0.000001 --zero_threshold ${zero_threshold} --batch_size ${BATCH_SIZE} --model ${MODEL} --strategy ${strategy} --dataset ${DATASET} --result_dir ${result} --download_data ${download_data}
for ((i=0; i<$task_num; i++));
do
rm -rf "${result}/broden_${MODEL}-${strategy}-${DATASET}-task${i}-${task_num}_layer4.pt"
done
rm -rf "${result}/broden_ViT-B16.pt"
rm -rf "${result}/20k_ViT-B16.pt"

sol=sol1
result=~/concept-driven-continual-learning/results/rebuttal/tab1/SEED_${SEED}/${MODEL}_${DATASET}_${task_num}task_20epoch/${strategy}_${sol}
python ~/concept-driven-continual-learning/continual_learning/train_all.py --task_num ${task_num} --pretrain ${pretrain} --ewc_lambda ${ewc_lambda} --sol ${sol} --probing_method CLIP-Dissect --seed ${SEED} --ck_dir ~/concept-driven-continual-learning/ck_dir/${MODEL}/${strategy}/SEED_${SEED}/ --l1_coeff 0.000001 --zero_threshold ${zero_threshold} --batch_size ${BATCH_SIZE} --model ${MODEL} --strategy ${strategy} --dataset ${DATASET} --result_dir ${result} --download_data ${download_data}
for ((i=0; i<$task_num; i++));
do
rm -rf "${result}/broden_${MODEL}-${strategy}-${DATASET}-task${i}-${task_num}_layer4.pt"
done
rm -rf "${result}/broden_ViT-B16.pt"
rm -rf "${result}/20k_ViT-B16.pt"

