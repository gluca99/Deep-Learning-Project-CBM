#!/usr/bin/env bash
nvidia-smi
source activate ~/concept-driven-continual-learning/conda_env/merge_env

echo "python training"
MODEL=resnet18
strategy=SRT #Continual Learning Methods
task_num=10 # Number of Task. | "5" :Table 2,7,10,23 | "10": Table 12| "20": Table 14
pretrain=none #True for pretrained backbone

DATASET=cifar100  
download_data=~/concept-driven-continual-learning/download_data

BATCH_SIZE=128
ewc_lambda=0.4 # prediction layer weight regularization (lambda). Different hyperparameter: Table 33 [0,1, 4]
l1_coeff=0.000001 # l1 sparse regularization (mu). Different hyperparameter: Table 32 [1e-5, 1e-7]
zero_threshold=0.15 # zero threshold (tau). Different hyperparameter: Table 35 [0.125, 0.2]

SEED=3456 # Task distribution. Table 23: "fix_order"
sol=sol0 #freeze-all
result=~/concept-driven-continual-learning/results/rebuttal/tab1/SEED_${SEED}/${MODEL}_${DATASET}_${task_num}task_20epoch/${strategy}_${sol}
python ~/concept-driven-continual-learning/continual_learning/train_all.py --task_num ${task_num} --pretrain ${pretrain} --ewc_lambda ${ewc_lambda} --sol ${sol} --probing_method CLIP-Dissect --seed ${SEED} --ck_dir ./concept-driven-continual-learning/ck_dir/${MODEL}/${strategy}/SEED_${SEED}/ --l1_coeff ${l1_coeff} --zero_threshold ${zero_threshold} --batch_size ${BATCH_SIZE} --model ${MODEL} --strategy ${strategy} --dataset ${DATASET} --result_dir ${result} --download_data ${download_data}
for ((i=0; i<$task_num; i++));
do
rm -rf "${result}/broden_${MODEL}-${strategy}-${DATASET}-task${i}-${task_num}_layer4.pt"
done
rm -rf "${result}/broden_ViT-B16.pt"
rm -rf "${result}/20k_ViT-B16.pt"

sol=sol1 #freeze-part
result=~/concept-driven-continual-learning/results/rebuttal/tab1/SEED_${SEED}/${MODEL}_${DATASET}_${task_num}task_20epoch/${strategy}_${sol}
python ~/concept-driven-continual-learning/continual_learning/train_all.py --task_num ${task_num} --pretrain ${pretrain} --ewc_lambda ${ewc_lambda} --sol ${sol} --probing_method CLIP-Dissect --seed ${SEED} --ck_dir ./concept-driven-continual-learning/ck_dir/${MODEL}/${strategy}/SEED_${SEED}/ --l1_coeff ${l1_coeff}  --zero_threshold ${zero_threshold} --batch_size ${BATCH_SIZE} --model ${MODEL} --strategy ${strategy} --dataset ${DATASET} --result_dir ${result} --download_data ${download_data}
for ((i=0; i<$task_num; i++));
do
rm -rf "${result}/broden_${MODEL}-${strategy}-${DATASET}-task${i}-${task_num}_layer4.pt"
done
rm -rf "${result}/broden_ViT-B16.pt"
rm -rf "${result}/20k_ViT-B16.pt"