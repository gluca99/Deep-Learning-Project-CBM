#!/bin/bash
source activate ~/concept-driven-continual-learning/conda_env/merge_env
DATASET=cifar10
SEED=3456
TASK_NUM=5
result=~/concept-driven-continual-learning/results/ST/SEED_${SEED}/resnet18_${DATASET}_5task_20epoch
ck_dir=~/concept-driven-continual-learning/ck_dir/SEED_${SEED}/



tasks=("SI" "LWF" "MIR" "GEM" "Naive" "EWC")

for t in "${tasks[@]}"; do 
for ((i=0; i<${TASK_NUM}; i++));
do

model=resnet18-${t}-${DATASET}-task${i}-${TASK_NUM}
result_dir=${result}/${t}
(cd ~/concept-driven-continual-learning/sandbox-clip-dissect-main; python ~/concept-driven-continual-learning/sandbox-clip-dissect-main/describe_neurons.py --target_model ${model} --ck_dir ${ck_dir} --activation_dir ${result} --result_dir ${result})

done
done
