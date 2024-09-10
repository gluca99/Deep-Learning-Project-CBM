#!/usr/bin/env bash
nvidia-smi
source activate /ceph/concept-driven-continual-learning/conda_env/merge_env

echo "gpt-time"
python ./sandbox-lf-cbm/GPT_init_concepts.py 
echo "concept set processing"
python ./sandbox-lf-cbm/GPT_conceptset_processor.py 

