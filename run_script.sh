#!/bin/bash
#SBATCH -A dl
#SBATCH -n 2
#SBATCH --time=00:00:05
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=6G
#SBATCH --job-name=dl
#SBATCH --output=out/%j.out
#SBATCH --error=err/%j.err

bash ${HOME}/Deep-Learning-Project-CBM/script_dir/CBM/cc_cbm.sh
