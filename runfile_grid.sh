#!/usr/bin/env bash
#SBATCH -A SNIC2020-33-8
#SBATCH -p alvis
#SBATCH -n 4
#SBATCH -t 0-12:00:00
#SBATCH --gpus-per-node=V100:1
#SBATCH --mail-user=emilio.jorge@chalmers.se --mail-type=END,FAIL
#module load intel/2019b Python/3.7.4
#module load intel/2018b Python/3.6.6 GCC
#--gpus-per-node=V100:1
module load CUDA/10.1.243
module load GCC/8.3.0

module load cuDNN/7.6.4.38
#module load CUDA/10.1.105
source venv/bin/activate

python3 image_captioning.py --lr $lr --dropout $dropout --cnn_top $cnn_top
