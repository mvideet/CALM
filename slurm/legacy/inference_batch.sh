#!/bin/bash
#SBATCH -J esc_ltu
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/esc_ltu_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/esc_ltu_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6,a5
##SBATCH --partition=a5,a6,2080
#SBATCH --exclude sls-a6-5
#SBATCH --mem=22G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=test
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate test

python -u /data/sls/u/urop/mvideet/sparse_audio/ltu/src/ltu/inference_batch_omni.py


