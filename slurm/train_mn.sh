#!/bin/bash
#SBATCH -J train_mn
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/train_mn_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/train_mn_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
##SBATCH --partition=a5,a6,2080
#SBATCH --exclude sls-a6-5
#SBATCH --mem=22G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=test
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate test

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs/src/meta_network/network/

srun hostname

python -u mlpv2.py

