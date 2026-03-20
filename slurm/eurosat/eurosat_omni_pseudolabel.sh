#!/bin/bash
#SBATCH -J eurosat_omni_pl
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/eurosat_omni_pl_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/eurosat_omni_pl_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# Pseudolabel generation: Qwen2.5-Omni on EuroSAT train set
# Produces pseudolabeled JSON that can be used as train data for SAV/CALM.

MODEL_NAME="qwen2.5_omni"
DATA_NAME="eurosat"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/eurosat/eurosat_train.json"
OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/eurosat"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task pseudolabel \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --output_dir   "${OUTPUT_DIR}" \
    --n_trials 8
