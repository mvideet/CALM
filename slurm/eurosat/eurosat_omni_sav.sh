#!/bin/bash
#SBATCH -J eurosat_omni_sav
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/eurosat_omni_sav_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/eurosat_omni_sav_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# SAV (supervised): Qwen2.5-Omni on EuroSAT

MODEL_NAME="qwen2.5_omni"
DATA_NAME="eurosat"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/eurosat/eurosat_train.json"
VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/eurosat/eurosat_val.json"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/eurosat/eurosat_test_6k.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task sav \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --test_path    "${TEST_PATH}" \
    --sav_num_heads 20 40 100 300 500 784 \
    --cache_dir "./cache/eurosat_omni_sav"
