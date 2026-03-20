#!/bin/bash
#SBATCH -J eurosat_omni_zs
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/eurosat_omni_zs_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/eurosat_omni_zs_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# Zero-shot inference: Qwen2.5-Omni on EuroSAT (6k test subset)

MODEL_NAME="qwen2.5_omni"
DATA_NAME="eurosat"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/eurosat/eurosat_test_6k.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task classify \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --test_path    "${TEST_PATH}" \
    --zero_shot_only
