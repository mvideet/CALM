#!/bin/bash
#SBATCH -J pets_omni_zs
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/pets_omni_zs_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/pets_omni_zs_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# Zero-shot inference: Qwen2.5-Omni on Oxford Pets

MODEL_NAME="qwen2.5_omni"
DATA_NAME="pets"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/pets/pets_test.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task classify \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --test_path    "${TEST_PATH}" \
    --zero_shot_only
