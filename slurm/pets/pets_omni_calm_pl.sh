#!/bin/bash
#SBATCH -J pets_omni_calm_pl
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/pets_omni_calm_pl_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/pets_omni_calm_pl_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# CALM + pseudo-labeling: Qwen2.5-Omni on Oxford Pets
# NOTE: Run pets_omni_pseudolabel.sh first.

MODEL_NAME="qwen2.5_omni"
DATA_NAME="pets"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/pets/pseudolabeled_pets_qwen2.5_omni_8trials.json"
VAL_PATH="${TRAIN_PATH}"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/pets/pets_test.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task classify \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --test_path    "${TEST_PATH}" \
    --weight_scheme margin_clamped \
    --tau 0.001 0.01 0.03 0.05 0.07 0.1 \
    --tau_w 0.1 0.3 0.5 1.0 2.0 \
    --top_k 5 10 20 40 100 300 500 784 \
    --last_n_tokens 1 \
    --n_trials 1 \
    --cache_dir "./cache/pets_omni_calm_pl"
