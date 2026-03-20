#!/bin/bash
#SBATCH -J pets_omni_sav_pl
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/pets_omni_sav_pl_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/pets_omni_sav_pl_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# SAV + pseudo-labeling: Qwen2.5-Omni on Oxford Pets
# NOTE: Run pets_omni_pseudolabel.sh first.

MODEL_NAME="qwen2.5_omni"
DATA_NAME="pets"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/pets/pseudolabeled_pets_qwen2.5_omni_8trials.json"
VAL_PATH="${TRAIN_PATH}"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/pets/pets_test.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task sav \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --test_path    "${TEST_PATH}" \
    --sav_num_heads 20 40 100 300 500 784 \
    --cache_dir "./cache/pets_omni_sav_pl"
