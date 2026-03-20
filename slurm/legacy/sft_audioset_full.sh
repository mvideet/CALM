#!/bin/bash
#SBATCH -J sft_audioset_full
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/sft_audioset_full_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/sft_audioset_full_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_20shot_train_individual_mcqs.json"
OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/checkpoints/sft_audioset_full"

EPOCHS=1
LR=1e-5
GRAD_ACCUM=8

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.sft \
    --train_path   "${TRAIN_PATH}" \
    --output_dir   "${OUTPUT_DIR}" \
    --epochs       "${EPOCHS}" \
    --lr           "${LR}" \
    --grad_accum   "${GRAD_ACCUM}" \
    --full_sft
