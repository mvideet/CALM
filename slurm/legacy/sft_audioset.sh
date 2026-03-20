#!/bin/bash
#SBATCH -J sft_audioset
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/sft_audioset_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/sft_audioset_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_20shot_train_individual_mcqs.json"
OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/checkpoints/sft_audioset_lora"

EPOCHS=1
LR=3e-4
LORA_RANK=4
LORA_ALPHA=8
GRAD_ACCUM=8

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.sft \
    --train_path   "${TRAIN_PATH}" \
    --output_dir   "${OUTPUT_DIR}" \
    --epochs       "${EPOCHS}" \
    --lr           "${LR}" \
    --lora_rank    "${LORA_RANK}" \
    --lora_alpha   "${LORA_ALPHA}" \
    --grad_accum   "${GRAD_ACCUM}"
