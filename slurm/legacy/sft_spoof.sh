#!/bin/bash
#SBATCH -J sft_spoof
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/sft_spoof_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/sft_spoof_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/LA_train.json"
OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/checkpoints/sft_spoof_lora"

MAX_STEPS=1000
LR=5e-4
LORA_RANK=2
LORA_ALPHA=4
GRAD_ACCUM=4

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.sft \
    --train_path   "${TRAIN_PATH}" \
    --output_dir   "${OUTPUT_DIR}" \
    --epochs       1 \
    --max_steps    "${MAX_STEPS}" \
    --lr           "${LR}" \
    --lora_rank    "${LORA_RANK}" \
    --lora_alpha   "${LORA_ALPHA}" \
    --grad_accum   "${GRAD_ACCUM}"
