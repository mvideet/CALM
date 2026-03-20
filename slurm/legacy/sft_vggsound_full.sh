#!/bin/bash
#SBATCH -J sft_vgg_full
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/sft_vgg_full_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/sft_vgg_full_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/checkpoints/sft_vggsound_full"

EPOCHS=1
LR=3e-5
GRAD_ACCUM=8

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.sft \
    --train_path   "${TRAIN_PATH}" \
    --output_dir   "${OUTPUT_DIR}" \
    --epochs       "${EPOCHS}" \
    --lr           "${LR}" \
    --grad_accum   "${GRAD_ACCUM}" \
    --full_sft
