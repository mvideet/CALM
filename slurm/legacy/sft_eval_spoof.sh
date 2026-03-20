#!/bin/bash
#SBATCH -J sft_eval_spoof
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/sft_eval_spoof_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/sft_eval_spoof_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

CHECKPOINT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/checkpoints/sft_spoof_lora"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/LA_eval.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.sft_eval \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --test_path      "${TEST_PATH}" \
    --spoof \
    --output_dir     "./results" \
    --debug_generations \
    --debug_n 100
