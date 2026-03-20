#!/bin/bash
#SBATCH -J sft_eval_vggsound
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/sft_eval_vggsound_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/sft_eval_vggsound_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

CHECKPOINT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/checkpoints/sft_vggsound_lora"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"
cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.sft_eval \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --test_path      "${TEST_PATH}" \
    --output_dir     "./results"
