#!/bin/bash
#SBATCH -J zero_shot_phi4_vgg
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/zero_shot_phi4_vgg_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/zero_shot_phi4_vgg_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# Zero-shot baseline: Phi-4 multimodal on VGGSound MCQ
# Runs model inference on TEST set only (no train, no CALM cache, no SAV).

DATA_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound"
TEST_PATH="${DATA_DIR}/vggsound_mcq_test.json"
DATA_NAME="vgg_sound_qa"
MODEL_NAME="phi4-multimodal"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

echo "=============================================="
echo "Zero-shot baseline: Phi-4 multimodal on VGGSound"
echo "=============================================="

python -u -m src.run --task classify \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --test_path    "${TEST_PATH}" \
    --zero_shot_only
