#!/bin/bash
#SBATCH -J unsup_vgg
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/unsup_vgg_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/unsup_vgg_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

DATA_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound"
MODEL_NAME="qwen2-audio-instruct"
DATA_NAME="vgg_sound_qa"
TRAIN_PATH="${DATA_DIR}/vggsound_mcq_train_20shot.json"
TEST_PATH="${DATA_DIR}/vggsound_mcq_test.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

echo "=============================================="
echo "Unsupervised HP selection (zero-shot pseudolabels on test set)"
echo "val = train, pseudolabels on test for HP selection"
echo "=============================================="

python -u -m src.run --task classify \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${TRAIN_PATH}" \
    --test_path    "${TEST_PATH}" \
    --weight_scheme margin_clamped \
    --tau 0.001 0.01 0.03 0.05 0.07 0.1 \
    --tau_w 0.1 0.3 0.5 1.0 2.0 \
    --top_k 5 10 20 40 100 300 500 768 \
    --last_n_tokens 1 \
    --n_trials 1 \
    --unsupervised \
    --cache_dir "./cache/unsupervised"
