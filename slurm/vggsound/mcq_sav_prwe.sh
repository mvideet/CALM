#!/bin/bash
#SBATCH -J hyperparam_search_vggsound_mcq
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/hyperparam_search_vggsound_mcq_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/hyperparam_search_vggsound_mcq_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# MODEL_NAME="phi4-multimodal"

MODEL_NAME="qwen2-audio-instruct"
DATA_NAME="vgg_sound_qa"
# DATA_NAME="esc_mcq"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_val.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_val.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test_split.json"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"


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
    --top_k 5 10 20 40 100 300 500 768 \
    --last_n_tokens 1 \
    --n_trials 1

python -u -m src.run --task sav \
    --model_name "${MODEL_NAME}" \
    --data_name "${DATA_NAME}" \
    --train_path "${TRAIN_PATH}" \
    --val_path "${VAL_PATH}" \
    --test_path "${TEST_PATH}" \
    --sav_num_heads 20 \
    --cache_dir "./cache/esc_mcq_prwe"
# 