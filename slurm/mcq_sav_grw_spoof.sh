#!/bin/bash
#SBATCH -J GRW_SPOOF
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/GRW_SPOOF%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/GRW_SPOOF%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6,a5
#SBATCH --exclude sls-a6-5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

MODEL_NAME="qwen2-audio-instruct"
# MODEL_NAME="qwen2.5_omni"
DATA_NAME="LA_spoof"

# LA_AvSpoof dataset paths
# Option 1: Use pseudolabeled training data
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/pseudolabeled_LA_spoof_qwen2.5_omni_train_8trials.json"
# Option 2: Use original training data (uncomment to use)
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/LA_train.json"

VAL_PATH="${TRAIN_PATH}"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/LA_eval.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# Run GRW-SAV for spoofing detection
python -u -m src.probabilistic.run_grw_spoof \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --test_path    "${TEST_PATH}" \
    --tau 0.001 0.01 0.03 0.05 0.1 1.0 \
    --tau_w 0.5 1.0 2.0 \
    --top_k 5 10 20 40 100 300 500 1024 \
    --last_n_tokens 1 \
    --n_trials 1

# Notes:
# - GRW uses global reliability (one weight per head) instead of per-class reliability
# - Simpler than PRWE but often works well for binary classification like spoofing
# - 784 heads in qwen2.5_omni, 1024 in qwen2-audio-instruct

