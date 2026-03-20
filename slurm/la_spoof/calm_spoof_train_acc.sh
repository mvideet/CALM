#!/bin/bash
#SBATCH -J calm_spoof_train
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/calm_spoof_train_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/calm_spoof_train_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# CALM on LA Spoof: evaluate on the TRAIN set itself
# to check how well the few-shot centroids classify the support examples.

MODEL_NAME="qwen2-audio-instruct"
DATA_NAME="LA_spoof"
DATA_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof"
TRAIN_PATH="${DATA_DIR}/LA_train_50pct.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task spoof \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${TRAIN_PATH}" \
    --test_path    "${TRAIN_PATH}" \
    --weight_scheme margin_clamped \
    --tau 0.001 0.01 0.03 0.05 0.07 0.1 \
    --tau_w 0.1 0.3 0.5 1.0 2.0 \
    --top_k 5 10 20 40 100 300 500 1024 \
    --last_n_tokens 1 \
    --n_trials 1
