#!/bin/bash
#SBATCH -J rand_topk_vgg
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/rand_topk_vgg_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/rand_topk_vgg_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1
DATA_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound"
TRAIN_PATH="${DATA_DIR}/vggsound_mcq_train_20shot.json"
TEST_PATH="${DATA_DIR}/vggsound_mcq_test.json"
DATA_NAME="vgg_sound_qa"
# MODEL_NAME="qwen2-audio-instruct"
MODEL_NAME="qwen2.5_omni"

RANDOM_HEAD_SEED=0

echo "=============================================="
echo "Model: ${MODEL_NAME}  top_k: 5 10 20 40 100 300 500 768  random_head_seed: ${RANDOM_HEAD_SEED}"
echo "Train: ${TRAIN_PATH}"
echo "BASELINE: Random head selection (not reliability-based)"
echo "=============================================="

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task classify \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --test_path    "${TEST_PATH}" \
    --weight_scheme margin_clamped \
    --tau 0.03 \
    --tau_w 0.5 \
    --top_k 5 10 20 40 100 300 500 768 \
    --random_topk \
    --random_head_seed "${RANDOM_HEAD_SEED}" \
    --last_n_tokens 1 \
    --n_trials 1 \
    --cache_dir "./cache/20shot"

echo ""
echo "=============================================="
echo "SAV Baseline"
echo "=============================================="

for HEADS in 5 10 20 40 100 300 500 768; do
    python -u -m src.run --task sav \
        --model_name   "${MODEL_NAME}" \
        --data_name    "${DATA_NAME}" \
        --train_path   "${TRAIN_PATH}" \
        --val_path     "${VAL_PATH}" \
        --test_path    "${TEST_PATH}" \
        --sav_num_heads "${HEADS}" \
        --cache_dir "./cache/20shot"
done
