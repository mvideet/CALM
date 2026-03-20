#!/bin/bash
#SBATCH -J pl_thresh
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/pl_thresh_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/pl_thresh_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-7%4

# 2 datasets x 4 thresholds = 8 jobs
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
THRESH_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

MODEL_NAME="qwen2-audio-instruct"
DATA_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data"
THRESH_TAGS=("conf0500" "conf0625" "conf0875" "conf1000")
THRESH_TAG="${THRESH_TAGS[$THRESH_IDX]}"

if [ "$DATASET_IDX" -eq 0 ]; then
    DATA_NAME="vgg_sound_qa"
    TRAIN_PATH="${DATA_DIR}/vggsound/pseudolabeled_vggsound_mcq_qwen2_8trials_${THRESH_TAG}.json"
    TEST_PATH="${DATA_DIR}/vggsound/vggsound_mcq_test.json"
    CACHE_TAG="vgg_pl_${THRESH_TAG}"
else
    DATA_NAME="esc_mcq"
    TRAIN_PATH="${DATA_DIR}/esc/pseudolabeled_esc_40shot_mcq_qwen2_8trials_${THRESH_TAG}.json"
    TEST_PATH="${DATA_DIR}/esc/esc_mcq_test.json"
    CACHE_TAG="esc_pl_${THRESH_TAG}"
fi

echo "=============================================="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATA_NAME}  Threshold: ${THRESH_TAG}"
echo "Train: ${TRAIN_PATH}"
echo "Test:  ${TEST_PATH}"
echo "=============================================="

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.run --task classify \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${TRAIN_PATH}" \
    --test_path    "${TEST_PATH}" \
    --weight_scheme margin_clamped \
    --tau 0.03 \
    --tau_w 0.5 \
    --top_k 20 \
    --last_n_tokens 1 \
    --n_trials 1 \
    --cache_dir "./cache/${CACHE_TAG}"
