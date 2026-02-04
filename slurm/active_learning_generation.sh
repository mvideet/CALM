#!/bin/bash
#SBATCH -J active_learning_gen
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/active_learning_gen%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/active_learning_gen%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6,a5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# MODEL_NAME="qwen2.5_omni"
MODEL_NAME="qwen2-audio-instruct"
DATA_NAME="vgg_sound_qa"
# DATA_NAME="esc_mcq"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train_40shot.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/pseudolabeled_vggsound_mcq_qwen2.5.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"

# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_test_individual_mcqs.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"
N_TRIALS=8
BOTTOM_PERCENT=0.15

if [[ "${DATA_NAME}" == "vgg_sound_qa" ]]; then
    OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound"
elif [[ "${DATA_NAME}" == "esc_mcq" ]]; then
    OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc"
elif [[ "${DATA_NAME}" == "audioset" ]]; then
    OUTPUT_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset"
else
    OUTPUT_DIR=""
fi

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m src.active_learning \
    --model_name "${MODEL_NAME}" \
    --data_name "${DATA_NAME}" \
    --train_path "${TRAIN_PATH}" \
    --n_trials "${N_TRIALS}" \
    --bottom_percent "${BOTTOM_PERCENT}" \
    --output_dir "${OUTPUT_DIR}"
