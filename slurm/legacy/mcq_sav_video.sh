#!/bin/bash
#SBATCH -J vggsound_sav_video
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/vggsound_sav_video%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/vggsound_sav_video%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=22G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=test
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate test

# MODEL_NAME="qwen2-audio-instruct"
MODEL_NAME="qwen2.5_omni"
DATA_NAME="vgg_sound_qa"
# DATA_NAME="esc_mcq"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/pseudolabeled_vggsound_mcq_qwen2.5.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVzs/data/vggsound/vggsound_mcq_train_40shot.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vgg_video_train.json"
VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vgg_video_test.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_test_individual_mcqs.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"
# WORST_HEAD_MULTIPLIER=1


cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# Set PyTorch CUDA memory allocation config to reduce fragmentation
# This helps prevent OOM errors and hanging when memory is fragmented
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use OpenCV backend for video reading (more reliable than decord, avoids hangs)
# VIDEO_READER_BACKEND=opencv will use OpenCV to load videos (bypasses qwen_omni_utils)
# VIDEO_READER_BACKEND=decord will use decord via qwen_omni_utils (default)
export VIDEO_READER_BACKEND=decord
# FORCE_QWENVL_VIDEO_READER is only used if VIDEO_READER_BACKEND is not "opencv"
export FORCE_QWENVL_VIDEO_READER=decord

# 2. Run your module with "-m src.run_vggsound" instead of calling the .py directly
python -u -m src.run_mcq_ans \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --n_trials     1 \
    --eval_zeroshot \
