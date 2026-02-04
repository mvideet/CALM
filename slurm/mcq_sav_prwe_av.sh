#!/bin/bash
#SBATCH -J av_prwe_vggsound
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/av_prwe_vggsound%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/av_prwe_vggsound%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6,a5
# SBATCH --exclude sls-a6-5
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
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
# If you don't have a dedicated val file, reuse SUPPORT for reliabilities (no test leakage)
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vgg_video_train.json"
VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vgg_video_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_test_individual_mcqs.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"


# Optional: provide a separate support set; otherwise uses train data for prototypes
# SUPPORT_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_support.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# Set PyTorch CUDA memory allocation config to reduce fragmentation
# This helps prevent OOM errors and hanging when memory is fragmented
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use OpenCV backend for video reading (more reliable than decord, avoids hangs)
# VIDEO_READER_BACKEND=opencv will use OpenCV to load videos (bypasses qwen_omni_utils)
# VIDEO_READER_BACKEND=decord will use decord via qwen_omni_utils (default)
export VIDEO_READER_BACKEND=opencv
# FORCE_QWENVL_VIDEO_READER is only used if VIDEO_READER_BACKEND is not "opencv"
export FORCE_QWENVL_VIDEO_READER=decord

# Run PRWE with clean splits (support=train by default; here we pass an explicit SUPPORT)
# Using run_mcq_prwe_av_vggsound.py which has audio_or_video="video" configured
python -u -m src.probabilistic.run_mcq_prwe_av_vggsound \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${TRAIN_PATH}" \
    --test_path    "${VAL_PATH}" \
    --weight_scheme margin_clamped brier_softmax\
    --tau 0.001 0.03 1.0 \
    --alpha 0.0\
    --tau_w 0.5 1.0 \
    --top_k 5 10 20 40 100 300 500 784 \
    --last_n_tokens 1 \
    --n_trials 1

#784 heads in qwen2.5_omni and 1024 in qwen2-audio-instruct