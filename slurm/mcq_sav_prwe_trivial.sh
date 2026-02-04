#!/bin/bash
#SBATCH -J prwe_trivial_vggsound
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/prwe_trivial_vggsound%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/prwe_trivial_vggsound%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
# SBATCH --exclude sls-a6-5
#SBATCH --mem=22G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=test
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate test

# MODEL_NAME="qwen2-audio-instruct"
MODEL_NAME="qwen2.5_omni"
DATA_NAME="vgg_sound_qa"
# DATA_NAME="esc_mcq"s
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/pseudolabeled_vggsound_mcq_qwen2.5.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
# # TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
# # VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"

# # TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_test_individual_mcqs.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"

# Optional: provide a separate support set; otherwise uses train data for prototypes
# SUPPORT_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_support.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# Run Simple Soft Voting methods (trivial PRWE alternatives)
python -u -m src.probabilistic.run_mcq_prwe_trivial \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --test_path    "${TEST_PATH}" \
    --methods concat topk_soft \
    --tau 0.001 0.03 \
    --top_k 5 10 20 40 100 300 500 784\
    --last_n_tokens 1 \
    --n_trials 20

# Methods:
#   - concat: Single cosine sim on concatenated head vectors
#   - uniform_soft: Average posteriors across all heads
#   - confidence_weighted: Weight heads by prediction confidence
#   - topk_soft: Uniform soft vote among top-K validated heads
#   - hard_vote: Traditional majority voting baseline
#
# Note: 784 heads in qwen2.5_omni and 1024 in qwen2-audio-instruct

