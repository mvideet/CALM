#!/bin/bash
#SBATCH -J grw_sav_vggsound_qwen2.5
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/grw_sav_vggsound_qwen2.5%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/grw_sav_vggsound_qwen2.5%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
# SBATCH --exclude sls-a6-5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=test
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate test

MODEL_NAME="qwen2-audio-instruct"
# MODEL_NAME="qwen2.5_omni"
# DATA_NAME="vgg_sound_qa"
DATA_NAME="esc_mcq"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/pseudolabeled_vggsound_mcq_qwen2.5.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
# If you don't have a dedicated val file, reuse SUPPORT for reliabilities (no test leakage)
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_20shot.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_test_individual_mcqs.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# # VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# TEST_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"


# Optional: provide a separate support set; otherwise uses train data for prototypes
# SUPPORT_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_support.json"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# Run GRW-SAV with clean splits (support=train by default; here we pass an explicit SUPPORT)
python -u -m src.probabilistic.run_grw_sav \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${TRAIN_PATH}" \
    --test_path    "${TEST_PATH}" \
    --tau 0.001 0.01 0.03 0.5 1 \
    --tau_w 0.5 1.0 7.0 \
    --top_k 5 10 20 40 100 300 500 1024 \
    --last_n_tokens 1 \
    --n_trials 1

#784 heads in qwen2.5_omni and 1024 in qwen2-audio-instruct

