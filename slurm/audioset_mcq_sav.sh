#!/bin/bash
#SBATCH -J audioset_mcq_sav
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/audioset_mcq_sav%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/audioset_mcq_sav%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6,a5
##SBATCH --partition=a5,a6,2080
#SBATCH --exclude sls-a6-5
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=test
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate test

MODEL_NAME="qwen2.5_omni"
# MODEL_NAME="qwen2.5_omni"
# DATA_NAME="vgg_sound_qa"
DATA_NAME="audioset"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/pseudolabeled_vggsound_mcq_qwen2.5.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/pseudolabeled_audioset_qwen2.5_omni_train_8trials.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/pseudolabeled_audioset_qwen2-audio-instruct_train_8trials.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/pseudolabeled_audioset_qwen2-audio-instruct_train_8trials.json"

TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/pseudolabeled_audioset_qwen2.5_omni_train_8trials.json"

VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_test_individual_mcqs.json"
cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# 2. Run your module with “-m src.run_vggsound” instead of calling the .py directly
python -u -m src.run_mcq_audioset \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}"
