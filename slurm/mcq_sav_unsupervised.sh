#!/bin/bash
#SBATCH -J mcq_sav_unsupervised
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/mcq_sav_unsupervised%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/mcq_sav_unsupervised%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6,a5
##SBATCH --partition=a5,a6,2080
#SBATCH --exclude sls-a6-5
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
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"

# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/pseudolabeled_vggsound_mcq_qwen2.5_8trials.json"
VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
EVAL_ZEROSHOT=False

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# 2. Run your module with “-m src.run_vggsound” instead of calling the .py directly
python -u -m src.run_mcq_ans_unsupervised \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    --eval_zeroshot "${EVAL_ZEROSHOT}"

