#!/bin/bash
#SBATCH -J pseudolabel_gen_spoofed
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/pseudolabel_gen_spoofed%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/pseudolabel_gen_spoofed%A_%a.err
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

# MODEL_NAME="qwen2-audio-instruct"
MODEL_NAME="qwen2.5_omni"
# DATA_NAME="vgg_sound_qa"
# DATA_NAME="esc_mcq"
DATA_NAME="audioset"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_train_40shot.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
N_TRIALS=8

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# 2. Run your module with “-m src.run_vggsound” instead of calling the .py directly
python -u -m src.generate_psuedolabel \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --n_trials     "${N_TRIALS}"
