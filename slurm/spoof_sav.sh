#!/bin/bash
#SBATCH -J mcq_spoof_sav
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/mcq_spoof_sav%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/mcq_spoof_sav%A_%a.err
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
DATA_NAME="LA_spoof"
# DATA_NAME="esc_mcq"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/pseudolabeled_vggsound_mcq_qwen2.5.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_mcq_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_train.json"
# VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/esc/esc_mcq_test.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/LA_train.json"
VAL_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/LA_eval.json"
# TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/pseudolabeled_LA_spoof_qwen2-audio-instruct_train_8trials.json"
TRAIN_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/LA_AvSpoof/pseudolabeled_LA_spoof_qwen2.5_omni_train_8trials.json"
EVAL_ZEROSHOT=False

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

ZEROSHOT_FLAG=""
if [ "${EVAL_ZEROSHOT}" = "True" ] || [ "${EVAL_ZEROSHOT}" = "true" ]; then
    ZEROSHOT_FLAG="--eval_zeroshot"
fi

python -u -m src.run_mcq_spoof \
    --model_name   "${MODEL_NAME}" \
    --data_name    "${DATA_NAME}" \
    --train_path   "${TRAIN_PATH}" \
    --val_path     "${VAL_PATH}" \
    ${ZEROSHOT_FLAG}

