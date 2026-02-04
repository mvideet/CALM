#!/bin/bash
#SBATCH -J return_embeddings
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/return_embeddings%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/return_embeddings%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # 1 GPU per array task
#SBATCH --requeue
#SBATCH --partition=a6,a5
# #SBATCH --mem=22G
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-3         
#SBATCH --chdir=/data/sls/u/urop/mvideet/sparse_audio/SAVs


# (Optional) Activate your conda/env here
# source /path/to/conda.sh
# conda activate test

# Configuration
DATASET_PATH="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound_mcq_train_20shot.json"
OUT_PREFIX="embeddings"
MODEL_NAME="qwen2.5_omni"
DATA_NAME="vgg_sound_qa"
N_TRIALS=20
SHOT=1
SPLITS=4

# Derive this task's split
SPLIT_ID=$SLURM_ARRAY_TASK_ID

# Count total samples once (you could cache this in a small file too)
# ---- count total rows ----------------------------------------------------
TOTAL=$(python3 - <<EOF
import json
with open("$DATASET_PATH") as f:
    data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        print(len(data['data']))
    else:
        print(len(data))
EOF
)
# Compute chunk size
CHUNK=$(( (TOTAL + SPLITS - 1) / SPLITS ))
START=$(( SPLIT_ID * CHUNK ))
END=$(( START + CHUNK ))
[ "$END" -gt "$TOTAL" ] && END=$TOTAL

echo "[$(date)] Task $SPLIT_ID running indices [$START,$END) on GPU ${CUDA_VISIBLE_DEVICES:-0}"
echo "Task \$SLURM_ARRAY_TASK_ID => CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"

python -m src.compute_vgg_embeddings \
  --dataset_path "$DATASET_PATH" \
  --start "$START" --end "$END" \
  --split_id "$SPLIT_ID" \
  --out_prefix "$OUT_PREFIX" \
  --model_name "$MODEL_NAME" \
  --data_name "$DATA_NAME" \
  --n_trials "$N_TRIALS" \
  --shot "$SHOT"
