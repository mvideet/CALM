#!/bin/bash
#SBATCH -J prwe_vgg_seeds
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/prwe_vgg_seeds%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/prwe_vgg_seeds%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=176G
#SBATCH --ntasks-per-node=1

DATA_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound"
VAL_PATH="${DATA_DIR}/vggsound_mcq_val.json"
TEST_PATH="${DATA_DIR}/vggsound_mcq_test_split.json"
DATA_NAME="vgg_sound_qa"
MODEL_NAME="qwen2-audio-instruct"

SEED_LIST=(123 456 101 201 301 401 501 601 701 801)
NUM_SEEDS=${#SEED_LIST[@]}
NUM_GPUS=4

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

LOG_DIR="/data/sls/u/urop/mvideet/sparse_audio/slurm/out/prwe_vgg_seeds"
mkdir -p "${LOG_DIR}"

run_seed() {
    local gpu=$1
    local seed=$2
    local train="${DATA_DIR}/vggsound_train_cleaned_${seed}.json"
    local log="${LOG_DIR}/seed_${seed}.log"

    echo "[GPU ${gpu}] Starting seed=${seed} -> ${log}"

    CUDA_VISIBLE_DEVICES=$gpu python -u -m src.run --task sav \
        --model_name   "${MODEL_NAME}" \
        --data_name    "${DATA_NAME}" \
        --train_path   "${train}" \
        --val_path     "${VAL_PATH}" \
        --test_path    "${TEST_PATH}" \
        --sav_num_heads 20 40 100 300 \
        --cache_dir "./cache/seed_${seed}" >> "${log}" 2>&1

    echo "[GPU ${gpu}] Finished seed=${seed}"
}

for (( start=0; start<NUM_SEEDS; start+=NUM_GPUS )); do
    pids=()
    for (( g=0; g<NUM_GPUS; g++ )); do
        idx=$(( start + g ))
        if (( idx >= NUM_SEEDS )); then
            break
        fi
        run_seed $g "${SEED_LIST[$idx]}" &
        pids+=($!)
    done
    wait "${pids[@]}"
    echo "--- Batch starting at index ${start} complete ---"
done

echo "All seeds done. Per-seed logs: ${LOG_DIR}/seed_*.log"
