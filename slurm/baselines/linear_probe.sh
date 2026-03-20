#!/bin/bash
#SBATCH -J linear_probe
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/linear_probe_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/linear_probe_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# Linear probing on Qwen2-Audio and Qwen2.5-Omni last hidden states.
# Runs both models on all 4 audio datasets.

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate test

DATA="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data"
cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

for MODEL in qwen2-audio-instruct qwen2.5_omni; do
    echo ""
    echo "========== MODEL: $MODEL =========="
    echo ""

    # ESC-50
    python -u -m scripts.linear_probe \
        --model "$MODEL" \
        --dataset "ESC-50" \
        --train_path "${DATA}/esc/esc50_mcq_40shot_train.json" \
        --test_path "${DATA}/esc/esc_mcq_test_split.json" \
        --n_trials 1

    # VGGSound
    python -u -m scripts.linear_probe \
        --model "$MODEL" \
        --dataset "VGGSound" \
        --train_path "${DATA}/vggsound/vggsound_mcq_train_5shot.json" \
        --test_path "${DATA}/vggsound/vggsound_mcq_test_split.json" \
        --n_trials 1

    # AudioSet (large - may take a while)
    python -u -m scripts.linear_probe \
        --model "$MODEL" \
        --dataset "AudioSet" \
        --train_path "${DATA}/audioset/audioset_20shot_train_individual_mcqs.json" \
        --test_path "${DATA}/audioset/audioset_test_individual_mcqs.json" \
        --n_trials 1

    # LA Spoof
    python -u -m scripts.linear_probe \
        --model "$MODEL" \
        --dataset "LA_Spoof" \
        --train_path "${DATA}/LA_AvSpoof/LA_train.json" \
        --test_path "${DATA}/LA_AvSpoof/LA_eval.json" \
        --n_trials 1
done

echo ""
echo "========== Linear probe complete =========="
