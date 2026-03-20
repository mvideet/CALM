#!/bin/bash
#SBATCH -J analyze_pl
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/analyze_pl_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/analyze_pl_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1

# GT vs pseudo-label agreement analysis across confidence thresholds.
# CPU only — no GPU needed.

DATA_DIR="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m scripts.analyze_pseudolabels \
    -i "${DATA_DIR}/vggsound/pseudolabeled_vggsound_mcq_qwen2_8trials.json" \
       "${DATA_DIR}/esc/pseudolabeled_esc_40shot_mcq_qwen2_8trials.json" \
       "${DATA_DIR}/audioset/pseudolabeled_audioset_qwen2-audio-instruct_train_8trials.json" \
       "${DATA_DIR}/LA_AvSpoof/pseudolabeled_LA_spoof_qwen2-audio-instruct_train_8trials.json"
