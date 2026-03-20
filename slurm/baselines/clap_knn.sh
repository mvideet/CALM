#!/bin/bash
#SBATCH -J clap_knn
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/clap_knn_%j.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/clap_knn_%j.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a5
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1

# CLAP KNN classification baseline across 4 audio datasets.
# Embeds train audio → per-class centroids, then classifies test audio
# by nearest cosine centroid.  Also runs text-centroid zero-shot.


source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate test
DATA="/data/sls/u/urop/mvideet/sparse_audio/SAVs/data"

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

python -u -m scripts.clap_knn \
    --batch_size 32 \
    --datasets \
        "ESC-50:${DATA}/esc/esc50_mcq_40shot_train.json:${DATA}/esc/esc_mcq_test_split.json" \
        "LA_Spoof:${DATA}/LA_AvSpoof/LA_train.json:${DATA}/LA_AvSpoof/LA_eval.json" \
        "VGGSound:${DATA}/vggsound/vggsound_mcq_train_20shot.json:${DATA}/vggsound/vggsound_mcq_test_split.json" \
        "AudioSet:${DATA}/audioset/audioset_20shot_train_individual_mcqs.json:${DATA}/audioset/audioset_test_individual_mcqs.json"
