#!/bin/bash
#SBATCH -J generateVggsound
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/generateVggsound%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/generateVggsound%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=2080
#SBATCH --mem=88G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=test
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate test

cd /data/sls/u/urop/mvideet/sparse_audio/SAVs/

seed_list=(123 456 101 201 301 401 501 601 701 801)

for seed in "${seed_list[@]}"; do
    python -u -m src.convert_dataset_vggsound --input /data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/vgg_train_cleaned.json \
    --label_csv /data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/class_labels_indices_vgg.csv \
    --output /data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vggsound_train_cleaned_${seed}.json \
    --samples_per_class 20 --seed ${seed}
done