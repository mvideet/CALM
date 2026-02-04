#!/bin/bash
#SBATCH -J test_video_loading
#SBATCH -o /data/sls/u/urop/mvideet/sparse_audio/slurm/out/test_video_loading%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/sparse_audio/slurm/err/test_video_loading%A_%a.err
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


cd /data/sls/u/urop/mvideet/sparse_audio/SAVs

# 2. Run your module with “-m src.run_vggsound” instead of calling the .py directly

python -m src.probabilistic.run_test_query_cache_only \
  --model_name qwen2.5_omni \
  --data_name vgg_sound_qa \
  --test_path /data/sls/u/urop/mvideet/sparse_audio/SAVs/data/vggsound/vgg_video_test.json \
  --audio_or_video video \
  --n_trials 1 \
  --last_n_tokens 1