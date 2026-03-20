# CALM: Class-conditional Attention vectors for Language Models

A training-free method for few-shot classification using reliability-weighted attention head activations from multimodal language models.

See [`SAVs/README.md`](SAVs/README.md) for full documentation, usage instructions, and API reference.

## Quick Start

```bash
cd SAVs
pip install -e .

# Zero-shot inference
python -m src.run --task classify --zero_shot_only \
    --model_name qwen2.5_omni --data_name eurosat \
    --test_path data/eurosat/eurosat_test_6k.json

# CALM few-shot classification
python -m src.run --task classify \
    --model_name qwen2-audio-instruct --data_name vgg_sound_qa \
    --train_path data/vggsound/vggsound_mcq_train_40shot.json \
    --val_path data/vggsound/vggsound_mcq_val_40shot.json \
    --test_path data/vggsound/vggsound_mcq_test.json \
    --tau 0.01 0.03 0.07 0.1 0.3 \
    --tau_w 0.1 0.3 0.5 1.0 2.0
```

## Repository Structure

```
SAVs/                       # Main package (source, scripts, data)
slurm/                      # SLURM experiment scripts
├── vggsound/               #   VGGSound experiments
├── esc50/                  #   ESC-50 experiments
├── audioset/               #   AudioSet experiments
├── la_spoof/               #   LA Spoof experiments
├── eurosat/                #   EuroSAT experiments
├── pets/                   #   Oxford Pets experiments
├── baselines/              #   CLAP KNN, linear probing
├── analysis/               #   Pseudolabel analysis
└── legacy/                 #   Deprecated scripts
```
