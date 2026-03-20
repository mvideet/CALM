# CALM: Class-conditional Attention vectors for Language Models

A training-free method for few-shot classification using reliability-weighted attention head activations from multimodal language models. Works across **audio** and **image** modalities without any fine-tuning.

## Method

1. **Build class prototypes** — extract per-head attention activations from few-shot support samples and average them per class.
2. **Estimate reliability** — use a held-out validation set to score each attention head's classification accuracy per class.
3. **Weighted voting** — at test time, aggregate per-head posteriors weighted by reliability.

The baseline **SAV** (Sparse Attention Vectors) replaces reliability weighting with simple top-k majority voting.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+, and a CUDA GPU.

## Supported Models

| Model | `--model_name` | Modality |
|-------|----------------|----------|
| Qwen2-Audio-7B-Instruct | `qwen2-audio-instruct` | Audio |
| Qwen2.5-Omni-7B | `qwen2.5_omni` | Audio + Image |
| Qwen2-VL-7B-Instruct | `qwen2-vl-instruct` | Image |
| Phi-4-multimodal-instruct | `phi4-multimodal` | Audio + Image |

## Supported Datasets

| Dataset | `--data_name` | Modality | Classes |
|---------|---------------|----------|---------|
| VGGSound | `vgg_sound_qa` | Audio | 309 |
| ESC-50 | `esc_mcq` | Audio | 50 |
| AudioSet | `audioset` | Audio | 527 |
| ASVspoof 2019 LA | `LA_spoof` | Audio | 2 |
| MLAAD | `mlaad` | Audio | 2 |
| EuroSAT | `eurosat` | Image | 10 |
| Oxford-IIIT Pets | `pets` | Image | 37 |

## Usage

All experiments are launched via the unified CLI:

```bash
python -m src.run --task <TASK> --model_name <MODEL> --data_name <DATASET> [OPTIONS]
```

### Tasks

#### Zero-shot inference

```bash
python -m src.run --task classify --zero_shot_only \
    --model_name qwen2.5_omni \
    --data_name eurosat \
    --test_path data/eurosat/eurosat_test_6k.json
```

#### CALM (few-shot classification with hyperparameter sweep)

```bash
python -m src.run --task classify \
    --model_name qwen2-audio-instruct \
    --data_name vgg_sound_qa \
    --train_path data/vggsound/vggsound_mcq_train_40shot.json \
    --val_path data/vggsound/vggsound_mcq_val_40shot.json \
    --test_path data/vggsound/vggsound_mcq_test.json \
    --tau 0.01 0.03 0.07 0.1 0.3 \
    --tau_w 0.1 0.3 0.5 1.0 2.0 \
    --top_k 5 10 20 40 100 300 500 784 \
    --n_trials 1 \
    --cache_dir ./cache/vggsound_calm
```

#### SAV baseline

```bash
python -m src.run --task sav \
    --model_name qwen2.5_omni \
    --data_name eurosat \
    --train_path data/eurosat/eurosat_train.json \
    --test_path data/eurosat/eurosat_test_6k.json \
    --sav_num_heads 5 10 20 50 100
```

#### Spoofing detection

```bash
python -m src.run --task spoof \
    --model_name qwen2-audio-instruct \
    --data_name LA_spoof \
    --train_path data/LA_AvSpoof/LA_train.json \
    --val_path data/LA_AvSpoof/LA_eval.json
```

#### Pseudolabel generation

```bash
python -m src.run --task pseudolabel \
    --model_name qwen2.5_omni \
    --data_name eurosat \
    --train_path data/eurosat/eurosat_train.json \
    --n_trials 8 \
    --min_confidence 0.5 \
    --output_dir ./pseudolabels
```

### Unsupervised HP selection

Select hyperparameters using only the model's own zero-shot predictions (no ground-truth labels required):

```bash
python -m src.run --task classify --unsupervised \
    --model_name qwen2-audio-instruct \
    --data_name vgg_sound_qa \
    --train_path data/vggsound/vggsound_mcq_train_40shot.json \
    --val_path data/vggsound/vggsound_mcq_val_40shot.json \
    --test_path data/vggsound/vggsound_mcq_test.json \
    --tau 0.01 0.03 0.07 0.1 0.3 \
    --tau_w 0.1 0.3 0.5 1.0 2.0
```

## Python API

```python
from src import (
    load_model, open_data,
    calm_prepare_cache, calm_compute_posteriors_from_cache,
    calm_compute_reliability, calm_build_weights_from_r,
    calm_eval_from_posteriors,
)

model = load_model("qwen2.5_omni", "eurosat")
train = open_data("eurosat", "data/eurosat/eurosat_train.json")
val   = open_data("eurosat", "data/eurosat/eurosat_val.json")
test  = open_data("eurosat", "data/eurosat/eurosat_test_6k.json")

cache = calm_prepare_cache(model, train, val, test, n_trials=1)

P_val  = calm_compute_posteriors_from_cache(cache, tau=0.07, split="val")
P_test = calm_compute_posteriors_from_cache(cache, tau=0.07, split="test")

r, _ = calm_compute_reliability(P_val, cache["val_labels_idx"], "margin_clamped")
w = calm_build_weights_from_r(r, weight_scheme="margin_clamped", tau_w=1.0)

accuracy = calm_eval_from_posteriors(P_test, w, cache["test_labels_idx"])
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tau` | 0.07 | Temperature for class posteriors (lower = sharper) |
| `--tau_w` | 1.0 | Temperature for head weighting |
| `--weight_scheme` | `margin_clamped` | Reliability estimation: `margin_clamped`, `margin_softmax`, `prob_softmax`, `brier_softmax` |
| `--n_trials` | 20 | Trials for activation averaging |
| `--top_k` | all | Top-k head selection per class |
| `--last_n_tokens` | 1 | Tokens to average from sequence end |
| `--sav_num_heads` | 20 | Heads for SAV majority voting |

## Data Format

Input JSON files follow this structure:

```json
[
  {
    "wav": "/path/to/audio.wav",
    "question": "What sound is this?\nA. cat\nB. dog\nC. bird\nD. car",
    "answer": "B",
    "mapped_label": "dog",
    "options": ["cat", "dog", "bird", "car"]
  }
]
```

For image datasets, replace `wav` with `image`:

```json
[
  {
    "image": "/path/to/image.jpg",
    "question": "What type of land use is shown?\nA. Forest\nB. Highway\nC. Industrial\nD. Residential",
    "answer": "A",
    "mapped_label": "forest",
    "options": ["Forest", "Highway", "Industrial", "Residential"]
  }
]
```

## Baselines

### CLAP KNN

K-nearest-neighbor classification using CLAP audio/text embeddings:

```bash
python -m scripts.clap_knn --datasets esc_mcq LA_spoof vgg_sound_qa audioset
```

### Linear Probing

Train a linear classifier on frozen model hidden states:

```bash
python -m scripts.linear_probe --model qwen2-audio-instruct --dataset esc_mcq
```

## Dataset Preparation

```bash
# EuroSAT (image, 10 classes)
python -m scripts.dataset_processing.prepare_eurosat \
    --data_dir data/EuroSAT_RGB --output_dir data/eurosat --train_shots 40

# Oxford Pets (image, 37 classes)
python -m scripts.dataset_processing.prepare_pets

# ASVspoof 2019 LA (audio, 2 classes)
python -m scripts.dataset_processing.prepare_asvspoof \
    --protocol PROTOCOL --audio_dir AUDIO --output out.json

# ESC-50 (audio, 50 classes)
python -m scripts.dataset_processing.prepare_esc50 \
    --train_files train.json --eval_file eval.json --audio_dir AUDIO --output_dir OUT

# AudioSet pruning (audio)
python -m scripts.dataset_processing.prune_audioset \
    --input input.json --output output.json --shots 20
```

## SLURM Experiments

Pre-configured experiment scripts in `../slurm/`, organized by dataset:

```
slurm/
├── vggsound/              # CALM, SAV, unsupervised, zero-shot (Phi-4)
├── esc50/                 # CALM, pseudolabel threshold ablation
├── audioset/              # CALM
├── la_spoof/              # CALM, spoof detection
├── eurosat/               # ZS, SAV, CALM, pseudolabels (Omni, Phi-4, Qwen2-VL)
├── pets/                  # ZS, SAV, CALM, pseudolabels (Omni)
├── baselines/             # CLAP KNN, linear probing
├── analysis/              # Pseudolabel analysis
└── legacy/                # Older/deprecated scripts
```

## Project Structure

```
SAVs/
├── src/
│   ├── __init__.py          # Package exports
│   ├── calm.py              # Core CALM algorithm
│   ├── model.py             # Model helpers (4 models)
│   ├── preprocess.py        # Data loading and formatting
│   ├── pseudolabel.py       # Pseudolabel generation via majority voting
│   ├── sav.py               # SAV baseline (top-k heads + majority vote)
│   └── run.py               # Unified CLI entry point
├── scripts/
│   ├── clap_knn.py          # CLAP KNN baseline
│   ├── linear_probe.py      # Linear probing baseline
│   ├── analyze_pseudolabels.py
│   ├── filter_pseudolabels.py
│   └── dataset_processing/  # Dataset preparation scripts
├── data/                    # Dataset JSON files and raw data
├── pyproject.toml
├── LICENSE
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE) for details.
