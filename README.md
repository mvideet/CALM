# CALM: Class-conditional Attention vectors for audio Language Models

A training-free method for few-shot audio classification using reliability-weighted attention head activations from audio language models.


## Installation

```bash
pip install -e .
```

## Quick Start

### Command Line

```bash
# Audio classification
python -m src.run --task classify \
    --model_name qwen2-audio-instruct \
    --data_name vgg_sound_qa \
    --train_path data/train.json \
    --val_path data/val.json \
    --test_path data/test.json

# Spoofing detection
python -m src.run --task spoof \
    --model_name qwen2-audio-instruct \
    --data_name LA_spoof \
    --train_path data/train.json \
    --val_path data/val.json

# Generate pseudolabels
python -m src.run --task pseudolabel \
    --model_name qwen2-audio-instruct \
    --data_name audioset \
    --train_path data/unlabeled.json \
    --output_dir ./pseudolabels
```

### Python API

```python
from src import (
    load_model,
    open_data,
    calm_prepare_cache,
    calm_compute_posteriors_from_cache,
    calm_compute_reliability,
    calm_build_weights_from_r,
    calm_eval_from_posteriors,
)

# Load model and data
model = load_model("qwen2-audio-instruct", "vgg_sound_qa")
train_data = open_data("vgg_sound_qa", "train.json")
val_data = open_data("vgg_sound_qa", "val.json")
test_data = open_data("vgg_sound_qa", "test.json")

# Build cache (extracts and caches activations)
cache = calm_prepare_cache(
    model,
    support_data=train_data,
    val_data=val_data,
    test_data=test_data,
    n_trials=20,
    cache_dir="./cache"
)

# Compute per-head posteriors
P_val = calm_compute_posteriors_from_cache(cache, tau=0.07, split="val")
P_test = calm_compute_posteriors_from_cache(cache, tau=0.07, split="test")

# Compute reliability weights
r, counts = calm_compute_reliability(P_val, cache["val_labels_idx"], "margin_clamped")
w = calm_build_weights_from_r(r, weight_scheme="margin_clamped", tau_w=1.0)

# Evaluate
accuracy = calm_eval_from_posteriors(P_test, w, test_labels_idx=cache["test_labels_idx"])
print(f"Accuracy: {accuracy:.4f}")
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.07 | Temperature for class posteriors (lower = sharper) |
| `tau_w` | 1.0 | Temperature for head weighting |
| `weight_scheme` | margin_clamped | Reliability estimation method |
| `n_trials` | 20 | Number of trials for activation averaging |
| `top_k` | None | Optional top-k head selection per class |
| `last_n_tokens` | 1 | Number of tokens to average |

### Weight Schemes

- `margin_clamped`: Clamped margin between correct class and runner-up (recommended)
- `margin_softmax`: Raw margin without clamping
- `prob_softmax`: Mean probability for correct class
- `brier_softmax`: Brier skill score

## Supported Models

| Model | Identifier |
|-------|------------|
| Qwen2-Audio-7B-Instruct | `qwen2-audio-instruct` |
| Qwen2.5-Omni-7B | `qwen2.5_omni` |

## Data Format

Input data should be JSON files with the following structure:

```json
[
  {
    "wav": "/path/to/audio.wav",
    "question": "What sound is this?",
    "answer": "dog barking",
    "label": "dog",
    "mapped_label": "dog",
    "options": ["cat", "dog", "bird", "car"]
  }
]
```

Required fields:
- `wav`: Path to audio file
- `mapped_label`: Class label for the sample

Optional fields:
- `question`, `answer`: For question-answering format
- `options`: Multiple choice options
- `label`: Original label (may differ from mapped_label)


## License

MIT License. See [LICENSE](LICENSE) for details.
