# CALM: Class-conditional Attention vectors for audio Language Models

CALM is a training-free method for extracting discriminative features from audio language models, enabling few-shot audio classification that outperforms fine-tuned baselines.

## Overview

Large audio language models excel at generative tasks but are not directly suited for discriminative tasks like classification. CALM extracts sparse attention head activations from these models and uses them as class-conditional features for classification.

**Key Features:**
- Training-free: No model fine-tuning required
- Few-shot: Works with just a handful of examples per class
- Interpretable: Uses less than 1% of attention heads as features
- Effective: Outperforms both few-shot and fine-tuned baselines

## Installation

```bash
git clone https://github.com/chancharikmitra/CALM.git
cd CALM

conda create -n calm python=3.10 -y
conda activate calm
pip install -e .
```

## Quickstart

### Data Format

Format your data as a JSON file with the following structure:

```json
[
  {"wav": "/path/to/audio1.wav", "question": "What sound is this?", "mapped_label": "dog_bark"},
  {"wav": "/path/to/audio2.wav", "question": "What sound is this?", "mapped_label": "car_horn"}
]
```

### Classification

```python
from src.utils import load_model, mllm_encode, mllm_classify
from src.preprocess import open_data

# Load model
model = load_model("qwen2-audio-instruct", "your_dataset")

# Load data
train_data = open_data("your_dataset", "/path/to/train.json")
test_data = open_data("your_dataset", "/path/to/test.json")

# Extract class-conditional attention vectors
config = {"N_TRIALS": 1}
class_embed = mllm_encode(model, train_data, num_head=20, config=config)

# Classify
for item in test_data:
    prediction = mllm_classify(item, model, class_embed)
    print(f"Predicted: {prediction}, Actual: {item['mapped_label']}")
```

### Command Line

```bash
python -m src.run_mcq_spoof \
    --model_name qwen2-audio-instruct \
    --data_name LA_spoof \
    --train_path /path/to/train.json \
    --val_path /path/to/test.json
```

## Supported Models

| Model | Identifier | Description |
|-------|------------|-------------|
| Qwen2-Audio-7B-Instruct | `qwen2-audio-instruct` | Audio-only language model |
| Qwen2.5-Omni-7B | `qwen2.5_omni` | Audio-visual-language model |

## Supported Datasets

- VGGSound (`vgg_sound`, `vgg_sound_qa`)
- ESC-50 (`esc_mcq`)
- AudioSet (`audioset`)
- ASVspoof/LA (`LA_spoof`)
- MLAAD (`mlaad`)

## How It Works

1. **Extract Activations**: For each training sample, extract attention head activations from the audio language model.

2. **Select Top Heads**: Evaluate each attention head's ability to discriminate between classes and select the top-k performing heads.

3. **Build Class Centroids**: Compute mean activations for each class using only the selected heads.

4. **Classify**: For new inputs, extract activations for the selected heads and find the nearest class centroid using cosine similarity with majority voting.

## API Reference

### `load_model(model_name, cur_dataset)`
Load a model and return its helper class.

### `mllm_encode(model, train_data, num_head, config)`
Extract class-conditional attention vectors from training data.

Returns a dict with:
- `activations`: Class centroids tensor
- `top_heads`: Selected (layer, head) tuples
- `int_to_str`: Class index to label mapping

### `mllm_classify(inputs, model, class_embed)`
Classify an input using the extracted embeddings.

### `mllm_classify_spoof(inputs, model, class_embed)`
Specialized classification for spoofing detection with confidence scores.

## Citation

If you find this work useful, please cite:

```bibtex
@article{mitra2024sparse,
  title={Sparse Attention Vectors: Generative Multimodal Model Features Are Discriminative Vision-Language Classifiers},
  author={Mitra, Chancharik and Huang, Brandon and Chai, Tianning and Lin, Zhiqiu and Arbelle, Assaf and Feris, Rogerio and Karlinsky, Leonid and Darrell, Trevor and Ramanan, Deva and Herzig, Roei},
  journal={arXiv preprint arXiv:2412.00142},
  year={2024}
}
```

## License

MIT License
