# Dataset Processing Scripts

Scripts for preparing datasets for CALM classification.

## Scripts

### `prune_audioset.py`
Prune AudioSet to N-shot per class.

```bash
python prune_audioset.py --input input.json --output output.json --shots 20
```

### `prepare_asvspoof.py`
Convert ASVspoof2019 LA to CALM format.

```bash
python prepare_asvspoof.py \
    --protocol ASVspoof2019.LA.cm.train.trn.txt \
    --audio_dir /path/to/flac \
    --output train.json
```

### `prepare_esc50.py`
Create N-shot ESC-50 dataset with multiple choice questions.

```bash
python prepare_esc50.py \
    --train_files train1.json train2.json \
    --eval_file eval.json \
    --label_csv labels.csv \
    --audio_dir /path/to/audio \
    --output_dir /path/to/output \
    --shots 40
```

## Data Format

Output JSON files follow this format:

```json
[
  {
    "wav": "/path/to/audio.wav",
    "question": "What sound is this?\nA. dog bark\nB. car horn\nC. music\nD. speech",
    "answer": "A",
    "mapped_label": "dog_bark"
  }
]
```

Required fields:
- `wav`: Path to audio file
- `question`: Question text (optional for some datasets)
- `answer`: Correct answer letter or label
- `mapped_label`: Human-readable class label for classification
