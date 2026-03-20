"""
CLAP-based K-Nearest Neighbor audio classification baseline.

For each dataset, builds class centroids from train-set audio embeddings,
then classifies test-set audio by nearest centroid (cosine similarity).
No MCQ questions are used — raw audio is embedded directly via CLAP.
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as nnF
from sklearn.metrics import f1_score
from tqdm import tqdm


def load_clap(device):
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    model = model.to(device).eval()
    return model, processor


def embed_audio_batch(wav_paths, model, processor, device, sr=48000):
    """Embed a batch of audio files, returning (N, D) tensor."""
    import librosa

    arrays = []
    valid_idx = []
    for i, p in enumerate(wav_paths):
        try:
            y, _ = librosa.load(p, sr=sr, mono=True)
            arrays.append(y)
            valid_idx.append(i)
        except Exception as e:
            warnings.warn(f"Could not load {p}: {e}")

    if not arrays:
        return torch.empty(0, 512, device=device), valid_idx

    inputs = processor(audios=arrays, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeds = model.get_audio_features(**inputs)
    embeds = nnF.normalize(embeds, dim=-1)
    return embeds, valid_idx


def embed_text(class_names, model, processor, device):
    """Embed class names as text, returning (C, D) tensor."""
    inputs = processor(text=class_names, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeds = model.get_text_features(**inputs)
    embeds = nnF.normalize(embeds, dim=-1)
    return embeds


def build_audio_centroids(train_data, label_key, model, processor, device, batch_size=32):
    """Build per-class centroids from train audio embeddings."""
    class_embeds = {}
    class_counts = {}

    wav_paths = [item["wav"] for item in train_data]
    labels = [item[label_key] for item in train_data]

    for start in tqdm(range(0, len(wav_paths), batch_size), desc="Embedding train audio"):
        batch_paths = wav_paths[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        embeds, valid_idx = embed_audio_batch(batch_paths, model, processor, device)

        for j, global_j in enumerate(valid_idx):
            lab = batch_labels[global_j]
            if lab not in class_embeds:
                class_embeds[lab] = embeds[j]
                class_counts[lab] = 1
            else:
                class_embeds[lab] = class_embeds[lab] + embeds[j]
                class_counts[lab] += 1

    classes = sorted(class_embeds.keys())
    centroids = torch.stack([class_embeds[c] / class_counts[c] for c in classes])
    centroids = nnF.normalize(centroids, dim=-1)
    return centroids, classes


def classify_test(test_data, label_key, centroids, class_list,
                  model, processor, device, batch_size=64):
    """Classify test audio via nearest centroid and return accuracy, macro-F1, and label arrays."""
    wav_paths = [item["wav"] for item in test_data]
    gt_labels = [item[label_key] for item in test_data]
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    y_true = []
    y_pred = []

    for start in tqdm(range(0, len(wav_paths), batch_size), desc="Classifying test audio"):
        batch_paths = wav_paths[start : start + batch_size]
        batch_gt = gt_labels[start : start + batch_size]
        embeds, valid_idx = embed_audio_batch(batch_paths, model, processor, device)

        if embeds.shape[0] == 0:
            continue

        sims = embeds @ centroids.T  # (B, C)
        preds = sims.argmax(dim=-1)

        for j, global_j in enumerate(valid_idx):
            gt = batch_gt[global_j]
            if gt not in class_to_idx:
                continue
            y_true.append(class_to_idx[gt])
            y_pred.append(preds[j].item())

    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / total if total > 0 else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if total > 0 else 0.0
    return acc, macro_f1, correct, total, y_true, y_pred


def run_dataset(name, train_path, test_path, model, processor, device, batch_size=32):
    """Run CLAP KNN for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"  Train:   {train_path}")
    print(f"  Test:    {test_path}")
    print(f"{'='*60}")

    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)
    if isinstance(train_data, dict):
        train_data = list(train_data.values())
    if isinstance(test_data, dict):
        test_data = list(test_data.values())

    label_key = "mapped_label"

    train_classes = set(item[label_key] for item in train_data)
    test_classes = set(item[label_key] for item in test_data)
    print(f"  Train samples: {len(train_data)}, classes: {len(train_classes)}")
    print(f"  Test  samples: {len(test_data)}, classes: {len(test_classes)}")

    # -- Audio-based centroids --
    print("\n[Audio centroids] Building centroids from train audio...")
    centroids, class_list = build_audio_centroids(
        train_data, label_key, model, processor, device, batch_size=batch_size
    )
    print(f"  Centroids shape: {centroids.shape}  ({len(class_list)} classes)")

    acc, macro_f1, correct, total, _, _ = classify_test(
        test_data, label_key, centroids, class_list,
        model, processor, device, batch_size=batch_size
    )
    print(f"\n  [Audio centroids] Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}  ({correct}/{total})")

    # -- Text-based centroids (zero-shot) --
    print("\n[Text centroids] Building centroids from class-name text embeddings...")
    text_centroids = embed_text(class_list, model, processor, device)
    text_acc, text_macro_f1, text_correct, text_total, _, _ = classify_test(
        test_data, label_key, text_centroids, class_list,
        model, processor, device, batch_size=batch_size
    )
    print(f"  [Text centroids] Accuracy: {text_acc:.4f}  Macro-F1: {text_macro_f1:.4f}  ({text_correct}/{text_total})")

    return {
        "dataset": name,
        "audio_centroid_acc": acc,
        "audio_centroid_macro_f1": macro_f1,
        "text_centroid_acc": text_acc,
        "text_centroid_macro_f1": text_macro_f1,
        "n_classes": len(class_list),
        "n_train": len(train_data),
        "n_test": total,
    }


def main():
    parser = argparse.ArgumentParser(description="CLAP KNN audio classification baseline")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset specs as name:train_path:test_path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    model, processor = load_clap(device)

    results = []
    for spec in args.datasets:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Skipping invalid spec '{spec}' — expected name:train_path:test_path")
            continue
        name, train_path, test_path = parts
        res = run_dataset(name, train_path, test_path, model, processor, device,
                          batch_size=args.batch_size)
        results.append(res)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  CLAP KNN Classification Summary")
    print(f"{'='*90}")
    print(f"  {'Dataset':<15} {'Classes':>8} {'Train':>7} {'Test':>7}  {'Audio Acc':>10}  {'Audio F1':>10}  {'Text Acc':>10}  {'Text F1':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*7} {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['dataset']:<15} {r['n_classes']:>8} {r['n_train']:>7} {r['n_test']:>7}"
              f"  {r['audio_centroid_acc']:>9.2%}  {r['audio_centroid_macro_f1']:>9.2%}"
              f"  {r['text_centroid_acc']:>9.2%}  {r['text_centroid_macro_f1']:>9.2%}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
