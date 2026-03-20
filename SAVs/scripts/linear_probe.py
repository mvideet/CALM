"""
Linear probing on Qwen2-Audio and Qwen2.5-Omni activations.

Extracts last-layer hidden state at the last token position, trains a
linear classifier (LogisticRegression) on train embeddings, evaluates
accuracy and macro-F1 on test.
"""
import argparse
import json
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.calm import load_model
from src.preprocess import open_data


def extract_hidden_state(model_helper, inputs, device):
    """Extract last hidden state (last layer, last token) from model forward."""
    model_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_helper.model(**model_input, output_hidden_states=True)
    # hidden_states: tuple of (num_layers+1) tensors, each (batch, seq_len, hidden_size)
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # (batch, hidden_size)
    return last_hidden.cpu().float().numpy()


def collect_embeddings(dataset, model_helper, split, device, n_trials=1, str_to_int=None):
    """Collect last hidden state embeddings for all items in dataset.

    If str_to_int is provided (e.g. from train), use it for label mapping and
    skip items with unknown labels. Otherwise build str_to_int from data.
    """
    embeddings = []
    labels = []
    label_key = "mapped_label"
    fixed_labels = str_to_int is not None

    items = list(dataset)
    if isinstance(dataset, dict):
        items = list(dataset.values())

    if str_to_int is None:
        str_to_int = {}

    for item in tqdm(items, desc=f"Extracting {split} embeddings"):
        lab = item.get(label_key, item.get("label", ""))
        lab_key = lab.lower() if isinstance(lab, str) else lab

        if fixed_labels and lab_key not in str_to_int:
            continue

        try:
            result = model_helper.format_func(
                all_data=items,
                cur_item=item,
                num_shot=0,
                model_helper=model_helper,
                split=split,
            )
            if len(result) == 5:
                tqs, ans, audio_list, _, _ = result
            else:
                tqs, ans, _, audio_list, _, _ = result
            inputs = model_helper.insert_audio(tqs, ans, audio_list)
        except Exception as e:
            warnings.warn(f"Skipping item: {e}")
            continue

        if lab_key not in str_to_int:
            str_to_int[lab_key] = len(str_to_int)

        emb_list = []
        for _ in range(n_trials):
            emb = extract_hidden_state(model_helper, inputs, device)
            emb_list.append(emb)
        emb_mean = np.mean(emb_list, axis=0)
        embeddings.append(emb_mean.squeeze())
        labels.append(str_to_int[lab_key])

    if not embeddings:
        return None, None, str_to_int, {v: k for k, v in str_to_int.items()}

    X = np.stack(embeddings, axis=0)
    y = np.array(labels)
    int_to_str = {v: k for k, v in str_to_int.items()}
    return X, y, str_to_int, int_to_str


def run_dataset(name, model_name, train_path, test_path, device, n_trials=1):
    """Run linear probing for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"  Model:   {model_name}")
    print(f"  Train:   {train_path}")
    print(f"  Test:    {test_path}")
    print(f"{'='*60}")

    data_name_map = {
        "ESC-50": "esc_mcq",
        "LA_Spoof": "LA_spoof",
        "VGGSound": "vgg_sound_qa",
        "AudioSet": "audioset",
    }
    data_name = data_name_map.get(name, name.lower().replace("-", "_").replace(" ", "_"))

    print("Loading model...")
    model = load_model(model_name, data_name)

    print("Loading data...")
    train_data = open_data(data_name, train_path)
    test_data = open_data(data_name, test_path)
    if isinstance(train_data, dict):
        train_data = list(train_data.values())
    if isinstance(test_data, dict):
        test_data = list(test_data.values())

    print(f"  Train: {len(train_data)}  Test: {len(test_data)}")

    print("Extracting train embeddings...")
    X_train, y_train, str_to_int, int_to_str = collect_embeddings(
        train_data, model, "train", device, n_trials=n_trials
    )
    if X_train is None:
        raise RuntimeError("No train embeddings extracted")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}  classes: {len(str_to_int)}")

    print("Extracting test embeddings...")
    X_test, y_test, _, _ = collect_embeddings(
        test_data, model, "test", device, n_trials=1, str_to_int=str_to_int
    )
    if X_test is None:
        raise RuntimeError("No test embeddings extracted")
    print(f"  X_test: {X_test.shape}  y_test: {y_test.shape}  (filtered to train classes)")

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train linear probe
    print("Training linear probe (LogisticRegression)...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    acc = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")

    return {
        "dataset": name,
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(str_to_int),
    }


def main():
    parser = argparse.ArgumentParser(description="Linear probing on Qwen2-Audio / Qwen2.5-Omni")
    parser.add_argument("--model", required=True, choices=["qwen2-audio-instruct", "qwen2.5_omni"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--n_trials", type=int, default=1, help="Trials for train embedding averaging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    res = run_dataset(
        args.dataset,
        args.model,
        args.train_path,
        args.test_path,
        device,
        n_trials=args.n_trials,
    )

    print(f"\n{'='*50}")
    print(f"  Linear Probe Summary")
    print(f"{'='*50}")
    print(f"  {res['dataset']} | {res['model']}")
    print(f"  Accuracy: {res['accuracy']:.4f}  Macro-F1: {res['macro_f1']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
