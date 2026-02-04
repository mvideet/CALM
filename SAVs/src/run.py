"""
CALM: Main entry point for audio classification.

Usage:
    python -m src.run \
        --model_name qwen2-audio-instruct \
        --data_name LA_spoof \
        --train_path /path/to/train.json \
        --val_path /path/to/test.json \
        --num_heads 20

For spoofing detection evaluation:
    python -m src.run \
        --model_name qwen2-audio-instruct \
        --data_name LA_spoof \
        --train_path /path/to/train.json \
        --val_path /path/to/test.json \
        --task spoof
"""
import argparse
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .utils import load_model, mllm_encode, mllm_classify, mllm_classify_spoof
from .preprocess import open_data

torch.set_grad_enabled(False)


def evaluate_classification(args):
    """Standard classification evaluation."""
    print(f"Loading model: {args.model_name}")
    model = load_model(args.model_name, args.data_name)
    
    print(f"Loading data...")
    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    config = {"N_TRIALS": args.n_trials}
    
    print(f"\nExtracting class-conditional attention vectors with {args.num_heads} heads...")
    class_embed = mllm_encode(model, train_data, num_head=args.num_heads, config=config)
    
    print(f"\nClassifying test samples...")
    predictions = []
    ground_truths = []
    
    for item in tqdm(test_data):
        pred = mllm_classify(item, model, class_embed)
        predictions.append(pred)
        ground_truths.append(item.get("mapped_label", item.get("label", "")))
    
    accuracy = accuracy_score(ground_truths, predictions)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(ground_truths, predictions))


def evaluate_spoof_detection(args):
    """Spoofing detection evaluation."""
    print(f"Loading model: {args.model_name}")
    model = load_model(args.model_name, args.data_name)
    
    print(f"Loading data...")
    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    config = {"N_TRIALS": args.n_trials}
    
    print(f"\nExtracting class-conditional attention vectors with {args.num_heads} heads...")
    class_embed = mllm_encode(model, train_data, num_head=args.num_heads, config=config)
    
    print(f"\nClassifying test samples...")
    predictions = []
    ground_truths = []
    
    for item in tqdm(test_data):
        pred, confidence = mllm_classify_spoof(item, model, class_embed)
        # Normalize prediction
        pred_normalized = "spoof" if "spoof" in pred.lower() else "bonafide"
        predictions.append(pred_normalized)
        ground_truths.append(item.get("mapped_label", item.get("label", "")))
    
    labels = ["bonafide", "spoof"]
    accuracy = accuracy_score(ground_truths, predictions)
    macro_f1 = f1_score(ground_truths, predictions, labels=labels, average='macro', zero_division=0)
    
    print("\n" + "=" * 60)
    print("SPOOFING DETECTION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(ground_truths, predictions, labels=labels))


def main():
    parser = argparse.ArgumentParser(description="CALM: Audio classification with attention vectors")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["qwen2-audio-instruct", "qwen2.5_omni"],
                        help="Model to use")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Dataset name (vgg_sound_qa, esc_mcq, audioset, LA_spoof, mlaad)")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training JSON file")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to test JSON file")
    parser.add_argument("--num_heads", type=int, default=20,
                        help="Number of attention heads to use (default: 20)")
    parser.add_argument("--n_trials", type=int, default=1,
                        help="Number of trials for activation averaging (default: 1)")
    parser.add_argument("--task", type=str, default="classify",
                        choices=["classify", "spoof"],
                        help="Task type: classify or spoof detection")
    
    args = parser.parse_args()
    
    if args.task == "spoof":
        evaluate_spoof_detection(args)
    else:
        evaluate_classification(args)


if __name__ == "__main__":
    main()
