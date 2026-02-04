"""
CALM: Unified entry point for audio classification tasks.

Usage:
    # Audio classification
    python -m src.run --task classify \
        --model_name qwen2-audio-instruct \
        --data_name vgg_sound_qa \
        --train_path /path/to/train.json \
        --val_path /path/to/val.json \
        --test_path /path/to/test.json

    # Spoofing detection
    python -m src.run --task spoof \
        --model_name qwen2-audio-instruct \
        --data_name LA_spoof \
        --train_path /path/to/train.json \
        --val_path /path/to/val.json

    # Pseudolabel generation
    python -m src.run --task pseudolabel \
        --model_name qwen2-audio-instruct \
        --data_name audioset \
        --train_path /path/to/unlabeled.json \
        --output_dir ./pseudolabels
"""
import argparse
import os
from datetime import datetime

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from .calm import (
    calm_build_weights_from_r,
    calm_compute_posteriors_from_cache,
    calm_compute_reliability,
    calm_eval_from_posteriors,
    calm_get_predictions,
    calm_prepare_cache,
    load_model,
)
from .preprocess import open_data

torch.set_grad_enabled(False)


def normalize_spoof_label(label: str) -> str:
    """Normalize spoofing labels to 'bonafide' or 'spoof'."""
    label_lower = (label or "").lower().strip()
    if any(k in label_lower for k in ["spoof", "fake", "synthetic", "artificial"]):
        return "spoof"
    if any(k in label_lower for k in ["bonafide", "genuine", "real", "authentic"]):
        return "bonafide"
    return "bonafide"


def run_classification(args):
    """Run audio classification with CALM."""
    print("=" * 60)
    print("CALM: Audio Classification")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print(f"Weight scheme: {args.weight_scheme}")
    print(f"Temperature (tau): {args.tau}")
    print(f"Weight temperature (tau_w): {args.tau_w}")
    print(f"N trials: {args.n_trials}")
    print()

    # Load model and data
    print("Loading model...")
    model = load_model(args.model_name, args.data_name)

    print("Loading data...")
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    test_data = open_data(args.data_name, args.test_path) if args.test_path else val_data

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(list(test_data))}")
    print()

    # Prepare cache
    print("Building CALM cache...")
    cache = calm_prepare_cache(
        model,
        support_data=train_data,
        val_data=val_data,
        test_data=test_data if args.test_path else None,
        last_n_tokens=args.last_n_tokens,
        n_trials=args.n_trials,
        cache_dir=args.cache_dir,
    )

    C, K, D = cache["prototypes"].shape
    print(f"  Classes (C): {C}")
    print(f"  Heads (K): {K}")
    print(f"  Dimension (D): {D}")
    print()

    # Compute posteriors
    print("Computing posteriors...")
    P_val = calm_compute_posteriors_from_cache(cache, tau=args.tau, split="val")
    P_test = calm_compute_posteriors_from_cache(
        cache, tau=args.tau, split="test" if args.test_path else "val"
    )

    # Compute reliability and weights
    print("Computing reliability weights...")
    r, counts = calm_compute_reliability(P_val, cache["val_labels_idx"], args.weight_scheme)
    w = calm_build_weights_from_r(
        r,
        weight_scheme=args.weight_scheme,
        tau_w=args.tau_w,
        top_k=args.top_k,
    )

    # Evaluate
    print("Evaluating...")
    test_labels = cache["test_labels_idx"] if args.test_path else cache["val_labels_idx"]
    accuracy = calm_eval_from_posteriors(P_test, w, test_labels_idx=test_labels)

    # Get predictions for detailed report
    pred_labels = calm_get_predictions(P_test, w, cache)
    test_items = list(test_data) if args.test_path else list(val_data)
    
    # Get original indices mapping
    test_meta = cache.get("qacts_test_n" if args.test_path else "qacts_val_n", {})
    original_indices = test_meta.get("original_indices", list(range(len(pred_labels))))

    ground_truths = []
    predictions = []
    for cached_idx, pred in enumerate(pred_labels):
        if cached_idx < len(original_indices):
            orig_idx = original_indices[cached_idx]
            if orig_idx < len(test_items):
                gt = test_items[orig_idx].get("mapped_label", test_items[orig_idx].get("label", ""))
                gt = gt.lower() if isinstance(gt, str) else gt
                ground_truths.append(gt)
                predictions.append(pred)

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print()
    print("Classification Report:")
    print(classification_report(ground_truths, predictions))

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            args.output_dir,
            f"calm_{args.data_name}_{args.model_name}_{timestamp}.txt"
        )
        with open(result_file, "w") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.data_name}\n")
            f.write(f"Weight scheme: {args.weight_scheme}\n")
            f.write(f"Tau: {args.tau}\n")
            f.write(f"Tau_w: {args.tau_w}\n")
            f.write(f"N trials: {args.n_trials}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"\n{classification_report(ground_truths, predictions)}")
        print(f"\nResults saved to: {result_file}")


def run_spoof_detection(args):
    """Run spoofing detection with CALM."""
    print("=" * 60)
    print("CALM: Spoofing Detection")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print()

    # Load model and data
    print("Loading model...")
    model = load_model(args.model_name, args.data_name)

    print("Loading data...")
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    test_data = open_data(args.data_name, args.test_path) if args.test_path else val_data

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(list(test_data))}")
    print()

    # Prepare cache
    print("Building CALM cache...")
    cache = calm_prepare_cache(
        model,
        support_data=train_data,
        val_data=val_data,
        test_data=test_data if args.test_path else None,
        last_n_tokens=args.last_n_tokens,
        n_trials=args.n_trials,
        cache_dir=args.cache_dir,
    )

    C, K, D = cache["prototypes"].shape
    print(f"  Classes (C): {C}")
    print(f"  Heads (K): {K}")
    print()

    # Compute posteriors
    print("Computing posteriors...")
    P_val = calm_compute_posteriors_from_cache(cache, tau=args.tau, split="val")
    P_test = calm_compute_posteriors_from_cache(
        cache, tau=args.tau, split="test" if args.test_path else "val"
    )

    # Compute reliability and weights
    print("Computing reliability weights...")
    r, counts = calm_compute_reliability(P_val, cache["val_labels_idx"], args.weight_scheme)
    w = calm_build_weights_from_r(
        r,
        weight_scheme=args.weight_scheme,
        tau_w=args.tau_w,
        top_k=args.top_k,
    )

    # Get predictions
    pred_labels = calm_get_predictions(P_test, w, cache)
    test_items = list(test_data) if args.test_path else list(val_data)

    # Get original indices mapping
    test_meta = cache.get("qacts_test_n" if args.test_path else "qacts_val_n", {})
    original_indices = test_meta.get("original_indices", list(range(len(pred_labels))))

    predictions = []
    ground_truths = []

    for cached_idx, pred in enumerate(pred_labels):
        if cached_idx < len(original_indices):
            orig_idx = original_indices[cached_idx]
            if orig_idx < len(test_items):
                item = test_items[orig_idx]
                gt = item.get("mapped_label", item.get("label", ""))

                predictions.append(normalize_spoof_label(pred))
                ground_truths.append(normalize_spoof_label(gt))

    # Compute metrics
    labels = ["bonafide", "spoof"]
    accuracy = accuracy_score(ground_truths, predictions)
    macro_f1 = f1_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        ground_truths, predictions, labels=labels, zero_division=0
    )

    # Results
    print()
    print("=" * 60)
    print("SPOOFING DETECTION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"F1 (bonafide): {f1s[0]:.4f}")
    print(f"F1 (spoof): {f1s[1]:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(ground_truths, predictions, labels=labels))

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            args.output_dir,
            f"calm_spoof_{args.data_name}_{args.model_name}_{timestamp}.txt"
        )
        with open(result_file, "w") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.data_name}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n")
            f.write(f"F1 (bonafide): {f1s[0]:.4f}\n")
            f.write(f"F1 (spoof): {f1s[1]:.4f}\n")
            f.write(f"\n{classification_report(ground_truths, predictions, labels=labels)}")
        print(f"\nResults saved to: {result_file}")


def run_pseudolabel(args):
    """Generate pseudolabels using model predictions."""
    from .pseudolabel import generate_pseudolabels

    print("=" * 60)
    print("CALM: Pseudolabel Generation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print(f"N trials: {args.n_trials}")
    print(f"Min confidence: {args.min_confidence}")
    print()

    # Create args for pseudolabel function
    class PseudolabelArgs:
        pass

    pl_args = PseudolabelArgs()
    pl_args.model_name = args.model_name
    pl_args.data_name = args.data_name
    pl_args.train_path = args.train_path
    pl_args.n_trials = args.n_trials
    pl_args.min_confidence = args.min_confidence
    pl_args.output_dir = args.output_dir

    generate_pseudolabels(pl_args)


def main():
    parser = argparse.ArgumentParser(
        description="CALM: Class-conditional Attention vectors for audio Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classify", "spoof", "pseudolabel"],
        help="Task to run: classify (audio classification), spoof (spoofing detection), pseudolabel (generate pseudolabels)",
    )

    # Model and data
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["qwen2-audio-instruct", "qwen2.5_omni"],
        help="Model to use",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help="Dataset name (vgg_sound_qa, esc_mcq, audioset, LA_spoof, mlaad, etc.)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to training/support JSON file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to validation JSON file (for reliability estimation)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to test JSON file (optional, defaults to val_path)",
    )

    # CALM hyperparameters
    parser.add_argument(
        "--tau",
        type=float,
        default=0.07,
        help="Temperature for class posteriors (default: 0.07)",
    )
    parser.add_argument(
        "--tau_w",
        type=float,
        default=1.0,
        help="Temperature for head weighting (default: 1.0)",
    )
    parser.add_argument(
        "--weight_scheme",
        type=str,
        default="margin_clamped",
        choices=["margin_clamped", "margin_softmax", "prob_softmax", "brier_softmax"],
        help="Weight scheme for reliability estimation (default: margin_clamped)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for activation averaging (default: 20)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional top-k head selection per class",
    )
    parser.add_argument(
        "--last_n_tokens",
        type=int,
        default=1,
        help="Number of tokens to average (default: 1)",
    )

    # Pseudolabel-specific
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for pseudolabels (default: 0.5)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory for output files (default: ./results)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for cached activations (default: ./cache)",
    )

    args = parser.parse_args()

    # Validate args
    if args.task in ["classify", "spoof"] and args.val_path is None:
        parser.error("--val_path is required for classify and spoof tasks")

    # Run task
    if args.task == "classify":
        run_classification(args)
    elif args.task == "spoof":
        run_spoof_detection(args)
    elif args.task == "pseudolabel":
        run_pseudolabel(args)


if __name__ == "__main__":
    main()
