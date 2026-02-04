from .prwe_utils import (
    load_model,
    prwe_prepare_cache,
    prwe_compute_posteriors_from_cache,
)
from .group_weighted_prwe import (
    grw_compute_reliability,
    grw_build_weights,
)
from ..model import *
from ..preprocess import *
from tqdm import tqdm
import torch
import argparse
import os
import csv
import re
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()


def normalize_spoof_prediction(prediction: str) -> str:
    """Normalize spoofing prediction to standard format"""
    p = (prediction or "").lower().strip()
    if any(k in p for k in ["spoof", "fake", "synthetic", "artificial", "no"]):
        return "spoof"
    if any(k in p for k in ["bonafide", "genuine", "real", "authentic", "yes"]):
        return "bonafide"
    # default to bonafide if unclear
    return "bonafide"


def grw_get_predictions(P_test, w):
    """
    Get individual predictions from GRW posteriors and weights.
    
    Args:
        P_test: Tensor of shape (T, K, C) - posteriors for T test samples
        w: Tensor of shape (K,) - global weights per head
    
    Returns:
        List of predicted class indices
    """
    if P_test is None or P_test.numel() == 0:
        return []
    # Weight posteriors by head weights and sum across heads
    # P_test: (T, K, C), w: (K,) -> weighted: (T, K, C)
    weighted = P_test * w.unsqueeze(0).unsqueeze(-1)  # (T, K, C)
    # Sum across heads to get class scores
    scores = weighted.sum(dim=1)  # (T, C)
    pred_idx = scores.argmax(dim=1)
    return pred_idx.cpu().tolist()


def eval_dataset(args):
    base_sav_dir = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/SAV_results/probabilistic/"
    # Create dataset-specific subdirectory
    sav_dir = os.path.join(base_sav_dir, args.data_name, args.model_name)
    os.makedirs(sav_dir, exist_ok=True)
    activation_hook_type = "LM_ATTN"

    # Load
    model = load_model(args.model_name, args.data_name)
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    support_data = open_data(args.data_name, args.support_path) if args.support_path else train_data
    test_data = open_data(args.data_name, args.test_path) if getattr(args, "test_path", None) else val_data

    # Summary file for sweep
    train_tag = os.path.splitext(os.path.basename(args.train_path))[0]
    val_tag = os.path.splitext(os.path.basename(args.val_path))[0]
    test_tag = os.path.splitext(os.path.basename(args.test_path))[0] if getattr(args, "test_path", None) else val_tag
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        sav_dir,
        f"grw_spoof_sweep_{args.data_name}_{args.model_name}_{train_tag}-{val_tag}-{test_tag}_"
        f"n{args.n_trials}_{activation_hook_type}_{date_tag}.txt"
    )
    summary_csv_file = os.path.join(
        sav_dir,
        f"grw_spoof_sweep_{args.data_name}_{args.model_name}_{train_tag}-{val_tag}-{test_tag}_"
        f"n{args.n_trials}_{activation_hook_type}_{date_tag}.csv"
    )
    
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Task: Audio Spoofing Detection (GRW)\n")
        f.write(f"Training Path: {args.train_path}\n")
        f.write(f"Validation Path: {args.val_path}\n")
        f.write(f"Test Path: {args.test_path or 'N/A'}\n")
        f.write("\nHyperparameter Sweep (GRW-SAV for Spoofing)\n")
        f.write("-" * 100 + "\n")
        f.write("    tau |  tau_w |  top_k | lastN | Accuracy | Macro F1 | F1 (bonafide) | F1 (spoof) |   C |   K |   D\n")
        f.write("-" * 100 + "\n")
    
    # Create CSV file for sweep results
    summary_csv = open(summary_csv_file, 'w', newline='', encoding='utf-8')
    summary_csv_writer = csv.writer(summary_csv)
    summary_csv_writer.writerow([
        "tau", "tau_w", "top_k", "lastN",
        "accuracy", "macro_f1", "f1_bonafide", "f1_spoof", "C", "K", "D"
    ])

    # Create directories for matrices
    base_matrix_dir = os.path.join(sav_dir, f"{args.model_name}_grw_spoof_matrices_n{args.n_trials}_{date_tag}")
    weight_matrix_dir = os.path.join(base_matrix_dir, "weight_matrix")
    reliability_matrix_dir = os.path.join(base_matrix_dir, "reliability_matrix")
    os.makedirs(weight_matrix_dir, exist_ok=True)
    os.makedirs(reliability_matrix_dir, exist_ok=True)

    # Prepare lists for sweep
    tau_list = args.tau if isinstance(args.tau, list) else [args.tau]
    tauw_list = args.tau_w if isinstance(args.tau_w, list) else [args.tau_w]
    topk_list = args.top_k if isinstance(args.top_k, list) else ([args.top_k] if args.top_k is not None else [None])
    lastn_list = args.last_n_tokens if isinstance(args.last_n_tokens, list) else [args.last_n_tokens]

    for lastN in lastn_list:
        # Build cache per lastN configuration
        cache = prwe_prepare_cache(model, support_data, val_data, test_data, heads=None, last_n_tokens=lastN, n_trials=args.n_trials)
        C, K, D = cache["prototypes"].shape
        int_to_str = cache["int_to_str"]

        for tau in tau_list:
            # Per-tau posteriors for val/test
            P_val = prwe_compute_posteriors_from_cache(cache, tau=tau, split="val")
            P_test = prwe_compute_posteriors_from_cache(cache, tau=tau, split="test")

            # Compute global reliability (one scalar per head, shape K)
            r = grw_compute_reliability(P_val, cache["val_labels_idx"])

            for tauw in tauw_list:
                for topk in topk_list:
                    # Build weights from global reliability
                    w = grw_build_weights(r, tau_w=tauw, top_k=topk)
                    
                    # Get individual predictions
                    pred_indices = grw_get_predictions(P_test, w)
                    
                    # Convert indices to labels
                    pred_labels = [int_to_str[idx] if 0 <= idx < C else "UNKNOWN" for idx in pred_indices]
                    
                    # Get mapping from cached indices to original test_data indices
                    test_meta = cache.get("qacts_test_n", None)
                    original_indices = test_meta.get("original_indices", None) if test_meta else None
                    
                    # Collect predictions and ground truths
                    predictions = []
                    ground_truths = []
                    
                    # Iterate through predictions (which correspond to cached items)
                    test_items = list(test_data)
                    for cached_idx, predicted_label in enumerate(pred_labels):
                        # Map cached index back to original test_data index
                        if original_indices is not None and cached_idx < len(original_indices):
                            original_idx = original_indices[cached_idx]
                        else:
                            # Fallback: assume sequential if no mapping available
                            original_idx = cached_idx
                        
                        if original_idx >= len(test_items):
                            continue
                        
                        item = test_items[original_idx]
                        ground_truth = item.get("mapped_label", item.get("label", ""))
                        
                        # Normalize prediction for spoofing detection
                        normalized_pred = normalize_spoof_prediction(predicted_label)
                        normalized_gt = normalize_spoof_prediction(ground_truth)
                        
                        predictions.append(normalized_pred)
                        ground_truths.append(normalized_gt)
                    
                    # Compute spoofing detection metrics
                    labels = ["bonafide", "spoof"]
                    macro_f1 = f1_score(ground_truths, predictions, labels=labels, average='macro', zero_division=0)
                    acc = accuracy_score(ground_truths, predictions)
                    precisions, recalls, f1s, support = precision_recall_fscore_support(
                        ground_truths, predictions, labels=labels, zero_division=0
                    )
                    
                    f1_bonafide = f1s[0] if len(f1s) > 0 else 0.0
                    f1_spoof = f1s[1] if len(f1s) > 1 else 0.0
                    
                    # Write to summary text file
                    with open(summary_file, 'a') as f:
                        f.write(f"{tau:6.3f} | {tauw:6.3f} | {str(topk if topk is not None else 'ALL'):>6} | {lastN:5d} | {acc:8.4f} | {macro_f1:8.4f} | {f1_bonafide:13.4f} | {f1_spoof:11.4f} | {C:3d} | {K:3d} | {D:3d}\n")
                    
                    # Write to summary CSV file
                    summary_csv_writer.writerow([
                        tau, tauw, topk if topk is not None else "ALL", lastN,
                        acc, macro_f1, f1_bonafide, f1_spoof, C, K, D
                    ])
                    
                    # Save matrices as numpy arrays
                    # r and w have shape (K,) - one scalar per head
                    r_np = r.cpu().numpy()
                    w_np = w.cpu().numpy()
                    
                    # Create filename
                    param_str = f"tau{tau}_tauw{tauw}_topk{str(topk if topk is not None else 'ALL')}_lastN{lastN}"
                    
                    # Save raw numpy arrays
                    np.save(os.path.join(weight_matrix_dir, f"weight_{param_str}.npy"), w_np)
                    np.save(os.path.join(reliability_matrix_dir, f"reliability_{param_str}.npy"), r_np)

    summary_csv.close()
    
    with open(summary_file, 'a') as f:
        f.write("\nArtifacts\n")
        f.write("-" * 100 + "\n")
        f.write(f"weight_matrix_dir: {weight_matrix_dir}\n")
        f.write(f"reliability_matrix_dir: {reliability_matrix_dir}\n")
        f.write(f"Summary CSV saved to: {summary_csv_file}\n")

    print(f"\nResults have been saved to {summary_file}.")
    print(f"CSV results saved to {summary_csv_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRW-SAV for Audio Spoofing Detection")
    parser.add_argument(
        "--model_name", type=str, default="your_audio_model_ckpt",
        help="Name or path of the multimodal/audio model to load"
    )
    parser.add_argument(
        "--data_name", type=str, default="LA_spoof",
        help="Dataset identifier (used by open_data to know how to parse it)"
    )
    parser.add_argument(
        "--train_path", type=str, required=True,
        help="Path to train-split metadata"
    )
    parser.add_argument(
        "--val_path", type=str, required=True,
        help="Path to val/test-split metadata"
    )
    parser.add_argument(
        "--support_path", type=str, default=None,
        help="Optional path for support split; if omitted, uses train data for prototypes"
    )
    parser.add_argument(
        "--test_path", type=str, default=None,
        help="Optional held-out test split for final evaluation (default: uses --val_path)"
    )
    parser.add_argument(
        "--tau", type=float, nargs='+', default=[0.07],
        help="Temperature(s) for per-head softmax over classes (list)"
    )
    parser.add_argument(
        "--tau_w", type=float, nargs='+', default=[1.0],
        help="Temperature(s) for softmax over heads when building weights (list)"
    )
    parser.add_argument(
        "--top_k", type=int, nargs='+', default=None,
        help="Optional list: restrict GRW-SAV to top-k heads when weighting"
    )
    parser.add_argument(
        "--last_n_tokens", type=int, nargs='+', default=[1],
        help="One or more N values; if >1, average head activations over last N tokens"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Number of trials for averaging activations when building prototypes and query activations"
    )
    args = parser.parse_args()
    eval_dataset(args)

