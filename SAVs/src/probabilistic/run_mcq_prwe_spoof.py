from .prwe_utils import (
    load_model,
    prwe_prepare_cache,
    prwe_compute_posteriors_from_cache,
    prwe_compute_reliability,
    prwe_apply_shrinkage,
    prwe_build_weights_from_r,
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


def normalize_answer(answer):
    """Normalize answer by removing option prefix (A., B., etc) and trimming whitespace"""
    answer = re.sub(r'^[A-D]\.\n*', '', answer.strip())
    return answer.lower().strip()


def normalize_spoof_prediction(prediction: str) -> str:
    """Normalize spoofing prediction to standard format"""
    p = (prediction or "").lower().strip()
    if any(k in p for k in ["spoof", "fake", "synthetic", "artificial", "no"]):
        return "spoof"
    if any(k in p for k in ["bonafide", "genuine", "real", "authentic", "yes"]):
        return "bonafide"
    # default to bonafide if unclear
    return "bonafide"


def prwe_get_predictions(P_test, w, cache):
    """
    Get individual predictions from PRWE posteriors and weights.
    Returns list of predicted class indices.
    """
    if P_test is None or P_test.numel() == 0:
        return []
    scores = (P_test * w.unsqueeze(0)).sum(dim=1)
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
        f"prwe_spoof_sweep_{args.data_name}_{args.model_name}_{train_tag}-{val_tag}-{test_tag}_"
        f"{activation_hook_type}_{date_tag}.txt"
    )
    summary_csv_file = os.path.join(
        sav_dir,
        f"prwe_spoof_sweep_{args.data_name}_{args.model_name}_{train_tag}-{val_tag}-{test_tag}_"
        f"n{args.n_trials}_{activation_hook_type}_{date_tag}.csv"
    )
    
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Task: Audio Spoofing Detection\n")
        f.write(f"Training Path: {args.train_path}\n")
        f.write(f"Validation Path: {args.val_path}\n")
        f.write(f"Test Path: {args.test_path or 'N/A'}\n")
        f.write("\nHyperparameter Sweep\n")
        f.write("-" * 140 + "\n")
        f.write("weight_scheme |    tau |  alpha | shrinkage |  tau_w | entmax |  top_k | lastN | Accuracy | Macro F1 | F1 (bonafide) | F1 (spoof) |   C |   K |   D\n")
        f.write("-" * 140 + "\n")
    
    # Create CSV file for sweep results
    summary_csv = open(summary_csv_file, 'w', newline='', encoding='utf-8')
    summary_csv_writer = csv.writer(summary_csv)
    summary_csv_writer.writerow([
        "weight_scheme", "tau", "alpha", "shrinkage", "tau_w", "entmax", "top_k", "lastN",
        "accuracy", "macro_f1", "f1_bonafide", "f1_spoof", "C", "K", "D"
    ])

    # Create directories for matrices
    base_matrix_dir = os.path.join(sav_dir, f"{args.model_name}_matrices_{date_tag}")
    weight_matrix_dir = os.path.join(base_matrix_dir, "weight_matrix")
    reliability_matrix_dir = os.path.join(base_matrix_dir, "reliability_matrix")
    os.makedirs(weight_matrix_dir, exist_ok=True)
    os.makedirs(reliability_matrix_dir, exist_ok=True)

    # Prepare lists for sweep
    tau_list = args.tau if isinstance(args.tau, list) else [args.tau]
    alpha_list = args.alpha if isinstance(args.alpha, list) else [args.alpha]
    tauw_list = args.tau_w if isinstance(args.tau_w, list) else [args.tau_w]
    ws_list = args.weight_scheme if isinstance(args.weight_scheme, list) else [args.weight_scheme]
    topk_list = args.top_k if isinstance(args.top_k, list) else ([args.top_k] if args.top_k is not None else [None])
    entmax_alpha_list = args.alpha_entmax if isinstance(args.alpha_entmax, list) else [args.alpha_entmax]
    shrinkage_list = [True, False]
    lastn_list = args.last_n_tokens if isinstance(args.last_n_tokens, list) else [args.last_n_tokens]

    for lastN in lastn_list:
        # Build PRWE-invariant cache per lastN configuration
        cache = prwe_prepare_cache(model, support_data, val_data, test_data, heads=None, last_n_tokens=lastN, n_trials=args.n_trials)
        C, K, D = cache["prototypes"].shape
        int_to_str = cache["int_to_str"]

        for tau in tau_list:
            # Per-tau posteriors for val/test
            P_val = prwe_compute_posteriors_from_cache(cache, tau=tau, split="val")
            P_test = prwe_compute_posteriors_from_cache(cache, tau=tau, split="test")

            for ws in ws_list:
                # Base reliabilities and counts from validation posteriors
                r, counts = prwe_compute_reliability(P_val, cache["val_labels_idx"], ws)
                
                for alpha in alpha_list:
                    for shrinkage in shrinkage_list:
                        if shrinkage:
                            r_hat = prwe_apply_shrinkage(r, counts, alpha)
                        else:
                            r_hat = r

                        for tauw in tauw_list:
                            for ent_alpha in entmax_alpha_list:
                                for topk in topk_list:
                                    w = prwe_build_weights_from_r(
                                        r_hat,
                                        weight_scheme=ws,
                                        tau_w=tauw,
                                        top_k=topk,
                                        alpha_entmax=ent_alpha,
                                    )
                                    
                                    # Get individual predictions
                                    pred_indices = prwe_get_predictions(P_test, w, cache)
                                    
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
                                        f.write(f"{ws:>13} | {tau:6.3f} | {alpha:6.3f} | {str(shrinkage):>5} | {tauw:6.3f} | {ent_alpha:6.2f} | {str(topk if topk is not None else 'ALL'):>6} | {lastN:5d} | {acc:8.4f} | {macro_f1:8.4f} | {f1_bonafide:13.4f} | {f1_spoof:11.4f} | {C:3d} | {K:3d} | {D:3d}\n")
                                    
                                    # Write to summary CSV file
                                    summary_csv_writer.writerow([
                                        ws, tau, alpha, shrinkage, tauw, ent_alpha, 
                                        topk if topk is not None else "ALL", lastN,
                                        acc, macro_f1, f1_bonafide, f1_spoof, C, K, D
                                    ])
                                    
                                    # Save matrices as numpy arrays
                                    # r_hat and w have shape (K, C) - Head x Class
                                    # Convert to numpy for saving
                                    r_hat_np = r_hat.cpu().numpy()  # (K, C)
                                    w_np = w.cpu().numpy()  # (K, C)
                                    
                                    # Create filename
                                    param_str = f"tau{tau}_alpha{alpha}_shrink{shrinkage}_tauw{tauw}_ent{ent_alpha}_topk{str(topk if topk is not None else 'ALL')}_ws{ws}_lastN{lastN}"
                                    
                                    # Save raw numpy arrays
                                    np.save(os.path.join(weight_matrix_dir, f"weight_{param_str}.npy"), w_np)
                                    np.save(os.path.join(reliability_matrix_dir, f"reliability_{param_str}.npy"), r_hat_np)

    summary_csv.close()
    
    with open(summary_file, 'a') as f:
        f.write("\nArtifacts\n")
        f.write("-" * 140 + "\n")
        f.write(f"weight_matrix_dir: {weight_matrix_dir}\n")
        f.write(f"reliability_matrix_dir: {reliability_matrix_dir}\n")
        f.write(f"Summary CSV saved to: {summary_csv_file}\n")

    print(f"\nResults have been saved to {summary_file}.")
    print(f"CSV results saved to {summary_csv_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="your_audio_model_ckpt",
        help="Name or path of the multimodal/audio model to load"
    )
    parser.add_argument(
        "--data_name", type=str, default="vggsound",
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
        "--alpha", type=float, nargs='+', default=[0.0],
        help="Reliability shrinkage strength values (list)"
    )
    parser.add_argument(
        "--weight_scheme", type=str, nargs='+', default=["margin_clamped"],
        help="One or more of {'margin_clamped','margin_softmax','prob_softmax','brier_softmax'}"
    )
    parser.add_argument(
        "--tau_w", type=float, nargs='+', default=[1.0],
        help="Temperature(s) for softmax over heads when using margin_softmax/prob_softmax (list)"
    )
    parser.add_argument(
        "--top_k", type=int, nargs='+', default=None,
        help="Optional list: restrict PRWE to top-k heads per class when weighting"
    )
    parser.add_argument(
        "--alpha_entmax", type=float, nargs='+', default=[1.0],
        help="Entmax alpha choices: 1.0 (softmax), 1.5 (entmax15), 2.0 (sparsemax)"
    )
    parser.add_argument("--shrinkage", action="store_true", help="If set, apply shrinkage to reliabilities")
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
