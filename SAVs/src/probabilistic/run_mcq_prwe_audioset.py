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
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()




def calculate_sample_metrics(sample_predictions, sample_ground_truths):
    """
    Calculate metrics for a single original sample.
    
    Args:
        sample_predictions: List of predicted labels for this sample
        sample_ground_truths: List of ground truth labels for this sample
    
    Returns:
        exact_match: 1 if all labels correct, 0 otherwise
        partial_accuracy: Fraction of labels correctly predicted
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)
        f1: 2 * precision * recall / (precision + recall)
    """
    pred_set = set(sample_predictions)
    gt_set = set(sample_ground_truths)
    
    if len(gt_set) == 0:
        return 0, 0, 0, 0, 0
    
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    exact_match = 1 if pred_set == gt_set else 0
    partial_accuracy = tp / len(gt_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return exact_match, partial_accuracy, precision, recall, f1


def calculate_average_precision(predicted_labels, ground_truth_labels, all_labels):
    """Calculate Average Precision for a single sample."""
    y_true = np.zeros(len(all_labels))
    y_scores = np.zeros(len(all_labels))
    
    for label in ground_truth_labels:
        if label in all_labels:
            idx = all_labels.index(label)
            y_true[idx] = 1
    
    for label in predicted_labels:
        if label in all_labels:
            idx = all_labels.index(label)
            y_scores[idx] = 1
    
    if np.sum(y_true) == 0:
        return 0.0
    
    tp = np.sum(y_true * y_scores)
    fp = np.sum((1 - y_true) * y_scores)
    fn = np.sum(y_true * (1 - y_scores))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision * recall


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
    sav_dir = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/SAV_results/probabilistic/"
    os.makedirs(sav_dir, exist_ok=True)
    activation_hook_type = "LM_ATTN"

    # Load
    model = load_model(args.model_name, args.data_name)
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    support_data = open_data(args.data_name, args.support_path) if args.support_path else train_data
    test_data = open_data(args.data_name, args.test_path) if getattr(args, "test_path", None) else val_data

    # Get all unique labels for mAP calculation
    all_labels = set()
    for item in test_data:
        if 'original_labels' in item:
            all_labels.update(item['original_labels'])
        else:
            label = item.get("correct_label", item.get("mapped_label", item.get("label", "")))
            if isinstance(label, list):
                all_labels.update(label)
            else:
                all_labels.add(label)
    all_labels = sorted(list(all_labels))
    print(f"Found {len(all_labels)} unique labels")

    # Summary file for sweep
    train_tag = os.path.splitext(os.path.basename(args.train_path))[0]
    val_tag = os.path.splitext(os.path.basename(args.val_path))[0]
    test_tag = os.path.splitext(os.path.basename(args.test_path))[0] if getattr(args, "test_path", None) else val_tag
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = (
        f"{sav_dir}prwe_audioset_sweep_{args.data_name}_{args.model_name}_{train_tag}-{val_tag}-{test_tag}_"
        f"n{args.n_trials}_{activation_hook_type}_{date_tag}.txt"
    )
    summary_csv_file = (
        f"{sav_dir}prwe_audioset_sweep_{args.data_name}_{args.model_name}_{train_tag}-{val_tag}-{test_tag}_"
        f"n{args.n_trials}_"
        f"{activation_hook_type}_{date_tag}.csv"
    )
    
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Training Path: {args.train_path}\n")
        f.write(f"Validation Path: {args.val_path}\n")
        f.write(f"Test Path: {args.test_path or 'N/A'}\n")
        f.write("\nHyperparameter Sweep\n")
        f.write("-" * 140 + "\n")
        f.write("weight_scheme |    tau |  alpha | shrinkage |  tau_w | entmax |  top_k | lastN | Q.Acc | S.EM | S.F1 | mAP |   C |   K |   D\n")
        f.write("-" * 140 + "\n")
    
    # Create CSV file for sweep results
    summary_csv = open(summary_csv_file, 'w', newline='', encoding='utf-8')
    summary_csv_writer = csv.writer(summary_csv)
    summary_csv_writer.writerow([
        "weight_scheme", "tau", "alpha", "shrinkage", "tau_w", "entmax", "top_k", "lastN",
        "question_accuracy", "sample_exact_match", "sample_f1", "mean_ap", "C", "K", "D"
    ])

    # Create directories for matrices
    base_matrix_dir = f"{sav_dir}_{args.model_name}_matrices_{date_tag}"
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
        # Build PRWE-invariant cache per lastN configuration (using audio-only mode)
        cache = prwe_prepare_cache(model, support_data, val_data, test_data, heads=None, last_n_tokens=lastN, audio_or_video="audio", n_trials=args.n_trials)
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
                                    
                                    # Track per-sample results
                                    sample_results = defaultdict(lambda: {
                                        'predictions': [],
                                        'ground_truths': [],
                                        'questions_answered': 0,
                                        'questions_correct': 0
                                    })
                                    
                                    question_level_correct = 0
                                    total_questions = 0
                                    
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
                                        ground_truth = item.get("correct_label", item.get("mapped_label", item.get("label", "")))
                                        
                                        # Case-insensitive comparison to handle label normalization differences
                                        # (e.g., training might have "conversation" while test has "Conversation")
                                        is_correct = (predicted_label.lower() == ground_truth.lower())
                                        
                                        global_item_index = item.get("original_sample_id", total_questions)
                                        
                                        # Always add prediction to track what the model predicted
                                    sample_results[global_item_index]['predictions'].append(predicted_label)
                                    sample_results[global_item_index]['questions_answered'] += 1
                                        
                                        # Set ground truths for this sample (only once)
                                        if not sample_results[global_item_index]['ground_truths']:
                                            original_labels = item.get("original_labels", [ground_truth])
                                            sample_results[global_item_index]['ground_truths'] = original_labels
                                        
                                        # Count total questions OUTSIDE the is_correct check!
                                        total_questions += 1
                                        
                                        if is_correct:
                                            question_level_correct += 1
                                            sample_results[global_item_index]['questions_correct'] += 1
                                    
                                    # Calculate sample-level metrics (OUTSIDE the prediction loop)
                                    sample_exact_matches = 0
                                    sample_f1s = []
                                    sample_aps = []
                                    invalid_samples = 0
                                    
                                    for sample_id, results in sample_results.items():
                                        predictions = results['predictions']
                                        ground_truths = results['ground_truths']
                                        
                                        if not ground_truths:
                                            invalid_samples += 1
                                            continue
                                        
                                        questions_total = results['questions_answered']
                                        questions_correct = results['questions_correct']
                                        exact_match = 1 if (questions_total > 0 and questions_correct == questions_total) else 0
                                        
                                        _em_tmp, partial_acc, precision, recall, f1 = calculate_sample_metrics(
                                            predictions, ground_truths
                                        )
                                        
                                        ap = calculate_average_precision(predictions, ground_truths, all_labels)
                                        
                                        sample_exact_matches += exact_match
                                        sample_f1s.append(f1)
                                        sample_aps.append(ap)
                                    
                                    # Calculate overall metrics
                                    valid_samples = len(sample_results) - invalid_samples
                                    question_accuracy = question_level_correct / total_questions if total_questions > 0 else 0
                                    sample_exact_match_accuracy = sample_exact_matches / valid_samples if valid_samples > 0 else 0
                                    mean_sample_f1 = np.mean(sample_f1s) if sample_f1s else 0
                                    mean_ap = np.mean(sample_aps) if sample_aps else 0
                                    
                                    # Write to summary text file
                                    with open(summary_file, 'a') as f:
                                        f.write(f"{ws:>13} | {tau:6.3f} | {alpha:6.3f} | {shrinkage:>5} | {tauw:6.3f} | {ent_alpha:6.2f} | {str(topk if topk is not None else 'ALL'):>6} | {lastN:5d} | {question_accuracy:5.3f} | {sample_exact_match_accuracy:5.3f} | {mean_sample_f1:5.3f} | {mean_ap:5.3f} | {C:3d} | {K:3d} | {D:3d}\n")
                                    
                                    # Write to summary CSV file
                                    summary_csv_writer.writerow([
                                        ws, tau, alpha, shrinkage, tauw, ent_alpha, 
                                        topk if topk is not None else "ALL", lastN,
                                        question_accuracy, sample_exact_match_accuracy, 
                                        mean_sample_f1, mean_ap, C, K, D
                                    ])
                                    
                                    # # Save matrices - for the sake of time, we don't need to save the matrices
                                    # param_str = f"tau{tau}_alpha{alpha}_shrink{shrinkage}_tauw{tauw}_ent{ent_alpha}_topk{str(topk if topk is not None else 'ALL')}_ws{ws}_lastN{lastN}"
                                    # r_hat_np = r_hat.cpu().numpy()
                                    # w_np = w.cpu().numpy()
                                    # class_labels = [int_to_str[i] for i in range(C)]
                                    
                                    # w_path = os.path.join(weight_matrix_dir, f"weight_{param_str}.png")
                                    # save_matrix_heatmap(w_np, class_labels, w_path, 
                                    #                     f"Weight Matrix (K={K}, C={C})\n{param_str}", 
                                    #                     xlabel="Class", ylabel="Head Index", vmin=0, vmax=1)
                                    
                                    # r_path = os.path.join(reliability_matrix_dir, f"reliability_{param_str}.png")
                                    # save_matrix_heatmap(r_hat_np, class_labels, r_path,
                                    #                     f"Reliability Matrix (K={K}, C={C})\n{param_str}",
                                    #                     xlabel="Class", ylabel="Head Index", vmin=None, vmax=None)
                                    
                                    # np.save(os.path.join(weight_matrix_dir, f"weight_{param_str}.npy"), w_np)
                                    # np.save(os.path.join(reliability_matrix_dir, f"reliability_{param_str}.npy"), r_hat_np)

    summary_csv.close()
    print("savingi everything in summary file", summary_file)
    with open(summary_file, 'a') as f:
        f.write("\nArtifacts\n")
        f.write("-" * 140 + "\n")
        f.write(f"weight_matrix_dir: {weight_matrix_dir}\n")
        f.write(f"reliability_matrix_dir: {reliability_matrix_dir}\n")
        f.write(f"Summary CSV saved to: {summary_csv_file}\n")


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
        "--tau", type=float, nargs='+', default=[0.07],
        help="Temperature(s) for per-head softmax over classes (list)"
    )
    parser.add_argument(
        "--alpha", type=float, nargs='+', default=[0.0],
        help="Reliability shrinkage strength values (list)"
    )
    parser.add_argument(
        "--test_path", type=str, default=None,
        help="Optional held-out test split for final evaluation (default: uses --val_path)"
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

