from .prwe_utils import (
    load_model,
    prwe_prepare_cache,
    prwe_compute_posteriors_from_cache,
)
from .group_weighted_prwe import (
    grw_compute_reliability,
    grw_build_weights,
    grw_eval_from_posteriors,
)
from ..model import *        # load_model, model.insert_audio, model.generate
from ..preprocess import *
from tqdm import tqdm
import torch
import argparse
import os
from datetime import datetime
import numpy as np

torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()


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
    # Use train data for support (prototypes) to avoid leakage
    support_data = open_data(args.data_name, args.support_path) if args.support_path else train_data
    # Held-out test split (if not provided, falls back to val)
    test_data = open_data(args.data_name, args.test_path) if getattr(args, "test_path", None) else val_data

    # Summary file for sweep (single consolidated table)
    train_tag = os.path.splitext(os.path.basename(args.train_path))[0]
    val_tag = os.path.splitext(os.path.basename(args.val_path))[0]
    test_tag = os.path.splitext(os.path.basename(args.test_path))[0] if getattr(args, "test_path", None) else val_tag
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        sav_dir,
        f"grw_sav_sweep_{args.data_name}_{args.model_name}_{train_tag}-{test_tag}_n{args.n_trials}_{date_tag}.txt"
    )
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Training Path: {args.train_path}\n")
        f.write(f"Validation Path: {args.val_path}\n")
        f.write(f"Test Path: {args.test_path or 'N/A'}\n")
        f.write("\nHyperparameter Sweep (GRW-SAV)\n")
        f.write("-" * 80 + "\n")
        f.write("    tau |  tau_w |  top_k | lastN | accuracy |   C |   K |   D\n")
        f.write("-" * 80 + "\n")

    # Create directories for matrices
    base_matrix_dir = os.path.join(sav_dir, f"{args.model_name}_grw_matrices_n{args.n_trials}_{date_tag}")
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
        # Build PRWE-invariant cache per lastN configuration
        cache = prwe_prepare_cache(model, support_data, val_data, test_data, heads=None, last_n_tokens=lastN, n_trials=args.n_trials)
        C, K, D = cache["prototypes"].shape

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
                    
                    # Evaluate on test set
                    acc = grw_eval_from_posteriors(P_test, w, test_labels_idx=cache["test_labels_idx"])
                    
                    with open(summary_file, 'a') as f:
                        f.write(f"{tau:6.3f} | {tauw:6.3f} | {str(topk if topk is not None else 'ALL'):>6} | {lastN:5d} | {acc:8.4f} | {C:3d} | {K:3d} | {D:3d}\n")
                    
                    # Save matrices as numpy arrays
                    # r and w have shape (K,) - one scalar per head
                    # Convert to numpy for saving
                    r_np = r.cpu().numpy()  # (K,)
                    w_np = w.cpu().numpy()  # (K,)
                    
                    # Create filename
                    param_str = f"tau{tau}_tauw{tauw}_topk{str(topk if topk is not None else 'ALL')}_lastN{lastN}"
                    
                    # Save raw numpy arrays
                    np.save(os.path.join(weight_matrix_dir, f"weight_{param_str}.npy"), w_np)
                    np.save(os.path.join(reliability_matrix_dir, f"reliability_{param_str}.npy"), r_np)

    print("Saving everything in summary file", summary_file)
    
    with open(summary_file, 'a') as f:
        f.write("\nArtifacts\n")
        f.write("-" * 80 + "\n")
        f.write(f"weight_matrix_dir: {weight_matrix_dir}\n")
        f.write(f"reliability_matrix_dir: {reliability_matrix_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRW-SAV: Global Reliability Weighted SAV")
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
        "--test_path", type=str, default=None,
        help="Optional held-out test split for final evaluation (default: uses --val_path)"
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

