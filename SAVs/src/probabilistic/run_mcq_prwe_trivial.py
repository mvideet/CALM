"""
Evaluation Script for Simple Soft Voting Methods

Compares multiple simple classification approaches:
1. concat          - Concatenated prototype (single cosine similarity)
2. uniform_soft    - Uniform soft voting across all heads
3. confidence_weighted - Self-weighted by prediction confidence
4. topk_soft       - Top-K heads with uniform soft voting
5. hard_vote       - Traditional majority voting (baseline)

Usage:
    python eval_simple_soft_vote.py \
        --model_name qwen2-audio-instruct \
        --data_name vggsound \
        --train_path /path/to/train.json \
        --val_path /path/to/val.json \
        --test_path /path/to/test.json \
        --tau 0.05 0.07 0.1 \
        --top_k 32 64 128
"""
#ended up not working, showing that prwe is better
import sys
import os
sys.path.insert(0, '/data/sls/u/urop/mvideet/sparse_audio/SAVs/src')
from probabilistic.probablistic_trivial import (
    load_model,
    prepare_simple_cache,
    eval_from_cache,
    select_top_heads_from_cache,
)
from preprocess import open_data

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
    """Main evaluation function."""
    
    # Setup directories
    base_dir = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/SAV_results/simple_soft_vote/"
    sav_dir = os.path.join(base_dir, args.data_name, args.model_name)
    os.makedirs(sav_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = load_model(args.model_name, args.data_name)
    
    # Load datasets
    print(f"Loading datasets...")
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    support_data = open_data(args.data_name, args.support_path) if args.support_path else train_data
    test_data = open_data(args.data_name, args.test_path) if args.test_path else val_data
    
    # Create summary file
    train_tag = os.path.splitext(os.path.basename(args.train_path))[0]
    val_tag = os.path.splitext(os.path.basename(args.val_path))[0]
    test_tag = os.path.splitext(os.path.basename(args.test_path))[0] if args.test_path else val_tag
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = os.path.join(
        sav_dir,
        f"simple_sweep_{args.data_name}_{args.model_name}_{train_tag}-{test_tag}_n{args.n_trials}_{date_tag}.txt"
    )
    
    # Write header
    with open(summary_file, 'w') as f:
        f.write(f"Simple Soft Voting Evaluation\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Train Path: {args.train_path}\n")
        f.write(f"Val Path: {args.val_path}\n")
        f.write(f"Test Path: {args.test_path or 'N/A'}\n")
        f.write(f"Date: {date_tag}\n")
        f.write(f"\n")
        f.write(f"Methods:\n")
        f.write(f"  - concat: Single cosine sim on concatenated head vectors\n")
        f.write(f"  - uniform_soft: Average posteriors across all heads\n")
        f.write(f"  - confidence_weighted: Weight heads by prediction confidence\n")
        f.write(f"  - topk_soft: Uniform soft vote among top-K validated heads\n")
        f.write(f"  - hard_vote: Traditional majority voting baseline\n")
        f.write(f"\n")
        f.write("=" * 80 + "\n\n")
    
    # Parse hyperparameter lists
    tau_list = args.tau if isinstance(args.tau, list) else [args.tau]
    topk_list = args.top_k if isinstance(args.top_k, list) else [args.top_k]
    lastn_list = args.last_n_tokens if isinstance(args.last_n_tokens, list) else [args.last_n_tokens]
    
    methods = args.methods if args.methods else ['concat', 'uniform_soft', 'confidence_weighted', 'topk_soft', 'hard_vote']
    
    # Results storage
    all_results = []
    
    for lastN in lastn_list:
        print(f"\n{'='*60}")
        print(f"Building cache with last_n_tokens={lastN}")
        print(f"{'='*60}")
        
        # Build cache once per lastN configuration
        cache = prepare_simple_cache(
            model, support_data, val_data, test_data,
            n_trials=args.n_trials, last_n_tokens=lastN
        )
        
        C, K, D = cache["C"], cache["K"], cache["D"]
        
        with open(summary_file, 'a') as f:
            f.write(f"\n--- last_n_tokens = {lastN} ---\n")
            f.write(f"Prototypes: C={C} classes, K={K} heads, D={D} dims\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'method':>20} | {'tau':>6} | {'top_k':>6} | {'accuracy':>10} | {'notes'}\n")
            f.write("-" * 70 + "\n")
        
        # Precompute top-K head selections for all K values
        topk_selections = {}
        for k in topk_list:
            if k is not None and 'topk_soft' in methods:
                print(f"Selecting top-{k} heads...")
                top_indices, head_accs = select_top_heads_from_cache(cache, k=k)
                topk_selections[k] = top_indices
                
                # Log top head accuracies
                top_acc_mean = head_accs[top_indices].mean().item()
                print(f"  Top-{k} heads mean accuracy: {top_acc_mean:.4f}")
        
        # Evaluate methods
        for method in methods:
            print(f"\nEvaluating: {method}")
            
            if method == 'concat':
                # No hyperparameters
                acc = eval_from_cache(cache, method, split="test")
                result = {'method': method, 'tau': None, 'top_k': None, 'lastN': lastN, 'accuracy': acc}
                all_results.append(result)
                
                with open(summary_file, 'a') as f:
                    f.write(f"{method:>20} | {'N/A':>6} | {'ALL':>6} | {acc:>10.4f} | single cosine sim\n")
                print(f"  {method}: {acc:.4f}")
            
            elif method == 'hard_vote':
                # No hyperparameters
                acc = eval_from_cache(cache, method, split="test")
                result = {'method': method, 'tau': None, 'top_k': None, 'lastN': lastN, 'accuracy': acc}
                all_results.append(result)
                
                with open(summary_file, 'a') as f:
                    f.write(f"{method:>20} | {'N/A':>6} | {'ALL':>6} | {acc:>10.4f} | majority voting\n")
                print(f"  {method}: {acc:.4f}")
            
            elif method in ['uniform_soft', 'confidence_weighted']:
                # Sweep over tau
                for tau in tau_list:
                    acc = eval_from_cache(cache, method, split="test", tau=tau)
                    result = {'method': method, 'tau': tau, 'top_k': None, 'lastN': lastN, 'accuracy': acc}
                    all_results.append(result)
                    
                    with open(summary_file, 'a') as f:
                        f.write(f"{method:>20} | {tau:>6.3f} | {'ALL':>6} | {acc:>10.4f} |\n")
                    print(f"  {method} (tau={tau}): {acc:.4f}")
            
            elif method == 'topk_soft':
                # Sweep over tau and top_k
                for tau in tau_list:
                    for k in topk_list:
                        if k is None:
                            continue
                        top_indices = topk_selections.get(k)
                        if top_indices is None:
                            continue
                        
                        acc = eval_from_cache(cache, method, split="test", tau=tau, top_k_indices=top_indices)
                        result = {'method': method, 'tau': tau, 'top_k': k, 'lastN': lastN, 'accuracy': acc}
                        all_results.append(result)
                        
                        with open(summary_file, 'a') as f:
                            f.write(f"{method:>20} | {tau:>6.3f} | {k:>6} | {acc:>10.4f} |\n")
                        print(f"  {method} (tau={tau}, k={k}): {acc:.4f}")
        
        # Clean up cache
        del cache
        torch.cuda.empty_cache()
    
    # Write summary
    with open(summary_file, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY - Best Results per Method\n")
        f.write("=" * 70 + "\n")
        
        # Group by method and find best
        method_best = {}
        for r in all_results:
            m = r['method']
            if m not in method_best or r['accuracy'] > method_best[m]['accuracy']:
                method_best[m] = r
        
        # Sort by accuracy
        sorted_methods = sorted(method_best.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        f.write(f"\n{'Rank':>4} | {'Method':>20} | {'Accuracy':>10} | {'tau':>6} | {'top_k':>6} | {'lastN':>6}\n")
        f.write("-" * 70 + "\n")
        
        for rank, (method, result) in enumerate(sorted_methods, 1):
            tau_str = f"{result['tau']:.3f}" if result['tau'] is not None else "N/A"
            topk_str = str(result['top_k']) if result['top_k'] is not None else "ALL"
            f.write(f"{rank:>4} | {method:>20} | {result['accuracy']:>10.4f} | {tau_str:>6} | {topk_str:>6} | {result['lastN']:>6}\n")
        
        f.write("\n")
        
        # Overall best
        best = max(all_results, key=lambda x: x['accuracy'])
        f.write(f"\nBest Overall: {best['method']} with accuracy {best['accuracy']:.4f}\n")
        f.write(f"  Parameters: tau={best['tau']}, top_k={best['top_k']}, last_n_tokens={best['lastN']}\n")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {summary_file}")
    print(f"{'='*60}")
    
    # Print quick summary
    print("\nQuick Summary (Best per Method):")
    for method, result in sorted_methods:
        print(f"  {method:>20}: {result['accuracy']:.4f}")
    
    return all_results


def compare_with_prwe(args, simple_results):
    """
    Optional: Load PRWE results and compare.
    """
    # This would load existing PRWE results and generate a comparison table
    # Left as a stub for future implementation
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate simple soft voting methods for prototype classification"
    )
    
    # Model and data
    parser.add_argument(
        "--model_name", type=str, default="qwen2-audio-instruct",
        help="Model name: qwen2-audio-instruct or qwen2.5_omni"
    )
    parser.add_argument(
        "--data_name", type=str, default="vggsound",
        help="Dataset identifier"
    )
    parser.add_argument(
        "--train_path", type=str, required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--val_path", type=str, required=True,
        help="Path to validation data (used for top-K selection)"
    )
    parser.add_argument(
        "--test_path", type=str, default=None,
        help="Path to test data (defaults to val if not provided)"
    )
    parser.add_argument(
        "--support_path", type=str, default=None,
        help="Path to support data for prototypes (defaults to train)"
    )
    
    # Methods to evaluate
    parser.add_argument(
        "--methods", type=str, nargs='+',
        default=['concat', 'uniform_soft', 'confidence_weighted', 'topk_soft', 'hard_vote'],
        help="Methods to evaluate"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--tau", type=float, nargs='+', default=[0.05, 0.07, 0.1, 0.15, 0.2],
        help="Temperature values for softmax"
    )
    parser.add_argument(
        "--top_k", type=int, nargs='+', default=[32, 64, 128, 256],
        help="Number of top heads for topk_soft method"
    )
    parser.add_argument(
        "--last_n_tokens", type=int, nargs='+', default=[1],
        help="Number of last tokens to average"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Number of trials for prototype averaging"
    )
    
    args = parser.parse_args()
    
    results = eval_dataset(args)