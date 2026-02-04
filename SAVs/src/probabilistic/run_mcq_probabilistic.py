from .utils import select_heads_with_mw, select_heads_with_hedge, evaluate_with_heads, load_model
from ..model import *        # load_model, model.insert_audio, model.generate
from ..preprocess import *
from tqdm import tqdm
import torch
import argparse
import os

torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()


def eval_dataset(args):
    # Print args
    print("All arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 50)

    # Defaults
    head_counts = args.top_k if args.top_k else [5]
    sav_dir = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/SAV_results/probabilistic/"
    os.makedirs(sav_dir, exist_ok=True)
    activation_hook_type = "LM_ATTN"

    # Load
    model = load_model(args.model_name, args.data_name)
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    test_data = val_data  # follow existing convention: evaluate on provided val_path

    # Summary file
    summary_file = f"{sav_dir}head_accuracies_prob_{args.data_name}_{args.model_name}_nc{args.num_candidates}_{activation_hook_type}_hedged.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Num Candidates (MW): {args.num_candidates}\n")
        f.write("Number of Heads | Accuracy\n")
        f.write("-" * 30 + "\n")

    # For each requested top_k, run Hedge selection and evaluate
    for k in tqdm(head_counts, desc="Testing different head counts (probabilistic Hedge)"):
        embed = select_heads_with_hedge(
            model,
            train_data,
            val_data,
            num_candidates=args.num_candidates,
            top_k=k,
            tau=args.tau,
            eta=args.eta,
            seed=args.seed,
        )

        acc = evaluate_with_heads(model, test_data, embed)
        print(f"Prob-Hedge Accuracy with {k} heads: {acc:.4f}")

        with open(summary_file, 'a') as f:
            f.write(f"{k} heads | {acc:.4f}\n")

    print(f"\nResults have been saved to {summary_file}.")


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
        "--top_k", type=int, nargs='+', default=None,
        help="List of head counts to test (e.g., --top_k 5 10 20 40 100)"
    )
    parser.add_argument(
        "--num_candidates", type=int, default=20,
        help="Number of candidate heads considered by MW (d)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.07,
        help="Temperature for per-head softmax over classes"
    )
    parser.add_argument(
        "--eta", type=float, default=1.0,
        help="Learning rate for Hedge updates"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for MW candidate sampling"
    )
    args = parser.parse_args()
    eval_dataset(args)


