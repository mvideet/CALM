from .utils import *        # must include: open_data, mllm_encode_audio, mllm_classify_audio, mllm_classify_spoof
from .model import *        # must include: load_model, model.insert_audio, model.generate
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
import csv
import re
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()  

def normalize_answer(answer):
    """Normalize answer by removing option prefix (A., B., etc) and trimming whitespace"""
    answer = re.sub(r'^[A-D]\.\n*', '', answer.strip())
    return answer.lower().strip()

# Local normalization for spoof/bonafide outputs

def normalize_spoof_prediction(prediction: str) -> str:
    p = (prediction or "").lower().strip()
    if any(k in p for k in ["spoof", "fake", "synthetic", "artificial", "no"]):
        return "spoof"
    if any(k in p for k in ["bonafide", "genuine", "real", "authentic", "yes"]):
        return "bonafide"
    # default to bonafide if unclear
    return "bonafide"

def eval_dataset(args):
    # Define range of heads to test
    print("All arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 50)
    head_counts = [20,40,100]
    # head_counts = [10]
    # head_counts = [300, 500, 640]
    sav_dir = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/SAV_results/"
    # shot=40 #for vggsound
    shot=1 #for esc
    zero_shot = args.eval_zeroshot  # Use command-line argument instead of hardcoded True
    model = load_model(args.model_name, args.data_name)
    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)
    activation_hook_type = "LM_ATTN"
    # Create summary file
    summary_file = f'{sav_dir}spoof_detection_results_{args.data_name}_{args.model_name}_{shot}shot_{activation_hook_type}.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Task: Audio Spoofing Detection\n")
        f.write(f"Activation Hook Type: {activation_hook_type}\n")
        f.write("=" * 60 + "\n\n")

    # Zero‐shot evaluation
    if zero_shot:
        print("Running zero-shot evaluation")
        predictions = []
        ground_truths = []
        
        for item in tqdm(test_data, desc="Zero-shot Evaluation"):
            tqs, ans, wavs, _, _ = model.format_func(
                all_data=None, cur_item=item, num_shot=0, model_helper=model, split="test"
            )
            model_input = model.insert_audio(tqs, ans, wavs)
            output = model.generate(model_input, max_new_tokens=32).strip()
            
            normalized_pred = normalize_spoof_prediction(output)
            ground_truth = item.get("mapped_label", item.get("label", ""))
            
            predictions.append(normalized_pred)
            ground_truths.append(ground_truth)

        # Compute Macro F1 and related metrics
        labels = ["bonafide", "spoof"]
        macro_f1 = f1_score(ground_truths, predictions, labels=labels, average='macro', zero_division=0)
        acc = accuracy_score(ground_truths, predictions)
        precisions, recalls, f1s, support = precision_recall_fscore_support(
            ground_truths, predictions, labels=labels, zero_division=0
        )
        
        print("\n" + "="*60)
        print("ZERO-SHOT SPOOFING DETECTION RESULTS")
        print("="*60)
        print(f"Total samples: {len(test_data)}")
        print("ACCURACY/F1 METRICS:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Class F1 (bonafide, spoof): {f1s.tolist()}")
        
        # Save results to summary file
        with open(summary_file, 'a') as f:
            f.write(f"Zero-shot Results:\n")
            f.write(f"  Accuracy: {acc:.4f}\n")
            f.write(f"  Macro F1: {macro_f1:.4f}\n")
            f.write(f"  Class F1 (bonafide, spoof): {f1s.tolist()}\n")
            f.write("-" * 50 + "\n")
        
        return

    # Create config for mllm_encode
    config = {'N_TRIALS': getattr(args, 'n_trials', 1)}  # Default to 1 trial if not specified
    
    # Store results for summary
    results = []
    
    # Embedding-based Evaluation for different head counts
    for num_head in tqdm(head_counts, desc="Testing different head counts"):
        print(f"\n--- Evaluating with {num_head} heads ---")
        
        multimodal_embeddings = mllm_encode(model, train_data, num_head=num_head, config=config)

        predictions = []
        ground_truths = []
        
        for item in tqdm(test_data, desc=f"SAV evaluation with {num_head} heads"):
            cur_class, _confidence = mllm_classify_spoof(item, model, multimodal_embeddings)
            ground_truth = item.get("mapped_label", item.get("label", ""))
            normalized_pred = normalize_spoof_prediction(cur_class)
            predictions.append(normalized_pred)
            ground_truths.append(ground_truth)

        # Compute Macro F1 and related metrics
        labels = ["bonafide", "spoof"]
        macro_f1 = f1_score(ground_truths, predictions, labels=labels, average='macro', zero_division=0)
        acc = accuracy_score(ground_truths, predictions)
        precisions, recalls, f1s, support = precision_recall_fscore_support(
            ground_truths, predictions, labels=labels, zero_division=0
        )
        
        print(f"\nSAV Results for {num_head} heads:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Class F1 (bonafide, spoof): {f1s.tolist()}")
        
        # Show first few predictions
        print(f"\nSample predictions (first 3):")
        for i in range(min(3, len(predictions))):
            print(f"  {i+1}: pred={predictions[i]}, gt={ground_truths[i]}")
        
        # Store results for summary
        results.append({
            'heads': num_head,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'f1_bonafide': f1s[0] if len(f1s) > 0 else 0.0,
            'f1_spoof': f1s[1] if len(f1s) > 1 else 0.0
        })
        
        # Print accuracy prominently after each head count
        print(f"\n{'='*60}")
        print(f"✓ COMPLETED {num_head} heads: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
        print(f"{'='*60}\n")
        
        # Write results to summary file
        with open(summary_file, 'a') as f:
            f.write(f"{num_head} heads:\n")
            f.write(f"  Accuracy: {acc:.4f}\n")
            f.write(f"  Macro F1: {macro_f1:.4f}\n")
            f.write(f"  Class F1 (bonafide, spoof): {f1s.tolist()}\n")
            f.write("-" * 30 + "\n")

    # Print summary table at the end
    print("\n" + "="*60)
    print("SUMMARY: All Head Count Results")
    print("="*60)
    print(f"{'Heads':<8} {'Accuracy':<12} {'Macro F1':<12} {'F1 (bonafide)':<15} {'F1 (spoof)':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['heads']:<8} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['f1_bonafide']:<15.4f} {r['f1_spoof']:<12.4f}")
    print("="*60)

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
        help="Path to VGGSound train‐split metadata (e.g. JSON or CSV listing audio_path+label)"
    )
    parser.add_argument(
        "--val_path", type=str, required=True,
        help="Path to VGGSound test‐split metadata"
    )
    parser.add_argument(
        "--eval_zeroshot", action="store_true",
        help="If set, run zero‐shot generation (model.insert_audio → model.generate)"
    )
    parser.add_argument(
        "--n_trials", type=int, default=1,
        help="Number of trials for averaging activations when building prototypes"
    )
    args = parser.parse_args()
    eval_dataset(args)
