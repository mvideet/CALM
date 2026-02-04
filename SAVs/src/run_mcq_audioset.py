from .utils import *        # must include: open_data, mllm_encode_audio, mllm_classify_audio
from .model import *        # must include: load_model, model.insert_audio, model.generate
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
import csv
import re
import json
import os
from collections import defaultdict
import numpy as np

torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()  

def normalize_answer(answer):
    """Normalize answer by removing option prefix (A., B., etc) and trimming whitespace"""
    # Remove option prefix if it exists (e.g., "A. ", "B. ", etc.)
    answer = re.sub(r'^[A-D]\.\s*', '', answer.strip())
    return answer.lower().strip()

def extract_answer_choice(output):
    """Extract the choice (A, B, C, D) from model output"""
    output = output.strip()
    # Look for patterns like "A", "A.", "A)", "(A)", etc.
    match = re.search(r'([A-D])', output.upper())
    if match:
        return match.group(1)
    return None

def map_choice_to_label(choice_letter, options):
    """Map choice letter (A, B, C, D) to actual label using options list"""
    if not choice_letter or not options:
        return None
    
    choice_index = ord(choice_letter.upper()) - ord('A')
    if 0 <= choice_index < len(options):
        return options[choice_index]
    return None

def extract_predicted_label(model_output, options, correct_label):
    """
    Extract the predicted label from model output, trying multiple approaches:
    1. Extract choice letter and map to label
    2. Direct text comparison with correct label
    3. Direct text comparison with options
    """
    model_output = model_output.strip()
    
    # Approach 1: Try to extract choice letter and map to actual label
    choice_letter = extract_answer_choice(model_output)
    if choice_letter and options:
        predicted_label = map_choice_to_label(choice_letter, options)
        if predicted_label:
            return predicted_label, choice_letter
    
    # Approach 2: Direct text comparison with correct label (normalized)
    normalized_output = normalize_answer(model_output)
    normalized_correct = normalize_answer(correct_label)
    if normalized_output == normalized_correct:
        return correct_label, choice_letter
    
    # Approach 3: Check if output matches any of the options (normalized)
    if options:
        for i, option in enumerate(options):
            normalized_option = normalize_answer(option)
            if normalized_output == normalized_option:
                choice_letter = chr(ord('A') + i)
                return option, choice_letter
    
    # If nothing matches, return the normalized output
    return normalized_output, choice_letter

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
    # Convert to sets for easier comparison
    pred_set = set(sample_predictions)
    gt_set = set(sample_ground_truths)
    
    # Skip samples with empty ground truth (invalid data)
    if len(gt_set) == 0:
        return 0, 0, 0, 0, 0
    
    # Calculate metrics
    tp = len(pred_set.intersection(gt_set))  # True positives
    fp = len(pred_set - gt_set)              # False positives  
    fn = len(gt_set - pred_set)              # False negatives
    
    exact_match = 1 if pred_set == gt_set else 0
    partial_accuracy = tp / len(gt_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return exact_match, partial_accuracy, precision, recall, f1

def calculate_average_precision(predicted_labels, ground_truth_labels, all_labels):
    """
    Calculate Average Precision for a single sample.
    """
    # Create binary vectors
    y_true = np.zeros(len(all_labels))
    y_scores = np.zeros(len(all_labels))
    
    # Set ground truth labels
    for label in ground_truth_labels:
        if label in all_labels:
            idx = all_labels.index(label)
            y_true[idx] = 1
    
    # Set predicted labels (binary: 1 if predicted, 0 otherwise)
    for label in predicted_labels:
        if label in all_labels:
            idx = all_labels.index(label)
            y_scores[idx] = 1
    
    # Calculate AP
    if np.sum(y_true) == 0:  # No positive labels
        return 0.0
    
    # Simple AP calculation for binary predictions
    tp = np.sum(y_true * y_scores)
    fp = np.sum((1 - y_true) * y_scores)
    fn = np.sum(y_true * (1 - y_scores))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision * recall  # Simplified AP for binary case

def zero_shot_eval_dataset(args):
    print("Running zero-shot evaluation")
    print("All arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 50)
    
    model = load_model(args.model_name, args.data_name)
    test_data = open_data(args.data_name, args.val_path)
    
    # Data structures to track per-sample results
    sample_results = defaultdict(lambda: {
        'predictions': [],
        'ground_truths': [],
        'questions_answered': 0,
        'questions_correct': 0
    })
    
    # Get all unique labels for mAP calculation
    all_labels = set()
    for item in test_data:
        if 'original_labels' in item:
            all_labels.update(item['original_labels'])
        else:
            # Fallback for single labels
            label = item.get("correct_label", item.get("mapped_label", item.get("label", "")))
            if isinstance(label, list):
                all_labels.update(label)
            else:
                all_labels.add(label)
    
    all_labels = sorted(list(all_labels))
    print(f"Found {len(all_labels)} unique labels")
    
    # Create detailed results file
    file_name = f'results_{args.data_name}_{args.model_name}_multiclass_analysis.csv'
    csv_file = open(file_name, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow([
        "question_id", "original_sample_id", "wav", "prompt", "model_output", "predicted_label",
        "ground_truth_label", "correct_answer_letter", "predicted_choice_letter", "correct_choice_letter", "is_correct"
    ])
    
    question_level_correct = 0
    total_questions = 0
    
    print("Evaluating individual questions...")
    for item in tqdm(test_data, desc="Zero-shot Evaluation"):
        # Get sample identification
        global_item_index = item.get("original_sample_id", total_questions)
        
        # Format the question
        tqs, ans, wavs, _, _ = model.format_func(
            all_data=None, cur_item=item, num_shot=0, model_helper=model, split="test"
        )
        prompt_str = " | ".join(tqs)
        
        # Get model prediction
        model_input = model.insert_audio(tqs, ans, wavs)
        output = model.generate(model_input, max_new_tokens=32).strip()
        
        # Extract information from the item
        ground_truth = item.get("correct_label", item.get("mapped_label", item.get("label", "")))
        options = item.get("options", [])  # Get the MCQ options
        correct_answer_letter = item.get("correct_answer", "")  # e.g., "B"
        
        # Use robust prediction extraction
        predicted_label, predicted_choice = extract_predicted_label(output, options, ground_truth)
        
        # Compare predicted label with ground truth (normalized)
        normalized_predicted = normalize_answer(predicted_label) if predicted_label else ""
        normalized_ground_truth = normalize_answer(ground_truth)
        
        is_correct = (normalized_predicted == normalized_ground_truth)
        
        # Debug print for first few samples
        if total_questions < 5:
            print(f"\n--- Debug Sample {total_questions} ---")
            print(f"Model output: '{output}'")
            print(f"Options: {options}")
            print(f"Predicted label: '{predicted_label}'")
            print(f"Ground truth: '{ground_truth}'")
            print(f"Normalized predicted: '{normalized_predicted}'")
            print(f"Normalized ground truth: '{normalized_ground_truth}'")
            print(f"Is correct: {is_correct}")
            print(f"Predicted choice: {predicted_choice}")
            print(f"Correct choice: {correct_answer_letter}")
            print("---")
        
        # Always track predictions (for proper F1/precision calculation)
        sample_results[global_item_index]['predictions'].append(predicted_label if predicted_label else "UNKNOWN")
        
        # Track sample-level information
        sample_results[global_item_index]['questions_answered'] += 1
        if is_correct:
            question_level_correct += 1
            sample_results[global_item_index]['questions_correct'] += 1
        
        # Add ground truth to sample results (get original labels if available)
        original_labels = item.get("original_labels", [ground_truth])
        
        # Set ground truths for this sample (only once)
        if not sample_results[global_item_index]['ground_truths']:
            sample_results[global_item_index]['ground_truths'] = original_labels
        
        # Write detailed results
        writer.writerow([
            total_questions, global_item_index, item.get("wav", ""), prompt_str,
            output, predicted_label, ground_truth, correct_answer_letter, predicted_choice,
            correct_answer_letter, is_correct
        ])
        total_questions += 1
    
    csv_file.close()
    
    # Calculate sample-level metrics
    print("\nCalculating sample-level metrics...")
    print(f"Total unique samples found: {len(sample_results)}")
    
    # Debug: Check a few samples
    debug_count = 0
    for sample_id, results in sample_results.items():
        if debug_count < 3:
            print(f"Sample {sample_id}: predictions={results['predictions']}, ground_truths={results['ground_truths']}")
            debug_count += 1
    
    sample_exact_matches = 0
    sample_partial_accuracies = []
    sample_precisions = []
    sample_recalls = []
    sample_f1s = []
    sample_aps = []
    
    # Track invalid samples
    invalid_samples = 0
    
    # Create sample-level results file
    sample_file = f'sample_results_{args.data_name}_{args.model_name}_multiclass.csv'
    with open(sample_file, 'w', newline='', encoding='utf-8') as f:
        sample_writer = csv.writer(f)
        sample_writer.writerow([
            "sample_id", "total_questions", "correct_questions", "question_accuracy",
            "predicted_labels", "ground_truth_labels", "exact_match", "partial_accuracy",
            "precision", "recall", "f1", "average_precision"
        ])
        
        for sample_id, results in sample_results.items():
            predictions = results['predictions']
            ground_truths = results['ground_truths']
            questions_correct = results['questions_correct']
            questions_total = results['questions_answered']
            
            # Skip samples with empty ground truths (data issues)
            if not ground_truths:
                invalid_samples += 1
                continue
                
            # Strict exact match per user spec: a sample is correct only if
            # ALL questions for that sample were answered correctly.
            questions_total = results['questions_answered']
            questions_correct = results['questions_correct']
            exact_match = 1 if (questions_total > 0 and questions_correct == questions_total) else 0

            # Calculate additional metrics
            _em_tmp, partial_acc, precision, recall, f1 = calculate_sample_metrics(
                predictions, ground_truths
            )
            
            # Calculate Average Precision
            ap = calculate_average_precision(predictions, ground_truths, all_labels)
            
            # Accumulate for overall metrics
            sample_exact_matches += exact_match
            sample_partial_accuracies.append(partial_acc)
            sample_precisions.append(precision)
            sample_recalls.append(recall)
            sample_f1s.append(f1)
            sample_aps.append(ap)
            
            # Write sample results
            sample_writer.writerow([
                sample_id, questions_total, questions_correct, 
                questions_correct / questions_total if questions_total > 0 else 0,
                "|".join(predictions), "|".join(ground_truths),
                exact_match, partial_acc, precision, recall, f1, ap
            ])
    
    # Calculate overall metrics
    total_samples = len(sample_results)
    valid_samples = total_samples - invalid_samples
    
    print(f"Invalid samples (empty ground truth): {invalid_samples}")
    print(f"Valid samples for evaluation: {valid_samples}")
    
    # Question-level accuracy
    question_accuracy = question_level_correct / total_questions if total_questions > 0 else 0
    
    # Sample-level metrics
    exact_match_accuracy = sample_exact_matches / valid_samples if valid_samples > 0 else 0
    mean_partial_accuracy = np.mean(sample_partial_accuracies) if sample_partial_accuracies else 0
    mean_precision = np.mean(sample_precisions) if sample_precisions else 0
    mean_recall = np.mean(sample_recalls) if sample_recalls else 0
    mean_f1 = np.mean(sample_f1s) if sample_f1s else 0
    # mAP (mean Average Precision) is the mean of the Average Precision scores across all samples
    # It provides a single-number metric that summarizes the precision-recall performance
    mean_ap = np.mean(sample_aps) if sample_aps else 0
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Questions: {total_questions}")
    print(f"Total Samples: {total_samples}")
    print(f"Valid Samples: {valid_samples}")
    print(f"Invalid Samples: {invalid_samples}")
    print(f"Questions per Sample (avg): {total_questions/valid_samples:.2f}" if valid_samples > 0 else "N/A")
    print()
    print("QUESTION-LEVEL METRICS:")
    print(f"Question Accuracy: {question_accuracy:.4f} ({question_level_correct}/{total_questions})")
    print()
    print("SAMPLE-LEVEL METRICS:")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f} ({sample_exact_matches}/{valid_samples})")
    print(f"Partial Accuracy: {mean_partial_accuracy:.4f}")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"F1-Score: {mean_f1:.4f}")
    print(f"mAP (mean Average Precision): {mean_ap:.4f}")
    
    # Save summary
    summary_file = f'summary_{args.data_name}_{args.model_name}_multiclass.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Total Questions: {total_questions}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Valid Samples: {valid_samples}\n")
        f.write(f"Invalid Samples: {invalid_samples}\n")
        f.write(f"Questions per Sample (avg): {total_questions/valid_samples:.2f}\n\n" if valid_samples > 0 else "Questions per Sample (avg): N/A\n\n")
        f.write("QUESTION-LEVEL METRICS:\n")
        f.write(f"Question Accuracy: {question_accuracy:.4f}\n\n")
        f.write("SAMPLE-LEVEL METRICS:\n")
        f.write(f"Exact Match Accuracy: {exact_match_accuracy:.4f}\n")
        f.write(f"Partial Accuracy: {mean_partial_accuracy:.4f}\n")
        f.write(f"Precision: {mean_precision:.4f}\n")
        f.write(f"Recall: {mean_recall:.4f}\n")
        f.write(f"F1-Score: {mean_f1:.4f}\n")
        f.write(f"mAP: {mean_ap:.4f}\n")
    
    print(f"\nDetailed results saved to: {file_name}")
    print(f"Sample-level results saved to: {sample_file}")
    print(f"Summary saved to: {summary_file}")




def eval_dataset(args):
    print("Running SAV (Sparse Attention Vectors) Evaluation...")
    print("All arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 50)
    
    head_counts = [5, 10, 20, 40, 100, 300, 500, 1024]
    sav_dir = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/SAV_results/"
    activation_hook_type = "LM_ATTN"
    
    model = load_model(args.model_name, args.data_name)
    test_data = open_data(args.data_name, args.val_path)
    train_data = open_data(args.data_name, args.train_path)
    
    # Create summary file for SAV results
    summary_file = f'{sav_dir}sav_head_accuracies_{args.data_name}_{args.model_name}_{activation_hook_type}.txt'
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(summary_file)
    
    with open(summary_file, 'a') as f:  # Use append mode
        if not file_exists:
            # Only write headers if file doesn't exist
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.data_name}\n")
            f.write(f"Activation Hook Type: {activation_hook_type}\n")
            f.write("Number of Heads | Question Accuracy | Sample Exact Match | Sample F1\n")
            f.write("-" * 70 + "\n")
    
    # Data structures to track metrics across all head counts
    all_head_results = {}
    
    # Get all unique labels for consistent evaluation
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
    print(f"Found {len(all_labels)} unique labels for SAV evaluation")

    # Create config dictionary with N_TRIALS
    config = {'N_TRIALS': args.n_trials}

    for num_head in tqdm(head_counts, desc="Testing different head counts"):
        print(f"\n--- Evaluating with {num_head} heads ---")
        
        # Prepare CSV output for current head count
        file_name = f'{sav_dir}results_{args.data_name}_{args.model_name}_{num_head}_heads_{activation_hook_type}.csv'
        csv_file = open(file_name, mode='w', newline='', encoding='utf-8')  # Use 'w' to overwrite
        writer = csv.writer(csv_file)
        writer.writerow([
            "question_id", "original_sample_id", "mode", "wav", 
            "predicted", "ground_truth", "is_correct"
        ])

        # Get embeddings for current head count
        print(f"Encoding training data with {num_head} heads...")
        multimodal_embeddings = mllm_encode(model, train_data, num_head=num_head, config=config)
        
        # Data structures for this head count
        sample_results = defaultdict(lambda: {
            'predictions': [],
            'ground_truths': [],
            'questions_answered': 0,
            'questions_correct': 0
        })
        
        question_level_correct = 0
        total_questions = 0
        
        print(f"Evaluating test data with {num_head} heads...")
        for item in tqdm(test_data, desc=f"SAV evaluation with {num_head} heads"):
            # Get sample identification
            global_item_index = item.get("original_sample_id", total_questions)
            
            # Get SAV prediction
            cur_class = mllm_classify(item, model, multimodal_embeddings)
            
            # Get ground truth
            ground_truth = item.get("correct_label", item.get("mapped_label", item.get("label", "")))
            
            # Check if prediction is correct
            is_correct = (cur_class == ground_truth)
            
            # Always track predictions (for proper F1/precision calculation)
                sample_results[global_item_index]['predictions'].append(cur_class)
            
            # Track sample-level information
            sample_results[global_item_index]['questions_answered'] += 1
            if is_correct:
                question_level_correct += 1
                sample_results[global_item_index]['questions_correct'] += 1
            
            # Set ground truths for this sample (only once)
            if not sample_results[global_item_index]['ground_truths']:
                original_labels = item.get("original_labels", [ground_truth])
                sample_results[global_item_index]['ground_truths'] = original_labels
            
            # Write detailed results
            writer.writerow([
                total_questions, global_item_index, "Sparse Attention Vectors",
                item.get("wav", ""), cur_class, ground_truth, is_correct
            ])
            total_questions += 1
        
        csv_file.close()
        
        # Calculate metrics for this head count
        question_accuracy = question_level_correct / total_questions if total_questions > 0 else 0
        
        # Calculate sample-level metrics
        sample_exact_matches = 0
        sample_f1s = []
        valid_samples = 0
        
        for sample_id, results in sample_results.items():
            predictions = results['predictions']
            ground_truths = results['ground_truths']
            
            # Skip samples with empty ground truths
            if not ground_truths:
                continue
                
            valid_samples += 1
            
            # Calculate metrics for this sample
            exact_match, _, _, _, f1 = calculate_sample_metrics(predictions, ground_truths)
            sample_exact_matches += exact_match
            sample_f1s.append(f1)
        
        # Calculate overall sample metrics
        sample_exact_match_accuracy = sample_exact_matches / valid_samples if valid_samples > 0 else 0
        mean_sample_f1 = np.mean(sample_f1s) if sample_f1s else 0
        
        # Store results for this head count
        all_head_results[num_head] = {
            'question_accuracy': question_accuracy,
            'sample_exact_match': sample_exact_match_accuracy,
            'sample_f1': mean_sample_f1,
            'total_questions': total_questions,
            'valid_samples': valid_samples,
            'question_correct': question_level_correct,
            'sample_exact_matches': sample_exact_matches
        }
        
        print(f"Results for {num_head} heads:")
        print(f"  Question Accuracy: {question_accuracy:.4f} ({question_level_correct}/{total_questions})")
        print(f"  Sample Exact Match: {sample_exact_match_accuracy:.4f} ({sample_exact_matches}/{valid_samples})")
        print(f"  Sample F1: {mean_sample_f1:.4f}")
        
        # Write to summary file
        with open(summary_file, 'a') as f:
            f.write(f"{num_head:3d} heads | {question_accuracy:.4f} | {sample_exact_match_accuracy:.4f} | {mean_sample_f1:.4f}\n")
    
    # Print final summary
    print("\n" + "="*80)
    print("SAV EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Heads':<6} {'Q.Acc':<7} {'S.EM':<7} {'S.F1':<7} {'Questions':<10} {'Samples':<8}")
    print("-" * 80)
    
    for num_head in head_counts:
        if num_head in all_head_results:
            r = all_head_results[num_head]
            print(f"{num_head:<6} {r['question_accuracy']:<7.4f} {r['sample_exact_match']:<7.4f} "
                  f"{r['sample_f1']:<7.4f} {r['total_questions']:<10} {r['valid_samples']:<8}")
    
    print(f"\nResults saved to: {summary_file}")
    print(f"Individual CSV files saved to: {sav_dir}")
    print("\nLegend:")
    print("  Q.Acc = Question-level Accuracy")
    print("  S.EM = Sample-level Exact Match")
    print("  S.F1 = Sample-level F1 Score")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials for averaging activations (default: 1)")
    parser.add_argument("--zero_shot", action="store_true", help="Whether to run zero-shot evaluation")
    args = parser.parse_args()
    print("All arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 50)
    if args.zero_shot:
        print("Running Zero-shot Evaluation...")
        zero_shot_eval_dataset(args) 
    else:
        print("Running SAV (Sparse Attention Vectors) Evaluation...")
        eval_dataset(args)