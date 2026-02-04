"""
Prepare ESC-50 dataset for audio classification.

This script creates N-shot train and test splits with multiple choice questions.

Usage:
    python prepare_esc50.py --data_dir DATA_DIR --label_csv LABEL_CSV --output_dir OUTPUT_DIR --shots 40
"""
import json
import argparse
import random
import pandas as pd
from collections import defaultdict


def load_esc_data(file_path):
    """Load ESC-50 data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get('data', data)


def generate_mcq(correct_label, all_labels, label_mapping, num_options=4):
    """Generate a multiple choice question."""
    correct_display = label_mapping.get(correct_label, correct_label)
    wrong_labels = [l for l in all_labels if l != correct_label]
    sampled_wrong = random.sample(wrong_labels, min(num_options - 1, len(wrong_labels)))
    wrong_displays = [label_mapping.get(l, l) for l in sampled_wrong]
    
    all_options = [correct_display] + wrong_displays
    random.shuffle(all_options)
    
    correct_idx = all_options.index(correct_display)
    correct_letter = chr(65 + correct_idx)
    
    options_text = " ".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(all_options)])
    question = f"What sound is this?\n{options_text}"
    
    return question, correct_letter, all_options


def prepare_esc50(train_files, eval_file, label_csv, audio_dir, output_dir, shots=40):
    """
    Create N-shot ESC-50 dataset.
    
    Args:
        train_files: List of training JSON files
        eval_file: Evaluation JSON file
        label_csv: Label mapping CSV
        audio_dir: Audio directory
        output_dir: Output directory
        shots: Shots per class
    """
    # Load data
    train_data = []
    for f in train_files:
        train_data.extend(load_esc_data(f))
    eval_data = load_esc_data(eval_file)
    
    # Load label mapping
    df = pd.read_csv(label_csv)
    label_mapping = dict(zip(df['mid'], df['display_name']))
    
    # Group by class
    class_data = defaultdict(list)
    for item in train_data:
        class_data[item['labels']].append(item)
    
    all_labels = list(class_data.keys())
    
    # Create train dataset
    train_dataset = []
    for class_label, items in class_data.items():
        random.shuffle(items)
        for item in items[:shots]:
            audio_path = item['wav']
            if audio_path.startswith('./'):
                audio_path = audio_path.replace('./', audio_dir + '/')
            
            question, answer, options = generate_mcq(item['labels'], all_labels, label_mapping)
            
            train_dataset.append({
                "wav": audio_path,
                "question": question,
                "answer": answer,
                "mapped_label": label_mapping.get(item['labels'], item['labels']),
                "options": options
            })
    
    random.shuffle(train_dataset)
    
    # Create test dataset
    test_dataset = []
    for item in eval_data:
        audio_path = item['wav']
        if audio_path.startswith('./'):
            audio_path = audio_path.replace('./', audio_dir + '/')
        
        question, answer, options = generate_mcq(item['labels'], all_labels, label_mapping)
        
        test_dataset.append({
            "wav": audio_path,
            "question": question,
            "answer": answer,
            "mapped_label": label_mapping.get(item['labels'], item['labels']),
            "options": options
        })
    
    # Save
    with open(f"{output_dir}/esc50_{shots}shot_train.json", 'w') as f:
        json.dump(train_dataset, f, indent=2)
    
    with open(f"{output_dir}/esc50_test.json", 'w') as f:
        json.dump(test_dataset, f, indent=2)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ESC-50 dataset")
    parser.add_argument("--train_files", nargs='+', required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shots", type=int, default=40)
    args = parser.parse_args()
    
    random.seed(42)
    prepare_esc50(
        args.train_files, args.eval_file, args.label_csv,
        args.audio_dir, args.output_dir, args.shots
    )
