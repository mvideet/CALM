#!/usr/bin/env python3
"""
Create a proper 40-shot ESC-50 dataset from the actual ESC-50 data files.
"""

import json
import random
import pandas as pd
from collections import defaultdict

def load_esc_data(file_path):
    """Load ESC-50 data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['data']

def load_label_mapping(csv_path):
    """Load label mapping from CSV."""
    df = pd.read_csv(csv_path)
    # Create mapping from mid to display_name
    mapping = dict(zip(df['mid'], df['display_name']))
    return mapping

def fix_audio_path(audio_path, audio_root_dir):
    """Fix audio file paths by replacing '.' with the actual root directory."""
    if audio_path.startswith('./'):
        # Replace './' with the audio root directory
        fixed_path = audio_path.replace('./', audio_root_dir + '/')
    elif audio_path.startswith('.'):
        # Replace '.' with the audio root directory
        fixed_path = audio_path.replace('.', audio_root_dir, 1)
    else:
        fixed_path = audio_path
    
    return fixed_path

def generate_multiple_choice_question(correct_label, all_labels, label_mapping, num_options=4):
    """Generate a multiple choice question."""
    # Get display name for correct label
    correct_display = label_mapping.get(correct_label, correct_label)
    
    # Get all other labels as potential wrong answers
    wrong_labels = [label for label in all_labels if label != correct_label]
    
    # Randomly sample wrong answers
    if len(wrong_labels) >= num_options - 1:
        sampled_wrong = random.sample(wrong_labels, num_options - 1)
    else:
        # If not enough wrong labels, repeat some
        sampled_wrong = random.choices(wrong_labels, k=num_options - 1)
    
    # Convert to display names
    wrong_displays = [label_mapping.get(label, label) for label in sampled_wrong]
    
    # Create options list with correct answer
    all_options = [correct_display] + wrong_displays
    random.shuffle(all_options)  # Shuffle to randomize position
    
    # Find the position of correct answer
    correct_answer_idx = all_options.index(correct_display)
    correct_answer_letter = chr(65 + correct_answer_idx)  # A, B, C, D...
    
    # Create question text
    options_text = " ".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(all_options)])
    question_text = f"What caption does the given audio belong to?\n{options_text}"
    
    return question_text, correct_answer_letter, all_options

def create_40shot_dataset():
    """Create 40-shot ESC-50 dataset."""
    
    # File paths
    train_file_1 = "/data/sls/scratch/sbhati/SSM/VMamba-ast/egs/esc50/data/datafiles/esc_train_data_1.json"
    train_file_2 = "/data/sls/scratch/sbhati/SSM/VMamba-ast/egs/esc50/data/datafiles/esc_train_data_2.json"
    eval_file = "/data/sls/scratch/sbhati/SSM/VMamba-ast/egs/esc50/data/datafiles/esc_eval_data_1.json"
    label_csv = "/data/sls/scratch/sbhati/SSM/VMamba-ast/egs/esc50/data/esc_class_labels_indices.csv"
    audio_root_dir = "/data/sls/scratch/sbhati/SSM/VMamba-ast/egs/esc50/data/ESC-50-master/audio_16k"
    
    # Load data
    print("Loading data files...")
    train_data_1 = load_esc_data(train_file_1)
    train_data_2 = load_esc_data(train_file_2)
    eval_data = load_esc_data(eval_file)
    
    print(f"Train data 1: {len(train_data_1)} samples")
    print(f"Train data 2: {len(train_data_2)} samples")
    print(f"Eval data: {len(eval_data)} samples")
    
    # Combine all training data
    all_train_data = train_data_1 + train_data_2
    print(f"Combined train data: {len(all_train_data)} samples")
    
    # Load label mapping
    label_mapping = load_label_mapping(label_csv)
    print(f"Loaded {len(label_mapping)} label mappings")
    
    # Group training data by class
    class_data = defaultdict(list)
    for item in all_train_data:
        class_data[item['labels']].append(item)
    
    print(f"Found {len(class_data)} classes in training data")
    for label, items in class_data.items():
        display_name = label_mapping.get(label, label)
        print(f"  {display_name} ({label}): {len(items)} samples")
    
    # Create 40-shot training dataset
    print(f"\nCreating 40-shot training dataset...")
    shot_dataset = []
    all_labels = list(class_data.keys())
    
    for class_label, items in class_data.items():
        # Shuffle items for this class
        random.shuffle(items)
        
        # Take up to 40 shots per class
        n_samples = min(40, len(items))
        selected_items = items[:n_samples]
        
        for item in selected_items:
            # Fix audio path
            fixed_audio_path = fix_audio_path(item['wav'], audio_root_dir)
            
            # Generate MCQ
            question_text, correct_answer, options = generate_multiple_choice_question(
                item['labels'], all_labels, label_mapping
            )
            
            # Create MCQ item
            mcq_item = {
                "wav": fixed_audio_path,
                "question": question_text,
                "answer": correct_answer,
                "label": item['labels'],
                "mapped_label": label_mapping.get(item['labels'], item['labels']),
                "options": options
            }
            
            shot_dataset.append(mcq_item)
        
        display_name = label_mapping.get(class_label, class_label)
        print(f"  Selected {n_samples} samples for class '{display_name}'")
    
    # Shuffle the final dataset
    random.shuffle(shot_dataset)
    
    # Save training dataset
    train_output = "esc50_40shot_train.json"
    with open(train_output, 'w') as f:
        json.dump(shot_dataset, f, indent=2)
    
    print(f"\nCreated 40-shot training dataset with {len(shot_dataset)} total samples")
    print(f"Saved to: {train_output}")
    
    # Create test dataset from eval data
    print(f"\nCreating test dataset from eval data...")
    test_dataset = []
    
    for item in eval_data:
        # Fix audio path
        fixed_audio_path = fix_audio_path(item['wav'], audio_root_dir)
        
        # Generate MCQ
        question_text, correct_answer, options = generate_multiple_choice_question(
            item['labels'], all_labels, label_mapping
        )
        
        # Create MCQ item
        mcq_item = {
            "wav": fixed_audio_path,
            "question": question_text,
            "answer": correct_answer,
            "label": item['labels'],
            "mapped_label": label_mapping.get(item['labels'], item['labels']),
            "options": options
        }
        
        test_dataset.append(mcq_item)
    
    # Save test dataset
    test_output = "esc50_test.json"
    with open(test_output, 'w') as f:
        json.dump(test_dataset, f, indent=2)
    
    print(f"Created test dataset with {len(test_dataset)} total samples")
    print(f"Saved to: {test_output}")
    
    # Print final statistics
    train_class_counts = defaultdict(int)
    for item in shot_dataset:
        train_class_counts[item['mapped_label']] += 1
    
    print(f"\nFinal training class distribution:")
    for label, count in sorted(train_class_counts.items()):
        print(f"  {label}: {count} samples")
    
    test_class_counts = defaultdict(int)
    for item in test_dataset:
        test_class_counts[item['mapped_label']] += 1
    
    print(f"\nTest class distribution:")
    for label, count in sorted(test_class_counts.items()):
        print(f"  {label}: {count} samples")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    create_40shot_dataset() 