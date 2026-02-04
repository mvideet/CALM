"""
Prune AudioSet dataset to N-shot per class.

Usage:
    python prune_audioset.py --input INPUT_JSON --output OUTPUT_JSON --shots 20
"""
import json
import argparse
from collections import defaultdict, Counter


def prune_audioset(input_file, output_file, shots_per_class=20):
    """
    Prune the AudioSet dataset to have at most N examples per class.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        shots_per_class: Maximum examples per class
    """
    print("Loading dataset...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Original dataset size: {len(data)} examples")

    class_counts = Counter(item['mapped_label'] for item in data)
    print(f"Found {len(class_counts)} unique classes")

    class_examples = defaultdict(list)
    for item in data:
        class_examples[item['mapped_label']].append(item)

    pruned_data = []
    for class_name, examples in class_examples.items():
        num_to_keep = min(len(examples), shots_per_class)
        pruned_data.extend(examples[:num_to_keep])

    print(f"Pruned dataset size: {len(pruned_data)} examples")

    with open(output_file, 'w') as f:
        json.dump(pruned_data, f, indent=2)

    print(f"Saved pruned dataset to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune AudioSet to N-shot")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--shots", type=int, default=20, help="Shots per class")
    args = parser.parse_args()
    
    prune_audioset(args.input, args.output, args.shots)
