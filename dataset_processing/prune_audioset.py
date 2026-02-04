import json
from collections import defaultdict, Counter

def prune_audioset_to_20shot(input_file, output_file, shots_per_class=20):
    """
    Prune the AudioSet dataset to have at most 'shots_per_class' examples per class.
    If a class has fewer than 'shots_per_class' examples, keep all of them.
    """
    print("Loading dataset...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Original dataset size: {len(data)} examples")

    # Count occurrences of each class
    class_counts = Counter()
    for item in data:
        class_counts[item['mapped_label']] += 1

    print(f"Found {len(class_counts)} unique classes")
    print(f"Most common classes: {class_counts.most_common(10)}")

    # Group examples by class
    class_examples = defaultdict(list)
    for item in data:
        class_examples[item['mapped_label']].append(item)

    # Prune to 20-shot (or maximum available)
    pruned_data = []
    for class_name, examples in class_examples.items():
        # Keep at most 'shots_per_class' examples, or all if fewer available
        num_to_keep = min(len(examples), shots_per_class)
        pruned_data.extend(examples[:num_to_keep])

        print(f"Class '{class_name}': {len(examples)} -> {num_to_keep} examples")

    print(f"\nPruned dataset size: {len(pruned_data)} examples")

    # Save pruned dataset
    with open(output_file, 'w') as f:
        json.dump(pruned_data, f, indent=2)

    print(f"Saved pruned dataset to {output_file}")

if __name__ == "__main__":
    input_file = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_train_individual_mcqs.json"
    output_file = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/data/audioset/audioset_20shot_train_individual_mcqs.json"

    prune_audioset_to_20shot(input_file, output_file)