"""
Pseudolabel generation for audio classification.

Uses model predictions with majority voting across multiple trials
to generate high-confidence pseudolabels for unlabeled data.
"""
import json
import os
import re
from collections import Counter

from tqdm import tqdm

from .calm import load_model
from .preprocess import open_data


def normalize_answer(answer: str) -> str:
    """Normalize answer by removing option prefix and trimming whitespace."""
    if not answer or not isinstance(answer, str):
        return ""

    answer = answer.strip()

    # Remove option prefix (A., B., etc.)
    answer = re.sub(r'^\s*[(]?[A-D][.)]\s*', '', answer, count=1)

    # Remove parentheses at start/end
    answer = re.sub(r'^\s*[(]', '', answer)
    answer = re.sub(r'[)]\s*$', '', answer)

    normalized = answer.lower().strip()

    if len(normalized) < 1:
        return ""

    return normalized


def extract_options_from_question(question_text: str) -> list:
    """Extract options from question text if options field is missing."""
    if not question_text:
        return []

    lines = [line.strip() for line in question_text.split('\n') if line.strip()]

    options = []
    for line in lines:
        if '?' in line:
            continue
        if len(line) > 1:
            options.append(line)

    return options


def generate_pseudolabels(args):
    """
    Generate pseudolabels using majority voting across multiple inference trials.

    Args:
        args: Namespace with:
            - model_name: Model identifier
            - data_name: Dataset name
            - train_path: Path to data to pseudolabel
            - n_trials: Number of inference trials per sample
            - min_confidence: Minimum confidence threshold
            - output_dir: Output directory (optional)

    Returns:
        Number of successfully pseudolabeled samples
    """
    # Initialize model and data
    model = load_model(args.model_name, args.data_name)
    train_data = open_data(args.data_name, args.train_path) or []

    n_trials = getattr(args, 'n_trials', 8)
    min_confidence = getattr(args, 'min_confidence', 0.5)
    output_dir = getattr(args, 'output_dir', '.')

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"pseudolabeled_{args.data_name}_{args.model_name}_{n_trials}trials.json"
    )

    pseudolabeled_data = []
    successful_labels = 0

    print(f"Generating pseudolabels for {len(train_data)} samples...")
    print(f"  Trials per sample: {n_trials}")
    print(f"  Minimum confidence: {min_confidence}")
    print()

    for item in tqdm(train_data, desc="Generating pseudolabels"):
        try:
            # Get model predictions
            result = model.format_func(
                all_data=None,
                cur_item=item,
                num_shot=0,
                model_helper=model,
                split="test"
            )

            if len(result) == 5:
                tqs, ans, wavs, _, _ = result
            else:
                tqs, ans, _, wavs, _, _ = result

            # Extract options
            options = item.get("options", [])
            if not options:
                options = extract_options_from_question(item.get("question", ""))

            if not options:
                continue

            # Run multiple trials
            trial_predictions = []

            for trial in range(n_trials):
                try:
                    model_input = model.insert_audio(tqs, ans, wavs)
                    output = model.generate(model_input, max_new_tokens=32)

                    if output is None:
                        trial_predictions.append("")
                        continue

                    output = output.strip()
                    normalized_output = normalize_answer(output)

                    # Handle long outputs (explanations)
                    if not normalized_output or len(normalized_output) > 50:
                        raw_lower = output.lower()
                        normalized_options = [normalize_answer(opt) for opt in options]

                        best_match = None
                        best_match_len = 0

                        for opt_norm in normalized_options:
                            if opt_norm and opt_norm in raw_lower:
                                if len(opt_norm) > best_match_len:
                                    best_match = opt_norm
                                    best_match_len = len(opt_norm)

                        if best_match:
                            normalized_output = best_match

                    if normalized_output:
                        trial_predictions.append(normalized_output)
                    else:
                        trial_predictions.append("")

                except Exception:
                    trial_predictions.append("")

            # Majority voting
            non_empty_predictions = [p for p in trial_predictions if p]

            if not non_empty_predictions:
                continue

            prediction_counts = Counter(non_empty_predictions)
            most_common_prediction, vote_count = prediction_counts.most_common(1)[0]
            confidence = vote_count / len(non_empty_predictions)

            if confidence < min_confidence:
                continue

            # Match to options
            normalized_options = [normalize_answer(opt) for opt in options]
            matched_option_idx = None

            if most_common_prediction in normalized_options:
                matched_option_idx = normalized_options.index(most_common_prediction)
            else:
                for i, opt_norm in enumerate(normalized_options):
                    if opt_norm and most_common_prediction:
                        if opt_norm in most_common_prediction or most_common_prediction in opt_norm:
                            if matched_option_idx is None or len(opt_norm) > len(normalized_options[matched_option_idx]):
                                matched_option_idx = i

            if matched_option_idx is None:
                continue

            matched_answer = options[matched_option_idx]

            # Build pseudolabel item
            pseudolabel_item = {
                "wav": item.get("wav", ""),
                "question": item.get("question", ""),
                "answer": matched_answer,
                "label": item.get("label", matched_answer),
                "mapped_label": matched_answer,
                "options": options,
                "pseudo_label_confidence": confidence,
                "correct_answer": matched_answer,
                "correct_answer_index": matched_option_idx,
            }

            # Preserve optional fields
            for field in ["original_sample_id", "original_labels", "label_index_in_original"]:
                if field in item:
                    pseudolabel_item[field] = item[field]

            pseudolabeled_data.append(pseudolabel_item)
            successful_labels += 1

        except Exception as e:
            continue

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pseudolabeled_data, f, ensure_ascii=False, indent=2)

    # Statistics
    if pseudolabeled_data:
        confidences = [item["pseudo_label_confidence"] for item in pseudolabeled_data]
        avg_confidence = sum(confidences) / len(confidences)
        high_confidence_count = sum(1 for c in confidences if c >= 0.8)
        label_counts = Counter([item.get("mapped_label", "") for item in pseudolabeled_data])
    else:
        avg_confidence = 0
        high_confidence_count = 0
        label_counts = Counter()

    print()
    print("=" * 60)
    print("PSEUDOLABEL GENERATION COMPLETE")
    print("=" * 60)
    print(f"Original samples: {len(train_data)}")
    print(f"Successfully labeled: {successful_labels}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"High confidence (>=80%): {high_confidence_count}")
    print()
    print("Class distribution:")
    for label, count in label_counts.most_common():
        pct = (count / len(pseudolabeled_data) * 100) if pseudolabeled_data else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()
    print(f"Output saved to: {output_file}")

    return len(pseudolabeled_data)
