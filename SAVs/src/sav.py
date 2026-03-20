"""
SAV: Sparse Attention Vectors baseline.

Selects top-k attention heads by per-head classification accuracy on the
training set, then classifies test queries via majority vote (each selected
head casts one vote for the class with highest cosine similarity).

Key functions:
- sav_build_full_cache: One-time expensive computation (prototypes, head
  scores, test activations)
- sav_evaluate_from_cache: Cheap evaluation for any num_heads using cached data
- sav_print_top_heads: Display selected heads
"""
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .calm import get_class_activations, get_last_mean_head_activations


def _extract_head_activations(item, model_helper, all_heads):
    """Extract per-head activations for a single item.

    Returns (K, D) tensor or None on failure.
    """
    try:
        mean_act = get_last_mean_head_activations(
            None, item, model_helper, N_TRIALS=1, shot=0,
        )
        return torch.stack([mean_act[layer, head, -1] for layer, head, *_ in all_heads])
    except Exception as e:
        warnings.warn(f"Failed to extract activations: {e}")
        return None


def sav_build_full_cache(train_data, test_data, model_helper):
    """One-time expensive computation: prototypes, head scores, test activations.

    After this, ``sav_evaluate_from_cache`` can sweep num_heads for free.

    Returns dict with keys: all_prototypes, head_scores, train_total,
    all_heads, str_to_int, int_to_str, test_activations, test_labels.
    """
    all_heads = list(model_helper.all_heads)
    K = len(all_heads)

    print("Step 1: Building class prototypes (all heads)...")
    all_prototypes, str_to_int, int_to_str = get_class_activations(
        train_data, model_helper, all_heads, n_trials=1,
    )

    print("Step 2: Scoring heads by train-set accuracy...")
    success_count = [0] * K
    total_count = 0

    for item in tqdm(train_data, desc="Scoring heads"):
        query_act = _extract_head_activations(item, model_helper, all_heads)
        if query_act is None:
            continue
        label_str = item.get("mapped_label", item.get("label", ""))
        label_key = label_str.lower() if isinstance(label_str, str) else label_str
        if label_key not in str_to_int:
            continue
        int_label = str_to_int[label_key]

        for h_idx in range(K):
            sims = F.cosine_similarity(
                all_prototypes[:, h_idx, :], query_act[h_idx].unsqueeze(0), dim=-1,
            )
            if sims.argmax().item() == int_label:
                success_count[h_idx] += 1
        total_count += 1

    print("Step 3: Caching test-set activations...")
    test_activations, test_labels = [], []

    for item in tqdm(test_data, desc="Test activations"):
        query_act = _extract_head_activations(item, model_helper, all_heads)
        if query_act is None:
            continue
        gt = item.get("mapped_label", item.get("label", ""))
        gt_key = gt.lower() if isinstance(gt, str) else gt
        if gt_key not in str_to_int:
            continue
        test_activations.append(query_act)
        test_labels.append(str_to_int[gt_key])

    print(f"  Cached {len(test_activations)} test items, {total_count} train items scored")

    return {
        "all_prototypes": all_prototypes,
        "head_scores": success_count,
        "train_total": total_count,
        "all_heads": all_heads,
        "str_to_int": str_to_int,
        "int_to_str": int_to_str,
        "test_activations": test_activations,
        "test_labels": test_labels,
    }


def sav_evaluate_from_cache(cache, num_heads):
    """Evaluate with *num_heads* top heads using cached data (no forward passes).

    Returns (accuracy, predictions_str, ground_truths_str).
    """
    all_prototypes = cache["all_prototypes"]
    int_to_str = cache["int_to_str"]
    test_activations = cache["test_activations"]
    test_labels = cache["test_labels"]

    arr = np.array(cache["head_scores"])
    topk_indices = np.argsort(arr)[-num_heads:][::-1].tolist()
    selected_prototypes = all_prototypes[:, topk_indices, :]

    predictions, ground_truths = [], []
    for query_act, gt_idx in zip(test_activations, test_labels):
        selected_act = query_act[topk_indices]
        head_votes = []
        for h_idx in range(num_heads):
            sims = F.cosine_similarity(
                selected_prototypes[:, h_idx, :], selected_act[h_idx].unsqueeze(0), dim=-1,
            )
            head_votes.append(sims.argmax().item())

        winning_idx = Counter(head_votes).most_common(1)[0][0]
        predictions.append(int_to_str.get(winning_idx, "UNKNOWN"))
        ground_truths.append(int_to_str.get(gt_idx, "UNKNOWN"))

    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / max(len(predictions), 1)
    return accuracy, predictions, ground_truths


def sav_print_top_heads(cache, num_heads):
    """Print the top-k heads selected for a given num_heads."""
    arr = np.array(cache["head_scores"])
    topk_indices = np.argsort(arr)[-num_heads:][::-1].tolist()
    all_heads = cache["all_heads"]
    total = cache["train_total"]

    print(f"\nSelected top {num_heads} heads (out of {len(all_heads)}):")
    for rank, idx in enumerate(topk_indices[:10]):
        acc = cache["head_scores"][idx] / max(total, 1)
        layer, head, *_ = all_heads[idx]
        print(f"  #{rank+1}: layer={layer}, head={head}, "
              f"acc={cache['head_scores'][idx]}/{total} ({acc:.3f})")
    if num_heads > 10:
        print(f"  ... ({num_heads - 10} more)")
