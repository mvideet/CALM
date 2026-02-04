"""
Simple Soft Voting Utilities for Prototype-based Classification

This module provides simpler alternatives to PRWE that often achieve
comparable performance with significantly less complexity.

Methods (in order of simplicity):
1. concat_prototype    - Single cosine similarity on concatenated vectors
2. uniform_soft_vote   - Average posteriors across all heads equally
3. confidence_weighted - Weight heads by their prediction confidence
4. topk_soft_vote      - Uniform soft vote among top-K validated heads
"""

from baukit import TraceDict
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


# =============================================================================
# Model Loading (reuse from prwe_utils)
# =============================================================================

def load_model(model_name, cur_dataset, lora_path=None):
    """
    Minimal loader for experiments. Supports:
    - qwen2-audio-instruct
    - qwen2.5_omni
    """
    if model_name == "qwen2-audio-instruct":
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device},
        )
        model.tie_weights()
        model.eval()
        model.requires_grad_(False)
        try:
            from ..model import Qwen2AudioHelper
        except ImportError:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from src.model import Qwen2AudioHelper
        model_helper = Qwen2AudioHelper(model, processor, cur_dataset)
    elif model_name == "qwen2.5_omni":
        from transformers import (
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto"
        )
        model.eval()
        model.requires_grad_(False)
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        try:
            from ..model import Qwen2OmniHelper
        except ImportError:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from src.model import Qwen2OmniHelper
        model_helper = Qwen2OmniHelper(model, processor, cur_dataset)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")
    return model_helper


# =============================================================================
# Activation Extraction (simplified from prwe_utils)
# =============================================================================

def gather_activations(inputs, model_helper):
    """Run forward pass and capture attention activations."""
    with TraceDict(
        model_helper.model,
        layers=model_helper.model_config['attn_hook_names'],
        retain_input=True,
        retain_output=True,
    ) as td:
        result = model_helper.forward(inputs)
    return td, result


def split_by_head(activations, model_config):
    """Reshape concatenated multi-head outputs into (batch, seq, n_heads, head_dim)."""
    if activations.dim() == 2:
        activations = activations.unsqueeze(1)
    new_shape = activations.size()[:-1] + (
        model_config['n_heads'],
        model_config['resid_dim'] // model_config['n_heads'],
    )
    return activations.view(*new_shape).to("cuda")


def get_head_activations(dataset, item, model_helper, n_trials=1, last_n_tokens=1):
    """
    Extract per-head activations for a single item.
    Returns: (n_layers, n_heads, head_dim) tensor
    """
    running_sum = None
    successful = 0
    
    for _ in range(n_trials):
        torch.cuda.empty_cache()
        sample = item if isinstance(item, dict) else item[0]
        
        try:
            tqs, ans, wavs, _, _ = model_helper.format_func(
                all_data=dataset, cur_item=sample, num_shot=0, 
                model_helper=model_helper, split="train"
            )
            inputs = model_helper.insert_audio(tqs, ans, wavs)
            if inputs is None:
                continue
            
            td, _ = gather_activations(inputs, model_helper)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to process item: {e}")
            continue
        
        layer_tensors = []
        for name in model_helper.model_config["attn_hook_names"]:
            layer_tensor = split_by_head(td[name].input, model_helper.model_config)
            layer_tensors.append(layer_tensor)
            del td[name]
        
        stacked = torch.stack(layer_tensors, dim=0)  # (n_layers, batch, seq, n_heads, head_dim)
        
        # Get last token(s) activation
        if last_n_tokens > 1:
            cur_act = stacked[:, -1, -last_n_tokens:, :, :].mean(dim=1)  # (n_layers, n_heads, head_dim)
        else:
            cur_act = stacked[:, -1, -1, :, :]  # (n_layers, n_heads, head_dim)
        
        cur_act = cur_act.permute(0, 1, 2)  # Keep as (n_layers, n_heads, head_dim)
        
        if running_sum is None:
            running_sum = cur_act
        else:
            running_sum += cur_act
        successful += 1
        
        del stacked
        torch.cuda.empty_cache()
    
    if successful == 0:
        raise RuntimeError("All trials failed")
    
    return running_sum / successful


# =============================================================================
# Prototype Building
# =============================================================================

def build_prototypes(dataset, model_helper, heads=None, n_trials=20, last_n_tokens=1):
    """
    Build class prototypes from dataset.
    
    Returns:
        prototypes: (C, K, D) tensor - C classes, K heads, D dimensions
        str_to_int: dict mapping label strings to indices
        int_to_str: dict mapping indices to label strings
    """
    str_to_int = {}
    int_to_str = {}
    class_sums = {}
    class_counts = {}
    
    # If no specific heads provided, we'll use all heads (determined by first sample)
    all_heads = heads if heads is not None else None
    
    for item in tqdm(dataset, desc="Building prototypes"):
        try:
            activations = get_head_activations(
                dataset, item, model_helper, 
                n_trials=n_trials, last_n_tokens=last_n_tokens
            )
            
            # activations shape: (n_layers, n_heads, head_dim)
            # Flatten to (n_layers * n_heads, head_dim) = (K, D)
            n_layers, n_heads, head_dim = activations.shape
            
            if all_heads is None:
                # Use all layer-head combinations
                K = n_layers * n_heads
                flat_act = activations.reshape(K, head_dim)
            else:
                # Use specified heads
                head_acts = []
                for (layer_idx, head_idx) in all_heads:
                    head_acts.append(activations[layer_idx, head_idx])
                flat_act = torch.stack(head_acts, dim=0)  # (K, D)
            
            label = item['mapped_label']
            
            if label not in str_to_int:
                idx = len(str_to_int)
                str_to_int[label] = idx
                int_to_str[idx] = label
                class_sums[label] = flat_act.clone()
                class_counts[label] = 1
            else:
                class_sums[label] += flat_act
                class_counts[label] += 1
                
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to process item: {e}")
            continue
    
    # Average to get prototypes
    C = len(str_to_int)
    K = flat_act.shape[0]
    D = flat_act.shape[1]
    
    prototypes = torch.zeros(C, K, D, device='cuda', dtype=flat_act.dtype)
    for label, idx in str_to_int.items():
        prototypes[idx] = class_sums[label] / class_counts[label]
    
    return prototypes, str_to_int, int_to_str


def get_query_activation(dataset, item, model_helper, heads=None, n_trials=1, last_n_tokens=1):
    """
    Get activation for a query item.
    Returns: (K, D) tensor
    
    Args:
        dataset: dataset for few-shot sampling
        item: query item
        model_helper: model helper object
        heads: optional list of (layer, head) tuples to use
        n_trials: number of trials to average over (default: 1 for inference)
        last_n_tokens: number of last tokens to average
    """
    activations = get_head_activations(
        dataset, item, model_helper, 
        n_trials=n_trials, last_n_tokens=last_n_tokens
    )
    
    n_layers, n_heads, head_dim = activations.shape
    
    if heads is None:
        return activations.reshape(n_layers * n_heads, head_dim)
    else:
        head_acts = []
        for (layer_idx, head_idx) in heads:
            head_acts.append(activations[layer_idx, head_idx])
        return torch.stack(head_acts, dim=0)


# =============================================================================
# Classification Methods
# =============================================================================

def classify_concat_prototype(query_act, prototypes):
    """
    Method 1: Concatenated Prototype (Simplest)
    
    Flatten all heads into a single vector and do one cosine similarity.
    
    Args:
        query_act: (K, D) query activations
        prototypes: (C, K, D) class prototypes
    
    Returns:
        pred_idx: predicted class index
        scores: (C,) similarity scores
    """
    # Flatten
    query_flat = query_act.reshape(-1)  # (K*D,)
    proto_flat = prototypes.reshape(prototypes.shape[0], -1)  # (C, K*D)
    
    # L2 normalize
    query_flat = F.normalize(query_flat, dim=-1)
    proto_flat = F.normalize(proto_flat, dim=-1)
    
    # Single cosine similarity
    scores = F.cosine_similarity(query_flat.unsqueeze(0), proto_flat, dim=-1)  # (C,)
    
    return scores.argmax().item(), scores


def classify_uniform_soft_vote(query_act, prototypes, tau=0.07):
    """
    Method 2: Uniform Soft Voting
    
    Each head produces a soft posterior, then average across heads.
    
    Args:
        query_act: (K, D) query activations
        prototypes: (C, K, D) class prototypes
        tau: temperature for softmax
    
    Returns:
        pred_idx: predicted class index
        avg_posterior: (C,) averaged posterior probabilities
    """
    # L2 normalize
    q = F.normalize(query_act, dim=-1)  # (K, D)
    p = F.normalize(prototypes, dim=-1)  # (C, K, D)
    
    # Per-head similarities: for each head k, compute similarity to all class prototypes
    # sims[k, c] = cosine_sim(q[k], p[c, k])
    sims = torch.einsum('kd,ckd->kc', q, p)  # (K, C)
    
    # Temperature-scaled softmax per head
    posteriors = torch.softmax(sims / tau, dim=-1)  # (K, C)
    
    # Uniform average across heads
    avg_posterior = posteriors.mean(dim=0)  # (C,)
    
    return avg_posterior.argmax().item(), avg_posterior


def classify_confidence_weighted(query_act, prototypes, tau=0.07):
    """
    Method 3: Confidence-Weighted Soft Voting
    
    Weight each head by how confident its prediction is.
    No validation data needed.
    
    Args:
        query_act: (K, D) query activations
        prototypes: (C, K, D) class prototypes
        tau: temperature for softmax
    
    Returns:
        pred_idx: predicted class index
        weighted_posterior: (C,) confidence-weighted posterior
    """
    # L2 normalize
    q = F.normalize(query_act, dim=-1)
    p = F.normalize(prototypes, dim=-1)
    
    # Per-head similarities
    sims = torch.einsum('kd,ckd->kc', q, p)  # (K, C)
    
    # Per-head posteriors
    posteriors = torch.softmax(sims / tau, dim=-1)  # (K, C)
    
    # Confidence = max probability per head
    confidence = posteriors.max(dim=-1).values  # (K,)
    
    # Normalize to get weights
    weights = confidence / confidence.sum()  # (K,)
    
    # Weighted average
    weighted_posterior = (posteriors * weights.unsqueeze(-1)).sum(dim=0)  # (C,)
    
    return weighted_posterior.argmax().item(), weighted_posterior


def classify_topk_soft_vote(query_act, prototypes, top_k_indices, tau=0.07):
    """
    Method 4: Top-K Soft Voting
    
    Use only the top-K heads (selected by validation accuracy),
    then do uniform soft voting among them.
    
    Args:
        query_act: (K, D) query activations
        prototypes: (C, K, D) class prototypes
        top_k_indices: list of head indices to use
        tau: temperature for softmax
    
    Returns:
        pred_idx: predicted class index
        avg_posterior: (C,) averaged posterior
    """
    # Subset to top-K heads
    query_subset = query_act[top_k_indices]  # (K', D)
    proto_subset = prototypes[:, top_k_indices, :]  # (C, K', D)
    
    # L2 normalize
    q = F.normalize(query_subset, dim=-1)
    p = F.normalize(proto_subset, dim=-1)
    
    # Per-head similarities
    sims = torch.einsum('kd,ckd->kc', q, p)  # (K', C)
    
    # Soft vote
    posteriors = torch.softmax(sims / tau, dim=-1)  # (K', C)
    avg_posterior = posteriors.mean(dim=0)  # (C,)
    
    return avg_posterior.argmax().item(), avg_posterior


def classify_hard_vote(query_act, prototypes):
    """
    Baseline: Traditional Hard Voting
    
    Each head votes for its most similar class, majority wins.
    
    Args:
        query_act: (K, D) query activations
        prototypes: (C, K, D) class prototypes
    
    Returns:
        pred_idx: predicted class index
        vote_counts: (C,) vote counts per class
    """
    from collections import Counter
    
    q = F.normalize(query_act, dim=-1)
    p = F.normalize(prototypes, dim=-1)
    
    sims = torch.einsum('kd,ckd->kc', q, p)  # (K, C)
    
    # Each head votes for argmax
    votes = sims.argmax(dim=-1).tolist()  # List of K class indices
    
    counter = Counter(votes)
    C = prototypes.shape[0]
    vote_counts = torch.zeros(C)
    for cls_idx, count in counter.items():
        vote_counts[cls_idx] = count
    
    pred_idx = counter.most_common(1)[0][0]
    
    return pred_idx, vote_counts


# =============================================================================
# Head Selection (for top-K methods)
# =============================================================================

def select_top_heads_by_accuracy(prototypes, val_dataset, model_helper, 
                                  str_to_int, k=64, last_n_tokens=1):
    """
    Select top-K heads based on individual classification accuracy on validation set.
    
    Returns:
        top_k_indices: list of K head indices with highest accuracy
        head_accuracies: (K,) accuracy per head
    """
    K = prototypes.shape[1]
    head_correct = torch.zeros(K)
    head_total = torch.zeros(K)
    
    p = F.normalize(prototypes, dim=-1)  # (C, K, D)
    
    for item in tqdm(val_dataset, desc="Selecting top heads"):
        try:
            query_act = get_query_activation(
                val_dataset, item, model_helper, 
                heads=None, last_n_tokens=last_n_tokens
            )
            q = F.normalize(query_act, dim=-1)  # (K, D)
            
            # Per-head predictions
            sims = torch.einsum('kd,ckd->kc', q, p)  # (K, C)
            preds = sims.argmax(dim=-1)  # (K,)
            
            # True label
            true_label = str_to_int.get(item['mapped_label'], -1)
            if true_label < 0:
                continue
            
            # Update counts
            head_correct += (preds == true_label).float().cpu()
            head_total += 1
            
        except Exception as e:
            continue
    
    head_accuracies = head_correct / head_total.clamp(min=1)
    top_k_indices = torch.topk(head_accuracies, k=min(k, K)).indices.tolist()
    
    return top_k_indices, head_accuracies


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_method(method_name, prototypes, test_dataset, model_helper,
                    str_to_int, int_to_str, tau=0.07, top_k_indices=None,
                    last_n_tokens=1):
    """
    Evaluate a classification method on a test dataset.
    
    Args:
        method_name: one of 'concat', 'uniform_soft', 'confidence_weighted', 
                     'topk_soft', 'hard_vote'
        prototypes: (C, K, D) class prototypes
        test_dataset: test data
        model_helper: model helper object
        str_to_int: label to index mapping
        int_to_str: index to label mapping
        tau: temperature (for soft methods)
        top_k_indices: required for 'topk_soft' method
        last_n_tokens: number of tokens to average
    
    Returns:
        accuracy: float
        predictions: list of (true_label, pred_label) tuples
    """
    correct = 0
    total = 0
    predictions = []
    
    for item in tqdm(test_dataset, desc=f"Evaluating {method_name}"):
        try:
            query_act = get_query_activation(
                test_dataset, item, model_helper,
                heads=None, last_n_tokens=last_n_tokens
            )
            
            true_label = item['mapped_label']
            true_idx = str_to_int.get(true_label, -1)
            if true_idx < 0:
                continue
            
            # Classify based on method
            if method_name == 'concat':
                pred_idx, _ = classify_concat_prototype(query_act, prototypes)
            elif method_name == 'uniform_soft':
                pred_idx, _ = classify_uniform_soft_vote(query_act, prototypes, tau)
            elif method_name == 'confidence_weighted':
                pred_idx, _ = classify_confidence_weighted(query_act, prototypes, tau)
            elif method_name == 'topk_soft':
                if top_k_indices is None:
                    raise ValueError("top_k_indices required for topk_soft method")
                pred_idx, _ = classify_topk_soft_vote(query_act, prototypes, top_k_indices, tau)
            elif method_name == 'hard_vote':
                pred_idx, _ = classify_hard_vote(query_act, prototypes)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            pred_label = int_to_str[pred_idx]
            predictions.append((true_label, pred_label))
            
            if pred_idx == true_idx:
                correct += 1
            total += 1
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to evaluate item: {e}")
            continue
    
    accuracy = correct / max(total, 1)
    return accuracy, predictions


# =============================================================================
# Cache-based Evaluation (for faster hyperparameter sweeps)
# =============================================================================

def prepare_simple_cache(model_helper, support_data, val_data, test_data=None,
                         n_trials=20, last_n_tokens=1):
    """
    Precompute all activations for faster evaluation.
    
    Returns:
        cache: dict with prototypes, query activations, and label mappings
    """
    # Build prototypes from support data
    prototypes, str_to_int, int_to_str = build_prototypes(
        support_data, model_helper, 
        n_trials=n_trials, last_n_tokens=last_n_tokens
    )
    
    # L2 normalize prototypes once
    prototypes_n = F.normalize(prototypes, dim=-1)
    
    C, K, D = prototypes.shape
    
    # Cache directory
    cache_root = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/cache"
    os.makedirs(cache_root, exist_ok=True)
    subdir = f"simple_C{C}_K{K}_D{D}_lastN{last_n_tokens}"
    run_dir = os.path.join(cache_root, subdir)
    os.makedirs(run_dir, exist_ok=True)
    
    def _cache_split(dataset, split_name):
        """Cache query activations to disk."""
        split_dir = os.path.join(run_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        labels_idx = []
        valid_idx = 0
        original_indices = []
        
        for idx, item in enumerate(tqdm(dataset, desc=f"Caching {split_name}")):
            try:
                qa = get_query_activation(
                    dataset, item, model_helper,
                    heads=None, n_trials=n_trials, last_n_tokens=last_n_tokens
                )
                qa_n = F.normalize(qa, dim=-1).cpu()
                torch.save(qa_n, os.path.join(split_dir, f"{split_name}_{valid_idx:06d}.pt"))
                
                y_str = item.get("mapped_label", item.get("label", ""))
                labels_idx.append(str_to_int.get(y_str, -1))
                original_indices.append(idx)
                valid_idx += 1
                
            except Exception as e:
                import warnings
                warnings.warn(f"Skipping item {idx} in {split_name}: {e}")
                continue
        
        meta = {
            "type": "disk",
            "dir": split_dir,
            "pattern": f"{split_name}_%06d.pt",
            "count": valid_idx,
            "K": K,
            "D": D,
            "original_indices": original_indices,
        }
        return meta, labels_idx
    
    val_meta, val_labels = _cache_split(val_data, "val")
    test_meta, test_labels = (None, None)
    if test_data is not None:
        test_meta, test_labels = _cache_split(test_data, "test")
    
    return {
        "prototypes": prototypes,
        "prototypes_n": prototypes_n,
        "str_to_int": str_to_int,
        "int_to_str": int_to_str,
        "val_meta": val_meta,
        "val_labels": val_labels,
        "test_meta": test_meta,
        "test_labels": test_labels,
        "C": C,
        "K": K,
        "D": D,
    }


def load_cached_activations(meta, indices=None):
    """Load cached activations from disk."""
    if meta is None:
        return None
    
    split_dir = meta["dir"]
    pattern = meta["pattern"]
    count = meta["count"]
    
    if indices is None:
        indices = range(count)
    
    activations = []
    for idx in indices:
        path = os.path.join(split_dir, pattern % idx)
        activations.append(torch.load(path))
    
    return torch.stack(activations, dim=0)  # (N, K, D)


def eval_from_cache(cache, method_name, split="test", tau=0.07, top_k_indices=None):
    """
    Evaluate a method using cached activations.
    
    Returns:
        accuracy: float
    """
    if split == "val":
        meta = cache["val_meta"]
        labels_idx = cache["val_labels"]
    else:
        meta = cache.get("test_meta")
        labels_idx = cache.get("test_labels")
    
    if meta is None or meta["count"] == 0:
        return 0.0
    
    prototypes = cache["prototypes"].cuda()
    prototypes_n = cache["prototypes_n"].cuda()
    
    correct = 0
    total = 0
    
    # Process in chunks to manage memory
    chunk_size = 64
    count = meta["count"]
    
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        
        # Load chunk of query activations
        query_acts = load_cached_activations(meta, range(start, end)).cuda()  # (B, K, D)
        chunk_labels = labels_idx[start:end]
        
        for i, query_act in enumerate(query_acts):
            true_idx = chunk_labels[i]
            if true_idx < 0:
                continue
            
            if method_name == 'concat':
                pred_idx, _ = classify_concat_prototype(query_act, prototypes)
            elif method_name == 'uniform_soft':
                pred_idx, _ = classify_uniform_soft_vote(query_act, prototypes, tau)
            elif method_name == 'confidence_weighted':
                pred_idx, _ = classify_confidence_weighted(query_act, prototypes, tau)
            elif method_name == 'topk_soft':
                pred_idx, _ = classify_topk_soft_vote(query_act, prototypes, top_k_indices, tau)
            elif method_name == 'hard_vote':
                pred_idx, _ = classify_hard_vote(query_act, prototypes)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            if pred_idx == true_idx:
                correct += 1
            total += 1
        
        del query_acts
        torch.cuda.empty_cache()
    
    return correct / max(total, 1)


def select_top_heads_from_cache(cache, k=64):
    """
    Select top-K heads using cached validation activations.
    
    Returns:
        top_k_indices: list of head indices
        head_accuracies: (K,) tensor
    """
    meta = cache["val_meta"]
    labels_idx = cache["val_labels"]
    prototypes_n = cache["prototypes_n"].cuda()
    
    K = cache["K"]
    head_correct = torch.zeros(K)
    head_total = 0
    
    count = meta["count"]
    chunk_size = 64
    
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        query_acts = load_cached_activations(meta, range(start, end)).cuda()
        chunk_labels = labels_idx[start:end]
        
        for i, query_act in enumerate(query_acts):
            true_idx = chunk_labels[i]
            if true_idx < 0:
                continue
            
            q = F.normalize(query_act, dim=-1)
            sims = torch.einsum('kd,ckd->kc', q, prototypes_n)
            preds = sims.argmax(dim=-1).cpu()
            
            head_correct += (preds == true_idx).float()
            head_total += 1
        
        del query_acts
        torch.cuda.empty_cache()
    
    head_accuracies = head_correct / max(head_total, 1)
    top_k_indices = torch.topk(head_accuracies, k=min(k, K)).indices.tolist()
    
    return top_k_indices, head_accuracies