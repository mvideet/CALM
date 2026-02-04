"""
Global Reliability Weighted SAV (GRW-SAV) Implementation

This is an intermediate baseline between SAV and PRWE that:
- Learns ONE scalar reliability per head (not class-conditional)
- Still uses soft weighting via softmax
- Sits cleanly between SAV (uniform weights) and PRWE (class-conditional weights)

Key difference from PRWE:
- PRWE: r has shape (K, C) - reliability per head per class
- GRW-SAV: r has shape (K,) - one global reliability per head
"""

import torch
import os
from tqdm import tqdm


def grw_compute_reliability(P_val: torch.Tensor, val_labels_idx) -> torch.Tensor:
    """
    Compute global (class-agnostic) reliability scores for each head.
    
    Formula: r^{(j)} = (1/N) * sum_i max(0, p_{y_i}^{(j)} - max_{c != y_i} p_c^{(j)})
    
    This computes the average clamped margin for each head across all validation samples.
    
    Args:
        P_val: Posterior probabilities, shape (T, K, C) where:
               T = number of validation samples
               K = number of heads
               C = number of classes
        val_labels_idx: List of ground truth class indices for each sample
        
    Returns:
        r: Global reliability scores, shape (K,)
    """
    T, K, C = P_val.shape
    device = P_val.device
    dtype = P_val.dtype
    
    # Initialize accumulator for reliability scores
    r_sum = torch.zeros(K, dtype=dtype, device=device)
    count = 0
    
    for t, y in enumerate(val_labels_idx):
        if not (0 <= y < C):
            continue  # Skip invalid labels
            
        # Get posteriors for this sample: shape (K, C)
        p_t = P_val[t]
        
        # p_{y_i}^{(j)} - probability of true class for each head: shape (K,)
        p_true = p_t[:, y]
        
        # max_{c != y_i} p_c^{(j)} - max probability of non-true classes: shape (K,)
        # Create a mask to exclude the true class
        mask = torch.ones(C, dtype=torch.bool, device=device)
        mask[y] = False
        p_other = p_t[:, mask]  # shape (K, C-1)
        p_max_other = p_other.max(dim=1).values  # shape (K,)
        
        # Clamped margin: max(0, p_true - p_max_other)
        margin = torch.clamp(p_true - p_max_other, min=0.0)
        
        r_sum += margin
        count += 1
    
    # Average over all valid samples
    if count > 0:
        r = r_sum / count
    else:
        r = r_sum  # All zeros if no valid samples
        
    return r


def grw_build_weights(r: torch.Tensor, tau_w: float = 1.0, top_k: int = None) -> torch.Tensor:
    """
    Convert global reliability scores to head weights via softmax.
    
    Formula: w^{(j)} = softmax(r^{(j)} / tau_w)
    
    Args:
        r: Global reliability scores, shape (K,)
        tau_w: Temperature for softmax (lower = sharper)
        top_k: If specified, only keep top-k heads before softmax
        
    Returns:
        w: Head weights, shape (K,)
    """
    K = r.shape[0]
    logits = r / max(float(tau_w), 1e-8)
    
    if isinstance(top_k, int) and 0 < top_k < K:
        # Only keep top-k heads
        top_idx = torch.topk(r, k=top_k).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[top_idx] = True
        min_val = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, min_val)
    
    w = torch.softmax(logits, dim=0)
    return w


def grw_predict(P_test: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Make predictions using GRW-SAV weighted voting.
    
    Formula: score_c = sum_j w^{(j)} * p_c^{(j)}(x)
    
    Args:
        P_test: Posterior probabilities, shape (T, K, C)
        w: Head weights, shape (K,)
        
    Returns:
        pred_idx: Predicted class indices, shape (T,)
    """
    # Weighted sum over heads: (T, K, C) * (K, 1) -> sum over K -> (T, C)
    scores = (P_test * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    pred_idx = scores.argmax(dim=1)
    return pred_idx


def grw_eval_from_posteriors(P_test: torch.Tensor, w: torch.Tensor, test_labels_idx=None):
    """
    Evaluate GRW-SAV on test set.
    
    Args:
        P_test: Posterior probabilities, shape (T, K, C)
        w: Head weights, shape (K,)
        test_labels_idx: Ground truth labels (optional, returns predictions if None)
        
    Returns:
        Accuracy (float) if test_labels_idx provided, else predictions tensor
    """
    if P_test is None or P_test.numel() == 0:
        return 0.0
    
    pred_idx = grw_predict(P_test, w)
    
    if test_labels_idx is None:
        return pred_idx
    
    # Compute accuracy
    correct = 0
    total = 0
    for i, y in enumerate(test_labels_idx):
        if 0 <= y < P_test.shape[2]:
            correct += int(pred_idx[i].item() == y)
            total += 1
    
    return correct / max(total, 1)


# ============================================================================
# Integration with existing PRWE cache system
# ============================================================================

def grw_from_cache(cache, tau: float, tau_w: float = 1.0, top_k: int = None, 
                   split: str = "test"):
    """
    Run GRW-SAV using cached posteriors from PRWE.
    
    This function reuses the cache created by prwe_prepare_cache() but computes
    global (not class-conditional) reliability weights.
    
    Args:
        cache: Cache dict from prwe_prepare_cache()
        tau: Temperature for computing posteriors
        tau_w: Temperature for weight softmax
        top_k: Number of top heads to keep (optional)
        split: "val" or "test"
        
    Returns:
        accuracy: Classification accuracy on the specified split
    """
    # Import the posterior computation function from PRWE
    from prwe import prwe_compute_posteriors_from_cache
    
    # Get validation posteriors for computing reliability
    P_val = prwe_compute_posteriors_from_cache(cache, tau, split="val")
    val_labels_idx = cache["val_labels_idx"]
    
    # Compute global reliability (one scalar per head)
    r = grw_compute_reliability(P_val, val_labels_idx)
    
    # Build weights via softmax
    w = grw_build_weights(r, tau_w=tau_w, top_k=top_k)
    
    # Get test posteriors and evaluate
    if split == "val":
        P_test = P_val
        test_labels_idx = val_labels_idx
    else:
        P_test = prwe_compute_posteriors_from_cache(cache, tau, split="test")
        test_labels_idx = cache.get("test_labels_idx", None)
    
    accuracy = grw_eval_from_posteriors(P_test, w, test_labels_idx)
    
    return accuracy, r, w


def grw_grid_search(cache, tau_values, tau_w_values, top_k_values=None):
    """
    Grid search over hyperparameters for GRW-SAV.
    
    Args:
        cache: Cache from prwe_prepare_cache()
        tau_values: List of temperature values for posteriors
        tau_w_values: List of temperature values for weight softmax
        top_k_values: List of top-k values (optional)
        
    Returns:
        results: Dict with best params and all results
    """
    from prwe import prwe_compute_posteriors_from_cache
    
    if top_k_values is None:
        top_k_values = [None]
    
    best_acc = 0.0
    best_params = {}
    all_results = []
    
    for tau in tqdm(tau_values, desc="GRW-SAV Grid Search"):
        # Compute posteriors once per tau
        P_val = prwe_compute_posteriors_from_cache(cache, tau, split="val")
        val_labels_idx = cache["val_labels_idx"]
        
        # Compute global reliability
        r = grw_compute_reliability(P_val, val_labels_idx)
        
        for tau_w in tau_w_values:
            for top_k in top_k_values:
                w = grw_build_weights(r, tau_w=tau_w, top_k=top_k)
                acc = grw_eval_from_posteriors(P_val, w, val_labels_idx)
                
                result = {
                    "tau": tau,
                    "tau_w": tau_w,
                    "top_k": top_k,
                    "val_acc": acc
                }
                all_results.append(result)
                
                if acc > best_acc:
                    best_acc = acc
                    best_params = result.copy()
    
    return {
        "best_params": best_params,
        "best_val_acc": best_acc,
        "all_results": all_results
    }


# # ============================================================================
# # Example usage and comparison script
# # ============================================================================

# if __name__ == "__main__":
#     """
#     Example usage comparing SAV, GRW-SAV, and PRWE.
    
#     This assumes you have a cache from prwe_prepare_cache().
#     """
#     print("GRW-SAV: Global Reliability Weighted SAV")
#     print("=" * 50)
#     print("\nKey Properties:")
#     print("  - One scalar reliability per head (not class-conditional)")
#     print("  - Soft weighting via softmax")
#     print("  - Intermediate between SAV and PRWE")
#     print("\nFormulas:")
#     print("  r^{(j)} = (1/N) * sum_i max(0, p_{y_i}^{(j)} - max_{c!=y_i} p_c^{(j)})")
#     print("  w^{(j)} = softmax(r^{(j)})")
#     print("  score_c = sum_j w^{(j)} * p_c^{(j)}(x)")
#     print("\n" + "=" * 50)
#     print("\nTo use:")
#     print("  from grw_sav import grw_from_cache")
#     print("  acc, r, w = grw_from_cache(cache, tau=0.1, tau_w=1.0)")