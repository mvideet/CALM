"""
CALM: Class-conditional Attention vectors for audio Language Models.

Core algorithm implementation for few-shot audio classification using
reliability-weighted attention head activations.

Key functions:
- load_model: Load audio language model
- calm_prepare_cache: Build class prototypes and cache query activations
- calm_compute_posteriors_from_cache: Compute per-head class posteriors
- calm_compute_reliability: Estimate per-head reliability scores
- calm_build_weights_from_r: Convert reliability to head weights
- calm_eval_from_posteriors: Evaluate classification accuracy
"""
import os
import warnings

import torch
import torch.nn.functional as F
from baukit import TraceDict
from tqdm import tqdm

from .model import Qwen2AudioHelper, Qwen2OmniHelper


def load_model(model_name: str, cur_dataset: str):
    """
    Load an audio language model.

    Args:
        model_name: Model identifier ('qwen2-audio-instruct' or 'qwen2.5_omni')
        cur_dataset: Dataset name for format function selection

    Returns:
        ModelHelper instance for the specified model
    """
    if model_name == "qwen2-audio-instruct":
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
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
        return Qwen2AudioHelper(model, processor, cur_dataset)

    elif model_name == "qwen2.5_omni":
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto"
        )
        model.eval()
        model.requires_grad_(False)
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        return Qwen2OmniHelper(model, processor, cur_dataset)

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Use 'qwen2-audio-instruct' or 'qwen2.5_omni'."
        )


def gather_last_attn_activations(inputs, model_helper):
    """
    Perform forward pass and extract attention activations.

    Args:
        inputs: Model inputs from model_helper.insert_audio()
        model_helper: ModelHelper instance

    Returns:
        tuple: (activations_dict, model_output)
    """
    with TraceDict(
        model_helper.model,
        layers=model_helper.model_config['attn_hook_names'],
        retain_input=True,
        retain_output=True,
    ) as td:
        result = model_helper.forward(inputs)
    return td, result


def split_activations_by_head(activations, model_config):
    """
    Split concatenated multi-head attention output into individual heads.

    Args:
        activations: Tensor from gather_last_attn_activations
        model_config: Model configuration dict

    Returns:
        Tensor reshaped to separate heads
    """
    if activations.dim() == 2:
        activations = activations.unsqueeze(1)

    new_shape = activations.size()[:-1] + (
        model_config['n_heads'],
        model_config['resid_dim'] // model_config['n_heads'],
    )
    activations = activations.view(*new_shape)
    return activations.to("cuda")


def get_last_mean_head_activations(
    entire_dataset,
    curr_item,
    model_helper,
    N_TRIALS: int = 20,
    shot: int = 0,
    no_mean: bool = False,
    split: str = "train"
):
    """
    Extract mean activation of the last input token across multiple trials.

    Args:
        entire_dataset: Full dataset for few-shot sampling (can be None for 0-shot)
        curr_item: Current item to process
        model_helper: ModelHelper instance
        N_TRIALS: Number of trials to average over
        shot: Number of few-shot examples (default 0)
        no_mean: If True, return all activations instead of mean
        split: Dataset split ('train', 'test', 'val')

    Returns:
        Tensor of shape (layer, head, seq, head_dim) or concatenated if no_mean=True
    """
    running_sum = None
    successful_trials = 0
    failed_trials = 0
    max_failed_trials = max(1, N_TRIALS // 2)

    for n in range(N_TRIALS):
        torch.cuda.empty_cache()

        if isinstance(curr_item, dict):
            sample = curr_item
        elif isinstance(curr_item, list) and len(curr_item) > 0:
            sample = curr_item[0]
        else:
            break

        try:
            result = model_helper.format_func(
                all_data=entire_dataset,
                cur_item=sample,
                num_shot=shot,
                model_helper=model_helper,
                split=split
            )

            # Handle format function output (audio-only)
            if len(result) == 5:
                tqs, ans, audio_list, _, _ = result
            else:
                tqs, ans, _, audio_list, _, _ = result

            inputs = model_helper.insert_audio(tqs, ans, audio_list)
            activations_td, result = gather_last_attn_activations(inputs, model_helper)

        except Exception as e:
            failed_trials += 1
            if failed_trials >= max_failed_trials:
                break
            continue

        del inputs
        torch.cuda.empty_cache()

        # Process layer activations
        layer_head_tensors = []
        for name in model_helper.model_config["attn_hook_names"]:
            layer_tensor = split_activations_by_head(
                activations_td[name].input, model_helper.model_config
            )
            layer_head_tensors.append(layer_tensor)
            del activations_td[name]

        del activations_td
        del result
        torch.cuda.empty_cache()

        stack_initial = torch.stack(layer_head_tensors, dim=0)
        del layer_head_tensors

        if stack_initial.shape[2] == 0:
            raise RuntimeError("Empty sequence detected in activations.")

        cur_activation = stack_initial[:, -1, :, :, :]
        cur_activation = cur_activation.permute(0, 2, 1, 3)

        del stack_initial
        torch.cuda.empty_cache()

        if no_mean:
            if running_sum is None:
                running_sum = cur_activation
            else:
                running_sum = torch.cat([running_sum, cur_activation], dim=0)
        else:
            if running_sum is None:
                running_sum = cur_activation
            else:
                running_sum += cur_activation
            del cur_activation

        successful_trials += 1
        torch.cuda.empty_cache()

    if successful_trials == 0:
        raise RuntimeError(f"All {N_TRIALS} trials failed. Cannot compute activations.")

    if no_mean:
        return running_sum
    else:
        mean_activations = running_sum / successful_trials
        del running_sum
        return mean_activations


def get_class_activations(
    train_dataset,
    model,
    attn_heads,
    last_n_tokens: int = 1,
    n_trials: int = 20
):
    """
    Compute class-conditional prototype activations from training data.

    Args:
        train_dataset: Training dataset
        model: ModelHelper instance
        attn_heads: List of (layer, head) tuples
        last_n_tokens: Number of tokens to average (default 1)
        n_trials: Number of trials for activation averaging

    Returns:
        tuple: (avg_activations, str_to_int, int_to_str)
            - avg_activations: (C, K, D) tensor of class prototypes
            - str_to_int: label string to index mapping
            - int_to_str: index to label string mapping
    """
    str_to_int = {}
    int_to_str = {}
    str_to_activation = {}
    str_to_count = {}

    for item in tqdm(train_dataset, desc="Building class prototypes"):
        try:
            mean_activations = get_last_mean_head_activations(
                train_dataset, item, model, N_TRIALS=n_trials, shot=0
            )
            head_act = []
            for head in attn_heads:
                if isinstance(last_n_tokens, int) and last_n_tokens > 1:
                    vec = mean_activations[head[0], head[1], -last_n_tokens:, :].mean(dim=0)
                else:
                    vec = mean_activations[head[0], head[1], -1]
                head_act.append(vec)
            head_act = torch.stack(head_act)

            label = item['mapped_label']
            label_key = label.lower() if isinstance(label, str) else label

            if label_key in str_to_activation:
                str_to_activation[label_key] += head_act
                str_to_count[label_key] += 1
            else:
                str_to_activation[label_key] = head_act
                int_label = len(str_to_activation) - 1
                str_to_int[label_key] = int_label
                int_to_str[int_label] = label_key
                str_to_count[label_key] = 1

        except Exception as e:
            warnings.warn(f"Skipping item (label: {item.get('mapped_label', 'unknown')}): {e}")
            continue

    avg_activations = []
    for key, item in str_to_activation.items():
        avg_activations.append(torch.div(item, str_to_count[key]))
    avg_activations = torch.stack(avg_activations)

    return avg_activations, str_to_int, int_to_str


def get_query_activations(
    query_input,
    model_helper,
    common_heads,
    last_n_tokens: int = 1,
    n_trials: int = 1
):
    """
    Get activations for a query input.

    Args:
        query_input: Input item (wrapped in list)
        model_helper: ModelHelper instance
        common_heads: List of (layer, head) tuples
        last_n_tokens: Number of tokens to average
        n_trials: Number of trials for averaging

    Returns:
        Tensor of head activations (K, D) or None if failed
    """
    try:
        mean_activations = get_last_mean_head_activations(
            None, query_input, model_helper, N_TRIALS=n_trials, shot=0
        )
        head_act = []
        for head in common_heads:
            if isinstance(last_n_tokens, int) and last_n_tokens > 1:
                vec = mean_activations[head[0], head[1], -last_n_tokens:, :].mean(dim=0)
            else:
                vec = mean_activations[head[0], head[1], -1]
            head_act.append(vec)
        head_act = torch.stack(head_act)
        return head_act
    except Exception as e:
        warnings.warn(f"Failed to process query: {e}. Returning None.")
        return None


def _l2norm(x, dim=-1, eps=1e-8):
    """L2 normalize along specified dimension."""
    denom = x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / denom


def calm_prepare_cache(
    model,
    support_data,
    val_data,
    test_data=None,
    heads=None,
    last_n_tokens: int = 1,
    n_trials: int = 20,
    cache_dir: str = "./cache"
):
    """
    Build class prototypes and cache query activations to disk.

    This is the main setup function for CALM. It:
    1. Computes class prototype vectors from support data
    2. Caches L2-normalized query activations for val/test splits

    Args:
        model: ModelHelper instance
        support_data: Support set for building prototypes
        val_data: Validation data for reliability estimation
        test_data: Optional test data for evaluation
        heads: List of (layer, head) tuples (default: all heads)
        last_n_tokens: Number of tokens to average
        n_trials: Number of trials for activation averaging
        cache_dir: Directory to store cached activations

    Returns:
        dict with keys:
            - prototypes_n: L2-normalized prototypes (C, K, D)
            - prototypes: Raw prototypes
            - heads: List of head tuples
            - int_to_str, str_to_int: Label mappings
            - qacts_val_n, qacts_test_n: Cache metadata
            - val_labels_idx, test_labels_idx: Label indices
    """
    if heads is None:
        heads = list(model.all_heads)

    prototypes, str_to_int, int_to_str = get_class_activations(
        support_data, model, heads,
        last_n_tokens=last_n_tokens,
        n_trials=n_trials
    )
    C, K, D = prototypes.shape
    device = prototypes.device

    os.makedirs(cache_dir, exist_ok=True)
    subdir = f"calm_C{C}_K{K}_D{D}_lastN{int(last_n_tokens)}"
    run_dir = os.path.join(cache_dir, subdir)
    os.makedirs(run_dir, exist_ok=True)

    def _collect_qacts_and_labels(dataset, split_name: str):
        items = list(dataset)
        labels_idx = []
        original_indices = []
        split_dir = os.path.join(run_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        valid_idx = 0

        for idx, it in enumerate(tqdm(items, desc=f"Caching {split_name} activations")):
            qa = get_query_activations(
                [it], model, heads,
                last_n_tokens=last_n_tokens,
                n_trials=n_trials
            )
            if qa is None:
                warnings.warn(f"Skipping item {idx} in {split_name}.")
                continue

            qa = _l2norm(qa, dim=1).to(dtype=prototypes.dtype, copy=False).cpu()
            torch.save(qa, os.path.join(split_dir, f"{split_name}_{valid_idx:06d}.pt"))

            y_str = it.get("mapped_label", it.get("label", ""))
            y_key = y_str.lower() if isinstance(y_str, str) else y_str
            labels_idx.append(str_to_int.get(y_key, -1))
            original_indices.append(idx)
            valid_idx += 1

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

    q_val_meta, val_labels_idx = _collect_qacts_and_labels(val_data, "val")
    q_test_meta, test_labels_idx = (None, None)
    if test_data is not None:
        q_test_meta, test_labels_idx = _collect_qacts_and_labels(test_data, "test")

    prot_n = _l2norm(prototypes, dim=2)

    return {
        "prototypes_n": prot_n,
        "prototypes": prototypes,
        "heads": heads,
        "int_to_str": int_to_str,
        "str_to_int": str_to_int,
        "qacts_val_n": q_val_meta,
        "val_labels_idx": val_labels_idx,
        "qacts_test_n": q_test_meta,
        "test_labels_idx": test_labels_idx,
    }


def calm_compute_posteriors_from_cache(cache: dict, tau: float, split: str):
    """
    Compute per-head class posteriors from cached activations.

    Args:
        cache: Cache dict from calm_prepare_cache
        tau: Temperature for softmax (lower = sharper)
        split: 'val' or 'test'

    Returns:
        Tensor of shape (T, K, C) - posteriors for T samples, K heads, C classes
    """
    prot_n = cache["prototypes_n"]

    if split == "val":
        q_src = cache["qacts_val_n"]
    elif split == "test":
        q_src = cache.get("qacts_test_n", None)
    else:
        raise ValueError("split must be 'val' or 'test'")

    if q_src is None:
        C, K = prot_n.shape[0], prot_n.shape[1]
        return torch.empty((0, K, C), dtype=prot_n.dtype, device=prot_n.device)

    device = prot_n.device
    dtype = prot_n.dtype

    if torch.is_tensor(q_src):
        if q_src.numel() == 0:
            C, K = prot_n.shape[0], prot_n.shape[1]
            return torch.empty((0, K, C), dtype=dtype, device=device)
        sims = torch.einsum("tjd,cjd->tjc", q_src.to(device=device, dtype=dtype), prot_n)
        logits = sims / float(tau)
        return torch.softmax(logits, dim=2)

    if isinstance(q_src, dict) and q_src.get("type") == "disk":
        count = int(q_src.get("count", 0))
        if count == 0:
            C, K = prot_n.shape[0], prot_n.shape[1]
            return torch.empty((0, K, C), dtype=dtype, device=device)

        split_dir = q_src["dir"]
        pattern = q_src["pattern"]
        chunk_size = 128
        out_chunks = []

        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            batch = []
            for idx in range(start, end):
                path = os.path.join(split_dir, pattern % idx)
                batch.append(torch.load(path))
            q_batch = torch.stack(batch, dim=0).to(device=device, dtype=dtype, non_blocking=True)
            sims = torch.einsum("tjd,cjd->tjc", q_batch, prot_n)
            logits = sims / float(tau)
            out_chunks.append(torch.softmax(logits, dim=2).to(dtype=torch.float32).cpu())
            del q_batch, sims, logits
            torch.cuda.empty_cache()

        return torch.cat(out_chunks, dim=0)

    raise ValueError("Unsupported qacts source type in cache")


def calm_compute_reliability(P_val: torch.Tensor, val_labels_idx, weight_scheme: str):
    """
    Compute per-head reliability scores from validation posteriors.

    Args:
        P_val: Validation posteriors (T, K, C)
        val_labels_idx: Ground truth class indices for validation samples
        weight_scheme: One of:
            - 'prob_softmax': Mean probability for correct class
            - 'margin_clamped': Clamped margin (correct - runner-up)
            - 'margin_softmax': Raw margin
            - 'brier_softmax': Brier skill score

    Returns:
        tuple: (r, counts)
            - r: Reliability matrix (K, C)
            - counts: Sample counts per class (C,)
    """
    T, K, C = P_val.shape
    device = P_val.device
    r = torch.zeros((K, C), dtype=P_val.dtype, device=device)
    counts = torch.zeros((C,), dtype=torch.long, device=device)

    idx_by_c = [[] for _ in range(C)]
    for t, y in enumerate(val_labels_idx):
        if 0 <= y < C:
            idx_by_c[y].append(t)

    for c in range(C):
        idx = idx_by_c[c]
        n_c = len(idx)
        counts[c] = n_c
        if n_c == 0:
            continue
        Pv = P_val[idx, :, :]

        if weight_scheme == "prob_softmax":
            r[:, c] = Pv[:, :, c].mean(dim=0)

        elif weight_scheme in ("margin_clamped", "margin_softmax"):
            k = min(2, C)
            top_vals, top_idx = torch.topk(Pv, k=k, dim=2)
            p_jc = Pv[:, :, c]

            if k == 1:
                max_other = torch.zeros_like(p_jc)
            else:
                max_other = torch.where(
                    top_idx[:, :, 0] == c,
                    top_vals[:, :, 1],
                    top_vals[:, :, 0]
                )

            margin = p_jc - max_other
            if weight_scheme == "margin_clamped":
                margin = torch.clamp(margin, min=0.0)
            r[:, c] = margin.mean(dim=0)

        elif weight_scheme == "brier_softmax":
            s2 = (Pv * Pv).sum(dim=2)
            r[:, c] = (2.0 * Pv[:, :, c] - s2).mean(dim=0)

        else:
            raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    return r, counts


def calm_apply_shrinkage(r: torch.Tensor, counts: torch.Tensor, alpha: float):
    """
    Apply James-Stein shrinkage to reliability estimates.

    Args:
        r: Raw reliability matrix (K, C)
        counts: Sample counts per class (C,)
        alpha: Shrinkage strength (0 = no shrinkage)

    Returns:
        Shrunk reliability matrix (K, C)
    """
    if alpha <= 0.0:
        return r
    r_bar = r.mean()
    n_c = counts.to(r.dtype).unsqueeze(0)
    shrink = n_c / (n_c + float(alpha))
    comp = 1.0 - shrink
    return r * shrink + r_bar * comp


def calm_build_weights_from_r(
    r_hat: torch.Tensor,
    weight_scheme: str,
    tau_w: float,
    top_k: int = None
):
    """
    Convert reliability scores to normalized head weights.

    Args:
        r_hat: Reliability matrix (K, C)
        weight_scheme: Weighting scheme
        tau_w: Temperature for weight softmax
        top_k: Optional top-k head selection per class

    Returns:
        Weight matrix (K, C) summing to 1 per class
    """
    K, C = r_hat.shape

    if weight_scheme in ("margin_softmax", "prob_softmax", "brier_softmax"):
        logits = r_hat / max(float(tau_w), 1e-8)

        if isinstance(top_k, int) and 0 < top_k < K:
            top_idx = torch.topk(r_hat, k=top_k, dim=0).indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(0, top_idx, True)
            min_val = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, min_val)

        return torch.softmax(logits, dim=0)

    else:
        r = r_hat.clone()
        mask = None

        if isinstance(top_k, int) and 0 < top_k < K:
            top_idx = torch.topk(r, k=top_k, dim=0).indices
            mask = torch.zeros_like(r, dtype=torch.bool)
            mask.scatter_(0, top_idx, True)
            r = r.masked_fill(~mask, 0.0)

        tw = float(tau_w)
        if tw != 1.0:
            eps = torch.finfo(r.dtype).tiny
            gamma = 1.0 / max(tw, 1e-8)
            r = torch.pow(r + eps, gamma)

        sums = r.sum(dim=0, keepdim=True)
        w = torch.zeros_like(r)
        nonzero = (sums > 1e-12).squeeze(0)

        if nonzero.any():
            w[:, nonzero] = r[:, nonzero] / sums[:, nonzero]

        if (~nonzero).any():
            if mask is not None:
                allowed = mask[:, ~nonzero].to(r.dtype)
                allowed_sums = allowed.sum(dim=0, keepdim=True).clamp_min(1.0)
                w[:, ~nonzero] = allowed / allowed_sums
            else:
                w[:, ~nonzero] = 1.0 / float(K)

        return w


def calm_eval_from_posteriors(
    P_test: torch.Tensor,
    w: torch.Tensor,
    test_labels_idx=None
):
    """
    Evaluate classification accuracy using weighted posteriors.

    Args:
        P_test: Test posteriors (T, K, C)
        w: Head weights (K, C)
        test_labels_idx: Optional ground truth indices

    Returns:
        If test_labels_idx provided: accuracy (float)
        Otherwise: predicted class indices (tensor)
    """
    if P_test is None or P_test.numel() == 0:
        return 0.0

    scores = (P_test * w.unsqueeze(0)).sum(dim=1)
    pred_idx = scores.argmax(dim=1)

    if test_labels_idx is None:
        return pred_idx

    correct = 0
    total = 0
    for i, y in enumerate(test_labels_idx):
        if 0 <= y < scores.shape[1]:
            correct += int(pred_idx[i].item() == y)
            total += 1

    return correct / max(total, 1)


def calm_get_predictions(P_test: torch.Tensor, w: torch.Tensor, cache: dict):
    """
    Get predicted class labels for test samples.

    Args:
        P_test: Test posteriors (T, K, C)
        w: Head weights (K, C)
        cache: Cache dict with int_to_str mapping

    Returns:
        List of predicted label strings
    """
    if P_test is None or P_test.numel() == 0:
        return []

    scores = (P_test * w.unsqueeze(0)).sum(dim=1)
    pred_idx = scores.argmax(dim=1).cpu().tolist()
    int_to_str = cache["int_to_str"]

    return [int_to_str.get(idx, "UNKNOWN") for idx in pred_idx]
