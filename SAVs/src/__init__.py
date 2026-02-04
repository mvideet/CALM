"""
CALM: Class-conditional Attention vectors for audio Language Models

A training-free method for few-shot audio classification using
reliability-weighted attention head activations from audio language models.

Example usage:
    from src import load_model, calm_prepare_cache, calm_eval_from_posteriors
    from src import open_data

    # Load model and data
    model = load_model("qwen2-audio-instruct", "vgg_sound_qa")
    train_data = open_data("vgg_sound_qa", "train.json")
    val_data = open_data("vgg_sound_qa", "val.json")

    # Build cache and evaluate
    cache = calm_prepare_cache(model, train_data, val_data)
    P_val = calm_compute_posteriors_from_cache(cache, tau=0.07, split="val")
    r, counts = calm_compute_reliability(P_val, cache["val_labels_idx"], "margin_clamped")
    w = calm_build_weights_from_r(r, weight_scheme="margin_clamped", tau_w=1.0)
    accuracy = calm_eval_from_posteriors(P_val, w, test_labels_idx=cache["val_labels_idx"])
"""
from .calm import (
    load_model,
    calm_prepare_cache,
    calm_compute_posteriors_from_cache,
    calm_compute_reliability,
    calm_apply_shrinkage,
    calm_build_weights_from_r,
    calm_eval_from_posteriors,
    calm_get_predictions,
    get_class_activations,
    get_query_activations,
    gather_last_attn_activations,
    split_activations_by_head,
    get_last_mean_head_activations,
)
from .preprocess import open_data, get_format_func
from .model import Qwen2AudioHelper, Qwen2OmniHelper, ModelHelper

__all__ = [
    # Core CALM functions
    "load_model",
    "calm_prepare_cache",
    "calm_compute_posteriors_from_cache",
    "calm_compute_reliability",
    "calm_apply_shrinkage",
    "calm_build_weights_from_r",
    "calm_eval_from_posteriors",
    "calm_get_predictions",
    # Lower-level functions
    "get_class_activations",
    "get_query_activations",
    "gather_last_attn_activations",
    "split_activations_by_head",
    "get_last_mean_head_activations",
    # Data loading
    "open_data",
    "get_format_func",
    # Model helpers
    "Qwen2AudioHelper",
    "Qwen2OmniHelper",
    "ModelHelper",
]

__version__ = "0.1.0"
