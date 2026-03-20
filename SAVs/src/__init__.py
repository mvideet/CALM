"""
CALM: Class-conditional Attention vectors for Language Models

A training-free method for few-shot classification using
reliability-weighted attention head activations from multimodal language models.
"""
from .calm import (
    load_model,
    calm_prepare_cache,
    calm_compute_posteriors_from_cache,
    calm_compute_reliability,
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
from .model import (
    ModelHelper,
    Qwen2AudioHelper,
    Qwen2OmniHelper,
    Qwen2VLHelper,
    Phi4MultimodalHelper,
)
from .sav import sav_build_full_cache, sav_evaluate_from_cache, sav_print_top_heads

__all__ = [
    # Core CALM
    "load_model",
    "calm_prepare_cache",
    "calm_compute_posteriors_from_cache",
    "calm_compute_reliability",
    "calm_build_weights_from_r",
    "calm_eval_from_posteriors",
    "calm_get_predictions",
    # Activation extraction
    "get_class_activations",
    "get_query_activations",
    "gather_last_attn_activations",
    "split_activations_by_head",
    "get_last_mean_head_activations",
    # Data loading
    "open_data",
    "get_format_func",
    # Model helpers
    "ModelHelper",
    "Qwen2AudioHelper",
    "Qwen2OmniHelper",
    "Qwen2VLHelper",
    "Phi4MultimodalHelper",
    # SAV baseline
    "sav_build_full_cache",
    "sav_evaluate_from_cache",
    "sav_print_top_heads",
]

__version__ = "0.1.0"
