"""
CALM: Class-conditional Attention vectors for audio Language Models

A training-free method for few-shot audio classification using attention head activations.
"""
from .utils import (
    load_model,
    mllm_encode,
    mllm_classify,
    mllm_classify_with_counts,
    mllm_classify_spoof,
)
from .preprocess import open_data, get_format_func
from .model import Qwen2AudioHelper, Qwen2OmniHelper, ModelHelper

__all__ = [
    "load_model",
    "mllm_encode",
    "mllm_classify",
    "mllm_classify_with_counts",
    "mllm_classify_spoof",
    "open_data",
    "get_format_func",
    "Qwen2AudioHelper",
    "Qwen2OmniHelper",
    "ModelHelper",
]

__version__ = "0.1.0"
