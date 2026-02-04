"""
Core utilities for CALM (Class-conditional Attention vectors for audio Language Models).

This module provides functions for:
- Model loading
- Attention activation extraction
- Class-conditional encoding (mllm_encode)
- Classification (mllm_classify)
"""
from baukit import TraceDict
from .model import Qwen2AudioHelper, Qwen2OmniHelper
from .preprocess import open_data
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoProcessor, logging

logging.set_verbosity_warning()


def load_model(model_name, cur_dataset):
    """
    Load a model and return its helper class.

    Args:
        model_name: Model identifier. Supported: 'qwen2.5_omni', 'qwen2-audio-instruct'
        cur_dataset: Dataset name for format function selection

    Returns:
        ModelHelper instance for the specified model
    """
    if model_name == "qwen2.5_omni":
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto"
        )
        model.eval()
        model.requires_grad_(False)
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        return Qwen2OmniHelper(model, processor, cur_dataset)
    
    elif model_name == "qwen2-audio-instruct":
        from transformers import Qwen2AudioForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device}
        )
        model.tie_weights()
        model.eval()
        model.requires_grad_(False)
        return Qwen2AudioHelper(model, processor, cur_dataset)
    
    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported models: 'qwen2.5_omni', 'qwen2-audio-instruct'"
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
        retain_output=True
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
        Tensor of shape (batch_size, n_tokens, n_heads, head_dim)
    """
    if activations.dim() == 2:
        activations = activations.unsqueeze(1)
    
    new_shape = activations.size()[:-1] + (
        model_config['n_heads'],
        model_config['resid_dim'] // model_config['n_heads']
    )
    activations = activations.view(*new_shape)
    return activations.to("cuda")


def get_last_mean_head_activations(entire_dataset, curr_item, model_helper, N_TRIALS,
                                    shot=4, no_mean=False, split="train"):
    """
    Extract mean activation of the last input token across multiple trials.

    Args:
        entire_dataset: Full dataset for few-shot sampling
        curr_item: Current item to process
        model_helper: ModelHelper instance
        N_TRIALS: Number of trials to average over
        shot: Number of few-shot examples
        no_mean: If True, return all activations instead of mean
        split: Dataset split ('train', 'test', 'val')

    Returns:
        Tensor of shape (layer, head, 1, head_dim) or concatenated if no_mean=True
    """
    running_sum = None
    successful_trials = 0
    
    for n in range(N_TRIALS):
        torch.cuda.empty_cache()
        
        if isinstance(curr_item, dict):
            sample = curr_item
        else:
            sample = curr_item[0]

        try:
            result = model_helper.format_func(
                all_data=entire_dataset,
                cur_item=sample,
                num_shot=shot,
                model_helper=model_helper,
                split=split,
            )

            # Handle format function output
            if len(result) == 5:
                tqs, ans, audio_list, _, _ = result
            else:
                tqs, ans, _, audio_list, _, _ = result

            inputs = model_helper.insert_audio(tqs, ans, audio_list)
            activations_td, _ = gather_last_attn_activations(inputs, model_helper)
            
        except Exception as e:
            if n == 0:
                raise
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
        torch.cuda.empty_cache()
        
        stack_initial = torch.stack(layer_head_tensors, dim=0)
        del layer_head_tensors

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


def get_class_activations(train_dataset, model, attn_heads, config):
    """
    Compute class-conditional activations from training data.

    Args:
        train_dataset: Training dataset
        model: ModelHelper instance
        attn_heads: List of (layer, head, -1) tuples
        config: Config dict with 'N_TRIALS'

    Returns:
        tuple: (avg_activations, str_to_int, int_to_str)
    """
    str_to_int = {}
    int_to_str = {}
    str_to_activation = {}
    str_to_count = {}

    for item in tqdm(train_dataset):
        N_TRIALS = config['N_TRIALS']
        mean_activations = get_last_mean_head_activations(
            train_dataset, item, model, N_TRIALS=N_TRIALS, shot=0
        )
        
        head_act = []
        for head in attn_heads:
            head_act.append(mean_activations[head[0], head[1], -1])
        head_act = torch.stack(head_act)

        label = item['mapped_label']
        if label in str_to_activation:
            str_to_activation[label] += head_act
            str_to_count[label] += 1
        else:
            str_to_activation[label] = head_act
            int_label = len(str_to_activation) - 1
            str_to_int[label] = int_label
            int_to_str[int_label] = label
            str_to_count[label] = 1

    avg_activations = []
    for key, item in str_to_activation.items():
        avg_activations.append(torch.div(item, str_to_count[key]))
    avg_activations = torch.stack(avg_activations)
    
    return avg_activations, str_to_int, int_to_str


def get_query_activations(query_input, model_helper, common_heads):
    """
    Get activations for a query input.

    Args:
        query_input: Input item (wrapped in list)
        model_helper: ModelHelper instance
        common_heads: List of (layer, head, -1) tuples

    Returns:
        Tensor of head activations
    """
    mean_activations = get_last_mean_head_activations(
        None, query_input, model_helper, N_TRIALS=1, shot=0
    )
    
    head_act = []
    for head in common_heads:
        head_act.append(mean_activations[head[0], head[1], -1])
    head_act = torch.stack(head_act)
    
    return head_act


def record_head_performance(sample_activations, cur_activation, label, success_count,
                            model_helper=None, common_heads=None):
    """
    Record per-head classification performance.

    Args:
        sample_activations: Class centroids (C, K, D)
        cur_activation: Query activations (K, D)
        label: Ground-truth class index
        success_count: List to update with correct predictions per head
        model_helper: Optional ModelHelper
        common_heads: Optional head tuples for debugging
    """
    C, K, D = sample_activations.shape
    assert cur_activation.shape == (K, D), f"Shape mismatch: {cur_activation.shape} vs {(K, D)}"

    votes = []
    for i in range(K):
        scores = F.cosine_similarity(
            sample_activations[:, i, :], cur_activation[i, :], dim=-1
        )
        pred_i = int(scores.argmax().item())
        votes.append(pred_i)

    for i, cls_idx in enumerate(votes):
        if cls_idx == label:
            success_count[i] += 1


def retrieve_examples(sample_activations, cur_activation):
    """
    Find most similar class using head voting.

    Args:
        sample_activations: (num_classes, num_heads, hidden_dim)
        cur_activation: (num_heads, hidden_dim)

    Returns:
        List of class indices ordered by vote count
    """
    all_sample = []
    for i in range(sample_activations.shape[1]):
        scores = F.cosine_similarity(
            sample_activations[:, i, :], cur_activation[i, :], dim=-1
        )
        all_sample.append(scores.argmax(dim=0).item())

    counter = Counter(all_sample)
    return [item[0] for item in counter.most_common()]


def retrieve_examples_with_counts(sample_activations, cur_activation):
    """
    Get vote counts per class.

    Args:
        sample_activations: (num_classes, num_heads, hidden_dim)
        cur_activation: (num_heads, hidden_dim)

    Returns:
        List of (class_index, vote_count) tuples sorted by votes
    """
    head_votes = []
    for i in range(cur_activation.shape[0]):
        scores = F.cosine_similarity(
            sample_activations[:, i, :], cur_activation[i, :], dim=-1
        )
        head_votes.append(scores.argmax(dim=0).item())

    counter = Counter(head_votes)
    return counter.most_common()


def mllm_encode(model, train_data, num_head, config):
    """
    Extract class-conditional attention vectors (CALM encoding).

    This is the main encoding function that:
    1. Extracts activations for all training samples
    2. Evaluates each attention head's classification accuracy
    3. Selects the top-k performing heads
    4. Returns class centroids for those heads

    Args:
        model: ModelHelper instance
        train_data: Training dataset
        num_head: Number of top heads to select
        config: Config dict with 'N_TRIALS'

    Returns:
        dict with keys:
            - 'activations': Class centroids (num_classes, num_heads, head_dim)
            - 'top_heads': List of selected (layer, head, -1) tuples
            - 'int_to_str': Mapping from class index to label string
    """
    all_heads = model.all_heads
    class_activations, str_to_int, int_to_str = get_class_activations(
        train_data, model, all_heads, config
    )
    success_count = [0 for _ in range(class_activations.shape[1])]

    for item in tqdm(train_data):
        query_activations = get_query_activations([item], model, all_heads).squeeze(dim=0)
        int_label = str_to_int[item['mapped_label']]
        record_head_performance(class_activations, query_activations, int_label, success_count)

    # Select top-k heads
    arr = np.array(success_count)
    topk_indices = np.argsort(arr)[-num_head:][::-1]

    top_heads = [all_heads[item] for item in topk_indices.tolist()]
    top_class_activations, str_to_int, int_to_str = get_class_activations(
        train_data, model, top_heads, config
    )
    
    return {
        "activations": top_class_activations,
        "top_heads": top_heads,
        "int_to_str": int_to_str
    }


def mllm_classify(inputs, model, class_embed):
    """
    Classify an input using CALM embeddings.

    Args:
        inputs: Input item to classify
        model: ModelHelper instance
        class_embed: Dict from mllm_encode()

    Returns:
        Predicted class label string
    """
    cur_activations = get_query_activations(
        [inputs], model, class_embed['top_heads']
    ).squeeze(dim=0)
    top_k_examples = retrieve_examples(class_embed['activations'], cur_activations)
    cur_int_label = top_k_examples[0]
    return class_embed['int_to_str'][cur_int_label]


def mllm_classify_with_counts(inputs, model, class_embed):
    """
    Classify with vote distribution.

    Args:
        inputs: Input item to classify
        model: ModelHelper instance
        class_embed: Dict from mllm_encode()

    Returns:
        tuple: (predicted_label, label_vote_counts dict)
    """
    cur_activations = get_query_activations([inputs], model, class_embed['top_heads'])
    votes_by_index = retrieve_examples_with_counts(
        class_embed['activations'], cur_activations
    )

    if not votes_by_index:
        return None, {}

    winning_index = votes_by_index[0][0]
    predicted_label = class_embed['int_to_str'][winning_index]
    label_vote_counts = {
        class_embed['int_to_str'][index]: count 
        for index, count in votes_by_index
    }

    return predicted_label, label_vote_counts


def retrieve_examples_spoof(sample_activations, cur_activation):
    """
    Spoofing detection vote counting.

    Args:
        sample_activations: (num_classes, num_heads, hidden_dim)
        cur_activation: (num_heads, hidden_dim)

    Returns:
        tuple: (predicted_class_index, vote_counts dict, total_heads)
    """
    all_sample = []
    for i in range(sample_activations.shape[1]):
        scores = F.cosine_similarity(
            sample_activations[:, i, :], cur_activation[i, :], dim=-1
        )
        all_sample.append(scores.argmax(dim=0).item())

    counter = Counter(all_sample)
    total_heads = len(all_sample)
    
    predicted_class_index = counter.most_common(1)[0][0] if counter else 0
    return predicted_class_index, dict(counter), total_heads


def mllm_classify_spoof(inputs, model, class_embed):
    """
    Classify for spoofing detection with confidence score.

    Args:
        inputs: Input item to classify
        model: ModelHelper instance
        class_embed: Dict from mllm_encode()

    Returns:
        tuple: (predicted_label, confidence_score)
    """
    cur_activations = get_query_activations(
        [inputs], model, class_embed['top_heads']
    ).squeeze(dim=0)
    predicted_class_index, vote_counts, total_heads = retrieve_examples_spoof(
        class_embed['activations'], cur_activations
    )
    
    predicted_label = class_embed['int_to_str'][predicted_class_index]
    votes_for_predicted = vote_counts.get(predicted_class_index, 0)
    confidence_score = votes_for_predicted / total_heads if total_heads > 0 else 0.0
    
    if predicted_label.lower() == "spoof":
        final_confidence = confidence_score
    else:
        final_confidence = 1.0 - confidence_score
    
    return predicted_label, final_confidence
