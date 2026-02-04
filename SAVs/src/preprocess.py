"""
Dataset preprocessing utilities for CALM.

Provides functions for loading datasets and formatting data for audio models.
"""
import json
import random


def open_data(dataset_name, path):
    """
    Load a dataset from a JSON file.

    Args:
        dataset_name: Name of the dataset
        path: Path to the JSON file

    Returns:
        List of dataset items, or None if unsupported
    """
    supported_datasets = [
        "vgg_sound", "vgg_sound_qa", "esc_mcq", 
        "audioset", "LA_spoof", "mlaad"
    ]

    with open(path, 'r') as json_file:
        if dataset_name in supported_datasets:
            dataset = json.load(json_file)
            if isinstance(dataset, dict):
                dataset = list(dataset.values())
        else:
            return None
    return dataset


def get_format_func(cur_dataset, label_csv=None):
    """
    Get the format function for a dataset.

    Args:
        cur_dataset: Dataset name
        label_csv: Optional pandas DataFrame with label mappings

    Returns:
        Format function or None if unsupported
    """
    if cur_dataset == "vgg_sound":
        return lambda all_data, cur_item=None, num_shot=0, model_helper=None, split="train", **kwargs: \
               format_vgg_sound(all_data, label_csv, cur_item, num_shot, model_helper, split)
    
    if cur_dataset in ["vgg_sound_qa", "esc_mcq", "audioset", "LA_spoof", "mlaad"]:
        def format_wrapper(all_data, cur_item=None, num_shot=0, model_helper=None, split="train", **kwargs):
            return format_audio_qa(all_data, cur_item, num_shot, model_helper, split)
        return format_wrapper
    
    return None


def format_vgg_sound(all_data, label_csv, cur_item=None, num_shot=0,
                     model_helper=None, split="train"):
    """
    Format VGGSound captioning data.

    Args:
        all_data: Full dataset for few-shot sampling
        label_csv: DataFrame with label mappings
        cur_item: Current item to process
        num_shot: Number of few-shot examples
        model_helper: ModelHelper instance (unused)
        split: Dataset split

    Returns:
        tuple: (qs_list, ans_list, audio_list, gt_label, question_id)
    """
    prompt = "Close-ended question: Write an audio caption describing the sound."
    
    if cur_item is None and all_data is not None:
        cur_item = random.choice(all_data)

    def label(mid):
        return label_csv.loc[label_csv.mid == mid, "display_name"].values[0]

    qs_list, ans_list, audio_list = [], [], []

    if all_data is None:
        qs_list.append(cur_item.get("question", prompt))
        audio_list.append(cur_item["wav"])
        ans_list.append(None)
        return qs_list, ans_list, audio_list, label(cur_item["labels"]), -1

    # Sample few-shot examples
    pool = [x for x in all_data if x is not cur_item]
    k = min(num_shot, len(pool))
    shots = random.sample(pool, k=k)
    
    for s in shots:
        qs_list.append(prompt)
        ans_list.append(label(s["labels"]))
        audio_list.append(s["wav"])
    
    # Add current item
    qs_list.append(cur_item.get("question", prompt))
    ans_list.append(None)
    audio_list.append(cur_item["wav"])
    
    assert len(qs_list) == num_shot + 1
    return qs_list, ans_list, audio_list, label(cur_item["labels"]), -1


def format_audio_qa(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    """
    Format audio QA/classification data.

    Works with VGGSound QA, ESC-50, AudioSet, LA-Spoof, and MLAAD datasets.

    Args:
        all_data: Full dataset for few-shot sampling
        cur_item: Current item to process
        num_shot: Number of few-shot examples
        model_helper: ModelHelper instance (unused)
        split: Dataset split

    Returns:
        tuple: (qs_list, ans_list, audio_list, gt_label, question_id)
    """
    base_prompt = "{} Answer with the class name."

    if cur_item is None:
        cur_item = random.choice(all_data)
    
    cur_question = cur_item.get('question', '')
    cur_label = cur_item.get('mapped_label', None)

    qs_list = []
    ans_list = []
    audio_list = []

    # Few-shot examples
    if num_shot > 0 and all_data is not None:
        pool = [x for x in all_data if x is not cur_item]
        k = min(num_shot, len(pool))
        samples = random.sample(pool, k)
        
        for sample in samples:
            prompt_text = base_prompt.format(sample.get('question', ''))
            qs_list.append(prompt_text)
            ans_list.append(sample.get('answer', None))
            audio_list.append(sample.get('wav'))

    # Add the final question
    final_prompt = base_prompt.format(cur_question)
    qs_list.append(final_prompt)
    ans_list.append(None)
    audio_list.append(cur_item.get('wav'))

    return qs_list, ans_list, audio_list, cur_label, -1
