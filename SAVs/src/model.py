"""
Model helpers for CALM (Class-conditional Attention vectors for audio Language Models).

Supported models:
- Qwen2-Audio-7B-Instruct
- Qwen2.5-Omni-7B
"""
import torch
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.functional as F
from .preprocess import get_format_func


class ModelHelper:
    """Base class for model helpers."""
    
    def __init__(self):
        """
        Attributes set by subclasses:
            model: The loaded model
            tokenizer: The loaded tokenizer
            processor: The audio processor
            model_config: Model architecture config with:
                - n_heads: Number of attention heads
                - n_layers: Number of layers
                - resid_dim: Hidden size
                - name_or_path: Model name or path
                - attn_hook_names: List of attention output projection hook names
                - layer_hook_names: List of layer hook names
                - mlp_hook_names: List of MLP projection hook names
            format_func: The format function for the current dataset
            cur_dataset: Name of the current dataset
            all_heads: List of (layer, head, -1) tuples for attention analysis
        """
        pass

    def insert_audio(self, questions, answers, audio_list):
        """
        Prepare audio inputs for the model.
        
        Args:
            questions: List of question strings
            answers: List of answer strings (None for inference)
            audio_list: List of audio file paths
            
        Returns:
            Model inputs ready for forward/generate
        """
        pass

    def forward(self, model_input, labels=None):
        """Forward pass wrapper."""
        pass

    def generate(self, model_input, max_new_tokens):
        """Generate text from audio input."""
        pass


class Qwen2OmniHelper(ModelHelper):
    """Helper for Qwen2.5-Omni-7B model."""
    
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer

        text_cfg = self.model.config.text_config
        n_heads = text_cfg.num_attention_heads
        n_layers = text_cfg.num_hidden_layers
        hidden = text_cfg.hidden_size
        
        attn_hook_names = [
            name for name, _ in self.model.named_modules()
            if name.startswith("model.layers.")
            and name.endswith(".self_attn.o_proj")
        ]
        attn_hook_names.sort(key=lambda n: int(n.split(".")[2]))
        
        layer_hook_names = [n.rsplit(".self_attn", 1)[0] for n in attn_hook_names]
        mlp_hook_names = [n.replace(".self_attn", ".mlp") for n in attn_hook_names]

        self.model_config = {
            "n_heads": n_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": hidden,
            "name_or_path": getattr(self.model.config, "_name_or_path", None),
            "attn_hook_names": attn_hook_names,
            "layer_hook_names": layer_hook_names,
            "mlp_hook_names": mlp_hook_names,
        }

        self.all_heads = [
            (layer_idx, head_idx, -1)
            for layer_idx in range(len(attn_hook_names))
            for head_idx in range(n_heads)
        ]

        # Load label mappings for supported datasets
        if cur_dataset in ['vgg_sound', 'vgg_sound_qa', 'esc_mcq', 'audioset', 'LA_spoof', 'mlaad']:
            self.mapping_df = pd.read_csv("/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/class_labels_indices_vgg.csv")
        elif cur_dataset == 'as':
            self.mapping_df = pd.read_csv("/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/class_labels_indices.csv")
        else:
            self.mapping_df = None
            
        self.format_func = get_format_func(cur_dataset, label_csv=self.mapping_df if cur_dataset in ['vgg_sound', 'as'] else None)
        self.cur_dataset = cur_dataset
        self.nonspecial_idx = 0
        self.question_lookup = None

    def insert_audio(self, questions, answers, audio_list):
        """Prepare audio inputs for Qwen2.5-Omni."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        no_audio = (not audio_list) or all(
            (a is None) or (isinstance(a, str) and a.strip() == "") 
            for a in audio_list
        )

        messages = []
        for q, a_path, a in zip(questions, (audio_list or [None] * len(questions)), answers or [None] * len(questions)):
            content = []
            if (not no_audio) and a_path:
                content.append({"type": "audio", "audio": a_path})
            content.append({"type": "text", "text": q})
            messages.append({"role": "user", "content": content})
            if a is not None:
                messages.append({"role": "assistant", "content": a})

        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(formatted, list):
            formatted = "".join(formatted)

        if not no_audio:
            wav_np_list = []
            for wav in audio_list:
                try:
                    waveform, sr = torchaudio.load(wav)
                    if waveform.shape[0] > 1:
                        wav_np = waveform.mean(0).cpu().numpy()
                    else:
                        wav_np = waveform.squeeze(0).cpu().numpy()
                    wav_np_list.append(wav_np)
                except Exception as e:
                    continue

            if len(wav_np_list) == 0:
                inputs = self.processor(
                    text=[formatted],
                    return_tensors="pt",
                ).to(device)
            else:
                inputs = self.processor(
                    text=[formatted],
                    audio=wav_np_list,
                    padding=True,
                    sampling_rate=16000,
                    return_tensors="pt",
                    use_audio_in_video=False,
                ).to(device)
        else:
            inputs = self.processor(
                text=[formatted],
                return_tensors="pt",
            ).to(device)

        return inputs

    @torch.inference_mode()
    def forward(self, model_input, labels=None):
        with torch.no_grad():
            result = self.model(**model_input)
        return result

    def generate(self, model_input, max_new_tokens, temperature=1.0, top_p=0.95):
        import random
        END = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        temperature = random.uniform(0.5, 1.5)
        seq = self.model.generate(
            **model_input,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=END
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_input.input_ids, seq)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def first_token_prob(self, model_input, temperature=1.1, top_p=0.95):
        """Get probabilities for multiple choice tokens A/B/C/D."""
        with torch.no_grad():
            logits = self.model(**model_input).logits[:, -1, :]
            if temperature > 0:
                logits = logits / max(float(temperature), 1e-8)
            probs = torch.softmax(logits, dim=-1)

            def get_token_ids(letter):
                token_ids = set()
                for var in [letter, f" {letter}", f"{letter}.", f"{letter})"]:
                    try:
                        token_ids.update(self.processor.tokenizer.encode(var, add_special_tokens=False))
                    except:
                        pass
                try:
                    tid = self.processor.tokenizer.convert_tokens_to_ids(letter)
                    if tid is not None and tid != self.processor.tokenizer.unk_token_id:
                        token_ids.add(tid)
                except:
                    pass
                return list(token_ids)

            option_probs = []
            for letter in ["A", "B", "C", "D"]:
                token_ids = get_token_ids(letter)
                prob = probs[:, token_ids].sum(dim=1).sum() if token_ids else torch.tensor(0.0)
                option_probs.append(prob)

            unnormalized_prob = [float(p.item()) for p in option_probs]

            total = sum(p.item() for p in option_probs)
            if total > 0:
                option_probs = [p / total for p in option_probs]
            else:
                option_probs = [torch.tensor(0.25) for _ in range(4)]

            normalized_prob = [float(p.item()) for p in option_probs]
            return unnormalized_prob, normalized_prob


class Qwen2AudioHelper(ModelHelper):
    """Helper for Qwen2-Audio-7B-Instruct model."""
    
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer

        text_cfg = self.model.config.text_config
        n_heads = text_cfg.num_attention_heads
        n_layers = text_cfg.num_hidden_layers
        hidden = text_cfg.hidden_size
        
        attn_hook_names = [
            name for name, _ in self.model.named_modules()
            if name.startswith("language_model.model.layers.")
            and name.endswith(".self_attn.o_proj")
        ]
        attn_hook_names.sort(key=lambda n: int(n.split(".")[3]))
        layer_hook_names = [n.rsplit(".self_attn", 1)[0] for n in attn_hook_names]
        mlp_hook_names = [n.replace(".self_attn", ".mlp") for n in attn_hook_names]

        self.model_config = {
            "n_heads": n_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": hidden,
            "name_or_path": getattr(self.model.config, "_name_or_path", None),
            "attn_hook_names": attn_hook_names,
            "layer_hook_names": layer_hook_names,
            "mlp_hook_names": mlp_hook_names,
        }

        self.all_heads = [
            (layer_idx, head_idx, -1)
            for layer_idx in range(len(attn_hook_names))
            for head_idx in range(n_heads)
        ]

        # Load label mappings for supported datasets
        if cur_dataset in ['vgg_sound', 'vgg_sound_qa', 'esc_mcq', 'audioset', 'LA_spoof', 'mlaad']:
            self.mapping_df = pd.read_csv("/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/class_labels_indices_vgg.csv")
        elif cur_dataset == 'as':
            self.mapping_df = pd.read_csv("/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/class_labels_indices.csv")
        else:
            self.mapping_df = None
            
        self.format_func = get_format_func(cur_dataset, label_csv=self.mapping_df if cur_dataset in ['vgg_sound', 'as'] else None)
        self.cur_dataset = cur_dataset
        self.nonspecial_idx = 0
        self.question_lookup = None

    def insert_audio(self, questions, answers, audio_list):
        """Prepare audio inputs for Qwen2-Audio."""
        # Handle empty audio list
        if not audio_list or audio_list[0] is None or (isinstance(audio_list[0], str) and not audio_list[0]):
            messages = []
            for q, a in zip(questions, answers or [None] * len(questions)):
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": q}],
                })
                if a is not None:
                    messages.append({"role": "assistant", "content": a})

            formatted = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(formatted, list):
                formatted = "".join(formatted)

            inputs = self.processor(
                text=[formatted],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            return inputs

        # Load audio files
        wav_np_list = []
        for audio in audio_list:
            try:
                waveform, sr = torchaudio.load(audio)
                if waveform.shape[0] > 1:
                    wav_np = waveform.mean(0).cpu().numpy()
                else:
                    wav_np = waveform.cpu().numpy()
                wav_np_list.append(wav_np)
            except Exception:
                continue

        # Fall back to text-only if all audio files failed
        if len(wav_np_list) == 0:
            messages = []
            for q, a in zip(questions, answers or [None] * len(questions)):
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": q}],
                })
                if a is not None:
                    messages.append({"role": "assistant", "content": a})

            formatted = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(formatted, list):
                formatted = "".join(formatted)

            inputs = self.processor(
                text=[formatted],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            return inputs

        # Build messages with audio
        messages = []
        for q, wav, a in zip(questions, audio_list, answers or [None] * len(questions)):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav},
                    {"type": "text", "text": q}
                ],
            })
            if a is not None:
                messages.append({"role": "assistant", "content": a})

        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(formatted, list):
            formatted = "".join(formatted)

        # Reload audio for processor
        wav_np_list = []
        for wav in audio_list:
            try:
                waveform = torchaudio.load(wav)[0]
                wav_np = waveform.mean(0).cpu().numpy()
                wav_np_list.append(wav_np)
            except Exception:
                continue

        inputs = self.processor(
            text=[formatted],
            audio=wav_np_list,
            padding=True,
            sampling_rate=16000,
            return_tensors="pt",
            use_audio_in_video=False,
        ).to("cuda")

        return inputs

    @torch.inference_mode()
    def forward(self, model_input, labels=None):
        with torch.no_grad():
            result = self.model(**model_input)
        return result

    def generate(self, model_input, max_new_tokens, temperature=1.1, top_p=0.95):
        END = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        seq = self.model.generate(
            **model_input,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=END
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_input.input_ids, seq)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def first_token_prob(self, model_input, temperature=1.1, top_p=0.95):
        """Get probabilities for multiple choice tokens A/B/C/D."""
        with torch.no_grad():
            logits = self.model(**model_input).logits
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / max(float(temperature), 1e-8)
            probs = torch.softmax(logits, dim=-1)
            
            a_tokens = self.processor.tokenizer.convert_tokens_to_ids("A")
            b_tokens = self.processor.tokenizer.convert_tokens_to_ids("B")
            c_tokens = self.processor.tokenizer.convert_tokens_to_ids("C")
            d_tokens = self.processor.tokenizer.convert_tokens_to_ids("D")
            
            a_prob = probs[:, a_tokens].sum()
            b_prob = probs[:, b_tokens].sum()
            c_prob = probs[:, c_tokens].sum()
            d_prob = probs[:, d_tokens].sum()
            
            unnormalized_prob = [float(a_prob.item()), float(b_prob.item()), float(c_prob.item()), float(d_prob.item())]
            
            total_prob = a_prob + b_prob + c_prob + d_prob
            a_prob = a_prob / total_prob
            b_prob = b_prob / total_prob
            c_prob = c_prob / total_prob
            d_prob = d_prob / total_prob
            
            normalized_prob = [float(a_prob.item()), float(b_prob.item()), float(c_prob.item()), float(d_prob.item())]
            return unnormalized_prob, normalized_prob
