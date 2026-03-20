"""
Model helpers for CALM.

Supported models:
- Qwen2-Audio-7B-Instruct
- Qwen2.5-Omni-7B
- Qwen2-VL-7B-Instruct
- Phi-4-multimodal-instruct (5.6B)
"""
import torch
import numpy as np
import pandas as pd
import torchaudio
from .preprocess import get_format_func

_VGG_LABEL_CSV = (
    "/data/sls/scratch/yuangong/cav-mae/pretrained_model/"
    "datafiles/vggsound/class_labels_indices_vgg.csv"
)
_AS_LABEL_CSV = (
    "/data/sls/scratch/yuangong/cav-mae/pretrained_model/"
    "datafiles/audioset/class_labels_indices.csv"
)


class ModelHelper:
    """Base class for model helpers.

    Subclasses must implement ``insert_audio``.  ``forward`` and
    ``first_token_prob`` have sensible defaults shared across all models.
    ``generate`` provides a default for the Qwen family; Phi-4 overrides it.
    """

    @staticmethod
    def _is_image_path(path):
        """Return True if *path* points to a common image file."""
        if not isinstance(path, str) or not path.strip():
            return False
        return path.lower().split("?")[0].rsplit(".", 1)[-1] in {
            "jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp", "ppm", "pgm",
        }

    # ── common initialisation ──────────────────────────────────────────

    def _init_common(self, model, processor, cur_dataset, model_config):
        """Shared setup called by every subclass ``__init__``."""
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = model_config
        self.cur_dataset = cur_dataset

        self.all_heads = [
            (layer_idx, head_idx, -1)
            for layer_idx in range(model_config["n_layers"])
            for head_idx in range(model_config["n_heads"])
        ]

        if cur_dataset in ("vgg_sound", "vgg_sound_qa", "esc_mcq",
                           "audioset", "LA_spoof", "mlaad"):
            self.mapping_df = pd.read_csv(_VGG_LABEL_CSV)
        elif cur_dataset == "as":
            self.mapping_df = pd.read_csv(_AS_LABEL_CSV)
        else:
            self.mapping_df = None

        self.format_func = get_format_func(
            cur_dataset,
            label_csv=self.mapping_df if cur_dataset in ("vgg_sound", "as") else None,
        )
        self.nonspecial_idx = 0
        self.question_lookup = None

    # ── shared methods ─────────────────────────────────────────────────

    @torch.inference_mode()
    def forward(self, model_input, labels=None):
        with torch.no_grad():
            return self.model(**model_input)

    def first_token_prob(self, model_input, temperature=1.1, top_p=0.95):
        """Return (unnormalised, normalised) probs for MCQ tokens A/B/C/D."""
        with torch.no_grad():
            logits = self.model(**model_input).logits[:, -1, :]
            if temperature > 0:
                logits = logits / max(float(temperature), 1e-8)
            probs = torch.softmax(logits, dim=-1)

            def _token_ids(letter):
                ids = set()
                for variant in (letter, f" {letter}", f"{letter}.", f"{letter})"):
                    try:
                        ids.update(self.tokenizer.encode(variant, add_special_tokens=False))
                    except Exception:
                        pass
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(letter)
                    if tid is not None and tid != self.tokenizer.unk_token_id:
                        ids.add(tid)
                except Exception:
                    pass
                return list(ids)

            option_probs = []
            for letter in ("A", "B", "C", "D"):
                tids = _token_ids(letter)
                p = probs[:, tids].sum(dim=1).sum() if tids else torch.tensor(0.0)
                option_probs.append(p)

            unnorm = [float(p.item()) for p in option_probs]
            total = sum(p.item() for p in option_probs)
            if total > 0:
                option_probs = [p / total for p in option_probs]
            else:
                option_probs = [torch.tensor(0.25)] * 4
            norm = [float(p.item()) for p in option_probs]
            return unnorm, norm

    def generate(self, model_input, max_new_tokens, temperature=1.0, top_p=0.95):
        """Generate text (default: Qwen-family with random temperature sampling)."""
        import random
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        temperature = random.uniform(0.5, 1.5)
        seq = self.model.generate(
            **model_input,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_input.input_ids, seq)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def insert_audio(self, questions, answers, media_list):
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# Qwen2.5-Omni-7B
# ═══════════════════════════════════════════════════════════════════════

class Qwen2OmniHelper(ModelHelper):
    """Helper for Qwen2.5-Omni-7B (audio + image)."""

    def __init__(self, model, processor, cur_dataset):
        text_cfg = model.config.text_config
        attn_hook_names = sorted(
            [n for n, _ in model.named_modules()
             if n.startswith("model.layers.") and n.endswith(".self_attn.o_proj")],
            key=lambda n: int(n.split(".")[2]),
        )
        model_config = {
            "n_heads": text_cfg.num_attention_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": text_cfg.hidden_size,
            "name_or_path": getattr(model.config, "_name_or_path", None),
            "attn_hook_names": attn_hook_names,
            "layer_hook_names": [n.rsplit(".self_attn", 1)[0] for n in attn_hook_names],
            "mlp_hook_names": [n.replace(".self_attn", ".mlp") for n in attn_hook_names],
        }
        self._init_common(model, processor, cur_dataset, model_config)

    def insert_audio(self, questions, answers, media_list):
        """Prepare audio OR image inputs for Qwen2.5-Omni."""
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        valid_media = [
            m for m in (media_list or [])
            if m is not None and isinstance(m, str) and m.strip()
        ]
        no_media = len(valid_media) == 0
        use_images = (not no_media) and self._is_image_path(valid_media[0])

        messages = []
        for q, m_path, a in zip(
            questions,
            media_list or [None] * len(questions),
            answers or [None] * len(questions),
        ):
            content = []
            has_media = m_path and isinstance(m_path, str) and m_path.strip()
            if (not no_media) and has_media:
                if use_images:
                    content.append({"type": "image", "image": f"file://{m_path}"})
                else:
                    content.append({"type": "audio", "audio": m_path})
            content.append({"type": "text", "text": q})
            messages.append({"role": "user", "content": content})
            if a is not None:
                messages.append({"role": "assistant", "content": a})

        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if isinstance(formatted, list):
            formatted = "".join(formatted)

        if no_media:
            return self.processor(
                text=[formatted], return_tensors="pt",
            ).to(device)

        if use_images:
            pil_images = []
            for img_path in (media_list or []):
                if img_path and isinstance(img_path, str) and img_path.strip():
                    try:
                        pil_images.append(Image.open(img_path).convert("RGB"))
                    except Exception:
                        continue
            if pil_images:
                return self.processor(
                    text=[formatted], images=pil_images,
                    padding=True, return_tensors="pt",
                ).to(device)
            return self.processor(text=[formatted], return_tensors="pt").to(device)

        wav_np_list = []
        for wav in (media_list or []):
            try:
                waveform, sr = torchaudio.load(wav)
                wav_np = waveform.mean(0).cpu().numpy() if waveform.shape[0] > 1 else waveform.squeeze(0).cpu().numpy()
                wav_np_list.append(wav_np)
            except Exception:
                continue
        if not wav_np_list:
            return self.processor(text=[formatted], return_tensors="pt").to(device)
        return self.processor(
            text=[formatted], audio=wav_np_list, padding=True,
            sampling_rate=16000, return_tensors="pt", use_audio_in_video=False,
        ).to(device)


# ═══════════════════════════════════════════════════════════════════════
# Qwen2-Audio-7B-Instruct
# ═══════════════════════════════════════════════════════════════════════

class Qwen2AudioHelper(ModelHelper):
    """Helper for Qwen2-Audio-7B-Instruct (audio only)."""

    def __init__(self, model, processor, cur_dataset):
        text_cfg = model.config.text_config
        attn_hook_names = sorted(
            [n for n, _ in model.named_modules()
             if n.startswith("language_model.model.layers.") and n.endswith(".self_attn.o_proj")],
            key=lambda n: int(n.split(".")[3]),
        )
        model_config = {
            "n_heads": text_cfg.num_attention_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": text_cfg.hidden_size,
            "name_or_path": getattr(model.config, "_name_or_path", None),
            "attn_hook_names": attn_hook_names,
            "layer_hook_names": [n.rsplit(".self_attn", 1)[0] for n in attn_hook_names],
            "mlp_hook_names": [n.replace(".self_attn", ".mlp") for n in attn_hook_names],
        }
        self._init_common(model, processor, cur_dataset, model_config)

    def insert_audio(self, questions, answers, audio_list):
        """Prepare audio inputs for Qwen2-Audio."""
        if not audio_list or audio_list[0] is None or (isinstance(audio_list[0], str) and not audio_list[0]):
            messages = []
            for q, a in zip(questions, answers or [None] * len(questions)):
                messages.append({"role": "user", "content": [{"type": "text", "text": q}]})
                if a is not None:
                    messages.append({"role": "assistant", "content": a})
            formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(formatted, list):
                formatted = "".join(formatted)
            return self.processor(text=[formatted], padding=True, return_tensors="pt").to("cuda")

        wav_np_list = []
        for audio in audio_list:
            try:
                waveform, sr = torchaudio.load(audio)
                wav_np = waveform.mean(0).cpu().numpy() if waveform.shape[0] > 1 else waveform.cpu().numpy()
                wav_np_list.append(wav_np)
            except Exception:
                continue

        if not wav_np_list:
            messages = []
            for q, a in zip(questions, answers or [None] * len(questions)):
                messages.append({"role": "user", "content": [{"type": "text", "text": q}]})
                if a is not None:
                    messages.append({"role": "assistant", "content": a})
            formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(formatted, list):
                formatted = "".join(formatted)
            return self.processor(text=[formatted], padding=True, return_tensors="pt").to("cuda")

        messages = []
        for q, wav, a in zip(questions, audio_list, answers or [None] * len(questions)):
            messages.append({
                "role": "user",
                "content": [{"type": "audio", "audio": wav}, {"type": "text", "text": q}],
            })
            if a is not None:
                messages.append({"role": "assistant", "content": a})

        formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(formatted, list):
            formatted = "".join(formatted)

        wav_np_list = []
        for wav in audio_list:
            try:
                waveform = torchaudio.load(wav)[0]
                wav_np_list.append(waveform.mean(0).cpu().numpy())
            except Exception:
                continue

        return self.processor(
            text=[formatted], audio=wav_np_list, padding=True,
            sampling_rate=16000, return_tensors="pt", use_audio_in_video=False,
        ).to("cuda")

    def generate(self, model_input, max_new_tokens, temperature=1.1, top_p=0.95):
        """Generate with fixed temperature (no random override)."""
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        seq = self.model.generate(
            **model_input,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_input.input_ids, seq)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]


# ═══════════════════════════════════════════════════════════════════════
# Qwen2-VL-7B-Instruct
# ═══════════════════════════════════════════════════════════════════════

class Qwen2VLHelper(ModelHelper):
    """Helper for Qwen2-VL-7B-Instruct (image only)."""

    def __init__(self, model, processor, cur_dataset):
        cfg = model.config
        text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
        attn_hook_names = sorted(
            [n for n, _ in model.named_modules()
             if n.startswith("model.layers.") and n.endswith(".self_attn.o_proj")],
            key=lambda n: int(n.split(".")[2]),
        )
        model_config = {
            "n_heads": text_cfg.num_attention_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": text_cfg.hidden_size,
            "name_or_path": getattr(cfg, "_name_or_path", None),
            "attn_hook_names": attn_hook_names,
            "layer_hook_names": [n.rsplit(".self_attn", 1)[0] for n in attn_hook_names],
            "mlp_hook_names": [n.replace(".self_attn.o_proj", ".mlp.down_proj") for n in attn_hook_names],
        }
        self._init_common(model, processor, cur_dataset, model_config)

    def insert_audio(self, questions, answers, media_list):
        """Prepare image inputs for Qwen2-VL."""
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        no_media = (not media_list) or all(
            (m is None) or (isinstance(m, str) and m.strip() == "") for m in media_list
        )

        messages = []
        for q, img_path, a in zip(
            questions,
            media_list or [None] * len(questions),
            answers or [None] * len(questions),
        ):
            content = []
            if (not no_media) and img_path:
                content.append({"type": "image", "image": f"file://{img_path}"})
            content.append({"type": "text", "text": q})
            messages.append({"role": "user", "content": content})
            if a is not None:
                messages.append({"role": "assistant", "content": a})

        formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(formatted, list):
            formatted = "".join(formatted)

        if not no_media:
            images = []
            for img_path in media_list:
                if img_path and isinstance(img_path, str) and img_path.strip():
                    try:
                        images.append(Image.open(img_path).convert("RGB"))
                    except Exception:
                        continue
            if images:
                return self.processor(
                    text=[formatted], images=images, padding=True, return_tensors="pt",
                ).to(device)
        return self.processor(text=[formatted], return_tensors="pt").to(device)


# ═══════════════════════════════════════════════════════════════════════
# Phi-4-multimodal-instruct
# ═══════════════════════════════════════════════════════════════════════

class Phi4MultimodalHelper(ModelHelper):
    """Helper for Phi-4-multimodal-instruct (5.6B, audio + image)."""

    def __init__(self, model, processor, generation_config, cur_dataset):
        self.generation_config = generation_config

        cfg = model.config
        attn_hook_names = [
            n for n, _ in model.named_modules()
            if n.endswith(".self_attn.o_proj")
            and any(c.isdigit() for c in n.split(".")[-3])
        ]
        attn_hook_names.sort(key=lambda n: int([s for s in n.split(".") if s.isdigit()][0]))

        model_config = {
            "n_heads": cfg.num_attention_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": cfg.hidden_size,
            "name_or_path": getattr(cfg, "_name_or_path", None),
            "attn_hook_names": attn_hook_names,
            "layer_hook_names": [n.rsplit(".self_attn", 1)[0] for n in attn_hook_names],
            "mlp_hook_names": [n.replace(".self_attn.o_proj", ".mlp.down_proj") for n in attn_hook_names],
        }
        self._init_common(model, processor, cur_dataset, model_config)

    def insert_audio(self, questions, answers, media_list):
        """Prepare audio OR image inputs for Phi-4-multimodal."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"

        valid_media = [
            m for m in (media_list or [])
            if m is not None and isinstance(m, str) and m.strip()
        ]
        no_media = len(valid_media) == 0
        use_images = (not no_media) and self._is_image_path(valid_media[0])

        if use_images:
            from PIL import Image
            parts, img_counter = [], 1
            for q, img_path, a in zip(
                questions,
                media_list or [None] * len(questions),
                answers or [None] * len(questions),
            ):
                has_img = img_path and isinstance(img_path, str) and img_path.strip()
                if has_img:
                    parts.append(f"{user_prompt}<|image_{img_counter}|>{q}{prompt_suffix}")
                    img_counter += 1
                else:
                    parts.append(f"{user_prompt}{q}{prompt_suffix}")
                parts.append(f"{assistant_prompt}{a}{prompt_suffix}" if a is not None else assistant_prompt)
            prompt = "".join(parts)

            pil_images = []
            for img_path in (media_list or []):
                if img_path and isinstance(img_path, str) and img_path.strip():
                    try:
                        pil_images.append(Image.open(img_path).convert("RGB"))
                    except Exception:
                        continue
            if pil_images:
                return self.processor(text=prompt, images=pil_images, return_tensors="pt").to(device)
            return self.processor(text=prompt, return_tensors="pt").to(device)

        # Audio branch
        parts = []
        for q, wav_path, a in zip(
            questions,
            media_list or [None] * len(questions),
            answers or [None] * len(questions),
        ):
            tag = f"<|audio_1|>" if (not no_media) and wav_path else ""
            parts.append(f"{user_prompt}{tag}{q}{prompt_suffix}")
            parts.append(f"{assistant_prompt}{a}{prompt_suffix}" if a is not None else assistant_prompt)
        prompt = "".join(parts)

        if not no_media:
            audio_tuples = []
            for wav_path in (media_list or []):
                if wav_path is None or (isinstance(wav_path, str) and not wav_path.strip()):
                    continue
                try:
                    waveform, sr = torchaudio.load(wav_path)
                    wav_np = waveform.mean(0).cpu().numpy() if waveform.shape[0] > 1 else waveform.squeeze(0).cpu().numpy()
                    audio_tuples.append((wav_np, sr))
                except Exception:
                    continue
            if audio_tuples:
                return self.processor(text=prompt, audios=audio_tuples, return_tensors="pt").to(device)
        return self.processor(text=prompt, return_tensors="pt").to(device)

    def generate(self, model_input, max_new_tokens, temperature=1.0, top_p=0.95):
        """Generate with Phi-4 generation_config."""
        import random
        temperature = random.uniform(0.5, 1.5)
        seq = self.model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_input["input_ids"], seq)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
