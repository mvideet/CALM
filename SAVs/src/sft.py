"""
SFT (Supervised Fine-Tuning) for Qwen2-Audio on VGGSound.

Supports LoRA (default) or full fine-tuning.

Usage (LoRA):
    python -m src.sft \
        --train_path /path/to/vggsound_mcq_train_20shot.json \
        --output_dir ./checkpoints/sft_vggsound_lora \
        --epochs 1 --lr 2e-4 --lora_rank 16

Usage (full SFT):
    python -m src.sft \
        --train_path /path/to/vggsound_mcq_train_20shot.json \
        --output_dir ./checkpoints/sft_vggsound_full \
        --full_sft --epochs 1 --lr 1e-5
"""
import argparse
import json
import math
import os
import time

import torch
import torchaudio
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


class VGGSoundSFTDataset(Dataset):
    """Dataset that yields raw sample dicts for collation in the training loop."""

    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)
        if isinstance(self.data, dict):
            self.data = list(self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_sft_inputs(sample, processor):
    """
    Build tokenised model inputs + labels for a single sample.

    The target text is ``"<letter>. <class_name>"`` (e.g. ``"D. scuba diving"``).
    Prompt tokens are masked with -100 so only the answer is supervised.

    Supports both VGGSound format (answer) and AudioSet format (correct_answer).
    """
    wav_path = sample["wav"]
    question = sample["question"]
    answer_letter = sample.get("answer") or sample.get("correct_answer")
    mapped_label = sample["mapped_label"]
    if isinstance(mapped_label, list):
        mapped_label = mapped_label[0] if mapped_label else ""
    if answer_letter and len(answer_letter) == 1 and answer_letter in "ABCDEFGH":
        target_text = f"{answer_letter}. {mapped_label}"
    else:
        target_text = mapped_label

    messages_with_answer = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav_path},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": target_text},
    ]

    messages_prompt_only = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav_path},
                {"type": "text", "text": question},
            ],
        },
    ]

    full_text = processor.apply_chat_template(
        messages_with_answer, tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.apply_chat_template(
        messages_prompt_only, tokenize=False, add_generation_prompt=True
    )
    if isinstance(full_text, list):
        full_text = "".join(full_text)
    if isinstance(prompt_text, list):
        prompt_text = "".join(prompt_text)

    waveform, _sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        wav_np = waveform.mean(0).cpu().numpy()
    else:
        wav_np = waveform.squeeze(0).cpu().numpy()

    full_inputs = processor(
        text=[full_text],
        audio=[wav_np],
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
        use_audio_in_video=False,
    )

    prompt_inputs = processor(
        text=[prompt_text],
        audio=[wav_np],
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
        use_audio_in_video=False,
    )

    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    return full_inputs, labels


def main():
    parser = argparse.ArgumentParser(description="SFT with LoRA on Qwen2-Audio")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft_vggsound_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--full_sft", action="store_true", help="Full fine-tuning (all params) instead of LoRA")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many steps (default: full epoch)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    mode = "full SFT" if args.full_sft else "LoRA"
    print("=" * 60)
    print(f"SFT ({mode}) — Qwen2-Audio-7B-Instruct")
    print("=" * 60)
    print(f"Train data: {args.train_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Grad accum: {args.grad_accum}")
    if not args.full_sft:
        print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": device},
    )
    model.tie_weights()

    if args.full_sft:
        for p in model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full SFT: {trainable:,} trainable parameters")
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    print()

    print("Loading dataset...")
    dataset = VGGSoundSFTDataset(args.train_path)
    print(f"  Samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    total_steps = len(loader) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1.0, float(warmup_steps))
        progress = float(current_step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Total steps: {total_steps}")
    if args.max_steps is not None:
        print(f"Max steps: {args.max_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Effective batch size: {args.grad_accum}")
    print()

    model.train()
    global_step = 0
    running_loss = 0.0
    skipped = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        optimizer.zero_grad()

        for step, sample_batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            sample = {k: v[0] if isinstance(v, list) else v for k, v in sample_batch.items()}

            try:
                inputs, labels = build_sft_inputs(sample, processor)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [skip] {e}")
                continue

            input_dict = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**input_dict, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum
            global_step += 1

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step={global_step}  loss={avg_loss:.4f}  lr={lr_now:.2e}  elapsed={elapsed:.0f}s")
                running_loss = 0.0

            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}, stopping.")
                break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    elapsed_total = time.time() - t0
    print()
    print(f"Training complete in {elapsed_total:.0f}s")
    if skipped > 0:
        print(f"Skipped {skipped} samples due to errors")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if args.full_sft:
        print(f"Full model saved to: {args.output_dir}")
    else:
        print(f"LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
