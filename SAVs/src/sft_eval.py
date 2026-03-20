"""
Direct inference evaluation for SFT-finetuned Qwen2-Audio model.

Supports both LoRA and full SFT checkpoints.

Usage (LoRA):
    python -m src.sft_eval \
        --checkpoint_dir ./checkpoints/sft_vggsound_lora \
        --test_path /path/to/vggsound_mcq_test.json

Usage (full SFT):
    python -m src.sft_eval \
        --checkpoint_dir ./checkpoints/sft_vggsound_full \
        --test_path /path/to/vggsound_mcq_test.json \
        --full_sft
"""
import argparse
import json
import os
from datetime import datetime

import torch
import torchaudio
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def load_sft_model(checkpoint_dir, full_sft=False):
    """Load SFT checkpoint: full model or base + LoRA adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if full_sft:
        processor = AutoProcessor.from_pretrained(checkpoint_dir)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device},
        )
        model.tie_weights()
    else:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device},
        )
        base_model.tie_weights()
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model = model.merge_and_unload()

    model.eval()
    model.requires_grad_(False)
    return model, processor, device


def generate_answer(model, processor, sample, device):
    """Generate a free-form answer for a single test sample."""
    wav_path = sample["wav"]
    question = sample["question"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav_path},
                {"type": "text", "text": question},
            ],
        },
    ]

    formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if isinstance(formatted, list):
        formatted = "".join(formatted)

    waveform, _sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        wav_np = waveform.mean(0).cpu().numpy()
    else:
        wav_np = waveform.squeeze(0).cpu().numpy()

    inputs = processor(
        text=[formatted],
        audio=[wav_np],
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
        use_audio_in_video=False,
    ).to(device)

    eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    with torch.no_grad():
        seq = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=20,
            eos_token_id=eos_id,
        )

    generated_ids = seq[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return text


def parse_answer_letter(text):
    """Extract A/B/C/D from the first character. Anything else is wrong."""
    text = text.strip()
    if text and text[0] in "ABCD":
        return text[0]
    return ""


def normalize_spoof_label(label: str) -> str:
    """Normalize spoofing labels to 'bonafide' or 'spoof'."""
    label_lower = (label or "").lower().strip()
    if any(k in label_lower for k in ["spoof", "fake", "synthetic", "artificial"]):
        return "spoof"
    if any(k in label_lower for k in ["bonafide", "genuine", "real", "authentic"]):
        return "bonafide"
    return "bonafide"


def parse_spoof_answer(text):
    """Extract bonafide/spoof from generated text. Returns normalized label or empty string."""
    text_lower = (text or "").lower().strip()
    if "spoof" in text_lower:
        return "spoof"
    if "bonafide" in text_lower or "genuine" in text_lower or "real" in text_lower:
        return "bonafide"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-finetuned Qwen2-Audio")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--full_sft", action="store_true", help="Checkpoint is full fine-tuned model, not LoRA")
    parser.add_argument("--spoof", action="store_true", help="Spoof detection mode: parse bonafide/spoof from text instead of A/B/C/D")
    parser.add_argument("--debug_generations", action="store_true", help="Print and save raw model outputs for debugging")
    parser.add_argument("--debug_n", type=int, default=100, help="Number of samples to print when --debug_generations (default: 100)")
    parser.add_argument("--log_every", type=int, default=100, help="Print running accuracy every N samples")
    args = parser.parse_args()

    mode = "full SFT" if args.full_sft else "LoRA"
    print("=" * 60)
    print(f"SFT Evaluation — Qwen2-Audio ({mode})")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Test data: {args.test_path}")
    print()

    print(f"Loading model ({mode})...")
    model, processor, device = load_sft_model(args.checkpoint_dir, full_sft=args.full_sft)
    print("  Model loaded and merged.")
    print()

    print("Loading test data...")
    with open(args.test_path) as f:
        test_data = json.load(f)
    if isinstance(test_data, dict):
        test_data = list(test_data.values())
    print(f"  Test samples: {len(test_data)}")
    print()

    predictions = []
    ground_truths = []
    pred_labels = []
    gt_labels = []
    debug_records = [] if args.debug_generations else None
    errors = 0

    parse_desc = "bonafide/spoof" if args.spoof else "first char A/B/C/D"
    print(f"Evaluating (generate + parse {parse_desc})...")
    for i, sample in enumerate(tqdm(test_data, desc="Inference")):
        gt_raw = sample.get("answer") or sample.get("correct_answer")
        gt_label = sample.get("mapped_label", gt_raw)
        if isinstance(gt_label, list):
            gt_label = gt_label[0] if gt_label else str(gt_raw)
        gt_norm = normalize_spoof_label(gt_label) if args.spoof else gt_raw

        try:
            raw_output = generate_answer(model, processor, sample, device)
            if args.spoof:
                pred_parsed = parse_spoof_answer(raw_output)
            else:
                pred_parsed = parse_answer_letter(raw_output)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [error] {e}")
            pred_parsed = ""
            raw_output = ""

        if args.spoof:
            pred_norm = pred_parsed if pred_parsed else ""
            gt_for_acc = gt_norm
            pred_for_acc = pred_norm
            gt_labels.append(gt_norm)
            pred_labels.append(pred_norm if pred_norm else raw_output or "(unparseable)")
        else:
            pred_for_acc = pred_parsed
            gt_for_acc = gt_raw
            options = sample.get("options", [])
            letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
            gt_idx = letter_to_idx.get(gt_raw, -1)
            pred_idx = letter_to_idx.get(pred_parsed, -1)
            gt_labels.append(options[gt_idx] if 0 <= gt_idx < len(options) else gt_label)
            pred_labels.append(options[pred_idx] if 0 <= pred_idx < len(options) else raw_output)

        predictions.append(pred_for_acc)
        ground_truths.append(gt_for_acc)

        if debug_records is not None:
            correct = pred_for_acc == gt_for_acc
            debug_records.append({
                "idx": i,
                "gt": gt_for_acc,
                "raw": raw_output,
                "parsed": pred_for_acc,
                "correct": correct,
            })

        if args.log_every > 0 and (i + 1) % args.log_every == 0:
            running_acc = accuracy_score(ground_truths, predictions)
            if args.spoof:
                _, _, running_f1_per_class, _ = precision_recall_fscore_support(
                    ground_truths, predictions, labels=["bonafide", "spoof"], zero_division=0
                )
                running_macro_f1 = (running_f1_per_class[0] + running_f1_per_class[1]) / 2
                tqdm.write(f"  [{i + 1}/{len(test_data)}] macro_f1={running_macro_f1:.4f} acc={running_acc:.4f} | bonafide_f1={running_f1_per_class[0]:.4f} spoof_f1={running_f1_per_class[1]:.4f}")
            else:
                tqdm.write(f"  [{i + 1}/{len(test_data)}] running accuracy: {running_acc:.4f} ({running_acc * 100:.2f}%)")

    accuracy = accuracy_score(ground_truths, predictions)

    print()
    print("=" * 60)
    print("SFT EVALUATION RESULTS")
    print("=" * 60)
    if args.spoof:
        prec, rec, f1_per_class, supp = precision_recall_fscore_support(
            ground_truths, predictions, labels=["bonafide", "spoof"], zero_division=0
        )
        macro_f1 = (f1_per_class[0] + f1_per_class[1]) / 2
        print(f"Macro F1: {macro_f1:.4f} (primary metric for imbalanced spoof data)")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  Bonafide F1: {f1_per_class[0]:.4f} (recall={rec[0]:.4f}, n={int(supp[0])})")
        print(f"  Spoof F1:    {f1_per_class[1]:.4f} (recall={rec[1]:.4f}, n={int(supp[1])})")
    else:
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Samples: {len(test_data)}, Errors: {errors}")
    print()
    print("Classification Report (by class name):")
    print(classification_report(gt_labels, pred_labels, zero_division=0))

    if debug_records:
        wrong = [r for r in debug_records if not r["correct"]]
        correct_records = [r for r in debug_records if r["correct"]]
        print()
        print("=" * 60)
        print("DEBUG: Sample generations (gt | parsed | raw)")
        print("=" * 60)
        n_print = min(args.debug_n, len(debug_records))
        n_wrong = min(n_print // 2, len(wrong))
        n_correct = min(n_print - n_wrong, len(correct_records))
        printed = 0
        for r in wrong[:n_wrong]:
            tqdm.write(f"  [WRONG] idx={r['idx']} gt={r['gt']!r} parsed={r['parsed']!r} raw={r['raw']!r}")
            printed += 1
        for r in correct_records[:n_correct]:
            tqdm.write(f"  [OK]    idx={r['idx']} gt={r['gt']!r} parsed={r['parsed']!r} raw={r['raw']!r}")
            printed += 1
        print(f"  (showing {printed} of {len(debug_records)} total; {len(wrong)} wrong)")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = os.path.join(args.output_dir, f"debug_generations_{timestamp}.json")
            with open(debug_file, "w") as f:
                json.dump(debug_records, f, indent=0)
            print(f"  Full generations saved to: {debug_file}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "sft_eval_spoof" if args.spoof else "sft_eval_vggsound"
        result_file = os.path.join(args.output_dir, f"{prefix}_{timestamp}.txt")
        with open(result_file, "w") as f:
            f.write(f"Checkpoint: {args.checkpoint_dir}\n")
            f.write(f"Test data: {args.test_path}\n")
            if args.spoof:
                f.write(f"Macro F1: {macro_f1:.4f} (primary)\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Bonafide F1: {f1_per_class[0]:.4f} (recall={rec[0]:.4f}, n={int(supp[0])})\n")
                f.write(f"Spoof F1: {f1_per_class[1]:.4f} (recall={rec[1]:.4f}, n={int(supp[1])})\n")
            else:
                f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Samples: {len(test_data)}, Errors: {errors}\n\n")
            f.write(classification_report(gt_labels, pred_labels, zero_division=0))
        print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
