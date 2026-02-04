"""
Prepare ASVspoof LA dataset for spoofing detection.

This script converts ASVspoof2019 LA protocol files to CALM-compatible JSON format.

Usage:
    python prepare_asvspoof.py --protocol PROTOCOL_FILE --audio_dir AUDIO_DIR --output OUTPUT_JSON
"""
import json
import argparse
import random


def prepare_asvspoof(protocol_file, audio_dir, output_file, sample_size=None):
    """
    Convert ASVspoof protocol file to JSON format.
    
    Args:
        protocol_file: Path to ASVspoof protocol file
        audio_dir: Directory containing audio files
        output_file: Output JSON path
        sample_size: Optional sample size for subset
    """
    with open(protocol_file, "r") as f:
        lines = f.readlines()

    data = []
    question = "Is this audio bonafide or spoofed?\nA. bonafide\nB. spoof"

    for line in lines:
        parts = line.strip().split(" ")
        speaker_id, utt_id = parts[0], parts[1]
        label = parts[-1]
        
        audio_path = f"{audio_dir}/{utt_id}.flac"
        mapped_label = "bonafide" if label == "bonafide" else "spoof"
        
        entry = {
            "wav": audio_path,
            "question": question,
            "answer": mapped_label,
            "mapped_label": mapped_label
        }
        data.append(entry)

    if sample_size:
        data = random.sample(data, min(sample_size, len(data)))

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Created dataset with {len(data)} samples")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ASVspoof dataset")
    parser.add_argument("--protocol", required=True, help="Protocol file path")
    parser.add_argument("--audio_dir", required=True, help="Audio directory")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--sample", type=int, help="Sample size (optional)")
    args = parser.parse_args()
    
    prepare_asvspoof(args.protocol, args.audio_dir, args.output, args.sample)
