"""
Filter a pseudolabel JSON by confidence threshold.

Usage:
    python -m scripts.filter_pseudolabels \
        --input data/vggsound/pseudolabeled_vggsound_mcq_qwen2_8trials.json \
        --thresholds 0.5 0.625 0.875 1.0
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter pseudolabels by confidence threshold")
    parser.add_argument("--input", "-i", required=True, help="Path to pseudolabeled JSON")
    parser.add_argument(
        "--thresholds", "-t", nargs="+", type=float, default=[0.5, 0.625, 0.875, 1.0],
        help="Confidence thresholds to filter at (default: 0.5 0.625 0.875 1.0)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    print(f"Loading {inp} ...")
    with open(inp) as f:
        data = json.load(f)
    print(f"  Total samples: {len(data):,}")

    for thresh in sorted(args.thresholds):
        filtered = [item for item in data if item["pseudo_label_confidence"] >= thresh]
        tag = f"{thresh:.3f}".replace(".", "")
        out_path = inp.with_name(f"{inp.stem}_conf{tag}{inp.suffix}")
        with open(out_path, "w") as f:
            json.dump(filtered, f, indent=2)
        print(f"  thresh >= {thresh:.3f}: {len(filtered):>7,} samples -> {out_path.name}")


if __name__ == "__main__":
    main()
