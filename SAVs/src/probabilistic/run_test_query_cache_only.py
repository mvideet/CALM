"""
Run ONLY the "Precomputing query activations (cache)" loop on the TEST split.

This is intentionally minimal and is meant for your manual workflow:
  - run this on the test dataset
  - if it stalls, the last printed "about to process" line tells you the bad file
  - remove that file from the dataset / delete the mp4
  - rerun until it completes

It does NOT compute prototypes / reliabilities / weights.
It only calls `get_query_activations(...)` for each test item and (optionally) saves them.
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
from tqdm import tqdm

from .prwe_utils import load_model, get_query_activations
from ..preprocess import open_data


def _resolve_video_path(it: dict) -> str:
    # Mirror preprocess.py logic: if "video" is missing, construct from "wav"
    vp = it.get("video")
    if vp:
        return str(vp)
    wav = it.get("wav")
    if not wav:
        return ""
    base = os.path.splitext(os.path.basename(str(wav)))[0]
    return f"/data/sls/placesaudio/datasets/VGGSound/video/{base}.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True, help="qwen2.5_omni or qwen2-audio-instruct")
    ap.add_argument("--data_name", type=str, default="vgg_sound_qa", help="dataset key for open_data/format_func")
    ap.add_argument("--test_path", type=str, required=True, help="Path to TEST json")
    ap.add_argument("--audio_or_video", type=str, default="video", choices=["audio", "video"], help="Force modality")
    ap.add_argument("--n_trials", type=int, default=20, help="Trials per item (passed through)")
    ap.add_argument("--last_n_tokens", type=int, default=1, help="If >1, average last N tokens")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only run first N items")
    ap.add_argument("--start_idx", type=int, default=0, help="Start index into test set (resume)")
    ap.add_argument("--save_dir", type=str, default="", help="Optional: save each activation tensor to this dir")
    ap.add_argument("--flush_every", type=int, default=1, help="Flush stdout every N items")
    args = ap.parse_args()

    model_helper = load_model(args.model_name, args.data_name)
    test_data = open_data(args.data_name, args.test_path)

    if args.start_idx and args.start_idx > 0:
        test_data = test_data[args.start_idx :]
    if args.limit and args.limit > 0:
        test_data = test_data[: args.limit]

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    heads = list(model_helper.all_heads)
    lastN = int(args.last_n_tokens)
    n_trials = int(args.n_trials)

    print("=" * 80)
    print("TEST QUERY CACHE ONLY")
    print(f"model_name={args.model_name}")
    print(f"data_name={args.data_name}")
    print(f"test_path={args.test_path}")
    print(f"audio_or_video={args.audio_or_video}")
    print(f"n_trials={n_trials}")
    print(f"last_n_tokens={lastN}")
    print(f"count={len(test_data)} (after start_idx/limit)")
    if args.save_dir:
        print(f"save_dir={args.save_dir}")
    print("=" * 80)
    sys.stdout.flush()

    for local_i, it in enumerate(tqdm(test_data, desc="Precomputing query activations (cache)")):
        global_i = args.start_idx + local_i

        vp = _resolve_video_path(it) if args.audio_or_video == "video" else ""
        wav = it.get("wav", "")
        label = it.get("mapped_label", it.get("label", ""))

        # This print is the key for your workflow: when it stalls, this is the last file.
        print(f"[about-to-process] idx={global_i} label={label} wav={wav} video={vp}")
        sys.stdout.flush()

        qa = get_query_activations(
            [it],
            model_helper,
            heads,
            last_n_tokens=lastN,
            audio_or_video=args.audio_or_video,
            n_trials=n_trials,
        )

        if qa is None:
            print(f"[WARN] idx={global_i} returned qa=None (skipping)")
            continue

        if args.save_dir:
            # Save per-item tensor so you can resume without recomputing if desired.
            out_path = os.path.join(args.save_dir, f"test_{global_i:06d}.pt")
            torch.save(qa.detach().cpu(), out_path)

        if args.flush_every > 0 and (local_i + 1) % args.flush_every == 0:
            sys.stdout.flush()

        # free GPU memory aggressively (you’re doing lots of decoding)
        del qa
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

