"""
Pseudo-label quality analysis: accuracy, coverage, failure modes, and
sensitivity to the confidence threshold.

Addresses reviewer concern:
  "the self-supervised pseudo-labeling extension is underdeveloped, with
   limited analysis of failure modes, label noise, or sensitivity to the
   confidence threshold."

Usage:
    python -m scripts.analyze_pseudolabels \
        --input data/vggsound/pseudolabeled_vggsound_mcq_qwen2_8trials.json

    # Analyze all pseudolabel files at once:
    python -m scripts.analyze_pseudolabels \
        --input data/vggsound/pseudolabeled_vggsound_mcq_qwen2_8trials.json \
               data/esc/pseudolabeled_esc_40shot_mcq_qwen2_8trials.json \
               data/audioset/pseudolabeled_audioset_qwen2-audio-instruct_train_8trials.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"^(the )?(given |this )?audio (clip )?(belongs to (the )?(class (of )?)?|is )", "", text)
    text = re.sub(r"[^a-z0-9 _/]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_prediction_to_option(pred, options):
    """Try to match a raw trial prediction to one of the MCQ options."""
    pred_n = _normalize(pred)
    if not pred_n:
        return None

    best, best_score = None, 0
    for opt in options:
        opt_n = _normalize(opt)
        if pred_n == opt_n:
            return opt
        if opt_n in pred_n or pred_n in opt_n:
            score = len(opt_n)
            if score > best_score:
                best, best_score = opt, score
    return best


def _answer_letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


# ──────────────────────────────────────────────────────────────────────
# Determine GT and pseudo-label per item
# ──────────────────────────────────────────────────────────────────────

def extract_gt_and_pl(item):
    """
    Returns (ground_truth, pseudo_label, confidence).

    Handles three formats:
      1. VGGSound / ESC  (has trial_predictions + single-letter answer)
      2. AudioSet         (has correct_answer, no trial_predictions)
      3. Spoof            (has correct_answer, binary labels)
    """
    confidence = item["pseudo_label_confidence"]

    # --- Format 2/3: AudioSet or Spoof (has correct_answer) ---
    if "correct_answer" in item:
        gt = item["correct_answer"]
        pl = item.get("mapped_label", item.get("label", item.get("answer", "")))
        return (_normalize(gt), _normalize(pl), confidence)

    # --- Format 1: VGGSound / ESC (has trial_predictions) ---
    options = item.get("options", [])
    answer = item.get("answer", "")

    # Ground truth
    if len(answer) == 1 and answer.isalpha():
        idx = _answer_letter_to_index(answer)
        gt = options[idx] if idx < len(options) else answer
    else:
        gt = item.get("mapped_label", answer)

    # Pseudo-label = majority vote of trial predictions
    trial_preds = item.get("trial_predictions", [])
    vote_counter = Counter()
    for pred in trial_preds:
        matched = _match_prediction_to_option(pred, options)
        if matched is not None:
            vote_counter[matched] += 1

    if vote_counter:
        pl = vote_counter.most_common(1)[0][0]
    else:
        pl = ""

    return (_normalize(gt), _normalize(pl), confidence)


# ──────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────

def analyze(data, tag=""):
    """Run full threshold-sensitivity analysis on a pseudolabel dataset."""

    records = []
    for item in data:
        gt, pl, conf = extract_gt_and_pl(item)
        records.append({"gt": gt, "pl": pl, "conf": conf, "correct": gt == pl})

    total = len(records)
    if total == 0:
        print("  (empty dataset, skipping)")
        return

    # Observed confidence levels
    conf_values = sorted(set(r["conf"] for r in records))

    # ── 1. Confidence histogram ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PSEUDO-LABEL ANALYSIS{f':  {tag}' if tag else ''}")
    print(f"  Total samples: {total:,}")
    print(f"{'='*70}")

    print("\n── Confidence distribution ─────────────────────────────")
    conf_counts = Counter(r["conf"] for r in records)
    for c in conf_values:
        n = conf_counts[c]
        bar = "█" * int(50 * n / total)
        print(f"  {c:.3f}  {n:>7,}  ({100*n/total:5.1f}%)  {bar}")

    # ── 2. Accuracy & coverage vs threshold ──────────────────────
    print("\n── Accuracy & coverage vs confidence threshold ────────")
    print(f"  {'Threshold':>10}  {'Retained':>9}  {'Coverage':>9}  {'Correct':>9}  {'Accuracy':>9}")
    print(f"  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")

    thresholds = [0.0] + conf_values
    seen = set()
    for t in thresholds:
        if t in seen:
            continue
        seen.add(t)
        subset = [r for r in records if r["conf"] >= t]
        n = len(subset)
        if n == 0:
            continue
        correct = sum(r["correct"] for r in subset)
        acc = correct / n
        cov = n / total
        print(f"  {t:>10.3f}  {n:>9,}  {cov:>8.1%}  {correct:>9,}  {acc:>8.2%}")

    # ── 3. Error analysis at each confidence level ───────────────
    print("\n── Error breakdown by confidence level ─────────────────")
    print(f"  {'Conf':>6}  {'N':>7}  {'Errors':>7}  {'Error%':>7}  {'Top error types (GT → PL)'}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*40}")

    for c in conf_values:
        bucket = [r for r in records if r["conf"] == c]
        errors = [r for r in bucket if not r["correct"]]
        n_err = len(errors)
        err_rate = n_err / len(bucket) if bucket else 0

        err_pairs = Counter((r["gt"], r["pl"]) for r in errors)
        top3 = err_pairs.most_common(3)
        top_str = ", ".join(f"{g}→{p} ({cnt})" for (g, p), cnt in top3) if top3 else "—"

        print(f"  {c:>6.3f}  {len(bucket):>7,}  {n_err:>7,}  {err_rate:>6.1%}  {top_str}")

    # ── 4. Per-class accuracy at high vs low confidence ──────────
    print("\n── Per-class accuracy: low (< 0.75) vs high (≥ 0.875) confidence ──")

    gt_classes = sorted(set(r["gt"] for r in records))
    if len(gt_classes) <= 60:
        print(f"  {'Class':>30}  {'N_low':>6}  {'Acc_low':>8}  {'N_high':>7}  {'Acc_high':>9}")
        print(f"  {'─'*30}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*9}")

        for cls in gt_classes:
            low = [r for r in records if r["gt"] == cls and r["conf"] < 0.75]
            high = [r for r in records if r["gt"] == cls and r["conf"] >= 0.875]

            acc_low = sum(r["correct"] for r in low) / len(low) if low else float("nan")
            acc_high = sum(r["correct"] for r in high) / len(high) if high else float("nan")

            acc_low_str = f"{acc_low:.1%}" if low else "  n/a"
            acc_high_str = f"{acc_high:.1%}" if high else "   n/a"

            print(f"  {cls:>30}  {len(low):>6}  {acc_low_str:>8}  {len(high):>7}  {acc_high_str:>9}")
    else:
        low_all = [r for r in records if r["conf"] < 0.75]
        high_all = [r for r in records if r["conf"] >= 0.875]
        acc_low = sum(r["correct"] for r in low_all) / len(low_all) if low_all else 0
        acc_high = sum(r["correct"] for r in high_all) / len(high_all) if high_all else 0
        print(f"  ({len(gt_classes)} classes — showing aggregate only)")
        print(f"  Low  conf (< 0.75):  N={len(low_all):>7,}  Acc={acc_low:.2%}")
        print(f"  High conf (≥ 0.875): N={len(high_all):>7,}  Acc={acc_high:.2%}")

    # ── 5. Class-coverage shift ──────────────────────────────────
    print("\n── Class coverage shift after thresholding (≥ 0.875) ──")
    all_class_dist = Counter(r["gt"] for r in records)
    high_records = [r for r in records if r["conf"] >= 0.875]
    high_class_dist = Counter(r["gt"] for r in high_records)

    drops = {}
    for cls in gt_classes:
        frac_all = all_class_dist[cls] / total
        frac_high = high_class_dist.get(cls, 0) / len(high_records) if high_records else 0
        drops[cls] = frac_high - frac_all

    worst_drop = sorted(drops.items(), key=lambda x: x[1])[:10]
    best_gain = sorted(drops.items(), key=lambda x: -x[1])[:10]

    print("  Classes most under-represented after thresholding:")
    for cls, delta in worst_drop:
        orig = 100 * all_class_dist[cls] / total
        print(f"    {cls:>30}  {orig:5.2f}% → {orig + 100*delta:5.2f}%  (Δ={100*delta:+.2f}pp)")

    if len(gt_classes) > 10:
        print("  Classes most over-represented after thresholding:")
        for cls, delta in best_gain:
            orig = 100 * all_class_dist[cls] / total
            print(f"    {cls:>30}  {orig:5.2f}% → {orig + 100*delta:5.2f}%  (Δ={100*delta:+.2f}pp)")

    # ── 6. Summary ───────────────────────────────────────────────
    overall_acc = sum(r["correct"] for r in records) / total
    high_acc = sum(r["correct"] for r in high_records) / len(high_records) if high_records else 0

    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Overall PL accuracy (no threshold): {overall_acc:.2%} on {total:,} samples")
    print(f"  PL accuracy (conf ≥ 0.875):         {high_acc:.2%} on {len(high_records):,} samples "
          f"({100*len(high_records)/total:.1f}% coverage)")
    perf = sum(1 for r in records if r["conf"] == 1.0)
    perf_acc = sum(r["correct"] for r in records if r["conf"] == 1.0) / perf if perf else 0
    print(f"  PL accuracy (conf = 1.0):            {perf_acc:.2%} on {perf:,} samples "
          f"({100*perf/total:.1f}% coverage)")
    print()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze pseudo-label quality vs confidence threshold",
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Path(s) to pseudolabeled JSON file(s)",
    )
    args = parser.parse_args()

    for path in args.input:
        tag = Path(path).stem
        print(f"\nLoading {path} ...")
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded {len(data):,} items.")
        analyze(data, tag=tag)


if __name__ == "__main__":
    main()
