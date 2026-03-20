"""
CALM: Unified entry point for audio classification tasks.

Usage:
    # Audio classification
    python -m src.run --task classify \
        --model_name qwen2-audio-instruct \
        --data_name vgg_sound_qa \
        --train_path /path/to/train.json \
        --val_path /path/to/val.json \
        --test_path /path/to/test.json

    # Spoofing detection
    python -m src.run --task spoof \
        --model_name qwen2-audio-instruct \
        --data_name LA_spoof \
        --train_path /path/to/train.json \
        --val_path /path/to/val.json

    # Pseudolabel generation
    python -m src.run --task pseudolabel \
        --model_name qwen2-audio-instruct \
        --data_name audioset \
        --train_path /path/to/unlabeled.json \
        --output_dir ./pseudolabels
"""
import argparse
import os
from datetime import datetime

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from .calm import (
    calm_build_weights_from_r,
    calm_compute_posteriors_from_cache,
    calm_compute_reliability,
    calm_eval_from_posteriors,
    calm_get_predictions,
    calm_prepare_cache,
    load_model,
)
from .preprocess import open_data

torch.set_grad_enabled(False)


def normalize_spoof_label(label: str) -> str:
    """Normalize spoofing labels to 'bonafide' or 'spoof'."""
    label_lower = (label or "").lower().strip()
    if any(k in label_lower for k in ["spoof", "fake", "synthetic", "artificial"]):
        return "spoof"
    if any(k in label_lower for k in ["bonafide", "genuine", "real", "authentic"]):
        return "bonafide"
    return "bonafide"


def _run_zero_shot_only(args):
    """Zero-shot inference on test set only. No CALM cache, no train/val processing."""
    print("=" * 60)
    print("Zero-shot baseline: inference on test set only")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print()

    print("Loading model...")
    model = load_model(args.model_name, args.data_name)

    test_path = args.test_path or args.val_path
    if not test_path:
        raise ValueError("--test_path or --val_path required for zero_shot_only")
    print("Loading test data...")
    test_data = open_data(args.data_name, test_path)
    test_items = list(test_data)
    print(f"  Test samples: {len(test_items)}")
    print()

    print("Running zero-shot inference on test set...")
    correct = 0
    total = 0
    test_split = "test" if args.test_path else "val"
    for item in tqdm(test_items, desc="Zero-shot inference"):
        try:
            result = model.format_func(
                all_data=None, cur_item=item, num_shot=0,
                model_helper=model, split=test_split
            )
            if len(result) == 5:
                tqs, ans, audio_list, _, _ = result
            else:
                tqs, ans, _, audio_list, _, _ = result
            inputs = model.insert_audio(tqs, ans, audio_list)
            _, norm_probs = model.first_token_prob(inputs)
            pred_idx = max(range(4), key=lambda i: norm_probs[i])
            options = item.get("options", [])
            mapped = (item.get("mapped_label") or "").lower()
            pred_label = options[pred_idx].lower() if pred_idx < len(options) else ""
            if mapped:
                total += 1
                if pred_label == mapped:
                    correct += 1
        except Exception:
            pass
        torch.cuda.empty_cache()

    acc = correct / max(total, 1)
    print()
    print("=" * 60)
    print("ZERO-SHOT RESULTS")
    print("=" * 60)
    print(f"Accuracy: {correct}/{total} ({acc:.4f})")
    print()


def run_classification(args):
    """Run audio classification with CALM. Supports sweeping tau, tau_w, top_k."""
    # Zero-shot only: skip CALM entirely, run inference on test set only
    if args.zero_shot_only:
        _run_zero_shot_only(args)
        return

    tau_list = args.tau
    tau_w_list = args.tau_w
    top_k_list = args.top_k if args.top_k is not None else [None]

    print("=" * 60)
    print("CALM: Audio Classification")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print(f"Weight scheme: {args.weight_scheme}")
    print(f"Tau values: {tau_list}")
    print(f"Tau_w values: {tau_w_list}")
    print(f"Top_k values: {top_k_list}")
    print(f"N trials: {args.n_trials}")
    if args.random_topk:
        print(f"Random reliability: ENABLED (seed={args.random_head_seed})")
    if args.unsupervised:
        print("Unsupervised HP selection: ENABLED (pseudolabels)")
    print()

    # Load model and data
    print("Loading model...")
    model = load_model(args.model_name, args.data_name)

    print("Loading data...")
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    test_data = open_data(args.data_name, args.test_path) if args.test_path else val_data

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(list(test_data))}")
    print()

    # Prepare cache (expensive — done once)
    print("Building CALM cache...")
    cache = calm_prepare_cache(
        model,
        support_data=train_data,
        val_data=val_data,
        test_data=test_data if args.test_path else None,
        last_n_tokens=args.last_n_tokens,
        n_trials=args.n_trials,
        cache_dir=args.cache_dir,
    )

    C, K, D = cache["prototypes"].shape
    print(f"  Classes (C): {C}")
    print(f"  Heads (K): {K}")
    print(f"  Dimension (D): {D}")
    print()

    # Unsupervised mode: zero-shot pseudolabel the TEST set for HP selection
    test_pseudo_labels = None
    if args.unsupervised:
        print("Unsupervised mode: generating zero-shot pseudolabels for test set...")
        str_to_int = cache["str_to_int"]
        _test_items = list(test_data) if args.test_path else list(val_data)
        _test_split = "test" if args.test_path else "val"
        _test_meta = cache.get(f"qacts_{_test_split}_n", {})
        _test_orig = _test_meta.get("original_indices", list(range(len(_test_items))))

        test_pseudo_labels = []
        _gt = cache["test_labels_idx"] if args.test_path else cache["val_labels_idx"]
        for orig_i in tqdm(_test_orig, desc="Zero-shot inference (test)"):
            item = _test_items[orig_i]
            try:
                result = model.format_func(
                    all_data=None, cur_item=item, num_shot=0,
                    model_helper=model, split=_test_split
                )
                if len(result) == 5:
                    tqs, ans, audio_list, _, _ = result
                else:
                    tqs, ans, _, audio_list, _, _ = result
                inputs = model.insert_audio(tqs, ans, audio_list)
                _, norm_probs = model.first_token_prob(inputs)
                pred_idx = max(range(4), key=lambda i: norm_probs[i])
                options = item.get("options", [])
                pred_label = options[pred_idx].lower() if pred_idx < len(options) else ""
                test_pseudo_labels.append(str_to_int.get(pred_label, -1))
            except Exception:
                test_pseudo_labels.append(-1)
            torch.cuda.empty_cache()

        pl_correct = sum(1 for p, g in zip(test_pseudo_labels, _gt) if p == g and g >= 0)
        pl_total = sum(1 for g in _gt if g >= 0)
        pl_valid = sum(1 for p in test_pseudo_labels if p >= 0)
        print(f"  Zero-shot accuracy on test: {pl_correct}/{pl_total} "
              f"({pl_correct / max(pl_total, 1):.4f})")
        print(f"  Valid pseudolabels: {pl_valid}/{len(test_pseudo_labels)}")
        print()

    # Sweep over all (tau, tau_w, top_k) combinations (cheap — uses cached activations)
    n_combos = len(tau_list) * len(tau_w_list) * len(top_k_list)
    print(f"Sweeping {n_combos} hyperparameter combination(s)...")
    print()

    test_labels = cache["test_labels_idx"] if args.test_path else cache["val_labels_idx"]
    test_items = list(test_data) if args.test_path else list(val_data)
    test_meta = cache.get("qacts_test_n" if args.test_path else "qacts_val_n", {})
    original_indices = test_meta.get("original_indices", list(range(len(test_items))))

    all_results = []

    has_separate_test = bool(args.test_path)

    # In unsupervised mode, HP selection uses pseudolabels on test set
    hp_selection_labels = test_pseudo_labels if args.unsupervised else None

    for tau in tau_list:
        P_val = calm_compute_posteriors_from_cache(cache, tau=tau, split="val")
        P_test = calm_compute_posteriors_from_cache(
            cache, tau=tau, split="test" if has_separate_test else "val"
        )

        r, counts = calm_compute_reliability(P_val, cache["val_labels_idx"], args.weight_scheme)

        for tau_w in tau_w_list:
            for top_k in top_k_list:
                w = calm_build_weights_from_r(
                    r,
                    weight_scheme=args.weight_scheme,
                    tau_w=tau_w,
                    top_k=top_k,
                    random_topk=args.random_topk,
                    random_head_seed=args.random_head_seed,
                )

                test_accuracy = calm_eval_from_posteriors(P_test, w, test_labels_idx=test_labels)

                if args.unsupervised:
                    pseudo_accuracy = calm_eval_from_posteriors(
                        P_test, w, test_labels_idx=hp_selection_labels
                    )
                    val_accuracy = pseudo_accuracy
                elif has_separate_test:
                    val_accuracy = calm_eval_from_posteriors(
                        P_val, w, test_labels_idx=cache["val_labels_idx"]
                    )
                else:
                    val_accuracy = test_accuracy

                all_results.append({
                    "tau": tau, "tau_w": tau_w, "top_k": top_k,
                    "val_accuracy": val_accuracy, "test_accuracy": test_accuracy,
                })

                top_k_str = str(top_k) if top_k is not None else "all"
                if args.unsupervised:
                    print(f"  tau={tau:<8g}  tau_w={tau_w:<8g}  top_k={top_k_str:<6s}  "
                          f"pseudo={val_accuracy:.4f}  real={test_accuracy:.4f}")
                elif has_separate_test:
                    print(f"  tau={tau:<8g}  tau_w={tau_w:<8g}  top_k={top_k_str:<6s}  "
                          f"val={val_accuracy:.4f}  test={test_accuracy:.4f}")
                else:
                    print(f"  tau={tau:<8g}  tau_w={tau_w:<8g}  top_k={top_k_str:<6s}  "
                          f"acc={test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Print summary table
    print()
    print("=" * 60)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 60)

    best = max(all_results, key=lambda x: x["val_accuracy"])

    if args.unsupervised:
        sel_col = "pseudo_acc"
        print(f"  {'tau':>8s}  {'tau_w':>8s}  {'top_k':>6s}  {sel_col:>10s}  {'real_acc':>10s}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*10}")
        for res in all_results:
            top_k_str = str(res["top_k"]) if res["top_k"] is not None else "all"
            marker = " *" if res is best else ""
            print(f"  {res['tau']:>8g}  {res['tau_w']:>8g}  {top_k_str:>6s}  "
                  f"{res['val_accuracy']:>9.4f}  {res['test_accuracy']:>9.4f}{marker}")
        print()
        print(f"  Best (by pseudolabel): tau={best['tau']}, tau_w={best['tau_w']}, "
              f"top_k={best['top_k']}")
        print(f"    pseudo_acc={best['val_accuracy']:.4f}, "
              f"real_acc={best['test_accuracy']:.4f}")
    elif has_separate_test:
        print(f"  {'tau':>8s}  {'tau_w':>8s}  {'top_k':>6s}  {'val_acc':>10s}  {'test_acc':>10s}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*10}")
        for res in all_results:
            top_k_str = str(res["top_k"]) if res["top_k"] is not None else "all"
            marker = " *" if res is best else ""
            print(f"  {res['tau']:>8g}  {res['tau_w']:>8g}  {top_k_str:>6s}  "
                  f"{res['val_accuracy']:>9.4f}  {res['test_accuracy']:>9.4f}{marker}")
        print()
        print(f"  Best (by val): tau={best['tau']}, tau_w={best['tau_w']}, "
              f"top_k={best['top_k']}")
        print(f"    val_acc={best['val_accuracy']:.4f}, "
              f"test_acc={best['test_accuracy']:.4f}")
    else:
        print(f"  {'tau':>8s}  {'tau_w':>8s}  {'top_k':>6s}  {'accuracy':>10s}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}")
        for res in all_results:
            top_k_str = str(res["top_k"]) if res["top_k"] is not None else "all"
            marker = " *" if res is best else ""
            print(f"  {res['tau']:>8g}  {res['tau_w']:>8g}  {top_k_str:>6s}  "
                  f"{res['test_accuracy']:>9.4f}{marker}")
        print()
        print(f"  Best: tau={best['tau']}, tau_w={best['tau_w']}, "
              f"top_k={best['top_k']}, acc={best['test_accuracy']:.4f}")

    # Detailed report for best config
    best_tau, best_tau_w, best_top_k = best["tau"], best["tau_w"], best["top_k"]
    P_val_best = calm_compute_posteriors_from_cache(cache, tau=best_tau, split="val")
    P_test_best = calm_compute_posteriors_from_cache(
        cache, tau=best_tau, split="test" if args.test_path else "val"
    )
    r_best, _ = calm_compute_reliability(P_val_best, cache["val_labels_idx"], args.weight_scheme)
    w_best = calm_build_weights_from_r(
        r_best, weight_scheme=args.weight_scheme, tau_w=best_tau_w, top_k=best_top_k,
        random_topk=args.random_topk, random_head_seed=args.random_head_seed,
    )
    pred_labels = calm_get_predictions(P_test_best, w_best, cache)

    ground_truths = []
    predictions = []
    for cached_idx, pred in enumerate(pred_labels):
        if cached_idx < len(original_indices):
            orig_idx = original_indices[cached_idx]
            if orig_idx < len(test_items):
                gt = test_items[orig_idx].get("mapped_label", test_items[orig_idx].get("label", ""))
                gt = gt.lower() if isinstance(gt, str) else gt
                ground_truths.append(gt)
                predictions.append(pred)

    print()
    print("Classification Report (best config):")
    print(classification_report(ground_truths, predictions))

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            args.output_dir,
            f"calm_{args.data_name}_{args.model_name}_{timestamp}.txt"
        )
        with open(result_file, "w") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.data_name}\n")
            f.write(f"Weight scheme: {args.weight_scheme}\n")
            f.write(f"N trials: {args.n_trials}\n\n")
            f.write("Sweep results:\n")
            if has_separate_test:
                f.write(f"  {'tau':>8s}  {'tau_w':>8s}  {'top_k':>6s}  "
                        f"{'val_acc':>10s}  {'test_acc':>10s}\n")
                for res in all_results:
                    top_k_str = str(res["top_k"]) if res["top_k"] is not None else "all"
                    f.write(f"  {res['tau']:>8g}  {res['tau_w']:>8g}  {top_k_str:>6s}  "
                            f"{res['val_accuracy']:>9.4f}  {res['test_accuracy']:>9.4f}\n")
                f.write(f"\nBest (by val): tau={best['tau']}, tau_w={best['tau_w']}, "
                        f"top_k={best['top_k']}\n")
                f.write(f"  val_acc={best['val_accuracy']:.4f}, "
                        f"test_acc={best['test_accuracy']:.4f}\n")
            else:
                f.write(f"  {'tau':>8s}  {'tau_w':>8s}  {'top_k':>6s}  {'accuracy':>10s}\n")
                for res in all_results:
                    top_k_str = str(res["top_k"]) if res["top_k"] is not None else "all"
                    f.write(f"  {res['tau']:>8g}  {res['tau_w']:>8g}  {top_k_str:>6s}  "
                            f"{res['test_accuracy']:>9.4f}\n")
                f.write(f"\nBest: tau={best['tau']}, tau_w={best['tau_w']}, "
                        f"top_k={best['top_k']}, acc={best['test_accuracy']:.4f}\n")
            f.write(f"\n{classification_report(ground_truths, predictions)}")
        print(f"\nResults saved to: {result_file}")


def run_spoof_detection(args):
    """Run spoofing detection with CALM. Supports sweeping tau, tau_w, top_k."""
    tau_list = args.tau
    tau_w_list = args.tau_w
    top_k_list = args.top_k if args.top_k is not None else [None]

    print("=" * 60)
    print("CALM: Spoofing Detection")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print(f"Tau values: {tau_list}")
    print(f"Tau_w values: {tau_w_list}")
    print(f"Top_k values: {top_k_list}")
    print()

    # Load model and data
    print("Loading model...")
    model = load_model(args.model_name, args.data_name)

    print("Loading data...")
    train_data = open_data(args.data_name, args.train_path)
    val_data = open_data(args.data_name, args.val_path)
    test_data = open_data(args.data_name, args.test_path) if args.test_path else val_data

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(list(test_data))}")
    print()

    # Prepare cache (expensive — done once)
    print("Building CALM cache...")
    cache = calm_prepare_cache(
        model,
        support_data=train_data,
        val_data=val_data,
        test_data=test_data if args.test_path else None,
        last_n_tokens=args.last_n_tokens,
        n_trials=args.n_trials,
        cache_dir=args.cache_dir,
    )

    C, K, D = cache["prototypes"].shape
    print(f"  Classes (C): {C}")
    print(f"  Heads (K): {K}")
    print()

    test_items = list(test_data) if args.test_path else list(val_data)
    test_meta = cache.get("qacts_test_n" if args.test_path else "qacts_val_n", {})

    n_combos = len(tau_list) * len(tau_w_list) * len(top_k_list)
    print(f"Sweeping {n_combos} hyperparameter combination(s)...")
    print()

    all_results = []

    for tau in tau_list:
        P_val = calm_compute_posteriors_from_cache(cache, tau=tau, split="val")
        P_test = calm_compute_posteriors_from_cache(
            cache, tau=tau, split="test" if args.test_path else "val"
        )
        r, counts = calm_compute_reliability(P_val, cache["val_labels_idx"], args.weight_scheme)

        for tau_w in tau_w_list:
            for top_k in top_k_list:
                w = calm_build_weights_from_r(
                    r,
                    weight_scheme=args.weight_scheme,
                    tau_w=tau_w,
                    top_k=top_k,
                    random_topk=args.random_topk,
                    random_head_seed=args.random_head_seed,
                )

                pred_labels = calm_get_predictions(P_test, w, cache)
                original_indices = test_meta.get("original_indices", list(range(len(pred_labels))))

                predictions = []
                ground_truths = []
                for cached_idx, pred in enumerate(pred_labels):
                    if cached_idx < len(original_indices):
                        orig_idx = original_indices[cached_idx]
                        if orig_idx < len(test_items):
                            item = test_items[orig_idx]
                            gt = item.get("mapped_label", item.get("label", ""))
                            predictions.append(normalize_spoof_label(pred))
                            ground_truths.append(normalize_spoof_label(gt))

                labels = ["bonafide", "spoof"]
                acc = accuracy_score(ground_truths, predictions)
                macro_f1_val = f1_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)

                all_results.append({
                    "tau": tau, "tau_w": tau_w, "top_k": top_k,
                    "accuracy": acc, "macro_f1": macro_f1_val,
                })

                top_k_str = str(top_k) if top_k is not None else "all"
                print(f"  tau={tau:<8g}  tau_w={tau_w:<8g}  top_k={top_k_str:<6s}  "
                      f"acc={acc:.4f}  f1={macro_f1_val:.4f}")

    # Summary
    print()
    print("=" * 60)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'tau':>8s}  {'tau_w':>8s}  {'top_k':>6s}  {'accuracy':>10s}  {'macro_f1':>10s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*10}")
    best = max(all_results, key=lambda x: x["macro_f1"])
    for res in all_results:
        top_k_str = str(res["top_k"]) if res["top_k"] is not None else "all"
        marker = " *" if res is best else ""
        print(f"  {res['tau']:>8g}  {res['tau_w']:>8g}  {top_k_str:>6s}  "
              f"{res['accuracy']:>9.4f}  {res['macro_f1']:>9.4f}{marker}")
    print()
    print(f"  Best: tau={best['tau']}, tau_w={best['tau_w']}, "
          f"top_k={best['top_k']}, acc={best['accuracy']:.4f}, f1={best['macro_f1']:.4f}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            args.output_dir,
            f"calm_spoof_{args.data_name}_{args.model_name}_{timestamp}.txt"
        )
        with open(result_file, "w") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.data_name}\n\n")
            f.write("Sweep results:\n")
            f.write(f"  {'tau':>8s}  {'tau_w':>8s}  {'top_k':>6s}  {'accuracy':>10s}  {'macro_f1':>10s}\n")
            for res in all_results:
                top_k_str = str(res["top_k"]) if res["top_k"] is not None else "all"
                f.write(f"  {res['tau']:>8g}  {res['tau_w']:>8g}  {top_k_str:>6s}  "
                        f"{res['accuracy']:>9.4f}  {res['macro_f1']:>9.4f}\n")
            f.write(f"\nBest: tau={best['tau']}, tau_w={best['tau_w']}, "
                    f"top_k={best['top_k']}, acc={best['accuracy']:.4f}, f1={best['macro_f1']:.4f}\n")
        print(f"\nResults saved to: {result_file}")


def run_pseudolabel(args):
    """Generate pseudolabels using model predictions."""
    from .pseudolabel import generate_pseudolabels

    print("=" * 60)
    print("CALM: Pseudolabel Generation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print(f"N trials: {args.n_trials}")
    print(f"Min confidence: {args.min_confidence}")
    print()

    # Create args for pseudolabel function
    class PseudolabelArgs:
        pass

    pl_args = PseudolabelArgs()
    pl_args.model_name = args.model_name
    pl_args.data_name = args.data_name
    pl_args.train_path = args.train_path
    pl_args.n_trials = args.n_trials
    pl_args.min_confidence = args.min_confidence
    pl_args.output_dir = args.output_dir

    generate_pseudolabels(pl_args)


def run_sav(args):
    """Run SAV baseline: top-k heads by train accuracy + majority vote.

    Supports sweeping over multiple num_heads values with a single cache build.
    """
    from .sav import sav_build_full_cache, sav_evaluate_from_cache, sav_print_top_heads

    heads_list = args.sav_num_heads if isinstance(args.sav_num_heads, list) else [args.sav_num_heads]

    print("=" * 60)
    print("SAV Baseline: Sparse Attention Vectors")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print(f"Num heads sweep: {heads_list}")
    print()

    print("Loading model...")
    model = load_model(args.model_name, args.data_name)

    print("Loading data...")
    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.test_path) if args.test_path else open_data(args.data_name, args.val_path)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(list(test_data))}")
    print()

    print("Building full SAV cache (prototypes + head scores + test activations)...")
    cache = sav_build_full_cache(train_data, test_data, model)

    C = cache["all_prototypes"].shape[0]
    K_all = len(cache["all_heads"])
    print(f"  Classes (C): {C}")
    print(f"  Total heads (K): {K_all}")
    print()

    all_results = []

    for num_heads in heads_list:
        sav_print_top_heads(cache, num_heads)
        accuracy, predictions, ground_truths = sav_evaluate_from_cache(cache, num_heads)

        all_results.append({
            "num_heads": num_heads,
            "accuracy": accuracy,
            "predictions": predictions,
            "ground_truths": ground_truths,
        })

        print()
        print("=" * 60)
        print(f"SAV RESULTS (num_heads={num_heads})")
        print("=" * 60)
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print()
        print("Classification Report:")
        print(classification_report(ground_truths, predictions))

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(
                args.output_dir,
                f"sav_{args.data_name}_{args.model_name}_{num_heads}heads_{timestamp}.txt"
            )
            with open(result_file, "w") as f:
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Dataset: {args.data_name}\n")
                f.write(f"Num heads: {num_heads}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"\n{classification_report(ground_truths, predictions)}")
            print(f"Results saved to: {result_file}")

    if len(heads_list) > 1:
        print()
        print("=" * 60)
        print("SAV SWEEP SUMMARY")
        print("=" * 60)
        print(f"  {'heads':>6s}  {'accuracy':>10s}")
        print(f"  {'─'*6}  {'─'*10}")
        for res in all_results:
            print(f"  {res['num_heads']:>6d}  {res['accuracy']:>9.4f}")
        best = max(all_results, key=lambda x: x["accuracy"])
        print(f"\n  Best: num_heads={best['num_heads']}, acc={best['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="CALM: Class-conditional Attention vectors for audio Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classify", "spoof", "pseudolabel", "sav"],
        help="Task to run: classify (CALM), spoof (spoofing detection), pseudolabel (generate pseudolabels), sav (SAV baseline)",
    )

    # Model and data
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["qwen2-audio-instruct", "qwen2.5_omni", "qwen2-vl-instruct", "phi4-multimodal"],
        help="Model to use",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help="Dataset name (vgg_sound_qa, esc_mcq, audioset, LA_spoof, mlaad, etc.)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Path to training/support JSON file (not needed for --zero_shot_only)",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to validation JSON file (for reliability estimation)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to test JSON file (optional, defaults to val_path)",
    )

    # CALM hyperparameters (accept multiple values for sweeps)
    parser.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=[0.07],
        help="Temperature(s) for class posteriors (default: 0.07). Multiple values trigger a sweep.",
    )
    parser.add_argument(
        "--tau_w",
        type=float,
        nargs="+",
        default=[1.0],
        help="Temperature(s) for head weighting (default: 1.0). Multiple values trigger a sweep.",
    )
    parser.add_argument(
        "--weight_scheme",
        type=str,
        default="margin_clamped",
        choices=["margin_clamped", "margin_softmax", "prob_softmax", "brier_softmax"],
        help="Weight scheme for reliability estimation (default: margin_clamped)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for activation averaging (default: 20)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=None,
        help="Optional top-k head selection per class. Multiple values trigger a sweep.",
    )
    parser.add_argument(
        "--random_topk",
        action="store_true",
        default=False,
        help="Use random reliability scores instead of computed ones (baseline ablation)",
    )
    parser.add_argument(
        "--random_head_seed",
        type=int,
        default=None,
        help="Seed for random reliability (for reproducibility). Only used with --random_topk.",
    )
    parser.add_argument(
        "--last_n_tokens",
        type=int,
        default=1,
        help="Number of tokens to average (default: 1)",
    )

    # SAV-specific
    parser.add_argument(
        "--sav_num_heads",
        type=int,
        nargs="+",
        default=[20],
        help="Number of heads to select for SAV baseline. Multiple values trigger a sweep (default: 20)",
    )

    # Unsupervised HP selection
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        default=False,
        help="Use pseudolabels (model's own predictions) instead of ground truth for "
             "reliability estimation and HP selection. Proves HPs can be chosen without labels.",
    )
    parser.add_argument(
        "--zero_shot_only",
        action="store_true",
        default=False,
        help="For classify task: run zero-shot inference only and exit. Skips CALM sweep.",
    )

    # Pseudolabel-specific
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for pseudolabels (default: 0.5)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory for output files (default: ./results)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for cached activations (default: ./cache)",
    )

    args = parser.parse_args()

    # Validate args
    if args.task in ["classify", "spoof"] and args.val_path is None and not (
        args.task == "classify" and args.zero_shot_only and args.test_path
    ):
        parser.error("--val_path is required for classify and spoof tasks")
    if args.train_path is None and args.task != "classify":
        parser.error("--train_path is required for spoof, pseudolabel, sav tasks")
    if args.task == "classify" and not args.zero_shot_only and args.train_path is None:
        parser.error("--train_path is required for classify (unless --zero_shot_only)")
    if args.task == "sav" and args.val_path is None and args.test_path is None:
        parser.error("--val_path or --test_path is required for sav task")

    # Run task
    if args.task == "classify":
        run_classification(args)
    elif args.task == "spoof":
        run_spoof_detection(args)
    elif args.task == "pseudolabel":
        run_pseudolabel(args)
    elif args.task == "sav":
        run_sav(args)


if __name__ == "__main__":
    main()
