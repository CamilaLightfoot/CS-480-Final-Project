import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from sklearn.metrics import roc_curve, auc
from models import get_model
from make_plots import generate_all_plots

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def compute_metrics(scores: np.ndarray, labels: np.ndarray):
    """
    Compute ROC AUC and EER from similarity scores and binary labels.

    labels: 1 = same identity (positive), 0 = different identity (negative)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    # Find index where |FPR - FNR| is minimized (approximate EER).
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[idx] + fnr[idx])

    return {
        "roc_auc": float(roc_auc),
        "eer": float(eer),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }

def evaluate_baseline(model, pairs_df: pd.DataFrame, model_name: str = "model"):
    """
    Evaluate baseline verification performance with clean images for both img1 and img2.

    Returns:
        metrics (dict): output of compute_metrics(scores, labels)

    Side effects:
        - Prints how many pairs were skipped due to detection/embedding errors.
        - Saves scores and labels to results/{model_name}_baseline_scores.npz for later ROC plotting.
    """

    scores = []
    labels = []
    skipped = 0

    for _, row in pairs_df.iterrows():
        img1 = row["img1"]
        img2 = row["img2"]
        label = int(row["label"])

        try:
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
        except Exception:
            skipped += 1
            continue

        score = cosine_similarity(emb1, emb2)
        scores.append(score)
        labels.append(label)

    if skipped > 0:
        print(f"[INFO] Skipped {skipped} pairs due to detection/embedding errors.")

    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    metrics = compute_metrics(scores, labels)

    print(
        f"  Baseline ROC AUC: {metrics['roc_auc']:.4f}\n"
        f"  Baseline EER:     {metrics['eer']:.4f}"
    )

    # Save scores and labels for ROC plotting later.
    os.makedirs("results", exist_ok=True)
    np.savez(
        f"results/{model_name}_baseline_scores.npz",
        scores=scores,
        labels=labels,
    )

    return metrics

def _ensure_tmp_dir(tmp_root: Path) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root

def _augmented_path(orig_path: str, aug: str, level, tmp_root: Path) -> str:
    """
    Create (if not already present) an augmented version of `orig_path`
    and return its path.

    aug ∈ {"blur", "occlusion", "brightness"}
    level:
        - blur: radius (float)
        - occlusion: integer index (we use same occlusion config; index is just to differentiate filenames)
        - brightness: factor (float, <1 darker, >1 brighter)
    """
    orig = Path(orig_path)
    tmp_root = _ensure_tmp_dir(tmp_root)

    suffix = orig.suffix if orig.suffix else ".jpg"
    out_name = f"{orig.stem}_{aug}_{level}{suffix}"
    out_path = tmp_root / out_name

    if out_path.exists():
        return str(out_path)

    img = Image.open(orig).convert("RGB")

    if aug == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=float(level)))
    elif aug == "occlusion":
        # Central square occlusion (~35% of width/height).
        w, h = img.size
        occ_w, occ_h = int(w * 0.35), int(h * 0.35)
        left = (w - occ_w) // 2
        top = (h - occ_h) // 2
        draw = ImageDraw.Draw(img)
        draw.rectangle([left, top, left + occ_w, top + occ_h], fill=(0, 0, 0))
    elif aug == "brightness":
        factor = float(level)
        img = ImageEnhance.Brightness(img).enhance(factor)
    else:
        raise ValueError(f"Unknown augmentation '{aug}'")

    img.save(out_path)
    return str(out_path)

def evaluate_robustness(
    model,
    pairs_df: pd.DataFrame,
    tmp_root: Path,
    blur_levels=(1.0, 2.0, 3.0),
    occlusion_levels=(1,),
    brightness_levels=(0.5, 0.8, 1.2, 1.5),
    model_name: str = "model",
):
    """
    For each augmentation and level, we perturb ONLY img2 in the pair. img1 remains clean. This simulates degraded probe images against clean gallery images.

    We catch detection/embedding errors (e.g., ArcFace detector failing) and skip those pairs, just like in evaluate_baseline().

    In addition to returning a metrics dict, this function also saves a JSON file with robustness metrics to results/{model_name}_robustness.json.
    """

    results = {}

    # Check blur robustness.
    for radius in blur_levels:
        scores, labels = [], []
        skipped = 0

        for _, row in pairs_df.iterrows():
            img1 = row["img1"]
            img2 = row["img2"]
            label = int(row["label"])

            try:
                emb1 = model.get_embedding(img1)
                aug2 = _augmented_path(img2, "blur", radius, tmp_root)
                emb2 = model.get_embedding(aug2)
            except Exception:
                skipped += 1
                continue

            scores.append(cosine_similarity(emb1, emb2))
            labels.append(label)

        if skipped > 0:
            print(f"[INFO] blur_r{radius}: skipped {skipped} pairs.")

        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        metrics = compute_metrics(scores, labels)

        # Store metrics.
        entry = dict(metrics)
        entry["skipped"] = int(skipped)
        entry["n_pairs"] = int(len(labels))
        key = f"blur_r{radius}"
        results[key] = entry

        print(
            f"[blur_r{radius}] ROC AUC: {metrics['roc_auc']:.4f}, "
            f"EER: {metrics['eer']:.4f}"
        )

    # Check occlusion robustness.
    for idx in occlusion_levels:
        scores, labels = [], []
        skipped = 0

        for _, row in pairs_df.iterrows():
            img1 = row["img1"]
            img2 = row["img2"]
            label = int(row["label"])

            try:
                emb1 = model.get_embedding(img1)
                aug2 = _augmented_path(img2, "occlusion", idx, tmp_root)
                emb2 = model.get_embedding(aug2)
            except Exception:
                skipped += 1
                continue

            scores.append(cosine_similarity(emb1, emb2))
            labels.append(label)

        if skipped > 0:
            print(f"[INFO] occlusion_{idx}: skipped {skipped} pairs.")

        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        metrics = compute_metrics(scores, labels)

        entry = dict(metrics)
        entry["skipped"] = int(skipped)
        entry["n_pairs"] = int(len(labels))
        key = f"occlusion_{idx}"
        results[key] = entry

        print(
            f"[occlusion_{idx}] ROC AUC: {metrics['roc_auc']:.4f}, "
            f"EER: {metrics['eer']:.4f}"
        )

    # Check brightness robustness.
    for factor in brightness_levels:
        scores, labels = [], []
        skipped = 0

        for _, row in pairs_df.iterrows():
            img1 = row["img1"]
            img2 = row["img2"]
            label = int(row["label"])

            try:
                emb1 = model.get_embedding(img1)
                aug2 = _augmented_path(img2, "brightness", factor, tmp_root)
                emb2 = model.get_embedding(aug2)
            except Exception:
                skipped += 1
                continue

            scores.append(cosine_similarity(emb1, emb2))
            labels.append(label)

        if skipped > 0:
            print(f"[INFO] brightness_{factor}: skipped {skipped} pairs.")

        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        metrics = compute_metrics(scores, labels)

        entry = dict(metrics)
        entry["skipped"] = int(skipped)
        entry["n_pairs"] = int(len(labels))
        key = f"brightness_{factor}"
        results[key] = entry

        print(
            f"[brightness_{factor}] ROC AUC: {metrics['roc_auc']:.4f}, "
            f"EER: {metrics['eer']:.4f}"
        )

    # Save robustness metrics to JSON.
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{model_name}_robustness.json")

    # Make sure everything is JSON-serializable.
    serializable = {}
    for key, val in results.items():
        serializable[key] = {
            "roc_auc": float(val["roc_auc"]),
            "eer": float(val["eer"]),
            **(
                {"threshold": float(val["threshold"])}
                if "threshold" in val
                else {}
            ),
            "skipped": int(val["skipped"]),
            "n_pairs": int(val["n_pairs"]),
        }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[INFO] Saved robustness metrics to {out_path}")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Face verification evaluation with model switch and robustness tests."
    )
    parser.add_argument(
        "--pairs_csv",
        type=str,
        required=True,
        help="CSV file with columns: img1, img2, label (1=same, 0=different).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="facenet | arcface | adaface | (omit to run all three)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string for PyTorch models (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="robust_tmp",
        help="Directory where augmented images will be cached.",
    )
    parser.add_argument(
        "--no_robustness",
        action="store_true",
        help="If set, skip robustness tests and only run baseline verification.",
    )

    args = parser.parse_args()

    # Load pairs CSV and sanity-check columns.
    pairs_df = pd.read_csv(args.pairs_csv)
    for col in ["img1", "img2", "label"]:
        if col not in pairs_df.columns:
            raise ValueError(f"Expected column '{col}' in {args.pairs_csv}")

    ALL_MODELS = ["facenet", "arcface", "adaface"]

    if args.model is None:
        models_to_run = ALL_MODELS
        print(f"[INFO] No model specified. Running all models: {models_to_run}")
    else:
        if args.model not in ALL_MODELS:
            raise ValueError(
                f"Invalid --model '{args.model}'. Must be one of {ALL_MODELS} or omitted."
            )
        models_to_run = [args.model]
        print(f"[INFO] Running single model: {args.model}")

    # Prepare tmp dir for augmentations once.
    tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Run evaluation for each selected model.
    for model_name in models_to_run:
        print("\n========================================")
        print(f"[INFO] Loading model: {model_name} on device {args.device}")
        print("========================================\n")

        model = get_model(model_name, device=args.device)

        print("[INFO] Running baseline verification evaluation...")
        baseline_metrics = evaluate_baseline(model, pairs_df, model_name=model_name)

        print(f"  [{model_name}] Baseline ROC AUC: {baseline_metrics['roc_auc']:.4f}, EER: {baseline_metrics['eer']:.4f}")

        if not args.no_robustness:
            print("[INFO] Running robustness evaluations (blur, occlusion, lighting)...")
            robustness = evaluate_robustness(
                model,
                pairs_df,
                tmp_root,
                model_name=model_name,
            )

            for key, metrics in robustness.items():
                print(f"  [{model_name} | {key}] ROC AUC: {metrics['roc_auc']:.4f}, EER: {metrics['eer']:.4f}"
                )
        else:
            print(f"[INFO] --no_robustness set; skipping robustness for {model_name}.")

    # After ALL models have run, generate plots.
    print("\n[INFO] All evaluations completed. Generating plots from 'results'...")
    generate_all_plots("results")
    print("[INFO] All done.")

if __name__ == "__main__":
    main()