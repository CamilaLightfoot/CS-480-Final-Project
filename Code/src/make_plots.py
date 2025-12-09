import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

models = ["FaceNet", "ArcFace", "AdaFace"]
model_keys = ["facenet", "arcface", "adaface"]


def load_baseline_scores(model_name: str, results_dir: str):
    path = os.path.join(results_dir, f"{model_name}_baseline_scores.npz")
    data = np.load(path)
    return data["scores"], data["labels"]


def load_robustness_metrics(model_name: str, results_dir: str):
    path = os.path.join(results_dir, f"{model_name}_robustness.json")
    with open(path, "r") as f:
        return json.load(f)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute EER (as a percentage) from similarity scores and binary labels.
    labels: 1 for genuine, 0 for impostor.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    diff = np.abs(fnr - fpr)
    idx = np.argmin(diff)
    eer = (fnr[idx] + fpr[idx]) / 2.0  # in [0,1]
    return float(eer * 100.0)


def generate_all_plots(results_dir: str = "results"):
    print(f"[INFO] Generating plots from '{results_dir}'...")

    # Create plots/ subdirectory inside results/.
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Baseline EER bar chart (computed from NPZ).
    baseline_eer_vals = []
    valid_model_labels = []

    for m_label, m_key in zip(models, model_keys):
        try:
            scores, labels = load_baseline_scores(m_key, results_dir)
            eer_pct = compute_eer(scores, labels)
            baseline_eer_vals.append(eer_pct)
            valid_model_labels.append(m_label)
        except FileNotFoundError:
            print(f"[WARN] Missing baseline scores for {m_key}; skipping in EER bar plot.")

    if baseline_eer_vals:
        plt.figure()
        x = np.arange(len(valid_model_labels))
        plt.bar(x, baseline_eer_vals)
        plt.xticks(x, valid_model_labels)
        plt.ylabel("EER (%)")
        plt.title("Baseline Equal Error Rate by Model (LFW)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, "baseline_eer.png")
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Saved {out_path}")
    else:
        print("[WARN] No baseline EER values available; baseline_eer.png not generated.")

    # Blur robustness (AUC) and 6. EER vs blur.
    blur_radii = [1.0, 2.0, 3.0]
    blur_keys = ["blur_r1.0", "blur_r2.0", "blur_r3.0"]

    # AUC vs blur.
    plt.figure()
    for m_label, m_key in zip(models, model_keys):
        try:
            rob = load_robustness_metrics(m_key, results_dir)
        except FileNotFoundError:
            print(f"[WARN] Missing robustness JSON for {m_key}; skipping in blur AUC plot.")
            continue

        auc_vals = [rob[k]["roc_auc"] for k in blur_keys]
        plt.plot(blur_radii, auc_vals, marker="o", label=m_label)

    plt.xlabel("Blur radius (Gaussian)")
    plt.ylabel("ROC AUC")
    plt.title("Blur Robustness (Probe Blurred, Gallery Clean)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "blur_auc.png")
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved {out_path}")

    # EER vs blur (convert fraction -> percent).
    plt.figure()
    for m_label, m_key in zip(models, model_keys):
        try:
            rob = load_robustness_metrics(m_key, results_dir)
        except FileNotFoundError:
            print(f"[WARN] Missing robustness JSON for {m_key}; skipping in EER vs blur plot.")
            continue

        eer_vals = [rob[k]["eer"] * 100.0 for k in blur_keys]
        plt.plot(blur_radii, eer_vals, marker="o", label=m_label)

    plt.xlabel("Blur radius (Gaussian)")
    plt.ylabel("EER (%)")
    plt.title("EER vs. Blur Radius (Probe Blurred, Gallery Clean)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "eer_vs_blur.png")
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved {out_path}")

    # Occlusion robustness (EER, from JSON).    
    plt.figure()
    occl_key = "occlusion_1"
    occl_eer_vals = []
    occl_labels = []

    for m_label, m_key in zip(models, model_keys):
        try:
            rob = load_robustness_metrics(m_key, results_dir)
        except FileNotFoundError:
            print(f"[WARN] Missing robustness JSON for {m_key}; skipping in occlusion plot.")
            continue

        occl_eer_vals.append(rob[occl_key]["eer"] * 100.0)
        occl_labels.append(m_label)

    if occl_eer_vals:
        x = np.arange(len(occl_labels))
        plt.bar(x, occl_eer_vals)
        plt.xticks(x, occl_labels)
        plt.ylabel("EER (%)")
        plt.title("Occlusion Robustness (Central Block Occlusion on Probe)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, "occlusion_eer.png")
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Saved {out_path}")
    else:
        print("[WARN] No occlusion EER values available; occlusion_eer.png not generated.")

    # Brightness robustness (AUC)
    brightness_levels = [0.5, 0.8, 1.2, 1.5]
    brightness_keys = ["brightness_0.5", "brightness_0.8", "brightness_1.2", "brightness_1.5"]

    plt.figure()
    for m_label, m_key in zip(models, model_keys):
        try:
            rob = load_robustness_metrics(m_key, results_dir)
        except FileNotFoundError:
            print(f"[WARN] Missing robustness JSON for {m_key}; skipping in brightness AUC plot.")
            continue

        auc_vals = [rob[k]["roc_auc"] for k in brightness_keys]
        plt.plot(brightness_levels, auc_vals, marker="o", label=m_label)

    plt.xlabel("Brightness scale factor")
    plt.ylabel("ROC AUC")
    plt.title("Brightness Robustness (Probe Scaled, Gallery Clean)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "brightness_auc.png")
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved {out_path}")

    # Baseline ROC curves from stored scores
    try:
        plt.figure()

        for m_label, m_key in zip(models, model_keys):
            scores, labels = load_baseline_scores(m_key, results_dir)
            fpr, tpr, _ = roc_curve(labels, scores)
            model_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{m_label} (AUC = {model_auc:.4f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Baseline ROC Curves on LFW")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, "roc_baseline_lfw.png")
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Saved {out_path}")

    except FileNotFoundError as e:
        print("[WARN] Could not generate ROC figure; missing NPZ file:")
        print("   ", e)
        print("[WARN] Run evaluate_verification.py with score saving enabled for each model.")