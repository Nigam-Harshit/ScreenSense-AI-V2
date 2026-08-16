"""
evaluate_yolo.py — Formal YOLO model evaluation for ScreenSense V2

Runs model.val() on the test split and prints a clean metrics table.
Also logs results to W&B if available.

Usage:
    python evaluate_yolo.py
    python evaluate_yolo.py --split valid    # run on validation set instead
    python evaluate_yolo.py --no-wandb       # skip W&B upload
"""

import argparse
import os
import json
from pathlib import Path

# ── W&B ──────────────────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[W&B] wandb not installed — skipping dashboard logging.")


def run_evaluation(split: str = "test", use_wandb: bool = True):
    from ultralytics import YOLO

    MODEL_PATH   = "models/best.pt"
    DATA_YAML    = "ScreenSenseAI.v2i.yolov8/data.yaml"
    CLASS_NAMES  = ["close_button", "maximize_button", "minimize_button"]

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(DATA_YAML):
        print(f"[ERROR] Dataset YAML not found at {DATA_YAML}")
        return

    print(f"\n{'─'*60}")
    print(f"  ScreenSense YOLO Evaluation — split: {split.upper()}")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Data   : {DATA_YAML}")
    print(f"{'─'*60}\n")

    model = YOLO(MODEL_PATH)

    # ── Run validation ────────────────────────────────────────────────────────
    # Use the split keyword so Ultralytics reads from data.yaml correctly.
    results = model.val(
        data=DATA_YAML,
        split=split,        # "test" or "valid"
        conf=0.25,          # lower threshold for val to see full recall curve
        iou=0.5,
        verbose=False,
        plots=True,         # saves confusion matrix + PR curve to runs/detect/val/
    )

    # ── Extract metrics ───────────────────────────────────────────────────────
    mp    = float(results.box.mp)    # mean precision across classes
    mr    = float(results.box.mr)    # mean recall
    map50 = float(results.box.map50) # mAP@0.5
    map50_95 = float(results.box.map) # mAP@0.5:0.95

    # Per-class metrics (shape: [num_classes])
    p_per_class    = results.box.p.tolist()
    r_per_class    = results.box.r.tolist()
    ap50_per_class = results.box.ap50.tolist()

    # ── Print results table ───────────────────────────────────────────────────
    col_w = 20
    print(f"\n{'─'*60}")
    print("  YOLO OBJECT DETECTION — EVALUATION RESULTS")
    print(f"{'─'*60}")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'─'*35}")
    print(f"  {'Mean Precision (mP)':<25} {mp:>10.4f}")
    print(f"  {'Mean Recall (mR)':<25} {mr:>10.4f}")
    print(f"  {'mAP @ 0.5':<25} {map50:>10.4f}")
    print(f"  {'mAP @ 0.5:0.95':<25} {map50_95:>10.4f}")
    print(f"{'─'*60}")
    print(f"\n  {'Class':<22} {'Precision':>10} {'Recall':>10} {'AP@0.5':>10}")
    print(f"  {'─'*52}")
    for cls_name, p, r, ap in zip(CLASS_NAMES, p_per_class, r_per_class, ap50_per_class):
        print(f"  {cls_name:<22} {p:>10.4f} {r:>10.4f} {ap:>10.4f}")
    print(f"{'─'*60}\n")

    # Locate saved plots
    save_dir = results.save_dir
    cm_path  = Path(save_dir) / "confusion_matrix.png"
    pr_path  = Path(save_dir) / "PR_curve.png"
    print(f"  Confusion matrix : {cm_path}")
    print(f"  PR curve         : {pr_path}")
    print(f"  Full results dir : {save_dir}\n")

    # ── Save metrics to JSON (for the evaluation harness in Week 2) ───────────
    metrics_dict = {
        "model":       MODEL_PATH,
        "split":       split,
        "mean_precision": mp,
        "mean_recall":    mr,
        "map50":          map50,
        "map50_95":       map50_95,
        "per_class": {
            cls: {"precision": p, "recall": r, "ap50": ap}
            for cls, p, r, ap in zip(CLASS_NAMES, p_per_class, r_per_class, ap50_per_class)
        }
    }
    os.makedirs("runs/eval", exist_ok=True)
    json_path = f"runs/eval/yolo_metrics_{split}.json"
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Metrics saved    : {json_path}\n")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if WANDB_AVAILABLE and use_wandb:
        run = wandb.init(
            project="screensense-v2",
            name=f"yolo-eval-{split}",
            config={"model": MODEL_PATH, "split": split, "conf_thresh": 0.25, "iou": 0.5}
        )

        wandb.log({
            "yolo/mean_precision": mp,
            "yolo/mean_recall":    mr,
            "yolo/mAP50":         map50,
            "yolo/mAP50_95":      map50_95,
        })

        # Per-class table
        table = wandb.Table(columns=["Class", "Precision", "Recall", "AP@0.5"])
        for cls, p, r, ap in zip(CLASS_NAMES, p_per_class, r_per_class, ap50_per_class):
            table.add_data(cls, round(p, 4), round(r, 4), round(ap, 4))
        wandb.log({"yolo/per_class_metrics": table})

        # Upload confusion matrix image if it exists
        if cm_path.exists():
            wandb.log({"yolo/confusion_matrix": wandb.Image(str(cm_path))})
        if pr_path.exists():
            wandb.log({"yolo/PR_curve": wandb.Image(str(pr_path))})

        run.finish()
        print(f"  [W&B] Results logged → {wandb.run.url if wandb.run else 'see dashboard'}")

    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ScreenSense YOLO model")
    parser.add_argument("--split",    default="test", choices=["test", "valid"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    run_evaluation(split=args.split, use_wandb=not args.no_wandb)
