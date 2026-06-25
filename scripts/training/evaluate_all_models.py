"""
Student B — Evaluate all trained models on validation and test splits.

Reads every best.pt from results/yolo_training/*/weights/best.pt
and runs model.val() on both splits, saving per-run metric JSON files.

Usage:
    python scripts/evaluate_all_models.py
"""

import json
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR    = PROJECT_ROOT / "results" / "yolo_training"
DATA_YAML    = PROJECT_ROOT / "dataset" / "mu-hall-yolov8" / "envodata.yaml"


def evaluate_model(run_folder: Path):
    weights = run_folder / "weights" / "best.pt"
    if not weights.exists():
        print(f"  ⚠ No best.pt in {run_folder.name}, skipping.")
        return

    model = YOLO(str(weights))
    results_out = {}

    for split in ["val", "test"]:
        print(f"    Evaluating {run_folder.name} on [{split}]...")
        t0 = time.time()
        metrics = model.val(
            data=str(DATA_YAML),
            split=split,
            imgsz=640,
            device="cpu",
            project=str(TRAIN_DIR),
            name=f"{run_folder.name}_eval_{split}",
            exist_ok=True,
            verbose=False,
        )
        elapsed = time.time() - t0
        results_out[split] = {
            "precision": round(float(metrics.box.mp), 4),
            "recall":    round(float(metrics.box.mr), 4),
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50-95":  round(float(metrics.box.map), 4),
            "eval_time_sec": round(elapsed, 1),
        }
        print(f"      mAP50={results_out[split]['mAP50']}  "
              f"mAP50-95={results_out[split]['mAP50-95']}  "
              f"({elapsed:.0f}s)")

    out = run_folder / "eval_summary.json"
    with open(out, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"    ✓ Saved: {out}\n")


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"YAML not found: {DATA_YAML}")

    folders = sorted(f for f in TRAIN_DIR.iterdir() if f.is_dir()
                     and (f / "weights" / "best.pt").exists())

    if not folders:
        print("No trained models found. Run train_all_experiments.py first.")
        return

    print(f"Found {len(folders)} model(s) to evaluate:\n")
    for f in folders:
        print(f"  {f.name}")
    print()

    for folder in folders:
        print(f"► {folder.name}")
        evaluate_model(folder)

    print("✓ All evaluations complete.")
    print("Next: run  python scripts/consolidate_results.py")


if __name__ == "__main__":
    main()
