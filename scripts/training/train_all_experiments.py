"""
Student B — YOLO Benchmarking & Sensitivity Analysis
Train all experiment variants on EnvoDat MU-Hall dataset.

Experiments:
  1. YOLOv8n  — 25 epochs, batch=8,  imgsz=640, lr=0.01  (baseline extended)
  2. YOLOv8s  — 25 epochs, batch=8,  imgsz=640, lr=0.01
  3. YOLOv8n  — 25 epochs, batch=16, imgsz=640, lr=0.01  (batch sensitivity)
  4. YOLOv8n  — 25 epochs, batch=8,  imgsz=640, lr=0.001 (lr sensitivity)

NOTE: If CPU-only, set EPOCHS lower (e.g. 10) and it will still produce
      valid comparison data. The analysis script works for any epoch count.

Usage:
    python scripts/train_all_experiments.py
    python scripts/train_all_experiments.py --epochs 10   # faster on CPU
    python scripts/train_all_experiments.py --exp 0 1     # run only exp 0 and 1
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO

# ── Experiment definitions ──────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "yolov8n_e25_b8_lr001",
        "model": "yolov8n.pt",
        "epochs": 25,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.01,
        "notes": "YOLOv8n baseline extended",
    },
    {
        "name": "yolov8s_e25_b8_lr001",
        "model": "yolov8s.pt",
        "epochs": 25,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.01,
        "notes": "YOLOv8s larger model",
    },
    {
        "name": "yolov8n_e25_b16_lr001",
        "model": "yolov8n.pt",
        "epochs": 25,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "notes": "YOLOv8n batch sensitivity (batch=16)",
    },
    {
        "name": "yolov8n_e25_b8_lr0001",
        "model": "yolov8n.pt",
        "epochs": 25,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.001,
        "notes": "YOLOv8n lr sensitivity (lr=0.001)",
    },
]

# ── YOLOv11 variants (use if ultralytics >= 8.3) ────────────────────────────
EXPERIMENTS_V11 = [
    {
        "name": "yolov11n_e25_b8_lr001",
        "model": "yolo11n.pt",
        "epochs": 25,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.01,
        "notes": "YOLOv11n nano",
    },
    {
        "name": "yolov11s_e25_b8_lr001",
        "model": "yolo11s.pt",
        "epochs": 25,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.01,
        "notes": "YOLOv11s small",
    },
]


def run_experiment(exp: dict, data_yaml: Path, results_dir: Path) -> dict:
    """Train one experiment and return its timing info."""
    print(f"\n{'='*60}")
    print(f"  Starting: {exp['name']}")
    print(f"  Model: {exp['model']}  Epochs: {exp['epochs']}  "
          f"Batch: {exp['batch']}  LR: {exp['lr0']}")
    print(f"{'='*60}\n")

    model = YOLO(exp["model"])
    t0 = time.time()

    model.train(
        data=str(data_yaml),
        epochs=exp["epochs"],
        imgsz=exp["imgsz"],
        batch=exp["batch"],
        lr0=exp["lr0"],
        project=str(results_dir),
        name=exp["name"],
        exist_ok=True,
        device="cpu",   # remove this line if you have CUDA
        workers=2,      # lower CPU pressure
        cache=False,
    )

    elapsed = time.time() - t0
    print(f"\n  ✓ Done in {elapsed/60:.1f} min — saved to {results_dir / exp['name']}")
    return {"name": exp["name"], "training_time_min": round(elapsed / 60, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all experiments (e.g. 10 for quick test)")
    parser.add_argument("--exp", type=int, nargs="+", default=None,
                        help="Run only these experiment indices (0-based)")
    parser.add_argument("--include-v11", action="store_true",
                        help="Also run YOLOv11 experiments (requires ultralytics>=8.3)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "dataset" / "mu-hall-yolov8" / "envodata.yaml"
    results_dir = project_root / "results" / "yolo_training"

    if not data_yaml.exists():
        raise FileNotFoundError(f"YAML not found: {data_yaml}\n"
                                "Make sure you run from inside the open_vocab_nav folder.")

    results_dir.mkdir(parents=True, exist_ok=True)

    exps = EXPERIMENTS[:]
    if args.include_v11:
        exps += EXPERIMENTS_V11

    if args.exp is not None:
        exps = [exps[i] for i in args.exp]

    if args.epochs is not None:
        for e in exps:
            e["epochs"] = args.epochs
        print(f"  ⚠ Epoch override: all experiments will run for {args.epochs} epochs")

    timings = []
    for exp in exps:
        t = run_experiment(exp, data_yaml, results_dir)
        timings.append(t)

    print("\n\n=== All experiments complete ===")
    for t in timings:
        print(f"  {t['name']:45s} → {t['training_time_min']} min")
    print("\nNext: run  python scripts/consolidate_results.py")


if __name__ == "__main__":
    main()
