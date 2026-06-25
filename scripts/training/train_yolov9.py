import argparse
import time
from pathlib import Path
from ultralytics import YOLO

EXPERIMENTS = {
    "t": {
        "name":   "yolov9t_e10_b8_lr001",
        "model":  "yolov9t.pt",
        "epochs": 10,
        "batch":  8,
        "imgsz":  640,
        "lr0":    0.01,
    },
    "s": {
        "name":   "yolov9s_e10_b8_lr001",
        "model":  "yolov9s.pt",
        "epochs": 10,
        "batch":  8,
        "imgsz":  640,
        "lr0":    0.01,
    },
    "m": {
        "name":   "yolov9m_e10_b8_lr001",
        "model":  "yolov9m.pt",
        "epochs": 10,
        "batch":  8,
        "imgsz":  640,
        "lr0":    0.01,
    },
}


def run_experiment(exp, data_yaml, results_dir):
    print("\n" + "="*60)
    print("  Starting : " + exp["name"])
    print("  Model    : " + exp["model"])
    print("  Epochs   : " + str(exp["epochs"]) + "  Batch: " + str(exp["batch"]) + "  LR: " + str(exp["lr0"]))
    print("="*60 + "\n")

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
        device="cpu",
        workers=2,
        cache=False,
    )

    elapsed = time.time() - t0
    print("\n  Done in " + str(round(elapsed/60, 1)) + " min")
    return {"name": exp["name"], "training_time_min": round(elapsed / 60, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["t", "s", "m", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_yaml    = project_root / "dataset" / "mu-hall-yolov8" / "envodata.yaml"
    results_dir  = project_root / "results" / "yolo_training"

    if not data_yaml.exists():
        raise FileNotFoundError(
            "YAML not found: " + str(data_yaml) +
            "\nMake sure you run from inside the open_vocab_nav folder:" +
            "\ncd C:/Users/Somya/Downloads/open_vocab_nav/open_vocab_nav" +
            "\npython scripts/train_yolov9.py"
        )

    results_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        selected = list(EXPERIMENTS.values())
    else:
        selected = [EXPERIMENTS[args.model]]

    if args.epochs is not None:
        for e in selected:
            e["epochs"] = args.epochs
        print("Epoch override: " + str(args.epochs) + " epochs")

    print("\nYOLOv9 Training — " + str(len(selected)) + " experiment(s) queued:")
    for e in selected:
        print("  - " + e["name"] + " (" + str(e["epochs"]) + " epochs)")
    print()

    timings = []
    for exp in selected:
        t = run_experiment(exp, data_yaml, results_dir)
        timings.append(t)

    total = sum(t["training_time_min"] for t in timings)
    print("\n=== All YOLOv9 experiments complete ===")
    for t in timings:
        print("  " + t["name"] + "  ->  " + str(t["training_time_min"]) + " min")
    print("\n  Total time: " + str(round(total/60, 1)) + " hours")
    print("\n  Next step: python scripts/consolidate_results.py")


if __name__ == "__main__":
    main()
