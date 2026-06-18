"""
train_yolo.py

Train YOLO models on the MU-CPS dataset.

Change MODEL_NAME and EXPERIMENT_NAME to run different
architectures or hyperparameter configurations.

Author: Tomas Agudelo
Project: Open Vocabulary Navigation
"""

from pathlib import Path
from ultralytics import YOLO
import time


# ============================================================
# AVAILABLE MODELS
# ============================================================

# YOLOv8
# yolov8n.pt
# yolov8s.pt
# yolov8m.pt
# yolov8l.pt
# yolov8x.pt

# YOLOv9
# yolov9t.pt
# yolov9s.pt
# yolov9m.pt
# yolov9c.pt
# yolov9e.pt

# YOLOv10
# yolov10n.pt
# yolov10s.pt
# yolov10m.pt
# yolov10b.pt
# yolov10l.pt
# yolov10x.pt

# YOLO11
# yolo11n.pt
# yolo11s.pt
# yolo11m.pt
# yolo11l.pt
# yolo11x.pt


# ============================================================
# EXPERIMENT
# ============================================================

MODEL_NAME = "yolov10n.pt"

EXPERIMENT_NAME = "batch6"


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 10
IMAGE_SIZE = 640
BATCH_SIZE = 6


# ============================================================
# OPTIMIZER
# ============================================================

INITIAL_LR = 0.01
FINAL_LR_FACTOR = 0.1

MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005


# ============================================================
# EARLY STOPPING
# ============================================================

PATIENCE = 50


# ============================================================
# HARDWARE
# ============================================================

DEVICE = "cpu"


# ============================================================
# MAIN
# ============================================================

def main():

    project_root = Path(__file__).resolve().parents[2]

    dataset_yaml = (
        project_root
        / "data"
        / "datasets"
        / "mu-cps-yolov8"
        / "envodat-mu-cps-yolov8.yaml"
    )

    output_dir = (
        project_root
        / "models"
        / "yolo"
    )

    model_id = Path(MODEL_NAME).stem

    run_name = (
        f"mu_cps_{model_id}_{EXPERIMENT_NAME}"
    )

    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)

    print(f"Model        : {MODEL_NAME}")
    print(f"Experiment   : {EXPERIMENT_NAME}")
    print(f"Dataset      : {dataset_yaml}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Image Size   : {IMAGE_SIZE}")
    print(f"Initial LR   : {INITIAL_LR}")
    print(f"Device       : {DEVICE}")
    print()

    model = YOLO(MODEL_NAME)

    start_time = time.time()

    model.train(
        data=str(dataset_yaml),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        lr0=INITIAL_LR,

        lrf=FINAL_LR_FACTOR,

        momentum=MOMENTUM,

        weight_decay=WEIGHT_DECAY,

        patience=PATIENCE,

        workers=2,

        device=DEVICE,

        project=str(output_dir),

        name=run_name,

        exist_ok=True
    )

    training_time = time.time() - start_time

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Training time: "
        f"{training_time / 3600:.2f} hours"
    )

    print(
        f"Results saved in:\n"
        f"{output_dir / run_name}"
    )


if __name__ == "__main__":
    main()