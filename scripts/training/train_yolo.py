"""
train_yolo.py

Train a YOLOv8 model on the MU-CPS dataset.

This script uses transfer learning from the official YOLOv8n
pre-trained weights and fine-tunes the detector using the
EnvoDat MU-CPS dataset.

Author: Tomas Agudelo
Project: Open Vocabulary Navigation
"""

from pathlib import Path
from ultralytics import YOLO


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

    print("=" * 60)
    print("YOLOv8 Training")
    print("=" * 60)

    print(f"Dataset: {dataset_yaml}")
    print(f"Output : {output_dir}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=30,
        imgsz=640,
        batch=2,
        workers=2,
        device="cpu",
        project=str(output_dir),
        name="mu_cps_yolov8n",
        exist_ok=True
    )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()