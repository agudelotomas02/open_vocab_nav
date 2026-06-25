import pandas as pd
from pathlib import Path
import yaml

# =========================
# CONFIG
# =========================
EXPERIMENTS_DIR = Path("models/yolo")
OUTPUT_CSV = Path("results/consolidated_results.csv")
OUTPUT_MD = Path("results/comparison_table.md")

# =========================
# FUNCIONES
# =========================
def extract_params_from_yaml(yaml_path):
    """Lee parámetros de args.yaml"""
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r") as f:
        try:
            data = yaml.safe_load(f)
            return {
                "model": data.get("model"),
                "epochs": data.get("epochs"),
                "batch_size": data.get("batch"),
                "lr": data.get("lr0")
            }
        except Exception:
            return {}

def consolidate_results():
    rows = []

    for exp_dir in EXPERIMENTS_DIR.glob("*"):
        if not exp_dir.is_dir():
            continue

        csv_file = exp_dir / "results.csv"
        yaml_file = exp_dir / "args.yaml"
        params = extract_params_from_yaml(yaml_file)

        # GPU/CPU según nombre
        gpu_cpu = "GPU" if "GPU" in exp_dir.name else "CPU"

        # YOLO version según carpeta
        yolo_version = None
        for prefix in ["yolov8", "yolov9", "yolov10", "yolov11"]:
            if prefix in exp_dir.name.lower():
                yolo_version = prefix
                break

        if not csv_file.exists():
            print(f"No results.csv found for {exp_dir}")
            continue

        df = pd.read_csv(csv_file)
        last_row = df.iloc[-1]

        row = {
            "yolo_version": yolo_version,
            "gpu_cpu": gpu_cpu,
            "experiment": exp_dir.name,
            "model": params.get("model"),
            "epochs": params.get("epochs"),
            "batch_size": params.get("batch_size"),
            "lr": params.get("lr"),
            "mAP50": last_row.get("metrics/mAP50(B)", None),
            "mAP50-95": last_row.get("metrics/mAP50-95(B)", None),
            "precision": last_row.get("metrics/precision(B)", None),
            "recall": last_row.get("metrics/recall(B)", None),
            "loss": last_row.get("val/box_loss", None)
        }
        rows.append(row)

    df_all = pd.DataFrame(rows)
    df_all.sort_values(["yolo_version", "gpu_cpu", "experiment"], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"Consolidated results saved to {OUTPUT_CSV}")

    # Markdown table
    md_table = df_all[["experiment","yolo_version","gpu_cpu","model","epochs","batch_size","lr",
                       "mAP50","mAP50-95","precision","recall","loss"]]
    md_table_str = md_table.to_markdown(index=False)
    OUTPUT_MD.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(md_table_str)
    print(f"Markdown table saved to {OUTPUT_MD}")
    print(md_table_str)


if __name__ == "__main__":
    consolidate_results()