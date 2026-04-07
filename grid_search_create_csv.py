import os
import csv
import glob
import re
import torch
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
# Point to the parent folder containing all the run directories
ROOT_DIR = "./runs/segment/grid_search_runs"
DATASET_YAML = "./17_03_dataset/data.yaml"
OUTPUT_CSV = "./clean_grid_search_results.csv"

# ── Helper: Parse Hyperparameters from Folder Name ───────────────────────────
def parse_run_name(name):
    """
    Extracts parameters from folder names like:
    ep50_bs8_lr0p01_s0_v0p4_mos1p0
    Returns a dict of parameters.
    """
    params = {
        "epochs": None, "batch": None, "lr0": None, 
        "hsv_s": None, "hsv_v": None, "mosaic": None
    }
    
    # Regex patterns for your specific naming convention
    patterns = {
        "epochs": r"ep(\d+)",
        "batch": r"bs(\d+)",
        "lr0": r"lr([0-9]+p[0-9]+)", # Captures 0p01, 0p001 etc.
        "hsv_s": r"_s([0-9]+p?[0-9]*)", # Captures s0, s0p35, s0p7
        "hsv_v": r"_v([0-9]+p[0-9]+)",
        "mosaic": r"mos([0-9]+p[0-9]+)"
    }
    
    try:
        # Epochs
        m = re.search(patterns["epochs"], name)
        if m: params["epochs"] = int(m.group(1))
        
        # Batch
        m = re.search(patterns["batch"], name)
        if m: params["batch"] = int(m.group(1))
        
        # Learning Rate (convert '0p01' -> '0.01')
        m = re.search(patterns["lr0"], name)
        if m: params["lr0"] = float(m.group(1).replace('p', '.'))
        
        # HSV Saturation (Handle 's0' vs 's0p35')
        # We look for _s followed by numbers until the next underscore or end
        m = re.search(r"_s([0-9]+(?:p[0-9]+)?)", name)
        if m: 
            val = m.group(1).replace('p', '.')
            params["hsv_s"] = float(val)
            
        # HSV Value
        m = re.search(patterns["hsv_v"], name)
        if m: params["hsv_v"] = float(m.group(1).replace('p', '.'))
        
        # Mosaic
        m = re.search(patterns["mosaic"], name)
        if m: params["mosaic"] = float(m.group(1).replace('p', '.'))
        
        return params
    except Exception as e:
        print(f"⚠️ Could not parse '{name}': {e}")
        return None

# ── Helper: Validate Model ───────────────────────────────────────────────────
def validate_model(weights_path, data_yaml):
    """Loads a model and runs validation on the test split."""
    try:
        model = YOLO(weights_path)
        # Run validation. plots=False speeds it up.
        metrics = model.val(data=data_yaml, split="test", verbose=False, plots=False)
        
        return {
            "seg_map50": round(float(metrics.seg.map50), 4),
            "seg_map50_95": round(float(metrics.seg.map), 4),
            "seg_precision": round(float(metrics.seg.p[0]), 4),
            "seg_recall": round(float(metrics.seg.r[0]), 4),
            "box_map50": round(float(metrics.box.map50), 4),
            "box_map50_95": round(float(metrics.box.map), 4),
            "box_precision": round(float(metrics.box.p[0]), 4),
            "box_recall": round(float(metrics.box.r[0]), 4),
            "inference_ms": round(float(metrics.speed["inference"]), 2),
            "status": "ok"
        }
    except Exception as e:
        print(f"  ❌ Error validating {os.path.basename(weights_path)}: {e}")
        return {
            "seg_map50": None, "seg_map50_95": None, "seg_precision": None, "seg_recall": None,
            "box_map50": None, "box_map50_95": None, "box_precision": None, "box_recall": None,
            "inference_ms": None, "status": f"error: {str(e)}"
        }

# ── Main Execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🔍 Scanning directory: {os.path.abspath(ROOT_DIR)}")
    
    if not os.path.exists(ROOT_DIR):
        print("❌ Directory not found. Please check the ROOT_DIR path.")
        exit()

    # Find all folders that contain a weights/best.pt file
    runs_data = []
    best_weights_list = glob.glob(os.path.join(ROOT_DIR, "*", "weights", "best.pt"))
    
    print(f"📦 Found {len(best_weights_list)} valid 'best.pt' files.")
    
    if len(best_weights_list) == 0:
        print("⚠️ No models found. Ensure training has completed successfully.")
        exit()

    # CSV Headers
    fieldnames = [
        "run_name", "run_type", "epochs", "batch", "lr0", "hsv_s", "hsv_v", "mosaic",
        "seg_map50", "seg_map50_95", "seg_precision", "seg_recall",
        "box_map50", "box_map50_95", "box_precision", "box_recall",
        "inference_ms", "status"
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, weights_path in enumerate(best_weights_list, 1):
            # Extract folder name (e.g., ep50_bs8_...)
            run_name = os.path.basename(os.path.dirname(os.path.dirname(weights_path)))
            
            print(f"[{i}/{len(best_weights_list)}] Processing: {run_name}")
            
            # 1. Parse Params from Name
            params = parse_run_name(run_name)
            if params is None:
                # Fallback for 'baseline' or unparseable names
                params = {k: None for k in ["epochs", "batch", "lr0", "hsv_s", "hsv_v", "mosaic"]}
                run_type = "baseline" if "baseline" in run_name else "unknown"
            else:
                run_type = "baseline" if "baseline" in run_name else "grid"

            # 2. Validate Model (Get Metrics)
            # We re-run validation here to ensure metrics are comparable (same test split, same IoU thresholds)
            metrics = validate_model(weights_path, DATASET_YAML)
            
            # 3. Compile Row
            row = {
                "run_name": run_name,
                "run_type": run_type,
                **params,
                **metrics
            }
            
            writer.writerow(row)
            
            if metrics["status"] == "ok":
                print(f"   ✅ mAP50-95: {metrics['seg_map50_95']}")
            else:
                print(f"   ❌ Failed validation")

    print(f"\n🎉 Done! Clean results saved to: {OUTPUT_CSV}")
    print(f"   You can now load this CSV in pandas or Excel for analysis.")