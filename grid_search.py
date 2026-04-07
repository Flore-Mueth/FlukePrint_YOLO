import itertools
import csv
import time
import os
import sys
import signal
import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_YAML = "./17_03_dataset/data.yaml"
RESULTS_CSV  = "./grid_search_results.csv"
RUN_DIR      = "./grid_search_runs"

# ── Parameter Grid ────────────────────────────────────────────────────────────
PARAM_GRID = {
    "epochs":  [50],
    "batch":   [8, 16],
    "lr0":     [0.001, 0.01],
    "hsv_h":   [0],              
    "hsv_s":   [0, 0.35, 0.7],
    "hsv_v":   [0.2, 0.4],
    "mosaic":  [0.5, 1.0],
    "degrees": [360],            
}

# ── Baseline ──────────────────────────────────────────────────────────────────
BASELINE = {
    "epochs": 100, "batch": 16, "lr0": 0.01,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "mosaic": 1.0, "degrees": 0.0,
}

# ── CSV Helpers ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "run_name", "run_type", "epochs", "batch", "lr0",
    "hsv_h", "hsv_s", "hsv_v", "mosaic", "degrees",
    "seg_map50", "seg_map50_95", "seg_precision", "seg_recall",
    "box_map50", "box_map50_95", "box_precision", "box_recall",
    "train_time_s", "inference_ms_per_img", "status",
]

def init_csv():
    if not Path(RESULTS_CSV).exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

def is_run_complete(run_name):
    """Check if run is in CSV with status 'ok' AND weights exist on disk."""
    if not Path(RESULTS_CSV).exists():
        return False
    try:
        df = pd.read_csv(RESULTS_CSV)
        if run_name in df['run_name'].values:
            row = df[df['run_name'] == run_name].iloc[0]
            if str(row['status']) == 'ok':
                weights_path = f"{RUN_DIR}/{run_name}/weights/best.pt"
                if os.path.exists(weights_path):
                    return True
    except Exception:
        pass
    return False

def save_row(row):
    """Appends a row to CSV. Creates file if missing."""
    file_exists = Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def get_metrics(m):
    """Extract seg + box metrics from a val() result object."""
    def safe(fn):
        try: 
            val = fn()
            return round(float(val), 4) if val is not None else None
        except Exception:
            return None
    
    # Handle cases where seg or box might be missing depending on model type
    seg_metrics = {}
    box_metrics = {}
    
    if hasattr(m, 'seg') and m.seg is not None:
        seg_metrics = {
            "seg_map50":     safe(lambda: m.seg.map50),
            "seg_map50_95":  safe(lambda: m.seg.map),
            "seg_precision": safe(lambda: m.seg.p[0]),
            "seg_recall":    safe(lambda: m.seg.r[0]),
        }
    else:
        seg_metrics = {k: None for k in ["seg_map50", "seg_map50_95", "seg_precision", "seg_recall"]}

    if hasattr(m, 'box') and m.box is not None:
        box_metrics = {
            "box_map50":     safe(lambda: m.box.map50),
            "box_map50_95":  safe(lambda: m.box.map),
            "box_precision": safe(lambda: m.box.p[0]),
            "box_recall":    safe(lambda: m.box.r[0]),
        }
    else:
        box_metrics = {k: None for k in ["box_map50", "box_map50_95", "box_precision", "box_recall"]}

    inf_time = None
    if hasattr(m, 'speed') and 'inference' in m.speed:
        inf_time = safe(lambda: m.speed['inference'])

    return {**seg_metrics, **box_metrics, "inference_ms_per_img": inf_time}

def clear_gpu():
    """Forces CUDA to clear memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ── Training Function ─────────────────────────────────────────────────────────
def run_training(run_name, run_type, params):
    # 1. Security Check
    if is_run_complete(run_name):
        print(f"  ✅ SKIP (Completed & Verified): {run_name}")
        return

    # Check for incomplete previous run
    if Path(RESULTS_CSV).exists():
        try:
            df = pd.read_csv(RESULTS_CSV)
            if run_name in df['run_name'].values:
                status = df[df['run_name'] == run_name]['status'].iloc[0]
                if str(status) not in ['ok', 'running']:
                     print(f"  ⚠️  RESUME (Found failed run): {run_name}")
        except Exception:
            pass

    print(f"\n{'='*60}\n  🚀 RUN  : {run_name}\n  PARAMS: {params}")

    row = {
        "run_name": run_name, "run_type": run_type, **params,
        "seg_map50": None, "seg_map50_95": None,
        "seg_precision": None, "seg_recall": None,
        "box_map50": None, "box_map50_95": None,
        "box_precision": None, "box_recall": None,
        "train_time_s": None, "inference_ms_per_img": None,
        "status": "running"
    }
    
    # Save "running" state immediately
    save_row(row)

    model = None
    try:
        model = YOLO("yolov8n-seg.pt")

        t0 = time.time()
        model.train(
            data=DATASET_YAML, project=RUN_DIR,
            name=run_name, exist_ok=True, verbose=False, **params
        )
        row["train_time_s"] = round(time.time() - t0, 1)

        best_weights_path = f"{RUN_DIR}/{run_name}/weights/best.pt"
        if not os.path.exists(best_weights_path):
            raise FileNotFoundError("Training finished but best.pt not found.")
            
        best = YOLO(best_weights_path)
        metrics = best.val(data=DATASET_YAML, split="test", verbose=False, plots=False)
        
        row.update(get_metrics(metrics))
        row["status"] = "ok"

    except KeyboardInterrupt:
        print(f"\n  ⛔ INTERRUPTED by user: {run_name}")
        row["status"] = "interrupted"
        save_row(row)
        sys.exit(0) # Stop the whole script gracefully

    except Exception as e:
        err_msg = str(e)
        print(f"  ❌ ERROR: {err_msg}")
        row["status"] = f"error: {err_msg[:50]}" # Truncate long errors
        save_row(row)
    
    finally:
        # Always save final state and clear GPU
        if row["status"] != "interrupted": # Don't overwrite interrupt status
             save_row(row)
        if model:
            del model
        clear_gpu()
        
    if row["status"] == "ok":
        print(f"  ✅ Done: mAP50-95={row['seg_map50_95']} | Time={row['train_time_s']}s")

# ── Main Execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🔍 Checking GPU...")
    if torch.cuda.is_available():
        print(f"   ✅ GPU Found: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  WARNING: No GPU detected. Running on CPU will be very slow.")

    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    init_csv()

    # 1. Baseline
    print("\n🏁 Starting Baseline...")
    run_training("baseline", "baseline", BASELINE)

    # 2. Grid Search
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total_runs = len(combos)
    
    print(f"\n📊 Starting Grid Search: {total_runs} combinations...")

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        
        # Secure Naming: Use 'p' for decimal points to avoid ambiguity
        name = (
            f"ep{params['epochs']}"
            f"_bs{params['batch']}"
            f"_lr{str(params['lr0']).replace('.','p')}"
            f"_s{str(params['hsv_s']).replace('.','p')}"
            f"_v{str(params['hsv_v']).replace('.','p')}"
            f"_mos{str(params['mosaic']).replace('.','p')}"
        )
        
        print(f"\n[{i:03d}/{total_runs}]")
        run_training(name, "grid", params)

    print("\n🎉 Grid search complete. Check grid_search_results.csv")
    
    # Optional: Auto-generate plot if pandas/matplotlib are available
    try:
        df = pd.read_csv(RESULTS_CSV)
        df_ok = df[df["status"] == "ok"].copy()
        if len(df_ok) > 0:
            print(f"\n📈 Generating summary plot for {len(df_ok)} successful runs...")
            # (Include your plotting logic from Cell 7 here if desired)
            # For brevity in this script, we just print the top 3
            top3 = df_ok.sort_values("seg_map50_95", ascending=False).head(3)
            print("\n🏆 Top 3 Configurations:")
            print(top3[["run_name", "seg_map50_95", "epochs", "batch", "lr0"]].to_string(index=False))
        else:
            print("\n⚠️ No successful runs found to plot.")
    except Exception as e:
        print(f"\n⚠️ Could not generate plot: {e}")