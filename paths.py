# -*- coding: utf-8 -*-
"""
Central Path Configuration
===========================
Single source of truth for all directory and file paths.
All scripts import from here to stay consistent.
"""
import os

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------
DATA_RAW_DIR  = os.path.join("data", "raw")
DATA_KIN_DIR  = os.path.join("data", "kinematic")
DATA_ANG_DIR  = os.path.join("data", "angles")
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"

ALL_DIRS = [DATA_RAW_DIR, DATA_KIN_DIR, DATA_ANG_DIR, MODELS_DIR, PLOTS_DIR]


def ensure_dirs():
    """Create all project directories if they don't exist."""
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 1 — Raw landmark CSVs
# ---------------------------------------------------------------------------
ISL_RAW_CSV  = os.path.join(DATA_RAW_DIR, "isl_raw_data.csv")
IMG_RAW_CSV  = os.path.join(DATA_RAW_DIR, "img_raw_data.csv")

# ---------------------------------------------------------------------------
# Phase 2 — Kinematic CSVs
# ---------------------------------------------------------------------------
ISL_KIN_CSV  = os.path.join(DATA_KIN_DIR, "isl_kinematic_data.csv")
IMG_KIN_CSV  = os.path.join(DATA_KIN_DIR, "img_kinematic_data.csv")

ISL_ANG_CSV  = os.path.join(DATA_ANG_DIR, "isl_angles_only.csv")
IMG_ANG_CSV  = os.path.join(DATA_ANG_DIR, "img_angles_only.csv")

# ---------------------------------------------------------------------------
# Phase 3 — Models, scalers, plots
# ---------------------------------------------------------------------------
def model_path(run_name):
    return os.path.join(MODELS_DIR, f"isl_{run_name}_mlp.h5")

def scaler_path(run_name):
    return os.path.join(MODELS_DIR, f"scaler_{run_name}.pkl")

def cm_path(run_name):
    return os.path.join(PLOTS_DIR, f"cm_{run_name}.png")

def history_path(run_name):
    return os.path.join(PLOTS_DIR, f"history_{run_name}.png")

def ablation_csv():
    return os.path.join(PLOTS_DIR, "ablation_summary.csv")
