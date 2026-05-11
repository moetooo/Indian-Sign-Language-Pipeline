# -*- coding: utf-8 -*-
"""
Unified ISL Pipeline
=====================
One script to rule them all — takes a CSV or image folder, runs Phase 1→2→3,
and produces named models ready for inference.

Usage:
  # From a CSV file (landmarks already extracted):
  python run_pipeline.py --input data/my_landmarks.csv --name myset

  # From an image dataset folder (a/ b/ c/ ... z/ subfolders):
  python run_pipeline.py --input dataset/my_images --name myimg

  # Only specific phases:
  python run_pipeline.py --input data.csv --name test --phase 2 3

  # Custom results directory:
  python run_pipeline.py --input data.csv --name exp1 --outdir results_exp1

This will produce:
  results/isl_{name}_raw_mlp.h5           + scaler_{name}_raw.pkl
  results/isl_{name}_kinematic_mlp.h5     + scaler_{name}_kinematic.pkl
  results/isl_{name}_angles_only_mlp.h5   + scaler_{name}_angles_only.pkl
"""

from src.utils.logger import logger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.common import _hand_cols, _pose_cols, HAND_LANDMARK_COUNT, POSE_INDICES, AXES, CLASS_LABELS



import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

from src.utils.paths import DATA_RAW_DIR, DATA_KIN_DIR, DATA_ANG_DIR, MODELS_DIR, PLOTS_DIR, ensure_dirs

# ---------------------------------------------------------------------------
# Constants (same schema as Phase 1)
# ---------------------------------------------------------------------------
EXCLUDE_FOLDERS = {"{", ".", ".."}

META_COLS = ["label", "source", "user_id"]






LH_COLS = _hand_cols("lh")
RH_COLS = _hand_cols("rh")
POSE_COLS = _pose_cols()
ALL_COLS = META_COLS + LH_COLS + RH_COLS + POSE_COLS


# ---------------------------------------------------------------------------
# Input detection
# ---------------------------------------------------------------------------
def detect_input_type(path):
    """Returns 'csv' or 'images' based on input path."""
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return "csv"
    elif os.path.isdir(path):
        # Check for letter subfolders
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) >= 10:
            return "images"
        else:
            logger.info(f"  ERROR: Directory has only {len(subdirs)} subfolders.")
            logger.info(f"         Expected letter subfolders (a/, b/, c/, ...)")
            sys.exit(1)
    else:
        logger.info(f"  ERROR: '{path}' is neither a CSV file nor a directory.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 1: Image → Landmarks CSV
# ---------------------------------------------------------------------------
def phase1_extract(image_dir, output_csv, name):
    """Extract hand landmarks from images using MediaPipe Hands."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Phase 1 — Landmark Extraction [{name}]")
    logger.info(f"{'='*60}")

    class_dirs = sorted([
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d)) and d not in EXCLUDE_FOLDERS
    ])
    total_files = sum(
        len([f for f in os.listdir(os.path.join(image_dir, d))
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for d in class_dirs
    )
    logger.info(f"  Input:   {image_dir}")
    logger.info(f"  Classes: {len(class_dirs)}")
    logger.info(f"  Images:  ~{total_files}")
    logger.info(f"  Output:  {output_csv}")

    mp_hands = mp.solutions.hands
    rows = []
    total = 0
    skipped = 0
    t0 = time.time()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.1,
    ) as hands:
        for ci, cls_name in enumerate(class_dirs):
            cls_dir = os.path.join(image_dir, cls_name)
            label = cls_name.upper()
            images = sorted([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            ok = 0
            fail = 0

            for img_name in images:
                img = cv2.imread(os.path.join(cls_dir, img_name))
                if img is None:
                    fail += 1
                    continue

                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    fail += 1
                    skipped += 1
                    continue

                rh = [0.0] * (HAND_LANDMARK_COUNT * 3)
                lh = [0.0] * (HAND_LANDMARK_COUNT * 3)

                for hi, hlm in enumerate(results.multi_hand_landmarks):
                    if results.multi_handedness and hi < len(results.multi_handedness):
                        side = results.multi_handedness[hi].classification[0].label
                    else:
                        side = "Right" if hi == 0 else "Left"
                    target = rh if side == "Right" else lh
                    for i, lm in enumerate(hlm.landmark):
                        target[i * 3] = lm.x
                        target[i * 3 + 1] = lm.y
                        target[i * 3 + 2] = lm.z

                pose = [0.0] * (len(POSE_INDICES) * 3)
                rows.append([label, f"{name}_img", "dataset"] + lh + rh + pose)
                ok += 1
                total += 1

            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            logger.info(f"    [{ci+1:2d}/{len(class_dirs)}] {label}: {ok} ok, {fail} skip  ({rate:.1f} img/s)")

    df = pd.DataFrame(rows, columns=ALL_COLS)
    df.to_csv(output_csv, index=False)
    logger.info(f"\n  Done: {total} rows, {skipped} skipped → {output_csv}")
    return output_csv


# ---------------------------------------------------------------------------
# Phase 2: Kinematic engineering
# ---------------------------------------------------------------------------
def phase2_kinematic(raw_csv, kin_csv, ang_csv, name):
    """Apply Phase 2 kinematic transforms."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Phase 2 — Kinematic Engineering [{name}]")
    logger.info(f"{'='*60}")

    from kinematic_engineer import run_pipeline
    import kinematic_engineer as ke

    # Override output paths
    orig_full = ke.OUT_FULL_PATH
    orig_ang = ke.OUT_ANGLES_PATH
    ke.OUT_FULL_PATH = kin_csv
    ke.OUT_ANGLES_PATH = ang_csv

    run_pipeline(raw_csv)

    ke.OUT_FULL_PATH = orig_full
    ke.OUT_ANGLES_PATH = orig_ang
    logger.info(f"  Outputs: {kin_csv}, {ang_csv}")


# ---------------------------------------------------------------------------
# Phase 3: Training
# ---------------------------------------------------------------------------
def phase3_train(raw_csv, kin_csv, ang_csv, name, outdir):
    """Train models with custom naming."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Phase 3 — Training [{name}]")
    logger.info(f"{'='*60}")

    import train_classifier as tc

    orig_runs = tc.ABLATION_RUNS
    orig_dir = tc.MODELS_DIR

    tc.MODELS_DIR = outdir
    tc.RESULTS_DIR = outdir
    os.makedirs(outdir, exist_ok=True)

    tc.ABLATION_RUNS = [
        (f"{name}_raw", raw_csv, "label", ["label", "source", "user_id"]),
        (f"{name}_kinematic", kin_csv, "label", ["label", "source", "user_id"]),
        (f"{name}_angles_only", ang_csv, "label", ["label", "source", "user_id"]),
    ]

    tc.main()

    tc.ABLATION_RUNS = orig_runs
    tc.MODELS_DIR = orig_dir
    tc.RESULTS_DIR = orig_dir

    logger.info(f"\n  Models saved to: {outdir}/")
    logger.info(f"    isl_{name}_raw_mlp.h5          + scaler_{name}_raw.pkl")
    logger.info(f"    isl_{name}_kinematic_mlp.h5    + scaler_{name}_kinematic.pkl")
    logger.info(f"    isl_{name}_angles_only_mlp.h5  + scaler_{name}_angles_only.pkl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified ISL Pipeline — CSV or Images → Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --input my_data.csv --name experiment1
  python run_pipeline.py --input dataset/images --name gestures
  python run_pipeline.py --input data.csv --name test --phase 2 3
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a landmarks CSV or image dataset directory"
    )
    parser.add_argument(
        "--name", required=True,
        help="Name prefix for models (e.g. 'exp1' → isl_exp1_raw_mlp.h5)"
    )
    parser.add_argument(
        "--phase", type=int, nargs="+", default=[1, 2, 3],
        help="Phases to run (default: 1 2 3)"
    )
    parser.add_argument(
        "--outdir", default="models",
        help="Output directory for models (default: models/)"
    )
    args = parser.parse_args()

    input_type = detect_input_type(args.input)
    raw_csv = os.path.join(DATA_RAW_DIR, f"{args.name}_raw_data.csv")
    kin_csv = os.path.join(DATA_KIN_DIR, f"{args.name}_kinematic_data.csv")
    ang_csv = os.path.join(DATA_ANG_DIR, f"{args.name}_angles_only.csv")

    ensure_dirs()

    logger.info(f"\n{'='*60}")
    logger.info(f"  Unified ISL Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"  Input:   {args.input} ({input_type})")
    logger.info(f"  Name:    {args.name}")
    logger.info(f"  Phases:  {args.phase}")
    logger.info(f"  Output:  {args.outdir}/")

    # Phase 1
    if 1 in args.phase:
        if input_type == "images":
            phase1_extract(args.input, raw_csv, args.name)
        elif input_type == "csv":
            # CSV already has landmarks — just copy/use as raw
            raw_csv = args.input
            logger.info(f"\n  Phase 1: Skipped (CSV input → using {raw_csv} directly)")
    else:
        if input_type == "csv":
            raw_csv = args.input

    # Phase 2
    if 2 in args.phase:
        if not os.path.exists(raw_csv):
            logger.info(f"  ERROR: {raw_csv} not found. Run Phase 1 first.")
            sys.exit(1)
        phase2_kinematic(raw_csv, kin_csv, ang_csv, args.name)

    # Phase 3
    if 3 in args.phase:
        if not os.path.exists(kin_csv):
            logger.info(f"  ERROR: {kin_csv} not found. Run Phase 1+2 first.")
            sys.exit(1)
        phase3_train(raw_csv, kin_csv, ang_csv, args.name, args.outdir)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Pipeline Complete! [{args.name}]")
    logger.info(f"{'='*60}")
    logger.info(f"\n  To use your models:")
    logger.info(f"    python realtime_inference.py --model {args.name}_raw")
    logger.info(f"    python realtime_inference.py --model {args.name}_kinematic")
    logger.info(f"    python realtime_inference.py --model {args.name}_angles_only")
    logger.info(f"\n  NOTE: Add your models to MODEL_CONFIGS in realtime_inference.py")
    logger.info(f"        to use them with live switching (keys 1-9).\n")


if __name__ == "__main__":
    main()
