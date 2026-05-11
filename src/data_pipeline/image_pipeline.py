# -*- coding: utf-8 -*-
"""
Image Dataset Pipeline — Phase 1 + 2 + 3
==========================================
Processes the Gesture Speech image dataset through the full ISL pipeline:
  1. Extract hand landmarks from images via MediaPipe
  2. Apply Phase 2 kinematic engineering
  3. Save CSVs ready for training

Usage:
    python image_pipeline.py                              # full run
    python image_pipeline.py --phase 1                    # extraction only
    python image_pipeline.py --phase 2                    # kinematic only
    python image_pipeline.py --phase 3                    # training only
    python image_pipeline.py --limit 50                   # limit images per class
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join("dataset", "dataset - Gesture Speech")
RAW_CSV     = os.path.join("data", "raw", "img_raw_data.csv")
KIN_CSV     = os.path.join("data", "kinematic", "img_kinematic_data.csv")
ANG_CSV     = os.path.join("data", "angles", "img_angles_only.csv")
RESULTS_DIR = "models"


# Excluded folders
EXCLUDE_FOLDERS = {"{"}

# Column names (matching Phase 1 schema)


META_COLS = ["label", "source", "user_id"]
LH_COLS   = _hand_cols("lh")
RH_COLS   = _hand_cols("rh")
POSE_COLS = _pose_cols()
ALL_FEATURE_COLS = LH_COLS + RH_COLS + POSE_COLS
ALL_COLS  = META_COLS + ALL_FEATURE_COLS

mp_holistic = mp.solutions.holistic


# ---------------------------------------------------------------------------
# Phase 1: Extract landmarks from images
# ---------------------------------------------------------------------------
def extract_from_images(dataset_dir, output_csv, limit_per_class=None):
    """Process all images in the dataset directory and extract landmarks."""
    logger.info(f"\n{'='*60}")
    logger.info("  Phase 1 — Image Landmark Extraction")
    logger.info(f"{'='*60}")
    logger.info(f"  Dataset: {dataset_dir}")
    logger.info(f"  Output:  {output_csv}")
    if limit_per_class:
        logger.info(f"  Limit:   {limit_per_class} images/class")

    # Discover classes
    class_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d not in EXCLUDE_FOLDERS
    ])
    logger.info(f"  Classes: {len(class_dirs)} ({class_dirs[:5]}...)")
    logger.info(f"  Using: MediaPipe Hands (static_image_mode=True)")

    rows = []
    total_images = 0
    skipped = 0
    t_start = time.time()

    # Use MediaPipe Hands instead of Holistic — much better for isolated hand images
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.1,
    ) as hands:

        for cls_idx, cls_name in enumerate(class_dirs):
            cls_dir = os.path.join(dataset_dir, cls_name)
            label = cls_name.upper()  # a -> A, b -> B, etc.

            # Get image files
            images = sorted([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if limit_per_class:
                images = images[:limit_per_class]

            cls_count = 0
            cls_skipped = 0

            for img_name in images:
                img_path = os.path.join(cls_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    cls_skipped += 1
                    continue

                # Process with MediaPipe Hands
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if not results.multi_hand_landmarks:
                    cls_skipped += 1
                    skipped += 1
                    continue

                # Extract hands — assign based on handedness
                rh = [0.0] * (HAND_LANDMARK_COUNT * 3)
                lh = [0.0] * (HAND_LANDMARK_COUNT * 3)

                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine handedness
                    if results.multi_handedness and hand_idx < len(results.multi_handedness):
                        hand_label = results.multi_handedness[hand_idx].classification[0].label
                    else:
                        hand_label = "Right" if hand_idx == 0 else "Left"

                    target = rh if hand_label == "Right" else lh
                    for i, lm in enumerate(hand_landmarks.landmark):
                        target[i * 3]     = lm.x
                        target[i * 3 + 1] = lm.y
                        target[i * 3 + 2] = lm.z

                # No pose data from hand-only images — fill with zeros
                pose = [0.0] * (len(POSE_INDICES) * 3)

                # Build row
                row = [label, "gesture_speech_img", "dataset"] + lh + rh + pose
                rows.append(row)
                cls_count += 1
                total_images += 1

            elapsed = time.time() - t_start
            rate = total_images / elapsed if elapsed > 0 else 0
            logger.info(f"    [{cls_idx+1:2d}/26] {label}: {cls_count} extracted, "
                  f"{cls_skipped} skipped  ({rate:.1f} img/s)")

    # Save CSV
    df = pd.DataFrame(rows, columns=ALL_COLS)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    elapsed = time.time() - t_start
    logger.info(f"\n  Total: {total_images} rows, {skipped} skipped")
    logger.info(f"  Time:  {elapsed:.1f}s ({total_images/elapsed:.1f} img/s)")
    logger.info(f"  Saved: {output_csv} ({df.shape})")
    logger.info(f"  Labels: {df['label'].value_counts().to_dict()}")
    return output_csv


# ---------------------------------------------------------------------------
# Phase 2: Kinematic engineering (reuses existing pipeline)
# ---------------------------------------------------------------------------
def run_kinematic(raw_csv, kin_csv, ang_csv):
    """Run Phase 2 kinematic engineering on the extracted landmarks."""
    logger.info(f"\n{'='*60}")
    logger.info("  Phase 2 — Kinematic Feature Engineering (Images)")
    logger.info(f"{'='*60}")

    from kinematic_engineer import run_pipeline

    # Temporarily override output paths
    import kinematic_engineer as ke
    orig_full = ke.OUT_FULL_PATH
    orig_ang  = ke.OUT_ANGLES_PATH
    ke.OUT_FULL_PATH   = kin_csv
    ke.OUT_ANGLES_PATH = ang_csv

    run_pipeline(raw_csv)

    # Restore
    ke.OUT_FULL_PATH   = orig_full
    ke.OUT_ANGLES_PATH = orig_ang


# ---------------------------------------------------------------------------
# Phase 3: Training (reuses existing training script)
# ---------------------------------------------------------------------------
def run_training():
    """Run Phase 3 training on the image-derived datasets."""
    logger.info(f"\n{'='*60}")
    logger.info("  Phase 3 — Training on Image Dataset")
    logger.info(f"{'='*60}")

    import train_classifier as tc

    # Override ablation runs to use image CSVs
    orig_runs = tc.ABLATION_RUNS
    tc.ABLATION_RUNS = [
        (
            "img_raw",
            RAW_CSV,
            "label",
            ["label", "source", "user_id"],
        ),
        (
            "img_kinematic",
            KIN_CSV,
            "label",
            ["label", "source", "user_id"],
        ),
        (
            "img_angles_only",
            ANG_CSV,
            "label",
            ["label", "source", "user_id"],
        ),
    ]

    tc.main()
    tc.ABLATION_RUNS = orig_runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Image Dataset Pipeline — Full Phase 1+2+3"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific phase only (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit images per class for Phase 1 (default: all)",
    )
    args = parser.parse_args()

    run_all = args.phase is None

    if run_all or args.phase == 1:
        extract_from_images(DATASET_DIR, RAW_CSV, limit_per_class=args.limit)

    if run_all or args.phase == 2:
        if not os.path.exists(RAW_CSV):
            logger.info(f"  ERROR: {RAW_CSV} not found. Run Phase 1 first.")
            sys.exit(1)
        run_kinematic(RAW_CSV, KIN_CSV, ANG_CSV)

    if run_all or args.phase == 3:
        if not os.path.exists(KIN_CSV):
            logger.info(f"  ERROR: {KIN_CSV} not found. Run Phase 1+2 first.")
            sys.exit(1)
        run_training()

    logger.info(f"\n{'='*60}")
    logger.info("  Image Pipeline Complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
