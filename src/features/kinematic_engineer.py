# -*- coding: utf-8 -*-
"""
Phase 2 — Kinematic Feature Engineering
========================================
Converts raw landmark data (isl_raw_data.csv) into an invariant geometric
representation:

1. Translation invariance  — center on mid-shoulder (webcam) or wrist (Kaggle)
2. Scale invariance        — normalize to [-1, 1]
3. Joint angles            — 15 bone-joint angles per hand (30 total)
4. Inter-finger spread     — 4 spread angles per hand (8 total)

Outputs:
  isl_kinematic_data.csv  — full dataset (185 columns)
  isl_angles_only.csv     — angles only  (41 columns)

Usage:
  python kinematic_engineer.py                     # full run
  python kinematic_engineer.py --dry-run           # preview only
  python kinematic_engineer.py --input custom.csv  # custom input
  python kinematic_engineer.py --skip-angles-only  # skip 2nd output
"""

from src.utils.logger import logger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.math_utils import angle_between, is_hand_present, is_pose_present
from src.utils.common import _hand_cols, _pose_cols, HAND_LANDMARK_COUNT, POSE_INDICES, AXES, CLASS_LABELS



import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — must match isl_detection.py schema
# ---------------------------------------------------------------------------
RAW_CSV_PATH     = os.path.join("data", "raw", "isl_raw_data.csv")
OUT_FULL_PATH    = os.path.join("data", "kinematic", "isl_kinematic_data.csv")
OUT_ANGLES_PATH  = os.path.join("data", "angles", "isl_angles_only.csv")


META_COLS = ["label", "source", "user_id"]

# Input column names (from Phase 1)


LH_COLS  = _hand_cols("lh")    # 63
RH_COLS  = _hand_cols("rh")    # 63
POSE_COLS = _pose_cols()        # 18

# Output column names — centered + normalized coordinates
CN_LH_COLS   = [f"cn_{c}" for c in LH_COLS]     # 63
CN_RH_COLS   = [f"cn_{c}" for c in RH_COLS]     # 63
CN_POSE_COLS = [f"cn_{c}" for c in POSE_COLS]   # 18

# Angle column names
LH_ANGLE_COLS  = [f"lh_angle_{i}" for i in range(15)]    # 15
RH_ANGLE_COLS  = [f"rh_angle_{i}" for i in range(15)]    # 15
LH_SPREAD_COLS = [f"lh_spread_{i}" for i in range(4)]    # 4
RH_SPREAD_COLS = [f"rh_spread_{i}" for i in range(4)]    # 4

ALL_ANGLE_COLS = LH_ANGLE_COLS + RH_ANGLE_COLS + LH_SPREAD_COLS + RH_SPREAD_COLS  # 38

OUT_FULL_COLS   = META_COLS + CN_LH_COLS + CN_RH_COLS + CN_POSE_COLS + ALL_ANGLE_COLS  # 185
OUT_ANGLES_COLS = META_COLS + ALL_ANGLE_COLS  # 41

# ---------------------------------------------------------------------------
# Finger bone chains for joint angle computation
# Each finger has 3 interior joints → 3 angles
# Triplet (a, b, c) means: angle at landmark b, between bones a→b and b→c
# ---------------------------------------------------------------------------
FINGER_JOINT_TRIPLETS = [
    # Thumb: (0,1,2), (1,2,3), (2,3,4)
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index: (0,5,6), (5,6,7), (6,7,8)
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle: (0,9,10), (9,10,11), (10,11,12)
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring: (0,13,14), (13,14,15), (14,15,16)
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky: (0,17,18), (17,18,19), (18,19,20)
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

# Inter-finger spread: angle between adjacent fingertip vectors from MCP
# (base, tip_a) vs (base, tip_b)
SPREAD_PAIRS = [
    # Thumb-Index:  vec(1→4) vs vec(5→8)
    (1, 4, 5, 8),
    # Index-Middle: vec(5→8) vs vec(9→12)
    (5, 8, 9, 12),
    # Middle-Ring:  vec(9→12) vs vec(13→16)
    (9, 12, 13, 16),
    # Ring-Pinky:   vec(13→16) vs vec(17→20)
    (13, 16, 17, 20),
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _landmarks_to_array(flat_values: np.ndarray) -> np.ndarray:
    """
    Reshape a flat array of 63 values (21 landmarks × 3) into (21, 3).
    """
    return flat_values.reshape(HAND_LANDMARK_COUNT, 3)




# ---------------------------------------------------------------------------
# Core transformations
# ---------------------------------------------------------------------------
def compute_joint_angles(landmarks_21x3: np.ndarray) -> list[float]:
    """
    Compute 15 joint angles for one hand from its 21 landmarks (shape 21×3).
    Returns list of 15 floats in radians.
    """
    if not is_hand_present(landmarks_21x3):
        return [0.0] * 15

    angles = []
    for a, b, c in FINGER_JOINT_TRIPLETS:
        v1 = landmarks_21x3[a] - landmarks_21x3[b]  # bone b→a
        v2 = landmarks_21x3[c] - landmarks_21x3[b]  # bone b→c
        angles.append(angle_between(v1, v2))

    return angles


def compute_spread_angles(landmarks_21x3: np.ndarray) -> list[float]:
    """
    Compute 4 inter-finger spread angles for one hand.
    Returns list of 4 floats in radians.
    """
    if not is_hand_present(landmarks_21x3):
        return [0.0] * 4

    spreads = []
    for base_a, tip_a, base_b, tip_b in SPREAD_PAIRS:
        va = landmarks_21x3[tip_a] - landmarks_21x3[base_a]
        vb = landmarks_21x3[tip_b] - landmarks_21x3[base_b]
        spreads.append(angle_between(va, vb))

    return spreads


def center_and_normalize(lh_flat: np.ndarray, rh_flat: np.ndarray,
                         pose_flat: np.ndarray):
    """
    Apply translation invariance and scale normalization.

    Translation:
      - If pose data exists: center = midpoint of left shoulder and right shoulder
      - Else: center = wrist(s) — midpoint of both if both present, else the one
        that exists.

    Scale:
      - Divide all coordinates by max absolute value across the row.

    Returns:
        (centered_lh_flat, centered_rh_flat, centered_pose_flat) — each a 1D array
    """
    lh = _landmarks_to_array(lh_flat.copy())      # (21, 3)
    rh = _landmarks_to_array(rh_flat.copy())       # (21, 3)
    pose = pose_flat.copy().reshape(len(POSE_INDICES), 3)  # (6, 3)

    lh_present = is_hand_present(lh)
    rh_present = is_hand_present(rh)
    pose_present = is_pose_present(pose)

    # ── Determine translation center ──
    if pose_present:
        # Mid-shoulder: average of pose landmark 11 (idx 0) and 12 (idx 1)
        center = (pose[0] + pose[1]) / 2.0
    else:
        # Fall back to wrist(s)
        if lh_present and rh_present:
            center = (lh[0] + rh[0]) / 2.0
        elif lh_present:
            center = lh[0].copy()
        elif rh_present:
            center = rh[0].copy()
        else:
            center = np.zeros(3)

    # ── Apply translation ──
    if lh_present:
        lh -= center
    if rh_present:
        rh -= center
    if pose_present:
        pose -= center

    # ── Scale normalization ──
    all_coords = np.concatenate([lh.flatten(), rh.flatten(), pose.flatten()])
    max_val = np.max(np.abs(all_coords))
    if max_val > 1e-9:
        lh /= max_val
        rh /= max_val
        pose /= max_val

    return lh.flatten(), rh.flatten(), pose.flatten()


# ---------------------------------------------------------------------------
# Row-level processing
# ---------------------------------------------------------------------------
def process_row(row: pd.Series) -> dict:
    """
    Process a single row from isl_raw_data.csv.
    Returns a dict with all output columns.
    """
    # Extract raw features
    lh_flat  = row[LH_COLS].values.astype(np.float64)
    rh_flat  = row[RH_COLS].values.astype(np.float64)
    pose_flat = row[POSE_COLS].values.astype(np.float64)

    # 1+2. Center and normalize
    cn_lh, cn_rh, cn_pose = center_and_normalize(lh_flat, rh_flat, pose_flat)

    # 3. Joint angles (use RAW landmarks, angles are invariant to translation/scale)
    lh_21x3 = _landmarks_to_array(lh_flat)
    rh_21x3 = _landmarks_to_array(rh_flat)
    lh_angles = compute_joint_angles(lh_21x3)
    rh_angles = compute_joint_angles(rh_21x3)

    # 4. Spread angles
    lh_spreads = compute_spread_angles(lh_21x3)
    rh_spreads = compute_spread_angles(rh_21x3)

    # Build output dict
    out = {}
    # Metadata
    out["label"]   = row["label"]
    out["source"]  = row["source"]
    out["user_id"] = row["user_id"]

    # Centered + normalized coordinates
    for col, val in zip(CN_LH_COLS, cn_lh):
        out[col] = val
    for col, val in zip(CN_RH_COLS, cn_rh):
        out[col] = val
    for col, val in zip(CN_POSE_COLS, cn_pose):
        out[col] = val

    # Angles
    for col, val in zip(LH_ANGLE_COLS, lh_angles):
        out[col] = val
    for col, val in zip(RH_ANGLE_COLS, rh_angles):
        out[col] = val
    for col, val in zip(LH_SPREAD_COLS, lh_spreads):
        out[col] = val
    for col, val in zip(RH_SPREAD_COLS, rh_spreads):
        out[col] = val

    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(input_path: str, dry_run: bool = False,
                 skip_angles_only: bool = False):
    """
    Read raw CSV, process all rows, write output files.
    """
    t_start = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"  Phase 2 - Kinematic Feature Engineering")
    logger.info(f"{'='*60}")
    logger.info(f"  Input:  {input_path}")
    logger.info(f"  Output: {OUT_FULL_PATH} (185 cols)")
    if not skip_angles_only:
        logger.info(f"          {OUT_ANGLES_PATH} (41 cols)")
    logger.info(f"  Dry run: {dry_run}\n")

    # ── Load input ──
    if not os.path.exists(input_path):
        logger.info(f"  ERROR: Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    n_rows = len(df)
    logger.info(f"  Loaded {n_rows:,} rows x {df.shape[1]} cols")
    logger.info(f"  Labels: {df['label'].nunique()} unique")
    logger.info(f"  Sources: {df['source'].value_counts().to_dict()}")

    # ── Check for pose data ──
    pose_cols_present = all(c in df.columns for c in POSE_COLS)
    if pose_cols_present:
        has_pose = (df[POSE_COLS].abs().sum(axis=1) > 1e-9)
        n_with_pose = has_pose.sum()
        logger.info(f"  Rows with pose data: {n_with_pose:,} / {n_rows:,}")
    else:
        logger.info(f"  WARNING: Pose columns not found in input!")

    if dry_run:
        logger.info(f"\n  DRY RUN - would process {n_rows:,} rows")
        logger.info(f"  Output schema (full):  {len(OUT_FULL_COLS)} columns")
        logger.info(f"  Output schema (angles): {len(OUT_ANGLES_COLS)} columns")
        logger.info(f"  Column samples (full):  {OUT_FULL_COLS[:5]} ... {OUT_FULL_COLS[-5:]}")
        logger.info(f"  Column samples (angles): {OUT_ANGLES_COLS[:5]} ... {OUT_ANGLES_COLS[-5:]}")
        return

    # ── Process rows ──
    logger.info(f"\n  Processing {n_rows:,} rows...")
    results = []
    report_interval = max(1, n_rows // 20)

    for idx, (_, row) in enumerate(df.iterrows()):
        results.append(process_row(row))
        if (idx + 1) % report_interval == 0 or idx == n_rows - 1:
            pct = (idx + 1) / n_rows * 100
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_rows - idx - 1) / rate if rate > 0 else 0
            logger.info(f"    [{pct:5.1f}%] {idx+1:,}/{n_rows:,} rows  "
                  f"({rate:.0f} rows/s, ETA {eta:.0f}s)")

    out_df = pd.DataFrame(results, columns=OUT_FULL_COLS)

    # ── Write full output ──
    out_df.to_csv(OUT_FULL_PATH, index=False, encoding="utf-8")
    logger.info(f"\n  Wrote {OUT_FULL_PATH}: {out_df.shape}")

    # ── Write angles-only output ──
    if not skip_angles_only:
        angles_df = out_df[OUT_ANGLES_COLS]
        angles_df.to_csv(OUT_ANGLES_PATH, index=False, encoding="utf-8")
        logger.info(f"  Wrote {OUT_ANGLES_PATH}: {angles_df.shape}")

    # ── Summary stats ──
    elapsed = time.time() - t_start
    feature_cols = [c for c in out_df.columns if c not in META_COLS]
    logger.info(f"\n  Summary:")
    logger.info(f"    Total time:     {elapsed:.1f}s")
    logger.info(f"    Rows:           {len(out_df):,}")
    logger.info(f"    Feature cols:   {len(feature_cols)}")
    logger.info(f"    NaN count:      {out_df[feature_cols].isna().sum().sum()}")

    # Angle stats
    angle_cols_in_df = [c for c in ALL_ANGLE_COLS if c in out_df.columns]
    angle_vals = out_df[angle_cols_in_df].values
    non_zero_angles = angle_vals[angle_vals > 1e-9]
    if len(non_zero_angles) > 0:
        logger.info(f"    Angle range:    [{non_zero_angles.min():.4f}, {non_zero_angles.max():.4f}] rad")
        logger.info(f"    Angle mean:     {non_zero_angles.mean():.4f} rad ({np.degrees(non_zero_angles.mean()):.1f} deg)")
    else:
        logger.info(f"    Angle range:    all zeros")

    # Coord stats
    coord_cols = CN_LH_COLS + CN_RH_COLS + CN_POSE_COLS
    coord_vals = out_df[coord_cols].values
    logger.info(f"    Coord range:    [{coord_vals.min():.6f}, {coord_vals.max():.6f}]")
    logger.info(f"    Max |coord|:    {np.max(np.abs(coord_vals)):.6f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Phase 2 COMPLETE")
    logger.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 - Kinematic Feature Engineering"
    )
    parser.add_argument("--input", default=RAW_CSV_PATH,
                        help=f"Input CSV path (default: {RAW_CSV_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing output files")
    parser.add_argument("--skip-angles-only", action="store_true",
                        help="Skip generating the angles-only CSV")
    args = parser.parse_args()

    run_pipeline(args.input, dry_run=args.dry_run,
                 skip_angles_only=args.skip_angles_only)


if __name__ == "__main__":
    main()
