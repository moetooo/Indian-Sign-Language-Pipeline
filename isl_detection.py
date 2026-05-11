"""
Phase 1 — ISL Data Capture Pipeline
=====================================
Uses MediaPipe Holistic (face disabled) to extract:
  • 21 left-hand landmarks  (x, y, z)
  • 21 right-hand landmarks (x, y, z)
  • 6 upper-body pose points (shoulders, elbows, wrists) (x, y, z)

Modes:
  --mode capture        Live webcam capture → isl_raw_data.csv
  --mode import_kaggle  Import Kaggle CSV datasets → isl_raw_data.csv
  --mode detect         Real-time inference using saved model

Usage:
  python isl_detection.py --mode capture --user user_01
  python isl_detection.py --mode import_kaggle [--dry-run]
  python isl_detection.py --mode detect
"""

import argparse
import csv
import copy
import itertools
import os
import string
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_CSV_PATH = os.path.join("data", "raw", "isl_raw_data.csv")
KAGGLE_CSV_1 = os.path.join("dataset", "Indian Sign Language Gesture Landmarks.csv")

CONFIDENCE_THRESHOLD = 0.7

# Sign classes
SIGN_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(1, 10)] + ["Neutral"]

# Upper-body pose landmark indices we care about
POSE_INDICES = [11, 12, 13, 14, 15, 16]  # L/R shoulder, elbow, wrist
POSE_NAMES   = ["shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r"]

# Column definitions
HAND_LANDMARK_COUNT = 21
AXES = ["x", "y", "z"]

def _hand_cols(prefix: str) -> list[str]:
    """Generate column names like lh_x0, lh_y0, lh_z0, ..., lh_z20."""
    cols = []
    for i in range(HAND_LANDMARK_COUNT):
        for ax in AXES:
            cols.append(f"{prefix}_{ax}{i}")
    return cols

def _pose_cols() -> list[str]:
    """Generate column names like pose_x11, pose_y11, pose_z11, ..., pose_z16."""
    cols = []
    for idx in POSE_INDICES:
        for ax in AXES:
            cols.append(f"pose_{ax}{idx}")
    return cols

META_COLS    = ["label", "source", "user_id"]
LH_COLS      = _hand_cols("lh")       # 63 columns
RH_COLS      = _hand_cols("rh")       # 63 columns
POSE_COLS    = _pose_cols()            # 18 columns
FEATURE_COLS = LH_COLS + RH_COLS + POSE_COLS  # 144 columns
ALL_COLS     = META_COLS + FEATURE_COLS        # 147 columns

# ---------------------------------------------------------------------------
# MediaPipe Holistic helper
# ---------------------------------------------------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_landmarks(results, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    Extract hand + upper-body landmarks from Holistic results.

    Returns:
        (left_hand, right_hand, pose_upper) — each a flat list of floats,
        or None if confidence is too low on BOTH hands.
    """
    # ----- Left hand (21 × 3 = 63) -----
    lh = [0.0] * (HAND_LANDMARK_COUNT * 3)
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark
        visibilities = [l.visibility if hasattr(l, "visibility") and l.visibility else
                        (l.presence if hasattr(l, "presence") and l.presence else 1.0)
                        for l in lm]
        avg_conf = np.mean(visibilities)
        if avg_conf >= confidence_threshold:
            for i, l in enumerate(lm):
                lh[i * 3]     = l.x
                lh[i * 3 + 1] = l.y
                lh[i * 3 + 2] = l.z
        lh_ok = avg_conf >= confidence_threshold
    else:
        lh_ok = False

    # ----- Right hand (21 × 3 = 63) -----
    rh = [0.0] * (HAND_LANDMARK_COUNT * 3)
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark
        visibilities = [l.visibility if hasattr(l, "visibility") and l.visibility else
                        (l.presence if hasattr(l, "presence") and l.presence else 1.0)
                        for l in lm]
        avg_conf = np.mean(visibilities)
        if avg_conf >= confidence_threshold:
            for i, l in enumerate(lm):
                rh[i * 3]     = l.x
                rh[i * 3 + 1] = l.y
                rh[i * 3 + 2] = l.z
        rh_ok = avg_conf >= confidence_threshold
    else:
        rh_ok = False

    # Skip frame if both hands failed confidence
    if not lh_ok and not rh_ok:
        return None

    # ----- Upper-body pose (6 × 3 = 18) -----
    pose = [0.0] * (len(POSE_INDICES) * 3)
    if results.pose_landmarks:
        for j, idx in enumerate(POSE_INDICES):
            lm = results.pose_landmarks.landmark[idx]
            vis = lm.visibility if lm.visibility else 0.0
            if vis >= confidence_threshold:
                pose[j * 3]     = lm.x
                pose[j * 3 + 1] = lm.y
                pose[j * 3 + 2] = lm.z

    return lh + rh + pose


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def _ensure_csv_header(path: str):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ALL_COLS)


def append_row(path: str, label: str, source: str, user_id: str, features: list[float]):
    """Append one row to the CSV."""
    _ensure_csv_header(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([label, source, user_id] + features)


# ---------------------------------------------------------------------------
# Mode: import_kaggle
# ---------------------------------------------------------------------------
def import_kaggle(dry_run: bool = False):
    """
    Import the external Kaggle datasets and the legacy keypoint.csv
    into the unified isl_raw_data.csv.
    """
    total_imported = 0

    # ── Dataset 1: Indian Sign Language Gesture Landmarks.csv ──────────
    if os.path.exists(KAGGLE_CSV_1):
        print(f"\n📂  Loading Kaggle dataset: {KAGGLE_CSV_1}")
        df = pd.read_csv(KAGGLE_CSV_1)
        print(f"    Shape: {df.shape}")
        print(f"    Labels: {sorted(df['target'].unique().tolist())}")

        rows = []
        for _, row in df.iterrows():
            label = str(row["target"]).upper().strip()

            # Build left hand features (x, y, z for 21 landmarks)
            lh = []
            for i in range(HAND_LANDMARK_COUNT):
                for ax in AXES:
                    col_name = f"left_hand_{ax}_{i}"
                    lh.append(float(row.get(col_name, 0.0)))

            # Build right hand features
            rh = []
            for i in range(HAND_LANDMARK_COUNT):
                for ax in AXES:
                    col_name = f"right_hand_{ax}_{i}"
                    rh.append(float(row.get(col_name, 0.0)))

            # Upper body → zeros (not available in this dataset)
            pose = [0.0] * (len(POSE_INDICES) * 3)

            features = lh + rh + pose
            rows.append([label, "kaggle_hand_landmarks", "kaggle"] + features)

        if dry_run:
            print(f"    ✅ Would import {len(rows)} rows")
            label_counts = {}
            for r in rows:
                label_counts[r[0]] = label_counts.get(r[0], 0) + 1
            for lbl in sorted(label_counts.keys()):
                print(f"       {lbl}: {label_counts[lbl]}")
        else:
            _ensure_csv_header(RAW_CSV_PATH)
            with open(RAW_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"    ✅ Imported {len(rows)} rows → {RAW_CSV_PATH}")
        total_imported += len(rows)
    else:
        print(f"⚠️  Kaggle dataset not found: {KAGGLE_CSV_1}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'DRY RUN — ' if dry_run else ''}Total: {total_imported} rows")
    if not dry_run and os.path.exists(RAW_CSV_PATH):
        df = pd.read_csv(RAW_CSV_PATH)
        print(f"CSV shape: {df.shape}")
        print(f"Columns ({len(df.columns)}): {df.columns.tolist()[:6]} ... {df.columns.tolist()[-3:]}")
        print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
        print(f"\nSource distribution:\n{df['source'].value_counts().to_string()}")


# ---------------------------------------------------------------------------
# Mode: capture (webcam)
# ---------------------------------------------------------------------------
def capture_webcam(user_id: str):
    """
    Live webcam capture with MediaPipe Holistic.
    Keys:
      a-z   → set label to that letter (uppercase)
      0-9   → set label to that digit
      n     → set label to 'Neutral'
      SPACE → toggle recording on/off
      ESC   → quit
    """
    print("\n🎥  Webcam Capture Mode")
    print(f"    User: {user_id}")
    print(f"    Output: {RAW_CSV_PATH}")
    print("    Controls:")
    print("      a-z / 0-9 → select label")
    print("      n          → Neutral class")
    print("      SPACE      → start/stop recording")
    print("      ESC        → quit and save\n")

    _ensure_csv_header(RAW_CSV_PATH)

    current_label = None
    recording = False
    sample_counts: dict[str, int] = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam!")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Draw landmarks on display frame
            display = frame.copy()

            # Draw pose (upper body only — we skip face)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # Draw hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            # Extract features
            features = extract_landmarks(results)

            # ── HUD overlay ──
            status_color = (0, 255, 0) if recording else (0, 0, 255)
            status_text = "● REC" if recording else "■ PAUSED"
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            label_text = f"Label: {current_label or '---'}"
            cv2.putText(display, label_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            count = sample_counts.get(current_label, 0) if current_label else 0
            cv2.putText(display, f"Samples: {count}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            conf_text = "Hands: OK" if features is not None else "Hands: LOW CONF"
            conf_color = (0, 255, 0) if features is not None else (0, 0, 255)
            cv2.putText(display, conf_text, (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

            # Show user_id
            cv2.putText(display, f"User: {user_id}", (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # ── Record if active ──
            if recording and current_label and features is not None:
                append_row(RAW_CSV_PATH, current_label, "webcam", user_id, features)
                sample_counts[current_label] = sample_counts.get(current_label, 0) + 1

            cv2.imshow("ISL Capture — Phase 1", display)

            # ── Key handling ──
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                recording = not recording
                state = "RECORDING" if recording else "PAUSED"
                print(f"  ⏺ {state} | Label: {current_label} | Samples: {sample_counts.get(current_label, 0)}")
            elif key == ord("n") or key == ord("N"):
                current_label = "Neutral"
                print(f"  🏷️  Label → Neutral")
            elif ord("a") <= key <= ord("z"):
                current_label = chr(key).upper()
                print(f"  🏷️  Label → {current_label}")
            elif ord("0") <= key <= ord("9"):
                current_label = chr(key)
                print(f"  🏷️  Label → {current_label}")

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    print("\n📊  Capture Summary:")
    for lbl in sorted(sample_counts.keys()):
        print(f"    {lbl}: {sample_counts[lbl]} samples")
    total = sum(sample_counts.values())
    print(f"    TOTAL: {total} samples saved to {RAW_CSV_PATH}")


# ---------------------------------------------------------------------------
# Mode: detect (real-time inference — backward compat)
# ---------------------------------------------------------------------------
def detect_realtime():
    """
    Real-time inference using the saved model.
    NOTE: The existing model.h5 uses the old 42-feature format.
    This mode preserves backward compat by extracting only left-hand x/y.
    Once the model is retrained in Phase 2 on the new 144-feature vector,
    this will be updated.
    """
    from tensorflow import keras

    model_path = "model.h5"
    if not os.path.exists(model_path):
        print(f"❌  Model not found: {model_path}")
        return

    model = keras.models.load_model(model_path)

    alphabet  = [str(i) for i in range(1, 10)]
    alphabet += list(string.ascii_uppercase)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam!")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            display = frame.copy()

            # For backward compat with old model, extract left hand x,y only
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # Prefer left hand; fall back to right
                hand_lm = results.left_hand_landmarks or results.right_hand_landmarks

                # Calculate pixel-based landmarks (old method)
                image_w, image_h = frame.shape[1], frame.shape[0]
                landmark_list = []
                for lm in hand_lm.landmark:
                    lx = min(int(lm.x * image_w), image_w - 1)
                    ly = min(int(lm.y * image_h), image_h - 1)
                    landmark_list.append([lx, ly])

                # Pre-process (relative coords + normalize)
                temp = copy.deepcopy(landmark_list)
                base_x, base_y = temp[0]
                for i, pt in enumerate(temp):
                    temp[i] = [pt[0] - base_x, pt[1] - base_y]
                flat = list(itertools.chain.from_iterable(temp))
                max_val = max(map(abs, flat)) if max(map(abs, flat)) != 0 else 1
                flat = [v / max_val for v in flat]

                df_input = pd.DataFrame([flat])
                predictions = model.predict(df_input, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]

                if predicted_class < len(alphabet):
                    label = alphabet[predicted_class]
                    cv2.putText(display, label, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    display,
                    hand_lm,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            cv2.imshow("ISL Detector — Phase 1", display)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ISL Data Capture & Inference Pipeline (Holistic)"
    )
    parser.add_argument(
        "--mode",
        choices=["capture", "import_kaggle", "detect", "detect_v2"],
        required=True,
        help=(
            "capture: webcam recording | import_kaggle: ingest CSV datasets | "
            "detect: legacy real-time inference | detect_v2: Phase 4 inference (recommended)"
        ),
    )
    parser.add_argument("--user", default="user_01", help="User ID for webcam capture (default: user_01)")
    parser.add_argument("--dry-run", action="store_true", help="Preview import without writing CSV")
    # Phase 4 detect_v2 options
    parser.add_argument("--model", default="kinematic",
                        choices=["source_kaggle", "raw", "kinematic", "angles_only"],
                        help="Model for detect_v2 (default: kinematic)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.85,
                        help="Confidence threshold for detect_v2 (default: 0.85)")
    parser.add_argument("--buffer", type=int, default=5,
                        help="Rolling buffer size for detect_v2 (default: 5)")

    args = parser.parse_args()

    if args.mode == "capture":
        capture_webcam(args.user)
    elif args.mode == "import_kaggle":
        import_kaggle(dry_run=args.dry_run)
    elif args.mode == "detect":
        detect_realtime()
    elif args.mode == "detect_v2":
        from realtime_inference import run_inference
        run_inference(
            model_name=args.model,
            camera_idx=args.camera,
            confidence_threshold=args.confidence,
            buffer_size=args.buffer,
        )


if __name__ == "__main__":
    main()
