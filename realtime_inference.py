# -*- coding: utf-8 -*-
"""
Phase 4 — Real-Time ISL Inference
====================================
Streams webcam → extracts landmarks → applies Phase 2 kinematic math →
predicts via trained MLP → displays with stability logic, latency, and FPS.

Usage:
    python realtime_inference.py                         # default (kinematic model)
    python realtime_inference.py --model raw             # use raw model
    python realtime_inference.py --camera 1              # webcam index
    python realtime_inference.py --confidence 0.90       # confidence threshold
    python realtime_inference.py --buffer 7              # rolling buffer size

Controls:
    ESC  — quit
"""

import argparse
import os
import sys
import time
import warnings
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp

# Suppress TF info logs before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
from tensorflow import keras

# Import kinematic math from Phase 2
from kinematic_engineer import (
    center_and_normalize,
    compute_joint_angles,
    compute_spread_angles,
    _landmarks_to_array,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
RESULTS_DIR = MODELS_DIR   # backward compat for imports
HAND_LANDMARK_COUNT = 21
AXES = ["x", "y", "z"]
POSE_INDICES = [11, 12, 13, 14, 15, 16]
CONFIDENCE_THRESHOLD_MP = 0.5   # MediaPipe detection confidence

# Model configurations: (run_name, expected_features, feature_type, dataset)
MODEL_CONFIGS = {
    # --- Kaggle CSV-trained models ---
    "source_kaggle": {"features": 126, "type": "kaggle_raw",    "dataset": "Kaggle CSV"},
    "raw":           {"features": 144, "type": "raw_landmarks", "dataset": "Kaggle CSV"},
    "kinematic":     {"features": 182, "type": "kinematic",     "dataset": "Kaggle CSV"},
    "angles_only":   {"features": 38,  "type": "angles_only",   "dataset": "Kaggle CSV"},
    # --- Image dataset-trained models ---
    "img_raw":          {"features": 144, "type": "raw_landmarks", "dataset": "Gesture Speech Images"},
    "img_kinematic":    {"features": 182, "type": "kinematic",     "dataset": "Gesture Speech Images"},
    "img_angles_only":  {"features": 38,  "type": "angles_only",   "dataset": "Gesture Speech Images"},
}

# Ordered list for key-switching (1-7)
MODEL_LIST = list(MODEL_CONFIGS.keys())

# Class labels (0-25 → A-Z)
CLASS_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# HUD colours (BGR)
COL_GREEN  = (0, 220, 0)
COL_WHITE  = (255, 255, 255)
COL_YELLOW = (0, 230, 255)
COL_RED    = (0, 0, 255)
COL_CYAN   = (255, 220, 0)
COL_BG     = (30, 30, 30)

# MediaPipe helpers
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Landmark extraction (mirrors isl_detection.py extract_landmarks)
# ---------------------------------------------------------------------------
def extract_landmarks_live(results):
    """
    Extract hand + upper-body landmarks from MediaPipe Holistic results.
    Returns flat numpy array of 144 floats (LH:63 + RH:63 + Pose:18),
    or None if no hands detected.
    """
    # ----- Left hand (21 × 3 = 63) -----
    lh = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float64)
    lh_ok = False
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i * 3]     = lm.x
            lh[i * 3 + 1] = lm.y
            lh[i * 3 + 2] = lm.z
        lh_ok = True

    # ----- Right hand (21 × 3 = 63) -----
    rh = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float64)
    rh_ok = False
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i * 3]     = lm.x
            rh[i * 3 + 1] = lm.y
            rh[i * 3 + 2] = lm.z
        rh_ok = True

    if not lh_ok and not rh_ok:
        return None

    # ----- Upper-body pose (6 × 3 = 18) -----
    pose = np.zeros(len(POSE_INDICES) * 3, dtype=np.float64)
    if results.pose_landmarks:
        for j, idx in enumerate(POSE_INDICES):
            lm = results.pose_landmarks.landmark[idx]
            vis = lm.visibility if lm.visibility else 0.0
            if vis >= CONFIDENCE_THRESHOLD_MP:
                pose[j * 3]     = lm.x
                pose[j * 3 + 1] = lm.y
                pose[j * 3 + 2] = lm.z

    return lh, rh, pose


# ---------------------------------------------------------------------------
# Feature builders per model type
# ---------------------------------------------------------------------------
def build_kinematic_features(lh, rh, pose):
    """
    Full Phase 2 pipeline: center+normalize → angles → spreads.
    Returns 182-dim feature vector.
    """
    # Center and normalize
    cn_lh, cn_rh, cn_pose = center_and_normalize(lh, rh, pose)

    # Joint angles (from RAW landmarks — angles are invariant)
    lh_21x3 = _landmarks_to_array(lh)
    rh_21x3 = _landmarks_to_array(rh)
    lh_angles  = compute_joint_angles(lh_21x3)
    rh_angles  = compute_joint_angles(rh_21x3)
    lh_spreads = compute_spread_angles(lh_21x3)
    rh_spreads = compute_spread_angles(rh_21x3)

    # Concatenate: cn_lh(63) + cn_rh(63) + cn_pose(18) + angles(30) + spreads(8) = 182
    features = np.concatenate([
        cn_lh, cn_rh, cn_pose,
        lh_angles, rh_angles,
        lh_spreads, rh_spreads,
    ])
    return features


def build_raw_features(lh, rh, pose):
    """Raw 144-dim landmark vector (LH + RH + Pose)."""
    return np.concatenate([lh, rh, pose])


def build_angles_only_features(lh, rh, _pose):
    """Angles + spreads only = 38-dim vector."""
    lh_21x3 = _landmarks_to_array(lh)
    rh_21x3 = _landmarks_to_array(rh)
    lh_angles  = compute_joint_angles(lh_21x3)
    rh_angles  = compute_joint_angles(rh_21x3)
    lh_spreads = compute_spread_angles(lh_21x3)
    rh_spreads = compute_spread_angles(rh_21x3)
    return np.concatenate([lh_angles, rh_angles, lh_spreads, rh_spreads])


def build_kaggle_features(lh, rh, _pose):
    """Source Kaggle format: 126 raw hand coords (no pose)."""
    return np.concatenate([lh, rh])


FEATURE_BUILDERS = {
    "kinematic":       build_kinematic_features,
    "raw":             build_raw_features,
    "angles_only":     build_angles_only_features,
    "source_kaggle":   build_kaggle_features,
    "img_raw":         build_raw_features,
    "img_kinematic":   build_kinematic_features,
    "img_angles_only": build_angles_only_features,
}


# ---------------------------------------------------------------------------
# HUD drawing helpers
# ---------------------------------------------------------------------------
def draw_hud(frame, prediction, confidence, fps, latency_ms,
             model_name, n_features, is_stable, buffer_counts):
    """Draw a professional heads-up display overlay."""
    h, w = frame.shape[:2]

    # ── Semi-transparent background bars ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), COL_BG, -1)            # top bar
    cv2.rectangle(overlay, (0, h - 40), (w, h), COL_BG, -1)        # bottom bar
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── Prediction (large, top-left) ──
    if prediction and is_stable:
        color = COL_GREEN if confidence >= 0.95 else COL_YELLOW
        cv2.putText(frame, prediction, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
        # Confidence bar
        conf_text = f"{confidence * 100:.1f}%"
        cv2.putText(frame, conf_text, (120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 100, 100), 4, cv2.LINE_AA)

    # ── FPS + Latency (top-right) ──
    fps_text = f"FPS: {fps:.0f}"
    lat_text = f"Latency: {latency_ms:.0f}ms"
    cv2.putText(frame, fps_text, (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, lat_text, (w - 220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_CYAN, 1, cv2.LINE_AA)

    # ── Bottom status bar ──
    stability_icon = "STABLE" if is_stable else "..."
    dataset = MODEL_CONFIGS.get(model_name, {}).get("dataset", "")
    status = f"Model: {model_name} | {n_features}f | {dataset} | {stability_icon}"
    cv2.putText(frame, status, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1, cv2.LINE_AA)

    # ── Model switch hint ──
    cv2.putText(frame, "Keys 1-7: switch model", (w - 200, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


def draw_landmarks(frame, results):
    """Draw hand landmarks on the frame (hands only — no face/body dots)."""
    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_inference(model_name: str, camera_idx: int,
                  confidence_threshold: float, buffer_size: int):
    """
    Main real-time inference loop.
    """
    print("=" * 60)
    print("  Phase 4 — Real-Time ISL Inference")
    print("=" * 60)

    # ── Load model and scaler ──
    model_path  = os.path.join(MODELS_DIR, f"isl_{model_name}_mlp.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{model_name}.pkl")

    if not os.path.exists(model_path):
        print(f"  ❌  Model not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"  ❌  Scaler not found: {scaler_path}")
        print(f"      Re-run train_classifier.py to generate scaler files.")
        sys.exit(1)

    print(f"  Model:      {model_path}")
    model = keras.models.load_model(model_path)

    print(f"  Scaler:     {scaler_path}")
    scaler = joblib.load(scaler_path)

    config = MODEL_CONFIGS[model_name]
    n_features = config["features"]
    feature_builder = FEATURE_BUILDERS[model_name]

    print(f"  Features:   {n_features} ({config['type']})")
    print(f"  Confidence: {confidence_threshold:.0%}")
    print(f"  Buffer:     {buffer_size} frames")
    print(f"  Camera:     {camera_idx}")
    print()

    # ── Rolling prediction buffer ──
    pred_buffer = deque(maxlen=buffer_size)
    conf_buffer = deque(maxlen=buffer_size)

    # ── FPS tracking ──
    fps_times = deque(maxlen=30)
    frame_count = 0

    # ── Open webcam ──
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"  ❌  Cannot open webcam (index={camera_idx})")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("  ✅  Webcam opened. Press ESC to quit.\n")

    # Create resizable window
    cv2.namedWindow("ISL Real-Time Inference - Phase 4", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Real-Time Inference - Phase 4", 640, 480)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            t_frame = time.perf_counter()

            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # ── Extract landmarks ──
            landmark_data = extract_landmarks_live(results)

            prediction = None
            confidence = 0.0
            both_hands = False

            if landmark_data is not None:
                lh, rh, pose = landmark_data

                # Check at least one hand is visible
                lh_visible = np.any(np.abs(lh) > 1e-9)
                rh_visible = np.any(np.abs(rh) > 1e-9)
                any_hand = lh_visible or rh_visible

                if not any_hand:
                    # Still update buffer with None to clear stale predictions
                    pred_buffer.append(None)
                    conf_buffer.append(0.0)
                else:
                    # ── Build feature vector ──
                    try:
                        features = feature_builder(lh, rh, pose)
                        features = np.array(features, dtype=np.float32).reshape(1, -1)

                        # ── Scale features ──
                        features_scaled = scaler.transform(features)

                        # ── Predict ──
                        proba = model.predict(features_scaled, verbose=0)[0]
                        pred_idx = np.argmax(proba)
                        confidence = float(proba[pred_idx])

                        if pred_idx < len(CLASS_LABELS):
                            prediction = CLASS_LABELS[pred_idx]

                        # ── Update rolling buffer ──
                        pred_buffer.append(prediction)
                        conf_buffer.append(confidence)

                    except Exception:
                        pred_buffer.append(None)
                        conf_buffer.append(0.0)
            else:
                # No hands detected — clear buffer gradually
                pred_buffer.append(None)
                conf_buffer.append(0.0)

            # ── Stability logic ──
            is_stable = False
            stable_prediction = None
            stable_confidence = 0.0

            if len(pred_buffer) >= 3:
                # Count non-None predictions
                valid_preds = [p for p in pred_buffer if p is not None]
                if valid_preds:
                    counter = Counter(valid_preds)
                    most_common, most_count = counter.most_common(1)[0]
                    # Majority vote: appears in ≥60% of buffer
                    majority_threshold = max(3, len(pred_buffer) * 0.6)
                    if most_count >= majority_threshold:
                        # Average confidence for the majority class
                        majority_confs = [
                            c for p, c in zip(pred_buffer, conf_buffer)
                            if p == most_common
                        ]
                        avg_conf = np.mean(majority_confs)
                        if avg_conf >= confidence_threshold:
                            is_stable = True
                            stable_prediction = most_common
                            stable_confidence = avg_conf

            # ── FPS computation ──
            t_now = time.perf_counter()
            fps_times.append(t_now)
            latency_ms = (t_now - t_frame) * 1000
            if len(fps_times) > 1:
                fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
            else:
                fps = 0.0

            # ── Draw ──
            draw_landmarks(frame, results)
            buffer_counts = Counter(p for p in pred_buffer if p is not None)
            draw_hud(
                frame,
                stable_prediction,
                stable_confidence,
                fps,
                latency_ms,
                model_name,
                n_features,
                is_stable,
                buffer_counts,
            )

            # ── Hand visibility warning ──
            if landmark_data is None or not any_hand:
                h, w = frame.shape[:2]
                msg = "Show your hand!" if landmark_data is not None else "No hands detected"
                cv2.putText(frame, msg, (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_RED, 2, cv2.LINE_AA)

            cv2.imshow("ISL Real-Time Inference - Phase 4", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # ── Live model switching (keys 1-7) ──
            if ord('1') <= key <= ord('7'):
                new_idx = key - ord('1')
                if new_idx < len(MODEL_LIST):
                    new_name = MODEL_LIST[new_idx]
                    if new_name != model_name:
                        new_model_path  = os.path.join(MODELS_DIR, f"isl_{new_name}_mlp.h5")
                        new_scaler_path = os.path.join(MODELS_DIR, f"scaler_{new_name}.pkl")
                        if os.path.exists(new_model_path) and os.path.exists(new_scaler_path):
                            model = keras.models.load_model(new_model_path)
                            scaler = joblib.load(new_scaler_path)
                            model_name = new_name
                            config = MODEL_CONFIGS[model_name]
                            n_features = config["features"]
                            feature_builder = FEATURE_BUILDERS[model_name]
                            pred_buffer.clear()
                            conf_buffer.clear()
                            print(f"  >> Switched to: {model_name} ({config['dataset']})")
                        else:
                            print(f"  >> Model files not found for: {new_name}")

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Session ended. {frame_count} frames processed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — Real-Time ISL Inference"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="kinematic",
        help="Which trained model to use (default: kinematic)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.85,
        help="Minimum confidence threshold for stable prediction (default: 0.85)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=5,
        help="Rolling buffer size for stability logic (default: 5)",
    )
    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        camera_idx=args.camera,
        confidence_threshold=args.confidence,
        buffer_size=args.buffer,
    )


if __name__ == "__main__":
    main()
