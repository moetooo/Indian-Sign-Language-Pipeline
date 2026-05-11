# -*- coding: utf-8 -*-
"""
Fingerspelling Mode — Real-Time Letter-by-Letter Translation
==============================================================
Uses the trained static letter model (angles_only) to detect A-Z
hand signs and build words/sentences via hold-to-confirm logic.

Usage:
    python realtime_fingerspell.py
    python realtime_fingerspell.py --model angles_only
    python realtime_fingerspell.py --hold-frames 12

Controls:
    ESC       — quit
    SPACE     — add space (finish current word)
    BACKSPACE — delete last character
    C         — clear everything
"""

from src.utils.logger import logger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.common import _hand_cols, _pose_cols, HAND_LANDMARK_COUNT, POSE_INDICES, AXES, CLASS_LABELS



import argparse
import os
import sys
import time
import warnings
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
from tensorflow import keras

from src.features.kinematic_engineer import (
    center_and_normalize,
    compute_joint_angles,
    compute_spread_angles,
    _landmarks_to_array,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = "models"

# Model configs
MODEL_CONFIGS = {
    "angles_only":       {"features": 38,  "type": "angles_only"},
    "kinematic":         {"features": 182, "type": "kinematic"},
    "raw":               {"features": 144, "type": "raw_landmarks"},
    "img_angles_only":   {"features": 38,  "type": "angles_only"},
    "img_kinematic":     {"features": 182, "type": "kinematic"},
    "img_raw":           {"features": 144, "type": "raw_landmarks"},
}

# HUD colours (BGR)
COL_GREEN  = (0, 220, 0)
COL_WHITE  = (255, 255, 255)
COL_YELLOW = (0, 230, 255)
COL_RED    = (0, 0, 255)
COL_CYAN   = (255, 220, 0)
COL_BG     = (30, 30, 30)
COL_GRAY   = (100, 100, 100)

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ---------------------------------------------------------------------------
# Custom Hand Visualization (Smoothed)
# ---------------------------------------------------------------------------
def draw_hands_smooth(frame, results, smooth_states, alpha=0.35):
    h, w = frame.shape[:2]
    base_r = max(1, int(w * 0.004))
    base_t = max(1, int(w * 0.0015))
    pad = int(w * 0.02)
    font_scale = w / 1300.0

    def draw_single_hand(hand_lms, label, key):
        if not hand_lms:
            smooth_states[key] = None
            return

        raw_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]

        if smooth_states.get(key) is None:
            smooth_states[key] = raw_norm
        
        smoothed_norm = []
        for i, (rx, ry, rz) in enumerate(raw_norm):
            px, py, pz = smooth_states[key][i]
            dist = ((rx - px)**2 + (ry - py)**2)**0.5
            current_alpha = 1.0 if dist > 0.1 else alpha
            
            sx = current_alpha * rx + (1 - current_alpha) * px
            sy = current_alpha * ry + (1 - current_alpha) * py
            sz = current_alpha * rz + (1 - current_alpha) * pz
            smoothed_norm.append((sx, sy, sz))
        
        smooth_states[key] = smoothed_norm

        pixel_pts = []
        min_x, max_x = w, 0
        min_y, max_y = h, 0
        for (sx, sy, _) in smoothed_norm:
            px_x, px_y = int(sx * w), int(sy * h)
            pixel_pts.append((px_x, px_y))
            if px_x < min_x: min_x = px_x
            if px_x > max_x: max_x = px_x
            if px_y < min_y: min_y = px_y
            if px_y > max_y: max_y = px_y

        min_x = max(0, min_x - pad)
        max_x = min(w, max_x + pad)
        min_y = max(0, min_y - pad)
        max_y = min(h, max_y + pad)

        # ── 1. Corner Brackets (behind skeleton) ──
        box_w = max_x - min_x
        box_h = max_y - min_y
        arm_len_x = min(int(0.15 * box_w), 40)
        arm_len_y = min(int(0.15 * box_h), 40)
        thick = 3
        color = (255, 255, 255)

        # Top-Left
        cv2.line(frame, (min_x, min_y), (min_x + arm_len_x, min_y), color, thick)
        cv2.line(frame, (min_x, min_y), (min_x, min_y + arm_len_y), color, thick)
        # Top-Right
        cv2.line(frame, (max_x, min_y), (max_x - arm_len_x, min_y), color, thick)
        cv2.line(frame, (max_x, min_y), (max_x, min_y + arm_len_y), color, thick)
        # Bottom-Left
        cv2.line(frame, (min_x, max_y), (min_x + arm_len_x, max_y), color, thick)
        cv2.line(frame, (min_x, max_y), (min_x, max_y - arm_len_y), color, thick)
        # Bottom-Right
        cv2.line(frame, (max_x, max_y), (max_x - arm_len_x, max_y), color, thick)
        cv2.line(frame, (max_x, max_y), (max_x, max_y - arm_len_y), color, thick)

        # ── 2. Hand Skeleton ──
        for connection in mp_holistic.HAND_CONNECTIONS:
            p1 = pixel_pts[connection[0]]
            p2 = pixel_pts[connection[1]]
            cv2.line(frame, p1, p2, (0, 0, 0), base_t + 2, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (255, 255, 255), base_t, cv2.LINE_AA)

        for p in pixel_pts:
            cv2.circle(frame, p, base_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, p, base_r, (255, 80, 50), -1, cv2.LINE_AA)

        # ── 3. Wrist-Anchored Label (Small & Transparent) ──
        wrist = pixel_pts[0]
        
        side = "r" if label == "Right" else "l"
        text = f"{side}_hand"

        small_font = font_scale * 0.6
        (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, small_font, 1)
        
        # Center horizontally at wrist, place above wrist
        text_x = max(0, wrist[0] - t_w // 2)
        text_y = wrist[1] - 30
        # If clipping top of screen, place below wrist
        if text_y - t_h - 10 < 0:
            text_y = wrist[1] + 40 + t_h

        # Transparent dark grey pill behind text
        rect_top_left = (text_x - 8, text_y - t_h - 6)
        rect_bottom_right = (text_x + t_w + 8, text_y + 6)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, rect_top_left, rect_bottom_right, (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Dimmer white text
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, small_font, (200, 200, 200), 1, cv2.LINE_AA)

    # Note: MediaPipe processes flipped frames, so results are inverted relative to user
    draw_single_hand(results.left_hand_landmarks, "Right", "lh")
    draw_single_hand(results.right_hand_landmarks, "Left", "rh")

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(results, feature_type):
    """Extract landmarks and build feature vector. Returns (features, hands_ok)."""
    # Left hand
    lh = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float64)
    lh_ok = False
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i*3], lh[i*3+1], lh[i*3+2] = lm.x, lm.y, lm.z
        lh_ok = True

    # Right hand
    rh = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float64)
    rh_ok = False
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i*3], rh[i*3+1], rh[i*3+2] = lm.x, lm.y, lm.z
        rh_ok = True

    if not lh_ok and not rh_ok:
        return None

    # Pose
    pose = np.zeros(len(POSE_INDICES) * 3, dtype=np.float64)
    if results.pose_landmarks:
        for j, idx in enumerate(POSE_INDICES):
            lm = results.pose_landmarks.landmark[idx]
            pose[j*3], pose[j*3+1], pose[j*3+2] = lm.x, lm.y, lm.z

    # Build features based on type
    if feature_type == "angles_only":
        lh_a = compute_joint_angles(_landmarks_to_array(lh))
        rh_a = compute_joint_angles(_landmarks_to_array(rh))
        lh_s = compute_spread_angles(_landmarks_to_array(lh))
        rh_s = compute_spread_angles(_landmarks_to_array(rh))
        return np.concatenate([lh_a, rh_a, lh_s, rh_s])

    elif feature_type == "kinematic":
        cn_lh, cn_rh, cn_pose = center_and_normalize(lh, rh, pose)
        lh_a = compute_joint_angles(_landmarks_to_array(lh))
        rh_a = compute_joint_angles(_landmarks_to_array(rh))
        lh_s = compute_spread_angles(_landmarks_to_array(lh))
        rh_s = compute_spread_angles(_landmarks_to_array(rh))
        return np.concatenate([cn_lh, cn_rh, cn_pose, lh_a, rh_a, lh_s, rh_s])

    else:  # raw_landmarks
        return np.concatenate([lh, rh, pose])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def draw_ui(frame, current_letter, letter_conf, hold_progress, sentence,
            current_word, fps, latency_ms, style_name="", top3=None):
    h, w = frame.shape[:2]

    # Semi-transparent bars
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), COL_BG, -1)
    cv2.rectangle(overlay, (0, h - 70), (w, h), COL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Current detected letter (large) ──
    if current_letter:
        color = COL_GREEN if letter_conf >= 0.90 else COL_YELLOW
        cv2.putText(frame, current_letter, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5, cv2.LINE_AA)
        # Confidence
        cv2.putText(frame, f"{letter_conf*100:.0f}%", (100, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2, cv2.LINE_AA)

        # Hold progress bar
        bar_x = 150
        bar_w = 200
        bar_y = 55
        bar_h = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      COL_GRAY, 1)
        fill_w = int(bar_w * hold_progress)
        if fill_w > 0:
            bar_color = COL_GREEN if hold_progress >= 1.0 else COL_YELLOW
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          bar_color, -1)
        cv2.putText(frame, "HOLD", (bar_x + bar_w + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "?", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COL_GRAY, 3, cv2.LINE_AA)

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps:.0f} | {latency_ms:.0f}ms", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_CYAN, 1, cv2.LINE_AA)

    # ── Mode label ──
    cv2.putText(frame, "FINGERSPELLING", (w - 180, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GREEN, 1, cv2.LINE_AA)

    # ── Bottom: sentence + current word ──
    display = sentence + current_word + "_"
    if len(display) > 60:
        display = "..." + display[-57:]
    cv2.putText(frame, display, (15, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "SPACE: space | BKSP: delete | C: clear | 1-6: dots | ESC: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, COL_GRAY, 1, cv2.LINE_AA)
    # Style label
    if style_name:
        cv2.putText(frame, f"Style: {style_name}", (w - 180, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GRAY, 1, cv2.LINE_AA)

    # ── Top 3 predictions panel (right side) ──
    if top3:
        panel_x = w - 160
        panel_y = 110
        cv2.putText(frame, "Top 3:", (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CYAN, 1, cv2.LINE_AA)
        for i, (letter, conf) in enumerate(top3):
            y = panel_y + 25 + i * 28
            # Letter + confidence text
            color = COL_GREEN if i == 0 and conf >= 0.90 else COL_WHITE if i == 0 else COL_GRAY
            cv2.putText(frame, f"{letter}: {conf*100:.0f}%", (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Mini bar
            bar_w = int(80 * conf)
            cv2.rectangle(frame, (panel_x + 70, y - 10), (panel_x + 70 + bar_w, y - 2),
                          color, -1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(model_name, camera_idx, confidence_thresh, hold_frames):
    logger.info("=" * 60)
    logger.info("  Fingerspelling Mode — Real-Time A-Z Detection")
    logger.info("=" * 60)

    config = MODEL_CONFIGS[model_name]
    model_path = os.path.join(MODELS_DIR, f"isl_{model_name}_mlp.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{model_name}.pkl")

    for p in [model_path, scaler_path]:
        if not os.path.exists(p):
            logger.info(f"  [error] Not found: {p}")
            sys.exit(1)

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    logger.info(f"  Model:       {model_name} ({config['features']} features)")
    logger.info(f"  Confidence:  {confidence_thresh:.0%}")
    logger.info(f"  Hold frames: {hold_frames} (~{hold_frames/30:.1f}s)")
    logger.info("")

    # State
    pred_buffer = deque(maxlen=hold_frames)
    sentence = ""
    current_word = ""
    last_confirmed = None
    confirmed_cooldown = 0
    smooth_states = {}

    fps_times = deque(maxlen=30)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.info("  [error] Cannot open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    logger.info("  [ok] Webcam opened. Press ESC to quit.\n")

    cv2.namedWindow("ISL Fingerspelling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Fingerspelling", 640, 480)

    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    ) as holistic:

        while cap.isOpened():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            # Extract features from RAW landmarks
            features = extract_features(results, config["type"])

            # Custom smoothed drawing
            draw_hands_smooth(frame, results, smooth_states, alpha=0.35)

            current_letter = None
            current_conf = 0.0
            hold_progress = 0.0
            top3 = None

            if confirmed_cooldown > 0:
                confirmed_cooldown -= 1

            if features is not None:
                try:
                    features = np.array(features, dtype=np.float32).reshape(1, -1)
                    features_scaled = scaler.transform(features)
                    proba = model.predict(features_scaled, verbose=0)[0]
                    pred_idx = np.argmax(proba)
                    current_conf = float(proba[pred_idx])

                    # Top 3 predictions
                    top3_idx = np.argsort(proba)[-3:][::-1]
                    top3 = [(CLASS_LABELS[i], float(proba[i])) for i in top3_idx if i < len(CLASS_LABELS)]

                    if pred_idx < len(CLASS_LABELS) and current_conf >= confidence_thresh:
                        current_letter = CLASS_LABELS[pred_idx]
                        pred_buffer.append(current_letter)
                        
                        # Log instantaneous prediction for debugging
                        if getattr(run, "last_raw", None) != current_letter:
                            logger.info(f"  ? {current_letter} ({current_conf*100:.0f}%)")
                            run.last_raw = current_letter
                    else:
                        pred_buffer.append(None)
                        if getattr(run, "last_raw", None) is not None:
                            logger.info("  ? [empty]")
                            run.last_raw = None

                    # Check if buffer has strong enough majority
                    valid = [p for p in pred_buffer if p is not None]
                    if valid:
                        counter = Counter(valid)
                        top_letter, top_count = counter.most_common(1)[0]
                        # Require 80% agreement in the buffer
                        agreement = top_count / max(len(list(pred_buffer)), 1)
                        hold_progress = top_count / hold_frames

                        if (top_count >= hold_frames and agreement >= 0.80
                                and confirmed_cooldown == 0):
                            if top_letter != last_confirmed:
                                current_word += top_letter
                                last_confirmed = top_letter
                                confirmed_cooldown = hold_frames + 5
                                pred_buffer.clear()
                                logger.info(f"  [ok] {top_letter}  →  {current_word}")
                except Exception as e:
                    logger.error(f"  [error] Inference error: {e}")
                    pred_buffer.append(None)
            else:
                pred_buffer.append(None)

            # Removed previous draw code.

            # FPS
            t1 = time.perf_counter()
            fps_times.append(t1)
            fps = (len(fps_times)-1)/(fps_times[-1]-fps_times[0]) if len(fps_times) > 1 else 0

            draw_ui(frame, current_letter, current_conf, hold_progress,
                    sentence, current_word, fps, (t1-t0)*1000,
                    "",
                    top3 if features is not None else None)

            cv2.imshow("ISL Fingerspelling", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE — finish word
                if current_word:
                    sentence += current_word + " "
                    logger.info(f"  📝 Word: {current_word}")
                    current_word = ""
                    last_confirmed = None
                    pred_buffer.clear()
            elif key == 8:  # BACKSPACE
                if current_word:
                    current_word = current_word[:-1]
                    last_confirmed = None
                elif sentence:
                    sentence = sentence.rstrip()
                    # Remove last word
                    if " " in sentence:
                        sentence = sentence[:sentence.rfind(" ")] + " "
                    else:
                        sentence = ""
            elif key == ord("c") or key == ord("C"):
                sentence = ""
                current_word = ""
                last_confirmed = None
                pred_buffer.clear()
                logger.info("  🔄 Cleared")

    cap.release()
    cv2.destroyAllWindows()
    final = sentence + current_word
    logger.info(f"\n  Final text: {final}")
    logger.info("  Session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerspelling Mode")
    parser.add_argument("--model", default="angles_only",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use (default: angles_only)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.90,
                        help="Min confidence (default: 0.90)")
    parser.add_argument("--hold-frames", type=int, default=15,
                        help="Frames to hold same letter to confirm (default: 15)")
    args = parser.parse_args()
    run(args.model, args.camera, args.confidence, args.hold_frames)
