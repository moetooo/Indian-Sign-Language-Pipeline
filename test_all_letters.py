# -*- coding: utf-8 -*-
"""
Guided Letter-by-Letter Test
==============================
Walks you through all 26 ISL letters (A-Z), one at a time.
For each letter:
  1. Shows which letter to sign on screen
  2. Gives you a countdown to hold the sign
  3. Captures the model's prediction
  4. Logs whether it matched

Usage:
    python test_all_letters.py
    python test_all_letters.py --hold 5        # 5 second hold per letter
    python test_all_letters.py --model raw     # use raw model
"""

import argparse
import os
import sys
import time
import warnings
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
from tensorflow import keras

from kinematic_engineer import (
    center_and_normalize,
    compute_joint_angles,
    compute_spread_angles,
    _landmarks_to_array,
)
from realtime_inference import (
    extract_landmarks_live,
    FEATURE_BUILDERS,
    MODEL_CONFIGS,
    CLASS_LABELS,
    RESULTS_DIR,
    draw_landmarks,
)

mp_holistic = mp.solutions.holistic

# Colours (BGR)
COL_WHITE  = (255, 255, 255)
COL_GREEN  = (0, 220, 0)
COL_RED    = (0, 0, 230)
COL_YELLOW = (0, 230, 255)
COL_CYAN   = (255, 220, 0)
COL_BG     = (30, 30, 30)
COL_GRAY   = (120, 120, 120)


def run_test(model_name="kinematic", camera_idx=0, hold_seconds=4):
    # Load model + scaler
    model_path  = os.path.join(RESULTS_DIR, f"isl_{model_name}_mlp.h5")
    scaler_path = os.path.join(RESULTS_DIR, f"scaler_{model_name}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler not found in {RESULTS_DIR}/")
        sys.exit(1)

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    feature_builder = FEATURE_BUILDERS[model_name]

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("Cannot open webcam!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    results_log = []  # (letter, predicted, correct)
    letter_idx = 0

    print("\n  Guided ISL Letter Test")
    print("  ======================")
    config = MODEL_CONFIGS[model_name]
    print(f"  Model: {model_name} | {config['features']}f | {config.get('dataset', '')}")
    print(f"  Hold: {hold_seconds}s per letter")
    print(f"  Press SPACE to start each letter, ESC to quit")
    print(f"\n  Available models (use --model <name>):")
    for i, (name, cfg) in enumerate(MODEL_CONFIGS.items(), 1):
        marker = " *" if name == model_name else ""
        print(f"    {i}. {name:<18} {cfg['features']:>3}f  {cfg.get('dataset','')}{marker}")
    print()

    # Create resizable window
    cv2.namedWindow("ISL Letter Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Letter Test", 640, 480)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while letter_idx < 26:
            target_letter = CLASS_LABELS[letter_idx]

            # === PHASE 1: Show instruction, wait for SPACE ===
            while True:
                success, frame = cap.read()
                if not success:
                    continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Dark overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), COL_BG, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Instructions
                cv2.putText(frame, f"Letter {letter_idx+1}/26", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_CYAN, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Show: {target_letter}", (w//2 - 80, h//2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, COL_YELLOW, 5, cv2.LINE_AA)
                cv2.putText(frame, "Press SPACE when ready", (w//2 - 160, h//2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2, cv2.LINE_AA)
                cv2.putText(frame, "ESC = quit | S = skip", (w//2 - 130, h//2 + 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GRAY, 1, cv2.LINE_AA)

                # Show progress so far
                if results_log:
                    correct = sum(1 for _, _, c in results_log if c)
                    cv2.putText(frame, f"Score: {correct}/{len(results_log)}", (w - 180, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_GREEN, 2, cv2.LINE_AA)

                cv2.imshow("ISL Letter Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    letter_idx = 99
                    break
                if key == ord(' '):
                    break
                if key == ord('s') or key == ord('S'):
                    results_log.append((target_letter, "SKIP", False))
                    letter_idx += 1
                    break

            if letter_idx >= 26:
                break

            if results_log and results_log[-1][1] == "SKIP":
                continue

            # === PHASE 2: Countdown + capture predictions ===
            predictions = []
            start_time = time.time()
            correct_streak_start = None  # tracks when correct prediction streak began
            wrong_streak_start = None    # tracks when wrong prediction streak began
            AUTO_ADVANCE_SECONDS = 5     # auto-advance after 5s of correct detection
            WRONG_SKIP_SECONDS = 15      # auto-skip after 15s of wrong detection

            while time.time() - start_time < hold_seconds:
                success, frame = cap.read()
                if not success:
                    continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                mp_results = holistic.process(frame_rgb)
                frame_rgb.flags.writeable = True

                # Draw landmarks
                draw_landmarks(frame, mp_results)

                # Extract + predict
                landmark_data = extract_landmarks_live(mp_results)
                pred_letter = "?"
                confidence = 0.0
                any_hand = False

                if landmark_data is not None:
                    lh, rh, pose = landmark_data
                    lh_visible = np.any(np.abs(lh) > 1e-9)
                    rh_visible = np.any(np.abs(rh) > 1e-9)
                    any_hand = lh_visible or rh_visible

                    if any_hand:
                        try:
                            features = feature_builder(lh, rh, pose)
                            features = np.array(features, dtype=np.float32).reshape(1, -1)
                            features_scaled = scaler.transform(features)
                            proba = model.predict(features_scaled, verbose=0)[0]
                            pred_idx = np.argmax(proba)
                            confidence = float(proba[pred_idx])
                            if pred_idx < len(CLASS_LABELS):
                                pred_letter = CLASS_LABELS[pred_idx]
                                predictions.append(pred_letter)
                        except Exception:
                            pass

                # Show warning if no hand visible
                if not any_hand:
                    cv2.putText(frame, "Show your hand!", (w // 2 - 120, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_RED, 2, cv2.LINE_AA)

                # --- Auto-advance logic ---
                if pred_letter == target_letter and confidence >= 0.85:
                    if correct_streak_start is None:
                        correct_streak_start = time.time()
                    streak_duration = time.time() - correct_streak_start
                    if streak_duration >= AUTO_ADVANCE_SECONDS:
                        break  # auto-advance!
                    wrong_streak_start = None  # reset wrong streak
                else:
                    correct_streak_start = None
                    # Track wrong streak (only after some predictions exist)
                    if len(predictions) >= 3 and pred_letter != target_letter:
                        if wrong_streak_start is None:
                            wrong_streak_start = time.time()
                        wrong_duration = time.time() - wrong_streak_start
                        if wrong_duration >= WRONG_SKIP_SECONDS:
                            break  # auto-skip, too long wrong
                    elif pred_letter == "?":
                        pass  # don't reset if no detection
                    else:
                        wrong_streak_start = None

                # HUD
                elapsed = time.time() - start_time
                remaining = max(0, hold_seconds - elapsed)

                # Top bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 90), COL_BG, -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Progress bar
                progress = elapsed / hold_seconds
                bar_w = int((w - 40) * progress)
                cv2.rectangle(frame, (20, 75), (w - 20, 85), COL_GRAY, -1)
                cv2.rectangle(frame, (20, 75), (20 + bar_w, 85), COL_GREEN, -1)

                cv2.putText(frame, f"Target: {target_letter}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_YELLOW, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Pred: {pred_letter} ({confidence*100:.0f}%)", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{remaining:.1f}s", (w - 80, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_CYAN, 2, cv2.LINE_AA)

                # Show lock indicator when correct prediction is being held
                if correct_streak_start is not None:
                    streak_s = time.time() - correct_streak_start
                    lock_pct = min(streak_s / AUTO_ADVANCE_SECONDS, 1.0)
                    lock_bar = int((w - 40) * lock_pct)
                    cv2.rectangle(frame, (20, 88), (20 + lock_bar, 93), COL_GREEN, -1)
                    cv2.putText(frame, f"LOCKED {streak_s:.1f}s/{AUTO_ADVANCE_SECONDS}s",
                                (w - 200, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GREEN, 1, cv2.LINE_AA)

                cv2.imshow("ISL Letter Test", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    letter_idx = 99
                    break

            if letter_idx >= 26:
                break

            # === PHASE 3: Evaluate majority prediction ===
            if predictions:
                counter = Counter(predictions)
                majority_pred, _ = counter.most_common(1)[0]
            else:
                majority_pred = "NONE"

            is_correct = (majority_pred == target_letter)
            results_log.append((target_letter, majority_pred, is_correct))

            icon = "OK" if is_correct else "WRONG"
            color = COL_GREEN if is_correct else COL_RED
            print(f"    [{icon}]  Target: {target_letter}  Predicted: {majority_pred}  "
                  f"({len(predictions)} samples)")

            # Brief result flash
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), COL_BG, -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                result_text = f"{target_letter} -> {majority_pred}"
                cv2.putText(frame, result_text, (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
                status = "CORRECT!" if is_correct else "WRONG"
                cv2.putText(frame, status, (w//2 - 80, h//2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                cv2.imshow("ISL Letter Test", frame)
                cv2.waitKey(1500)  # Show result for 1.5 seconds

            letter_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # === SUMMARY ===
    if results_log:
        print(f"\n  {'='*50}")
        print(f"  RESULTS SUMMARY")
        print(f"  {'='*50}")
        correct = sum(1 for _, _, c in results_log if c)
        total = len(results_log)
        skipped = sum(1 for _, p, _ in results_log if p == "SKIP")
        tested = total - skipped
        print(f"  Tested:  {tested}/26")
        print(f"  Correct: {correct}/{tested}  ({correct/tested*100:.0f}%)" if tested > 0 else "")
        print(f"  Skipped: {skipped}")
        print()
        print(f"  {'Letter':<8} {'Predicted':<10} {'Result'}")
        print(f"  {'-'*30}")
        for letter, pred, ok in results_log:
            icon = "OK" if ok else ("SKIP" if pred == "SKIP" else "MISS")
            print(f"  {letter:<8} {pred:<10} {icon}")
        print(f"  {'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Guided ISL Letter Test")
    parser.add_argument("--model", default="kinematic",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hold", type=int, default=30,
                        help="Seconds to hold each sign (default: 30)")
    args = parser.parse_args()

    run_test(model_name=args.model, camera_idx=args.camera,
             hold_seconds=args.hold)


if __name__ == "__main__":
    main()
