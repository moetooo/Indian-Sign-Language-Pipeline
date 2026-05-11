# -*- coding: utf-8 -*-
"""
Phase 3 — ISL Classifier Training & Ablation Study
====================================================
Trains identical Deep MLP classifiers on four feature representations and
compares their performance.

Ablation runs:
    1. source_kaggle  — 126 raw Kaggle landmarks
    2. raw            — 144 Phase-1 features
    3. kinematic      — 182 Phase-2 centred coords + angles + spreads
    4. angles_only    — 38  Phase-2 angles + spreads only

Outputs (in results/):
    - isl_{run}_mlp.h5          saved Keras model
    - cm_{run}.png              confusion matrix heatmap
    - history_{run}.png         training loss/accuracy curves
    - ablation_summary.csv      comparison table
"""

from src.utils.logger import logger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
from src.utils.paths import MODELS_DIR, PLOTS_DIR, ensure_dirs

RESULTS_DIR = MODELS_DIR   # backward compat for external callers
RANDOM_STATE = 42
TEST_SIZE = 0.15      # 15% test
VAL_SIZE = 0.176      # 15% of total  (0.176 of remaining 85% ≈ 15%)
BATCH_SIZE = 128
MAX_EPOCHS = 50

# Ablation run definitions
# Each entry: (run_name, csv_path, label_col, drop_cols)
ABLATION_RUNS = [
    (
        "source_kaggle",
        os.path.join("dataset", "Indian Sign Language Gesture Landmarks.csv"),
        "target",
        ["target", "uses_two_hands"],
    ),
    (
        "raw",
        os.path.join("data", "raw", "isl_raw_data.csv"),
        "label",
        ["label", "source", "user_id"],
    ),
    (
        "kinematic",
        os.path.join("data", "kinematic", "isl_kinematic_data.csv"),
        "label",
        ["label", "source", "user_id"],
    ),
    (
        "angles_only",
        os.path.join("data", "angles", "isl_angles_only.csv"),
        "label",
        ["label", "source", "user_id"],
    ),
]


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────
def build_mlp(input_dim: int, num_classes: int) -> keras.Model:
    """
    Build a Deep MLP with BatchNorm and Dropout.

    Architecture:
        Input → 512 → 256 → 128 → 64 → num_classes (softmax)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_labels, run_name, save_path):
    """Save a confusion matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {run_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"    Saved: {save_path}")


def plot_training_history(history, run_name, save_path):
    """Save training loss & accuracy curves as PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {run_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Train Acc", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy — {run_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"    Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────
def run_experiment(run_name, csv_path, label_col, drop_cols):
    """
    Load data, train model, evaluate, save outputs for one ablation run.
    Returns a dict of metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  RUN: {run_name}")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────
    t0 = time.time()
    logger.info(f"  Loading data...")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"    Shape: {df.shape}")

    # Clean labels: drop NaN, convert all to string
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""]

    y_raw = df[label_col].values
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    logger.info(f"    Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # ── Encode labels ─────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    class_labels = [str(c) for c in le.classes_]
    logger.info(f"    Classes: {num_classes}  ({class_labels[:5]}{'...' if num_classes > 5 else ''})")

    # ── Split ─────────────────────────────────────────────────────
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )
    logger.info(f"    Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ── Scale ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── Save scaler for inference ─────────────────────────────────
    s_path = os.path.join(MODELS_DIR, f"scaler_{run_name}.pkl")
    joblib.dump(scaler, s_path)
    logger.info(f"    Scaler saved: {s_path}")

    # ── Build & train ─────────────────────────────────────────────
    logger.info(f"  Building model (input_dim={X_train.shape[1]}, classes={num_classes})...")
    model = build_mlp(X_train.shape[1], num_classes)
    model.summary(print_fn=lambda x: None)  # suppress summary

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(f"  Training (max {MAX_EPOCHS} epochs, batch={BATCH_SIZE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    # ── Evaluate ──────────────────────────────────────────────────
    logger.info(f"  Evaluating on test set...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"\n  ── Results ──")
    logger.info(f"    Accuracy:        {acc:.4f}")
    logger.info(f"    Precision (M):   {prec:.4f}")
    logger.info(f"    Recall (M):      {rec:.4f}")
    logger.info(f"    F1 (weighted):   {f1:.4f}")
    logger.info(f"    F1 (macro):      {f1_macro:.4f}")
    logger.info(f"    Train time:      {train_time:.1f}s")
    logger.info(f"    Epochs run:      {len(history.history['loss'])}")

    # ── Save model ────────────────────────────────────────────────
    m_path = os.path.join(MODELS_DIR, f"isl_{run_name}_mlp.h5")
    model.save(m_path)
    logger.info(f"    Model saved: {m_path}")

    # ── Plots ─────────────────────────────────────────────────────
    c_path = os.path.join(PLOTS_DIR, f"cm_{run_name}.png")
    plot_confusion_matrix(y_test, y_pred, class_labels, run_name, c_path)

    h_path = os.path.join(PLOTS_DIR, f"history_{run_name}.png")
    plot_training_history(history, run_name, h_path)

    # ── Classification report (to console) ────────────────────────
    logger.info(f"\n  Classification Report ({run_name}):")
    logger.info(classification_report(y_test, y_pred, target_names=class_labels))

    return {
        "run": run_name,
        "features": X.shape[1],
        "samples": X.shape[0],
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "epochs": len(history.history["loss"]),
        "train_time_s": round(train_time, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  Phase 3 — ISL Classifier Training & Ablation Study")
    logger.info("=" * 60)
    logger.info(f"  TensorFlow: {tf.__version__}")
    logger.info(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    logger.info(f"  Models dir: {MODELS_DIR}/")
    logger.info(f"  Plots dir:  {PLOTS_DIR}/")
    logger.info("")

    ensure_dirs()

    results = []
    for run_name, csv_path, label_col, drop_cols in ABLATION_RUNS:
        if not os.path.exists(csv_path):
            logger.info(f"\n  [SKIP] {csv_path} not found — skipping {run_name}")
            continue
        metrics = run_experiment(run_name, csv_path, label_col, drop_cols)
        results.append(metrics)

    # ── Summary table ─────────────────────────────────────────────
    if results:
        logger.info(f"\n{'='*60}")
        logger.info("  ABLATION STUDY SUMMARY")
        logger.info(f"{'='*60}")
        df_summary = pd.DataFrame(results)
        logger.info(df_summary.to_string(index=False))

        summary_path = os.path.join(PLOTS_DIR, "ablation_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"\n  Summary saved: {summary_path}")

        # Highlight best
        best_idx = df_summary["f1_weighted"].idxmax()
        best = df_summary.loc[best_idx]
        logger.info(f"\n  ★ Best run: {best['run']}  (F1-weighted = {best['f1_weighted']:.4f})")
    else:
        logger.info("\n  No runs completed. Check that CSV files exist.")

    logger.info(f"\n{'='*60}")
    logger.info("  Phase 3 complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
