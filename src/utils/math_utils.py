# -*- coding: utf-8 -*-
"""
Mathematical Utilities for Kinematic Feature Engineering
======================================================
Shared math functions for computing joint angles, normalized distances,
and spatial invariant representations for the MediaPipe hand landmarks.
"""
import numpy as np

# A small epsilon to avoid division by zero
EPSILON = 1e-9

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle (radians) between two 3D vectors using the dot-product formula.
    Returns 0.0 if either vector has zero magnitude.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < EPSILON or norm2 < EPSILON:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp to [-1, 1] to avoid numerical issues with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))

def is_hand_present(landmarks_21x3: np.ndarray) -> bool:
    """Check if a hand has actual data (not all zeros)."""
    return np.any(np.abs(landmarks_21x3) > EPSILON)

def is_pose_present(pose_flat: np.ndarray) -> bool:
    """Check if pose (upper body) data is present."""
    return np.any(np.abs(pose_flat) > EPSILON)
