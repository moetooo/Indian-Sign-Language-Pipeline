# -*- coding: utf-8 -*-
"""
Common Constants and Utility Functions
"""
import copy

HAND_LANDMARK_COUNT = 21
POSE_INDICES = [11, 12, 13, 14, 15, 16]
AXES = ["x", "y", "z"]
CLASS_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

def _hand_cols(prefix: str) -> list[str]:
    cols = []
    for i in range(HAND_LANDMARK_COUNT):
        for ax in AXES:
            cols.append(f"{prefix}_{ax}{i}")
    return cols

def _pose_cols() -> list[str]:
    cols = []
    for idx in POSE_INDICES:
        for ax in AXES:
            cols.append(f"pose_{ax}{idx}")
    return cols
