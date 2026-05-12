# ISL Detection — Complete Project Walkthrough

## Overview

This project implements a complete **Indian Sign Language (ISL) Detection** system that recognizes 26 ASL/ISL alphabet letters (A-Z) in real-time via webcam. The system is built as a **4-phase pipeline**: Data Capture → Feature Engineering → Model Training → Real-Time Inference.

---

## Phase 1 — Data Capture & Landmark Extraction

**Purpose:** Extract hand + upper-body landmarks from video/images using MediaPipe.

### What happens:
1. **MediaPipe Holistic** processes each frame/image
2. Extracts **21 landmarks per hand** (left + right) × 3 axes (x, y, z) = **126 hand values**
3. Extracts **6 upper-body pose landmarks** (shoulders, elbows, wrists) × 3 axes = **18 pose values**
4. Total: **144 features per sample** + 3 metadata columns (label, source, user_id) = **147 columns**

### Data sources used:
| Source | Method | Samples |
|--------|--------|---------|
| Kaggle CSV (pre-extracted) | Direct import | ~78,000 |
| Gesture Speech Images | MediaPipe Hands on 31,200 JPGs | ~28,600 |
| Live webcam | Real-time capture | User-defined |

### Output:
- `data/raw/isl_raw_data.csv` — Kaggle-sourced landmarks
- `data/raw/img_raw_data.csv` — Image-extracted landmarks

---

## Phase 2 — Kinematic Feature Engineering

**Purpose:** Transform raw landmarks into position/scale-invariant features.

### Three transformations applied:

#### 1. Translation Invariance (Centering)
- Single hand: wrist set to (0, 0, 0)
- Both hands: midpoint of both wrists set to (0, 0, 0)
- Pose: mid-shoulder set to (0, 0, 0)

#### 2. Scale Invariance (Normalization)
- All coordinates divided by the maximum absolute value per sample
- Result: all values in [-1, 1], max = 1.0

#### 3. Joint Angles & Spread Angles
- **15 joint angles per hand** — flexion/extension of finger joints (MCP, PIP, DIP)
- **4 spread angles per hand** — inter-finger abduction angles
- Total: **38 angle features** (invariant to position, scale, rotation)

### Output:
- `data/kinematic/isl_kinematic_data.csv` — 185 cols (centered coords + angles)
- `data/angles/isl_angles_only.csv` — 41 cols (angles only)

---

## Phase 3 — Model Training & Ablation Study

**Purpose:** Train Deep MLP classifiers on different feature sets and compare.

### Architecture (identical for all runs):
```
Input → Dense(512, ReLU) → BatchNorm → Dropout(0.3)
      → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
      → Dense(128, ReLU) → BatchNorm → Dropout(0.2)
      → Dense(26, Softmax) → Output
```

### Training config:
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Cross-Entropy
- EarlyStopping (patience=5), ReduceLROnPlateau (patience=3)
- 80/15/15 train/val/test split, max 50 epochs

### Results — 7 Models:

| # | Model | Features | Dataset | Accuracy | F1 |
|---|-------|----------|---------|----------|-----|
| 1 | `source_kaggle` | 126 | Kaggle CSV | 99.92% | 0.9992 |
| 2 | `raw` | 144 | Kaggle CSV | 99.92% | 0.9992 |
| 3 | `kinematic` | 182 | Kaggle CSV | 99.79% | 0.9979 |
| 4 | `angles_only` | 38 | Kaggle CSV | 99.75% | 0.9975 |
| 5 | `img_raw` | 144 | Images | 99.16% | 0.9916 |
| 6 | `img_kinematic` | 182 | Images | 98.91% | 0.9891 |
| 7 | `img_angles_only` | 38 | Images | 96.47% | 0.9647 |

### Output:
- `models/*.h5` — trained model weights
- `models/*.pkl` — fitted StandardScaler objects
- `plots/cm_*.png` — confusion matrices
- `plots/history_*.png` — training loss/accuracy curves
- `plots/ablation_summary.csv` — comparison table

---

## Phase 4 — Real-Time Inference

**Purpose:** Live webcam → prediction with stability logic.

### Pipeline per frame:
```
Webcam Frame → MediaPipe Holistic → Extract Landmarks
             → Build Feature Vector → Scale (StandardScaler)
             → MLP Prediction → Stability Buffer → HUD Display
```

### Stability logic:
- **Rolling buffer** of last 5 predictions
- **Majority vote**: must appear in ≥60% of buffer
- **Confidence threshold**: average confidence must be ≥85%
- Only shows prediction when all criteria met

### Controls:
| Key | Action |
|-----|--------|
| ESC | Quit |
| 1-7 | Switch between models live |

### HUD shows:
- Predicted letter (large, color-coded by confidence)
- Confidence percentage
- FPS and latency
- Active model name and dataset
- Hand visibility warning

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| MLP over CNN | Landmarks are already extracted — no spatial/image processing needed |
| MediaPipe Hands for images | Holistic model failed on rendered 3D hand images (2% detection). Hands model achieved 92% |
| Kinematic features | Position-invariant features generalize better across users and environments |
| `angles_only` as best for webcam | Only 38 features, fully invariant to hand position/scale/rotation |
| StandardScaler persistence | Ensures inference uses exact same scaling as training |
