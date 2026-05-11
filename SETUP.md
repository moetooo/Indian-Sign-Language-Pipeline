# ISL Detection — Setup & Usage Guide

## Requirements

- **Python 3.9** (tested and recommended)
- **Conda** (Miniconda or Anaconda)
- **Webcam** (for Phase 1 capture and Phase 4 inference)
- **Windows 10/11** (tested), also works on Linux/macOS

---

## Installation

### 1. Clone the project
```bash
git clone https://github.com/moetooo/Indian-Sign-Language-Pipeline.git
cd Indian-Sign-Language-Detection
```

### 2. Create a Conda environment with Python 3.9
```powershell
conda create -n isl_py39 python=3.9 -y
```

### 3. Activate the environment
```powershell
conda activate isl_py39
```

### 4. Install dependencies
```powershell
pip install -r requirements.txt
```

### Key dependencies:
| Package | Purpose |
|---------|---------|
| `mediapipe` | Hand/pose landmark detection |
| `opencv-python` | Webcam capture and image processing |
| `tensorflow` | Deep learning model training & inference |
| `scikit-learn` | Data splitting, scaling, metrics |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Plots and confusion matrices |
| `joblib` | Scaler persistence |

> **Note:** The project requires `tensorflow==2.13.1` and `mediapipe==0.10.11`.
> These have been tested to work together on **Python 3.9 + Windows**.
> Python 3.10+ may encounter protobuf version conflicts.

---

## Quick Start (Using Pre-Trained Models)

If models are already trained (files exist in `models/`):

### Real-time inference (webcam)
```powershell
conda activate isl_py39
python realtime_inference.py                    # default: kinematic model
python realtime_inference.py --model angles_only # most robust for webcam
python realtime_inference.py --model img_raw     # image-trained model
```
- Press **1-7** to switch models live
- Press **ESC** to quit

### Guided letter test (A-Z)
```powershell
python test_all_letters.py                        # default: kinematic
python test_all_letters.py --model angles_only    # recommended
python test_all_letters.py --model img_raw --hold 15
```
- Press **SPACE** to start each letter
- Press **S** to skip, **ESC** to quit
- Auto-advances after 5s of correct detection

---

## Full Pipeline (From Scratch)

### Option A: Using the unified pipeline script

```powershell
# From a CSV file
python run_pipeline.py --input "dataset\Indian Sign Language Gesture Landmarks.csv" --name kaggle

# From an image dataset
python run_pipeline.py --input "dataset\dataset - Gesture Speech" --name img

# Only specific phases
python run_pipeline.py --input data.csv --name test --phase 2 3
```

### Option B: Running each phase manually

#### Phase 1 — Import Kaggle data
```powershell
python isl_detection.py --mode import_kaggle
```
Output: `data/raw/isl_raw_data.csv`

#### Phase 1b — Extract from images (optional)
```powershell
python image_pipeline.py --phase 1
```
Output: `data/raw/img_raw_data.csv`

#### Phase 2 — Kinematic engineering
```powershell
python kinematic_engineer.py
```
Output: `data/kinematic/isl_kinematic_data.csv`, `data/angles/isl_angles_only.csv`

#### Phase 3 — Train models
```powershell
python train_classifier.py
```
Output: `models/*.h5`, `models/*.pkl`, `plots/*.png`

#### Phase 4 — Run inference
```powershell
python realtime_inference.py --model kinematic
```

---

## Available Models

| Key | `--model` value | Features | Best for |
|-----|----------------|----------|----------|
| 1 | `source_kaggle` | 126 | Testing on Kaggle data |
| 2 | `raw` | 144 | Raw landmark comparison |
| 3 | `kinematic` | 182 | General use (centered + angles) |
| 4 | `angles_only` | 38 | **Best webcam performance** |
| 5 | `img_raw` | 144 | Image dataset comparison |
| 6 | `img_kinematic` | 182 | Image + kinematic |
| 7 | `img_angles_only` | 38 | Image + angles |

---

## CLI Reference

### `isl_detection.py`
```
--mode    capture | import_kaggle | detect | detect_v2
--user    User ID for webcam capture (default: user1)
--camera  Webcam index (default: 0)
```

### `realtime_inference.py`
```
--model       Model name from MODEL_CONFIGS (default: kinematic)
--camera      Webcam index (default: 0)
--confidence  Min confidence threshold (default: 0.85)
--buffer      Rolling buffer size (default: 5)
```

### `test_all_letters.py`
```
--model   Model name (default: kinematic)
--camera  Webcam index (default: 0)
--hold    Seconds per letter (default: 30)
```

### `run_pipeline.py`
```
--input   CSV file path or image directory (required)
--name    Model name prefix (required)
--phase   Phases to run: 1 2 3 (default: all)
--outdir  Output directory for models (default: models/)
```

### `image_pipeline.py`
```
--phase   Specific phase: 1, 2, or 3 (default: all)
--limit   Max images per class for Phase 1
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mediapipe==0.9.1.0` not found | Use updated `requirements.txt` (uses `0.10.11`) |
| `protobuf` / `builder` import error | Ensure `tensorflow==2.13.1` is installed (not 2.11) |
| `grpcio` build fails | Use **Python 3.9** via Conda — not 3.13 |
| "Model not found" | Run Phase 3 first, or check `models/` directory |
| Poor webcam predictions | Use `angles_only` model, ensure good lighting |
| "Show your hand!" warning | Move hand into camera frame |
| Window too small/big | Drag window edges to resize |
| Slow inference | Lower `--buffer` size, or use `angles_only` (fewer features) |
| ImportError | Activate conda env and run `pip install -r requirements.txt` |
