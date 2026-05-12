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
> **Important:** Always run from the project root using the `-m` flag to ensure internal package paths work.

```powershell
conda activate isl_py39
python -m src.inference.realtime_inference                    # default: kinematic model
python -m src.inference.realtime_inference --model angles_only # most robust for webcam
```
- **Live HUD**: Displays prediction, confidence, FPS, and latency.
- **Terminal Logs**: Live detection events (`[ok] Hands detected`, `--> Detected: A`) are logged to console.
- **Controls**: Press **1-7** to switch models live; **ESC** to quit.

### Fingerspelling Mode (A-Z sentence builder)
```powershell
python -m src.inference.realtime_fingerspell --model angles_only
```
- **Hold-to-confirm**: Hold a sign for ~0.5s to type the letter.
- **Controls**: **SPACE** for space, **BACKSPACE** to delete, **C** to clear.

---

## Troubleshooting: "ModuleNotFoundError: No module named 'src'"

If you get this error, it means you are likely running the script directly (e.g., `python src/inference/realtime_inference.py`). 

**Fix:** Use the `-m` syntax shown above:
```powershell
# Correct way:
python -m src.inference.realtime_inference
```

Alternatively, you can set the `PYTHONPATH` for your session:
```powershell
# PowerShell:
$env:PYTHONPATH = "."
python src/inference/realtime_inference.py
```

---

## Logging & Debugging

The project uses a unified logging system. All key events (loading, training, detection status, inference results) are sent to the **Console** for real-time feedback.

---

## Full Pipeline (From Scratch)

### Option A: Using the unified pipeline script

```powershell
# From a CSV file
python -m scripts.run_pipeline --input "dataset\Indian Sign Language Gesture Landmarks.csv" --name kaggle

# From an image dataset
python -m scripts.run_pipeline --input "dataset\dataset - Gesture Speech" --name img
```

### Option B: Running each phase manually

#### Phase 1 — Data Collection / Import
```powershell
python -m src.data_pipeline.isl_detection --mode import_kaggle
```
Output: `data/raw/isl_raw_data.csv`

#### Phase 2 — Kinematic Feature Engineering
```powershell
python -m src.features.kinematic_engineer
```
Output: `data/kinematic/isl_kinematic_data.csv`

#### Phase 3 — Model Training
```powershell
python -m src.modeling.train_classifier
```
Output: `models/isl_{run}_mlp.h5` and `plots/`

#### Phase 1b — Extract from images (optional)
```powershell
python -m src.data_pipeline.image_pipeline --phase 1
```
Output: `data/raw/img_raw_data.csv`

#### Phase 2 — Kinematic engineering
```powershell
python -m src.features.kinematic_engineer
```
Output: `data/kinematic/isl_kinematic_data.csv`, `data/angles/isl_angles_only.csv`

#### Phase 3 — Train models
```powershell
python -m src.modeling.train_classifier
```
Output: `models/*.h5`, `models/*.pkl`, `plots/*.png`

#### Phase 4 — Run inference
```powershell
python -m src.inference.realtime_inference --model kinematic

---

### Dynamic Sign Pipeline (Phases 5–7) — Body-Aware

> These phases are **fully isolated** from Phases 1–4. They use separate
> scripts, data directories, and model files.

#### Phase 5 — Temporal Feature Engineering (Body-Aware, 240-dim)
```powershell
python -m src.features.temporal_engineer_bodyaware               # both data sources
python -m src.features.temporal_engineer_bodyaware --source video # .MOV files only
python -m src.features.temporal_engineer_bodyaware --source npy   # .npy files only
```
Output: `data/dynamic_bodyaware/` (tensors + `label_map_body.json`)

#### Phase 6 — Train Dynamic Model (Body-Aware)
```powershell
python -m src.modeling.train_dynamic_bodyaware              # GRU (default)
python -m src.modeling.train_dynamic_bodyaware --use_lstm    # LSTM instead
```
Output: `models/isl_bodyaware_lstm.h5`, `plots/bodyaware_confusion.png`

#### Phase 7 — Real-Time Dynamic Inference (Body-Aware)
```powershell
python -m src.inference.realtime_dynamic_bodyaware
python -m src.inference.realtime_dynamic_bodyaware --camera 1
```

#### Consistency Test (240-dim parity)
```powershell
python -m tests.test_kinematics_parity
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
| `ModuleNotFoundError` | Use `-m` flag (e.g. `python -m src.inference.realtime_inference`) |
| `mediapipe==0.9.1.0` not found | Use updated `requirements.txt` (uses `0.10.11`) |
| `protobuf` / `builder` import error | Ensure `tensorflow==2.13.1` is installed (not 2.11) |
| `grpcio` build fails | Use **Python 3.9** via Conda — not 3.13 |
| "Model not found" | Run Phase 3 first, or check `models/` directory |
| Poor webcam predictions | Use `angles_only` model, ensure good lighting |
| Hud is missing | Ensure you are running with `python -m` from the root |
| Window too small/big | Drag window edges to resize |
| Slow inference | Lower `--buffer` size, or use `angles_only` (fewer features) |
| ImportError | Activate conda env and run `pip install -r requirements.txt` |
