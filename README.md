# ISL-Vision-Engine: A Comprehensive Indian Sign Language Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-brightgreen)
![Status](https://img.shields.io/badge/Status-Phase%204%20(Static%20Complete)-success)



> A robust, real-time Indian Sign Language (ISL) recognition system. **Part 1** achieves **99.9% accuracy** on static alphabet classification using Kinematic Feature Engineering and Deep MLPs. **Part 2** will extend this to continuous temporal translation (sentences) using LSTMs and sliding windows.

---

## 🌟 What is this project? (Overview)

Most traditional Sign Language Recognition systems feed raw images (pixels) into heavy Convolutional Neural Networks (CNNs). This requires massive GPU power, enormous datasets, and usually fails when the user moves slightly or the camera angle changes.

**We took a different approach.** 

By utilizing Google's **MediaPipe**, we extract the 3D human skeleton directly. We then apply **Kinematic Feature Engineering** — mathematically transforming raw coordinates into **Joint Angles** and **Normalized Distances**. 

This means our network doesn't look at *where* your hand is, it looks at the *pure geometric shape* of your hand. Because of this:
*   We achieve **99.9% real-world accuracy**.
*   The model uses only **38 input features** (not millions of pixels).
*   It is completely **Position, Scale, and Rotation Invariant**.
*   It trains in **~60 seconds** on a standard CPU.
*   Inference runs at **30+ FPS** on basic webcams.

---

## 📂 File Structure & Architecture

The project is structured as a modular 4-phase pipeline.

```text
Indian-Sign-Language-Pipeline/
├── data/                          ← Generated data (Raw + Kinematic + Angles CSVs)
├── models/                        ← Trained MLP models (.h5) & Scalers (.pkl)
├── plots/                         ← Confusion matrices & training histories
├── dataset/                       ← Source datasets (Kaggle CSV)

├── paths.py                       ← Central path configuration
├── isl_detection.py               ← Phase 1: Data Collection & Kaggle Import
├── kinematic_engineer.py          ← Phase 2: Math/Feature Engineering
├── train_classifier.py            ← Phase 3: Ablation Study & MLP Training
├── realtime_inference.py          ← Phase 4: OpenCV Live Webcam Inference
├── test_all_letters.py            ← Guided A-Z Letter Test Script
├── run_pipeline.py                ← Single-command execution (Phases 1-3)

```

---

## 💻 Setup & Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone https://github.com/username/Indian-Sign-Language-Pipeline.git
cd Indian-Sign-Language-Pipeline

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Live Webcam Inference (Quick Start)
If models are already trained (located in the `models/` directory):

```powershell
# Run real-time webcam inference (Press 1-4 to switch models live)

python src/inference/realtime_inference.py --model angles_only

# Take the guided A-Z ISL Letter Test against the AI
python tests/test_all_letters.py --model angles_only
```

### 3. Training the Pipeline from Scratch
To run the automated data ingestion, feature engineering, and model training:

```powershell
# Run full pipeline on the Kaggle CSV
python scripts/run_pipeline.py --input "dataset\Indian Sign Language Gesture Landmarks.csv" --name kaggle

# Or run specific phases manually
python src/data_pipeline/isl_detection.py --mode import_kaggle
python src/features/kinematic_engineer.py
python src/modeling/train_classifier.py
```

---

## 🗺️ The 7-Phase Roadmap

### Part 1: Static ISL Recognition (✅ Current)
*   **Phase 1 — Data Collection:** MediaPipe Holistic landmark extraction resulting in 78,000+ CSV samples from Kaggle dataset.

*   **Phase 2 — Kinematic Feature Engineering:** Centering on the wrist, scaling/normalization, and calculating 38 precise Joint & Spread Angles per hand.
*   **Phase 3 — Deep MLP Ablation Study:** Training, comparing, and evaluating 4 distinct models across 4 unique mathematical feature variants.

*   **Phase 4 — Real-Time Inference:** Production-ready OpenCV GUI featuring live model-switching via keyboard, a 5-frame rolling stability buffer, and >85% confidence gating.

### Part 2: Dynamic ISL Translation (🔜 Planned)
*   **Phase 5 — Temporal Data Engineering:** Reshaping flat CSV data into `[Samples, 30_Frames, Features]` sequences using large-scale ISL-CSLTR video datasets.
*   **Phase 6 — Temporal Sequence Modeling:** Training a lightweight multi-layer LSTM to recognize continuous signs over 1-second video bursts.
*   **Phase 7 — Continuous Translation Engine:** Developing a real-time sliding window with **Neutral-Pose Word Boundary Detection** to intelligently string signs into full sentences, validated against the iSign Benchmark.
