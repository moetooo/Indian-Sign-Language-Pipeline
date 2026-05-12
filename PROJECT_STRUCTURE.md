# Project File Structure & Architecture Documentation

This document provides a comprehensive, in-depth explanation of the entire file structure for the **Indian-Sign-Language-Detection** project. It details every file, directory, their purpose in the larger 7-Phase architecture, and provides actionable suggestions for better project management.

---

## 📂 Root Directory Architecture

The root directory acts as the control center, containing all execution scripts, pipeline phases, and project documentation. The core logic is broken down into 7 sequential phases transitioning from static alphabet recognition (MLP) to dynamic sentence translation (LSTM).

### 🐍 Core Python Scripts (The Pipeline)

*   **`isl_detection.py` (Phase 1)**
    *   **Purpose:** The data collection and extraction engine. 
    *   **Details:** It utilizes Google's MediaPipe Holistic model to extract 3D coordinates (x, y, z) for hand and upper-body poses. It can ingest data from live webcam feeds or import massive datasets directly from the Kaggle CSV format. It outputs "raw" landmark data into CSV files.

*   **`kinematic_engineer.py` (Phase 2)**
    *   **Purpose:** The mathematical feature engineering core for static gestures.
    *   **Details:** It reads the raw landmarks generated in Phase 1 and converts them into rotation-invariant, scale-invariant geometric features. Instead of absolute spatial coordinates, it computes Joint Angles, Spread Angles, and Normalized Distances centered on the wrist. This dramatically reduces input dimensionality and boosts model accuracy to ~99.9%.
*   **`train_classifier.py` (Phase 3)**
    *   **Purpose:** The Deep Learning training script for static alphabet classification (A-Z).
    *   **Details:** It defines and trains a Multi-Layer Perceptron (MLP) model using TensorFlow/Keras. It reads the kinematic CSVs, normalizes the data, trains the network, evaluates it, and generates confusion matrices. It also includes functions for ablation studies to compare the performance of raw coordinates vs. engineered angles.
*   **`realtime_inference.py` (Phase 4)**
    *   **Purpose:** Live webcam application for static gesture prediction.
    *   **Details:** Utilizes OpenCV and the trained Phase 3 MLP to capture live video, extract MediaPipe landmarks, apply the exact same kinematic math in real-time, and display the predicted letter and confidence on a Heads-Up Display (HUD).
*   **`temporal_engineer.py` (Phase 5)**
    *   **Purpose:** The data pre-processor for dynamic, temporal sequence learning.
    *   **Details:** Processes raw `.npy` recording sequences of continuous signs. It applies the kinematic math to every single frame, interpolates short/long videos into a standardized length (e.g., 30 frames), performs speed augmentation, and outputs 3D tensors suitable for Recurrent Neural Networks.
*   **`train_dynamic.py` (Phase 6)**
    *   **Purpose:** The Deep Learning training script for dynamic sentence translation.
    *   **Details:** Reads the 3D tensors from Phase 5 and trains a 2-layer Long Short-Term Memory (LSTM) network. It learns the sequential time-series patterns for 100+ ISL sentences. It evaluates performance and saves the final `.h5` LSTM model.
*   **`realtime_dynamic.py` (Phase 7)**
    *   **Purpose:** Live webcam continuous translation.
    *   **Details:** Employs a sliding-window approach over the webcam feed, continuously feeding the last ~1 second of frames into the LSTM. When the model reaches a high-confidence threshold, it outputs the translated ISL sentence to a live subtitle bar.

### 🛠️ Supporting Scripts & Tools

*   **`realtime_fingerspell.py`**
    *   **Purpose:** Specialized real-time spelling application.
    *   **Details:** Instead of isolated letters or full sentences, this script focuses on constructing continuous words and sentences letter-by-letter. It features a UI tracking the built word, a hold-progress bar to confirm a letter, and auto-spacing.
*   **`run_pipeline.py`**
    *   **Purpose:** The master orchestrator for Phases 1 to 3.
    *   **Details:** Provides a single, unified CLI command to automatically chain data extraction, kinematic engineering, and static model training back-to-back.

*   **`test_all_letters.py`**
    *   **Purpose:** A gamified testing/validation script.
    *   **Details:** Guides the user through a sequential test, challenging them to sign all letters from A to Z in front of the webcam, verifying the signs against the trained AI model in real-time.
*   **`paths.py`**
    *   **Purpose:** Centralized configuration map.
    *   **Details:** Prevents hardcoding issues by storing all directory paths, file names, and dynamically generating unique run names. Included in almost all other scripts.
*   **`test_mutable.py`**
    *   **Purpose:** A minimal scratchpad/testing script.
    *   **Details:** A quick technical validation to check if Google MediaPipe `NormalizedLandmarkList` objects are mutable in the current Python environment.

### 📝 Documentation & Configs

*   **`README.md`**: The master overview. Contains project architecture, methodology explanation, installation steps, and the execution roadmap.
*   **`SETUP.md`**: Detailed instructions on setting up Conda/Pip virtual environments and hardware acceleration (CUDA/CuDNN) for TensorFlow.
*   **`phases_5_6_7_plan.md`**: Deep dive into the methodology, file outputs, and logic driving the temporal LSTM architecture.
*   **`requirements.txt`**: The pip dependency file containing exact library versions (e.g., `mediapipe`, `tensorflow`, `opencv-python`).
*   **`.gitignore`**: Defines which large data folders, model binaries, and cache files Git should ignore.

---

## 📁 Sub-Directories

*   **`data/`**
    *   **Purpose:** The central data hub for the static pipeline (Phases 1-3).
    *   **Contents:** Stores massive CSV files. `raw_dataset.csv` contains raw MediaPipe coordinates, while `kinematic_dataset.csv` and `angles_dataset.csv` contain the mathematically engineered features. It also houses a `dynamic/` subfolder containing the `.npy` 3D tensors generated in Phase 5.
*   **`dataset/`**
    *   **Purpose:** The raw, unadulterated source materials.
    *   **Contents:** Where the initial Kaggle CSVs or massive folders of raw hand-sign images are downloaded before processing begins.
*   **`models/`**
    *   **Purpose:** The binary repository.
    *   **Contents:** Contains all trained neural networks (`.h5` files), SciKit-Learn standard scalers (`.pkl`), and label encoders. Models are usually saved with unique timestamps or configuration names (e.g., `angles_only.h5`).
*   **`npy_dataset/`**
    *   **Purpose:** The raw recording database for Phase 5-7.
    *   **Contents:** Hundreds of subdirectories named after sentences (e.g., `bring water for me`, `can i help you`). Inside are raw NumPy arrays (`.npy`), each representing a recorded video sequence of hand movements over time.
*   **`plots/`**
    *   **Purpose:** Visual artifacts for evaluation.
    *   **Contents:** Auto-generated during training phases. Includes Confusion Matrices (`.png`), classification reports (`.txt`), and loss/accuracy curves.
*   **`viva/`**
    *   **Purpose:** Academic and presentation defense materials.
    *   **Contents:** Contains rich markdown files (`VIVA_DEEP_DIVE.md`, `VIVA_EXHAUSTIVE_QUESTIONS.md`, etc.) designed for project walkthroughs, architectural justification, and answering theoretical defense questions.
*   **`venv/` & `__pycache__/`**
    *   **Purpose:** System directories. `venv/` holds the isolated Python environment, and `__pycache__/` contains compiled Python bytecode for faster execution.

---

## 🚀 Suggestions for Better Structure & File Management

As the project has evolved from simple static signs to complex dynamic LSTMs, the root directory has become slightly crowded. Adopting standard Software Engineering practices will make the repository significantly more professional and maintainable.

### 1. Adopt a Standard `src/` Layout
Currently, all 13 Python scripts sit in the root directory. Moving functional code into a modular `src/` (or `isl_engine/`) package prevents clutter and makes importing functions cleaner.
```text
Indian-Sign-Language-Detection/
├── src/
│   ├── data_pipeline/         # isl_detection.py

│   ├── features/              # kinematic_engineer.py, temporal_engineer.py
│   ├── modeling/              # train_classifier.py, train_dynamic.py
│   ├── inference/             # realtime_inference.py, realtime_dynamic.py, realtime_fingerspell.py
│   └── utils/                 # paths.py
├── scripts/                   # run_pipeline.py, test_all_letters.py
```

### 2. Centralized Configuration File (`config.yaml`)
Variables like `confidence_threshold = 0.85`, `sequence_length = 30`, and hardcoded camera indices (`camera_idx = 0`) are scattered across multiple scripts. Consolidating `paths.py` and these magic numbers into a central `config.yaml` or `config.json` will allow for tweaking the pipeline without touching the Python code.

### 3. Implement a Command-Line Interface (CLI)
Instead of having multiple standalone scripts (`python run_pipeline.py`, `python train_dynamic.py`), unify the project under a single entry point using `argparse` or `click`.
*   *Example Usage:* 
    *   `python main.py extract --type static`
    *   `python main.py train --model lstm --epochs 50`
    *   `python main.py realtime --mode fingerspell`

### 4. Isolate Testing & Scratchpads
Scripts like `test_mutable.py` and `test_all_letters.py` should be migrated to a dedicated `tests/` directory. If they are meant for unit testing, rewrite them using `pytest` to automatically ensure pipeline stability when code is changed.

### 5. Introduce Logging over Print Statements
In Phase 3 and Phase 6, training can take a long time. Replacing standard `print()` functions with Python's built-in `logging` module will allow you to save detailed execution traces to a `logs/` directory, helping immensely with debugging failed runs or silent MediaPipe crashes.

### 6. Introduce a `notebooks/` Directory
If you use Jupyter Notebooks for exploratory data analysis (EDA), visualizing the 3D tensors, or prototyping new models, create a `notebooks/` folder. This keeps experimental data science work strictly separated from production `.py` pipeline code.