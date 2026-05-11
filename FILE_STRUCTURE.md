# ISL Detection — File Structure & Workflow

## Directory Structure

```
Indian-Sign-Language-Detection/
│
├── 📁 data/                          ← Generated data (Phase 1 & 2 outputs)
│   ├── raw/                          ← Phase 1: Raw landmark CSVs
│   │   ├── isl_raw_data.csv            (78K rows × 147 cols, from Kaggle)
│   │   └── img_raw_data.csv            (28K rows × 147 cols, from images)
│   ├── kinematic/                    ← Phase 2: Centered + normalized + angles
│   │   ├── isl_kinematic_data.csv      (78K rows × 185 cols)
│   │   └── img_kinematic_data.csv      (28K rows × 185 cols)
│   └── angles/                       ← Phase 2: Angles only
│       ├── isl_angles_only.csv         (78K rows × 41 cols)
│       └── img_angles_only.csv         (28K rows × 41 cols)
│
├── 📁 models/                        ← Phase 3: Trained models + scalers
│   ├── isl_source_kaggle_mlp.h5        (126f, Kaggle CSV)
│   ├── isl_raw_mlp.h5                  (144f, Kaggle CSV)
│   ├── isl_kinematic_mlp.h5            (182f, Kaggle CSV)
│   ├── isl_angles_only_mlp.h5          (38f, Kaggle CSV)
│   ├── isl_img_raw_mlp.h5              (144f, Gesture Images)
│   ├── isl_img_kinematic_mlp.h5        (182f, Gesture Images)
│   ├── isl_img_angles_only_mlp.h5      (38f, Gesture Images)
│   └── scaler_*.pkl                    (StandardScaler for each model)
│
├── 📁 plots/                         ← Phase 3: Visualizations
│   ├── cm_*.png                        (Confusion matrices)
│   ├── history_*.png                   (Training curves)
│   └── ablation_summary.csv            (Comparison table)
│
├── 📁 dataset/                       ← Source datasets (input, not generated)
│   ├── Indian Sign Language Gesture Landmarks.csv
│   └── dataset - Gesture Speech/       (26 letter folders × 1200 images)
│
├── 📄 paths.py                       ← Central path config (all dirs defined here)
├── 📄 isl_detection.py               ← Phase 1: Main CLI, webcam capture, Kaggle import
├── 📄 kinematic_engineer.py          ← Phase 2: Feature engineering math
├── 📄 train_classifier.py            ← Phase 3: MLP training + ablation study
├── 📄 realtime_inference.py          ← Phase 4: Live webcam inference
├── 📄 test_all_letters.py            ← Testing: Guided A-Z letter test
├── 📄 run_pipeline.py                ← Unified: CSV or images → full pipeline
├── 📄 image_pipeline.py              ← Image-specific: images → landmarks → training
├── 📄 requirements.txt               ← Python dependencies
├── 📄 base_paper.pdf                 ← Reference research paper
└── 📄 .gitignore                     ← Excludes data/, models/, plots/ from git
```

---

## Data Flow Diagram

```
                         ┌─────────────────────┐
                         │   Source Datasets    │
                         │  (dataset/ folder)   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             Kaggle CSV      Image Folder      Webcam Live
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐ ┌─────────────────┐ ┌────────────┐
        │isl_detection.py│ │image_pipeline.py│ │isl_detection│
        │  import_kaggle │ │ MediaPipe Hands │ │  capture    │
        └───────┬───────┘ └────────┬────────┘ └─────┬──────┘
                │                  │                 │
                ▼                  ▼                 ▼
        ┌─────────────────────────────────────────────────┐
        │           data/raw/*.csv  (Phase 1)             │
        │         147 columns per row (144 features)      │
        └─────────────────────┬───────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │         kinematic_engineer.py  (Phase 2)        │
        │    Center → Normalize → Angles → Spreads        │
        └──────────┬──────────────────────┬───────────────┘
                   │                      │
                   ▼                      ▼
        data/kinematic/*.csv       data/angles/*.csv
          (185 columns)              (41 columns)
                   │                      │
                   └──────────┬───────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │         train_classifier.py  (Phase 3)          │
        │   Deep MLP × 4 feature sets = ablation study    │
        └──────────┬──────────────────────┬───────────────┘
                   │                      │
                   ▼                      ▼
           models/*.h5              plots/*.png
           models/*.pkl             plots/ablation_summary.csv
                   │
                   ▼
        ┌─────────────────────────────────────────────────┐
        │       realtime_inference.py  (Phase 4)          │
        │  Webcam → MediaPipe → Features → MLP → Display  │
        └─────────────────────────────────────────────────┘
```

---

## File Dependencies

```
paths.py ──────────────────┐
                           │
isl_detection.py ──────────┤ (standalone, imports nothing from project)
                           │
kinematic_engineer.py ─────┤ (standalone math functions)
     ▲                     │
     │                     │
train_classifier.py ───────┤ (imports from paths.py)
     ▲                     │
     │                     │
realtime_inference.py ─────┤ (imports from kinematic_engineer.py)
     ▲                     │
     │                     │
test_all_letters.py ───────┘ (imports from realtime_inference.py)

run_pipeline.py ───────────── (imports kinematic_engineer + train_classifier)
image_pipeline.py ─────────── (imports kinematic_engineer + train_classifier)
```

---

## What Each Script Does

| Script | Lines | Input | Output | When to Run |
|--------|-------|-------|--------|-------------|
| `isl_detection.py` | ~495 | Kaggle CSV / Webcam | `data/raw/isl_raw_data.csv` | Once for data collection |
| `kinematic_engineer.py` | ~413 | Raw CSV | Kinematic + Angles CSVs | Once after Phase 1 |
| `train_classifier.py` | ~366 | Any CSV | Models + Plots | Once to train models |
| `realtime_inference.py` | ~520 | Webcam + Model | Live prediction | Anytime for inference |
| `test_all_letters.py` | ~350 | Webcam + Model | A-Z test results | For evaluation |
| `run_pipeline.py` | ~320 | CSV or image dir | Everything | One-shot full pipeline |
| `image_pipeline.py` | ~280 | Image directory | Everything (images) | For image datasets |
| `paths.py` | ~63 | — | — | Imported by other scripts |
