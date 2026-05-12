# ISL Detection — Base Paper Comparison & Viva Preparation

## Paper Reference

> **"Indian Sign Language Recognition Using MediaPipe Holistic"**
> Authors: Kaushal Goyal, Dr. Velmathi G
> Institution: Vellore Institute of Technology, Chennai
> Keywords: ISL, CNN, LSTM, MediaPipe Holistic, static & gesture signs

The base paper has **two parts**: static letter recognition (CNN) and gesture phrase recognition (LSTM). Our project is split accordingly:
- **Part 1 (Current):** Static ISL letter recognition — compared against the paper's CNN/LSTM static results
- **Part 2 (Future):** Dynamic ISL phrase recognition — our planned Phase 5-7 roadmap, compared against the paper's LSTM gesture approach

---

# Part 1: Static ISL (Current Project vs Base Paper Static)

## Side-by-Side Comparison

| Aspect | Base Paper (Static) | Our Project (Part 1) |
|--------|-------------------|---------------------|
| **Scope** | 26 static alphabet letters | 26 static alphabet letters |
| **Landmark Extraction** | MediaPipe Holistic → 1662 keypoints (face + hands + pose) | MediaPipe Holistic → 144 keypoints (hands + upper pose, NO face) |
| **Face Landmarks** | ✅ 468 face points included | ❌ Excluded (irrelevant for hand signs) |
| **Data Storage** | NumPy arrays (.npy, 30 frames × 1662 per video) | CSV (1 row × 144 values per sample) |
| **Dataset Size** | ~1,800 samples (60 videos × 30 frames, self-collected) | **78,000+ samples** (Kaggle CSV) |

| **Model** | CNN (3 Conv2D layers) + LSTM (3 LSTM layers) | Deep MLP (512→256→128→26) |
| **Best Static Accuracy** | CNN: 97.75% F1 / LSTM: 85.57% F1 | **MLP: 99.92% accuracy** |
| **Feature Engineering** | None (raw keypoints only) | 3-stage: Centering → Normalization → Joint Angles |
| **Feature Types Tested** | 1 (raw coordinates) | **4 types:** raw, kinematic, angles_only, source_kaggle |
| **Position Invariance** | ❌ Not addressed | ✅ Centered on wrist midpoint |
| **Scale Invariance** | ❌ Not addressed | ✅ Normalized to [-1, 1] |
| **Rotation Invariance** | ❌ Not addressed | ✅ Joint angles independent of orientation |
| **Real-Time Inference** | Django web app + webcam | Standalone OpenCV with live model switching (1-7 keys) |
| **Stability Logic** | None mentioned | 5-frame rolling buffer + majority vote + 85% confidence |
| **Ablation Study** | CNN vs LSTM only | **4 models × 4 feature types** |

| **Regularization** | L2 + Dropout(0.2) + BatchNorm | Dropout(0.2/0.3) + BatchNorm + EarlyStopping + ReduceLROnPlateau |
| **Training Epochs** | 60-70 epochs | 50 max with EarlyStopping (patience=5) |
| **Reproducibility** | Self-collected only | Public Kaggle CSV |


---

## What We Improved (Static Recognition)

### 1. 40× Larger Dataset
- **Paper:** ~1,800 samples (self-collected webcam)
- **Ours:** 78,000+ samples from public Kaggle dataset
- **Impact:** Prevents overfitting, enables true generalization


### 2. Kinematic Feature Engineering
The paper feeds **raw coordinates** directly — sensitive to where/how the hand appears. We added:

| Transform | What it does | Why it matters |
|-----------|-------------|---------------|
| **Centering** | Wrist → origin (0,0,0) | Hand position doesn't affect prediction |
| **Normalization** | Scale to [-1, 1] | Hand size/distance irrelevant |
| **Joint Angles** | 38 angular features | Fully rotation-invariant |

> **Key result:** `angles_only` uses just **38 features** (vs 1,662 in paper) yet achieves **99.75% accuracy**.

### 3. Correct Architecture for the Task
- **Paper uses CNN** on keypoints → CNN is designed for 2D image grids, not 1D landmark vectors
- **Paper uses LSTM** on static signs → LSTM models temporal sequences, but static frames have no temporal component
- **We use MLP** → correct architecture for tabular/vector data

### 4. Removed 90% Noise (Face Landmarks)
Paper extracts 468 face points (1,404 values). For hand-based letter recognition, face data is pure noise. Removing it:
- Cut data size by ~90%
- Reduced model complexity
- Zero accuracy loss

### 5. Real-Time Stability System
| Feature | Paper | Ours |
|---------|-------|------|
| Flickering prevention | None | 5-frame rolling buffer |
| Prediction consistency | None | 60% majority vote required |
| Confidence filtering | None | 85% threshold |
| Live model switching | None | Keys 1-7 |

### 6. Scientific Ablation Study
Paper: CNN vs LSTM — 2 comparisons.
Ours: **4 models across 4 feature types** = proper ablation proving which features matter.


---

## Accuracy Comparison (Static Only)

| Metric | Base Paper CNN | Base Paper LSTM | Our Best (source_kaggle) | Our angles_only |
|--------|---------------|----------------|------------------------|----------------|
| Accuracy/F1 | 97.75% | 85.57% | **99.92%** | 99.75% |
| Features used | 1,662 | 1,662 | 126 | **38** |
| Training samples | ~1,800 | ~1,800 | 78,000+ | 78,000+ |
| Training time | Not specified | Not specified | ~63 seconds | ~63 seconds |

---

# Part 2: Dynamic ISL & Future Expansions (Planned)

## Base Paper's Dynamic Approach

The paper's gesture recognition uses:
- **LSTM with 3 layers** (64 → 128 → 64 units)
- **30 frames per video**, 1662 features per frame
- **5 gesture phrases** (words/sentences)
- **100% accuracy** on their small 5-class gesture test (likely overfitting on tiny dataset)
- **Sigmoid output** for multi-class (should be softmax)

### Limitations of the paper's dynamic approach:
1. Only 5 gesture classes — too small to generalize
2. 1,662 features per frame including face (unnecessary for most gestures)
3. No word boundary detection
4. No continuous translation (isolated gesture only)
5. Likely overfit on small self-collected dataset

---

## Our Planned Dynamic Roadmap

We will upgrade from single-frame classification to **continuous motion understanding** for words and phrases, addressing all the paper's limitations.

### Phase 5 — Temporal Data Engineering (Capturing Time)

**Goal:** Reshape data from `[Samples, Features]` → `[Samples, Time, Features]`

| Aspect | Paper's Approach | Our Plan |
|--------|-----------------|----------|
| Data shape | [Samples, 30, 1662] | [Samples, 30, Features] using kinematic features |
| Dataset | 5 words, self-collected | **ISL-CSLTR** (40,000+ videos, 320 classes) + **INCLUDE v3** (263 signs, real-world webcam) |
| Feature engineering | None | Phase 2 kinematic math applied to **every frame** |
| Storage | .npy files | `dynamic_isl_data.npy` |
| Augmentation | None | Temporal augmentation (speed simulation) + padding |
| Robustness testing | None | Physical robustness metrics from INCLUDE dataset |

**Key upgrade:** Apply our proven kinematic transforms (centering, normalization, angles) to every frame of the temporal sequence. This means the LSTM sees position/scale-invariant features over time, not raw coordinates.

### Phase 6 — Dynamic Model (LSTM Brain)

**Goal:** Train a sequence model for gesture phrase recognition

| Aspect | Paper's LSTM | Our Planned LSTM |
|--------|-------------|-----------------|
| Architecture | LSTM(64) → LSTM(128) → LSTM(64) → Dense(26, sigmoid) | Input → LSTM(64-128, return_seq) → LSTM(32-64) → Dropout(30-40%) → Dense(Softmax) |
| Classes | 5 gesture phrases | **320+ sign classes** |
| Features per frame | 1,662 (raw + face) | 38-182 (kinematic, no face) |
| Regularization | L2 + Dropout(0.2) | Dropout(30-40%) + EarlyStopping |
| Training data | ~300 sequences | **40,000+ video sequences** |
| Output | `model.h5` (generic) | `isl_dynamic_lstm.h5` |

**Defence point:** LSTM is chosen as lightweight compared to heavy attention models — appropriate for real-time inference on consumer hardware.

### Phase 7 — Continuous Translation Engine

**Goal:** Real-time sliding window for continuous sign-to-text

| Feature | Paper | Our Plan |
|---------|-------|----------|
| Input method | Isolated gesture → predict | Continuous sliding window via `deque(maxlen=30)` |
| Word boundaries | None | **Neutral Pose Detection** as spacebar (prevents output spam) |
| Language model | None | **ISLVT** gloss-grammar pairs for intelligent sign parsing |
| Output | Single word prediction | **Continuous sentence generation** |
| Validation | None | **iSign Benchmark** (118K sentence pairs) |

**Key innovation:** Word Boundary Logic — detecting when the signer returns to neutral pose to mark the end of one sign and the start of the next. This solves the continuous recognition problem that the paper doesn't address.

### Future Phase — Advanced Architectures

**Goal:** Compare LSTM against modern alternatives

| Method | Purpose | Advantage |
|--------|---------|-----------|
| **Dynamic Time Warping (DTW)** | Geometric matching baseline | No training needed, works with few samples |
| **Transformer** | Attention-based sequence modeling | Better at long-range dependencies than LSTM |
| **Vision Transformer (ViT)** | End-to-end from frames | Eliminates hand-crafted features entirely |

This establishes state-of-the-art context and proves our approach is competitive with cutting-edge methods.

---

## Part 2 vs Base Paper Dynamic — Planned Improvements Summary

| Metric | Base Paper (Dynamic) | Our Planned Part 2 |
|--------|---------------------|-------------------|
| Classes | 5 gesture phrases | **320+ sign classes** |
| Dataset | ~300 self-collected sequences | **40,000+ videos** (ISL-CSLTR + INCLUDE v3) |
| Features per frame | 1,662 (with face noise) | **38-182** (kinematic, position-invariant) |
| Architecture | LSTM only | LSTM + DTW + Transformer/ViT comparison |
| Word boundaries | None | Neutral Pose Detection |
| Translation mode | Isolated gesture | **Continuous sliding window** |
| Language model | None | ISLVT grammar-gloss pairs |
| Sentence validation | None | iSign Benchmark (118K pairs) |
| Real-time capable | Basic | Optimized `deque` sliding window |

---

# Viva Q&A — Questions and Answers

## Static Recognition (Part 1 — Current)

### Q1: How is your project different from the base paper?
**A:** Three major differences: (1) We engineered kinematic features (centering, normalization, joint angles) making our model position/scale/rotation invariant — the paper uses raw coordinates. (2) We use a Deep MLP instead of CNN/LSTM which is architecturally appropriate for landmark vectors. (3) We trained on 106K+ samples vs. ~1,800, achieving 99.92% accuracy vs. their 97.75%.


### Q2: Why didn't you use CNN like the paper?
**A:** CNN is designed for 2D spatial data (images with pixel grids). Our input is a 1D vector of landmark coordinates — there's no spatial grid structure for convolution to exploit. An MLP is the correct architecture for dense tabular data. Using CNN on landmark vectors is like trying to find edges in a spreadsheet.

### Q3: Why didn't you use LSTM for static signs?
**A:** LSTM captures temporal dependencies in sequences. Static sign recognition classifies **single frames independently** — there's no temporal sequence to model. LSTM is correct for gesture phrases (which we plan in Part 2), not for single-frame letter recognition. Even the paper shows LSTM only achieves 85.57% on static signs vs CNN's 97.75%, confirming this mismatch.

### Q4: Why did you remove face landmarks?
**A:** Face landmarks (468 points × 3 axes = 1,404 features) are relevant for facial expressions in gesture recognition but add zero predictive value for **hand-based static letters**. Removing them cut data size by ~90% with no accuracy loss, making the model faster and more focused.

### Q5: What are kinematic features? Why do they matter?
**A:** Kinematic features describe hand geometry independent of position/scale/orientation:
- **Centering:** Subtracts wrist position → hand always at origin
- **Normalization:** Divides by max extent → hand size doesn't matter
- **Joint Angles:** Measures actual finger bends → camera angle irrelevant

The same sign produces identical features whether the hand is close/far, left/right, tilted or straight.

### Q6: What is the ablation study? Why 7 models?
**A:** An ablation study systematically varies components to measure their impact. We trained 7 models:
- 4 feature types (source_kaggle / raw / kinematic / angles_only) on Kaggle data


This proves that kinematic features generalize better and that 38 angle features match 144 raw features while being more robust.

### Q7: Why is `angles_only` best for real-time webcam?
**A:** It uses **only 38 joint/spread angle features**, completely invariant to hand position, camera distance, hand orientation, and single/both hand visibility. Raw coordinates change dramatically with position/scale, causing misclassification when the hand moves.

### Q8: How does real-time stability work?
**A:** Three mechanisms: (1) Rolling buffer stores last 5 predictions, (2) Majority vote requires ≥60% agreement, (3) Confidence threshold requires ≥85% average. This prevents flickering between letters.

### Q9: If the paper uses 1662 features and you use 38, how are you more accurate?
**A:** More features ≠ better accuracy. The paper's 1,662 include 468 face landmarks (irrelevant), full body pose (mostly irrelevant), and raw coordinates (position-dependent). Our 38 angle features capture only the **geometric essence** of hand shape — pure signal, zero noise.

### Q10: What dataset did you use?
**A:** Kaggle "Indian Sign Language Gesture Landmarks" CSV — ~78K pre-extracted samples.


---

## Dynamic Recognition (Part 2 — Planned)

### Q11: How will you handle gesture/dynamic signs?
**A:** Phase 5-7 roadmap:
- **Phase 5:** Reshape data to temporal sequences [Samples, 30 frames, Features] using ISL-CSLTR (40K+ videos, 320 classes)
- **Phase 6:** Train LSTM sequence model with kinematic features per frame
- **Phase 7:** Continuous translation engine with sliding window and neutral pose word boundaries

### Q12: Why LSTM for dynamic signs and not CNN?
**A:** For temporal sequences, LSTM is specifically designed to capture time dependencies — which finger positions change over time during a gesture. CNN would treat frames independently. Even the base paper confirms LSTM achieves 100% on gesture recognition vs CNN's 90%.

### Q13: How will you prevent output spam in continuous recognition?
**A:** **Neutral Pose Detection** acts as a "spacebar" — when the signer returns hands to a neutral resting position between signs, the system marks a word boundary. This prevents the same sign from being output repeatedly.

### Q14: What is the advantage of applying kinematic features to every frame?
**A:** The base paper feeds raw 1,662-dimension vectors per frame. We'll apply centering, normalization, and angle extraction to **each of the 30 frames**, so the LSTM sees position-invariant features over time. This should dramatically improve robustness since the LSTM learns **how finger angles change**, not **where the hand moves on screen**.

### Q15: How does your dynamic approach compare to the paper's?
**A:** Key upgrades: (1) 320+ classes vs 5, (2) 40K+ training sequences vs ~300, (3) Kinematic features vs raw, (4) Continuous sliding window vs isolated gesture, (5) DTW/Transformer comparison for SOTA context, (6) ISLVT language model for grammar-aware parsing.

---

## General / Cross-Cutting Questions

### Q16: Can your system work for full sentences?
**A:** Currently (Part 1): No — static letters only. Planned (Part 2): Yes — continuous sliding window + neutral pose boundaries + ISLVT language model will enable real-time sentence-level translation.

### Q17: What are the limitations of your current system?
**A:**
1. Static letters only (A-Z), no gesture phrases yet
2. Requires clear hand visibility in camera
3. Similar hand shapes (H/U, M/N) can confuse the model
4. Depends on MediaPipe detection quality and lighting
5. No text-to-sign conversion (one-way only)

### Q18: What is the full project roadmap?
**A:**
| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Done | Landmark extraction (CSV + images + webcam) |
| Phase 2 | ✅ Done | Kinematic feature engineering |
| Phase 3 | ✅ Done | MLP training + ablation study (4 models) |

| Phase 4 | ✅ Done | Real-time inference with live model switching |
| Phase 5 | 🔜 Planned | Temporal data engineering (30-frame sequences) |
| Phase 6 | 🔜 Planned | Dynamic LSTM model training |
| Phase 7 | 🔜 Planned | Continuous translation engine |
| Future | 🔜 Planned | DTW + Transformer/ViT comparison |

---

## Summary: Before vs After

| Metric | Base Paper (Static) | Our Part 1 (Current) | Base Paper (Dynamic) | Our Part 2 (Planned) |
|--------|-------------------|---------------------|---------------------|---------------------|
| Accuracy | 97.75% (CNN) | **99.92%** | 100% (5 classes) | TBD (320+ classes) |
| Dataset | ~1,800 | **78,000+** | ~300 sequences | **40,000+ videos** |

| Features | 1,662 (with noise) | **38** (pure signal) | 1,662 | **38-182** (kinematic) |
| Engineering | None | Centering+Norm+Angles | None | Per-frame kinematic |
| Architecture | CNN/LSTM (mismatched) | MLP (correct) | LSTM only | LSTM + DTW + Transformer |
| Word boundaries | N/A | N/A | None | Neutral Pose Detection |
| Translation | Isolated | Isolated | Isolated | **Continuous** |
| Models compared | 2 | **4** | 1 | **3+** (LSTM/DTW/ViT) |

