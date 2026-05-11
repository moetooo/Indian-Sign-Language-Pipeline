# Phases 5-6-7 — Implementation Plan ✅

All 3 scripts have been created. Here's how to run them.

---

## How to Run

```powershell
conda activate isl_py39
pip install seaborn    # one-time, needed for Phase 6 plots

# Phase 5 — Process npy_dataset into 3D tensors (~2-5 min)
python src/features/temporal_engineer.py

# Phase 6 — Train LSTM (~5-20 min depending on GPU)
python src/modeling/train_dynamic.py

# Phase 7 — Live webcam translation
python src/inference/realtime_dynamic.py
```

---

## Files Created

| File | Phase | What it does |
|------|-------|-------------|
| `temporal_engineer.py` | 5 | npy_dataset → kinematic math per frame → pad to 30 frames → save tensors |
| `train_dynamic.py` | 6 | Load tensors → 2-layer LSTM → train → evaluate → save model |
| `realtime_dynamic.py` | 7 | Webcam → sliding window → LSTM predict → subtitle UI |

## Output Files

| Phase | Output |
|-------|--------|
| 5 | `data/dynamic/X_dynamic.npy`, `y_dynamic.npy`, `class_names.npy` |
| 6 | `models/isl_dynamic_lstm.h5`, `scaler_dynamic.pkl`, `label_encoder_dynamic.pkl` |
| 6 | `plots/dynamic_confusion.png`, `plots/dynamic_classification_report.txt` |

---

## Plain English Summary

> **Phase 5** takes your 687 raw video recordings and converts each one into a standardized "30-frame movie" of 182 mathematical features per frame (joint angles, centered coordinates). All videos become the same length via interpolation. Speed augmentation triples the dataset.
>
> **Phase 6** trains an LSTM neural network — a type of AI with "memory" — to watch these 30-frame sequences and learn which pattern of hand movement corresponds to which ISL sentence (out of 101 possible).
>
> **Phase 7** connects your webcam to this trained LSTM. It continuously feeds the last 1 second of hand movement into the model, and when it confidently recognizes a sentence, it adds it to a subtitle bar on screen — like live captions for sign language.

---

## Future Scope: Fingerspelling Integration

The [ISL-Fingerspelling dataset](https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling) (Kirandevraj et al., AACL-IJCNLP 2025) contains **1,308 videos** of continuous ISL fingerspelling with character-level transcriptions.

**How it could extend this project:**

1. **Hybrid Architecture** — When the sentence-level LSTM detects an unknown/low-confidence gesture, switch to a fingerspelling decoder that recognizes letter-by-letter hand spelling. This handles proper nouns, place names, and out-of-vocabulary words.

2. **CTC Decoder** — Unlike sentence classification (fixed labels), fingerspelling requires a **Connectionist Temporal Classification (CTC)** loss function that outputs variable-length character sequences from variable-length inputs.

3. **Pipeline Design:**
   ```
   Webcam → MediaPipe → Kinematic Math
         ↓                        ↓
   Sentence LSTM (101 classes)   Fingerspelling CTC-LSTM (A-Z characters)
         ↓                        ↓
         └── Confidence Router ──→ Subtitle Output
   ```

4. **Thesis Mention:** *"The system could be extended with fingerspelling recognition using datasets like ISL-Fingerspelling (Kirandevraj et al., 2025) to handle proper nouns and out-of-vocabulary words via a CTC-based decoder, enabling unrestricted continuous ISL-to-text translation."*
